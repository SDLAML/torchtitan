# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from unittest.mock import patch

import pytest
import torch

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.opt_moe import moe_opt_moe_configs
from torchtitan.models.opt_moe.config_registry import polar_si_config
from torchtitan.models.opt_moe.parallelize import parallelize_opt_moe
from torchtitan.models.opt_moe.utils.polar import (
    CosineLinear,
    PolarLinear,
    square_polar,
)
from torchtitan.optimizers.disco import DiSCO
from torchtitan.optimizers.utils import (
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
)


class _LocalParallelDims:
    fsdp_enabled = False
    ep_enabled = False
    dp_replicate_enabled = False
    tp_enabled = False
    world_mesh = "unit-test"

    def get_optional_mesh(self, _name):
        return None


def _tiny_config():
    cfg = copy.deepcopy(moe_opt_moe_configs["synth-proxy-1layer-polar-si"])
    cfg.dim = 8
    cfg.vocab_size = 16
    cfg.layer.feed_forward.hidden_dim = 8
    cfg.layer.attention.n_heads = 2
    cfg.layer.attention.n_kv_heads = 2
    cfg.layer.attention.head_dim = 4
    cfg.rope.dim = 4
    cfg.rope.max_seq_len = 8
    return cfg


def _spectral_radiality(weight, update):
    # CPU-compatible evaluation of the same norm derivative as radial_helper.
    left, _, right = torch.linalg.svd(weight.double(), full_matrices=False)
    update = update.double()
    return (left[:, 0] @ update @ right[0]) / torch.linalg.matrix_norm(update, 2)


@pytest.mark.parametrize("orthogonal", [False, True])
def test_square_polar_gradcheck_including_repeated_singular_values(orthogonal):
    torch.manual_seed(11)
    weight = torch.randn(4, 4, dtype=torch.float64)
    if orthogonal:
        weight = torch.eye(4, dtype=torch.float64)
    weight.requires_grad_()
    assert torch.autograd.gradcheck(square_polar, (weight,), atol=1e-6, rtol=1e-4)


def test_polar_stretch_invariance_and_gradient_tangency():
    torch.manual_seed(12)
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    a = torch.randn(6, 6, dtype=torch.float64)
    h = a.mT @ a + torch.eye(6, dtype=torch.float64)
    weight = (q @ h).requires_grad_()
    effective = square_polar(weight)
    torch.testing.assert_close(effective, q)
    torch.testing.assert_close(square_polar(q @ (h + 2 * torch.eye(6))), q)
    (effective * torch.randn_like(effective)).sum().backward()
    gradient = weight.grad
    assert gradient.norm() > 0
    assert abs(_spectral_radiality(weight, gradient)) < 1e-11
    torch.testing.assert_close(
        q.mT @ gradient + gradient.mT @ q,
        torch.zeros_like(weight),
        atol=1e-11,
        rtol=0,
    )


def test_polar_rejects_rectangular_weights():
    with pytest.raises(ValueError, match="square"):
        PolarLinear(8, 16)
    with pytest.raises(ValueError, match="square"):
        square_polar(torch.zeros(4, 8))


def test_normalized_output_has_independent_row_invariance():
    torch.manual_seed(13)
    head = CosineLinear(6, 10, initial_logit_scale=2.0).double()
    inputs = torch.randn(8, 6, dtype=torch.float64)
    original = head(inputs)
    with torch.no_grad():
        head.weight.mul_(torch.linspace(0.5, 2.0, 10).unsqueeze(1))
    torch.testing.assert_close(head(inputs), original)
    loss = torch.nn.functional.cross_entropy(head(inputs), torch.arange(8))
    loss.backward()
    torch.testing.assert_close(
        (head.weight * head.weight.grad).sum(dim=1),
        torch.zeros(10, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert head.logit_scale.grad.abs() > 0


def test_meta_initialization_and_existing_flavor_are_unchanged():
    cfg = _tiny_config()
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    model.init_weights(buffer_device=torch.device("cpu"))
    assert model.output.logit_scale.item() == 0.0
    assert sum(isinstance(m, PolarLinear) for m in model.modules()) == 7
    assert "output.weight" in dict(model.named_parameters())
    assert "output.logit_scale" in dict(model.named_parameters())
    old = moe_opt_moe_configs["synth-proxy-1layer-si"]
    assert old.layer.feed_forward.hidden_dim == 1024
    assert not old.layer.feed_forward.polar_weights
    assert not old.layer.attention.polar_weights
    assert not old.normalized_output


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_fullgraph_cpu_model_matches_eager_forward_and_backward(backend, tmp_path):
    # Avoid all-zero ReLU outputs: eps=1e-30 RMSNorm can give NaN gradients
    # there even in eager mode, independently of the polar implementation.
    torch.manual_seed(14)
    eager = _tiny_config().build()
    eager.init_weights(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        for module in eager.modules():
            if isinstance(module, PolarLinear):
                # Exactly repeated singular values exercise the custom backward.
                module.weight.copy_(torch.eye(module.in_features))
    compiled = copy.deepcopy(eager)
    config = polar_si_config()
    config.training.seq_len = 8
    config.compile.enable = True
    config.compile.backend = backend
    parallelize_opt_moe(
        compiled,
        parallel_dims=ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=1
        ),
        training=config.training,
        model_converters=config.model_converters,
        parallelism=config.parallelism,
        compile_config=config.compile,
        ac_config=config.activation_checkpoint,
        dump_folder=str(tmp_path),
    )
    assert hasattr(compiled.layers["0"], "_orig_mod")
    tokens = torch.arange(16).reshape(2, 8)
    labels = torch.roll(tokens, 1, dims=1)
    for _ in range(2):
        eager.zero_grad()
        compiled.zero_grad()
        expected, _ = eager(tokens)
        actual, _ = compiled(tokens)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4)
        for logits in (expected, actual):
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten()
            ).backward()
        with torch.no_grad():
            for original, traced in zip(
                eager.parameters(), compiled.parameters(), strict=True
            ):
                assert torch.isfinite(original.grad).all()
                assert torch.isfinite(traced.grad).all()
                torch.testing.assert_close(
                    traced.grad, original.grad, atol=2e-6, rtol=1e-4
                )
                original.add_(original.grad, alpha=-0.01)
                traced.add_(traced.grad, alpha=-0.01)


@pytest.mark.skipif(
    os.environ.get("TORCHTITAN_TEST_POLAR_CUDA") != "1",
    reason="CUDA integration test is opt-in via TORCHTITAN_TEST_POLAR_CUDA=1",
)
def test_actual_disco_steps_with_triton_backend():
    device = "cuda"
    torch.manual_seed(14)
    model = _tiny_config().build().to(device)
    model.init_weights(buffer_device=torch.device(device))
    config = polar_si_config()
    config.optimizer.lr = 0.05
    assert config.optimizer.zeropower_backend == "polar_express_triton"
    kwargs = create_disco_optimizer_kwargs_from_optimizer_config(
        config.optimizer, _LocalParallelDims()
    )
    groups, kwargs = create_disco_param_groups(model, kwargs)
    with patch("torchtitan.optimizers.disco.dist.get_rank", return_value=0):
        optimizer = DiSCO(groups, **kwargs)
    assert optimizer.communication_dtype == torch.float32
    assert optimizer.scale_param_names == ["output.logit_scale"]
    assert optimizer.embed_param_names == ["tok_embeddings.weight", "output.weight"]
    assert len(optimizer.ddp_params) == 7
    tokens = torch.arange(16, device=device).reshape(2, 8)
    labels = torch.roll(tokens, 1, dims=1)

    # Exercise the existing compiled Triton backend and the actual radial logger.
    for _ in range(3):
        optimizer.zero_grad()
        logits, _ = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten()
        )
        assert torch.isfinite(loss)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for module in model.modules():
            if isinstance(module, PolarLinear):
                q = square_polar(module.weight.detach()).double()
                gradient = module.weight.grad.double()
                relative_error = (q.mT @ gradient + gradient.mT @ q).norm() / gradient.norm()
                assert relative_error < 2e-5
        optimizer.calculate_norm_at_next_step(
            ["rms_to_rms", "rms_to_inf", "l1_to_rms"], gram_level=0
        )
        optimizer.step()
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
            if p.ndim != 2:
                continue
            weight = before[name]
            update = p.detach() - weight
            assert update.norm() > 0
            geometry = {
                "tok_embeddings.weight": "l1_to_rms",
                "output.weight": "rms_to_inf",
            }.get(name, "rms_to_rms")
            key = (
                f"track_radial_update_radiality_{geometry}/"
                f"{name.removesuffix('.weight')}"
            )
            logged = optimizer.get_norms_at_current_step()[key]
            assert torch.isfinite(logged), (key, logged.item())
            if geometry != "rms_to_rms":
                # Embedding/output use their existing FP32 row normalization.
                assert abs(logged) < 2e-5, (key, logged.item())
            # Hidden gradients are tangent above; BF16 dualization can introduce
            # radiality, so the hidden update has no FP64 near-zero assertion.
    assert model.output.logit_scale.item() != 0.0
