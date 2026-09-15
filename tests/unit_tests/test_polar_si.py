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
from torchtitan.models.opt_moe.config_registry import moe_template_config
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


def _test_config():
    config = moe_template_config()
    config.activation_checkpoint.mode = "none"
    config.optimizer.name = "DiSCO"
    config.optimizer.zeropower_backend = "polar_express_triton"
    config.optimizer.momentum = 1.0
    config.optimizer.weight_decay = 0.0
    config.optimizer.eps = 1e-20
    config.optimizer.extra_param_group_split_rules = [
        {
            "str_match": "tok_embeddings.weight",
            "norm_factor": "embed_sqrt",
            "backend": "identity",
        },
        {
            "str_match": "output.weight",
            "norm_factor": "unembed_sqrt",
            "backend": "identity",
        },
        {
            "str_match": r"^output\.logit_scale$",
            "norm_factor": "sign",
            "backend": "identity",
        },
    ]
    return config


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


@pytest.mark.parametrize("layer_cls", [PolarLinear, CosineLinear])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_bf16_projection_preserves_parameter_gradients(layer_cls, input_dtype):
    torch.manual_seed(17)
    layer = layer_cls(16, 16)
    inputs = torch.randn(2, 3, 16).to(input_dtype).requires_grad_()
    leaves = (inputs, *layer.parameters())
    reference = layer(inputs)
    upstream = torch.randn_like(reference)
    expected_grads = torch.autograd.grad((reference * upstream).sum(), leaves)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(inputs)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    actual_grads = torch.autograd.grad((actual.float() * upstream).sum(), leaves)
    for leaf, actual_grad, expected_grad in zip(
        leaves, actual_grads, expected_grads, strict=True
    ):
        assert actual_grad.dtype == leaf.dtype
        assert torch.isfinite(actual_grad).all()
        relative_error = (actual_grad.float() - expected_grad.float()).norm()
        relative_error = relative_error / expected_grad.float().norm()
        assert relative_error < 0.02
    assert all(p.dtype == torch.float32 for p in layer.parameters())

    if isinstance(layer, PolarLinear):
        q = square_polar(layer.weight.detach()).double()
        grad = actual_grads[1].double()
        tangency_error = (q.mT @ grad + grad.mT @ q).norm() / grad.norm()
        assert tangency_error < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_scale_before_projection_matches_reference_gradients(dtype):
    torch.manual_seed(18)
    head = CosineLinear(8, 19, initial_logit_scale=2.3).to(dtype)
    inputs = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    hidden = torch.nn.functional.normalize(inputs, dim=-1, eps=1e-30)
    weight = torch.nn.functional.normalize(head.weight, dim=-1, eps=1e-30)
    reference = torch.nn.functional.linear(hidden, weight) * head.logit_scale.exp()
    actual = head(inputs)
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn_like(reference)
    leaves = (inputs, head.weight, head.logit_scale)
    expected_grads = torch.autograd.grad((reference * upstream).sum(), leaves)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), leaves)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("enable_amp", [False, True])
def test_output_head_does_not_save_full_logits_for_backward(enable_amp):
    head = CosineLinear(8, 19, initial_logit_scale=2.3)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=enable_amp),
    ):
        logits = head(inputs)
    assert tuple(logits.shape) not in saved_shapes
    assert (6, 19) not in saved_shapes
    logits.float().square().mean().backward()
    assert torch.isfinite(head.logit_scale.grad)


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


def test_fixed_output_scale_initialization_and_optimizer_step():
    torch.manual_seed(19)
    cfg = _tiny_config()
    cfg.output_logit_scale = 2.3
    cfg.output_logit_scale_trainable = False
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    model.init_weights(buffer_device=torch.device("cpu"))
    head = model.output
    assert not head.logit_scale.requires_grad
    assert head.logit_scale.exp().item() == pytest.approx(cfg.output_logit_scale)
    assert "output.logit_scale" in model.state_dict()

    config = _test_config()
    kwargs = create_disco_optimizer_kwargs_from_optimizer_config(
        config.optimizer, _LocalParallelDims()
    )
    with patch("torch.distributed.get_rank", return_value=0):
        groups, _ = create_disco_param_groups(model, kwargs)
    assert all(head.logit_scale is not p for group in groups for p in group["params"])

    inputs = torch.randn(2, 3, cfg.dim, requires_grad=True)
    expected = torch.nn.functional.linear(
        torch.nn.functional.normalize(inputs, dim=-1, eps=1e-30),
        torch.nn.functional.normalize(head.weight, dim=-1, eps=1e-30),
    ) * cfg.output_logit_scale
    logits = head(inputs)
    torch.testing.assert_close(logits, expected)
    original_weight = head.weight.detach().clone()
    original_scale = head.logit_scale.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.1)
    logits.square().mean().backward()
    assert head.logit_scale.grad is None
    assert torch.isfinite(inputs.grad).all() and inputs.grad.norm() > 0
    optimizer.step()
    torch.testing.assert_close(head.logit_scale, original_scale, rtol=0, atol=0)
    assert not torch.equal(head.weight, original_weight)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("enable_amp", [False, True])
def test_fullgraph_cpu_model_matches_eager_forward_and_backward(
    backend, enable_amp, tmp_path
):
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
    config = _test_config()
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
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enable_amp):
            expected, _ = eager(tokens)
            actual, _ = compiled(tokens)
        expected_dtype = torch.bfloat16 if enable_amp else torch.float32
        assert expected.dtype == actual.dtype == expected_dtype
        atol, rtol = (0.01, 0.02) if enable_amp else (2e-6, 1e-4)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for logits in (expected, actual):
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), labels.flatten()
            ).backward()
        with torch.no_grad():
            for original, traced in zip(
                eager.parameters(), compiled.parameters(), strict=True
            ):
                assert torch.isfinite(original.grad).all()
                assert torch.isfinite(traced.grad).all()
                torch.testing.assert_close(
                    traced.grad, original.grad, atol=atol, rtol=rtol
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
    config = _test_config()
    config.optimizer.lr = 0.05
    assert config.optimizer.zeropower_backend == "polar_express_triton"
    kwargs = create_disco_optimizer_kwargs_from_optimizer_config(
        config.optimizer, _LocalParallelDims()
    )
    kwargs["communication_dtype"] = torch.float32
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
