# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical invariants of the square-polar and broken-symmetry models."""

import pytest
import torch
from torch.nn import functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.hf.configuration_opt_moe import OptMoeConfig
from torchtitan.models.opt_moe.hf.modeling_opt_moe import OptMoeForCausalLM
from torchtitan.models.opt_moe.polar import CosineLinear, PolarLinear, square_polar
from torchtitan.models.opt_moe.state_dict_adapter import hf_config


class _CausalAttention(torch.nn.Module):
    # FlexAttention backward requires an accelerator.
    def forward(self, q, k, v, *, scale, **kwargs):
        return F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=True,
            scale=scale,
        ).transpose(0, 1)


def test_polar_gradcheck_at_repeated_singular_values():
    w = torch.eye(4, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(square_polar, (w,))
    with pytest.raises(ValueError, match="square"):
        square_polar(torch.ones(2, 3))


@pytest.mark.parametrize(
    "flavor, input_invariant, hidden_invariant, output_scale",
    [
        ("s-1layer", True, True, 32.0),
        ("s-1layer-logit-scale-256", True, True, 256.0),
        ("s-1layer-broken-input", False, True, 32.0),
        ("s-1layer-broken-hidden", True, False, 32.0),
        ("s-1layer-broken-input-hidden", False, False, 32.0),
        ("s-1layer-broken-input-output", False, True, None),
        ("s-1layer-broken-all", False, False, None),
    ],
)
def test_independent_embedding_hidden_and_head_rescalings(
    flavor, input_invariant, hidden_invariant, output_scale
):
    torch.manual_seed(42)
    cfg = build_model_config(
        flavor,
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        dim=16,
        head_dim=8,
        vocab_size=32,
        seq_len=12,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    layer = model.layers["0"]
    assert isinstance(model.lm_head, CosineLinear) == (output_scale is not None)
    if output_scale is not None:
        assert not model.lm_head.logit_scale.requires_grad
        torch.testing.assert_close(
            model.lm_head.logit_scale.exp().squeeze(), torch.tensor(output_scale)
        )
    assert isinstance(layer.feed_forward.w13, PolarLinear) == hidden_invariant
    layer.attention.inner_attention = _CausalAttention()
    tokens = torch.arange(12)

    def forward():
        return model(tokens, positions=tokens)

    with torch.no_grad():
        before = forward()
        weight = model.tok_embeddings.weight
        weight.mul_(torch.linspace(0.4, 2.0, len(weight)).unsqueeze(1))
        after = forward()
        assert torch.allclose(before, after, atol=2e-4, rtol=1e-5) == input_invariant
        # Rescale each interleaved Q/K/V matrix independently and only gate in w13.
        qkv = layer.attention.qkv_linear.wqkv.weight.view(2, 3, 8, 16)
        qkv.mul_(torch.tensor([0.5, 2.0, 3.0]).view(1, 3, 1, 1))
        layer.feed_forward.w13.weight[0].mul_(2.0)
        after_hidden = forward()
        assert (
            torch.allclose(after, after_hidden, atol=2e-4, rtol=1e-5)
            == hidden_invariant
        )
        model.lm_head.weight[0].mul_(2.0)
        assert torch.allclose(after_hidden, forward(), atol=2e-4, rtol=1e-5) == (
            output_scale is not None
        )
    F.cross_entropy(forward(), tokens.roll(-1)).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad
    )


def test_polar_and_cosine_bfloat16_autocast_backward():
    torch.manual_seed(42)
    for layer in (
        PolarLinear.Config(in_features=8, out_features=8).build(),
        CosineLinear.Config(
            in_features=8,
            out_features=16,
            initial_logit_scale=1.0,
            logit_scale_trainable=True,
        ).build(),
    ):
        x = torch.randn(5, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x)
        assert output.dtype == torch.bfloat16
        output.float().square().mean().backward()
        assert all(
            p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
            for p in layer.parameters()
        )


@pytest.mark.parametrize("trainable", [False, True])
def test_hf_reload_preserves_logit_scale(tmp_path, trainable):
    cfg = build_model_config(
        "s-1layer",
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        dim=16,
        head_dim=8,
        vocab_size=32,
        seq_len=12,
        output_logit_scale_trainable=trainable,
    )
    hf = OptMoeForCausalLM(OptMoeConfig(**hf_config(cfg)))
    if trainable:
        with torch.no_grad():
            hf.lm_head.logit_scale.fill_(1.2345)
    hf.save_pretrained(tmp_path)
    loaded = OptMoeForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    scale = loaded.lm_head.logit_scale
    assert scale.requires_grad == trainable
    assert scale.dtype == torch.float32
    torch.testing.assert_close(scale, hf.lm_head.logit_scale, rtol=0, atol=0)
