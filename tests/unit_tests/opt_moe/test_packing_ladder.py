"""Packing correctness for OPT MoE, as a ladder of configurations.

Each rung adds exactly one feature, so a failure says which feature broke it:

    dense/MoE  x  full/sliding-window attention  x  document mask on/off

The ``attn_mask_type="causal"`` rungs are negative controls: without the
document mask, packing invariance MUST fail. A suite whose every assertion
passes proves little -- these rungs demonstrate the check is actually sensitive
to the cross-document attention it is meant to catch.

Two properties are checked at every rung.

**Packing invariance** -- a document's outputs must not depend on where it sits
in the packed sequence. This is what makes packed training equivalent to
training each document on its own, and it is implementation-independent: it
does not treat any previous branch as ground truth. It fails if attention can
reach across document boundaries.

**RoPE offset invariance** -- shifting every position in a document by a
constant must not change anything, because RoPE is relative:
``<R(m)q, R(n)k>`` depends only on ``m - n``. This is why resetting positions
per document is *not* what makes packing correct; the document mask is. The
reset exists because upstream derives document boundaries from
``positions == 0``, not because RoPE needs it. Asserting it here keeps that
reasoning honest rather than folkloric.
"""

import torch

from torchtitan.models.common.attention import get_causal_mask_mod
from torchtitan.models.opt_moe import OPTMoEModel, OPTMoETransformerBlock
from torchtitan.models.opt_moe.gated_norm_swattention import GatedNormSWAttention
from torchtitan.models.opt_moe.norm_ffn import FeedForward
from torchtitan.models.opt_moe.norm_moe import MoE
from torchtitan.hf_datasets.mixed_text_datasets import _document_positions
from torchtitan.models.common.rope import CosSinRoPE

EOS = 5
SEQ = 256
DIM = 128
HEAD_DIM = 64
N_LAYERS = 2
SWA_WINDOW = 32


def build(*, moe: bool, swa: bool, doc_mask: bool, seq_len: int = SEQ):
    """Smallest model exercising one rung of the ladder."""
    attention = GatedNormSWAttention.Config(
        n_heads=2,
        n_kv_heads=1,
        head_dim=HEAD_DIM,
        qk_norm=True,
        norm_eps=1e-30,
        sliding_window_size=SWA_WINDOW if swa else -1,
        attn_backend="flex",
        attn_mask_type="block_causal" if doc_mask else "causal",
    )
    layer = OPTMoETransformerBlock.Config(
        n_dense_layers=0 if moe else N_LAYERS,
        attention=attention,
        feed_forward=None if moe else FeedForward.Config(hidden_dim=256),
        moe=(
            MoE.Config(hidden_dim=64, num_experts=4, num_shared_experts=1, top_k=2)
            if moe
            else None
        ),
    )
    config = OPTMoEModel.Config(
        n_layers=N_LAYERS,
        dim=DIM,
        vocab_size=512,
        layer=layer,
        swa_pattern=("S" if swa else "F") * N_LAYERS,
        rope=CosSinRoPE.Config(dim=HEAD_DIM, max_context_length=seq_len, theta=10000.0),
    )
    config._expand_layers()
    model = config.build()
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    model.eval()
    return model, config


def _forward(model, tokens, positions, masks):
    with torch.no_grad():
        out, _ = model(tokens, positions=positions, attention_masks=masks)
    return out.float()


def check_packing_invariance(model, doc, fillers):
    """Same document, different offsets in the pack -> same outputs."""
    outs = []
    for prefix in fillers:
        toks = list(prefix) + list(doc)
        toks += [7] * (SEQ - len(toks) - 1) + [EOS]
        assert len(toks) == SEQ
        t = torch.tensor(toks, dtype=torch.int64, device="cuda")
        pos = _document_positions(t.unsqueeze(0).cpu(), EOS).cuda()
        masks = model.get_attention_masks(positions=pos)
        out = _forward(model, t, pos, masks)
        outs.append(out[len(prefix) : len(prefix) + len(doc)])
    scale = outs[0].abs().max().item()
    return max((o - outs[0]).abs().max().item() for o in outs[1:]), scale


def check_rope_offset_invariance(model, config, length=48):
    """Constant position shift, mask held fixed -> same outputs."""
    toks = torch.randint(10, 200, (length,), generator=torch.Generator().manual_seed(1))
    toks = toks.cuda()
    attn_cfg = config.first_attention

    def run(offset):
        pos = torch.arange(length, device="cuda") + offset
        mask = model._create_flex_attention_mask(
            pos, attn_cfg, [get_causal_mask_mod()]
        )
        return _forward(model, toks, pos, {"full": mask, "swa": mask})

    base = run(0)
    scale = base.abs().max().item()
    return max((run(k) - base).abs().max().item() for k in (1, 17, 100)), scale


def main() -> int:
    g = torch.Generator().manual_seed(0)
    doc = torch.randint(10, 200, (24,), generator=g).tolist() + [EOS]
    fillers = [
        [],
        torch.randint(10, 200, (40,), generator=g).tolist() + [EOS],
        torch.randint(10, 200, (91,), generator=g).tolist() + [EOS],
    ]

    rungs = []
    for moe in (False, True):
        for swa in (False, True):
            for doc_mask in (True, False):
                name = (
                    f"{'MoE  ' if moe else 'dense'} + "
                    f"{'sliding window' if swa else 'full attention'} + "
                    f"{'doc mask' if doc_mask else 'NO doc mask'}"
                )
                rungs.append((name, dict(moe=moe, swa=swa, doc_mask=doc_mask)))

    failures = 0
    print(f"{'configuration':<44} {'packing':>18} {'rope offset':>16}")
    print("-" * 80)
    for name, kwargs in rungs:
        model, config = build(**kwargs)
        pack_d, pack_s = check_packing_invariance(model, doc, fillers)
        rope_d, rope_s = check_rope_offset_invariance(model, config)

        holds = pack_d < 2e-3 * pack_s
        # Without the document mask, tokens can attend into earlier documents,
        # so the same document must NOT produce the same outputs.
        expect_holds = kwargs["doc_mask"]
        pack_ok = holds == expect_holds
        rope_ok = rope_d < 2e-3 * rope_s

        failures += (not pack_ok) + (not rope_ok)
        verdict = "holds" if holds else "violated"
        tag = "OK" if pack_ok else "UNEXPECTED"
        print(
            f"{name:<44} {verdict:>9} {tag:>8} "
            f"{'OK' if rope_ok else 'FAIL':>16}"
        )
        print(f"{'':<44}   rel={pack_d/pack_s:.1e}      rel={rope_d/rope_s:.1e}")
        del model
        torch.cuda.empty_cache()

    print()
    print("expected: packing holds WITH the document mask, and is violated without it;")
    print("          rope offset invariance holds everywhere (RoPE is relative).")
    print()
    print("ALL CHECKS AS EXPECTED" if failures == 0 else f"{failures} CHECK(S) UNEXPECTED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
