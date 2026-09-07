"""`_document_positions` must agree with upstream's packing convention.

Our loader is the torchdata one, so it cannot reuse grain's packing directly --
but the *convention* must be upstream's, not our own. Upstream derives packed
positions in `components/data/packing.py::_packing_output_to_text_sequence`;
this pins our torch implementation to that numpy one so the two cannot drift.

Run: python tests/unit_tests/opt_moe/test_document_positions.py
"""

import numpy as np
import torch

from torchtitan.hf_datasets.mixed_text_datasets import _document_positions

EOS = 5


def upstream_positions(row: np.ndarray, eos_id: int) -> np.ndarray:
    """Upstream's derivation, verbatim from _packing_output_to_text_sequence.

    Upstream starts from per-document positions produced by the packer; here we
    synthesize the same input from eos boundaries so the two are comparable.
    """
    starts = np.zeros(len(row), dtype=bool)
    starts[0] = True
    starts[1:] = row[:-1] == eos_id
    seed_positions = np.where(starts, 0, 1)  # any non-zero off a boundary

    boundaries = seed_positions == 0
    token_indices = np.arange(len(boundaries), dtype=np.int64)
    segment_starts = np.maximum.accumulate(np.where(boundaries, token_indices, 0))
    return token_indices - segment_starts


def main() -> int:
    rng = np.random.default_rng(0)
    cases = {
        "no eos at all": np.array([1, 2, 3, 4, 5000, 9]),
        "single trailing eos": np.array([1, 2, 3, EOS]),
        "leading eos": np.array([EOS, 1, 2, 3]),
        "adjacent eos": np.array([1, EOS, EOS, 2, 3]),
        "all eos": np.array([EOS] * 6),
        "eos at both ends": np.array([EOS, 1, 2, EOS]),
        "long random": rng.integers(0, 8, size=257),
    }
    failures = 0
    for name, row in cases.items():
        row = row.astype(np.int64)
        mine = _document_positions(
            torch.from_numpy(row).unsqueeze(0), EOS
        ).numpy()
        theirs = upstream_positions(row, EOS)
        ok = np.array_equal(mine, theirs)
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            print(f"        row   : {row.tolist()}")
            print(f"        mine  : {mine.tolist()}")
            print(f"        theirs: {theirs.tolist()}")

    # eos_id=None must degrade to a single document per row.
    plain = _document_positions(torch.zeros(2, 5, dtype=torch.int64), None)
    ok = plain.tolist() == [0, 1, 2, 3, 4] * 2
    failures += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} eos_id=None -> one document per row")

    print()
    print("MATCHES UPSTREAM" if failures == 0 else f"{failures} MISMATCH(ES)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
