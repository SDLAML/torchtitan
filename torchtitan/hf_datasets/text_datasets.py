# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from os import PathLike
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch

from datasets import Dataset, load_dataset
from datasets.data_files import DataFilesDict, DataFilesList
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV = "TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS"
TORCHTITAN_MIX_STATE_MAPPING_ENV = "TORCHTITAN_MIX_STATE_MAPPING"
_MIX_STATE_MAPPING_RULE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def infer_dataloader_snapshot_every_n_steps(
    checkpoint_enabled: bool, checkpoint_interval: int, gradient_accumulation_steps: int
) -> int:
    if checkpoint_enabled:
        return checkpoint_interval * gradient_accumulation_steps
    else:
        return 999999999999


def list_tree_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_tree_to_tuple(x) for x in obj)
    return obj


def _env_is_set(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value != ""


def _state_value_to_tensor(value, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"Checkpoint field {name!r} must be 1-D, got {tensor.ndim}-D")
    return tensor.to(dtype=dtype)


def _copy_checkpoint_vector(
    target: torch.Tensor,
    value,
    name: str,
    mapping: dict[int, int] | None = None,
    *,
    zero_unmapped: bool = False,
) -> None:
    loaded = _state_value_to_tensor(value, target.dtype, name)
    if mapping is None:
        target.copy_(loaded)
        return

    restored = torch.zeros_like(target) if zero_unmapped else target.clone()
    for current_idx, checkpoint_idx in mapping.items():
        restored[current_idx] = loaded[checkpoint_idx]
    target.copy_(restored)


def _parse_mix_state_mapping(
    mapping_spec: str,
    *,
    num_current_datasets: int,
    num_checkpoint_datasets: int,
) -> dict[int, int]:
    """Parse CURRENT_INDEX-CKPT_INDEX mapping rules from an env var."""
    rules = [rule.strip() for rule in re.split(r"[|,]", mapping_spec) if rule.strip()]
    if not rules:
        raise ValueError(
            f"{TORCHTITAN_MIX_STATE_MAPPING_ENV} must contain at least one mapping rule"
        )

    mapping: dict[int, int] = {}
    used_checkpoint_indices: set[int] = set()
    for rule in rules:
        match = _MIX_STATE_MAPPING_RULE.fullmatch(rule)
        if match is None:
            raise ValueError(
                f"Invalid {TORCHTITAN_MIX_STATE_MAPPING_ENV} rule {rule!r}; "
                "expected CURRENT_INDEX-CKPT_INDEX"
            )

        current_idx, checkpoint_idx = (int(index) for index in match.groups())

        if current_idx < 0 or current_idx >= num_current_datasets:
            raise ValueError(
                f"Invalid {TORCHTITAN_MIX_STATE_MAPPING_ENV} rule {rule!r}; "
                f"current dataset index {current_idx} is out of range "
                f"[0, {num_current_datasets})"
            )
        if checkpoint_idx < 0 or checkpoint_idx >= num_checkpoint_datasets:
            raise ValueError(
                f"Invalid {TORCHTITAN_MIX_STATE_MAPPING_ENV} rule {rule!r}; "
                f"checkpoint dataset index {checkpoint_idx} is out of range "
                f"[0, {num_checkpoint_datasets})"
            )
        if current_idx in mapping:
            raise ValueError(
                f"Invalid {TORCHTITAN_MIX_STATE_MAPPING_ENV}; "
                f"current dataset index {current_idx} is mapped more than once"
            )
        if checkpoint_idx in used_checkpoint_indices:
            raise ValueError(
                f"Invalid {TORCHTITAN_MIX_STATE_MAPPING_ENV}; "
                f"checkpoint dataset index {checkpoint_idx} is mapped more than once"
            )

        mapping[current_idx] = checkpoint_idx
        used_checkpoint_indices.add(checkpoint_idx)

    return mapping


def _process_simple_text(sample: dict[str, Any], key: str) -> str:
    """Process a simple custom dataset's sample text."""
    return sample[key]


def _prepared_data_files(
    dataset_path: str,
    dataset_files: str | Sequence[str] | None,
    dataset_split: str,
    dataset_streaming: bool,
) -> DataFilesDict | None:
    base_path = Path(dataset_path)
    if dataset_files is None:
        manifest_name = (
            "manifest.parquet.txt" if dataset_streaming else "manifest.json.txt"
        )
        manifest_path = base_path / manifest_name
        if not manifest_path.is_file():
            return None
        files = [
            line.strip()
            for line in manifest_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not files:
            raise ValueError(f"Manifest file {manifest_path} is empty")
    elif isinstance(dataset_files, (str, PathLike)):
        files = [str(dataset_files)]
    else:
        files = [str(path) for path in dataset_files]

    files = [
        str(path if Path(path).is_absolute() else base_path / path) for path in files
    ]
    # DataFilesList requires one origin-metadata entry per file. Using empty
    # tuples keeps the manifest lightweight and avoids HF's per-file fs.info()
    # pass during startup.
    origin_metadata = [()] * len(files)
    return DataFilesDict({dataset_split: DataFilesList(files, origin_metadata)})


def _load_simple_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    dataset_files: str | Sequence[str] | DataFilesDict | None = None,
    dataset_split: str = "train",
    dataset_streaming: bool = False,
):
    """Load a simple custom dataset with its configuration."""
    if dataset_files is not None and isinstance(dataset_files, DataFilesDict):
        return load_dataset(
            "parquet" if dataset_streaming else "json",
            data_files=dataset_files,
            split=dataset_split,
            streaming=dataset_streaming,
        )

    return load_dataset(
        dataset_path,
        name=dataset_name,
        data_files=dataset_files,
        split=dataset_split,
        streaming=dataset_streaming,
    )


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return _load_simple_dataset(
        dataset_path,
        dataset_name="en",
        dataset_files=None,
        dataset_split=split,
        dataset_streaming=True,
    )


DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, dataset_split="train"),
        sample_processor=partial(_process_simple_text, key="text"),
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=partial(_load_simple_dataset, dataset_split="train"),
        sample_processor=partial(_process_simple_text, key="text"),
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, dataset_split="validation"),
        sample_processor=partial(_process_simple_text, key="text"),
    ),
    "simple_custom": None,
}


def _validate_dataset(
    dataset_name: str,
    dataset_path: str | None,
    dataset_inner_name: str | None,
    dataset_files: str | Sequence[str] | None,
    dataset_split: str,
    dataset_streaming: bool,
    dataset_key: str,
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    if config is None:
        # that goes to simple_custom, we need to read everything from the config
        assert dataset_path is not None
        config = DatasetConfig(
            path=dataset_path,
            loader=lambda path: _load_simple_dataset(
                path,
                dataset_inner_name,
                dataset_files,
                dataset_split,
                dataset_streaming,
            ),
            sample_processor=lambda sample: _process_simple_text(sample, dataset_key),
        )
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        dataset_inner_name: str | None = None,
        dataset_files: str | Sequence[str] | None = None,
        dataset_split: str = "train",
        dataset_streaming: bool = False,
        dataset_key: str = "text",
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, sample_processor = _validate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_inner_name=dataset_inner_name,
            dataset_files=dataset_files,
            dataset_split=dataset_split,
            dataset_streaming=dataset_streaming,
            dataset_key=dataset_key,
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.infinite = infinite
        self._sample_processor = sample_processor

        # Variables for checkpointing
        self._sample_idx = 0

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        while True:
            num_yielded = 0
            for sample in self._get_data_iter():
                self._sample_idx += 1
                # Use the dataset-specific text processor
                try:
                    sample_text = self._sample_processor(sample)
                except Exception:
                    # bad row / missing key / load error -> skip, but state is correct
                    continue

                # Skip None / empty
                if sample_text is None:
                    continue
                if isinstance(sample_text, str) and not sample_text.strip():
                    continue

                try:
                    sample_tokens = self._tokenizer.encode(
                        sample_text, add_bos=True, add_eos=True
                    )
                except Exception:
                    # tokenization error -> skip, state is correct
                    continue

                num_yielded += 1
                yield sample_tokens

            if not self.infinite:
                logger.warning(
                    f"HuggingFaceDataset {self.dataset_name} from {self.dataset_path} has run out of data"
                )
                break
            elif num_yielded == 0:
                # "HuggingFaceDataset (shard on this rank) yielded 0 samples. Stopping iteration to prevent infinite loop."
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"HuggingFaceDataset {self.dataset_name} from {self.dataset_path} is being re-looped"
                )
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class MixedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        datasets: list[IterableDataset],
        dp_rank: int,
        weights: list[float] | None,
        seed: int | None = 0,
        normalize_by_length: bool = False,
        seq_len: int | None = None,
        drop_long_samples: bool = False,
        dataset_aliases: list[str] | None = None,
    ):
        self.datasets = datasets
        self.dataset_aliases = (
            [str(i) for i in range(len(self.datasets))]
            if dataset_aliases is None
            else dataset_aliases
        )
        if len(self.dataset_aliases) != len(self.datasets):
            raise ValueError(
                "dataset_aliases must have the same length as datasets "
                f"get len(datasets) = {len(self.datasets)} and "
                f"len(dataset_aliases) = {len(self.dataset_aliases)}"
            )

        _initial_weights = [1.0] * len(self.datasets) if weights is None else weights
        self.weights = torch.tensor(
            _initial_weights, dtype=torch.float64
        ).share_memory_()

        self.num_docs_sampled = torch.zeros(
            len(self.datasets), dtype=torch.int64
        ).share_memory_()

        self.num_tokens_sampled = torch.zeros(
            len(self.datasets), dtype=torch.int64
        ).share_memory_()

        # Flags for  "exhausted"  datasets
        self.removed = torch.zeros(len(self.datasets), dtype=torch.bool).share_memory_()

        self._dataset_indices = list(range(len(self.datasets)))
        self._sample_idx = 0
        self._data_iters = None
        base_seed = 0 if seed is None else seed
        self._rng = Random(base_seed + dp_rank)
        self._dp_rank = dp_rank

        # When normalize_by_length=True, _sample_dataset rescales weights by per-dataset
        # average doc length (= num_tokens_sampled / num_docs_sampled) before
        # sampling, converting document-count weights into token-fraction weights.
        self.normalize_by_length = normalize_by_length
        self.seq_len = seq_len
        self.drop_long_samples = drop_long_samples

    @property
    def dataset_name(self):
        return "mixed"

    @property
    def dataset_path(self):
        return ",".join(
            str(getattr(dataset, "dataset_path", None)) for dataset in self.datasets
        )

    @property
    def normed_weights(self):
        weights_sum = sum(self.weights)
        return [w / weights_sum for w in self.weights]

    def _init_data_iters(self):
        self._data_iters = [iter(dataset) for dataset in self.datasets]

    def _sample_dataset(self, sample_idx: int):
        if self.normalize_by_length:
            # avg_len[i] = num_tokens[i] / num_sampled[i]
            # +1 in both numerator and denominator: cold start (0/0) → 1.0, negligible
            # bias after sufficient data. Eliminates any division-by-zero check.
            counts = self.num_docs_sampled.to(torch.float64)
            avg_len = (self.num_tokens_sampled.to(torch.float64) + 1) / (counts + 1)
            rescaled = self.weights / avg_len
            rescaled = rescaled / rescaled.sum()
            return self._rng.choices(self._dataset_indices, weights=rescaled.tolist())[
                0
            ]
        return self._rng.choices(self._dataset_indices, weights=self.weights.tolist())[
            0
        ]

    def set_weights(self, weights: list[float]):
        assert len(weights) == len(self.datasets), (
            "weights must have the same length as datasets"
        )
        w = torch.tensor(weights, dtype=torch.float64)
        w[self.removed] = 0.0
        self.weights.copy_(w)

    def _get_next(self, dataset_index: int):
        data_iter = self._data_iters[dataset_index]
        try:
            return next(data_iter)
        except StopIteration:
            dataset = self.datasets[dataset_index]
            logger.warning(
                f"Removing {dataset.dataset_name} | {dataset.dataset_path} from data mix."
            )
            self.weights[dataset_index] = 0.0
            self.removed[dataset_index] = True
            return None

    def __iter__(self):
        if self._data_iters is None:
            self._init_data_iters()
        while True:
            sample = None
            # Handle exhausted data iterators.
            while sample is None:
                if all(w == 0.0 for w in self.weights):
                    self._data_iters = None
                    return
                dataset_index = self._sample_dataset(self._sample_idx)
                sample = self._get_next(dataset_index)
                if (
                    sample is not None
                    and self.drop_long_samples
                    and self.seq_len is not None
                    and len(sample) > self.seq_len + 1
                ):
                    sample = None  # discard; loop picks next without updating counters

            self.num_docs_sampled[dataset_index] += 1
            # For mix_in_seq=False, samples are (dict, tensor) tuples from inner
            # GreedyPackedDataset — len() returns 2, not the token count. Use
            # self.seq_len directly in that case.
            self.num_tokens_sampled[dataset_index] += (
                self.seq_len
                if isinstance(sample, tuple) and self.seq_len is not None
                else len(sample)
            )
            self._sample_idx += 1
            yield sample

            if all(w == 0.0 for w in self.weights):
                logger.warning(
                    "Data mix is empty (all sampling weights have been set to zero); "
                    "stopping iteration."
                )
                break
        # Unset data iterators so they will be re-initialized.
        self._data_iters = None

    def load_state_dict(self, state_dict):
        """Restore mixed-dataset progress, optionally adapting checkpoint order.

        By default, the checkpoint must contain the same number of datasets as
        the current config and sub-dataset states are restored by position.

        Two env vars are intentionally supported for resume-time mix changes:
        - TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS: keep the current/config weights
          instead of restoring checkpoint weights.
        - TORCHTITAN_MIX_STATE_MAPPING: remap sub-dataset state by index using
          CURRENT_INDEX-CKPT_INDEX rules, e.g. "0-0|1-2". Current datasets not
          mentioned in the mapping are treated as new and keep fresh state.
        """
        self._sample_idx = state_dict["sample_idx"]
        dataset_states = state_dict["datasets"]

        if not isinstance(dataset_states, list):
            raise TypeError(
                f"Unsupported datasets state type: {type(dataset_states)}. "
                "This checkpoint was likely produced by an older version; please restart from scratch."
            )

        mapping_spec = os.environ.get(TORCHTITAN_MIX_STATE_MAPPING_ENV)
        dataset_state_mapping = None
        if mapping_spec is not None and mapping_spec.strip():
            dataset_state_mapping = _parse_mix_state_mapping(
                mapping_spec,
                num_current_datasets=len(self.datasets),
                num_checkpoint_datasets=len(dataset_states),
            )
            logger.info(
                f"Loading mixed dataset checkpoint state with "
                f"{TORCHTITAN_MIX_STATE_MAPPING_ENV}={mapping_spec!r}"
            )
        elif len(dataset_states) != len(self.datasets):
            raise ValueError(
                f"Checkpoint has {len(dataset_states)} dataset states, but current config has {len(self.datasets)}."
            )

        # Weights are the only checkpoint field that can be fully ignored. This
        # lets a resumed run intentionally use newly configured mix weights.
        if _env_is_set(TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV):
            logger.info(
                f"Keeping configured mixed dataset weights because "
                f"{TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV} is set"
            )
        else:
            _copy_checkpoint_vector(
                self.weights, state_dict["weights"], "weights", dataset_state_mapping
            )

        # Removed flags and sampling counters follow the mapping. Unmapped
        # current datasets start fresh instead of inheriting unrelated state.
        loaded_removed = state_dict.get("removed", None)
        if loaded_removed is None:
            # Old checkpoints: nothing was sticky; start with "nothing removed".
            self.removed.zero_()
        else:
            _copy_checkpoint_vector(
                self.removed,
                loaded_removed,
                "removed",
                dataset_state_mapping,
                zero_unmapped=dataset_state_mapping is not None,
            )

        self.weights[self.removed] = 0.0

        # NOTE: num_docs_sampled and num_tokens_sampled are sticky.
        _copy_checkpoint_vector(
            self.num_docs_sampled,
            state_dict["num_docs_sampled"],
            "num_docs_sampled",
            dataset_state_mapping,
            zero_unmapped=dataset_state_mapping is not None,
        )
        _copy_checkpoint_vector(
            self.num_tokens_sampled,
            state_dict["num_tokens_sampled"],
            "num_tokens_sampled",
            dataset_state_mapping,
            zero_unmapped=dataset_state_mapping is not None,
        )

        state_dict["rng_state"] = list_tree_to_tuple(state_dict["rng_state"])
        self._rng.setstate(state_dict["rng_state"])
        # Restore only mapped sub-datasets. Without an explicit mapping, this is
        # the original by-order restore path.
        restore_mapping = dataset_state_mapping or {
            index: index for index in range(len(self.datasets))
        }
        for current_idx, checkpoint_idx in sorted(restore_mapping.items()):
            self.datasets[current_idx].load_state_dict(dataset_states[checkpoint_idx])

        # Unset data iterators so they will be re-initialized.
        self._data_iters = None

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "weights": self.weights.tolist(),
            "removed": self.removed.tolist(),
            "num_docs_sampled": self.num_docs_sampled.tolist(),
            "num_tokens_sampled": self.num_tokens_sampled.tolist(),
            "datasets": [dataset.state_dict() for dataset in self.datasets],
            "rng_state": self._rng.getstate(),
        }


class GreedyPackedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: IterableDataset,
        seq_len: int = 2048,
        infinite: bool = False,
        num_mtp_tokens: int = 0,
        drop_long_samples: bool = False,
        eos_id: int | None = None,
    ) -> None:
        self._data = dataset
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_mtp_tokens = num_mtp_tokens
        self.drop_long_samples = drop_long_samples
        self.eos_id = eos_id

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    @property
    def dataset_name(self):
        return self._data.dataset_name

    @property
    def dataset_path(self):
        return self._data.dataset_path

    def _get_data_iter(self):
        # We don't use the sample index because we defer skipping to the
        # sub-dataset.
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len + self.num_mtp_tokens

        while True:
            num_yielded = 0
            for sample_tokens in self._get_data_iter():
                num_yielded += 1
                self._sample_idx += 1

                if self.drop_long_samples and len(sample_tokens) > max_buffer_token_len:
                    continue

                self._token_buffer.extend(sample_tokens)

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    # if self.eos_id is not None:
                    #     # Ignore the artificial EOS -> next-doc-start transition
                    #     # introduced by concatenative packing.
                    #     eos_mask = input == self.eos_id
                    #     if eos_mask.any():
                    #         label = label.clone()
                    #         label[eos_mask] = IGNORE_INDEX
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(
                    f"GreedyPackedDataset {self.dataset_name} from {self.dataset_path} has run out of data"
                )
                break
            elif num_yielded == 0:
                # "GreedyPackedDataset (shard on this rank) yielded 0 samples. Stopping iteration to prevent infinite loop."
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(
                    f"GreedyPackedDataset {self.dataset_name} from {self.dataset_path} is being re-looped"
                )
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._token_buffer = state_dict["token_buffer"]
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
            "dataset": self._data.state_dict(),
        }


class BestFitPackedDataset(IterableDataset, Stateful):
    """Pack complete documents into fixed-length sequences using best-fit-decreasing.

    Buffers `pool_size` documents from the upstream dataset, sorts them by length
    descending, and assigns each to the open bin (sequence) with the smallest
    remaining space that still fits it. Closed bins are padded to `seq_len` and
    yielded in a randomly shuffled order so that batch ordering is decorrelated
    from BFD's length-sort order (otherwise long-doc datasets dominate the first
    batches after each pool refill).

    Label positions corresponding to pad tokens are set to `IGNORE_INDEX` so
    padding does not contribute to the training loss.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        seq_len: int = 2048,
        infinite: bool = False,
        num_mtp_tokens: int = 0,
        drop_long_samples: bool = False,
        eos_id: int | None = None,
        pad_id: int = 0,
        pool_size: int = 4096,
        seed: int = 0,
    ) -> None:
        self._data = dataset
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_mtp_tokens = num_mtp_tokens
        self.drop_long_samples = drop_long_samples
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.pool_size = pool_size

        # Variables for checkpointing
        self._sample_idx = 0
        self._pool: list[np.ndarray] = []
        self._yield_queue: list[np.ndarray] = []
        self._rng = np.random.default_rng(seed=seed)

    @property
    def dataset_name(self):
        return self._data.dataset_name

    @property
    def dataset_path(self):
        return self._data.dataset_path

    def _get_data_iter(self):
        return iter(self._data)

    def _max_buffer_len(self) -> int:
        return 1 + self.seq_len + self.num_mtp_tokens

    def _bfd_pack(self, docs: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
        max_len = self._max_buffer_len()
        docs.sort(key=len, reverse=True)

        # Each bin: [list_of_doc_arrays, total_len_used]
        bins: list[list] = []
        for doc in docs:
            L = len(doc)
            best_idx = -1
            best_rem = max_len + 1
            for i, b in enumerate(bins):
                rem = max_len - b[1]
                if rem >= L and rem < best_rem:
                    best_idx = i
                    best_rem = rem
            if best_idx == -1:
                if L > max_len:
                    # Should have been filtered by drop_long_samples upstream.
                    continue
                bins.append([[doc], L])
            else:
                bins[best_idx][0].append(doc)
                bins[best_idx][1] += L

        # Materialize each bin into a (buf, used) pair. `used` is the number of
        # real tokens in the bin; positions [used, max_len) are padding. We pass
        # `used` through to `_emit` so it can mask the exact pad-target labels
        # — necessary when pad_id == eos_id, otherwise inferring pad positions
        # from token value would also mask the model's chance to learn the EOS
        # tokens at the end of real documents.
        out: list[tuple[np.ndarray, int]] = []
        for doc_list, used in bins:
            buf = np.empty(max_len, dtype=np.int64)
            off = 0
            for d in doc_list:
                buf[off : off + len(d)] = d
                off += len(d)
            if off < max_len:
                buf[off:] = self.pad_id
            out.append((buf, used))
        return out

    def _emit(self, packed_with_used: tuple[np.ndarray, int]):
        packed, used = packed_with_used
        x = torch.from_numpy(packed)
        input_ = x[:-1]
        label = x[1:].clone()
        # Label index i corresponds to predicting packed[i+1]. Positions
        # [used, max_len) are pad, so label indices i with i+1 >= used
        # (i.e. i >= used - 1) are pad targets and must be masked.
        max_len = packed.shape[0]
        if used < max_len:
            label[max(used - 1, 0) :] = IGNORE_INDEX
        return {"input": input_}, label

    def __iter__(self):
        max_len = self._max_buffer_len()

        while True:
            num_yielded = 0

            # Drain any bins left over from a previous epoch / checkpoint restore.
            while self._yield_queue:
                yield self._emit(self._yield_queue.pop(0))

            for sample_tokens in self._get_data_iter():
                num_yielded += 1
                self._sample_idx += 1

                if self.drop_long_samples and len(sample_tokens) > max_len:
                    continue

                self._pool.append(np.asarray(sample_tokens, dtype=np.int64))

                if len(self._pool) >= self.pool_size:
                    bins = self._bfd_pack(self._pool)
                    self._pool = []
                    self._rng.shuffle(bins)
                    self._yield_queue.extend(bins)
                    while self._yield_queue:
                        yield self._emit(self._yield_queue.pop(0))

            # Data exhausted — flush remaining pool.
            if self._pool:
                bins = self._bfd_pack(self._pool)
                self._pool = []
                self._rng.shuffle(bins)
                self._yield_queue.extend(bins)
            while self._yield_queue:
                yield self._emit(self._yield_queue.pop(0))

            if not self.infinite:
                logger.warning(
                    f"BestFitPackedDataset {self.dataset_name} from {self.dataset_path} has run out of data"
                )
                break
            elif num_yielded == 0:
                # No samples yielded by upstream this epoch; stop to avoid infinite loop.
                break
            else:
                self._sample_idx = 0
                logger.warning(
                    f"BestFitPackedDataset {self.dataset_name} from {self.dataset_path} is being re-looped"
                )
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._pool = [np.asarray(a, dtype=np.int64) for a in state_dict["pool"]]
        self._yield_queue = [
            (np.asarray(buf, dtype=np.int64), int(used))
            for buf, used in state_dict["yield_queue"]
        ]
        self._rng.bit_generator.state = state_dict["rng_state"]
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "pool": self._pool,
            "yield_queue": self._yield_queue,
            "rng_state": self._rng.bit_generator.state,
            "dataset": self._data.state_dict(),
        }


def _normalize_list(
    xs: list[str | None] | None,
    length: int,
    duplicate: bool = False,
) -> list[str | None]:
    if xs is None:
        xs = [None] * length
    elif duplicate and len(xs) == 1:
        xs = [xs[0] for _ in range(length)]
    return xs


def _coerce_to_list(
    xs: str | Sequence[str] | None,
) -> list[str] | None:
    if xs is None:
        return None
    if isinstance(xs, str):
        return [xs]
    return list(xs)


def _replace_none_with_literal(
    xs: str | Sequence[str] | None,
) -> list[str | None] | None:
    xs = _coerce_to_list(xs)
    if xs is None:
        xs = None
    else:
        xs = [None if x == "None" else x for x in xs]
    return xs


def _resolve_dataset_aliases(dataset_aliases: list[str | None]) -> list[str]:
    return [
        alias if alias is not None else str(i)
        for i, alias in enumerate(dataset_aliases)
    ]


class HuggingFaceTextDataLoader(ParallelAwareDataloader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        infinite: bool = True
        """Whether to loop the dataset indefinitely"""

        dataset: list[str] = field(default_factory=lambda: ["c4_test"])
        """Dataset to use"""

        dataset_alias: list[str | None] | None = None
        """
        Optional aliases used for data-mix logging.
        Entries with string "None" will be replaced with the Python literal `None`.
        """

        dataset_path: list[str] | None = None
        """
        Path to the dataset in the file system. If provided, data will be
        loaded from this path instead of downloaded.
        Entries with string "None" will be replaced with the Python literal `None`.
        """

        dataset_seed: int | None = None
        """
        Choose the base RNG seed used for data shuffling. By default,
        use the same as `training.seed`.
        """

        dataset_shuffle_buffer_size: int = 0
        """Buffer size of windowed shuffling buffer. 0 means no shuffling (the default)."""

        dataset_weights: list[float] | None = None
        """
        Probability of sampling from each dataset, separated by commas.
        If not given, sample uniformly.
        """

        dataset_mix_in_seq: bool = False
        """
        Whether to also mix datasets in the sequence dimension during
        packing. If not given, only mix in batch dimension.
        """

        dataset_inner_name: list[str] | None = None
        """
        Dataset name to use (`name` argument of `datasets.load_dataset`).
        Entries with string "None" will be replaced with the Python literal `None`.
        """

        dataset_files: list[str] | None = None
        """Dataset files to use (only necessary for certain types of datasets)"""

        dataset_split: list[str] = field(default_factory=lambda: ["train"])
        """Dataset split to use"""

        dataset_streaming: bool = False
        """Whether to stream the dataset"""

        dataset_key: list[str] = field(default_factory=lambda: ["text"])
        """Key to use for extracting the relevant text data from the dataset's samples"""

        data_mixing_scheduler_configs: str | None = None
        """Path to the mixing scheduler configs file
        The mixing scheduler configs file should be a JSON file with the following format:
        {
            "0": [weights_for_dataset_0, weights_for_dataset_1, ...],
        }
        or:
        {
            "@0%": [weights_for_dataset_0, weights_for_dataset_1, ...],
            "@10%": [weights_for_dataset_0, weights_for_dataset_1, ...],
        }
        Milestone keys must all be step strings or all percentage strings, and the
        initial milestone must be "0" or "@0%".
        """

        drop_long_samples: bool = False
        """Whether to drop samples longer than the sequence length"""

        pack_strategy: str = "greedy"
        """How to pack documents into sequences. "greedy" concatenates documents
        and cuts at seq_len boundaries (documents may split across sequences).
        "best_fit" buffers documents and uses best-fit-decreasing packing so that
        every sequence holds only complete documents (with some padding).
        """

        packing_pool_size: int = 4096
        """Only used when pack_strategy="best_fit". Number of documents buffered
        per rank before each BFD pass. Larger pools give tighter packing and
        lower per-batch ratio variance at the cost of more CPU memory and a
        longer warmup before the first batch.
        """

        pad_id: int | None = None
        """Only used when pack_strategy="best_fit". Token id written into padding
        positions; label positions corresponding to a pad token are masked with
        IGNORE_INDEX so padding does not contribute to the loss. If None, defaults
        to ``tokenizer.eos_id`` — works with FlexAttention's EOS-based document
        mask out of the box (each pad token becomes a one-token "document").
        """

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int = 1,
        seed: int | None = None,
        **kwargs,
    ):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        dataset_name = _coerce_to_list(config.dataset)
        if dataset_name is None:
            raise ValueError("dataset config must contain at least one dataset")
        dataset_alias = _replace_none_with_literal(config.dataset_alias)
        dataset_path = _replace_none_with_literal(config.dataset_path)
        dataset_streaming = config.dataset_streaming
        dataset_weights = config.dataset_weights
        dataset_mix_in_seq = config.dataset_mix_in_seq
        dataset_inner_name = _replace_none_with_literal(config.dataset_inner_name)
        dataset_files = _coerce_to_list(config.dataset_files)
        dataset_split = _coerce_to_list(config.dataset_split)
        dataset_key = config.dataset_key
        infinite = config.infinite

        normed_list_length = len(dataset_name)
        dataset_alias = _normalize_list(dataset_alias, normed_list_length)
        dataset_path = _normalize_list(dataset_path, normed_list_length)
        dataset_inner_name = _normalize_list(dataset_inner_name, normed_list_length)
        dataset_split = _normalize_list(
            dataset_split, normed_list_length, duplicate=True
        )
        dataset_key = _normalize_list(
            _coerce_to_list(dataset_key), normed_list_length, duplicate=True
        )
        dataset_weights = (
            [1.0] * normed_list_length
            if dataset_weights is None
            # Convert to floats.
            else list(map(float, dataset_weights))
        )
        resolved_dataset_aliases = _resolve_dataset_aliases(dataset_alias)
        drop_long_samples = config.drop_long_samples
        pack_strategy = config.pack_strategy
        if pack_strategy not in ("greedy", "best_fit"):
            raise ValueError(
                f"pack_strategy must be 'greedy' or 'best_fit', got {pack_strategy!r}"
            )
        packing_pool_size = config.packing_pool_size
        pad_id = config.pad_id
        if pack_strategy == "best_fit" and pad_id is None:
            if tokenizer.eos_id is None:
                raise ValueError(
                    "pack_strategy='best_fit' requires a pad_id, but config.pad_id is "
                    "unset and tokenizer.eos_id is None. Set config.pad_id explicitly."
                )
            pad_id = tokenizer.eos_id
        # Deterministic per-rank seed for the BFD output shuffle. Different
        # ranks get different shuffles; the same rank is reproducible across
        # resumes once load_state_dict restores the RNG state.
        bfd_seed = (seed or 0) * 1_000_003 + dp_rank

        if len(dataset_name) > 1:
            assert dataset_files is None, (
                "cannot supply dataset files when using multiple datasets"
            )
        prepared_dataset_files = [
            (
                None
                if d_path is None
                else _prepared_data_files(
                    d_path, dataset_files, d_split, dataset_streaming
                )
            )
            for d_path, d_split in zip(dataset_path, dataset_split)
        ]
        for d in [
            dataset_alias,
            dataset_path,
            dataset_inner_name,
            dataset_split,
            dataset_key,
            dataset_weights,
            prepared_dataset_files,
        ]:
            assert len(d) == normed_list_length, (
                f"list {d} does not match length of list of datasets (length = {normed_list_length})"
            )
        hf_datasets = []
        for d_name, d_path, d_inner_name, d_split, d_key, d_files in zip(
            dataset_name,
            dataset_path,
            dataset_inner_name,
            dataset_split,
            dataset_key,
            prepared_dataset_files,
        ):
            hf_ds = HuggingFaceDataset(
                dataset_name=d_name,
                dataset_path=d_path,
                tokenizer=tokenizer,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=infinite,
                dataset_inner_name=d_inner_name,
                dataset_files=d_files,
                dataset_split=d_split,
                dataset_streaming=dataset_streaming,
                dataset_key=d_key,
            )
            if not dataset_mix_in_seq:
                if pack_strategy == "greedy":
                    hf_ds = GreedyPackedDataset(
                        dataset=hf_ds,
                        seq_len=seq_len,
                        infinite=infinite,
                        drop_long_samples=drop_long_samples,
                        eos_id=tokenizer.eos_id,
                    )
                else:  # best_fit
                    hf_ds = BestFitPackedDataset(
                        dataset=hf_ds,
                        seq_len=seq_len,
                        infinite=infinite,
                        drop_long_samples=drop_long_samples,
                        eos_id=tokenizer.eos_id,
                        pad_id=pad_id,
                        pool_size=packing_pool_size,
                        seed=bfd_seed,
                    )
            hf_datasets.append(hf_ds)

        if torch.distributed.is_initialized():
            # do a final barrier to ensure all datasets are loaded before mixing
            torch.distributed.barrier()
        # First pack, then mix → data is only mixed in batch dimension.
        # First mix, then pack → data is also mixed inside packed sample.
        hf_ds = MixedDataset(
            hf_datasets,
            dp_rank,
            dataset_weights,
            seed=seed,
            normalize_by_length=dataset_mix_in_seq,
            seq_len=seq_len,
            drop_long_samples=drop_long_samples,
            dataset_aliases=resolved_dataset_aliases,
        )

        if dataset_mix_in_seq:
            if pack_strategy == "greedy":
                hf_ds = GreedyPackedDataset(
                    dataset=hf_ds,
                    seq_len=seq_len,
                    infinite=infinite,
                    drop_long_samples=drop_long_samples,
                    eos_id=tokenizer.eos_id,
                )
            else:  # best_fit
                hf_ds = BestFitPackedDataset(
                    dataset=hf_ds,
                    seq_len=seq_len,
                    infinite=infinite,
                    drop_long_samples=drop_long_samples,
                    eos_id=tokenizer.eos_id,
                    pad_id=pad_id,
                    pool_size=packing_pool_size,
                    seed=bfd_seed,
                )

        if len(dataset_name) == 1:
            snapshot_every_n_steps = 1
        else:
            snapshot_every_n_steps = snapshot_every_n_steps

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
            "generator": rng,
            "snapshot_every_n_steps": snapshot_every_n_steps,
        }

        super().__init__(
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
