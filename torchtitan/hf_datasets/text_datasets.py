# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from random import Random
from typing import Any

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


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


def _process_simple_text(sample: dict[str, Any], key: str) -> str:
    """Process a simple custom dataset's sample text."""
    return sample[key]


def _load_simple_dataset(
    dataset_path: str,
    dataset_name: str | None,
    dataset_files: str | Sequence[str] | None,
    dataset_split: str,
    dataset_streaming: bool,
):
    """Load a simple custom dataset with its configuration."""
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
    ):
        self.datasets = datasets

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
        self._rng = Random(seed + dp_rank)
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
        assert len(weights) == len(
            self.datasets
        ), "weights must have the same length as datasets"
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
        self._sample_idx = state_dict["sample_idx"]
        loaded_weights = state_dict["weights"]
        if isinstance(loaded_weights, torch.Tensor):
            self.weights.copy_(loaded_weights)
        else:
            self.weights.copy_(torch.tensor(loaded_weights, dtype=torch.float64))

        loaded_removed = state_dict.get("removed", None)
        if loaded_removed is None:
            # Old checkpoints: nothing was sticky; start with "nothing removed".
            self.removed.zero_()
        else:
            if isinstance(loaded_removed, torch.Tensor):
                self.removed.copy_(loaded_removed.to(dtype=torch.bool))
            else:
                self.removed.copy_(torch.tensor(loaded_removed, dtype=torch.bool))

        self.weights[self.removed] = 0.0

        # NOTE: num_docs_sampled and num_tokens_sampled are sticky.
        loaded_counts = state_dict["num_docs_sampled"]
        if isinstance(loaded_counts, torch.Tensor):
            self.num_docs_sampled.copy_(loaded_counts.to(dtype=torch.int64))
        else:
            self.num_docs_sampled.copy_(torch.tensor(loaded_counts, dtype=torch.int64))

        loaded_tokens = state_dict["num_tokens_sampled"]
        if isinstance(loaded_tokens, torch.Tensor):
            self.num_tokens_sampled.copy_(loaded_tokens.to(dtype=torch.int64))
        else:
            self.num_tokens_sampled.copy_(
                torch.tensor(loaded_tokens, dtype=torch.int64)
            )

        state_dict["rng_state"] = list_tree_to_tuple(state_dict["rng_state"])
        self._rng.setstate(state_dict["rng_state"])
        # Restore sub-datasets.
        dataset_states = state_dict["datasets"]

        if not isinstance(dataset_states, list):
            raise TypeError(
                f"Unsupported datasets state type: {type(dataset_states)}. "
                "This checkpoint was likely produced by an older version; please restart from scratch."
            )

        if len(dataset_states) != len(self.datasets):
            raise ValueError(
                f"Checkpoint has {len(dataset_states)} dataset states, but current config has {len(self.datasets)}."
            )
        for dataset, ds_state in zip(self.datasets, dataset_states):
            dataset.load_state_dict(ds_state)

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
                    if self.eos_id is not None:
                        # Ignore the artificial EOS -> next-doc-start transition
                        # introduced by concatenative packing.
                        eos_mask = input == self.eos_id
                        if eos_mask.any():
                            label = label.clone()
                            label[eos_mask] = IGNORE_INDEX
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


def _replace_none_with_literal(xs: list[str] | None) -> list[str | None] | None:
    if xs is None:
        xs = None
    else:
        xs = [None if x == "None" else x for x in xs]
    return xs


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

        dataset_name = config.dataset
        dataset_path = _replace_none_with_literal(config.dataset_path)
        dataset_streaming = config.dataset_streaming
        dataset_weights = config.dataset_weights
        dataset_mix_in_seq = config.dataset_mix_in_seq
        dataset_inner_name = _replace_none_with_literal(config.dataset_inner_name)
        dataset_files = config.dataset_files
        dataset_split = config.dataset_split
        dataset_key = config.dataset_key
        infinite = config.infinite

        normed_list_length = len(dataset_name)
        dataset_path = _normalize_list(dataset_path, normed_list_length)
        dataset_inner_name = _normalize_list(dataset_inner_name, normed_list_length)
        dataset_split = _normalize_list(dataset_split, normed_list_length)
        dataset_key = _normalize_list(dataset_key, normed_list_length)
        dataset_weights = (
            [1.0] * normed_list_length
            if dataset_weights is None
            # Convert to floats.
            else list(map(float, dataset_weights))
        )
        drop_long_samples = config.drop_long_samples

        if len(dataset_name) > 1:
            assert (
                dataset_files is None
            ), "cannot supply dataset files when using multiple datasets"
        for d in [
            dataset_path,
            dataset_inner_name,
            dataset_split,
            dataset_key,
            dataset_weights,
        ]:
            assert (
                len(d) == normed_list_length
            ), f"list {d} does not match length of list of datasets (length = {normed_list_length})"
        hf_datasets = []
        for d_name, d_path, d_inner_name, d_split, d_key in zip(
            dataset_name,
            dataset_path,
            dataset_inner_name,
            dataset_split,
            dataset_key,
        ):
            hf_ds = HuggingFaceDataset(
                dataset_name=d_name,
                dataset_path=d_path,
                tokenizer=tokenizer,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=infinite,
                dataset_inner_name=d_inner_name,
                dataset_files=dataset_files,
                dataset_split=d_split,
                dataset_streaming=dataset_streaming,
                dataset_key=d_key,
            )
            if not dataset_mix_in_seq:
                hf_ds = GreedyPackedDataset(
                    dataset=hf_ds,
                    seq_len=seq_len,
                    infinite=infinite,
                    drop_long_samples=drop_long_samples,
                    eos_id=tokenizer.eos_id,
                )
            hf_datasets.append(hf_ds)

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
        )

        if dataset_mix_in_seq:
            hf_ds = GreedyPackedDataset(
                dataset=hf_ds,
                seq_len=seq_len,
                infinite=infinite,
                drop_long_samples=drop_long_samples,
                eos_id=tokenizer.eos_id,
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
