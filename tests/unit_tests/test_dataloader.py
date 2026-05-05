# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from random import Random
from types import SimpleNamespace

from torch.utils.data import IterableDataset

from torchtitan.components.data_mix_scheduler import build_data_mix_scheduler

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    HuggingFaceTextDataLoader,
    MixedDataset,
    TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV,
    TORCHTITAN_MIX_STATE_MAPPING_ENV,
)


class DummyDataset(IterableDataset):
    """A simple dummy dataset for testing."""

    def __iter__(self):
        for i in range(100):
            yield {"input": i}, i


class DummyTokenizer(BaseTokenizer):
    """A dummy tokenizer for testing that implements BaseTokenizer interface."""

    def __init__(self):
        super().__init__()
        self.eos_id = 2

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        # Simple encoding: convert each character to its ASCII value
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, 1)  # BOS token
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        # Simple decoding: convert ASCII values back to characters
        return "".join(chr(t) for t in token_ids if t > 2)

    def get_vocab_size(self) -> int:
        return 256  # ASCII range


class DummyMixedComponent(IterableDataset):
    def __init__(self, name: str):
        self.dataset_name = name
        self.dataset_path = f"/tmp/{name}"

    def __iter__(self):
        yield [1, 2, 3]


class StatefulDummyMixedComponent(DummyMixedComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict

    def state_dict(self):
        return {"name": self.dataset_name}


class TestParallelAwareDataloader(unittest.TestCase):
    def test_dataloader_yields_correct_batches(self):
        """Test that the dataloader correctly yields batched data from the dataset."""
        dataset = DummyDataset()
        batch_size = 4

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
        )

        batches = list(dataloader)

        # DummyDataset yields 100 items, so we expect 25 batches of size 4
        self.assertEqual(len(batches), 25)

        # Check first batch structure and values
        first_batch_input, first_batch_label = batches[0]
        self.assertEqual(len(first_batch_input["input"]), batch_size)
        self.assertEqual(len(first_batch_label), batch_size)

        # Verify first batch contains expected values (0, 1, 2, 3)
        self.assertEqual(first_batch_input["input"].tolist(), [0, 1, 2, 3])
        self.assertEqual(first_batch_label.tolist(), [0, 1, 2, 3])

        # Check last batch
        last_batch_input, last_batch_label = batches[-1]
        self.assertEqual(last_batch_input["input"].tolist(), [96, 97, 98, 99])
        self.assertEqual(last_batch_label.tolist(), [96, 97, 98, 99])

    def test_validate_kwargs_rejects_invalid_kwargs(self):
        """Test that passing invalid kwargs raises ValueError."""
        dataset = DummyDataset()

        with self.assertRaises(ValueError) as context:
            ParallelAwareDataloader(
                dataset,
                dp_rank=0,
                dp_world_size=1,
                invalid_arg=42,
            )

        self.assertIn("Invalid dataloader kwargs", str(context.exception))
        self.assertIn("invalid_arg", str(context.exception))

    def test_config_batch_size_overwritten_by_explicit_batch_size(self):
        """Test that batch_size in config kwargs is overwritten by explicit batch_size."""
        dataset = DummyDataset()

        config_kwargs = {"batch_size": 2, "num_workers": 0}

        explicit_batch_size = 8

        # Merge kwargs with explicit args taking precedence (same pattern as in dataset files)
        dataloader_kwargs = {
            **config_kwargs,
            "batch_size": explicit_batch_size,
        }

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            **dataloader_kwargs,
        )

        # Verify that batch_size is the explicit one, not the config one
        self.assertEqual(dataloader.batch_size, explicit_batch_size)

    def test_build_dataloader_with_trainer_config(self):
        """Verify batch_size from training.local_batch_size is correctly used."""
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=2,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=512,
            local_batch_size=8,
        )

        self.assertEqual(dataloader.batch_size, 8)
        self.assertEqual(dataloader.num_workers, 2)

    def test_single_dataset_scalar_alias_is_normalized(self):
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            dataset_alias="wiki",
            num_workers=0,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=32,
            local_batch_size=1,
        )

        self.assertEqual(dataloader.dataset.dataset_aliases, ["wiki"])

    def test_dataset_alias_none_entries_fall_back_to_index(self):
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset=["c4_test", "c4_test"],
            dataset_alias=["wiki", None],
            num_workers=0,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=32,
            local_batch_size=1,
        )

        self.assertEqual(dataloader.dataset.dataset_aliases, ["wiki", "1"])

    def test_dataset_alias_length_mismatch_raises(self):
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset=["c4_test", "c4_test"],
            dataset_alias=["wiki"],
            num_workers=0,
        )

        with self.assertRaises(AssertionError):
            HuggingFaceTextDataLoader(
                dl_config,
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tokenizer,
                seq_len=32,
                local_batch_size=1,
            )


class TestMixedDatasetResumeOverrides(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV, None)
        os.environ.pop(TORCHTITAN_MIX_STATE_MAPPING_ENV, None)

    def _build_checkpoint_state(self, num_datasets=3):
        return {
            "sample_idx": 7,
            "weights": [float(i + 1) for i in range(num_datasets)],
            "removed": [i == 1 for i in range(num_datasets)],
            "num_docs_sampled": [10 * (i + 1) for i in range(num_datasets)],
            "num_tokens_sampled": [100 * (i + 1) for i in range(num_datasets)],
            "datasets": [{"checkpoint_dataset": i} for i in range(num_datasets)],
            "rng_state": Random(123).getstate(),
        }

    def _build_mixed_dataset(self, names, weights=None):
        if weights is None:
            weights = [1.0 for _ in names]
        return MixedDataset(
            datasets=[StatefulDummyMixedComponent(name) for name in names],
            dp_rank=0,
            weights=weights,
        )

    def test_load_state_dict_defaults_to_checkpoint_order(self):
        mixed_dataset = self._build_mixed_dataset(["a", "b"], weights=[0.5, 0.5])

        mixed_dataset.load_state_dict(self._build_checkpoint_state(num_datasets=2))

        self.assertEqual(mixed_dataset._sample_idx, 7)
        self.assertEqual(mixed_dataset.weights.tolist(), [1.0, 0.0])
        self.assertEqual(mixed_dataset.removed.tolist(), [False, True])
        self.assertEqual(mixed_dataset.num_docs_sampled.tolist(), [10, 20])
        self.assertEqual(mixed_dataset.num_tokens_sampled.tolist(), [100, 200])
        self.assertEqual(
            [dataset.loaded_state for dataset in mixed_dataset.datasets],
            [{"checkpoint_dataset": 0}, {"checkpoint_dataset": 1}],
        )

    def test_skip_checkpoint_weights_preserves_configured_weights(self):
        os.environ[TORCHTITAN_MIX_SKIP_CKPT_WEIGHTS_ENV] = "1"
        mixed_dataset = self._build_mixed_dataset(["a", "b"], weights=[0.8, 0.2])

        mixed_dataset.load_state_dict(self._build_checkpoint_state(num_datasets=2))

        self.assertEqual(mixed_dataset.weights.tolist(), [0.8, 0.0])
        self.assertEqual(mixed_dataset.removed.tolist(), [False, True])

    def test_state_mapping_loads_only_mapped_dataset_states(self):
        os.environ[TORCHTITAN_MIX_STATE_MAPPING_ENV] = "0-0|1-2"
        mixed_dataset = self._build_mixed_dataset(
            ["a", "c", "d"], weights=[0.4, 0.5, 0.6]
        )

        mixed_dataset.load_state_dict(self._build_checkpoint_state(num_datasets=3))

        self.assertEqual(mixed_dataset.weights.tolist(), [1.0, 3.0, 0.6])
        self.assertEqual(mixed_dataset.removed.tolist(), [False, False, False])
        self.assertEqual(mixed_dataset.num_docs_sampled.tolist(), [10, 30, 0])
        self.assertEqual(mixed_dataset.num_tokens_sampled.tolist(), [100, 300, 0])
        self.assertEqual(
            [dataset.loaded_state for dataset in mixed_dataset.datasets],
            [{"checkpoint_dataset": 0}, {"checkpoint_dataset": 2}, None],
        )

    def test_state_mapping_rejects_invalid_rules(self):
        invalid_mappings = [
            "not-a-rule",
            "0-0|0-1",
            "0-1|1-1",
            "0-5",
        ]
        for mapping in invalid_mappings:
            with self.subTest(mapping=mapping):
                os.environ[TORCHTITAN_MIX_STATE_MAPPING_ENV] = mapping
                mixed_dataset = self._build_mixed_dataset(["a", "b"])
                with self.assertRaises(ValueError):
                    mixed_dataset.load_state_dict(
                        self._build_checkpoint_state(num_datasets=2)
                    )


class TestDataMixSchedulerDatasetNames(unittest.TestCase):
    def _build_scheduler(self, *, dataset_aliases=None, scheduler_names=None):
        mixed_dataset = MixedDataset(
            datasets=[DummyMixedComponent("a"), DummyMixedComponent("b")],
            dp_rank=0,
            weights=[0.7, 0.3],
            dataset_aliases=dataset_aliases,
        )
        dataloader = SimpleNamespace(dataset=mixed_dataset)

        if scheduler_names is None:
            return build_data_mix_scheduler(dataloader, None, training_steps=100)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(
                {
                    "names": scheduler_names,
                    "@0%": [0.7, 0.3],
                },
                tmp,
            )
            tmp_path = tmp.name

        self.addCleanup(lambda: os.unlink(tmp_path))
        return build_data_mix_scheduler(dataloader, tmp_path, training_steps=100)

    def test_default_scheduler_names_use_indices(self):
        scheduler = self._build_scheduler()

        self.assertEqual(scheduler.datasets_names, ["0", "1"])

    def test_dataset_aliases_are_used_when_scheduler_names_absent(self):
        scheduler = self._build_scheduler(dataset_aliases=["wiki", "code"])

        self.assertEqual(scheduler.datasets_names, ["wiki", "code"])

    def test_scheduler_names_override_dataset_aliases(self):
        scheduler = self._build_scheduler(
            dataset_aliases=["wiki", "code"],
            scheduler_names=["sched_wiki", "sched_code"],
        )

        self.assertEqual(scheduler.datasets_names, ["sched_wiki", "sched_code"])

    def test_scheduler_name_none_falls_back_per_index(self):
        scheduler = self._build_scheduler(
            dataset_aliases=["wiki", "code"],
            scheduler_names=["sched_wiki", None],
        )

        self.assertEqual(scheduler.datasets_names, ["sched_wiki", "code"])

    def test_scheduler_names_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._build_scheduler(
                dataset_aliases=["wiki", "code"],
                scheduler_names=["sched_only"],
            )


if __name__ == "__main__":
    unittest.main()
