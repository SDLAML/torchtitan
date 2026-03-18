# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import re

import torch

from torchtitan.components.dataloader import BaseDataLoader

__all__ = [
    "DataMixScheduler",
]


PERCENT_STEP_KEY_PATTERN = re.compile(r"^@([0-9]+(?:\.[0-9]+)?)%$")


class DataMixScheduler:
    """
    Lets make this stateless for now, to make the change of data mix easier.

    """

    def __init__(
        self,
        dataloader,
        mixing_configs,
        datasets_names,
    ):
        self.dataloader = dataloader
        self._mixed_dataset = _resolve_mixed_dataset(dataloader.dataset)
        self.mixing_configs = mixing_configs
        self.datasets_names = datasets_names
        self.step_milestones = sorted(mixing_configs.keys(), reverse=True)

    def get_weights_at_step(self, current_step: int):
        for step in self.step_milestones:
            if current_step >= step:
                return self.mixing_configs[step]
        # In theory this should never happen since we assume
        # there is at least a step 0, but just in case:
        # fall back to the earliest config
        first_step = self.step_milestones[0]
        return self.mixing_configs[first_step]

    def convert_mixing_configs_to_json(self):
        configs_dict = {}
        for key, value in self.mixing_configs.items():
            if key == "names":
                configs_dict[key] = value
                continue
            if isinstance(value, torch.Tensor):
                configs_dict[key] = value.cpu().tolist()
            elif isinstance(value, list) or isinstance(value, tuple):
                configs_dict[key] = value
            elif isinstance(value, int) or isinstance(value, float):
                configs_dict[key] = [value]
            else:
                raise ValueError(f"Unsupported type: {type(value)}")

        return configs_dict

    def get_log_dict_at_step(self, current_step: int):
        all_weights = self.get_weights_at_step(current_step)
        data_mix_log, data_sampled_log = {}, {}
        for data_i in range(len(self.datasets_names)):
            data_mix_log[f"data_mixing/{self.datasets_names[data_i]}"] = all_weights[
                data_i
            ]
            data_sampled_log[
                f"data_sampled/{self.datasets_names[data_i]}"
            ] = self._mixed_dataset.num_tokens_per_dataset[data_i]
        return data_mix_log, data_sampled_log

    def step(self, current_step: int):
        current_weights = self.get_weights_at_step(current_step)
        self._mixed_dataset.set_weights(current_weights)

    def dump_mixing_configs(self, dump_folder: str):
        if torch.distributed.get_rank() == 0:
            # Save model args to dump folder.
            os.makedirs(dump_folder, exist_ok=True)
            data_mix_scheduler_save_path = os.path.join(
                dump_folder,
                "data_mix_scheduler_"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M")
                + ".json",
            )
            with open(data_mix_scheduler_save_path, "w") as f:
                json.dump(
                    self.convert_mixing_configs_to_json(),
                    f,
                    indent=4,
                )


class DummyDataMixScheduler:
    def __init__(self):
        self.mixing_configs = {0: [1]}

    def get_weights_at_step(self, current_step: int):
        return [1]

    def get_log_dict_at_step(self, current_step: int):
        data_mix_log = {"data_mixing/not_mixed_datasets": torch.tensor(1)}
        data_sampled_log = {"data_sampled/not_mixed_datasets": torch.tensor(0)}
        return data_mix_log, data_sampled_log

    def convert_mixing_configs_to_json(self):
        return {"0": [1]}

    def step(self, current_step: int):
        pass

    def dump_mixing_configs(self, dump_folder: str):
        pass


def _parse_percentage_step_key(step_key: str) -> float | None:
    match = PERCENT_STEP_KEY_PATTERN.fullmatch(step_key)
    if match is None:
        return None
    return float(match.group(1))


def _load_mixing_configs(mixing_scheduler_configs: str, training_steps: int):
    with open(mixing_scheduler_configs) as config_file:
        mixing_configs = json.load(config_file)
    datasets_names = mixing_configs.pop("names", None)
    if not mixing_configs:
        raise ValueError("mixing_configs must contain at least one milestone entry")

    if all(isinstance(key, str) and key.isdigit() for key in mixing_configs):
        parsed_mixing_configs = {
            int(key): value for key, value in mixing_configs.items()
        }
        return parsed_mixing_configs, datasets_names

    if not all(
        isinstance(key, str) and _parse_percentage_step_key(key) is not None
        for key in mixing_configs
    ):
        raise ValueError(
            "mixing_configs milestone keys must be either non-negative integer strings "
            'like "500" or percentage strings like "@10%"'
        )

    if training_steps < 0:
        raise ValueError(
            "training_steps must be non-negative when using percentage milestones"
        )

    rendered_mixing_configs = {}
    saw_zero_percent = False
    for key, value in mixing_configs.items():
        percent = _parse_percentage_step_key(key)
        if percent > 100:
            raise ValueError(
                f"Percentage milestone {key!r} must be less than or equal to 100%"
            )

        rendered_step = int(training_steps * percent / 100)
        if rendered_step in rendered_mixing_configs:
            raise ValueError(
                f"Percentage milestone {key!r} collides with an existing rendered step "
                f"{rendered_step}"
            )
        rendered_mixing_configs[rendered_step] = value
        saw_zero_percent = saw_zero_percent or percent == 0

    if not saw_zero_percent:
        raise ValueError("Percentage-based mixing_configs must contain an '@0%' entry")

    return rendered_mixing_configs, datasets_names


def build_data_mix_scheduler(
    dataloader: BaseDataLoader,
    mixing_scheduler_configs: str | None,
    training_steps: int,
):
    mixed_dataset = _resolve_mixed_dataset(dataloader.dataset)
    if mixed_dataset is None:
        return DummyDataMixScheduler()
    mixing_configs, datasets_names = None, None
    if mixing_scheduler_configs:
        if os.path.isfile(mixing_scheduler_configs):
            mixing_configs, datasets_names = _load_mixing_configs(
                mixing_scheduler_configs, training_steps
            )

    """
    mixing_configs should be organized like:
    {
        0: [weights_for_dataset_0, weights_for_dataset_1, ...],
        500: [weights_for_dataset_0, weights_for_dataset_1, ...],
        step: [weights_for_dataset_0, weights_for_dataset_1, ...],
    }
    or:
    {
        "@0%": [weights_for_dataset_0, weights_for_dataset_1, ...],
        "@10%": [weights_for_dataset_0, weights_for_dataset_1, ...],
    }
    """

    if mixing_configs is None:
        mixing_configs = {
            0: mixed_dataset.weights.tolist(),
        }

    if datasets_names is None:
        datasets_names = [str(i) for i in range(len(mixed_dataset.datasets))]
    elif isinstance(datasets_names, str):
        datasets_names = [datasets_names]
    if len(datasets_names) != len(mixed_dataset.datasets):
        raise ValueError(
            f"datasets_names must have the same length as datasets get len(datasets) = "
            f"{len(mixed_dataset.datasets)} and len(datasets_names) = "
            f"{len(datasets_names)} but got datasets_names = {datasets_names}"
        )
    assert (
        0 in mixing_configs
    ), "mixing_configs must contain at least one entry for step 0"

    for step, weights in mixing_configs.items():
        assert len(weights) == len(mixed_dataset.datasets), (
            f"weights must have the same length as datasets get len(datasets) = "
            f"{len(mixed_dataset.datasets)} and len(weights) = "
            f"{len(weights)}"
        )

    return DataMixScheduler(dataloader, mixing_configs, datasets_names)


def _resolve_mixed_dataset(dataset):
    current = dataset
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "weights") and hasattr(current, "datasets"):
            return current
        current = getattr(current, "_data", None)

    return None
