# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.loss import MoEAuxLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.optimizers.container import OptimizersContainer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.mixed_text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer
from . import model_registry


def moe_template_config() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/gpt-oss-120b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("bsc-1B-7B-opt-g"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="simple_custom",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        # OPT MoE returns (logits, load_balance_loss); this unpacks the tuple and
        # adds the auxiliary term straight-through.
        loss=MoEAuxLoss.Config(),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        # The validator defaults to the grain loader; point it at the same
        # torchdata-backed loader the training path uses so held-out configs
        # can set the same dataset_* fields.
        validator=Validator.Config(
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset="simple_custom",
                pack_strategy="best_fit",
            ),
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        # Upstream replaced the mode/selective_ac_option pair with a config
        # class per policy; per-op SAC is now SelectiveAC.Config.
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        # scripts/checkpoint_conversion/convert_to_hf.py reconstructs the model
        # spec from this file, so opt_moe runs always write it.
        debug=DebugConfig(save_config_file="job_config.json"),
    )
