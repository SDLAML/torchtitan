# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)

from torchtitan.experiments.sft.auto_tokenizer import HuggingFaceAutoTokenizer
from torchtitan.experiments.sft.configs import SFTTrainerConfig

# Re-export existing config factories
from torchtitan.experiments.sft.opt_moe import (  # noqa: F401
    sft_opt_moe_full_multiturn,
    sft_opt_moe_proxy_gsm8k,
    sft_opt_moe_proxy_multiturn,
)
from torchtitan.experiments.sft.sft_text_datasets import SFTDataLoader
from torchtitan.models.opt_moe import model_registry


def berliner_sft_multiturn() -> SFTTrainerConfig:
    """SFT on opt_moe proxy model with Berliner-SFT chat template and multi-turn data."""
    return SFTTrainerConfig(
        hf_assets_path="/e/project1/trustllm-eu/wang55/torchtitan_assets/berliner-sft",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_registry("bsc-1B-7B-opt-g-proxy"),
        tokenizer=HuggingFaceAutoTokenizer.Config(
            eos_token="<|endoftext|>",
            pad_token_id=200008,
            pad_token="<|pad|>",
        ),
        dataloader=SFTDataLoader.Config(
            dataset="berliner_sft",
            apply_chat_template=True,
            chat_template_kwargs={
                "truncate_history_thinking": False,
            },
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=1000,
            max_norm=1.0,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            folder="checkpoint",
            interval=500,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )
