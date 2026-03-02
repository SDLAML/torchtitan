# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig

from torchtitan.experiments.sft.auto_tokenizer import HuggingFaceAutoTokenizer
from torchtitan.experiments.sft.configs import SFTTrainerConfig
from torchtitan.experiments.sft.sft_text_datasets import SFTDataLoader
from torchtitan.models.opt_moe import model_registry


def sft_opt_moe_proxy_multiturn() -> SFTTrainerConfig:
    """SFT on opt_moe proxy model (bsc-1B-7B-opt-g-proxy) with multi-turn chat data."""
    return SFTTrainerConfig(
        hf_assets_path="./assets/hf/opt_moe",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_registry("bsc-1B-7B-opt-g-proxy"),
        tokenizer=HuggingFaceAutoTokenizer.Config(
            eos_token="<|end_of_text|>",
            pad_token_id=128014,
            pad_token="<|i_am_pad|>",
        ),
        dataloader=SFTDataLoader.Config(
            dataset="multi_turn",
            dataset_path="allenai/Dolci-Think-SFT-7B",
            apply_chat_template=True,
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


def sft_opt_moe_proxy_gsm8k() -> SFTTrainerConfig:
    """SFT on opt_moe proxy model (bsc-1B-7B-opt-g-proxy) with GSM8K question-answer data."""
    return SFTTrainerConfig(
        hf_assets_path="./assets/hf/opt_moe",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_registry("bsc-1B-7B-opt-g-proxy"),
        tokenizer=HuggingFaceAutoTokenizer.Config(
            eos_token="<|end_of_text|>",
            pad_token_id=128014,
            pad_token="<|i_am_pad|>",
        ),
        dataloader=SFTDataLoader.Config(
            dataset="question_answer",
            dataset_path="openai/gsm8k",
            dataset_subset="main",
            apply_chat_template=False,
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


def sft_opt_moe_full_multiturn() -> SFTTrainerConfig:
    """SFT on opt_moe full model (bsc-1B-7B-opt-g) with multi-turn chat data."""
    return SFTTrainerConfig(
        hf_assets_path="./assets/hf/opt_moe",
        metrics=MetricsProcessor.Config(log_freq=10),
        model_spec=model_registry("bsc-1B-7B-opt-g"),
        tokenizer=HuggingFaceAutoTokenizer.Config(
            eos_token="<|end_of_text|>",
            pad_token_id=128014,
            pad_token="<|i_am_pad|>",
        ),
        dataloader=SFTDataLoader.Config(
            dataset="multi_turn",
            dataset_path="allenai/Dolci-Think-SFT-7B",
            apply_chat_template=True,
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
