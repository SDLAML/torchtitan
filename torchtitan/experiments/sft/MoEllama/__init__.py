# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.sft.auto_tokenizer import build_auto_tokenizer

from torchtitan.experiments.sft.sft_text_datasets import (
    build_sft_text_dataloader,
    build_sft_validation_dataloader,
)

from torchtitan.models.MoEllama import moe_llama_configs, Transformer
from torchtitan.models.MoEllama.hf_assests import setup_hf
from torchtitan.models.MoEllama.infra.parallelize import parallelize_llama
from torchtitan.models.MoEllama.model.state_dict_adapter import MoEllamaStateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=moe_llama_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_sft_text_dataloader,
        build_tokenizer_fn=build_auto_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_sft_validation_dataloader,
        state_dict_adapter=MoEllamaStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    )
