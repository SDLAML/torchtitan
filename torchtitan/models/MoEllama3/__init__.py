# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is used for backward compatibility with the old MoEllama3 model for repo <=0.3.0
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.models.MoEllama import (
    build_lr_schedulers,
    build_optimizers_with_moe_load_balancing,
    build_text_dataloader,
    moe_llama_configs,
    parallelize_llama,
    pipeline_llm,
    Transformer,
)
from torchtitan.models.MoEllama.hf_assests import setup_hf
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
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=MoEllamaStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    )
