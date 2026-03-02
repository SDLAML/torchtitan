# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.sft.auto_tokenizer import HuggingFaceAutoTokenizer
from torchtitan.experiments.sft.sft_text_datasets import SFTDataLoader
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class SFTTrainerConfig(Trainer.Config):
    dataloader: SFTDataLoader.Config = field(
        default_factory=SFTDataLoader.Config
    )
    tokenizer: HuggingFaceAutoTokenizer.Config = field(
        default_factory=HuggingFaceAutoTokenizer.Config
    )
