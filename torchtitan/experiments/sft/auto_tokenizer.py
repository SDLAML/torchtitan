# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger
from transformers import AutoTokenizer as HF_AutoTokenizer


class HuggingFaceAutoTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        eos_token: str = ""
        """EOS token string."""
        pad_token_id: int = -1
        """PAD token ID."""
        pad_token: str | None = None
        """PAD token string override. If None, inferred from pad_token_id."""

    def __init__(
        self,
        config: Config,
        *,
        tokenizer_path: str,
    ):
        super().__init__()
        eos_token = config.eos_token
        pad_token_id = config.pad_token_id
        pad_token = config.pad_token

        self.tokenizer = HF_AutoTokenizer.from_pretrained(
            tokenizer_path, eos_token=eos_token, use_fast=True
        )
        self.tokenizer_path = tokenizer_path
        self.chat_template = self.tokenizer.chat_template

        self.vocab_size = len(self.tokenizer)

        assert (
            pad_token_id < self.vocab_size
        ), f"PAD token ID is out of range: {pad_token_id} >= {self.vocab_size}"
        assert (
            pad_token_id != self.tokenizer.eos_token_id
        ), "PAD token ID is the same as EOS token ID, this can cause problems with varlen/flex attention in dynamic packing."

        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

        self.pad_id = pad_token_id

        self.maybe_original_token_at_pad_id = self.tokenizer._convert_id_to_token(
            pad_token_id
        )
        if pad_token is not None:
            self.pad_token = pad_token
        else:
            self.pad_token = self.maybe_original_token_at_pad_id

        logger.info(
            f"[SFT AutoTokenizer] Using EOS token: {self.eos_token} - EOS ID: {self.eos_id} "
            f"[SFT AutoTokenizer] Using PAD token: {self.pad_token} - PAD ID: {self.pad_id}"
        )

    def apply_chat_template(self, messages: list[dict], **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, text: str, *args, **kwargs):
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, tokens: list[int], *args, **kwargs):
        decoded = self.tokenizer.decode(tokens, *args, **kwargs)
        # this is an ad-hoc for debugging purpose
        if self.pad_id in tokens:
            decoded = decoded.replace(
                self.maybe_original_token_at_pad_id, self.pad_token
            )
        return decoded

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __call__(self, text: str, *args, **kwargs):
        return self.tokenizer(text, *args, **kwargs)
