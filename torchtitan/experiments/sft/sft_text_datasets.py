# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# code are heavily borrowed from
# [1] https://github.com/volcengine/verl/blob/main/verl/utils/dataset/multiturn_sft_dataset.py
# [2] https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L35
# [3] https://github.com/volcengine/verl/blob/main/verl/utils/dataset/sft_dataset.py#L33
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


def _build_multi_turn_messages_from_row_dict(
    row_dict: dict,
    messages_key: str = "messages",
    tools_key: str = "tools",
    thinking_key: str = "thinking",
):
    """Build multi-turn messages from row dictionary."""
    # return the message, tools, and enable_thinking
    message = row_dict[messages_key]
    tools = row_dict.get(tools_key, None)
    enable_thinking = row_dict.get(thinking_key, None)
    if isinstance(enable_thinking, str):
        enable_thinking = enable_thinking == "on"
    return message, tools, enable_thinking


def _build_prompt_response_messages_from_row_dict(
    row_dict: dict,
    prompt_key: str = "prompt",
    response_key: str = "response",
):
    """Build one turn messages from row dictionary."""
    # return the message, tools, and enable_thinking
    message = [
        {"role": "user", "content": row_dict[prompt_key]},
        {"role": "assistant", "content": row_dict[response_key]},
    ]
    return message, None, None


def _build_dolci_instruct_sft_messages_from_row_dict(
    row_dict: dict,
    messages_key: str = "messages",
    tools_key: str = "functions",
    **kwargs,
):
    """Build Dolci-Instruct-SFT messages from row dictionary."""
    messages = row_dict[messages_key]
    for msg in messages:
        if msg.get("content") is None:
            msg["content"] = ""
    tools = messages[0].get(tools_key, None)

    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            # Handle cases where the string is malformed or empty
            logger.warning(f"Failed to parse tools JSON: {tools[:50]}...")
            tools = None
    return messages, tools, False


def _build_berliner_sft_messages_from_row_dict(
    row_dict: dict,
    messages_key: str = "messages",
    tools_key: str = "tools",
    enable_thinking_key: str = "enable_thinking",
):
    """Build messages from Berliner-SFT Parquet row format.

    All columns are stored as JSON strings that must be parsed.
    ``enable_thinking`` is pre-computed: True if any <think> block
    in the messages contains >= 10 chars of non-whitespace text.
    """
    messages = json.loads(row_dict[messages_key])
    # Normalize tool_call arguments so the template's |items filter always gets a dict.
    # Arguments may arrive as: JSON string, already-parsed dict, list, or None.
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function") or tc
                if "arguments" not in func:
                    continue
                args = func["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool_call arguments: {str(args)[:80]}..."
                        )
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                func["arguments"] = args
    tools_str = row_dict.get(tools_key, "[]")
    try:
        tools = json.loads(tools_str) if tools_str else []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tools JSON: {str(tools_str)[:80]}...")
        tools = []
    enable_thinking = bool(row_dict.get(enable_thinking_key, False))
    # the thinking trace is already rendered in the message
    return messages, tools, False


DATASET_MESSAGE_BUILDERS = {
    "multi_turn": _build_multi_turn_messages_from_row_dict,
    "prompt_response": _build_prompt_response_messages_from_row_dict,
    "question_answer": partial(
        _build_prompt_response_messages_from_row_dict,
        prompt_key="question",
        response_key="answer",
    ),
    "Dolci-Instruct-SFT": _build_dolci_instruct_sft_messages_from_row_dict,
    "berliner_sft": _build_berliner_sft_messages_from_row_dict,
}


def extract_system_prompt_and_generation(tokenizer):
    """Derive system and generation prompt token chunks from the tokenizer's chat template."""
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


class SFTDataset(IterableDataset, Stateful):
    """
    Iterable dataset that tokenizes a conversational stream for SFT training.
    """

    def __init__(
        self,
        dataset,
        tokenizer: BaseTokenizer,
        message_builder: Callable,
        sft_config,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self.message_builder = message_builder
        self.infinite = infinite

        self.sft_config = sft_config

        self.pad_mode = sft_config.pad_mode
        self.ignore_input_ids_mismatch = sft_config.ignore_input_ids_mismatch
        self.drop_long_samples = getattr(sft_config, "drop_long_samples", False)
        self.packing_pool_size = getattr(sft_config, "packing_pool_size", 4096)
        self.chat_template_prefix_workers = max(
            1, int(getattr(sft_config, "chat_template_prefix_workers", 4))
        )
        self._chat_template_prefix_executor: ThreadPoolExecutor | None = None

        if self.pad_mode == "best_fit_packing" and not self.drop_long_samples:
            raise ValueError(
                "pad_mode='best_fit_packing' requires drop_long_samples=True. "
                "Best-fit packing cannot truncate or split conversations, so "
                "conversations longer than seq_len+1 must be dropped."
            )

        self.max_length = seq_len
        self.apply_chat_template_kwargs = sft_config.chat_template_kwargs

        # logging chat template kwargs if is not empty
        if self.apply_chat_template_kwargs:
            logger.info(
                f"[sft_text_datasets.py] Chat template kwargs: {self.apply_chat_template_kwargs}"
            )

        self.apply_chat_template = sft_config.apply_chat_template
        if self.apply_chat_template:
            assert self.tokenizer.chat_template is not None, (
                f"Chat template is not set for the tokenizer {self.tokenizer.tokenizer_path}, "
                f"please set it in the tokenizer config file"
            )
            (
                self.system_prompt,
                self.generation_prompt,
            ) = extract_system_prompt_and_generation(self.tokenizer)
        else:
            self.system_prompt = torch.tensor([], dtype=torch.long)
            self.generation_prompt = torch.tensor([], dtype=torch.long)

        logger.info(
            f"[sft_text_datasets.py] Infer system_prompt: {self.tokenizer.decode(self.system_prompt)}"
        )
        logger.info(
            f"[sft_text_datasets.py] Infer generation_prompt: {self.tokenizer.decode(self.generation_prompt)}"
        )

        self.pad_id = self.tokenizer.pad_id
        self.pad_token = self.tokenizer.pad_token
        self.eos_id = self.tokenizer.eos_id

        self.buffer_max_length = self.max_length
        # Stateful variables
        self._sample_idx = 0
        self._buffer = self._reset_buffer()

        # State for best_fit_packing mode
        self._pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._n_dropped_oversized: int = 0
        # Per-rank deterministic RNG for shuffling BFD output. Different ranks
        # get different shuffles; reproducible across resumes via rng_state.
        _seed = getattr(sft_config, "dataset_seed", None) or 0
        self._rng = np.random.default_rng(seed=_seed * 1_000_003 + dp_rank)

        # Per-dataset sampling counters (1-element shared-memory tensors so
        # worker-process increments are visible to the main process when
        # DataMixScheduler/SingleDatasetMixScheduler reads them at log time).
        self.num_docs_sampled = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.num_tokens_sampled = torch.zeros(1, dtype=torch.int64).share_memory_()
        # Pick a dataset_alias for wandb keys. Priority: explicit config →
        # last component of dataset_path → message-builder name → "sft".
        # Using basename avoids ugly "data_docs//abs/path/to/data" keys.
        _explicit_alias = getattr(sft_config, "dataset_alias", None)
        _path = getattr(sft_config, "dataset_path", None)
        _path_basename = os.path.basename(_path.rstrip("/")) if _path else None
        self.dataset_alias = (
            _explicit_alias
            or _path_basename
            or getattr(sft_config, "dataset", None)
            or "sft"
        )

    def _reset_buffer(self):
        """Reset the greedy packing buffer to empty tensors."""
        return {
            "input_ids": [],
            "position_ids": [],
            "labels": [],
            "segment_lens": [],
            "current_len": 0,
        }

    def _get_data_iter(self):
        """Return an iterator over the local partition, resuming for map-style datasets."""
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def load_state_dict(self, state_dict):
        self._buffer = state_dict["buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "dataset" in state_dict
            self._data.load_state_dict(state_dict["dataset"])

        # Restore best_fit_packing state if present (backward compatible:
        # checkpoints written before this field is absent simply leave the
        # in-memory defaults set by __init__).
        if "pool" in state_dict:
            self._pool = list(state_dict["pool"])
        if "n_dropped_oversized" in state_dict:
            self._n_dropped_oversized = int(state_dict["n_dropped_oversized"])
        if "rng_state" in state_dict:
            self._rng.bit_generator.state = state_dict["rng_state"]
        # Sampling counters: copy_ into the existing shared-memory tensors so
        # forked worker processes still see the same storage.
        if "num_docs_sampled" in state_dict:
            self.num_docs_sampled.copy_(state_dict["num_docs_sampled"])
        if "num_tokens_sampled" in state_dict:
            self.num_tokens_sampled.copy_(state_dict["num_tokens_sampled"])

    def state_dict(self):
        _state_dict = {"buffer": self._buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["dataset"] = self._data.state_dict()

        # best_fit_packing state. Always persisted (cheap when empty); only
        # used by the best_fit_packing branch on load.
        _state_dict["pool"] = self._pool
        _state_dict["n_dropped_oversized"] = self._n_dropped_oversized
        _state_dict["rng_state"] = self._rng.bit_generator.state

        # Sampling counters (clone to detach from shared memory for the
        # serialized snapshot).
        _state_dict["num_docs_sampled"] = self.num_docs_sampled.clone()
        _state_dict["num_tokens_sampled"] = self.num_tokens_sampled.clone()

        return _state_dict

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Tokenize one conversation turn while applying template overrides."""
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        if self.apply_chat_template:
            # logger.info(f"[sft_text_datasets.py] Applying chat template to message: {message}")
            inputs = self.tokenizer.apply_chat_template(
                [message],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_attention_mask=False,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
            inputs = dict(inputs)
            input_ids = inputs.pop("input_ids")[0]
        else:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    [item["text"] for item in content if item["type"] == "text"]
                )
            enc = self.tokenizer(
                content,
                add_special_tokens=False,
                return_tensors="pt",
                return_attention_mask=False,
            )
            input_ids = enc["input_ids"][0]

            if message["role"] == "assistant":
                input_ids = torch.cat([input_ids, input_ids.new_tensor([self.eos_id])])

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(input_ids)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(input_ids)

        return input_ids, loss_mask

    def _process_with_single_shot(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        enable_thinking: bool | None,
        drop_if_longer_than: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-shot tokenization with per-assistant-turn span detection for loss masking.

        Makes 1 + 2k template calls (k = number of assistant turns) instead of n+1 for
        per-turn or O(n²) for prefix-diff. input_ids matches the single-shot output
        exactly so sanity_check is always satisfied.

        If drop_if_longer_than is set and len(input_ids) > drop_if_longer_than, returns
        (input_ids, None) immediately — skipping the 2k prefix calls so no work is
        wasted on conversations that will be dropped anyway.
        """
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        def _tok(msgs):
            out = self.tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_attention_mask=False,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
            return dict(out)["input_ids"][0]

        # 1 call: correct input_ids guaranteed to match single-shot output
        input_ids = _tok(messages)

        # Early exit for oversized conversations: skip the 2k prefix calls entirely.
        if drop_if_longer_than is not None and len(input_ids) > drop_if_longer_than:
            return input_ids, None

        loss_mask = torch.zeros_like(input_ids)

        gen_len = len(self.generation_prompt)

        # 2 calls per assistant turn to locate its token span in input_ids.
        # Requires chat_template_kwargs to include truncate_history_thinking=False so that
        # prefix calls (_tok(messages[:i])) render identically to the full conversation call.
        assistant_indices = [
            i for i, message in enumerate(messages) if message["role"] == "assistant"
        ]
        if self.chat_template_prefix_workers <= 1 or len(assistant_indices) < 2:
            for i in assistant_indices:
                prefix_len = len(_tok(messages[:i])) if i > 0 else 0
                turn_end = len(_tok(messages[: i + 1]))
                # mask generation-prompt header; train on the rest of the assistant turn
                loss_mask[prefix_len + gen_len : turn_end] = 1
        else:
            executor = self._get_chat_template_prefix_executor()
            span_futures = {}
            for i in assistant_indices:
                prefix_future = (
                    executor.submit(_tok, messages[:i]) if i > 0 else None
                )
                end_future = executor.submit(_tok, messages[: i + 1])
                span_futures[i] = (prefix_future, end_future)

            for i in assistant_indices:
                prefix_future, end_future = span_futures[i]
                prefix_len = len(prefix_future.result()) if prefix_future else 0
                turn_end = len(end_future.result())
                loss_mask[prefix_len + gen_len : turn_end] = 1

        return input_ids, loss_mask

    def _get_chat_template_prefix_executor(self) -> ThreadPoolExecutor:
        """Return the lazy per-worker pool used for prefix chat-template calls."""
        if self._chat_template_prefix_executor is None:
            self._chat_template_prefix_executor = ThreadPoolExecutor(
                max_workers=self.chat_template_prefix_workers,
                thread_name_prefix="sft-chat-template-prefix",
            )
        return self._chat_template_prefix_executor

    def sanity_check(
        self,
        input_ids: torch.Tensor,
        messages: list[dict],
        tools: list[dict],
        enable_thinking: bool,
    ):
        """Ensure concatenated per-turn templates match a single-shot template invocation."""
        if not self.apply_chat_template:
            return
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            chat_template_decode = self.tokenizer.decode(
                input_ids, skip_special_tokens=False
            )
            manual_decode = self.tokenizer.decode(
                inputs["input_ids"].squeeze(0), skip_special_tokens=False
            )
            GREEN = "\033[92m"
            CYAN = "\033[96m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"

            def one_line(x):
                return str(x).replace("\n", "\\n").replace("\r", "\\r")

            logger.info(f"{YELLOW}raw: {one_line(messages)}{RESET}")
            logger.info(
                f"{GREEN}chat_template_decode: {one_line(chat_template_decode)}{RESET}"
            )
            logger.info(f"{CYAN}manual_decode: {one_line(manual_decode)}{RESET}")

            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)

    def _process_one_row(self, row_dict: dict):
        """Convert a dataset row into model-ready tensors with causal labels."""
        messages, tools, enable_thinking = self.message_builder(row_dict)

        # tokenize each message
        if self.apply_chat_template:
            # Single-shot: input_ids IS the template output, sanity_check trivially passes.
            # Pass drop_if_longer_than so oversized conversations skip the expensive 2k
            # prefix calls and return (input_ids, None) — caught below.
            drop_threshold = self.max_length + 1 if self.drop_long_samples else None
            input_ids, loss_mask = self._process_with_single_shot(
                messages, tools, enable_thinking, drop_if_longer_than=drop_threshold
            )
            if loss_mask is None:
                return None
        else:
            input_ids, loss_mask = [], []
            for i, message in enumerate(messages):
                _input_ids, _loss_mask = self._process_single_message(
                    index=i,
                    message=message,
                    # here we assume the definition of tools is given only in system message.
                    tools=tools if i == 0 else None,
                    enable_thinking=enable_thinking,
                )
                input_ids.append(_input_ids)
                loss_mask.append(_loss_mask)
            input_ids = torch.cat(input_ids, dim=0)
            loss_mask = torch.cat(loss_mask, dim=0)

        # when chat template is applied, append the EOS token to the input_ids and loss_mask
        # but only append EOS if the last token is not EOS
        if self.apply_chat_template and input_ids[-1].item() != self.eos_id:
            # otherwise, we append a [no-gradient] EOS token to make FlexAttn/VarlenAttn work
            # if the last token is already EOS, we do nothing
            # this path potentially needs add <im_end> to Stop Criteria for inference
            # and needs <im_end> to be different from EOS token
            input_ids = torch.cat(
                [input_ids, input_ids.new_tensor([self.eos_id])], dim=0
            )
            loss_mask = torch.cat([loss_mask, loss_mask.new_tensor([0])], dim=0)

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)
        # comment out these two lines to log the actual text for debugging purpose
        # actaul_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        # logger.info(f"actual_text: {actaul_text} ||-> last mask : {loss_mask[-3:]}")

        # handle padding
        sequence_length = input_ids.shape[0]
        target_length = self.max_length + 1

        # Drop oversized conversations when drop_long_samples=True. This is the
        # only legal behavior for best_fit_packing (enforced at init); the other
        # modes opt into it via the flag.
        if sequence_length > target_length and self.drop_long_samples:
            return None

        # Count of real (non-pad, post-shift) tokens the model will see from
        # this conversation. Computed BEFORE any right_padding inflation so
        # num_tokens_sampled stays accurate for all pad modes.
        real_token_count = min(sequence_length - 1, self.max_length)

        # Calculate valid length (unpadded) of the sequence for the model
        # Note: We slice input_ids[:-1] later, so the valid length for training is len - 1
        # If truncated, it is target_length - 1
        if self.pad_mode == "right_padding":
            if sequence_length < target_length:
                # Pad sequences
                pad_token_id = self.pad_id
                padded_input_ids = torch.full(
                    (target_length - sequence_length,),
                    pad_token_id,
                    dtype=input_ids.dtype,
                )
                padded_loss_mask = torch.zeros(
                    (target_length - sequence_length,), dtype=loss_mask.dtype
                )

                input_ids = torch.cat((input_ids, padded_input_ids))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(
                    position_ids, (0, target_length - sequence_length), value=0
                )
            elif sequence_length > target_length:
                # "right_trunc":
                input_ids = input_ids[:target_length]
                loss_mask = loss_mask[:target_length]
                position_ids = position_ids[:target_length]

        elif self.pad_mode in ("greedy_packing", "best_fit_packing"):
            # Packing modes: actual bin-assembly happens in __iter__ /
            # _greedy_pack_buffer / _best_fit_pack_pool. Truncate if longer
            # than max_length+1 (only reachable when drop_long_samples=False,
            # which is rejected at init for best_fit_packing).
            if len(input_ids) > target_length:
                input_ids = input_ids[:target_length]
                loss_mask = loss_mask[:target_length]
                position_ids = position_ids[:target_length]
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        labels = input_ids[1:].clone()
        labels[loss_mask[1:] == 0] = IGNORE_INDEX
        input_ids = input_ids[:-1]
        position_ids = position_ids[:-1]

        return input_ids, labels, position_ids, real_token_count

    def _greedy_pack_buffer(self):
        if not self._buffer["input_ids"]:
            return None

        # Concatenate buffer
        input_ids = torch.cat(self._buffer["input_ids"], dim=0)
        labels = torch.cat(self._buffer["labels"], dim=0)
        positions = torch.cat(self._buffer["position_ids"], dim=0)

        L = int(input_ids.numel())
        T = int(self.buffer_max_length)
        if L < T:
            pad_len = T - L
            input_ids = F.pad(input_ids, (0, pad_len), value=self.pad_id)
            positions = F.pad(positions, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)

        return {
            "input": input_ids,
            "positions": positions,
        }, labels

    def _best_fit_pack_pool(self) -> list[dict]:
        """Sort the pool by length descending and best-fit-assign into bins.

        Each bin is a dict ``{"ids": [...], "labels": [...], "pos": [...],
        "used": int}``. Returns the list of closed bins and clears the pool.
        """
        T = int(self.buffer_max_length)
        # Sort by conversation length descending so larger conversations are
        # placed first and small ones slot into the leftover space.
        self._pool.sort(key=lambda t: t[0].shape[0], reverse=True)

        bins: list[dict] = []
        for input_ids, labels, positions in self._pool:
            L = int(input_ids.shape[0])
            # Oversized conversations are dropped upstream; defensive guard.
            assert L <= T, f"oversized conversation ({L} > {T}) reached BFD"
            best_idx = -1
            best_rem = T + 1
            for i, b in enumerate(bins):
                rem = T - b["used"]
                if rem >= L and rem < best_rem:
                    best_idx = i
                    best_rem = rem
            if best_idx == -1:
                bins.append(
                    {
                        "ids": [input_ids],
                        "labels": [labels],
                        "pos": [positions],
                        "used": L,
                    }
                )
            else:
                bins[best_idx]["ids"].append(input_ids)
                bins[best_idx]["labels"].append(labels)
                bins[best_idx]["pos"].append(positions)
                bins[best_idx]["used"] += L
        self._pool = []
        return bins

    def _emit_bin(self, b: dict):
        """Materialize one BFD bin into the (dict, label) yield contract.

        Uses a single pre-allocated tensor per output (input_ids, labels,
        positions) so memory traffic is O(T) rather than the O(T^2) growth of
        repeated torch.cat.
        """
        T = int(self.buffer_max_length)
        used = b["used"]
        input_ids = torch.empty(T, dtype=b["ids"][0].dtype)
        labels = torch.empty(T, dtype=b["labels"][0].dtype)
        positions = torch.empty(T, dtype=b["pos"][0].dtype)
        off = 0
        for ids, lbl, pos in zip(b["ids"], b["labels"], b["pos"]):
            n = ids.shape[0]
            input_ids[off : off + n] = ids
            labels[off : off + n] = lbl
            positions[off : off + n] = pos
            off += n
        if used < T:
            input_ids[used:] = self.pad_id
            labels[used:] = IGNORE_INDEX
            positions[used:] = 0
        return {"input": input_ids, "positions": positions}, labels

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                processed = self._process_one_row(sample)
                self._sample_idx += 1
                if processed is None:
                    # Oversized conversation dropped (only when drop_long_samples=True).
                    self._n_dropped_oversized += 1
                    if self._n_dropped_oversized % 64 == 0:
                        logger.warning(
                            f"[sft {self.pad_mode}] dropped "
                            f"{self._n_dropped_oversized} oversized "
                            f"conversations (> {self.buffer_max_length} tokens)"
                        )
                    continue
                input_ids, labels, positions, real_token_count = processed
                new_len = input_ids.shape[0]

                # Per-dataset sampling stats. Counts real conversation tokens
                # excluding any pad introduced by right_padding — matches
                # pretrain MixedDataset.num_tokens_sampled semantics.
                self.num_docs_sampled[0] += 1
                self.num_tokens_sampled[0] += int(real_token_count)

                if self.pad_mode == "right_padding":
                    # Yield consistent dict structure immediately
                    return_dict = {
                        "input": input_ids,
                        "positions": positions,
                    }
                    yield return_dict, labels
                    continue

                if self.pad_mode == "best_fit_packing":
                    self._pool.append((input_ids, labels, positions))
                    if len(self._pool) >= self.packing_pool_size:
                        bins = self._best_fit_pack_pool()
                        self._rng.shuffle(bins)
                        for b in bins:
                            yield self._emit_bin(b)
                    continue

                # greedy_packing path
                if self._buffer["current_len"] + new_len > self.buffer_max_length:
                    if self._buffer["current_len"] > 0:
                        yield self._greedy_pack_buffer()

                    self._buffer = self._reset_buffer()
                    self._buffer["input_ids"].append(input_ids)
                    self._buffer["position_ids"].append(positions)
                    self._buffer["labels"].append(labels)
                    self._buffer["current_len"] = new_len
                else:
                    self._buffer["input_ids"].append(input_ids)
                    self._buffer["position_ids"].append(positions)
                    self._buffer["labels"].append(labels)
                    self._buffer["current_len"] += new_len

            # Inner stream exhausted. For best_fit_packing, flush whatever
            # is left in the pool so partial bins aren't lost between
            # epochs (and so the dataset terminates cleanly when not infinite).
            if self.pad_mode == "best_fit_packing" and self._pool:
                bins = self._best_fit_pack_pool()
                self._rng.shuffle(bins)
                for b in bins:
                    yield self._emit_bin(b)

            if not self.infinite:
                logger.warning("Dataset has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning("Dataset is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)


class SFTDataLoader(ParallelAwareDataloader):
    """Configurable SFT dataloader wrapping SFTDataset.

    Follows the Configurable pattern so that Trainer can call
    ``config.dataloader.build(dp_world_size=..., dp_rank=..., tokenizer=..., ...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        # inherited from BaseDataLoader.Config:
        #   dataset: str = ""          (message builder name, e.g. "multi_turn")
        #   dataset_path: str | None = None  (HuggingFace dataset path)
        # inherited from ParallelAwareDataloader.Config:
        #   num_workers, persistent_workers, pin_memory, prefetch_factor

        dataset_subset: str | None = None
        """HuggingFace dataset subset / name (passed as `name` to load_dataset)."""

        dataset_split: str = "train"
        """Dataset split to use."""

        dataset_streaming: bool = True
        """Whether to stream the dataset."""

        dataset_seed: int | None = None
        """RNG seed for data shuffling (falls back to trainer seed when None)."""

        # SFT-specific fields (mirrors SFTConfig)
        apply_chat_template: bool = False
        """Apply tokenizer chat template to messages."""

        pad_mode: Literal["right_padding", "greedy_packing", "best_fit_packing"] = (
            "greedy_packing"
        )
        """How to pad/pack sequences into a fixed-length batch.
        - right_padding: each conversation padded individually to seq_len+1.
        - greedy_packing: first-fit-no-split. Conversations concatenated into a
          bin until the next won't fit, then bin is padded and yielded.
        - best_fit_packing: buffer `packing_pool_size` conversations, sort by
          length, best-fit-assign to bins. Lower padding than greedy. Requires
          drop_long_samples=True (oversized convs are dropped, never truncated).
        """

        packing_pool_size: int = 4096
        """Only used by best_fit_packing. Conversations buffered per rank before
        each BFD pass. Larger pool → tighter packing and lower per-batch ratio
        noise at the cost of more CPU memory."""

        drop_long_samples: bool = False
        """If True, conversations longer than buffer_max_length (= seq_len + 1)
        are dropped instead of truncated. Tracked via _n_dropped_oversized.
        Required for pad_mode='best_fit_packing'."""

        dataset_alias: str | None = None
        """Label used for per-dataset sampling stats in metrics
        (data_docs/<alias>, data_tokens/<alias>). If None, falls back to
        dataset_path, then to the message-builder dataset name."""

        chat_template_kwargs: dict = field(default_factory=dict)
        """Extra kwargs forwarded to tokenizer.apply_chat_template."""

        chat_template_prefix_workers: int = 4
        """Thread workers for per-assistant prefix chat-template renders.
        Set to 1 to use the serial prefix-render path."""

        ignore_input_ids_mismatch: bool = False
        """Ignore input_ids mismatch when applying chat template per-turn."""

    def __init__(
        self,
        config: "SFTDataLoader.Config",
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
        dataset = load_dataset(
            config.dataset_path,
            config.dataset_subset,
            split=config.dataset_split,
            streaming=config.dataset_streaming,
        )

        message_builder = DATASET_MESSAGE_BUILDERS[config.dataset]
        hf_ds = SFTDataset(
            dataset=dataset,
            message_builder=message_builder,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=True,
            sft_config=config,
        )

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        super().__init__(
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            generator=rng,
            snapshot_every_n_steps=snapshot_every_n_steps,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )
