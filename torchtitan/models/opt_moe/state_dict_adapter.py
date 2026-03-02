# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from collections import defaultdict
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.protocols.model import BaseModel
from torchtitan.models.utils import MoEStateDictAdapter

logger = logging.getLogger()


class OPTMoEStateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # self.model_config and self.hf_assets_path already set by MoEStateDictAdapter

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.gate_proj.weight": "layers.{}.attention.gate_proj.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.router.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.expert_bias": "layers.{}.moe.expert_bias",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # Both native (apply_rotary_emb_cos_sin / rotate_half) and HF use the
        # "consecutive halves" RoPE convention, so wq/wk weights are copied verbatim.
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "load_balance_loss" in key:
                continue
            if "tokens_per_expert" in key:
                continue
            if "router_entropy" in key:
                continue
            if "acc_fwd_times" in key:
                continue

            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key not in to_hf_map:
                    logger.warning("Skipping unknown state dict key: %s", key)
                    continue

                new_key = to_hf_map[abstract_key]

                if "moe.experts" in key:
                    # Store the GroupedExperts Weight metadata for from_hf()
                    if isinstance(value, DTensor):
                        self.grouped_expert_weight_placements[abstract_key] = (
                            value.placements
                        )
                        self.grouped_expert_weight_shape[abstract_key] = value.shape

                        # Split GroupedExperts weight to local individual expert weights
                        local_expert_fqn = self._get_local_experts_weights(
                            new_key,
                            abstract_key,
                            layer_num,
                            value,
                        )
                        hf_state_dict.update(local_expert_fqn)

                    else:
                        # keep this path for offline conversion
                        split_values = self._split_experts_weights(
                            value, self.model_config.layer.moe.num_experts
                        )

                        for expert_num in range(
                            self.model_config.layer.moe.num_experts
                        ):
                            expert_new_key = new_key.format(layer_num, expert_num)
                            hf_state_dict[expert_new_key] = split_values[
                                expert_num
                            ].squeeze()
                else:
                    if new_key is None:
                        continue
                    new_key = new_key.format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    logger.warning("Skipping unknown state dict key: %s", key)
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        # Temporary storage for HF MoE expert weights before regrouping:
        # keyed by (layer_num, native_key_template) e.g. ("0", "layers.{}.moe.experts.w1")
        grouped_experts: dict[tuple[str, str], dict[int, Any]] = defaultdict(dict)

        # Guard for dense models that have no MoE layers (layer.moe is None)
        num_experts = (
            self.model_config.layer.moe.num_experts
            if self.model_config.layer.moe is not None
            else 0
        )

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # collect all numeric indices (layer, expert, ...)
                nums = re.findall(r"\d+", key)
                if not nums:
                    # shouldn't happen, but be defensive
                    continue
                layer_num = nums[0]

                # Generalise *all* numeric indices so MoE patterns match too
                abstract_key = re.sub(r"(\d+)", "{}", key)

                # --- MoE experts (two indices: layer + expert) ---
                if ".mlp.experts." in key:
                    if len(nums) < 2:
                        logger.warning(
                            "Found MoE expert key without expert index: %s", key
                        )
                        continue
                    expert_num = int(nums[1])

                    new_key_template = self.from_hf_map.get(abstract_key, None)
                    if new_key_template is None:
                        # nothing to do (unknown key)
                        continue

                    grouped_experts[(layer_num, new_key_template)][expert_num] = value
                    continue  # don't write directly into state_dict yet

                # --- Non-expert layer parameters (attention, FFN, router, shared_experts, etc.) ---
                # Both models use "consecutive halves" RoPE — q_proj/k_proj copied verbatim.
                new_key_template = self.from_hf_map.get(abstract_key, None)
                if new_key_template is None:
                    # e.g. rotary_emb.inv_freq or unknown keys
                    continue

                new_key = new_key_template.format(layer_num)
                state_dict[new_key] = value
            else:
                new_key = self.from_hf_map.get(key, None)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        # --- Rebuild grouped-expert weights from per-expert HF tensors ---
        for (layer_num, native_key_template), experts_dict in grouped_experts.items():
            # Expect indices [0, num_experts-1]; warn if incomplete
            missing = [i for i in range(num_experts) if i not in experts_dict]
            if missing:
                logger.warning(
                    "Missing experts %s for layer %s param %s when regrouping MoE weights",
                    missing,
                    layer_num,
                    native_key_template,
                )

            # Order by expert index; only keep those we actually have
            ordered_expert_ids = sorted(experts_dict.keys())
            ordered_weights = [experts_dict[i] for i in ordered_expert_ids]

            # Stack along expert dimension to invert _split_experts_weights
            grouped_weight = torch.stack(ordered_weights, dim=0)

            native_key = native_key_template.format(layer_num)
            state_dict[native_key] = grouped_weight

        return state_dict
