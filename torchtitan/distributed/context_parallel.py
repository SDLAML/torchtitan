# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _ContextParallel,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.distributed.tensor.experimental._context_parallel._attention import (
    flex_cp_allgather,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
)
from torchtitan.tools.logging import logger


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    attention_type: str,
) -> None:
    """
    Apply context parallelism to attention modules.

    CP splits the sequence dimension across devices to enable training with
    longer sequences. This function applies CP to the provided attention
    modules.

    Args:
        attention_modules: Sequence of attention modules to apply CP to
        cp_mesh: Device mesh for context parallel dimension
        attention_type: Type of attention mechanism. Must be one of:
            - "sdpa": scaled_dot_product_attention()
            - "flex": flex_attention()
            - "varlen": varlen_attn() (not yet implemented)

    Raises:
        NotImplementedError: If attention_type is "varlen"
    """
    # Apply context parallelism to every attention module
    # TODO: make seq_dim configurable once the implementation doesn't assume 2
    # internally.
    match attention_type:
        case "flex":
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
            )
        case "sdpa":
            # Enable the DTensor dispatcher to route SDPA operations to the
            # Context Parallel implementation. This is required for CP to work
            # with SDPA (but not FlexAttention).
            # Note: Use _disable_context_parallel_dispatcher() if you need to
            # turn this off. In TorchTitan, we currently don't disable the CP
            # dispatcher.
            _enable_context_parallel_dispatcher()
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
            )
        case "varlen":
            raise NotImplementedError(
                "Variable-length attention CP is not yet supported"
            )
        case _:
            raise ValueError(
                f"Invalid attention_type '{attention_type}'. "
                f"Must be one of: 'sdpa', 'flex', 'varlen'"
            )

    for attention_module in attention_modules:
        parallelize_module(
            module=attention_module,
            device_mesh=cp_mesh,
            parallelize_plan=cp_plan,
        )

    logger.info("Applied Context Parallel to the model")


def apply_cp_to_forward(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Wrap inner_attention.forward with CP logic.

    For FlexAttention: allgathers K/V from all CP ranks so each device computes
    attention with local Q and global K/V. Naturally handles SWA — the block_mask
    is Q-sharded/KV-full after cp_shard, which matches local Q vs global K/V.

    For SDPA: wraps Q/K/V as CP-sharded DTensors before calling forward.

    Must be called BEFORE parallelize() / apply_moe_ep_tp() so the CP wrapper
    runs inside any local_map boundary on local tensors.
    """
    first = attention_modules[0]
    if isinstance(first, FlexAttentionWrapper):
        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())

                def cp_forward(q, k, v, **kwargs):
                    k = k.contiguous()
                    v = v.contiguous()
                    global_k, global_v = flex_cp_allgather(k, v, 2, pg_name)
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, ScaledDotProductAttentionWrapper):
        _enable_context_parallel_dispatcher()

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                placement = [Shard(2)]

                def cp_forward(q, k, v, **kwargs):
                    if not isinstance(q, DTensor):
                        q = DTensor.from_local(q, mesh, placement, run_check=False)
                    if not isinstance(k, DTensor):
                        k = DTensor.from_local(k, mesh, placement, run_check=False)
                    if not isinstance(v, DTensor):
                        v = DTensor.from_local(v, mesh, placement, run_check=False)
                    output = orig_fn(q, k, v, **kwargs)
                    return output.to_local() if isinstance(output, DTensor) else output

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, VarlenAttentionWrapper):
        raise NotImplementedError("Variable-length attention CP is not yet supported")
    else:
        raise NotImplementedError(
            f"CP forward wrapping not supported for {type(first).__name__}"
        )

    logger.info("Applied Context Parallel (forward wrapping) to the model")


def apply_cp_to_forward_fused_kv_gather(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Same as apply_cp_to_forward, but with fused KV allgather.

    For FlexAttention: concatenates K and V along the head dimension, does a
    single allgather, then splits back. This halves NCCL kernel launches per
    attention layer (2 allgathers + 2 reduce_scatters → 1 + 1).
    """
    first = attention_modules[0]
    if isinstance(first, FlexAttentionWrapper):
        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())
                cp_size = mesh.size()

                def cp_forward(q, k, v, **kwargs):
                    kv = torch.cat([k, v], dim=1).contiguous()
                    # gather_dim=0 avoids _maybe_view_chunk_cat, which is on
                    # Dynamo's MOD_SKIPLIST and breaks compile + AC.
                    kv_gathered = funcol.all_gather_tensor(kv, 0, pg_name)
                    if isinstance(kv_gathered, funcol.AsyncCollectiveTensor):
                        kv_gathered = kv_gathered.wait()
                    # [B*CP, H_kv*2, S/CP, D] -> [B, H_kv*2, S, D]
                    kv_full = torch.cat(kv_gathered.chunk(cp_size, dim=0), dim=2)
                    global_k, global_v = kv_full.chunk(2, dim=1)
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, ScaledDotProductAttentionWrapper):
        _enable_context_parallel_dispatcher()

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                placement = [Shard(2)]

                def cp_forward(q, k, v, **kwargs):
                    if not isinstance(q, DTensor):
                        q = DTensor.from_local(q, mesh, placement, run_check=False)
                    if not isinstance(k, DTensor):
                        k = DTensor.from_local(k, mesh, placement, run_check=False)
                    if not isinstance(v, DTensor):
                        v = DTensor.from_local(v, mesh, placement, run_check=False)
                    output = orig_fn(q, k, v, **kwargs)
                    return output.to_local() if isinstance(output, DTensor) else output

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, VarlenAttentionWrapper):
        raise NotImplementedError("Variable-length attention CP is not yet supported")
    else:
        raise NotImplementedError(
            f"CP forward wrapping not supported for {type(first).__name__}"
        )

    logger.info(
        "Applied Context Parallel (forward wrapping, fused KV gather) to the model"
    )


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Prepare inputs, labels, and attention masks for Context Parallel forward pass.

    This function prepares tensors for context parallel by:
    1. Creating position indices based on input sequence length
    2. Sharding inputs, labels, and positions across the CP mesh
    3. Sharding attention masks if present

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary that may contain 'attention_masks' to be sharded
        cp_mesh: Device mesh for context parallel dimension
        device: Device to create position tensor on
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = torch.arange(
        0, inputs.shape[1], dtype=torch.int32, device=device
    ).expand(inputs.shape)
    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh,
        (inputs, labels, positions),
        attention_masks,
        load_balancer_type,
    )
    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


def _shard_block_mask_cp(mask: BlockMask, cp_mesh: DeviceMesh) -> BlockMask:
    """Shard BlockMask for allgather CP by slicing Q-block rows.

    Slices existing block-level metadata rather than materializing a dense
    [B, H, Q_SHARD, KV_LEN] tensor, which OOMs at large sequence lengths.
    BlockMask.from_kv_blocks() auto-computes q_num_blocks/q_indices (backward)
    via _transpose_ordered().
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    Q_LEN, KV_LEN = mask.seq_lengths
    Q_SHARD_LEN = Q_LEN // cp_size
    BS_Q = mask.BLOCK_SIZE[0] if isinstance(mask.BLOCK_SIZE, tuple) else mask.BLOCK_SIZE
    q_block_start = cp_rank * (Q_SHARD_LEN // BS_Q)
    q_block_end = q_block_start + Q_SHARD_LEN // BS_Q
    q_offset = cp_rank * Q_SHARD_LEN

    kv_num = mask.kv_num_blocks[:, :, q_block_start:q_block_end].clone()
    # Keep full last dimension — _ordered_to_dense uses shape[-1] as the dense
    # matrix width; trimming it below the max KV index value causes OOB crashes.
    kv_idx = mask.kv_indices[:, :, q_block_start:q_block_end, :].clone()

    full_kv_num = None
    full_kv_idx = None
    if mask.full_kv_num_blocks is not None:
        full_kv_num = mask.full_kv_num_blocks[:, :, q_block_start:q_block_end].clone()
        full_kv_idx = mask.full_kv_indices[:, :, q_block_start:q_block_end, :].clone()

    orig_mod = mask.mask_mod

    def local_mask_mod(b, h, q_idx, kv_idx_arg):
        return orig_mod(b, h, q_idx + q_offset, kv_idx_arg)

    return BlockMask.from_kv_blocks(
        kv_num,
        kv_idx,
        full_kv_num,
        full_kv_idx,
        BLOCK_SIZE=mask.BLOCK_SIZE,
        mask_mod=local_mask_mod,
        seq_lengths=(Q_SHARD_LEN, KV_LEN),
    )


def _shard_block_mask_cp_headtail(mask: BlockMask, cp_mesh: DeviceMesh) -> BlockMask:
    """Shard BlockMask for allgather CP with headtail load balancing.

    Rank r gets Q blocks from head chunk r and tail chunk (2*CP-1-r), matching
    _HeadTailLoadBalancer. KV indices are remapped from original order to
    headtail-allgathered order. Pads kv_idx to KV_BLOCKS width when needed
    (e.g. SWA masks where the original width < KV_BLOCKS) so that
    _ordered_to_dense stays in-bounds after remapping.
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_size = cp_mesh.size()
    Q_LEN, KV_LEN = mask.seq_lengths
    BS_Q = mask.BLOCK_SIZE[0] if isinstance(mask.BLOCK_SIZE, tuple) else mask.BLOCK_SIZE
    BS_KV = (
        mask.BLOCK_SIZE[1] if isinstance(mask.BLOCK_SIZE, tuple) else mask.BLOCK_SIZE
    )

    Q_BLOCKS = Q_LEN // BS_Q
    KV_BLOCKS = KV_LEN // BS_KV
    q_chunk = Q_BLOCKS // (2 * cp_size)
    kv_chunk = KV_BLOCKS // (2 * cp_size)

    q_head_start = cp_rank * q_chunk
    q_tail_start = (2 * cp_size - 1 - cp_rank) * q_chunk
    Q_SHARD_LEN = 2 * q_chunk * BS_Q

    # orig_to_ht[b] = headtail position of original KV block b.
    # Headtail order: [rank0_head_chunk, rank0_tail_chunk, rank1_head_chunk, ...]
    # Rank r head: orig [r*kv_chunk .. (r+1)*kv_chunk) → ht [r*2*kv_chunk .. r*2*kv_chunk+kv_chunk)
    # Rank r tail: orig [(2CP-1-r)*kv_chunk .. (2CP-r)*kv_chunk) → ht [r*2*kv_chunk+kv_chunk .. (r+1)*2*kv_chunk)
    device = mask.kv_indices.device
    orig_to_ht = torch.empty(KV_BLOCKS, dtype=mask.kv_indices.dtype, device=device)
    for r in range(cp_size):
        orig_to_ht[r * kv_chunk : (r + 1) * kv_chunk] = torch.arange(
            r * 2 * kv_chunk, r * 2 * kv_chunk + kv_chunk, device=device
        )
        tail_orig = (2 * cp_size - 1 - r) * kv_chunk
        orig_to_ht[tail_orig : tail_orig + kv_chunk] = torch.arange(
            r * 2 * kv_chunk + kv_chunk, (r + 1) * 2 * kv_chunk, device=device
        )

    def _select_and_remap(
        blk_num: torch.Tensor, blk_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num = torch.cat(
            [
                blk_num[:, :, q_head_start : q_head_start + q_chunk],
                blk_num[:, :, q_tail_start : q_tail_start + q_chunk],
            ],
            dim=2,
        ).clone()
        idx_orig = torch.cat(
            [
                blk_idx[:, :, q_head_start : q_head_start + q_chunk, :],
                blk_idx[:, :, q_tail_start : q_tail_start + q_chunk, :],
            ],
            dim=2,
        ).clone()
        idx = orig_to_ht[idx_orig]
        # After remap, values span [0, KV_BLOCKS); _ordered_to_dense uses
        # shape[-1] as the dense width, so it must be >= KV_BLOCKS.
        if idx.shape[-1] < KV_BLOCKS:
            idx = torch.nn.functional.pad(idx, (0, KV_BLOCKS - idx.shape[-1]))
        return num, idx

    kv_num, kv_idx = _select_and_remap(mask.kv_num_blocks, mask.kv_indices)

    full_kv_num = full_kv_idx = None
    if mask.full_kv_num_blocks is not None:
        full_kv_num, full_kv_idx = _select_and_remap(
            mask.full_kv_num_blocks, mask.full_kv_indices
        )

    orig_mod = mask.mask_mod
    q_head_tok = q_head_start * BS_Q
    q_tail_tok = q_tail_start * BS_Q
    half_q_tok = q_chunk * BS_Q

    def local_mask_mod(b, h, q_local, kv_ht):
        # Q: map local headtail token index to original token index
        q_orig = torch.where(
            q_local < half_q_tok,
            q_local + q_head_tok,
            q_local - half_q_tok + q_tail_tok,
        )
        # KV: map headtail token index to original token index (analytical inverse)
        # rank_r = ht_block // (2*kv_chunk); is_tail = (ht_block // kv_chunk) % 2
        kv_blk_ht = kv_ht // BS_KV
        kv_off = kv_ht % BS_KV
        rank_r = kv_blk_ht // (2 * kv_chunk)
        is_tail = (kv_blk_ht // kv_chunk) % 2 == 1
        pos = kv_blk_ht % kv_chunk
        kv_blk_orig = torch.where(
            is_tail,
            (2 * cp_size - 1 - rank_r) * kv_chunk + pos,
            rank_r * kv_chunk + pos,
        )
        kv_orig = kv_blk_orig * BS_KV + kv_off
        return orig_mod(b, h, q_orig, kv_orig)

    return BlockMask.from_kv_blocks(
        kv_num,
        kv_idx,
        full_kv_num,
        full_kv_idx,
        BLOCK_SIZE=mask.BLOCK_SIZE,
        mask_mod=local_mask_mod,
        seq_lengths=(Q_SHARD_LEN, KV_LEN),
    )


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    """
    Shard inputs and attention masks across the context parallel mesh.

    This function distributes input tensors across devices in the CP mesh
    along the sequence dimension, enabling efficient processing. It optionally
    uses a load balancer to handle uneven computation workload.

    Args:
        cp_mesh: Device mesh for context parallel dimension
        inputs: Tuple of input tensors to be sharded along the sequence
            dimension
        attention_masks: Attention masks to be sharded. Supports None,
            BlockMask, or dict[str, BlockMask]
        load_balancer_type: Type of load balancer to use. Options:
            - "headtail": Use HeadTailLoadBalancer (for SDPA)
            - "ptrr": Use PTRRLoadBalancer (for FlexAttention)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Sharded attention masks (BlockMask or
              dict[str, BlockMask]) or None

    Raises:
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None or a dict
    """
    seq_len = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)

    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail":
                # For SDPA, we use the _HeadTailLoadBalancer.
                load_balancer = _HeadTailLoadBalancer(
                    seq_len, cp_world_size, cp_mesh.device_type
                )
            case "ptrr":
                # For FlexAttention, we use _PTRRLoadBalancer.
                # _PTRRLoadBalancer requires attention_masks to be a BlockMask.
                # For dict[str, BlockMask], _PTRRLoadBalancer currently doesn't
                # support the case where there are multiple masks.
                if attention_masks is None or isinstance(attention_masks, dict):
                    raise ValueError(
                        "PTRRLoadBalancer requires attention_masks to be a "
                        "BlockMask, but got None or dict[str, BlockMask]"
                    )
                if not isinstance(attention_masks, BlockMask):
                    raise ValueError(
                        f"PTRRLoadBalancer requires attention_masks to be a "
                        f"BlockMask, but got {type(attention_masks)}"
                    )
                load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)
            case _:
                raise ValueError(
                    f"Invalid load_balancer_type '{load_balancer_type}'. "
                    f"Must be one of: 'headtail', 'ptrr', or None"
                )

    inputs = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=tuple(input_seq_dim for _ in inputs),
            load_balancer=load_balancer,
        ),
    )

    # BlockMask: shard Q dimension only; KV stays global for allgather CP.
    # Use our slicing-based functions to avoid torch.compile subprocess crashes
    # and OOM from materialising dense [B, H, Q_SHARD, KV_LEN] tensors.
    if attention_masks is not None:
        assert isinstance(attention_masks, (BlockMask, dict))
        if load_balancer_type == "headtail":
            shard_mask = lambda m: _shard_block_mask_cp_headtail(m, cp_mesh)
        else:
            shard_mask = lambda m: _shard_block_mask_cp(m, cp_mesh)
        if isinstance(attention_masks, BlockMask):
            attention_masks = shard_mask(attention_masks)
        else:
            attention_masks = {k: shard_mask(v) for k, v in attention_masks.items()}

    return inputs, attention_masks
