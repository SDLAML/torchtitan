# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import (
    apply_cp_to_forward,
    # apply_cp_to_forward_fused_kv_gather,
)
from torchtitan.distributed.dual_pipe_v import (
    DualPipeExpertParallel,
    get_dual_pipe_v_flag,
)
from torchtitan.distributed.expert_parallel import (
    BaseExpertParallel,
    DeepEPExpertParallel,
    ExpertParallel,
    ExpertTensorParallel,
)
from torchtitan.models.llama3.parallelize import apply_ddp
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.models.opt_moe import norm_moe as moe_module
from torchtitan.models.opt_moe.model import OPTMoEModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}


def parallelize_opt_moe(
    model: OPTMoEModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    tp_only_attention = parallelism.tensor_parallel_only_attention
    enable_approx_mid_norm_for_tensor_parallel = (
        parallelism.enable_approx_mid_norm_for_tensor_parallel
    )
    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        if (
            parallelism.enable_async_tensor_parallel
            and not enable_approx_mid_norm_for_tensor_parallel
        ):
            has_real_mid_norm = any(
                not isinstance(block.attention.mid_norm, nn.Identity)
                for block in model.layers.values()
            )
            if has_real_mid_norm:
                raise RuntimeError(
                    "Async TP is incompatible with non-approximate mid_norm. "
                    "Use --parallelism.enable_approx_mid_norm_for_tensor_parallel "
                    "or disable async tensor parallel."
                )

        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            cp_enabled=parallel_dims.cp_enabled,
            tensor_parallel_only_attention=tp_only_attention,
            enable_approx_mid_norm_for_tensor_parallel=enable_approx_mid_norm_for_tensor_parallel,
        )
    # I dont think we need to apply TP for MOE?

    ep_backend = parallelism.expert_parallel_comm_backend
    if not parallel_dims.ep_enabled and not parallel_dims.etp_enabled:
        ep_backend = "standard"

    if ep_backend == "deepep":
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # Import deepep module to register custom ops before accessing them
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(
            parallelism=parallelism, ac_config=ac_config, parallel_dims=parallel_dims
        )

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
            use_deepep=use_deepep,
            tp_only_attention=tp_only_attention,
            enable_approx_mid_norm_for_tensor_parallel=enable_approx_mid_norm_for_tensor_parallel,
        )

    if parallel_dims.cp_enabled:
        # pyrefly: ignore [missing-attribute]
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )
        # apply_cp_to_forward_fused_kv_gather(
        #     [block.attention.inner_attention for block in model.layers.values()],
        #     parallel_dims.get_mesh("cp"),
        # )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        if parallel_dims.ep_enabled:
            apply_compile(model, compile_config, parallel_dims.ep_enabled)
        else:
            apply_compile_wo_ep(model, compile_config)

    if parallel_dims.fsdp_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    cp_enabled: bool,
    tensor_parallel_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    positions_sharding = Replicate() if cp_enabled else None

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None, None, positions_sharding),
                desired_input_layouts=(Replicate(), None, None, positions_sharding),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
        }
        # gate_proj is nn.Identity when gated_attention_type is None, nn.Linear otherwise.
        if transformer_block.attention.gated_attention_type is not None:
            # Gate projection is head/channel aligned, so shard it colwise with heads.
            layer_plan["attention.gate_proj"] = colwise_parallel()

        attention_mid_norm = getattr(
            transformer_block.attention, "mid_norm", nn.Identity()
        )
        if not isinstance(attention_mid_norm, nn.Identity):
            attn_norm_name = "attention.mid_norm"
            if enable_approx_mid_norm_for_tensor_parallel:
                layer_plan[attn_norm_name] = SequenceParallel(sequence_dim=-1)
            else:
                layer_plan[attn_norm_name] = PrepareMidNormInputOutput()

        # dont want to bother the Mid-norm for now
        if not transformer_block.moe_enabled and not tensor_parallel_only_attention:
            layer_plan.update(
                {
                    "ffn_norm": SequenceParallel(),
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

            feed_forward_mid_norm = getattr(
                transformer_block.feed_forward, "mid_norm", nn.Identity()
            )
            if not isinstance(feed_forward_mid_norm, nn.Identity):
                ffn_norm_name = "feed_forward.mid_norm"
                if enable_approx_mid_norm_for_tensor_parallel:
                    layer_plan[ffn_norm_name] = SequenceParallel(sequence_dim=-1)
                else:
                    layer_plan[ffn_norm_name] = PrepareMidNormInputOutput()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    dual_pipe_v: bool = False,
    use_deepep: bool = False,
    tp_only_attention: bool = False,
    enable_approx_mid_norm_for_tensor_parallel: bool = False,
):
    assert ep_mesh is not None or tp_mesh is not None

    # When tp_only_attention=True, suppress TP for MoE expert layers — only attention
    # layers receive TP.  Using the regular TP mesh for expert computation is not
    # supported; users who have TP enabled but don't want it for experts must set
    # tp_only_attention=True.  Expert-specific tensor parallelism uses the dedicated
    # etp_mesh (a separate parallelism dimension).
    if tp_only_attention:
        tp_mesh = None
    elif ep_mesh is None and tp_mesh is not None:
        raise NotImplementedError(
            "TP for MoE experts via the regular TP mesh is not supported. "
            "Use --parallelism.tensor_parallel_only_attention to restrict TP to "
            "attention layers, or configure an EP mesh for expert parallelism."
        )

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        experts_mesh: DeviceMesh | None = None
        experts_plan = None

        if ep_mesh is not None and etp_mesh is not None:
            # EP + ETP: shard experts across both EP and ETP dimensions.
            assert ep_etp_mesh is not None
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        elif ep_mesh is not None:
            # EP only: dispatch tokens to experts across EP ranks.
            experts_mesh = ep_mesh
            if use_deepep:
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts
                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                experts_plan = ExpertParallel()
                logger.info("Applying ExpertParallel to MoE layer")

            # Set metadata used for logging / debug display inside GroupedExperts.
            transformer_block.moe.experts.ep_enable = True
            total_experts = transformer_block.moe.experts.num_experts
            ep_world_size = ep_mesh.size()
            transformer_block.moe.experts.expert_per_rank = (
                total_experts // ep_world_size
            )
            transformer_block.moe.experts.ep_size = ep_world_size

        # else: ep_mesh is None and tp_mesh is None (tp_only_attention=True with no EP)
        #       → no expert parallelism for this layer.

        if dual_pipe_v and isinstance(experts_plan, BaseExpertParallel):
            experts_plan = DualPipeExpertParallel(experts_plan)

        if experts_mesh is not None:
            parallelize_module(
                module=transformer_block.moe.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )

            # ETP shards w1/w3 on the hidden-dim.  mid_norm inside GroupedExperts
            # normalises over the full hidden-dim, so it must see the gathered
            # tensor across ETP ranks — exactly the same issue as attention mid_norm
            # under TP.  Apply the same two-choice fix using the ETP-only mesh (not
            # ep_etp_mesh: the gather is only across ETP, not EP).
            if etp_mesh is not None:
                experts_mid_norm = getattr(
                    transformer_block.moe.experts, "mid_norm", nn.Identity()
                )
                if not isinstance(experts_mid_norm, nn.Identity):
                    etp_norm_plan = (
                        SequenceParallel(sequence_dim=-1)
                        if enable_approx_mid_norm_for_tensor_parallel
                        else PrepareMidNormInputOutput()
                    )
                    parallelize_module(
                        module=transformer_block.moe.experts,
                        device_mesh=etp_mesh,
                        parallelize_plan={"mid_norm": etp_norm_plan},
                    )


def apply_compile_wo_ep(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block, backend=compile_config.backend, fullgraph=True
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_compile(model: nn.Module, compile_config: CompileConfig, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=compile_config.backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    # Patch some globals only once (apply_compile is called multiple times for PP setup)
    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in moe_module._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        moe_module._run_experts_grouped_mm = torch.compile(
            moe_module._run_experts_grouped_mm,
            backend=compile_config.backend,
            fullgraph=True,
        )

        if ep_enabled:
            # pyrefly: ignore [missing-attribute]
            compiled_fn = moe_module._run_experts_grouped_mm

            # keep function logic in sync with `already_patched` above
            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
                activation: Callable,
                mid_norm: nn.Module,
            ) -> torch.Tensor:
                # dynamic number of tokens in expert parallel
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(
                    w1, w2, w3, x, num_tokens_per_expert, activation, mid_norm
                )

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")


class PrepareMidNormInputOutput(torch.distributed.tensor.parallel.ParallelStyle):
    """
    when `norm_everywhere=True`, we need to particularly handle
    the mid-norm in mid of FFN. (and norm before out-proj in attention)

    We need to
    1. Replicate[gather] the input to the norm layer,
    2. Run the norm layer
    3. Shard the output back

    But it does not work with async TP and compile together. (for FFN)

    ###########
    Insteard, Can we use SequenceParallel(dim=-1) here?
    it seems to be working + loss and norm are aligned
    """

    def __init__(
        self,
        shard_dim: int = -1,
    ):
        # fixed layouts for the MLP mid-norm case
        self._in_layout = (Shard(shard_dim),)
        self._desired_in = (Replicate(),)
        self._out_layout = (Replicate(),)
        self._desired_out = (Shard(shard_dim),)

    def _prep_in(self, inputs, mesh):
        x, *rest = inputs
        if not isinstance(x, torch.distributed.tensor.DTensor):
            x = torch.distributed.tensor.DTensor.from_local(
                x, mesh, self._in_layout, run_check=False
            )
        if self._in_layout != self._desired_in:
            x = x.redistribute(placements=self._desired_in)
        return (x.to_local(), *rest)  # hand local tensor to module

    def _prep_out(self, outputs, mesh):
        if not isinstance(outputs, torch.distributed.tensor.DTensor):
            outputs = torch.distributed.tensor.DTensor.from_local(
                outputs, mesh, self._out_layout, run_check=False
            )
        if self._out_layout != self._desired_out:
            outputs = outputs.redistribute(placements=self._desired_out)
        return outputs.to_local()  # keep local shard

    def _apply(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        module.register_forward_pre_hook(  # gather before norm
            lambda m, i: self._prep_in(i, mesh)
        )
        module.register_forward_hook(  # re-shard after norm
            lambda m, i, o: self._prep_out(o, mesh)
        )
        return module
