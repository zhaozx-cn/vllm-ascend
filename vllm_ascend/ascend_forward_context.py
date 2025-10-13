import math
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import (BatchDescriptor, get_forward_context,
                                  set_forward_context)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_sp

if TYPE_CHECKING:
    from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
else:
    WeightPrefetchMethod = None


class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    AllGatherEP = 3
    NaiveMulticast = 4
    All2AllSeq = 5


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    NAIVE_MULTICAST = 3


# TODO(zzzzwwjj): add soc_version to choose branch
def _get_fused_moe_state(ep_size: int, with_prefill: bool,
                         is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        return FusedMoEState.AllGatherEP
    elif ep_size == 1:
        if with_prefill:
            return FusedMoEState.NaiveMulticast
        else:
            return FusedMoEState.AllGather
    # NOTE: mc2 need ep_size >= 16 & all2all can't use in torchair graph.
    elif ep_size < 16 or with_prefill:
        return FusedMoEState.All2All
    else:
        return FusedMoEState.MC2


@contextmanager
def set_ascend_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        with_prefill: bool = True,
        in_profile_run: bool = False,
        reserved_mc2_mask: Optional[torch.Tensor] = None,
        moe_comm_type: Optional[MoECommType] = None,
        num_actual_tokens: Optional[int] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: Optional[BatchDescriptor] = None,
        prefetch_stream: torch.npu.Stream = None,
        model_instance: torch.nn.Module = None,
        weight_prefetch_method: Optional[WeightPrefetchMethod] = None):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(
            attn_metadata,
            vllm_config,
            virtual_engine=virtual_engine,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
    ):
        forward_context = get_forward_context()

        from vllm_ascend.ops.moe.moe_comm_method import get_moe_comm_method
        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        forward_context.with_prefill = with_prefill
        tp_world_size = get_tensor_model_parallel_world_size()
        ep_size = (get_ep_group().world_size if
                   vllm_config.parallel_config.enable_expert_parallel else 1)

        is_deepseek_v3_r1 = hasattr(
            vllm_config.model_config.hf_config, 'n_routed_experts'
        ) and vllm_config.model_config.hf_config.n_routed_experts == 256
        fused_moe_state = _get_fused_moe_state(ep_size, with_prefill,
                                               is_deepseek_v3_r1)
        forward_context.fused_moe_state = fused_moe_state
        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # set for sequence parallelism, 1000 is the batch size concurrency threshold for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency exceeds this threshold,
        # the performance benefits can be maximized. Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.
        sp_enabled = enable_sp(vllm_config) and \
            tp_world_size > 1 and \
            num_tokens is not None and num_tokens > 1000

        if get_ascend_config(
        ).enable_shared_expert_dp and num_tokens is not None:
            sp_enabled = True
        if sp_enabled:
            pad_size = (tp_world_size -
                        (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size
        forward_context.sp_enabled = sp_enabled
        forward_context.num_tokens = num_tokens

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if model_instance is not None and \
            hasattr(model_instance, "model") and \
            hasattr(model_instance.model, "start_layer"):
            forward_context.layer_idx = model_instance.model.start_layer

        # TODO(rjg-lyh): refactor mlp weight prefetch method
        # set for mlp weight prefetch
        prefetch_mlp_enabled = envs_ascend.VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE and \
            envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP and \
            forward_context.layer_idx is not None and \
            num_tokens is not None and num_tokens < 500
        if prefetch_mlp_enabled:
            forward_context.prefetch_stream = prefetch_stream
            forward_context.model_instance = model_instance
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False
        forward_context.prefetch_mlp_enabled = prefetch_mlp_enabled
        # TODO(yuzhup): integrate moe weight prefetch method
        forward_context.weight_prefetch_method = weight_prefetch_method

        # TODO(rjg-lyh): The current implementation is somewhat brute force and not elegant.
        # It will be improved later by implementing operator fusion through the FX graph.
        #
        # set for addrmsnorm+quant fusion.
        # this optim now just support dense models due to the specific operators used.
        # Once the necessary conditions are met, support for MOE models will also be added.
        from vllm_ascend.quantization.quant_config import AscendQuantConfig
        addrmsnorm_quant_fusion_enabled = isinstance(vllm_config.quant_config, AscendQuantConfig) and \
            vllm_config.model_config.hf_config.model_type in ["llama", "qwen2", "qwen3"] and \
            forward_context.layer_idx is not None
        if addrmsnorm_quant_fusion_enabled:
            forward_context.model_instance = model_instance
            forward_context.num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers
            forward_context.fusion_linear = "gate_up_dense" if forward_context.layer_idx == 0 else "qkv_dense"
        forward_context.addrmsnorm_quant_fusion_enabled = addrmsnorm_quant_fusion_enabled

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item(
            )
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(
                max_tokens_across_dp / tp_world_size) * tp_world_size

            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[:forward_context.
                                             padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass
