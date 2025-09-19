# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# # Adapted from
# # vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py
# # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# # vllm-project/vllm/vllm/model_executor/models/deepseek_v2.py
# """Inference-only DeepseekV2/DeepseekV3 model."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, ModelConfig, VllmConfig)
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.deepseek_v2 import \
    DeepseekV2ForCausalLM  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import \
    yarn_get_mscale  # noqa: E501
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLAAttention,
    get_spec_layer_idx_from_weight_name)
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v2 import (
    CustomDeepseekV2MLP, CustomDeepseekV2RowParallelLinear,
    CustomDeepseekV2RowParallelLinearReplaceAllreduce)
from vllm_ascend.models.layers.mla import AscendMLAModules
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.utils import dispose_tensor

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO
VLLM_ASCEND_GATEDP_ENABLED = envs_ascend.VLLM_ASCEND_GATEDP_ENABLED


class CustomDeepseekDBOMLP(CustomDeepseekV2MLP):
    pass


class CustomDeepseekDBOMoE(nn.Module):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe and \
            self.torchair_graph_enabled

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = AscendFusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)

        if config.n_shared_experts is not None:
            self.all_reduce_merge = self.experts.all_reduce_merge
            reduce_results = not self.all_reduce_merge
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
            self.shared_experts = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=reduce_results,
                force_replicate=self.enable_multistream_moe
                or enable_shared_expert_dp,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None  # type: ignore
        CustomDeepseekDBOMoE.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size
        self.dp_rank = get_dp_group().rank_in_group

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()
        self.rm_router_logits = self.experts.rm_router_logits

    def forward(self,
                hidden_states: torch.Tensor,
                attn_metadata: Optional[AttentionMetadata] = None,
                replace_allreduce: bool = False) -> torch.Tensor:

        forward_context = get_forward_context()
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.

        enable_force_load_balance = forward_context.in_profile_run

        is_prefill = forward_context.with_prefill

        # router_logits: (num_tokens, n_experts)
        router_logits = None
        if not self.rm_router_logits and not self.enable_multistream_moe:
            router_logits, _ = self.gate(hidden_states)

        experts_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=CustomDeepseekDBOMoE.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=self.shared_experts,
            gate=self.gate,
            replace_allreduce=replace_allreduce)

        hidden_states = (
            experts_hidden_states[0] * self.routed_scaling_factor +
            experts_hidden_states[1])
        if self.all_reduce_merge:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            is_prefill = forward_context.with_prefill
            flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
            if not is_prefill or not flashcomm1_ds_prefill:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states

    # To achieve better overlap performance for gate dp, we split the moe forward into two parts
    # after the allgather comm
    def _forward_ms_dp_gate_moe_pre_comm(
        self,
        hidden_states: torch.Tensor,
    ):

        # router_logits: (num_tokens, n_experts)
        router_logits = None
        if not self.rm_router_logits and not self.enable_multistream_moe:
            router_logits, _ = self.gate(hidden_states)

        return self.experts._forward_fused_moe_pre_comm(
            hidden_states=hidden_states,
            router_logits=router_logits,
            gate=self.gate)

    def _forward_ms_dp_gate_moe_post_comp(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            pertoken_scale: Optional[torch.Tensor] = None,
            replace_allreduce: bool = False,
            num_tokens: int = 0):

        forward_context = get_forward_context()
        enable_force_load_balance = forward_context.in_profile_run
        is_prefill = forward_context.with_prefill

        experts_hidden_states = self.experts._forward_fused_moe_post_comp(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            top_k=CustomDeepseekDBOMoE.top_k,
            shared_experts=self.shared_experts,
            pertoken_scale=pertoken_scale,
            replace_allreduce=replace_allreduce,
            num_tokens=num_tokens,
        )

        hidden_states = (
            experts_hidden_states[0] * self.routed_scaling_factor +
            experts_hidden_states[1])
        if self.all_reduce_merge:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            is_prefill = forward_context.with_prefill
            flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
            if not is_prefill or not flashcomm1_ds_prefill:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states


class CustomDeepseekDBOMLAAttention(DeepseekV2MLAAttention):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % self.tp_size == 0
        self.num_local_heads = num_heads // self.tp_size
        self.layers = config.num_hidden_layers
        self.first_k_dense_replace = config.first_k_dense_replace

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        ascend_config = get_ascend_config()
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        if (config.n_routed_experts is not None
                and self.debug_layer_idx >= config.first_k_dense_replace
                and self.debug_layer_idx % config.moe_layer_freq == 0
                and self.enable_shared_expert_dp):
            # TODO: currently dbo may be not compatible with enable_shared_expert_dp
            self.o_proj = CustomDeepseekV2RowParallelLinearReplaceAllreduce(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = CustomDeepseekV2RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                reduce_results=not envs_ascend.VLLM_ASCEND_FC1_ENABLED,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        mla_modules = AscendMLAModules(
            q_a_proj=self.q_a_proj if self.q_lora_rank is not None else None,
            q_a_layernorm=self.q_a_layernorm
            if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            rotary_emb=self.rotary_emb,
        )

        self.mla_attn = MultiHeadLatentAttention(
            self.hidden_size,
            self.enable_shared_expert_dp,
            self.debug_layer_idx,
            self.first_k_dense_replace,
            self.tp_size,
            mla_modules,
            self.num_local_heads,
            self.scaling,
            self.layers,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.q_lora_rank,
            self.qk_nope_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, kv_cache, attn_metadata)


class CustomDeepseekDBODecoderLayer(DeepseekV2DecoderLayer):

    def __init__(self,
                 config: PretrainedConfig,
                 prefix: str,
                 model_config: ModelConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        self.layers = config.num_hidden_layers
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_size = get_dp_group().world_size
        ascend_config = get_ascend_config()
        # TODO: enable mla in vllm-ascend
        if model_config.use_mla:
            attn_cls = CustomDeepseekDBOMLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = CustomDeepseekDBOMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe_layer = True
        else:
            self.mlp = CustomDeepseekDBOMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                reduce_results=not envs_ascend.VLLM_ASCEND_FC1_ENABLED,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe_layer = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.first_k_dense_replace = config.first_k_dense_replace
        self.tp_group = get_tp_group().device_group
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

    def post_attention_process(self, hidden_states, residual, is_prefill):
        if self.tp_size > 1:
            if is_prefill:
                forward_context = get_forward_context()
                num_padding_tokens = forward_context.pad_size
                # Pad hidden_states to make it divisible by tp_size to avoid cross-ring AllGatherV on 910B2C
                if num_padding_tokens > 0:
                    hidden_states = nn.functional.pad(
                        hidden_states, (0, 0, 0, num_padding_tokens))
                output = get_tp_group().reduce_scatter(hidden_states, dim=0)
                dispose_tensor(hidden_states)
                hidden_states = output
                if self.layer_idx == 0:
                    residual = nn.functional.pad(residual,
                                                 (0, 0, 0, num_padding_tokens))
                    residual_parts = torch.chunk(residual, self.tp_size, dim=0)
                    residual = residual_parts[self.tp_rank]
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
                if not envs_ascend.VLLM_ASCEND_GATEDP_ENABLED or not self.is_moe_layer:
                    hidden_states = get_tp_group().all_gather(hidden_states, 0)
                    # unpad
                    if num_padding_tokens > 0:
                        hidden_states = hidden_states[:-num_padding_tokens]
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
            return hidden_states, residual
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            return hidden_states, residual

    def post_mlp_process(self, hidden_states, residual, is_prefill):
        if self.tp_size > 1:
            if is_prefill:
                forward_context = get_forward_context()
                num_padding_tokens = forward_context.pad_size
                # Pad hidden_states to make it divisible by tp_size to avoid cross-ring AllGatherV on 910B2C
                if num_padding_tokens > 0:
                    hidden_states = nn.functional.pad(
                        hidden_states, (0, 0, 0, num_padding_tokens))
                output = get_tp_group().reduce_scatter(hidden_states, dim=0)
                dispose_tensor(hidden_states)
                hidden_states = output
                hidden_states = hidden_states + residual
                residual = hidden_states
            else:
                if isinstance(self.mlp, CustomDeepseekV2MLP):
                    hidden_states = tensor_model_parallel_all_reduce(
                        hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        replace_allreduce: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        forward_context = get_forward_context()
        is_prefill = forward_context.with_prefill
        flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
                hidden_states = self.input_layernorm(hidden_states)
            else:
                previous_hidden_states, previous_residual = hidden_states, residual
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
                # Dispose hidden_states and residual from the previous layer
                # to save npu memory because they're no longer used.
                dispose_tensor(previous_hidden_states)
                dispose_tensor(previous_residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
            hidden_states, residual = self.post_attention_process(
                hidden_states, residual, is_prefill)
        else:
            # Fully Connected
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        tp_size = get_tensor_model_parallel_world_size()
        if self.enable_shared_expert_dp and (
                self.layer_idx == self.first_k_dense_replace
                or self.layer_idx == self.layers) and tp_size > 1:
            num_tokens, _ = residual.shape
            if num_tokens % tp_size:
                residual = nn.functional.pad(residual,
                                             (0, 0, 0, -num_tokens % tp_size))
            chunk_residual = torch.tensor_split(residual, tp_size, dim=0)
            tp_rank = get_tensor_model_parallel_rank()
            residual = chunk_residual[tp_rank]

        if isinstance(self.mlp, CustomDeepseekDBOMoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
            hidden_states, residual = self.post_mlp_process(
                hidden_states, residual, is_prefill)

        if isinstance(
                self.mlp,
                CustomDeepseekDBOMLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        # for last layer of main model and mtp layer.
        if self.enable_shared_expert_dp and self.layer_idx >= (
                self.layers - 1) and tp_size > 1:
            hidden_states = get_tp_group().all_gather(hidden_states, 0)
            residual = get_tp_group().all_gather(residual, 0)

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is not None:
                num_tokens = attn_metadata.num_actual_tokens
            else:
                num_tokens = hidden_states.shape[0]

            if num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:num_tokens]
                residual = residual[:num_tokens]

        return hidden_states, residual

    # should split ops in Decoder Layer
    def _forward_ms_op_input_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        forward_context = get_forward_context()
        is_prefill = forward_context.with_prefill
        flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
                hidden_states = self.input_layernorm(hidden_states)
            else:
                previous_hidden_states, previous_residual = hidden_states, residual
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
                # Dispose hidden_states and residual from the previous layer
                # to save npu memory because they're no longer used.
                dispose_tensor(previous_hidden_states)
                dispose_tensor(previous_residual)
        return hidden_states, residual

    def _update_forward_context(self, batch_idx, attn_metadata,
                                cu_dbo_tokens_across_dp,
                                max_dbo_tokens_across_dp, dbo_num_tokens,
                                dbo_pad_tokens):
        forward_context = get_forward_context()
        forward_context.attn_metadata = attn_metadata[batch_idx]
        forward_context.pad_size = dbo_pad_tokens[batch_idx]
        if self.dp_size > 1:
            forward_context.dp_metadata.cu_tokens_across_dp_cpu = cu_dbo_tokens_across_dp[
                batch_idx]
            forward_context.dp_metadata.max_tokens_across_dp_cpu = max_dbo_tokens_across_dp[
                batch_idx]

    def _forward_ms_layer_flashcomm1_stream(
            self,
            positions: List[torch.Tensor],
            hidden_states: List[torch.Tensor],
            hidden_states_or_q_c: List[torch.Tensor],
            kv_c_normed: List[torch.Tensor],
            kv_no_split: List[torch.Tensor],
            kv_num_tokens: List[torch.Tensor],
            previous_hidden_states: Optional[torch.Tensor],
            previous_residual: Optional[torch.Tensor],
            residual: List[torch.Tensor],
            attn_metadata: List[AttentionMetadata],
            cu_dbo_tokens_across_dp: List[torch.Tensor],
            max_dbo_tokens_across_dp: List[torch.Tensor],
            next_layer: Any,
            is_prefill: List[bool],
            next_kvcache: Optional[torch.Tensor] = None,
            kv_cache: Optional[torch.Tensor] = None,
            is_first_layer: bool = False,
            is_last_layer: bool = False,
            dbo_num_tokens=None,
            dbo_pad_tokens=None,
            replace_allreduce=False):
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        assert layer_index >= 0 and ms_metadata is not None
        num_micro_batchs = ms_metadata.ms_config.num_micro_batches
        #assert isinstance(self.mlp, CustomDeepseekDBOMoE)
        assert len(positions) == num_micro_batchs
        assert len(hidden_states) == num_micro_batchs
        assert residual is not None
        assert attn_metadata is not None
        assert dbo_num_tokens is not None

        # gatedp
        router_logits = [None] * num_micro_batchs
        pertoken_scale = [None] * num_micro_batchs
        gatedp_num_tokens = [None] * num_micro_batchs

        # block 1 : attention
        # block 2 : attn tp communication
        # the attn computation of microbatch 1 can be overlapped with the moe
        # communication in the previous layer, and the attn computation of microbatch 2
        # can be overlapped with the attn communication of microbatch 1
        for i in range(num_micro_batchs):

            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_AR_FINISH],
            )

            self._update_forward_context(i, attn_metadata,
                                         cu_dbo_tokens_across_dp,
                                         max_dbo_tokens_across_dp,
                                         dbo_num_tokens, dbo_pad_tokens)

            if i == 0:
                with set_multistream_context(context, i):
                    hidden_states[i], residual[i], router_logits[
                        i], pertoken_scale[i], gatedp_num_tokens[
                            i] = self._forward_dbo_stream_attn(
                                hidden_states=hidden_states[i],
                                hidden_states_or_q_c=hidden_states_or_q_c[i],
                                residual=residual[i],
                                kv_cache=kv_cache,
                                kv_c_normed=kv_c_normed[i],
                                kv_no_split=kv_no_split[i],
                                kv_num_tokens=kv_num_tokens[i],
                                positions=positions[i],
                                attn_metadata=attn_metadata[i],
                                is_first_layer=is_first_layer,
                                batch_index=i)
            else:

                with torch.npu.stream(context.comm_stream):
                    ms_metadata.ms_events[layer_index][i - 1][
                        MSEventKey.ATTN_COM_FINISH].wait()
                    hidden_states[i], residual[i], router_logits[
                        i], pertoken_scale[i], gatedp_num_tokens[
                            i] = self._forward_dbo_stream_attn(
                                hidden_states=hidden_states[i],
                                hidden_states_or_q_c=hidden_states_or_q_c[i],
                                residual=residual[i],
                                kv_cache=kv_cache,
                                kv_c_normed=kv_c_normed[i],
                                kv_no_split=kv_no_split[i],
                                kv_num_tokens=kv_num_tokens[i],
                                positions=positions[i],
                                attn_metadata=attn_metadata[i],
                                is_first_layer=is_first_layer,
                                batch_index=i)

        for i in range(num_micro_batchs):

            # the following kernels will be submitted to the comm stream to overlap the computation of the
            # moe computation of next microbatch and the attn computation of next layer
            # TODO add func update_context()
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH],
            )

            self._update_forward_context(i, attn_metadata,
                                         cu_dbo_tokens_across_dp,
                                         max_dbo_tokens_across_dp,
                                         dbo_num_tokens, dbo_pad_tokens)

            if i == 0:
                with set_multistream_context(context, i):
                    hidden_states[i], residual[i], hidden_states_or_q_c[
                        i], kv_c_normed[i], kv_no_split[i], kv_num_tokens[
                            i] = self._forward_dbo_stream_mlp(
                                hidden_states=hidden_states[i],
                                residual=residual[i],
                                router_logits=router_logits[i],
                                kv_cache=next_kvcache,
                                next_layer=next_layer,
                                positions=positions[i],
                                attn_metadata=attn_metadata[i],
                                is_prefill=is_prefill[i],
                                is_last_layer=is_last_layer,
                                batch_index=i,
                                pertoken_scale=pertoken_scale[i],
                                gatedp_num_tokens=gatedp_num_tokens[i])
            else:
                with torch.npu.stream(context.comm_stream):
                    ms_metadata.ms_events[layer_index][i - 1][
                        MSEventKey.FFN_COM_FINISH].wait()
                    hidden_states[i], residual[i], hidden_states_or_q_c[
                        i], kv_c_normed[i], kv_no_split[i], kv_num_tokens[
                            i] = self._forward_dbo_stream_mlp(
                                hidden_states=hidden_states[i],
                                residual=residual[i],
                                router_logits=router_logits[i],
                                kv_cache=next_kvcache,
                                next_layer=next_layer,
                                positions=positions[i],
                                attn_metadata=attn_metadata[i],
                                is_prefill=is_prefill[i],
                                is_last_layer=is_last_layer,
                                batch_index=i,
                                pertoken_scale=pertoken_scale[i],
                                gatedp_num_tokens=gatedp_num_tokens[i])
                    if is_last_layer:
                        ms_metadata.ms_events[layer_index][i][
                            MSEventKey.FFN_AR_FINISH].record()

        if previous_hidden_states is not None:
            dispose_tensor(previous_hidden_states)
        return hidden_states, residual, hidden_states_or_q_c, kv_c_normed, kv_no_split, kv_num_tokens

    def _forward_dbo_stream_attn(self, hidden_states: torch.Tensor,
                                 hidden_states_or_q_c: torch.Tensor,
                                 residual: Optional[torch.Tensor],
                                 kv_cache: Any, kv_c_normed: Any,
                                 kv_no_split: Any, positions: torch.Tensor,
                                 kv_num_tokens: Any, attn_metadata: Any,
                                 is_first_layer: bool, batch_index: int):
        current_ms_metadata = get_multistream_comm_context()
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        forward_context = get_forward_context()
        is_prefill = forward_context.with_prefill
        flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill

        # gatedp
        router_logits = None
        pertoken_scale = None
        gatedp_num_tokens = None

        # TODO: refactor
        if is_first_layer:
            hidden_states, residual = self._forward_ms_op_input_layernorm(
                hidden_states=hidden_states, residual=residual)
            hidden_states_or_q_c, hidden_states, kv_c_normed, kv_no_split, kv_num_tokens = self.self_attn.mla_attn._forward_mla_preprocess(
                positions, hidden_states, kv_cache, attn_metadata)

        hidden_states = self.self_attn.mla_attn._forward_mla_attn(
            hidden_states, hidden_states_or_q_c, kv_num_tokens, kv_c_normed,
            kv_no_split, attn_metadata)

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        if current_ms_metadata is not None:
            current_ms_metadata.before_comm_event.record()
        else:
            ms_metadata.ms_events[layer_index][batch_index - 1][
                MSEventKey.ATTN_AR_FINISH].wait()

        if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
            hidden_states, residual = self.post_attention_process(
                hidden_states, residual, is_prefill)
        else:
            # Fully Connected
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        tp_size = get_tensor_model_parallel_world_size()
        if self.enable_shared_expert_dp and (
                self.layer_idx == self.first_k_dense_replace
                or self.layer_idx == self.layers) and tp_size > 1:
            num_tokens, _ = residual.shape
            if num_tokens % tp_size:
                residual = nn.functional.pad(residual,
                                             (0, 0, 0, -num_tokens % tp_size))
            chunk_residual = torch.tensor_split(residual, tp_size, dim=0)
            tp_rank = get_tensor_model_parallel_rank()
            residual = chunk_residual[tp_rank]

        # for gatedp, we move the allgather in fused_moe here
        if isinstance(self.mlp,
                      CustomDeepseekDBOMoE) and VLLM_ASCEND_GATEDP_ENABLED:
            hidden_states, router_logits, pertoken_scale, gatedp_num_tokens = self.mlp._forward_ms_dp_gate_moe_pre_comm(
                hidden_states=hidden_states)
        if current_ms_metadata is not None:
            current_ms_metadata.after_comm_event.record()

        return hidden_states, residual, router_logits, pertoken_scale, gatedp_num_tokens

    def _forward_dbo_stream_mlp(self,
                                hidden_states: torch.Tensor,
                                residual: Optional[torch.Tensor],
                                router_logits: torch.Tensor,
                                kv_cache: Any,
                                next_layer: Any,
                                positions: torch.Tensor,
                                attn_metadata: Any,
                                is_prefill: bool,
                                is_last_layer: bool,
                                batch_index: int = 0,
                                pertoken_scale: Optional[torch.Tensor] = None,
                                gatedp_num_tokens: int = 0):
        forward_context = get_forward_context()
        is_prefill = forward_context.with_prefill
        flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
        current_ms_metadata = get_multistream_comm_context()
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        hidden_states_or_q_c = None
        kv_num_tokens = None
        kv_no_split = None
        kv_c_normed = None

        if isinstance(self.mlp,
                      CustomDeepseekDBOMoE) and VLLM_ASCEND_GATEDP_ENABLED:
            hidden_states = self.mlp._forward_ms_dp_gate_moe_post_comp(
                hidden_states=hidden_states,
                router_logits=router_logits,
                pertoken_scale=pertoken_scale,
                num_tokens=gatedp_num_tokens)
        elif isinstance(self.mlp, CustomDeepseekDBOMoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)

        if current_ms_metadata is not None:
            current_ms_metadata.before_comm_event.record()
        else:
            ms_metadata.ms_events[layer_index][batch_index - 1][
                MSEventKey.FFN_AR_FINISH].wait()

        if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
            hidden_states, residual = self.post_mlp_process(
                hidden_states, residual, is_prefill)
        if isinstance(
                self.mlp,
                CustomDeepseekDBOMLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        # for last layer of main model and mtp layer.
        tp_size = get_tensor_model_parallel_world_size()
        if self.enable_shared_expert_dp and self.layer_idx >= (
                self.layers - 1) and tp_size > 1:
            hidden_states = get_tp_group().all_gather(hidden_states, 0)
            residual = get_tp_group().all_gather(residual, 0)

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is not None:
                num_tokens = attn_metadata.num_actual_tokens
            else:
                num_tokens = hidden_states.shape[0]

            if num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:num_tokens]
                residual = residual[:num_tokens]

        # TODO: refactor
        if not is_last_layer:
            hidden_states, residual = next_layer._forward_ms_op_input_layernorm(
                hidden_states, residual)
            hidden_states_or_q_c, hidden_states, kv_c_normed, kv_no_split, kv_num_tokens = next_layer.self_attn.mla_attn._forward_mla_preprocess(
                positions, hidden_states, kv_cache, attn_metadata)

        if current_ms_metadata is not None:
            current_ms_metadata.after_comm_event.record()

        return hidden_states, residual, hidden_states_or_q_c, kv_c_normed, kv_no_split, kv_num_tokens


class CustomDeepseekDBOModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.first_k_dense_replace = config.first_k_dense_replace

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomDeepseekDBODecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        # tbo related members
        if VLLM_ASCEND_ENABLE_DBO:
            self.dp_size = parallel_config.data_parallel_size
            self.dp_rank = parallel_config.data_parallel_rank
            self.use_mla = model_config.use_mla
            self.multistream_config = MultiStreamConfig()
            multistream_metadata = make_multistream_metadata_ds(
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                causal_lm=getattr(config, "causal_lm", True),
                multistream_config=self.multistream_config,
            )
            self.ms_pre_layer = MultiStreamPreTransformerLayer(
                multistream_metadata)
            self.ms_post_layer = MultiStreamPostTransformerLayer(
                multistream_metadata)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        replace_allreduce = hidden_states.shape[0] % self.tp_size == 0
        can_run_dbo = VLLM_ASCEND_ENABLE_DBO and self.can_run_ms()
        if can_run_dbo:
            previous_hidden_states, previous_residual = hidden_states, residual
            attn_metadata, [positions, hidden_states,
                            residual] = self.ms_pre_layer(
                                [positions, hidden_states, residual], )

            is_prefill = [False] * 2
            for i in range(2):
                if attn_metadata[i] is None:
                    # for profile run
                    is_prefill[i] = True
                else:
                    is_prefill[i] = attn_metadata[i].num_prefills > 0
                    if hasattr(attn_metadata[i], 'with_prefill_across_dp'):
                        is_prefill[i] = is_prefill[i] or attn_metadata[
                            i].with_prefill_across_dp

            hidden_states, residual = self._forward_ms_layers(
                positions=positions,
                hidden_states=hidden_states,
                previous_hidden_states=previous_hidden_states,
                previous_residual=previous_residual,
                residual=residual,
                attn_metadata=attn_metadata,
                moe_start_layer=self.start_layer,
                kv_caches=kv_caches,
                is_prefill=is_prefill,
                replace_allreduce=replace_allreduce,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    kv_caches[i - self.start_layer]
                    if kv_caches is not None else None,
                    attn_metadata,
                    replace_allreduce=replace_allreduce)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if not can_run_dbo:
            forward_context = get_forward_context()
            is_prefill = forward_context.with_prefill
            flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
            if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
                hidden_states = self.norm(hidden_states)
                hidden_states = get_tp_group().all_gather(hidden_states, 0)
                pad_size = forward_context.pad_size
                if pad_size > 0:
                    hidden_states = hidden_states[:-pad_size]
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def can_run_ms(self):
        attn_metadata = get_forward_context().attn_metadata
        # enable prefill overlap
        if attn_metadata is None or attn_metadata.num_prefills == 0 or not attn_metadata.enable_dbo_across_dp:
            return False
        return True

    def _forward_ms_layers(
        self,
        positions: List[torch.Tensor],
        hidden_states: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor],
        previous_residual: Optional[torch.Tensor],
        residual: List[torch.Tensor],
        attn_metadata: List[AttentionMetadata],
        moe_start_layer: int,
        kv_caches: Optional[List[torch.Tensor]] = None,
        is_prefill: List[bool] = [False, False],
        replace_allreduce: bool = False,
    ):

        if moe_start_layer == self.end_layer:
            return hidden_states, residual

        if self.dp_size > 1:
            cu_dbo_tokens_across_dp = []
            max_dbo_tokens_across_dp = []
            # update the cu_dbo_tokens_across_dp
            for i in range(self.multistream_config.num_micro_batches):
                batchsize = attn_metadata[i].num_input_tokens
                num_dbo_tokens_across_dp = self._num_dbo_tokens_across_dp(
                    batchsize, self.dp_size, self.dp_rank)
                max_dbo_token = torch.max(num_dbo_tokens_across_dp)
                cu_dbo_token = torch.cumsum(num_dbo_tokens_across_dp, dim=0)
                cu_dbo_tokens_across_dp.append(cu_dbo_token)
                max_dbo_tokens_across_dp.append(max_dbo_token)
        else:
            cu_dbo_tokens_across_dp = [None] * 2
            max_dbo_tokens_across_dp = [None] * 2
        dbo_num_tokens = [
            attn_metadata[i].num_input_tokens
            for i in range(self.multistream_config.num_micro_batches)
        ]
        tp_world_size = get_tensor_model_parallel_world_size()
        dbo_pad_tokens = [
            (tp_world_size - dbo_num_tokens[i] % tp_world_size) % tp_world_size
            for i in range(self.multistream_config.num_micro_batches)
        ]

        hidden_states_or_q_c = [None] * 2
        kv_c_normed = [None] * 2
        kv_num_tokens = [None] * 2
        kv_no_split = [None] * 2
        # the rest layers
        for i in range(moe_start_layer, self.end_layer):
            layer = self.layers[i]
            #, hidden_states_or_q_c, kv_c_normed, k_pe
            hidden_states, residual, hidden_states_or_q_c, kv_c_normed, kv_no_split, kv_num_tokens = layer._forward_ms_layer_flashcomm1_stream(
                positions=positions,
                hidden_states=hidden_states,
                hidden_states_or_q_c=hidden_states_or_q_c,
                kv_c_normed=kv_c_normed,
                kv_no_split=kv_no_split,
                kv_num_tokens=kv_num_tokens,
                residual=residual,
                next_layer=self.layers[i +
                                       1] if i != self.end_layer - 1 else None,
                next_kvcache=kv_caches[i - self.start_layer + 1]
                if kv_caches is not None and i != self.end_layer - 1 else None,
                previous_hidden_states=previous_hidden_states,
                previous_residual=previous_residual,
                attn_metadata=attn_metadata,
                max_dbo_tokens_across_dp=max_dbo_tokens_across_dp,
                cu_dbo_tokens_across_dp=cu_dbo_tokens_across_dp,
                kv_cache=kv_caches[i - self.start_layer]
                if kv_caches is not None else None,
                is_prefill=is_prefill,
                is_first_layer=True if i == self.start_layer else False,
                is_last_layer=True if i == self.end_layer - 1 else False,
                dbo_num_tokens=dbo_num_tokens,
                dbo_pad_tokens=dbo_pad_tokens,
                replace_allreduce=replace_allreduce)

            previous_hidden_states = previous_residual = None
            advance_step_multistream_layer_context()

        layer_index, ms_metadata, _ = get_multistream_layer_context()
        for i in range(2):

            ms_metadata.try_wait_event(layer_index - 1, i,
                                       MSEventKey.FFN_AR_FINISH)

            forward_context = get_forward_context()
            is_prefill = forward_context.with_prefill
            flashcomm1_ds_prefill = forward_context.flashcomm1_ds_prefill
            if flashcomm1_ds_prefill and is_prefill and self.tp_size > 1:
                hidden_states[i] = self.norm(hidden_states[i])
                hidden_states[i] = get_tp_group().all_gather(
                    hidden_states[i], 0)
                if dbo_pad_tokens[i] > 0:
                    hidden_states[i] = hidden_states[i][:-dbo_pad_tokens[i]]
            else:
                hidden_states[i], _ = self.norm(hidden_states[i], residual[i])

        [hidden_states,
         residual] = self.ms_post_layer([hidden_states, residual], )
        return hidden_states, residual

    def _num_dbo_tokens_across_dp(self, num_tokens: int, dp_size: int,
                                  dp_rank: int) -> torch.Tensor:
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor


class CustomDeepseekDBOForCausalLM(DeepseekV2ForCausalLM):
    # add `packed_modules_mapping` in `DeepseekV2ForCausalLM` to support weight merging
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomDeepseekDBOModel(vllm_config=vllm_config,
                                            prefix=maybe_prefix(
                                                prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    # NOTE: This `load_weights` is mainly copied from
    # https://github.com/vllm-project/vllm/commit/07b8fae219b1fff51ef115c38c44b51395be5bb5
    # to fix CI, and it is different from the implementation in main
    # TODO: support eplb style load_weights
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = AscendFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "module" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id,
                                  return_success=False)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states
