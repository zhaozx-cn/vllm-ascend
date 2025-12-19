from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2Model,
                                                    _get_llama_4_scaling)
from vllm.sequence import IntermediateTensors


def forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    # Note(wxy): This is a hack fix to avoid graph mode error for torch 2.8
    # We'll find a better way to remove this patch.
    try:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling")
    except AttributeError:
        llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer):
        # Collect auxiliary hidden states if specified
        if layer_idx in self.aux_hidden_state_layers:
            aux_hidden_state = (
                hidden_states + residual if residual is not None else hidden_states
            )
            aux_hidden_states.append(aux_hidden_state)
        hidden_states, residual = layer(positions, hidden_states, residual,
                llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states

def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    super().__init__()

    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    self.config = config
    self.device = current_platform.device_type

    self.vocab_size = config.vocab_size
    self.is_v32 = hasattr(config, "index_topk")
    if self.is_v32:
        topk_tokens = config.index_topk
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )
    else:
        topk_indices_buffer = None

    if get_pp_group().is_first_rank:
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
    else:
        self.embed_tokens = PPMissingLayer()
    self.start_layer, self.end_layer, self.layers = make_layers(
        config.num_hidden_layers,
        lambda prefix: DeepseekV2DecoderLayer(
            vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
        ),
        prefix=f"{prefix}.layers",
    )

    if get_pp_group().is_last_rank:
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    else:
        self.norm = PPMissingLayer()
    self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], config.hidden_size
    )
    self.aux_hidden_state_layers: tuple[int, ...] = ()

DeepseekV2Model.__init__ = __init__
DeepseekV2Model.forward = forward

def set_aux_hidden_state_layers(self, layers: tuple[int]) -> None:
    self.model.aux_hidden_state_layers = layers

def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, int, int]:
    num_layers = len(self.model.layers)
    return 2, num_layers // 2, num_layers - 3

DeepseekV2ForCausalLM.set_aux_hidden_state_layers = set_aux_hidden_state_layers
DeepseekV2ForCausalLM.get_eagle3_aux_hidden_state_layers = get_eagle3_aux_hidden_state_layers


