from typing import Optional

import torch
from vllm.v1.outputs import SamplerOutput as SamplerOutputV1
from vllm.v1.sample.rejection_sampler import \
    RejectionSampler as RejectionSamplerV1
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadataV1
from vllm.v1.outputs import SamplerOutput as SamplerOutputV1




def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Apply top-k and top-p optimized for TPU.
    This algorithm avoids using torch.scatter which is extremely slow on TPU.
    This is achieved by finding a "cut-off" element in the original logit, and
    after thresholding the logit using this cut-off, the remaining elements
    shall constitute the top-p set.
    Note: in the case of tie (i.e. multipple cut-off elements present in the
    logit), all tie elements are included in the top-p set. In other words,
    this function does not break ties. Instead, these tie tokens have equal
    chance of being chosen during final sampling, so we can consider the tie
    being broken then.
    """
    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)
    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)
        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))
    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one
        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))
    return logits


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    batch_size = input_ptr.shape[0]
    start_idx = torch.zeros(batch_size,
                            device=cu_num_tokens_ptr.device,
                            dtype=torch.int32)
    start_idx[1:] = cu_num_tokens_ptr[:-1]
    end_idx = cu_num_tokens_ptr

    num_tokens = end_idx - start_idx
    src_val = torch.where(input_ptr == replace_from, replace_to, input_ptr)
    offset = torch.arange(MAX_NUM_TOKENS,
                          device=num_tokens.device).unsqueeze(0)
    mask = offset < num_tokens.unsqueeze(1)
    output_slice = start_idx.unsqueeze(1) + offset
    output_ptr[output_slice[mask]] = src_val.unsqueeze(1).expand(
        batch_size, MAX_NUM_TOKENS)[mask]


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_pytorch(
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=32,  # To avoid recompilation.
    )
    return expanded_x


def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadataV1,
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    logits = logits.to(torch.float32)
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=-1,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    logits = apply_top_k_top_p_npu(logits, top_k, top_p)
    output_prob = logits.softmax(dim=-1, dtype=torch.float32)
    return output_prob


class SimpleSampler(RejectionSamplerV1):

    def __init__(self, main_sampler, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.previous_frequency_penalties = []
        self.previous_repetition_penalties = []
        self.previous_presence_penalties = []
        self.main_sampler = main_sampler
        self.minus_one = None

    def forward(self, input_ids, logits, logits_indices, sampling_metadata,
                num_decodes, num_prefills, bonus_token_ids, target_logits,
                metadata):
        if num_decodes != 0 and num_prefills != 0:
            raise ("Chunked prefill is not supported in current version.")
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raise ("Logprobs gathered is not supported in current version")
        if self.minus_one is None:
            # prepare const on npu
            self.minus_one = -torch.ones(
                1, 1, device=input_ids.device, dtype=input_ids.dtype)
        batch_size = num_decodes + num_prefills
        logits_indices = logits_indices.to(torch.int32)
        num_sampling_tokens_per_req = (logits_indices.numel() // batch_size)
        if self.main_sampler is None or sampling_metadata.all_greedy:
            forward_tokens = logits.argmax(dim=-1).to(
                dtype=input_ids.dtype,
                device=input_ids.device).view(batch_size, -1)
        else:
            if num_sampling_tokens_per_req == 2:
                forward_tokens = torch.empty_like(
                    logits_indices,
                    dtype=input_ids.dtype,
                    device=input_ids.device).view(batch_size, -1)
                target_probs = compute_probs(
                    target_logits,
                    metadata.cu_num_draft_tokens,
                    sampling_metadata,
                )
                target_argmax = target_probs.argmax(dim=-1)
                forward_tokens[:, 0] = target_argmax.view(-1)
                forward_tokens[:, 1] = bonus_token_ids.view(-1)
            else:
                start_indices = torch.arange(
                    batch_size,
                    device=logits.device) * num_sampling_tokens_per_req
                forward_tokens = torch.empty_like(
                    logits_indices,
                    dtype=input_ids.dtype,
                    device=input_ids.device).view(batch_size, -1)
                for i in range(num_sampling_tokens_per_req):
                    sampler_output = self.main_sampler(
                        logits=logits[start_indices],
                        sampling_metadata=sampling_metadata,
                    )
                    start_indices += 1
                    forward_tokens[:,
                                   i] = sampler_output.sampled_token_ids.view(
                                       -1)
                    sampler_output.sampled_token_ids = None

        if num_prefills > 0:
            mtp_input_tokens = torch.empty_like(input_ids)
            mtp_input_tokens[:-1] = input_ids[1:]  # for prefill
        else:
            mtp_input_tokens = input_ids.clone()
        mtp_input_tokens[logits_indices] = forward_tokens.view(-1)
        # Create output buffer.
        # output_token_ids:
        # if accepted [input_ids[-1], forward_tokens_result]
        # else [forward_tokens_result, -1]
        # all prefill
        if num_decodes == 0:
            last_accepted_index = torch.arange(batch_size,
                                               dtype=torch.int32,
                                               device=logits_indices.device)
            output_token_ids = forward_tokens.view(-1, 1)
            accepted_num = 0
        else:
            accepted = input_ids[logits_indices].view(
                batch_size, -1)[:, 1:] == forward_tokens.view(
                    batch_size, -1)[:, :-1]  # bool [batch_size, 1]
            #if model_extra_config.operator_opt_config.control_accept_rate >= 0 and model_extra_config.operator_opt_config.control_accept_rate <= 1:
            #    accepted = torch.empty_like(accepted, dtype=torch.float32).uniform_() < model_extra_config.operator_opt_config.control_accept_rate
            padding_zero = torch.zeros((batch_size, 1),
                                       dtype=torch.int32,
                                       device=input_ids.device)
            accepted_mask = accepted.to(dtype=torch.int32)
            accepted_mask = torch.cat((accepted_mask, padding_zero), dim=1)
            accepted_num = accepted_mask.argmin(dim=1).to(dtype=torch.int32)
            offset = torch.arange(num_sampling_tokens_per_req,
                                  device=accepted_num.device,
                                  dtype=torch.int32)
            output_token_ids = torch.where(
                offset[None, :] <= accepted_num[:, None], forward_tokens,
                self.minus_one)
            last_accepted_index = torch.arange(
                batch_size, device=input_ids.device,
                dtype=torch.int32) * num_sampling_tokens_per_req + accepted_num
        sampler_output = SamplerOutputV1(sampled_token_ids=output_token_ids,
                                         logprobs_tensors=None,
                                         logprobs_tensors_for_trace=None)
        return sampler_output, mtp_input_tokens, last_accepted_index, accepted_num
