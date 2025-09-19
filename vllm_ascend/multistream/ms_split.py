from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState

from .base import MSAttentionMetadataSplitConfig


def compute_split_seq_index(
    query_lens: Optional[list[int]],
    attn_state: AscendAttentionState,
    num_tokens: int,
    imbalance_ratio: float = 0.1,
) -> list[int]:
    if attn_state != AscendAttentionState.DecodeOnly:
        assert query_lens is not None
        total_tokens = sum(query_lens)
        # the first index in last split
        tokens, split_index = 0, 0
        for value in query_lens:
            tokens += value
            split_index += 1
            if tokens >= total_tokens // 2:
                delta_1 = abs(tokens - total_tokens // 2)
                delta_2 = abs(tokens - total_tokens // 2 - value)
                if delta_1 <= delta_2:
                    # use delta_1
                    delta = delta_1
                else:
                    # use delta_2
                    delta = delta_2
                    split_index -= 1
                    tokens -= value
                if delta <= total_tokens * imbalance_ratio:
                    return [tokens, split_index]
                else:
                    return [0, 0]
    else:
        tokens = num_tokens // 2
        return [tokens, tokens]
    return [0, 0]


def split_attn_tensor_type(
    input_tensor: torch.Tensor,
    index: int,
) -> List[torch.Tensor]:
    return [input_tensor[:index], input_tensor[index:]]


def split_attn_int_type(
    var: int,
    index: int,
) -> List[torch.Tensor]:
    return [min(var, index), max(var - index, 0)]


def model_input_split_v1_mla_attn(
    attn_metadata,
    _metadata_cls,
    ms_split_config: MSAttentionMetadataSplitConfig,
) -> List[Any]:
    assert 0 < ms_split_config.num_micro_batches < 3
    if attn_metadata is None:
        return [attn_metadata]
    [token_index,
     seq_index] = compute_split_seq_index(attn_metadata.query_lens,
                                          attn_metadata.attn_state,
                                          attn_metadata.num_decode_tokens)
    if token_index == 0 or seq_index == 0 or seq_index == len(
            attn_metadata.query_lens):
        return [attn_metadata]

    query_start_loc_cpu = np.zeros(shape=(len(attn_metadata.query_lens) + 1, ),
                                   dtype=int)
    np.cumsum(attn_metadata.query_lens, out=query_start_loc_cpu[1:])
    if attn_metadata.num_prefills > 0:
        prefill_query_start_loc = np.zeros(
            shape=(len(attn_metadata.prefill.query_lens) + 1, ), dtype=int)
        np.cumsum(attn_metadata.prefill.query_lens,
                  out=prefill_query_start_loc[1:])

    # split attn metadata
    [slot_mapping_pre,
     slot_mapping_post] = split_attn_tensor_type(attn_metadata.slot_mapping,
                                                 token_index)
    [num_decodes_pre,
     num_decodes_post] = split_attn_int_type(attn_metadata.num_decodes,
                                             seq_index)
    [num_decode_tokens_pre, num_decode_tokens_post
     ] = split_attn_int_type(attn_metadata.num_decode_tokens, token_index)
    [num_prefills_pre, num_prefills_post
     ] = split_attn_int_type(attn_metadata.num_prefills,
                             max(0, seq_index - attn_metadata.num_decodes))
    seq_lens = attn_metadata.prefill.seq_lens if attn_metadata.num_prefills > 0 else attn_metadata.decode.seq_lens
    [seq_lens_pre, seq_lens_post] = split_attn_tensor_type(seq_lens, seq_index)

    query_start_loc_pre = query_start_loc_post = None
    if attn_metadata.query_start_loc is not None:
        query_start_loc_pre = attn_metadata.query_start_loc[:seq_index + 1]
        query_start_loc_post = deepcopy(
            attn_metadata.query_start_loc[seq_index:]
        ) - attn_metadata.query_start_loc[seq_index]
    [block_table_pre,
     block_table_post] = split_attn_tensor_type(attn_metadata.block_tables,
                                                seq_index)

    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
        # the attn_mla kernel in torch npu only accept 128*128 attn mask
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = attn_metadata.attn_state
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # should be none in decode only state
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = AscendAttentionState.DecodeOnly
    else:
        # chunked prefill
        attn_mask_pre = attn_mask_post = None
        if num_prefills_pre > 0:
            attn_state_pre = attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_state_post = AscendAttentionState.ChunkedPrefill
            if attn_metadata.attn_mask is not None:
                attn_mask_pre = attn_metadata.attn_mask[:token_index, :max(
                    seq_lens_pre)].contiguous()
                attn_mask_post = attn_metadata.attn_mask[
                    token_index:, :max(seq_lens_post)].contiguous()
        else:
            attn_state_pre = AscendAttentionState.DecodeOnly
            attn_state_post = AscendAttentionState.ChunkedPrefill
            if attn_metadata.attn_mask is not None:
                attn_mask_post = attn_metadata.attn_mask[
                    token_index:, :max(seq_lens_post)].contiguous()
    from vllm_ascend.attention.mla_v1 import (AscendMLADecodeMetadata,
                                              AscendMLAPrefillMetadata)
    if num_prefills_pre > 0:
        # split metadata.prefill
        prefill = attn_metadata.prefill
        chunked_context = prefill.chunked_context
        [input_positions_pre, input_positions_post] = split_attn_tensor_type(
            prefill.input_positions,
            token_index - attn_metadata.num_decode_tokens)
        [block_tables_pre, block_tables_post
         ] = split_attn_tensor_type(prefill.block_table,
                                    seq_index - attn_metadata.num_decodes)
        [prefill_query_lens_pre, prefill_query_lens_post
         ] = split_attn_tensor_type(prefill.query_lens,
                                    seq_index - attn_metadata.num_decodes)
        [prefill_cos_pre, prefill_cos_post
         ] = split_attn_tensor_type(prefill.cos,
                                    token_index - attn_metadata.num_decodes)
        [prefill_sin_pre, prefill_sin_post
         ] = split_attn_tensor_type(prefill.sin,
                                    token_index - attn_metadata.num_decodes)
        prefill_query_start_loc_pre = prefill.query_start_loc[:seq_index + 1 -
                                                              attn_metadata.
                                                              num_decodes]
        prefill_query_start_loc_post = deepcopy(
            prefill.query_start_loc[seq_index - attn_metadata.num_decodes:]
        ) - prefill.query_start_loc[seq_index - attn_metadata.num_decodes]
        context_len_pre = seq_lens_pre[attn_metadata.num_decodes:]
        context_len_post = seq_lens_post
        prefill_max_query_len_pre = max(prefill_query_lens_pre)
        prefill_max_query_len_post = max(prefill_query_lens_post)
        # chunked prefill metadata
        if chunked_context is not None:
            chunked_len = prefill.seq_lens[attn_metadata.
                                           num_decodes:] - prefill.query_lens
            [chunked_len_pre, chunked_len_post
             ] = split_attn_tensor_type(chunked_len,
                                        seq_index - attn_metadata.num_decodes)
            max_chunked_len_pre = chunked_len_pre.to("cpu").max().item()
            max_chunked_len_post = chunked_len_post.to("cpu").max().item()

            chunked_starts_pre = chunked_context.starts[:, :seq_index -
                                                        attn_metadata.
                                                        num_decodes]
            # we have no chunk seq lens, so we use chunk cu seq lens here
            chunked_seq_lens = torch.diff(chunked_context.cu_seq_lens,
                                          dim=1).to("cpu")
            # split
            chunked_seq_lens_pre = chunked_seq_lens[:, :seq_index -
                                                    attn_metadata.num_decodes]
            chunked_cu_seq_lens_pre = chunked_context.cu_seq_lens[:, :
                                                                  seq_index -
                                                                  attn_metadata
                                                                  .
                                                                  num_decodes +
                                                                  1]
            chunked_seq_tot_pre = chunked_seq_lens_pre.sum(dim=1).tolist()

            # TODO: currently after split the last several elements in seq_tot may be 0 and encounter error in mla attn
            # i.e.: tensor([4,4,4],[1,4,1],[0,0,0]) -> seq_tot: [12,6,0]
            # so we remove them, note that the element in the middle of seq_tot should not be zero.
            chunked_seq_tot_pre = [x for x in chunked_seq_tot_pre if x != 0]

            chunked_max_seq_lens_pre = chunked_seq_lens_pre.max(
                dim=1).values.tolist()

            chunked_starts_post = chunked_context.starts[:, seq_index -
                                                         attn_metadata.
                                                         num_decodes:]

            chunked_seq_lens_post = chunked_seq_lens[:,
                                                     seq_index - attn_metadata.
                                                     num_decodes:]
            chunked_cu_seq_lens_post = chunked_context.cu_seq_lens[:,
                                                                   seq_index -
                                                                   attn_metadata
                                                                   .num_decodes
                                                                   +
                                                                   1:] - chunked_context.cu_seq_lens[:,
                                                                                                     seq_index
                                                                                                     -
                                                                                                     attn_metadata
                                                                                                     .
                                                                                                     num_decodes].unsqueeze(
                                                                                                         1
                                                                                                     )
            # check device
            zero = torch.zeros((chunked_cu_seq_lens_post.size(0), 1),
                               device=chunked_cu_seq_lens_post.device,
                               dtype=torch.int32)
            chunked_cu_seq_lens_post = torch.cat(
                (zero, chunked_cu_seq_lens_post), dim=1)
            chunked_seq_tot_post = chunked_seq_lens_post.sum(dim=1).tolist()
            chunked_max_seq_lens_post = chunked_seq_lens_post.max(
                dim=1).values.tolist()
            chunked_seq_tot_post = [x for x in chunked_seq_tot_post if x != 0]

            chunked_prefill_metadata_pre = None if max_chunked_len_pre == 0 else AscendMLAPrefillMetadata.ChunkedContextMetadata(
                cu_seq_lens=chunked_cu_seq_lens_pre,
                starts=chunked_starts_pre,
                seq_tot=chunked_seq_tot_pre,
                max_seq_lens=chunked_max_seq_lens_pre,
                workspace=chunked_context.workspace,
                chunk_seq_lens=chunked_seq_lens_pre,
            )
            chunked_prefill_metadata_post = None if max_chunked_len_post == 0 else AscendMLAPrefillMetadata.ChunkedContextMetadata(
                cu_seq_lens=chunked_cu_seq_lens_post,
                starts=chunked_starts_post,
                seq_tot=chunked_seq_tot_post,
                max_seq_lens=chunked_max_seq_lens_post,
                workspace=chunked_context.workspace,
                chunk_seq_lens=chunked_seq_lens_post,
            )

        else:
            chunked_prefill_metadata_pre = chunked_prefill_metadata_post = None

        # construct Prefill metadata
        prefill_pre = AscendMLAPrefillMetadata(
            attn_mask=attn_mask_pre,
            query_lens=prefill_query_lens_pre,
            seq_lens=seq_lens_pre,
            query_start_loc=prefill_query_start_loc_pre,
            input_positions=input_positions_pre,
            context_lens=context_len_pre,
            block_table=block_tables_pre,
            max_query_len=prefill_max_query_len_pre,
            max_seq_lens=context_len_pre.max().item(),
            chunked_context=chunked_prefill_metadata_pre,
            cos=prefill_cos_pre,
            sin=prefill_sin_pre,
        )
        prefill_post = AscendMLAPrefillMetadata(
            attn_mask=attn_mask_post,
            query_lens=prefill_query_lens_post,
            seq_lens=seq_lens_post,
            query_start_loc=prefill_query_start_loc_post,
            input_positions=input_positions_post,
            context_lens=context_len_post,
            block_table=block_tables_post,
            max_query_len=prefill_max_query_len_post,
            max_seq_lens=context_len_post.max().item(),
            chunked_context=chunked_prefill_metadata_post,
            cos=prefill_cos_post,
            sin=prefill_sin_post,
        )
        decode_pre = attn_metadata.decode
        decode_post = None
    else:
        # prefill is None, split metadata.decode
        [input_positions_pre, input_positions_post
         ] = split_attn_tensor_type(attn_metadata.decode.input_positions,
                                    token_index)
        [block_tables_pre, block_tables_post
         ] = split_attn_tensor_type(attn_metadata.decode.block_table,
                                    seq_index)
        [decode_cos_pre,
         decode_cos_post] = split_attn_tensor_type(attn_metadata.decode.cos,
                                                   token_index)
        [decode_sin_pre,
         decode_sin_post] = split_attn_tensor_type(attn_metadata.decode.sin,
                                                   token_index)
        [decode_seq_lens_pre,
         decode_seq_lens_post] = split_attn_tensor_type(seq_lens, seq_index)
        decode_pre = AscendMLADecodeMetadata(
            input_positions=input_positions_pre,
            block_table=block_tables_pre,
            seq_lens=decode_seq_lens_pre,
            max_seq_lens=max(decode_seq_lens_pre),
            seq_lens_list=decode_seq_lens_pre.tolist(),
            cos=decode_cos_pre,
            sin=decode_sin_pre,
        )
        decode_post = AscendMLADecodeMetadata(
            input_positions=input_positions_post,
            block_table=block_tables_post,
            seq_lens=decode_seq_lens_post,
            max_seq_lens=max(decode_seq_lens_post),
            seq_lens_list=decode_seq_lens_post.tolist(),
            cos=decode_cos_post,
            sin=decode_sin_post,
        )
        prefill_pre = None
        prefill_post = prefill
    # construct metadata
    from vllm_ascend.attention.mla_v1 import AscendMLAPrefillMetadata
    attention_metadata_pre = _metadata_cls(
        num_actual_tokens=token_index,
        num_input_tokens=token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_pre,
        seq_lens=seq_lens_pre,
        query_start_loc=query_start_loc_pre,
        block_tables=block_table_pre,
        num_decodes=num_decodes_pre,
        num_prefills=num_prefills_pre,
        num_decode_tokens=num_decode_tokens_pre,
        attn_state=attn_state_pre,
        attn_mask=attn_mask_pre,
        prefill=prefill_pre,
        decode=decode_pre,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    attention_metadata_post = _metadata_cls(
        num_actual_tokens=attn_metadata.num_actual_tokens - token_index,
        num_input_tokens=attn_metadata.num_input_tokens - token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_post,
        seq_lens=seq_lens_post,
        query_start_loc=query_start_loc_post,
        block_tables=block_table_post,
        num_decodes=num_decodes_post,
        num_prefills=num_prefills_post,
        num_decode_tokens=num_decode_tokens_post,
        attn_mask=attn_mask_post,
        attn_state=attn_state_post,
        prefill=prefill_post,
        decode=decode_post,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    return [attention_metadata_pre, attention_metadata_post]
