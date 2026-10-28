# SPDX-License-Identifier: Apache-2.0
"""Paged QSA prefill through CANN's MLA sparse attention primitive."""

import torch
import torch_npu

from vllm_ascend.ops.triton.qwen4_exp.native_indices import compact_native_indices


def qsa_native_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Preserve QK and PV using Q'=[Q,0] and the shared KV'=[K,V].

    CANN's MLA primitive requires a 512-wide shared key/value and a power-of-two
    head group. The unused output half and padded query heads are discarded.
    Query/key RoPE inputs are zero because QSA has already applied its RoPE.
    Separate caches are packed only on this bounded prefill path.
    """
    rows, heads, width = query.shape
    assert width == 256 and key_cache.shape[2:] == (1, 256)
    assert key_cache.shape == value_cache.shape
    packed_kv = torch.cat((key_cache, value_cache), dim=-1)
    padded_heads = 1 << (heads - 1).bit_length()
    packed_query = torch.zeros(rows, padded_heads, 512, dtype=query.dtype, device=query.device)
    packed_query[:, :heads, :width].copy_(query)
    query_rope = torch.zeros(rows, padded_heads, 64, dtype=query.dtype, device=query.device)
    key_rope = torch.zeros((*key_cache.shape[:-1], 64), dtype=key_cache.dtype, device=key_cache.device)

    safe_indices, nonempty = compact_native_indices(
        indices, block_table, token_to_req, key_cache.shape[0], key_cache.shape[1]
    )
    native, _, _ = torch_npu.npu_sparse_flash_attention(
        packed_query,
        packed_kv,
        packed_kv,
        safe_indices.unsqueeze(1),
        width**-0.5,
        block_table=block_table.clamp(0, key_cache.shape[0] - 1),
        actual_seq_lengths_query=query_start_loc[1:].int(),
        actual_seq_lengths_kv=seq_lens.int(),
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
    )
    # Empty selections and graph-padding rows have zero attention by contract.
    result = torch.where(nonempty[:, None, None], native[:, :heads, width:], 0)
    out.copy_(result)
    return out
