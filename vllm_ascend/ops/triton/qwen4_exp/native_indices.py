# SPDX-License-Identifier: Apache-2.0
"""Stable physical-page filtering for native QSA attention."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _native_validity(
    indices, table, requests, flags,
    index_stride, index_column_stride, table_stride, table_column_stride,
    request_stride, count, table_width, num_requests, cache_pages,
    PAGE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + row * request_stride)
    token = tl.load(indices + row * index_stride + col * index_column_stride, col < count, -1)
    page = tl.maximum(token, 0) // PAGE
    valid = (col < count) & (token >= 0) & (request >= 0) & (request < num_requests) & (page < table_width)
    # Clamp addresses as well as masking the read; invalid physical/logical
    # pages must never form an out-of-allocation address on Ascend.
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    safe_page = tl.minimum(page, table_width - 1)
    physical = tl.load(table + safe_request * table_stride + safe_page * table_column_stride, valid, -1)
    valid &= (physical >= 0) & (physical < cache_pages)
    tl.store(flags + row * count + col, valid.to(tl.int32), col < count)


def compact_native_indices(indices, block_table, token_to_req, cache_pages, page_size):
    """Return valid indices in original order and a per-row nonempty mask."""
    rows, columns = indices.shape
    if rows == 0 or columns == 0 or block_table.shape[0] == 0 or block_table.shape[1] == 0:
        return (torch.full((rows, columns), -1, dtype=torch.int32, device=indices.device),
                torch.zeros(rows, dtype=torch.bool, device=indices.device))
    flags = torch.empty((rows, columns), dtype=torch.int32, device=indices.device)
    tile = min(512, max(32, triton.next_power_of_2(columns)))
    grid = (rows, triton.cdiv(columns, tile))
    _native_validity[grid](
        indices, block_table, token_to_req, flags,
        indices.stride(0), indices.stride(1), block_table.stride(0), block_table.stride(1),
        token_to_req.stride(0), columns, block_table.shape[1],
        block_table.shape[0], cache_pages,
        PAGE=page_size, BLOCK=tile,
        enable_auto_bind_sub_block=False,
        disable_tightly_coupled_buffer_reuse=True,
        multibuffer=False,
        code_motion=False,
    )
    # CANN's scan supports graph capture; Triton scan currently lowers to
    # launch-time host-initialized locks, which are not capture-safe.
    counts = flags.cumsum(-1, dtype=torch.int32)
    valid = flags.bool()
    # Use the last in-range element as the sentinel destination for invalid
    # entries. Valid destinations occupy [0, counts[-1] - 1], and whenever an
    # invalid entry exists counts[-1] < columns, so columns - 1 is collision
    # free. This avoids allocating and then slicing/contiguously copying a
    # [rows, columns + 1] temporary on every native-prefill call.
    destination = torch.where(valid, counts - 1, columns - 1).long()
    output = torch.full((rows, columns), -1, dtype=torch.int32, device=indices.device)
    output.scatter_(1, destination, torch.where(valid, indices, -1).int())
    return output, counts[:, -1] > 0
