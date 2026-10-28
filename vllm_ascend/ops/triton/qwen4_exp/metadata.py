# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend QSA metadata using bounded register reductions for request lookup."""

from vllm.triton_utils import tl, triton

_REQUEST_TILE = 16
_TOKEN_TILE = 128


@triton.jit
def _qsa_metadata_kernel(
    loc,
    lens,
    table,
    common_slots,
    requests,
    positions,
    slots,
    work,
    table_stride,
    table_width,
    num_reqs,
    num_tokens,
    num_mapped,
    work_rows,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    RING: tl.constexpr,
    TOKENS: tl.constexpr,
    REQUESTS: tl.constexpr,
    WORK: tl.constexpr,
):
    row = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    req = tl.arange(0, REQUESTS)
    starts = tl.load(loc + req, req < num_reqs, 0)
    ends = tl.load(loc + req + 1, req < num_reqs, 0)
    seqs = tl.load(lens + req, req < num_reqs, 0)
    owner = tl.sum(
        ((row[:, None] >= starts[None, :]) & (row[:, None] < ends[None, :]) & (req[None, :] < num_reqs)).to(tl.int32)
        * req[None, :],
        1,
    )
    active = (row < num_mapped) & (row < num_tokens)
    begin = tl.load(loc + owner, active, 0)
    end = tl.load(loc + owner + 1, active, 0)
    length = tl.load(lens + owner, active, 0)
    position = tl.where(active, length - (end - begin) + row - begin, -1)
    tl.store(requests + row, tl.where(active, owner, 0), row < num_tokens)
    tl.store(positions + row, position, row < num_tokens)
    if RING > 0:
        valid = active & (position >= 0) & (row + RING >= end)
        page = tl.load(table + owner * table_stride, valid, -1).to(tl.int64)
        slot = page * RING + position % RING
        valid &= page >= 0
    else:
        logical = tl.maximum(position, 0) // RATIO
        page_index = logical // PAGE
        valid = active & (position >= 0) & ((position + 1) % RATIO == 0) & (page_index < table_width)
        page = tl.load(table + owner * table_stride + page_index, valid, -1).to(tl.int64)
        slot = page * PAGE + logical % PAGE
        valid &= (page >= 0) & (tl.load(common_slots + row, active, -1) >= 0)
    tl.store(slots + row, tl.where(valid, slot, -1), row < num_tokens)
    if RING == 0 and RATIO > 1:
        counts = tl.where(
            (req < num_reqs) & (ends > starts), tl.maximum(seqs // RATIO - (seqs - (ends - starts)) // RATIO, 1), 0
        )
        # Avoid scan/gather lowering: prefix work offsets by a small matrix.
        work_starts = tl.sum(tl.where(req[:, None] > req[None, :], counts[None, :], 0), 1)
        work_ends = work_starts + counts
        offset = tl.program_id(0) * WORK + tl.arange(0, WORK)
        owns = (
            (offset[:, None] >= work_starts[None, :])
            & (offset[:, None] < work_ends[None, :])
            & (req[None, :] < num_reqs)
        )
        owner = tl.sum(tl.where(owns, req[None, :], 0), 1)
        owner_start = tl.sum(tl.where(owns, work_starts[None, :], 0), 1)
        valid = offset < tl.sum(counts, 0)
        tl.store(work + offset * 2, tl.where(valid, owner, -1), offset < work_rows)
        tl.store(work + offset * 2 + 1, tl.where(valid, offset - owner_start, -1), offset < work_rows)


def build_qsa_metadata(
    common,
    req_buffer,
    pos_buffer,
    slot_buffer,
    *,
    storage_block_size,
    compress_ratio,
    circular_buffer_size=0,
    k_work_metadata_buffer=None,
    request_capacity=None,
    num_reqs_actual=None,
):
    count = common.num_reqs if num_reqs_actual is None else num_reqs_actual
    if not 0 < count <= common.num_reqs:
        raise ValueError("QSA actual request count must be positive and fit the batch")
    if (
        common.seq_lens.device.type != "npu"
        or max(request_capacity or count, count) > _REQUEST_TILE
        or common.block_table_tensor.stride(1) != 1
    ):
        # Large request matrices and noncontiguous page columns retain the
        # reference path. The fused path is validated for at most 16 requests.
        from vllm_ascend.models.qwen4_exp.common.qsa_cache import _build_qsa_metadata_torch

        return _build_qsa_metadata_torch(
            common,
            req_buffer,
            pos_buffer,
            slot_buffer,
            storage_block_size=storage_block_size,
            compress_ratio=compress_ratio,
            circular_buffer_size=circular_buffer_size,
            k_work_metadata_buffer=k_work_metadata_buffer,
            request_capacity=request_capacity,
            num_reqs_actual=num_reqs_actual,
        )
    rows = common.num_actual_tokens
    mapped = int(common.query_start_loc_cpu[count])
    if mapped > rows:
        raise ValueError("QSA active query rows exceed the token capacity")
    requests, positions, slots = req_buffer[:rows], pos_buffer[:rows], slot_buffer[:rows]
    work = k_work_metadata_buffer
    work_rows = 0 if work is None else work.shape[0]
    _qsa_metadata_kernel[(max(triton.cdiv(rows, _TOKEN_TILE), triton.cdiv(work_rows, _TOKEN_TILE), 1),)](
        common.query_start_loc,
        common.seq_lens,
        common.block_table_tensor,
        common.slot_mapping,
        requests,
        positions,
        slots,
        work if work is not None else slots,
        common.block_table_tensor.stride(0),
        common.block_table_tensor.shape[1],
        count,
        rows,
        mapped,
        work_rows,
        PAGE=storage_block_size,
        RATIO=compress_ratio,
        RING=circular_buffer_size,
        TOKENS=_TOKEN_TILE,
        # Smaller request dimensions can crash the Ascend register-reduction
        # lowering. Mask a fixed 16-wide tile even for a single request.
        REQUESTS=_REQUEST_TILE,
        WORK=_TOKEN_TILE,
        num_stages=1,
        num_warps=4,
        enable_auto_bind_sub_block=False,
        disable_tightly_coupled_buffer_reuse=True,
        multibuffer=False,
        code_motion=False,
    )
    if not circular_buffer_size and compress_ratio == 1:
        slots = common.slot_mapping[:rows]
    return requests, positions, slots
