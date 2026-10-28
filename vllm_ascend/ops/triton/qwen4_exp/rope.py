# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _qsa_rope_kernel(
    x_ptr,
    positions_ptr,
    cache_ptr,
    out_ptr,
    ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    X_ROW: tl.constexpr,
    X_HEAD: tl.constexpr,
    X_DIM: tl.constexpr,
    POS_AXIS: tl.constexpr,
    POS_ROW: tl.constexpr,
    CACHE_ROW: tl.constexpr,
    CACHE_DIM: tl.constexpr,
    MROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    NEOX: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    within_row = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = within_row < HEADS * DIM
    head = within_row // DIM
    col = within_row % DIM
    offsets = row * HEADS * DIM + within_row
    half = ROTARY_DIM // 2
    if NEOX:
        freq = col % half
        first = col < half
        partner = tl.where(first, col + half, col - half)
    else:
        freq = col // 2
        first = col % 2 == 0
        partner = col ^ 1
    axis = tl.full((BLOCK,), 0, tl.int32)
    if MROPE:
        if INTERLEAVED:
            axis = tl.where((freq % 3 == 1) & (freq < SECTION_H * 3), 1, axis)
            axis = tl.where((freq % 3 == 2) & (freq < SECTION_W * 3), 2, axis)
        else:
            axis = tl.where(freq < SECTION_T, 0, tl.where(freq < SECTION_T + SECTION_H, 1, 2))
    pos0 = tl.load(positions_ptr + row * POS_ROW)
    if MROPE:
        pos1 = tl.load(positions_ptr + POS_AXIS + row * POS_ROW)
        pos2 = tl.load(positions_ptr + 2 * POS_AXIS + row * POS_ROW)
        pos = tl.where(axis == 0, pos0, tl.where(axis == 1, pos1, pos2))
    else:
        pos = pos0
    # Match torch indexing for the -1 position used by inactive compression rows.
    pos = tl.where(pos < 0, pos + CACHE_ROWS, pos)
    cos = tl.load(cache_ptr + pos * CACHE_ROW + freq * CACHE_DIM, valid & (col < ROTARY_DIM), 0).to(tl.float32)
    sin = tl.load(cache_ptr + pos * CACHE_ROW + (freq + half) * CACHE_DIM, valid & (col < ROTARY_DIM), 0).to(tl.float32)
    x = tl.load(x_ptr + row * X_ROW + head * X_HEAD + col * X_DIM, valid, 0)
    other = tl.load(x_ptr + row * X_ROW + head * X_HEAD + partner * X_DIM, valid & (col < ROTARY_DIM), 0)
    # The reference rounds each product to the input dtype before add/sub.
    xc = (x.to(tl.float32) * cos).to(x.dtype).to(tl.float32)
    ys = (other.to(tl.float32) * sin).to(x.dtype).to(tl.float32)
    rotated = tl.where(first, xc - ys, xc + ys).to(x.dtype)
    tl.store(out_ptr + offsets, tl.where(col < ROTARY_DIM, rotated, x), valid)


def qsa_apply_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
    mrope_section: tuple[int, int, int] | None = None,
    mrope_interleaved: bool = False,
) -> torch.Tensor:
    """Fuse QSA RoPE gather/composition while preserving intermediate rounding."""
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    if not tensor.numel():
        return out
    sections = mrope_section or (0, 0, 0)
    block_size = triton.next_power_of_2(tensor.shape[2])
    _qsa_rope_kernel[(tensor.shape[0], triton.cdiv(tensor.shape[1] * tensor.shape[2], block_size))](
        tensor,
        positions,
        cache,
        out,
        *tensor.shape,
        rotary_dim,
        cache.shape[0],
        *tensor.stride(),
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.stride(-1),
        *cache.stride(),
        positions.ndim == 2,
        mrope_interleaved,
        is_neox_style,
        *sections,
        BLOCK=block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
