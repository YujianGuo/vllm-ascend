# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Bit-exact BF16 gather from registered Host memory, Ascend implementation.

The Ascend Triton backend cannot cast a *runtime* integer to a pointer:
``tl.cast(x, tl.pointer_type(...))`` on a runtime value fails with
``Casting pointers with unmatched bitwidth`` even for a plain device address.
The base address is therefore baked in as a ``tl.constexpr`` while the row
indices stay runtime values. A registered Host region is a stable 42-bit
device alias for the process lifetime, so one specialization is compiled per
process and reused by every graph replay.

Byte spans beyond 2**32 are addressed with int64 row indices, so the per-rank
shard does not need the chunked pointer table the CUDA/A3 scheme uses.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _host_gather(
    ids,
    output,
    rows,
    BASE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids + row)
    valid = (index >= 0) & (index < rows)
    local = tl.where(valid, index, 0)
    base = tl.cast(BASE, tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, BLOCK)
    value = tl.load(base + local * DIM + col, valid & (col < DIM), 0)
    tl.store(output + row * DIM + col, value, col < DIM)


def ple_host_gather(
    ids: torch.Tensor,
    output: torch.Tensor,
    base: int,
    rows: int,
    dim: int,
) -> torch.Tensor:
    """Gather ``ids`` rows out of the Host region mapped at ``base``.

    ``ids`` holds *local* row indices already reduced into this rank's
    share; out-of-range entries read row 0 and are zeroed by the caller.
    """
    count = ids.numel()
    if count:
        if output.shape[0] != count or output.shape[1] != dim:
            raise ValueError(
                f"PLE Host gather output must be [{count}, {dim}], got {tuple(output.shape)}"
            )
        _host_gather[(count,)](
            ids,
            output,
            rows,
            BASE=base,
            DIM=dim,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=4,
        )
    return output


def _fake(
    ids: torch.Tensor,
    output: torch.Tensor,
    base: int,
    rows: int,
    dim: int,
) -> torch.Tensor:
    return output


direct_register_custom_op(
    op_name="qwen_ple_host_gather",
    op_func=ple_host_gather,
    mutates_args=["output"],
    fake_impl=_fake,
)
