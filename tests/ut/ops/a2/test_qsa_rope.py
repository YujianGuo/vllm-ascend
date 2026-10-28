# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.qsa import apply_qsa_rope
from vllm_ascend.ops.triton.qwen4_exp.rope import qsa_apply_rope


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,heads", [(1, 1), (8, 4), (32, 4), (64, 1), (257, 4)])
@pytest.mark.parametrize("mode", ["1d", "interleaved", "sections"])
@pytest.mark.parametrize("neox", [True, False])
@torch.inference_mode()
def test_qsa_rope_preserves_product_rounding(dtype, rows, heads, mode, neox):
    """Cache gather and partial rotary must match the unfused CPU formula exactly."""
    torch.manual_seed(17)
    cache = torch.randn(4096, 64, dtype=dtype)
    x = torch.randn(rows, heads + 1, 128, dtype=dtype)[:, :heads]
    positions = torch.randint(0, 4096, (rows, 3)).T
    positions[:, -1] = -1  # Inactive compression row uses torch negative indexing.
    if mode == "1d":
        positions = positions[0]
    section = (11, 11, 10)
    rotary = SimpleNamespace(
        rotary_dim=64,
        is_neox_style=neox,
        mrope_section=section,
        mrope_interleaved=mode == "interleaved",
        _match_cos_sin_cache_dtype=lambda _: cache,
    )
    expected = apply_qsa_rope(rotary, positions, x)
    # Preserve the sliced projection and transposed position strides on device.
    x_npu = torch.empty(rows, heads + 1, 128, dtype=dtype, device="npu")[:, :heads]
    x_npu.copy_(x)
    positions_npu = positions.T.contiguous().npu().T if positions.ndim == 2 else positions.npu()
    actual = qsa_apply_rope(x_npu, positions_npu, cache.npu(), 64, neox, section, mode == "interleaved")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_rope_graph_reads_current_positions():
    """A captured small-batch kernel must not retain prior request positions."""
    torch.manual_seed(17)
    x = torch.randn(32, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(4096, 64, dtype=x.dtype, device="npu")
    positions = torch.randint(0, 4096, (32, 3), device="npu").T
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        qsa_apply_rope(x, positions, cache, 64, True, (11, 11, 10), True)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = qsa_apply_rope(x, positions, cache, 64, True, (11, 11, 10), True)
    for _ in range(3):
        positions.copy_(torch.randint(0, 4096, positions.shape, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        rotary = SimpleNamespace(
            rotary_dim=64,
            is_neox_style=True,
            mrope_section=(11, 11, 10),
            mrope_interleaved=True,
            _match_cos_sin_cache_dtype=lambda _: cache.cpu(),
        )
        torch.testing.assert_close(actual.cpu(), apply_qsa_rope(rotary, positions.cpu(), x.cpu()), rtol=0, atol=0)
