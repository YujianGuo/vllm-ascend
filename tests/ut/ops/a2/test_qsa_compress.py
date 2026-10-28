# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops import qsa_compress_groups_with_ratio as reference
from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_compress_groups_with_ratio as candidate


def _make_case(lengths, starts, ring, with_rope, strided, inactive):
    rows = sum(lengths)
    count = len(lengths)
    raw = torch.randn(rows, 1, 256 if strided else 128, dtype=torch.bfloat16)
    if strided:
        raw = raw[..., ::2]
    raw_positions = torch.zeros(rows, 1, 3, dtype=torch.int64)
    qloc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    req = torch.repeat_interleave(torch.arange(count, dtype=torch.int32), torch.tensor(lengths))
    logical = torch.cat([torch.arange(start, start + length) for start, length in zip(starts, lengths)])
    for axis in range(3):
        raw_positions[:, 0, axis] = logical + axis * 17
    slots = torch.where((logical + 1) % 4 == 0, logical // 4, -1).to(torch.int64)
    pages = count + 2
    cache = torch.randn(pages, ring, 1, 136 if strided else 128, dtype=torch.bfloat16)[..., :128]
    table = torch.arange(1, count + 1, dtype=torch.int32)[:, None].expand(-1, 5).clone()
    rope = torch.zeros(pages, ring, 1, 6 if strided else 3, dtype=torch.int64)
    if strided:
        rope = rope[..., ::2]
    for i, start in enumerate(starts):
        for position in range(max(0, start - ring), start):
            rope[i + 1, position % ring, 0] = torch.tensor([position, position + 17, position + 34])
    if inactive:
        req[-1] = -1
        slots[-1] = -1
    args = [raw, raw_positions, cache, table, req, qloc, logical, slots, 4, rope if with_rope else None]
    npu_args = []
    for value in args:
        if isinstance(value, torch.Tensor):
            # Preserve noncontiguous views, including packed-page stride.
            device_value = torch.empty_strided(value.shape, value.stride(), device="npu", dtype=value.dtype)
            device_value.copy_(value)
            npu_args.append(device_value)
        else:
            npu_args.append(value)
    return args, npu_args


@pytest.mark.parametrize(
    "lengths,starts",
    [
        ([1], [128]),
        ([4], [0]),
        ([4], [126]),
        ([4, 4], [125, 255]),
        ([16, 16], [119, 507]),
        ([32, 32], [4095, 8191]),
        ([257], [5]),
    ],
)
@pytest.mark.parametrize("with_rope", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_qsa_compress_paged_history_matches_cpu(lengths, starts, with_rope, strided):
    torch.manual_seed(17)
    args, npu_args = _make_case(lengths, starts, 8, with_rope, strided, True)
    output = candidate(*npu_args)
    for actual, expected in zip(output, reference(*args)):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_compress_graph_reuses_changed_current_and_history_keys():
    torch.manual_seed(17)
    args, npu_args = _make_case([4, 4], [125, 255], 8, True, True, True)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        candidate(*npu_args)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = candidate(*npu_args)
    for scale in (2, 0, 3):
        changed = args.copy()
        changed[0] = args[0] * scale
        changed[2] = args[2] * scale
        npu_args[0].copy_(changed[0])
        npu_args[2].copy_(changed[2])
        graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(output, reference(*changed)):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
