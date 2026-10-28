# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_mqa_paged


def _inputs(rows, context):
    torch.manual_seed(29)
    # Noncontiguous query and cache views exercise the paged strides.
    query = torch.randn(rows, 5, 128, dtype=torch.bfloat16, device="npu")[:, :4]
    cache = torch.randn(1024, 32, 2, 128, dtype=torch.bfloat16, device="npu")[:, :, :1]
    table = torch.randperm(1024, dtype=torch.int32, device="npu").view(1, -1).repeat(4, 1)
    requests = torch.arange(rows, dtype=torch.int32, device="npu") % 4
    if rows > 1:
        requests[-1] = -1
    positions = torch.full((rows,), context - 1, dtype=torch.int32, device="npu")
    lengths = torch.tensor([context, context - 1, context // 2, 0], dtype=torch.int32, device="npu")
    return query, cache, table, requests, positions, lengths


def _check_reference(inputs, output):
    query, cache, table, requests, positions, lengths = inputs
    logits, visible = output
    requests = requests.cpu().tolist()
    positions = positions.cpu().tolist()
    lengths = lengths.cpu().tolist()
    for row in sorted({0, 1, 2, 3, len(requests) - 1}):
        if row >= len(requests):
            continue
        request = requests[row]
        count = min((positions[row] + 1) // 4, lengths[request] // 4) if request >= 0 else 0
        assert visible[row].item() == count
        assert torch.isneginf(logits[row, count:]).all().item()
        if count:
            keys = cache[table[request].long()].reshape(-1, 128)[:count].float()
            expected = (keys @ query[row].float().T).relu().sum(-1) / math.sqrt(128)
            torch.testing.assert_close(logits[row, :count], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rows,context", [(1, 384), (8, 128), (32, 384), (64, 8192), (64, 131072), (256, 384)])
@torch.inference_mode()
def test_qsa_mqa_matches_paged_fp32_reference(rows, context):
    inputs = _inputs(rows, context)
    output = qsa_mqa_paged(*inputs, compress_ratio=4)
    torch.npu.synchronize()
    _check_reference(inputs, output)


@torch.inference_mode()
def test_qsa_mqa_graph_changes_live_context_without_stale_logits():
    inputs = _inputs(32, 128)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        qsa_mqa_paged(*inputs, compress_ratio=4)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = qsa_mqa_paged(*inputs, compress_ratio=4)
    for context in (128, 8192, 131072, 4, 384):
        inputs[4].fill_(context - 1)
        inputs[5].fill_(context)
        graph.replay()
        torch.npu.synchronize()
        _check_reference(inputs, output)
