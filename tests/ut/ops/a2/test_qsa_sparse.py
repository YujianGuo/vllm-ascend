# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_sparse_paged_attention


def _make_case(rows, context, kv_heads, dimension):
    torch.manual_seed(41)
    query = torch.randn(rows, kv_heads * 3 + 1, dimension, dtype=torch.bfloat16)[:, :-1]
    key = torch.randn(7, 768, kv_heads, dimension + 16, dtype=torch.bfloat16)[..., :dimension]
    value = torch.randn_like(key)
    table = torch.tensor([[6, 2, 4, 1], [3, 5, 0, 2]], dtype=torch.int32)
    requests = torch.arange(rows, dtype=torch.int32) % 2
    indices = torch.full((rows, 2051), -1, dtype=torch.int32)
    for row in range(rows):
        count = max(context - row % 4, 0)
        indices[row, :count] = torch.randperm(context, dtype=torch.int32)[:count]
    if rows > 1:
        requests[-1] = -1
    if rows > 3:
        # A valid logical token whose physical page is invalid must be masked.
        table[1, 1] = -1
    cpu = (query, key, value, indices, table, requests)
    device = []
    for tensor in cpu:
        copy = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="npu")
        copy.copy_(tensor)
        device.append(copy)
    return cpu, device


def _reference(args):
    query, key, value, indices, table, requests = args
    result = torch.zeros_like(query, dtype=torch.float32)
    for row, request in enumerate(requests.tolist()):
        if not 0 <= request < table.shape[0]:
            continue
        selected = indices[row]
        selected = selected[selected >= 0].long()
        pages = table[request, selected // key.shape[1]].long()
        valid = (pages >= 0) & (pages < key.shape[0])
        selected, pages = selected[valid], pages[valid]
        if not selected.numel():
            continue
        for head in range(query.shape[1]):
            kv_head = head // (query.shape[1] // key.shape[2])
            keys = key[pages, selected % key.shape[1], kv_head].float()
            values = value[pages, selected % key.shape[1], kv_head].float()
            score = keys @ query[row, head].float() / query.shape[-1] ** 0.5
            result[row, head] = score.softmax(0) @ values
    return result


@pytest.mark.parametrize(
    "rows,context,kv_heads,dimension",
    [
        (1, 0, 1, 128),
        (1, 1, 1, 256),
        (8, 63, 2, 128),
        (8, 65, 1, 256),
        (32, 128, 1, 256),
        (64, 384, 1, 256),
        (8, 769, 2, 128),
        (32, 2051, 1, 256),
    ],
)
@torch.inference_mode()
def test_sparse_qsa_strided_768_token_pages_match_fp32(rows, context, kv_heads, dimension):
    cpu, device = _make_case(rows, context, kv_heads, dimension)
    actual = qsa_sparse_paged_attention(*device).cpu()
    # The kernel rounds probabilities to BF16 before the value reduction,
    # matching the prior Cube path. Compare against unrounded FP32 softmax.
    torch.testing.assert_close(actual.float(), _reference(cpu), rtol=0.02, atol=0.005)


@torch.inference_mode()
def test_sparse_qsa_graph_changes_live_tiles_and_page_ownership():
    cpu, device = _make_case(8, 769, 1, 256)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        qsa_sparse_paged_attention(*device)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = qsa_sparse_paged_attention(*device)
    for context in (128, 0, 2051, 65, 769):
        cpu[3].fill_(-1)
        cpu[3][:, :context] = torch.arange(context, dtype=torch.int32)
        cpu[4].copy_(cpu[4].flip(0))
        cpu[0].mul_(-1)
        for host, target in zip(cpu, device):
            target.copy_(host)
        for _ in range(16):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu().float(), _reference(cpu), rtol=0.02, atol=0.005)
