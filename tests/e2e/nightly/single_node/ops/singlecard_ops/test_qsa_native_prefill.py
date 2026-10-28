# SPDX-License-Identifier: Apache-2.0
"""Paged native QSA must preserve attention, request isolation, and graph inputs."""

import pytest
import torch

from vllm_ascend.ops.qwen_qsa_native import qsa_native_prefill


def _reference(query, key, value, indices, table, requests):
    query, key, value, indices, table, requests = [
        tensor.cpu() for tensor in (query, key, value, indices, table, requests)
    ]
    output = torch.zeros_like(query, dtype=torch.float32)
    bound = torch.zeros_like(output)
    # BF16 roundoff in unnormalized probabilities and final output:
    # bound the former by the probability-weighted absolute values, including
    # margin for online normalization and float32 reduction/exp roundoff.
    unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
    for row, request in enumerate(requests.tolist()):
        if not 0 <= request < table.shape[0]:
            continue
        tokens = indices[row]
        valid = (tokens >= 0) & (tokens // key.shape[1] < table.shape[1])
        tokens = tokens[valid].long()
        physical = table[request, tokens // key.shape[1]].long()
        valid = (physical >= 0) & (physical < key.shape[0])
        tokens, physical = tokens[valid], physical[valid]
        if not tokens.numel():
            continue
        selected_k = key[physical, tokens % key.shape[1], 0].float()
        selected_v = value[physical, tokens % key.shape[1], 0].float()
        probabilities = (query[row].float() @ selected_k.T / 16).softmax(-1)
        output[row] = probabilities @ selected_v
        bound[row] = 4 * unit_roundoff * (probabilities @ selected_v.abs()) + 2e-6
    return output, bound


@pytest.mark.parametrize("scenario", ["mixed", "padding", "empty", "invalid_pages", "short"])
@torch.inference_mode()
def test_native_qsa_prefill_attention_and_changing_graph_inputs(scenario):
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(20260906)
    query = torch.randn(16, 3, 256, dtype=torch.bfloat16, generator=generator).npu()
    key = torch.randn(16, 768, 1, 256, dtype=query.dtype, generator=generator).npu()
    value = torch.randn(key.shape, dtype=key.dtype, generator=generator).npu()
    table = torch.randperm(16, generator=generator).int().repeat(4, 1).npu()
    starts = torch.tensor([0, 1, 5, 8, 16], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([8000, 8001, 8002, 8003], dtype=torch.int32, device="npu")
    requests = torch.tensor([0] + [1] * 4 + [2] * 3 + [3] * 8, dtype=torch.int32, device="npu")
    indices = torch.randperm(4096, generator=generator)[:2051].int().repeat(16, 1).npu()
    if scenario == "padding":
        starts.copy_(torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device="npu"))
        seq_lens[3] = 0
        requests.copy_(torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [-1] * 4, dtype=requests.dtype, device="npu"))
        indices[12:] = -1
    elif scenario == "empty":
        indices[5] = -1
    elif scenario == "invalid_pages":
        table[:, 2] = -1
        table[:, 3] = 99999
        indices[:, 31] = 999999
    elif scenario == "short":
        indices[:, 63:] = -1

    def launch():
        return qsa_native_prefill(
            query, key, value, indices, table, starts, seq_lens, requests, torch.empty_like(query)
        )

    launch()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = launch()
    for _ in range(3):
        query.copy_(torch.randn(query.shape, dtype=query.dtype, generator=generator).npu())
        key.copy_(torch.randn(key.shape, dtype=key.dtype, generator=generator).npu())
        value.copy_(torch.randn(value.shape, dtype=value.dtype, generator=generator).npu())
        table.copy_(table.roll(1, dims=1))
        indices.copy_(torch.where(indices >= 0, (indices + 31) % 4096, indices))
        graph.replay()
        torch.npu.synchronize()
        actual = output.cpu().float()
        eager = launch().cpu().float()
        assert torch.equal(actual, eager)
        expected, bound = _reference(query, key, value, indices, table, requests)
        assert torch.isfinite(actual).all()
        assert torch.all((actual - expected).abs() <= bound), (scenario, (actual - expected).abs().max())
