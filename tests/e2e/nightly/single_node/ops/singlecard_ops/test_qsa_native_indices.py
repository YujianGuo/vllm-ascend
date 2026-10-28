# SPDX-License-Identifier: Apache-2.0
"""Native index compaction preserves order, duplicates and invalid-row masks."""
import pytest
import torch

from vllm_ascend.ops.triton.qwen4_exp.native_indices import compact_native_indices


@pytest.mark.parametrize("columns", [1, 63, 512, 2051, 4096])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_native_indices_exact_and_graph_reuse(columns, strided):
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(20260907)
    source = torch.randint(-800, 17000, (17, columns * 2), generator=rng, dtype=torch.int32)
    source[:, :8] = torch.tensor([0, 0, -1, 767, 768, 9999999, 769, -900])[: min(8, columns * 2)]
    host_indices = source[:, ::2] if strided else source[:, :columns].contiguous()
    host_table = torch.tensor([[0, 1, -1, 999999, 4, 5, 6, 7]] * 4, dtype=torch.int32)
    host_requests = torch.tensor([-1, 0, 1, 2, 3, 4, 999] + [0] * 10, dtype=torch.int32)
    indices = source.npu()[:, ::2] if strided else host_indices.npu()
    table_storage = host_table.repeat_interleave(2, dim=1).npu()
    table = table_storage[:, ::2] if strided else host_table.npu()
    requests = host_requests.npu()

    def launch():
        return compact_native_indices(indices, table, requests, 8, 768)

    for _ in range(3):
        launch()
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output, nonempty = launch()
    for shift in [0, 1, 3]:
        new = host_indices.roll(shift, dims=1)
        indices.copy_(new.npu())
        graph.replay()
        actual, mask = output.cpu(), nonempty.cpu()
        expected = torch.full_like(new, -1)
        for row, request in enumerate(host_requests.tolist()):
            valid = []
            if 0 <= request < host_table.shape[0]:
                for token in new[row].tolist():
                    page = token // 768
                    if token >= 0 and page < host_table.shape[1] and 0 <= host_table[request, page] < 8:
                        valid.append(token)
            expected[row, :len(valid)] = torch.tensor(valid, dtype=expected.dtype)
        assert torch.equal(actual, expected)
        assert torch.equal(mask, (expected >= 0).any(-1))
