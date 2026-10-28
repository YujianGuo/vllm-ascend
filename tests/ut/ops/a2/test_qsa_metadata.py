# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic graph parity for the Ascend metadata fusion candidate."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.models.qwen4_exp.common import qsa_cache
from vllm_ascend.ops.triton.qwen4_exp.metadata import build_qsa_metadata as build_metadata


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("actual_reqs", [1, 8, 16])
@torch.inference_mode()
def test_qsa_metadata_masks_inactive_padded_rows(ring, actual_reqs, monkeypatch):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=16))
    spec = (
        AscendCircularBufferSpec(block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
        if ring
        else MLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, compress_ratio=4)
    )
    loc = torch.arange(17, dtype=torch.int32) * 4
    common = CommonAttentionMetadata(
        query_start_loc=loc.npu(),
        query_start_loc_cpu=loc,
        seq_lens=(torch.arange(16, dtype=torch.int32) * 257 + 128).npu(),
        num_reqs=16,
        num_actual_tokens=64,
        max_query_len=4,
        max_seq_len=4096,
        block_table_tensor=torch.arange(16, dtype=torch.int32, device="npu")[:, None].expand(-1, 32).contiguous(),
        slot_mapping=torch.arange(64, dtype=torch.int64, device="npu"),
    )
    builder = qsa_cache.QSAMetadataBuilder(spec, ["test"], config, torch.device("npu"))
    monkeypatch.setattr(qsa_cache, "build_qsa_metadata", qsa_cache._build_qsa_metadata_torch)
    fields = ("logical_positions", "token_to_req", "slot_mapping", "k_work_metadata")
    reference = builder.build(0, common, num_reqs_actual=actual_reqs)
    expected = {name: getattr(reference, name).cpu().clone() for name in fields}
    monkeypatch.setattr(qsa_cache, "build_qsa_metadata", build_metadata)
    actual = builder.build(0, common, num_reqs_actual=actual_reqs)
    for name in fields:
        if name == "token_to_req":
            # Inactive rows are gated by positions/slots; the fused builder
            # consistently maps them to zero, independent of graph padding.
            torch.testing.assert_close(
                getattr(actual, name)[: 4 * actual_reqs].cpu(), expected[name][: 4 * actual_reqs]
            )
            assert torch.all(getattr(actual, name)[4 * actual_reqs :] == 0)
        else:
            torch.testing.assert_close(getattr(actual, name).cpu(), expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("ring", [False, True])
@torch.inference_mode()
def test_qsa_metadata_graph_changes_lengths_and_page_owners(ring, monkeypatch):
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=16))
    spec = (
        AscendCircularBufferSpec(block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
        if ring
        else MLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, compress_ratio=4)
    )
    loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    m = CommonAttentionMetadata(
        query_start_loc=loc.npu(),
        query_start_loc_cpu=loc,
        seq_lens=torch.tensor([132, 260], device="npu", dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=8,
        max_seq_len=4096,
        block_table_tensor=torch.ones((2, 32), device="npu", dtype=torch.int32),
        slot_mapping=torch.arange(8, device="npu", dtype=torch.int64),
    )
    candidate = qsa_cache.QSAMetadataBuilder(spec, ["test"], config, torch.device("npu"))
    reference = qsa_cache.QSAMetadataBuilder(spec, ["test"], config, torch.device("npu"))
    monkeypatch.setattr(qsa_cache, "build_qsa_metadata", build_metadata)
    candidate.build(0, m)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        result = candidate.build(0, m)
    for starts, lengths, pages in [
        ([0, 4, 8], [132, 260], [2, 3]),
        ([0, 0, 8], [0, 775], [-1, 10]),
        ([0, 8, 8], [128, 0], [12, -1]),
        ([0, 3, 8], [4095, 515], [7, 8]),
    ]:
        m.query_start_loc_cpu.copy_(torch.tensor(starts, dtype=torch.int32))
        m.query_start_loc.copy_(m.query_start_loc_cpu)
        m.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        m.block_table_tensor.copy_(torch.tensor(pages, dtype=torch.int32)[:, None].expand(-1, 32))
        for _ in range(64):
            graph.replay()
        torch.npu.synchronize()
        monkeypatch.setattr(qsa_cache, "build_qsa_metadata", qsa_cache._build_qsa_metadata_torch)
        cm = copy.copy(m)
        cm._token_to_req_indices_cache = None
        expected = reference.build(0, cm)
        for field in ("logical_positions", "token_to_req", "slot_mapping", "k_work_metadata"):
            torch.testing.assert_close(getattr(result, field).cpu(), getattr(expected, field).cpu(), rtol=0, atol=0)
