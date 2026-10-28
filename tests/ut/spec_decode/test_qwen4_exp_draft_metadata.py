# SPDX-License-Identifier: Apache-2.0
"""Verify cache-owner isolation and rejected-token advancement on CPU."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata as CommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm_ascend.spec_decode.eagle_proposer import AscendQwen4ExpMTPProposer


def make_proposer():
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4))
    specs = [
        FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256, dtype=torch.bfloat16),
        MLAAttentionSpec(block_size=768, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, compress_ratio=4),
        AscendCircularBufferSpec(block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16),
    ]
    groups = []
    for name, gid, spec in zip(("main", "compressed", "raw"), (1, 1, 0), specs):
        builder = QSAMetadataBuilder(spec, [name], config, torch.device("cpu"))
        groups.append(
            SimpleNamespace(layer_names=[name], kv_cache_group_id=gid, get_metadata_builder=lambda b=builder: b)
        )
    tables = {1: torch.tensor([[2, 4], [3, 5]], dtype=torch.int32), 0: torch.tensor([[10], [11]], dtype=torch.int32)}
    proposer = SimpleNamespace(
        _step_metadata_builders={},
        draft_attn_groups=groups,
        num_speculative_tokens=3,
        use_cuda_graph=False,
        compact_draft_steps=True,
        _per_group_block_tables=tables,
        kv_cache_gid=1,
        block_size=768,
        max_model_len=4096,
        query_start_loc_group=[torch.empty(65, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.empty(64, dtype=torch.int32) for _ in range(3)],
        slot_mapping_group=[torch.empty(64, dtype=torch.int64) for _ in range(3)],
        arange=torch.arange(65, dtype=torch.int32),
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([132, 260], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=260,
        block_table_tensor=tables[1],
        slot_mapping=torch.tensor([1664, 1665, 1666, 1667, 2560, 2561, 2562, 2563]),
    )
    # The target builder has already cached its four-token-per-request layout.
    common.token_to_req_indices(torch.empty(8, dtype=torch.int32))
    proposer._build_qwen_draft_metadata = MethodType(AscendQwen4ExpMTPProposer._build_qwen_draft_metadata, proposer)
    return proposer, common


def test_qwen_draft_metadata_isolates_owners_and_steps():
    proposer, common = make_proposer()
    steps, _ = AscendQwen4ExpMTPProposer.build_draft_attn_metadata(proposer, common, 8, 8, 2)
    steps.extend(
        AscendQwen4ExpMTPProposer.build_later_draft_attn_metadata(
            proposer, common, 2, 2, torch.tensor([128, 257]), None
        )
    )
    torch.testing.assert_close(
        steps[0]["raw"].logical_positions, torch.tensor([128, 129, 130, 131, 256, 257, 258, 259])
    )
    for index in (1, 2):
        expected = torch.tensor([128, 257]) + index
        for name in ("main", "compressed", "raw"):
            torch.testing.assert_close(steps[index][name].logical_positions, expected)
            torch.testing.assert_close(steps[index][name].token_to_req, torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(steps[index]["main"].slot_mapping[:2], torch.tensor([2, 3]) * 768 + expected)
        torch.testing.assert_close(steps[index]["raw"].slot_mapping, torch.tensor([10, 11]) * 8 + expected % 8)
    for name in ("main", "compressed", "raw"):
        assert len({step[name].logical_positions.data_ptr() for step in steps}) == 3


def test_graph_metadata_keeps_owners_and_addresses_after_request_completion():
    proposer, common = make_proposer()
    proposer.use_cuda_graph = True
    pointers = []
    for active, positions in ((2, [128, 257]), (1, [383]), (2, [767, 257])):
        common.seq_lens[:active] = torch.tensor(positions) + 4
        steps, _ = AscendQwen4ExpMTPProposer.build_draft_attn_metadata(proposer, common, 8, active * 4, active)
        steps.extend(
            AscendQwen4ExpMTPProposer.build_later_draft_attn_metadata(
                proposer,
                common,
                active,
                8,
                torch.tensor(positions),
                CUDAGraphMode.FULL,
            )
        )
        for index in (1, 2):
            for owner in ("raw", "main", "compressed"):
                metadata = steps[index][owner]
                torch.testing.assert_close(metadata.logical_positions[:active], torch.tensor(positions) + index)
                assert torch.all(metadata.logical_positions[active:] == -1)
                assert torch.all(metadata.slot_mapping[active:2] == -1)
        pointers.append(
            {
                (index, owner, field): getattr(step[owner], field).data_ptr()
                for index, step in enumerate(steps)
                for owner in ("raw", "main", "compressed")
                for field in ("logical_positions", "token_to_req", "slot_mapping", "seq_lens", "query_start_loc")
            }
        )
    assert all(pointer == pointers[0] for pointer in pointers)


@pytest.mark.parametrize("enabled,greedy", [(True, True), (True, False), (False, True)])
def test_repeat_candidates_use_accepted_row_before_buffer_reuse(monkeypatch, enabled, greedy):
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    proposer = SimpleNamespace(use_repeated_proposals=enabled, _qwen_probabilistic=False)
    # Request 0 rejected both later rows, request 1 accepted its first draft.
    target_ids = torch.tensor([42, 9, 10, 11, 6, 77, 8, 9])
    next_ids = torch.tensor([42, 80])
    proposals = torch.tensor([[2, 3, 4], [5, 6, 7]])

    def draft(*args, **kwargs):
        target_ids.fill_(0)
        return proposals.clone()

    monkeypatch.setattr(AscendSpecDecodeBaseProposer, "_propose", draft)
    result = AscendQwen4ExpMTPProposer._propose(
        proposer,
        target_ids,
        None,
        None,
        next_ids,
        torch.tensor([0, 5]),
        None,
        None,
        SimpleNamespace(all_greedy=greedy),
    )
    expected = proposals.clone()
    if enabled and greedy:
        expected[0].fill_(42)
    torch.testing.assert_close(result, expected)
