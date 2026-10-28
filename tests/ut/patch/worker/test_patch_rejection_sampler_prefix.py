# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the async rejection-output parser patch."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from vllm.v1.outputs import LogprobsTensors


REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "patch_rejection_sampler",
    REPO / "vllm_ascend/patch/worker/patch_rejection_sampler.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parser_keeps_only_contiguous_prefix():
    output = torch.tensor([[101, -1, 202, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    parsed, logprobs = module._parse_output_contiguous_prefix(output, vocab_size=1000)
    assert parsed == [[101], []]
    assert logprobs is None


def test_parser_filters_invalid_ids_and_discarded_rows():
    output = torch.tensor([[101, 1001, 202], [303, -1, 404]], dtype=torch.int32)
    parsed, _ = module._parse_output_contiguous_prefix(
        output, vocab_size=1000, discard_req_indices=np.asarray([1], dtype=np.int64)
    )
    assert parsed == [[101], []]


def test_parser_keeps_logprob_cumulative_lengths_in_sync():
    output = torch.tensor([[101, -1, 202], [303, 304, -1]], dtype=torch.int32)
    tensors = LogprobsTensors(
        torch.tensor([[101], [0], [0], [303], [304], [0]], dtype=torch.int32),
        torch.tensor([[-0.1], [0.0], [0.0], [-0.2], [-0.3], [0.0]], dtype=torch.float32),
        torch.tensor([1, 0, 0, 2, 3, 0], dtype=torch.int32),
    )
    parsed, logprobs = module._parse_output_contiguous_prefix(
        output, vocab_size=1000, logprobs_tensors=tensors
    )
    assert parsed == [[101], [303, 304]]
    assert logprobs is not None
    assert logprobs.cu_num_generated_tokens == [0, 1, 3]
