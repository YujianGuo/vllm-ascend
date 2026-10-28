# SPDX-License-Identifier: Apache-2.0
"""Regression for padded-selector vector expansion at decode row counts."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def production_expander():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/qwen4_exp/qsa.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_expand_qsa_padded_blocks_npu")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_QSA_TENSOR_EXPAND_MIN_ROWS" for target in node.targets
            )
        )
    ]
    calls = []

    class Tail:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                # Cases end at complete groups, so the tail is empty.
                calls.append("tail")

            return launch

    def fallback(*args):
        calls.append("fallback")
        return args[-1]

    namespace = dict(
        torch=torch,
        triton=SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
        _expand_qsa_tail_kernel=Tail(),
        expand_qsa_block_indices_npu=fallback,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace["_expand_qsa_padded_blocks_npu"], calls


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_decode_sizes_expand_masked_blocks_with_tensor_path(production_expander, rows):
    expand, calls = production_expander
    blocks = torch.tensor([[3, 0, -1, -1]], dtype=torch.int32).repeat(rows, 1)
    output = torch.full((rows, 19), 12345, dtype=torch.int32)
    actual = expand(blocks, torch.full((rows,), 15), torch.full((rows,), 16), torch.arange(rows), 4, 16, output)
    expected = torch.tensor([[12, 13, 14, 15, 0, 1, 2, 3] + [-1] * 11], dtype=torch.int32).repeat(rows, 1)
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)
    assert calls == ["tail"]
    blocks.fill_(-1)
    expand(blocks, torch.full((rows,), 15), torch.zeros(rows), torch.full((rows,), -1), 4, 16, output)
    assert torch.all(output == -1)


def test_noncontiguous_selector_retains_stride_aware_fallback(production_expander):
    expand, calls = production_expander
    blocks = torch.zeros((4, 8), dtype=torch.int32).transpose(0, 1)
    output = torch.zeros((8, 19), dtype=torch.int32)
    actual = expand(blocks, torch.full((8,), 15), torch.full((8,), 16), torch.arange(8), 4, 16, output)
    assert actual is output
    assert calls == ["fallback"]
