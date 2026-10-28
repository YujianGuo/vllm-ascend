# SPDX-License-Identifier: Apache-2.0
"""Greedy verification of compressed top-k logits must emit vocabulary IDs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_greedy_verification_maps_compressed_columns_to_vocabulary_ids():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/sample/rejection_sampler.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "rejection_sample"
    )
    outer = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not sampling_metadata.all_random"
    )
    branch = ast.Module(body=[outer.body[0]], type_ignores=[])
    code = compile(ast.fix_missing_locations(branch), str(source), "exec")
    logits = torch.tensor([[9.0, 1.0, 3.0], [2.0, 8.0, 4.0]])
    indices = torch.tensor([[248046, 17, 140], [99999, 123456, 900]])
    namespace = dict(
        target_logits=logits,
        target_indices=indices,
        greedy_token_ids=None,
        get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=True),
        greedy_sample=lambda value: value.argmax(-1),
    )
    exec(code, namespace)
    assert namespace["target_argmax"].tolist() == [248046, 123456]
    # Pure-greedy requests still use the original distributed argmax branch.
    namespace["target_indices"] = None
    exec(code, namespace)
    assert namespace["target_argmax"].tolist() == [0, 1]
