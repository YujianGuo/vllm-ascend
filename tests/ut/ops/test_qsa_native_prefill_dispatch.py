# SPDX-License-Identifier: Apache-2.0
"""Keep unsupported layouts and small QSA forwards on the existing path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("rows", "context", "query_length", "kv_heads", "device_type", "expected"),
    [
        (128, 8192, 128, 1, "A3", "triton"),
        (256, 512, 256, 1, "A3", "triton"),
        (256, 8192, 256, 1, "A3", "native"),
        (4096, 8192, 4096, 1, "A3", "native"),
        (4096, 8192, 4096, 2, "A3", "triton"),
        (4096, 8192, 4096, 1, "A2", "triton"),
        (4096, 0, 4096, 1, "A3", "native"),
        (4096, 0, 256, 1, "A3", "triton"),
        (1024, 0, 1024, 1, "A3", "native"),
        (1023, 0, 1023, 1, "A3", "triton"),
        (4096, 0, None, 1, "A3", "triton"),
    ],
)
def test_native_prefill_dispatch_preserves_metadata_and_output(
    rows, context, query_length, kv_heads, device_type, expected
):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/qsa.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendQSAImpl")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_qsa")
    calls = []

    def native(*args):
        calls.append(("native", args))
        return args[-1]

    def triton(*args):
        calls.append(("triton", args))
        return args[-1]

    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), forward]
    namespace = dict(
        torch=torch,
        _QSA_NATIVE_PREFILL_MIN_ROWS=256,
        _QSA_NATIVE_PREFILL_MIN_CONTEXT=1024,
        get_ascend_device_type=lambda: device_type,
        AscendDeviceType=SimpleNamespace(A3="A3"),
        canonicalize_singleton_dim_strides=lambda value: value,
        _split_qsa_kv_cache=lambda value, width: value,
        qsa_native_prefill=native,
        qsa_sparse_paged_attention=triton,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    query = torch.zeros(rows, 3, 256, dtype=torch.bfloat16)
    output = torch.ones_like(query)
    key = torch.zeros(1, 768, kv_heads, 256, dtype=query.dtype)
    cache = (key, torch.zeros_like(key))
    layer = SimpleNamespace(topk_indices_buffer=torch.zeros(rows, 2051, dtype=torch.int32))
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        max_seq_len=context,
        max_query_len=query_length,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens=torch.tensor([context], dtype=torch.int32),
    )
    request = torch.zeros(rows, dtype=torch.int32)
    owner = SimpleNamespace(head_size=256, num_heads=3, num_kv_heads=kv_heads)
    actual = namespace["forward_qsa"](owner, layer, query, None, None, cache, metadata, output, request)
    assert len(calls) == 1 and calls[0][0] == expected
    assert actual.data_ptr() == output.data_ptr()
    assert metadata.max_seq_len == context
    if expected == "native":
        assert calls[0][1][5] is metadata.query_start_loc
        assert calls[0][1][6] is metadata.seq_lens
    assert not torch.count_nonzero(output)
