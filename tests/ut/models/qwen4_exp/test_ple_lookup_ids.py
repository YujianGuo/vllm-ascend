# SPDX-License-Identifier: Apache-2.0
"""Exact PLE lookup IDs against a scalar oracle, including graph padding."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/nvidia/ple_layer.py"
EOS = 248044
MULTIPLIERS = [30000000000001, 20000000000003, 10000000000009]
SIZES = [20000003, 20000033, 20000047, 20000059]
OFFSETS = [0, 20000003, 40000036, 60000083]


def load_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpNGramEmbedding")
    names = {"forward_impl", "_shift_apply", "_shift_precompute"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch, "is_offload_process": lambda: False}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


def scalar_ids(tokens, context):
    result = []
    joined = context + tokens
    for position in range(len(context), len(joined)):
        last_eos = max((i for i in range(position) if joined[i] == EOS), default=-1)
        values = [joined[position]]
        for shift in (1, 2):
            values.append(joined[position - shift] if position - shift > last_eos else EOS)
        ids = []
        for ngram in (2, 3):
            mixed = 0
            for index in range(ngram):
                mixed ^= (values[index] * MULTIPLIERS[index]) & ((1 << 64) - 1)
            if mixed >= 1 << 63:
                mixed -= 1 << 64
            for head in range(2):
                index = (ngram - 2) * 2 + head
                ids.append(mixed % SIZES[index] + OFFSETS[index])
        result.append(ids)
    return result


def make_embedding(methods):
    def obj_ngram_embedding(ids):
        return ids.unsqueeze(-1)

    return SimpleNamespace(
        positions_buffer=torch.arange(4096),
        # Production keeps one unused sentinel element after the maximum
        # contiguous pack workspace.
        padded_buffer=torch.full((16 * 4096 + 1,), -99, dtype=torch.long),
        max_num_reqs=16,
        eos_token_id=EOS,
        ngram_size=3,
        heads_per_ngram=2,
        layer_multipliers=torch.tensor(MULTIPLIERS),
        ngram_heads_vocab_sizes=torch.tensor(SIZES),
        ngram_heads_offsets=torch.tensor(OFFSETS),
        ngram_embedding=obj_ngram_embedding,
        tp_size=1,
        shard_indices=None,
        _lookup_ngram_embedding=lambda ids, num_tokens: obj_ngram_embedding(
            ids
        ).reshape(num_tokens, -1),
        _shift_apply=methods["_shift_apply"],
        _shift_precompute=methods["_shift_precompute"],
    )


def test_lookup_ids_match_scalar_oracle_after_shape_and_token_changes():
    torch.set_num_threads(1)
    methods = load_methods()
    obj = make_embedding(methods)
    generator = torch.Generator().manual_seed(20260906)
    # Reuse workspace across shrinking/growing buckets and emptied rows.
    for lengths, padded in [
        ([1] * 16, 16),
        ([4] * 13, 64),
        ([0, 3, 0, 9], 16),
        ([4030, 1, 3], 4096),
        ([1], 8),
        ([4] * 16, 64),
    ]:
        contexts = torch.randint(0, EOS, (16, 2), generator=generator)
        contexts[::2, 1] = EOS
        tokens = torch.randint(0, EOS, (padded,), generator=generator)
        tokens[::7] = EOS
        query = [0]
        expected = []
        for row, length in enumerate(lengths):
            expected.extend(scalar_ids(tokens[query[-1] : query[-1] + length].tolist(), contexts[row].tolist()))
            query.append(query[-1] + length)
        query.extend([query[-1]] * (17 - len(query)))
        actual = methods["forward_impl"](obj, None, tokens, torch.tensor(query), contexts)
        assert actual[: sum(lengths)].tolist() == expected


def test_max_workspace_bucket_keeps_forced_dynamic_token_dim():
    """The 4096 bucket must not collapse a forced-dynamic token dimension."""
    methods = load_methods()
    obj = make_embedding(methods)
    query_start_loc = torch.arange(17, dtype=torch.int64) * 256
    contexts = torch.full((16, 2), EOS, dtype=torch.int64)

    def run(tokens):
        return methods["forward_impl"](
            obj,
            None,
            tokens,
            query_start_loc,
            contexts,
        )

    tokens = torch.arange(4096, dtype=torch.int64)
    torch._dynamo.mark_dynamic(tokens, 0)
    try:
        output = torch.compile(run, backend="eager", fullgraph=True)(tokens)
        assert output.shape == (4096, 4)
        assert obj.padded_buffer[: 16 * 4096].view(16, 4096).is_contiguous()
    finally:
        torch._dynamo.reset()
