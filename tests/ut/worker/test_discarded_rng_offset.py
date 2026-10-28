# SPDX-License-Identifier: Apache-2.0
"""Discarded prefills must not consume any seeded sampling RNG draws."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"


class FakeGenerator:
    def __init__(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset


def test_restore_exact_offset_only_for_discarded_seeded_rows():
    cls = next(
        n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner"
    )
    sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sample")
    snapshot = next(
        n
        for n in sample.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_discarded_generator_offsets" for t in n.targets)
    )
    bookkeeping = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_bookkeeping_sync")
    restore = next(
        n
        for n in bookkeeping.body
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == "discard_sampled_tokens_req_indices"
    )
    for increment in (4, 12, 128):
        generators = {0: FakeGenerator(72), 1: FakeGenerator(180)}
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(generators=generators),
            discard_request_indices=SimpleNamespace(np=np.array([0, 2])),
            num_discarded_requests=2,
        )
        namespace = dict(self=runner, discard_sampled_tokens_req_indices=np.array([0, 2]))
        for _ in range(7):
            for node in [snapshot]:
                exec(
                    compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"),
                    namespace,
                )
            for generator in generators.values():
                generator.offset += increment
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[restore], type_ignores=[])), str(SOURCE), "exec"),
                namespace,
            )
            assert generators[0].offset == 72
        assert generators[1].offset == 180 + 7 * increment
