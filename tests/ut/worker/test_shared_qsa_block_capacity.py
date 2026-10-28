# SPDX-License-Identifier: Apache-2.0
"""CPU block-table capacity regression for shared QSA cache owners."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


class Uniform:
    def __init__(self, specs):
        self.kv_cache_specs = specs


class Mamba:
    pass


path = ROOT / "vllm_ascend/worker/block_table.py"
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BlockTable")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ["__init__", "append_row"]]
env = {
    "np": np,
    "torch": torch,
    "get_dcp_group": lambda: NS(world_size=1, rank_in_group=0),
    "UniformTypeKVCacheSpecs": Uniform,
    "MambaSpec": Mamba,
    "cdiv": lambda a, b: (a + b - 1) // b,
}
module = ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)


class Table:
    def _make_buffer(self, *size, dtype):
        return NS(np=np.zeros(size, dtype=np.int32))


Table.__init__ = env["__init__"]
Table.append_row = env["append_row"]


class TestSharedCapacity(unittest.TestCase):
    def make(self, specs):
        return Table(
            768,
            16,
            171,
            4096,
            False,
            torch.device("cpu"),
            kernel_sizes=[768],
            num_speculative_tokens=3,
            kv_cache_group=NS(kv_cache_spec=Uniform(specs)),
        )

    def test_compressed_owner_first(self):
        t = self.make({"compressed": NS(compress_ratio=4), "main": NS()})
        self.assertEqual(t.max_num_blocks_per_req, 171)
        for start in range(0, 171, 5):
            t.append_row(list(range(start, min(start + 5, 171))), 0)
        self.assertEqual(t.num_blocks_per_row[0], 171)

    def test_full_owner_first(self):
        t = self.make({"main": NS(), "compressed": NS(compress_ratio=4)})
        self.assertEqual(t.max_num_blocks_per_req, 171)

    def test_homogeneous_legacy_compression_preserved(self):
        t = self.make({"a": NS(compress_ratio=4), "b": NS(compress_ratio=4)})
        self.assertEqual(t.max_num_blocks_per_req, 43)

    def test_all_uncompressed(self):
        self.assertEqual(self.make({"a": NS(), "b": NS()}).max_num_blocks_per_req, 171)


if __name__ == "__main__":
    unittest.main(verbosity=2)
