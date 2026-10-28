# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; do not initialize the serving runtime or NPU."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("qwen_ple_history", REPO / "vllm_ascend/worker/qwen_ple_history.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
QwenPleTokenHistory = module.QwenPleTokenHistory


class ReadyRows:
    def __init__(self, rows):
        self.rows = rows
        self.ready = False
        self.waits = 0

    def synchronize(self):
        self.ready = True
        self.waits += 1

    def tolist(self):
        assert self.ready, "Read before D2H completion"
        return self.rows


class TestQwenPleHistory(unittest.TestCase):
    def setUp(self):
        self.history = QwenPleTokenHistory()

    def commit(self, ids, positions, rows, invalid=(), finished=()):
        copy = ReadyRows(rows)
        self.history.stage(ids, positions, invalid, copy, copy)
        self.history.commit_pending(finished)
        # The wait is deferred: ``commit_pending`` must not touch the copy,
        # and the first reader is what materializes it.
        self.assertEqual(copy.waits, 0)
        self.history.materialize()
        self.assertEqual(copy.waits, 1)
        # Materializing again is a no-op rather than a second wait.
        self.history.materialize()
        self.assertEqual(copy.waits, 1)

    def test_async_decode_uses_real_previous_ids(self):
        self.commit(["a"], [3], [[101]])
        self.assertEqual(self.history.context("a", 2, [9, -1]), [9, 101])
        self.commit(["a"], [4], [[202]])
        self.assertEqual(self.history.context("a", 3, [-1, -1]), [101, 202])

    def test_partial_acceptance_and_bonus_are_committed(self):
        self.commit(["a"], [3], [[101, 202, -1, -1]])
        self.assertEqual(self.history.context("a", 3, [-1, -1]), [101, 202])
        self.assertEqual(self.history.committed_context_end("a"), 4)
        # Position 4 is still covered by the history; position 5 is beyond it
        # and gets the benign filler instead of raising.
        self.assertEqual(
            self.history.context("a", 4, [-1, -1]),
            [202, module._PLE_CONTEXT_FILLER_TOKEN],
        )

    def test_next_commit_uses_actual_accepted_length(self):
        self.commit(["a"], [3], [[101, 202, -1, -1]])
        self.commit(["a"], [5], [[303, 404, 505, 606]])
        self.assertEqual(self.history.context("a", 5, [-1, -1]), [303, 404])

    def test_request_row_permutation(self):
        self.commit(["a", "b"], [3, 10], [[101], [201]])
        self.commit(["b", "a"], [11, 4], [[202], [102]])
        self.assertEqual(self.history.context("a", 3, [-1, -1]), [101, 102])
        self.assertEqual(self.history.context("b", 10, [-1, -1]), [201, 202])

    def test_staging_snapshots_mutable_request_maps(self):
        ids, positions, invalid = ["a"], [3], []
        copy = ReadyRows([[101]])
        self.history.stage(ids, positions, invalid, copy, copy)
        ids[0], positions[0] = "b", 99
        invalid.append(0)
        self.history.commit_pending(())
        self.assertEqual(self.history.context("a", 3, [-1]), [101])

    def test_discarded_prefill_does_not_commit_sample(self):
        self.commit(["a"], [2], [[999]], invalid=[0])
        self.assertEqual(self.history.context("a", 0, [7, 8]), [7, 8])

    def test_finished_request_and_id_reuse(self):
        self.commit(["a"], [3], [[101]])
        self.commit(["a"], [4], [[202]], finished={"a"})
        self.assertEqual(self.history.context("a", 3, [777]), [777])
        self.commit(["a"], [3], [[303]])
        self.assertEqual(self.history.context("a", 3, [-1]), [303])

    def test_paused_request_keeps_history(self):
        self.commit(["a"], [3], [[101]])
        self.commit(["b"], [6], [[201]])
        self.history.commit_pending(())
        self.assertEqual(self.history.context("a", 3, [-1]), [101])

    def test_recompute_replaces_abandoned_suffix(self):
        self.commit(["a"], [3], [[101, 202, 303, 404]])
        self.commit(["a"], [4], [[909]])
        self.assertEqual(self.history.context("a", 3, [-1, -1]), [101, 909])
        # Position 5 is beyond the committed history; filled, not raised.
        self.assertEqual(
            self.history.context("a", 5, [-1]),
            [module._PLE_CONTEXT_FILLER_TOKEN],
        )

    def test_prefix_hit_uses_real_prompt_context(self):
        self.assertEqual(self.history.context("a", 4094, [77, 88]), [77, 88])

    def test_missing_history_is_not_silently_substituted(self):
        # Without any committed history the caller's fallback IDs are kept as
        # long as they are valid; unresolvable -1 placeholders become filler.
        self.assertEqual(self.history.context("a", 3, [77, 88]), [77, 88])
        self.assertEqual(
            self.history.context("a", 3, [-1, -1]),
            [module._PLE_CONTEXT_FILLER_TOKEN, module._PLE_CONTEXT_FILLER_TOKEN],
        )

    def test_output_gap_bridges_entry_without_discarding_history(self):
        # A position gap no longer discards the committed prefix. Unknown
        # positions are represented by the benign filler token.
        self.commit(["a"], [3], [[101]])
        self.commit(["a"], [5], [[303]])
        self.assertEqual(self.history.context("a", 3, [-1, -1, -1]), [101, 0, 303])

    def test_output_rewind_gap_merges_coverage(self):
        # F01 dual-machine regression: after a rewind the next output lands
        # before the existing history start. The old coverage must be kept
        # (merged), not discarded, so the context window just below the new
        # position stays resolvable.
        self.commit(["a"], [5], [[303]])
        self.commit(["a"], [4], [[202]])
        # Merged history covers [4, 6); position 5 was replaced by the
        # recomputed suffix, position 4 holds the rewound commit.
        self.assertEqual(self.history.context("a", 4, [-1, -1]), [202, 303])

    def test_context_window_below_history_start_is_filled(self):
        # The exact F01 crash signature: the next decode step queries a
        # context window that starts before the (reinitialized) history.
        # This must degrade to a filler token, never raise.
        self.commit(["a"], [5], [[303]])
        self.assertEqual(
            self.history.context("a", 3, [-1, -1]),
            [module._PLE_CONTEXT_FILLER_TOKEN, module._PLE_CONTEXT_FILLER_TOKEN],
        )
        # One step later the window overlaps the history again.
        self.assertEqual(
            self.history.context("a", 4, [-1, -1]),
            [module._PLE_CONTEXT_FILLER_TOKEN, 303],
        )

    def test_rewind_gap_hole_in_front_is_filled(self):
        # After a rewind merge the region in front of the old start stays
        # unknown; queries there get filler tokens instead of crashing.
        self.commit(["a"], [5], [[303]])
        self.commit(["a"], [4], [[202]])
        self.assertEqual(
            self.history.context("a", 2, [-1, -1]),
            [module._PLE_CONTEXT_FILLER_TOKEN, module._PLE_CONTEXT_FILLER_TOKEN],
        )

    def test_malformed_acceptance_prefix_is_rejected(self):
        # A torn/malformed async row is isolated to PLE. The contiguous
        # prefix remains useful, while the worker must not crash globally.
        self.commit(["a"], [3], [[101, -1, 202]])
        self.assertEqual(self.history.context("a", 3, [-1]), [101])

    def test_negative_corruption_is_not_committed(self):
        self.commit(["a"], [3], [[101, -2, 202]])
        self.assertEqual(self.history.context("a", 3, [-1]), [101])

    def test_unconsumed_output_cannot_be_overwritten(self):
        copy = ReadyRows([[101]])
        self.history.stage(["a"], [3], [], copy, copy)
        with self.assertRaises(RuntimeError):
            self.history.stage(["b"], [4], [], copy, copy)

    def test_reader_materializes_without_an_explicit_call(self):
        # Every entry point must observe the staged commit; a caller that
        # never calls materialize() must still see the committed token.
        copy = ReadyRows([[101]])
        self.history.stage(["a"], [3], [], copy, copy)
        self.history.commit_pending(())
        self.assertEqual(self.history.context("a", 3, [-1]), [101])
        self.assertEqual(copy.waits, 1)

    def test_finished_request_is_excluded_by_the_deferred_commit(self):
        # ``finished`` is recorded by commit_pending but can only be applied
        # when the rows are read, so the exclusion must survive the deferral.
        self.commit(["a"], [3], [[101]])
        copy = ReadyRows([[202]])
        self.history.stage(["a"], [4], [], copy, copy)
        self.history.commit_pending({"a"})
        self.assertEqual(self.history.context("a", 3, [777]), [777])

    def test_forget_is_applied_after_the_pending_commit(self):
        # The model runner forgets reused IDs in the same step it commits.
        # Applying that forget eagerly would only remove history that is about
        # to be rewritten; the deferred commit must still drop the old segment
        # so a re-issued ID starts clean.
        self.commit(["a"], [3], [[101]])
        copy = ReadyRows([[202]])
        self.history.stage(["a"], [4], [], copy, copy)
        self.history.commit_pending(())
        self.history.forget(["a"])
        self.assertEqual(self.history.context("a", 4, [555]), [555])
        # The forget survives into the next commit's bookkeeping.
        self.commit(["a"], [5], [[303]])
        self.assertEqual(self.history.context("a", 5, [-1]), [303])

    def test_materialize_is_idempotent(self):
        copy = ReadyRows([[101]])
        self.history.stage(["a"], [3], [], copy, copy)
        self.history.commit_pending(())
        self.history.materialize()
        self.history.materialize()
        self.assertEqual(copy.waits, 1)
        # A later stage/commit cycle still waits exactly once.
        copy2 = ReadyRows([[202]])
        self.history.stage(["a"], [4], [], copy2, copy2)
        self.history.commit_pending(())
        self.history.materialize()
        self.assertEqual(copy2.waits, 1)
        self.assertEqual(self.history.context("a", 3, [-1, -1]), [101, 202])

    def test_streaming_prefix_replacement(self):
        self.commit(["a"], [3], [[101]])
        self.history.forget(["a"])
        self.assertEqual(self.history.context("a", 3, [555]), [555])

    def test_preemption_resume_uses_restored_canonical_tokens(self):
        self.commit(["a"], [3], [[101, 202, 303]])
        self.history.forget({"a"})
        self.assertEqual(self.history.context("a", 3, [808, 909]), [808, 909])
        self.commit(["a"], [5], [[707]])
        self.assertEqual(self.history.context("a", 4, [909, -1]), [909, 707])

    def test_output_position_snapshot_for_prefill_and_verification(self):
        source = (REPO / "vllm_ascend/worker/model_runner_v1.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
        assignment = next(
            n
            for n in ast.walk(sample)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ple_output_positions" for t in n.targets)
            and isinstance(n.value, ast.ListComp)
        )
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["prefill", "decode", "mtp3", "mtp1"], num_computed_tokens_cpu=[100, 200, 300, 400]
            )
        )
        scheduler = SimpleNamespace(
            num_scheduled_tokens={"prefill": 28, "decode": 1, "mtp3": 4, "mtp1": 2},
            scheduled_spec_decode_tokens={"mtp3": [-1, -1, -1], "mtp1": [-1]},
        )
        env = {"self": runner, "scheduler_output": scheduler, "ple_history": QwenPleTokenHistory()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), str(REPO), "exec"), env)
        self.assertEqual(env["ple_output_positions"], [128, 201, 301, 401])

    def test_next_output_position_prefers_committed_boundary(self):
        # Async scheduling may report the all-drafts-accepted boundary even
        # when rejection left only the accepted prefix plus bonus token.
        self.commit(["a"], [3], [[101, 202, -1, -1]])
        self.assertEqual(self.history.next_output_position("a", 5), 5)
        self.assertEqual(self.history.next_output_position("a", 8), 5)
        # A lower boundary is an actual rewind and must remain scheduler-led
        # so the abandoned suffix can be replaced.
        self.assertEqual(self.history.next_output_position("a", 4), 4)

    def test_next_output_position_keeps_scheduler_for_new_request(self):
        self.assertEqual(self.history.next_output_position("new", 42), 42)

    def test_logprobs_snapshot_preserves_optional_cu_metadata(self):
        # The async output ownership fix must clone tensor fields only;
        # cu_num_generated_tokens is an optional Python list (often None).
        source = (REPO / "vllm_ascend/worker/model_runner_v1.py").read_text()
        self.assertIn("logprobs_tensors_snapshot.cu_num_generated_tokens", source)

    def test_partition_invariant_committed_context(self):
        self.commit(["a"], [3], [[101, 202, 303]])
        batch_context = self.history.context("a", 4, [-1, -1])
        self.history = QwenPleTokenHistory()
        for i, token in enumerate([101, 202, 303]):
            self.commit(["a"], [3 + i], [[token]])
        self.assertEqual(self.history.context("a", 4, [-1, -1]), batch_context)

    def test_production_ple_prepare_with_cpu_placeholders(self):
        import numpy as np
        import torch

        source = (REPO / "vllm_ascend/worker/model_runner_v1.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        prepare = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_add_qwen4_exp_ple_inputs"
        )
        pad = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_qwen4_exp_ple_graph_inputs"
        )
        env = {"np": np, "torch": torch}
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), pad, prepare]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(REPO), "exec"), env)
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(ple_layer_ids=[1], ngram_size=3, eos_token_id=999)
            ),
            max_num_reqs=2,
            device=torch.device("cpu"),
            input_ids=SimpleNamespace(gpu=torch.tensor([303], dtype=torch.int32)),
            query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.int32)),
            input_batch=SimpleNamespace(
                req_ids=["a"], num_computed_tokens_cpu=np.array([5]), token_ids_cpu=np.array([[7, 8, 9, -1, -1, -1]])
            ),
            _qwen4_exp_ple_history=self.history,
            _qwen4_exp_ngram_context_buffer=None,
            _qwen4_exp_query_start_loc_buffer=None,
        )
        self.commit(["a"], [3], [[101, 202, 303, -1]])
        kwargs = {}
        env["_maybe_add_qwen4_exp_ple_inputs"](runner, kwargs, num_tokens=1, num_reqs=1, num_reqs_padded=1)
        self.assertEqual(kwargs["ngram_context"].tolist(), [[101, 202], [999, 999]])

    def test_production_ple_rewinds_optimistic_rejected_suffix(self):
        import numpy as np
        import torch

        source = (REPO / "vllm_ascend/worker/model_runner_v1.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        prepare = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_add_qwen4_exp_ple_inputs"
        )
        pad = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_qwen4_exp_ple_graph_inputs"
        )
        env = {"np": np, "torch": torch}
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), pad, prepare]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(REPO), "exec"), env)
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(ple_layer_ids=[1], ngram_size=3, eos_token_id=999)
            ),
            max_num_reqs=2,
            device=torch.device("cpu"),
            input_ids=SimpleNamespace(gpu=torch.tensor([303], dtype=torch.int32)),
            query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.int32)),
            input_batch=SimpleNamespace(
                req_ids=["a"],
                # Scheduler assumed all three drafts were accepted. History
                # proves that only one draft plus the bonus token survived.
                num_computed_tokens_cpu=np.array([6]),
                token_ids_cpu=np.array([[7, 8, 9, -1, -1, -1, -1]]),
            ),
            _qwen4_exp_ple_history=self.history,
            _qwen4_exp_ngram_context_buffer=None,
            _qwen4_exp_query_start_loc_buffer=None,
        )
        self.commit(["a"], [3], [[101, 202, -1, -1]])
        kwargs = {}
        env["_maybe_add_qwen4_exp_ple_inputs"](
            runner,
            kwargs,
            num_tokens=1,
            num_reqs=1,
            num_reqs_padded=1,
        )
        # The final valid token is the next input and is excluded from the
        # preceding PLE context.
        self.assertEqual(kwargs["ngram_context"].tolist(), [[9, 101], [999, 999]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
