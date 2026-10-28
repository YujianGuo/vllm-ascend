# SPDX-License-Identifier: Apache-2.0
"""Mixed greedy rows must keep full-vocabulary tie semantics after compression."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def production_forward():
    # Once installed under tests/ut/sample this resolves the real implementation.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/sample/rejection_sampler.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendRejectionSampler")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    reject = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rejection_sample")
    outer = next(
        node
        for node in reject.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not sampling_metadata.all_random"
    )
    selection = ast.fix_missing_locations(ast.Module(body=[outer.body[0]], type_ignores=[]))
    select_code = compile(selection, str(source), "exec")

    class Sampler:
        def __call__(self, **kwargs):
            return SimpleNamespace(sampled_token_ids=torch.tensor([[0]]))

        def greedy_sample(self, logits):
            return logits.argmax(-1)

    def compress(logits, *args):
        # Top-k is allowed to exclude the lowest-ID tied maximum entirely.
        indices = torch.tensor([[3, 2]])
        return logits.gather(1, indices), indices

    def reject_candidates(*args, **kwargs):
        logits, indices = args[5]
        namespace = dict(
            target_logits=logits,
            target_indices=indices,
            greedy_token_ids=kwargs.get("greedy_token_ids"),
            get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=True),
        )
        exec(select_code, namespace)
        return namespace["target_argmax"].unsqueeze(-1)

    namespace = dict(
        torch=torch,
        MAX_SPEC_LEN=128,
        replace=lambda value, **kwargs: value,
        apply_sampling_constraints=compress,
        rejection_sample=reject_candidates,
        SamplerOutput=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), forward]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    sampler = SimpleNamespace(
        _log_rejection_sampler_entry=lambda *args: None,
        _log_rejection_sampler_exit=lambda *args: None,
        sampler=Sampler(),
        is_processed_logprobs_mode=False,
        apply_logits_processors=lambda logits, *args: logits,
        top_k=20,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
    )
    metadata = SimpleNamespace(
        max_spec_len=1,
        bonus_logits_indices=torch.tensor([1]),
        target_logits_indices=torch.tensor([0]),
        draft_token_ids=torch.tensor([3]),
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1]),
    )
    sampling = SimpleNamespace(all_greedy=False, all_random=False, max_num_logprobs=None)
    return namespace["forward"], sampler, metadata, sampling


def test_tied_maximum_excluded_by_compact_topk_keeps_global_id(production_forward):
    forward, sampler, metadata, sampling = production_forward
    logits = torch.tensor([[0.0, 3.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    actual = forward(sampler, metadata, None, logits, sampling)
    assert actual.sampled_token_ids.tolist() == [[1]]


def test_greedy_selection_uses_processed_logits(production_forward):
    forward, sampler, metadata, sampling = production_forward
    sampler.apply_logits_processors = lambda logits, *args: logits + torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    logits = torch.tensor([[0.0, 3.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    actual = forward(sampler, metadata, None, logits, sampling)
    assert actual.sampled_token_ids.tolist() == [[0]]
