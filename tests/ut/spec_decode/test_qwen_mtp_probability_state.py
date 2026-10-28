# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for real-proposal ownership and idle DP companion rows."""

import ast
import hashlib
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def production_methods():
    # Load the actual methods without initializing distributed/NPU workers.
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/spec_decode/eagle_proposer.py"
    probability = root / "vllm_ascend/spec_decode/qwen_mtp_probability.py"
    spec = importlib.util.spec_from_file_location("qwen_probability_state_test", probability)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proposer = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendQwen4ExpMTPProposer"
    )
    methods = [
        node
        for node in proposer.body
        if isinstance(node, ast.FunctionDef) and node.name in {"compute_draft_token_ids", "_propose"}
    ]

    def fake_base(self, *args, **kwargs):
        # Mimic graph writes into persistent storage, with no model execution.
        self._qwen_prob_buffer.fill_(1 / 7)
        return torch.ones((len(self.runner.input_batch.req_ids), 3), dtype=torch.long)

    namespace = dict(
        torch=torch,
        hashlib=hashlib,
        random=random,
        sample_probability=module.sample_probability,
        AscendSpecDecodeBaseProposer=SimpleNamespace(_propose=fake_base),
        greedy_sample=lambda logits: logits.argmax(-1),
        get_ascend_config=lambda: SimpleNamespace(enable_reduce_sample=True),
        get_tp_group=lambda: SimpleNamespace(all_gather=lambda value, dim: value),
    )
    code = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)
    return namespace


def make_proposer(request_ids):
    runner = SimpleNamespace(
        max_num_reqs=16,
        requests={
            key: SimpleNamespace(sampling_params=SimpleNamespace(seed=4 + i)) for i, key in enumerate(["a", "b"])
        },
        input_batch=SimpleNamespace(req_ids=request_ids),
        model_config=SimpleNamespace(get_vocab_size=lambda: 7),
    )
    return SimpleNamespace(
        _qwen_probabilistic=True,
        _qwen_probability_step=0,
        num_speculative_tokens=3,
        runner=runner,
        model=SimpleNamespace(compute_logits=lambda value: value),
        _qwen_temperature=torch.zeros(16),
        _qwen_uniforms=torch.full((16, 3), 0.5),
        _qwen_prob_buffer=torch.zeros(16, 3, 7),
        _qwen_draft_rng={},
        use_repeated_proposals=False,
        device="cpu",
    )


def propose(methods, obj, *, greedy=False):
    batch = len(obj.runner.input_batch.req_ids)
    metadata = SimpleNamespace(all_greedy=greedy, temperature=torch.full((batch,), 0.0 if greedy else 1.0))
    return methods["_propose"](obj, None, None, None, None, None, None, None, metadata)


def test_idle_companion_may_exceed_real_request_capacity(production_methods):
    obj = make_proposer(["a"])
    logits = torch.arange(22 * 7).view(22, 7).float()
    actual = production_methods["compute_draft_token_ids"](obj, logits)
    assert torch.equal(actual, logits.argmax(-1))
    assert torch.count_nonzero(obj._qwen_prob_buffer) == 0


def test_real_probabilities_survive_dummy_overwrite_and_clear_for_greedy(production_methods):
    obj = make_proposer(["a"])
    propose(production_methods, obj)
    saved = obj._last_draft_probs.clone()
    assert obj._last_draft_probs.data_ptr() != obj._qwen_prob_buffer.data_ptr()
    obj._qwen_prob_buffer.zero_()
    assert torch.equal(obj._last_draft_probs, saved)
    torch.testing.assert_close(saved, torch.full((1, 3, 7), 1 / 7))
    propose(production_methods, obj, greedy=True)
    assert obj._last_draft_probs is None


def test_request_rng_survives_reorder_and_does_not_consume_target_rng(production_methods):
    first = make_proposer(["a", "b"])
    reordered = make_proposer(["b", "a"])
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    for _ in range(3):
        propose(production_methods, first)
        propose(production_methods, reordered)
        assert torch.equal(first._qwen_uniforms[:2], reordered._qwen_uniforms[[1, 0]])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    del first.runner.requests["b"]
    first.runner.input_batch.req_ids = ["a"]
    propose(production_methods, first)
    assert set(first._qwen_draft_rng) == {"a"}


def test_excess_draft_step_fails_before_overwriting_probabilities(production_methods):
    obj = make_proposer(["a"])
    obj._qwen_probability_step = obj.num_speculative_tokens
    with pytest.raises(RuntimeError, match="More probability writes"):
        production_methods["compute_draft_token_ids"](obj, torch.zeros(1, 7))
    assert torch.count_nonzero(obj._qwen_prob_buffer) == 0


def test_uniform_staging_preserves_exact_request_rng_and_padding(production_methods):
    obj = make_proposer(["a", "b"])
    generators = [random.Random(seed ^ 0x514D5450) for seed in [4, 5]]
    for _ in range(4):
        expected = torch.tensor([[rng.random() for _ in range(3)] for rng in generators]).clamp_max(1 - 2**-24)
        propose(production_methods, obj)
        assert torch.equal(obj._qwen_uniforms[:2], expected)
        assert torch.equal(obj._qwen_uniforms[2:], torch.full((14, 3), 0.5))
