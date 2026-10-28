# SPDX-License-Identifier: Apache-2.0
"""CPU contract for categorical MTP proposals and their rejection probabilities."""

import importlib.util
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/qwen_mtp_probability.py"
spec = importlib.util.spec_from_file_location("qwen_mtp_probability", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sample_probability = module.sample_probability


def test_inverse_cdf_tracks_the_reported_proposal_distribution():
    torch.set_num_threads(1)
    count = 20000
    q = torch.tensor([0.04, 0.11, 0.2, 0.25, 0.4])
    draws = (torch.arange(count, dtype=torch.float32) + 0.5) / count
    ids, probs = sample_probability(q.log().expand(count, -1), torch.ones(count), draws)
    actual = torch.bincount(ids, minlength=5).float() / count
    torch.testing.assert_close(probs[0], q)
    torch.testing.assert_close(actual, q, atol=1 / count, rtol=0)


def test_greedy_rows_and_request_reordering():
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0], [0.0, 1.0, 2.0]])
    temperature = torch.tensor([0.0, 0.7, 1.3])
    uniforms = torch.tensor([0.01, 0.8, 0.4])
    ids, probs = sample_probability(logits, temperature, uniforms)
    order = torch.tensor([2, 0, 1])
    new_ids, new_probs = sample_probability(logits[order], temperature[order], uniforms[order])
    assert ids[0] == logits[0].argmax()
    assert torch.equal(ids[order], new_ids)
    torch.testing.assert_close(probs[order], new_probs)


def test_supplied_uniforms_change_draw_without_changing_probabilities():
    logits = torch.tensor([[0.3, 0.4, 0.1]])
    a, p = sample_probability(logits, torch.ones(1), torch.tensor([0.01]))
    b, q = sample_probability(logits, torch.ones(1), torch.tensor([0.99]))
    assert a.item() != b.item()
    assert torch.equal(p, q)


def test_standard_rejection_recovers_target_with_disjoint_support():
    p = torch.tensor([0.0, 0.3, 0.0, 0.55, 0.15], dtype=torch.float64)
    q = torch.tensor([0.04, 0.11, 0.2, 0.25, 0.4], dtype=torch.float64)
    accepted = torch.minimum(p, q)
    residual = (p - q).clamp_min(0)
    actual = accepted + (1 - accepted.sum()) * residual / residual.sum()
    torch.testing.assert_close(actual, p, atol=1e-15, rtol=1e-15)
