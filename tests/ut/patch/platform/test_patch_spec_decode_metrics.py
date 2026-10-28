# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler

import vllm_ascend.patch.platform.patch_spec_decode_metrics  # noqa: F401


def _scheduler():
    return SimpleNamespace(log_stats=True, num_spec_tokens=3)


def test_invalid_only_wave_does_not_increment_drafts():
    stats = Scheduler.make_spec_decoding_stats(
        _scheduler(), None, 3, 0, {"req": 3}, "req"
    )
    assert stats is None


def test_partial_invalid_wave_uses_effective_denominator():
    stats = Scheduler.make_spec_decoding_stats(
        _scheduler(), None, 3, 2, {"req": 1}, "req"
    )
    assert stats is not None
    assert stats.num_drafts == 1
    assert stats.num_draft_tokens == 2
    assert stats.num_accepted_tokens == 2
    assert stats.num_accepted_tokens_per_pos == [1, 1, 0]


def test_accepted_tokens_are_clamped_to_valid_candidates():
    stats = Scheduler.make_spec_decoding_stats(
        _scheduler(), None, 3, 3, {"req": 2}, "req"
    )
    assert stats is not None
    assert stats.num_draft_tokens == 1
    assert stats.num_accepted_tokens == 1
    assert stats.num_accepted_tokens_per_pos == [1, 0, 0]
