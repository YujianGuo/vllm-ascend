# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fix speculative-decoding metric denominators for grammar-filtered drafts.

The scheduler pads grammar-invalid draft candidates with ``-1`` and reports
their count through ``num_invalid_spec_tokens``.  Upstream subtracts those
tokens from ``num_draft_tokens`` but still increments ``num_drafts`` for a
wave that can contain no valid draft token.  This makes the token acceptance
rate (accepted/effective-draft-tokens) disagree with the per-position vector
(accepted-position/number-of-waves).

Keep vLLM untouched: vLLM-Ascend installs a narrow wrapper around the scheduler
method and forwards effective counts to the original implementation.  The
standard vLLM per-position metric remains unconditional-per-valid-wave, while
the aggregate token metric uses the same valid-token population.
"""

from functools import wraps

from vllm.v1.core.sched.scheduler import Scheduler


_ORIGINAL_MAKE_SPEC_DECODING_STATS = Scheduler.make_spec_decoding_stats


@wraps(_ORIGINAL_MAKE_SPEC_DECODING_STATS)
def _patched_make_spec_decoding_stats(
    self,
    spec_decoding_stats,
    num_draft_tokens: int,
    num_accepted_tokens: int,
    num_invalid_spec_tokens,
    request_id: str,
):
    if not self.log_stats or not num_draft_tokens:
        return None

    invalid_tokens = 0
    if num_invalid_spec_tokens:
        invalid_tokens = max(0, num_invalid_spec_tokens.get(request_id, 0))
    effective_draft_tokens = max(num_draft_tokens - invalid_tokens, 0)

    # A grammar-filtered wave with no valid candidate is not a draft attempt;
    # do not create/increment ``num_drafts`` for it.  Preserve any stats already
    # accumulated in the current scheduler update.
    if effective_draft_tokens == 0:
        return spec_decoding_stats

    # Accepted tokens are a prefix of the valid draft candidates.  Clamp here
    # so malformed/stale async output cannot make acceptance counters exceed
    # their effective denominator.
    effective_accepted_tokens = min(max(num_accepted_tokens, 0), effective_draft_tokens)
    return _ORIGINAL_MAKE_SPEC_DECODING_STATS(
        self,
        spec_decoding_stats=spec_decoding_stats,
        num_draft_tokens=effective_draft_tokens,
        num_accepted_tokens=effective_accepted_tokens,
        num_invalid_spec_tokens=None,
        request_id=request_id,
    )


if not getattr(Scheduler.make_spec_decoding_stats, "_vllm_ascend_metrics_patched", False):
    _patched_make_spec_decoding_stats._vllm_ascend_metrics_patched = True  # type: ignore[attr-defined]
    Scheduler.make_spec_decoding_stats = _patched_make_spec_decoding_stats

