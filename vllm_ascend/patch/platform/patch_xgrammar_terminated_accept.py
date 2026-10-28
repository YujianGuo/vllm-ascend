# SPDX-License-Identifier: Apache-2.0
"""Stop treating post-completion tokens as grammar errors.

With speculative decoding (MTP), one verification step hands
``grammar.accept_tokens`` a whole batch: the accepted draft tokens plus the
sampler's bonus token. xgrammar reports a *completed* document separately from
a *terminated* one:

* ``matcher.is_completed()`` becomes ``True`` as soon as the grammar's root
  rule is fully matched (e.g. the JSON document closed).
* ``matcher.is_terminated()`` only becomes ``True`` once the stop/EOS token
  itself is accepted.

So for a batch such as ``['"}', '!']`` the closing brace completes the
document, ``is_terminated()`` is still ``False`` and the trailing bonus token
``'!'`` fails ``accept_token``. Upstream ``accept_tokens`` then returns
``False``, the scheduler flags the request ``FINISHED_ERROR`` and the client
gets a 500 even though the model produced a perfectly valid JSON document.

Under asynchronous scheduling the speculative drafts are produced in the
worker and are *not* re-validated against the grammar by
``update_draft_token_ids_in_output`` (that path is skipped when
``async_scheduling`` is on). Once the document is complete those drafts are
still appended to the request output, which is why a completed document is
followed by trailing junk tokens.

This patch makes ``accept_tokens`` completion-aware:

* it stops advancing the FSM as soon as the document is complete, so a
  rejection after that point is reported as success instead of an error;
* it records ``_completed_upto`` -- how many tokens of the current batch were
  consumed before/at completion -- so the scheduler can drop the trailing
  post-completion tokens and finish the request cleanly.

A rejection on an *incomplete* document is a genuine grammar violation and
still returns ``False`` so the scheduler can terminate the request.

``validate_tokens`` gets the same treatment. It probes ``matcher.accept_token``
directly and had no terminated guard, so the scheduler's draft-trimming path
kept feeding post-completion tokens to an already-terminated matcher. xgrammar
then logged "The matcher has terminated after accepting the stop token, but is
trying to accept new token with id N" on every batch. The wording is cosmetic
(the probe result is discarded either way), but the call is pure overhead and
the spam hides real errors, so this patch stops probing once the grammar is
terminated.
"""

import logging

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

logger = logging.getLogger(__name__)


def _patched_accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
    # Reset per call: None means "document not completed by this batch".
    self._completed_upto = None

    if self._is_terminated:
        # The stop token was already accepted (previous step). Later output is
        # sampler traffic (bonus or draft suffix tokens), not a violation.
        self._completed_upto = 0
        return True

    for i, token in enumerate(tokens):
        if not self.matcher.accept_token(token):
            if self.matcher.is_completed():
                # Rejected only because the document is already whole: this is
                # the bonus/draft suffix that followed the closing token.
                self._completed_upto = i
                self._is_terminated = self.matcher.is_terminated()
                return True
            logger.error(
                "Failed to advance FSM for request %s for tokens %s. "
                "Please file an issue.",
                request_id,
                token,
            )
            return False
        self.num_processed_tokens += 1
        if self.matcher.is_completed():
            # The document closed on this token; anything after it in this
            # batch is post-completion sampler traffic.
            self._completed_upto = i + 1
            self._is_terminated = self.matcher.is_terminated()
            return True

    self._is_terminated = self.matcher.is_terminated()
    return True


def _patched_validate_tokens(self, tokens: list[int]) -> list[int]:
    """Filter speculative drafts without feeding a terminated matcher.

    ``validate_tokens`` calls ``matcher.accept_token`` directly, bypassing the
    ``_is_terminated`` guard that ``accept_tokens`` has. The scheduler uses it
    to trim draft tokens (``update_draft_token_ids_in_output`` /
    ``update_draft_token_ids``), and it also runs from the Ascend structured
    output patch while probing a candidate token. Once the grammar has already
    accepted its stop token, every further ``accept_token`` call makes xgrammar
    log "The matcher has terminated after accepting the stop token, but is
    trying to accept new token with id N" at the C++ level (grammar_matcher.cc).
    Nothing can be accepted after termination, so return the empty prefix
    instead of probing the matcher.

    The FSM must not advance, so any token accepted before termination (or
    before the loop decides to stop) is rolled back exactly as upstream does.
    """
    if self._is_terminated:
        return []

    accepted_tokens = []
    for token in tokens:
        if not self.matcher.accept_token(token):
            break
        accepted_tokens.append(token)
        if self.matcher.is_terminated():
            # The stop token was consumed on this token; later drafts are
            # post-completion sampler traffic and must not reach the matcher.
            break
    if len(accepted_tokens) > 0:
        # Rollback the FSM to the initial state
        self.matcher.rollback(len(accepted_tokens))
    return accepted_tokens


XgrammarGrammar.accept_tokens = _patched_accept_tokens
XgrammarGrammar.validate_tokens = _patched_validate_tokens
