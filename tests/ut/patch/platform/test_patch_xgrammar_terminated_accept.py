# SPDX-License-Identifier: Apache-2.0
"""Tests for the completion-aware xgrammar accept/validate patch.

Regression target: with speculative decoding a verification step hands the
grammar the accepted draft tokens plus the sampler's bonus token. Once the
document is complete, feeding the trailing tokens to the matcher makes xgrammar
log "The matcher has terminated after accepting the stop token, but is trying
to accept new token with id N" (grammar_matcher.cc:612) on every batch, and in
the ``accept_tokens`` path used to surface as a 500 for a valid document.
"""

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

from vllm_ascend.patch.platform import patch_xgrammar_terminated_accept  # noqa: F401


class FakeMatcher:
    """Minimal grammar matcher stub recording every ``accept_token`` probe."""

    def __init__(self, accept_upto: int | None = None, terminated_after: int | None = None):
        self.calls: list = []
        self.accepted = 0
        self.rolled_back = 0
        self._accept_upto = accept_upto
        self._terminated_after = terminated_after
        self._terminated = False

    def accept_token(self, token) -> bool:
        self.calls.append(token)
        if self._accept_upto is not None and self.accepted >= self._accept_upto:
            return False
        self.accepted += 1
        if self._terminated_after is not None and self.accepted >= self._terminated_after:
            self._terminated = True
        return True

    def is_terminated(self) -> bool:
        return self._terminated

    def is_completed(self) -> bool:
        return self._terminated

    def rollback(self, num_tokens: int) -> None:
        self.rolled_back += num_tokens
        self.accepted -= num_tokens


def _make_grammar(matcher: FakeMatcher) -> XgrammarGrammar:
    grammar = XgrammarGrammar.__new__(XgrammarGrammar)
    grammar.matcher = matcher
    grammar._is_terminated = False
    grammar.num_processed_tokens = 0
    grammar._completed_upto = None
    return grammar


def test_validate_tokens_skips_terminated_matcher():
    """A terminated grammar must not probe the matcher at all."""
    matcher = FakeMatcher()
    grammar = _make_grammar(matcher)
    grammar._is_terminated = True

    assert grammar.validate_tokens([198, 199]) == []
    assert matcher.calls == []


def test_accept_tokens_skips_terminated_matcher():
    """accept_tokens keeps its pre-existing terminated guard."""
    matcher = FakeMatcher()
    grammar = _make_grammar(matcher)
    grammar._is_terminated = True

    assert grammar.accept_tokens("req", [198]) is True
    assert matcher.calls == []


def test_validate_tokens_rolls_back_accepted_prefix():
    """Upstream semantics: accepted tokens are probed then rolled back."""
    matcher = FakeMatcher()
    grammar = _make_grammar(matcher)

    assert grammar.validate_tokens([1, 2, 3]) == [1, 2, 3]
    assert matcher.calls == [1, 2, 3]
    assert matcher.rolled_back == 3
    assert grammar.num_processed_tokens == 0


def test_validate_tokens_stops_at_rejection():
    """A rejected draft ends the accepted prefix without a rollback."""
    matcher = FakeMatcher(accept_upto=1)
    grammar = _make_grammar(matcher)

    assert grammar.validate_tokens([7, 8, 9]) == [7]
    assert matcher.calls == [7, 8]
    assert matcher.rolled_back == 1


def test_validate_tokens_stops_after_matcher_terminates():
    """Once the stop token lands mid-batch, later drafts are not probed."""
    matcher = FakeMatcher(terminated_after=1)
    grammar = _make_grammar(matcher)

    assert grammar.validate_tokens([5, 6, 7]) == [5]
    # Only the token that terminated the grammar was probed.
    assert matcher.calls == [5]
    assert matcher.rolled_back == 1


def test_accept_tokens_accepts_post_completion_suffix():
    """A completed document followed by bonus tokens is a success, not an error."""
    matcher = FakeMatcher(accept_upto=1, terminated_after=1)
    grammar = _make_grammar(matcher)

    assert grammar.accept_tokens("req", [5, 198]) is True
    assert grammar.num_processed_tokens == 1
    assert grammar._completed_upto == 1
    assert grammar._is_terminated is True


def test_accept_tokens_rejects_incomplete_violation():
    """A genuine grammar violation on an incomplete document still fails."""
    matcher = FakeMatcher(accept_upto=0)
    grammar = _make_grammar(matcher)

    assert grammar.accept_tokens("req", [9]) is False
