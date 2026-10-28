# SPDX-License-Identifier: Apache-2.0
"""Committed token history for PLE in the V1 asynchronous model runner."""

import logging
from array import array
from collections.abc import Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Benign filler for PLE n-gram context positions that cannot be resolved
# (uncommitted placeholders, or holes left by a history rewind). Must be a
# valid non-negative token id; 0 (an "!" BPE fragment) keeps the hash in range
# and never matches a real stop sequence in practice.
_PLE_CONTEXT_FILLER_TOKEN = 0


class QwenPleTokenHistory:
    """Keep accepted IDs separate from the scheduler's optimistic placeholders.

    Pending rows and absolute positions are captured before batch reordering.
    Only the already existing async output copy is read; this class schedules
    no additional device transfer.

    That copy is launched at the end of a step and only awaited when its rows
    are actually about to be read. ``commit_pending`` runs while the next
    forward's inputs are still being prepared, so an eager wait there blocks
    the host before the device has been given any work -- the previous step's
    copy has had only the (short) step boundary to complete. This class
    therefore stages the commit and performs the wait in :meth:`materialize`,
    which every reader calls; preparation of the next forward then overlaps
    the copy and, in steady state, the event is already signalled.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, tuple[int, array]] = {}
        self._pending: tuple[tuple[str, ...], tuple[int, ...], frozenset[int], Any, Any] | None = None
        # Forget work that belongs to the deferred commit, kept in the order an
        # eager commit would have applied it.
        self._deferred_finished: set[str] = set()
        self._deferred_forget: set[str] = set()

    def stage(
        self,
        req_ids: Sequence[str],
        first_output_positions: Sequence[int],
        invalid_req_indices: Sequence[int],
        sampled_token_ids_cpu: Any,
        copy_ready_event: Any,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("PLE token history has an unconsumed async output")
        if len(req_ids) != len(first_output_positions):
            raise ValueError("PLE output positions must match the request rows")
        self._pending = (
            tuple(req_ids),
            tuple(int(p) for p in first_output_positions),
            frozenset(invalid_req_indices),
            sampled_token_ids_cpu,
            copy_ready_event,
        )

    def commit_pending(self, finished_req_ids: Iterable[str]) -> None:
        """Record the commit of the staged async output and its forget set.

        The device->host transfer is deliberately *not* awaited here. Eagerly
        waiting at this point blocks the host at the top of the step, before
        the forward has been dispatched, which is exactly where the previous
        step's copy is least likely to have completed. The wait happens in
        :meth:`materialize`, which every reader calls, so the input
        preparation in between overlaps the copy.

        Ordering is preserved exactly as an eager commit would have applied
        it: :meth:`forget` is called by the model runner *after* this method,
        so its requests are merged here and the whole set is applied in one
        place at materialization time.
        """
        self._deferred_finished = set(finished_req_ids)
        self._deferred_forget |= self._deferred_finished

    def materialize(self) -> None:
        """Commit the staged output, waiting for its copy to reach the host.

        Safe and cheap to call repeatedly: after the first call the commit is
        a no-op, so readers can delegate to it unconditionally.
        """
        if self._pending is not None:
            req_ids, positions, invalid, tokens_cpu, event = self._pending
            finished = self._deferred_finished
            event.synchronize()
            rows = tokens_cpu.tolist()
            if len(rows) != len(req_ids):
                raise ValueError("PLE async output has unexpected request rows")
            for row, (req_id, position) in enumerate(zip(req_ids, positions)):
                if row in invalid or req_id in finished:
                    continue
                tokens = rows[row]
                # Any negative value marks the end of the usable prefix.  In
                # normal rejection output this is ``-1``; treating other
                # negatives the same way prevents a corrupted sentinel from
                # entering the PLE hash state.
                count = next((i for i, token in enumerate(tokens) if token < 0), len(tokens))
                accepted = tokens[:count]
                if any(t != -1 for t in tokens[count:]):
                    # A malformed/torn async copy must not take down every DP
                    # worker.  The output path will independently filter the
                    # same row; for PLE, only the contiguous prefix is safe to
                    # commit and the suffix is treated as unavailable.
                    logger.warning(
                        "PLE sampled-token row is not a valid prefix for req_id=%s "
                        "(row=%d, tokens=%s); committing prefix length %d only.",
                        req_id,
                        row,
                        tokens,
                        count,
                    )
                if not accepted:
                    continue
                if position < 0:
                    raise ValueError("PLE output positions must be non-negative")
                if req_id not in self._tokens:
                    self._tokens[req_id] = (position, array("i", accepted))
                    continue
                start, history = self._tokens[req_id]
                offset = position - start
                if not 0 <= offset <= len(history):
                    # Gap detected. Speculative-decoding history can diverge
                    # from the scheduler's expected positions after a request
                    # is terminated, reordered or rewound (preemption / prefix
                    # cache eviction), and a single desynchronized request must
                    # not take the whole engine down.
                    if offset < 0:
                        # Rewound: the new accepted segment overlays an
                        # earlier region. Merge instead of replacing so the
                        # old coverage is kept; the rewound hole in front of
                        # the old start stays unresolved (-1) and is filled
                        # by context() when queried.
                        pad = start - position
                        merged = array("i", [-1]) * pad + history
                        merged[offset + pad : offset + pad + len(accepted)] = array("i", accepted)
                        self._tokens[req_id] = (position, merged)
                    else:
                        # Jumped forward: preserve the committed prefix and
                        # bridge the unknown interval with the same benign
                        # filler used by context().  Dropping the whole entry
                        # caused every subsequent async MTP step to rebuild
                        # PLE state and produced severe concurrent slowdown.
                        missing = offset - len(history)
                        if missing > 0:
                            history.extend([_PLE_CONTEXT_FILLER_TOKEN] * missing)
                        history.extend(array("i", accepted))
                        logger.warning(
                            "PLE committed token history bridged a gap for req_id=%s "
                            "(position=%d, start=%d, offset=%d, missing=%d).",
                            req_id,
                            position,
                            start,
                            offset,
                            max(missing, 0),
                        )
                    continue
                # Recomputed suffixes replace history rather than retaining
                # tokens from an abandoned continuation.
                history[offset:] = array("i", accepted)
            self._pending = None
        if self._deferred_forget:
            self._forget_now(self._deferred_forget)
            self._deferred_forget = set()
        self._deferred_finished = set()

    def forget(self, req_ids: Iterable[str]) -> None:
        """Queue requests for removal as part of the deferred commit.

        These calls happen after ``commit_pending`` in the same step, so their
        requests must be applied by the *pending* materialization; forgetting
        them immediately would let the commit that is still on the device
        rewrite history for an ID that was just reused by a new request.

        A single ``forget`` call is only ever split across two commits if it
        spans a ``stage`` boundary, and ``_update_states`` issues each forget
        set in one call, so the requests of one caller always travel together.
        """
        self._deferred_forget |= set(req_ids)

    def _forget_now(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._tokens.pop(req_id, None)

    def committed_context_end(self, req_id: str) -> int | None:
        """Return the real computed-token boundary after the latest output.

        A speculative output row contains the accepted prefix followed by the
        token that will be used as the first input of the next target step.
        That final token is excluded from the preceding PLE context, so a
        committed segment ending at ``start + len(history) - 1`` proves the
        context boundary.

        This lets PLE recover when async scheduling still exposes the
        all-drafts-accepted boundary in ``num_computed_tokens_cpu``.  It uses
        the sampled-token copy that ``commit_pending`` already synchronized;
        no additional device-to-host transfer is needed.
        """
        self.materialize()
        segment = self._tokens.get(req_id)
        if segment is None:
            return None
        start, history = segment
        if not history:
            return None
        return start + len(history) - 1

    def next_output_position(self, req_id: str, scheduler_position: int) -> int:
        """Return the position at which the next target output is committed.

        Async speculative scheduling advances ``num_computed_tokens``
        optimistically (as if every draft token were accepted).  The sampled
        token copy, however, records the actual accepted prefix.  Once a
        request has a committed history, that prefix is the authoritative
        boundary for PLE: the final accepted token is the next target input,
        therefore the following output belongs at ``end + 1``.

        For requests without a committed row (new/prefill requests), retain
        the scheduler-derived position.  A mismatch is diagnostic only; the
        filler bridge in :meth:`commit_pending` remains the safety net for a
        genuinely discontinuous output.
        """
        scheduler_position = int(scheduler_position)
        committed_end = self.committed_context_end(req_id)
        if committed_end is None:
            return scheduler_position
        committed_position = committed_end + 1
        # Only correct a forward scheduler drift.  A lower scheduler boundary
        # is a real rewind (for example after KV-load recompute or rejection
        # rollback); retaining that position lets ``commit_pending`` replace
        # the abandoned suffix instead of appending tokens past the rewind.
        if scheduler_position < committed_position:
            logger.debug(
                "PLE output position keeps scheduler rewind for req_id=%s "
                "(scheduler_position=%d, committed_position=%d, committed_end=%d).",
                req_id,
                scheduler_position,
                committed_position,
                committed_end,
            )
            return scheduler_position
        if committed_position != scheduler_position:
            logger.debug(
                "PLE output position corrected from scheduler boundary for "
                "req_id=%s (scheduler_position=%d, committed_position=%d, "
                "committed_end=%d).",
                req_id,
                scheduler_position,
                committed_position,
                committed_end,
            )
        return committed_position

    def context(self, req_id: str, start_position: int, fallback_ids: Sequence[int]) -> list[int]:
        self.materialize()
        result = [int(token) for token in fallback_ids]
        segment = self._tokens.get(req_id)
        if segment is not None:
            start, history = segment
            for i in range(len(result)):
                offset = start_position + i - start
                if 0 <= offset < len(history) and history[offset] >= 0:
                    result[i] = history[offset]
        unresolved = [start_position + i for i, token in enumerate(result) if token < 0]
        if unresolved:
            # The context window can extend below the committed history after
            # a rewind (preemption / prefix cache eviction). The PLE n-gram
            # hashes tolerate a stale or filler token far better than a
            # crashed worker, so resolve the holes with a benign filler
            # instead of raising. -1 must never reach the NPU-side hash
            # computation.
            history_range = None
            if segment is not None:
                start, history = segment
                history_range = (start, start + len(history))
            logger.warning(
                "PLE context has unresolved placeholders for req_id=%s "
                "(context_range=(%d, %d), unresolved_positions=%s, "
                "history_range=%s). Filling with filler token; the affected "
                "request may see a slightly degraded n-gram context.",
                req_id,
                start_position,
                start_position + len(result),
                unresolved,
                history_range,
            )
            for i, token in enumerate(result):
                if token < 0:
                    result[i] = _PLE_CONTEXT_FILLER_TOKEN
        return result
