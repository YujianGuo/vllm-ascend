# SPDX-License-Identifier: Apache-2.0
"""Fix out-of-range stop-token indexing when the LM head is TP-sharded.

``MinTokensLogitsProcessor`` (active whenever ``min_tokens`` is requested, and
always wired because the API server adds the model's EOS ids to
``all_stop_token_ids``) writes ``-inf`` into ``logits`` with
``logits.index_put_((req_idx, stop_token_ids), ...)``. With TP > 1 the LM head
is sharded, so ``logits`` only spans this rank's slice of the vocabulary:
indexing with global token ids either corrupts unrelated rows or trips the
device-side index bounds check (observed as
``Index out of range in dimension 1: index value 248044 exceeds bounds 31040``
on Ascend, where 31040 = 248320 / 8). The same global-id indexing is applied by
the spec-decode variant ``apply_with_spec_decode``.

The processor keeps per-request global stop ids on the CPU side, so remapping
them to local column ids (and dropping ids owned by other ranks) can happen at
state-update time without touching the hot ``apply`` path.
"""

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    pad_vocab_size,
    vocab_range_from_global_vocab_size,
)
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor


def _patched_init(self, vllm_config, device, is_pin_memory: bool):
    _original_init(self, vllm_config, device, is_pin_memory)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    if tp_size > 1:
        vocab_size = vllm_config.model_config.get_vocab_size()
        org_vocab_size_padded = pad_vocab_size(
            vocab_size, DEFAULT_VOCAB_PADDING_SIZE
        )
        rank = get_tensor_model_parallel_rank()
        self._shard_start, self._shard_end = vocab_range_from_global_vocab_size(
            org_vocab_size_padded, rank, tp_size
        )
    else:
        self._shard_start = 0
        self._shard_end = None


_original_init = MinTokensLogitsProcessor.__init__


def _local_stop_id(self, tok_id: int) -> int | None:
    """Map a global token id onto this rank's sharded logits columns.

    Returns None when the token lives in another rank's vocabulary shard and
    therefore must not be masked here.
    """
    if self._shard_end is None:
        return tok_id
    if self._shard_start <= tok_id < self._shard_end:
        return tok_id - self._shard_start
    return None


def _patched_update_state(self, batch_update):
    _original_update_state(self, batch_update)
    # ``_original_update_state`` already rebuilt ``logits_slice``; remap the
    # global stop ids it stored into this shard's local columns.
    if self._shard_end is None or not self.min_toks:
        return
    reqs, tok_ids = self.logits_slice
    if reqs.numel() == 0:
        return
    global_ids = tok_ids.tolist()
    local_reqs: list[int] = []
    local_ids: list[int] = []
    for req, tok in zip(reqs.tolist(), global_ids):
        local = self._local_stop_id(int(tok))
        if local is None:
            continue
        local_reqs.append(int(req))
        local_ids.append(local)
    self.logits_slice = (
        self._device_tensor(local_reqs, torch.int32),
        self._device_tensor(local_ids, torch.int32),
    )


def _patched_apply_with_spec_decode(self, logits, num_draft_tokens):
    if not self.min_toks:
        return logits
    import numpy as np
    from vllm.utils.torch_utils import async_tensor_h2d

    num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
    cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

    entries = [
        (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
        for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
        if stop_tok_ids
    ]
    if not entries:
        return logits

    all_rows: list[np.ndarray] = []
    all_toks: list[np.ndarray] = []
    for req_idx, min_tok, current_len, stop_toks in entries:
        remaining = min_tok - current_len
        n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))
        if n_mask > 0:
            local_toks = [
                local
                for tok in stop_toks
                if (local := _local_stop_id(self, tok)) is not None
            ]
            if not local_toks:
                continue
            offset = cumsum[req_idx]
            row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
            all_rows.append(np.repeat(row_indices, len(local_toks)))
            all_toks.append(np.tile(local_toks, n_mask))

    if all_rows:
        logits_slice = (
            async_tensor_h2d(np.concatenate(all_rows), device=self.device),
            async_tensor_h2d(np.concatenate(all_toks), device=self.device),
        )
        logits.index_put_(logits_slice, self.neg_inf_tensor)
    return logits


import torch  # noqa: E402  (used by _patched_update_state)

_original_update_state = MinTokensLogitsProcessor.update_state
_original_apply_with_spec_decode = MinTokensLogitsProcessor.apply_with_spec_decode

MinTokensLogitsProcessor.__init__ = _patched_init
MinTokensLogitsProcessor._local_stop_id = _local_stop_id
MinTokensLogitsProcessor.update_state = _patched_update_state
MinTokensLogitsProcessor.apply_with_spec_decode = _patched_apply_with_spec_decode
