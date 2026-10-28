# mypy: ignore-errors

import itertools
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
from vllm.v1.worker.mamba_utils import MambaCopyBuffers, MambaSpecDecodeGPUContext

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.mamba.postprocess import (
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
)
from vllm_ascend.utils import is_310p


def _can_launch_triton_batch_memcpy() -> bool:
    return not is_310p()


def _batch_memcpy_triton(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def _tensor_view_from_data_ptr(state: torch.Tensor, start_addr: int, num_elements: int) -> torch.Tensor:
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")

    element_offset = byte_offset // element_size
    flat_state = state.view(-1)
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def _get_mamba_spec_for_layer(
    kv_cache_group: KVCacheGroupSpec, layer_name: str
) -> MambaSpec:
    kv_cache_spec = kv_cache_group.kv_cache_spec
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
    assert isinstance(kv_cache_spec, MambaSpec)
    return kv_cache_spec


def get_mamba_groups(
    kv_cache_config: KVCacheConfig,
) -> dict[MambaSpec, list[int]]:
    """Group Mamba managers by their per-layer state contract."""
    mamba_groups: dict[MambaSpec, set[int]] = {}
    for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        group_spec = kv_cache_group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = next(iter(group_spec.kv_cache_specs.values()))
        if not isinstance(group_spec, MambaSpec):
            continue
        for layer_name in kv_cache_group.layer_names:
            layer_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
            mamba_groups.setdefault(layer_spec, set()).add(group_id)
    assert mamba_groups, "no mamba layers in the model"
    return {spec: sorted(group_ids) for spec, group_ids in mamba_groups.items()}


def get_mamba_group_ids(mamba_groups: dict[MambaSpec, list[int]]) -> list[int]:
    return sorted(
        {group_id for group_ids in mamba_groups.values() for group_id in group_ids}
    )


def _copy_funcs_for_layer(
    kv_cache_group: KVCacheGroupSpec,
    layer_name: str,
    copy_funcs,
) -> tuple[MambaStateCopyFunc, ...]:
    if isinstance(copy_funcs, dict):
        layer_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
        return copy_funcs[layer_spec.mamba_type]
    return copy_funcs


def validate_mamba_state_copy_funcs(
    mamba_groups: dict[MambaSpec, list[int]], copy_funcs
) -> None:
    for mamba_spec in mamba_groups:
        if isinstance(copy_funcs, dict):
            assert mamba_spec.mamba_type in copy_funcs, (
                f"missing state copy funcs for {mamba_spec.mamba_type}"
            )
            state_copy_funcs = copy_funcs[mamba_spec.mamba_type]
        else:
            state_copy_funcs = copy_funcs
        assert len(state_copy_funcs) == len(mamba_spec.shapes), (
            f"{mamba_spec.mamba_type} declares {len(mamba_spec.shapes)} states, "
            f"but provides {len(state_copy_funcs)} state copy funcs"
        )


def _get_tensor_copy_pairs(copy_bufs: mamba_utils.MambaCopyBuffers) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if copy_bufs.offset == 0 or not hasattr(copy_bufs, "_tensor_copy_pairs"):
        copy_bufs._tensor_copy_pairs = []
    return copy_bufs._tensor_copy_pairs


def _collect_mamba_copy_meta_torch(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    tensor_copy_pairs = _get_tensor_copy_pairs(copy_bufs)
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
        layer_names = kv_cache_group.layer_names
        for layer_name in layer_names:
            state_copy_funcs = _copy_funcs_for_layer(
                kv_cache_group, layer_name, mamba_state_copy_funcs
            )
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, state_copy_funcs):
                copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                dst_state = _tensor_view_from_data_ptr(state, state[dest_block_id].data_ptr(), copy_spec.num_elements)
                tensor_copy_pairs.append((src_state, dst_state))
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def _collect_mamba_copy_meta(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
        for layer_name in kv_cache_group.layer_names:
            state_copy_funcs = _copy_funcs_for_layer(
                kv_cache_group, layer_name, mamba_state_copy_funcs
            )
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            assert len(kv_caches) == len(state_copy_funcs), (
                f"{layer_name} exposes {len(kv_caches)} states, but has "
                f"{len(state_copy_funcs)} copy funcs"
            )
            for state, state_copy_func in zip(kv_caches, state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )
                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def _do_mamba_copy_block_torch(copy_bufs: mamba_utils.MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        if hasattr(copy_bufs, "_tensor_copy_pairs"):
            copy_bufs._tensor_copy_pairs = []
        return

    tensor_copy_pairs = getattr(copy_bufs, "_tensor_copy_pairs", None)
    if tensor_copy_pairs is None or len(tensor_copy_pairs) != n:
        raise RuntimeError("Mamba tensor copy metadata is incomplete.")

    for src_state, dst_state in tensor_copy_pairs:
        dst_state.copy_(src_state.clone())
    copy_bufs._tensor_copy_pairs = []


def _postprocess_mamba_align_gpu_cpu_fallback(
    *,
    bufs: "mamba_utils.MambaBuffers",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    """CPU fallback for 310P where the Triton fused postprocess is unavailable."""
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    # stage_postprocess_inputs_to_gpu has already materialized the same
    # per-request values into the CpuGpuBuffer numpy views. 310P cannot use the
    # Triton fused kernel, so reuse the CPU views to mirror its decision logic.
    mamba_state_idx = ctx.mamba_state_idx_buf.np
    num_scheduled_tokens = ctx.num_scheduled_tokens_buf.np
    num_computed_tokens = ctx.num_computed_tokens_buf.np
    num_draft_tokens = ctx.num_draft_tokens_buf.np
    block_size = ctx.block_size

    # Upstream initializes num_accepted_tokens_out from the real accepted-token
    # counts, then only overwrites entries where src and dest are the same
    # block. Preserve that default so the next preprocess keeps the right
    # accept_token_bias when multiple draft tokens were accepted.
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])
    num_accepted_tokens = input_batch.num_accepted_tokens_cpu
    for i in range(num_reqs):
        num_tokens_running_state = num_computed_tokens[i] + num_scheduled_tokens[i] - num_draft_tokens[i]
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[i] - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens < num_tokens_running_state:
            continue

        src_block_idx = mamba_state_idx[i]
        dest_block_idx = aligned_new_computed_tokens // block_size - 1
        accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
        if src_block_idx == dest_block_idx:
            # Match the fused kernel: once the running state remains in the
            # same block, the next preprocess should start from token bias 0.
            num_accepted_tokens_cpu_tensor[i] = 1
            if accept_token_bias == 0:
                continue

        # The upstream fused kernel also copies Mamba state in this postprocess
        # step. Do the same with tensor views so 310P avoids Triton without
        # changing where conv/temporal state lands before the next iteration.
        for mamba_group_id in ctx.mamba_group_ids:
            block_ids = input_batch.block_table[mamba_group_id].get_numpy_array()[i]
            dest_block_id = block_ids[dest_block_idx]
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                    copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                    src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                    dst_state = _tensor_view_from_data_ptr(
                        state, state[dest_block_id].data_ptr(), copy_spec.num_elements
                    )
                    dst_state.copy_(src_state.clone())


def _batch_memcpy_unavailable(src_ptrs, dst_ptrs, sizes):
    raise RuntimeError(
        "Pointer-based Mamba batch memcpy requires Triton and is not available "
        "on 310P. Use the tensor-copy fallback path instead."
    )


if _can_launch_triton_batch_memcpy():
    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy_triton
    mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel
    mamba_utils.precopy_mamba_align_fused_kernel = precopy_mamba_align_fused_kernel
    mamba_utils.collect_mamba_copy_meta = _collect_mamba_copy_meta
else:
    mamba_utils.batch_memcpy = _batch_memcpy_unavailable
    mamba_utils.collect_mamba_copy_meta = _collect_mamba_copy_meta_torch
    mamba_utils.do_mamba_copy_block = _do_mamba_copy_block_torch
    mamba_utils.postprocess_mamba_align_gpu = _postprocess_mamba_align_gpu_cpu_fallback

# Ascend NPU does not support DT_UINT64 in aclnnInplaceZero.
# MambaCopyBuffers.create() uses torch.uint64 for src_ptrs/dst_ptrs,
# which triggers a runtime error. Remap to int64 at the source.
@classmethod
def _patched_create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
    mamba_groups = get_mamba_groups(kv_cache_config)
    validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
    mamba_spec = next(iter(mamba_groups))
    assert all(
        spec.block_size == mamba_spec.block_size
        and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
        and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
        for spec in mamba_groups
    ), "all mamba groups must share cache scheduling parameters"

    mamba_group_ids = get_mamba_group_ids(mamba_groups)
    entries_per_req = sum(
        len(_copy_funcs_for_layer(kv_cache_group, layer_name, copy_funcs))
        for group_id in mamba_group_ids
        for kv_cache_group in (kv_cache_config.kv_cache_groups[group_id],)
        for layer_name in kv_cache_group.layer_names
    )
    num_entries = max_num_reqs * entries_per_req
    return cls(
        src_ptrs=make_buffer(num_entries, dtype=torch.int64),
        dst_ptrs=make_buffer(num_entries, dtype=torch.int64),
        sizes=make_buffer(num_entries, dtype=torch.int32),
        mamba_group_ids=mamba_group_ids,
        mamba_spec=mamba_spec,
    )


MambaCopyBuffers.create = _patched_create


@classmethod
def _patched_spec_decode_context_create(
    cls,
    max_num_reqs,
    kv_cache_config,
    num_state_types,
    device,
    make_buffer,
):
    del num_state_types
    mamba_groups = get_mamba_groups(kv_cache_config)
    mamba_spec = next(iter(mamba_groups))
    assert all(
        spec.block_size == mamba_spec.block_size
        and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
        and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
        for spec in mamba_groups
    ), "all mamba groups must share cache scheduling parameters"
    mamba_group_ids = get_mamba_group_ids(mamba_groups)
    total_states = sum(
        len(_get_mamba_spec_for_layer(kv_cache_group, layer_name).shapes)
        for group_id in mamba_group_ids
        for kv_cache_group in (kv_cache_config.kv_cache_groups[group_id],)
        for layer_name in kv_cache_group.layer_names
    )

    # Upstream represents the launch width as num_layers * num_state_types.
    # Mixed GDN/PLE layers do not have a rectangular state matrix, so retain
    # the exact flattened state count as total_states * 1.
    context = cls(
        state_base_addrs=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_block_strides=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_elem_sizes=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_inner_sizes=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_conv_widths=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_group_indices=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_dim_row_count=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_dim_row_stride=torch.zeros(total_states, dtype=torch.int64, device=device),
        block_size=mamba_spec.block_size,
        num_layers=total_states,
        num_state_types=1,
        mamba_group_ids=mamba_group_ids,
        num_groups=len(mamba_group_ids),
        num_accepted_tokens_out=torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        ),
        block_table_ptrs=torch.zeros(
            len(mamba_group_ids), dtype=torch.int64, device=device
        ),
        mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        is_initialized=False,
    )
    # Backport vLLM f51193b9ae's V1 align pre-copy staging. The v0.26
    # dataclass does not declare these fields yet, so attach them here.
    context.precopy_src_col_buf = make_buffer(max_num_reqs, dtype=torch.int32)
    context.precopy_token_bias_buf = make_buffer(max_num_reqs, dtype=torch.int32)
    # v0.26's kernel requires an explicit mapping. V1 state arrays already use
    # batch order, making the identity mapping the correct one.
    context.precopy_idx_mapping = torch.arange(
        max_num_reqs, dtype=torch.int32, device=device
    )
    return context


MambaSpecDecodeGPUContext.create = _patched_spec_decode_context_create


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs,
    copy_bufs: MambaCopyBuffers,
    align_ctx: MambaSpecDecodeGPUContext | None = None,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    # assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    num_reqs = len(input_batch.req_ids)
    if align_ctx is not None:
        if num_reqs == 0:
            return
        if not align_ctx.is_initialized:
            # The v0.26 initializer accepts one tuple, while Qwen4Exp exposes
            # GDN=(conv, temporal) and PLE=(conv,). Their contracts are prefix
            # compatible, so the longest tuple identifies every physical
            # state encountered by the initializer.
            if isinstance(mamba_state_copy_funcs, dict):
                init_copy_funcs = max(mamba_state_copy_funcs.values(), key=len)
                assert all(
                    funcs == init_copy_funcs[: len(funcs)]
                    for funcs in mamba_state_copy_funcs.values()
                ), "mixed Mamba copy funcs must be prefix-compatible"
            else:
                init_copy_funcs = mamba_state_copy_funcs
            align_ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                init_copy_funcs,
                [
                    input_batch.block_table[group_id].get_device_tensor(num_reqs)
                    for group_id in align_ctx.mamba_group_ids
                ],
            )
        align_ctx.precopy_src_col_buf.np[:num_reqs] = -1
        align_ctx.precopy_token_bias_buf.np[:num_reqs] = 0

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if align_ctx is not None:
            align_ctx.mamba_state_idx_buf.np[i] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            # A speculative/structured-output row can legitimately have no
            # committed token in the previous step (the rejection sampler
            # leaves the row fully padded with PLACEHOLDER_TOKEN_ID).  The
            # upstream align kernel maps this case to bias=0, which copies
            # the state from the beginning of the previous block.  Do not
            # assert here: -1 would crash every worker before the next
            # scheduler step and turn a recoverable empty-accept row into a
            # DP/TP engine failure.
            accept_token_bias = max(int(input_batch.num_accepted_tokens_cpu[i]) - 1, 0)
            if align_ctx is not None:
                align_ctx.precopy_src_col_buf.np[i] = prev_state_idx
                align_ctx.precopy_token_bias_buf.np[i] = accept_token_bias
            else:
                mamba_utils.collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    mamba_group_ids,
                    prev_state_idx,
                    curr_state_idx,
                    accept_token_bias,
                    req_state,
                    forward_context,
                )
            input_batch.num_accepted_tokens_cpu[i] = 1
    if align_ctx is not None:
        align_ctx.mamba_state_idx_buf.copy_to_gpu(num_reqs)
        align_ctx.precopy_src_col_buf.copy_to_gpu(num_reqs)
        align_ctx.precopy_token_bias_buf.copy_to_gpu(num_reqs)
        align_ctx.run_fused_precopy(
            num_reqs=num_reqs,
            state_idx_gpu=align_ctx.mamba_state_idx_buf.gpu,
            src_col_gpu=align_ctx.precopy_src_col_buf.gpu,
            token_bias_gpu=align_ctx.precopy_token_bias_buf.gpu,
            idx_mapping=align_ctx.precopy_idx_mapping[:num_reqs],
        )
    # The scalar path intentionally remains deferred until KV transfer is
    # loaded. Speculative align mode always supplies align_ctx and therefore
    # performs the required pre-copy above.


mamba_utils.preprocess_mamba = preprocess_mamba
