# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.mamba_utils import MambaCopyBuffers, MambaSpecDecodeGPUContext

from vllm_ascend.patch.worker.patch_mamba_utils import (
    _patched_create,
    _patched_spec_decode_context_create,
    get_mamba_groups,
    preprocess_mamba,
    validate_mamba_state_copy_funcs,
)


def _mixed_mamba_config() -> KVCacheConfig:
    scheduling = {
        "block_size": 16,
        "mamba_cache_mode": "align",
        "num_speculative_blocks": 0,
    }
    gdn_spec = MambaSpec(
        shapes=((4, 8), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        **scheduling,
    )
    ple_spec = MambaSpec(
        shapes=((12, 8),),
        dtypes=(torch.bfloat16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        **scheduling,
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["gdn"], gdn_spec),
            KVCacheGroupSpec(["ple"], ple_spec),
        ],
    )


def test_mixed_gdn_ple_copy_buffers_use_per_type_state_counts():
    kv_cache_config = _mixed_mamba_config()
    copy_funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: (
            MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        ),
        MambaAttentionBackendEnum.SHORT_CONV: (
            MambaStateCopyFuncCalculator.short_conv_state_copy_func()
        ),
    }

    mamba_groups = get_mamba_groups(kv_cache_config)
    assert {spec.mamba_type for spec in mamba_groups} == {
        MambaAttentionBackendEnum.GDN_ATTN,
        MambaAttentionBackendEnum.SHORT_CONV,
    }

    copy_buffers = _patched_create.__func__(
        MambaCopyBuffers,
        max_num_reqs=4,
        kv_cache_config=kv_cache_config,
        copy_funcs=copy_funcs,
        make_buffer=lambda n, dtype: torch.empty(n, dtype=dtype),
    )
    assert copy_buffers.mamba_group_ids == [0, 1]
    assert copy_buffers.src_ptrs.numel() == 4 * (2 + 1)
    assert copy_buffers.dst_ptrs.dtype == torch.int64


def test_mixed_gdn_ple_rejects_wrong_ple_copy_contract():
    kv_cache_config = _mixed_mamba_config()
    gdn_copy_funcs = MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
    copy_funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: gdn_copy_funcs,
        MambaAttentionBackendEnum.SHORT_CONV: gdn_copy_funcs,
    }

    with pytest.raises(AssertionError, match="SHORT_CONV declares 1 states"):
        validate_mamba_state_copy_funcs(
            get_mamba_groups(kv_cache_config), copy_funcs
        )


def test_mixed_gdn_ple_spec_decode_context_uses_exact_state_count():
    kv_cache_config = _mixed_mamba_config()
    context = _patched_spec_decode_context_create.__func__(
        MambaSpecDecodeGPUContext,
        max_num_reqs=4,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=torch.device("cpu"),
        make_buffer=lambda n, dtype: torch.empty(n, dtype=dtype),
    )
    assert context.mamba_group_ids == [0, 1]
    assert context.num_layers * context.num_state_types == 3
    assert context.state_base_addrs.numel() == 3
    assert context.state_group_indices.numel() == 3
    assert context.precopy_src_col_buf.numel() == 4
    assert context.precopy_token_bias_buf.numel() == 4
    assert torch.equal(context.precopy_idx_mapping, torch.arange(4, dtype=torch.int32))


class _FakeCpuGpuBuffer:
    def __init__(self, size: int):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = object()

    def copy_to_gpu(self, size=None):
        return self.gpu


class _FakeAlignContext:
    def __init__(self, size: int):
        self.is_initialized = True
        self.mamba_group_ids = [0]
        self.mamba_state_idx_buf = _FakeCpuGpuBuffer(size)
        self.precopy_src_col_buf = _FakeCpuGpuBuffer(size)
        self.precopy_token_bias_buf = _FakeCpuGpuBuffer(size)
        self.precopy_idx_mapping = torch.arange(size, dtype=torch.int32)
        self.calls = []

    def run_fused_precopy(self, **kwargs):
        self.calls.append(kwargs)


def test_preprocess_mamba_uses_fused_precopy_for_multi_accept_boundary():
    req_ids = ["fresh", "same", "cross", "zero"]
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"fresh": 1, "same": 1, "cross": 1, "zero": 1},
    )
    input_batch = SimpleNamespace(
        req_ids=req_ids,
        num_accepted_tokens_cpu=np.array([1, 2, 2, 0], dtype=np.int32),
    )
    requests = {
        "fresh": SimpleNamespace(num_computed_tokens=0),
        "same": SimpleNamespace(num_computed_tokens=5),
        "cross": SimpleNamespace(num_computed_tokens=8),
        "zero": SimpleNamespace(num_computed_tokens=8),
    }
    state_indices = {"same": 1, "cross": 0, "zero": 0}
    copy_bufs = SimpleNamespace(
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(block_size=4, num_speculative_blocks=1),
        offset=0,
    )
    context = _FakeAlignContext(len(req_ids))

    preprocess_mamba(
        scheduler_output=scheduler_output,
        kv_cache_config=SimpleNamespace(),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        mamba_state_idx=state_indices,
        input_batch=input_batch,
        requests=requests,
        forward_context={},
        mamba_state_copy_funcs=(),
        copy_bufs=copy_bufs,
        align_ctx=context,
    )

    assert context.mamba_state_idx_buf.np.tolist() == [0, 1, 2, 2]
    assert context.precopy_src_col_buf.np.tolist() == [-1, -1, 0, 0]
    assert context.precopy_token_bias_buf.np.tolist() == [0, 0, 1, 0]
    assert input_batch.num_accepted_tokens_cpu.tolist() == [1, 2, 1, 1]
    assert len(context.calls) == 1
    assert torch.equal(
        context.calls[0]["idx_mapping"], torch.arange(4, dtype=torch.int32)
    )
