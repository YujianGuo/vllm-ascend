# SPDX-License-Identifier: Apache-2.0
"""Cleanup contract for the PLE Host region.

Every partial-failure path must release exactly what it acquired: freeing
without registering is correct, unregistering without a successful register
is not, and a cleanup failure must never replace the original error.
"""

import ctypes

import pytest

from vllm_ascend.models.qwen4_exp.ple_host_memory import (
    ACL_HOST_REG_MAPPED,
    ACL_HOST_REG_PINNED,
    PleHostMemory,
)


class FakeRuntime:
    """Records ACL calls; optionally fails one of them."""

    def __init__(self, fail_at=None, null_device_pointer=False):
        self.calls = []
        self.fail_at = fail_at
        self.null_device_pointer = null_device_pointer

    def _make(self, name):
        def call(*args):
            self.calls.append(name)
            if name == self.fail_at:
                return 1
            if name == "aclrtHostGetDevicePointer" and self.null_device_pointer:
                return 0  # succeeds but leaves the pointer null
            return 0

        return call

    def __getattr__(self, name):
        if name.startswith("aclrt"):
            return self._make(name)
        raise AttributeError(name)


def test_malloc_pointer_failure_frees_nothing():
    rt = FakeRuntime(fail_at="aclrtMallocHost")
    with pytest.raises(RuntimeError):
        PleHostMemory(4, 8, runtime=rt, device=0)
    assert rt.calls == ["aclrtMallocHost"]


def test_register_failure_frees_host_but_does_not_unregister():
    rt = FakeRuntime(fail_at="aclrtHostRegisterV2")
    with pytest.raises(RuntimeError):
        PleHostMemory(4, 8, runtime=rt, device=0)
    assert "aclrtHostUnregister" not in rt.calls
    assert rt.calls[-1] == "aclrtFreeHost"


def test_mapping_failure_unregisters_then_frees():
    rt = FakeRuntime(fail_at="aclrtHostGetDevicePointer")
    with pytest.raises(RuntimeError):
        PleHostMemory(4, 8, runtime=rt, device=0)
    assert rt.calls[-2:] == ["aclrtHostUnregister", "aclrtFreeHost"]


def test_null_device_pointer_is_rejected_and_released():
    rt = FakeRuntime(null_device_pointer=True)
    with pytest.raises(RuntimeError, match="null device address"):
        PleHostMemory(4, 8, runtime=rt, device=0)
    assert rt.calls[-2:] == ["aclrtHostUnregister", "aclrtFreeHost"]


def test_register_flags_match_acl_header():
    # acl_rt.h: MAPPED 0x2, PINNED 0x10000000 -> 0x10000002.
    assert (ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED) == 0x10000002
