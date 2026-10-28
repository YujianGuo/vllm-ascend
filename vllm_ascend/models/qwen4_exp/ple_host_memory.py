# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owned, registered BF16 Host storage for the read-only PLE weights.

The region is registered (not merely allocated) so that the NPU can read it
directly through the device alias returned by ``aclrtHostGetDevicePointer``.
Acl APIs are declared with explicit argtypes/restype; without them ctypes
truncates the 64-bit address arguments and the region is silently aliased.

Lifetime contract: the registration must outlive every graph replay that
reads it, so the owner is held until process exit and released only through
``close()``. ``close()`` synchronizes the owning device first and, if that
fails, deliberately skips unmapping rather than risking an in-flight read.
"""

import atexit
import ctypes

import torch

# acl_rt.h: ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000
ACL_HOST_REG_FLAGS = ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED


class PleHostMemory:
    """A pinned, device-mapped BF16 Host buffer of ``rows x dim``."""

    def __init__(
        self,
        rows: int,
        dim: int,
        *,
        runtime=None,
        device: int | None = None,
    ) -> None:
        if rows <= 0 or dim <= 0:
            raise ValueError("PLE Host storage dimensions must be positive")
        self.rows = int(rows)
        self.dim = int(dim)
        self.device = torch.npu.current_device() if device is None else device
        self.runtime = runtime if runtime is not None else ctypes.CDLL("libascendcl.so")
        self._declare(self.runtime)

        self.base = ctypes.c_void_p()
        self.device_alias = 0
        self.closed = False
        self.allocated = False
        self.registered = False
        self.tensor: torch.Tensor | None = None
        self._raw = None

        size = self.rows * self.dim * 2
        try:
            self._call("aclrtMallocHost", ctypes.byref(self.base), size)
            self.allocated = True
            self._call(
                "aclrtHostRegisterV2",
                ctypes.c_void_p(self.base.value),
                ctypes.c_uint64(size),
                ctypes.c_uint32(ACL_HOST_REG_FLAGS),
            )
            self.registered = True
            ptr = ctypes.c_void_p()
            self._call(
                "aclrtHostGetDevicePointer",
                ctypes.c_void_p(self.base.value),
                ctypes.byref(ptr),
                ctypes.c_uint32(0),
            )
            if not ptr.value:
                raise RuntimeError("PLE Host mapping returned a null device address")
            self.device_alias = int(ptr.value)
            self._raw = (ctypes.c_ubyte * size).from_address(self.base.value)
            self.tensor = torch.frombuffer(self._raw, dtype=torch.bfloat16).reshape(
                self.rows, self.dim
            )
            if self.tensor.shape != (self.rows, self.dim):
                raise RuntimeError("PLE Host buffer shape mismatch")
        except BaseException:
            # Best-effort cleanup: a failure here must not replace the error
            # that actually caused the construction to fail.
            try:
                self._release()
            except Exception:
                pass
            raise
        atexit.register(self.close)

    @staticmethod
    def _declare(runtime) -> None:
        signatures = {
            "aclrtMallocHost": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "aclrtFreeHost": [ctypes.c_void_p],
            "aclrtHostRegisterV2": [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint32,
            ],
            "aclrtHostGetDevicePointer": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
            ],
            "aclrtHostUnregister": [ctypes.c_void_p],
        }
        for name, args in signatures.items():
            fn = getattr(runtime, name)
            fn.argtypes, fn.restype = args, ctypes.c_int

    def _call(self, name, *args):
        code = getattr(self.runtime, name)(*args)
        if code != 0:
            raise RuntimeError(f"PLE Host offload: {name} failed with ACL error {code}")

    def _release(self) -> None:
        # Unregister only if registration actually succeeded, so a failure in
        # ``aclrtHostRegisterV2`` does not mask the original error with a
        # bogus unregister failure. ``aclrtFreeHost`` is required whenever the
        # alloc succeeded, whether or not the register did.
        if self.registered:
            self._call("aclrtHostUnregister", ctypes.c_void_p(self.base.value))
            self.registered = False
        if self.allocated:
            self._call("aclrtFreeHost", self.base)
            self.allocated = False
        self.base = ctypes.c_void_p()
        self.device_alias = 0
        self.tensor = None
        self._raw = None
        self.closed = True

    def close(self) -> None:
        if not self.closed:
            # Fail closed: never unmap a region a replay may still be reading.
            try:
                torch.npu.synchronize(self.device)
            except Exception:
                return
            self._release()
            atexit.unregister(self.close)
