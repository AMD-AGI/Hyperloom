# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load standalone assembly with an explicit HIP kernel ABI and stream."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

ArgumentType = Literal["ptr", "i32", "u32", "i64", "u64", "f32", "f64"]

_ARGUMENT_TYPES = {
    "ptr": ctypes.c_void_p,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}
_INTEGER_RANGES = {
    "ptr": (0, 2**64),
    "i32": (-(2**31), 2**31),
    "u32": (0, 2**32),
    "i64": (-(2**63), 2**63),
    "u64": (0, 2**64),
}


class HipError(RuntimeError):
    """A HIP module operation failed."""


def _check(status: int, operation: str) -> None:
    if status:
        raise HipError(f"{operation} failed with HIP error {status}")


def _dimensions(values: tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if len(values) != 3 or any(not isinstance(v, int) or not 0 < v < 2**32 for v in values):
        raise ValueError(f"{name} must contain three positive uint32 dimensions")
    return values


class HipKernel:
    """Own one freshly loaded kernel; the caller supplies its verified ABI.

    Initialize the intended HIP device before construction. Pointer arguments
    are device addresses, not tensors. No tensor shape, stride, dtype, symbol,
    argument layout, or launch geometry is inferred from a source language.
    Match these to the AMDHSA metadata and validate them in the kernel driver.

    Build/load before graph capture. Every launch receives fresh arguments and
    an explicit stream handle. Retain this object while any captured graph can
    run; call close only after GPU work finishes and those graphs are retired.
    Module unloading is explicit to avoid invalidating an in-flight graph from
    a Python finalizer. The HIP context releases unclosed modules at process exit.
    """

    def __init__(self, code_object: Path | str, symbol: str, argument_types: Sequence[ArgumentType]) -> None:
        self.argument_types = tuple(argument_types)
        if any(kind not in _ARGUMENT_TYPES for kind in self.argument_types):
            raise ValueError(f"Unsupported HIP argument type in {self.argument_types!r}")
        if not symbol or "\0" in symbol:
            raise ValueError("symbol must be a nonempty kernel name without NUL bytes")
        self._image = ctypes.create_string_buffer(Path(code_object).read_bytes())
        self._lib = ctypes.CDLL("libamdhip64.so")
        signatures = {
            "hipGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "hipModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            "hipModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            "hipModuleUnload": [ctypes.c_void_p],
            "hipModuleLaunchKernel": [
                ctypes.c_void_p,
                *([ctypes.c_uint] * 7),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        }
        for name, argtypes in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        self._device = self._current_device()
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._closed = False
        _check(self._lib.hipModuleLoadData(ctypes.byref(self._module), self._image), "hipModuleLoadData")
        status = self._lib.hipModuleGetFunction(ctypes.byref(self._function), self._module, symbol.encode())
        if status:
            self._lib.hipModuleUnload(self._module)
            self._closed = True
            _check(status, "hipModuleGetFunction")

    def _current_device(self) -> int:
        device = ctypes.c_int()
        _check(self._lib.hipGetDevice(ctypes.byref(device)), "hipGetDevice")
        return device.value

    def _require_device(self) -> None:
        if self._current_device() != self._device:
            raise HipError(f"Kernel belongs to HIP device {self._device}; activate that device before use")

    def launch(
        self,
        arguments: Sequence[int | float],
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        stream: int,
        shared_memory_bytes: int = 0,
    ) -> None:
        """Enqueue the kernel without synchronizing or caching argument addresses."""
        if self._closed:
            raise HipError("Kernel module is closed")
        if len(arguments) != len(self.argument_types):
            raise ValueError(f"Expected {len(self.argument_types)} kernel arguments, got {len(arguments)}")
        grid, block = _dimensions(grid, "grid"), _dimensions(block, "block")
        if not isinstance(stream, int) or not 0 <= stream < 2**64:
            raise ValueError("stream must be a uint64 HIP stream handle")
        if not isinstance(shared_memory_bytes, int) or not 0 <= shared_memory_bytes < 2**32:
            raise ValueError("shared_memory_bytes must fit uint32")
        packed = []
        for kind, value in zip(self.argument_types, arguments, strict=True):
            if kind in _INTEGER_RANGES:
                low, high = _INTEGER_RANGES[kind]
                if not isinstance(value, int) or not low <= value < high:
                    raise ValueError(f"Argument {value!r} does not fit {kind}")
                packed.append(_ARGUMENT_TYPES[kind](value))
            else:
                packed.append(ctypes.c_float(value) if kind == "f32" else ctypes.c_double(value))
        pointers = (ctypes.c_void_p * len(packed))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in packed)
        )
        self._require_device()
        _check(
            self._lib.hipModuleLaunchKernel(
                self._function, *grid, *block, shared_memory_bytes, ctypes.c_void_p(stream), pointers, None
            ),
            "hipModuleLaunchKernel",
        )

    def close(self) -> None:
        """Unload after synchronization and graph retirement; repeated calls are harmless."""
        if not self._closed:
            self._require_device()
            _check(self._lib.hipModuleUnload(self._module), "hipModuleUnload")
            self._closed = True
