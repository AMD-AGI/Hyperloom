# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the standalone HIP launch boundary without loading a GPU runtime."""

from __future__ import annotations

import ctypes
from unittest.mock import Mock

import pytest

from kernelforge.assembly import hip


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path / "kernel.co"
    path.write_bytes(b"first image")
    library = Mock()
    device = [0]
    images, launches, unloaded = [], [], []

    def get_device(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = device[0]
        return 0

    def load(pointer, image):
        images.append(bytes(image))
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 100 + len(images)
        return 0

    def symbol(pointer, module, name):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = module.value + 100
        return 0

    library.hipGetDevice.side_effect = get_device
    library.hipModuleLoadData.side_effect = load
    library.hipModuleGetFunction.side_effect = symbol
    library.hipModuleUnload.side_effect = lambda module: unloaded.append(module.value) or 0
    library.hipModuleLaunchKernel.return_value = 0
    monkeypatch.setattr(hip.ctypes, "CDLL", lambda name: library)
    return path, library, device, images, launches, unloaded


def test_fresh_images_typed_arguments_and_stream_are_forwarded(runtime):
    path, library, _, images, launches, unloaded = runtime
    kinds = ["ptr", "i32", "u32", "i64", "u64", "f32", "f64"]
    first = hip.HipKernel(path, "entry", kinds)
    path.write_bytes(b"second image")
    second = hip.HipKernel(path, "entry", kinds)

    def launch(function, gx, gy, gz, bx, by, bz, shared, stream, pointers, extra):
        values = [
            ctypes.cast(pointer, ctypes.POINTER(hip._ARGUMENT_TYPES[kind]))[0] for kind, pointer in zip(kinds, pointers)
        ]
        launches.append((function.value, (gx, gy, gz), (bx, by, bz), shared, stream.value, values))
        assert extra is None
        return 0

    library.hipModuleLaunchKernel.side_effect = launch
    args = [0xFFFF00000001, -17, 2**32 - 1, -(2**62), 2**63 + 1, 0.125, 0.0625]
    for kernel, pointer in [(first, args[0]), (second, 0x100000001), (first, 0x200000001)]:
        args[0] = pointer
        kernel.launch(args, grid=(2, 3, 1), block=(64, 1, 1), stream=0x123456789, shared_memory_bytes=128)
        assert launches[-1][1:] == ((2, 3, 1), (64, 1, 1), 128, 0x123456789, args)
    assert [item[0] for item in launches] == [201, 202, 201]
    assert images == [b"first image\0", b"second image\0"]
    first.close()
    first.close()
    assert unloaded == [101]
    second.launch(args, grid=(1, 1, 1), block=(64, 1, 1), stream=0)
    with pytest.raises(hip.HipError, match="closed"):
        first.launch(args, grid=(1, 1, 1), block=(64, 1, 1), stream=0)
    second.close()


@pytest.mark.parametrize("value", [-1, 2**64, 1.5])
def test_pointer_overflow_is_rejected_before_launch(runtime, value):
    path, library, *_ = runtime
    kernel = hip.HipKernel(path, "entry", ["ptr"])
    with pytest.raises(ValueError, match="does not fit ptr"):
        kernel.launch([value], grid=(1, 1, 1), block=(64, 1, 1), stream=0)
    library.hipModuleLaunchKernel.assert_not_called()
    kernel.close()


def test_device_and_launch_contract_fail_closed(runtime):
    path, library, device, *_ = runtime
    kernel = hip.HipKernel(path, "entry", ["i32"])
    options = {"grid": (1, 1, 1), "block": (64, 1, 1), "stream": 0}
    for args, overrides in [
        ([], {}),
        ([2**31], {}),
        ([1], {"grid": (0, 1, 1)}),
        ([1], {"stream": -1}),
        ([1], {"shared_memory_bytes": -1}),
    ]:
        with pytest.raises(ValueError):
            kernel.launch(args, **(options | overrides))
    device[0] = 1
    with pytest.raises(hip.HipError, match="device 0"):
        kernel.launch([1], **options)
    library.hipModuleLaunchKernel.assert_not_called()
    device[0] = 0
    library.hipModuleLaunchKernel.return_value = 700
    with pytest.raises(hip.HipError, match="hipModuleLaunchKernel.*700"):
        kernel.launch([1], **options)
    kernel.close()


def test_symbol_lookup_failure_releases_loaded_module(runtime):
    path, library, *_, unloaded = runtime
    library.hipModuleGetFunction.side_effect = lambda *args: 500
    with pytest.raises(hip.HipError, match="hipModuleGetFunction.*500"):
        hip.HipKernel(path, "missing", [])
    assert unloaded == [101]


def test_load_failure_is_propagated_without_using_a_function(runtime):
    path, library, *_ = runtime
    library.hipModuleLoadData.side_effect = lambda *args: 200
    with pytest.raises(hip.HipError, match="hipModuleLoadData.*200"):
        hip.HipKernel(path, "entry", [])
    library.hipModuleGetFunction.assert_not_called()
    library.hipModuleLaunchKernel.assert_not_called()
