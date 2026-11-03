"""Loading the runtime's C API and the marker library through ctypes.

`ruppell run` puts the paths in the environment (RUPPELL_TOOL_LIB,
RUPPELL_MARKER_LIB) and the ROCm profiler SDK has already loaded the tool
library into this process; loading the same path again returns that handle.

On CUDA the driver loads the tool at cuInit (CUDA_INJECTION64_PATH), which
PyTorch defers until the first CUDA call; `tool()` calls cuInit first so the
tool is loaded and configured before anything asks whether it is active.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

_tool: Optional[ctypes.CDLL] = None
_marker: Optional[ctypes.CDLL] = None


def tool() -> Optional[ctypes.CDLL]:
    global _tool
    if _tool is None:
        path = os.environ.get("RUPPELL_TOOL_LIB")
        if not path:
            return None
        if os.environ.get("CUDA_INJECTION64_PATH") == path:
            rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
            if rc != 0:
                raise RuntimeError(f"cuInit failed ({rc}): the CUDA driver did not start, so the tool is not loaded")
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        lib.ruppell_active.restype = ctypes.c_int
        lib.ruppell_push.argtypes = [ctypes.c_uint64]
        lib.ruppell_push.restype = ctypes.c_int
        lib.ruppell_pop.restype = ctypes.c_int
        lib.ruppell_define.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ruppell_define.restype = ctypes.c_int
        lib.ruppell_capture_key.argtypes = [ctypes.c_uint64]
        lib.ruppell_capture_key.restype = ctypes.c_int
        lib.ruppell_step_fields.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        lib.ruppell_step_fields.restype = ctypes.c_int
        lib.ruppell_step_cpu.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ruppell_step_cpu.restype = ctypes.c_int
        lib.ruppell_flush.argtypes = []
        lib.ruppell_flush.restype = ctypes.c_int
        lib.ruppell_detail.argtypes = [ctypes.c_int]
        lib.ruppell_detail.restype = ctypes.c_int
        if hasattr(lib, "ruppell_counters"):  # gray-parrot's counter steps (the ROCm tool's)
            lib.ruppell_counters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
            lib.ruppell_counters.restype = ctypes.c_int
            lib.ruppell_counter_groups.argtypes = []
            lib.ruppell_counter_groups.restype = ctypes.c_int
        if hasattr(lib, "ruppell_detail_on_stream"):  # the CUDA tool's: "off" is queued on a stream
            lib.ruppell_detail_on_stream.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.ruppell_detail_on_stream.restype = ctypes.c_int
        if hasattr(lib, "ruppell_push_collective"):  # a tool built before it pairs no calls
            lib.ruppell_push_collective.argtypes = [ctypes.c_uint64]
            lib.ruppell_push_collective.restype = ctypes.c_int
        if hasattr(lib, "ruppell_call_enter"):  # collective calls as ranges (calls.bin)
            lib.ruppell_call_enter.argtypes = [ctypes.c_uint64]
            lib.ruppell_call_enter.restype = ctypes.c_int
            lib.ruppell_call_exit.argtypes = []
            lib.ruppell_call_exit.restype = ctypes.c_int
        _tool = lib
    return _tool


def marker() -> ctypes.CDLL:
    global _marker
    if _marker is None:
        path = os.environ.get("RUPPELL_MARKER_LIB")
        if not path:
            raise RuntimeError("RUPPELL_MARKER_LIB is not set: start the program with `python -m ruppell run`")
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        lib.ruppell_marker_launch.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        lib.ruppell_marker_launch.restype = ctypes.c_int
        _marker = lib
    return _marker
