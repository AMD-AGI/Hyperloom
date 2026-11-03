"""The CPU op tree (RUPPELL_CPU_OPS; off by default): aten ops recorded on host
threads through PyTorch's RecordFunction callbacks (runtime/src/cpu_ops.cpp),
timestamped by the platform tool, which also times ruppell's scopes (the
module tree). `ruppell.init()` starts it when the variable is set:

    RUPPELL_CPU_OPS=1       ops with their input dims and types
    RUPPELL_CPU_OPS=names   ops by name only (no input capture, cheaper)

Each op's autograd sequence number is recorded too (RUPPELL_CPU_OPS_SEQ=0 turns
it off): a forward op with the scope it ran in, a backward function (autograd's
engine thread, where no module hook runs) with its number, so the analysis can
attribute backward kernels to the forward scope they differentiate.

The library is built for one torch (runtime/build_cpu_ops.sh writes the
version beside it); loading it into another is refused.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

from ruppell import _lib

_ops: Optional[ctypes.CDLL] = None


def mode() -> Optional[str]:
    v = os.environ.get("RUPPELL_CPU_OPS", "")
    if v in ("", "0"):
        return None
    return "names" if v == "names" else "inputs"


def start() -> bool:
    """Start recording ops in this process; False when not asked for."""
    global _ops
    m = mode()
    if m is None or _ops is not None:
        return _ops is not None
    tool = _lib.tool()
    if tool is None or tool.ruppell_cpu_ops_enabled() != 1:
        raise RuntimeError("RUPPELL_CPU_OPS is set but the tool is not recording the CPU op tree in this process")
    import torch  # the library links libtorch: the traced process's must be loaded first

    path = os.environ.get("RUPPELL_CPU_OPS_LIB") or os.path.join(
        os.path.dirname(os.environ["RUPPELL_TOOL_LIB"]), "libruppell_cpu_ops.so")
    stamp = os.path.join(os.path.dirname(path), "CPU_OPS_TORCH")
    built_for = open(stamp).read().strip() if os.path.exists(stamp) else None
    if built_for != torch.__version__:
        raise RuntimeError(f"{path} was built for torch {built_for}, this process runs {torch.__version__}; "
                           "rebuild it with runtime/build_cpu_ops.sh using this python")
    lib = ctypes.CDLL(path)
    lib.ruppell_cpu_ops_start.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    lib.ruppell_cpu_ops_start.restype = ctypes.c_int
    lib.ruppell_cpu_ops_count.restype = ctypes.c_uint64
    lib.ruppell_cpu_ops_pause.argtypes = [ctypes.c_int]
    lib.ruppell_cpu_ops_pause.restype = ctypes.c_int

    def addr(name: str) -> int:
        return ctypes.cast(getattr(tool, name), ctypes.c_void_p).value

    if os.environ.get("RUPPELL_CPU_OPS_SEQ", "1") != "0" and hasattr(tool, "ruppell_op_seq")             and hasattr(lib, "ruppell_cpu_ops_seq"):
        lib.ruppell_cpu_ops_seq.argtypes = [ctypes.c_void_p]
        lib.ruppell_cpu_ops_seq(addr("ruppell_op_seq"))
    rc = lib.ruppell_cpu_ops_start(addr("ruppell_op_enter"), addr("ruppell_op_exit"), addr("ruppell_op_define"),
                                   1 if m == "inputs" else 0)
    if rc != 0:
        raise RuntimeError(f"ruppell_cpu_ops_start failed ({rc})")
    _ops = lib
    return True


def pause(paused: bool) -> None:
    """Pause or resume recording ops (detail sampling: ruppell.detail calls it,
    so ops are recorded in the steps whose kernels are, and at startup)."""
    if _ops is not None:
        _ops.ruppell_cpu_ops_pause(1 if paused else 0)


def count() -> int:
    """Ops recorded so far in this process."""
    return int(_ops.ruppell_cpu_ops_count()) if _ops is not None else 0
