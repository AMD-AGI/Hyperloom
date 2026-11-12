# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Importing this module installs the single-process torch.distributed shim.

A .pth startup hook (`import vllm_windows_rocm.bootstrap`) makes this run before any
`import vllm`, so vLLM sees a working (single-process) torch.distributed from the start.

It also pre-sets environment variables that upstream components resolve at import time
and otherwise fail on silently:

- ROCM_HOME: triton's AMD driver resolves the SDK root by shelling out to
  `rocm-sdk path --root` at import; when that CLI is not on PATH (the usual case with a
  pip-installed SDK), the HIP include dir is never added and the first Triton JIT kernel
  dies with `fatal error: 'hip/hip_runtime.h' file not found`. Deriving it from the
  `rocm_sdk` Python package (no PATH needed) and exporting it here makes that path work.
- TMP/TEMP: tvm-ffi and the triton JIT compile through paths that break on non-ASCII
  characters; if the user profile lives under a non-ASCII path (e.g. C:\\Users\\PC-Clément),
  the temp dirs default there. Pointing TMP/TEMP at an ASCII location (LOCALAPPDATA when
  it is ASCII, else <system drive>\\vllm_win_tmp) fixes every temp-path failure seen so
  far. A warning is printed when no ASCII option is found.
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# The plugin logs what it patches (native ops loaded, KV-scale override, CK prefill, the
# 2D-decode switch) at INFO on its own logger, which nobody configures: without a handler
# those lines are dropped and a user cannot tell whether a hook took. Attach one on stderr
# (stdout must stay machine-readable for vLLM subprocesses, issue #17) and stop propagation
# so vLLM root handlers cannot print them twice. This module runs in every interpreter on
# the machine (sitecustomize), so the level stays at WARNING here; the platform raises it
# to INFO once a vLLM engine is being configured. VLLM_WIN_LOG_LEVEL overrides both.
_pkg_logger = logging.getLogger("vllm_windows_rocm")
if not _pkg_logger.handlers:
    _h = logging.StreamHandler()  # stderr
    _h.setFormatter(logging.Formatter("vllm-win %(levelname)s %(name)s: %(message)s"))
    _pkg_logger.addHandler(_h)
    _pkg_logger.setLevel(os.environ.get("VLLM_WIN_LOG_LEVEL", "WARNING").upper())
    _pkg_logger.propagate = False

from .torchdist_shim import apply

# Keep the validated torch/Triton fallbacks as the default for Windows ROCm entrypoints.
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
# SDPA on ROCm: PyTorch only lists gfx90a/gfx942/gfx1100/gfx1201 as supported for its aotriton
# flash kernels; every other RDNA3/3.5 part (gfx1101, gfx1102, gfx1150, gfx1151) is gated behind
# this flag and silently falls back to the math kernel, which materializes the whole score
# matrix (+16.5 GiB per layer at 8k on a 28-head model). The torch-rocm Windows wheels ship
# the gfx11xx aotriton image, so enabling it is safe; an explicit user value is kept.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

apply()


def _is_ascii(path: str) -> bool:
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _ensure_rocm_home() -> None:
    if os.environ.get("ROCM_HOME") or os.environ.get("HIP_PATH"):
        return  # user pinned it; never override an explicit choice
    try:
        import rocm_sdk
    except ImportError:
        return
    try:
        libs = rocm_sdk.find_libraries("amdhip64")
    except Exception:  # noqa: BLE001
        return
    if not libs:
        return
    d = os.path.dirname(str(libs[0]))
    # .../_rocm_sdk_devel/bin/amdhip64_7.dll -> the _rocm_sdk_* package dir is the root
    while d and not os.path.basename(d).startswith("_rocm_sdk"):
        nd = os.path.dirname(d)
        if nd == d:
            return
        d = nd
    if d and os.path.isfile(os.path.join(d, "include", "hip", "hip_runtime.h")):
        os.environ["ROCM_HOME"] = d


def _ensure_ascii_tmp() -> None:
    cur = os.environ.get("TMP") or os.environ.get("TEMP") or tempfile.gettempdir()
    if _is_ascii(cur):
        return
    sysdrive = os.environ.get("SystemDrive", "C:")
    cands = [os.environ.get("LOCALAPPDATA", ""), os.path.join(sysdrive, "\\")]
    for base in cands:
        if not base or not _is_ascii(base):
            continue
        target = os.path.join(base, "vllm_win_tmp")
        try:
            os.makedirs(target, exist_ok=True)
        except OSError:
            continue
        for var in ("TMP", "TEMP", "TMPDIR"):
            os.environ[var] = target
        logger.warning("TMP/TEMP was under a non-ASCII path (%s); redirected to %s "
                       "(non-ASCII paths break native DLL/JIT compilation)", cur, target)
        return
    logger.warning("TMP/TEMP is under a non-ASCII path and no ASCII fallback was found; "
                   "triton JIT compilation may fail. Set TMP/TEMP to an ASCII directory.")


_ensure_rocm_home()
_ensure_ascii_tmp()
