# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
# Installed into the active interpreter's site-packages (a venv or the system Python) by
# tools/patch_vllm.py. site.py auto-imports it at EVERY interpreter startup, including vLLM's
# model-inspection subprocess (`python -m vllm.model_executor.models.registry`, which does NOT
# load the platform plugin). Two independent fixes, each guarded so a non-vllm / non-tvm python
# never breaks startup. Nothing here may print: some of those interpreters write
# machine-readable output to stdout (issue #17).
import os as _os
import sys as _sys

# 1. Apply the single-process torch.distributed shim at startup, so model inspection of an
# un-cached architecture (e.g. ERNIE-4.5) does not fail with
# ModuleNotFoundError: torch._C._distributed_c10d on this USE_DISTRIBUTED=0 Windows torch.
try:
    import vllm_windows_rocm.bootstrap  # noqa: F401  (import triggers torchdist_shim.apply())
except Exception:
    pass


# 2. apache-tvm-ffi (used by xgrammar >= 0.2) cannot load a library whose path has non-ASCII
# characters ("Failed to load dynamic shared library", e.g. a venv under an accented Windows
# user name). load_module is wrapped to pass the 8.3 short form of such paths, which is pure
# ASCII; GetShortPathNameW returns an uppercase extension and tvm_ffi's loader registry is
# case-sensitive (.DLL != .dll), so the extension is lowercased again. ASCII paths go through
# untouched. The wrapper is only installed when a path tvm_ffi loads from can be non-ASCII
# (interpreter prefix, user site-packages, the home directory holding ~/.cache/tvm-ffi,
# TVM_FFI_CACHE_DIR): importing tvm_ffi costs ~70-140 ms, which every Python process on the
# machine would otherwise pay for nothing. VLLM_WIN_TVMFFI_SHORTPATH=1 / =0 forces it on / off.
def _ascii_short_path(path):
    """8.3 short form of an existing Windows path with a lowercased extension, or None when the
    path does not exist or its short form is not ASCII (8.3 names disabled on that volume)."""
    import ctypes

    s = _os.fspath(path)
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    n = get_short(s, None, 0)  # required size, terminating null included
    if not n:
        return None
    buf = ctypes.create_unicode_buffer(n)
    if not 0 < get_short(s, buf, n) < n:
        return None
    short = buf.value
    if not short.isascii():
        return None
    root, ext = _os.path.splitext(short)
    return root + ext.lower()


def _tvm_ffi_paths_may_be_non_ascii() -> bool:
    forced = _os.environ.get("VLLM_WIN_TVMFFI_SHORTPATH", "")
    if forced in ("0", "1"):
        return forced == "1"
    import site

    candidates = (
        _sys.prefix,
        getattr(site, "USER_SITE", None) or "",
        _os.path.expanduser("~"),
        _os.environ.get("TVM_FFI_CACHE_DIR", ""),
    )
    return not all(c.isascii() for c in candidates)


def _install_tvm_ffi_short_paths() -> None:
    import functools

    import tvm_ffi
    from tvm_ffi import module as tvm_module

    orig = tvm_module.load_module
    if getattr(orig, "_vllm_win_short_path", False):
        return

    @functools.wraps(orig)
    def load_module(path, *args, **kwargs):
        try:
            if not _os.fspath(path).isascii():
                path = _ascii_short_path(path) or path
        except Exception:
            pass
        return orig(path, *args, **kwargs)

    load_module._vllm_win_short_path = True
    # Both bindings: tvm_ffi.libinfo (xgrammar's path) imports load_module from the module at
    # call time, while tvm_ffi.cpp's JIT loader calls the package-level re-export.
    tvm_module.load_module = load_module
    tvm_ffi.load_module = load_module


if _sys.platform == "win32":
    try:
        if _tvm_ffi_paths_may_be_non_ascii():
            _install_tvm_ffi_short_paths()
    except Exception:
        pass
