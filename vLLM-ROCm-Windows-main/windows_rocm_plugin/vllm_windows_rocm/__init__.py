# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Out-of-tree vLLM platform plugin for native Windows + AMD ROCm (RDNA3 / gfx1100).

vLLM resolves the active platform by calling every registered `vllm.platform_plugins`
entry point; a plugin returning a class qualname wins over the built-in detectors.
On native Windows the built-in ROCm detector fails (it needs the `amdsmi` package, which
is Linux-only), and the CUDA detector fails (torch.version.cuda is None on a HIP build),
so without this plugin vLLM falls back to UnspecifiedPlatform. This plugin forces our
WindowsRocmPlatform whenever we are on win32 with a HIP-enabled torch.
"""
import sys

# Installing the single-process torch.distributed shim on import guarantees it is in place
# before vLLM touches torch.distributed (the .pth hook makes this happen at interpreter start).
try:
    from . import bootstrap  # noqa: F401  (side effect: applies the shim)
except Exception:
    pass


def register() -> str | None:
    try:
        import torch

        if sys.platform == "win32" and getattr(torch.version, "hip", None):
            return "vllm_windows_rocm.platform.WindowsRocmPlatform"
    except Exception:
        pass
    return None
