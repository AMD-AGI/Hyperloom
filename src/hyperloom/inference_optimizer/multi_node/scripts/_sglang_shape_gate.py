# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side SGLang >= 0.5.18 no-patch kernel-shape-tool gate.

stdlib-only; shipped beside pod launcher scripts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_KERNEL_SHAPE_TOOL_REL: tuple[str, ...] = ("TraceLens", "TraceUtils", "kernel_shape_tool")
_SGLANG_SITECUSTOMIZE_MIN_VERSION: tuple[int, ...] = (0, 5, 18)


def _sglang_shape_mode() -> str:
    """Return the active SGLang kernel-shape integration mode."""
    override = os.environ.get("HYPERLOOM_SGLANG_SHAPE_MODE", "auto").strip().lower()
    if override in {"patch", "patched"}:
        return "patched"
    if override == "sitecustomize":
        return "sitecustomize"
    version = ""
    try:
        import sglang  # type: ignore

        version = (getattr(sglang, "__version__", "") or "").strip()
    except Exception:  # noqa: BLE001
        version = os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "").strip()
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", version)
    if not m:
        return "patched"
    vt = tuple(int(p) for p in m.group(1).split("."))
    return "sitecustomize" if vt >= _SGLANG_SITECUSTOMIZE_MIN_VERSION else "patched"


def _maybe_activate_kernel_shape_tool(env: dict[str, str]) -> None:
    """Add the no-patch kernel_shape_tool to PYTHONPATH for SGLang >= 0.5.18."""
    root = (env.get("TRACELENS_ROOT") or os.environ.get("TRACELENS_ROOT") or "").strip()
    if not root or _sglang_shape_mode() != "sitecustomize":
        return
    tool = Path(root).joinpath(*_KERNEL_SHAPE_TOOL_REL)
    if not tool.is_dir():
        import sys

        sys.stderr.write(f"WARN kernel_shape_tool not found at {tool}; SGLang shape discovery disabled\n")
        return
    existing = (env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = f"{tool}{os.pathsep}{existing}" if existing else str(tool)
    env.setdefault("TRACELENS_SHAPE_DISCOVERY", "1")
