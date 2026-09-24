# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side SGLang shape-mode gate (stdlib-only; bundled into each pod launcher)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_KERNEL_SHAPE_TOOL_REL = ("TraceLens", "TraceUtils", "kernel_shape_tool")
_SGLANG_SITECUSTOMIZE_MIN_VERSION = (0, 5, 18)


def sglang_shape_mode() -> str:
    """Return ``"sitecustomize"`` for SGLang >= 0.5.18 (no-patch shape tool), else ``"patched"``."""
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


def activate_kernel_shape_tool(env: dict[str, str]) -> None:
    """In sitecustomize mode, put the no-patch kernel_shape_tool on the child's ``PYTHONPATH``."""
    root = (env.get("TRACELENS_ROOT") or os.environ.get("TRACELENS_ROOT") or "").strip()
    if not root or sglang_shape_mode() != "sitecustomize":
        return
    tool = Path(root).joinpath(*_KERNEL_SHAPE_TOOL_REL)
    if not tool.is_dir():
        sys.stderr.write(f"WARN kernel_shape_tool not found at {tool}; SGLang shape discovery disabled\n")
        return
    existing = (env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = f"{tool}{os.pathsep}{existing}" if existing else str(tool)
    env.setdefault("TRACELENS_SHAPE_DISCOVERY", "1")
