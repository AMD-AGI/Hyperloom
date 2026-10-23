# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which chip this run targets."""

from __future__ import annotations

import re
import subprocess

from hyperloom.common.gpu_identity import AMD_GPU_BOARDS, board_by_identifier

_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """Return the canonical lowercase ``gfx*`` arch, or ``""`` when unresolvable.

    Board names are folded in, so a caller reporting ``MI355X`` and a probe reporting ``gfx950`` agree.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    board = board_by_identifier(raw)
    if board:
        return board.gfx
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0).lower()
    return next((b.gfx for b in AMD_GPU_BOARDS.values() if any(name in raw for name in b.identifiers)), "")


def arch_phrase(gpu_arch: str) -> str:
    """How to name the target GPU in a prompt."""
    arch = (gpu_arch or "").strip().lower()
    marketing = "/".join(board.product.upper() for board in AMD_GPU_BOARDS.values() if board.gfx == arch)
    if not arch:
        return "an AMD ROCm GPU"
    return f"AMD {marketing} ({arch})" if marketing else f"an AMD GPU ({arch})"


def detect_arch(timeout_s: float = 15.0) -> str:
    """Best-effort local arch via ``rocminfo``; ``""`` when undetectable."""
    try:
        completed = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = _GFX_RE.search(completed.stdout or "")
    return match.group(0).lower() if match else ""


__all__ = ["arch_phrase", "canon_arch", "detect_arch"]
