# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which AMD GPU a run targets: the one mapping between product names and gfx targets, and the local probe."""

from __future__ import annotations

import re
import subprocess

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS, board_named_in


def _spelling(value: str) -> str:
    """``value`` lowercase with spaces and dashes as underscores, the one spelling names are compared in."""
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


# Product names fold onto their gfx target so a caller reporting ``MI355X`` and a probe reporting ``gfx950`` agree.
_ARCH_BY_GPU_TYPE = {
    _spelling(name): board.gfx for board in AMD_GPU_BOARDS.values() for name in (board.product, *board.aliases)
}
# The product name each gfx target is recorded under, so one chip has one name in KB identities.
_GPU_TYPE_BY_ARCH = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    "gfx1151": "radeon8060s",
    "gfx1201": "r9700",
}
# A concrete target only: rocminfo also lists generic ISAs such as ``gfx12-generic`` beside the chip's own.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b(?!-)", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """The canonical lowercase gfx target ``value`` names, product name or gfx alike; ``""`` when it names none.

    Lenient: a spelling that contains a gfx target, or names a board as a whole word the way
    :func:`hyperloom_common.gpu_identity.board_named_in` reads probe output, folds onto it (``mi300x_vf`` is
    ``gfx942``; ``R9700S`` and ``MI300XL`` are other products and name none). Use it to pick a target, not to key
    stored records; :func:`canonical_gpu_type` does that.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0).lower()
    board = board_named_in(re.sub(r"[_-]+", " ", raw))
    return board.gfx if board else ""


def canonical_gpu_type(value: str) -> str:
    """The product name a known chip is recorded under; any other value keeps its own normalized spelling.

    Exact, because the result keys KB records: only a known product name or gfx target maps, so a spelling that
    merely contains one (``mi300x_vf``, ``gfx942:sramecc+``) keeps the records it was stored under.
    """
    normalized = _spelling(value)
    arch = _ARCH_BY_GPU_TYPE.get(normalized, normalized)
    return _GPU_TYPE_BY_ARCH.get(arch, normalized)


def product_arch(value: str) -> str:
    """The gfx target of ``value`` when it is spelled exactly as a known product name; ``""`` otherwise."""
    return _ARCH_BY_GPU_TYPE.get(str(value or "").strip().lower(), "")


def detect_arch(timeout_s: float = 15.0) -> str:
    """The local gfx target rocminfo reports; ``""`` when it cannot run, fails, or reports none or several."""
    try:
        completed = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    targets = {match.lower() for match in _GFX_RE.findall(completed.stdout or "")}
    return targets.pop() if len(targets) == 1 else ""


__all__ = ["canon_arch", "canonical_gpu_type", "detect_arch", "product_arch"]
