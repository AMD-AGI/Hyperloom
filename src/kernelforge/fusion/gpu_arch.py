# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which chip this run targets."""

from __future__ import annotations

import re
import subprocess

# Canonical arch tokens are lowercase ``gfx*``; marketing names are folded in so a caller reporting ``MI355X`` and a
# probe reporting ``gfx950`` agree.
_ARCH_ALIASES = {
    "gfx942": "gfx942",
    "gfx950": "gfx950",
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
    "radeon8065s": "gfx1151",
}
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """Return the canonical lowercase ``gfx*`` arch, or ``""`` when unresolvable."""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw in _ARCH_ALIASES:
        return _ARCH_ALIASES[raw]
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0).lower()
    for alias, canonical in _ARCH_ALIASES.items():
        if alias in raw:
            return canonical
    return ""


def detect_arch(timeout_s: float = 15.0) -> str:
    """Best-effort local arch via ``rocminfo``; ``""`` when undetectable."""
    try:
        completed = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    match = _GFX_RE.search(completed.stdout or "")
    return match.group(0).lower() if match else ""


__all__ = ["canon_arch", "detect_arch"]
