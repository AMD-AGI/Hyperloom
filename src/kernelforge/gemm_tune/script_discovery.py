# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find aiter's official tuner scripts by looking, not by assuming a path.

The previous design stored one hardcoded relative path per tuner. That is the
exact shape of the failure this work started from: aiter moved the bf16 dense
tuner out of ``gradlib/`` into ``csrc/gemm_a16w16/``, the constant kept pointing
at the old location, and 14 runs died on ``unrecognized arguments`` having tuned
nothing. A constant cannot survive an upstream move, and aiter moves things.

So resolution happens in two layers:

1. **Hints** -- the known relative paths, tried in preference order. This keeps
   the common case exact and cheap, and lets a tuner prefer the real tuner script
   over a thin shim wrapping it.
2. **Patterns** -- a filename glob under ``csrc/``. When every hint misses,
   the script is *searched for*. A move to a new directory then costs nothing.

Neither layer guesses at arguments: whatever is found still goes through
``script_probe`` before being called.

The scan also yields an inventory of every ``*_tune.py`` aiter ships, including
the ones forge has not wired up yet (``batched_gemm_a8w8_tune.py``,
``batched_gemm_bf16_tune.py``, ``opus_gemm_tune.py``). Those are Tier-1 stock:
official scripts we simply have not connected, which is a very different thing
from "no official tuner exists" -- and only the latter justifies dropping to a
lower tier.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Everything aiter-location-related lives in a leaf module: ``utils`` needs the
# same tables and the same csrc lookup, and keeping either here made the two
# files import each other. The hints and patterns are re-exported so callers and
# tests can keep importing them from this module.
from . import aiter_script_map
from .aiter_script_map import TUNER_SCRIPT_HINTS, TUNER_SCRIPT_PATTERNS

log = logging.getLogger(__name__)

__all__ = [
    "TUNER_SCRIPT_HINTS",
    "TUNER_SCRIPT_PATTERNS",
    "discover_tuner_script",
    "inventory",
    "unwired_scripts",
]


def _default_csrc() -> Path | None:
    # Looked up through the module rather than bound at import time, so a test
    # that patches ``aiter_script_map.resolve_aiter_csrc`` takes effect here.
    return aiter_script_map.resolve_aiter_csrc()


# Scanning csrc/ is cheap but not free, and a tuning session resolves several
# tuners against the same tree.
_INVENTORY_CACHE: dict[str, dict[str, Path]] = {}


def _glob_first(csrc: Path, pattern: str) -> Path | None:
    """First file matching ``pattern``, deterministically ordered.

    Sorted so two hosts with the same aiter tree resolve to the same script;
    ``Path.glob`` order is filesystem-dependent otherwise.
    """
    try:
        matches = sorted(p for p in csrc.glob(pattern) if p.is_file())
    except OSError as exc:
        log.debug("glob %s under %s failed: %s", pattern, csrc, exc)
        return None
    return matches[0] if matches else None


def discover_tuner_script(tuner_name: str, csrc: Path | None = None) -> Path | None:
    """Locate the official aiter script for ``tuner_name``.

    Hints first (exact and cheap), then a filename search (survives a move).
    Returns ``None`` when aiter ships no such script -- which is the only signal
    that legitimately sends a tuner down to Tier 2.
    """
    root = csrc if csrc is not None else _default_csrc()
    if root is None:
        return None

    for rel in TUNER_SCRIPT_HINTS.get(tuner_name, ()):
        candidate = root / rel
        if candidate.is_file():
            return candidate

    for pattern in TUNER_SCRIPT_PATTERNS.get(tuner_name, ()):
        found = _glob_first(root, pattern)
        if found is not None:
            log.info(
                "%s: no hinted path matched; found %s by search (aiter layout changed?)",
                tuner_name,
                found,
            )
            return found
    return None


def inventory(csrc: Path | None = None, *, use_cache: bool = True) -> dict[str, Path]:
    """Every ``*_tune.py`` aiter ships, keyed by filename stem.

    Used to tell "aiter has no tuner for this" apart from "aiter has one and we
    have not wired it up". Only the first justifies dropping a tier.
    """
    root = csrc if csrc is not None else _default_csrc()
    if root is None:
        return {}
    key = str(root)
    if use_cache and key in _INVENTORY_CACHE:
        return dict(_INVENTORY_CACHE[key])
    try:
        found = {p.stem: p for p in sorted(root.glob("**/*_tune.py")) if p.is_file()}
    except OSError as exc:
        log.warning("aiter script inventory scan failed under %s: %s", root, exc)
        return {}
    if use_cache:
        _INVENTORY_CACHE[key] = found
    return dict(found)


def unwired_scripts(csrc: Path | None = None) -> dict[str, Path]:
    """Official scripts present on disk that no forge tuner currently drives.

    Reported rather than acted on: connecting one is a deliberate change, but a
    silent gap between "what aiter offers" and "what we call" is how the bf16
    tuner stayed pointed at a dead path.
    """
    all_scripts = inventory(csrc)
    wired = {script.stem for name in TUNER_SCRIPT_HINTS if (script := discover_tuner_script(name, csrc)) is not None}
    return {stem: path for stem, path in all_scripts.items() if stem not in wired}
