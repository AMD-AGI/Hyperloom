# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find aiter's official tuner scripts by looking, not by assuming a path."""

from __future__ import annotations

import logging
from pathlib import Path

# Everything aiter-location-related lives in a leaf module: ``utils`` needs the same tables and the same csrc lookup,
# and keeping either here made the two files import each other.
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
    # Looked up through the module rather than bound at import time, so a test that patches
    # ``aiter_script_map.resolve_aiter_csrc`` takes effect here.
    return aiter_script_map.resolve_aiter_csrc()


# Scanning csrc/ is cheap but not free, and a tuning session resolves several tuners against the same tree.
_INVENTORY_CACHE: dict[str, dict[str, Path]] = {}


def _glob_first(csrc: Path, pattern: str) -> Path | None:
    """First file matching ``pattern``, deterministically ordered."""
    try:
        matches = sorted(p for p in csrc.glob(pattern) if p.is_file())
    except OSError as exc:
        log.debug("glob %s under %s failed: %s", pattern, csrc, exc)
        return None
    return matches[0] if matches else None


def discover_tuner_script(tuner_name: str, csrc: Path | None = None) -> Path | None:
    """Locate the official aiter script for ``tuner_name``."""
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
    """Every ``*_tune.py`` aiter ships, keyed by filename stem."""
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
    """Official scripts present on disk that no forge tuner currently drives."""
    all_scripts = inventory(csrc)
    wired = {script.stem for name in TUNER_SCRIPT_HINTS if (script := discover_tuner_script(name, csrc)) is not None}
    return {stem: path for stem, path in all_scripts.items() if stem not in wired}
