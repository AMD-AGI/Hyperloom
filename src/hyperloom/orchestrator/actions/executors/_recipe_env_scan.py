# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Static detection of environment names an InferenceX recipe script re-exports unconditionally."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ._magpie_patcher import _resolve_inferencex_benchmarks_dir

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}``/``${NAME-...}`` guard on the
# same line overwrites whatever the caller set, so a recipe author's own
# unconditional export always wins over an --extra-env of the same name.
_UNGUARDED_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)


def recipe_overwritten_env_names(inferencex_path: str, benchmark_script: str) -> frozenset[str]:
    """Env names ``benchmark_script`` re-exports unconditionally (empty when unreadable)."""
    script_name = str(benchmark_script or "").strip()
    if not script_name:
        return frozenset()
    benchmarks_dir = _resolve_inferencex_benchmarks_dir(inferencex_path or None)
    if benchmarks_dir is None:
        return frozenset()
    try:
        match = next(benchmarks_dir.rglob(script_name))
    except StopIteration:
        return frozenset()
    try:
        text = match.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    return frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


__all__ = ["recipe_overwritten_env_names"]
