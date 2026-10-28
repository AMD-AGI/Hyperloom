# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens's bundled per-platform arch specs, read as roofline specs for boards ``HW_SPECS`` has no row for."""

from __future__ import annotations

import json
import os
from collections.abc import Container, Mapping
from pathlib import Path
from typing import Any

#: Where a TraceLens checkout keeps its per-platform arch specs.
ARCH_SPEC_DIR = Path("TraceLens/Agent/Analysis/utils/arch")

#: The ``source`` TraceLens's microbenchmark writes into a spec it measured; ceilings use published figures only.
_MEASURED_SOURCE = "benchmark derived"


def bundled_arch_specs(
    gpu_type: str | None, precision_tags: Mapping[str, tuple[str, ...]], *, published: Container[str]
) -> dict[str, dict[str, Any]]:
    """``{gpu_type: spec}`` from the TraceLens checkout at ``$TRACELENS_ROOT``, shaped as a ``HW_SPECS`` row.

    Empty when *gpu_type* has a *published* row, TraceLens bundles no usable spec for it, or the spec was measured.
    *precision_tags* maps each TraceLens ``matrix_<tag>`` peak to the precision tags it is read under.
    """
    key = (gpu_type or "").strip().lower()
    root = os.environ.get("TRACELENS_ROOT", "").strip()
    if not key or not root or key in published:
        return {}
    payload = _read_spec(Path(root) / ARCH_SPEC_DIR, key)
    bandwidth = _positive(payload.get("mem_bw_gbps"))
    peaks = payload.get("max_achievable_tflops")
    if not bandwidth or not isinstance(peaks, dict) or _MEASURED_SOURCE in str(payload.get("source", "")).lower():
        return {}
    table = {
        alias: tflops
        for tag, aliases in precision_tags.items()
        if (tflops := _positive(peaks.get(f"matrix_{tag}")))
        for alias in aliases
    }
    return {key: {"hbm_bw_gbps": bandwidth, "hbm_gb": _positive(payload.get("memory_gb")), "peak_tflops": table}}


def _read_spec(arch_dir: Path, gpu_type: str) -> dict[str, Any]:
    """The spec whose file name matches *gpu_type* ignoring case and separators (``Radeon_8060S`` for ``radeon8060s``)."""
    for path in sorted(arch_dir.glob("*.json")):
        if _platform_key(path.stem) != _platform_key(gpu_type):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _platform_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _positive(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0
