# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Align tuned-GEMM shapes with the M keys aiter actually looks up.

aiter resolves a tuned config by trying three lookup keys in order (see
``aiter/ops/gemm_op_a8w8.py::get_CKGEMM_config`` and
``csrc/py_itfs_cu/gemm_common.cu::getPaddedM``):

1. the raw ``M``,
2. ``gl=0`` fine-grained padding — round M up to 16 (M<=256), 32 (M<=1024),
   64 (M<=4096) or 128,
3. ``gl=1`` coarse padding — ``nextPow2(M)``, clamped to 8192 when
   ``M > 8192 and N > 4096``.

Runtime M is the number of tokens in a scheduled batch, so prefill M is
data-dependent and effectively never repeats between two runs. Handing the
tuner the raw M values sampled from one run therefore produces a CSV whose keys
no runtime lookup can reach: a later run asking for M=1082 pads to 1088 and
misses a row keyed on M=1076, so aiter falls back to its default config and the
measured micro speedup contributes nothing end to end.

Keying the CSV on the padded values instead makes each tuned row cover the whole
bucket that pads onto it, which is what turns a micro-level win into an
end-to-end one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

# Emitted by aiter on every tuned-config lookup miss.
_AITER_SHAPE_MISS_RE = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+)(?:[^\n]*?)not found tuned config",
)
# Emitted by aiter (only under AITER_LOG_TUNED_CONFIG) on a lookup hit.
_AITER_SHAPE_HIT_RE = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+), found padded_M: (\d+)",
)

Shape = tuple[int, int, int]


def aiter_padded_m_fine(m: int) -> int:
    """Return aiter's ``gl=0`` padded M (fine-grained lookup key)."""
    if m <= 256:
        return (m + 15) // 16 * 16
    if m <= 1024:
        return (m + 31) // 32 * 32
    if m <= 4096:
        return (m + 63) // 64 * 64
    return (m + 127) // 128 * 128


def _next_pow2(m: int) -> int:
    if m <= 1:
        return 1
    return 1 << (m - 1).bit_length()


def aiter_padded_m_coarse(m: int, n: int) -> int:
    """Return aiter's ``gl=1`` padded M (coarse power-of-two lookup key)."""
    if m > 8192 and n > 4096:
        return 8192
    return _next_pow2(m)


def aiter_lookup_keys(shape: Shape) -> tuple[Shape, Shape, Shape]:
    """Return the three (M, N, K) keys aiter tries, in lookup order."""
    m, n, k = shape
    return (
        (m, n, k),
        (aiter_padded_m_fine(m), n, k),
        (aiter_padded_m_coarse(m, n), n, k),
    )


def align_shapes_to_aiter_keys(
    shapes: Iterable[Shape],
    *,
    max_shapes: int = 64,
) -> tuple[list[Shape], dict[str, Any]]:
    """Replace observed M values with the M keys aiter will look up.

    For every observed shape both padded keys are emitted: the fine-grained one
    covers the immediate neighbourhood of the observed operating point, and the
    power-of-two one acts as a wide net for M values that land in a different
    fine bucket on a later run.

    Args:
        shapes: Observed ``(M, N, K)`` triples, typically harvested from a
            profile trace or a server log.
        max_shapes: Upper bound on the returned shape count, so a wide observed
            M distribution cannot blow up the tuning budget. Larger M is kept
            first because those GEMMs dominate runtime.

    Returns:
        The aligned shapes plus a report describing what changed.
    """
    observed = sorted({(int(m), int(n), int(k)) for m, n, k in shapes if min(m, n, k) > 0})
    if not observed:
        return [], {"observed": 0, "aligned": 0, "dropped": 0, "unchanged": True}

    aligned: set[Shape] = set()
    for m, n, k in observed:
        aligned.add((aiter_padded_m_fine(m), n, k))
        aligned.add((aiter_padded_m_coarse(m, n), n, k))

    # Keep one row per (N, K) before trimming so no projection loses coverage.
    by_nk: dict[tuple[int, int], list[int]] = {}
    for m, n, k in aligned:
        by_nk.setdefault((n, k), []).append(m)
    kept: set[Shape] = set()
    for (n, k), ms in by_nk.items():
        kept.add((max(ms), n, k))
    remaining = sorted(aligned - kept, key=lambda s: s[0], reverse=True)
    budget = max(len(kept), max_shapes)
    for shape in remaining:
        if len(kept) >= budget:
            break
        kept.add(shape)

    result = sorted(kept)
    return result, {
        "observed": len(observed),
        "aligned": len(result),
        "dropped": max(0, len(aligned) - len(result)),
        "unchanged": result == observed,
        "observed_m": sorted({m for m, _, _ in observed})[:32],
        "aligned_m": sorted({m for m, _, _ in result})[:32],
    }


def load_shapes_json(path: str | Path) -> list[Shape]:
    """Read a forge shapes JSON file into ``(M, N, K)`` triples."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("shapes") or []
    if not isinstance(data, list):
        return []
    out: list[Shape] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        keys = {str(key).upper(): value for key, value in row.items()}
        try:
            shape = (int(keys["M"]), int(keys["N"]), int(keys["K"]))
        except (KeyError, TypeError, ValueError):
            continue
        if min(shape) > 0:
            out.append(shape)
    return out


def write_shapes_json(shapes: Iterable[Shape], destination: Path) -> str:
    """Write ``shapes`` as a forge-compatible shapes JSON, returning its path."""
    payload = [{"M": m, "N": n, "K": k} for m, n, k in sorted(shapes)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(destination)


def parse_aiter_shape_lookups(log_text: str) -> tuple[set[Shape], set[Shape]]:
    """Return the ``(missed, hit)`` GEMM shapes aiter reported in a server log."""
    missed = {(int(m), int(n), int(k)) for m, n, k in _AITER_SHAPE_MISS_RE.findall(log_text or "")}
    hit = {(int(m), int(n), int(k)) for m, n, k, _padded in _AITER_SHAPE_HIT_RE.findall(log_text or "")}
    return missed, hit


def tuned_csv_shapes(path: str | Path) -> set[Shape]:
    """Return the ``(M, N, K)`` keys present in an aiter tuned-GEMM CSV."""
    out: set[Shape] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    if not lines:
        return out
    header = [col.strip().upper() for col in lines[0].split(",")]
    try:
        mi, ni, ki = header.index("M"), header.index("N"), header.index("K")
    except ValueError:
        return out
    width = max(mi, ni, ki) + 1
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) < width:
            continue
        try:
            shape = (int(cols[mi]), int(cols[ni]), int(cols[ki]))
        except ValueError:
            continue
        if min(shape) > 0:
            out.add(shape)
    return out


def tuned_config_coverage(
    tuned_shapes: Iterable[Shape],
    requested_shapes: Iterable[Shape],
) -> dict[str, Any]:
    """Report how many requested shapes a tuned CSV can actually serve.

    Replays aiter's three-step lookup against the CSV keys, so the result
    answers "will this artifact ever be used?" rather than "did the micro
    benchmark look good?".
    """
    tuned = {(int(m), int(n), int(k)) for m, n, k in tuned_shapes}
    requested = sorted({(int(m), int(n), int(k)) for m, n, k in requested_shapes})
    if not requested:
        return {
            "requested": 0,
            "covered": 0,
            "coverage_pct": None,
            "tuned_rows": len(tuned),
        }
    covered = [shape for shape in requested if any(key in tuned for key in aiter_lookup_keys(shape))]
    return {
        "requested": len(requested),
        "covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / len(requested), 2),
        "tuned_rows": len(tuned),
        "uncovered_sample": [{"M": m, "N": n, "K": k} for m, n, k in requested if (m, n, k) not in set(covered)][:10],
    }
