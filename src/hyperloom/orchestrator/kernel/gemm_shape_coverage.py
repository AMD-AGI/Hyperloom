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

# Emitted by aiter on every tuned-config lookup miss, naming the table consulted.
_AITER_SHAPE_MISS_RE = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+)(?:[^\n]*?)not found tuned config in (\S+?),",
)
# Emitted by aiter (only under AITER_LOG_TUNED_CONFIG) on a lookup hit.
_AITER_SHAPE_HIT_RE = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+), found padded_M: (\d+)",
)

Shape = tuple[int, int, int]

#: Columns present in an aiter MoE CSV row (its untuned CSV and the runtime
#: tuple after gfx/cu_num). Locates and validates a row's fields; a *problem's*
#: identity is the narrower :data:`_FMOE_PROBLEM_COLUMNS`.
_FMOE_DISPATCH_COLUMNS = (
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)


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


#: Smallest ladder rung. Any M below it pads up to 16 through the ``gl=0``
#: lookup, so starting here still covers M=1..16.
_LADDER_MIN_M = 16
#: aiter clamps the coarse key at 8192 for wide-N GEMMs, so no rung above it can
#: ever be reached.
_LADDER_MAX_M = 8192


def _pow2_ladder(max_m: int) -> list[int]:
    """Return the power-of-two M rungs from 16 up to ``nextPow2(max_m)``."""
    top = min(_LADDER_MAX_M, max(_LADDER_MIN_M, _next_pow2(max_m)))
    rungs = []
    rung = _LADDER_MIN_M
    while rung <= top:
        rungs.append(rung)
        rung *= 2
    return rungs


def align_shapes_to_aiter_keys(
    shapes: Iterable[Shape],
    *,
    max_shapes: int = 64,
    max_m: int = 0,
) -> tuple[list[Shape], dict[str, Any]]:
    """Re-key observed shapes onto the M values aiter will actually look up.

    Two row families are emitted per observed ``(N, K)`` projection:

    ``ladder``
        Every power-of-two M rung up to the observed maximum. Because the
        ``gl=1`` lookup key is ``nextPow2(M)`` and the ``gl=0`` key pads anything
        below 16 up to 16, a complete ladder makes *every* M resolvable — which
        matters because a profile trace only ever samples a few operating points
        and cannot see decode M at all (decode replays inside a CUDA graph and
        emits no op events).
    ``fine``
        The ``gl=0`` padded key for each observed M. These are tried before the
        ladder, so where the profile does carry evidence the tuner's answer for
        that neighbourhood wins over the coarser rung.

    Args:
        shapes: Observed ``(M, N, K)`` triples from a profile trace or server log.
        max_shapes: Upper bound on the returned shape count, bounding tuning
            time. The ladder is preserved ahead of the fine rows because it is
            what guarantees coverage; ladder rungs are then dropped smallest
            first, since small-M GEMMs contribute least to throughput.
        max_m: Optional upper bound on the M the workload can schedule (e.g.
            ``max_num_batched_tokens``). Extends the ladder past the observed
            maximum when the profile under-sampled prefill.

    Returns:
        The aligned shapes plus a report describing what changed.
    """
    observed = sorted({(int(m), int(n), int(k)) for m, n, k in shapes if min(m, n, k) > 0})
    if not observed:
        return [], {"observed": 0, "aligned": 0, "dropped": 0, "unchanged": True}

    nk_pairs = sorted({(n, k) for _m, n, k in observed})
    ladder = _pow2_ladder(max(max(m for m, _, _ in observed), int(max_m or 0)))

    ladder_rows = {(m, n, k) for n, k in nk_pairs for m in ladder}
    fine_rows = {(aiter_padded_m_fine(m), n, k) for m, n, k in observed} - ladder_rows

    budget = max(len(nk_pairs), int(max_shapes))
    kept = set(ladder_rows)
    if len(kept) > budget:
        # Drop the smallest rungs first, but never leave an (N, K) with no row.
        for m in ladder:
            if len(kept) <= budget:
                break
            for n, k in nk_pairs:
                if len(kept) <= budget:
                    break
                if len([1 for _m, _n, _k in kept if (_n, _k) == (n, k)]) > 1:
                    kept.discard((m, n, k))
    for shape in sorted(fine_rows, key=lambda s: s[0], reverse=True):
        if len(kept) >= budget:
            break
        kept.add(shape)

    result = sorted(kept)
    return result, {
        "observed": len(observed),
        "aligned": len(result),
        "dropped": max(0, len(ladder_rows | fine_rows) - len(result)),
        "unchanged": result == observed,
        "nk_pairs": len(nk_pairs),
        "ladder_m": ladder,
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
    missed = {(int(m), int(n), int(k)) for m, n, k, _table in _AITER_SHAPE_MISS_RE.findall(log_text or "")}
    hit = {(int(m), int(n), int(k)) for m, n, k, _padded in _AITER_SHAPE_HIT_RE.findall(log_text or "")}
    return missed, hit


def parse_aiter_consulted_tables(log_text: str) -> set[str]:
    """Return the tuned-config files the runtime actually looked in.

    aiter keys each quantisation variant to its own table and env var (plain
    block-scale vs ``..._BPRESHUFFLE``, for instance). When the server dispatches
    to a variant the tuner did not target, the tuned CSV is never consulted at
    all -- a different failure from a CSV that is consulted but has no matching
    row, and one worth naming separately.
    """
    return {table for _m, _n, _k, table in _AITER_SHAPE_MISS_RE.findall(log_text or "")}


def _normalize_fmoe_field(name: str, value: str) -> str:
    """Normalize one MoE dispatch field for stable CSV/log comparison."""
    text = str(value or "").strip()
    if name in {"use_g1u1", "doweight_stage1"}:
        if text in {"1", "True", "true"}:
            return "True"
        if text in {"0", "False", "false"}:
            return "False"
    return text


#: Identity of one fused-MoE problem. ``token`` is excluded because the tuner
#: sweeps it while the runtime asks for whichever batch size it is running;
#: requiring them equal reported zero coverage for a table that does serve the
#: problem. Matches ``_FMOE_SHAPE_FIELDS``, which already omitted it.
_FMOE_PROBLEM_COLUMNS = tuple(
    name for name in _FMOE_DISPATCH_COLUMNS if name != "token"
)


def fmoe_dispatch_key(fields: dict[str, str]) -> tuple[str, ...]:
    """Return the lookup key identifying one fused-MoE problem (token-agnostic)."""
    return tuple(
        _normalize_fmoe_field(name, fields.get(name, ""))
        for name in _FMOE_PROBLEM_COLUMNS
    )


def tuned_fmoe_csv_keys(path: str | Path) -> set[tuple[str, ...]]:
    """Return the fused-MoE dispatch keys present in an aiter MoE tuned CSV."""
    out: set[tuple[str, ...]] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    if not lines:
        return out
    header = [col.strip() for col in lines[0].split(",")]
    try:
        indexes = {name: header.index(name) for name in _FMOE_DISPATCH_COLUMNS}
    except ValueError:
        return out
    width = max(indexes.values()) + 1
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) < width:
            continue
        fields = {name: cols[index] for name, index in indexes.items()}
        if all(str(fields.get(name) or "").strip() for name in _FMOE_DISPATCH_COLUMNS):
            out.add(fmoe_dispatch_key(fields))
    return out


def fmoe_tuned_config_coverage(
    tuned_keys: Iterable[tuple[str, ...]],
    requested_keys: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Report how many runtime MoE dispatches a tuned fmoe CSV can serve."""
    tuned = {tuple(key) for key in tuned_keys}
    requested = [fmoe_dispatch_key(key) for key in requested_keys]
    if not requested:
        return {
            "requested": 0,
            "covered": 0,
            "coverage_pct": None,
            "tuned_rows": len(tuned),
        }
    covered = [key for key in requested if key in tuned]
    covered_set = set(covered)
    return {
        "requested": len(requested),
        "covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / len(requested), 2),
        "tuned_rows": len(tuned),
        "uncovered_sample": [
            dict(zip(_FMOE_PROBLEM_COLUMNS, key, strict=True))
            for key in requested
            if key not in covered_set
        ][:10],
    }


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
