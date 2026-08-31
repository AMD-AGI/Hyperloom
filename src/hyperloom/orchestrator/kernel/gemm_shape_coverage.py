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
#
# ``K:`` and ``found padded_M:`` are *not* adjacent in practice: aiter prints the
# dispatch kwargs between them, and only some of those are comma-separated --
#
#   shape is M:16384, N:4608, K:8192 dtype='torch.bfloat16' otype='torch.bfloat16'
#   bias=False, scaleAB=False, bpreshuffle=False found padded_M: 8192, N:4608, ...
#
# An earlier pattern required a comma straight after ``K:<n>`` and so matched
# none of the 5024 hit lines in a production Qwen3 log, which scored a fully
# served tuned table as ``no_shape_key_matched`` and reverted it. The trailing
# ``is tuned`` anchor keeps the lazy ``[^\n]*?`` from spanning an unrelated line.
_AITER_SHAPE_HIT_RE = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+)[^\n]*?found padded_M: (\d+), N:\d+, K:\d+ is tuned",
)

Shape = tuple[int, int, int]
FmoeDispatchKey = tuple[str, ...]

# Short aliases and canonical torch dtype strings seen in fused-MoE logs/CSVs.
# Only listed forms are normalized; anything else is preserved for exact matching.
_FMoe_Q_DTYPE_ALIASES: dict[str, str] = {
    "fp4": "torch.float4_e2m1fn_x2",
    "torch.float4_e2m1fn_x2": "torch.float4_e2m1fn_x2",
    "torch.float8_e4m3fn": "torch.float8_e4m3fn",
    "torch.float8_e4m3fnuz": "torch.float8_e4m3fnuz",
    "torch.float8_e5m2": "torch.float8_e5m2",
}

# ``get_2stage_cfgs`` indexes tuned rows on all fourteen columns below (see
# ``aiter/fused_moe.py::_INDEX_COLS``). Runtime logs the same tuple after gfx
# was added; the descriptor between ``using 2stage`` and ``for`` is either
# ``default`` or ``(kernelName1='…', kernelName2='…')``.
FMOE_INDEX_COLS = (
    "gfx",
    "cu_num",
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

_KERNEL_DESCRIPTOR_RE = re.compile(r"kernelName1='(?P<kn1>[^']*)'.*?kernelName2='(?P<kn2>[^']*)'")


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


def _split_fmoe_tuple(raw: str) -> list[str]:
    """Split a fused-MoE dispatch tuple, respecting single-quoted fields."""
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in raw:
        if ch == "'":
            in_quote = not in_quote
            continue
        if ch == "," and not in_quote:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_fmoe_q_dtype(value: str) -> str:
    """Map known q-dtype aliases to the canonical string aiter logs."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return text
    if text in _FMoe_Q_DTYPE_ALIASES:
        return _FMoe_Q_DTYPE_ALIASES[text]
    lowered = text.lower()
    if lowered in _FMoe_Q_DTYPE_ALIASES:
        return _FMoe_Q_DTYPE_ALIASES[lowered]
    return text


def _normalize_fmoe_field(name: str, value: str) -> str:
    text = str(value or "").strip().strip("'\"")
    if name == "gfx":
        return text
    if name == "act_type" and "." in text:
        # Runtime logs ``ActivationType.Swiglu``; CSVs may store the suffix.
        text = text.rsplit(".", 1)[-1]
    if name == "dtype":
        lowered = text.lower()
        if lowered.startswith("torch.bfloat"):
            return "bf16"
        if lowered.startswith("torch.float16"):
            return "fp16"
    if name in ("q_dtype_a", "q_dtype_w"):
        return _normalize_fmoe_q_dtype(text)
    if name in ("use_g1u1", "doweight_stage1"):
        lowered = text.lower()
        if lowered in ("true", "1"):
            return "1"
        if lowered in ("false", "0"):
            return "0"
    return text


def _extract_paren_group(text: str, start: int) -> str | None:
    """Return the parenthesised group starting at ``start``, or ``None``."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx]
    return None


def _parse_fused_moe_dispatch_line(line: str) -> dict[str, str] | None:
    """Parse one ``[fused_moe] using … for (…)`` line."""
    marker = "[fused_moe] using "
    pos = line.find(marker)
    if pos < 0:
        return None
    rest = line[pos + len(marker) :]
    for_sep = " for ("
    for_pos = rest.find(for_sep)
    if for_pos < 0:
        return None

    stage_desc = rest[:for_pos].strip()
    keys_raw = _extract_paren_group(rest, for_pos + len(" for "))
    if keys_raw is None:
        return None

    descriptor = "default"
    kernel_name1 = ""
    kernel_name2 = ""
    if not stage_desc.endswith("default"):
        desc_open = stage_desc.rfind("(")
        if desc_open < 0:
            return None
        descriptor = stage_desc[desc_open:].strip()
        kn_match = _KERNEL_DESCRIPTOR_RE.search(descriptor)
        if kn_match is None:
            return None
        kernel_name1 = kn_match.group("kn1")
        kernel_name2 = kn_match.group("kn2")

    record = _fmoe_dispatch_record(_split_fmoe_tuple(keys_raw))
    if record is None:
        return None
    record["descriptor"] = descriptor
    record["kernelName1"] = kernel_name1
    record["kernelName2"] = kernel_name2
    return record


def _fmoe_dispatch_record(parts: list[str]) -> dict[str, str] | None:
    """Map a fused-MoE dispatch tuple to the fourteen-column lookup schema."""
    if len(parts) < len(FMOE_INDEX_COLS):
        return None
    record: dict[str, str] = {}
    for idx, name in enumerate(FMOE_INDEX_COLS):
        record[name] = _normalize_fmoe_field(name, parts[idx])
    if not all(record.get(field) for field in FMOE_INDEX_COLS[:7]):
        return None
    return record


def fmoe_dispatch_lookup_key(record: dict[str, str]) -> FmoeDispatchKey:
    """Return the fourteen-column lookup key ``get_2stage_cfgs`` uses."""
    return tuple(record[field] for field in FMOE_INDEX_COLS)


def parse_aiter_fused_moe_dispatches(log_text: str) -> list[dict[str, str]]:
    """Return every fused-MoE dispatch the server logged."""
    seen: set[tuple[str, str, FmoeDispatchKey]] = set()
    out: list[dict[str, str]] = []
    for line in (log_text or "").splitlines():
        record = _parse_fused_moe_dispatch_line(line)
        if record is None:
            continue
        dedupe = (
            record.get("descriptor") or "",
            record.get("kernelName1") or "",
            fmoe_dispatch_lookup_key(record),
        )
        if dedupe in seen:
            continue
        seen.add(dedupe)
        out.append(record)
    return out


def resolve_fmoe_candidate_csv(path: str | Path) -> Path | None:
    """Resolve the bare candidate CSV used for runtime attribution.

    E2E envs often point at ``merged_<name>.csv`` (for example
    ``merged_candidate_fmoe.csv`` or ``merged_tuned_fmoe.csv``). Attribution
    must use the sibling bare file when it exists; the merged superset must
    not impersonate a candidate row the tuner never produced.
    """
    resolved = Path(path)
    if not resolved.is_file():
        return None
    if resolved.name.startswith("merged_"):
        bare = resolved.parent / resolved.name[len("merged_") :]
        return bare if bare.is_file() else None
    return resolved


FMOE_INTEGRATE_RUN = "integrate-gemm_tune_fmoe_ck"


def log_has_fused_moe_activity(log_text: str) -> bool:
    """Return True when server.log shows fused-MoE activity we might parse."""
    text = log_text or ""
    return "[aiter] [fused_moe]" in text or "Mxfp4 MoE backend" in text


def aiter_log_tuned_config_enabled(envs: dict[str, str]) -> bool:
    """Mirror dense apply verification: dispatch attribution needs the flag."""
    raw = str(envs.get("AITER_LOG_TUNED_CONFIG", "1")).strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def read_latest_integrate_server_log(
    session_dir: Path,
    integrate_name: str = FMOE_INTEGRATE_RUN,
) -> tuple[Path, str] | None:
    """Return the newest ``server.log`` under an integrate run, if readable.

    Retries land in ``<integrate_name>-2``/``-3`` siblings rather than inside
    the original directory, so those are scanned too; otherwise a retried run is
    judged on the attempt it replaced.
    """
    parent = session_dir / "runs" / "integrate"
    run_dirs = [parent / integrate_name, *sorted(parent.glob(f"{integrate_name}-*"))]
    logs = sorted(
        (log_path for run_dir in run_dirs for log_path in run_dir.rglob("server.log")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    if not logs:
        return None
    try:
        return logs[-1], logs[-1].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def tuned_fmoe_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Return candidate rows with lookup keys and kernel names."""
    out: list[dict[str, str]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    if not lines:
        return out
    header = [col.strip() for col in lines[0].split(",")]
    try:
        indices = {name: header.index(name) for name in FMOE_INDEX_COLS}
    except ValueError:
        return out
    kn1_idx = header.index("kernelName1") if "kernelName1" in header else None
    kn2_idx = header.index("kernelName2") if "kernelName2" in header else None
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) <= max(indices.values()):
            continue
        record = {name: _normalize_fmoe_field(name, cols[index]) for name, index in indices.items()}
        if not all(record.get(field) for field in FMOE_INDEX_COLS[:7]):
            continue
        if kn1_idx is not None and kn1_idx < len(cols):
            record["kernelName1"] = cols[kn1_idx].strip()
        if kn2_idx is not None and kn2_idx < len(cols):
            record["kernelName2"] = cols[kn2_idx].strip()
        out.append(record)
    return out


def _fmoe_candidate_row_for_dispatch(
    record: dict[str, str],
    candidate_rows: dict[FmoeDispatchKey, dict[str, str]],
) -> dict[str, str] | None:
    return candidate_rows.get(fmoe_dispatch_lookup_key(record))


def fmoe_tuned_config_coverage(
    candidate_rows: Iterable[dict[str, str]],
    requested_dispatches: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Report how many logged dispatches hit a candidate row *and* kernel pair."""
    by_key = {fmoe_dispatch_lookup_key(row): row for row in candidate_rows}
    requested = list(requested_dispatches)
    if not requested:
        return {
            "requested": 0,
            "covered": 0,
            "coverage_pct": None,
            "tuned_rows": len(by_key),
        }
    covered: list[dict[str, str]] = []
    uncovered: list[dict[str, str]] = []
    default_count = 0
    kernel_mismatch = 0
    for record in requested:
        if record.get("descriptor") == "default":
            default_count += 1
            uncovered.append(record)
            continue
        row = _fmoe_candidate_row_for_dispatch(record, by_key)
        if row is None:
            uncovered.append(record)
            continue
        if record.get("kernelName1") == row.get("kernelName1") and record.get("kernelName2") == row.get("kernelName2"):
            covered.append(record)
        else:
            kernel_mismatch += 1
            uncovered.append(record)
    total = len(requested)
    return {
        "requested": total,
        "covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / total, 2) if total else None,
        "tuned_rows": len(by_key),
        "runtime_default": default_count,
        "kernel_name_mismatch": kernel_mismatch,
        "uncovered_sample": [
            {
                "gfx": record.get("gfx"),
                "token": record.get("token"),
                "descriptor": record.get("descriptor"),
                "kernelName1": record.get("kernelName1"),
                "kernelName2": record.get("kernelName2"),
            }
            for record in uncovered[:10]
        ],
    }


def parse_aiter_consulted_tables(log_text: str) -> set[str]:
    """Return the tuned-config files the runtime actually looked in.

    aiter keys each quantisation variant to its own table and env var (plain
    block-scale vs ``..._BPRESHUFFLE``, for instance). When the server dispatches
    to a variant the tuner did not target, the tuned CSV is never consulted at
    all -- a different failure from a CSV that is consulted but has no matching
    row, and one worth naming separately.
    """
    return {table for _m, _n, _k, table in _AITER_SHAPE_MISS_RE.findall(log_text or "")}


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
    known_covered: Iterable[Shape] | None = None,
) -> dict[str, Any]:
    """Report how many requested shapes a tuned CSV can actually serve.

    Replays aiter's three-step lookup against the CSV keys, so the result
    answers "will this artifact ever be used?" rather than "did the micro
    benchmark look good?".

    Args:
        tuned_shapes: ``(M, N, K)`` keys present in the tuned CSV.
        requested_shapes: ``(M, N, K)`` triples the runtime asked for.
        known_covered: Shapes the runtime *itself* reported as resolved against
            the tuned table. These count as covered without replaying the
            ladder: aiter has already stated which row it used, and that is
            better evidence than our reconstruction of its padding rules. Any
            drift between the two -- a new rung, a variant with different
            clamping -- would otherwise score a served shape as uncovered and
            revert a run that worked.
    """
    tuned = {(int(m), int(n), int(k)) for m, n, k in tuned_shapes}
    requested = sorted({(int(m), int(n), int(k)) for m, n, k in requested_shapes})
    confirmed = {(int(m), int(n), int(k)) for m, n, k in (known_covered or ())}
    if not requested:
        return {
            "requested": 0,
            "covered": 0,
            "coverage_pct": None,
            "tuned_rows": len(tuned),
        }
    covered = [
        shape for shape in requested if shape in confirmed or any(key in tuned for key in aiter_lookup_keys(shape))
    ]
    return {
        "requested": len(requested),
        "covered": len(covered),
        "coverage_pct": round(100.0 * len(covered) / len(requested), 2),
        "tuned_rows": len(tuned),
        "uncovered_sample": [{"M": m, "N": n, "K": k} for m, n, k in requested if (m, n, k) not in set(covered)][:10],
    }
