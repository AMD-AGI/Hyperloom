# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured roofline snapshot extraction for final report / dashboards."""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import dict_rows, first_str
from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.roofline_ceiling import (
    ComputeTarget,
    DecodeShape,
    RooflineBreakdown,
    apply_runtime_dtype,
    resolve_compute_peak_provenance,
    resolve_runtime_dtype,
)

log = logging.getLogger(__name__)

_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|",
    re.MULTILINE,
)
_PCT_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
DEFAULT_SATURATION_WITHIN_PCT: float = 95.0


def saturation_within_threshold_pct() -> float:
    """Return the configured roofline saturation threshold percentage."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_SATURATION_WITHIN_PCT", "").strip()
    if not raw:
        return DEFAULT_SATURATION_WITHIN_PCT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SATURATION_WITHIN_PCT
    return val if 0.0 < val <= 100.0 else DEFAULT_SATURATION_WITHIN_PCT


def _parse_pct(raw: str | None) -> float | None:
    """Extract a leading percentage number from a raw table cell."""
    if raw is None:
        return None
    m = _PCT_NUM_RE.search(str(raw).replace(",", ""))
    if not m:
        return None
    try:
        return round(float(m.group(1)), 2)
    except (TypeError, ValueError):
        return None


def _parse_executive_table(text: str) -> dict[str, str]:
    """Parse a markdown Executive Summary table into a label→value map."""
    rows: dict[str, str] = {}
    for m in _TABLE_ROW_RE.finditer(text):
        label = m.group("label").strip()
        value = m.group("value").strip()
        if label.lower() in ("metric", "--------"):
            continue
        rows[label] = value
    return rows


def _parse_top_bottleneck(raw: str | None) -> str | None:
    """Strip the trailing ``(pct%)`` annotation from a bottleneck label."""
    if not raw:
        return None
    name = raw.split("(")[0].strip()
    return name or None


def _exposed_memcpy_pct(tracelens_dir: Path) -> float | None:
    """Exposed memcpy share from ``perf_report_csvs/gpu_timeline.csv``; analysis.md omits it."""
    try:
        with (tracelens_dir / "perf_report_csvs" / "gpu_timeline.csv").open(encoding="utf-8", newline="") as fh:
            row = next((r for r in csv.DictReader(fh) if r.get("type") == "exposed_memcpy_time"), None)
    except OSError:
        return None
    return _parse_pct(row.get("percent")) if row else None


def _shares_from_md(path: Path) -> dict[str, Any]:
    """Workload shares from the markdown Executive Summary table."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    rows = _parse_executive_table(text)
    return {
        "compute_pct": _parse_pct(rows.get("Compute %")),
        "idle_pct": _parse_pct(rows.get("Idle %")),
        "comm_pct": _parse_pct(rows.get("Exposed Communication %") or rows.get("Communication %")),
        "top_bottleneck": _parse_top_bottleneck(rows.get("Top Bottleneck Category")),
    }


def _shares_from_json(path: Path) -> dict[str, Any]:
    """Workload shares from ``analysis.json``: ``executive_summary.metrics`` and the first top operation."""
    data = read_json(path, default={}, require_dict=True)
    metrics = (data.get("executive_summary") or {}).get("metrics") or {}
    top_operations = data.get("top_operations")
    return {
        "top_bottleneck": top_operations[0]["category"] if top_operations else None,
        "compute_pct": metrics.get("compute_pct"),
        "idle_pct": metrics.get("idle_pct"),
        "comm_pct": metrics.get("exposed_communication_pct"),
    }


def extract_workload_summary(analysis_md_path: str | Path) -> dict[str, Any]:
    """Best-effort workload-level metrics from the report's Executive Summary (``analysis.md`` or ``analysis.json``)."""
    path = Path(analysis_md_path)
    out: dict[str, Any] = {
        "compute_pct": None,
        "idle_pct": None,
        "comm_pct": None,
        "memcpy_pct": None,
        "top_bottleneck": None,
    }
    if not path.is_file():
        return out
    out.update(_shares_from_json(path) if path.suffix == ".json" else _shares_from_md(path))
    out["memcpy_pct"] = _exposed_memcpy_pct(path.parent)
    return out


def _top_kernel_from_json(path: Path) -> dict[str, Any] | None:
    """The ``compute_optimizations`` member with the highest ``pct_e2e`` of an ``analysis.json``."""
    data = read_json(path, default={}, require_dict=True)
    members = [m for task in data.get("compute_optimizations") or [] for m in task["members"]]
    if not members:
        return None
    best = max(members, key=lambda m: m["pct_e2e"])
    return {
        "name": best["operation_cell"],
        "gpu_pct": round(best["pct_e2e"], 2),
        "efficiency_pct": _parse_pct(None if best["efficiency_percent"] is None else str(best["efficiency_percent"])),
        "bound_type": best["bound"] or "",
    }


def extract_top_kernel(analysis_md_path: str | Path) -> dict[str, Any] | None:
    """Return the highest-share operation: from ``category_data`` metrics, or from an ``analysis.json`` path."""
    md_path = Path(analysis_md_path)
    if md_path.suffix == ".json":
        return _top_kernel_from_json(md_path)
    cat_dir = md_path.parent / "category_data"
    if not cat_dir.is_dir():
        return None

    best: dict[str, Any] | None = None
    best_pct = -1.0

    for metrics_path in sorted(cat_dir.glob("*_metrics.json")):
        data = read_json(metrics_path, default=None, require_dict=True)
        if data is None:
            continue
        category = str(data.get("category") or metrics_path.stem.replace("_metrics", ""))
        for op in data.get("operations") or []:
            if not isinstance(op, dict):
                continue
            pct_raw = op.get("percent_of_total")
            try:
                pct = float(pct_raw)
            except (TypeError, ValueError):
                continue
            if pct <= best_pct:
                continue
            eff = op.get("efficiency") if isinstance(op.get("efficiency"), dict) else {}
            best_pct = pct
            best = {
                "name": str(op.get("name") or ""),
                "gpu_pct": round(pct, 2),
                "efficiency_pct": _parse_pct(
                    str(eff.get("efficiency_percent")) if eff.get("efficiency_percent") is not None else None
                ),
                "bound_type": str(eff.get("bound_type") or op.get("bound_type") or ""),
                "category": category,
            }
    if best and not best.get("name"):
        return None
    return best


def within_roofline_pct(*, peak: float, achieved: float) -> float | None:
    """``round(achieved/peak*100, 2)``, or None when either input is non-positive."""
    if peak <= 0 or achieved <= 0:
        return None
    return round(achieved / peak * 100.0, 2)


def _clamp_within(raw: float | None) -> tuple[float | None, float | None]:
    """Split a raw within-roofline ratio into the reported and overshoot values."""
    if raw is None:
        return None, None
    within = min(float(raw), 100.0)
    return within, round(100.0 - within, 2)


def attach_perfmodel_breakdown(snapshot: dict[str, Any], state: Any, *, arm: str) -> None:
    """Add ``roofline_provenance`` (+ ``perfmodel_breakdown`` when the PerfModel succeeds) for *arm*."""
    from .roofline_ceiling import compute_roofline_from_perfmodel, load_model_meta, resolve_runtime_workload

    runtime = resolve_runtime_workload(state, arm=arm)
    if framework_registry.is_scriptable(runtime.framework):
        return
    meta = load_model_meta(runtime.model_path, precision_hint=runtime.precision)
    if meta is None:
        return
    rt = resolve_runtime_dtype(state, meta, arm=arm)
    meta = apply_runtime_dtype(meta, rt)
    compute_precision_tag = rt.compute_precision_tag or runtime.precision or "bf16"
    pm_bd = compute_roofline_from_perfmodel(
        meta=meta,
        target=ComputeTarget(gpu_type=runtime.gpu_type, num_gpus=runtime.tp, precision_tag=compute_precision_tag),
        shape=DecodeShape(isl=runtime.isl, osl=runtime.osl, concurrency=runtime.concurrency),
    )
    snapshot["roofline_provenance"] = {
        "formula": "perfmodel" if pm_bd is not None else "legacy",
        **resolve_compute_peak_provenance(runtime.gpu_type, compute_precision_tag),
        "runtime_weight_dtype": rt.weight_dtype_tag,
        "runtime_weight_dtype_bytes": rt.weight_dtype_bytes,
        "runtime_activation_dtype_bytes": rt.activation_dtype_bytes,
        "quantization": rt.quantization,
        "dtype_source": rt.source,
        "effective_concurrency": runtime.concurrency,
        "runtime_tp": runtime.tp,
        "runtime_isl": runtime.isl,
        "runtime_osl": runtime.osl,
        "runtime_precision": runtime.precision,
        "runtime_framework": runtime.framework,
    }
    if pm_bd is not None:
        snapshot["perfmodel_breakdown"] = {
            "decode_tok_per_s": pm_bd.decode_tok_per_s,
            "prefill_tok_per_s": pm_bd.prefill_tok_per_s,
            "decode_mem_tok_per_s": pm_bd.decode_mem_tok_per_s,
            "decode_cmp_tok_per_s": pm_bd.decode_cmp_tok_per_s,
            "bound_kind": pm_bd.bound_kind,
            "hbm_bw_gbps": pm_bd.hbm_bw_gbps,
            "peak_achievable_tflops": pm_bd.peak_achievable_tflops,
            "ops": [
                {
                    "name": op.name,
                    "flops": op.flops,
                    "bytes_moved": op.bytes_moved,
                    "ai": op.ai,
                    "time_s": op.time_s,
                    "bound": op.bound,
                    "pct_time": op.pct_time,
                }
                for op in pm_bd.ops
            ],
        }


@dataclass(frozen=True)
class RooflineReading:
    """One arm's achieved throughput (in ``throughput_unit``) against its modelled ceiling."""

    ceiling: RooflineBreakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
    achieved_tok_per_sec: float = 0.0
    throughput_unit: str = "tok/s"


def build_roofline_snapshot(
    *,
    snapshot_id: int | None,
    ts: str,
    analysis_md_path: str,
    reading: RooflineReading = RooflineReading(),
    framework: str = "",
) -> dict[str, Any]:
    """Materialise one side (baseline or latest) of the comparison."""
    ceiling = reading.ceiling
    peak, achieved = ceiling.peak_tok_per_sec, reading.achieved_tok_per_sec
    within_raw = within_roofline_pct(peak=peak, achieved=achieved)
    within, gap = _clamp_within(within_raw)
    snap: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "ts": ts or "",
        # Framework tag so the report layer renders the achieved metric's unit.
        "framework": str(framework or ""),
        # sidecar pointer — overwritten by record_trace_analyze.
        "kernel_roofline_path": "",
        "compute_pct": None,
        "idle_pct": None,
        "comm_pct": None,
        "memcpy_pct": None,
        "top_bottleneck": None,
        "top_kernel": None,
        # Primary decode ceiling plus memory/compute sides; None when unavailable.
        "theoretical_peak_tok_per_sec": (float(peak) if peak > 0 else None),
        "roofline_mem_ceiling_tok_per_sec": (float(ceiling.mem_tok_per_sec) if ceiling.mem_tok_per_sec > 0 else None),
        "roofline_cmp_ceiling_tok_per_sec": (float(ceiling.cmp_tok_per_sec) if ceiling.cmp_tok_per_sec > 0 else None),
        "roofline_bound_kind": (str(ceiling.bound_kind) if ceiling.bound_kind else "unknown"),
        # Unit for the *_tok_per_sec fields ("tok/s" text-gen, "img/s" xDiT).
        "throughput_unit": (str(reading.throughput_unit) if reading.throughput_unit else "tok/s"),
        "achieved_tok_per_sec": (float(achieved) if achieved > 0 else None),
        "within_roofline_pct": within,
        "gap_to_roofline_pct": gap,
        # Uncapped ratio + the flag derived from it: an achieved throughput above the modelled ceiling is a
        # ceiling-model problem the report must show.
        "within_roofline_pct_uncapped": within_raw,
        "roofline_ceiling_exceeded": bool(within_raw is not None and within_raw > 100.0),
    }
    if not analysis_md_path:
        return snap
    wl = extract_workload_summary(analysis_md_path)
    snap["compute_pct"] = wl.get("compute_pct")
    snap["idle_pct"] = wl.get("idle_pct")
    snap["comm_pct"] = wl.get("comm_pct")
    snap["memcpy_pct"] = wl.get("memcpy_pct")
    snap["top_bottleneck"] = wl.get("top_bottleneck")
    top_k = extract_top_kernel(analysis_md_path)
    if top_k:
        snap["top_kernel"] = {
            "name": top_k.get("name"),
            "gpu_pct": top_k.get("gpu_pct"),
            "efficiency_pct": top_k.get("efficiency_pct"),
            "bound_type": top_k.get("bound_type") or None,
        }
    return snap


def _num_delta(latest: float | None, baseline: float | None) -> float | None:
    """Return ``latest - baseline`` rounded to two decimals."""
    if latest is None or baseline is None:
        return None
    return round(latest - baseline, 2)


#: Relative tolerance for treating two modelled ceilings as the same ceiling.
_CEILING_REL_TOL: float = 0.01


def _ceiling_of(snapshot: dict[str, Any]) -> float | None:
    """Return one snapshot's modelled ceiling in its throughput unit, or ``None``."""
    peak = snapshot.get("theoretical_peak_tok_per_sec")
    if isinstance(peak, (int, float)) and peak > 0:
        return float(peak)
    return None


def ceilings_comparable(
    baseline: dict[str, Any] | None,
    latest: dict[str, Any] | None,
) -> bool:
    """Whether both snapshots were measured against the same modelled ceiling."""
    base_ceiling = _ceiling_of(baseline or {})
    latest_ceiling = _ceiling_of(latest or {})
    if base_ceiling is None or latest_ceiling is None:
        return True
    largest = max(base_ceiling, latest_ceiling)
    return abs(base_ceiling - latest_ceiling) <= largest * _CEILING_REL_TOL


def build_roofline_comparison_from_history(
    snapshots: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Build the ``roofline_comparison`` block from :attr:`SharedState.roofline_snapshots` (preferred entry point for building the comparison block from snapshot history)."""
    snapshots = list(snapshots or [])
    if not snapshots:
        return None
    baseline = dict(snapshots[0])
    latest = dict(snapshots[-1])
    base_id = baseline.get("snapshot_id")
    latest_id = latest.get("snapshot_id")
    same_snapshot = isinstance(base_id, int) and isinstance(latest_id, int) and base_id == latest_id
    mode = "single_snapshot" if same_snapshot else "before_after"
    out: dict[str, Any] = {
        "mode": mode,
        "baseline": baseline,
        "latest": latest,
    }
    if mode == "before_after":
        base_eff = (baseline.get("top_kernel") or {}).get("efficiency_pct")
        lat_eff = (latest.get("top_kernel") or {}).get("efficiency_pct")
        comparable = ceilings_comparable(baseline, latest)
        out["ceilings_comparable"] = comparable
        out["delta"] = {
            "compute_pct": _num_delta(
                latest.get("compute_pct"),
                baseline.get("compute_pct"),
            ),
            "idle_pct": _num_delta(
                latest.get("idle_pct"),
                baseline.get("idle_pct"),
            ),
            "comm_pct": _num_delta(
                latest.get("comm_pct"),
                baseline.get("comm_pct"),
            ),
            "top_kernel_efficiency_pct": _num_delta(lat_eff, base_eff),
            # Saturation deltas are only meaningful against one shared ceiling; across a moved ceiling they would
            # report a denominator change as a saturation change.
            "within_roofline_pct": (
                _num_delta(
                    latest.get("within_roofline_pct"),
                    baseline.get("within_roofline_pct"),
                )
                if comparable
                else None
            ),
            "gap_to_roofline_pct": (
                _num_delta(
                    latest.get("gap_to_roofline_pct"),
                    baseline.get("gap_to_roofline_pct"),
                )
                if comparable
                else None
            ),
        }
    return out


def _fmt_delta(val: float | None) -> str:
    """Format a signed delta cell with one decimal place."""
    if val is None:
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}"


def _fmt_tput(v: float | None, framework: str = "") -> str:
    """Format the achieved primary-metric cell for the roofline table."""
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    if framework_registry.is_scriptable(framework):
        return framework_registry.format_primary_metric(framework, float(v))
    return f"{float(v):.1f} tok/s"


def _fmt_pct_cell(v: float | None) -> str:
    """Format a percentage cell; ``—`` when missing."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{float(v):.1f}%"


def _pct_cells(snaps: list[dict[str, Any]], key: str) -> list[str]:
    """One :func:`_fmt_pct_cell` per snapshot for ``key``."""
    return [_fmt_pct_cell(snap.get(key)) for snap in snaps]


_CEILING_LABEL = "Roofline ceiling"


def _unit_of(snapshot: dict[str, Any]) -> str:
    """The throughput unit a snapshot's ceiling and achieved values are in."""
    return str(snapshot.get("throughput_unit") or "tok/s")


def _ceiling_note(snapshot: dict[str, Any]) -> str:
    """The ceiling explainer, worded from the snapshot's bound kind and throughput unit."""
    kind = snapshot.get("roofline_bound_kind")
    basis = f"{kind}-roofline ceiling" if kind in ("memory", "compute") else "modelled ceiling"
    return (
        f"_The **{_CEILING_LABEL}** below is the {basis} "
        "(see `roofline_ceiling.compute_roofline_breakdown_from_state`). "
        f"It is an upper bound on the measured throughput ({_unit_of(snapshot)}): real runs "
        "stay under it because of comm overhead and kernel efficiency below 100%. "
        "**Within roofline %** = measured / ceiling; **Gap to roofline %** = 100 − Within._"
    )


def _ceiling_lines(
    baseline: dict[str, Any],
    latest: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    """Render the ceiling explainer and header above the metrics table."""
    base_ceiling = _ceiling_of(baseline)
    latest_ceiling = _ceiling_of(latest)
    if mode == "before_after" and base_ceiling is not None and latest_ceiling is not None:
        if not ceilings_comparable(baseline, latest):
            return [
                _ceiling_note(baseline),
                "",
                f"**{_CEILING_LABEL} (Base):** {base_ceiling:.1f} {_unit_of(baseline)}",
                f"**{_CEILING_LABEL} (Opt):** {latest_ceiling:.1f} {_unit_of(latest)}",
                "_Ceilings differ (runtime dtype / quantization changed): each side's "
                "within % and gap % are measured against its own ceiling, so the Δ "
                "column is withheld._",
                "",
            ]
    value, side = (base_ceiling, baseline) if base_ceiling is not None else (latest_ceiling, latest)
    if value is None:
        return []
    return [
        _ceiling_note(side),
        "",
        f"**{_CEILING_LABEL}:** "
        f"{value:.1f} {_unit_of(side)}  "
        f"_(single-source ceiling; baseline / latest compared against it)_",
        "",
    ]


#: Snapshot key holding the uncapped achieved/ceiling ratio.
_UNCAPPED_KEY = "within_roofline_pct_uncapped"


def _ceiling_exceeded(snapshot: dict[str, Any]) -> bool:
    """Whether a snapshot measured above its own modelled ceiling."""
    return bool(snapshot.get("roofline_ceiling_exceeded"))


def _ceiling_exceeded_lines(
    baseline: dict[str, Any],
    latest: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    """Render the warning for a snapshot that measured above its ceiling."""
    sides: list[str] = []
    if _ceiling_exceeded(baseline):
        sides.append("Base" if mode == "before_after" else "this snapshot")
    if _ceiling_exceeded(latest):
        sides.append("Opt")
    if not sides:
        return []
    ratios = " / ".join(
        _fmt_pct_cell(snap.get(_UNCAPPED_KEY)) for snap in (baseline, latest) if _ceiling_exceeded(snap)
    )
    return [
        f"> **Ceiling model exceeded ({', '.join(sides)}):** measured throughput reached "
        f"{ratios} of the modelled ceiling, so **Within roofline %** is capped at 100% and "
        f"**Gap to roofline %** at 0%. The theoretical peak is understated — check the "
        f"runtime dtype / quantization / GPU spec behind it before reading the saturation "
        f"numbers.",
        "",
    ]


def format_roofline_metrics_table(cmp: dict[str, Any]) -> list[str]:
    """Render the compact Base / Opt / Δ markdown table (session-constant ceiling rendered once above the Base/Opt columns)."""
    baseline = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    delta = cmp.get("delta") or {}
    mode = cmp.get("mode") or "single_snapshot"
    single = mode == "single_snapshot"
    sides = [baseline] if single else [baseline, latest]
    kernels = [snap.get("top_kernel") or {} for snap in sides]

    def row(label: str, cells: list[str], delta_value: float | None = None) -> str:
        """One table row: a cell per side, then the Δ cell when two snapshots are compared."""
        return f"| {label} | {' | '.join(cells if single else [*cells, _fmt_delta(delta_value)])} |"

    # The ceiling is usually session-constant, so surface it once above the table; when the two sides model different
    # ceilings (a dtype / quantization change) both are reported and the within/gap columns are flagged as not
    # directly comparable.
    lines = _ceiling_lines(baseline, latest, mode=mode)
    if single:
        lines += ["| Metric | Value |", "|--------|-------|"]
    else:
        lines += ["| Metric | Base | Opt | Δ |", "|--------|------|-----|---|"]
    lines.append(row("Compute %", _pct_cells(sides, "compute_pct"), delta.get("compute_pct")))
    lines.append(row("Idle %", _pct_cells(sides, "idle_pct"), delta.get("idle_pct")))
    lines.append(row("Comm %", _pct_cells(sides, "comm_pct"), delta.get("comm_pct")))
    lines.append(row("Top bottleneck", [str(snap.get("top_bottleneck") or "—") for snap in sides]))
    lines.append(
        row("Top kernel efficiency", _pct_cells(kernels, "efficiency_pct"), delta.get("top_kernel_efficiency_pct"))
    )
    if any(kernel.get("name") for kernel in kernels):
        lines.append(row("Top kernel", [f"`{kernel.get('name') or '—'}`" for kernel in kernels]))
    lines.append(
        row(
            "Achieved output_throughput",
            [_fmt_tput(snap.get("achieved_tok_per_sec"), snap.get("framework")) for snap in sides],
        )
    )
    for label, key in (("Within roofline %", "within_roofline_pct"), ("Gap to roofline %", "gap_to_roofline_pct")):
        lines.append(row(label, _pct_cells(sides, key), delta.get(key)))
    exceeded = _ceiling_exceeded_lines(baseline, {} if single else latest, mode=mode)
    if exceeded:
        lines.append(row("Within roofline % (uncapped)", _pct_cells(sides, _UNCAPPED_KEY)))
    lines.append("")
    lines.extend(exceeded)
    return lines


#: Dominant roofline direction → (specialist domain, kb tag). Shared by the
#: profiler digest and the coordinator's bottleneck-redirect advisory.
BOTTLENECK_DOMAIN_HINTS: dict[str, tuple[str, str]] = {
    "comm": ("comm_specialist", "communication"),
    "idle": ("system_specialist", "systems"),
    "compute": ("kernel_switch_specialist", "kernel_agent"),
    "memory": ("serving_specialist", "framework"),
}

_DIRECTION_PCT_KEYS = (("compute", "compute_pct"), ("idle", "idle_pct"), ("comm", "comm_pct"))


def dominant_direction(snapshot: dict[str, Any] | None) -> tuple[str, float]:
    """Return ``(direction, pct)`` for the most-saturated direction in one snapshot."""
    if not isinstance(snapshot, dict):
        return "", 0.0
    candidates: dict[str, float] = {}
    for direction, key in _DIRECTION_PCT_KEYS:
        val = snapshot.get(key)
        if isinstance(val, (int, float)):
            candidates[direction] = float(val)
    bound_kind = str(snapshot.get("roofline_bound_kind") or "").strip().lower()
    if bound_kind == "memory":
        candidates["memory"] = max(candidates.get("compute", 0.0), 0.0) + 0.01
    if not candidates:
        return "", 0.0
    best = max(candidates.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def direction_saturation(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Classify whether the latest dominant roofline direction is near ceiling."""
    direction, pct = dominant_direction(snapshot)
    snap = snapshot if isinstance(snapshot, dict) else {}
    within = snap.get("within_roofline_pct")
    gap = snap.get("gap_to_roofline_pct")
    threshold = saturation_within_threshold_pct()
    saturated = isinstance(within, (int, float)) and float(within) >= threshold
    hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
    return {
        "direction": direction,
        "direction_pct": round(float(pct), 2),
        "within_pct": float(within) if isinstance(within, (int, float)) else None,
        "gap_pct": float(gap) if isinstance(gap, (int, float)) else None,
        "saturated": bool(saturated),
        "threshold_pct": threshold,
        "bound_kind": snap.get("roofline_bound_kind"),
        "domain_hint": {"domain": hint[0], "tag": hint[1]} if hint else {},
    }


def build_profiler_digest(
    snapshots: list[dict[str, Any]] | None,
    trace_analyze: dict[str, Any] | None,
) -> str:
    """Render a compact, bottleneck-focused profiler block for prompt injection."""
    snaps = dict_rows(snapshots)
    ta = trace_analyze if isinstance(trace_analyze, dict) else {}
    if not snaps and not ta:
        return ""
    latest = snaps[-1] if snaps else {}

    bound_kind = str(latest.get("roofline_bound_kind") or "").strip() or "unknown"
    lines: list[str] = [
        f"bound_kind={bound_kind}  "
        f"compute={_fmt_pct_cell(latest.get('compute_pct'))}  "
        f"idle={_fmt_pct_cell(latest.get('idle_pct'))}  "
        f"comm={_fmt_pct_cell(latest.get('comm_pct'))}  "
        f"memcpy={_fmt_pct_cell(latest.get('memcpy_pct'))}"
    ]

    if len(snaps) >= 2:
        prev = snaps[-2]
        parts: list[str] = []
        for label, key in _DIRECTION_PCT_KEYS:
            d = _num_delta(latest.get(key), prev.get(key))
            if d is not None:
                parts.append(f"{label} {_fmt_delta(d)}pp")
        if parts:
            lines.append("delta_vs_prev: " + "  ".join(parts))

    rows: list[str] = []
    for entry in ta.get("hot_kernels_top15", [])[:3]:
        name = first_str(entry.get("name"), entry.get("kernel_id"), "?")
        seg = f"  {name}  {_fmt_pct_cell(entry.get('gpu_pct'))} gpu"
        eff = entry.get("efficiency_percent")
        if isinstance(eff, (int, float)):
            seg += f"  (eff {float(eff):.1f}%)"
        rows.append(seg)
    if not rows and latest.get("top_bottleneck"):
        rows.append(f"  {latest.get('top_bottleneck')}")
    if rows:
        lines.append("top_bottlenecks:")
        lines.extend(rows)

    direction, _pct_val = dominant_direction(latest)
    lever = BOTTLENECK_DOMAIN_HINTS.get(direction)
    if lever:
        lines.append(f"suggested_lever (dominant={direction}): {lever[0]}")

    reusable = ta.get("reusable_native_kernel_ids")
    if reusable:
        lines.append(f"reusable_native_kernel_ids={[str(r) for r in reusable[:12]]}")

    return "\n".join(lines)
