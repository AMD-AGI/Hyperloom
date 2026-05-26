"""Structured roofline snapshot extraction for final report / dashboards.

Parses TraceLens ``analysis.md`` Executive Summary tables and
``category_data/*_metrics.json`` for a compact before/after comparison
shape consumed by ``report.py`` and downstream frontends.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|",
    re.MULTILINE,
)
_PCT_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def _parse_pct(raw: str | None) -> float | None:
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
    rows: dict[str, str] = {}
    for m in _TABLE_ROW_RE.finditer(text):
        label = m.group("label").strip()
        value = m.group("value").strip()
        if label.lower() in ("metric", "--------"):
            continue
        rows[label] = value
    return rows


def _parse_top_bottleneck(raw: str | None) -> str | None:
    if not raw:
        return None
    # ``MoE_fused (28.78%)`` -> ``MoE_fused``
    name = raw.split("(")[0].strip()
    return name or None


def extract_workload_summary(analysis_md_path: str | Path) -> dict[str, Any]:
    """Best-effort workload-level metrics from Executive Summary table."""
    path = Path(analysis_md_path)
    out: dict[str, Any] = {
        "compute_pct": None,
        "idle_pct": None,
        "comm_pct": None,
        "top_bottleneck": None,
    }
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    rows = _parse_executive_table(text)
    out["compute_pct"] = _parse_pct(rows.get("Compute %"))
    out["idle_pct"] = _parse_pct(rows.get("Idle %"))
    out["comm_pct"] = _parse_pct(
        rows.get("Exposed Communication %") or rows.get("Communication %")
    )
    out["top_bottleneck"] = _parse_top_bottleneck(
        rows.get("Top Bottleneck Category")
    )
    return out


# ---------------------------------------------------------------------------
# F3-4 — saturation per direction (soft advisory feed)
# ---------------------------------------------------------------------------
#: Direction → list of Executive Summary table label aliases. The
#: aliases mirror what TraceLens' analyzer emits today; missing labels
#: degrade silently to ``0.0`` so a partial / experimental analysis.md
#: never produces a false saturation hint.
_SATURATION_LABEL_MAP: dict[str, tuple[str, ...]] = {
    "compute": ("Compute %", "Compute Bound %", "Compute Bound"),
    "memory": ("Memory %", "Memory Bound %", "Memory Bound"),
    "host_overhead": ("Idle %", "Host Overhead %", "GPU Idle %"),
    "comm": ("Exposed Communication %", "Communication %", "Comm %"),
}

#: Saturation threshold (%) above which the prompt-side advisory
#: surfaces a direction. Single source of truth for the threshold so
#: the SharedState advisory renderer + tests share one constant.
SATURATION_ADVISORY_THRESHOLD_PCT: float = 80.0


def derive_saturation_per_direction(analysis_md_text: str) -> dict[str, float]:
    """Parse Executive Summary cells from analysis.md and return
    ``{direction: saturation_pct}`` for the four canonical directions
    (compute / memory / host_overhead / comm).

    F3-4 (Roofline-v2): the resulting dict is appended to
    :attr:`SharedState.roofline_saturation_history` after every
    successful ``roofline`` action so the prompt advisory renderer can
    flag directions ≥ :data:`SATURATION_ADVISORY_THRESHOLD_PCT` as
    "diminishing returns" hints.

    Soft contract: any direction whose label is missing from the
    Executive Summary (or whose value cannot be parsed as a percentage)
    degrades silently to ``0.0`` — an empty / partial analysis.md must
    never produce a false saturation hint that could mis-steer the
    LLM toward dropping a direction.
    """
    out: dict[str, float] = {k: 0.0 for k in _SATURATION_LABEL_MAP}
    if not analysis_md_text:
        return out
    rows = _parse_executive_table(analysis_md_text)
    for direction, aliases in _SATURATION_LABEL_MAP.items():
        for alias in aliases:
            raw = rows.get(alias)
            pct = _parse_pct(raw)
            if pct is not None:
                out[direction] = float(pct)
                break
    return out


def _tracelens_dir_for_analysis_md(analysis_md_path: Path) -> Path:
    # ``.../tracelens/analysis.md`` -> ``.../tracelens/``
    if analysis_md_path.name == "analysis.md" and analysis_md_path.parent.name:
        return analysis_md_path.parent
    return analysis_md_path.parent


def extract_top_kernel(analysis_md_path: str | Path) -> dict[str, Any] | None:
    """Return the highest ``percent_of_total`` operation across category metrics."""
    md_path = Path(analysis_md_path)
    cat_dir = _tracelens_dir_for_analysis_md(md_path) / "category_data"
    if not cat_dir.is_dir():
        return None

    best: dict[str, Any] | None = None
    best_pct = -1.0

    for metrics_path in sorted(cat_dir.glob("*_metrics.json")):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
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
                    str(eff.get("efficiency_percent"))
                    if eff.get("efficiency_percent") is not None
                    else None
                ),
                "bound_type": str(eff.get("bound_type") or op.get("bound_type") or ""),
                "category": category,
            }
    if best and not best.get("name"):
        return None
    return best


def build_roofline_snapshot(
    *,
    snapshot_id: int | None,
    ts: str,
    analysis_md_path: str,
) -> dict[str, Any]:
    """Materialise one side (baseline or latest) of the comparison."""
    snap: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "ts": ts or "",
        "compute_pct": None,
        "idle_pct": None,
        "comm_pct": None,
        "top_bottleneck": None,
        "top_kernel": None,
    }
    if not analysis_md_path:
        return snap
    wl = extract_workload_summary(analysis_md_path)
    snap["compute_pct"] = wl.get("compute_pct")
    snap["idle_pct"] = wl.get("idle_pct")
    snap["comm_pct"] = wl.get("comm_pct")
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


def _snapshot_id_from_meta(meta: dict[str, Any]) -> int | None:
    for key in ("snapshot_id", "roofline_snapshot_id"):
        raw = meta.get(key)
        if isinstance(raw, int):
            return raw
    return None


def _num_delta(latest: float | None, baseline: float | None) -> float | None:
    if latest is None or baseline is None:
        return None
    return round(latest - baseline, 2)


def build_roofline_comparison(
    baseline_meta: dict[str, Any],
    latest_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Build ``roofline_comparison`` block for ``final.json``."""
    base_path = str(baseline_meta.get("analysis_md_path") or "")
    latest_path = str(latest_meta.get("analysis_md_path") or "")
    if not base_path and not latest_path:
        return None

    baseline = build_roofline_snapshot(
        snapshot_id=_snapshot_id_from_meta(baseline_meta),
        ts=str(baseline_meta.get("ts") or ""),
        analysis_md_path=base_path,
    )
    latest = build_roofline_snapshot(
        snapshot_id=_snapshot_id_from_meta(latest_meta),
        ts=str(latest_meta.get("ts") or ""),
        analysis_md_path=latest_path or base_path,
    )

    same_snapshot = (
        base_path == latest_path
        or (
            isinstance(baseline.get("snapshot_id"), int)
            and isinstance(latest.get("snapshot_id"), int)
            and baseline["snapshot_id"] == latest["snapshot_id"]
        )
    )
    mode = "single_snapshot" if same_snapshot else "before_after"

    out: dict[str, Any] = {
        "mode": mode,
        "baseline": baseline,
        "latest": latest,
    }

    if mode == "before_after":
        base_eff = (baseline.get("top_kernel") or {}).get("efficiency_pct")
        lat_eff = (latest.get("top_kernel") or {}).get("efficiency_pct")
        out["delta"] = {
            "compute_pct": _num_delta(latest.get("compute_pct"), baseline.get("compute_pct")),
            "idle_pct": _num_delta(latest.get("idle_pct"), baseline.get("idle_pct")),
            "comm_pct": _num_delta(latest.get("comm_pct"), baseline.get("comm_pct")),
            "top_kernel_efficiency_pct": _num_delta(lat_eff, base_eff),
        }
    return out


def _fmt_delta(val: float | None) -> str:
    if val is None:
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}"


def format_roofline_metrics_table(cmp: dict[str, Any]) -> list[str]:
    """Render the compact Base / Opt / Δ markdown table."""

    def cell(v: float | None) -> str:
        return f"{v:.1f}%" if isinstance(v, float) else "—"

    baseline = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    delta = cmp.get("delta") or {}
    mode = cmp.get("mode") or "single_snapshot"

    lines: list[str] = []
    if mode == "single_snapshot":
        snap = baseline
        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Compute % | {cell(snap.get('compute_pct'))} |",
            f"| Idle % | {cell(snap.get('idle_pct'))} |",
            f"| Comm % | {cell(snap.get('comm_pct'))} |",
            f"| Top bottleneck | {snap.get('top_bottleneck') or '—'} |",
        ])
        tk = snap.get("top_kernel") or {}
        lines.append(f"| Top kernel efficiency | {cell(tk.get('efficiency_pct'))} |")
        if tk.get("name"):
            lines.append(f"| Top kernel | `{tk.get('name')}` |")
        lines.append("")
        return lines

    lines.extend([
        "| Metric | Base | Opt | Δ |",
        "|--------|------|-----|---|",
        f"| Compute % | {cell(baseline.get('compute_pct'))} | "
        f"{cell(latest.get('compute_pct'))} | "
        f"{_fmt_delta(delta.get('compute_pct'))} |",
        f"| Idle % | {cell(baseline.get('idle_pct'))} | "
        f"{cell(latest.get('idle_pct'))} | "
        f"{_fmt_delta(delta.get('idle_pct'))} |",
        f"| Comm % | {cell(baseline.get('comm_pct'))} | "
        f"{cell(latest.get('comm_pct'))} | "
        f"{_fmt_delta(delta.get('comm_pct'))} |",
        f"| Top bottleneck | {baseline.get('top_bottleneck') or '—'} | "
        f"{latest.get('top_bottleneck') or '—'} | — |",
    ])
    btk = baseline.get("top_kernel") or {}
    ltk = latest.get("top_kernel") or {}
    lines.append(
        f"| Top kernel efficiency | {cell(btk.get('efficiency_pct'))} | "
        f"{cell(ltk.get('efficiency_pct'))} | "
        f"{_fmt_delta(delta.get('top_kernel_efficiency_pct'))} |"
    )
    if btk.get("name") or ltk.get("name"):
        lines.append(
            f"| Top kernel | `{btk.get('name') or '—'}` | "
            f"`{ltk.get('name') or '—'}` | — |"
        )
    lines.append("")
    return lines
