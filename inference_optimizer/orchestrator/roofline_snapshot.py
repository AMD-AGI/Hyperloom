# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    """Extract a leading percentage number from a raw table cell.

    Args:
        raw (str | None): The raw cell text (e.g. ``"28.78%"``), or ``None``.

    Returns:
        float | None: The parsed number rounded to two decimals, or ``None``
            when the input is missing or unparseable.
    """
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
    """Parse a markdown Executive Summary table into a label→value map.

    Skips header and separator rows.

    Args:
        text (str): The markdown text containing the two-column table.

    Returns:
        dict[str, str]: Mapping of row label to its raw value cell.
    """
    rows: dict[str, str] = {}
    for m in _TABLE_ROW_RE.finditer(text):
        label = m.group("label").strip()
        value = m.group("value").strip()
        if label.lower() in ("metric", "--------"):
            continue
        rows[label] = value
    return rows


def _parse_top_bottleneck(raw: str | None) -> str | None:
    """Strip the trailing ``(pct%)`` annotation from a bottleneck label.

    Args:
        raw (str | None): Raw cell such as ``"MoE_fused (28.78%)"``.

    Returns:
        str | None: The bare category name, or ``None`` when empty.
    """
    if not raw:
        return None
    # ``MoE_fused (28.78%)`` -> ``MoE_fused``
    name = raw.split("(")[0].strip()
    return name or None


def extract_workload_summary(analysis_md_path: str | Path) -> dict[str, Any]:
    """Best-effort workload-level metrics from Executive Summary table.

    Args:
        analysis_md_path (str | Path): Path to a TraceLens ``analysis.md``.

    Returns:
        dict[str, Any]: Mapping with ``compute_pct`` / ``idle_pct`` /
            ``comm_pct`` / ``top_bottleneck`` keys; values are ``None`` when
            the file is missing or a metric cannot be parsed.
    """
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

    Args:
        analysis_md_text (str): The raw ``analysis.md`` text.

    Returns:
        dict[str, float]: Mapping of each canonical direction to its
            saturation percentage; missing/unparseable directions are ``0.0``.
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
    """Return the TraceLens output directory containing an ``analysis.md``.

    Args:
        analysis_md_path (Path): Path to an ``analysis.md`` file.

    Returns:
        Path: The parent directory holding the TraceLens artifacts.
    """
    # ``.../tracelens/analysis.md`` -> ``.../tracelens/``
    if analysis_md_path.name == "analysis.md" and analysis_md_path.parent.name:
        return analysis_md_path.parent
    return analysis_md_path.parent


def extract_top_kernel(analysis_md_path: str | Path) -> dict[str, Any] | None:
    """Return the highest ``percent_of_total`` operation across category metrics.

    Scans the sibling ``category_data/*_metrics.json`` files for the operation
    with the largest share of total GPU time.

    Args:
        analysis_md_path (str | Path): Path to a TraceLens ``analysis.md``.

    Returns:
        dict[str, Any] | None: The top kernel descriptor (``name`` / ``gpu_pct``
            / ``efficiency_pct`` / ``bound_type`` / ``category``), or ``None``
            when no category data is available or no named op was found.
    """
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


def _compute_within_and_gap(
    *,
    peak: float,
    achieved: float,
) -> tuple[float | None, float | None]:
    """Return ``(within_roofline_pct, gap_to_roofline_pct)``; both
    ``None`` when either input is non-positive so the renderer stays
    placeholder-friendly.

    Args:
        peak (float): The theoretical roofline ceiling (tok/s).
        achieved (float): The achieved throughput (tok/s).

    Returns:
        tuple[float | None, float | None]: The within-roofline and
            gap-to-roofline percentages, or ``(None, None)`` when either input
            is non-positive.
    """
    if peak <= 0 or achieved <= 0:
        return None, None
    within = round(achieved / peak * 100.0, 2)
    return within, round(100.0 - within, 2)


def build_roofline_snapshot(
    *,
    snapshot_id: int | None,
    ts: str,
    analysis_md_path: str,
    theoretical_peak_tok_per_sec: float = 0.0,
    achieved_tok_per_sec: float = 0.0,
    mem_ceiling_tok_per_sec: float = 0.0,
    cmp_ceiling_tok_per_sec: float = 0.0,
    bound_kind: str = "unknown",
) -> dict[str, Any]:
    """Materialise one side (baseline or latest) of the comparison.

    ``theoretical_peak_tok_per_sec`` is the primary decode roofline ceiling
    produced by ``roofline_ceiling.compute_roofline_breakdown_from_state``.
    The decomposition is also
    persisted via ``roofline_mem_ceiling_tok_per_sec`` /
    ``roofline_cmp_ceiling_tok_per_sec`` / ``roofline_bound_kind`` so
    reports can show which side dominated (``"memory"`` / ``"compute"``
    / ``"unknown"``). ``achieved_tok_per_sec`` is the benchmark
    ``output_throughput`` measured at the time the snapshot was recorded
    (baseline_tput for snapshot #1, current_best.tput afterwards).
    All five default to 0/"unknown" so legacy callers that don't pass
    them produce ``None`` in the derived percentage fields, preserving
    the existing placeholder-friendly renderer contract.

    Args:
        snapshot_id (int | None): Monotonic snapshot id, or ``None``.
        ts (str): ISO timestamp the snapshot was recorded.
        analysis_md_path (str): Path to the TraceLens ``analysis.md``; empty
            skips workload/kernel extraction.
        theoretical_peak_tok_per_sec (float): Two-sided roofline ceiling
            ``min(T_mem, T_cmp)``. Defaults to ``0.0``.
        achieved_tok_per_sec (float): Measured output throughput. Defaults to
            ``0.0``.
        mem_ceiling_tok_per_sec (float): Memory-bound ceiling. Defaults to
            ``0.0``.
        cmp_ceiling_tok_per_sec (float): Compute-bound ceiling. Defaults to
            ``0.0``.
        bound_kind (str): Dominant side (``"memory"`` / ``"compute"`` /
            ``"unknown"``). Defaults to ``"unknown"``.

    Returns:
        dict[str, Any]: The assembled snapshot dict; derived percentage fields
            are ``None`` when their inputs are non-positive.
    """
    within, gap = _compute_within_and_gap(
        peak=theoretical_peak_tok_per_sec,
        achieved=achieved_tok_per_sec,
    )
    snap: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "ts": ts or "",
        # 9fe4609 sidecar pointer — caller (SharedState.record_trace_analyze)
        # overwrites with the real path; stays empty for offline /
        # synthetic callers that only have an analysis.md.
        "kernel_roofline_path": "",
        "compute_pct": None,
        "idle_pct": None,
        "comm_pct": None,
        "top_bottleneck": None,
        "top_kernel": None,
        # Primary decode roofline ceiling plus its memory/compute sides.
        # All ``None`` when the ceiling is unavailable.
        "theoretical_peak_tok_per_sec": (
            float(theoretical_peak_tok_per_sec)
            if theoretical_peak_tok_per_sec > 0 else None
        ),
        "roofline_mem_ceiling_tok_per_sec": (
            float(mem_ceiling_tok_per_sec)
            if mem_ceiling_tok_per_sec > 0 else None
        ),
        "roofline_cmp_ceiling_tok_per_sec": (
            float(cmp_ceiling_tok_per_sec)
            if cmp_ceiling_tok_per_sec > 0 else None
        ),
        "roofline_bound_kind": (
            str(bound_kind) if bound_kind else "unknown"
        ),
        "achieved_tok_per_sec": (
            float(achieved_tok_per_sec)
            if achieved_tok_per_sec > 0 else None
        ),
        "within_roofline_pct": within,
        "gap_to_roofline_pct": gap,
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
    """Extract an integer snapshot id from a metadata dict.

    Args:
        meta (dict[str, Any]): Snapshot metadata.

    Returns:
        int | None: The first integer-valued id found, or ``None``.
    """
    for key in ("snapshot_id", "roofline_snapshot_id"):
        raw = meta.get(key)
        if isinstance(raw, int):
            return raw
    return None


def _num_delta(latest: float | None, baseline: float | None) -> float | None:
    """Return ``latest - baseline`` rounded to two decimals.

    Args:
        latest (float | None): The latest value.
        baseline (float | None): The baseline value.

    Returns:
        float | None: The rounded delta, or ``None`` if either input is
            ``None``.
    """
    if latest is None or baseline is None:
        return None
    return round(latest - baseline, 2)


def build_roofline_comparison_from_history(
    snapshots: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Build the ``roofline_comparison`` block straight from
    :attr:`SharedState.roofline_snapshots` (preferred entry point post
    PR #321).

    Each entry in ``snapshots`` is the already-parsed dict written by
    :meth:`SharedState.record_trace_analyze` (shape:
    ``snapshot_id`` / ``ts`` / ``analysis_md_path`` / ``trace_input``
    / ``compute_pct`` / ``idle_pct`` / ``comm_pct`` / ``top_bottleneck``
    / ``top_kernel``). Returns ``None`` when history is empty so the
    report-side caller can omit the section.

    The list is append-only with ``snapshots[0]`` carrying the baseline
    snapshot; ``snapshots[-1]`` carries the most recent watermark
    refresh. Same snapshot_id (i.e. history length 1) → single_snapshot
    mode; distinct ids → before_after with a populated ``delta`` block.

    Args:
        snapshots (list[dict[str, Any]] | None): Append-only snapshot history.

    Returns:
        dict[str, Any] | None: The ``roofline_comparison`` block, or ``None``
            when history is empty.
    """
    snapshots = list(snapshots or [])
    if not snapshots:
        return None
    baseline = dict(snapshots[0])
    latest = dict(snapshots[-1])
    base_id = baseline.get("snapshot_id")
    latest_id = latest.get("snapshot_id")
    same_snapshot = (
        isinstance(base_id, int)
        and isinstance(latest_id, int)
        and base_id == latest_id
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
            "compute_pct": _num_delta(
                latest.get("compute_pct"), baseline.get("compute_pct"),
            ),
            "idle_pct": _num_delta(
                latest.get("idle_pct"), baseline.get("idle_pct"),
            ),
            "comm_pct": _num_delta(
                latest.get("comm_pct"), baseline.get("comm_pct"),
            ),
            "top_kernel_efficiency_pct": _num_delta(lat_eff, base_eff),
            "within_roofline_pct": _num_delta(
                latest.get("within_roofline_pct"),
                baseline.get("within_roofline_pct"),
            ),
            # Dashboard surfaces gap-to-roofline shrinkage as the main
            # "X% within roofline" delta (negative = closer to ceiling).
            "gap_to_roofline_pct": _num_delta(
                latest.get("gap_to_roofline_pct"),
                baseline.get("gap_to_roofline_pct"),
            ),
        }
    return out


def build_roofline_comparison(
    baseline_meta: dict[str, Any],
    latest_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Build ``roofline_comparison`` block for ``final.json``.

    Args:
        baseline_meta (dict[str, Any]): Baseline snapshot metadata (must carry
            ``analysis_md_path`` / ``ts`` / ``snapshot_id``).
        latest_meta (dict[str, Any]): Latest snapshot metadata.

    Returns:
        dict[str, Any] | None: The comparison block, or ``None`` when neither
            side has an ``analysis_md_path``.
    """
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
            "within_roofline_pct": _num_delta(
                latest.get("within_roofline_pct"),
                baseline.get("within_roofline_pct"),
            ),
            # Dashboard surfaces gap-to-roofline shrinkage as the main
            # "X% within roofline" delta (negative = closer to ceiling).
            "gap_to_roofline_pct": _num_delta(
                latest.get("gap_to_roofline_pct"),
                baseline.get("gap_to_roofline_pct"),
            ),
        }
    return out


def _fmt_delta(val: float | None) -> str:
    """Format a signed delta cell with one decimal place.

    Args:
        val (float | None): The delta value, or ``None``.

    Returns:
        str: A signed string (e.g. ``"+1.2"``), or ``"—"`` when ``None``.
    """
    if val is None:
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}"


def _fmt_tput(v: float | None) -> str:
    """Format ``output_throughput`` cell; one decimal place + tok/s unit.

    Args:
        v (float | None): The throughput value (tok/s).

    Returns:
        str: The formatted cell, or ``"—"`` when missing/non-positive.
    """
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    return f"{float(v):.1f} tok/s"


def _fmt_pct_cell(v: float | None) -> str:
    """Format a percentage cell; ``—`` when missing.

    Args:
        v (float | None): The percentage value.

    Returns:
        str: The formatted cell, or ``"—"`` when not numeric.
    """
    if not isinstance(v, (int, float)):
        return "—"
    return f"{float(v):.1f}%"


def format_roofline_metrics_table(cmp: dict[str, Any]) -> list[str]:
    """Render the compact Base / Opt / Δ markdown table.

    The decode-roofline ceiling (``theoretical_peak_tok_per_sec``) is a
    session-level constant — same value on every history snapshot — so
    we render it once as a single full-width row above the Base/Opt
    columns. Achieved / within / gap then vary across snapshots and
    use the regular Base/Opt/Δ layout.

    Args:
        cmp (dict[str, Any]): A ``roofline_comparison`` block (``mode`` /
            ``baseline`` / ``latest`` / optional ``delta``).

    Returns:
        list[str]: The rendered markdown lines for the metrics table.
    """

    def cell(v: float | None) -> str:
        """Format a percentage value for a table cell.

        Args:
            v (float | None): The percentage value.

        Returns:
            str: The formatted cell, or ``"—"`` when not a float.
        """
        return f"{v:.1f}%" if isinstance(v, float) else "—"

    baseline = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    delta = cmp.get("delta") or {}
    mode = cmp.get("mode") or "single_snapshot"

    # Ceiling stays constant across snapshots; surface once before the
    # Base/Opt table. Fall back to whichever side carries it.
    peak = baseline.get("theoretical_peak_tok_per_sec")
    if not isinstance(peak, (int, float)) or peak <= 0:
        peak = latest.get("theoretical_peak_tok_per_sec")
    ceiling_lines: list[str] = []
    if isinstance(peak, (int, float)) and peak > 0:
        ceiling_lines.append(
            f"**Theoretical peak (decode memory-roofline ceiling):** "
            f"{float(peak):.1f} tok/s  "
            f"_(single-source ceiling; baseline / latest compared against it)_"
        )
        ceiling_lines.append("")

    lines: list[str] = list(ceiling_lines)
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
        lines.append(
            f"| Achieved output_throughput | "
            f"{_fmt_tput(snap.get('achieved_tok_per_sec'))} |"
        )
        lines.append(
            f"| Within roofline % | "
            f"{_fmt_pct_cell(snap.get('within_roofline_pct'))} |"
        )
        lines.append(
            f"| Gap to roofline % | "
            f"{_fmt_pct_cell(snap.get('gap_to_roofline_pct'))} |"
        )
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
    lines.append(
        f"| Achieved output_throughput | "
        f"{_fmt_tput(baseline.get('achieved_tok_per_sec'))} | "
        f"{_fmt_tput(latest.get('achieved_tok_per_sec'))} | — |"
    )
    lines.append(
        f"| Within roofline % | "
        f"{_fmt_pct_cell(baseline.get('within_roofline_pct'))} | "
        f"{_fmt_pct_cell(latest.get('within_roofline_pct'))} | "
        f"{_fmt_delta(delta.get('within_roofline_pct'))} |"
    )
    lines.append(
        f"| Gap to roofline % | "
        f"{_fmt_pct_cell(baseline.get('gap_to_roofline_pct'))} | "
        f"{_fmt_pct_cell(latest.get('gap_to_roofline_pct'))} | "
        f"{_fmt_delta(delta.get('gap_to_roofline_pct'))} |"
    )
    lines.append("")
    return lines
