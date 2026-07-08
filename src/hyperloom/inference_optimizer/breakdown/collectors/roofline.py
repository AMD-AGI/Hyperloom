# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from ._common import (
    _load_json_safe,
    _to_float,
    _to_int,
)



# §13b Roofline (single-path + watermark refresh model)
def collect_roofline(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Shape ``state.roofline_snapshots`` into the per-session roofline comparison the ``Roofline`` renderer expects.

    Returns ``[]`` when no snapshots exist or on parse failure
    (best-effort; errors recorded in ``warnings``).

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place on build
            failure).

    Returns:
        list[dict[str, Any]]: A single-element comparison list (baseline /
        latest / optional delta), or ``[]`` when no snapshots or on failure.
    """
    snapshots = state.get("roofline_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return []
    try:
        # Lazy import to avoid the orchestrator → breakdown → orchestrator circular path.
        from hyperloom.orchestrator.kernel.roofline_snapshot import (
            build_roofline_comparison_from_history,
        )

        cmp = build_roofline_comparison_from_history(snapshots)
    except Exception as exc:  # noqa: BLE001 — defensive
        warnings.append(
            f"collect_roofline: failed to build comparison from "
            f"roofline_snapshots ({len(snapshots)} entries): "
            f"{type(exc).__name__}: {exc}"
        )
        return []
    if not cmp:
        return []
    entry: dict[str, Any] = {
        "source_path": "state.json#roofline_snapshots",
        "mode": cmp.get("mode") or "single_snapshot",
        "baseline": cmp.get("baseline") or {},
        "latest": cmp.get("latest") or {},
    }
    delta = cmp.get("delta")
    if isinstance(delta, dict) and delta:
        entry["delta"] = delta
    return [entry]


# source_files map
_KERNEL_ROOFLINE_REL_PATH = "reports/kernel_roofline.json"


# Roofline — optimization-progress curve (Dashboard integration spec §2).
# Conservative achievable fraction of vendor-peak bandwidth Hyperloom
# targets (vendor specs are theoretical maxima); see spec §2.
DEFAULT_ROOFLINE_TARGET_RATIO = 0.70


def collect_roofline_progress(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the ``roofline_progress`` section feeding the optimization-progress chart (spec §2); never raises.

    Renamed from ``roofline`` to avoid clashing with :func:`collect_roofline`'s
    list-shaped key. Pure over ``state`` + ``manifest``. Output: ``trajectory[]``
    (baseline + KEEPs, ts-sorted), ceiling/target reference lines from the latest
    snapshot, headline current-best numbers, and a ``snapshots[]`` passthrough.

    Args:
        session_dir (Path): Absolute session root (kept for a uniform collector
            signature; the body is pure over state/manifest).
        state (dict[str, Any]): Parsed ``state.json``.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place on a
            trajectory / current-best mismatch).

    Returns:
        dict[str, Any]: The ``roofline_progress`` section (trajectory, ceiling
        / target reference lines, headline numbers, and normalized snapshots).
    """
    # Trajectory: baseline + each KEEP.
    baseline_tput = _to_float(state.get("baseline_tput")) or 0.0
    trajectory: list[dict[str, Any]] = []
    if baseline_tput > 0:
        trajectory.append(
            {
                "ts": str(manifest.get("created_at_utc") or ""),
                "tput": baseline_tput,
                "label": "baseline",
                "action": "baseline",
                "gain_pct": 0.0,
                "flags": "",
                "extra_envs": {},
            }
        )

    stack = state.get("optimization_stack") or []
    if isinstance(stack, list):
        # Already in promotion order; sort by ts as a guard against legacy prepends.
        ordered = sorted(
            (e for e in stack if isinstance(e, dict)),
            key=lambda e: str(e.get("ts") or ""),
        )
        for entry in ordered:
            tput = _to_float(entry.get("tput"))
            if tput is None or tput <= 0:
                continue
            gain_pct = ((tput - baseline_tput) / baseline_tput * 100.0) if baseline_tput > 0 else 0.0
            trajectory.append(
                {
                    "ts": str(entry.get("ts") or ""),
                    "tput": tput,
                    "label": str(entry.get("variant_name") or entry.get("action") or ""),
                    "action": str(entry.get("action") or ""),
                    "gain_pct": round(gain_pct, 4),
                    "flags": str(entry.get("candidate_extra_server_args") or ""),
                    "extra_envs": dict(entry.get("extra_envs") or {}),
                }
            )

    # Reference lines: ceiling + target.
    snapshots_raw = state.get("roofline_snapshots") or []
    snapshots: list[dict[str, Any]] = []
    if isinstance(snapshots_raw, list):
        for snap in snapshots_raw:
            if isinstance(snap, dict):
                snapshots.append(_normalize_roofline_snapshot(snap))

    # Use the LATEST snapshot (ceiling refines as the pipeline reruns).
    latest_snap = snapshots[-1] if snapshots else None
    ceiling_tok = _to_float(latest_snap.get("theoretical_peak_tok_per_sec")) if latest_snap else None
    ceiling_available = ceiling_tok is not None and ceiling_tok > 0
    target_tok = round(ceiling_tok * DEFAULT_ROOFLINE_TARGET_RATIO, 4) if ceiling_available else None

    # Headline numbers.
    current_best_tput = trajectory[-1]["tput"] if trajectory else 0.0
    cumulative_gain_pct = _to_float(state.get("cumulative_gain")) or 0.0
    pct_of_ceiling = (
        round(current_best_tput / ceiling_tok * 100.0, 4) if ceiling_available and current_best_tput > 0 else None
    )
    pct_of_target = (
        round(current_best_tput / target_tok * 100.0, 4)
        if (target_tok and target_tok > 0 and current_best_tput > 0)
        else None
    )

    # Scriptable/diffusion (xDiT) image models have no tok/s decode ceiling; their
    # roofline lives in the latency domain (ideal per-image compute floor vs the
    # measured e2e latency). Surface it through DEDICATED ms fields + a
    # ``ceiling_kind`` discriminator so the list page can render a roofline % for
    # image models without overloading the tok/s fields (which stay null).
    ceiling_kind = "throughput" if ceiling_available else "none"
    latency_ceiling_ms: float | None = None
    achieved_latency_ms: float | None = None
    latency_ceiling_available = False
    pct_of_latency_ceiling: float | None = None
    if not ceiling_available and latest_snap:
        ideal_ms = _to_float(latest_snap.get("roofline_ideal_ms"))
        measured_ms = _to_float(latest_snap.get("e2e_mean_ms"))
        if ideal_ms is not None and ideal_ms > 0 and measured_ms is not None and measured_ms > 0:
            latency_ceiling_ms = round(ideal_ms, 4)
            achieved_latency_ms = round(measured_ms, 4)
            latency_ceiling_available = True
            ceiling_kind = "latency"
            # Latency "closer to the ceiling" means the measured e2e approaches
            # the ideal floor, so the ratio is ideal/measured (higher = nearer,
            # mirroring the serving achieved/peak semantics; caps well under 100).
            pct_of_latency_ceiling = round(ideal_ms / measured_ms * 100.0, 4)

    out: dict[str, Any] = {
        "ceiling_kind": ceiling_kind,
        "ceiling_tok_per_sec": ceiling_tok,
        "target_tok_per_sec": target_tok,
        "ceiling_ratio_target": DEFAULT_ROOFLINE_TARGET_RATIO,
        "ceiling_available": ceiling_available,
        # Independent latency-domain ceiling for scriptable/diffusion models.
        "latency_ceiling_ms": latency_ceiling_ms,
        "achieved_latency_ms": achieved_latency_ms,
        "latency_ceiling_available": latency_ceiling_available,
        "current_best_pct_of_latency_ceiling": pct_of_latency_ceiling,
        "trajectory": trajectory,
        "baseline_tput": baseline_tput,
        "current_best_tput": current_best_tput,
        "cumulative_gain_pct": round(cumulative_gain_pct, 4),
        "current_best_pct_of_ceiling": pct_of_ceiling,
        "current_best_pct_of_target": pct_of_target,
        "roofline_failure_streak": _to_int(state.get("roofline_failure_streak")) or 0,
        "snapshots": snapshots,
    }
    if latest_snap:
        out["snapshot_top_bottleneck"] = str(latest_snap.get("top_bottleneck") or "")
        within = _to_float(latest_snap.get("within_roofline_pct"))
        if within is not None:
            out["snapshot_within_roofline_pct"] = within
        gap = _to_float(latest_snap.get("gap_to_roofline_pct"))
        if gap is not None:
            out["snapshot_gap_to_roofline_pct"] = gap

    # Sanity check: trajectory tail vs state.current_best.tput; divergence
    # means the stack wasn't fully promoted (resume mid-promotion).
    cb_tput = _to_float((state.get("current_best") or {}).get("tput"))
    if (
        cb_tput is not None
        and cb_tput > 0
        and current_best_tput > 0
        and abs(cb_tput - current_best_tput) / max(cb_tput, 1.0) > 0.001
    ):
        warnings.append(
            f"roofline.current_best_tput ({current_best_tput:.2f}) does not "
            f"match state.current_best.tput ({cb_tput:.2f}); the trajectory "
            f"may be missing a promotion event."
        )
    return out


def _normalize_roofline_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Coerce one ``state.roofline_snapshots[]`` entry to the schema.

    Args:
        snap (dict[str, Any]): A raw roofline snapshot from state.

    Returns:
        dict[str, Any]: The snapshot with stable types and a normalized
        ``top_kernel`` sub-dict.
    """
    top_kernel_raw = snap.get("top_kernel") or {}
    top_kernel: dict[str, Any] = {}
    if isinstance(top_kernel_raw, dict):
        top_kernel = {
            "name": str(top_kernel_raw.get("name") or ""),
            "bound_type": str(top_kernel_raw.get("bound_type") or ""),
            "efficiency_pct": _to_float(top_kernel_raw.get("efficiency_pct")) or 0.0,
            "gpu_pct": _to_float(top_kernel_raw.get("gpu_pct")) or 0.0,
        }
    return {
        "snapshot_id": _to_int(snap.get("snapshot_id")) or 0,
        "ts": str(snap.get("ts") or ""),
        "achieved_tok_per_sec": _to_float(snap.get("achieved_tok_per_sec")) or 0.0,
        "theoretical_peak_tok_per_sec": _to_float(snap.get("theoretical_peak_tok_per_sec")) or 0.0,
        "within_roofline_pct": _to_float(snap.get("within_roofline_pct")) or 0.0,
        "gap_to_roofline_pct": _to_float(snap.get("gap_to_roofline_pct")) or 0.0,
        # Scriptable/diffusion (xDiT) latency-roofline pair — the ms analogue of
        # the tok/s ceiling. Preserved (was previously dropped) so the progress
        # collector can surface an independent latency ceiling for image models.
        "e2e_mean_ms": _to_float(snap.get("e2e_mean_ms")),
        "roofline_ideal_ms": _to_float(snap.get("roofline_ideal_ms")),
        "roofline_bound_kind": str(snap.get("roofline_bound_kind") or "unknown"),
        "roofline_mem_ceiling_tok_per_sec": _to_float(snap.get("roofline_mem_ceiling_tok_per_sec")),
        "roofline_cmp_ceiling_tok_per_sec": _to_float(snap.get("roofline_cmp_ceiling_tok_per_sec")),
        "compute_pct": _to_float(snap.get("compute_pct")) or 0.0,
        "idle_pct": _to_float(snap.get("idle_pct")) or 0.0,
        "comm_pct": _to_float(snap.get("comm_pct")) or 0.0,
        "top_bottleneck": str(snap.get("top_bottleneck") or ""),
        "top_kernel": top_kernel,
        "analysis_md_path": str(snap.get("analysis_md_path") or ""),
        "kernel_roofline_path": str(snap.get("kernel_roofline_path") or ""),
        "trace_input": str(snap.get("trace_input") or ""),
    }


def collect_kernel_roofline(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``reports/kernel_roofline.json`` into the ``kernel_roofline`` section (spec §1).

    Missing file → ``{}`` (quiet); malformed → ``{}`` + warning; non-list
    ``kernels`` → ``[]``. Entries are type-coerced so upstream drift
    doesn't break consumers.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on
            malformed input).

    Returns:
        dict[str, Any]: The ``kernel_roofline`` section, or ``{}`` when the
        file is absent / not a JSON object.
    """
    path = session_dir / _KERNEL_ROOFLINE_REL_PATH
    if not path.exists():
        # Quiet on absence; most sessions never run the roofline pipeline.
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(f"kernel_roofline: {_KERNEL_ROOFLINE_REL_PATH} is not a JSON object")
        return {}

    raw_kernels = blob.get("kernels")
    if raw_kernels is None:
        kernels: list[dict[str, Any]] = []
    elif not isinstance(raw_kernels, list):
        warnings.append("kernel_roofline.kernels is not a list; dropping entries")
        kernels = []
    else:
        kernels = [_normalize_kernel_roofline_entry(k) for k in raw_kernels if isinstance(k, dict)]

    out: dict[str, Any] = {
        "schema_version": _to_int(blob.get("schema_version")) or 1,
        "source": str(blob.get("source") or ""),
        "analysis_md_path": str(blob.get("analysis_md_path") or ""),
        "kernel_candidates_path": str(blob.get("kernel_candidates_path") or ""),
        "trace_input": str(blob.get("trace_input") or ""),
        "trace_input_type": str(blob.get("trace_input_type") or ""),
        "kernels": kernels,
    }
    return out


def _normalize_kernel_roofline_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one kernel roofline entry to the schema shape with stable types.

    Args:
        raw (dict[str, Any]): A raw kernel entry from ``kernel_roofline.json``.

    Returns:
        dict[str, Any]: The entry with all fields coerced to stable types.
    """
    return {
        "kernel_id": str(raw.get("kernel_id") or ""),
        "name": str(raw.get("name") or ""),
        "source_file": str(raw.get("source_file") or ""),
        "kernel_category": str(raw.get("kernel_category") or ""),
        "bound_type": str(raw.get("bound_type") or ""),
        "arithmetic_intensity": _to_float(raw.get("arithmetic_intensity")) or 0.0,
        "flops_per_byte": _to_float(raw.get("flops_per_byte")) or 0.0,
        "efficiency_percent": _to_float(raw.get("efficiency_percent")) or 0.0,
        "gpu_pct": _to_float(raw.get("gpu_pct")) or 0.0,
        "call_count": _to_int(raw.get("call_count")) or 0,
        "duration_us": _to_float(raw.get("duration_us")) or 0.0,
        "reusable_native_kernel": bool(raw.get("reusable_native_kernel")),
        "rocprof_roofline": raw.get("rocprof_roofline"),
    }

