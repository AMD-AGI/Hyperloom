# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_RooflineStateMixin`` — trace-analyze and roofline-snapshot recording for :class:`..shared_state.SharedState`."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_int
from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)


def first_positive_tput(d: Any) -> float:
    """Return the first positive ``tput``/``output_throughput`` from a dict."""
    src = d if isinstance(d, dict) else {}
    for key in ("tput", "output_throughput"):
        val = src.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return 0.0


# How many hot / skipped kernels ``record_trace_analyze`` keeps in the trace summary (matches the ``*_top15`` field
# names).
_TRACE_HOT_KERNEL_TOP_N = 15

# A projected hot-kernel row carries roofline evidence when any of these is populated.
_ROOFLINE_SIGNAL_KEYS: tuple[str, ...] = (
    "bound_type",
    "arithmetic_intensity",
    "flops_per_byte",
    "efficiency_percent",
    "compute_utilization_pct",
    "bandwidth_utilization_pct",
    "suggestion",
    "roofline_name",
)


def _saturation_delta(sat: dict[str, Any], prev_sat: dict[str, Any], key: str) -> float | None:
    current, previous = sat.get(key), prev_sat.get(key)
    if current is None or previous is None:
        return None
    return round(float(current) - float(previous), 2)


def _skipped_kernels_top(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The hottest skipped (non-routable) candidates, so the LLM sees the operators it cannot optimize."""
    skipped = result.get("skipped_kernels")
    skipped_rows = sorted(
        (e for e in skipped if isinstance(e, dict)) if isinstance(skipped, list) else (),
        key=lambda e: float(e.get("gpu_pct") or 0.0),
        reverse=True,
    )
    return [
        {
            "kernel_id": e.get("kernel_id"),
            "name": e.get("name"),
            "skip_reason": e.get("skip_reason") or "",
            "gpu_pct": e.get("gpu_pct"),
        }
        for e in skipped_rows[:_TRACE_HOT_KERNEL_TOP_N]
    ]


def _hot_kernel_row(entry: dict[str, Any]) -> dict[str, Any]:
    arithmetic_intensity = entry.get("arithmetic_intensity")
    if arithmetic_intensity is None:
        arithmetic_intensity = entry.get("flops_per_byte")
    efficiency_percent = entry.get("efficiency_percent")
    if efficiency_percent is None:
        efficiency_percent = entry.get("efficiency_pct")
    return {
        "kernel_id": entry.get("kernel_id"),
        "name": entry.get("name"),
        # TraceLens kernel_category bucket ("" when absent).
        "kernel_category": entry.get("kernel_category") or "",
        "gpu_pct": entry.get("gpu_pct"),
        "call_count": entry.get("call_count"),
        "duration_us": entry.get("duration_us"),
        "shapes": entry.get("shapes") or [],
        "tracelens_pitem_rank": entry.get("tracelens_pitem_rank"),
        "impact_score": entry.get("impact_score"),
        "bottleneck": entry.get("bottleneck"),
        "bound_type": entry.get("bound_type"),
        "arithmetic_intensity": arithmetic_intensity,
        "flops_per_byte": entry.get("flops_per_byte"),
        "efficiency_percent": efficiency_percent,
        "compute_utilization_pct": entry.get("compute_utilization_pct"),
        "bandwidth_utilization_pct": entry.get("bandwidth_utilization_pct"),
        "suggestion": entry.get("suggestion") or "",
        "roofline_name": entry.get("roofline_name"),
        "source_file": entry.get("source_file"),
        "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
        "kernel_contract": entry.get("kernel_contract"),
        "is_multigpu": entry.get("is_multigpu") is True,
        # Names the deterministic extractor the row came from.
        "candidate_source": entry.get("candidate_source") or "",
        "recommended_backends": entry.get("recommended_backends") or [],
        "recommended_actions": entry.get("recommended_actions") or [],
        # Vendor-playbook fields; see _vendor_operator_playbooks.py.
        "patch_strategy": entry.get("patch_strategy") or "",
        "vendor_playbook_group_id": entry.get("vendor_playbook_group_id") or "",
        "vendor_playbook_aggregate_gpu_pct": entry.get("vendor_playbook_aggregate_gpu_pct"),
        "vendor_playbook_min_gpu_pct_floor": entry.get("vendor_playbook_min_gpu_pct_floor"),
    }


# Session-level kernel-roofline report the analyzer writes for a non-close run; read back when the trace_analyze
# envelope arrives without its payload keys.
_DEFAULT_ROOFLINE_REPORT_NAME = "kernel_roofline_current.json"

# roofline_snapshots history cap (record_trace_analyze).
_ROOFLINE_SNAPSHOTS_CAP = 50


class _RooflineStateMixin:
    def _resolve_baseline_achieved_tput(self) -> float:
        """Baseline throughput for a baseline-arm roofline snapshot."""
        if isinstance(self.baseline_tput, (int, float)) and self.baseline_tput > 0:
            return float(self.baseline_tput)
        return first_positive_tput(self.last_baseline)

    def _resolve_current_best_achieved_tput(self) -> float:
        """Optimized-arm throughput for a current_best roofline snapshot."""
        return first_positive_tput(self.current_best)

    def _roofline_throughput_unit(self) -> str:
        """Return the throughput unit for roofline snapshots of this workload."""
        from hyperloom.inference_optimizer import framework_registry

        framework = str(getattr(self, "framework", "") or "").strip().lower()
        return framework_registry.throughput_unit(framework)

    def record_trace_analyze(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Write ``last_trace_analyze`` (single writer); ``roofline_snapshot_id`` increments from the previous nested value, restarting from 1 when the cache was cleared."""
        if not isinstance(result, dict):
            return
        result = dict(result)
        # Both callers build the payload as a dict.
        task = payload
        artifacts = result.get("artifact_paths")
        if not isinstance(artifacts, dict):
            artifacts = {}
        candidates_path, kernel_roofline_path = self._recover_from_roofline_report(
            result,
            task,
            candidates_path=result.get("candidates_path") or artifacts.get("kernel_candidates") or "",
            kernel_roofline_path=result.get("kernel_roofline_path") or artifacts.get("kernel_roofline") or "",
        )
        steady_state_trace = result.get("steady_state_trace") or artifacts.get("tracelens_steady_state_trace") or ""
        summary, kernel_roofline, reusable_ids = self._build_hot_kernel_summaries(result)
        warning_rows = result.get("trace_health_warnings")
        warning_rows = warning_rows if isinstance(warning_rows, list) else []
        task_groups = result.get("task_groups")
        # Monotonic snapshot counter: read previous value + 1.
        recorded = self.last_trace_analyze if isinstance(self.last_trace_analyze, dict) else {}
        analysis_md_path = result.get("trace_report_path") or ""
        analysis_md_text = ""
        if analysis_md_path:
            # Stored verbatim; the prompt path strips base64 data-URLs.
            with suppress(OSError, ValueError):
                analysis_md_text = Path(analysis_md_path).read_text(encoding="utf-8", errors="replace")
        self.last_trace_analyze = {
            "trace_input": str(task.get("trace_input") or task.get("trace_dir") or ""),
            "steady_state_trace": str(steady_state_trace),
            "candidates_path": str(candidates_path),
            "kernel_roofline_path": str(kernel_roofline_path),
            "hot_kernels_top15": summary,
            "kernel_roofline_top15": kernel_roofline,
            "skipped_kernels_top": _skipped_kernels_top(result),
            "task_groups": task_groups if isinstance(task_groups, list) else [],
            "reusable_native_kernel_ids": reusable_ids,
            "trace_health_warnings": [dict(e) for e in warning_rows if isinstance(e, dict) and e.get("code")],
            "analysis_md_path": str(analysis_md_path),
            "analysis_md_text": analysis_md_text,
            "roofline_snapshot_id": to_int(recorded.get("roofline_snapshot_id"), 0) + 1,
            "roofline_baseline_gain_at_snapshot": float(self.cumulative_gain_validated),
            "ts": now_iso(),
        }

        self._append_roofline_snapshot_history(payload)

    def _recover_from_roofline_report(
        self,
        result: dict[str, Any],
        task: dict[str, Any],
        *,
        candidates_path: Any,
        kernel_roofline_path: Any,
    ) -> tuple[Any, Any]:
        """Fill what the envelope lost from the session roofline report; the candidates and roofline paths after it."""
        disk = self._read_session_roofline_report(task, kernel_roofline_path)
        if not disk:
            return candidates_path, kernel_roofline_path
        report = disk["payload"]
        kernel_roofline_path = kernel_roofline_path or str(disk["path"])
        candidates_path = candidates_path or str(report.get("kernel_candidates_path") or "")
        result["trace_report_path"] = result.get("trace_report_path") or str(report.get("analysis_md_path") or "")
        if not (result.get("hot_kernels") or result.get("hot_kernels_top15")):
            result["hot_kernels"] = disk["kernels"]
            log.warning(
                "record_trace_analyze: envelope carried no hot kernels; "
                "recovered %d from %s. The analysis succeeded — the result "
                "envelope lost its payload in transit.",
                len(disk["kernels"]),
                disk["path"],
            )
        return candidates_path, kernel_roofline_path

    def _read_session_roofline_report(
        self,
        payload: dict[str, Any],
        kernel_roofline_path: str,
    ) -> "dict[str, Any] | None":
        """Read the session-level kernel-roofline report written by TraceLens."""
        path = Path(kernel_roofline_path) if kernel_roofline_path else None
        if path is None:
            session_dir = getattr(self, "_session_dir", None)
            if not session_dir:
                return None
            name = str((payload or {}).get("roofline_output_name") or "").strip()
            path = Path(session_dir) / "reports" / (name or _DEFAULT_ROOFLINE_REPORT_NAME)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(report, dict):
            return None
        rows = [row for row in (report.get("kernels") or []) if isinstance(row, dict)]
        rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
        return {"path": path, "payload": report, "kernels": rows}

    def _build_hot_kernel_summaries(
        self,
        result: dict[str, Any],
    ) -> "tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]":
        """Build ``(summary, kernel_roofline, reusable_ids)`` from the top-N hot kernels."""
        hot = result.get("hot_kernels")
        summary: list[dict[str, Any]] = []
        kernel_roofline: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        for entry in hot[:_TRACE_HOT_KERNEL_TOP_N] if isinstance(hot, list) else []:
            if not isinstance(entry, dict):
                continue
            row = _hot_kernel_row(entry)
            summary.append(row)
            if any(row.get(key) not in (None, "", []) for key in _ROOFLINE_SIGNAL_KEYS):
                kernel_roofline.append(dict(row))
            if row["reusable_native_kernel"] and row["kernel_id"]:
                reusable_ids.append(str(row["kernel_id"]))
        return summary, kernel_roofline, reusable_ids

    def _append_roofline_snapshot_history(self, payload: dict[str, Any]) -> None:
        """Append a compact roofline snapshot of ``last_trace_analyze`` for report-side comparison.

        Runs after the canonical write above, so a failure here cannot undo it.
        """
        from hyperloom.inference_optimizer.roofline_ceiling import compute_roofline_breakdown_from_state
        from hyperloom.inference_optimizer.roofline_snapshot import (
            RooflineReading,
            attach_perfmodel_breakdown,
            build_roofline_snapshot,
        )

        latest = self.last_trace_analyze
        snapshot_id = latest["roofline_snapshot_id"]
        analysis_md_path = str(latest["analysis_md_path"])
        kernel_roofline_path = str(latest["kernel_roofline_path"])
        # Resolve which arm this snapshot measures first so the ceiling is anchored to the same arm as achieved.
        forced_arm = str((payload or {}).get("roofline_arm") or "").strip()
        # Unknown arm values fall through to current_best inference; warn.
        if forced_arm and forced_arm not in ("baseline", "current_best"):
            log.warning(
                "record_trace_analyze: ignoring unknown roofline_arm=%r; falling back to current_best inference",
                forced_arm,
            )
            forced_arm = ""
        cb_tput = self.current_best.get("tput")
        snapshot_arm = forced_arm or ("current_best" if first_positive_tput({"tput": cb_tput}) else "baseline")
        achieved_tput = (
            self._resolve_current_best_achieved_tput()
            if snapshot_arm == "current_best"
            else self._resolve_baseline_achieved_tput()
        )
        fw = str(getattr(self, "framework", "") or "")
        history_entry = build_roofline_snapshot(
            snapshot_id=snapshot_id,
            ts=latest["ts"],
            analysis_md_path=analysis_md_path,
            reading=RooflineReading(
                # Primary decode ceiling plus memory/compute sides (PerfModel bottom-up).
                ceiling=compute_roofline_breakdown_from_state(self, arm=snapshot_arm),
                achieved_tok_per_sec=achieved_tput,
                throughput_unit=self._roofline_throughput_unit(),
            ),
            framework=fw,
        )
        # Per-op PerfModel breakdown for dashboard visualization (serving runs only).
        attach_perfmodel_breakdown(history_entry, self, arm=snapshot_arm)
        history_entry["trace_input"] = str(latest["trace_input"])
        history_entry["macro_cycle"] = int(getattr(self, "macro_cycle", 0) or 0)
        history_entry["analysis_md_path"] = analysis_md_path
        # Sidecar artifact pointer for per-kernel roofline data.
        history_entry["kernel_roofline_path"] = kernel_roofline_path
        if not isinstance(self.roofline_snapshots, list):
            self.roofline_snapshots = []
        self.roofline_snapshots.append(history_entry)
        self._record_bottleneck_saturation(history_entry, snapshot_id)
        # R3: retire the pending switch once the top bottleneck has drifted.
        self.maybe_clear_bottleneck_switch_on_drift(
            str(history_entry.get("top_bottleneck") or ""),
        )
        if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
            # Always keep snapshot #1 as the report's baseline anchor.
            base = self.roofline_snapshots[0]
            tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1) :]
            self.roofline_snapshots = [base, *tail]

    def _record_bottleneck_saturation(self, history_entry: dict[str, Any], snapshot_id: Any) -> None:
        """Stamp the saturation of this snapshot's dominant direction and how it shifted since the previous cycle."""
        from hyperloom.inference_optimizer.roofline_snapshot import direction_saturation

        sat = direction_saturation(history_entry)
        hint = sat.get("domain_hint") if isinstance(sat.get("domain_hint"), dict) else {}
        domain_key = str(hint.get("domain") or sat.get("direction") or "unknown")
        latest_cycle = int(history_entry.get("macro_cycle") or 0)
        prev_cycle_snapshot: dict[str, Any] = {}
        for prev in reversed(self.roofline_snapshots[:-1]):
            if int(prev.get("macro_cycle") or 0) < latest_cycle:
                prev_cycle_snapshot = prev
                break
        if not prev_cycle_snapshot and len(self.roofline_snapshots) >= 2:
            prev_cycle_snapshot = self.roofline_snapshots[-2]
        prev_sat = direction_saturation(prev_cycle_snapshot) if prev_cycle_snapshot else {}
        if not isinstance(self.saturated_directions, dict):
            self.saturated_directions = {}
        self.saturated_directions[domain_key] = {
            **sat,
            "domain": domain_key,
            "tag": str(hint.get("tag") or ""),
            "macro_cycle": latest_cycle,
            "snapshot_id": snapshot_id,
            "top_bottleneck": history_entry.get("top_bottleneck"),
        }
        self.bottleneck_shift = {
            "from": prev_sat.get("direction", ""),
            "to": sat.get("direction", ""),
            "from_domain": (prev_sat.get("domain_hint") or {}).get("domain", ""),
            "to_domain": domain_key,
            "prev_cycle": prev_cycle_snapshot.get("macro_cycle"),
            "cycle": latest_cycle,
            "within_delta": _saturation_delta(sat, prev_sat, "within_pct"),
            "gap_delta": _saturation_delta(sat, prev_sat, "gap_pct"),
            "bound_kind_changed": (
                bool(prev_sat) and str(prev_sat.get("bound_kind") or "") != str(sat.get("bound_kind") or "")
            ),
            "current": sat,
            "previous": prev_sat,
        }
