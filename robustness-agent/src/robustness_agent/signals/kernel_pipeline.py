# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-pipeline / external-backend health signals (F1-F5).

* **F1 ``ray_pending_starvation``** — non-zero ``ray status`` Pending across
  ``min_pending_ticks`` consecutive ticks (cluster quota ledger wedged).
* **F2 ``geak_budget_starvation``** — same kernel_id's GEAK attempt SIGTERM'd across
  ``min_geak_sigterm_attempts`` rows; budget too short for ``select_patch``.
* **F4 ``cursor_auth_storm``** — ``backend=cursor`` + 401 marker across
  ``min_cursor_401_hits`` rows (``--backends cursor`` without ``CURSOR_API_KEY``).
* **F5 ``kernel_opt_no_progress``** — ``min_kernels_with_no_progress`` kernel_ids land
  all backend attempts on PARTIAL/REVERT (no KEEP); prune kernel_opt toward params/sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



# "GEAK budget killed the run" markers in report_text (matched case-insensitive).
_GEAK_SIGTERM_MARKERS: tuple[str, ...] = (
    "sigterm",
    "signal 15",
    "budget exhausted",
    "timed out before select_patch",
    "killed by deadline",
)

# "cursor authentication failed" markers (401 / unauthorized).
_CURSOR_AUTH_MARKERS: tuple[str, ...] = (
    "401",
    "unauthorized",
    "authentication failed",
    "primus.00009",
)


@dataclass
class KernelPipelineConfig:
    """Tunables for :func:`evaluate_kernel_pipeline_signals`."""

    # F1 — pending count above this for N consecutive ticks → fire.
    pending_count_threshold: int = 1
    min_pending_ticks: int = 3
    # F2 — same kernel_id has GEAK backend SIGTERM'd this many times.
    min_geak_sigterm_attempts: int = 2
    # F4 — cursor 401 marker count threshold.
    min_cursor_401_hits: int = 3
    # F5 — kernel_ids with no PARTIAL→KEEP progression across the
    # recent oob_attempts window.
    min_kernels_with_no_progress: int = 3


# ---------------------------------------------------------------------------
# F1 — Ray pending starvation (stateful — counts consecutive ticks)
# ---------------------------------------------------------------------------

class RayPendingDetector:
    """Track ``ray status`` pending across ticks; ``min_pending_ticks`` consecutive ticks above threshold = dispatcher wedged."""

    def __init__(
        self,
        config: KernelPipelineConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore persisted pending counters.

        Args:
            config (KernelPipelineConfig | None): Tunables; defaults to
                :class:`KernelPipelineConfig` when ``None``.
            state_view (DetectorStateView | None): Disk-backed state view used
                to load/persist ``consecutive_hits`` and ``last_pending``.
        """
        self._config = config or KernelPipelineConfig()
        self._state_view = state_view
        # Disk-backed counter — see GpuLeakDetector for the same
        # reasoning. Without this, F1 (≥3 consecutive ticks) cannot
        # fire under the subprocess-per-tick transport.
        loaded = state_view.load() if state_view is not None else {}
        try:
            self._consecutive_hits: int = max(
                0, int(loaded.get("consecutive_hits", 0))
            )
        except (TypeError, ValueError):
            self._consecutive_hits = 0
        try:
            self._last_pending: int = max(
                0, int(loaded.get("last_pending", 0))
            )
        except (TypeError, ValueError):
            self._last_pending = 0

    def _persist(self) -> None:
        """Write the pending counters to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({
            "consecutive_hits": self._consecutive_hits,
            "last_pending": self._last_pending,
        })

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        """Advance the pending streak and fire F1 once it crosses threshold.

        Resets the streak when Ray data is missing, the head is unhealthy, or
        the pending count is at/below the configured threshold.

        Args:
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data including ``local_ray``.

        Returns:
            list[Symptom]: A single ``ray_pending_starvation`` symptom once the
                consecutive-tick threshold is crossed, otherwise an empty list.
        """
        ray_info = data.local_ray
        if not isinstance(ray_info, dict) or not ray_info:
            # No Ray data this tick → don't accumulate.
            self._consecutive_hits = 0
            self._persist()
            return []
        if not ray_info.get("healthy"):
            # Ray head dead is its own signal (A6); we don't pile
            # pending-starvation on top.
            self._consecutive_hits = 0
            self._persist()
            return []
        pending = ray_info.get("pending_tasks")
        if not isinstance(pending, int) or pending <= self._config.pending_count_threshold:
            self._consecutive_hits = 0
            self._last_pending = 0
            self._persist()
            return []
        self._consecutive_hits += 1
        self._last_pending = pending
        self._persist()
        if self._consecutive_hits < self._config.min_pending_ticks:
            return []
        return [
            Symptom(
                name="ray_pending_starvation",
                severity=SymptomSeverity.HIGH,
                summary=(
                    f"ray reports {pending} pending tasks for "
                    f"{self._consecutive_hits} consecutive tick(s) "
                    f"(>= {self._config.min_pending_ticks}); cluster "
                    f"quota likely wedged"
                ),
                evidence={
                    "pending_tasks": pending,
                    "consecutive_ticks": self._consecutive_hits,
                    "threshold": self._config.min_pending_ticks,
                },
                subject={},
                source="local",
                suggestion=(
                    "prune_branch(kernel_opt) until ray clears; "
                    "shrink concurrency or wait out the cluster quota anomaly"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# F2 — GEAK budget starvation
# ---------------------------------------------------------------------------

def _geak_budget_symptoms(
    data: SourceData, cfg: KernelPipelineConfig,
) -> list[Symptom]:
    """F2: fire ``geak_budget_starvation`` for kernels whose GEAK runs SIGTERM.

    Args:
        data (SourceData): Collected source data including the decision-audit
            ``oob_attempts``.
        cfg (KernelPipelineConfig): Tunables (provides the SIGTERM-attempt
            threshold).

    Returns:
        list[Symptom]: One ``geak_budget_starvation`` symptom per offending
            kernel, possibly empty.
    """
    audit = data.local_decision_audit
    if not isinstance(audit, dict):
        return []
    attempts = audit.get("oob_attempts") or []
    if not isinstance(attempts, list):
        return []
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for entry in attempts:
        if not isinstance(entry, dict):
            continue
        backend = str(entry.get("backend") or "").lower()
        if "geak" not in backend:
            continue
        report = str(entry.get("report_text") or "").lower()
        if not any(m in report for m in _GEAK_SIGTERM_MARKERS):
            continue
        kernel_id = str(entry.get("kernel_id") or "unknown")
        by_kernel.setdefault(kernel_id, []).append(entry)
    out: list[Symptom] = []
    for kernel_id, rows in by_kernel.items():
        if len(rows) < cfg.min_geak_sigterm_attempts:
            continue
        out.append(
            Symptom(
                name="geak_budget_starvation",
                severity=SymptomSeverity.HIGH,
                summary=(
                    f"GEAK backend was SIGTERM'd on kernel_id={kernel_id!r} "
                    f"in {len(rows)} consecutive attempts "
                    f"(>= {cfg.min_geak_sigterm_attempts}); "
                    f"select_patch never completes within budget"
                ),
                evidence={
                    "kernel_id": kernel_id,
                    "attempt_count": len(rows),
                    "threshold": cfg.min_geak_sigterm_attempts,
                    "last_report_head": (
                        str(rows[-1].get("report_text") or "")[:200]
                    ),
                },
                subject={"kernel_id": kernel_id},
                source="local",
                suggestion=(
                    "extend GEAK budget for this kernel OR prune it from "
                    "the kernel_opt rotation — current budget cannot "
                    "produce a verdict"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# F4 — Cursor 401 storm
# ---------------------------------------------------------------------------

def _cursor_auth_storm_symptoms(
    data: SourceData, cfg: KernelPipelineConfig,
) -> list[Symptom]:
    """F4: fire ``cursor_auth_storm`` when the cursor backend hits repeated 401s.

    Args:
        data (SourceData): Collected source data including the decision-audit
            ``oob_attempts``.
        cfg (KernelPipelineConfig): Tunables (provides the 401-hit threshold).

    Returns:
        list[Symptom]: A one-element list with the ``cursor_auth_storm`` symptom
            once the threshold is reached, otherwise an empty list.
    """
    audit = data.local_decision_audit
    if not isinstance(audit, dict):
        return []
    attempts = audit.get("oob_attempts") or []
    if not isinstance(attempts, list):
        return []
    hits: list[dict[str, Any]] = []
    for entry in attempts:
        if not isinstance(entry, dict):
            continue
        backend = str(entry.get("backend") or "").lower()
        if "cursor" not in backend:
            continue
        report = str(entry.get("report_text") or "").lower()
        if not any(m in report for m in _CURSOR_AUTH_MARKERS):
            continue
        hits.append(entry)
    if len(hits) < cfg.min_cursor_401_hits:
        return []
    return [
        Symptom(
            name="cursor_auth_storm",
            severity=SymptomSeverity.MEDIUM,
            summary=(
                f"cursor backend returned 401/auth-failed on "
                f"{len(hits)} attempts (>= {cfg.min_cursor_401_hits}); "
                f"CURSOR_API_KEY likely missing or revoked"
            ),
            evidence={
                "hit_count": len(hits),
                "threshold": cfg.min_cursor_401_hits,
                "kernel_ids": list({
                    str(h.get("kernel_id") or "") for h in hits
                })[:5],
            },
            subject={"backend": "cursor"},
            source="local",
            suggestion=(
                "drop cursor from --backends until CURSOR_API_KEY is "
                "rotated; the auto-skip path only kicks in when the "
                "env var is unset, not when explicit user override is in play"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# F5 — Kernel-opt no-progress
# ---------------------------------------------------------------------------

def _kernel_opt_no_progress_symptoms(
    data: SourceData, cfg: KernelPipelineConfig,
) -> list[Symptom]:
    """Identify kernels where every backend attempt landed at
    PARTIAL/REVERT (no KEEP) across the recent attempt window.

    Only kernels with at least two distinct backends attempted count, so
    one-shot kernels that haven't had time to fail are not flagged.

    Args:
        data (SourceData): Collected source data including the decision-audit
            ``oob_attempts`` and ``recent_integrate``.
        cfg (KernelPipelineConfig): Tunables (provides the no-progress kernel
            count threshold).

    Returns:
        list[Symptom]: A one-element list with the ``kernel_opt_no_progress``
            symptom when enough kernels show no progress, otherwise an empty
            list.
    """
    audit = data.local_decision_audit
    if not isinstance(audit, dict):
        return []
    attempts = audit.get("oob_attempts") or []
    integrate = audit.get("recent_integrate") or []
    if not isinstance(attempts, list) and not isinstance(integrate, list):
        return []

    # Roll up per kernel from oob_attempts + recent_integrate; ``has_keep`` only on a positive speedup OR a KEEP decision.
    rollups: dict[str, dict[str, Any]] = {}
    if isinstance(attempts, list):
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            kernel_id = str(entry.get("kernel_id") or "")
            if not kernel_id:
                continue
            roll = rollups.setdefault(kernel_id, {
                "kernel_id": kernel_id,
                "backends": set(),
                "has_keep": False,
                "attempt_count": 0,
            })
            roll["attempt_count"] += 1
            backend = str(entry.get("backend") or "").lower()
            if backend:
                roll["backends"].add(backend)
            ms = entry.get("microbench_speedup")
            if isinstance(ms, (int, float)) and ms >= 1.2:
                roll["has_keep"] = True
    if isinstance(integrate, list):
        for entry in integrate:
            if not isinstance(entry, dict):
                continue
            kernel_id = str(entry.get("kernel_id") or "")
            if not kernel_id:
                continue
            roll = rollups.setdefault(kernel_id, {
                "kernel_id": kernel_id,
                "backends": set(),
                "has_keep": False,
                "attempt_count": 0,
            })
            if str(entry.get("decision") or "") == "KEEP":
                roll["has_keep"] = True

    # Only count kernels with at least 2 distinct backends attempted —
    # one-shot kernels haven't had time to fail across the pipeline.
    bad_kernels = [
        roll for roll in rollups.values()
        if not roll["has_keep"] and len(roll["backends"]) >= 2
    ]
    if len(bad_kernels) < cfg.min_kernels_with_no_progress:
        return []
    return [
        Symptom(
            name="kernel_opt_no_progress",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"{len(bad_kernels)} kernel_ids have all backend attempts "
                f"land on PARTIAL/REVERT with no KEEP "
                f"(>= {cfg.min_kernels_with_no_progress}); the kernel_opt "
                f"pipeline cannot optimise this model"
            ),
            evidence={
                "kernel_count": len(bad_kernels),
                "threshold": cfg.min_kernels_with_no_progress,
                "kernels": [
                    {"kernel_id": roll["kernel_id"],
                     "backends": sorted(roll["backends"]),
                     "attempt_count": roll["attempt_count"]}
                    for roll in bad_kernels[:5]
                ],
            },
            subject={},
            source="local",
            suggestion=(
                "prune_branch(kernel_opt); the budget will land further "
                "wins on params/sweep instead"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Public entry point — module-level helper (stateful F1 lives in the class)
# ---------------------------------------------------------------------------

def evaluate_kernel_pipeline_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: KernelPipelineConfig | None = None,
) -> list[Symptom]:
    """Stateless slice of F (F2 / F4 / F5); F1 is stateful in :class:`RayPendingDetector`."""
    cfg = config or KernelPipelineConfig()
    out: list[Symptom] = []
    out.extend(_geak_budget_symptoms(data, cfg))
    out.extend(_cursor_auth_storm_symptoms(data, cfg))
    out.extend(_kernel_opt_no_progress_symptoms(data, cfg))
    return out


__all__ = [
    "KernelPipelineConfig",
    "RayPendingDetector",
    "evaluate_kernel_pipeline_signals",
]
