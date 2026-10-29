# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL_AGENT phase handler for the fusion, GEMM, and GEAK lanes."""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging as _logging
import sqlite3
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from . import machine_state as _phase_state
from .kernel_round_budget import controller_round_stop, measurement_round_sec
from hyperloom.common.coerce import first_str, to_int
from hyperloom.common.env import env_bool, env_float
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_NO_PROMOTE
from hyperloom.inference_optimizer.breakdown.recorder import tool_versions
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    ROUTE_FORGE,
    ROUTE_GEAK,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import (
    FusionRun,
    GeakRebenchAttempt,
    GemmTuningRun,
    KernelExit,
    KernelReprofile,
    KernelRewriteRun,
    LaneRunFields,
)
from ..actions.stop_attribution import stopped_by_the_run_class
from ..state.analysis_failures import analysis_skip_reason, remembered_trace_failure
from ..state.task_registry import TERMINAL_STATES, TaskNotFound, TaskSpec
from ..bus.message_bus import Message
from ..kernel import kernel_decisions
from ..collaborator import CoordinatorCollaborator
from .kernel_geak_dispatch import _GeakDispatchMixin
from .kernel_geak_promotion import _GeakPromotionMixin, _covered_step
from .kernel_gemm_tuning import _GemmTuningMixin, _first_text

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import KernelEventRecorder
    from .machine import Transition

log = _logging.getLogger(__name__)

# What the phase, not the Controller subprocess, decides about a rewrite-controller attempt.
_CONTROLLER_ATTEMPT_FACTS = ("macro_cycle", "handoff_dir", "budget_minutes", "hard_timeout_sec")

ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline

# Consecutive roofline failures tolerated before the watermark stops re-arming.
MAX_ROOFLINE_FAILURE_RETRIES: int = 3

# How many times a session re-runs forge-fusion after it aborted on
# infrastructure (no git workspace, harness could not be authored). Such a run
# judged nothing, so reporting it as a result would be wrong and it has to stay
# retryable -- but the causes do not all heal mid-session, and a retry re-runs
# LLM discovery before failing in the same place. Two is one free recovery from
# a transient cause plus the original attempt.
MAX_FUSION_INFRA_RETRIES = 2

# One lane ceiling covers both fusion pipelines, so a round can leave targets unfunded.
MAX_FUSION_WITHHELD_RETRIES = 2


def _withheld_targets(result: object) -> int:
    """How many discovered targets a fusion round's lane ceiling never funded."""
    if not isinstance(result, dict):
        return 0
    nomination = result.get("nomination")
    if not isinstance(nomination, dict):
        return 0
    withheld = nomination.get("withheld")
    return max(0, withheld) if isinstance(withheld, int) and not isinstance(withheld, bool) else 0


@dataclass(frozen=True)
class KernelExitFacts:
    """What the phase machine's KERNEL exit reads from the task queue."""

    agent_in_flight: bool


class KernelPhase(_GeakDispatchMixin, _GeakPromotionMixin, _GemmTuningMixin, CoordinatorCollaborator):
    """KERNEL_AGENT phase handler: drives kernel-level optimization tasks."""

    # Relative-change floor for the Forge kernel-entry reprofile: effectively "any change", absorbing float noise.
    _REPROFILE_CHANGE_TOL: float = 1e-5

    async def exit_facts(self) -> KernelExitFacts:
        """Advance the idle streak for this tick and report the facts the KERNEL exit decision reads."""
        await self._track_idle_streak()
        return KernelExitFacts(agent_in_flight=await self.agent_in_flight())

    async def agent_in_flight(self) -> bool:
        """Whether KERNEL is the phase and its ``kernel_agent`` task is queued or running; reads only."""
        if str(self.shared_state.phase or "").upper() != _phase_state.PHASE_KERNEL_AGENT:
            return False
        return any(task.kind == "kernel_agent" for task in [*await self.tasks.queued(), *await self.tasks.running()])

    async def _inflight_task_ids(self) -> tuple[str, ...]:
        """Return the ids of queued/running tasks doing KERNEL-lane work."""
        kinds = _phase_state.PHASE_ALLOWED_ACTIONS[_phase_state.PHASE_KERNEL_AGENT]
        tasks = list(await self.tasks.queued()) + list(await self.tasks.running())
        return tuple(
            sorted(str(task.task_id) for task in tasks if str(getattr(task, "kind", "") or "").strip() in kinds)
        )

    async def _track_idle_streak(self) -> None:
        """Advance or reset the KERNEL idle-streak counters for this tick."""
        state = self.shared_state
        if str(state.phase or "").upper() != _phase_state.PHASE_KERNEL_AGENT:
            state.kernel_idle_ticks = 0
            state.kernel_progress_fingerprint = ""
            state.kernel_idle_since_unix = 0.0
            return
        now = time.time()
        inflight = await self._inflight_task_ids()
        fingerprint = _phase_state.compute_kernel_progress_fingerprint(
            state,
            inflight_task_ids=inflight,
        )
        if fingerprint != str(state.kernel_progress_fingerprint or ""):
            state.kernel_progress_fingerprint = fingerprint
            state.kernel_idle_ticks = 0
            state.kernel_idle_since_unix = now
            return
        if inflight or _phase_state.kernel_inline_step_running(state, now_unix=now):
            state.kernel_idle_since_unix = now
            return
        # Only reachable after a tick that opened the streak above, so ``kernel_idle_since_unix`` is already stamped
        # whenever the counter is non-zero — the pairing the guard's wall-clock floor relies on.
        state.kernel_idle_ticks = int(state.kernel_idle_ticks or 0) + 1

    _kernel_timeline_recorder: "KernelEventRecorder | None" = None

    @staticmethod
    def _serving_config_signature(serving_config: Any) -> str:
        """Stable identity string for a ``serving_config`` sub-dict, or '' when empty."""
        if not isinstance(serving_config, Mapping) or not serving_config:
            return ""
        raw_envs = serving_config.get("extra_envs") or {}
        envs = (
            {str(key): str(value) for key, value in raw_envs.items() if str(key).strip()}
            if isinstance(raw_envs, Mapping)
            else {}
        )
        payload = {
            "extra_server_args": str(serving_config.get("extra_server_args") or "").strip(),
            "extra_envs": envs,
        }
        if not any((payload["extra_server_args"], envs)):
            return ""
        return "hyperloom-profile-config:" + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _current_profile_config_signature(self) -> str:
        """Return a stable identity for the optimized serving configuration."""
        serving_config = self.shared_state.profile_workload_context().get("serving_config")
        return self._serving_config_signature(serving_config)

    def _profile_config_changed(self, signature: str) -> bool:
        """Whether the latest trace predates the current backend/config."""
        if not signature:
            return False
        recorded = self.shared_state.last_profile_workload
        if not isinstance(recorded, Mapping) or not recorded:
            # No workload recorded for the trace: defer to _profile_workload_changed, which owns the "stale trace with
            # no workload metadata" decision.
            return False
        previous = self._serving_config_signature(recorded.get("serving_config"))
        return previous != signature

    def _profile_workload_changed(self) -> bool:
        """Whether the latest trace predates the active serving workload."""
        if self.shared_state.last_profile_status and not self.shared_state.profile_trace_usable():
            return True
        recorded = self.shared_state.last_profile_workload
        if not isinstance(recorded, dict) or not recorded:
            return bool(
                self.shared_state.last_profile_trace
                or self.shared_state.last_trace_analyze
                or self.shared_state.roofline_snapshots
            )
        identity = self.shared_state.profile_workload_identity
        return identity(recorded) != identity(self.shared_state.profile_workload_context())

    def _reprofile_idempotency_reason(self, profile_signature: Any, target_tput: float) -> str:
        identity = json.dumps(
            {
                "config": profile_signature,
                "target_tput": round(float(target_tput), 6),
                "workload": self.shared_state.profile_workload_context(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        stack_len = int(self.shared_state.cumulative_gain_validated_stack_len or 0)
        return f"kernel_entry_g{stack_len}_{hashlib.sha256(identity).hexdigest()[:12]}"

    def _adopt_reprofile_snapshot(self, *, before: float, snapshot_id_before: int, snapshots_before: int) -> bool:
        state = self.shared_state
        after = self._last_measured_roofline_tput()
        # A reprofile that reproduces the same tput under the same snapshot id still landed: the third signal is
        # that the run appended a snapshot at all, and without it a same-tput reprofile never adopts its workload.
        landed = (
            after != before
            or int(state.roofline_snapshot_id or 0) != snapshot_id_before
            or len(state.roofline_snapshots or []) != snapshots_before
        )
        if after <= 0 or not landed:
            log.warning("kernel-entry reprofile produced no new snapshot; the phase targets the existing trace")
            return landed
        state.last_roofline_tput = after
        # Record the workload (incl. serving_config) that this trace reflects; _profile_config_changed derives the
        # config signature from it, so last_profile_args stays the plain-args field it is everywhere else.
        state.last_profile_workload = state.profile_workload_context()
        state.save(self.session_dir)
        return landed

    async def _maybe_reprofile_for_kernel(self) -> None:
        """Reprofile inline when projected tput diverges from the last measured trace, so the phase targets the live bottleneck."""
        before = self._last_measured_roofline_tput()
        cur = self.current_tput_from_validated_gain()
        profile_signature = self._current_profile_config_signature()
        config_changed = self._profile_config_changed(profile_signature)
        workload_changed = self._profile_workload_changed()
        recorder = self.timeline()

        def _note_reprofile(reprofile: KernelReprofile) -> None:
            if recorder is None:
                return
            recorder.record_reprofile(reprofile)

        if cur <= 0:
            _note_reprofile(KernelReprofile(ran=False, skipped_reason="no_projected_tput"))
            return
        # With a measured trace, reprofile only on a material gain or a change in what is being measured.
        if (
            before > 0
            and abs(cur - before) / before < self._REPROFILE_CHANGE_TOL
            and not config_changed
            and not workload_changed
        ):
            _note_reprofile(KernelReprofile(ran=False, skipped_reason="within_tolerance"))
            return
        trigger = "config_changed" if config_changed else ("workload_changed" if workload_changed else "gain")
        snapshot_id_before = int(self.shared_state.roofline_snapshot_id or 0)
        snapshots_before = len(self.shared_state.roofline_snapshots or [])
        idempotency_reason = self._reprofile_idempotency_reason(profile_signature, cur)
        task_kind = self._coord.phase_prelude.internal_analysis_kind()
        params = self._coord.phase_prelude.internal_analysis_params(
            reason=idempotency_reason, inline_event=recorder.event_id if recorder is not None else ""
        )
        denied = self._coord.dispatcher.time_budget_denial_for_action(task_kind)
        if params is None or denied is not None:
            _note_reprofile(
                KernelReprofile(
                    ran=False,
                    task_kind=task_kind,
                    trigger=trigger,
                    skipped_reason=denied.rule if denied else analysis_skip_reason(self.shared_state),
                    idempotency_reason=idempotency_reason,
                    snapshot_id_before=snapshot_id_before,
                )
            )
            return
        # profile_lane conflicts with the benchmark_lane the kernel_agent task holds, so the reprofile is a step of
        # that task rather than a task of its own.
        reprofile_task = _covered_step(task_kind, params, idempotency_key=f"internal-analysis-{idempotency_reason}")
        try:
            await self.sub.execute_covered(reprofile_task)
        except Exception:  # isolation boundary: the kernel-entry reprofile step, recorded as dispatch_failed
            log.exception("kernel-entry reprofile failed; the phase proceeds on the existing snapshot")
            _note_reprofile(
                KernelReprofile(
                    ran=True,
                    task_kind=task_kind,
                    trigger=trigger,
                    skipped_reason="dispatch_failed",
                    idempotency_reason=idempotency_reason,
                    snapshot_id_before=snapshot_id_before,
                )
            )
            return
        snapshot_landed = self._adopt_reprofile_snapshot(
            before=before, snapshot_id_before=snapshot_id_before, snapshots_before=snapshots_before
        )
        _note_reprofile(
            KernelReprofile(
                ran=True,
                task_kind=task_kind,
                trigger=trigger,
                idempotency_reason=idempotency_reason,
                snapshot_landed=snapshot_landed,
                snapshot_id_before=snapshot_id_before,
                snapshot_id_after=int(self.shared_state.roofline_snapshot_id or 0),
                task_id=str(getattr(reprofile_task, "task_id", "") or ""),
            )
        )
        if snapshot_landed:
            self._record_kernel_discovered_from_cache(provenance="reprofile_snapshot")

    def geak_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the GEAK e2e optimizer."""
        return self.shared_state.kernel_optimizer == "geak"

    def timeline(self) -> Any:
        """The in-flight kernel timeline recorder, or ``None``."""
        return self._kernel_timeline_recorder

    def _open_kernel_timeline(self, *, route: str, route_reason: str, from_phase: str) -> None:
        """Open the kernel timeline event for this KERNEL entry."""
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import make_kernel_recorder

        state = self.shared_state
        recorder = make_kernel_recorder(
            macro_cycle=int(state.macro_cycle or 0),
            route=route,
            route_reason=route_reason,
            resumed=str(from_phase or "") == "resume",
        )
        self._kernel_timeline_recorder = recorder
        if recorder is None:
            return
        state = self.shared_state
        stack = state.optimization_stack if isinstance(state.optimization_stack, list) else []
        self._kernel_stack_at_entry = [dict(item) for item in stack if isinstance(item, dict)]
        cached = state.last_trace_analyze or {}
        recorder.begin(
            stack_depth_in=state.cumulative_gain_validated_stack_len or None,
            tput_before=state.current_best.get("tput"),
            session_baseline_tput=state.baseline_tput or None,
            snapshot=cached,
            snapshot_staleness="absent" if not cached else "fresh",
        )
        self._record_kernel_discovered_from_cache(provenance="entry_snapshot")

    def _record_kernel_discovered_from_cache(self, *, provenance: str) -> None:
        """Record the profiling table the visit inherited or just produced."""
        recorder = self.timeline()
        if recorder is None:
            return
        cached = self.shared_state.last_trace_analyze or {}
        recorder.record_discovered_kernels(cached, provenance=provenance)

    def _record_kernel_rewrite_controller_timeline(self, result: dict[str, Any]) -> None:
        """Record settled Controller integrations as Forge kernel rewrites."""
        integration = result.get("integration")
        rows = integration.get("results") if isinstance(integration, dict) else None
        recorder = self.timeline()
        if recorder is None or not isinstance(rows, list):
            return
        cycle = int(result.get("macro_cycle") or self.shared_state.macro_cycle or 0)
        recorded_any = False
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            kernel_id = first_str(row.get("operator_id"))
            if not kernel_id:
                continue
            status = first_str(row.get("status"), "unknown").lower()
            if status == "kept":
                decision = "KEEP"
            elif status.startswith("reverted"):
                decision = "REVERT"
            elif status.startswith("skipped") or stopped_by_the_run_class(status):
                decision = "SKIPPED"
            else:
                decision = "FAILED"
            gain_pct = row.get("gain_pct")
            speedup = (
                1.0 + float(gain_pct) / 100.0
                if isinstance(gain_pct, (int, float)) and not isinstance(gain_pct, bool)
                else None
            )
            recorder.record_kernel_rewrite(
                KernelRewriteRun(
                    lane=LaneRunFields(
                        run_id=f"controller-c{cycle}-{index}",
                        status=status,
                        micro_decision=decision,
                        failure_reason=first_str(row.get("reason")),
                        # The integration status is this route's failure taxonomy
                        # (``reverted_apply_conflict``, ``skipped_dirty_worktree``),
                        # and an integration row carries no separate class.
                        error_class="" if status == "kept" else status,
                    ),
                    kernel_id=kernel_id,
                    kernel_name=kernel_id.split(":")[2] if len(kernel_id.split(":")) > 2 else "",
                    backends_tried=["forge"],
                    adopted_backend="forge" if decision == "KEEP" else "",
                    speedup=speedup or None,
                )
            )
            recorded_any = True
        if recorded_any:
            tool_versions.record_tool_version(self.session_dir, tool="forge")

    def _record_gemm_tuning_timeline(self, result: dict[str, Any]) -> None:
        """Record a settled GEMM campaign in the Forge lane -- one row per tuner that ran.

        A campaign can run several tuners (e.g. ``fmoe_ck`` and ``a4w4_blockscale``) in one cycle;
        each has its own status, shape coverage and failure. Collapsing them into one row joined
        `tuner` names with a comma and let one tuner's `failure_reason` land on a row whose
        `gain_pct` came from a different, successful tuner -- a row that read as both "complete"
        and "failed" at once. The campaign's own validated gain (``result["e2e_gain_pct"]``, set by
        ``_validate_gemm_tuning_e2e`` only after e2e validation of the winning candidate) is
        attached to the tuner whose artifact was actually applied (``tuned_file``); a tuner's own
        best micro speedup is never substituted for it, since a per-shape timing ratio is not the
        axis the KEEP verdict was graded on.
        """
        recorder = self.timeline()
        if recorder is None:
            return
        try:
            self._record_gemm_tuner_rows(recorder, result)
        except Exception:
            log.debug("kernel timeline: GEMM tuning record failed", exc_info=True)

    def _record_gemm_tuner_rows(self, recorder: Any, result: dict[str, Any]) -> None:
        """Write one timeline row per tuner the campaign ran, and the campaign's own tool version."""
        tuner_rows = [row for row in result.get("tuners_run") or [] if isinstance(row, dict)]
        shape_capture = result.get("shape_capture") or {}
        base_run_id = (
            _first_text(result, "task_id")
            or Path(_first_text(result, "workspace")).name
            or f"gemm-c{int(self.shared_state.macro_cycle or 0)}"
        )
        tuned_file = _first_text(result, "tuned_file", "config_path")
        # Without per-tuner rows the campaign itself is the one row. It names these facts the way the campaign
        # envelope does, so it enters the loop under the per-tuner names and every field keeps its own source.
        campaign_row = {
            "tuner": result.get("tuner"),
            "total_shapes": result.get("shapes_total", shape_capture.get("shape_count")),
            "improved_shapes": result.get("shapes_tuned"),
            "artifact": tuned_file,
            "elapsed_s": result.get("duration_sec"),
            "error": result.get("error"),
            "skip_reason": result.get("skip_reason"),
            "error_class": result.get("error_class"),
        }
        for row in tuner_rows or [campaign_row]:
            tuner_name = _first_text(row, "tuner")
            # The campaign's e2e-validated gain names the axis the KEEP was graded on; it
            # belongs on the tuner whose artifact was actually applied, not on every row
            # this cycle produced.
            artifact = _first_text(row, "artifact", "env_value")
            applied = not tuner_rows or artifact == tuned_file != ""
            recorder.record_gemm_tuning_run(
                GemmTuningRun(
                    lane=LaneRunFields(
                        run_id=f"{base_run_id}-{tuner_name}" if tuner_rows and tuner_name else base_run_id,
                        status=_first_text(row, "status", default=_first_text(result, "status", default="unknown")),
                        micro_decision=_first_text(result, "micro_decision", "decision"),
                        integrate_ref=_first_text(result, "integration_id"),
                        started_at=_first_text(result, "started_at"),
                        ended_at=_first_text(result, "ended_at", "ts"),
                        duration_sec=row.get("elapsed_s", result.get("duration_sec")),
                        error_class=_first_text(row, "error_class"),
                        failure_reason=_first_text(row, "error", "skip_reason", "error_class"),
                    ),
                    shapes_total=row.get("total_shapes", result.get("shapes_total", shape_capture.get("shape_count"))),
                    shapes_tuned=row.get("improved_shapes"),
                    config_path=artifact,
                    # _validate_gemm_tuning_e2e writes the validated gain to e2e_gain_pct, not
                    # gain_pct -- no producer on this path sets a bare "gain_pct" key.
                    gain_pct=result.get("e2e_gain_pct") if applied else None,
                    graded_objective=_first_text(result, "graded_objective") if applied else "",
                    tuner=tuner_name,
                )
            )
        backend = _first_text(result, "backend", "engine").lower()
        if backend:
            tool_versions.record_tool_version(self.session_dir, tool=backend)

    def _record_fusion_timeline(self, result: dict[str, Any]) -> None:
        """Record a settled fusion campaign in the Forge lane."""
        recorder = self.timeline()
        if recorder is None or not isinstance(result, dict):
            return
        recorder.record_fusion_run(
            FusionRun(
                lane=LaneRunFields(
                    run_id=first_str(result.get("fusion_run_id")),
                    status=first_str(result.get("status"), "unknown"),
                    micro_decision=first_str(result.get("micro_decision"), result.get("decision")),
                    integrate_ref=first_str(result.get("integration_id")),
                    started_at=first_str(result.get("started_at")),
                    ended_at=first_str(result.get("ended_at"), result.get("ts")),
                    duration_sec=result.get("duration_sec"),
                    error_class=first_str(result.get("error_class")),
                    failure_reason=first_str(result.get("error"), result.get("skip_reason"), result.get("error_class")),
                ),
                pattern=first_str(result.get("pattern"), result.get("fusion_pattern")),
                target_module=first_str(result.get("target_module"), result.get("kernel_name")),
                applied=bool(result.get("kept")),
                gain_pct=result.get("gain_pct"),
                patch_path=first_str(result.get("patch_path"), result.get("source_patch")),
            )
        )
        tool_versions.record_tool_version(self.session_dir, tool="forge")
        agent_backend = first_str(result.get("agent_backend")).lower()
        if agent_backend:
            tool_versions.record_tool_version(self.session_dir, tool=agent_backend)

    def close_kernel_timeline(self, tr: "Transition") -> None:
        """Close the kernel timeline event when the phase is left."""
        exit_reason = tr.reason
        recorder = self.timeline()
        if recorder is None:
            return
        self._kernel_timeline_recorder = None
        state = self.shared_state
        stack_before = getattr(self, "_kernel_stack_at_entry", None) or []
        stack_after = [dict(item) for item in (state.optimization_stack or []) if isinstance(item, dict)]
        if len(stack_after) >= len(stack_before):
            stack_added = stack_after[len(stack_before) :]
            stack_removed = []
        else:
            stack_added = [item for item in stack_after if item not in stack_before]
            stack_removed = [item for item in stack_before if item not in stack_after]
        recorder.finish(
            KernelExit(
                exit_reason=exit_reason,
                tput_after=state.current_best.get("tput"),
                cumulative_gain_validated_out=state.cumulative_gain_validated or None,
                stack_depth_out=state.cumulative_gain_validated_stack_len or None,
                stack_added=stack_added,
                stack_removed=stack_removed,
            )
        )

    async def on_enter_kernel(self, tr: "Transition") -> None:
        """Open the KERNEL timeline and enqueue the ``kernel_agent`` task that carries the phase's work."""
        state = self.shared_state
        if not self._coord.phase_machine.kernel_enabled():
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False (from=%s)",
                tr.from_phase or "<unknown>",
            )
            return
        self._open_kernel_timeline(
            route=ROUTE_GEAK if self.geak_enabled() else ROUTE_FORGE,
            route_reason=f"kernel_optimizer={str(state.kernel_optimizer or '')}",
            from_phase=tr.from_phase,
        )
        lanes, lease_ttl = self._coord.dispatcher.registry_lanes_ttl("kernel_agent")
        # Leases do not expire on their TTL, so it only records how long the holder expects to keep the lanes.
        remaining = _phase_state.phase_budget_remaining_seconds(state)
        ttl = int(remaining) if remaining is not None and remaining > 0 else lease_ttl
        # A resumed session re-enters the phase whose earlier task already settled, so only a live row is reused.
        base_key = f"kernel_agent_c{int(state.macro_cycle or 0)}"
        attempt = 0
        while True:
            task, was_existing = await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind="kernel_agent",
                    params={"from_phase": str(tr.from_phase or "")},
                    idempotency_key=base_key if attempt == 0 else f"{base_key}-r{attempt}",
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="coordinator",
                )
            )
            if not (was_existing and task.state in TERMINAL_STATES):
                break
            attempt += 1
        log.info("KERNEL entry: kernel_agent task=%s (%s)", task.task_id, task.state)

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        """Report whether GEMM tuning is disabled via the env escape hatch.

        Returns:
            bool: ``True`` when ``INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING`` is set.
        """
        return env_bool("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING")

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Decide whether GEMM tuning must run before kernel_opt.

        Eligible on any supported framework (sglang / vllm / vllm-aiter), with no
        precision or MoE pre-filter.

        Returns:
            bool: ``True`` when GEMM tuning should run before source-level
                ``kernel_opt``.
        """
        if self._skip_gemm_tuning():
            return False
        ss = self.shared_state
        framework = str(ss.framework or "").strip().lower()

        # kernelforge gemm-tune handles any precision (bf16/fp16/fp8/fp4/mxfp4),
        # dense or MoE, on sglang/vllm. Real e2e KEEPs span all of these —
        # including bf16 *dense* (+11.1%) — so we must NOT pre-filter on
        # precision/MoE here, or a category that can optimize gets silently
        # blocked. Gate only on a supported framework and let forge itself
        # return no_improvement when a shape can't be beaten.
        eligible = framework in ("sglang", "vllm", "vllm-aiter")

        if not eligible:
            return False
        last = ss.last_gemm_tuning or {}
        status = str(last.get("status") or "").strip().lower()
        # The fp8 -> bf16 dense retry runs inside a single gemm call (the
        # tuner router selects the bf16 pass as a fallback), so a completed run's
        # status is terminal -- there is no pending second attempt to re-trigger.
        return status not in {
            "ok",
            "succeeded",
            "success",
            "complete",
            "completed",
            "skipped",
            "failed",
        }

    async def run_agent(self, ctx: Any) -> dict[str, Any]:
        """Run the KERNEL_AGENT phase's work under the ``kernel_agent`` task's lanes.

        Args:
            ctx: The runner context; ``ctx.task.params["from_phase"]`` names the
                phase the KERNEL entry came from.

        Returns:
            A result payload naming the route that ran.
        """
        from_phase = str((ctx.task.params or {}).get("from_phase") or "")
        if self.geak_enabled():
            # GEAK owns the whole KERNEL_AGENT phase: one in-process e2e run seeded with the best config so far, then
            # hand straight to SWEEP.
            await self._run_geak_kernel_phase(from_phase=from_phase)
            return {"status": "ok", "route": "geak"}
        if not self._gemm_tuning_required_before_kernel_opt():
            await self._finish_kernel_entry()
            return {"status": "ok", "route": "forge_no_gemm"}

        # Refresh the snapshot before GEMM tuning targets the bottleneck.
        await self._maybe_reprofile_for_kernel()
        log.info(
            "KERNEL entry: running GEMM tuning before source-level kernel_opt",
        )
        self._coord.phase_machine.record_phase_entry_evidence(
            gemm_tuning={"status": "running", "source": "kernel_entry_auto"},
        )
        run_gemm_tuning_handler = None
        try:
            from ..kernel.request_handlers import run_gemm_tuning_handler

            # The fp8 -> bf16 dense retry now lives inside the tuner router: an fp8 run whose tuning comes back empty
            # runs the bf16 dense pass in the same call (router selects it as a fallback).
            result = await run_gemm_tuning_handler(
                {
                    "task_id": "kernel_entry_gemm_tuning",
                    "reason": "kernel_entry_auto",
                    "macro_cycle": int(self.shared_state.macro_cycle or 0),
                },
                session_dir=self.session_dir,
            )
        except Exception as exc:  # isolation boundary: the kernel-entry GEMM tuning stage, recorded as failed
            log.exception("KERNEL entry GEMM tuning failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self._handle_gemm_tuning_result(result)

        status = str(result.get("status") or "unknown")
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "run_gemm_tuning_done",
                    "status": status,
                    "result": result,
                    "source": "kernel_entry_auto",
                },
            )
        )
        self._coord.phase_machine.record_phase_entry_evidence(
            gemm_tuning={
                "status": "done" if status in {"ok", "complete", "succeeded"} else status,
                "source": "kernel_entry_auto",
                "best_speedup": result.get("best_speedup"),
                "tuned_file": result.get("tuned_file"),
            },
        )
        # Capture explore + GEMM-tuning gains before the entry batch.
        await self._finish_kernel_entry()
        return {"status": "ok", "route": "forge_gemm"}

    def _kernel_rewrite_controller_timeouts(self) -> tuple[int, int]:
        """Return the Controller soft budget and Hyperloom hard timeout."""
        candidates: list[float] = []
        session_remaining = _phase_state.session_remaining_seconds(self.shared_state)
        if session_remaining is not None:
            candidates.append(max(0.0, float(session_remaining) - self.shared_state.closing_reserve_sec()))
        phase_remaining = _phase_state.phase_budget_remaining_seconds(self.shared_state)
        if phase_remaining is not None:
            candidates.append(max(0.0, float(phase_remaining)))
        # The phase still measures the Controller's patches E2E after the Controller exits.
        hard_timeout = int(min(candidates) - measurement_round_sec(self.shared_state)) if candidates else 90 * 60
        return max(0, hard_timeout - 30), max(0, hard_timeout)

    def reject_geak_promotion(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        current_best_tput: float,
        reason: str,
        attempt_id: str = "geak_final_validation",
    ) -> None:
        """Close a measured candidate without recording an adoption."""
        rejected_result = {**result, "revalidation_status": REBENCH_NO_PROMOTE, "revalidation_error": reason}
        rejected_result["final_validation"] = {
            "decision": "REJECTED",
            "reason": reason,
            "measured_tput": measured_tput,
            "current_best_tput": current_best_tput,
        }
        self.shared_state.geak_result = rejected_result
        self.shared_state.geak_pending = {}
        KernelPhase.reject_geak_kernel_journey(
            self,
            rejected_result,
            measured_tput=measured_tput,
            current_best_tput=current_best_tput,
            provenance="geak_promote_rejected",
            rejection_reason=reason,
        )
        recorder = KernelPhase.timeline(self)
        if recorder is not None:
            recorder.record_geak_rebench_attempt(
                GeakRebenchAttempt(
                    attempt_id=attempt_id,
                    base_tput=current_best_tput,
                    measured_tput=measured_tput if measured_tput > 0 else None,
                    decision=REBENCH_NO_PROMOTE,
                    decision_reason=reason,
                    status=REBENCH_NO_PROMOTE,
                )
            )
            recorder.record_geak_rebench_conclusion(final_status=REBENCH_NO_PROMOTE, final_error=reason)

    async def _finish_kernel_entry(self) -> None:
        """Run the gated kernel lanes, write the handoff, and delegate rewrite control."""
        await self._maybe_reprofile_for_kernel()
        await self._maybe_run_forge_fusion_before_kernel_opt()
        from hyperloom.inference_optimizer.session.session_paths import (
            next_forge_attempt_dir,
        )

        # One fresh directory per entry rather than per macro cycle: the controller refuses an output root it has
        # already initialized, and the handoff rides inside it so each attempt keeps the evidence it was given.
        attempt_dir = next_forge_attempt_dir(
            self.session_dir,
            int(self.shared_state.macro_cycle or 0),
        )
        handoff_dir = attempt_dir / "handoff"
        # Sealed before the handoff is written and before the controller starts,
        # which is the last moment the serving trees stand still: reprofile,
        # fusion and collective have all finished, and every uncommitted change
        # they left is part of what the server is now running. Committing it is
        # what lets a campaign name its own baseline -- the diff's starting
        # point, and the state a borrowed repository is handed back at.
        baselines: dict[str, object] = {}
        try:
            from ..kernel.campaign_baseline import seal_campaign_baseline

            baselines = seal_campaign_baseline(
                self.shared_state,
                session_id=str(self.shared_state.session_id or self.session_dir.name),
                macro_cycle=int(self.shared_state.macro_cycle or 0),
            )
            if baselines:
                log.info(
                    "KERNEL entry: sealed campaign baselines %s",
                    {repo: baseline.commit for repo, baseline in baselines.items()},
                )
        except (OSError, ValueError, subprocess.SubprocessError):
            log.exception("KERNEL entry: sealing the campaign baseline failed")
        from ..kernel.forge_handoff import write_forge_handoff

        env_spec = self._coord.writeback.build_env_spec()
        handoff_dir = write_forge_handoff(
            self.shared_state,
            env_spec=env_spec,
            handoff_dir=handoff_dir,
            baselines=baselines,
        )
        log.info("KERNEL entry: wrote Forge handoff to %s", handoff_dir)
        await self._run_kernel_rewrite_controller(handoff_dir, attempt_dir, baselines)

    async def _controller_attempt_result(self, handoff_dir: Path, output_dir: Path) -> dict[str, Any]:
        from ..kernel.controller_submit import run_controller_subprocess

        controller_budget_sec, hard_timeout_sec = self._kernel_rewrite_controller_timeouts()
        budget_minutes = controller_budget_sec / 60.0
        attempt = {
            "macro_cycle": int(self.shared_state.macro_cycle or 0),
            "handoff_dir": str(handoff_dir),
            "budget_minutes": budget_minutes,
            "hard_timeout_sec": hard_timeout_sec,
            "patch_count": 0,
            "task_count": 0,
            "output_dir": str(output_dir),
        }
        if controller_budget_sec <= 0 or hard_timeout_sec <= 0:
            return {
                **attempt,
                "status": "no_result",
                "reason": "no KERNEL phase budget remains for the rewrite controller",
            }
        try:
            result = await asyncio.to_thread(
                run_controller_subprocess,
                handoff_dir=handoff_dir,
                output_dir=output_dir,
                budget_minutes=budget_minutes,
                hard_timeout_sec=hard_timeout_sec,
            )
        except (OSError, ValueError) as error:
            log.exception("KERNEL entry: kernel rewrite controller failed")
            return {**attempt, "status": "failed", "reason": f"controller invocation failed: {error}"}
        # The four facts the phase owns are re-pinned over the child's own output: the Controller writes
        # ``state.json`` itself and must not be able to report a budget other than the one it was given.
        return {**attempt, **result, **{key: attempt[key] for key in _CONTROLLER_ATTEMPT_FACTS}}

    async def _integrate_controller_output(self, result: dict[str, Any], output_dir: Path) -> None:
        if int(result.get("patch_count") or 0) <= 0:
            result["integration"] = {
                "status": "not_run",
                "reason": "Controller published no patches",
                "kept_count": 0,
                "reverted_count": 0,
                "skipped_count": 0,
            }
            return
        from ..kernel.controller_patch_integration import KeepSession, integrate_controller_patches

        try:
            integration = await integrate_controller_patches(
                patches_root=str(result.get("patches_root") or output_dir / "result" / "patches"),
                session=KeepSession(
                    session_dir=self.session_dir,
                    shared_state=self.shared_state,
                    record_keep=self._coord.writeback.record_integrate_keep,
                ),
                admit_round=lambda rounds_run: controller_round_stop(self.shared_state, rounds_run),
            )
            result["integration"] = integration.to_dict()
        except Exception as error:  # isolation boundary: the Controller patch integration step, recorded as failed
            log.exception("KERNEL entry: Controller patch integration failed")
            result["integration"] = {"status": "failed", "reason": str(error), "kept_count": 0}

    def _file_controller_integration_failure(self, result: dict[str, Any]) -> None:
        integration = result["integration"]
        skipped = [
            str(row.get("reason") or "")
            for row in integration.get("results") or []
            if isinstance(row, dict) and str(row.get("status") or "").startswith("skipped")
        ]
        status = str(integration.get("status") or "")
        if status not in {"failed", "no_patch_admitted"} and not skipped:
            return
        cycle = int(self.shared_state.macro_cycle or 0)
        self.shared_state.record_action_failure(
            action="kernel_rewrite_controller",
            task_id=str(result.get("run_id") or f"forge-cycle-{cycle}"),
            result={
                "error_class": status or "patches_skipped",
                "error": "; ".join(x for x in ([str(integration.get("reason") or "")] + skipped) if x)[:800],
            },
        )

    async def _run_kernel_rewrite_controller(
        self, handoff_dir: Path, output_dir: Path, baselines: dict[str, object]
    ) -> None:
        """Run one Controller attempt without preselecting operators."""
        from ..kernel.controller_submit import record_controller_llm_usage

        result = await self._controller_attempt_result(handoff_dir, output_dir)
        # The Controller cannot reach this ledger from its own process, so its forge-loops' spend is filed here
        # now that the child has exited.
        record_controller_llm_usage(result=result, session_dir=self.session_dir)
        # Before integration reads any HEAD. A hard timeout kills the process
        # tree, so a borrowed repository can still be sitting on a campaign
        # branch, and integration refuses a publication whose base commit is
        # not the HEAD it finds -- which would discard exactly the patches
        # incremental publication saved from the kill.
        try:
            from ..kernel.campaign_baseline import reclaim_campaign_repositories

            reclaimed = reclaim_campaign_repositories(baselines)
            if reclaimed:
                result["reclaimed_repositories"] = reclaimed
        except ValueError:
            log.exception("KERNEL entry: reclaiming the campaign repositories failed")
        await self._integrate_controller_output(result, output_dir)
        self._record_kernel_rewrite_controller_timeline(result)
        self.shared_state.kernel_rewrite_controller_result = result
        self._file_controller_integration_failure(result)
        self.shared_state.save(self.session_dir)
        # The summary rides a ``response`` message the inbox dumps raw once.
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "kernel_rewrite_controller_done",
                    "status": result.get("status", "failed"),
                    "result": result,
                    "source": "kernel_entry_auto",
                },
            )
        )

    def _fusion_required_before_kernel_opt(self) -> bool:
        """Gate the forge-fusion step in KERNEL entry."""
        if env_bool("HYPERLOOM_SKIP_FUSION"):
            return False
        framework = str(self.shared_state.framework or "sglang").strip().lower()
        if framework not in ("sglang", "vllm", "vllm-aiter"):
            return False
        trace = str(self.shared_state.last_profile_trace or "").strip()
        if not trace:
            log.info("KERNEL entry: skip forge-fusion (no decode trace yet)")
            return False
        last = self.shared_state.last_fusion
        if isinstance(last, dict) and str(last.get("status") or "").strip() in ("ok", "complete", "kept"):
            # A round that kept nothing and left targets unfunded answers only for the ones it ran, so it re-arms
            # fusion until the retry cap is spent.
            if not last.get("kept") and _withheld_targets(last) > 0:
                return to_int(self.shared_state.fusion_withheld_retries, 0) < MAX_FUSION_WITHHELD_RETRIES
            return False
        if isinstance(last, dict) and last.get("infrastructure_abort"):
            # An abort judged nothing, so it must stay retryable -- but not forever.
            spent = to_int(self.shared_state.fusion_infra_aborts, 0)
            if spent >= MAX_FUSION_INFRA_RETRIES:
                log.info(
                    "KERNEL entry: skip forge-fusion (aborted on infrastructure %d time(s): %s)",
                    spent,
                    last.get("error_class") or "unknown",
                )
                return False
        return True

    async def _maybe_run_forge_fusion_before_kernel_opt(self) -> None:
        """Run the independently gated forge-fusion stage before kernel_opt."""
        if not self._fusion_required_before_kernel_opt():
            return
        await self._run_forge_fusion()
        await self._maybe_reprofile_for_kernel()

    async def _run_forge_fusion(self) -> None:
        """Run autonomous kernel fusion during KERNEL entry."""
        log.info("KERNEL entry: running forge-fusion (autonomous kernel fusion)")
        try:
            from ..kernel.request_handlers import run_fusion_handler

            result = await run_fusion_handler(
                {"task_id": "kernel_entry_fusion", "reason": "kernel_entry_auto"},
                session_dir=self.session_dir,
            )
        except Exception as exc:  # isolation boundary: the kernel-entry fusion stage, recorded as failed
            log.exception("KERNEL entry forge-fusion failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "engine": "forge_fusion",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self._handle_fusion_result(result)

    async def _handle_fusion_result(self, result: dict) -> None:
        """Record the forge-fusion result + surface it on the bus."""
        status = first_str(result.get("status"), "unknown") if isinstance(result, dict) else "failed"
        if isinstance(result, dict) and result.get("kept") and result.get("requires_e2e_validation"):
            from ..kernel.nomination_result import parse_outcome

            skew = parse_outcome(result).schema_error
            if skew:
                # A KEEP whose envelope the contract cannot read judged nothing, so it is reported as infrastructure
                # rather than latching the lane.
                result["status"] = "failed"
                result["error_class"] = "nomination_envelope_skew"
                result["error"] = skew
                result["infrastructure_abort"] = True
                status = "failed"
        if isinstance(result, dict) and result.get("infrastructure_abort"):
            # Counted on the session, not on the record: ``last_fusion`` is replaced by every run, so a timeout or a
            # handler crash landing between two aborts would carry no count forward and hand the cap back a clean
            # slate on every other entry.
            spent = to_int(self.shared_state.fusion_infra_aborts, 0)
            self.shared_state.fusion_infra_aborts = spent + 1
        if isinstance(result, dict) and not result.get("kept") and _withheld_targets(result) > 0:
            # Counted on the session for the same reason as the aborts above.
            withheld_spent = to_int(self.shared_state.fusion_withheld_retries, 0)
            self.shared_state.fusion_withheld_retries = withheld_spent + 1
        if isinstance(result, dict) and not first_str(result.get("fusion_run_id")).strip():
            cycle = int(self.shared_state.macro_cycle or 0)
            result["fusion_run_id"] = f"fusion-c{cycle}-{time.time_ns():x}"
        self.shared_state.last_fusion = result if isinstance(result, dict) else {"status": status}
        if isinstance(result, dict):
            self._record_fusion_timeline(result)
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent",
                    "orchestration",
                    "response",
                    {
                        "in_reply_to": "",
                        "kind": "run_fusion_done",
                        "status": status,
                        "result": result,
                        "source": "kernel_entry_auto",
                    },
                )
            )
        except (sqlite3.Error, TypeError, ValueError):
            log.exception("failed to post run_fusion_done bus message")
        # A KEPT fusion is handed to integrate for the e2e re-baseline decision.
        if isinstance(result, dict) and result.get("kept") and result.get("requires_e2e_validation"):
            await self._integrate_fusion(result)
        # One save after the queue is filled: a failed save cannot leave a kept fusion recorded without its siblings.
        self.shared_state.save(self.session_dir)

    async def _integrate_fusion(self, result: dict) -> None:
        """Queue every KEPT forge-fusion sibling for the shared e2e integrate lane."""
        from ..kernel.nomination_result import parse_outcome, summarize_dropped_patches

        outcome = parse_outcome(result)
        # Counted before the empty check: an all-refused envelope and a run that kept nothing are the same ``queued``
        # figure and differ only in reasons.
        refused = summarize_dropped_patches(outcome.dropped)
        if outcome.is_empty:
            if outcome.schema_error:
                log.warning(
                    "KERNEL entry: fusion KEPT but its nomination envelope was unreadable (%s); refused=%s",
                    outcome.schema_error,
                    refused or "none",
                )
            else:
                log.info(
                    "KERNEL entry: fusion KEPT but nominated no usable sibling; nothing to queue (refused=%s)",
                    refused or "none",
                )
            return
        keep_pct = env_float("HYPERLOOM_FUSION_KEEP_PCT", default=1.0)
        queued = 0
        for patch in outcome.patches:
            record = kernel_decisions.enqueue_nominated_patch(
                self.shared_state,
                patch=patch,
                keep_threshold_pct=keep_pct,
            )
            if record is not None:
                queued += 1
        log.info(
            "KERNEL entry: queued %d/%d fusion sibling(s) for SWEEP-entry integrate (refused=%s)",
            queued,
            len(outcome.patches),
            refused or "none",
        )

    def current_tput_from_validated_gain(self) -> float:
        """Project current tput from ``baseline_tput * (1 + cumulative_gain_validated/100)``; 0.0 when baseline unknown (watermark not-yet-armed)."""
        state = self.shared_state
        try:
            base = float(state.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return 0.0
        try:
            gain = float(state.cumulative_gain_validated or 0.0)
        except (TypeError, ValueError):
            gain = 0.0
        return base * (1.0 + gain / 100.0)

    def _last_measured_roofline_tput(self) -> float:
        """Measured tok/s of the most recent roofline snapshot; 0.0 when none."""
        snaps = self.shared_state.roofline_snapshots or []
        for snap in reversed(snaps):
            if not isinstance(snap, dict):
                continue
            try:
                tput = float(snap.get("achieved_tok_per_sec") or 0.0)
            except (TypeError, ValueError):
                tput = 0.0
            if tput > 0:
                return tput
        return 0.0

    def _needs_roofline_for_watermark(self) -> bool:
        """True iff projected tput crossed the watermark over ``last_roofline_tput`` (False until PRELUDE roofline ran, or while auto_roofline_pending_task_id is in-flight)."""
        state = self.shared_state
        if str(state.gpu_trace_unsupported_reason or ""):
            return False
        try:
            last_rl = float(state.last_roofline_tput or 0.0)
        except (TypeError, ValueError):
            last_rl = 0.0
        if (state.auto_roofline_pending_task_id or "").strip():
            return False
        if last_rl <= 0:
            try:
                failure_streak = int(state.roofline_failure_streak or 0)
            except (TypeError, ValueError):
                failure_streak = 0
            if failure_streak <= 0:
                return False
            if failure_streak > MAX_ROOFLINE_FAILURE_RETRIES:
                return False
            try:
                last_rl = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                last_rl = 0.0
            if last_rl <= 0:
                return False
        cur = self.current_tput_from_validated_gain()
        if cur <= 0:
            return False
        return cur / last_rl >= ROOFLINE_WATERMARK_RATIO

    async def release_finished_roofline_gate(self) -> None:
        """Drop an in-flight marker that names a roofline which already finished."""
        pending = (self.shared_state.auto_roofline_pending_task_id or "").strip()
        if not pending:
            return
        try:
            task = await self.tasks.get(pending)
        except TaskNotFound:
            task = None
        if task is not None and str(getattr(task, "state", "")) not in TERMINAL_STATES:
            return
        if self._coord.dispatcher.owes_result(pending):
            return
        self.shared_state.auto_roofline_pending_task_id = ""
        log.info("watermark-roofline: released in-flight gate held by finished task=%s", pending)

    async def maybe_enqueue_watermark_roofline(
        self,
        *,
        reason: str,
    ) -> bool:
        """Enqueue a fresh roofline if the watermark crossed; idempotency-keyed via ``reason`` and the roofline tput it crossed from, stamps auto_roofline_pending_task_id. Returns True when enqueued."""
        await self.release_finished_roofline_gate()
        if not self._needs_roofline_for_watermark():
            return False
        # Crossings share a reason; the anchor separates this crossing's task from an earlier one's.
        anchor = self.shared_state.last_roofline_tput
        try:
            task = await self._coord.phase_prelude.enqueue_internal_analysis_task(
                reason=reason,
                idem_scope=f"-from{anchor:.0f}" if anchor > 0 else "",
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            log.exception(
                "watermark-roofline (%s): failed to enqueue: %r",
                reason,
                exc,
            )
            return False
        if task is None:
            return False
        self.shared_state.auto_roofline_pending_task_id = task.task_id
        log.info(
            "watermark-roofline (%s): enqueued task=%s (cur=%.2f, last_roofline=%.2f, ratio>=%.2f)",
            reason,
            task.task_id,
            self.current_tput_from_validated_gain(),
            float(self.shared_state.last_roofline_tput or 0.0),
            ROOFLINE_WATERMARK_RATIO,
        )
        return True

    def cached_kernel_request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return the result to serve without running the handler: the trace's last successful analysis
        (``last_trace_analyze``), else a failure the same trace is known to reproduce; ``None`` to run it.
        """
        if kind != "trace_analyze":
            return None
        return self._cached_trace_analysis(payload) or remembered_trace_failure(self.shared_state, payload)

    def _cached_trace_analysis(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """The last successful analysis of ``payload``'s trace, when its candidates are still on disk."""
        cached = self.shared_state.last_trace_analyze or {}
        if not isinstance(cached, dict) or not cached:
            return None
        trace_input = payload.get("trace_input") or payload.get("trace_dir")
        if not trace_input or trace_input != cached.get("trace_input"):
            return None
        candidates_path = cached.get("candidates_path")
        if not candidates_path or not Path(candidates_path).exists():
            return None
        return {
            "status": "ok",
            "candidates_path": candidates_path,
            "hot_kernels_top15": cached.get("hot_kernels_top15", []),
            "reusable_native_kernel_ids": cached.get("reusable_native_kernel_ids", []),
            "cached_at": cached.get("ts"),
            "note": "served from shared_state.last_trace_analyze cache",
        }
