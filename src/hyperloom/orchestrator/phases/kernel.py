# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL_AGENT phase handler for collective, fusion, GEMM, and GEAK lanes."""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging as _logging
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from . import geak_rebench as _geak_rebench
from . import machine_state as _phase_state
from hyperloom.common.io import atomic_write_json
from hyperloom.common.perf_metric import graded_axes_of
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    LEVER_KERNEL,
)
from ..kernel import collective_recovery as _collective_recovery
from hyperloom.inference_optimizer.breakdown.recorder import tool_versions
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    ROUTE_COLLECTIVE_ONLY,
    ROUTE_FORGE,
    ROUTE_GEAK,
)
from ..actions.stop_attribution import stopped_by_the_run_class
from ..state.optimization_journal import (
    KIND_GEMM_TUNING,
    OUTCOME_KEEP,
    JournalEntry,
)
from ..state.task_registry import TERMINAL_STATES
from ..bus.message_bus import Message
from ..loop.coordinator_helpers import (
    _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
    _MAX_ROOFLINE_FAILURE_RETRIES,
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    _geak_spec_name,
    geak_is_cand_tag,
    geak_spec_is_env,
    _resolve_roofline_watermark_ratio,
    _accepted_config_as_variant,
    _coerce_tp,
    _resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_tp,
    _resolve_serving_fidelity,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)

# Last-resort location of the aiter checkout inside the standard serving container.
_CONTAINER_AITER_CONFIG_DIR = Path("/sgl-workspace/aiter/aiter/configs")

# How many times a session re-runs forge-fusion after it aborted on
# infrastructure (no git workspace, harness could not be authored). Such a run
# judged nothing, so reporting it as a result would be wrong and it has to stay
# retryable -- but the causes do not all heal mid-session, and a retry re-runs
# LLM discovery before failing in the same place. Two is one free recovery from
# a transient cause plus the original attempt.
MAX_FUSION_INFRA_RETRIES = 2

# One lane ceiling covers both fusion pipelines, so a round can leave targets unfunded.
MAX_FUSION_WITHHELD_RETRIES = 2


def _as_int(value: object) -> int:
    """Read a counter that round-tripped through JSON, defaulting to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _withheld_targets(result: object) -> int:
    """How many discovered targets a fusion round's lane ceiling never funded."""
    if not isinstance(result, dict):
        return 0
    nomination = result.get("nomination")
    if not isinstance(nomination, dict):
        return 0
    withheld = nomination.get("withheld")
    return max(0, withheld) if isinstance(withheld, int) and not isinstance(withheld, bool) else 0


def _as_float(value: object, default: float) -> float:
    """Read a measurement that round-tripped through JSON, defaulting on junk."""
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


# Which table each aiter config env var is resolved under at serving time.
_AITER_ENV_TO_TABLE: dict[str, str] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
    "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
}


def _integrate_server_logs(session_dir: Path, tuner_name: str) -> list[Path]:
    """Server logs for a tuner's integrate run, retries included, oldest first."""
    from ..kernel.gemm_shape_coverage import integrate_server_logs

    return integrate_server_logs(session_dir, f"integrate-gemm_tune_{tuner_name}")


def _candidate_tuned_file(env: Any, env_var: str) -> str:
    """Return the tuned artifact a candidate's env points at."""
    if not isinstance(env, dict):
        return ""
    value = env.get(env_var)
    if value in (None, ""):
        value = next((v for v in env.values() if v not in (None, "")), "")
    return str(value or "")


def _paired_measurement_basis(verdict: Any) -> str:
    """How the promoted gain was measured, so the ledger cannot overstate it."""
    if verdict is None:
        return "e2e_rebench_unpaired"
    if getattr(verdict, "candidate_wins", False):
        return "e2e_paired"
    return f"e2e_paired_{getattr(verdict, 'reason', 'unknown')}"


def _collective_comm_share(state: Any) -> tuple[float | None, str]:
    """Return the communication share gating the lane, and its provenance."""
    comm_pct = state.current_comm_pct()
    if comm_pct is not None:
        return float(comm_pct), "roofline"
    from ..kernel.request_handlers import select_collective_candidate

    try:
        candidate = select_collective_candidate(state)
    except (OSError, TypeError, ValueError) as exc:
        log.info("KERNEL entry: collective fallback share unavailable: %s", exc)
        return None, "unavailable"
    if not candidate:
        return None, "unavailable"
    return float(candidate["gpu_pct"]), "candidate_gpu_pct"


def _derive_collective_attempt_id(result: dict[str, Any]) -> str:
    """Compute the stable identity for one logical Collective campaign."""
    identity = {
        key: result.get(key)
        for key in (
            "analysis_key",
            "experiment_id",
            "patch",
            "kernel_id",
            "status",
            "error_class",
        )
        if result.get(key) not in (None, "")
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "collective-" + hashlib.sha256(encoded).hexdigest()[:24]


def _geak_decline_status(decline_reason: Any) -> str:
    """Map a 2b decline reason to the status left on ``geak_pending``."""
    reason = str(decline_reason or "").strip().lower()
    return "overlay_unloadable" if reason == "geak_overlay_unloadable" else "rebench_declined"


def _record_geak_integration(entry: dict[str, Any], *, kernel_id: str, macro_cycle: int) -> None:
    """Mirror one GEAK adoption onto the kernel event as an integrate row.

    GEAK adopts by writing the per-kernel ledger directly rather than through
    the integrate queue, so without this the timeline holds no gate row for a
    GEAK adoption at all -- and the basis the gain was measured on, along with
    whether it could be pinned on this one kernel, lives only on that ledger.

    The integration id is synthesized from the kernel, because the queue that
    would have minted one was never involved. It keys the row, so it carries
    no ``:``: that is the fragment key's own separator, and a row whose key
    contains one is dropped on the way to the event.

    Args:
        entry (dict[str, Any]): The per-kernel ledger entry just written.
        kernel_id (str): The kernel the adoption is for.
        macro_cycle (int): The cycle the adoption settled in.
    """
    if not kernel_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_integrate_verdict

        record_integrate_verdict(
            macro_cycle=macro_cycle,
            integration_id=f"geak-{kernel_id}",
            kernel_id=kernel_id,
            decision=str(entry.get("last_decision") or ""),
            status=str(entry.get("last_status") or ""),
            attempt_count=entry.get("attempt_count"),
            gain_pct=entry.get("best_gain_pct"),
            basis=str(entry.get("basis") or ""),
            alignment_status=str(entry.get("alignment_status") or ""),
            gain_attributed=bool(entry.get("validated", True)),
            settled_at=str(entry.get("updated_at") or ""),
        )
    except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
        log.debug("kernel timeline: geak integration record failed", exc_info=True)


class KernelPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

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
        recorded = getattr(self.shared_state, "last_profile_workload", None)
        if not isinstance(recorded, Mapping) or not recorded:
            # No workload recorded for the trace: defer to _profile_workload_changed, which owns the "stale trace with
            # no workload metadata" decision.
            return False
        previous = self._serving_config_signature(recorded.get("serving_config"))
        return previous != signature

    def _profile_workload_changed(self) -> bool:
        """Whether the latest trace predates the active serving workload."""
        status = str(getattr(self.shared_state, "last_profile_status", "") or "").strip().lower()
        if status and status != "succeeded":
            return True
        recorded = getattr(self.shared_state, "last_profile_workload", None)
        if not isinstance(recorded, dict) or not recorded:
            return bool(
                getattr(self.shared_state, "last_profile_trace", "")
                or getattr(self.shared_state, "last_trace_analyze", None)
                or getattr(self.shared_state, "roofline_snapshots", None)
            )
        identity = self.shared_state.profile_workload_identity
        return identity(recorded) != identity(self.shared_state.profile_workload_context())

    async def _maybe_reprofile_for_kernel(self) -> None:
        """Reprofile inline when projected tput diverges from the last measured trace, so the phase targets the live bottleneck."""
        before = self._last_measured_roofline_tput()
        cur = self._current_tput_from_validated_gain()
        profile_signature = self._current_profile_config_signature()
        config_changed = self._profile_config_changed(profile_signature)
        workload_changed = self._profile_workload_changed()
        recorder = self._kernel_timeline()

        def _note_reprofile(**fields: Any) -> None:
            if recorder is None:
                return
            try:
                recorder.record_reprofile(**fields)
            except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
                log.debug("kernel timeline: reprofile record failed", exc_info=True)

        if cur <= 0:
            _note_reprofile(ran=False, skipped_reason="no_projected_tput")
            return
        # With a measured trace, reprofile only on a material gain or a change in what is being measured.
        if (
            before > 0
            and abs(cur - before) / before < self._REPROFILE_CHANGE_TOL
            and not config_changed
            and not workload_changed
        ):
            _note_reprofile(ran=False, skipped_reason="within_tolerance")
            return
        trigger = "config_changed" if config_changed else ("workload_changed" if workload_changed else "gain")
        if config_changed or workload_changed:
            log.info("kernel-entry reprofile: active runtime context changed")
        snapshots_before = len(getattr(self.shared_state, "roofline_snapshots", None) or [])
        snapshot_id_before = int(getattr(self.shared_state, "roofline_snapshot_id", 0) or 0)
        stack_len = int(getattr(self.shared_state, "cumulative_gain_validated_stack_len", 0) or 0)
        profile_identity = json.dumps(
            {
                "config": profile_signature,
                "target_tput": round(float(cur), 6),
                "workload": self.shared_state.profile_workload_context(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        profile_fingerprint = hashlib.sha256(profile_identity.encode("utf-8")).hexdigest()[:12]
        idempotency_reason = f"kernel_entry_g{stack_len}_{profile_fingerprint}"
        task_kind = self._internal_analysis_kind()
        try:
            # The re-profile is this entry's own sub-step, not an action of its own: it exists to decide whether the
            # analysis the lanes will target is stale, and it is never dispatched by anything else.
            reprofile_task = await self._enqueue_internal_analysis_task(
                reason=idempotency_reason,
                inline_event=recorder.event_id if recorder is not None else "",
            )
            # An idempotent reuse can return a task that already reached a terminal state (its snapshot from a prior
            # cycle is still valid). run_task would then attempt succeeded->running -> IllegalTransition, so reuse the
            # existing snapshot instead of re-running.
            if str(getattr(reprofile_task, "state", "")) in TERMINAL_STATES:
                log.info(
                    "kernel-entry reprofile reuses terminal analysis task (state=%s); the phase targets the existing snapshot",
                    reprofile_task.state,
                )
                _note_reprofile(
                    ran=False,
                    task_kind=task_kind,
                    trigger=trigger,
                    skipped_reason="terminal_task_reused",
                    idempotency_reason=idempotency_reason,
                    snapshot_id_before=snapshot_id_before,
                )
                return
            await self.run_task_registered(reprofile_task)
        except Exception:  # noqa: BLE001 — never block the phase on a reprofile failure
            log.exception("kernel-entry reprofile failed; the phase proceeds on the existing snapshot")
            _note_reprofile(
                ran=True,
                task_kind=task_kind,
                trigger=trigger,
                skipped_reason="dispatch_failed",
                idempotency_reason=idempotency_reason,
                snapshot_id_before=snapshot_id_before,
            )
            return
        # Advance the anchor only when a new snapshot actually landed.
        after = self._last_measured_roofline_tput()
        snapshots_after = len(getattr(self.shared_state, "roofline_snapshots", None) or [])
        snapshot_id_after = int(getattr(self.shared_state, "roofline_snapshot_id", 0) or 0)
        snapshot_landed = (
            after != before or snapshots_after != snapshots_before or snapshot_id_after != snapshot_id_before
        )
        if after > 0 and snapshot_landed:
            self.shared_state.last_roofline_tput = after
            self.shared_state.last_profile_status = "succeeded"
            # Record the workload (incl. serving_config) that this trace reflects; _profile_config_changed derives the
            # config signature from it, so last_profile_args stays the plain-args field it is everywhere else.
            self.shared_state.last_profile_workload = self.shared_state.profile_workload_context()
            self.shared_state.save(self.session_dir)
        else:
            log.warning("kernel-entry reprofile produced no new snapshot; the phase targets the existing trace")
        _note_reprofile(
            ran=True,
            task_kind=task_kind,
            trigger=trigger,
            idempotency_reason=idempotency_reason,
            snapshot_landed=bool(snapshot_landed),
            snapshot_id_before=snapshot_id_before,
            snapshot_id_after=snapshot_id_after,
            task_id=str(getattr(reprofile_task, "task_id", "") or ""),
        )
        if snapshot_landed:
            self._record_kernel_discovered_from_cache(provenance="reprofile_snapshot")

    def _geak_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the GEAK e2e optimizer."""
        from ..kernel.request_handlers import geak_selected

        return geak_selected()

    def _kernel_timeline(self) -> Any:
        """The in-flight kernel timeline recorder, or ``None``."""
        return getattr(self, "_kernel_timeline_recorder", None)

    def _open_kernel_timeline(self, *, route: str, route_reason: str, from_phase: str) -> None:
        """Open the kernel timeline event for this KERNEL entry."""
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import make_kernel_recorder

        state = self.shared_state
        recorder = make_kernel_recorder(
            macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            route=route,
            route_reason=route_reason,
            resumed=str(from_phase or "") == "resume",
            code_revision=str(getattr(state, "code_revision", "") or ""),
        )
        self._kernel_timeline_recorder = recorder
        if recorder is None:
            return
        state = self.shared_state
        stack = state.optimization_stack if isinstance(getattr(state, "optimization_stack", None), list) else []
        self._kernel_stack_at_entry = [dict(item) for item in stack if isinstance(item, dict)]
        cached = getattr(state, "last_trace_analyze", None) or {}
        current_best = state.current_best if isinstance(getattr(state, "current_best", None), dict) else {}
        try:
            recorder.begin(
                stack_depth_in=getattr(state, "cumulative_gain_validated_stack_len", None),
                tput_before=current_best.get("tput"),
                session_baseline_tput=getattr(state, "baseline_tput", None),
                snapshot=cached,
                snapshot_staleness="absent" if not cached else "fresh",
            )
        except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
            log.debug("kernel timeline: begin failed", exc_info=True)
        else:
            self._record_kernel_discovered_from_cache(provenance="entry_snapshot")

    def _record_kernel_discovered_from_cache(self, *, provenance: str) -> None:
        """Record the profiling table the visit inherited or just produced."""
        recorder = self._kernel_timeline()
        if recorder is None:
            return
        cached = getattr(self.shared_state, "last_trace_analyze", None) or {}
        try:
            recorder.record_discovered_kernels(cached, provenance=provenance)
        except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
            log.debug("kernel timeline: discovered kernels record failed", exc_info=True)

    def _close_kernel_timeline(self, *, verdict: str = "", exit_reason: str = "") -> None:
        """Close the kernel timeline event when the phase is left."""
        recorder = self._kernel_timeline()
        if recorder is None:
            return
        self._kernel_timeline_recorder = None
        state = self.shared_state
        current_best = state.current_best if isinstance(getattr(state, "current_best", None), dict) else {}
        stack_before = getattr(self, "_kernel_stack_at_entry", None) or []
        stack_after = [dict(item) for item in (state.optimization_stack or []) if isinstance(item, dict)]
        if len(stack_after) >= len(stack_before):
            stack_added = stack_after[len(stack_before) :]
            stack_removed = []
        else:
            stack_added = [item for item in stack_after if item not in stack_before]
            stack_removed = [item for item in stack_before if item not in stack_after]
        try:
            recorder.finish(
                verdict=verdict,
                exit_reason=exit_reason,
                tput_after=current_best.get("tput"),
                cumulative_gain_validated_out=getattr(state, "cumulative_gain_validated", None),
                stack_depth_out=getattr(state, "cumulative_gain_validated_stack_len", None),
                stack_added=stack_added,
                stack_removed=stack_removed,
            )
        except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
            log.debug("kernel timeline: finish failed", exc_info=True)

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """Run deterministic KERNEL-entry optimization and re-profile gates."""
        if not self._kernel_enabled():
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False (from=%s)",
                from_phase or "<unknown>",
            )
            return
        collective_only = self._collective_only_mode()
        geak_enabled = False if collective_only else self._geak_enabled()
        if collective_only:
            self._open_kernel_timeline(
                route=ROUTE_COLLECTIVE_ONLY,
                route_reason="collective_only_mode",
                from_phase=from_phase,
            )
            self.shared_state.collective_only_mode = True
            self.shared_state.save(self.session_dir)
            await self._maybe_reprofile_for_kernel()
            await self._maybe_run_collective_before_kernel_opt()
            return
        self._open_kernel_timeline(
            route=ROUTE_GEAK if geak_enabled else ROUTE_FORGE,
            route_reason=f"kernel_optimizer={str(getattr(self.shared_state, 'kernel_optimizer', '') or '')}",
            from_phase=from_phase,
        )
        if geak_enabled:
            # GEAK owns the whole KERNEL_AGENT phase: one in-process e2e run seeded with the best config so far, then
            # hand straight to SWEEP.
            await self._run_geak_kernel_phase(from_phase=from_phase)
            return
        if not self._gemm_tuning_required_before_kernel_opt():
            await self._finish_kernel_entry()
            return

        # Refresh the snapshot before GEMM tuning targets the bottleneck.
        await self._maybe_reprofile_for_kernel()
        log.info(
            "KERNEL entry: running GEMM tuning before source-level kernel_opt",
        )
        self._record_phase_entry_evidence(
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
                    "macro_cycle": int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                },
                session_dir=self.session_dir,
            )
        except Exception as exc:  # noqa: BLE001
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
                priority=1,
            )
        )
        self._record_phase_entry_evidence(
            gemm_tuning={
                "status": "done" if status in {"ok", "complete", "succeeded"} else status,
                "source": "kernel_entry_auto",
                "best_speedup": result.get("best_speedup"),
                "tuned_file": result.get("tuned_file"),
            },
        )
        # Capture explore + GEMM-tuning gains before the entry batch.
        await self._finish_kernel_entry()

    @staticmethod
    def _read_recipe_bench_envs(recipe_path: str) -> dict[str, Any]:
        """Read recipe ``benchmark.envs``; return ``{}`` if unavailable."""
        try:
            import yaml

            if recipe_path and Path(recipe_path).is_file():
                cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
                envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
                return dict(envs) if isinstance(envs, dict) else {}
        except Exception:  # noqa: BLE001
            log.warning("geak handoff: could not read recipe %r", recipe_path, exc_info=True)
        return {}

    @classmethod
    def _resolve_bench_protocol(cls, recipe_path: str, *, envs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

        Reads the materialized baseline recipe's ``benchmark.envs`` (falling back
        to the process env) and returns only the keys that resolve, so absent
        values leave GEAK on its standalone defaults. Never raises.

        Args:
            recipe_path: Path to the baseline recipe YAML.
            envs: An already-parsed ``benchmark.envs`` for that path. Pass it
                when the caller needs the envs too, so the YAML is read once and
                both consumers see the same snapshot.
        """
        envs = cls._read_recipe_bench_envs(recipe_path) if envs is None else envs

        def _pick(key: str, cast: Callable[[str], Any]) -> Any:
            raw = envs.get(key)
            if raw is None or str(raw).strip() == "":
                raw = os.environ.get(key, "")
            raw = str(raw).strip()
            if not raw:
                return None
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return None

        protocol: dict[str, Any] = {}
        for proto_key, env_key, cast in (
            ("random_range_ratio", "RANDOM_RANGE_RATIO", float),
            ("num_prompts", "NUM_PROMPTS", int),
            ("num_warmups", "NUM_WARMUPS", int),
            ("seed", "SEED", int),
        ):
            val = _pick(env_key, cast)
            if val is not None:
                protocol[proto_key] = val
        return protocol

    def _geak_timeouts(self) -> tuple[int, int, bool]:
        """Resolve the GEAK e2e timeouts from the live run budget."""
        # Standalone fallback ONLY: the 12h (43200s) default applies when no run deadline is set (budget_known=False).
        env_default_timeout = int(os.environ.get("GEAK_E2E_TIMEOUT_S", "43200"))
        deadline = self._run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline - time.monotonic()
        grace = self.shared_state.closing_reserve_sec()
        margin = float(os.environ.get("GEAK_BUDGET_MARGIN_S", "300"))
        # Reserve the closing window: kill the subprocess with at least ``grace`` left.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget: cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(
            self.shared_state,
            budget_pct=self._phase_budget_pct,
        )
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # The runner self-stops ``margin`` before the hard subprocess kill, which reserves the closing-grace window.
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

    def _kernel_rewrite_controller_timeouts(self) -> tuple[int, int]:
        """Return the Controller soft budget and Hyperloom hard timeout."""
        candidates: list[float] = []
        session_remaining = _phase_state.session_remaining_seconds(self.shared_state)
        if session_remaining is not None:
            candidates.append(
                max(
                    0.0,
                    float(session_remaining) - self.shared_state.closing_reserve_sec(),
                )
            )
        phase_remaining = _phase_state.phase_budget_remaining_seconds(
            self.shared_state,
            budget_pct=self._phase_budget_pct,
        )
        if phase_remaining is not None:
            candidates.append(max(0.0, float(phase_remaining)))
        phase_cap = _phase_state.phase_cap_seconds(
            self.shared_state,
            budget_pct=self._phase_budget_pct,
        )
        if phase_cap is not None:
            candidates.append(
                max(
                    0.0,
                    float(phase_cap) - _phase_state.phase_cumulative_seconds(self.shared_state),
                )
            )
        hard_timeout = int(min(candidates)) if candidates else 90 * 60
        return max(0, hard_timeout - 30), max(0, hard_timeout)

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run)."""
        state = self.shared_state
        cb = state.current_best or {}
        try:
            env_spec = self.build_env_spec()
        except Exception:  # noqa: BLE001 — a legacy handoff remains runnable
            log.exception("geak: build_env_spec failed; handoff is unverified")
            env_spec = {}
        spec_config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        accepted_flags = str(spec_config.get("extra_server_args") or cb.get("extra_server_args") or "")
        extra_envs = spec_config.get("extra_envs") or cb.get("extra_envs") or {}
        accepted_env = " ".join(f"{k}={v}" for k, v in dict(extra_envs).items())
        state_measurement = getattr(state, "current_best_measurement", None)
        measurement = (
            state_measurement
            if isinstance(state_measurement, Mapping) and state_measurement
            else (
                cb.get("measurement") if isinstance(cb, Mapping) and isinstance(cb.get("measurement"), Mapping) else {}
            )
        )
        expected_identity = str(env_spec.get("launch_identity") or "")
        measured_identity = str(measurement.get("declared_launch_identity") or measurement.get("launch_identity") or "")
        identity_matches = bool(expected_identity and expected_identity == measured_identity)
        launch_evidence = measurement.get("launch_evidence")
        launch_evidence = dict(launch_evidence) if isinstance(launch_evidence, Mapping) else {}
        observed_flags = str(
            measurement.get("resolved_server_launch_flags") or launch_evidence.get("observed_server_launch_flags") or ""
        ).strip()
        observed_server_identity = measurement.get("observed_server_identity") or launch_evidence.get(
            "observed_server_identity"
        )
        observed_server_identity = (
            {str(key): value for key, value in sorted(observed_server_identity.items())}
            if isinstance(observed_server_identity, Mapping)
            else {}
        )
        if identity_matches and (observed_flags or observed_server_identity):
            reference_verification_status = "verified_observed"
        elif identity_matches and (
            str(launch_evidence.get("requested_server_args") or "").strip()
            or bool(launch_evidence.get("requested_server_env"))
            or str(launch_evidence.get("recipe_digest") or "").strip()
        ):
            reference_verification_status = "verified_declared_only"
        else:
            reference_verification_status = "unverified"
        reference_verified = reference_verification_status == "verified_observed"
        observed_identity = str(measurement.get("observed_launch_identity") or "")
        if not observed_identity and identity_matches and (observed_flags or observed_server_identity):
            observed_payload = json.dumps(
                {
                    "declared_launch_identity": measured_identity,
                    "observed_server_launch_flags": observed_flags,
                    "observed_server_identity": observed_server_identity,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            observed_identity = f"sha256:{hashlib.sha256(observed_payload).hexdigest()}"
        same_config_tput = float(measurement.get("tput") or 0.0) if reference_verified else 0.0
        workload = {
            "isl": int(getattr(state, "isl", 0) or int(os.environ.get("ISL", "1024"))),
            "osl": int(getattr(state, "osl", 0) or int(os.environ.get("OSL", "1024"))),
            "conc": int(getattr(state, "conc", 0) or int(os.environ.get("CONC", "64"))),
        }
        # Forward the benchmark settings and GPU placement used by Hyperloom.
        _recipe_path = str(getattr(state, "baseline_config_path", "") or "")
        # Parse once so every handoff field uses the same recipe snapshot.
        _recipe_envs = self._read_recipe_bench_envs(_recipe_path)
        bench_protocol = self._resolve_bench_protocol(_recipe_path, envs=_recipe_envs)
        # Preserve the run's actual GPU pin; {} means the whole machine.
        gpu_pin = _resolve_gpu_pin(recipe_envs=_recipe_envs)
        # Resolve TP and GPU ids together so the values cannot disagree.
        _tp = _coerce_tp(_recipe_envs.get("TP"), os.environ.get("TP"))
        # Clamp ids to the visible mask, then TP to the resulting device set.
        _gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=gpu_pin, tp=_tp)
        _tp = _resolve_handoff_tp(gpu_ids=_gpu_ids, tp=_tp)
        _gpu_ids_space = _resolve_handoff_gpu_ids_space(gpu_pin=gpu_pin)
        if _gpu_ids_space == "none":
            # An empty visibility mask means the ids are placeholders, not devices.
            log.error(
                "geak handoff: %s is set but empty; the run has no visible GPUs. "
                "gpu_ids=%s is a placeholder (gpu_ids_space=none), not a device set.",
                gpu_pin.get("var"),
                _gpu_ids,
            )
        # Reuse baseline server settings so GEAK measures the same engine config.
        try:
            from ..kernel.roofline_ceiling import read_baseline_server_args

            _baseline_srv_args = read_baseline_server_args(state) or ""
        except Exception:  # noqa: BLE001 — accessor is best-effort
            _baseline_srv_args = ""
        _current_best_server_args = str(spec_config.get("server_launch_flags") or _baseline_srv_args)
        _serving_fidelity = _resolve_serving_fidelity(
            baseline_server_args=_current_best_server_args,
            state_max_model_len=int(getattr(state, "max_model_len", 0) or 0),
        )

        handoff = {
            # v2 adds baseline_env_spec; v3 adds actual GPU-pinning metadata.
            "schema_version": 3,
            "model_path": str(getattr(state, "model_path", "") or os.environ.get("MODEL_PATH", "")),
            "framework": str(os.environ.get("FRAMEWORK", "") or "sglang"),
            "gpu_type": str(getattr(state, "gpu_type", "") or os.environ.get("GPU_TYPE", "")),
            "tp": _tp,
            "workload": workload,
            "accepted_flags": accepted_flags,
            "accepted_env": accepted_env,
            "launch_recipe": str(getattr(state, "baseline_config_path", "") or ""),
            "raw_baseline_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            # Orchestrator throughput of the SAME config GEAK seeds its baseline with, so run_e2e can compute a pure
            # measurement divergence. 0.0 => no accepted config yet (falls back to raw baseline downstream).
            "orchestrator_best_tput_same_config": same_config_tput,
            "same_config_reference_status": "verified" if reference_verified else "unverified",
            "same_config_reference_identity": measured_identity,
            "same_config_expected_identity": expected_identity,
            "same_config_reference_workspace": str(measurement.get("benchmark_workspace") or ""),
            # Additive identity semantics.
            "same_config_reference_verification_status": reference_verification_status,
            "same_config_reference_declared_identity": measured_identity,
            "same_config_reference_observed_identity": observed_identity,
            # GEAK compares this map with its parsed ServerArgs.
            "same_config_observed_identity": observed_server_identity,
            "observed_server_identity": observed_server_identity,
            "measurement_evidence": launch_evidence,
            "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
            # Serving-launch fidelity (both optional; unset => GEAK adapter default).
            "max_model_len": int(getattr(state, "max_model_len", 0) or int(os.environ.get("MAX_MODEL_LEN", "0") or 0)),
            "mem_fraction": float(
                getattr(state, "mem_fraction", 0.0) or float(os.environ.get("GPU_MEMORY_UTILIZATION", "0") or 0.0)
            ),
            "exp_root": str(self.session_dir / "geak"),
            # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the in-progress on-disk artifacts while a new
            # cycle gets a fresh dir.
            "eval_dir": str(self.session_dir / "geak" / f"e2e_cycle{int(getattr(state, 'macro_cycle', 0) or 0)}"),
            # Align GEAK's bench CLIENT to Hyperloom's exact one so final/sweep numbers are cross-harness comparable.
            "bench_client": "auto",
            "e2e_metric": "output",
            "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
            # The serving/optimization device set, as HIP-level ids (what the
            # consumer exports as HIP_VISIBLE_DEVICES). Logical positions inside
            # an inherited ROCR mask, a HIP/CUDA mask as-is, else 0..tp-1.
            "gpu_ids": _gpu_ids,
            # Which coordinate system ``gpu_ids`` is in: "logical" (positions
            # inside the ROCR mask the child inherits), "absolute" (whole-
            # machine device ids), or "none" (the mask is set but empty — the
            # ids are placeholders and must not be launched on). Exporting them
            # as HIP_VISIBLE_DEVICES is correct in the first two; a consumer
            # that instead writes ROCR itself needs to know which it holds.
            "gpu_ids_space": _gpu_ids_space,
        }
        if gpu_pin:
            # ABSOLUTE ids + the var they came from, so a consumer that writes
            # ROCR_VISIBLE_DEVICES itself re-applies the same pin instead of
            # resetting the child to card 0.
            handoff["gpu_pin"] = gpu_pin
        if bench_protocol:
            handoff["bench_protocol"] = bench_protocol
        # Only forward resolved fidelity knobs; absence => GEAK adapter default.
        handoff.update(_serving_fidelity)
        # Full layered environment and its matching measurement identity.
        if env_spec:
            handoff["baseline_env_spec"] = env_spec

        out_dir = self.session_dir / "geak"
        out_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = out_dir / "handoff.json"
        handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        recorder = self._kernel_timeline()
        if recorder is not None:
            try:
                recorder.enter_stage("geak_delegation")
                recorder.record_geak_handoff(handoff)
            except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
                log.debug("kernel timeline: geak handoff record failed", exc_info=True)

        from ..kernel.request_handlers import _kernel_agent_tool_path

        def _read_geak_result(path: Path) -> dict[str, Any]:
            if not path.is_file():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}

        def _promote_recovered_result(
            result: dict[str, Any],
            *,
            recovered_from: str,
            runner_timeout_s: int | None = None,
        ) -> None:
            state.geak_result = result
            self._record_geak_measurement(result)
            # Rebench-first: record the recovered win as an UNVALIDATED candidate;
            # the caller enqueues the main-flow rebench that writes the headline.
            self._record_geak_candidate(result)
            self._record_geak_kernel_journey(result)
            evidence = {
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "recovered_from": recovered_from,
            }
            if runner_timeout_s is not None:
                evidence["runner_timeout_s"] = runner_timeout_s
            self._record_phase_entry_evidence(geak=evidence)
            # Set the wind-down hint BEFORE the durable save (it is in-memory only).
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
            state.save(self.session_dir)

        def _finish_skip(result: dict[str, Any]) -> None:
            """Record a (failed/skipped) GEAK outcome + wind down to SWEEP."""
            state.geak_result = result
            self._record_phase_entry_evidence(
                geak={
                    "status": result.get("status"),
                    "error_class": result.get("error_class"),
                    "error": (str(result.get("error") or "")[:500] or None),
                }
            )
            # Persist the wind-down hint durably.
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
            state.save(self.session_dir)

        async def _replay_succeeded_rebench(task_id: str) -> bool:
            """Replay a persisted delegated result lost before state writeback."""
            try:
                settled_task = await self.tasks.get(task_id)
                for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                    payload = msg.payload if isinstance(msg.payload, dict) else {}
                    if str(payload.get("task_id") or "") != task_id:
                        continue
                    if str(payload.get("kind") or "") != "explore":
                        continue
                    if str(payload.get("state") or "").lower() != "succeeded":
                        continue
                    result = payload.get("result")
                    if not isinstance(result, dict) or not self._is_promotable_result("explore", result):
                        return False

                    replay_pending = dict(state.geak_pending) if isinstance(state.geak_pending, dict) else {}
                    replay_pending["status"] = "awaiting_rebench"
                    replay_pending["revalidation_task_id"] = task_id
                    replay_pending.pop("revalidation_error", None)
                    state.geak_pending = replay_pending
                    state.save(self.session_dir)
                    await self._promote_to_shared_state(
                        "explore",
                        result,
                        task=settled_task,
                    )
                    return True
            except Exception:  # noqa: BLE001 - recovery is best-effort
                log.exception(
                    "geak: failed to replay delegated result for succeeded rebench %s",
                    task_id,
                )
            return False

        async def _enqueue_geak_revalidation(*, reason: str) -> bool:
            """Enqueue and persist the rebench that keeps a GEAK win pending."""
            # Reserve the pending slot BEFORE the task exists.
            cycle = int(getattr(state, "macro_cycle", 0) or 0)
            placeholder_keys = _geak_rebench.geak_revalidation_placeholder_keys(cycle)
            inflight = await _geak_rebench.find_inflight_geak_rebench_task(self.tasks)
            if inflight is not None and inflight.state in {"queued", "running"}:
                pending = dict(state.geak_pending) if isinstance(state.geak_pending, dict) else {}
                pending["status"] = "awaiting_rebench"
                pending["revalidation_task_id"] = inflight.task_id
                pending.pop("revalidation_error", None)
                state.geak_pending = pending
                state.save(self.session_dir)
                log.info(
                    "geak: revalidation already in flight (%s); skipping duplicate enqueue",
                    inflight.task_id,
                )
                return True

            reserved = dict(state.geak_pending) if isinstance(state.geak_pending, dict) else {}
            reserved["status"] = "awaiting_rebench"
            if not str(reserved.get("revalidation_task_id") or "").strip():
                reserved["revalidation_task_id"] = _geak_rebench.geak_revalidate_idempotency_key(cycle)
            reserved.pop("revalidation_error", None)
            state.geak_pending = reserved
            state.save(self.session_dir)

            try:
                summary = await self._enqueue_internal_stack_rebench(reason=reason)
            except Exception as exc:  # noqa: BLE001 - defensive
                log.exception("geak: enqueue same-harness revalidation failed")
                summary = {"skipped": True, "reason": repr(exc)}

            # The dispatcher refuses to launch a rebench whose only material is an overlay that cannot load — that run
            # would measure plain baseline and credit GEAK for the noise.
            if isinstance(summary, dict) and summary.get("fallback") == "geak_harness":
                log.warning(
                    "geak: 2b declined (%s); validating through the GEAK harness instead",
                    summary.get("reason"),
                )
                try:
                    fb = await self._validate_geak_via_geak_harness(reason=str(summary.get("reason") or "2b_declined"))
                except Exception as exc:  # noqa: BLE001 - defensive
                    log.exception("geak: GEAK-harness validation failed")
                    fb = {"validated": False, "reason": repr(exc)}
                if bool(fb.get("validated")):
                    # 2a promotes and clears geak_pending itself.
                    return True
                pending = dict(state.geak_pending) if isinstance(state.geak_pending, dict) else {}
                pending["status"] = _geak_decline_status((summary or {}).get("reason"))
                pending.pop("revalidation_task_id", None)
                pending["revalidation_error"] = str(fb.get("reason") or summary.get("reason") or "")[:500]
                state.geak_pending = pending
                state.save(self.session_dir)
                return False

            task_id = str(summary.get("task_id") or "") if isinstance(summary, dict) else ""
            task_state = str(summary.get("task_state") or "queued").strip().lower() if task_id else ""
            existing = bool(isinstance(summary, dict) and summary.get("existing"))
            pending = dict(state.geak_pending) if isinstance(state.geak_pending, dict) else {}
            if task_id and task_state in {"queued", "running"}:
                pending["status"] = "awaiting_rebench"
                pending["revalidation_task_id"] = task_id
                state.geak_pending = pending
                state.save(self.session_dir)
                return True

            if task_id and existing and task_state == "succeeded":
                # create_or_return_existing returned a task that already ran under this cycle's idempotency key
                # (#1240).
                prior_geak_result = state.geak_result if isinstance(getattr(state, "geak_result", None), dict) else {}
                settled_status = str(prior_geak_result.get("revalidation_status") or "")
                if settled_status in {"no_material", "no_promote"} or self._geak_win_already_recorded():
                    state.geak_pending = {}
                    state.save(self.session_dir)
                    return True
                if await _replay_succeeded_rebench(task_id):
                    log.info(
                        "geak: replayed persisted result for already-succeeded rebench %s",
                        task_id,
                    )
                    return True
                pending["status"] = "rebench_unavailable"
                if pending.get("revalidation_task_id") in placeholder_keys:
                    pending.pop("revalidation_task_id", None)
                pending["revalidation_error"] = (
                    f"rebench task {task_id} already succeeded but its verdict could not be reconciled"
                )[:500]
                state.geak_pending = pending
                state.save(self.session_dir)
                log.warning(
                    "geak: same-harness revalidation unavailable; candidate remains audit-only (%s)",
                    pending["revalidation_error"],
                )
                return False

            pending["status"] = "rebench_unavailable"
            # Drop reservation placeholders (current + legacy) so no stale id outlives the slot.
            if pending.get("revalidation_task_id") in placeholder_keys:
                pending.pop("revalidation_task_id", None)
            if task_state == "cancelled":
                default_reason = f"rebench cancelled before completion ({task_id or 'unknown'})"
            else:
                default_reason = f"rebench task settled without a usable result (state={task_state or 'unknown'})"
            pending["revalidation_error"] = str((summary or {}).get("reason") or default_reason)[:500]
            state.geak_pending = pending
            state.save(self.session_dir)
            log.warning(
                "geak: same-harness revalidation unavailable; candidate remains audit-only (%s)",
                pending["revalidation_error"],
            )
            return False

        # Crash-recovery: a validated result.json written before a coordinator crash is promoted on resume, guarded by
        # ``_geak_win_already_recorded`` so a prior cycle's result.json does not short-circuit a fresh entry.
        result_path = out_dir / "result.json"
        recovered = _read_geak_result(result_path)
        # Tombstone a result already adjudicated by 2b so stale result.json cannot re-enqueue a settled candidate on a
        # later KERNEL entry.
        prev_geak = (
            self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        )
        already_adjudicated = str(prev_geak.get("revalidation_status") or "") in {
            "no_material",
            "no_promote",
        }
        if recovered.get("status") == "ok" and not self._geak_win_already_recorded() and not already_adjudicated:
            log.info(
                "GEAK result.json exists but state has no recorded win "
                "(crash before handback); promoting recovered result."
            )
            _promote_recovered_result(recovered, recovered_from="existing_result_json")
            if recovered.get("status") == "ok":
                await _enqueue_geak_revalidation(reason="geak_e2e_win_recovered")
            return

        try:
            runner = _kernel_agent_tool_path("backends/geak_runner.py")
        except Exception as exc:  # noqa: BLE001
            log.exception("GEAK runner not resolvable; skipping KERNEL")
            _finish_skip({"status": "error", "error_class": "runner_not_found", "error": repr(exc)})
            return

        # Budget-aware timeouts: shrink to the remaining run deadline and always reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._geak_timeouts()
        min_run = int(os.environ.get("GEAK_MIN_RUN_S", "600"))
        if budget_known and runner_timeout < min_run:
            log.warning(
                "GEAK: only %ds budget remains (< min %ds); skipping e2e "
                "and winding down to SWEEP so the closing report runs in time.",
                runner_timeout,
                min_run,
            )
            _finish_skip(
                {
                    "status": "skipped",
                    "error_class": "insufficient_budget",
                    "error": (
                        f"only {runner_timeout}s of KERNEL budget remained "
                        f"(< min {min_run}s); skipped to protect the closing "
                        f"report window"
                    ),
                    "runner_timeout_s": runner_timeout,
                }
            )
            return

        cmd = [
            sys.executable,
            str(runner),
            str(handoff_path),
            str(out_dir),
            "--timeout-s",
            str(runner_timeout),
        ]
        log.info(
            "KERNEL entry: delegating to GEAK e2e (from=%s) runner_timeout=%ds kill_timeout=%ds budget_known=%s cmd=%s",
            from_phase or "<unknown>",
            runner_timeout,
            kill_timeout,
            budget_known,
            " ".join(cmd),
        )

        # Run in its own process group so a timeout can SIGTERM the whole runner -> run_e2e -> vllm/node tree (grace
        # to flush result.json), then SIGKILL, instead of orphaning run_e2e + its servers.
        term_grace = int(os.environ.get("GEAK_TERM_GRACE_S", "180"))

        def _run() -> subprocess.CompletedProcess:
            runner_env = dict(os.environ)
            runner_env["E2E_METRIC"] = "output"
            # Only injection point needed for the whole GEAK chain: geak_runner and run_e2e both hand their full
            # environment to the child, so the tag reaches the Claude CLI that actually spends.
            from hyperloom.common.llm_attribution import inject_env

            inject_env(runner_env, component="geak", operation="optimize_kernel")
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=runner_env,
                start_new_session=True,
            )

            def _killpg(sig: int) -> None:
                try:
                    os.killpg(os.getpgid(p.pid), sig)
                except (ProcessLookupError, PermissionError):
                    # Process already exited; nothing to signal.
                    pass

            try:
                out, err = p.communicate(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                _killpg(signal.SIGTERM)
                try:
                    out, err = p.communicate(timeout=term_grace)
                except subprocess.TimeoutExpired:
                    _killpg(signal.SIGKILL)
                    out, err = p.communicate()
                raise subprocess.TimeoutExpired(
                    cmd,
                    kill_timeout,
                    output=out,
                    stderr=err,
                )
            return subprocess.CompletedProcess(cmd, p.returncode, out, err)

        try:
            proc = await asyncio.to_thread(_run)
            stderr_tail = (proc.stderr or "")[-2000:]
            if proc.returncode != 0:
                log.warning("GEAK runner rc=%s: %s", proc.returncode, stderr_tail)
        except subprocess.TimeoutExpired:
            log.warning(
                "GEAK runner exceeded kill_timeout=%ds; SIGTERM'd to let it flush, then reclaimed the closing window",
                kill_timeout,
            )
            # The graceful SIGTERM gives run_e2e a window to flush result.json; keep a real win instead of discarding
            # the phase as a timeout.
            recovered = _read_geak_result(result_path)
            if recovered.get("status") == "ok":
                log.info(
                    "GEAK flushed an OK result.json under SIGTERM grace; promoting the recovered win despite the cap."
                )
                _promote_recovered_result(
                    recovered,
                    recovered_from="sigterm_flushed_result_json",
                    runner_timeout_s=runner_timeout,
                )
                # Rebench-first: enqueue the main-flow rebench (candidate stays pending if a budget cap prevents it
                # from running).
                await _enqueue_geak_revalidation(reason="geak_e2e_win_sigterm_recovered")
                return
            _finish_skip(
                {
                    "status": "error",
                    "error_class": "timeout",
                    "error": (f"GEAK e2e killed after {kill_timeout}s (budget-capped); closing window preserved"),
                    "runner_timeout_s": runner_timeout,
                    "kill_timeout_s": kill_timeout,
                }
            )
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("GEAK runner crashed")
            _finish_skip({"status": "error", "error_class": "runner_crashed", "error": repr(exc)})
            return

        result: dict[str, Any] = _read_geak_result(result_path)
        if not result:
            _finish_skip(
                {
                    "status": "error",
                    "error_class": "no_result_json",
                    "error": (f"runner rc={proc.returncode} produced no parseable result.json at {result_path}"),
                    "stderr_tail": stderr_tail,
                }
            )
            return
        # Carry the actual exit code so the breakdown can audit a nonzero rc.
        result.setdefault("returncode", proc.returncode)
        state.geak_result = result
        self._record_geak_measurement(result)

        # Invariant guard: a GEAK run whose baseline ref failed to reproduce ``orchestrator_best_tput_same_config``
        # optimized against a phantom baseline, so its gain is non-comparable — never promote it.
        if str(result.get("status") or "") == "baseline_reproduction_failed":
            log.warning(
                "GEAK baseline_reproduction_failed: ref did not match "
                "orchestrator best (%s); refusing to promote a phantom-baseline gain",
                result.get("error"),
            )
            _finish_skip(
                {
                    "status": "baseline_reproduction_failed",
                    "error_class": "baseline_reproduction_failed",
                    "error": (
                        str(result.get("error") or "")[:500]
                        or "GEAK baseline ref != orchestrator best (env_spec mismatch)"
                    ),
                    "ref_tput": result.get("ref_tput"),
                    "orchestrator_best_tput_same_config": result.get("orchestrator_best_tput_same_config"),
                }
            )
            return

        # Rebench-first: record the win as an UNVALIDATED candidate only; the headline is written later from the
        # measured rebench.
        self._record_geak_candidate(result)
        self._record_geak_kernel_journey(result)
        # Enqueue the same-harness config-identity rebench — the ONLY path that
        # writes the headline. Until it lands the candidate stays pending.
        if str(result.get("status") or "") == "ok":
            await _enqueue_geak_revalidation(reason="geak_e2e_win")
        elif _geak_has_accepted_kernel(result):
            # A no_gain headline over an accepted, parity-checked kernel still deserves the measurement — the rebench
            # is what decides, and without it the kernel is lost with no number attached to it.
            await _enqueue_geak_revalidation(reason="geak_e2e_accepted_kernel")
        self._record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "runner_timeout_s": runner_timeout,
            }
        )
        state.save(self.session_dir)
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "geak_e2e_done",
                    "status": str(result.get("status") or "unknown"),
                    "speedup": result.get("throughput_speedup"),
                    "result_path": str(result_path),
                },
                priority=1,
            )
        )
        # KERNEL is a one-shot under GEAK: wind down to SWEEP (persist the hint).
        state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
        state.save(self.session_dir)

    def _geak_win_already_recorded(self) -> bool:
        """Whether a GEAK e2e win is already in this session's state."""
        return any(
            isinstance(item, dict) and item.get("action") == "geak_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    @staticmethod
    def _parse_geak_accepted_config(
        result: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """Parse ``result.accepted_config`` into (flags, env dict)."""
        return _accepted_config_as_variant(result.get("accepted_config"))

    def _record_geak_candidate(self, result: dict[str, Any]) -> None:
        """Record a GEAK e2e win as an UNVALIDATED candidate (no headline)."""
        if not isinstance(result, dict):
            return
        # ``no_gain`` is GEAK's verdict on its own headline number, not on the kernels it accepted.
        if result.get("status") not in ("ok",) and not _geak_has_accepted_kernel(result):
            return
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        if new_tput <= 0:
            return
        accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)
        base = float(self.shared_state.baseline_tput or 0.0)
        self_gain = ((new_tput - base) / base * 100.0) if base > 0 else None
        am = result.get("alignment_metrics") or {}
        self.shared_state.geak_pending = {
            "status": "awaiting_rebench",
            # Audit-only self-reported numbers (not the headline until rebench).
            "self_reported_tput": new_tput,
            "self_reported_speedup": result.get("throughput_speedup"),
            "self_reported_gain_pct": self_gain,
            "self_reported_basis": result.get("final_throughput_basis"),
            # Reproducible config the rebench launches from.
            "accepted_flags": accepted_flags,
            "accepted_envs": dict(parsed_envs),
            # Carry the kernels and the basis they were judged on into the pending record, so a later promotion can
            # name what it adopted without re-reading result.json.
            "accepted_kernels": result.get("accepted_kernels") or [],
            "geak_status": str(result.get("status") or ""),
            "baseline_alignment_status": str((result.get("baseline_alignment") or {}).get("status") or ""),
            "final_overlay": result.get("final_overlay") or "",
            "final_launch_script": result.get("final_launch_script"),
            "bench_script": result.get("bench_script"),
            "eval_dir": result.get("eval_dir"),
            # GEAK's own within-harness speedups, for the report's audit cross-check.
            "alignment": {
                "hot_geak_speedup": am.get("hot_geak_speedup"),
                "cold_geak_speedup": am.get("cold_geak_speedup"),
                "hot_speedup": am.get("hot_speedup"),
                "cold_speedup": am.get("cold_speedup"),
                "final_basis": am.get("final_basis") or result.get("final_throughput_basis"),
                "geak_throughput_speedup": result.get("throughput_speedup"),
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        recorder = self._kernel_timeline()
        if recorder is not None:
            try:
                recorder.record_geak_claim(
                    self.shared_state.geak_pending,
                    specs=self._geak_acceptance_specs(result),
                )
                recorder.record_geak_product(
                    accepted_flags=accepted_flags,
                    accepted_envs=dict(parsed_envs),
                    accepted_config=result.get("accepted_config"),
                    final_overlay=result.get("final_overlay") or "",
                    final_launch_script=result.get("final_launch_script") or "",
                    bench_script=result.get("bench_script") or "",
                    final_patch=result.get("final_patch") or "",
                )
            except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
                log.debug("kernel timeline: geak claim record failed", exc_info=True)
        # Surface a large cross-harness measurement divergence as a warning only.
        bb = result.get("baseline_basis") or {}
        mdiv = bb.get("measurement_divergence_pct")
        try:
            mdiv_f = abs(float(mdiv)) if mdiv is not None else None
        except (TypeError, ValueError):
            mdiv_f = None
        if mdiv_f is not None and mdiv_f > _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT:
            log.warning(
                "geak candidate: large cross-harness measurement divergence "
                "%.2f%% (|.|>%.1f%%) - candidate held out of headline until a "
                "main-flow rebench validates it",
                float(mdiv),
                _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
            )

    @staticmethod
    def _geak_acceptance_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return every GEAK acceptance, tagged with the queue that proposed it."""
        lanes: list[tuple[str, Any]] = [
            *(("kernelQueue", row) for row in (result.get("accepted_kernels") or [])),
            *(("headQueue", row) for row in (result.get("accepted_heads") or [])),
        ]
        out: list[dict[str, Any]] = []
        index: dict[tuple[str, str], int] = {}
        for lane, raw in lanes:
            if not isinstance(raw, dict):
                continue
            stated = raw.get("e2e_delta_pct")
            try:
                delta = None if stated is None or stated == "" else float(stated)
            except (TypeError, ValueError):
                delta = None
            name = _geak_spec_name(raw)
            if not name:
                continue
            row = {**raw, "lane": lane, "alias_collapsed": False}
            if geak_spec_is_env(raw):
                row["kind"] = "env"
            if delta is None:
                # Collapsing is a claim that two rows measured the same thing, and an absent or unparseable delta is
                # no evidence for it.
                out.append(row)
                continue
            twin = (str(raw.get("op_kind") or ""), f"{delta:.4f}")
            position = index.get(twin)
            if position is None:
                index[twin] = len(out)
                out.append(row)
                continue
            kept = out[position]
            kept["alias_collapsed"] = True
            # The collapsed twin's name is the one a reader may hold, so it is
            # carried on the survivor rather than dropped with the row.
            aliases = {*(kept.get("aliases") or []), _geak_spec_name(kept), name}
            if geak_is_cand_tag(_geak_spec_name(kept)) and not geak_is_cand_tag(name):
                row["alias_collapsed"] = True
                out[position] = row
                kept = row
            kept["aliases"] = sorted({a for a in aliases if a and a != _geak_spec_name(kept)})
        return out

    @staticmethod
    def _geak_stack_entry_extra(result: dict[str, Any], *, overlay_loaded: bool | None) -> dict[str, Any]:
        """Build the ``geak_e2e`` stack entry, carrying only kernels proven to have run."""
        proven = overlay_loaded is True
        return {
            "accepted_kernels": (result.get("accepted_kernels") or []) if proven else [],
            "accepted_heads": (result.get("accepted_heads") or []) if proven else [],
            "report_path": result.get("report_path"),
            "source": "geak_e2e",
            "overlay_loaded": overlay_loaded,
        }

    def _promote_geak_from_candidate(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        provenance: str = "geak_e2e_promote",
        overlay_loaded: bool | None = None,
        measurement_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the GEAK headline from a MEASURED main-flow rebench."""
        if not isinstance(result, dict):
            return
        try:
            measured = float(measured_tput)
        except (TypeError, ValueError):
            return
        if measured <= 0:
            return
        # KEEP guard (aligns GEAK with forge / integrate_patch): a measured rebench that does not beat the current
        # best must NOT overwrite the headline / stack / gain.
        cb_now = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
        cb_tput = cb_now.get("tput")
        if isinstance(cb_tput, (int, float)) and cb_tput > 0 and measured <= float(cb_tput):
            log.info(
                "geak promote skipped: measured %.3f did not beat current_best %.3f",
                measured,
                float(cb_tput),
            )
            self._reject_geak_kernel_journey(
                result,
                measured_tput=measured,
                current_best_tput=float(cb_tput),
                provenance="geak_promote_rejected",
            )
            try:
                rejected_result = dict(result)
                rejected_result["final_validation"] = {
                    "decision": "REJECTED",
                    "reason": "rebench_did_not_beat_current_best",
                    "measured_tput": measured,
                    "current_best_tput": float(cb_tput),
                }
            except Exception:  # noqa: BLE001
                log.debug(
                    "geak v4 final validation rejection recording failed",
                    exc_info=True,
                )
            self.shared_state.geak_pending = {}
            self.shared_state.resume_pending_revalidation = False
            return
        accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)

        # The lever is stamped here, not guessed from the task kind: GEAK promotes on a proven kernel overlay OR on a
        # config/env-only win, and only this site holds the overlay proof.
        entry_extra = self._geak_stack_entry_extra(result, overlay_loaded=overlay_loaded)
        kernel_proven = bool(entry_extra.get("accepted_kernels") or entry_extra.get("accepted_heads"))

        promotion_measurement = {
            "name": "geak_e2e",
            "candidate_extra_server_args": accepted_flags,
            "extra_envs": dict(parsed_envs),
            "final_overlay": result.get("final_overlay") or "",
            "source_phase": "KERNEL_AGENT",
            "lever_kind": LEVER_KERNEL if kernel_proven else LEVER_CONFIG,
            "ttft_mean_ms": result.get("ttft_ms"),
            "tpot_mean_ms": result.get("tpot_ms"),
            **graded_axes_of(result),
            "workspace": result.get("eval_dir"),
        }
        if isinstance(measurement_provenance, Mapping):
            for key in (
                "launch_evidence",
                "launch_evidence_path",
                "server_log_path",
                "workspace",
                "single_workspace",
            ):
                value = measurement_provenance.get(key)
                if value not in (None, "", {}):
                    promotion_measurement[key] = value
        self._lift_to_current_best(
            "geak_e2e",
            measured,
            promotion_measurement,
            entry_extra=entry_extra,
        )

        base = float(self.shared_state.baseline_tput or 0.0)
        # Where the session stood before GEAK ran: the anchor both the journey rejection and the route-level residual
        # measure from.
        pre_geak = float(cb_tput) if isinstance(cb_tput, (int, float)) and cb_tput > 0 else base
        self._record_geak_adopted_kernels(
            result,
            measured_tput=measured,
            baseline_tput=base,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
        )
        if overlay_loaded is not True:
            # The journey is replayed before the main-flow rebench and can therefore contain GEAK-internal KEEPs for
            # kernels that were not present in the configuration that produced ``measured``.
            self._reject_geak_kernel_journey(
                result,
                measured_tput=measured,
                current_best_tput=pre_geak,
                provenance=provenance,
                rejection_reason="overlay_not_proven_loaded",
            )
        if base > 0:
            self._update_cumulative_gain_validated(
                measured,
                result,
                source="geak_e2e_promote",
            )
        self.shared_state.resume_pending_revalidation = False
        self.shared_state.geak_pending = {}

    @staticmethod
    def _geak_journey_path(result: dict[str, Any]) -> str:
        """Resolve the journey file for a GEAK result."""
        if not isinstance(result, dict):
            return ""
        path = str(result.get("kernel_journey_path") or "")
        if not path:
            eval_dir = str(result.get("eval_dir") or "")
            if eval_dir:
                path = str(Path(eval_dir) / "kernel_journey.json")
        return path if path and Path(path).is_file() else ""

    @classmethod
    def _load_geak_journey(cls, result: dict[str, Any]) -> dict[str, Any]:
        """Read the journey file, or return ``{}`` when it is unusable."""
        path = cls._geak_journey_path(result)
        if not path:
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.debug("geak journey read failed", exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _geak_journey_kernels(cls, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the journey's kernel records, or ``[]`` when unreadable."""
        journey = cls._load_geak_journey(result)
        return [kernel for kernel in journey.get("kernels") or [] if isinstance(kernel, dict)]

    def _record_geak_adopted_kernels(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        baseline_tput: float,
        provenance: str,
        overlay_loaded: bool | None,
    ) -> None:
        """Write one adoption row per accepted GEAK kernel."""
        if not isinstance(result, dict):
            return
        # Both acceptance lanes, ``env`` selections excluded and alias twins collapsed.
        specs = _geak_accepted_kernel_specs(result)
        if not specs:
            return
        rows = [
            {
                "kernel_id": str(k.get("short_name") or k.get("kernel_id") or k.get("cand_tag") or "").strip(),
                "spec": k,
            }
            for k in specs
        ]

        rebench_gain: float | None = None
        if baseline_tput > 0 and measured_tput > 0:
            rebench_gain = (measured_tput - baseline_tput) / baseline_tput * 100.0
        # One kernel, overlay proven loaded, one measured number: the gain is attributable.
        attributable = bool(overlay_loaded) and len(rows) == 1
        am = result.get("alignment_metrics") or {}
        basis = str(am.get("final_basis") or result.get("final_throughput_basis") or "")
        alignment_status = str((result.get("baseline_alignment") or {}).get("status") or "")
        ts = datetime.now(timezone.utc).isoformat()
        ledger = self.shared_state.kernel_integrate_attempts
        if not isinstance(ledger, dict):
            return
        for row in rows:
            kid = row["kernel_id"]
            spec = row["spec"]
            entry = dict(ledger.get(kid) or {})
            attempts = list(entry.get("attempts") or [])
            attempt_decision = "KEEP" if attributable else "UNATTRIBUTED"
            attempt_status = "ok" if attributable else "unvalidated"
            attempts.append(
                {
                    "decision": attempt_decision,
                    "status": attempt_status,
                    "new_tput": measured_tput,
                    "gain_pct": rebench_gain if attributable else None,
                    "decision_reason": provenance,
                    "artifact_kind": str(spec.get("kind") or "authored"),
                    "ts": ts,
                    "cycle": int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                }
            )
            # Max over attempts, matching the canonical ledger writer in ``_kernel_decisions.py`` -- ``by_kernel`` and
            # ``kernel_lifecycle`` read this one field from both writers, so a second, worse rebench must not lower
            # the kernel's best.
            gains = [
                float(a["gain_pct"])
                for a in attempts
                if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
            ]
            entry.update(
                {
                    "key": kid,
                    "kernel_id": kid,
                    "source": "geak_e2e",
                    "attempts": attempts,
                    "attempt_count": len(attempts),
                    "best_gain_pct": max(gains) if gains else None,
                    "last_decision": attempt_decision,
                    "last_status": attempt_status,
                    "validated": attributable,
                    "overlay_loaded": bool(overlay_loaded),
                    "basis": basis,
                    "alignment_status": alignment_status,
                    # GEAK's own same-config A/B, kept beside the orchestrator number so the two are never confused
                    # for each other.
                    "geak_same_config_delta_pct": spec.get("e2e_delta_pct"),
                    "geak_isolated_speedup": spec.get("isolated"),
                    "updated_at": ts,
                }
            )
            ledger[kid] = entry
            # GEAK adopts by writing this ledger directly rather than through
            # the integrate queue, so without this the kernel timeline holds no
            # gate row for a GEAK adoption at all -- and the basis the gain was
            # measured on lives only here.
            _record_geak_integration(
                entry,
                kernel_id=kid,
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
            )
        self.shared_state.kernel_integrate_attempts = ledger
        log.info(
            "geak: recorded %d adopted kernel(s) in the per-kernel ledger (overlay_loaded=%r attributable=%r gain=%r)",
            len(rows),
            overlay_loaded,
            attributable,
            rebench_gain if attributable else None,
        )

    def _record_geak_measurement(self, result: dict[str, Any]) -> None:
        """Record the latency and parity GEAK's harness measured for this run.

        Called where ``geak_result`` is set rather than beside the candidate
        record, because a run that measured a latency but accepted nothing
        never reaches the candidate path and would otherwise report none of it.

        Best-effort: a failed record never breaks the phase.
        """
        if not isinstance(result, dict) or not result:
            return
        recorder = self._kernel_timeline()
        if recorder is None:
            return
        try:
            recorder.record_geak_measurement(result)
        except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
            log.debug("kernel timeline: geak measurement record failed", exc_info=True)

    def _record_geak_kernel_journey(self, result: dict[str, Any]) -> None:
        """Record what GEAK-e2e's ``kernel_journey.json`` says about its run.

        Two facts come out of the file. The attempts themselves go onto the
        kernel timeline event, which is what the breakdown reads. The tool
        builds are the other: GEAK reports the build of every tool its run went
        through, and this journey is the only place they reach the optimizer at
        all, so they are recorded even though nothing else in the file is.

        Best-effort: a missing or partial file never breaks the phase.
        """
        journey = self._load_geak_journey(result)
        if not journey:
            return

        recorder = self._kernel_timeline()
        if recorder is not None:
            try:
                recorder.record_geak_attempts(journey)
            except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
                log.debug("kernel timeline: geak attempts record failed", exc_info=True)

        for tool, meta in (journey.get("versions") or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                tool_versions.record_tool_version(
                    self.session_dir,
                    tool=str(tool),
                    root=str(meta.get("root_dir") or "") or None,
                    version=str(meta.get("version") or meta.get("commit") or "") or None,
                )
            except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
                log.debug("kernel timeline: geak tool version record failed", exc_info=True)

    def _reject_geak_kernel_journey(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        current_best_tput: float,
        provenance: str,
        rejection_reason: str = "rebench_did_not_beat_current_best",
    ) -> None:
        """Replace provisional GEAK e2e KEEPs after a failed final rebench."""

        # Named on the class, not through ``self``: Coordinator does not
        # delegate this method, so callers bind it with a Coordinator as
        # ``self`` and an attribute lookup there would not find the helper.
        for kernel in KernelPhase._geak_journey_kernels(result):
            kernel_id = str(kernel.get("kernel_id") or "")
            e2e = kernel.get("e2e")
            if not kernel_id or not isinstance(e2e, dict):
                continue
            decision = str(e2e.get("decision") or "").upper()
            if decision not in {"KEEP", "ADOPTED"}:
                continue
            evidence = dict(e2e)
            evidence.update(
                {
                    "self_reported_e2e_gain_pct": e2e.get("e2e_gain_pct"),
                    "revalidation_measured_tput": measured_tput,
                    "revalidation_current_best_tput": current_best_tput,
                    "revalidation_provenance": provenance,
                    "rejection_reason": rejection_reason,
                }
            )

    def _runtime_uses_aiter_fused_moe(self) -> bool:
        """Return whether the served model dispatches MoE through aiter."""
        from ..kernel.request_handlers import _resolve_forge_server_log

        try:
            log_path = _resolve_forge_server_log(self.shared_state, self.session_dir)
        except Exception:  # noqa: BLE001 - detection is best-effort
            return False
        if not log_path:
            return False
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return "[aiter] [fused_moe]" in text or "Mxfp4 MoE backend" in text

    def _gemm_tuned_config_coverage(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Report whether the validated aiter CSV was reachable by the server."""
        try:
            return self._gemm_tuned_config_coverage_impl(tuner_name, envs)
        except Exception:  # noqa: BLE001
            log.warning(
                "tuned-config coverage failed for %s; treating it as undetermined",
                tuner_name,
                exc_info=True,
            )
            return None

    def _gemm_tuned_config_coverage_impl(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Replay aiter's lookup against the round's log (see the caller)."""
        if tuner_name == "fmoe_ck":
            return self._fmoe_tuned_config_coverage(envs)
        from ..kernel.gemm_shape_coverage import (
            parse_aiter_consulted_tables,
            parse_aiter_shape_lookups,
            parse_aiter_shape_lookups_for_tables,
            tuned_config_coverage,
            tuned_csv_shapes,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        logs = _integrate_server_logs(self.session_dir, tuner_name)
        if not logs:
            return None
        try:
            log_text = logs[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        def _unreadable(kind: str) -> None:
            """Log that the artifact could not be read, so the caller stays out of it."""
            log.warning(
                "gemm E2E: tuner=%s %s tuned CSV yielded no keys from %s; "
                "coverage is undetermined and will not block the KEEP",
                tuner_name,
                kind,
                csv_paths,
            )

        all_missed, all_hit = parse_aiter_shape_lookups(log_text)
        all_requested = all_missed | all_hit
        if not all_requested:
            return None
        wanted = {Path(path).name for path in csv_paths}
        missed, hit = parse_aiter_shape_lookups_for_tables(log_text, wanted)
        requested = missed | hit
        scoped_to_candidate = bool(requested)
        if not scoped_to_candidate:
            # Preserve the existing artifact-not-consulted diagnostic when the server performed lookups, but none
            # against this candidate's table.
            missed, hit = all_missed, set()
            requested = all_requested
        tuned: set[tuple[int, int, int]] = set()
        for path in csv_paths:
            tuned |= tuned_csv_shapes(path)
        if not tuned:
            _unreadable("dense")
            return None
        report = tuned_config_coverage(tuned, requested, known_covered=hit)
        report["server_log"] = str(logs[-1])
        report["runtime_lookup_miss"] = len(missed)
        report["runtime_lookup_hit"] = len(hit)
        report["artifact_applied"] = bool(report.get("covered"))
        consulted = parse_aiter_consulted_tables(log_text)
        report["consulted_tables"] = sorted(consulted)[:8]
        if consulted and not (wanted & {Path(name).name for name in consulted}):
            # The runtime resolved a different quantisation variant's table, so the tuner targeted a kernel this
            # server never dispatches to.
            report["artifact_applied"] = False
            report["not_applied_reason"] = "artifact_table_not_consulted"
        elif not report["artifact_applied"]:
            report["not_applied_reason"] = "no_shape_key_matched"
        return report

    def _fmoe_tuned_config_coverage(
        self,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Report whether a ``tuned_fmoe.csv`` covers logged fused-MoE dispatches."""
        from ..kernel.gemm_shape_coverage import (
            aiter_log_tuned_config_enabled,
            fmoe_tuned_config_coverage,
            log_has_fused_moe_activity,
            parse_aiter_fused_moe_dispatches,
            read_latest_integrate_server_log,
            resolve_fmoe_candidate_csv,
            tuned_fmoe_csv_rows,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        loaded = read_latest_integrate_server_log(self.session_dir)
        if loaded is None:
            return None
        log_path, log_text = loaded
        candidate_path = resolve_fmoe_candidate_csv(csv_paths[0])
        dispatches = parse_aiter_fused_moe_dispatches(log_text)
        hit_logging = aiter_log_tuned_config_enabled(envs)
        report: dict[str, Any] = {
            "server_log": str(log_path),
            "requested": len(dispatches),
        }
        if not dispatches:
            if log_has_fused_moe_activity(log_text):
                report["artifact_applied"] = False
                report["not_applied_reason"] = "fused_moe_parse_inconclusive"
                report["conclusive"] = False
            elif not hit_logging:
                report["artifact_applied"] = False
                report["not_applied_reason"] = "fused_moe_logging_disabled"
                report["conclusive"] = False
            else:
                report["artifact_applied"] = False
                report["not_applied_reason"] = "no_fused_moe_dispatch"
                report["conclusive"] = True
            report["runtime_lookup_miss"] = 0
            report["runtime_lookup_hit"] = 0
            return report
        if candidate_path is None:
            report["artifact_applied"] = False
            report["not_applied_reason"] = "candidate_csv_missing"
            report["runtime_lookup_miss"] = len(dispatches)
            report["runtime_lookup_hit"] = 0
            report["conclusive"] = False
            return report
        candidate_rows = tuned_fmoe_csv_rows(candidate_path)
        report["candidate_csv"] = str(candidate_path)
        report.update(fmoe_tuned_config_coverage(candidate_rows, dispatches))
        report["artifact_applied"] = bool(report.get("covered"))
        report["conclusive"] = True
        report["runtime_lookup_miss"] = report.get("requested", 0) - report.get("covered", 0)
        report["runtime_lookup_hit"] = report.get("covered", 0)
        if not report["artifact_applied"]:
            if report.get("runtime_default"):
                report["not_applied_reason"] = "runtime_default_config"
            elif report.get("kernel_name_mismatch"):
                report["not_applied_reason"] = "kernel_name_mismatch"
            else:
                report["not_applied_reason"] = "no_shape_key_matched"
        return report

    async def _confirm_gemm_gain_paired(
        self,
        stacked_envs: dict[str, str],
        *,
        baseline_tput: float,
        budget_minutes: int,
        extra_server_args: str = "",
    ):
        """Re-measure baseline and tuned stack interleaved, and judge the pairs."""
        from ..kernel.request_handlers import integrate_handler
        from ..measurement.paired import assess_paired, interleaved_plan

        try:
            n_pairs = int(os.environ.get("HYPERLOOM_GEMM_PAIRED_PAIRS", "0") or 0)
        except ValueError:
            n_pairs = 0
        if n_pairs <= 0 or not stacked_envs or baseline_tput <= 0:
            return None

        pairs: list[tuple[float, float]] = []
        pending: float | None = None
        for idx, side in enumerate(interleaved_plan(n_pairs)):
            envs = {} if side == "A" else dict(stacked_envs)
            # The B leg has to be served the same way the KEEP was: fmoe_ck only takes effect under
            # --moe-runner-backend aiter, and without it the tuned table is never read, so B measures the same thing
            # as A and the confirmation reports within_noise for a gain that is real.
            side_args = extra_server_args if side == "B" else ""
            try:
                res = await integrate_handler(
                    {
                        "task_id": f"gemm_paired_{side}{idx}",
                        "kernel_id": f"gemm_paired_{side}{idx}",
                        "source": "forge_gemm_paired",
                        "base_tput": baseline_tput,
                        "extra_server_args": side_args,
                        "extra_envs": envs,
                        # Measure, do not decide: the verdict comes from the pairs, so a per-round KEEP/REVERT here
                        # would be noise promoted to a decision.
                        "keep_threshold_pct": 100.0,
                        "budget_minutes": budget_minutes,
                        "mode": "env_only",
                    },
                    session_dir=self.session_dir,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("forge gemm paired confirmation aborted at %s%d: %s", side, idx, exc)
                break
            tput = float(res.get("new_tput") or 0.0)
            if tput <= 0:
                log.warning("forge gemm paired confirmation: %s%d produced no throughput", side, idx)
                break
            if side == "A":
                pending = tput
            elif pending is not None:
                pairs.append((pending, tput))
                pending = None

        verdict = assess_paired(pairs)
        log.info(
            "forge gemm paired confirmation: %d pair(s) -> %s (median delta %s%%)",
            len(pairs),
            verdict.reason,
            verdict.median_delta_pct,
        )
        return verdict

    def _gemm_apply_verdict(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Did the tuned table reach the server's merge list and get read?"""
        if tuner_name == "fmoe_ck":
            return self._fmoe_apply_verdict(envs)
        from ..measurement.apply_verification import verify_applied

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        logs = _integrate_server_logs(self.session_dir, tuner_name)
        if not logs:
            # Say so.
            log.warning(
                "forge gemm E2E: no server.log under %s (retries included); apply verification cannot run for %s",
                self.session_dir / "runs" / "integrate" / f"integrate-gemm_tune_{tuner_name}",
                tuner_name,
            )
            return None

        # The deployed file is named after the candidate, so the runtime's own table name has to travel with it or the
        # arrival check compares merged_tuned_dense_bf16.csv against bf16_tuned_gemm.csv and concludes the artifact
        # never landed.
        table_names = [name for key in envs if (name := _AITER_ENV_TO_TABLE.get(key))]
        # aiter prints a hit line only under this flag; every serving run now sets it by default, but an operator
        # value in the candidate env wins, and then a zero-hit result means nothing.
        raw_flag = str(envs.get("AITER_LOG_TUNED_CONFIG", "1")).strip().lower()
        hit_logging = raw_flag not in ("", "0", "false", "no", "off")

        try:
            return verify_applied(
                logs[-1],
                csv_paths,
                hit_logging=hit_logging,
                runtime_table_names=table_names,
            ).to_dict()
        except Exception:  # noqa: BLE001 - verification must never fail the run
            log.warning("apply verification failed for %s", tuner_name, exc_info=True)
            return None

    def _fmoe_apply_verdict(
        self,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Apply verdict for ``fmoe_ck`` based on fused-MoE dispatch, not dense GEMM."""
        from ..kernel.gemm_shape_coverage import (
            aiter_log_tuned_config_enabled,
            fmoe_tuned_config_coverage,
            log_has_fused_moe_activity,
            parse_aiter_fused_moe_dispatches,
            read_latest_integrate_server_log,
            resolve_fmoe_candidate_csv,
            tuned_fmoe_csv_rows,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        loaded = read_latest_integrate_server_log(self.session_dir)
        if loaded is None:
            run_dir = self.session_dir / "runs" / "integrate" / "integrate-gemm_tune_fmoe_ck"
            log.warning(
                "forge gemm E2E: no server.log under %s; apply verification cannot run for fmoe_ck",
                run_dir,
            )
            return None
        log_path, log_text = loaded
        hit_logging = aiter_log_tuned_config_enabled(envs)
        candidate_path = resolve_fmoe_candidate_csv(csv_paths[0])
        dispatches = parse_aiter_fused_moe_dispatches(log_text)
        if not dispatches:
            if log_has_fused_moe_activity(log_text):
                return {
                    "verdict": "fused_moe_parse_inconclusive",
                    "hits": 0,
                    "misses": 0,
                    "blocks_keep": False,
                    "conclusive": False,
                    "merged_tables": [],
                    "unmerged_artifacts": list(csv_paths),
                    "detail": ("server.log contains fused-MoE activity but no dispatch lines could be parsed"),
                }
            if not hit_logging:
                return {
                    "verdict": "fused_moe_logging_disabled",
                    "hits": 0,
                    "misses": 0,
                    "blocks_keep": False,
                    "conclusive": False,
                    "merged_tables": [],
                    "unmerged_artifacts": list(csv_paths),
                    "detail": ("AITER_LOG_TUNED_CONFIG is off; fused-MoE dispatch attribution cannot run"),
                }
            return {
                "verdict": "no_fused_moe_dispatch",
                "hits": 0,
                "misses": 0,
                "blocks_keep": True,
                "conclusive": True,
                "merged_tables": [],
                "unmerged_artifacts": list(csv_paths),
                "detail": ("server.log exists but contains no [aiter] [fused_moe] dispatch lines"),
            }

        if candidate_path is None:
            return {
                "verdict": "candidate_csv_missing",
                "hits": 0,
                "misses": len(dispatches),
                "blocks_keep": False,
                "conclusive": False,
                "merged_tables": [],
                "unmerged_artifacts": list(csv_paths),
                "detail": (
                    "env points at a merged fmoe CSV but the sibling bare "
                    "candidate file is absent; cannot attribute runtime "
                    "kernel names to the tuner candidate"
                ),
            }

        candidate_rows = tuned_fmoe_csv_rows(candidate_path)
        coverage = fmoe_tuned_config_coverage(candidate_rows, dispatches)
        covered = int(coverage.get("covered") or 0)
        requested = int(coverage.get("requested") or 0)
        if covered > 0:
            return {
                "verdict": "served",
                "hits": covered,
                "misses": requested - covered,
                "blocks_keep": False,
                "conclusive": True,
                "merged_tables": [],
                "unmerged_artifacts": [],
                "detail": (
                    f"{covered} fused-MoE dispatch(es) match candidate kernelName1/kernelName2 in {candidate_path.name}"
                ),
            }
        if coverage.get("runtime_default"):
            return {
                "verdict": "runtime_default_config",
                "hits": 0,
                "misses": requested,
                "blocks_keep": True,
                "conclusive": True,
                "merged_tables": [],
                "unmerged_artifacts": [],
                "detail": ("runtime served default heuristics; tuned candidate kernels were not selected"),
            }
        if coverage.get("kernel_name_mismatch"):
            return {
                "verdict": "kernel_name_mismatch",
                "hits": 0,
                "misses": requested,
                "blocks_keep": True,
                "conclusive": True,
                "merged_tables": [],
                "unmerged_artifacts": [],
                "detail": (
                    "lookup key matched bundled rows but runtime kernelName1/kernelName2 differ from candidate_fmoe.csv"
                ),
            }
        return {
            "verdict": "no_shape_key_matched",
            "hits": 0,
            "misses": requested,
            "blocks_keep": True,
            "conclusive": True,
            "merged_tables": [],
            "unmerged_artifacts": [],
            "detail": (f"0 of {requested} fused-MoE dispatch(es) resolve to a candidate row"),
        }

    def _merge_gemm_candidate_with_runtime(self, env_var: str, candidate_csv_path: str) -> str | None:
        """Merge a GEMM candidate CSV with the runtime config."""
        import csv
        import importlib.util
        import math
        import re

        def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = [dict(row) for row in reader]
            return fieldnames, rows

        def _read_header(path: Path) -> list[str]:
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle).fieldnames or [])

        candidate_path = Path(candidate_csv_path)
        if not candidate_path.is_file():
            return None

        runtime_filename = _AITER_ENV_TO_TABLE.get(env_var)
        if not runtime_filename:
            return None
        tuned_stem = Path(runtime_filename).stem
        untuned_stem = (
            re.sub(r"(?:_)?tuned$", "_untuned", tuned_stem)
            if re.search(r"(?:_)?tuned$", tuned_stem)
            else tuned_stem.replace("tuned", "untuned")
        )

        config_dirs: list[Path] = []
        explicit_root = os.environ.get("AITER_ROOT_DIR", "").strip()
        if explicit_root:
            root = Path(explicit_root)
            config_dirs.extend((root / "aiter" / "configs", root / "configs"))
        try:
            spec = importlib.util.find_spec("aiter")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            config_dirs.append(Path(spec.origin).resolve().parent / "configs")
        config_dirs.append(_CONTAINER_AITER_CONFIG_DIR)
        config_dirs.extend(sorted(self.session_dir.glob("runs/specialist/*/worktree/aiter/configs")))

        seen_dirs: set[str] = set()
        unique_config_dirs: list[Path] = []
        for config_dir in config_dirs:
            key = str(config_dir)
            if key not in seen_dirs and config_dir.is_dir():
                seen_dirs.add(key)
                unique_config_dirs.append(config_dir)

        try:
            candidate_columns, candidate_rows = _read_csv(candidate_path)
            runtime_cache_dir = Path(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR",
                    "/tmp/aiter_configs",
                )
            )
            runtime_path = runtime_cache_dir / runtime_filename
            source_paths: list[Path] = []
            source_config_dir: Path | None = None
            if runtime_path.is_file():
                source_paths = [runtime_path]
            else:
                for config_dir in unique_config_dirs:
                    base_path = config_dir / runtime_filename
                    model_paths = sorted(
                        path
                        for path in (config_dir / "model_configs").glob(f"*{tuned_stem}*.csv")
                        if path.is_file() and "untuned" not in path.name
                    )
                    paths = ([base_path] if base_path.is_file() else []) + model_paths
                    if paths:
                        source_paths = paths
                        source_config_dir = config_dir
                        break
            if not source_paths:
                log.warning(
                    "gemm E2E: no complete aiter config found for %s; candidate-only validation would be unsafe",
                    env_var,
                )
                return None
            if source_config_dir is None:
                source_config_dir = next(
                    (config_dir for config_dir in unique_config_dirs if (config_dir / f"{untuned_stem}.csv").is_file()),
                    None,
                )

            source_columns: list[list[str]] = []
            source_row_sets: list[list[dict[str, str]]] = []
            for path in source_paths:
                columns, rows = _read_csv(path)
                source_columns.append(columns)
                source_row_sets.append(rows)

            all_columns = list(source_columns[0])
            for columns in [*source_columns[1:], candidate_columns]:
                for column in columns:
                    if column not in all_columns:
                        insert_at = all_columns.index("tflops") if "tflops" in all_columns else len(all_columns)
                        all_columns.insert(insert_at, column)
            fill_defaults = {"xbf16": "0", "run_1stage": "0", "ksplit": "0"}

            def _normalize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
                normalized = []
                for row in rows:
                    new_row = {}
                    for column in all_columns:
                        value = row.get(column)
                        if value is None:
                            value = fill_defaults.get(column, "0")
                        new_row[column] = value
                    normalized.append(new_row)
                return normalized

            runtime_rows: list[dict[str, str]] = []
            for rows in source_row_sets:
                runtime_rows.extend(_normalize(rows))
            candidate_rows = _normalize(candidate_rows)

            key_cols: list[str] = []
            if source_config_dir is not None:
                untuned_path = source_config_dir / f"{untuned_stem}.csv"
                if untuned_path.is_file():
                    untuned_columns = _read_header(untuned_path)
                    key_cols.extend(column for column in untuned_columns if column in all_columns)
            if not key_cols:
                key_cols.extend(column for column in ("M", "N", "K") if column in all_columns)
            for column in ("gfx", "cu_num", "_tag"):
                if column in all_columns and column not in key_cols:
                    key_cols.append(column)
            if not key_cols:
                log.warning(
                    "gemm E2E: cannot derive dispatch keys for %s",
                    env_var,
                )
                return None

            def _dispatch_key(row: dict[str, str]) -> tuple[str, ...]:
                return tuple(row.get(column, "") for column in key_cols)

            def _deduplicate_dispatch_rows(rows: list[dict[str, str]], label: str) -> list[dict[str, str]] | None:
                counts: dict[tuple[str, ...], int] = {}
                for row in rows:
                    key = _dispatch_key(row)
                    counts[key] = counts.get(key, 0) + 1
                if not any(count > 1 for count in counts.values()):
                    return rows
                if "us" not in all_columns:
                    log.warning(
                        "gemm E2E: %s has duplicate dispatch keys for %s but no 'us' column to select the fastest row",
                        label,
                        env_var,
                    )
                    return None

                def _us(row: dict[str, str]) -> float:
                    try:
                        return float(row.get("us", ""))
                    except (TypeError, ValueError):
                        return math.inf

                # Keep the fastest (smallest us) row per dispatch key; ties keep the first row seen (stable), NaN-like
                # values sort last.
                best: dict[tuple[str, ...], dict[str, str]] = {}
                order: list[tuple[str, ...]] = []
                for row in rows:
                    key = _dispatch_key(row)
                    if key not in best:
                        best[key] = row
                        order.append(key)
                    elif _us(row) < _us(best[key]):
                        best[key] = row
                deduplicated = [best[key] for key in order]
                log.info(
                    "gemm E2E: removed %d duplicate %s row(s) for %s",
                    len(rows) - len(deduplicated),
                    label,
                    env_var,
                )
                return deduplicated

            runtime_rows = _deduplicate_dispatch_rows(runtime_rows, "runtime config")
            candidate_rows = _deduplicate_dispatch_rows(candidate_rows, "candidate")
            if runtime_rows is None or candidate_rows is None:
                return None

            # Drop rows from runtime that the candidate improves, then concat.
            candidate_keys = {_dispatch_key(row) for row in candidate_rows}
            kept_from_runtime = [row for row in runtime_rows if _dispatch_key(row) not in candidate_keys]
            merged_rows = kept_from_runtime + candidate_rows

            merged_path = candidate_path.parent / f"merged_{candidate_path.name}"
            with merged_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(merged_rows)
            log.info(
                "gemm E2E: merged %d candidate rows into %d rows from %d aiter config file(s) -> %d total (%s)",
                len(candidate_rows),
                len(runtime_rows),
                len(source_paths),
                len(merged_rows),
                merged_path,
            )
            return str(merged_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("gemm E2E: merge failed (%s); rejecting candidate", exc)
            return None

    def _ck_blockscale_switch_eligible(self, result: dict[str, Any]) -> bool:
        """Whether the fp8 block-scale CK backend switch should be E2E-validated."""
        if not isinstance(result, dict):
            return False
        from ..kernel.request_handlers import _resolve_gemm_tuning_backend

        backend = str(result.get("backend") or _resolve_gemm_tuning_backend({})).strip().lower()
        if backend != "forge":
            return False
        framework = str(getattr(self.shared_state, "framework", "") or "").strip().lower()
        if framework != "sglang":
            return False
        if not self._ck_switch_precision_is_fp8(result):
            return False

        from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type
        from ..actions.executors._workload_envs import _GFX942_GPU_TYPES

        gpu = _resolve_amd_gpu_type(getattr(self.shared_state, "gpu_type", "") or "")
        if gpu not in _GFX942_GPU_TYPES:
            return False

        # Block-scale fp8 only, asserted positively via ``weight_block_size``.
        from hyperloom.inference_optimizer.model_config_utils import _fp8_is_block_scale

        model_path = str(getattr(self.shared_state, "model_path", "") or os.environ.get("MODEL_PATH", ""))
        return _fp8_is_block_scale(model_path)

    def _ck_switch_precision_is_fp8(self, result: dict[str, Any]) -> bool:
        """Whether the workload runs fp8, resolved from any available signal."""
        if str(getattr(self.shared_state, "precision", "") or "").strip().lower() == "fp8":
            return True
        if isinstance(result, dict) and str(result.get("precision") or "").strip().lower() == "fp8":
            return True
        try:
            from ..kernel.request_handlers import _resolve_forge_precision_and_quant

            precision, _ = _resolve_forge_precision_and_quant(self.shared_state, {})
            if str(precision or "").strip().lower() == "fp8":
                return True
        except Exception:  # noqa: BLE001 - best-effort runtime resolution
            pass
        return False

    def _sync_profile_state_after_gemm_roofline(self, result: dict[str, Any]) -> None:
        """Merge a handler-owned Roofline fallback into the live Coordinator state."""
        shape_capture = result.get("shape_capture") if isinstance(result, dict) else None
        if not isinstance(shape_capture, dict) or shape_capture.get("capture_mode") != "block_fp8_profile":
            return
        source_trace = str(shape_capture.get("source_profile_trace") or "").strip()
        if not source_trace:
            return
        from copy import deepcopy

        from ..state.shared_state import SharedState

        persisted = SharedState.load_or_init(self.session_dir)
        persisted_trace = str((persisted.last_trace_analyze or {}).get("steady_state_trace") or "").strip()
        if persisted_trace != source_trace:
            log.warning(
                "GEMM Roofline state sync skipped: persisted steady trace %r does not match result %r",
                persisted_trace,
                source_trace,
            )
            return
        for field_name in (
            "last_profile_trace",
            "last_profile_status",
            "last_profile_args",
            "last_profile_workload",
            "last_profile_workload_action",
            "last_trace_analyze",
            "roofline_snapshots",
            "baseline_eager_fallback",
        ):
            setattr(
                self.shared_state,
                field_name,
                deepcopy(getattr(persisted, field_name)),
            )
        # Lifecycle is append-only telemetry owned by both states; union it so neither the inline Roofline's rows nor
        # the live state's are dropped.
        self.shared_state.merge_lifecycle_events(persisted.lifecycle)

    async def _handle_gemm_tuning_result(self, result: dict[str, Any]) -> None:
        """Record and post-process a run_gemm_tuning result from any entrypoint."""
        self._sync_profile_state_after_gemm_roofline(result)
        self.shared_state.record_gemm_tuning(result)
        try:
            await self._validate_gemm_tuning_e2e(result)
        except Exception as exc:  # noqa: BLE001
            # Validation spans server restarts, log parsing and CSV merges, and is reached from two entrypoints that
            # only guard the tuning call itself.
            log.exception("gemm E2E validation raised; recording it as a fault")
            e2e = result.setdefault("e2e_results", {})
            if isinstance(e2e, dict):
                faults = e2e.setdefault("faults", [])
                if isinstance(faults, list):
                    faults.append(
                        {
                            "tuner": "*",
                            "error_class": "e2e_validation_exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            # The bridge stamped KEEP + the raw combined env on the micro result; the normal exit rewrites both so
            # Orchestration never bundles an integrate against an unmeasured candidate.
            result["decision"] = "REVERT"
            result["requires_e2e_validation"] = False
            result["e2e_validated"] = False
            result["micro_decision"] = "e2e_validation_exception"
            for stale in ("recommended_env", "extra_envs"):
                if result.get(stale):
                    result[stale] = {}
            # ``record_gemm_tuning`` above stored a SHALLOW COPY, so the scalar rewrites just made
            # (decision/micro_decision/...) do not reach the recorded entry on their own -- only the normal exit
            # re-syncs it.
            self._replace_latest_gemm_tuning_attempt(result)
        self.shared_state.save(self.session_dir)

    def _journal_gemm_tuning_keep(
        self,
        entry: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """Mirror an adopted GEMM-tuning stack entry as an optimization_journal KEEP row."""
        try:
            journal = self._ensure_journal()
            variant_name = str(entry.get("variant_name") or "gemm_tuning")
            backend = str(entry.get("backend") or "").strip().lower()
            try:
                tput = float(entry["tput"]) if entry.get("tput") is not None else None
            except (TypeError, ValueError):
                tput = None
            try:
                gain_pct = float(entry["gain_pct"]) if entry.get("gain_pct") is not None else None
            except (TypeError, ValueError):
                gain_pct = None
            metrics: dict[str, Any] = {}
            if entry.get("tuned_file"):
                metrics["tuned_file"] = str(entry.get("tuned_file"))
            journal.append_entry(
                JournalEntry(
                    phase=self._journal_entry_phase(),
                    iter=int(self.shared_state.tick or 0),
                    kind=KIND_GEMM_TUNING,
                    change=variant_name,
                    outcome=OUTCOME_KEEP,
                    gain_pct=gain_pct,
                    throughput_after=tput,
                    task_id=str(task_id or ""),
                    variant_name=variant_name,
                    ts=str(entry.get("ts") or ""),
                    provenance=f"gemm_tuning:{backend}" if backend else "gemm_tuning",
                    tick=int(self.shared_state.tick or 0),
                    metrics=metrics,
                )
            )
        except Exception:  # noqa: BLE001 — journaling is best-effort
            log.exception("gemm_tuning journal append failed")

    def _writeback_gemm_result_json(self, entry: dict[str, Any]) -> None:
        """Overwrite ``<workspace>/result.json`` with the E2E-adjudicated envelope."""
        workspace = str(entry.get("workspace") or "").strip()
        if not workspace:
            return
        path = Path(workspace) / "result.json"
        try:
            if not path.parent.is_dir():
                return
            atomic_write_json(path, entry, make_parents=False)
        except (OSError, TypeError, ValueError):
            log.warning("gemm result.json writeback failed for %s", path, exc_info=True)

    def _replace_latest_gemm_tuning_attempt(self, result: dict[str, Any]) -> None:
        """Sync the latest GEMM history row, and publish the verdict to disk."""
        if not isinstance(result, dict):
            return
        entry = dict(result)
        attempts = list(getattr(self.shared_state, "gemm_tuning_attempts", []) or [])
        if attempts and isinstance(attempts[-1], dict):
            entry.setdefault("ts", attempts[-1].get("ts"))
            attempts[-1] = entry
        else:
            entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
            attempts.append(entry)
        self.shared_state.gemm_tuning_attempts = attempts
        self.shared_state.last_gemm_tuning = entry
        self._writeback_gemm_result_json(entry)

    @staticmethod
    def _gemm_canonical_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Read the per-tuner candidates the producer already decided on."""
        rows = result.get("candidates")
        if not isinstance(rows, list):
            return []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_env = row.get("env")
            envs = (
                {str(key): str(value) for key, value in raw_env.items() if str(key).strip() and str(value).strip()}
                if isinstance(raw_env, dict)
                else {}
            )
            if not envs:
                # Nothing to apply, so nothing an e2e run could validate.
                continue
            # The singular pair is what the CK-switch dedup and the promote path read; it is only unambiguous for a
            # single-variable candidate.
            env_var, env_value = next(iter(envs.items())) if len(envs) == 1 else ("", "")
            candidates.append(
                {
                    "tuner": str(row.get("tuner") or "unknown"),
                    "env_var": env_var,
                    "env_value": env_value,
                    "envs": envs,
                    "micro_speedup": _as_float(row.get("best_micro_speedup"), 1.0),
                }
            )
        return candidates

    def _gemm_e2e_candidates(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Reduce a GEMM tuning result to the env sets worth E2E-validating."""
        candidates = self._gemm_canonical_candidates(result)
        # The rebuild's status gate cannot see a forced candidate, so it runs only when the producer named none: the
        # pre-``candidates[]`` envelope and GEAK.
        for t in [] if candidates else (result.get("tuners_run") or []):
            if not isinstance(t, dict):
                continue
            # partial_output is a real artifact: the tuner wrote fewer rows than shapes it was given (the grouped
            # batch budget ran out), but the rows it did write are deployable.
            if t.get("status") not in ("ok", "partial_output"):
                continue
            # improved_shapes can never exceed 0 for tuners with no comparable baseline -- TunableOp never times the
            # untuned dispatch, the candidate-CSV fallback has no per-shape Pre/Post table, and a hipblaslt-only bf16
            # run has no torch candidate to measure against.
            if (
                not bool(t.get("candidate"))
                and int(t.get("improved_shapes") or 0) <= 0
                and int(t.get("unverified_shapes") or 0) <= 0
            ):
                continue
            env_var = str(t.get("env_var") or "").strip()
            env_value = str(t.get("env_value") or "").strip()
            raw_envs = t.get("env_vars") or {}
            envs = (
                {str(key): str(value) for key, value in raw_envs.items() if str(key).strip() and str(value).strip()}
                if isinstance(raw_envs, dict)
                else {}
            )
            if env_var and env_value:
                envs.setdefault(env_var, env_value)
            if envs:
                candidates.append(
                    {
                        "tuner": t.get("tuner") or "unknown",
                        "env_var": env_var,
                        "env_value": env_value,
                        "envs": envs,
                        "micro_speedup": float(t.get("best_micro_speedup") or 1.0),
                    }
                )

        if not candidates and str(result.get("backend") or "").strip().lower() != "forge":
            tuned_file = str(result.get("tuned_file") or "").strip()
            try:
                micro_speedup = float(result.get("best_speedup") or 0.0)
            except (TypeError, ValueError):
                micro_speedup = 0.0
            keeps = str(result.get("decision") or "").strip().upper() == "KEEP" and str(
                result.get("status") or ""
            ).strip().lower() in {"ok", "complete", "completed", "succeeded", "success"}
            if tuned_file and micro_speedup > 1.0 and keeps:
                env_var = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
                candidates.append(
                    {
                        "tuner": "a8w8_blockscale_tuned_gemm",
                        "env_var": env_var,
                        "env_value": tuned_file,
                        "envs": {env_var: tuned_file},
                        "micro_speedup": micro_speedup,
                    }
                )

        # Standalone fp8 block-scale CK backend switch: inject as its own candidate so the loop E2E-validates baseline
        # Triton vs CK.
        if self._ck_blockscale_switch_eligible(result):
            if not any(c.get("env_var") == "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" for c in candidates):
                candidates.append(
                    {
                        "tuner": "ck_blockscale_backend_switch",
                        "env_var": "SGLANG_FP8_BLOCKSCALE_CK_MAX_M",
                        "env_value": "256",
                        "envs": {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"},
                        "micro_speedup": 1.0,
                    }
                )
        return candidates

    async def _validate_gemm_tuning_e2e(self, result: dict[str, Any]) -> None:
        """Sequentially E2E-validate each tuning candidate's env independently."""
        from ..kernel.request_handlers import integrate_handler
        from hyperloom.common.model_paths import resolve_session_model_path

        backend = str(result.get("backend") or "geak").strip().lower()
        candidates = self._gemm_e2e_candidates(result)
        if not candidates:
            log.info("gemm tuning: no candidates to E2E validate")
            # Close the books here too.
            result["decision"] = "REVERT"
            result["requires_e2e_validation"] = False
            result["e2e_validated"] = False
            # Only when the tuners left no verdict of their own: ``micro_decision`` is a routing key downstream, not a
            # label, so an existing one stands.
            if not str(result.get("micro_decision") or "").strip():
                result["micro_decision"] = "no_e2e_candidates"
            # ``recommended_env``/``extra_envs`` stay as the tuners left them: they are the raw record of what was
            # produced, and the eligibility checks downstream already read them as "must be empty".
            self._replace_latest_gemm_tuning_attempt(result)
            return

        baseline_tput = float(self.shared_state.baseline_tput or 0.0)
        running_tput = float((self.shared_state.current_best or {}).get("tput") or baseline_tput)
        stacked_envs: dict[str, str] = {}
        kept: list[dict[str, Any]] = []
        reverted: list[dict[str, Any]] = []
        faults: list[dict[str, Any]] = []
        # Set by the last KEEP; the attempt row claims this exact string.
        adopted_tuned_file = ""
        try:
            from ..actions.executors.explore import _compute_explore_variant_timeout

            per_tuner_timeout_sec = _compute_explore_variant_timeout(
                baseline_runtime_sec=float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0),
                kill_ratio=float(getattr(self.shared_state, "explore_overtime_kill_ratio", 1.5) or 1.5),
            )
        except Exception:  # noqa: BLE001 - conservative fallback
            per_tuner_timeout_sec = 15 * 60
        per_tuner_budget_minutes = max(1, int((per_tuner_timeout_sec + 59) // 60))

        # fmoe_ck is only meaningful with --moe-runner-backend aiter, and aiter's CK fused-MoE rejects a
        # non-128-aligned intermediate_size_per_partition.
        from hyperloom.inference_optimizer.cli.model_gate import (
            model_supports_aiter_ck_fused_moe,
        )

        # ``_runtime_uses_aiter_fused_moe`` resolves the serving log -- which now byte-scans the whole runs/ tree for
        # aiter evidence -- and then reads it whole, ~17MB on the fleet.
        triton_moe_inert = any(c.get("tuner") == "vllm_moe_triton" for c in candidates) and await asyncio.to_thread(
            self._runtime_uses_aiter_fused_moe
        )

        for cand in candidates:
            tuner_name = cand["tuner"]
            if tuner_name == "vllm_moe_triton" and triton_moe_inert:
                log.warning(
                    "gemm E2E: skipping %s — the server dispatches MoE through "
                    "aiter, which never reads VLLM_TUNED_CONFIG_FOLDER, so the tuned "
                    "Triton config cannot take effect",
                    tuner_name,
                )
                reverted.append({**cand, "reason": "aiter_moe_runtime_triton_config_inert"})
                continue
            if tuner_name == "fmoe_ck" and not model_supports_aiter_ck_fused_moe(
                str(getattr(self.shared_state, "model_path", "") or ""),
                int(getattr(self.shared_state, "tp", 0) or 0),
            ):
                log.info(
                    "gemm E2E: skipping %s — aiter CK fused-MoE cannot serve "
                    "this model at tp=%s (intermediate size is not 128-aligned)",
                    tuner_name,
                    getattr(self.shared_state, "tp", 0),
                )
                reverted.append({**cand, "reason": "aiter_ck_moe_shape_unsupported"})
                continue
            # Merge candidate CSV with the runtime config so that shapes NOT in the candidate keep their existing
            # tuned entries.
            env = dict(cand["envs"])
            merge_failure_reason = ""
            merge_failure_env = ""
            for env_var, env_value in list(env.items()):
                if not env_var.startswith("AITER_CONFIG"):
                    continue
                if not Path(env_value).is_file():
                    merge_failure_reason = "candidate_artifact_missing"
                    merge_failure_env = env_var
                    break
                merged_path = self._merge_gemm_candidate_with_runtime(
                    env_var,
                    env_value,
                )
                if merged_path:
                    env[env_var] = merged_path
                else:
                    merge_failure_reason = "complete_aiter_config_unavailable"
                    merge_failure_env = env_var
                    break
            if merge_failure_reason:
                log.error(
                    "gemm E2E: refusing aiter candidate for %s (%s: %s)",
                    tuner_name,
                    merge_failure_env,
                    merge_failure_reason,
                )
                reverted.append(
                    {
                        **cand,
                        "reason": merge_failure_reason,
                        "failed_env_var": merge_failure_env,
                    }
                )
                continue
            extra_server_args = (
                "--moe-runner-backend aiter"
                if tuner_name == "fmoe_ck"
                and str(getattr(self.shared_state, "framework", "") or "").lower() == "sglang"
                else ""
            )
            # Merge with previously KEEP'd envs.
            test_envs = dict(stacked_envs)
            test_envs.update(env)

            log.info(
                "gemm E2E: validating tuner=%s env=%s (base_tput=%.1f)",
                tuner_name,
                cand["env_var"],
                running_tput,
            )

            from ..state.kernel_decision_settings import _MAX_INTEGRATE_FAULT_ATTEMPTS

            integrate_verdict: dict[str, Any] | None = None
            run_stopped = False
            integrate_payload = {
                "task_id": f"gemm_tune_e2e_{tuner_name}",
                "kernel_id": f"gemm_tune_{tuner_name}",
                "source": "forge_gemm_tuning",
                "base_tput": running_tput,
                "model_path": resolve_session_model_path(
                    state_model_path=str(getattr(self.shared_state, "model_path", "") or ""),
                    for_serving=True,
                ),
                "extra_server_args": extra_server_args,
                "extra_envs": test_envs,
                "keep_threshold_pct": 3.0,
                "budget_minutes": per_tuner_budget_minutes,
                "mode": "env_only",
            }
            for fault_attempt in range(1, _MAX_INTEGRATE_FAULT_ATTEMPTS + 1):
                try:
                    integrate_result = await integrate_handler(
                        integrate_payload,
                        session_dir=self.session_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    if fault_attempt < _MAX_INTEGRATE_FAULT_ATTEMPTS:
                        log.warning(
                            "gemm E2E: integrate raised for %s (fault attempt %d/%d): %s",
                            tuner_name,
                            fault_attempt,
                            _MAX_INTEGRATE_FAULT_ATTEMPTS,
                            exc,
                        )
                        continue
                    log.warning(
                        "gemm E2E: integrate raised for %s: %s",
                        tuner_name,
                        exc,
                    )
                    faults.append(
                        {
                            **cand,
                            "reason": "integrate_fault:handler_exception",
                            "fault": True,
                            "error_class": "handler_exception",
                            "error": repr(exc),
                            "fault_attempts": fault_attempt,
                        }
                    )
                    break

                stopped = stopped_by_the_run_class(integrate_result.get("error_class"))
                if stopped is not None:
                    log.info(
                        "gemm E2E: %s left unmeasured — %s",
                        tuner_name,
                        stopped.interrupted,
                    )
                    run_stopped = True
                    break

                if self.shared_state._is_integrate_fault(integrate_result):
                    error_class = str(integrate_result.get("error_class") or "integrate_fault").strip()
                    if fault_attempt < _MAX_INTEGRATE_FAULT_ATTEMPTS:
                        log.warning(
                            "gemm E2E: retrying tuner=%s after integrate fault %s (attempt %d/%d)",
                            tuner_name,
                            error_class,
                            fault_attempt,
                            _MAX_INTEGRATE_FAULT_ATTEMPTS,
                        )
                        continue
                    log.warning(
                        "gemm E2E: tuner=%s integrate fault (%s) — unmeasured, not a REVERT verdict",
                        tuner_name,
                        error_class,
                    )
                    faults.append(
                        {
                            **cand,
                            "reason": f"integrate_fault:{error_class}",
                            "fault": True,
                            "error_class": error_class,
                            "integrate_status": integrate_result.get("status"),
                            "error": integrate_result.get("error"),
                            "fault_attempts": fault_attempt,
                        }
                    )
                    break

                integrate_verdict = integrate_result
                break

            if run_stopped:
                break
            if integrate_verdict is None:
                continue

            decision = str(integrate_verdict.get("decision") or "").upper()
            new_tput = float(integrate_verdict.get("new_tput") or 0.0)
            gain_pct = float(integrate_verdict.get("gain_pct") or 0.0)

            log.info(
                "gemm E2E: tuner=%s decision=%s new_tput=%.1f gain=%.2f%%",
                tuner_name,
                decision,
                new_tput,
                gain_pct,
            )

            # Two independent ways the artifact can fail to take effect, neither of which the throughput delta can
            # see: the keys are unreachable (coverage), and the table never reached the server (apply verdict).
            apply_blockers: list[str] = []

            # Off the event loop for the same reason: this reads the integrate run's server.log in full and parses
            # every tuned CSV named in the candidate env.
            coverage = await asyncio.to_thread(self._gemm_tuned_config_coverage, tuner_name, env)
            if coverage is not None:
                cand = {**cand, "tuned_config_coverage": coverage}
                if not coverage.get("artifact_applied") and coverage.get("conclusive", True):
                    apply_blockers.append(str(coverage.get("not_applied_reason") or "no_shape_key_matched"))
                    log.error(
                        "gemm E2E: tuner=%s produced an artifact the runtime never "
                        "applied — 0 of %d requested shape(s) resolve to a tuned row; "
                        "the %.2f%% e2e delta measures nothing about the tuning",
                        tuner_name,
                        coverage.get("requested") or 0,
                        gain_pct,
                    )
                elif not coverage.get("artifact_applied"):
                    log.info(
                        "gemm E2E: tuner=%s tuned-config coverage inconclusive — %s",
                        tuner_name,
                        coverage.get("not_applied_reason"),
                    )
                else:
                    log.info(
                        "gemm E2E: tuner=%s tuned-config coverage %.2f%% (%d/%d shapes)",
                        tuner_name,
                        coverage.get("coverage_pct") or 0.0,
                        coverage.get("covered") or 0,
                        coverage.get("requested") or 0,
                    )

            applied = self._gemm_apply_verdict(tuner_name, env)
            if applied is not None:
                cand = {**cand, "apply_verdict": applied}
                if applied.get("blocks_keep"):
                    apply_blockers.append(str(applied.get("verdict") or "not_applied"))
                    log.error(
                        "forge gemm E2E: tuner=%s apply verdict=%s — %s",
                        tuner_name,
                        applied.get("verdict"),
                        applied.get("detail"),
                    )
                elif not applied.get("conclusive"):
                    # "Cannot tell" is not "did not apply": hit lines need AITER_LOG_TUNED_CONFIG=1, and treating
                    # their absence as a failure would revert every arm that ran without it.
                    log.info(
                        "forge gemm E2E: tuner=%s apply verdict=%s (not conclusive) — %s",
                        tuner_name,
                        applied.get("verdict"),
                        applied.get("detail"),
                    )

            if decision == "KEEP" and new_tput > running_tput and not apply_blockers:
                stacked_envs.update(env)
                running_tput = new_tput
                kept.append(
                    {
                        **cand,
                        "envs": dict(env),
                        "tput": new_tput,
                        "gain_pct": gain_pct,
                    }
                )
                # The one place this path names its artifact.
                adopted_tuned_file = _candidate_tuned_file(env, cand.get("env_var", ""))

                lifted = self._lift_to_current_best(
                    "gemm_tuning",
                    new_tput,
                    {
                        "name": f"{backend}_{tuner_name}",
                        "candidate_extra_server_args": extra_server_args,
                        "extra_envs": dict(env),
                        "source_phase": "KERNEL_AGENT",
                        **graded_axes_of(result),
                        "workspace": result.get("workspace"),
                    },
                    entry_extra={
                        "tuned_file": adopted_tuned_file,
                        "gain_pct": gain_pct,
                        "backend": backend,
                        "source": "kernel_entry_auto",
                    },
                )
                if lifted:
                    self._journal_gemm_tuning_keep(
                        self.shared_state.optimization_stack[-1],
                        task_id=f"gemm_tune_e2e_{tuner_name}",
                    )
            else:
                reason = f"decision={decision}, gain={gain_pct:.2f}%"
                if apply_blockers:
                    # Distinguish "the tuning did not pay off" from "the tuned artifact was never reachable", which is
                    # a wiring defect.
                    reason = f"tuned_config_never_applied[{'+'.join(apply_blockers)}] ({reason})"
                reverted.append({**cand, "reason": reason})

        # The watermark covers the whole run, so it waits for the last KEEP.
        if kept:
            total_gain = (running_tput - baseline_tput) / baseline_tput * 100.0 if baseline_tput > 0 else 0.0
            # One end-to-end measurement is not enough on this fleet: three rounds of a single unchanged configuration
            # spanned 58%.
            paired = await self._confirm_gemm_gain_paired(
                stacked_envs,
                baseline_tput=baseline_tput,
                budget_minutes=per_tuner_budget_minutes,
                extra_server_args=("--moe-runner-backend aiter" if "AITER_CONFIG_FMOE" in stacked_envs else ""),
            )
            if baseline_tput > 0:
                self._update_cumulative_gain_validated(
                    running_tput,
                    result,
                    source="forge_gemm_tuning_e2e",
                    measurement_basis=_paired_measurement_basis(paired),
                )
            # Name the artifact this run adopted, so the breakdown can tell it was.
            if adopted_tuned_file:
                result["tuned_file"] = adopted_tuned_file
            log.info(
                "gemm E2E: %d tuners KEEP (total gain=+%.2f%%), %d REVERT",
                len(kept),
                total_gain,
                len(reverted),
            )
        elif faults:
            stacked_envs = {}
            total_gain = 0.0
            log.info(
                "gemm E2E: %d tuner(s) hit integrate fault(s), no E2E verdict",
                len(faults),
            )
        else:
            stacked_envs = {}
            total_gain = 0.0
            log.info(
                "gemm E2E: all %d tuners REVERT, no E2E gain",
                len(reverted),
            )

        # Rewrite the stored result to the E2E-validated outcome so Orchestration never sees the raw combined
        # recommended_env and issues a bundled integrate.
        result["e2e_results"] = {"kept": kept, "reverted": reverted, "faults": faults}
        result["recommended_env_raw"] = dict(result.get("recommended_env") or {})
        result["extra_envs_raw"] = dict(result.get("extra_envs") or {})
        result["recommended_env"] = dict(stacked_envs)
        result["extra_envs"] = dict(stacked_envs)
        if faults and not kept and not reverted:
            result["e2e_gain_pct"] = None
        else:
            result["e2e_gain_pct"] = round(float(total_gain), 4)
        result["e2e_validated"] = True
        result["requires_e2e_validation"] = False
        if kept:
            result["status"] = "complete"
            result["decision"] = "KEEP"
        elif reverted:
            result["status"] = "complete"
            result["decision"] = "REVERT"
            result["micro_decision"] = "candidate_no_e2e_gain"
        elif faults:
            result["status"] = "failed"
            result["decision"] = "REVERT"
            result["micro_decision"] = "integrate_fault"
        else:
            result["status"] = "complete"
            result["decision"] = "REVERT"
            result["micro_decision"] = "candidate_no_e2e_gain"
        self._replace_latest_gemm_tuning_attempt(result)

    async def _finish_kernel_entry(self) -> None:
        """Run the gated kernel lanes, write the handoff, and delegate rewrite control."""
        await self._maybe_reprofile_for_kernel()
        await self._maybe_run_forge_fusion_before_kernel_opt()
        await self._maybe_run_collective_before_kernel_opt()
        from hyperloom.inference_optimizer.session.session_paths import (
            next_forge_attempt_dir,
        )

        # One fresh directory per entry rather than per macro cycle: the controller refuses an output root it has
        # already initialized, and the handoff rides inside it so each attempt keeps the evidence it was given.
        attempt_dir = next_forge_attempt_dir(
            self.session_dir,
            int(getattr(self.shared_state, "macro_cycle", 0) or 0),
        )
        handoff_dir = attempt_dir / "handoff"
        try:
            from ..kernel.forge_handoff import write_forge_handoff

            try:
                env_spec = self.build_env_spec()
            except Exception:  # noqa: BLE001
                log.exception("KERNEL entry: could not build Forge serving environment")
                env_spec = {}
            handoff_dir = write_forge_handoff(
                self.session_dir,
                self.shared_state,
                env_spec=env_spec,
                handoff_dir=handoff_dir,
            )
            log.info("KERNEL entry: wrote Forge handoff to %s", handoff_dir)
        except Exception:  # noqa: BLE001
            log.exception("KERNEL entry: Forge handoff generation failed")
        await self._run_kernel_rewrite_controller(handoff_dir, attempt_dir)

    async def _run_kernel_rewrite_controller(self, handoff_dir: Path, output_dir: Path) -> None:
        """Run one Controller attempt without preselecting operators."""
        from hyperloom.common.inline_step_heartbeat import inline_step_heartbeat

        from ..kernel.controller_submit import (
            record_controller_llm_usage,
            run_controller_subprocess,
        )

        cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        controller_budget_sec, hard_timeout_sec = self._kernel_rewrite_controller_timeouts()

        def _stamp(when: float) -> None:
            self.shared_state.kernel_inline_step_seen_unix = when

        def _clear() -> None:
            self.shared_state.kernel_inline_step_seen_unix = 0.0

        # The heartbeat spans patch integration as well as the subprocess.
        async with inline_step_heartbeat(
            stamp=_stamp,
            interval_sec=_phase_state.KERNEL_HEARTBEAT_SEC,
            clear=_clear,
        ):
            if controller_budget_sec <= 0 or hard_timeout_sec <= 0:
                result = {
                    "status": "no_result",
                    "reason": "no KERNEL phase budget remains for the rewrite controller",
                    "patch_count": 0,
                    "task_count": 0,
                    "output_dir": str(output_dir),
                }
            else:
                try:
                    result = await asyncio.to_thread(
                        run_controller_subprocess,
                        handoff_dir=handoff_dir,
                        output_dir=output_dir,
                        budget_minutes=controller_budget_sec / 60.0,
                        hard_timeout_sec=hard_timeout_sec,
                    )
                except Exception as error:  # noqa: BLE001
                    log.exception("KERNEL entry: kernel rewrite controller failed")
                    result = {
                        "status": "failed",
                        "reason": f"controller invocation failed: {error}",
                        "patch_count": 0,
                        "task_count": 0,
                        "output_dir": str(output_dir),
                    }

            result = {
                **result,
                "macro_cycle": cycle,
                "handoff_dir": str(handoff_dir),
                "budget_minutes": controller_budget_sec / 60.0,
                "hard_timeout_sec": hard_timeout_sec,
            }
            # The Controller cannot reach this ledger from its own process, so its forge-loops' spend is filed here
            # now that the child has exited.
            record_controller_llm_usage(result=result, session_dir=self.session_dir)
            if int(result.get("patch_count") or 0) > 0:
                try:
                    from ..kernel.controller_patch_integration import (
                        integrate_controller_patches,
                    )

                    integration = await integrate_controller_patches(
                        patches_root=str(result.get("patches_root") or output_dir / "result" / "patches"),
                        session_dir=self.session_dir,
                        shared_state=self.shared_state,
                    )
                    result["integration"] = integration.to_dict()
                except Exception as error:  # noqa: BLE001
                    log.exception("KERNEL entry: Controller patch integration failed")
                    result["integration"] = {
                        "status": "failed",
                        "reason": str(error),
                        "kept_count": 0,
                    }
            else:
                result["integration"] = {
                    "status": "not_run",
                    "reason": "Controller published no patches",
                    "kept_count": 0,
                    "reverted_count": 0,
                    "skipped_count": 0,
                }
        self.shared_state.kernel_optimizer = "forge"
        self.shared_state.kernel_rewrite_controller_result = result
        # The summary rides a ``response`` message the inbox dumps raw once.
        _integration = result.get("integration")
        if isinstance(_integration, dict):
            _skipped = [
                str(r.get("reason") or "")
                for r in (_integration.get("results") or [])
                if isinstance(r, dict) and str(r.get("status") or "").startswith("skipped")
            ]
            _status = str(_integration.get("status") or "")
            if _status in {"failed", "no_patch_admitted"} or _skipped:
                self.shared_state.record_action_failure(
                    action="kernel_rewrite_controller",
                    task_id=str(result.get("run_id") or f"forge-cycle-{cycle}"),
                    result={
                        "error_class": _status or "patches_skipped",
                        "error": "; ".join(x for x in ([str(_integration.get("reason") or "")] + _skipped) if x)[:800],
                    },
                )
        self.shared_state.set_pending_escalate_hint(
            _phase_state.ESCALATE_HINT_SKIP_TO_SWEEP,
        )
        self.shared_state.save(self.session_dir)
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
                priority=1,
            )
        )

    def _fusion_required_before_kernel_opt(self) -> bool:
        """Gate the forge-fusion step in KERNEL entry."""
        import os

        if str(os.environ.get("HYPERLOOM_SKIP_FUSION", "")).strip().lower() in ("1", "true", "yes", "on"):
            return False
        framework = str(getattr(self.shared_state, "framework", "") or "sglang").strip().lower()
        if framework not in ("sglang", "vllm", "vllm-aiter"):
            return False
        trace = str(getattr(self.shared_state, "last_profile_trace", "") or "").strip()
        if not trace:
            log.info("KERNEL entry: skip forge-fusion (no decode trace yet)")
            return False
        last = getattr(self.shared_state, "last_fusion", None)
        if isinstance(last, dict) and str(last.get("status") or "").strip() in ("ok", "complete", "kept"):
            # A round that kept nothing and left targets unfunded answers only for the ones it ran, so it re-arms
            # fusion until the retry cap is spent.
            if not last.get("kept") and _withheld_targets(last) > 0:
                return _as_int(getattr(self.shared_state, "fusion_withheld_retries", 0)) < MAX_FUSION_WITHHELD_RETRIES
            return False
        if isinstance(last, dict) and last.get("infrastructure_abort"):
            # An abort judged nothing, so it must stay retryable -- but not forever.
            spent = _as_int(getattr(self.shared_state, "fusion_infra_aborts", 0))
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

    #: Exposed communication below this share of E2E is not worth a tuning round.
    COLLECTIVE_COMM_PCT_FLOOR = 1.0
    #: Floor for the fallback share, which counts one kernel's whole GPU time
    #: rather than the exposed part of all communication. A collective the
    #: compute overlaps entirely still scores here, so the bar is higher.
    COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR = 3.0

    def _collective_only_mode(self) -> bool:
        """Return whether KERNEL should run only the Collective lane."""
        state_value = getattr(
            self.shared_state,
            "collective_only_mode",
            False,
        )
        if not isinstance(state_value, bool):
            raise ValueError("collective_only_mode must be boolean")
        return state_value or str(os.environ.get("HYPERLOOM_COLLECTIVE_ONLY", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _collective_required_before_kernel_opt(self) -> bool:
        """Return whether the current trace warrants a collective campaign."""
        if str(os.environ.get("HYPERLOOM_SKIP_COLLECTIVE", "")).strip().lower() in ("1", "true", "yes", "on"):
            return False
        tp = getattr(self.shared_state, "tp", 0)
        if isinstance(tp, bool) or not isinstance(tp, int):
            raise ValueError("Collective TP must be an integer")
        if tp <= 1:
            return False
        analysis = getattr(self.shared_state, "last_trace_analyze", None)
        if analysis in (None, {}):
            log.info("KERNEL entry: skip collective (no trace analysis yet)")
            return False
        if not isinstance(analysis, dict):
            raise ValueError("last_trace_analyze must be a mapping")
        comm_pct, comm_source = _collective_comm_share(self.shared_state)
        if comm_pct is None:
            log.info(
                "KERNEL entry: skip collective (no roofline comm share, and no "
                "source-resolved collective candidate to fall back on)",
            )
            return False
        floor = (
            self.COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR
            if comm_source == "candidate_gpu_pct"
            else self.COLLECTIVE_COMM_PCT_FLOOR
        )
        if comm_pct < floor:
            log.info(
                "KERNEL entry: skip collective (comm share %.2f%% from %s < %.2f%% floor)",
                comm_pct,
                comm_source,
                floor,
            )
            return False
        last = getattr(self.shared_state, "last_collective", None)
        if last is not None and not isinstance(last, dict):
            raise ValueError("last_collective must be a mapping")
        if last:
            status = str(last.get("status") or "").strip()
            if status in ("ok", "complete", "kept"):
                return False
            if status == "skipped":
                from ..kernel.request_handlers import collective_analysis_key

                if str(last.get("analysis_key") or "") == collective_analysis_key(self.shared_state):
                    return False
        return True

    async def _maybe_run_collective_before_kernel_opt(self) -> None:
        """Run or resume collective optimization before kernel_opt."""
        if _phase_state.collective_integration_pending(self.shared_state):
            last = self.shared_state.last_collective
            await self._integrate_collective(last)
        elif self._collective_required_before_kernel_opt():
            await self._run_forge_collective()
        if self._collective_only_mode() and not (_phase_state.collective_integration_pending(self.shared_state)):
            self.shared_state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
            self.shared_state.save(self.session_dir)

    async def _run_forge_collective(self) -> None:
        """Tune the hottest rewritable multi-GPU collective during KERNEL entry."""
        log.info("KERNEL entry: running collective tuning (multi-GPU comm kernel)")
        try:
            from ..kernel.request_handlers import run_collective_handler

            result = await run_collective_handler(
                {"task_id": "kernel_entry_collective", "reason": "kernel_entry_auto"},
                session_dir=self.session_dir,
            )
        except Exception as exc:  # noqa: BLE001 - preserve a durable lane verdict
            log.exception("KERNEL entry collective tuning failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "engine": "forge_collective",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self._handle_collective_result(result)

    async def _handle_collective_result(self, result: dict | None) -> None:
        """Record a collective run, publish it, and integrate a validated patch."""
        if not isinstance(result, dict):
            raise TypeError("Collective handler result must be a mapping")
        recorded = dict(result)
        kept = recorded.setdefault("kept", False)
        requires_e2e = recorded.setdefault(
            "requires_e2e_validation",
            False,
        )
        if not isinstance(kept, bool) or not isinstance(requires_e2e, bool):
            raise ValueError("Collective handler E2E flags must be boolean")
        if kept != requires_e2e:
            raise ValueError("Collective handler E2E flags are inconsistent")
        if not str(recorded.get("collective_attempt_id") or "").strip():
            recorded["collective_attempt_id"] = _derive_collective_attempt_id(recorded)
        if kept:
            recorded["patch_cleanup_status"] = "pending"
            if not str(recorded.get("integration_id") or "").strip():
                seed = recorded["collective_attempt_id"] + ":" + str(recorded.get("patch") or "")
                recorded["integration_id"] = (
                    "collective-integration-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
                )
        self.shared_state.record_collective(recorded, self.session_dir)
        self._record_collective_timeline(recorded)
        log.info(
            "collective tuning: status=%s decision=%s speedup=%s kernel=%s",
            result.get("status"),
            result.get("decision"),
            result.get("kernel_speedup"),
            result.get("kernel_name"),
        )
        status = str(result.get("status") or "unknown")
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent",
                    "orchestration",
                    "response",
                    {
                        "in_reply_to": "",
                        "kind": "run_collective_done",
                        "status": status,
                        "result": recorded,
                        "source": "kernel_entry_auto",
                    },
                    priority=1,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to post run_collective_done bus message")
        if kept:
            await self._integrate_collective(recorded)

    def _record_collective_timeline(self, recorded: dict[str, Any]) -> None:
        """Record the settled collective campaign on the kernel timeline.

        A collective KEEP is withheld until the E2E gate rules on it, and a
        campaign the gate rejects never reaches the stack ledger. Recording the
        run here is what keeps it auditable: this is the moment the campaign's
        own verdict is final, whatever integrate later decides.

        Args:
            recorded: The settled campaign, as handed to ``record_collective``.
        """
        recorder = self._kernel_timeline()
        if recorder is None:
            return
        speedup = recorded.get("kernel_speedup")
        gain_pct = (
            (speedup - 1.0) * 100.0 if isinstance(speedup, (int, float)) and not isinstance(speedup, bool) else None
        )
        withheld = bool(recorded.get("requires_e2e_validation"))
        try:
            recorder.record_collective_run(
                run_id=str(recorded.get("collective_attempt_id") or ""),
                status=str(recorded.get("status") or ""),
                op=str(recorded.get("collective_op") or ""),
                world_size=recorded.get("world_size"),
                gain_pct=gain_pct,
                withheld=withheld,
                withhold_reason="pending_e2e_validation" if withheld else "",
                micro_decision=str(recorded.get("decision") or ""),
                ended_at=str(recorded.get("ts") or ""),
                duration_sec=recorded.get("duration_sec"),
                failure_reason=str(recorded.get("error_class") or ""),
            )
        except Exception:  # noqa: BLE001 — observability cannot change kernel behavior
            log.debug("kernel timeline: collective run record failed", exc_info=True)

    async def _run_collective_integration(
        self,
        result: dict,
        inputs: "_collective_recovery.IntegrationInputs",
        *,
        preapplied: dict,
        backup_root: Path,
        apply_checkpoint: Path,
    ) -> dict:
        """Run the E2E integrate round, or describe why it could not run."""
        from ..kernel.request_handlers import (
            integrate_handler,
            materialize_unified_patch_snapshot,
        )

        patch = inputs.patch
        target_file = inputs.target_file

        def _failed(error_class: str, error: str) -> dict:
            return {
                "status": "failed",
                "decision": "REVERT",
                "error_class": error_class,
                "error": error,
                "patch_path": patch,
                "target_file": target_file,
                "apply_result": preapplied or {},
            }

        if not patch or not target_file:
            return _failed(
                "collective_patch_missing",
                "collective KEEP is missing patch or target_file",
            )

        snapshot_dir = str(result.get("snapshot_dir") or "").strip()
        if not snapshot_dir and patch.endswith(".patch") and inputs.kernel_repo:
            try:
                snapshot_dir = await asyncio.to_thread(
                    materialize_unified_patch_snapshot,
                    patch_path=patch,
                    repo_root=inputs.kernel_repo,
                    snapshot_dir=Path(patch).parent / "collective_snapshot",
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("KERNEL entry collective snapshot materialization failed")
                return _failed(exc.__class__.__name__, repr(exc))

        try:
            keep_threshold = float(os.environ.get("HYPERLOOM_COLLECTIVE_KEEP_PCT", "1.0"))
            if not math.isfinite(keep_threshold) or keep_threshold < 0:
                raise ValueError("HYPERLOOM_COLLECTIVE_KEEP_PCT must be finite and non-negative")
            integ = await integrate_handler(
                {
                    "task_id": "collective_e2e",
                    "kernel_id": "forge_collective",
                    "source": "forge_collective",
                    "patch_path": patch,
                    "target_file": target_file,
                    "kernel_repo": inputs.kernel_repo,
                    "snapshot_dir": snapshot_dir,
                    "backup_root": str(backup_root),
                    "apply_checkpoint_path": str(apply_checkpoint),
                    "preapplied_apply_result": preapplied,
                    "extra_envs": inputs.extra_envs,
                    "defer_patch_finalize": True,
                    "integration_id": inputs.integration_id,
                    "keep_threshold_pct": keep_threshold,
                },
                session_dir=self.session_dir,
            )
            if not isinstance(integ, dict):
                raise TypeError("Collective integration result must be a mapping")
            return integ
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry collective integrate failed")
            return _failed(exc.__class__.__name__, repr(exc))

    async def _settle_collective_integration(
        self,
        integ: dict,
        *,
        apply_checkpoint: Path,
        backup_root: Path,
        integration_id: str,
        recovery_uncertain: bool,
    ) -> str:
        """Resolve the decision and finish the revert a non-KEEP owes."""
        from ..kernel.request_handlers import _maybe_revert_kernel_patch

        apply_result = integ.get("apply_result")
        manifest_path = str(apply_result.get("manifest_path") or "").strip() if isinstance(apply_result, dict) else ""
        if not manifest_path and apply_checkpoint.is_file():
            try:
                apply_result, _manifest_status = _collective_recovery.load_apply_checkpoint(
                    apply_checkpoint,
                    backup_root,
                )
                integ["apply_result"] = apply_result
            except Exception as exc:  # noqa: BLE001
                recovery_uncertain = True
                integ.update(
                    {
                        "status": "failed",
                        "decision": "NEEDS_REVIEW",
                        "error_class": "collective_apply_checkpoint_invalid",
                        "error": repr(exc),
                    }
                )

        decision = str(integ.get("decision") or "").strip().upper()
        if decision not in {"KEEP", "REVERT", "NEEDS_REVIEW"}:
            integ.update(
                {
                    "status": "failed",
                    "decision": "NEEDS_REVIEW",
                    "error_class": "collective_integration_decision_invalid",
                    "error": f"Invalid integration decision: {decision!r}",
                }
            )
            decision = "NEEDS_REVIEW"
            recovery_uncertain = True
        integ["integration_id"] = integration_id

        apply_result = integ.get("apply_result")
        if not isinstance(apply_result, dict):
            apply_result = {}
            integ["apply_result"] = apply_result
        manifest_path = str(apply_result.get("manifest_path") or "").strip()
        if decision == "KEEP":
            return decision

        revert_result = integ.get("revert_result")
        if manifest_path and not _collective_recovery.patch_lifecycle_complete(revert_result):
            integ["revert_result"] = await asyncio.to_thread(
                _maybe_revert_kernel_patch,
                apply_result,
            )
        revert_complete = not manifest_path or _collective_recovery.patch_lifecycle_complete(integ.get("revert_result"))
        integration_complete = revert_complete and not recovery_uncertain
        integ["patch_cleanup_status"] = "complete" if integration_complete else "recovery_required"
        integ["patch_cleanup_action"] = "" if integration_complete else "revert"
        return decision

    async def _integrate_collective(self, result: dict) -> None:
        """Apply a collective patch and adopt it only after an E2E KEEP."""
        from ..kernel.request_handlers import (
            _maybe_finalize_kernel_patch,
            _maybe_revert_kernel_patch,
        )
        from hyperloom.inference_optimizer.session.session_paths import patches_dir

        inputs = _collective_recovery.validate_integration_inputs(
            result,
            self.shared_state,
        )
        integration_id = inputs.integration_id
        current_envs = inputs.extra_envs
        patch_root = patches_dir(
            self.session_dir,
            "forge_collective_" + hashlib.sha256(integration_id.encode("utf-8")).hexdigest()[:16],
        )
        backup_root = patch_root / "backup"
        apply_checkpoint = patch_root / "apply_checkpoint.json"
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        recovered = await _collective_recovery.recover_apply_state(
            result,
            checkpoint=apply_checkpoint,
            backup_root=backup_root,
            patch=inputs.patch,
            target_file=inputs.target_file,
        )
        integ = recovered.integ
        if integ is None:
            integ = await self._run_collective_integration(
                result,
                inputs,
                preapplied=recovered.preapplied,
                backup_root=backup_root,
                apply_checkpoint=apply_checkpoint,
            )
        decision = await self._settle_collective_integration(
            integ,
            apply_checkpoint=apply_checkpoint,
            backup_root=backup_root,
            integration_id=integration_id,
            recovery_uncertain=recovered.uncertain,
        )
        apply_result = integ["apply_result"]

        state_snapshot = {
            "optimization_stack": list(self.shared_state.optimization_stack or []),
            "gain_per_stack_entry": list(self.shared_state.gain_per_stack_entry or []),
            "current_best": dict(self.shared_state.current_best or {}),
            "cumulative_gain_validated": (self.shared_state.cumulative_gain_validated),
            "cumulative_gain_validated_ts": (self.shared_state.cumulative_gain_validated_ts),
            "cumulative_gain_validated_stack_len": (self.shared_state.cumulative_gain_validated_stack_len),
        }
        if decision == "KEEP":
            try:
                self._promote_collective_integrate_keep(
                    result,
                    integ,
                    extra_envs=current_envs,
                )
            except Exception as exc:  # noqa: BLE001
                for field, value in state_snapshot.items():
                    setattr(self.shared_state, field, value)
                revert_result = await asyncio.to_thread(
                    _maybe_revert_kernel_patch,
                    apply_result,
                )
                revert_complete = _collective_recovery.patch_lifecycle_complete(revert_result)
                revert_action = "" if revert_complete else "revert"
                integ.update(
                    {
                        "status": "failed",
                        "decision": "REVERT",
                        "error_class": "collective_promotion_invalid",
                        "error": repr(exc),
                        "revert_result": revert_result,
                        "patch_cleanup_status": ("complete" if revert_complete else "recovery_required"),
                        "patch_cleanup_action": revert_action,
                    }
                )
                decision = "REVERT"

        gain = integ.get("gain_pct")
        log.info(
            "KERNEL entry: collective integrate decision=%s gain_pct=%s",
            decision,
            gain,
        )
        if decision == "KEEP":
            integ["patch_cleanup_status"] = "recovery_required"
            integ["patch_cleanup_action"] = "finalize"
        try:
            self.shared_state.record_collective_integration(
                integ,
                self.session_dir,
                integration_id=integration_id,
            )
        except Exception:
            if decision == "KEEP":
                for field, value in state_snapshot.items():
                    setattr(self.shared_state, field, value)
            raise

        if decision == "KEEP":
            finalize_result = integ.get("finalize_result")
            # Settled, not complete: an already-finalized manifest must not be finalized again even when its sweep was
            # partial.
            if not _collective_recovery.patch_finalize_settled(finalize_result):
                finalize_result = await asyncio.to_thread(
                    _maybe_finalize_kernel_patch,
                    apply_result,
                )
                integ["finalize_result"] = finalize_result
            finalize_complete = _collective_recovery.patch_finalize_settled(finalize_result)
            finalize_action = "" if finalize_complete else "finalize"
            integ["patch_cleanup_status"] = "complete" if finalize_complete else "recovery_required"
            integ["patch_cleanup_action"] = finalize_action
            self.shared_state.record_collective_integration(
                integ,
                self.session_dir,
                integration_id=integration_id,
            )

        if integ["patch_cleanup_status"] == "complete":
            apply_checkpoint.unlink(missing_ok=True)
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent",
                    "orchestration",
                    "response",
                    {
                        "in_reply_to": "",
                        "kind": "collective_integrate_done",
                        "status": integ.get("status", "failed"),
                        "decision": decision,
                        "gain_pct": gain,
                        "result": integ,
                        "source": "kernel_entry_auto",
                    },
                    priority=1,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to post collective_integrate_done bus message")

    def _promote_collective_integrate_keep(
        self,
        collective_result: dict,
        integrate_result: dict,
        *,
        extra_envs: dict[str, str] | None = None,
    ) -> None:
        """Promote an E2E-validated Collective KEEP through the current_best lift."""
        if not isinstance(collective_result, dict) or not isinstance(integrate_result, dict):
            raise TypeError("Collective promotion inputs must be mappings")
        if str(integrate_result.get("decision") or "").strip().upper() != "KEEP":
            return
        if str(integrate_result.get("status") or "").strip().lower() != "ok":
            raise ValueError("Collective KEEP requires a successful integration")
        apply_result = integrate_result.get("apply_result")
        if (
            not isinstance(apply_result, dict)
            or apply_result.get("status") != "ok"
            or not str(apply_result.get("manifest_path") or "").strip()
        ):
            raise ValueError("Collective KEEP is missing an apply manifest")
        new_tput_raw = integrate_result.get("new_tput")
        incremental_gain_raw = integrate_result.get("gain_pct")
        baseline_tput_raw = self.shared_state.baseline_tput
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (
                new_tput_raw,
                incremental_gain_raw,
                baseline_tput_raw,
            )
        ):
            raise ValueError("Collective KEEP is missing numeric E2E measurements")
        try:
            new_tput = float(new_tput_raw)
            incremental_gain = float(incremental_gain_raw)
            baseline_tput = float(baseline_tput_raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Collective KEEP is missing numeric E2E measurements") from exc
        if not math.isfinite(new_tput) or new_tput <= 0:
            raise ValueError("Collective KEEP new_tput must be positive")
        if not math.isfinite(incremental_gain) or incremental_gain <= 0:
            raise ValueError("Collective KEEP gain_pct must be positive")
        if not math.isfinite(baseline_tput) or baseline_tput <= 0:
            raise ValueError("Collective KEEP baseline_tput must be positive")

        patch = str(collective_result.get("patch") or integrate_result.get("patch_path") or "").strip()
        if not patch:
            raise ValueError("Collective KEEP is missing patch_path")
        integration_id = str(
            collective_result.get("integration_id") or integrate_result.get("integration_id") or ""
        ).strip()
        if not integration_id:
            raise ValueError("Collective KEEP is missing integration_id")
        if not isinstance(self.shared_state.optimization_stack, list):
            raise ValueError("optimization_stack must be a list")
        existing = {
            str(item.get("patch_path") or "")
            for item in (self.shared_state.optimization_stack or [])
            if isinstance(item, dict) and item.get("action") == "collective"
        }
        if patch in existing:
            return
        envs = dict(extra_envs or integrate_result.get("extra_envs") or {})
        extra_args = str(integrate_result.get("extra_server_args") or "")
        lifted = self._lift_to_current_best(
            "collective",
            new_tput,
            {
                "name": "forge_collective",
                "candidate_extra_server_args": extra_args,
                "extra_envs": envs,
                "source_phase": "KERNEL_AGENT",
                "provenance": "forge_collective",
                **graded_axes_of(integrate_result.get("bench_result") or integrate_result),
                "workspace": integrate_result.get("workspace"),
            },
            entry_extra={
                "backend": "forge",
                "engine": "forge_collective",
                "source": "kernel_entry_auto",
                "integration_id": integration_id,
                "kernel_id": str(collective_result.get("kernel_id") or ""),
                "kernel_name": str(collective_result.get("kernel_name") or ""),
                "gain_pct": incremental_gain,
                "patch_path": patch,
                "target_file": collective_result.get("source_file") or integrate_result.get("target_file"),
                "kernel_speedup": collective_result.get("kernel_speedup"),
                "gpu_pct": collective_result.get("gpu_pct"),
                "collective_op": collective_result.get("collective_op"),
                "world_size": collective_result.get("world_size"),
            },
        )
        if not lifted:
            return
        ts = datetime.now(timezone.utc).isoformat()
        self._update_cumulative_gain_validated(
            new_tput,
            integrate_result,
            source="collective_promote",
            ts=ts,
        )

    async def _run_forge_fusion(self) -> None:
        """Run autonomous kernel fusion during KERNEL entry."""
        log.info("KERNEL entry: running forge-fusion (autonomous kernel fusion)")
        try:
            from ..kernel.request_handlers import run_fusion_handler

            result = await run_fusion_handler(
                {"task_id": "kernel_entry_fusion", "reason": "kernel_entry_auto"},
                session_dir=self.session_dir,
            )
        except Exception as exc:  # noqa: BLE001
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
        status = str(result.get("status") or "unknown") if isinstance(result, dict) else "failed"
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
            spent = _as_int(getattr(self.shared_state, "fusion_infra_aborts", 0))
            try:
                self.shared_state.fusion_infra_aborts = spent + 1
            except Exception:  # noqa: BLE001 - state shape tolerant, as below
                pass
        if isinstance(result, dict) and not result.get("kept") and _withheld_targets(result) > 0:
            # Counted on the session for the same reason as the aborts above.
            withheld_spent = _as_int(getattr(self.shared_state, "fusion_withheld_retries", 0))
            try:
                self.shared_state.fusion_withheld_retries = withheld_spent + 1
            except Exception:  # noqa: BLE001 - state shape tolerant, as below
                pass
        try:
            if isinstance(result, dict) and not str(result.get("fusion_run_id") or "").strip():
                cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
                result["fusion_run_id"] = f"fusion-c{cycle}-{time.time_ns():x}"
            self.shared_state.last_fusion = result if isinstance(result, dict) else {"status": status}
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 - state shape tolerant (best-effort idempotency record)
            pass
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
                    priority=1,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to post run_fusion_done bus message")
        # A KEPT fusion is handed to integrate for the e2e re-baseline decision.
        if isinstance(result, dict) and result.get("kept") and result.get("requires_e2e_validation"):
            await self._integrate_fusion(result)

    async def _integrate_fusion(self, result: dict) -> None:
        """Queue every KEPT forge-fusion sibling for the shared e2e integrate lane."""
        import os

        from ..kernel._kernel_decisions import enqueue_nominated_patch
        from ..kernel.nomination_result import parse_outcome

        from ..kernel.request_handlers import _summarize_dropped_patches

        outcome = parse_outcome(result)
        # Counted before the empty check: an all-refused envelope and a run that kept nothing are the same ``queued``
        # figure and differ only in reasons.
        refused = _summarize_dropped_patches(outcome.dropped)
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
        try:
            keep_pct = float(os.environ.get("HYPERLOOM_FUSION_KEEP_PCT", "3.0"))
        except (TypeError, ValueError):
            keep_pct = 3.0
        queued = 0
        for patch in outcome.patches:
            record = enqueue_nominated_patch(
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
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 - best-effort persist; drain reloads state
            pass

    def _current_tput_from_validated_gain(self) -> float:
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
        snaps = getattr(self.shared_state, "roofline_snapshots", None) or []
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
        try:
            last_rl = float(state.last_roofline_tput or 0.0)
        except (TypeError, ValueError):
            last_rl = 0.0
        if (state.auto_roofline_pending_task_id or "").strip():
            return False
        if last_rl <= 0:
            try:
                failure_streak = int(getattr(state, "roofline_failure_streak", 0) or 0)
            except (TypeError, ValueError):
                failure_streak = 0
            if failure_streak <= 0:
                return False
            if failure_streak > _MAX_ROOFLINE_FAILURE_RETRIES:
                return False
            try:
                last_rl = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                last_rl = 0.0
            if last_rl <= 0:
                return False
        cur = self._current_tput_from_validated_gain()
        if cur <= 0:
            return False
        return cur / last_rl >= _resolve_roofline_watermark_ratio()

    async def _release_finished_roofline_gate(self) -> None:
        """Drop an in-flight marker that names a roofline which already finished."""
        pending = (self.shared_state.auto_roofline_pending_task_id or "").strip()
        if not pending:
            return
        try:
            task = await self.tasks.get(pending)
        except Exception:  # noqa: BLE001 — a missing row is itself finished
            task = None
        if task is not None and str(getattr(task, "state", "")) not in TERMINAL_STATES:
            return
        self.shared_state.auto_roofline_pending_task_id = ""
        log.info(
            "watermark-roofline: released in-flight gate held by finished task=%s",
            pending,
        )

    async def _maybe_enqueue_watermark_roofline(
        self,
        *,
        reason: str,
    ) -> bool:
        """Enqueue a fresh roofline if the watermark crossed; idempotency-keyed via ``reason``, stamps auto_roofline_pending_task_id. Returns True when enqueued."""
        await self._release_finished_roofline_gate()
        if not self._needs_roofline_for_watermark():
            return False
        try:
            task = await self._enqueue_internal_analysis_task(reason=reason)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "watermark-roofline (%s): failed to enqueue: %r",
                reason,
                exc,
            )
            return False
        self.shared_state.auto_roofline_pending_task_id = task.task_id
        log.info(
            "watermark-roofline (%s): enqueued task=%s (cur=%.2f, last_roofline=%.2f, ratio>=%.2f)",
            reason,
            task.task_id,
            self._current_tput_from_validated_gain(),
            float(self.shared_state.last_roofline_tput or 0.0),
            self._ROOFLINE_WATERMARK_RATIO,
        )
        return True

    def _cached_kernel_request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return a cached programmatic_handler result if applicable (cache key last_trace_analyze)."""
        if kind != "trace_analyze":
            return None
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
