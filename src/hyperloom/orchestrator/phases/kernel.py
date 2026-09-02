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
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    LEVER_KERNEL,
)
from ..kernel import collective_recovery as _collective_recovery
from ..actions.stop_attribution import stopped_by_the_run_class
from ..kernel._recorder_trace import trace_recording_skipped
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
    _resolve_roofline_watermark_ratio,
    _accepted_config_as_variant,
    _resolve_serving_fidelity,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)

# Last-resort location of the aiter checkout inside the standard serving
# container. Module-level (not an inline literal) so tests can point it at a
# non-existent path and exercise the "no complete aiter config anywhere" branch
# on a developer box that happens to have the real checkout mounted.
_CONTAINER_AITER_CONFIG_DIR = Path("/sgl-workspace/aiter/aiter/configs")

# How much of the route-level lift must survive the share the per-kernel ledger
# already claims before the residual is worth recording as its own attempt.
# 0.1% is measurement noise, and a noise-sized keep in the gain ledger reads as
# an optimization that never happened.
_GEAK_RESIDUAL_MIN_RATIO = 1.001

# How many times a session re-runs forge-fusion after it aborted on
# infrastructure (no git workspace, harness could not be authored). Such a run
# judged nothing, so reporting it as a result would be wrong and it has to stay
# retryable -- but the causes do not all heal mid-session, and a retry re-runs
# LLM discovery before failing in the same place. Two is one free recovery from
# a transient cause plus the original attempt.
MAX_FUSION_INFRA_RETRIES = 2


def _as_int(value: object) -> int:
    """Read a counter that round-tripped through JSON, defaulting to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# Which table each aiter config env var is resolved under at serving time. Two
# callers need it: the merge step, which has to find the runtime table to merge
# our candidate into, and the apply check, which has to recognise our artifact
# in the runtime's own lookup lines (the deployed file carries the candidate's
# name, not the table's). They were separate copies until one of them was
# almost edited alone -- and a name that drifts reads as "the artifact never
# arrived", which reverts a candidate that was fine.
#
# A third copy lives in KernelForge's TUNER_ENV_VARS and cannot be shared
# across repositories; ``test_aiter_env_table_matches_kernelforge`` asserts the
# two agree wherever forge is importable.
#
# Note AITER_CONFIG_GEMM_A4W4, not the "_BLOCKSCALE" variant: aiter reads
# fp4/mxfp4 (gfx950-only) configs under that name (jit/core.py), and the
# suffixed key was a dead one that silently dropped every tuned fp4 GEMM.
_AITER_ENV_TO_TABLE: dict[str, str] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
    "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
}


def _safe_mtime(path: Path) -> float:
    """Return ``path``'s mtime, or ``0`` when it cannot be read.

    Sorting server logs by mtime races the round that is still writing them, and
    an ``exists()`` guard does not close the window. Ordering is a heuristic for
    picking the newest log, so a vanished file is worth sorting last rather than
    aborting the check that owns it.
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _candidate_tuned_file(env: Any, env_var: str) -> str:
    """Return the tuned artifact a candidate's env points at.

    One KEEP is described by three different path strings -- the durable copy in
    aiter's config tree, the tuner-workspace original, and the E2E merge product
    -- so an attempt row cannot re-derive the one the stack ends up holding, and
    reconstructing it matched none of them: every forge KEEP read as unadopted.

    Reading the newest stack entry back is not the way out either. The stack
    append is skipped when ``(action, variant_name)`` already matches, and a GEMM
    variant is named ``<backend>_<tuner>`` -- so a second macro cycle re-tuning
    the same tuner finds its entry present, appends nothing, and the newest entry
    is the previous round's. The attempt would then claim that round's artifact
    along with its gain: the same misreport as before, inverted.

    Both the stack entry and the attempt row take the value from here, which
    makes them the same string by construction rather than by lookup.
    """
    if not isinstance(env, dict):
        return ""
    value = env.get(env_var)
    if value in (None, ""):
        value = next((v for v in env.values() if v not in (None, "")), "")
    return str(value or "")


def _paired_measurement_basis(verdict: Any) -> str:
    """How the promoted gain was measured, so the ledger cannot overstate it.

    A gain from ``base_tput`` (measured earlier) against ``new_tput`` (measured
    now) is a comparison of two *blocks*, and drift between them is folded into
    the result. Recording that distinction is what lets a reader tell a
    confirmed number from a plausible one; without it both arrive as
    ``e2e_rebench`` and look equally solid.
    """
    if verdict is None:
        return "e2e_rebench_unpaired"
    if getattr(verdict, "candidate_wins", False):
        return "e2e_paired"
    return f"e2e_paired_{getattr(verdict, 'reason', 'unknown')}"


def _collective_comm_share(state: Any) -> tuple[float | None, str]:
    """Return the communication share gating the lane, and its provenance.

    ``current_comm_pct`` reads the roofline snapshot, whose exposed-comm bucket
    comes from the TraceLens internal extension (``TRACELENS_INTERNAL_ROOT``).
    A checkout without it has no such value, which would disable the whole lane
    behind nothing but a log line, so fall back to the hottest source-resolved
    collective's own GPU-time share. That share is the weaker signal -- it
    counts one kernel rather than all exposed communication -- but the trace
    always carries it.
    """
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
    """Compute the stable identity for one logical Collective campaign.

    This mints the value; readers take it off the record instead of recomputing.

    ``workspace`` is deliberately excluded: every attempt gets a fresh
    ``attempt-<time_ns>`` directory, so hashing it would make the identity a
    timestamp and a replayed or salvaged campaign would never deduplicate.
    """
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
    """Map a 2b decline reason to the status left on ``geak_pending``.

    ``rebench_unavailable`` is not reused here, because the two states are
    different facts that lead to different actions. ``rebench_unavailable``
    means the rebench never got to run -- a scheduling or dispatch problem, and
    the candidate should be retried. A decline means the rebench was refused on
    purpose and the GEAK-harness fallback did not rescue it. When the refusal
    was the overlay, retrying changes nothing: the kernel cannot install.

    Collapsing the two would overwrite a live diagnostic. The field is already
    in use across the campaign and carries only two error strings, so a third
    meaning folded into it is unreadable.

    The status is derived from the reason rather than hardcoded, so a future
    ``geak_harness`` fallback with a different cause does not silently inherit
    the overlay label.

    Args:
        decline_reason (Any): ``reason`` from the 2b dispatcher's summary.

    Returns:
        str: ``"overlay_unloadable"`` when the overlay was the refusal,
        ``"rebench_declined"`` for every other refusal.
    """
    reason = str(decline_reason or "").strip().lower()
    return "overlay_unloadable" if reason == "geak_overlay_unloadable" else "rebench_declined"


class KernelPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    @staticmethod
    def _serving_config_signature(serving_config: Any) -> str:
        """Stable identity string for a ``serving_config`` sub-dict, or '' when empty.

        Reuses the exact ``serving_config`` shape built by
        ``SharedState.profile_workload_context`` so the reprofile gate and the
        recorded-trace workload are normalized identically (no second, drifting
        copy of the args/env rules).
        """
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
        """Whether the latest trace predates the current backend/config.

        Compares the current serving-config signature against the one implied by
        the recorded profile workload (``last_profile_workload['serving_config']``,
        written by the roofline executor and writeback). This is apples-to-apples:
        it never confuses the plain ``last_profile_args`` string with a config
        signature, so a trace already profiled under the current config does not
        spuriously force a kernel-entry reprofile.
        """
        if not signature:
            return False
        recorded = getattr(self.shared_state, "last_profile_workload", None)
        if not isinstance(recorded, Mapping) or not recorded:
            # No workload recorded for the trace: defer to
            # _profile_workload_changed, which owns the "stale trace with no
            # workload metadata" decision.
            return False
        previous = self._serving_config_signature(recorded.get("serving_config"))
        return previous != signature

    def _profile_workload_changed(self) -> bool:
        """Whether the latest trace predates the active serving workload.

        Compares only the fields that identify the profiled workload. The rest
        of the context records how the profile task was parameterized, and the
        two writers disagree there by construction: the roofline path records
        through ``record_profile_workload(task_params)`` and fills them, while
        the kernel-entry path records through ``profile_workload_context()`` and
        leaves them empty. A whole-dict comparison therefore reported a change
        on every first entry -- costing a full re-profile and a second TraceLens
        pass, roughly fifty minutes, with the serving configuration provably
        unchanged -- and then stopped reporting one, because the re-profile it
        forced had rewritten the record in the other writer's shape.

        The serving configuration is not compared here; that is
        :meth:`_profile_config_changed`, which reads it from ``current_best`` on
        both sides and is symmetric for the same reason this now is.
        """
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
        """Reprofile inline when projected tput diverges from the last measured trace, so GEAK targets the live bottleneck."""
        before = self._last_measured_roofline_tput()
        cur = self._current_tput_from_validated_gain()
        profile_signature = self._current_profile_config_signature()
        config_changed = self._profile_config_changed(profile_signature)
        workload_changed = self._profile_workload_changed()
        if cur <= 0:
            return
        # With a measured trace, reprofile only on a material gain or a change in
        # what is being measured. Backend/env changes invalidate shapes even at
        # equal tput, so a staleness signal is a reason to reprofile: a missed
        # reprofile silently points GEAK at a bottleneck that no longer exists.
        #
        # Staleness is judged from the recorded serving config, NOT from
        # ``current_profile_workload_context()``: that context derives its
        # runtime fields from the profile task params while the recorded
        # ``serving_config`` derives them from ``current_best``, so the two
        # disagree whenever a profile was recorded without runtime params -- and
        # the gate would then reprofile on every entry, forever.
        if (
            before > 0
            and abs(cur - before) / before < self._REPROFILE_CHANGE_TOL
            and not config_changed
            and not workload_changed
        ):
            return
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
        try:
            reprofile_task = await self._enqueue_internal_analysis_task(
                reason=f"kernel_entry_g{stack_len}_{profile_fingerprint}"
            )
            # An idempotent reuse can return a task that already reached a
            # terminal state (its snapshot from a prior cycle is still valid).
            # run_task would then attempt succeeded->running -> IllegalTransition,
            # so reuse the existing snapshot instead of re-running.
            if str(getattr(reprofile_task, "state", "")) in TERMINAL_STATES:
                log.info(
                    "kernel-entry reprofile reuses terminal analysis task (state=%s); GEAK targets existing snapshot",
                    reprofile_task.state,
                )
                return
            await self.run_task_registered(reprofile_task)
        except Exception:  # noqa: BLE001 — never block GEAK on a reprofile failure
            log.exception("kernel-entry reprofile failed; GEAK proceeds on existing snapshot")
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
            # Record the workload (incl. serving_config) that this trace reflects;
            # _profile_config_changed derives the config signature from it, so
            # last_profile_args stays the plain-args field it is everywhere else.
            self.shared_state.last_profile_workload = self.shared_state.profile_workload_context()
            self.shared_state.save(self.session_dir)
        else:
            log.warning("kernel-entry reprofile produced no new snapshot; GEAK targets existing trace")

    def _geak_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the GEAK e2e optimizer.

        ``KERNEL_OPT_BACKEND_ORDER`` is the only source of truth: anything
        other than an exact ``forge`` leaves GEAK owning the whole phase.
        """
        from ..kernel.request_handlers import geak_selected

        return geak_selected()

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """Run deterministic KERNEL-entry optimization and re-profile gates.

        Args:
            from_phase: The phase being left, used only for logging.
        """
        if not self._kernel_enabled():
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False (from=%s)",
                from_phase or "<unknown>",
            )
            return
        collective_only = self._collective_only_mode()
        geak_enabled = False if collective_only else self._geak_enabled()
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            selected = "geak" if geak_enabled else "kernel_agent_forge"
            instrument.record_kernel_strategy_selection(
                self.session_dir,
                selected_strategy=selected,
                actual_path=selected,
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
            )
            if not geak_enabled:
                instrument.record_native_kernel_run_start(
                    self.session_dir,
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    payload={
                        "kernel_optimizer": str(getattr(self.shared_state, "kernel_optimizer", "") or ""),
                        "from_phase": from_phase,
                    },
                )
        except Exception:  # noqa: BLE001
            log.debug("kernel v4 strategy selection recording failed", exc_info=True)
        if collective_only:
            self.shared_state.collective_only_mode = True
            self.shared_state.save(self.session_dir)
            await self._maybe_reprofile_for_kernel()
            await self._maybe_run_collective_before_kernel_opt()
            return
        if geak_enabled:
            # GEAK owns the whole KERNEL_AGENT phase: one in-process e2e run
            # seeded with the best config so far, then hand straight to SWEEP.
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

            # The fp8 -> bf16 dense retry now lives inside the tuner router: an
            # fp8 run whose tuning comes back empty runs the bf16 dense pass in
            # the same call (router selects it as a fallback). There is no longer
            # a second subprocess to resume or launch here.
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
    def _resolve_bench_protocol(recipe_path: str) -> dict[str, Any]:
        """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

        Reads the materialized baseline recipe's ``benchmark.envs`` (falling back
        to the process env) and returns only the keys that resolve, so absent
        values leave GEAK on its standalone defaults. Never raises.
        """
        envs: dict[str, Any] = {}
        try:
            import yaml

            if recipe_path and Path(recipe_path).is_file():
                cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
                envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
        except Exception:  # noqa: BLE001
            log.warning("bench_protocol: could not read recipe %r", recipe_path, exc_info=True)
            envs = {}

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
        """Resolve the GEAK e2e timeouts from the live run budget.

        The KERNEL_AGENT phase-entry hook runs GEAK synchronously, so the run is
        capped to always finish with at least the closing-grace window left, and
        the runner's own budget is shrunk by a safety margin on top of that.

        Returns:
            tuple[int, int, bool]: ``(runner_timeout_s, kill_timeout_s,
            budget_known)``. ``runner_timeout_s`` is passed to the runner as its
            own e2e budget; ``kill_timeout_s`` is the hard subprocess kill
            (always ≤ remaining − closing_grace so the closing report can run).
            ``budget_known`` is ``False`` only when no run deadline is set
            (e.g. a unit test invoking the hook directly), where the env default
            is used verbatim.
        """
        # Standalone fallback ONLY: the 12h (43200s) default applies when no run
        # deadline is set (budget_known=False). A Hyperloom-driven run sources the
        # budget from the live deadline / phase allocation instead.
        env_default_timeout = int(os.environ.get("GEAK_E2E_TIMEOUT_S", "43200"))
        deadline = self._run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline - time.monotonic()
        grace = self.shared_state.closing_reserve_sec()
        margin = float(os.environ.get("GEAK_BUDGET_MARGIN_S", "300"))
        # Reserve the closing window: kill the subprocess with at least ``grace`` left.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget:
        # cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(
            self.shared_state,
            budget_pct=self._phase_budget_pct,
        )
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # The runner self-stops ``margin`` before the hard subprocess kill, which
        # reserves the closing-grace window.
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run).

        Builds a handoff from the best config so far, runs the GEAK
        runner out-of-process (it owns all Claude-SDK / Workflow detail),
        records the optimized launch/bench scripts + throughput into state, then
        signals SWEEP via the ``skip_to_sweep`` escalate hint.
        """
        state = self.shared_state
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="runner_started",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result={"from_phase": from_phase},
                status="running",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 start recording failed", exc_info=True)
        cb = state.current_best or {}
        accepted_flags = str(cb.get("extra_server_args") or "")
        extra_envs = cb.get("extra_envs") or {}
        accepted_env = " ".join(f"{k}={v}" for k, v in dict(extra_envs).items())
        workload = {
            "isl": int(getattr(state, "isl", 0) or int(os.environ.get("ISL", "1024"))),
            "osl": int(getattr(state, "osl", 0) or int(os.environ.get("OSL", "1024"))),
            "conc": int(getattr(state, "conc", 0) or int(os.environ.get("CONC", "64"))),
        }
        # Forward the SAME bench knobs Hyperloom benched with so GEAK's internal
        # e2e measures identically; source = the baseline recipe's benchmark.envs
        # (process-env fallback). Only resolved keys are sent.
        bench_protocol = self._resolve_bench_protocol(str(getattr(state, "baseline_config_path", "") or ""))
        # Serving-launch fidelity: forward the SAME max-model-len / gpu-mem-util
        # the baseline served with so GEAK launches the identical engine and its
        # baseline matches raw_baseline_tput. Resolver parses these from the raw
        # baseline server-args (dedicated state.max_model_len wins; env last).
        try:
            from ..kernel.roofline_ceiling import read_baseline_server_args

            _baseline_srv_args = read_baseline_server_args(state) or ""
        except Exception:  # noqa: BLE001 — accessor is best-effort
            _baseline_srv_args = ""
        _serving_fidelity = _resolve_serving_fidelity(
            baseline_server_args=_baseline_srv_args,
            state_max_model_len=int(getattr(state, "max_model_len", 0) or 0),
        )

        handoff = {
            # v2 adds ``baseline_env_spec`` (the full layered env of current_best);
            # v1-only consumers ignore it and degrade to the flags/env-only baseline.
            "schema_version": 2,
            "model_path": str(getattr(state, "model_path", "") or os.environ.get("MODEL_PATH", "")),
            "framework": str(os.environ.get("FRAMEWORK", "") or "sglang"),
            "gpu_type": str(getattr(state, "gpu_type", "") or os.environ.get("GPU_TYPE", "")),
            "tp": int(os.environ.get("TP", "1") or 1),
            "workload": workload,
            "accepted_flags": accepted_flags,
            "accepted_env": accepted_env,
            "launch_recipe": str(getattr(state, "baseline_config_path", "") or ""),
            "raw_baseline_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            # Orchestrator throughput of the SAME config GEAK seeds its baseline
            # with, so run_e2e can compute a pure measurement divergence. 0.0 =>
            # no accepted config yet (falls back to raw baseline downstream).
            "orchestrator_best_tput_same_config": float((state.current_best or {}).get("tput") or 0.0)
            if isinstance(getattr(state, "current_best", None), dict)
            else 0.0,
            # Serving-launch fidelity (both optional; unset => GEAK adapter default).
            "max_model_len": int(getattr(state, "max_model_len", 0) or int(os.environ.get("MAX_MODEL_LEN", "0") or 0)),
            "mem_fraction": float(
                getattr(state, "mem_fraction", 0.0) or float(os.environ.get("GPU_MEMORY_UTILIZATION", "0") or 0.0)
            ),
            "exp_root": str(self.session_dir / "geak"),
            # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the
            # in-progress on-disk artifacts while a new cycle gets a fresh dir.
            "eval_dir": str(self.session_dir / "geak" / f"e2e_cycle{int(getattr(state, 'macro_cycle', 0) or 0)}"),
            # Align GEAK's bench CLIENT to Hyperloom's exact one so final/sweep
            # numbers are cross-harness comparable.
            "bench_client": "auto",
            "e2e_metric": "output",
            "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
            # Pin the serving GPU set: explicit visibility mask, else 0..tp-1.
            "gpu_ids": (
                os.environ.get("HIP_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
                or ",".join(str(i) for i in range(int(os.environ.get("TP", "1") or 1)))
            ),
        }
        if bench_protocol:
            handoff["bench_protocol"] = bench_protocol
        # Only forward resolved fidelity knobs; absence => GEAK adapter default.
        handoff.update(_serving_fidelity)
        # Full layered environment of current_best so GEAK's baseline ref ==
        # orchestrator best (config + source-patch snapshots + overlay).
        try:
            handoff["baseline_env_spec"] = self.build_env_spec()
        except Exception:  # noqa: BLE001 — env_spec is additive; never block handoff
            log.exception("geak: build_env_spec failed; handoff stays v1-compatible")

        out_dir = self.session_dir / "geak"
        out_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = out_dir / "handoff.json"
        handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

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
            # Rebench-first: record the recovered win as an UNVALIDATED candidate;
            # the caller enqueues the main-flow rebench that writes the headline.
            self._record_geak_candidate(result)
            self._record_geak_kernel_journey(result)
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                instrument.record_geak_operation(
                    self.session_dir,
                    stage="runner_result",
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    result={**result, "recovered_from": recovered_from},
                    status=str(result.get("status") or "unknown"),
                )
                instrument.record_geak_operation(
                    self.session_dir,
                    stage="candidate",
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    result=result,
                    status="running",
                )
            except Exception:  # noqa: BLE001
                log.debug("geak recovered v4 result recording failed", exc_info=True)
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
            """Record a (failed/skipped) GEAK outcome + wind down to SWEEP.

            Always records the normalized outcome into ``geak_result``,
            mirrors the failure reason onto the phase-entry evidence (so the
            session-breakdown surfaces WHY the e2e run did not land), then sets
            the ``skip_to_sweep`` hint so the coordinator never deadlocks.
            """
            state.geak_result = result
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                instrument.record_geak_operation(
                    self.session_dir,
                    stage="failed",
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    result=result,
                    status=str(result.get("status") or "failed"),
                )
            except Exception:  # noqa: BLE001
                log.debug("geak v4 failure recording failed", exc_info=True)
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
            # Reserve the pending slot BEFORE the task exists. The rebench runs
            # as an ``explore`` task; non-CLOSE phase boundaries may spare it
            # via ``spare_geak_rebench_on_phase_transition`` so the rebench can
            # finish after KERNEL winds down. Publishing the reservation after
            # enqueue left a window where KERNEL could exit and cancel the row.
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

            # The dispatcher refuses to launch a rebench whose only material is
            # an overlay that cannot load — that run would measure plain
            # baseline and credit GEAK for the noise. GEAK's own harness replays
            # the optimized config from result.json, so the kernel engages by
            # construction there; take that route instead of losing the win.
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
                # create_or_return_existing returned a task that already ran
                # under this cycle's idempotency key (#1240). Reconcile the
                # reservation from the persisted verdict rather than replacing
                # it with the misleading "settled before dispatch" status.
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

        # Crash-recovery: a validated result.json written before a coordinator
        # crash is promoted on resume, guarded by ``_geak_win_already_recorded``
        # so a prior cycle's result.json does not short-circuit a fresh entry.
        result_path = out_dir / "result.json"
        recovered = _read_geak_result(result_path)
        # Tombstone a result already adjudicated by 2b so stale result.json
        # cannot re-enqueue a settled candidate on a later KERNEL entry.
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

        # Budget-aware timeouts: shrink to the remaining run deadline and always
        # reserve the closing-grace window.
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

        # Run in its own process group so a timeout can SIGTERM the whole
        # runner -> run_e2e -> vllm/node tree (grace to flush result.json), then
        # SIGKILL, instead of orphaning run_e2e + its servers.
        term_grace = int(os.environ.get("GEAK_TERM_GRACE_S", "180"))

        def _run() -> subprocess.CompletedProcess:
            runner_env = dict(os.environ)
            runner_env["E2E_METRIC"] = "output"
            # Only injection point needed for the whole GEAK chain: geak_runner
            # and run_e2e both hand their full environment to the child, so the
            # tag reaches the Claude CLI that actually spends.
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
            # The graceful SIGTERM gives run_e2e a window to flush result.json;
            # keep a real win instead of discarding the phase as a timeout.
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
                # Rebench-first: enqueue the main-flow rebench (candidate stays
                # pending if a budget cap prevents it from running).
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
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="runner_result",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result=result,
                status=str(result.get("status") or "unknown"),
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 runner-result recording failed", exc_info=True)

        # Invariant guard: a GEAK run whose baseline ref failed to reproduce
        # ``orchestrator_best_tput_same_config`` optimized against a phantom
        # baseline, so its gain is non-comparable — never promote it.
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

        # Rebench-first: record the win as an UNVALIDATED candidate only; the
        # headline is written later from the measured rebench.
        self._record_geak_candidate(result)
        self._record_geak_kernel_journey(result)
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="candidate",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result=result,
                status="running",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 candidate recording failed", exc_info=True)
        # Enqueue the same-harness config-identity rebench — the ONLY path that
        # writes the headline. Until it lands the candidate stays pending.
        if str(result.get("status") or "") == "ok":
            await _enqueue_geak_revalidation(reason="geak_e2e_win")
        elif _geak_has_accepted_kernel(result):
            # A no_gain headline over an accepted, parity-checked kernel still
            # deserves the measurement — the rebench is what decides, and
            # without it the kernel is lost with no number attached to it.
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
        """Whether a GEAK e2e win is already in this session's state.

        Gates crash-recovery from an existing ``result.json`` so a prior cycle's
        win is not re-promoted on a later KERNEL entry.
        """
        return any(
            isinstance(item, dict) and item.get("action") == "geak_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    @staticmethod
    def _parse_geak_accepted_config(
        result: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """Parse ``result.accepted_config`` into (flags, env dict).

        Turns the bench-style ``{"flags":.., "env":..}`` blob into a reproducible
        (server-args, real-env) pair: any ``KEY=VAL`` token in ``env`` becomes a
        real env var; any ``--flag`` token folds into flags.

        Shares :func:`_accepted_config_as_variant` with the material gate and the
        2b dispatch, so the env mapping written to ``geak_pending`` and to
        ``current_best`` is the one the executor will actually run. This is the
        path that hands ``current_best`` the raw ``accepted_config``: filtering
        only at the comparison would leave a blocked name on the stored side and
        make an unchanged config read as a difference.
        """
        return _accepted_config_as_variant(result.get("accepted_config"))

    def _record_geak_candidate(self, result: dict[str, Any]) -> None:
        """Record a GEAK e2e win as an UNVALIDATED candidate (no headline).

        Stores the accepted config + the optimizer's own (audit-only)
        throughput/speedup under ``geak_pending`` without touching
        ``current_best`` / ``optimization_stack`` / ``cumulative_gain_validated*``. The
        headline is written later from a measured rebench by
        ``_promote_geak_from_candidate``; the config is captured verbatim as the
        source the rebench launches from.
        """
        if not isinstance(result, dict):
            return
        # ``no_gain`` is GEAK's verdict on its own headline number, not on the
        # kernels it accepted. A run can report no_gain on the promoted basis
        # while carrying an accepted kernel with a positive, parity-checked
        # same-config A/B — and dropping the whole result here means that kernel
        # never reaches a rebench and never appears anywhere. Admit it as a
        # candidate; the rebench downstream is still what decides.
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
            # Carry the kernels and the basis they were judged on into the
            # pending record, so a later promotion can name what it adopted
            # without re-reading result.json.
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
    def _geak_stack_entry_extra(result: dict[str, Any], *, overlay_loaded: bool | None) -> dict[str, Any]:
        """Build the ``geak_e2e`` stack entry, carrying only kernels proven to have run.

        ``accepted_kernels`` / ``accepted_heads`` are GEAK's self-report. They are
        evidence that a kernel *ran* only if the overlay carrying it was proven loaded
        for this measurement, which is exactly the call
        :meth:`_record_geak_adopted_kernels` already makes for the per-kernel ledger.

        The stack entry is the other reader: ``_geak_contribution`` classifies the
        dashboard row from these lanes alone. Copying the lanes unconditionally let the
        two disagree — a rebench that stripped a dead overlay promoted on its config
        gain, the ledger correctly said unattributable, and the dashboard still filed
        the row under ``kernel`` because the entry named one. So the lanes travel only
        with the proof, and the proof travels with them.

        Args:
            result: GEAK's ``result.json`` payload.
            overlay_loaded: Whether the overlay was proven loaded. ``None`` means the
                caller could not tell, which is not proof and so is not credited.

        Returns:
            dict[str, Any]: The ``entry_extra`` for :meth:`_lift_to_current_best`.
        """
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
    ) -> None:
        """Write the GEAK headline from a MEASURED main-flow rebench.

        The single headline writer: lifts ``current_best`` (config/overlay/scripts
        + the measured tput), appends the ``geak_e2e`` optimization_stack entry +
        gain ledger, and stamps ``cumulative_gain`` / ``cumulative_gain_validated``
        as the same-harness total ``(measured - baseline)/baseline``. Clears
        ``geak_pending`` and the revalidation flag.

        Args:
            result: GEAK's ``result.json`` payload.
            measured_tput: The rebench-measured throughput (tok/s).
            provenance: Which validation path measured it.
            overlay_loaded: Whether the authored-kernel overlay was proven
                loaded for the measurement. ``None`` means the caller could not
                tell. Only a ``True`` here lets an accepted kernel be written
                into the adoption ledger: a flags-only rebench measured no
                kernel, so crediting one would be an invention.
        """
        if not isinstance(result, dict):
            return
        try:
            measured = float(measured_tput)
        except (TypeError, ValueError):
            return
        if measured <= 0:
            return
        # KEEP guard (aligns GEAK with forge / integrate_patch): a measured
        # rebench that does not beat the current best must NOT overwrite the
        # headline / stack / gain. Backstops every promote entry point (2a, 2b,
        # crash-recovery) so a low-but-valid measurement can never lower best.
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
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                rejected_result = dict(result)
                rejected_result["final_validation"] = {
                    "decision": "REJECTED",
                    "reason": "rebench_did_not_beat_current_best",
                    "measured_tput": measured,
                    "current_best_tput": float(cb_tput),
                }
                instrument.record_geak_operation(
                    self.session_dir,
                    stage="final_validation_failed",
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    result=rejected_result,
                    status="failed",
                    validated=False,
                    measured_tput=measured,
                    validation_source="geak_promote_rejected",
                )
            except Exception:  # noqa: BLE001
                log.debug(
                    "geak v4 final validation rejection recording failed",
                    exc_info=True,
                )
            self.shared_state.geak_pending = {}
            self.shared_state.resume_pending_revalidation = False
            return
        accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)

        # The lever is stamped here, not guessed from the task kind: GEAK
        # promotes on a proven kernel overlay OR on a config/env-only win, and
        # only this site holds the overlay proof. Reuse the same proof
        # ``_geak_stack_entry_extra`` applies so ``lever_buckets`` and
        # ``_geak_contribution`` cannot classify one row two ways.
        entry_extra = self._geak_stack_entry_extra(result, overlay_loaded=overlay_loaded)
        kernel_proven = bool(entry_extra.get("accepted_kernels") or entry_extra.get("accepted_heads"))

        self._lift_to_current_best(
            "geak_e2e",
            measured,
            {
                "name": "geak_e2e",
                "candidate_extra_server_args": accepted_flags,
                "extra_envs": dict(parsed_envs),
                "final_overlay": result.get("final_overlay") or "",
                "source_phase": "KERNEL_AGENT",
                "lever_kind": LEVER_KERNEL if kernel_proven else LEVER_CONFIG,
                "ttft_mean_ms": result.get("ttft_ms"),
                "tpot_mean_ms": result.get("tpot_ms"),
                "workspace": result.get("eval_dir"),
            },
            entry_extra=entry_extra,
        )

        base = float(self.shared_state.baseline_tput or 0.0)
        # Where the session stood before GEAK ran: the anchor both the journey
        # rejection and the route-level residual measure from.
        pre_geak = float(cb_tput) if isinstance(cb_tput, (int, float)) and cb_tput > 0 else base
        self._record_geak_adopted_kernels(
            result,
            measured_tput=measured,
            baseline_tput=base,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
        )
        if overlay_loaded is not True:
            # The journey is replayed before the main-flow rebench and can
            # therefore contain GEAK-internal KEEPs for kernels that were not
            # present in the configuration that produced ``measured``.  Once
            # the final validation proves no overlay was loaded, withdraw those
            # provisional per-kernel adoptions.  The validated win still lands
            # below as one route-level config attempt.
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
                source="geak_e2e_promote",
            )
        self.shared_state.resume_pending_revalidation = False
        self.shared_state.geak_pending = {}
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="final_validation",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result=result,
                status="succeeded",
                validated=True,
                measured_tput=measured,
                validation_source="geak_orch_harness",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 final validation recording failed", exc_info=True)

        # The route operation above is diagnostic context and is deliberately
        # excluded from the canonical optimization-attempt ledger. Record the
        # validated route-level win separately so env/flag/CSV wins, and
        # multi-kernel wins that cannot be divided honestly, still reach the
        # GEAK dashboard bucket. Record only the part the per-kernel ledger did
        # NOT already claim: the journey's own attributable KEEPs are summed by
        # ``record_kernel_e2e``, so crediting the full route delta again would
        # double-count them, while suppressing the whole attempt because one
        # kernel was attributable would drop every other percentage point the
        # route measured.
        try:
            # A journey KEEP is attributable only when the final measurement
            # proved that its overlay was loaded.  Otherwise the same-harness
            # route attempt owns the complete measured delta.
            claimed_delta = self._geak_journey_attributed_delta(result) if overlay_loaded is True else 0.0
            # Anchor the route attempt where the per-kernel ledger stops, so
            # the two records partition the measured lift instead of
            # overlapping. Both records divide by the same session baseline, so
            # holding back the ledger's ABSOLUTE tok/s makes the two
            # ``(after - started_from) / baseline`` terms telescope to exactly
            # the measured route lift, leaving nothing for
            # ``unattributed_gain_pct`` to absorb.
            residual_before = pre_geak + claimed_delta
            if base > 0 and pre_geak > 0 and measured > residual_before * _GEAK_RESIDUAL_MIN_RATIO:
                # Only an ``AITER_CONFIG_*`` env names a GEMM tuning table. Any
                # other csv-valued env (a profile dump, a shape list) says
                # nothing about the lane, so it must not reclassify the kind.
                is_gemm = any(str(key).upper().startswith("AITER_CONFIG_") for key in dict(parsed_envs or {}))
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                instrument.record_geak_e2e_attempt(
                    self.session_dir,
                    kind="gemm_tuning" if is_gemm else "kernel_optimization",
                    throughput_before=residual_before,
                    throughput_after=measured,
                    baseline_tput=base,
                    # ``local_gain_pct`` is measured against the attempt's own
                    # starting point, not the session baseline.
                    gain_pct=(measured - residual_before) / residual_before * 100.0,
                    attribution_eligible=True,
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    accepted_config=result.get("accepted_config"),
                    provenance=provenance,
                    result=result,
                )
        except Exception:  # noqa: BLE001
            log.debug("geak e2e attempt recording failed", exc_info=True)

    @staticmethod
    def _geak_journey_path(result: dict[str, Any]) -> str:
        """Resolve the journey file for a GEAK result.

        Three readers need it on the promote path, and each rediscovering the
        ``kernel_journey_path`` / ``eval_dir`` fallback is three chances to
        disagree about which file they read.

        Args:
            result: GEAK's ``result.json`` payload.

        Returns:
            str: The journey path, or ``""`` when there is no readable file.
        """
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
        """Read the journey file, or return ``{}`` when it is unusable.

        Args:
            result: GEAK's ``result.json`` payload.

        Returns:
            dict[str, Any]: The parsed journey; empty on any failure, which
            every caller must read as "the journey says nothing".
        """
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
        """Return the journey's kernel records, or ``[]`` when unreadable.

        Args:
            result: GEAK's ``result.json`` payload.

        Returns:
            list[dict[str, Any]]: The ``kernels`` array; empty on any failure.
        """
        journey = cls._load_geak_journey(result)
        return [kernel for kernel in journey.get("kernels") or [] if isinstance(kernel, dict)]

    @classmethod
    def _geak_journey_attributed_delta(cls, result: dict[str, Any]) -> float:
        """Return the tok/s the per-kernel ledger already credits.

        A journey KEEP with a validated ``(base_tput, new_tput)`` pair is
        credited by ``collect_recorded_optimizations`` as
        ``(new_tput - base_tput) / session_baseline``: an ABSOLUTE tok/s delta
        over the one session denominator. So the share to hold back from the
        route-level attempt is that same absolute delta, summed.

        It is deliberately not a speedup RATIO. GEAK measures its journey on
        its own harness at its own working point, so ``base_tput`` is not the
        session's ``current_best`` (see ``record_kernel_e2e``: the executor's
        percentage is "measured against whatever baseline it happened to hold
        at the time"). Scaling ``current_best`` by ``new/base`` would withhold
        a number no record ever claimed, and the difference would silently
        reappear as ``validation.unattributed_gain_pct``.

        Args:
            result: GEAK's ``result.json`` payload.

        Returns:
            float: The already-claimed tok/s, ``0.0`` when the journey holds no
            attributable KEEP (nothing is claimed, so the route owns it all).
        """
        delta = 0.0
        for kernel in cls._geak_journey_kernels(result):
            e2e = kernel.get("e2e") if isinstance(kernel.get("e2e"), dict) else {}
            decision = str(e2e.get("decision") or "").upper()
            base_tput = e2e.get("base_tput")
            new_tput = e2e.get("new_tput")
            if (
                e2e.get("validated") is True
                and decision in {"KEEP", "ADOPTED"}
                and isinstance(base_tput, (int, float))
                and isinstance(new_tput, (int, float))
                and base_tput > 0
                and new_tput > 0
            ):
                # A regression is never "claimed gain": clamp at 0 so a slower
                # KEEP cannot inflate the route-level residual.
                delta += max(0.0, float(new_tput) - float(base_tput))
        return delta

    def _record_geak_adopted_kernels(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        baseline_tput: float,
        provenance: str,
        overlay_loaded: bool | None,
    ) -> None:
        """Write one adoption row per accepted GEAK kernel.

        GEAK's win is recorded in two disjoint places today. The per-ACTION
        ledger (``optimization_stack`` + ``geak_pending``) carries the headline;
        the per-KERNEL ledger (``state.kernel_integrate_attempts``) is what
        ``by_kernel``, ``kernel_lifecycle.adopted``, the attribution split and
        the timeline all read. GEAK writes only the first, so an adopted kernel
        exists in the headline and nowhere a report can name it. This writes the
        second, from the same promotion, so both agree by construction.

        The gain recorded is the ORCHESTRATOR-measured rebench gain over
        baseline, never GEAK's self-reported ``e2e_delta_pct``. When several
        kernels rode in on one rebench, or the overlay was not proven loaded,
        the gain cannot be attributed to any single kernel: the row is written
        with a null gain and ``validated: False`` rather than an invented share.
        """
        if not isinstance(result, dict):
            return
        # Both acceptance lanes, ``env`` selections excluded and alias twins
        # collapsed. See ``_geak_accepted_kernel_specs``.
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
        # One kernel, overlay proven loaded, one measured number: the gain is
        # attributable. Anything else is a joint measurement.
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
            # Max over attempts, matching the canonical ledger writer in
            # ``_kernel_decisions.py`` -- ``by_kernel`` and
            # ``kernel_lifecycle`` read this one field from both writers, so a
            # second, worse rebench must not lower the kernel's best. ``None``
            # is kept rather than that writer's ``0.0`` default: here it means
            # "not attributable", which is not the same claim as "no gain".
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
                    # GEAK's own same-config A/B, kept beside the orchestrator
                    # number so the two are never confused for each other.
                    "geak_same_config_delta_pct": spec.get("e2e_delta_pct"),
                    "geak_isolated_speedup": spec.get("isolated"),
                    "updated_at": ts,
                }
            )
            ledger[kid] = entry
        self.shared_state.kernel_integrate_attempts = ledger
        log.info(
            "geak: recorded %d adopted kernel(s) in the per-kernel ledger (overlay_loaded=%r attributable=%r gain=%r)",
            len(rows),
            overlay_loaded,
            attributable,
            rebench_gain if attributable else None,
        )

    def _record_geak_kernel_journey(self, result: dict[str, Any]) -> None:
        """Replay GEAK-e2e's kernel_journey.json into the breakdown recorder.

        GEAK-e2e emits a ``kernel_journey.json`` whose per-kernel sub-objects are
        shaped exactly as the recorder's ``record_kernel_{dispatch,backend_result,
        e2e}`` inputs; replay them verbatim so the assembler folds the e2e
        optimizer's kernels into ``kernel_journey``. Best-effort: a missing/partial
        file never breaks the phase.
        """
        journey = self._load_geak_journey(result)
        if not journey:
            return

        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        sdir = self.session_dir
        commit = str(getattr(self.shared_state, "code_revision", "") or "")
        # Replay GEAK-e2e's discovery substream so the assembler backfills each
        # kernel's discovery-sourced fields; GEAK profiles via rocprofv3 (route
        # ``bypass``), ``tool="geak"`` for version provenance.
        for run in journey.get("discovery_runs") or []:
            if not isinstance(run, dict):
                continue
            try:
                instrument.record_kernel_discovery(
                    sdir,
                    source=str(run.get("source") or "bypass"),
                    status=str(run.get("status") or "success"),
                    hot_kernels=list(run.get("hot_kernels") or []),
                    scan=run.get("scan") if isinstance(run.get("scan"), dict) else None,
                    tool="geak",
                    route_strategy="geak",
                )
            except Exception:  # noqa: BLE001
                log.debug("geak kernel_journey discovery replay failed", exc_info=True)
        for k in journey.get("kernels") or []:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            if not kid:
                continue
            disp = k.get("dispatch") if isinstance(k.get("dispatch"), dict) else {}
            try:
                instrument.record_kernel_dispatch(
                    sdir,
                    kernel_id=kid,
                    dispatched=bool(disp.get("dispatched", True)),
                    backends=list(disp.get("backends") or []),
                    skip_reason=str(disp.get("skip_reason") or ""),
                    orchestration_commit=commit,
                    task_group=disp.get("task_group"),
                    route_strategy="geak",
                )
                br = k.get("backend_result")
                if isinstance(br, dict):
                    instrument.record_kernel_backend_result(
                        sdir,
                        br,
                        route_strategy="geak",
                    )
                e2e = k.get("e2e")
                if isinstance(e2e, dict):
                    instrument.record_kernel_e2e(
                        sdir,
                        kernel_id=kid,
                        integrated=bool(e2e.get("integrated", False)),
                        e2e_gain_pct=e2e.get("e2e_gain_pct"),
                        validated=e2e.get("validated"),
                        decision=str(e2e.get("decision") or ""),
                        patch_path=e2e.get("patch_path"),
                        target_file=e2e.get("target_file"),
                        extra_server_args=str(e2e.get("extra_server_args") or ""),
                        result=e2e,
                        route_strategy="geak",
                        # Replaying must land on the reading it originally
                        # recorded, not count itself as a fresh one.
                        occurrence=e2e.get("occurrence"),
                    )
            except Exception:  # noqa: BLE001
                log.debug("geak kernel_journey replay failed for %s", kid, exc_info=True)
        for tool, meta in (journey.get("versions") or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                instrument.record_tool_version(
                    sdir,
                    tool=str(tool),
                    root=str(meta.get("root_dir") or "") or None,
                    version=str(meta.get("version") or meta.get("commit") or "") or None,
                )
            except Exception:  # noqa: BLE001
                pass

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

        from hyperloom.inference_optimizer.breakdown.recorder import instrument

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
            try:
                instrument.record_kernel_e2e(
                    self.session_dir,
                    kernel_id=kernel_id,
                    integrated=False,
                    e2e_gain_pct=None,
                    validated=False,
                    decision="REVERT",
                    # Same route as the KEEP that this withdraws. Leaving it on the default
                    # re-parented the kernel under a synthetic Forge route at the moment it was
                    # revoked, so a withdrawn GEAK kernel ended up filed under an optimizer that
                    # never touched it.
                    route_strategy="geak",
                    patch_path=e2e.get("patch_path"),
                    target_file=e2e.get("target_file"),
                    extra_server_args=str(e2e.get("extra_server_args") or ""),
                    result=evidence,
                    # This is a second look at a kernel that was already kept,
                    # and ``evidence`` still carries the original integrate's
                    # identity. Without a namespace of its own, the reading
                    # that rejects the kernel would land on the reading that
                    # adopted it.
                    occurrence="revalidation",
                )
            except Exception:  # noqa: BLE001
                log.debug(
                    "geak kernel_journey rejection replay failed for %s",
                    kernel_id,
                    exc_info=True,
                )

    def _runtime_uses_aiter_fused_moe(self) -> bool:
        """Return whether the served model dispatches MoE through aiter.

        vLLM's Triton ``fused_moe`` reads ``VLLM_TUNED_CONFIG_FOLDER``; aiter's
        fused MoE does not. When aiter owns the MoE the Triton tuner's JSON is
        unreachable, so validating it burns two full benchmark rounds on a config
        the server cannot load.
        """
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
        """Report whether the validated aiter CSV was reachable by the server.

        A tuned CSV only helps when aiter's padded (M, N, K) lookup can resolve a
        row for the shapes the server actually asks for. When it cannot, the run
        still boots and benchmarks fine, so the gate sees an honest "no gain" and
        the real cause -- an artifact the runtime never applied -- stays invisible.
        Replaying the lookup against the round's ``server.log`` separates the two.

        Its result can block a KEEP, so an unexpected failure must not: it would
        turn a diagnostic into the very false REVERT this replaces. Any
        exception degrades to "undetermined", matching ``_gemm_apply_verdict``.
        """
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
        """Replay aiter's lookup against the round's log (see the caller).

        For ``fmoe_ck``, delegates to ``_fmoe_tuned_config_coverage``, which
        matches fused-MoE dispatch lines against ``candidate_fmoe.csv`` rather
        than dense ``(M, N, K)`` GEMM lookups.
        """
        if tuner_name == "fmoe_ck":
            return self._fmoe_tuned_config_coverage(envs)
        from ..kernel.gemm_shape_coverage import (
            parse_aiter_consulted_tables,
            parse_aiter_shape_lookups,
            tuned_config_coverage,
            tuned_csv_shapes,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        run_dir = self.session_dir / "runs" / "integrate" / f"integrate-gemm_tune_{tuner_name}"
        logs = sorted(run_dir.rglob("server.log"), key=_safe_mtime)
        if not logs:
            return None
        try:
            log_text = logs[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        def _unreadable(kind: str) -> None:
            """Log that the artifact could not be read, so the caller stays out of it.

            A CSV we cannot parse is an absence of evidence, not evidence the
            runtime ignored the table. Returning a 0% report would let that
            absence block a KEEP whose throughput genuinely improved -- the
            same conflation this change set exists to remove.
            """
            log.warning(
                "gemm E2E: tuner=%s %s tuned CSV yielded no keys from %s; "
                "coverage is undetermined and will not block the KEEP",
                tuner_name,
                kind,
                csv_paths,
            )

        missed, hit = parse_aiter_shape_lookups(log_text)
        requested = missed | hit
        if not requested:
            return None
        tuned: set[tuple[int, int, int]] = set()
        for path in csv_paths:
            tuned |= tuned_csv_shapes(path)
        if not tuned:
            _unreadable("dense")
            return None
        report = tuned_config_coverage(tuned, requested)
        report["server_log"] = str(logs[-1])
        report["runtime_lookup_miss"] = len(missed)
        report["runtime_lookup_hit"] = len(hit)
        report["artifact_applied"] = bool(report.get("covered"))
        consulted = parse_aiter_consulted_tables(log_text)
        report["consulted_tables"] = sorted(consulted)[:8]
        wanted = {Path(path).name for path in csv_paths}
        if consulted and not (wanted & {Path(name).name for name in consulted}):
            # The runtime resolved a different quantisation variant's table, so
            # the tuner targeted a kernel this server never dispatches to.
            report["artifact_applied"] = False
            report["not_applied_reason"] = "artifact_table_not_consulted"
        elif not report["artifact_applied"]:
            report["not_applied_reason"] = "no_shape_key_matched"
        return report

    def _fmoe_tuned_config_coverage(
        self,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Report whether a ``tuned_fmoe.csv`` covers logged fused-MoE dispatches.

        Dense BF16 GEMM lookups in the same log are ignored: they belong to
        linears the ``fmoe_ck`` tuner never wrote, and treating
        ``bf16_tuned_gemm.csv`` as evidence produced false
        ``artifact_table_not_consulted`` blockers on MoE models.
        """
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
        """Re-measure baseline and tuned stack interleaved, and judge the pairs.

        ``running_tput`` is compared against a ``baseline_tput`` measured earlier
        in the session, so any drift between the two -- clocks, temperature, a
        neighbour's workload -- is indistinguishable from the tuning. One
        controlled repeat on this fleet moved 16% with nothing changed, and three
        rounds of one unchanged configuration spanned 58%.

        Interleaving is the only thing that separates them, and it costs two
        extra benchmark rounds per pair, so it is opt-in via
        ``HYPERLOOM_GEMM_PAIRED_PAIRS``. When it does not run the gain is still
        promoted -- it is the best number available -- but it is *labelled* as an
        unpaired block comparison rather than passed off as a paired one.
        """
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
            # The B leg has to be served the same way the KEEP was: fmoe_ck only
            # takes effect under --moe-runner-backend aiter, and without it the
            # tuned table is never read, so B measures the same thing as A and
            # the confirmation reports within_noise for a gain that is real.
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
                        # Measure, do not decide: the verdict comes from the
                        # pairs, so a per-round KEEP/REVERT here would be noise
                        # promoted to a decision.
                        "keep_threshold_pct": 100.0,
                        "budget_minutes": budget_minutes,
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
        """Did the tuned table reach the server's merge list and get read?

        Complements ``_gemm_tuned_config_coverage``, which replays the shape
        lookup against the CSV we wrote. That answers "could this table have
        served the requests"; it cannot see the case where the table never
        arrived and the server loaded its bundled default instead, because the
        CSV on our disk still contains the right rows either way.

        For ``fmoe_ck``, delegates to ``_fmoe_apply_verdict``, which attributes
        fused-MoE kernel pairs from dispatch lines instead of dense merge/hit
        logging.
        """
        if tuner_name == "fmoe_ck":
            return self._fmoe_apply_verdict(envs)
        from ..measurement.apply_verification import verify_applied

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        run_dir = self.session_dir / "runs" / "integrate" / f"integrate-gemm_tune_{tuner_name}"
        logs = sorted(run_dir.rglob("server.log"), key=_safe_mtime)
        if not logs:
            # Say so. This whole change exists to stop checks from failing
            # quietly, and a missing log is the one way this one can.
            log.warning(
                "forge gemm E2E: no server.log under %s; apply verification cannot run for %s",
                run_dir,
                tuner_name,
            )
            return None

        # The deployed file is named after the candidate, so the runtime's own
        # table name has to travel with it or the arrival check compares
        # merged_tuned_dense_bf16.csv against bf16_tuned_gemm.csv and concludes
        # the artifact never landed.
        table_names = [name for key in envs if (name := _AITER_ENV_TO_TABLE.get(key))]
        # aiter prints a hit line only under this flag; every serving run now
        # sets it by default, but an operator value in the candidate env wins,
        # and then a zero-hit result means nothing.
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
        """Apply verdict for ``fmoe_ck`` based on fused-MoE dispatch, not dense GEMM.

        Dense ``bf16_tuned_gemm.csv`` consulted-table lines in the same log must
        not drive ``not_merged`` for a tuner that only deploys ``tuned_fmoe.csv``.
        """
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
        """Merge a GEMM candidate CSV with the runtime config.

        aiter's complete config is the merged superset of its top-level table and
        all matching ``model_configs/*.csv`` tables. The candidate CSV only has
        the shapes the tuner improved. Using it alone as the env override drops
        all other shapes' tuned entries, causing regression.

        Prefer the live ``/tmp/aiter_configs`` table when it exists. That cache is
        normally removed with the serving process, so fall back to rebuilding the
        same table from the installed aiter package. Overlay the candidate by the
        untuned schema's dispatch keys and write one self-contained CSV for E2E.

        Implemented on the stdlib ``csv`` module on purpose: this runs in the
        orchestrator process, which must not carry a hard pandas dependency
        (pandas is not declared in ``pyproject.toml`` and is absent from the
        ``.[test,ci]`` CI environment -- importing it there raises
        ``ModuleNotFoundError`` and every candidate is silently rejected).
        Values are carried through as text, so a config round-trips byte-for-byte
        instead of being re-formatted by a dataframe writer.

        Returns the merged file path, or None if merging fails.
        """
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

                # Keep the fastest (smallest us) row per dispatch key; ties keep
                # the first row seen (stable), NaN-like values sort last.
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
        """Whether the fp8 block-scale CK backend switch should be E2E-validated.

        The CK backend switch (``SGLANG_FP8_BLOCKSCALE_CK_MAX_M``) routes the fp8
        block-scale GEMM from the Triton default to the aiter CK
        ``gemm_a8w8_blockscale`` kernel on gfx942; it is independent of the a8w8
        table tuner result and must be flipped + E2E-validated as its own
        candidate. Gated strictly to the forge backend on a
        sglang + fp8 + gfx942 + block-scale workload (block-scale asserted
        positively via ``weight_block_size``).

        Args:
            result (dict[str, Any]): The GEMM tuning handler result.

        Returns:
            bool: ``True`` only when the CK switch is the relevant lever.
        """
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
        """Whether the workload runs fp8, resolved from any available signal.

        Accepts fp8 from, in order: ``shared_state.precision``, the forge
        ``result`` envelope's resolved precision, or the runtime
        ``--quantization`` resolved from the actual server args.

        Args:
            result (dict[str, Any]): The GEMM tuning handler result.

        Returns:
            bool: ``True`` when any signal resolves to fp8.
        """
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
        """Merge a handler-owned Roofline fallback into the live Coordinator state.

        The handler runs its inline Roofline against a throwaway ``SharedState``
        loaded from disk, so the refreshed profile fields only exist in
        ``state.json`` until they are merged back here. Any save of the live
        state between the handler returning and this merge would clobber them,
        so callers must invoke this before persisting the live state. Repeated
        calls are idempotent.
        """
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
        # Lifecycle is append-only telemetry owned by both states; union it so
        # neither the inline Roofline's rows nor the live state's are dropped.
        self.shared_state.merge_lifecycle_events(persisted.lifecycle)

    async def _handle_gemm_tuning_result(self, result: dict[str, Any]) -> None:
        """Record and post-process a run_gemm_tuning result from any entrypoint.

        Both the KERNEL-entry auto hook and orchestration-issued
        ``run_gemm_tuning`` requests converge here so no backend bypasses
        per-candidate E2E validation.
        """
        self._sync_profile_state_after_gemm_roofline(result)
        self.shared_state.record_gemm_tuning(result)
        try:
            await self._validate_gemm_tuning_e2e(result)
        except Exception as exc:  # noqa: BLE001
            # Validation spans server restarts, log parsing and CSV merges, and
            # is reached from two entrypoints that only guard the tuning call
            # itself. An unexpected failure here has to read as "this candidate
            # was never measured", not take the KERNEL phase down with it --
            # tuning that produced nothing measurable is the outcome this whole
            # change exists to record honestly.
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
            # The bridge stamped KEEP + the raw combined env on the micro result;
            # the normal exit rewrites both so Orchestration never bundles an
            # integrate against an unmeasured candidate. This arm was not
            # measured, so it reads as REVERT.
            result["decision"] = "REVERT"
            result["requires_e2e_validation"] = False
            result["e2e_validated"] = False
            result["micro_decision"] = "e2e_validation_exception"
            for stale in ("recommended_env", "extra_envs"):
                if result.get(stale):
                    result[stale] = {}
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_gemm_tuning_operation(
                self.session_dir,
                payload={
                    "task_id": str(result.get("task_id") or "kernel_entry_gemm_tuning"),
                    "macro_cycle": int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                },
                result=result,
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
            )
        except Exception:  # noqa: BLE001
            log.debug("gemm v4 finalized-result recording failed", exc_info=True)
        self.shared_state.save(self.session_dir)

    def _journal_gemm_tuning_keep(
        self,
        entry: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """Mirror an adopted GEMM-tuning stack entry as an optimization_journal KEEP row.

        Emits a KEEP journal row carrying the end-to-end ``throughput_after`` plus
        the originating ``task_id`` so the GEMM tuning point shows up on the
        phase_timeline alongside every other attempt. Best-effort.

        Args:
            entry: The ``optimization_stack`` entry just appended for this
                GEMM-tuning adoption (carries variant_name / tput / gain_pct /
                backend / tuned_file / ts).
            task_id: Originating task id used to join per-step token spend.
        """
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

    def _replace_latest_gemm_tuning_attempt(self, result: dict[str, Any]) -> None:
        """Sync the latest GEMM history row after forge E2E rewrites ``result``."""
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

    def _gemm_e2e_candidates(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Reduce a GEMM tuning result to the env sets worth E2E-validating.

        Selection is by result shape: ``tuners_run`` entries name their own env
        vars, whereas a bare ``tuned_file`` is only meaningful under the GEAK
        a8w8 tuner's env var.

        Args:
            result (dict[str, Any]): The GEMM tuning handler result.

        Returns:
            list[dict[str, Any]]: Candidates with ``tuner`` / ``env_var`` /
                ``env_value`` / ``envs`` / ``micro_speedup``.
        """
        candidates: list[dict[str, Any]] = []
        # The list is already priority-sorted by forge CLI (fmoe_ck first).
        for t in result.get("tuners_run") or []:
            if not isinstance(t, dict):
                continue
            # partial_output is a real artifact: the tuner wrote fewer rows than
            # shapes it was given (the grouped batch budget ran out), but the
            # rows it did write are deployable.
            if t.get("status") not in ("ok", "partial_output"):
                continue
            # improved_shapes can never exceed 0 for tuners with no comparable
            # baseline -- TunableOp never times the untuned dispatch, the
            # candidate-CSV fallback has no per-shape Pre/Post table, and a
            # hipblaslt-only bf16 run has no torch candidate to measure against.
            # They report unverified_shapes instead, so gating on improved_shapes
            # alone would drop exactly the artifacts that need e2e to say
            # anything at all about them.
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

        # Standalone fp8 block-scale CK backend switch: inject as its own
        # candidate so the loop E2E-validates baseline Triton vs CK.
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
        """Sequentially E2E-validate each tuning candidate's env independently.

        Like kernel_opt's per-kernel integrate: try each candidate's env one by
        one, measured against ``current_best``. KEEPs accumulate (stacked envs);
        REVERTs are discarded, so one bad candidate cannot drag down the set.
        A round the run stopped ends the sweep with its tuners unrecorded.
        """
        from ..kernel.request_handlers import integrate_handler
        from hyperloom.common.model_paths import resolve_session_model_path

        backend = str(result.get("backend") or "geak").strip().lower()
        candidates = self._gemm_e2e_candidates(result)
        if not candidates:
            log.info("gemm tuning: no candidates to E2E validate")
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

        # fmoe_ck is only meaningful with --moe-runner-backend aiter, and aiter's
        # CK fused-MoE rejects a non-128-aligned intermediate_size_per_partition.
        # Validating it anyway costs a full cold start that can only end in a
        # dead server.
        from hyperloom.inference_optimizer.cli.model_gate import (
            model_supports_aiter_ck_fused_moe,
        )

        triton_moe_inert = (
            any(c.get("tuner") == "vllm_moe_triton" for c in candidates) and self._runtime_uses_aiter_fused_moe()
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
            # Merge candidate CSV with the runtime config so that shapes NOT in
            # the candidate keep their existing tuned entries. Without this, the
            # E2E validation would run with ONLY the candidate's shapes tuned,
            # causing regression on all other shapes that lose their config.
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

            # Two independent ways the artifact can fail to take effect, neither
            # of which the throughput delta can see: the keys are unreachable
            # (coverage), and the table never reached the server (apply verdict).
            # Both are positive findings, not absences of evidence -- so they
            # block the KEEP rather than merely annotating it. Crediting a gain
            # here would attribute run-to-run drift to tuning that provably did
            # not run.
            apply_blockers: list[str] = []

            coverage = self._gemm_tuned_config_coverage(tuner_name, env)
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
                    # "Cannot tell" is not "did not apply": hit lines need
                    # AITER_LOG_TUNED_CONFIG=1, and treating their absence as a
                    # failure would revert every arm that ran without it.
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
                # The one place this path names its artifact. The stack entry
                # below and the attempt row further down both read it, so the
                # breakdown's string match cannot be defeated by a stack append
                # that was skipped as already-applied.
                adopted_tuned_file = _candidate_tuned_file(env, cand.get("env_var", ""))

                lifted = self._lift_to_current_best(
                    "gemm_tuning",
                    new_tput,
                    {
                        "name": f"{backend}_{tuner_name}",
                        "candidate_extra_server_args": extra_server_args,
                        "extra_envs": dict(env),
                        "source_phase": "KERNEL_AGENT",
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
                    # Distinguish "the tuning did not pay off" from "the tuned
                    # artifact was never reachable", which is a wiring defect.
                    # The second is worth reporting even when the run also
                    # happened to measure a gain -- especially then.
                    reason = f"tuned_config_never_applied[{'+'.join(apply_blockers)}] ({reason})"
                reverted.append({**cand, "reason": reason})

        # The watermark covers the whole run, so it waits for the last KEEP.
        if kept:
            total_gain = (running_tput - baseline_tput) / baseline_tput * 100.0 if baseline_tput > 0 else 0.0
            # One end-to-end measurement is not enough on this fleet: three
            # rounds of a single unchanged configuration spanned 58%. Re-run
            # the baseline interleaved with the tuned stack so drift shows up
            # as drift. Opt-in, and when it does not run the gain is still
            # promoted -- it is the best number available -- but labelled as an
            # unpaired block comparison rather than passed off as a paired one.
            paired = await self._confirm_gemm_gain_paired(
                stacked_envs,
                baseline_tput=baseline_tput,
                budget_minutes=per_tuner_budget_minutes,
                extra_server_args=("--moe-runner-backend aiter" if "AITER_CONFIG_FMOE" in stacked_envs else ""),
            )
            if paired is not None:
                result["paired_confirmation"] = paired.to_dict()
            if baseline_tput > 0:
                self._update_cumulative_gain_validated(
                    running_tput,
                    source="forge_gemm_tuning_e2e",
                    measurement_basis=_paired_measurement_basis(paired),
                )
            # Name the artifact this run adopted, so the breakdown can tell it
            # was. Forge never set ``tuned_file`` (it reports per-tuner envs
            # instead), which left the history row's path empty and the adoption
            # lookup matching on "". The value is the one the stack entry above
            # carries, taken from the same call rather than looked up.
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

        # Rewrite the stored result to the E2E-validated outcome so Orchestration
        # never sees the raw combined recommended_env and issues a bundled integrate.
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
        """Close out KERNEL entry on either route: re-profile, run the
        independently gated stages, then dispatch whatever kernel_opt work the
        candidate table already justifies.

        The dispatch used to sit on the GEMM route alone, so skipping GEMM
        tuning silently removed the phase's own kernel_opt as well. The two
        settings are unrelated -- one tunes GEMM shape tables, the other
        rewrites source-level kernels -- and nothing in the log connected them,
        so a run could hold eight routable candidates, clear the dispatch floor,
        and still reach SWEEP having optimized nothing, waiting on an
        orchestration request that never came.

        What the dispatch needs is untried routable candidates. That is what it
        asks for, on both routes.
        """
        await self._maybe_reprofile_for_kernel()
        await self._maybe_run_forge_fusion_before_kernel_opt()
        await self._maybe_run_collective_before_kernel_opt()
        if self._kernel_opt_work_remains():
            await self._run_kernel_opt_nomination()

    def _kernel_opt_work_remains(self) -> bool:
        """Whether KERNEL entry should call forge to nominate a kernel itself.

        The switch scopes to this entry call alone. ``kernel_opt`` stays in the
        phase's allowed actions either way, so orchestration can still request
        it; opting out only means the phase stops asking on its own.

        Returns:
            bool: ``True`` when the ``auto_kernel_opt_enabled`` flag is set and
                there are untried hot reusable kernels remaining.
        """
        if not bool(getattr(self.shared_state, "auto_kernel_opt_enabled", True)):
            return False
        return bool(self.shared_state.untried_hot_reusable_kernels())

    async def _run_kernel_opt_nomination(self) -> None:
        """Call forge once at KERNEL entry to nominate and optimize a kernel.

        The which-kernel selection now lives in forge nomination, so this entry
        no longer picks a candidate set or fans out across it -- it hands the
        latest trace context (``candidates_path``) to a single
        ``run_optimization_handler`` call and lets forge decide what to touch.

        The call is fired for its side effect on the phase-exit latch as much as
        for the patch it may produce: once forge has looked at the trace and
        returned -- whether or not it nominated anything -- the pass is complete
        for this macro cycle (see :func:`mark_kernel_auto_pass_complete`). Skip
        it only when there is no trace to hand over.
        """
        cached = self.shared_state.last_trace_analyze or {}
        candidates_path = str(cached.get("candidates_path") or "")
        if not candidates_path:
            log.info("KERNEL entry: skip kernel_opt; no candidates_path")
            return
        log.info(
            "KERNEL entry: calling forge to nominate source-level kernel_opt",
        )
        try:
            from hyperloom.common.inline_step_heartbeat import inline_step_heartbeat

            from ..kernel.request_handlers import run_optimization_handler
            from .machine_state import KERNEL_HEARTBEAT_SEC

            # This step awaits a subprocess that can run for an hour. Without a
            # re-stamped progress marker the idle guard cannot tell a working
            # phase from a stuck one, and the phase is unobservable throughout.
            def _stamp(when: float) -> None:
                self.shared_state.kernel_inline_step_seen_unix = when

            async with inline_step_heartbeat(stamp=_stamp, interval_sec=KERNEL_HEARTBEAT_SEC):
                result = await run_optimization_handler(
                    {
                        "candidates_path": candidates_path,
                        "session_id": self.session_dir.name,
                    },
                    session_dir=self.session_dir,
                    record_partial=self._record_kernel_opt_partial,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry run_optimization after GEMM failed")
            result = {
                "status": "failed",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        finally:
            # Stamped even for an empty nomination or a failure: a hot kernel
            # forge looked at and passed over leaves no ledger row and stays
            # "untried" forever, so the phase-exit predicate never goes quiet
            # unless a completed pass answers for it (design doc s9). This is
            # the one place that records the pass ran this cycle.
            from .machine_state import mark_kernel_auto_pass_complete

            mark_kernel_auto_pass_complete(self.shared_state)
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "run_optimization_done",
                    "status": result.get("status", "ok") if isinstance(result, dict) else "failed",
                    "result": result,
                    "source": "kernel_entry_auto",
                },
                priority=1,
            )
        )
        if isinstance(result, dict):
            self.shared_state.record_kernel_opt(result)
        self.shared_state.save(self.session_dir)

    def _fusion_required_before_kernel_opt(self) -> bool:
        """Gate the forge-fusion step in KERNEL entry.

        Runs only when: not disabled by ``HYPERLOOM_SKIP_FUSION``, the framework is
        fusion-eligible (sglang/vllm), a decode trace exists to discover from, no
        fusion already succeeded this session (idempotent re-entry), and forge-fusion
        has not spent its retries aborting on infrastructure.
        """
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
            return False
        if isinstance(last, dict) and last.get("infrastructure_abort"):
            # An abort judged nothing, so it must stay retryable -- but not
            # forever. ``no_git_workspace`` does not heal mid-session, and every
            # retry re-runs LLM discovery before failing in the same place, so an
            # uncapped retry spends gateway budget to relearn the same answer.
            #
            # Capping is not the old bug returning: the record still reads
            # ``failed`` with an ``error_class``, so the run is reported as
            # infrastructure that gave up, not as "this model has no fusion
            # opportunity".
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

    async def _run_collective_integration(
        self,
        result: dict,
        inputs: "_collective_recovery.IntegrationInputs",
        *,
        preapplied: dict,
        backup_root: Path,
        apply_checkpoint: Path,
    ) -> dict:
        """Run the E2E integrate round, or describe why it could not run.

        Every failure path returns a REVERT result rather than raising, so the
        caller always has a decision to settle and a patch state to unwind.
        """
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
        """Resolve the decision and finish the revert a non-KEEP owes.

        Mutates ``integ`` in place and returns the settled decision. A patch the
        session cannot prove reverted stays flagged ``recovery_required`` so the
        next run picks it up.
        """
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
            # Settled, not complete: an already-finalized manifest must not be
            # finalized again even when its sweep was partial.
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
        """Promote an E2E-validated Collective KEEP through the current_best lift.

        A no-op when the patch is already stacked, or when the lift refuses a
        winner that does not beat the live throughput anchor.
        """
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
            source="collective_promote",
            ts=ts,
        )
        total_gain = (new_tput - baseline_tput) / baseline_tput * 100.0
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_collective_promotion(
                self.session_dir,
                integration_id=integration_id,
                kernel_id=str(collective_result.get("kernel_id") or ""),
                baseline_tput=baseline_tput,
                new_tput=new_tput,
                gain_pct=incremental_gain,
                patch_path=patch,
                target_file=str(collective_result.get("source_file") or integrate_result.get("target_file") or ""),
                collective_op=str(collective_result.get("collective_op") or ""),
                world_size=collective_result.get("world_size"),
                kernel_speedup=collective_result.get("kernel_speedup"),
                configuration=envs,
                ts=ts,
            )
            instrument.record_session_validation(
                self.session_dir,
                baseline_tput=baseline_tput,
                validated_tput=new_tput,
                validated_gain_pct=total_gain,
                stack_len=self.shared_state.cumulative_gain_validated_stack_len,
                source="collective_promote",
                measurement_basis="e2e_rebench",
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("record_collective_promotion failed", exc_info=True)
            trace_recording_skipped(
                "kernel_collective",
                reason="caller raised before the recorder",
                entity=integration_id,
                error=exc,
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
        """Record the forge-fusion result + surface it on the bus.

        Hands a KEPT fusion (source patch + env flags) to ``integrate_handler``
        for the real e2e re-baseline / adopt decision.

        Stamps a ``fusion_run_id`` before the state write. ``last_fusion`` is
        overwritten on every run while ``last_fusion_integrate`` is only
        overwritten when integration actually runs, so without an id shared by
        the pair a later run silently inherits the previous round's e2e
        verdict. Readers must treat the two as one run only when the ids match.
        """
        status = str(result.get("status") or "unknown") if isinstance(result, dict) else "failed"
        if isinstance(result, dict) and result.get("infrastructure_abort"):
            # Counted on the session, not on the record: ``last_fusion`` is
            # replaced by every run, so a timeout or a handler crash landing
            # between two aborts would carry no count forward and hand the cap
            # back a clean slate on every other entry.
            spent = _as_int(getattr(self.shared_state, "fusion_infra_aborts", 0))
            try:
                self.shared_state.fusion_infra_aborts = spent + 1
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
        """Queue every KEPT forge-fusion sibling for the shared e2e integrate lane.

        Each self-nominated recipe is one independent sibling patch. Rather than
        integrate inline here (one patch, one blocking cold-boot re-baseline), we
        write each sibling as a ``status="pending"`` record and let the SWEEP-entry
        drain (``_drain_pending_keep_integrates``) run them through the same lane
        the rewrite/gemm lanes use -- so same-file collapse, cross-file
        independence, and the patch budget all apply for free.

        forge-fusion is env-gated: the fused path is inert until its env flag is
        set, so ``enqueue_nominated_patch`` stamps the flag onto the record and the
        drain's resolver merges it into the re-baseline server's envs. The
        fusion-specific keep bar (``HYPERLOOM_FUSION_KEEP_PCT``, default 3.0) and
        the ``action="fusion"`` promotion label ride on the record too.

        An empty ``patches`` array (a run that kept nothing) is a valid no-op.
        """
        import os

        from ..kernel._kernel_decisions import enqueue_nominated_patch
        from ..kernel.nomination_result import parse_outcome

        outcome = parse_outcome(result)
        if outcome.is_empty:
            log.info("KERNEL entry: fusion KEPT but nominated no usable sibling; nothing to queue")
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
            "KERNEL entry: queued %d/%d fusion sibling(s) for SWEEP-entry integrate",
            queued,
            len(outcome.patches),
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 - best-effort persist; drain reloads state
            pass

    def _current_tput_from_validated_gain(self) -> float:
        """Project current tput from ``baseline_tput * (1 + cumulative_gain_validated/100)``; 0.0 when baseline unknown (watermark not-yet-armed).

        Returns:
            The projected current throughput, or ``0.0`` when the baseline is
            unknown.
        """
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
        """True iff projected tput crossed the watermark over ``last_roofline_tput`` (False until PRELUDE roofline ran, or while auto_roofline_pending_task_id is in-flight).

        Returns:
            ``True`` when a fresh roofline is warranted because projected tput
            crossed the watermark ratio; ``False`` otherwise (including the
            bootstrap and in-flight re-arm guards, and once the failure streak
            has exhausted ``_MAX_ROOFLINE_FAILURE_RETRIES``).
        """
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
        """Drop an in-flight marker that names a roofline which already finished.

        ``auto_roofline_pending_task_id`` gates the watermark so two rooflines
        never run at once, and it is cleared when the task reports back. A task
        that was deduplicated into an already-finished attempt reports nothing,
        so the marker it left behind gated the watermark permanently — and it
        survives into the next process, because the marker is persisted state.
        Whoever resumed the session inherited a gate that nothing could open.
        """
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
        """Enqueue a fresh roofline if the watermark crossed; idempotency-keyed via ``reason``, stamps auto_roofline_pending_task_id. Returns True when enqueued.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            ``True`` if a roofline task was enqueued, else ``False``.
        """
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
        """Return a cached programmatic_handler result if applicable (cache key last_trace_analyze).

        Args:
            kind: The kernel request kind; only ``trace_analyze`` is cacheable.
            payload: The merged request payload; its ``trace_input`` /
                ``trace_dir`` must match the cached entry for a hit.

        Returns:
            A synthesized cached result dict on a cache hit, else ``None``.
        """
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
