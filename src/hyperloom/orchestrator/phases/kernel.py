# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL_AGENT phase handler for the fusion, GEMM, and GEAK lanes."""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging as _logging
import math
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import uuid
from concurrent.futures import CancelledError as FuturesCancelledError
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from . import geak_rebench as _geak_rebench, machine_state as _phase_state
from .kernel_round_budget import controller_round_stop, integrate_within_phase, measurement_round_sec, rebench_round_sec
from hyperloom.common.coerce import first_str, to_float, to_int
from hyperloom.common.env import env_bool, env_float, env_int
from hyperloom.common import gain_math
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_FALLBACK_FAILED, REBENCH_NO_PROMOTE
from hyperloom.common.io import WriteOptions, atomic_write_json
from hyperloom.common.jsonio import read_json
from hyperloom.common.launch_log_evidence import recipe_server_launch
from hyperloom.common.perf_metric import axis_of, graded_axes_of
from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_KERNEL,
)
from hyperloom.inference_optimizer.breakdown.recorder import tool_versions
from ..actions.executors._recipe_script import resolve_launch_server_script
from ..actions.executors._workload_envs import GEAK_METRIC_OUTPUT, geak_metric_axis
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    ROUTE_FORGE,
    ROUTE_GEAK,
    kernel_event_id,
    record_geak_attempts,
    reject_geak_attempts,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import (
    FusionRun,
    GeakDelegation,
    GeakProduct,
    GeakRebenchAttempt,
    GemmTuningRun,
    IntegrateVerdict,
    KernelExit,
    KernelReprofile,
    KernelRewriteRun,
    LaneRunFields,
)
from ..actions.stop_attribution import stopped_by_the_run_class
from hyperloom.inference_optimizer.session.optimization_journal import (
    KIND_GEMM_TUNING,
    OUTCOME_KEEP,
    JournalEntry,
)
from ..state.analysis_failures import analysis_skip_reason, remembered_trace_failure
from ..state.shared_state import resolve_graded_comparison
from ..state.task_registry import TERMINAL_STATES, Task, TaskNotFound, TaskSpec
from ..bus.message_bus import Message
from ..kernel.geak_config import (
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    geak_acceptance_specs,
    geak_spec_name,
    _accepted_config_as_variant,
    _accepted_config_controls,
)
from ..kernel import kernel_decisions
from ..kernel.lane_budget import INSUFFICIENT_BUDGET
from ..actions.executors._gpu_pin import (
    coerce_tp,
    resolve_gpu_pin,
    _resolve_handoff_gpu_identity,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_physical_gpus,
    _resolve_handoff_tp,
    _resolve_serving_fidelity,
)
from ..collaborator import CoordinatorCollaborator

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import KernelEventRecorder
    from .machine import Transition

log = _logging.getLogger(__name__)

# |measurement_divergence_pct| above this (GEAK vs orchestrator, same config) is logged as a measurement-mismatch
# warning at geak promote.
_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT: float = 3.0

_GEAK_RUNNER_MODULE = "hyperloom.orchestrator.kernel.geak_runner"

# What the phase, not the Controller subprocess, decides about a rewrite-controller attempt.
_CONTROLLER_ATTEMPT_FACTS = ("macro_cycle", "handoff_dir", "budget_minutes", "hard_timeout_sec")

# How long the GEAK handoff waits for the servers it stopped to release their VRAM.
_GEAK_GPU_FREE_WAIT_SEC: float = 120.0

ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline

# Consecutive roofline failures tolerated before the watermark stops re-arming.
MAX_ROOFLINE_FAILURE_RETRIES: int = 3

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


def _record_gemm_e2e_envelope(
    result: dict[str, Any],
    stacked_envs: dict[str, str],
    *,
    kept: list[dict[str, Any]],
    reverted: list[dict[str, Any]],
    faults: list[dict[str, Any]],
) -> None:
    # Orchestration must never see the raw combined recommended_env and issue a bundled integrate, so the
    # E2E-validated envs replace it and the tuners' own record moves to the ``_raw`` keys.
    result["e2e_results"] = {"kept": kept, "reverted": reverted, "faults": faults}
    result["recommended_env_raw"] = dict(result.get("recommended_env") or {})
    result["extra_envs_raw"] = dict(result.get("extra_envs") or {})
    result["recommended_env"] = dict(stacked_envs)
    result["extra_envs"] = dict(stacked_envs)


def _bench_envs(bench: Mapping[str, Any]) -> dict[str, Any]:
    envs = bench.get("envs")
    return dict(envs) if isinstance(envs, dict) else {}


def _first_text(mapping: Mapping[str, Any], *keys: str, default: str = "") -> str:
    # The first of ``keys`` whose value renders as non-empty text.
    for key in keys:
        text = str(mapping.get(key) or "")
        if text:
            return text
    return default


def _served_replay_result(path: str, root: Path) -> dict[str, Any] | None:
    """The orchestrator's own replay result at ``path``, or ``None`` when it is not a served measurement."""
    # GEAK's own results are not the orchestrator's baseline, and the
    # overlay probe is not a served measurement. Match path COMPONENTS
    # below the runs root, never the whole string: a substring test
    # excludes every result the moment an ancestor directory happens
    # to contain "geak" (a campaign rooted at .../hlgeak_24h_run/ hits
    # this), which silently degrades the shape to the 1024 default
    # this method exists to replace.
    try:
        rel_parts = Path(path).relative_to(root).parts
    except ValueError:
        return None
    if "geak" in rel_parts or "_baseline_source_overlay" in rel_parts:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    completed = int(raw.get("completed") or 0)
    tin = int(raw.get("total_input_tokens") or 0)
    tout = int(raw.get("total_output_tokens") or 0)
    tput = float(raw.get("output_throughput") or 0.0)
    if completed <= 0 or tin <= 0 or tout <= 0 or tput <= 0:
        return None
    return raw


def _read_dispatch_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _dispatch_key(row: Mapping[str, str], key_cols: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in key_cols)


def _merged_dispatch_columns(column_sets: list[list[str]]) -> list[str]:
    all_columns = list(column_sets[0])
    for columns in column_sets[1:]:
        for column in columns:
            if column not in all_columns:
                insert_at = all_columns.index("tflops") if "tflops" in all_columns else len(all_columns)
                all_columns.insert(insert_at, column)
    return all_columns


def _normalized_dispatch_rows(rows: list[dict[str, str]], all_columns: list[str]) -> list[dict[str, str]]:
    fill = {"xbf16": "0", "run_1stage": "0", "ksplit": "0"}
    return [
        {column: fill.get(column, "0") if row.get(column) is None else row[column] for column in all_columns}
        for row in rows
    ]


def _dispatch_key_columns(all_columns: list[str], keys_from: Path | None, untuned_stem: str) -> list[str]:
    key_cols: list[str] = []
    untuned_path = keys_from / f"{untuned_stem}.csv" if keys_from is not None else None
    if untuned_path is not None and untuned_path.is_file():
        key_cols = [column for column in _read_dispatch_csv(untuned_path)[0] if column in all_columns]
    if not key_cols:
        key_cols = [column for column in ("M", "N", "K") if column in all_columns]
    for column in ("gfx", "cu_num", "_tag"):
        if column in all_columns and column not in key_cols:
            key_cols.append(column)
    return key_cols


def _fastest_dispatch_row(row: Mapping[str, str]) -> float:
    try:
        return float(row.get("us", ""))
    except (TypeError, ValueError):
        return math.inf


def _deduplicate_dispatch_rows(
    rows: list[dict[str, str]], key_cols: list[str], all_columns: list[str], label: str, env_var: str
) -> list[dict[str, str]] | None:
    keys = [_dispatch_key(row, key_cols) for row in rows]
    if len(set(keys)) == len(keys):
        return rows
    if "us" not in all_columns:
        log.warning(
            "gemm E2E: %s has duplicate dispatch keys for %s but no 'us' column to select the fastest row",
            label,
            env_var,
        )
        return None
    # Keep the fastest (smallest us) row per dispatch key; ties keep the first row seen (stable), NaN-like
    # values sort last.
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for key, row in zip(keys, rows):
        if key not in best or _fastest_dispatch_row(row) < _fastest_dispatch_row(best[key]):
            best[key] = row
    log.info("gemm E2E: removed %d duplicate %s row(s) for %s", len(rows) - len(best), label, env_var)
    return list(best.values())


def _integrate_tput_and_gain(verdict: Mapping[str, Any]) -> tuple[float, float]:
    measurement = verdict.get("bench_result") or verdict
    tput = float(measurement.get("output_throughput", verdict.get("new_tput")) or 0.0)
    return tput, float(verdict.get("gain_pct") or 0.0)


def _candidate_envs(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(key).strip() and str(value).strip()}


def _observed_identity_digest(declared: str, flags: str, server_identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "declared_launch_identity": declared,
            "observed_server_launch_flags": flags,
            "observed_server_identity": server_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _withheld_targets(result: object) -> int:
    """How many discovered targets a fusion round's lane ceiling never funded."""
    if not isinstance(result, dict):
        return 0
    nomination = result.get("nomination")
    if not isinstance(nomination, dict):
        return 0
    withheld = nomination.get("withheld")
    return max(0, withheld) if isinstance(withheld, int) and not isinstance(withheld, bool) else 0


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


def _integrate_run_name(tuner_name: str) -> str:
    """Name of the integrate run that serves a tuner's candidate."""
    return f"integrate-gemm_tune_{tuner_name}"


def _integrate_server_logs(session_dir: Path, tuner_name: str) -> list[Path]:
    """Server logs for a tuner's integrate run, retries included, oldest first."""
    from ..kernel.gemm_shape_coverage import integrate_server_logs

    return integrate_server_logs(session_dir, _integrate_run_name(tuner_name))


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
        return "e2e_paired_entry_reference_to_tuned"
    return f"e2e_paired_entry_reference_to_tuned_{getattr(verdict, 'reason', 'unknown')}"


def _covered_step(kind: str, params: dict[str, Any], *, idempotency_key: str) -> Task:
    """An unpersisted task for a step run under the ``kernel_agent`` task's lanes."""
    return Task(task_id=uuid.uuid4().hex, kind=kind, state="running", params=params, idempotency_key=idempotency_key)


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
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_integrate_verdict

    record_integrate_verdict(
        macro_cycle=macro_cycle,
        verdict=IntegrateVerdict(
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
        ),
    )


def geak_absent_backends(eval_dir: str) -> dict[str, str]:
    """``{backend: probe}`` for the backends GEAK's environment report lists as absent.

    GEAK records them in ``<eval_dir>/env_report.json`` and deep in its final
    report; they are provisioning gaps, and nothing else carries them to the
    optimizer's log or reports.
    """
    if not eval_dir:
        return {}
    try:
        report = json.loads((Path(eval_dir) / "env_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    absent = report.get("absent_backends") if isinstance(report, dict) else None
    if not isinstance(absent, dict):
        return {}
    return {
        str(name): str(entry.get("probe") or "") if isinstance(entry, dict) else str(entry or "")
        for name, entry in absent.items()
    }


class EnvSpecUnserializable(Exception):
    """The accepted launch configuration could not be serialized into an env spec.

    Only this failure is an ``invalid_env_spec``. Everything else the GEAK handoff
    raises while it is built -- a native-AgentX proposal refusing an unverified Magpie
    launch, a malformed env var, a state field that is not a number -- is a different
    fault and propagates to the caller rather than being reported as one.
    """


@dataclass(frozen=True)
class KernelExitFacts:
    """What the phase machine's KERNEL exit reads from the task queue."""

    agent_in_flight: bool


@dataclass(frozen=True)
class GeakRunnerFacts:
    """How the GEAK runner process ran, as the timeline and the phase evidence report it.

    These are properties of the dispatch, not of the product GEAK shipped, so
    they travel beside ``result.json``'s content and never merge into it: the
    persisted ``geak_result`` has to stay byte-comparable with the file
    :func:`geak_candidate_matches` reads back on the next KERNEL entry.
    """

    started_at: str = ""
    duration_sec: float | None = None
    runner_timeout_sec: int | None = None
    kill_timeout_sec: int | None = None


@dataclass(frozen=True)
class GeakRunnerOutcome:
    """What one GEAK runner dispatch leaves for the phase to settle."""

    result: dict[str, Any] | None = None
    facts: GeakRunnerFacts = GeakRunnerFacts()
    recovered_from: str = ""


class KernelPhase(CoordinatorCollaborator):
    """KERNEL_AGENT phase handler: drives kernel-level optimization tasks."""

    # Relative-change floor for the Forge kernel-entry reprofile: effectively "any change", absorbing float noise.
    _REPROFILE_CHANGE_TOL: float = 1e-5

    async def exit_facts(self) -> KernelExitFacts:
        """Advance the idle streak for this tick and report the facts the KERNEL exit decision reads."""
        await self._track_idle_streak()
        in_kernel = str(self.shared_state.phase or "").upper() == _phase_state.PHASE_KERNEL_AGENT
        return KernelExitFacts(agent_in_flight=in_kernel and await self._agent_in_flight())

    async def _inflight_task_ids(self) -> tuple[str, ...]:
        """Return the ids of queued/running tasks doing KERNEL-lane work."""
        kinds = _phase_state.PHASE_ALLOWED_ACTIONS[_phase_state.PHASE_KERNEL_AGENT]
        tasks = list(await self.tasks.queued()) + list(await self.tasks.running())
        return tuple(
            sorted(str(task.task_id) for task in tasks if str(getattr(task, "kind", "") or "").strip() in kinds)
        )

    async def _agent_in_flight(self) -> bool:
        """Whether the ``kernel_agent`` task is queued or running."""
        tasks = list(await self.tasks.queued()) + list(await self.tasks.running())
        return any(task.kind == "kernel_agent" for task in tasks)

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

    @staticmethod
    def _resolve_bench_protocol(envs: Mapping[str, Any]) -> dict[str, Any]:
        """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

        Reads an already-parsed ``benchmark.envs`` (falling back to the process
        env) and returns only the keys that resolve, so absent values leave GEAK
        on its standalone defaults. Never raises.
        """

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

    @staticmethod
    def _recipe_benchmark(recipe_path: str) -> dict[str, Any]:
        """Parse a materialized recipe once and hand back its ``benchmark`` mapping.

        Both handoff resolvers below need the same mapping out of the same file,
        so the read, the YAML parse and the shape validation live here instead of
        being repeated with their own fallbacks on each side.

        Returns ``{}`` for a missing, unreadable or malformed recipe, which each
        caller already treats as "nothing to advertise". Never raises.
        """
        import yaml

        try:
            if not recipe_path or not Path(recipe_path).is_file():
                return {}
            cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
            if not isinstance(cfg, dict):
                return {}
            bench = cfg.get("benchmark") or {}
            return bench if isinstance(bench, dict) else {}
        except (OSError, ValueError, yaml.YAMLError):
            log.warning("recipe: could not read %r", recipe_path, exc_info=True)
            return {}

    def _observed_replay_shape(self, target_tput: float) -> dict[str, Any]:
        """Measure the trace replay's real sequence shape from OUR OWN baseline.

        The AgentX handoff's ``workload.isl/osl`` are the CLI's synthetic
        defaults (1024/1024) because a trace replay has no single ISL -- the
        corpus spans roughly 89k at p50 past 500k at p99. GEAK does not measure
        with them (the aiperf client ignores them outright), but the KERNEL
        agents still read them as the analytic serving call model when they
        synthesize GEMM/attention shapes. Left at 1024 they aim two orders of
        magnitude below the load, so the search tunes a regime nobody serves.

        Rather than hardcode corpus percentiles, derive the average shape from
        the baseline result THIS run already measured: the canonical result
        records ``total_input_tokens``/``total_output_tokens`` over ``completed``
        requests. That keeps the number sourced from a measurement instead of a
        constant that silently rots when the corpus is re-pinned.

        Returns ``{}`` whenever no usable result is on disk, so an unmeasurable
        run degrades to today's behavior rather than to a fabricated shape.
        """
        try:
            import glob as _glob

            from hyperloom.inference_optimizer.session.session_paths import runs_root

            root = runs_root(self.session_dir)
            best: tuple[float, dict[str, Any], str] | None = None
            for rp in _glob.glob(str(Path(root) / "**" / "inferencex_result.json"), recursive=True):
                raw = _served_replay_result(rp, root)
                if raw is None:
                    continue
                tput = float(raw["output_throughput"])
                # Prefer the result whose throughput IS the baseline we handed
                # over, so the shape describes the same measurement.
                err = abs(tput - target_tput) / target_tput if target_tput > 0 else 1.0
                if best is None or err < best[0]:
                    best = (err, raw, rp)
            if best is None:
                return {}
            err, raw, path = best
            completed = int(raw["completed"])
            isl = int(round(int(raw["total_input_tokens"]) / completed))
            osl = int(round(int(raw["total_output_tokens"]) / completed))
            if isl <= 0 or osl <= 0:
                return {}
            return {
                "observed_isl": isl,
                "observed_osl": osl,
                "observed_requests": completed,
                "observed_source": path,
                # 1.0 => no baseline throughput to match against, so this is the
                # best available replay result rather than a confirmed same-run one.
                "observed_tput_match_err": round(err, 6),
            }
        except (OverflowError, TypeError, ValueError):
            log.warning("workload_spec: could not derive observed replay shape", exc_info=True)
            return {}

    @staticmethod
    def _resolve_launch_server_script(bench: Mapping[str, Any]) -> str:
        """Name the server-phase script GEAK should launch through Magpie.

        GEAK infers its launcher from the recipe's ``benchmark_script``, which on
        every non-AgentX run IS a server launcher. The AgentX switch replaces
        that field with the aiperf client -- which boots a server, replays the
        corpus for ``AGENTX_DURATION``, then tears the server down in its exit
        trap. Run under ``MAGPIE_RUN_PHASE=server`` it therefore returns no pid,
        and GEAK's bench aborts before it measures a single repeat.

        Naming the builtin the client itself delegates to keeps GEAK on the
        Magpie launch path -- the reason that launcher exists, since the platform
        kernel preset, ``--trust-remote-code`` and the gpu-mem-util default are
        not flags and so cannot be recovered from the accepted-flags handoff --
        while letting its own bench repeats run again.

        Resolution mirrors ``mlperf_agentic_client.sh``: the same ``AGENTX_SERVER_SCRIPT``
        override, the same ``{framework}_{gpu}.sh`` fallback, the same
        ``<checkout>/benchmarks/`` directory and deliberately no recursive
        search, so what we advertise is the path that would have booted the
        server rather than merely a plausible one. Recipe-recorded values beat
        the ambient env because the recipe is the record of what actually ran.

        Returns "" for any non-AgentX recipe -- there GEAK's own derivation names
        the script that really launched the baseline, which is strictly better
        than anything re-derived here -- and "" whenever the builtin cannot be
        confirmed on disk, leaving current behaviour untouched. Never raises.
        """
        try:
            from hyperloom.common.agentx_workload import is_agentx_client_script

            # Only the AgentX client misleads the inference; anything else in
            # this field is the launcher GEAK should keep deriving for itself.
            if not is_agentx_client_script(str(bench.get("benchmark_script") or "").strip()):
                return ""
            return resolve_launch_server_script(bench)
        except OSError:
            log.warning("launch_server_script: could not resolve from the recipe", exc_info=True)
            return ""

    def _geak_timeouts(self) -> tuple[int, int, bool]:
        """Resolve the GEAK e2e timeouts from the live run budget."""
        # Standalone fallback ONLY: the 12h (43200s) default applies when no run deadline is set (budget_known=False).
        env_default_timeout = env_int("GEAK_E2E_TIMEOUT_S", default=43200)
        deadline = self._coord.run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline.remaining()
        grace = self.shared_state.closing_reserve_sec()
        margin = env_float("GEAK_BUDGET_MARGIN_S", default=300.0)
        # Reserve the closing window: kill the subprocess with at least ``grace`` left.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget: cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(self.shared_state)
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # A GEAK result reaches the headline only through this session's own re-measurement of it, which runs in this
        # phase after GEAK returns. Which results skip it is known only then, so every hand-off leaves room for it.
        rebench_sec = rebench_round_sec(self.shared_state)
        log.info("GEAK: holding back %.0fs of the KERNEL budget to re-measure GEAK's result", rebench_sec)
        kill_budget -= rebench_sec
        # The runner self-stops ``margin`` before the hard subprocess kill, which reserves the closing-grace window.
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

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

    async def _free_gpus_for_geak(self, cards: list[int] | None) -> str | None:
        """Stop what this session left on GEAK's GPUs, then wait for them to drain; return why they did not, if so.

        The ``kernel_agent`` task holds every serving lane by now, but a lane only records that the round holding it
        ended, not that the server the round booted went with it.
        """
        from ..actions.executors._server_lifecycle import free_gpus_for_handoff

        gpus = await asyncio.to_thread(
            free_gpus_for_handoff, self.session_dir, cards, timeout_sec=_GEAK_GPU_FREE_WAIT_SEC
        )
        if gpus.reaped:
            log.warning(
                "geak handoff: stopped %d server(s) this session left running: pids=%s", len(gpus.reaped), gpus.reaped
            )
        if gpus.unverified:
            log.warning("geak handoff: cannot confirm GPUs %s are free (%s); delegating anyway", cards, gpus.unverified)
        if not gpus.busy:
            return None
        busy = ", ".join(f"GPU{card} {used:.0f}/{total:.0f} MiB" for card, used, total in gpus.busy)
        error = f"GPUs still hold VRAM after this session's leftover servers were stopped: {busy}"
        if gpus.holders:
            error += f"; serving processes still up: {'; '.join(gpus.holders)}"
        log.error("geak handoff: %s; not delegating", error)
        return error

    async def _wait_for_gpus_before_rebench(self, cards: list[int] | None) -> None:
        """Let the GPUs GEAK hands back drain before the rebench boots a server on them.

        A card still holding the VRAM of a process GEAK's run left behind fails the server's memory-balance check at
        boot, and the candidate is then dropped unvalidated. The rebench is attempted either way: this only waits.
        """
        from ..actions.executors._server_lifecycle import free_gpus_for_handoff

        gpus = await asyncio.to_thread(
            free_gpus_for_handoff, self.session_dir, cards, timeout_sec=_GEAK_GPU_FREE_WAIT_SEC
        )
        if gpus.reaped:
            log.warning(
                "geak rebench: stopped %d server(s) this session left running: pids=%s", len(gpus.reaped), gpus.reaped
            )
        if gpus.busy:
            busy = ", ".join(f"GPU{card} {used:.0f}/{total:.0f} MiB" for card, used, total in gpus.busy)
            holders = f"; serving processes still up: {'; '.join(gpus.holders)}" if gpus.holders else ""
            log.error("geak rebench: GPUs still hold VRAM after GEAK returned: %s%s; measuring anyway", busy, holders)
        elif gpus.unverified:
            log.warning("geak rebench: cannot confirm GPUs %s are free (%s)", cards, gpus.unverified)

    async def _launch_geak_runner(
        self, handoff_path: Path, *, timeouts: tuple[int, int, bool], from_phase: str
    ) -> subprocess.CompletedProcess:
        """Run geak_runner on ``handoff_path`` under :meth:`_geak_timeouts`; raise TimeoutExpired at its kill cap.

        geak_runner gives run_e2e its own SIGTERM grace to flush result.json before ``runner_timeout`` +
        ``GEAK_FLUSH_GRACE_S``, inside the kill cap. The launch marks the run, so whatever it left running is
        reaped when it returns, however it detached.
        """
        from ..actions.executors._subprocess_kill import REAP_MARK_ENV, RoundWatch, run_with_session_kill

        runner_timeout, kill_timeout, budget_known = timeouts
        cmd = [
            sys.executable,
            "-m",
            _GEAK_RUNNER_MODULE,
            str(handoff_path),
            str(handoff_path.parent),
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
        # GEAK must search and gate on the axis this session keeps candidates on: a kernel accepted on another
        # axis need not move that one. geak_metric_axis owns the mapping and its rationale.
        metric, _ = geak_metric_axis(
            benchmark_mode=str(self.shared_state.benchmark_mode or ""),
            grading=self.shared_state.grading or None,
        )
        runner_env = dict(os.environ)
        runner_env["E2E_METRIC"] = metric
        runner_env[REAP_MARK_ENV] = uuid.uuid4().hex
        # Only injection point needed for the whole GEAK chain: geak_runner and run_e2e both hand their full
        # environment to the child, so the tag reaches the Claude CLI that actually spends.
        from hyperloom.common.llm_attribution import inject_env

        inject_env(runner_env, component="geak", operation="optimize_kernel")
        return await asyncio.to_thread(
            run_with_session_kill, cmd, env=runner_env, watch=RoundWatch(timeout=kill_timeout)
        )

    def _accepted_launch_spec(self, *, agentx: bool) -> tuple[dict[str, Any], Mapping[str, Any]]:
        try:
            env_spec = self._coord.writeback.build_env_spec()
        except (OSError, TypeError, ValueError) as exc:
            raise EnvSpecUnserializable(str(exc)) from exc
        measurement = self.shared_state.current_best_measurement
        config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        if agentx and not str(config.get("server_launch_flags") or "").strip():
            evidence = measurement.get("launch_evidence")
            server_log = str(evidence.get("actual_server_log_path") or "") if isinstance(evidence, Mapping) else ""
            recipe_launch = recipe_server_launch(server_log)
            if recipe_launch.flags or recipe_launch.env:
                env_spec = self._coord.writeback.build_env_spec(
                    measurement=measurement,
                    server_launch_flags=recipe_launch.flags,
                    server_launch_envs=recipe_launch.env,
                )
        return env_spec, measurement

    @staticmethod
    def _reference_verification_status(evidence: Mapping[str, Any], *, identity_matches: bool, observed: bool) -> str:
        if identity_matches and observed:
            return "verified_observed"
        declared = (
            str(evidence.get("requested_server_args") or "").strip()
            or bool(evidence.get("requested_server_env"))
            or str(evidence.get("recipe_digest") or "").strip()
        )
        return "verified_declared_only" if identity_matches and declared else "unverified"

    def _same_config_reference(
        self, measurement: Mapping[str, Any], *, expected_identity: str, agentx: bool
    ) -> dict[str, Any]:
        measured_identity = _first_text(measurement, "declared_launch_identity")
        identity_matches = bool(expected_identity and expected_identity == measured_identity)
        evidence = measurement.get("launch_evidence")
        evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
        observed_flags = str(
            measurement.get("resolved_server_launch_flags") or evidence.get("observed_server_launch_flags") or ""
        ).strip()
        server_identity = measurement.get("observed_server_identity") or evidence.get("observed_server_identity")
        server_identity = (
            {str(key): value for key, value in sorted(server_identity.items())}
            if isinstance(server_identity, Mapping)
            else {}
        )
        observed = bool(observed_flags or server_identity)
        status = (
            "unverified_workload"
            if agentx
            else self._reference_verification_status(evidence, identity_matches=identity_matches, observed=observed)
        )
        observed_identity = _first_text(measurement, "observed_launch_identity")
        if not observed_identity and identity_matches and observed:
            observed_identity = _observed_identity_digest(measured_identity, observed_flags, server_identity)
        verified = status == "verified_observed"
        return {
            # Zero means no verified same-config reference.
            "orchestrator_best_tput_same_config": float(measurement.get("tput") or 0.0) if verified else 0.0,
            "same_config_reference_status": "verified" if verified else "unverified",
            "same_config_reference_identity": measured_identity,
            "same_config_expected_identity": expected_identity,
            "same_config_reference_workspace": _first_text(measurement, "benchmark_workspace"),
            # Additive identity semantics.
            "same_config_reference_verification_status": status,
            "same_config_reference_declared_identity": measured_identity,
            "same_config_reference_observed_identity": observed_identity,
            # GEAK compares this map with its parsed ServerArgs.
            "same_config_observed_identity": server_identity,
            "observed_server_identity": server_identity,
            "measurement_evidence": evidence,
        }

    @staticmethod
    def _geak_device_fields(recipe_envs: Mapping[str, Any]) -> dict[str, Any]:
        gpu_pin = resolve_gpu_pin(recipe_envs=recipe_envs)
        tp = coerce_tp(recipe_envs.get("TP"), os.environ.get("TP"))
        gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=gpu_pin, tp=tp)
        space = _resolve_handoff_gpu_ids_space(gpu_pin=gpu_pin)
        if space == "none":
            # An empty visibility mask means the ids are placeholders, not devices.
            log.error(
                "geak handoff: %s is set but empty; the run has no visible GPUs. "
                "gpu_ids=%s is a placeholder (gpu_ids_space=none), not a device set.",
                gpu_pin.get("var"),
                gpu_ids,
            )
        return {
            "tp": _resolve_handoff_tp(gpu_ids=gpu_ids, tp=tp),
            # The serving/optimization device set, as HIP-level ids (what the
            # consumer exports as HIP_VISIBLE_DEVICES). Logical positions inside
            # an inherited ROCR mask, a HIP/CUDA mask as-is, else 0..tp-1.
            "gpu_ids": gpu_ids,
            # Which coordinate system ``gpu_ids`` is in: "logical" (positions
            # inside the ROCR mask the child inherits), "absolute" (whole-
            # machine device ids), or "none" (the mask is set but empty -- the
            # ids are placeholders and must not be launched on). Exporting them
            # as HIP_VISIBLE_DEVICES is correct in the first two; a consumer
            # that instead writes ROCR itself needs to know which it holds.
            "gpu_ids_space": space,
            # ABSOLUTE ids + the var they came from, so a consumer that writes
            # ROCR_VISIBLE_DEVICES itself re-applies the same pin instead of
            # resetting the child to card 0.
            **({"gpu_pin": gpu_pin} if gpu_pin else {}),
        }

    def _geak_serving_fidelity(self, spec_config: Mapping[str, Any], accepted_flags: str) -> dict[str, Any]:
        server_args = str(spec_config.get("server_launch_flags") or "")
        if not server_args:
            # Reuse baseline server settings so GEAK measures the same engine config.
            from hyperloom.inference_optimizer.roofline_ceiling import read_baseline_server_args

            inherited = read_baseline_server_args(self.shared_state) or ""
            from hyperloom.inference_optimizer.grid_server_args import compose_server_args

            server_args = compose_server_args(
                inherited_args=inherited,
                variant_extra_args=accepted_flags,
                remove_args=spec_config.get("remove_args"),
                args_mode=str(spec_config.get("args_mode") or "append"),
            )
        return _resolve_serving_fidelity(
            baseline_server_args=server_args,
            state_max_model_len=int(self.shared_state.max_model_len or 0),
        )

    def _geak_workload_spec(self, bench: Mapping[str, Any]) -> dict[str, Any]:
        # Only AgentX materializations carry ``benchmark.workload_spec``; every other recipe
        # omits it so GEAK keeps today's synthetic path unchanged.
        spec = bench.get("workload_spec")
        if not isinstance(spec, dict) or not spec:
            return {}
        # Give the kernel agents the shape they must actually optimize for;
        # absence leaves GEAK on the handoff's synthetic isl/osl.
        observed = self._observed_replay_shape(float(self.shared_state.baseline_tput or 0.0))
        if not observed:
            return dict(spec)
        log.info(
            "workload_spec: observed replay shape isl=%d osl=%d over %d requests (%s)",
            observed["observed_isl"],
            observed["observed_osl"],
            observed["observed_requests"],
            observed["observed_source"],
        )
        return {**spec, **observed}

    def _geak_session_fields(self, *, agentx: bool) -> dict[str, Any]:
        state = self.shared_state
        # GEAK's E2E_METRIC and the workload spec's metric_basis are the same decision, so they
        # resolve through one helper: a handoff that named a different axis than the one KEEP is
        # decided on would have GEAK searching against a reference it was never measured against.
        e2e_metric, _ = geak_metric_axis(benchmark_mode=str(state.benchmark_mode or ""), grading=state.grading or None)
        return {
            # v2 adds baseline_env_spec; v3 adds actual GPU-pinning metadata.
            "schema_version": 3,
            "model_path": str(state.model_path or os.environ.get("MODEL_PATH", "")),
            "framework": str(state.framework or os.environ.get("FRAMEWORK", "") or "sglang"),
            **_resolve_handoff_gpu_identity(state.gpu_type or os.environ.get("GPU_TYPE", "")),
            "workload": {"isl": int(state.isl), "osl": int(state.osl), "conc": int(state.conc or 64)},
            "launch_recipe": str(state.baseline_config_path or ""),
            # AgentX canonical throughput is not a reference for GEAK's proxy workload.
            "raw_baseline_tput": 0.0 if agentx else float(state.baseline_tput or 0.0),
            "exp_root": str(self.session_dir / "geak"),
            # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the in-progress on-disk artifacts while a new
            # cycle gets a fresh dir.
            "eval_dir": str(self.session_dir / "geak" / f"e2e_cycle{int(state.macro_cycle or 0)}"),
            # GEAK owns client selection; AgentX results are proposal proxies.
            "bench_client": "auto",
            "e2e_metric": e2e_metric,
            "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
        }

    def _build_geak_handoff(self, *, agentx: bool) -> dict[str, Any]:
        cb = self.shared_state.current_best
        env_spec, measurement = self._accepted_launch_spec(agentx=agentx)
        spec_config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        accepted_flags = str(spec_config.get("extra_server_args", cb.get("extra_server_args")) or "")
        extra_envs = spec_config.get("extra_envs", cb.get("extra_envs")) or {}
        recipe_bench = self._recipe_benchmark(str(self.shared_state.baseline_config_path or ""))
        recipe_envs = _bench_envs(recipe_bench)
        # Absent => GEAK keeps deriving the launcher from the recipe, which is
        # correct everywhere except AgentX (see _resolve_launch_server_script).
        launch_server_script = self._resolve_launch_server_script(recipe_bench)
        handoff = {
            **self._geak_session_fields(agentx=agentx),
            **self._geak_device_fields(recipe_envs),
            "accepted_flags": accepted_flags,
            "accepted_env": shlex.join(f"{k}={v}" for k, v in dict(extra_envs).items()),
            **self._same_config_reference(
                measurement, expected_identity=str(env_spec.get("launch_identity") or ""), agentx=agentx
            ),
            "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
            **self._geak_serving_fidelity(spec_config, accepted_flags),
        }
        for key, value in (
            ("bench_protocol", self._resolve_bench_protocol(recipe_envs)),
            ("workload_spec", self._geak_workload_spec(recipe_bench)),
            ("launch_server_script", launch_server_script),
            # Full layered environment and its matching measurement identity.
            ("baseline_env_spec", env_spec),
        ):
            if value:
                handoff[key] = value
        if agentx:
            log.info("GEAK results remain proposal proxies; canonical AgentX validation remains in Hyperloom.")
            # Force native only when the recipe builtin could not be named. The
            # AgentX client delegates its server phase to that builtin, so with it
            # in hand (set as ``launch_server_script`` above) GEAK derives the
            # Magpie launcher -- which writes a pid and recovers the non-flag
            # presets -- instead of its adapter default; without it there is no
            # Magpie server phase to launch. Flags/env stay aligned either way:
            # baseline_env_spec still folds the recorded EXTRA_ATOM_ARGS.
            if not launch_server_script:
                handoff["bench_launcher"] = "native"

        from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy

        seed_native_geak_proxy(handoff, benchmark=recipe_bench, measurement=measurement)
        return handoff

    def _geak_result_path(self) -> Path:
        return self.session_dir / "geak" / "result.json"

    def _geak_settled_replay(self, candidate: dict[str, Any], *, agentx: bool) -> bool:
        """Whether ``candidate`` is a result this session already settled."""
        return _geak_rebench.geak_candidate_is_adjudicated(
            self.shared_state.geak_result, candidate, harness_can_replay=not agentx
        )

    def _settle_geak_skip(self, result: dict[str, Any]) -> None:
        # A settled verdict is left standing: a later failure records itself without
        # retiring the candidate that already adjudicated.
        state = self.shared_state
        if not _geak_rebench.geak_verdict_is_terminal(state.geak_result):
            state.geak_result = result
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "error_class": result.get("error_class"),
                "error": (str(result.get("error") or "")[:500] or None),
            }
        )
        state.save(self.session_dir)

    def _finish_geak_skip(
        self, result: dict[str, Any], *, handoff: dict[str, Any], facts: GeakRunnerFacts = GeakRunnerFacts()
    ) -> None:
        self._record_geak_delegation_timeline(result, handoff=handoff, facts=facts)
        self._settle_geak_skip(result)

    def _promote_recovered_geak_result(
        self,
        result: dict[str, Any],
        *,
        handoff: dict[str, Any],
        recovered_from: str,
        facts: GeakRunnerFacts = GeakRunnerFacts(),
    ) -> bool:
        state = self.shared_state
        self._record_geak_delegation_timeline(result, handoff=handoff, recovered_from_disk=True, facts=facts)
        previous = state.geak_result or {}
        if previous.get("kernel_event_id") and _geak_rebench.geak_candidate_matches(previous, result):
            result["kernel_event_id"] = previous["kernel_event_id"]
        state.geak_result = result
        self._record_geak_measurement(result)
        # Rebench-first: record the recovered win as an UNVALIDATED candidate;
        # the caller enqueues the main-flow rebench that writes the headline.
        if not self._record_geak_candidate(result):
            state.save(self.session_dir)
            return False
        self._record_geak_kernel_journey(result)
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "recovered_from": recovered_from,
                **({"runner_timeout_s": facts.runner_timeout_sec} if facts.runner_timeout_sec is not None else {}),
            }
        )
        state.save(self.session_dir)
        return True

    def _refuse_geak_for_budget(self, handoff: dict[str, Any], *, runner_timeout: int, min_run: int) -> None:
        log.warning(
            "GEAK: only %ds budget remains (< min %ds); skipping e2e "
            "and winding down to SWEEP so the closing report runs in time.",
            runner_timeout,
            min_run,
        )
        self._finish_geak_skip(
            {
                "status": "skipped",
                "error_class": INSUFFICIENT_BUDGET,
                "error": (
                    f"only {runner_timeout}s of KERNEL budget remained "
                    f"(< min {min_run}s); skipped to protect the closing report window"
                ),
                "runner_timeout_s": runner_timeout,
            },
            handoff=handoff,
        )

    def _settle_geak_runner_exit(
        self, proc: subprocess.CompletedProcess, *, handoff: dict[str, Any], agentx: bool, facts: GeakRunnerFacts
    ) -> dict[str, Any] | None:
        result_path = self._geak_result_path()
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            log.warning("GEAK runner rc=%s: %s", proc.returncode, stderr_tail)
        result = read_json(result_path, {}, require_dict=True)
        if not result:
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "no_result_json",
                    "error": f"runner rc={proc.returncode} produced no parseable result.json at {result_path}",
                    "stderr_tail": stderr_tail,
                    "returncode": proc.returncode,
                },
                handoff=handoff,
                facts=facts,
            )
            return None
        if agentx and self._geak_settled_replay(result, agentx=agentx):
            # The runner left the candidate this session already settled, so it shipped no product: recording it
            # would retire the verdict and re-enqueue the revalidation that produced it. The file stays for the
            # next run to overwrite.
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "no_new_geak_product",
                    "error": f"runner rc={proc.returncode} left the already-adjudicated result.json at {result_path}",
                    "stderr_tail": stderr_tail,
                },
                handoff=handoff,
            )
            return None
        # Carry the actual exit code so the breakdown can audit a nonzero rc. Nothing else is added: the persisted
        # record stays the product GEAK wrote, so the next KERNEL entry recognizes its own settled candidate.
        result.setdefault("returncode", proc.returncode)
        return result

    async def _dispatch_geak_runner(
        self, handoff: dict[str, Any], *, from_phase: str, agentx: bool
    ) -> GeakRunnerOutcome:
        out_dir = self.session_dir / "geak"
        # Budget-aware timeouts: shrink to the remaining run deadline and always reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._geak_timeouts()
        min_run = env_int("GEAK_MIN_RUN_S", default=600)
        if budget_known and runner_timeout < min_run:
            self._refuse_geak_for_budget(handoff, runner_timeout=runner_timeout, min_run=min_run)
            return GeakRunnerOutcome()
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()

        def _facts() -> GeakRunnerFacts:
            return GeakRunnerFacts(
                started_at=started_at,
                duration_sec=time.monotonic() - started,
                runner_timeout_sec=runner_timeout,
                kill_timeout_sec=kill_timeout,
            )

        try:
            proc = await self._launch_geak_runner(
                out_dir / "handoff.json",
                timeouts=(runner_timeout, kill_timeout, budget_known),
                from_phase=from_phase,
            )
        except subprocess.TimeoutExpired:
            log.warning(
                "GEAK runner exceeded kill_timeout=%ds; SIGTERM'd to let it flush, then reclaimed the closing window",
                kill_timeout,
            )
            # The graceful SIGTERM gives run_e2e a window to flush result.json; keep a real win instead of discarding
            # the phase as a timeout.
            facts = _facts()
            flushed = read_json(self._geak_result_path(), {}, require_dict=True)
            if flushed.get("status") == "ok" and not (agentx and self._geak_settled_replay(flushed, agentx=agentx)):
                log.info(
                    "GEAK flushed an OK result.json under SIGTERM grace; promoting the recovered win despite the cap."
                )
                return GeakRunnerOutcome(flushed, facts, "sigterm_flushed_result_json")
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "timeout",
                    "error": f"GEAK e2e killed after {kill_timeout}s (budget-capped); closing window preserved",
                    "runner_timeout_s": runner_timeout,
                    "kill_timeout_s": kill_timeout,
                },
                handoff=handoff,
                facts=facts,
            )
            return GeakRunnerOutcome()
        except (OSError, ValueError) as exc:
            log.exception("GEAK runner crashed")
            self._finish_geak_skip(
                {"status": "error", "error_class": "runner_crashed", "error": repr(exc)},
                handoff=handoff,
                facts=_facts(),
            )
            return GeakRunnerOutcome()
        facts = _facts()
        return GeakRunnerOutcome(
            self._settle_geak_runner_exit(proc, handoff=handoff, agentx=agentx, facts=facts), facts
        )

    def _open_geak_handoff(self, *, agentx: bool) -> dict[str, Any] | None:
        try:
            handoff = self._build_geak_handoff(agentx=agentx)
        except EnvSpecUnserializable as exc:
            log.exception("geak: cannot serialize the accepted launch configuration")
            recorder = self.timeline()
            if recorder is not None:
                recorder.finish_failed(stage="geak_handoff", error_class="invalid_env_spec", message=str(exc))
            self._settle_geak_skip({"status": "error", "error_class": "invalid_env_spec", "error": str(exc)})
            return None
        out_dir = self.session_dir / "geak"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
        recorder = self.timeline()
        if recorder is not None:
            recorder.enter_stage("geak_delegation")
            recorder.record_geak_handoff(handoff)
        return handoff

    async def _promote_crashed_geak_result(
        self, handoff: dict[str, Any], *, agentx: bool, cards: list[int] | None
    ) -> bool:
        # A validated result.json written before a coordinator crash is promoted on resume, guarded by
        # ``_geak_win_already_recorded`` so a prior cycle's result.json does not short-circuit a fresh entry. A
        # result this session already settled is not recovered; new candidate evidence remains eligible.
        recovered = read_json(self._geak_result_path(), {}, require_dict=True)
        if (
            recovered.get("status") != "ok"
            or self._geak_win_already_recorded()
            or self._geak_settled_replay(recovered, agentx=agentx)
        ):
            return False
        log.info(
            "GEAK result.json exists but state has no recorded win (crash before handback); promoting recovered result."
        )
        if self._promote_recovered_geak_result(recovered, handoff=handoff, recovered_from="existing_result_json"):
            await self._revalidate_geak_candidate(reason="geak_e2e_win_recovered", handoff_cards=cards)
        return True

    def _refuse_phantom_baseline_gain(self, result: dict[str, Any]) -> None:
        # A GEAK run whose baseline ref failed to reproduce ``orchestrator_best_tput_same_config`` optimized
        # against a phantom baseline, so its gain is non-comparable -- never promote it.
        log.warning(
            "GEAK baseline_reproduction_failed: ref did not match "
            "orchestrator best (%s); refusing to promote a phantom-baseline gain",
            result.get("error"),
        )
        self._settle_geak_skip(
            {
                "status": "baseline_reproduction_failed",
                "error_class": "baseline_reproduction_failed",
                "error": (
                    str(result.get("error") or "")[:500] or "GEAK baseline ref != orchestrator best (env_spec mismatch)"
                ),
                "ref_tput": result.get("ref_tput"),
                "orchestrator_best_tput_same_config": result.get("orchestrator_best_tput_same_config"),
                **({"absent_backends": result["absent_backends"]} if result.get("absent_backends") else {}),
            }
        )

    async def _publish_geak_e2e_done(self, result: dict[str, Any], *, facts: GeakRunnerFacts) -> None:
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "runner_timeout_s": facts.runner_timeout_sec,
            }
        )
        self.shared_state.save(self.session_dir)
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
                    "result_path": str(self._geak_result_path()),
                },
            )
        )
        self.shared_state.save(self.session_dir)

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run)."""
        state = self.shared_state
        agentx = not _geak_rebench.geak_harness_replays_workload(state)
        handoff = self._open_geak_handoff(agentx=agentx)
        if handoff is None:
            return
        cards = _resolve_handoff_physical_gpus(
            gpu_ids=handoff["gpu_ids"], gpu_ids_space=handoff["gpu_ids_space"], gpu_pin=handoff.get("gpu_pin") or {}
        )
        if await self._promote_crashed_geak_result(handoff, agentx=agentx, cards=cards):
            return

        still_busy = await self._free_gpus_for_geak(cards)
        if still_busy:
            self._finish_geak_skip(
                {"status": "error", "error_class": "gpus_busy_at_handoff", "error": still_busy}, handoff=handoff
            )
            return

        outcome = await self._dispatch_geak_runner(handoff, from_phase=from_phase, agentx=agentx)
        result, facts = outcome.result, outcome.facts
        if result is None:
            return
        if outcome.recovered_from:
            if self._promote_recovered_geak_result(
                result, handoff=handoff, recovered_from=outcome.recovered_from, facts=facts
            ):
                await self._revalidate_geak_candidate(reason="geak_e2e_win_sigterm_recovered", handoff_cards=cards)
            return

        self._record_geak_delegation_timeline(result, handoff=handoff, facts=facts)
        state.geak_result = result
        self._record_geak_measurement(result)
        status = str(result.get("status") or "")
        if status == "baseline_reproduction_failed":
            self._refuse_phantom_baseline_gain(result)
            return

        # Rebench-first: record the win as an UNVALIDATED candidate only; the headline is written later from the
        # measured rebench. The same-harness rebench is the only path that writes it, and a no_gain headline over an
        # accepted, parity-checked kernel still earns the measurement -- without it the kernel is lost with no number.
        if not self._record_geak_candidate(result):
            state.save(self.session_dir)
            return
        self._record_geak_kernel_journey(result)
        if status == "ok":
            await self._revalidate_geak_candidate(reason="geak_e2e_win", handoff_cards=cards)
        elif _geak_has_accepted_kernel(result):
            await self._revalidate_geak_candidate(reason="geak_e2e_accepted_kernel", handoff_cards=cards)
        await self._publish_geak_e2e_done(result, facts=facts)

    async def _revalidate_geak_candidate(self, *, reason: str, handoff_cards: list[int] | None) -> None:
        """Measure the GEAK candidate on the orchestrator harness and settle its verdict.

        The 2b rebench runs as a step of the ``kernel_agent`` task, under the
        lanes it holds, and its result goes through the same promotion a
        dispatched explore would. A candidate the grid cannot carry goes to the
        GEAK-harness replay (2a) instead. ``handoff_cards`` are drained only
        when a measurement follows.
        """
        state = self.shared_state
        params = self._coord.writeback.geak_rebench_params(reason=reason)
        from .geak_native_revalidation import ORIGIN, finish_invalid_source, revalidate_source

        if params.get("reason") == "geak_native_source_invalid":
            finish_invalid_source(self._coord, str(params["error"]))
            return
        skip_reason = params.get("reason") if params.get("skipped") else None
        if skip_reason == "geak_invalid_config":
            return
        if skip_reason == "geak_no_material":
            self._coord.writeback.settle_geak_no_material()
            state.save(self.session_dir)
            return
        await self._wait_for_gpus_before_rebench(handoff_cards)
        if params.get("origin") == ORIGIN:
            await revalidate_source(self._coord, params)
            return
        if skip_reason is not None:
            await self._revalidate_on_geak_harness(decline_reason=str(skip_reason))
            return
        task = _covered_step("explore", params, idempotency_key=f"geak-revalidate-c{int(state.macro_cycle or 0)}")
        try:
            result = await self.sub.execute_covered(task)
        except FuturesCancelledError as exc:
            state.geak_pending = {
                **(state.geak_pending or {}),
                "status": "rebench_cancelled",
                "revalidation_error": str(exc)[:500],
            }
            state.save(self.session_dir)
            raise
        if self._coord.writeback.is_promotable_result("explore", result):
            await self._coord.writeback.promote_to_shared_state("explore", result, task=task)
        else:
            await self._coord.writeback.handle_unpromotable_result(task, result)

    async def _revalidate_on_geak_harness(self, *, decline_reason: str) -> None:
        """Replay the candidate through GEAK's own harness (2a); record the decline when it does not validate."""
        state = self.shared_state
        log.warning("geak: 2b declined (%s); validating through the GEAK harness instead", decline_reason)
        fb = await self._coord.writeback.validate_geak_via_geak_harness(reason=decline_reason)
        # Both are verdicts 2a has already recorded.
        if fb.get("validated") or fb.get("status") == REBENCH_NO_PROMOTE:
            return
        error = str(fb.get("reason") or decline_reason)[:500]
        state.geak_pending = {
            **(state.geak_pending or {}),
            "status": _geak_decline_status(decline_reason),
            "revalidation_error": error,
        }
        if (
            not _geak_rebench.geak_harness_replays_workload(state)
            and fb.get("status") == _geak_rebench.INCOMPARABLE_REVALIDATION
        ):
            state.geak_result = {
                **state.geak_result,
                "revalidation_status": REBENCH_FALLBACK_FAILED,
                "revalidation_error_class": _geak_rebench.INCOMPARABLE_REVALIDATION,
                "revalidation_error": error,
                # This refusal is reusable only while its overlay remains unloadable.
                "revalidation_blocked_overlay": str(state.geak_result.get("final_overlay") or ""),
            }
        state.save(self.session_dir)

    def _geak_win_already_recorded(self) -> bool:
        """Whether a GEAK e2e win is already in this session's state."""
        return any(
            isinstance(item, dict) and item.get("action") == "geak_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    def _record_geak_claim_and_product(
        self, result: dict[str, Any], accepted_flags: str, parsed_envs: dict[str, str]
    ) -> None:
        recorder = self.timeline()
        if recorder is None:
            return
        recorder.record_geak_claim(self.shared_state.geak_pending, specs=geak_acceptance_specs(result))
        recorder.record_geak_product(
            GeakProduct(
                accepted_flags=accepted_flags,
                accepted_envs=dict(parsed_envs),
                accepted_config=result.get("accepted_config"),
                final_overlay=result.get("final_overlay") or "",
                final_launch_script=result.get("final_launch_script") or "",
                bench_script=result.get("bench_script") or "",
                final_patch=result.get("final_patch") or "",
            )
        )

    def _record_geak_candidate(self, result: dict[str, Any]) -> bool:
        """Validate the return and record any measured claim without a headline gain."""
        if not isinstance(result, dict):
            return False
        result.setdefault("kernel_event_id", kernel_event_id(int(self.shared_state.macro_cycle or 0)))
        try:
            accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False
        if result.get("status") != "ok" and not _geak_has_accepted_kernel(result):
            return True
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        if new_tput <= 0:
            return True
        basis_facts = result.get("baseline_basis") or {}
        self.shared_state.geak_pending = self._geak_pending_record(
            result, basis_facts, accepted_flags=accepted_flags, parsed_envs=parsed_envs, new_tput=new_tput
        )
        self._record_geak_claim_and_product(result, accepted_flags, parsed_envs)
        # Surface a large cross-harness measurement divergence as a warning only.
        divergence = to_float(basis_facts.get("measurement_divergence_pct"), 0.0)
        if abs(divergence) > _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT:
            log.warning(
                "geak candidate: large cross-harness measurement divergence %.2f%% (|.|>%.1f%%) - candidate held "
                "out of headline until a main-flow rebench validates it",
                divergence,
                _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
            )
        return True

    def _geak_pending_record(
        self,
        result: dict[str, Any],
        basis_facts: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
        new_tput: float,
    ) -> dict[str, Any]:
        from .geak_native_revalidation import matching_review

        basis = str(result.get("metric_basis") or GEAK_METRIC_OUTPUT[1])
        base = (
            float(self.shared_state.baseline_tput or 0.0)
            if basis == GEAK_METRIC_OUTPUT[1]
            else axis_of(self.shared_state.baseline_perf, basis)
        )
        # synthetic isl/osl at face value reads as roughly +175% with nothing
        # optimized. The downstream rebench would eventually reject it, but the
        # number would already be recorded, so refuse to compute it here.
        # An ABSENT verdict means an older GEAK that only ever ran the synthetic
        # path, which stays comparable and keeps today's value exactly.
        comparability = basis_facts.get("workload_comparability") or {}
        comparable = comparability.get("comparable") is not False
        am = result.get("alignment_metrics") or {}
        return {
            "status": "awaiting_rebench",
            # Audit-only self-reported numbers (not the headline until rebench).
            "self_reported_tput": new_tput,
            "self_reported_speedup": result.get("throughput_speedup"),
            "self_reported_gain_pct": gain_math.incremental_gain_pct(new_tput, base) if comparable else None,
            # Why the gain above is a number or None, so a reader never has to
            # guess whether an absent gain means "no gain" or "not comparable".
            "workload_comparability": comparability or None,
            "self_reported_basis": result.get("final_throughput_basis"),
            # Reproducible config the rebench launches from.
            "accepted_flags": accepted_flags,
            "accepted_envs": dict(parsed_envs),
            **_accepted_config_controls(result.get("accepted_config")),
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
            **matching_review(self.shared_state, result),
        }

    @staticmethod
    def _geak_stack_entry_extra(result: dict[str, Any], *, overlay_loaded: bool | None) -> dict[str, Any]:
        """Build the ``geak_e2e`` stack entry, carrying only kernels proven to have run."""
        proven = overlay_loaded is True
        return {
            "backend": "geak",
            "accepted_kernels": (result.get("accepted_kernels") or []) if proven else [],
            "accepted_heads": (result.get("accepted_heads") or []) if proven else [],
            "report_path": result.get("report_path"),
            "source": "geak_e2e",
            "overlay_loaded": overlay_loaded,
        }

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

    def _geak_promotion_measurement(
        self,
        result: dict[str, Any],
        graded_measurement: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
        kernel_proven: bool,
    ) -> dict[str, Any]:
        measurement = {
            "name": "geak_e2e",
            "candidate_extra_server_args": accepted_flags,
            "extra_server_args": accepted_flags,
            "extra_envs": dict(parsed_envs),
            **_accepted_config_controls(result.get("accepted_config")),
            "final_overlay": result.get("final_overlay") or "",
            "source_phase": "KERNEL_AGENT",
            "lever_kind": LEVER_KERNEL if kernel_proven else LEVER_CONFIG,
            "ttft_mean_ms": result.get("ttft_ms"),
            "tpot_mean_ms": result.get("tpot_ms"),
            **graded_axes_of(graded_measurement),
            "workspace": result.get("eval_dir"),
        }
        if graded_measurement is result:
            return measurement
        for key in (
            "extra_server_args",
            "effective_extra_server_args",
            "extra_envs",
            "candidate_extra_server_args",
            "candidate_extra_envs",
            "recipe_delta",
            "remove_args",
            "unset_envs",
            "args_mode",
            "fingerprint",
        ):
            if key in graded_measurement:
                measurement[key] = graded_measurement[key]
        for key in (
            "accuracy",
            "gpu_power_avg_w",
            "gpu_power_by_gpu_w",
            "launch_evidence",
            "launch_evidence_path",
            "server_log_path",
            "workspace",
            "single_workspace",
        ):
            value = graded_measurement.get(key)
            if value not in (None, "", {}):
                measurement[key] = value
        return measurement

    def _compose_geak_launch_recipe(
        self,
        measurement: dict[str, Any],
        accepted_config: Mapping[str, Any],
        *,
        accepted_flags: str,
        parsed_envs: dict[str, str],
    ) -> None:
        from hyperloom.common.coerce import to_str_list
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
        from hyperloom.inference_optimizer.framework_registry import server_args_env_name
        from hyperloom.inference_optimizer.grid_server_args import compose_server_args, remove_server_args

        cb_now = self.shared_state.current_best
        accepted_controls = _accepted_config_controls(accepted_config)
        prior_controls = _accepted_config_controls(cb_now)
        launch_controls = {**prior_controls, **accepted_controls}
        for key in ("remove_args", "unset_envs"):
            values = list(dict.fromkeys(to_str_list(prior_controls.get(key)) + to_str_list(accepted_controls.get(key))))
            if values:
                launch_controls[key] = values
        complete = accepted_controls.get("args_mode") == "replace"
        if complete and "remove_args" in accepted_config:
            launch_controls["remove_args"] = to_str_list(accepted_config["remove_args"])
        if not launch_controls:
            return
        inherited_args = ""
        if not complete and prior_controls.get("args_mode") != "replace":
            recipe_envs = _bench_envs(self._recipe_benchmark(str(self.shared_state.baseline_config_path or "")))
            inherited_args = str(recipe_envs.get(server_args_env_name(self.shared_state.framework)) or "")
        measurement["extra_server_args"] = compose_server_args(
            inherited_args=inherited_args,
            base_extra_args="" if complete else cb_now.get("extra_server_args"),
            variant_extra_args=accepted_flags,
            remove_args=launch_controls.get("remove_args"),
            args_mode="replace" if complete else "append",
        )
        if not complete and launch_controls.get("remove_args"):
            # A legacy delta can re-enable a removed flag. The retained
            # snapshot is complete, so stale removals would prune it again.
            accepted_identity = canonical_fingerprint(accepted_flags, {})
            launch_controls["remove_args"] = [
                spec
                for spec in launch_controls["remove_args"]
                if canonical_fingerprint(remove_server_args(accepted_flags, [spec]), {}) == accepted_identity
            ]
        launch_envs = dict(cb_now.get("extra_envs") or {})
        for key in accepted_controls.get("unset_envs", []):
            launch_envs.pop(key, None)
        launch_envs.update(parsed_envs)
        measurement["extra_envs"] = launch_envs
        measurement.update(launch_controls)
        measurement["args_mode"] = "replace"

    def _record_geak_lift_attribution(
        self,
        result: dict[str, Any],
        *,
        measured: float,
        provenance: str,
        overlay_loaded: bool | None,
        pre_geak_tput: float,
    ) -> None:
        """Attribute the lifted GEAK result to the kernels that can carry it."""
        base = float(self.shared_state.baseline_tput or 0.0)
        # ``pre_geak_tput`` is read before the lift, which overwrites ``current_best`` with ``measured``: the journey
        # rejection and the route-level residual both measure from where the session stood before GEAK ran.
        self.record_geak_adopted_kernels(
            result,
            measured_tput=measured,
            baseline_tput=base,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
        )
        if overlay_loaded is not True:
            # The journey is replayed before the main-flow rebench and can therefore contain GEAK-internal KEEPs for
            # kernels that were not present in the configuration that produced ``measured``.
            self.reject_geak_kernel_journey(
                result,
                measured_tput=measured,
                current_best_tput=pre_geak_tput if pre_geak_tput > 0 else base,
                provenance=provenance,
                rejection_reason="overlay_not_proven_loaded",
            )

    def promote_geak_from_candidate(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        provenance: str = "geak_e2e_promote",
        overlay_loaded: bool | None = None,
        measurement_provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Promote a measured GEAK candidate and return whether the lift succeeded."""
        if not isinstance(result, dict):
            return False
        measured = to_float(measured_tput, 0.0)
        if measured <= 0:
            return False
        # Read before the lift overwrites ``current_best``; see :meth:`_record_geak_lift_attribution`.
        pre_geak_tput = to_float(self.shared_state.current_best.get("tput"), 0.0)
        try:
            accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False

        # The lever is stamped here, not guessed from the task kind: GEAK promotes on a proven kernel overlay OR on a
        # config/env-only win, and only this site holds the overlay proof.
        entry_extra = self._geak_stack_entry_extra(result, overlay_loaded=overlay_loaded)
        graded_measurement = measurement_provenance if isinstance(measurement_provenance, Mapping) else result
        promotion_measurement = self._geak_promotion_measurement(
            result,
            graded_measurement,
            accepted_flags=accepted_flags,
            parsed_envs=parsed_envs,
            kernel_proven=bool(entry_extra.get("accepted_kernels") or entry_extra.get("accepted_heads")),
        )
        if graded_measurement is result or "extra_server_args" not in graded_measurement:
            self._compose_geak_launch_recipe(
                promotion_measurement,
                result.get("accepted_config") or {},
                accepted_flags=accepted_flags,
                parsed_envs=parsed_envs,
            )
        if not self._coord.writeback.lift_to_current_best(
            "geak_e2e", measured, promotion_measurement, entry_extra=entry_extra
        ):
            self.reject_geak_promotion(
                result,
                measured_tput=measured,
                current_best_tput=pre_geak_tput,
                reason="graded_comparison_rejected",
            )
            return False
        self._record_geak_lift_attribution(
            result,
            measured=measured,
            provenance=provenance,
            overlay_loaded=overlay_loaded,
            pre_geak_tput=pre_geak_tput,
        )
        self._coord.writeback.validate(measured, graded_measurement, source="geak_e2e_promote")
        self.shared_state.geak_pending = {}
        return True

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

    def record_geak_adopted_kernels(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        baseline_tput: float,
        provenance: str,
        overlay_loaded: bool | None,
    ) -> None:
        """Write one adoption row per accepted GEAK kernel."""
        from .geak_adoption import AdoptionMeasurement, adoption_entry

        if not isinstance(result, dict):
            return
        # Both acceptance lanes, ``env`` selections excluded and alias twins collapsed.
        specs = _geak_accepted_kernel_specs(result)
        if not specs:
            return
        rows = [{"kernel_id": geak_spec_name(k), "spec": k} for k in specs]

        rebench_gain = gain_math.gain_pct(measured_tput, baseline_tput)
        # One kernel, overlay proven loaded, one measured number: the gain is attributable.
        attributable = bool(overlay_loaded) and len(rows) == 1
        am = result.get("alignment_metrics") or {}
        basis = str(am.get("final_basis") or result.get("final_throughput_basis") or "")
        alignment_status = str((result.get("baseline_alignment") or {}).get("status") or "")
        ts = datetime.now(timezone.utc).isoformat()
        ledger = self.shared_state.kernel_integrate_attempts
        if not isinstance(ledger, dict):
            return
        cycle = int(self.shared_state.macro_cycle or 0)
        measurement = AdoptionMeasurement(
            measured_tput=measured_tput,
            rebench_gain=rebench_gain,
            attributable=attributable,
            provenance=provenance,
            ts=ts,
            cycle=cycle,
            overlay_loaded=overlay_loaded,
            basis=basis,
            alignment_status=alignment_status,
        )
        for row in rows:
            kid = row["kernel_id"]
            spec = row["spec"]
            entry = adoption_entry(
                ledger.get(kid) or {},
                kid=kid,
                spec=spec,
                measurement=measurement,
            )
            ledger[kid] = entry
            # GEAK adopts by writing this ledger directly rather than through
            # the integrate queue, so without this the kernel timeline holds no
            # gate row for a GEAK adoption at all -- and the basis the gain was
            # measured on lives only here.
            _record_geak_integration(
                entry,
                kernel_id=kid,
                macro_cycle=cycle,
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
        """
        if not isinstance(result, dict) or not result:
            return
        recorder = self.timeline()
        if recorder is None:
            return
        recorder.record_geak_measurement(result)

    def _record_geak_delegation_timeline(
        self,
        result: dict[str, Any],
        *,
        handoff: dict[str, Any],
        facts: GeakRunnerFacts,
        recovered_from_disk: bool = False,
    ) -> None:
        """Record the delegated GEAK runner's terminal state."""
        eval_dir = str(result.get("eval_dir") or handoff.get("eval_dir") or "")
        absent = geak_absent_backends(eval_dir)
        if absent:
            result["absent_backends"] = absent
            log.warning(
                "GEAK could not use these backends (provisioning gaps, not measured no-wins): %s",
                "; ".join(f"{name}: {probe}" for name, probe in absent.items()),
            )
        recorder = self.timeline()
        if recorder is None:
            return
        versions = result.get("versions")
        versions = versions if isinstance(versions, dict) else {}
        recorder.record_geak_delegation(
            GeakDelegation(
                runner_status=first_str(result.get("status"), "unknown"),
                started_at=facts.started_at or first_str(result.get("started_at")),
                ended_at=str(result.get("ended_at") or datetime.now(timezone.utc).isoformat()),
                duration_sec=facts.duration_sec if facts.duration_sec is not None else result.get("duration_sec"),
                error_class=first_str(result.get("error_class")),
                error=first_str(result.get("error")),
                returncode=result.get("returncode"),
                runner_timeout_sec=(
                    facts.runner_timeout_sec if facts.runner_timeout_sec is not None else result.get("runner_timeout_s")
                ),
                kill_timeout_sec=(
                    facts.kill_timeout_sec if facts.kill_timeout_sec is not None else result.get("kill_timeout_s")
                ),
                exp_root=str(result.get("exp_root") or handoff.get("exp_root") or ""),
                eval_dir=eval_dir,
                report_path=first_str(result.get("report_path")),
                versions=versions,
                recovered_from_disk=recovered_from_disk,
                stages_reached=result.get("stages_reached"),
            )
        )

    def _record_geak_kernel_journey(self, result: dict[str, Any]) -> None:
        """Record what GEAK-e2e's ``kernel_journey.json`` says about its run.

        Two facts come out of the file. The attempts themselves go onto the
        kernel timeline event, which is what the breakdown reads. The tool
        builds are the other: GEAK reports the build of every tool its run went
        through, and this journey is the only place they reach the optimizer at
        all, so they are recorded even though nothing else in the file is.

        A missing or unreadable journey file records nothing and returns.
        """
        journey = self._load_geak_journey(result)
        if not journey:
            return

        record_geak_attempts(
            event=str(result.get("kernel_event_id") or kernel_event_id(int(self.shared_state.macro_cycle or 0))),
            journey=journey,
        )

        for tool, meta in (journey.get("versions") or {}).items():
            if not isinstance(meta, dict):
                continue
            tool_versions.record_tool_version(
                self.session_dir,
                tool=str(tool),
                root=str(meta.get("root_dir") or "") or None,
                version=str(meta.get("version") or meta.get("commit") or "") or None,
            )

    def reject_geak_kernel_journey(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        current_best_tput: float,
        provenance: str,
        rejection_reason: str = "rebench_did_not_beat_current_best",
    ) -> None:
        """Revoke the persisted provisional GEAK KEEPs after a final rebench."""
        reject_geak_attempts(
            event=str(result.get("kernel_event_id") or kernel_event_id(int(self.shared_state.macro_cycle or 0))),
            measured_tput=measured_tput,
            current_best_tput=current_best_tput,
            provenance=provenance,
            rejection_reason=rejection_reason,
        )

    def _runtime_uses_aiter_fused_moe(self) -> bool:
        """Return whether the served model dispatches MoE through aiter."""
        from ..kernel.serving_log_evidence import resolve_forge_server_log

        log_path = resolve_forge_server_log(self.shared_state, self.session_dir)
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
        """Report whether the validated aiter CSV was reachable by the server, or ``None`` when undetermined."""
        try:
            return self._gemm_tuned_config_coverage_impl(tuner_name, envs)
        except (OSError, ValueError):  # parses server logs and tuner CSVs whose format varies by aiter version
            log.warning("tuned-config coverage failed for %s; treating it as undetermined", tuner_name, exc_info=True)
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
        loaded = read_latest_integrate_server_log(self.session_dir, _integrate_run_name("fmoe_ck"))
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
        reference: dict[str, Any],
        candidate: dict[str, Any],
        *,
        config_path: str,
    ):
        """Re-measure the frozen GEMM entry recipe and tuned stack interleaved."""
        from ..kernel.request_handlers import integrate_handler
        from ..measurement.paired import assess_paired, interleaved_plan

        n_pairs = env_int("HYPERLOOM_GEMM_PAIRED_PAIRS", default=0)
        if n_pairs <= 0 or float(reference.get("tput") or 0.0) <= 0:
            return None

        pairs: list[tuple[float, float]] = []
        pending: float | None = None
        for idx, side in enumerate(interleaved_plan(n_pairs)):
            recipe = reference if side == "A" else candidate
            envs = dict(recipe.get("extra_envs") or {})
            overlay = str(recipe.get("final_overlay") or "")
            if overlay and overlay not in str(envs.get("PYTHONPATH") or "").split(":"):
                envs["PYTHONPATH"] = ":".join(filter(None, (overlay, envs.get("PYTHONPATH"))))
            res = await integrate_handler(
                {
                    "task_id": f"gemm_paired_{side}{idx}",
                    "kernel_id": f"gemm_paired_{side}{idx}",
                    "source": "forge_gemm_paired",
                    "base_tput": reference["tput"],
                    "config_path": config_path,
                    "paired_reference": reference,
                    "extra_server_args": str(recipe.get("extra_server_args") or ""),
                    "extra_envs": envs,
                    "remove_args": recipe.get("remove_args", []),
                    "unset_envs": recipe.get("unset_envs", []),
                    "args_mode": recipe.get("args_mode", "append"),
                    # Measure, do not decide: the verdict comes from the pairs, so a per-round KEEP/REVERT here
                    # would be noise promoted to a decision.
                    "keep_threshold_pct": 100.0,
                    "mode": "env_only",
                },
                session_dir=self.session_dir,
            )
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
        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths or tuner_name == "fmoe_ck":
            return None
        from ..kernel.gemm_shape_coverage import aiter_log_tuned_config_enabled
        from ..measurement.apply_verification import verify_applied

        logs = _integrate_server_logs(self.session_dir, tuner_name)
        if not logs:
            # Say so.
            log.warning(
                "forge gemm E2E: no server.log under %s (retries included); apply verification cannot run for %s",
                self.session_dir / "runs" / "integrate" / _integrate_run_name(tuner_name),
                tuner_name,
            )
            return None

        # The deployed file is named after the candidate, so the runtime's own table name has to travel with it or the
        # arrival check compares merged_tuned_dense_bf16.csv against bf16_tuned_gemm.csv and concludes the artifact
        # never landed.
        table_names = [name for key in envs if (name := _AITER_ENV_TO_TABLE.get(key))]
        # aiter prints a hit line only under this flag; every serving run now sets it by default, but an operator
        # value in the candidate env wins, and then a zero-hit result means nothing.
        hit_logging = aiter_log_tuned_config_enabled(envs)

        return verify_applied(
            logs[-1],
            csv_paths,
            hit_logging=hit_logging,
            runtime_table_names=table_names,
        ).to_dict()

    def _aiter_config_dirs(self) -> list[Path]:
        import importlib.util

        config_dirs: list[Path] = []
        explicit_root = os.environ.get("AITER_ROOT_DIR", "").strip()
        if explicit_root:
            config_dirs.extend((Path(explicit_root) / "aiter" / "configs", Path(explicit_root) / "configs"))
        try:
            spec = importlib.util.find_spec("aiter")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            config_dirs.append(Path(spec.origin).resolve().parent / "configs")
        config_dirs.append(_CONTAINER_AITER_CONFIG_DIR)
        config_dirs.extend(sorted(self.session_dir.glob("runs/specialist/*/worktree/aiter/configs")))
        unique: dict[str, Path] = {}
        for config_dir in config_dirs:
            if config_dir.is_dir():
                unique.setdefault(str(config_dir), config_dir)
        return list(unique.values())

    @staticmethod
    def _aiter_source_tables(
        config_dirs: list[Path], *, runtime_filename: str, tuned_stem: str, untuned_stem: str
    ) -> tuple[list[Path], Path | None]:
        # The complete runtime tables to merge the candidate into, plus the dir that defines their keys.
        cache_dir = os.environ.get("INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR", "/tmp/aiter_configs")
        runtime_path = Path(cache_dir) / runtime_filename
        if runtime_path.is_file():
            keys_from = next((d for d in config_dirs if (d / f"{untuned_stem}.csv").is_file()), None)
            return [runtime_path], keys_from
        for config_dir in config_dirs:
            base_path = config_dir / runtime_filename
            model_paths = sorted(
                path
                for path in (config_dir / "model_configs").glob(f"*{tuned_stem}*.csv")
                if path.is_file() and "untuned" not in path.name
            )
            paths = ([base_path] if base_path.is_file() else []) + model_paths
            if paths:
                return paths, config_dir
        return [], None

    def _merge_gemm_candidate_with_runtime(self, env_var: str, candidate_csv_path: str) -> str | None:
        """Merge a GEMM candidate CSV with the runtime config."""
        import csv
        import re

        candidate_path = Path(candidate_csv_path)
        runtime_filename = _AITER_ENV_TO_TABLE.get(env_var)
        if not runtime_filename or not candidate_path.is_file():
            return None
        tuned_stem = Path(runtime_filename).stem
        untuned_stem = (
            re.sub(r"(?:_)?tuned$", "_untuned", tuned_stem)
            if re.search(r"(?:_)?tuned$", tuned_stem)
            else tuned_stem.replace("tuned", "untuned")
        )
        config_dirs = self._aiter_config_dirs()
        try:
            candidate_columns, candidate_rows = _read_dispatch_csv(candidate_path)
            source_paths, keys_from = self._aiter_source_tables(
                config_dirs, runtime_filename=runtime_filename, tuned_stem=tuned_stem, untuned_stem=untuned_stem
            )
            if not source_paths:
                log.warning(
                    "gemm E2E: no complete aiter config found for %s; candidate-only validation would be unsafe",
                    env_var,
                )
                return None
            sources = [_read_dispatch_csv(path) for path in source_paths]
            all_columns = _merged_dispatch_columns([columns for columns, _ in sources] + [candidate_columns])
            runtime_rows: list[dict[str, str]] = []
            for _, rows in sources:
                runtime_rows.extend(_normalized_dispatch_rows(rows, all_columns))
            candidate_rows = _normalized_dispatch_rows(candidate_rows, all_columns)
            key_cols = _dispatch_key_columns(all_columns, keys_from, untuned_stem)
            if not key_cols:
                log.warning("gemm E2E: cannot derive dispatch keys for %s", env_var)
                return None
            kept = _deduplicate_dispatch_rows(runtime_rows, key_cols, all_columns, "runtime config", env_var)
            tuned = _deduplicate_dispatch_rows(candidate_rows, key_cols, all_columns, "candidate", env_var)
            if kept is None or tuned is None:
                return None
            # Drop rows from runtime that the candidate improves, then concat.
            candidate_keys = {_dispatch_key(row, key_cols) for row in tuned}
            merged_rows = [row for row in kept if _dispatch_key(row, key_cols) not in candidate_keys] + tuned
            merged_path = candidate_path.parent / f"merged_{candidate_path.name}"
            with merged_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(merged_rows)
            log.info(
                "gemm E2E: merged %d candidate rows into %d rows from %d aiter config file(s) -> %d total (%s)",
                len(tuned),
                len(kept),
                len(source_paths),
                len(merged_rows),
                merged_path,
            )
            return str(merged_path)
        except (OSError, ValueError, csv.Error) as exc:
            log.warning("gemm E2E: merge failed (%s); rejecting candidate", exc)
            return None

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
            "last_profile_launch_evidence_path",
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
        if result.get("error_class") == INSUFFICIENT_BUDGET:
            # Never started, so no attempt to record; a recorded ``skipped`` would latch GEMM off for later entries.
            return
        self._sync_profile_state_after_gemm_roofline(result)
        kernel_decisions.record_gemm_tuning(self.shared_state, result)
        try:
            await self._validate_gemm_tuning_e2e(result)
        except Exception as exc:  # isolation boundary: one GEMM campaign's e2e validation, recorded as a REVERT fault
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
        self._record_gemm_tuning_timeline(result)
        self.shared_state.save(self.session_dir)

    def _journal_gemm_tuning_keep(
        self,
        entry: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """Mirror an adopted GEMM-tuning stack entry as an optimization_journal KEEP row."""
        try:
            journal = self._coord.recipe_journal.ensure_journal()
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
                    phase=self._coord.recipe_journal.journal_entry_phase(),
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
        except (OSError, TypeError, ValueError):
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
            atomic_write_json(path, entry, options=WriteOptions())
        except (OSError, TypeError, ValueError):
            log.warning("gemm result.json writeback failed for %s", path, exc_info=True)

    def _replace_latest_gemm_tuning_attempt(self, result: dict[str, Any]) -> None:
        """Sync the latest GEMM history row, and publish the verdict to disk."""
        if not isinstance(result, dict):
            return
        entry = dict(result)
        attempts = list(self.shared_state.gemm_tuning_attempts or [])
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
            envs = _candidate_envs(row.get("env"))
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
                    "micro_speedup": to_float(row.get("best_micro_speedup")) or 1.0,
                }
            )
        return candidates

    def _close_gemm_e2e_without_candidates(self, result: dict[str, Any]) -> None:
        log.info("gemm tuning: no candidates to E2E validate")
        result["decision"] = "REVERT"
        result["requires_e2e_validation"] = False
        result["e2e_validated"] = False
        # Only when the tuners left no verdict of their own: ``micro_decision`` is a routing key downstream, not a
        # label, so an existing one stands.
        if not _first_text(result, "micro_decision").strip():
            result["micro_decision"] = "no_e2e_candidates"
        # ``recommended_env``/``extra_envs`` stay as the tuners left them: they are the raw record of what was
        # produced, and the eligibility checks downstream already read them as "must be empty".
        self._replace_latest_gemm_tuning_attempt(result)

    def _gemm_candidate_skip_reason(self, tuner_name: str, *, triton_moe_inert: bool) -> str:
        from hyperloom.inference_optimizer.model_config_utils import model_supports_aiter_ck_fused_moe

        if tuner_name == "vllm_moe_triton" and triton_moe_inert:
            log.warning(
                "gemm E2E: skipping %s -- the server dispatches MoE through "
                "aiter, which never reads VLLM_TUNED_CONFIG_FOLDER, so the tuned "
                "Triton config cannot take effect",
                tuner_name,
            )
            return "aiter_moe_runtime_triton_config_inert"
        # fmoe_ck is only meaningful with --moe-runner-backend aiter, and aiter's CK fused-MoE rejects a
        # non-128-aligned intermediate_size_per_partition.
        if tuner_name == "fmoe_ck" and not model_supports_aiter_ck_fused_moe(
            str(self.shared_state.model_path or ""), int(self.shared_state.tp or 0)
        ):
            log.info(
                "gemm E2E: skipping %s -- aiter CK fused-MoE cannot serve "
                "this model at tp=%s (intermediate size is not 128-aligned)",
                tuner_name,
                self.shared_state.tp or 0,
            )
            return "aiter_ck_moe_shape_unsupported"
        return ""

    def _merged_candidate_envs(self, cand: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any] | None]:
        # Each aiter table merged into the runtime one, plus the refusal row if one of them is unusable.
        env = dict(cand["envs"])
        for env_var, env_value in list(env.items()):
            if not env_var.startswith("AITER_CONFIG"):
                continue
            if Path(env_value).is_file():
                merged_path = self._merge_gemm_candidate_with_runtime(env_var, env_value)
                if merged_path:
                    env[env_var] = merged_path
                    continue
                reason = "complete_aiter_config_unavailable"
            else:
                reason = "candidate_artifact_missing"
            log.error("gemm E2E: refusing aiter candidate for %s (%s: %s)", cand["tuner"], env_var, reason)
            return env, {**cand, "reason": reason, "failed_env_var": env_var}
        return env, None

    def _gemm_candidate_server_args(self, tuner_name: str) -> str:
        if tuner_name == "fmoe_ck" and str(self.shared_state.framework or "").lower() == "sglang":
            return "--moe-runner-backend aiter"
        return ""

    def _gemm_integrate_payload(
        self, tuner_name: str, test_envs: dict[str, str], *, base_tput: float
    ) -> dict[str, Any]:
        from hyperloom.common.model_paths import resolve_session_model_path

        return {
            "task_id": f"gemm_tune_e2e_{tuner_name}",
            "kernel_id": f"gemm_tune_{tuner_name}",
            "source": "forge_gemm_tuning",
            "base_tput": base_tput,
            "model_path": resolve_session_model_path(state_model_path=str(self.shared_state.model_path or "")),
            "extra_server_args": self._gemm_candidate_server_args(tuner_name),
            "extra_envs": test_envs,
            "keep_threshold_pct": 1.0,
            "mode": "env_only",
        }

    async def _integrate_gemm_candidate(
        self,
        cand: dict[str, Any],
        test_envs: dict[str, str],
        jit_backup_dir: Path,
        *,
        base_tput: float,
    ) -> dict[str, Any] | None:
        # The integrate verdict, a row carrying ``fault``, or ``None`` when the run itself was stopped.
        from ..actions.executors._aiter_jit import drop_serving_so_for_envs, prepare_serving_so_for_csvs
        from ..state.kernel_decision_settings import _MAX_INTEGRATE_FAULT_ATTEMPTS

        tuner_name = cand["tuner"]
        payload = self._gemm_integrate_payload(tuner_name, test_envs, base_tput=base_tput)
        # The native handler reloads both the recipe and grading anchor from disk.
        self.shared_state.save(self.session_dir)

        async def _fault(attempt: int, error_class: str, **fields: Any) -> dict[str, Any]:
            await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=jit_backup_dir)
            return {
                **cand,
                "reason": f"integrate_fault:{error_class}",
                "fault": True,
                "error_class": error_class,
                "fault_attempts": attempt,
                **fields,
            }

        for attempt in range(1, _MAX_INTEGRATE_FAULT_ATTEMPTS + 1):
            last = attempt >= _MAX_INTEGRATE_FAULT_ATTEMPTS
            await asyncio.to_thread(prepare_serving_so_for_csvs, test_envs, backup_dir=jit_backup_dir)
            try:
                integrate_result = await integrate_within_phase(
                    payload, state=self.shared_state, session_dir=self.session_dir
                )
            except Exception as exc:  # isolation boundary: one integrate attempt for one GEMM candidate
                log.warning(
                    "gemm E2E: integrate raised for %s (fault attempt %d/%d): %s",
                    tuner_name,
                    attempt,
                    _MAX_INTEGRATE_FAULT_ATTEMPTS,
                    exc,
                    exc_info=True,
                )
                if last:
                    return await _fault(attempt, "handler_exception", error=repr(exc))
                continue
            stopped = stopped_by_the_run_class(integrate_result.get("error_class"))
            if stopped is not None:
                log.info("gemm E2E: %s left unmeasured -- %s", tuner_name, stopped.interrupted)
                return None
            if not self.shared_state._is_integrate_fault(integrate_result):
                return integrate_result
            error_class = str(integrate_result.get("error_class") or "integrate_fault").strip()
            log.warning(
                "gemm E2E: tuner=%s integrate fault %s (attempt %d/%d)",
                tuner_name,
                error_class,
                attempt,
                _MAX_INTEGRATE_FAULT_ATTEMPTS,
            )
            if last:
                return await _fault(
                    attempt,
                    error_class,
                    integrate_status=integrate_result.get("status"),
                    error=integrate_result.get("error"),
                )
        return None

    async def _gemm_apply_blockers(
        self, cand: dict[str, Any], env: dict[str, str], *, gain_pct: float
    ) -> tuple[list[str], dict[str, Any]]:
        # Two independent ways the artifact can fail to take effect, neither of which the throughput delta can
        # see: the keys are unreachable (coverage), and the table never reached the server (apply verdict).
        tuner_name = cand["tuner"]
        blockers: list[str] = []
        # Off the event loop for the same reason: this reads the integrate run's server.log in full and parses
        # every tuned CSV named in the candidate env.
        coverage = await asyncio.to_thread(self._gemm_tuned_config_coverage, tuner_name, env)
        if coverage is not None:
            cand = {**cand, "tuned_config_coverage": coverage}
            if coverage.get("artifact_applied"):
                log.info(
                    "gemm E2E: tuner=%s tuned-config coverage %.2f%% (%d/%d shapes)",
                    tuner_name,
                    coverage.get("coverage_pct") or 0.0,
                    coverage.get("covered") or 0,
                    coverage.get("requested") or 0,
                )
            elif coverage.get("conclusive", True):
                blockers.append(str(coverage.get("not_applied_reason") or "no_shape_key_matched"))
                log.error(
                    "gemm E2E: tuner=%s produced an artifact the runtime never "
                    "applied -- 0 of %d requested shape(s) resolve to a tuned row; "
                    "the %.2f%% e2e delta measures nothing about the tuning",
                    tuner_name,
                    coverage.get("requested") or 0,
                    gain_pct,
                )
            else:
                log.info(
                    "gemm E2E: tuner=%s tuned-config coverage inconclusive -- %s",
                    tuner_name,
                    coverage.get("not_applied_reason"),
                )
        applied = self._gemm_apply_verdict(tuner_name, env)
        if applied is None:
            return blockers, cand
        cand = {**cand, "apply_verdict": applied}
        if applied.get("blocks_keep"):
            blockers.append(str(applied.get("verdict") or "not_applied"))
            log.error(
                "forge gemm E2E: tuner=%s apply verdict=%s -- %s",
                tuner_name,
                applied.get("verdict"),
                applied.get("detail"),
            )
        elif not applied.get("conclusive"):
            # "Cannot tell" is not "did not apply": hit lines need AITER_LOG_TUNED_CONFIG=1, and treating
            # their absence as a failure would revert every arm that ran without it.
            log.info(
                "forge gemm E2E: tuner=%s apply verdict=%s (not conclusive) -- %s",
                tuner_name,
                applied.get("verdict"),
                applied.get("detail"),
            )
        return blockers, cand

    def _keep_gemm_candidate(
        self, cand: dict[str, Any], env: dict[str, str], verdict: Mapping[str, Any], result: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        from ..measurement.integrate_performance import integrate_measurement_fields

        backend = _first_text(result, "backend").strip().lower() or "forge"
        tuner_name = cand["tuner"]
        tput, gain_pct = _integrate_tput_and_gain(verdict)
        tuned_file = _candidate_tuned_file(env, cand["env_var"])
        lifted = self._coord.writeback.lift_to_current_best(
            "gemm_tuning",
            tput,
            {
                "name": f"{backend}_{tuner_name}",
                "candidate_extra_server_args": self._gemm_candidate_server_args(tuner_name),
                "extra_envs": dict(env),
                "source_phase": "KERNEL_AGENT",
                **integrate_measurement_fields(verdict.get("bench_result") or verdict),
            },
            entry_extra={
                "tuned_file": tuned_file,
                "gain_pct": gain_pct,
                "backend": backend,
                "source": "kernel_entry_auto",
            },
        )
        if not lifted:
            return None
        self._journal_gemm_tuning_keep(self.shared_state.optimization_stack[-1], task_id=f"gemm_tune_e2e_{tuner_name}")
        return {**cand, "envs": dict(env), "tput": tput, "gain_pct": gain_pct, "tuned_file": tuned_file}

    async def _settle_gemm_e2e_verdict(
        self,
        result: dict[str, Any],
        accepted_measurement: Mapping[str, Any],
        paired_reference: dict[str, Any],
        *,
        running_tput: float,
        paired_config_path: str,
    ) -> None:
        kept = result["e2e_results"]["kept"]
        reverted = result["e2e_results"]["reverted"]
        faults = result["e2e_results"]["faults"]
        # The watermark covers the whole run, so it waits for the last KEEP.
        result["graded_objective"] = None
        total_gain: float | None = 0.0
        if kept:
            total_gain = None
            # Paired checks isolate the GEMM increment, not cumulative session gain.
            paired = await self._confirm_gemm_gain_paired(
                paired_reference,
                deepcopy(self.shared_state.current_best),
                config_path=paired_config_path,
            )
            if self._coord.writeback.validate(
                running_tput,
                accepted_measurement,
                source="forge_gemm_tuning_e2e",
                measurement_basis=_paired_measurement_basis(paired),
            ):
                total_gain = self.shared_state.cumulative_gain_validated
                result["graded_objective"] = resolve_graded_comparison(
                    self.shared_state,
                    {**accepted_measurement, "output_throughput": running_tput},
                    against_baseline=True,
                ).objective
            # Name the artifact this run adopted, so the breakdown can tell it was.
            if kept[-1]["tuned_file"]:
                result["tuned_file"] = kept[-1]["tuned_file"]
            log.info(
                "gemm E2E: %d tuners KEEP (total gain=%s), %d REVERT",
                len(kept),
                f"{total_gain:+.2f}%" if total_gain is not None else "unavailable",
                len(reverted),
            )
        elif faults:
            log.info("gemm E2E: %d tuner(s) hit integrate fault(s), no E2E verdict", len(faults))
        else:
            log.info("gemm E2E: all %d tuners REVERT, no E2E gain", len(reverted))
        if total_gain is None or (faults and not kept and not reverted):
            result["e2e_gain_pct"] = None
        else:
            result["e2e_gain_pct"] = round(float(total_gain), 4)
        result["e2e_validated"] = True
        result["requires_e2e_validation"] = False
        result["decision"] = "KEEP" if kept else "REVERT"
        if kept:
            result["status"] = "complete"
        elif faults and not reverted:
            result["status"] = "failed"
            result["micro_decision"] = "integrate_fault"
        else:
            result["status"] = "complete"
            result["micro_decision"] = "candidate_no_e2e_gain"
        self._replace_latest_gemm_tuning_attempt(result)

    async def _validate_gemm_tuning_e2e(self, result: dict[str, Any]) -> None:
        """Sequentially E2E-validate each tuning candidate's env independently."""
        from ..actions.executors._aiter_jit import drop_serving_so_for_envs

        candidates = self._gemm_canonical_candidates(result)
        if not candidates:
            self._close_gemm_e2e_without_candidates(result)
            return
        running_tput = float(self.shared_state.current_best.get("tput") or self.shared_state.baseline_tput or 0.0)
        paired_reference = deepcopy(self.shared_state.current_best)
        paired_reference.setdefault("tput", running_tput)
        paired_config_path = str(self.shared_state.baseline_config_path or "")
        jit_backup_dir = self.session_dir / "runs" / "aiter_jit_backup"
        stacked_envs: dict[str, str] = {}
        kept: list[dict[str, Any]] = []
        reverted: list[dict[str, Any]] = []
        faults: list[dict[str, Any]] = []
        accepted_measurement: dict[str, Any] = {}
        # ``_runtime_uses_aiter_fused_moe`` resolves the serving log -- which now byte-scans the whole runs/ tree for
        # aiter evidence -- and then reads it whole, ~17MB on the fleet.
        triton_moe_inert = any(c.get("tuner") == "vllm_moe_triton" for c in candidates) and await asyncio.to_thread(
            self._runtime_uses_aiter_fused_moe
        )
        for cand in candidates:
            tuner_name = cand["tuner"]
            skip_reason = self._gemm_candidate_skip_reason(tuner_name, triton_moe_inert=triton_moe_inert)
            if skip_reason:
                reverted.append({**cand, "reason": skip_reason})
                continue
            # Merge candidate CSV with the runtime config so that shapes NOT in the candidate keep their existing
            # tuned entries.
            env, refusal = self._merged_candidate_envs(cand)
            if refusal is not None:
                reverted.append(refusal)
                continue
            # Merge with previously KEEP'd envs.
            test_envs = {**stacked_envs, **env}
            log.info("gemm E2E: validating tuner=%s env=%s (base_tput=%.1f)", tuner_name, cand["env_var"], running_tput)
            verdict = await self._integrate_gemm_candidate(cand, test_envs, jit_backup_dir, base_tput=running_tput)
            if verdict is None:
                break
            if verdict.get("fault"):
                faults.append(verdict)
                continue
            measurement = verdict.get("bench_result") or verdict
            new_tput, gain_pct = _integrate_tput_and_gain(verdict)
            decision = str(verdict.get("decision") or "").upper()
            log.info(
                "gemm E2E: tuner=%s decision=%s new_tput=%.1f gain=%.2f%%", tuner_name, decision, new_tput, gain_pct
            )
            blockers, cand = await self._gemm_apply_blockers(cand, env, gain_pct=gain_pct)
            if decision == "KEEP" and new_tput > 0 and not blockers:
                keep_row = self._keep_gemm_candidate(cand, env, verdict, result)
                if keep_row is not None:
                    stacked_envs.update(env)
                    running_tput = new_tput
                    accepted_measurement = measurement
                    kept.append(keep_row)
                    continue
            reason = f"decision={decision}, gain={gain_pct:.2f}%"
            if blockers:
                # Distinguish no gain from a tuned artifact the runtime never reached.
                reason = f"tuned_config_never_applied[{'+'.join(blockers)}] ({reason})"
            reverted.append({**cand, "reason": reason})
            await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=jit_backup_dir)

        _record_gemm_e2e_envelope(result, stacked_envs, kept=kept, reverted=reverted, faults=faults)
        await self._settle_gemm_e2e_verdict(
            result,
            accepted_measurement,
            paired_reference,
            running_tput=running_tput,
            paired_config_path=paired_config_path,
        )

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
            self.session_dir,
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
        self.shared_state.auto_roofline_pending_task_id = ""
        log.info(
            "watermark-roofline: released in-flight gate held by finished task=%s",
            pending,
        )

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
