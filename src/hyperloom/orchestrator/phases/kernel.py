# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL_AGENT phase handler for the fusion, GEMM, and GEAK lanes."""

from __future__ import annotations
import asyncio
import hashlib
import importlib.util
import json
import logging as _logging
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import CancelledError as FuturesCancelledError
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from . import _aiter_config_merge
from . import _geak_handoff
from . import _kernel_timeline
from . import geak_rebench as _geak_rebench
from .geak_adoption import AdoptionMeasurement, adoption_entry
from . import machine_state as _phase_state
from ._kernel_timeline import RunnerRun
from .kernel_round_budget import controller_round_stop, integrate_within_phase, measurement_round_sec, rebench_round_sec
from hyperloom.common.env import env_bool, env_float, env_int
from hyperloom.common.io import WriteOptions, atomic_write_json
from hyperloom.common.perf_metric import axis_of, graded_axes_of
from hyperloom.orchestrator.lever import (
    LEVER_CONFIG,
    LEVER_KERNEL,
)
from hyperloom.inference_optimizer.breakdown.recorder import tool_versions
from ..actions.executors._workload_envs import GEAK_METRIC_OUTPUT, geak_metric_axis
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    ROUTE_FORGE,
    ROUTE_GEAK,
    kernel_event_id,
    record_geak_attempts,
    reject_geak_attempts,
)
from ..actions.stop_attribution import stopped_by_the_run_class
from hyperloom.inference_optimizer.session.optimization_journal import (
    KIND_GEMM_TUNING,
    OUTCOME_KEEP,
    JournalEntry,
)
from ..state.analysis_failures import analysis_skip_reason, remembered_trace_failure
from ..state.shared_state import resolve_graded_comparison
from ..state.task_registry import TERMINAL_STATES, Task, TaskNotFound
from ..bus.message_bus import Message
from ..kernel.geak_config import (
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    geak_spec_name,
    geak_is_cand_tag,
    geak_spec_is_env,
    _accepted_config_as_variant,
    _accepted_config_controls,
)
from ..kernel.lane_budget import INSUFFICIENT_BUDGET
from ..actions.executors._gpu_pin import (
    _resolve_handoff_physical_gpus,
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


def _aiter_config_dirs(session_dir: Path) -> list[Path]:
    """Every existing directory the runtime's aiter config tables could be read from, nearest first."""
    dirs: list[Path] = []
    explicit_root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if explicit_root:
        root = Path(explicit_root)
        dirs.extend((root / "aiter" / "configs", root / "configs"))
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        dirs.append(Path(spec.origin).resolve().parent / "configs")
    dirs.append(_CONTAINER_AITER_CONFIG_DIR)
    dirs.extend(sorted(session_dir.glob("runs/specialist/*/worktree/aiter/configs")))
    seen: set[str] = set()
    unique: list[Path] = []
    for config_dir in dirs:
        if str(config_dir) not in seen and config_dir.is_dir():
            seen.add(str(config_dir))
            unique.append(config_dir)
    return unique


def _controller_produced_nothing(status: str, reason: str, output_dir: Path) -> dict[str, Any]:
    """The result a Controller attempt that published nothing reports."""
    return {
        "status": status,
        "reason": reason,
        "patch_count": 0,
        "task_count": 0,
        "output_dir": str(output_dir),
    }


def _reclaim_borrowed_repositories(result: dict[str, Any], baselines: Mapping[str, object]) -> None:
    """Return every borrowed campaign repository to the HEAD integration will read."""
    try:
        from ..kernel.campaign_baseline import reclaim_campaign_repositories

        reclaimed = reclaim_campaign_repositories(baselines)
    except Exception:
        log.exception("KERNEL entry: reclaiming the campaign repositories failed")
        return
    if reclaimed:
        result["reclaimed_repositories"] = reclaimed


class _GemmE2ERun(NamedTuple):
    """The conditions every candidate of one GEMM e2e validation run is measured under."""

    backend: str
    budget_minutes: int
    reference: dict[str, Any]
    config_path: str
    inert_triton: bool
    jit_backup_dir: Path


class _MergedCandidateEnvs(NamedTuple):
    """A candidate's env once its tuned tables were merged into the runtime's, or why they were not."""

    envs: dict[str, str]
    failure_reason: str
    failed_env_var: str


class _IntegrateOutcome(NamedTuple):
    """What one candidate's integrate attempts settled on."""

    verdict: dict[str, Any] | None
    run_stopped: bool


class _GemmMeasured(NamedTuple):
    """What one candidate's integrate measured."""

    decision: str
    new_tput: float
    gain_pct: float
    measurement: dict[str, Any]


class _ApplyCheck(NamedTuple):
    """A candidate with what the apply checks learned about it, and what blocks a KEEP."""

    candidate: dict[str, Any]
    blockers: list[str]


class _GemmE2ELedger:
    """What a GEMM e2e validation run has settled so far, candidate by candidate."""

    def __init__(self, entry_tput: float) -> None:
        self.running_tput = entry_tput
        self.stacked_envs: dict[str, str] = {}
        self.kept: list[dict[str, Any]] = []
        self.reverted: list[dict[str, Any]] = []
        self.faults: list[dict[str, Any]] = []
        self.accepted_measurement: dict[str, Any] = {}
        #: Set by the last KEEP; the attempt row claims this exact string.
        self.adopted_tuned_file = ""

    def record_keep(self, cand: Mapping[str, Any], *, env: dict[str, str], measured: _GemmMeasured) -> None:
        """Adopt this candidate's env as the stack the next candidate is measured on top of."""
        self.stacked_envs.update(env)
        self.running_tput = measured.new_tput
        self.accepted_measurement = measured.measurement
        self.adopted_tuned_file = _candidate_tuned_file(env, cand.get("env_var", ""))
        self.kept.append({**cand, "envs": dict(env), "tput": measured.new_tput, "gain_pct": measured.gain_pct})


def _gemm_measured(verdict: Mapping[str, Any]) -> _GemmMeasured:
    """Read what an integrate verdict measured for the candidate it judged."""
    measurement = verdict.get("bench_result") or verdict
    return _GemmMeasured(
        decision=str(verdict.get("decision") or "").upper(),
        new_tput=float(measurement.get("output_throughput", verdict.get("new_tput")) or 0.0),
        gain_pct=float(verdict.get("gain_pct") or 0.0),
        measurement=measurement,
    )


def _handler_exception_fault(cand: Mapping[str, Any], exc: Exception, *, attempts: int) -> dict[str, Any]:
    """The fault row an integrate that raised leaves behind."""
    return {
        **cand,
        "reason": "integrate_fault:handler_exception",
        "fault": True,
        "error_class": "handler_exception",
        "error": repr(exc),
        "fault_attempts": attempts,
    }


def _integrate_fault(
    cand: Mapping[str, Any], integrate_result: Mapping[str, Any], error_class: str, *, attempts: int
) -> dict[str, Any]:
    """The fault row an integrate that judged nothing leaves behind."""
    return {
        **cand,
        "reason": f"integrate_fault:{error_class}",
        "fault": True,
        "error_class": error_class,
        "integrate_status": integrate_result.get("status"),
        "error": integrate_result.get("error"),
        "fault_attempts": attempts,
    }


def _coverage_blocker(tuner: str, coverage: Mapping[str, Any], gain_pct: float) -> str:
    """Why the tuned rows were unreachable at dispatch; '' when they were reached or cannot be told."""
    if coverage.get("artifact_applied"):
        log.info(
            "gemm E2E: tuner=%s tuned-config coverage %.2f%% (%d/%d shapes)",
            tuner,
            coverage.get("coverage_pct") or 0.0,
            coverage.get("covered") or 0,
            coverage.get("requested") or 0,
        )
        return ""
    if not coverage.get("conclusive", True):
        log.info(
            "gemm E2E: tuner=%s tuned-config coverage inconclusive — %s", tuner, coverage.get("not_applied_reason")
        )
        return ""
    log.error(
        "gemm E2E: tuner=%s produced an artifact the runtime never "
        "applied — 0 of %d requested shape(s) resolve to a tuned row; "
        "the %.2f%% e2e delta measures nothing about the tuning",
        tuner,
        coverage.get("requested") or 0,
        gain_pct,
    )
    return str(coverage.get("not_applied_reason") or "no_shape_key_matched")


def _apply_verdict_blocker(tuner: str, applied: Mapping[str, Any]) -> str:
    """Why the apply check says the tuned table never reached the server; '' when it did."""
    if applied.get("blocks_keep"):
        log.error(
            "forge gemm E2E: tuner=%s apply verdict=%s — %s",
            tuner,
            applied.get("verdict"),
            applied.get("detail"),
        )
        return str(applied.get("verdict") or "not_applied")
    if not applied.get("conclusive"):
        # "Cannot tell" is not "did not apply": hit lines need AITER_LOG_TUNED_CONFIG=1, and treating
        # their absence as a failure would revert every arm that ran without it.
        log.info(
            "forge gemm E2E: tuner=%s apply verdict=%s (not conclusive) — %s",
            tuner,
            applied.get("verdict"),
            applied.get("detail"),
        )
    return ""


def _gemm_revert_reason(measured: _GemmMeasured, blockers: list[str]) -> str:
    """Why a measured candidate was not adopted."""
    reason = f"decision={measured.decision}, gain={measured.gain_pct:.2f}%"
    if not blockers:
        return reason
    # Distinguish no gain from a tuned artifact the runtime never reached.
    return f"tuned_config_never_applied[{'+'.join(blockers)}] ({reason})"


def _gemm_tuning_verdict(ledger: _GemmE2ELedger) -> dict[str, str]:
    """The status and decision a settled GEMM campaign reports."""
    if ledger.kept:
        return {"status": "complete", "decision": "KEEP"}
    if ledger.faults and not ledger.reverted:
        return {"status": "failed", "decision": "REVERT", "micro_decision": "integrate_fault"}
    return {"status": "complete", "decision": "REVERT", "micro_decision": "candidate_no_e2e_gain"}


class _AcceptedLaunch(NamedTuple):
    """The launch a GEAK run accepted, as a promotion records it."""

    flags: str
    envs: dict[str, str]


#: What the run that produced the number recorded about the launch it measured.
_MEASURED_LAUNCH_KEYS: tuple[str, ...] = (
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
)

#: Evidence of that run, carried only where the measurement actually holds some.
_MEASURED_EVIDENCE_KEYS: tuple[str, ...] = (
    "accuracy",
    "gpu_power_avg_w",
    "gpu_power_by_gpu_w",
    "launch_evidence",
    "launch_evidence_path",
    "server_log_path",
    "workspace",
    "single_workspace",
)


def _as_measured_number(value: object) -> float:
    """A measured number read back off a JSON record; zero for anything that is not one."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _geak_promotion_measurement(
    result: Mapping[str, Any], accepted: _AcceptedLaunch, *, kernel_proven: bool, graded: Mapping[str, Any]
) -> dict[str, Any]:
    """The measurement row a promoted GEAK candidate is lifted into the current best with."""
    return {
        "name": "geak_e2e",
        "candidate_extra_server_args": accepted.flags,
        "extra_server_args": accepted.flags,
        "extra_envs": dict(accepted.envs),
        **_accepted_config_controls(result.get("accepted_config")),
        "final_overlay": result.get("final_overlay") or "",
        "source_phase": "KERNEL_AGENT",
        "lever_kind": LEVER_KERNEL if kernel_proven else LEVER_CONFIG,
        "ttft_mean_ms": result.get("ttft_ms"),
        "tpot_mean_ms": result.get("tpot_ms"),
        **graded_axes_of(graded),
        "workspace": result.get("eval_dir"),
    }


def _carry_measured_launch(measurement: dict[str, Any], measured: Mapping[str, Any]) -> None:
    """Carry onto the promotion what the run that produced its number recorded about itself."""
    for key in _MEASURED_LAUNCH_KEYS:
        if key in measured:
            measurement[key] = measured[key]
    for key in _MEASURED_EVIDENCE_KEYS:
        value = measured.get(key)
        if value not in (None, "", {}):
            measurement[key] = value


def _merged_launch_controls(
    prior: Mapping[str, Any], accepted: Mapping[str, Any], accepted_config: Mapping[str, Any]
) -> dict[str, Any]:
    """The launch controls a promotion carries: the accepted ones over what current best already held."""
    from hyperloom.common.coerce import to_str_list

    controls = {**prior, **accepted}
    for key in ("remove_args", "unset_envs"):
        values = list(dict.fromkeys(to_str_list(prior.get(key)) + to_str_list(accepted.get(key))))
        if values:
            controls[key] = values
    if accepted.get("args_mode") == "replace" and "remove_args" in accepted_config:
        controls["remove_args"] = to_str_list(accepted_config["remove_args"])
    return controls


def _removals_the_accepted_flags_keep(remove_args: list[str], accepted_flags: str) -> list[str]:
    """Drop the removals that would prune a flag the accepted launch itself carries."""
    from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
    from hyperloom.inference_optimizer.grid_server_args import remove_server_args

    accepted_identity = canonical_fingerprint(accepted_flags, {})
    return [
        spec
        for spec in remove_args
        if canonical_fingerprint(remove_server_args(accepted_flags, [spec]), {}) == accepted_identity
    ]


def _promoted_launch_envs(
    current_best: Mapping[str, Any], *, unset: Iterable[str], accepted_envs: Mapping[str, str]
) -> dict[str, str]:
    """The env the promoted launch runs with: the current best's, minus what the acceptance unset."""
    envs = dict(current_best.get("extra_envs") or {})
    for key in unset:
        envs.pop(key, None)
    envs.update(accepted_envs)
    return envs


class _RooflineAnchor(NamedTuple):
    """The roofline measurement a phase is anchored on, and the snapshot it came from."""

    tput: float
    snapshot_count: int
    snapshot_id: int


class _ReprofileDecision(NamedTuple):
    """Why a kernel entry does or does not reprofile before it picks its targets."""

    trigger: str
    skipped_reason: str


def _fusion_awaits_e2e(fusion: Mapping[str, Any]) -> bool:
    """Whether a KEPT fusion still has to prove itself end to end."""
    return bool(fusion.get("kept")) and bool(fusion.get("requires_e2e_validation"))


def _fail_unreadable_fusion_envelope(fusion: dict[str, Any]) -> None:
    """Report a KEEP whose nomination envelope the contract cannot read as infrastructure."""
    if not _fusion_awaits_e2e(fusion):
        return
    from ..kernel.nomination_result import parse_outcome

    skew = parse_outcome(fusion).schema_error
    if not skew:
        return
    fusion.update(
        {
            "status": "failed",
            "error_class": "nomination_envelope_skew",
            "error": skew,
            "infrastructure_abort": True,
        }
    )


def _as_float(value: object, default: float) -> float:
    """Read a measurement that round-tripped through JSON, defaulting on junk."""
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _applicable_envs(raw: object) -> dict[str, str]:
    """The env assignments of a candidate that a run could actually export: both halves present."""
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(key).strip() and str(value).strip()}


def _tuner_left_a_deployable_artifact(row: Mapping[str, Any]) -> bool:
    """Whether a tuner run wrote rows an e2e validation could apply."""
    if row.get("status") not in ("ok", "partial_output"):
        return False
    return (
        bool(row.get("candidate"))
        or int(row.get("improved_shapes") or 0) > 0
        or int(row.get("unverified_shapes") or 0) > 0
    )


def _rebuilt_gemm_candidates(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the candidates a tuner run describes but never named in ``candidates[]``."""
    candidates: list[dict[str, Any]] = []
    for row in result.get("tuners_run") or []:
        if not isinstance(row, dict) or not _tuner_left_a_deployable_artifact(row):
            continue
        env_var = str(row.get("env_var") or "").strip()
        env_value = str(row.get("env_value") or "").strip()
        envs = _applicable_envs(row.get("env_vars"))
        if env_var and env_value:
            envs.setdefault(env_var, env_value)
        if envs:
            candidates.append(
                {
                    "tuner": row.get("tuner") or "unknown",
                    "env_var": env_var,
                    "env_value": env_value,
                    "envs": envs,
                    "micro_speedup": _as_float(row.get("best_micro_speedup"), 1.0),
                }
            )
    return candidates


def _assumed_blockscale_candidates(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The a8w8-blockscale candidate a pre-``candidates[]`` KEEP envelope implies, if any."""
    if str(result.get("backend") or "").strip().lower() == "forge":
        return []
    tuned_file = str(result.get("tuned_file") or "").strip()
    micro_speedup = _as_float(result.get("best_speedup"), 0.0)
    if not tuned_file or micro_speedup <= 1.0 or not _envelope_recorded_a_keep(result):
        return []
    env_var = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
    return [
        {
            "tuner": "a8w8_blockscale_tuned_gemm",
            "env_var": env_var,
            "env_value": tuned_file,
            "envs": {env_var: tuned_file},
            "micro_speedup": micro_speedup,
        }
    ]


def _charge_fusion_retry_caps(state: Any, fusion: Mapping[str, Any]) -> None:
    """Charge this fusion run against the session's abort and withheld-target retry caps."""
    if fusion.get("infrastructure_abort"):
        state.fusion_infra_aborts = _as_int(state.fusion_infra_aborts) + 1
    if not fusion.get("kept") and _withheld_targets(fusion) > 0:
        state.fusion_withheld_retries = _as_int(state.fusion_withheld_retries) + 1


def _save_last_fusion(state: Any, session_dir: Path, record: dict[str, Any]) -> None:
    """Keep this run's record as the session's last fusion."""
    try:
        state.last_fusion = record
        state.save(session_dir)
    except Exception:  # noqa: BLE001 - state shape tolerant (best-effort idempotency record)
        pass


async def _announce_fusion_done(bus: Any, result: Any, *, status: str) -> None:
    """Surface the settled fusion run on the bus for whoever is waiting on it."""
    try:
        await bus.append_and_seq(
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
    except Exception:
        log.exception("failed to post run_fusion_done bus message")


def _write_geak_adoptions(
    state: Any, ledger: dict[str, Any], specs: list[dict[str, Any]], measurement: AdoptionMeasurement
) -> None:
    """Record each accepted kernel's adoption in the per-kernel ledger and on the kernel timeline."""
    cycle = int(state.macro_cycle or 0)
    for spec in specs:
        kid = _adopted_kernel_id(spec)
        entry = adoption_entry(ledger.get(kid) or {}, kid=kid, spec=spec, measurement=measurement)
        ledger[kid] = entry
        # GEAK adopts by writing this ledger directly rather than through
        # the integrate queue, so without this the kernel timeline holds no
        # gate row for a GEAK adoption at all -- and the basis the gain was
        # measured on lives only here.
        _record_geak_integration(entry, kernel_id=kid, macro_cycle=cycle)
    state.kernel_integrate_attempts = ledger


class _AdoptionOrigin(NamedTuple):
    """Where an adoption's number came from, and whether it can be pinned on one kernel."""

    provenance: str
    overlay_loaded: bool | None
    attributable: bool


def _adopted_kernel_id(spec: Mapping[str, Any]) -> str:
    """The id an accepted GEAK kernel is recorded under in the per-kernel ledger."""
    return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()


def _relative_gain_pct(measured_tput: float, baseline_tput: float) -> float | None:
    """What a measurement gained on its baseline, as a percentage; ``None`` without both numbers."""
    if baseline_tput <= 0 or measured_tput <= 0:
        return None
    return (measured_tput - baseline_tput) / baseline_tput * 100.0


def _geak_adoption_measurement(
    result: Mapping[str, Any],
    *,
    measured_tput: float,
    rebench_gain: float | None,
    cycle: int,
    origin: _AdoptionOrigin,
) -> AdoptionMeasurement:
    """The one rebench outcome every kernel in this adopted stack is recorded against."""
    alignment = result.get("alignment_metrics") or {}
    return AdoptionMeasurement(
        measured_tput=measured_tput,
        rebench_gain=rebench_gain,
        attributable=origin.attributable,
        provenance=origin.provenance,
        ts=datetime.now(timezone.utc).isoformat(),
        cycle=cycle,
        overlay_loaded=origin.overlay_loaded,
        basis=str(alignment.get("final_basis") or result.get("final_throughput_basis") or ""),
        alignment_status=str((result.get("baseline_alignment") or {}).get("status") or ""),
    )


def _acceptance_lane_rows(result: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every acceptance GEAK recorded, paired with the queue that proposed it."""
    for lane, key in (("kernelQueue", "accepted_kernels"), ("headQueue", "accepted_heads")):
        for row in result.get(key) or []:
            if isinstance(row, dict):
                yield lane, row


def _stated_e2e_delta(raw: Mapping[str, Any]) -> float | None:
    """The e2e delta an acceptance row states, or ``None`` when it states none this reader can use."""
    stated = raw.get("e2e_delta_pct")
    if stated is None or stated == "":
        return None
    try:
        return float(stated)
    except (TypeError, ValueError):
        return None


def _collapse_onto_twin(out: list[dict[str, Any]], position: int, row: dict[str, Any], name: str) -> None:
    """Fold a second acceptance of the same measurement onto the one already kept."""
    kept = out[position]
    kept["alias_collapsed"] = True
    # The collapsed twin's name is the one a reader may hold, so it is
    # carried on the survivor rather than dropped with the row.
    aliases = {*(kept.get("aliases") or []), geak_spec_name(kept), name}
    if geak_is_cand_tag(geak_spec_name(kept)) and not geak_is_cand_tag(name):
        row["alias_collapsed"] = True
        out[position] = row
        kept = row
    kept["aliases"] = sorted({alias for alias in aliases if alias and alias != geak_spec_name(kept)})


class _RunnerExit(NamedTuple):
    """How the delegated GEAK runner process ended, and where it was to leave its result."""

    returncode: int
    stderr_tail: str
    result_path: Path


class _RunnerClock(NamedTuple):
    """When the delegated runner was started, and the timeouts it was started under."""

    started_at: str
    started_monotonic: float
    runner_timeout: int
    kill_timeout: int

    def observed(self) -> RunnerRun:
        """What the phase has observed of the runner process it started, as of now."""
        return RunnerRun(
            started_at=self.started_at,
            duration_sec=time.monotonic() - self.started_monotonic,
            runner_timeout_sec=self.runner_timeout,
            kill_timeout_sec=self.kill_timeout,
        )


class _GeakDelegation(NamedTuple):
    """The handoff one GEAK e2e delegation runs from, and where it was written."""

    payload: dict[str, Any]
    path: Path


def _geak_runner_command(handoff_path: Path, out_dir: Path, runner_timeout: int) -> list[str]:
    """The command line that runs one GEAK e2e delegation."""
    return [
        sys.executable,
        "-m",
        _GEAK_RUNNER_MODULE,
        str(handoff_path),
        str(out_dir),
        "--timeout-s",
        str(runner_timeout),
    ]


def _read_geak_result(path: Path) -> dict[str, Any]:
    """The result the GEAK runner wrote, or ``{}`` when there is none to read."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the whole process group ``proc`` leads; a process that already exited is not an error."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def _communicate_or_kill_tree(
    proc: subprocess.Popen, cmd: list[str], *, kill_timeout: int, term_grace: int
) -> subprocess.CompletedProcess:
    """Wait for ``proc``; past ``kill_timeout`` SIGTERM its tree, give it ``term_grace``, then SIGKILL."""
    try:
        out, err = proc.communicate(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        _signal_process_group(proc, signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=term_grace)
        except subprocess.TimeoutExpired:
            _signal_process_group(proc, signal.SIGKILL)
            out, err = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, kill_timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


class _GeakClaim(NamedTuple):
    """What a GEAK run reported about the configuration it accepted, before any rebench."""

    accepted_flags: str
    accepted_envs: dict[str, str]
    tput: float
    gain_pct: float | None
    comparability: dict[str, Any]


def _geak_pending_record(result: Mapping[str, Any], claim: _GeakClaim) -> dict[str, Any]:
    """The pending record the main-flow rebench reads this candidate back from."""
    alignment = result.get("alignment_metrics") or {}
    return {
        "status": "awaiting_rebench",
        # Audit-only self-reported numbers (not the headline until rebench).
        "self_reported_tput": claim.tput,
        "self_reported_speedup": result.get("throughput_speedup"),
        "self_reported_gain_pct": claim.gain_pct,
        # Why the gain above is a number or None, so a reader never has to
        # guess whether an absent gain means "no gain" or "not comparable".
        "workload_comparability": claim.comparability or None,
        "self_reported_basis": result.get("final_throughput_basis"),
        # Reproducible config the rebench launches from.
        "accepted_flags": claim.accepted_flags,
        "accepted_envs": claim.accepted_envs,
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
            "hot_geak_speedup": alignment.get("hot_geak_speedup"),
            "cold_geak_speedup": alignment.get("cold_geak_speedup"),
            "hot_speedup": alignment.get("hot_speedup"),
            "cold_speedup": alignment.get("cold_speedup"),
            "final_basis": alignment.get("final_basis") or result.get("final_throughput_basis"),
            "geak_throughput_speedup": result.get("throughput_speedup"),
        },
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _warn_on_measurement_divergence(result: Mapping[str, Any]) -> None:
    """Warn when GEAK's harness and this session's disagree on the same measurement."""
    stated = (result.get("baseline_basis") or {}).get("measurement_divergence_pct")
    try:
        divergence = abs(float(stated)) if stated is not None else None
    except (TypeError, ValueError):
        return
    if divergence is None or divergence <= _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT:
        return
    log.warning(
        "geak candidate: large cross-harness measurement divergence "
        "%.2f%% (|.|>%.1f%%) - candidate held out of headline until a "
        "main-flow rebench validates it",
        float(stated),
        _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
    )


def _envelope_recorded_a_keep(result: Mapping[str, Any]) -> bool:
    """Whether a pre-``candidates[]`` envelope settled on a KEEP of a completed run."""
    decision = str(result.get("decision") or "").strip().upper()
    status = str(result.get("status") or "").strip().lower()
    return decision == OUTCOME_KEEP and status in _COMPLETED_TUNING_STATUSES


#: How a GEMM tuning envelope spells "the run finished" before it names its candidates.
_COMPLETED_TUNING_STATUSES: frozenset[str] = frozenset({"ok", "complete", "completed", "succeeded", "success"})

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


@dataclass(frozen=True)
class KernelExitFacts:
    """What the phase machine's KERNEL exit reads from the task queue."""

    agent_in_flight: bool


class KernelPhase(CoordinatorCollaborator):
    """KERNEL_AGENT phase handler: drives kernel-level optimization tasks."""

    # Relative-change floor for the pre-GEAK reprofile: effectively "any change", absorbing float noise.
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
        status = str(self.shared_state.last_profile_status or "").strip().lower()
        if status and status != "succeeded":
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

    async def _maybe_reprofile_for_kernel(self) -> None:
        """Reprofile inline when projected tput diverges from the last measured trace, so the phase targets the live bottleneck."""
        signature = self._current_profile_config_signature()
        config_changed = self._profile_config_changed(signature)
        workload_changed = self._profile_workload_changed()
        projected = self.current_tput_from_validated_gain()
        decision = self._reprofile_decision(projected, config_changed=config_changed, workload_changed=workload_changed)
        recorder = self.timeline()

        def _note_reprofile(**fields: Any) -> None:
            if recorder is None:
                return
            recorder.record_reprofile(**fields)

        if not decision.trigger:
            _note_reprofile(ran=False, skipped_reason=decision.skipped_reason)
            return
        if config_changed or workload_changed:
            log.info("kernel-entry reprofile: active runtime context changed")
        anchor = self._roofline_anchor()
        reason = self._reprofile_idempotency_reason(signature, projected)
        task_kind = self._coord.phase_prelude.internal_analysis_kind()
        params = self._coord.phase_prelude.internal_analysis_params(
            reason=reason, inline_event=recorder.event_id if recorder is not None else ""
        )
        denied = self._coord.dispatcher.time_budget_denial_for_action(task_kind)
        if params is None or denied is not None:
            _note_reprofile(
                ran=False,
                task_kind=task_kind,
                trigger=decision.trigger,
                skipped_reason=denied.rule if denied else analysis_skip_reason(self.shared_state),
                idempotency_reason=reason,
                snapshot_id_before=anchor.snapshot_id,
            )
            return
        # profile_lane conflicts with the benchmark_lane the kernel_agent task holds, so the reprofile is a step of
        # that task rather than a task of its own.
        reprofile_task = _covered_step(task_kind, params, idempotency_key=f"internal-analysis-{reason}")
        try:
            await self.sub.execute_covered(reprofile_task)
        except Exception:
            log.exception("kernel-entry reprofile failed; the phase proceeds on the existing snapshot")
            _note_reprofile(
                ran=True,
                task_kind=task_kind,
                trigger=decision.trigger,
                skipped_reason="dispatch_failed",
                idempotency_reason=reason,
                snapshot_id_before=anchor.snapshot_id,
            )
            return
        # Advance the anchor only when a new snapshot actually landed.
        after = self._roofline_anchor()
        landed = after != anchor
        if landed and after.tput > 0:
            self._adopt_reprofiled_snapshot(after.tput)
        else:
            log.warning("kernel-entry reprofile produced no new snapshot; the phase targets the existing trace")
        _note_reprofile(
            ran=True,
            task_kind=task_kind,
            trigger=decision.trigger,
            idempotency_reason=reason,
            snapshot_landed=landed,
            snapshot_id_before=anchor.snapshot_id,
            snapshot_id_after=after.snapshot_id,
            task_id=str(getattr(reprofile_task, "task_id", "") or ""),
        )
        if landed:
            self._record_kernel_discovered_from_cache(provenance="reprofile_snapshot")

    def _reprofile_decision(
        self, projected_tput: float, *, config_changed: bool, workload_changed: bool
    ) -> _ReprofileDecision:
        """Whether the measured trace still describes what the phase is about to optimize."""
        if projected_tput <= 0:
            return _ReprofileDecision(trigger="", skipped_reason="no_projected_tput")
        measured = self._last_measured_roofline_tput()
        # With a measured trace, reprofile only on a material gain or a change in what is being measured.
        unmoved = measured > 0 and abs(projected_tput - measured) / measured < self._REPROFILE_CHANGE_TOL
        if unmoved and not config_changed and not workload_changed:
            return _ReprofileDecision(trigger="", skipped_reason="within_tolerance")
        if config_changed:
            return _ReprofileDecision(trigger="config_changed", skipped_reason="")
        return _ReprofileDecision(trigger="workload_changed" if workload_changed else "gain", skipped_reason="")

    def _roofline_anchor(self) -> _RooflineAnchor:
        """The roofline measurement this phase is currently anchored on."""
        state = self.shared_state
        return _RooflineAnchor(
            tput=self._last_measured_roofline_tput(),
            snapshot_count=len(state.roofline_snapshots or []),
            snapshot_id=int(state.roofline_snapshot_id or 0),
        )

    def _adopt_reprofiled_snapshot(self, measured_tput: float) -> None:
        """Move the phase's roofline anchor onto the trace the reprofile just produced."""
        state = self.shared_state
        state.last_roofline_tput = measured_tput
        state.last_profile_status = "succeeded"
        # Record the workload (incl. serving_config) that this trace reflects; _profile_config_changed derives the
        # config signature from it, so last_profile_args stays the plain-args field it is everywhere else.
        state.last_profile_workload = state.profile_workload_context()
        state.save(self.session_dir)

    def _reprofile_idempotency_reason(self, profile_signature: str, projected_tput: float) -> str:
        """A key that stays the same for a reprofile of the same config, target and workload."""
        state = self.shared_state
        identity = json.dumps(
            {
                "config": profile_signature,
                "target_tput": round(float(projected_tput), 6),
                "workload": state.profile_workload_context(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return f"kernel_entry_g{int(state.cumulative_gain_validated_stack_len or 0)}_{fingerprint}"

    def geak_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the GEAK e2e optimizer."""
        from ..kernel.request_handlers import geak_selected

        return geak_selected()

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
        _kernel_timeline.record_controller_rewrites(
            self.timeline(),
            self.session_dir,
            result,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
        )

    def _record_gemm_tuning_timeline(self, result: dict[str, Any]) -> None:
        """Record a settled GEMM campaign in the Forge lane."""
        recorder = self.timeline()
        if recorder is None or not isinstance(result, dict):
            return
        try:
            _kernel_timeline.record_gemm_campaign(
                recorder, self.session_dir, result, macro_cycle=int(self.shared_state.macro_cycle or 0)
            )
        except Exception:
            log.debug("kernel timeline: GEMM tuning record failed", exc_info=True)

    def _record_fusion_timeline(self, result: dict[str, Any]) -> None:
        """Record a settled fusion campaign in the Forge lane."""
        recorder = self.timeline()
        if recorder is None or not isinstance(result, dict):
            return
        _kernel_timeline.record_fusion_campaign(recorder, self.session_dir, result)

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
            exit_reason=exit_reason,
            tput_after=state.current_best.get("tput"),
            cumulative_gain_validated_out=state.cumulative_gain_validated or None,
            stack_depth_out=state.cumulative_gain_validated_stack_len or None,
            stack_added=stack_added,
            stack_removed=stack_removed,
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
                kind="kernel_agent",
                params={"from_phase": str(tr.from_phase or "")},
                idempotency_key=base_key if attempt == 0 else f"{base_key}-r{attempt}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
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
        except Exception as exc:
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
        phase_cap = _phase_state.phase_cap_seconds(self.shared_state)
        if phase_cap is not None:
            candidates.append(max(0.0, float(phase_cap) - _phase_state.phase_cumulative_seconds(self.shared_state)))
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

    def _session_is_agentx(self) -> bool:
        """Whether this session measures the AgentX workload."""
        from hyperloom.common.perf_metric import is_agentx_mode
        from ..actions.executors._workload_envs import agentx_enabled

        benchmark_mode = str(self.shared_state.benchmark_mode or "").strip()
        return is_agentx_mode(benchmark_mode) if benchmark_mode else agentx_enabled()

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run)."""
        agentx = self._session_is_agentx()
        delegation = self._open_geak_delegation(agentx=agentx)
        if delegation is None:
            return
        handoff = delegation.payload
        result_path = _geak_handoff.geak_dir(self.session_dir) / "result.json"
        if await self._promote_geak_result_left_on_disk(handoff, agentx=agentx, result_path=result_path):
            return
        handoff_cards = _resolve_handoff_physical_gpus(
            gpu_ids=handoff["gpu_ids"], gpu_ids_space=handoff["gpu_ids_space"], gpu_pin=handoff.get("gpu_pin")
        )
        still_busy = await self._free_gpus_for_geak(handoff_cards)
        if still_busy:
            self._finish_geak_skip(
                {"status": "error", "error_class": "gpus_busy_at_handoff", "error": still_busy}, handoff=handoff
            )
            return
        # Budget-aware timeouts: shrink to the remaining run deadline and always reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._geak_timeouts()
        if self._refuse_geak_for_budget(handoff, runner_timeout=runner_timeout, budget_known=budget_known):
            return
        cmd = _geak_runner_command(delegation.path, _geak_handoff.geak_dir(self.session_dir), runner_timeout)
        log.info(
            "KERNEL entry: delegating to GEAK e2e (from=%s) runner_timeout=%ds kill_timeout=%ds budget_known=%s cmd=%s",
            from_phase or "<unknown>",
            runner_timeout,
            kill_timeout,
            budget_known,
            " ".join(cmd),
        )
        # Read before the spawn: raising after it would leave a tree nothing can signal.
        term_grace = env_int("GEAK_TERM_GRACE_S", default=180)
        clock = _RunnerClock(
            started_at=datetime.now(timezone.utc).isoformat(),
            started_monotonic=time.monotonic(),
            runner_timeout=runner_timeout,
            kill_timeout=kill_timeout,
        )
        try:
            proc = await asyncio.to_thread(self._run_geak_runner, cmd, kill_timeout=kill_timeout, term_grace=term_grace)
        except subprocess.TimeoutExpired:
            await self._keep_what_the_killed_geak_runner_flushed(
                handoff, agentx=agentx, result_path=result_path, run=clock.observed(), cards=handoff_cards
            )
            return
        except Exception as exc:
            log.exception("GEAK runner crashed")
            self._finish_geak_skip(
                {"status": "error", "error_class": "runner_crashed", "error": repr(exc)},
                handoff=handoff,
                run=clock.observed(),
            )
            return
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            log.warning("GEAK runner rc=%s: %s", proc.returncode, stderr_tail)
        result = _read_geak_result(result_path)
        exited = _RunnerExit(returncode=proc.returncode, stderr_tail=stderr_tail, result_path=result_path)
        if self._refuse_geak_product(result, exited, handoff=handoff, agentx=agentx, run=clock.observed()):
            return
        # Carry the actual exit code so the breakdown can audit a nonzero rc.
        result.setdefault("returncode", proc.returncode)
        await self._settle_geak_run(
            result, handoff=handoff, run=clock.observed(), result_path=result_path, cards=handoff_cards
        )

    def _refuse_geak_product(
        self,
        result: Mapping[str, Any],
        exited: _RunnerExit,
        *,
        handoff: Mapping[str, Any],
        agentx: bool,
        run: RunnerRun,
    ) -> bool:
        """Refuse a runner exit that shipped no new product; report whether it was refused."""
        if not result:
            self._finish_geak_skip(
                {
                    "status": "error",
                    "error_class": "no_result_json",
                    "error": (
                        f"runner rc={exited.returncode} produced no parseable result.json at {exited.result_path}"
                    ),
                    "stderr_tail": exited.stderr_tail,
                    "returncode": exited.returncode,
                },
                handoff=handoff,
                run=run,
            )
            return True
        if not (agentx and self._geak_result_already_settled(result, agentx=agentx)):
            return False
        # The runner left the candidate this session already settled, so it
        # shipped no product: recording it would retire the verdict and
        # re-enqueue the revalidation that produced it. The file stays for
        # the next run to overwrite.
        self._finish_geak_skip(
            {
                "status": "error",
                "error_class": "no_new_geak_product",
                "error": (
                    f"runner rc={exited.returncode} left the already-adjudicated result.json at {exited.result_path}"
                ),
                "stderr_tail": exited.stderr_tail,
            },
            handoff=handoff,
        )
        return True

    def _open_geak_delegation(self, *, agentx: bool) -> _GeakDelegation | None:
        """Write the handoff this delegation runs from; wind the phase down when it cannot be built."""
        try:
            env_spec = self._coord.writeback.build_env_spec()
        except (OSError, TypeError, ValueError) as exc:
            log.exception("geak: cannot serialize the accepted launch configuration")
            recorder = self.timeline()
            if recorder is not None:
                recorder.finish_failed(stage="geak_handoff", error_class="invalid_env_spec", message=str(exc))
            self._wind_down_geak_to_sweep({"status": "error", "error_class": "invalid_env_spec", "error": str(exc)})
            return None
        state = self.shared_state
        reference = _geak_handoff.same_config_reference(state, env_spec, writeback=self._coord.writeback, agentx=agentx)
        payload = _geak_handoff.build_handoff(
            state, reference, session_dir=self.session_dir, agentx=agentx, e2e_metric=self._geak_e2e_metric()
        )
        path = _geak_handoff.write_handoff(payload, session_dir=self.session_dir)
        recorder = self.timeline()
        if recorder is not None:
            recorder.enter_stage("geak_delegation")
            recorder.record_geak_handoff(payload)
        return _GeakDelegation(payload=payload, path=path)

    def _wind_down_geak_to_sweep(self, result: dict[str, Any]) -> None:
        """Carry a (failed/skipped) GEAK outcome into the phase exit toward SWEEP."""
        state = self.shared_state
        prev = state.geak_result if isinstance(state.geak_result, dict) else {}
        if not _geak_rebench.geak_verdict_is_terminal(prev):
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
        self, result: dict[str, Any], *, handoff: Mapping[str, Any], run: RunnerRun = RunnerRun()
    ) -> None:
        """Record what the delegated runner left behind and wind the phase down to SWEEP."""
        self._record_geak_delegation_timeline(result, handoff=handoff, run=run)
        self._wind_down_geak_to_sweep(result)

    def _geak_result_already_settled(self, candidate: Mapping[str, Any], *, agentx: bool) -> bool:
        """Whether ``candidate`` is a result this session already settled."""
        prev = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        return _geak_rebench.geak_candidate_is_adjudicated(prev, candidate, harness_can_replay=not agentx)

    def _refuse_geak_for_budget(self, handoff: Mapping[str, Any], *, runner_timeout: int, budget_known: bool) -> bool:
        """Wind the phase down when too little budget remains to run GEAK; report whether it did."""
        min_run = env_int("GEAK_MIN_RUN_S", default=600)
        if not budget_known or runner_timeout >= min_run:
            return False
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
                    f"(< min {min_run}s); skipped to protect the closing "
                    f"report window"
                ),
                "runner_timeout_s": runner_timeout,
            },
            handoff=handoff,
            run=RunnerRun(runner_timeout_sec=runner_timeout),
        )
        return True

    def _run_geak_runner(self, cmd: list[str], *, kill_timeout: int, term_grace: int) -> subprocess.CompletedProcess:
        """Run the GEAK runner to completion in a process group of its own."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._geak_runner_env(),
            start_new_session=True,
        )
        return _communicate_or_kill_tree(proc, cmd, kill_timeout=kill_timeout, term_grace=term_grace)

    def _geak_e2e_metric(self) -> str:
        """The axis GEAK searches and gates on, for both the runner env and the handoff it reads."""
        # GEAK must search and gate on the axis this session keeps candidates on: a kernel accepted on another
        # axis need not move that one. geak_metric_axis owns the mapping and its rationale.
        metric, _ = geak_metric_axis(
            benchmark_mode=str(self.shared_state.benchmark_mode or ""),
            grading=self.shared_state.grading or None,
        )
        return metric

    def _geak_runner_env(self) -> dict[str, str]:
        """The environment the whole GEAK chain inherits, on this session's metric and spend tag."""
        from hyperloom.common.llm_attribution import inject_env

        runner_env = dict(os.environ)
        runner_env["E2E_METRIC"] = self._geak_e2e_metric()
        # Only injection point needed for the whole GEAK chain: geak_runner and run_e2e both hand their full
        # environment to the child, so the tag reaches the Claude CLI that actually spends.
        inject_env(runner_env, component="geak", operation="optimize_kernel")
        return runner_env

    async def _promote_geak_result_left_on_disk(
        self, handoff: Mapping[str, Any], *, agentx: bool, result_path: Path
    ) -> bool:
        """Adopt a validated result.json a crashed coordinator left behind; report whether it did."""
        recovered = _read_geak_result(result_path)
        if recovered.get("status") != "ok" or self._geak_win_already_recorded():
            return False
        if self._geak_result_already_settled(recovered, agentx=agentx):
            return False
        log.info(
            "GEAK result.json exists but state has no recorded win (crash before handback); promoting recovered result."
        )
        if self._promote_recovered_geak_result(recovered, handoff=handoff, recovered_from="existing_result_json"):
            await self._revalidate_geak_candidate(reason="geak_e2e_win_recovered")
        return True

    async def _keep_what_the_killed_geak_runner_flushed(
        self, handoff: Mapping[str, Any], *, agentx: bool, result_path: Path, run: RunnerRun, cards: list[int] | None
    ) -> None:
        """Keep a win the killed runner still flushed, or record the timeout."""
        log.warning(
            "GEAK runner exceeded kill_timeout=%ds; SIGTERM'd to let it flush, then reclaimed the closing window",
            run.kill_timeout_sec,
        )
        recovered = _read_geak_result(result_path)
        flushed_a_win = recovered.get("status") == "ok" and not (
            agentx and self._geak_result_already_settled(recovered, agentx=agentx)
        )
        if flushed_a_win:
            log.info("GEAK flushed an OK result.json under SIGTERM grace; promoting the recovered win despite the cap.")
            if self._promote_recovered_geak_result(
                recovered, handoff=handoff, recovered_from="sigterm_flushed_result_json", run=run
            ):
                await self._wait_for_gpus_before_rebench(cards)
                await self._revalidate_geak_candidate(reason="geak_e2e_win_sigterm_recovered")
            return
        self._finish_geak_skip(
            {
                "status": "error",
                "error_class": "timeout",
                "error": f"GEAK e2e killed after {run.kill_timeout_sec}s (budget-capped); closing window preserved",
                "runner_timeout_s": run.runner_timeout_sec,
                "kill_timeout_s": run.kill_timeout_sec,
            },
            handoff=handoff,
            run=run,
        )

    def _promote_recovered_geak_result(
        self,
        result: dict[str, Any],
        *,
        handoff: Mapping[str, Any],
        recovered_from: str,
        run: RunnerRun = RunnerRun(),
    ) -> bool:
        """Adopt a result the runner left on disk as this cycle's unvalidated GEAK candidate."""
        state = self.shared_state
        self._record_geak_delegation_timeline(result, handoff=handoff, run=run._replace(recovered_from_disk=True))
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
        evidence = {
            "status": result.get("status"),
            "throughput_speedup": result.get("throughput_speedup"),
            "final_throughput_tok_s": result.get("final_throughput_tok_s"),
            "eval_dir": result.get("eval_dir"),
            "report_path": result.get("report_path"),
            "recovered_from": recovered_from,
        }
        if run.runner_timeout_sec is not None:
            evidence["runner_timeout_s"] = run.runner_timeout_sec
        self._coord.phase_machine.record_phase_entry_evidence(geak=evidence)
        state.save(self.session_dir)
        return True

    async def _settle_geak_run(
        self,
        result: dict[str, Any],
        *,
        handoff: Mapping[str, Any],
        run: RunnerRun,
        result_path: Path,
        cards: list[int] | None,
    ) -> None:
        """Record a completed GEAK run's product and enqueue the rebench that can validate it."""
        state = self.shared_state
        self._record_geak_delegation_timeline(result, handoff=handoff, run=run)
        state.geak_result = result
        self._record_geak_measurement(result)
        if str(result.get("status") or "") == "baseline_reproduction_failed":
            self._refuse_phantom_baseline_gain(result)
            return
        # Rebench-first: record the win as an UNVALIDATED candidate only; the headline is written later from the
        # measured rebench.
        if not self._record_geak_candidate(result):
            state.save(self.session_dir)
            return
        self._record_geak_kernel_journey(result)
        # The same-harness rebench is the only path that writes the headline.
        if str(result.get("status") or "") == "ok":
            await self._wait_for_gpus_before_rebench(cards)
            await self._revalidate_geak_candidate(reason="geak_e2e_win")
        elif _geak_has_accepted_kernel(result):
            # A no_gain headline over an accepted, parity-checked kernel still deserves the measurement — the rebench
            # is what decides, and without it the kernel is lost with no number attached to it.
            await self._wait_for_gpus_before_rebench(cards)
            await self._revalidate_geak_candidate(reason="geak_e2e_accepted_kernel")
        self._coord.phase_machine.record_phase_entry_evidence(
            geak={
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "runner_timeout_s": run.runner_timeout_sec,
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
            )
        )
        state.save(self.session_dir)

    def _refuse_phantom_baseline_gain(self, result: Mapping[str, Any]) -> None:
        """Refuse a run whose baseline ref never reproduced the configuration it was given."""
        log.warning(
            "GEAK baseline_reproduction_failed: ref did not match "
            "orchestrator best (%s); refusing to promote a phantom-baseline gain",
            result.get("error"),
        )
        self._wind_down_geak_to_sweep(
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

    async def _revalidate_geak_candidate(self, *, reason: str) -> None:
        """Measure the GEAK candidate on the orchestrator harness and settle its verdict.

        The 2b rebench runs as a step of the ``kernel_agent`` task, under the
        lanes it holds, and its result goes through the same promotion a
        dispatched explore would. A candidate the grid cannot carry goes to the
        GEAK-harness replay (2a) instead.
        """
        state = self.shared_state
        params = self._coord.writeback.geak_rebench_params(reason=reason)
        from .geak_native_revalidation import ORIGIN, finish_invalid_source, revalidate_source

        if params.get("origin") == ORIGIN:
            await revalidate_source(self._coord, params)
            return
        if params.get("reason") == "geak_native_source_invalid":
            finish_invalid_source(self._coord, str(params["error"]))
            return
        skip_reason = params.get("reason") if params.get("skipped") else None
        if skip_reason == "geak_invalid_config":
            return
        if skip_reason == "geak_no_material":
            state.geak_result = {**state.geak_result, "revalidation_status": "no_material"}
            state.geak_pending = {}
            state.save(self.session_dir)
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
        if fb.get("validated") or fb.get("status") == "no_promote":
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
                "revalidation_status": "fallback_failed",
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

    @staticmethod
    def _parse_geak_accepted_config(
        result: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """Parse ``result.accepted_config`` into (flags, env dict)."""
        return _accepted_config_as_variant(result.get("accepted_config"))

    def _record_geak_candidate(self, result: dict[str, Any]) -> bool:
        """Validate the return and record any measured claim without a headline gain."""
        if not isinstance(result, dict):
            return False
        result.setdefault("kernel_event_id", kernel_event_id(int(self.shared_state.macro_cycle or 0)))
        from .geak_native_revalidation import matching_review

        reviewed_product = matching_review(self.shared_state, result)
        try:
            accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False
        if result.get("status") not in ("ok",) and not _geak_has_accepted_kernel(result):
            return True
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        if new_tput <= 0:
            return True
        claim = self._geak_claim(result, accepted_flags=accepted_flags, accepted_envs=dict(parsed_envs), tput=new_tput)
        self.shared_state.geak_pending = {**_geak_pending_record(result, claim), **reviewed_product}
        self._record_geak_claim_on_timeline(result, claim)
        _warn_on_measurement_divergence(result)
        return True

    def _geak_claim(
        self, result: Mapping[str, Any], *, accepted_flags: str, accepted_envs: dict[str, str], tput: float
    ) -> _GeakClaim:
        """What this GEAK run reported about the configuration it accepted."""
        basis = str(result.get("metric_basis") or GEAK_METRIC_OUTPUT[1])
        base = (
            float(self.shared_state.baseline_tput or 0.0)
            if basis == GEAK_METRIC_OUTPUT[1]
            else axis_of(self.shared_state.baseline_perf, basis)
        )
        comparability = (result.get("baseline_basis") or {}).get("workload_comparability") or {}
        comparable = base > 0 and comparability.get("comparable") is not False
        return _GeakClaim(
            accepted_flags=accepted_flags,
            accepted_envs=accepted_envs,
            tput=tput,
            gain_pct=((tput - base) / base * 100.0) if comparable else None,
            comparability=comparability,
        )

    def _record_geak_claim_on_timeline(self, result: Mapping[str, Any], claim: _GeakClaim) -> None:
        """Put the unvalidated claim, and the product it arrived with, on the kernel timeline."""
        recorder = self.timeline()
        if recorder is None:
            return
        recorder.record_geak_claim(self.shared_state.geak_pending, specs=self._geak_acceptance_specs(result))
        recorder.record_geak_product(
            accepted_flags=claim.accepted_flags,
            accepted_envs=claim.accepted_envs,
            accepted_config=result.get("accepted_config"),
            final_overlay=result.get("final_overlay") or "",
            final_launch_script=result.get("final_launch_script") or "",
            bench_script=result.get("bench_script") or "",
            final_patch=result.get("final_patch") or "",
        )

    @staticmethod
    def _geak_acceptance_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return every GEAK acceptance, tagged with the queue that proposed it."""
        out: list[dict[str, Any]] = []
        index: dict[tuple[str, str], int] = {}
        for lane, raw in _acceptance_lane_rows(result):
            name = geak_spec_name(raw)
            if not name:
                continue
            row = {**raw, "lane": lane, "alias_collapsed": False}
            if geak_spec_is_env(raw):
                row["kind"] = "env"
            delta = _stated_e2e_delta(raw)
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
            _collapse_onto_twin(out, position, row, name)
        return out

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
        rejected_result = dict(result)
        rejected_result["revalidation_status"] = "no_promote"
        rejected_result["revalidation_error"] = reason
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
                attempt_id=attempt_id,
                base_tput=current_best_tput,
                measured_tput=measured_tput if measured_tput > 0 else None,
                decision="no_promote",
                decision_reason=reason,
                status="no_promote",
            )
            recorder.record_geak_rebench_conclusion(final_status="no_promote", final_error=reason)

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
        try:
            measured = float(measured_tput)
        except (TypeError, ValueError):
            return False
        if measured <= 0:
            return False
        cb_tput = self.shared_state.current_best.get("tput")
        try:
            accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)
        except ValueError as exc:
            self.reject_geak_promotion(result, measured_tput=0.0, current_best_tput=0.0, reason=str(exc))
            return False

        # The lever is stamped here, not guessed from the task kind: GEAK promotes on a proven kernel overlay OR on a
        # config/env-only win, and only this site holds the overlay proof.
        entry_extra = self._geak_stack_entry_extra(result, overlay_loaded=overlay_loaded)
        measured_launch = measurement_provenance if isinstance(measurement_provenance, Mapping) else None
        accepted = _AcceptedLaunch(flags=accepted_flags, envs=dict(parsed_envs))
        graded_measurement = measured_launch if measured_launch is not None else result
        promotion_measurement = _geak_promotion_measurement(
            result,
            accepted,
            kernel_proven=bool(entry_extra.get("accepted_kernels") or entry_extra.get("accepted_heads")),
            graded=graded_measurement,
        )
        if measured_launch is not None:
            _carry_measured_launch(promotion_measurement, measured_launch)
        if measured_launch is None or "extra_server_args" not in measured_launch:
            self._compose_promoted_launch(promotion_measurement, result, accepted)
        lifted = self._coord.writeback.lift_to_current_best(
            "geak_e2e",
            measured,
            promotion_measurement,
            entry_extra=entry_extra,
        )
        if not lifted:
            KernelPhase.reject_geak_promotion(
                self,
                result,
                measured_tput=measured,
                current_best_tput=_as_measured_number(cb_tput),
                reason="graded_comparison_rejected",
            )
            return False

        self._attribute_promoted_geak_kernels(
            result,
            measured=measured,
            entered_at=_as_measured_number(cb_tput),
            provenance=provenance,
            overlay_loaded=overlay_loaded,
        )
        self._coord.writeback.validate(measured, graded_measurement, source="geak_e2e_promote")
        self.shared_state.geak_pending = {}
        return True

    def _attribute_promoted_geak_kernels(
        self,
        result: dict[str, Any],
        *,
        measured: float,
        entered_at: float,
        provenance: str,
        overlay_loaded: bool | None,
    ) -> None:
        """Pin a promoted GEAK run's gain on the kernels it can actually be attributed to."""
        base = float(self.shared_state.baseline_tput or 0.0)
        self.record_geak_adopted_kernels(
            result, measured_tput=measured, baseline_tput=base, provenance=provenance, overlay_loaded=overlay_loaded
        )
        if overlay_loaded is True:
            return
        # The journey is replayed before the main-flow rebench and can therefore contain GEAK-internal KEEPs for
        # kernels that were not present in the configuration that produced ``measured``.
        self.reject_geak_kernel_journey(
            result,
            measured_tput=measured,
            # Where the session stood before GEAK ran: the anchor both the journey rejection and the
            # route-level residual measure from.
            current_best_tput=entered_at or base,
            provenance=provenance,
            rejection_reason="overlay_not_proven_loaded",
        )

    def _compose_promoted_launch(
        self, measurement: dict[str, Any], result: Mapping[str, Any], accepted: _AcceptedLaunch
    ) -> None:
        """Derive the launch the promotion adopts when the measurement recorded none of its own."""
        from hyperloom.inference_optimizer.grid_server_args import compose_server_args

        current_best = self.shared_state.current_best
        accepted_controls = _accepted_config_controls(result.get("accepted_config"))
        prior_controls = _accepted_config_controls(current_best)
        controls = _merged_launch_controls(prior_controls, accepted_controls, result.get("accepted_config") or {})
        if not controls:
            return
        complete = accepted_controls.get("args_mode") == "replace"
        inherits_recipe_args = not complete and prior_controls.get("args_mode") != "replace"
        measurement["extra_server_args"] = compose_server_args(
            inherited_args=self._recipe_inherited_server_args() if inherits_recipe_args else "",
            base_extra_args="" if complete else current_best.get("extra_server_args"),
            variant_extra_args=accepted.flags,
            remove_args=controls.get("remove_args"),
            args_mode="replace" if complete else "append",
        )
        if not complete and controls.get("remove_args"):
            controls["remove_args"] = _removals_the_accepted_flags_keep(controls["remove_args"], accepted.flags)
        measurement["extra_envs"] = _promoted_launch_envs(
            current_best, unset=accepted_controls.get("unset_envs", []), accepted_envs=accepted.envs
        )
        measurement.update(controls)
        measurement["args_mode"] = "replace"

    def _recipe_inherited_server_args(self) -> str:
        """The server args the baseline recipe set, which a launch that is not a replace inherits."""
        from hyperloom.inference_optimizer.framework_registry import server_args_env_name

        recipe_envs = _geak_handoff.read_recipe_bench_envs(str(self.shared_state.baseline_config_path or ""))
        return str(recipe_envs.get(server_args_env_name(self.shared_state.framework)) or "")

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
        if not isinstance(result, dict):
            return
        # Both acceptance lanes, ``env`` selections excluded and alias twins collapsed.
        specs = _geak_accepted_kernel_specs(result)
        ledger = self.shared_state.kernel_integrate_attempts
        if not specs or not isinstance(ledger, dict):
            return
        gain = _relative_gain_pct(measured_tput, baseline_tput)
        # One kernel, overlay proven loaded, one measured number: the gain is attributable.
        origin = _AdoptionOrigin(
            provenance=provenance, overlay_loaded=overlay_loaded, attributable=bool(overlay_loaded) and len(specs) == 1
        )
        state = self.shared_state
        cycle = int(state.macro_cycle or 0)
        measurement = _geak_adoption_measurement(
            result, measured_tput=measured_tput, rebench_gain=gain, cycle=cycle, origin=origin
        )
        _write_geak_adoptions(state, ledger, specs, measurement)
        log.info(
            "geak: recorded %d adopted kernel(s) in the per-kernel ledger (overlay_loaded=%r attributable=%r gain=%r)",
            len(specs),
            overlay_loaded,
            origin.attributable,
            gain if origin.attributable else None,
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
        self, result: dict[str, Any], *, handoff: dict[str, Any], run: RunnerRun = RunnerRun()
    ) -> None:
        """Record the delegated GEAK runner's terminal state."""
        _kernel_timeline.note_absent_backends(result, handoff=handoff)
        recorder = self.timeline()
        if recorder is None:
            return
        _kernel_timeline.record_geak_delegation(recorder, result, handoff=handoff, run=run)

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
        from ..kernel.request_handlers import _resolve_forge_server_log

        log_path = _resolve_forge_server_log(self.shared_state, self.session_dir)
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
        except Exception:  # parses server logs and tuner CSVs whose format varies by aiter version
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
        reference: dict[str, Any],
        candidate: dict[str, Any],
        *,
        config_path: str,
        budget_minutes: int,
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
                    "budget_minutes": budget_minutes,
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
        if tuner_name == "fmoe_ck":
            return self._fmoe_apply_verdict(envs)
        from ..kernel.gemm_shape_coverage import aiter_log_tuned_config_enabled
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
        hit_logging = aiter_log_tuned_config_enabled(envs)

        return verify_applied(
            logs[-1],
            csv_paths,
            hit_logging=hit_logging,
            runtime_table_names=table_names,
        ).to_dict()

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
        candidate_path = Path(candidate_csv_path)
        runtime_filename = _AITER_ENV_TO_TABLE.get(env_var)
        if not candidate_path.is_file() or not runtime_filename:
            return None
        config_dirs = _aiter_config_dirs(self.session_dir)
        tuned_stem = Path(runtime_filename).stem
        try:
            sources = _aiter_config_merge.runtime_config_sources(
                config_dirs, filename=runtime_filename, tuned_stem=tuned_stem
            )
            if not sources.paths:
                log.warning(
                    "gemm E2E: no complete aiter config found for %s; candidate-only validation would be unsafe",
                    env_var,
                )
                return None
            return _aiter_config_merge.merge_candidate_into_runtime(
                candidate_path,
                sources,
                env_var=env_var,
                untuned_stem=_aiter_config_merge.untuned_table_stem(tuned_stem),
                config_dirs=config_dirs,
            )
        except Exception as exc:  # noqa: BLE001
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
        self.shared_state.record_gemm_tuning(result)
        try:
            await self._validate_gemm_tuning_e2e(result)
        except Exception as exc:
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
        except Exception:
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
            envs = _applicable_envs(row.get("env"))
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
        # The rebuild's status gate cannot see a forced candidate, so it runs only when the producer named none: the
        # pre-``candidates[]`` envelope and GEAK.
        named = self._gemm_canonical_candidates(result) or _rebuilt_gemm_candidates(result)
        return named or _assumed_blockscale_candidates(result)

    async def _validate_gemm_tuning_e2e(self, result: dict[str, Any]) -> None:
        """Sequentially E2E-validate each tuning candidate's env independently."""
        candidates = self._gemm_e2e_candidates(result)
        if not candidates:
            self._close_gemm_tuning_with_no_candidates(result)
            return
        run = await self._gemm_e2e_run(result, candidates)
        ledger = _GemmE2ELedger(self._gemm_entry_tput())
        for cand in candidates:
            if not await self._validate_one_gemm_candidate(cand, run, ledger):
                break
        await self._settle_gemm_tuning(result, run, ledger)

    def _close_gemm_tuning_with_no_candidates(self, result: dict[str, Any]) -> None:
        """Close the books on a campaign that produced nothing an e2e run could validate."""
        log.info("gemm tuning: no candidates to E2E validate")
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

    def _gemm_entry_tput(self) -> float:
        """The throughput the phase enters a GEMM validation run at."""
        baseline_tput = float(self.shared_state.baseline_tput or 0.0)
        return float((self.shared_state.current_best or {}).get("tput") or baseline_tput)

    async def _gemm_e2e_run(self, result: Mapping[str, Any], candidates: list[dict[str, Any]]) -> _GemmE2ERun:
        """The conditions every candidate of one GEMM validation run is measured under."""
        from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts

        # ``_runtime_uses_aiter_fused_moe`` resolves the serving log -- which now byte-scans the whole runs/ tree for
        # aiter evidence -- and then reads it whole, ~17MB on the fleet.
        inert_triton = any(c.get("tuner") == "vllm_moe_triton" for c in candidates) and await asyncio.to_thread(
            self._runtime_uses_aiter_fused_moe
        )
        per_tuner_timeout_sec = resolve_benchmark_timeouts()[1]
        reference = deepcopy(self.shared_state.current_best or {})
        reference.setdefault("tput", self._gemm_entry_tput())
        return _GemmE2ERun(
            backend=str(result.get("backend") or "geak").strip().lower(),
            budget_minutes=max(1, int((per_tuner_timeout_sec + 59) // 60)),
            reference=reference,
            config_path=str(self.shared_state.baseline_config_path or ""),
            inert_triton=inert_triton,
            jit_backup_dir=self.session_dir / "runs" / "aiter_jit_backup",
        )

    async def _validate_one_gemm_candidate(
        self, cand: dict[str, Any], run: _GemmE2ERun, ledger: _GemmE2ELedger
    ) -> bool:
        """Measure one tuning candidate end to end; report whether the run should carry on."""
        from ..actions.executors._aiter_jit import drop_serving_so_for_envs

        tuner = cand["tuner"]
        skipped = self._gemm_candidate_skip_reason(tuner, inert_triton=run.inert_triton)
        if skipped:
            ledger.reverted.append({**cand, "reason": skipped})
            return True
        merged = self._merged_candidate_envs(cand)
        if merged.failure_reason:
            log.error(
                "gemm E2E: refusing aiter candidate for %s (%s: %s)",
                tuner,
                merged.failed_env_var,
                merged.failure_reason,
            )
            ledger.reverted.append({**cand, "reason": merged.failure_reason, "failed_env_var": merged.failed_env_var})
            return True
        env = merged.envs
        # Merge with previously KEEP'd envs.
        test_envs = {**ledger.stacked_envs, **env}
        log.info("gemm E2E: validating tuner=%s env=%s (base_tput=%.1f)", tuner, cand["env_var"], ledger.running_tput)
        outcome = await self._integrate_gemm_candidate(cand, test_envs=test_envs, run=run, ledger=ledger)
        if outcome.run_stopped:
            return False
        if outcome.verdict is None:
            return True
        measured = _gemm_measured(outcome.verdict)
        log.info(
            "gemm E2E: tuner=%s decision=%s new_tput=%.1f gain=%.2f%%",
            tuner,
            measured.decision,
            measured.new_tput,
            measured.gain_pct,
        )
        check = await self._gemm_apply_blockers(cand, env, measured.gain_pct)
        if measured.decision == "KEEP" and measured.new_tput > 0 and not check.blockers:
            if self._lift_gemm_candidate(check.candidate, env, measured, run):
                ledger.record_keep(check.candidate, env=env, measured=measured)
                self._journal_gemm_tuning_keep(
                    self.shared_state.optimization_stack[-1], task_id=f"gemm_tune_e2e_{tuner}"
                )
                return True
        ledger.reverted.append({**check.candidate, "reason": _gemm_revert_reason(measured, check.blockers)})
        await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=run.jit_backup_dir)
        return True

    def _gemm_candidate_skip_reason(self, tuner: str, *, inert_triton: bool) -> str:
        """Why a tuner's artifact cannot take effect on this runtime at all; '' when it can."""
        if tuner == "vllm_moe_triton" and inert_triton:
            log.warning(
                "gemm E2E: skipping %s — the server dispatches MoE through "
                "aiter, which never reads VLLM_TUNED_CONFIG_FOLDER, so the tuned "
                "Triton config cannot take effect",
                tuner,
            )
            return "aiter_moe_runtime_triton_config_inert"
        # fmoe_ck is only meaningful with --moe-runner-backend aiter, and aiter's CK fused-MoE rejects a
        # non-128-aligned intermediate_size_per_partition.
        from hyperloom.inference_optimizer.model_config_utils import model_supports_aiter_ck_fused_moe

        if tuner == "fmoe_ck" and not model_supports_aiter_ck_fused_moe(
            str(self.shared_state.model_path or ""),
            int(self.shared_state.tp or 0),
        ):
            log.info(
                "gemm E2E: skipping %s — aiter CK fused-MoE cannot serve "
                "this model at tp=%s (intermediate size is not 128-aligned)",
                tuner,
                self.shared_state.tp or 0,
            )
            return "aiter_ck_moe_shape_unsupported"
        return ""

    def _merged_candidate_envs(self, cand: Mapping[str, Any]) -> _MergedCandidateEnvs:
        """Merge every tuned table a candidate names into the runtime config it has to join."""
        env = dict(cand["envs"])
        for env_var, env_value in list(env.items()):
            if not env_var.startswith("AITER_CONFIG"):
                continue
            if not Path(env_value).is_file():
                return _MergedCandidateEnvs(env, "candidate_artifact_missing", env_var)
            merged_path = self._merge_gemm_candidate_with_runtime(env_var, env_value)
            if not merged_path:
                return _MergedCandidateEnvs(env, "complete_aiter_config_unavailable", env_var)
            env[env_var] = merged_path
        return _MergedCandidateEnvs(env, "", "")

    def _gemm_extra_server_args(self, tuner: str) -> str:
        """The server flag a tuner's artifact needs before the runtime will dispatch through it."""
        if tuner == "fmoe_ck" and str(self.shared_state.framework or "").lower() == "sglang":
            return "--moe-runner-backend aiter"
        return ""

    async def _integrate_gemm_candidate(
        self, cand: Mapping[str, Any], *, test_envs: dict[str, str], run: _GemmE2ERun, ledger: _GemmE2ELedger
    ) -> _IntegrateOutcome:
        """Measure one candidate's env end to end, retrying the faults that judged nothing."""
        from hyperloom.common.model_paths import resolve_session_model_path

        from ..actions.executors._aiter_jit import drop_serving_so_for_envs, prepare_serving_so_for_csvs
        from ..state.kernel_decision_settings import _MAX_INTEGRATE_FAULT_ATTEMPTS

        tuner = cand["tuner"]
        payload = {
            "task_id": f"gemm_tune_e2e_{tuner}",
            "kernel_id": f"gemm_tune_{tuner}",
            "source": "forge_gemm_tuning",
            "base_tput": ledger.running_tput,
            "model_path": resolve_session_model_path(
                state_model_path=str(self.shared_state.model_path or ""),
                for_serving=True,
            ),
            "extra_server_args": self._gemm_extra_server_args(tuner),
            "extra_envs": test_envs,
            "keep_threshold_pct": 1.0,
            "budget_minutes": run.budget_minutes,
            "mode": "env_only",
        }
        # The native handler reloads both the recipe and grading anchor from disk.
        self.shared_state.save(self.session_dir)
        for attempt in range(1, _MAX_INTEGRATE_FAULT_ATTEMPTS + 1):
            last_attempt = attempt >= _MAX_INTEGRATE_FAULT_ATTEMPTS
            await asyncio.to_thread(prepare_serving_so_for_csvs, test_envs, backup_dir=run.jit_backup_dir)
            try:
                integrate_result = await integrate_within_phase(
                    payload, state=self.shared_state, session_dir=self.session_dir
                )
            except Exception as exc:  # noqa: BLE001
                if not last_attempt:
                    log.warning(
                        "gemm E2E: integrate raised for %s (fault attempt %d/%d): %s",
                        tuner,
                        attempt,
                        _MAX_INTEGRATE_FAULT_ATTEMPTS,
                        exc,
                    )
                    continue
                log.warning("gemm E2E: integrate raised for %s: %s", tuner, exc)
                ledger.faults.append(_handler_exception_fault(cand, exc, attempts=attempt))
                await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=run.jit_backup_dir)
                return _IntegrateOutcome(verdict=None, run_stopped=False)
            stopped = stopped_by_the_run_class(integrate_result.get("error_class"))
            if stopped is not None:
                log.info("gemm E2E: %s left unmeasured — %s", tuner, stopped.interrupted)
                return _IntegrateOutcome(verdict=None, run_stopped=True)
            if not self.shared_state._is_integrate_fault(integrate_result):
                return _IntegrateOutcome(verdict=integrate_result, run_stopped=False)
            error_class = str(integrate_result.get("error_class") or "integrate_fault").strip()
            if not last_attempt:
                log.warning(
                    "gemm E2E: retrying tuner=%s after integrate fault %s (attempt %d/%d)",
                    tuner,
                    error_class,
                    attempt,
                    _MAX_INTEGRATE_FAULT_ATTEMPTS,
                )
                continue
            log.warning(
                "gemm E2E: tuner=%s integrate fault (%s) — unmeasured, not a REVERT verdict",
                tuner,
                error_class,
            )
            ledger.faults.append(_integrate_fault(cand, integrate_result, error_class, attempts=attempt))
            await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=run.jit_backup_dir)
            return _IntegrateOutcome(verdict=None, run_stopped=False)
        return _IntegrateOutcome(verdict=None, run_stopped=False)

    async def _gemm_apply_blockers(self, cand: dict[str, Any], env: dict[str, str], gain_pct: float) -> _ApplyCheck:
        """Check whether the tuned table actually reached the runtime, by both checks that can see it."""
        tuner = cand["tuner"]
        blockers: list[str] = []
        # Off the event loop for the same reason: this reads the integrate run's server.log in full and parses
        # every tuned CSV named in the candidate env.
        coverage = await asyncio.to_thread(self._gemm_tuned_config_coverage, tuner, env)
        if coverage is not None:
            cand = {**cand, "tuned_config_coverage": coverage}
            unreachable = _coverage_blocker(tuner, coverage, gain_pct)
            if unreachable:
                blockers.append(unreachable)
        applied = self._gemm_apply_verdict(tuner, env)
        if applied is not None:
            cand = {**cand, "apply_verdict": applied}
            blocked = _apply_verdict_blocker(tuner, applied)
            if blocked:
                blockers.append(blocked)
        return _ApplyCheck(candidate=cand, blockers=blockers)

    def _lift_gemm_candidate(
        self, cand: Mapping[str, Any], env: dict[str, str], measured: _GemmMeasured, run: _GemmE2ERun
    ) -> bool:
        """Adopt a measured GEMM candidate as the current best; report whether it was taken."""
        from ..measurement.integrate_performance import integrate_measurement_fields

        tuner = cand["tuner"]
        return bool(
            self._coord.writeback.lift_to_current_best(
                "gemm_tuning",
                measured.new_tput,
                {
                    "name": f"{run.backend}_{tuner}",
                    "candidate_extra_server_args": self._gemm_extra_server_args(tuner),
                    "extra_envs": dict(env),
                    "source_phase": "KERNEL_AGENT",
                    **integrate_measurement_fields(measured.measurement),
                },
                entry_extra={
                    "tuned_file": _candidate_tuned_file(env, cand.get("env_var", "")),
                    "gain_pct": measured.gain_pct,
                    "backend": run.backend,
                    "source": "kernel_entry_auto",
                },
            )
        )

    async def _settle_gemm_tuning(self, result: dict[str, Any], run: _GemmE2ERun, ledger: _GemmE2ELedger) -> None:
        """Rewrite the stored result to the E2E-validated outcome."""
        # The watermark covers the whole run, so it waits for the last KEEP.
        result["graded_objective"] = None
        if ledger.kept:
            total_gain = await self._grade_gemm_keeps(result, run, ledger)
        else:
            total_gain = 0.0
            ledger.stacked_envs = {}
            if ledger.faults:
                log.info("gemm E2E: %d tuner(s) hit integrate fault(s), no E2E verdict", len(ledger.faults))
            else:
                log.info("gemm E2E: all %d tuners REVERT, no E2E gain", len(ledger.reverted))
        result["e2e_results"] = {"kept": ledger.kept, "reverted": ledger.reverted, "faults": ledger.faults}
        result["recommended_env_raw"] = dict(result.get("recommended_env") or {})
        result["extra_envs_raw"] = dict(result.get("extra_envs") or {})
        result["recommended_env"] = dict(ledger.stacked_envs)
        result["extra_envs"] = dict(ledger.stacked_envs)
        unmeasured = bool(ledger.faults) and not ledger.kept and not ledger.reverted
        result["e2e_gain_pct"] = None if total_gain is None or unmeasured else round(float(total_gain), 4)
        result["e2e_validated"] = True
        result["requires_e2e_validation"] = False
        result.update(_gemm_tuning_verdict(ledger))
        self._replace_latest_gemm_tuning_attempt(result)

    async def _grade_gemm_keeps(self, result: dict[str, Any], run: _GemmE2ERun, ledger: _GemmE2ELedger) -> float | None:
        """Confirm the adopted GEMM increment against the configuration the phase entered on."""
        # Paired checks isolate the GEMM increment, not cumulative session gain.
        paired = await self._confirm_gemm_gain_paired(
            run.reference,
            deepcopy(self.shared_state.current_best),
            config_path=run.config_path,
            budget_minutes=run.budget_minutes,
        )
        total_gain: float | None = None
        if self._coord.writeback.validate(
            ledger.running_tput,
            ledger.accepted_measurement,
            source="forge_gemm_tuning_e2e",
            measurement_basis=_paired_measurement_basis(paired),
        ):
            total_gain = self.shared_state.cumulative_gain_validated
            result["graded_objective"] = resolve_graded_comparison(
                self.shared_state,
                {**ledger.accepted_measurement, "output_throughput": ledger.running_tput},
                against_baseline=True,
            ).objective
        # Name the artifact this run adopted, so the breakdown can tell it was.
        if ledger.adopted_tuned_file:
            result["tuned_file"] = ledger.adopted_tuned_file
        log.info(
            "gemm E2E: %d tuners KEEP (total gain=%s), %d REVERT",
            len(ledger.kept),
            f"{total_gain:+.2f}%" if total_gain is not None else "unavailable",
            len(ledger.reverted),
        )
        return total_gain

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
        except Exception:
            log.exception("KERNEL entry: sealing the campaign baseline failed")
        try:
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
        except Exception:
            log.exception("KERNEL entry: Forge handoff generation failed")
        await self._run_kernel_rewrite_controller(handoff_dir, attempt_dir, baselines)

    async def _run_kernel_rewrite_controller(
        self,
        handoff_dir: Path,
        output_dir: Path,
        baselines: dict[str, object] | None = None,
    ) -> None:
        """Run one Controller attempt without preselecting operators."""
        from ..kernel.controller_submit import record_controller_llm_usage

        cycle = int(self.shared_state.macro_cycle or 0)
        budget_sec, hard_timeout_sec = self._kernel_rewrite_controller_timeouts()
        attempt = await self._controller_attempt(
            handoff_dir, output_dir, budget_sec=budget_sec, hard_timeout_sec=hard_timeout_sec
        )
        result = {
            **attempt,
            "macro_cycle": cycle,
            "handoff_dir": str(handoff_dir),
            "budget_minutes": budget_sec / 60.0,
            "hard_timeout_sec": hard_timeout_sec,
        }
        # The Controller cannot reach this ledger from its own process, so its forge-loops' spend is filed here
        # now that the child has exited.
        record_controller_llm_usage(result=result, session_dir=self.session_dir)
        _reclaim_borrowed_repositories(result, baselines or {})
        result["integration"] = await self._integrate_controller_patches(result, output_dir)
        self._record_kernel_rewrite_controller_timeline(result)
        self.shared_state.kernel_optimizer = "forge"
        self.shared_state.kernel_rewrite_controller_result = result
        self._record_controller_patches_not_landed(result, cycle=cycle)
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

    async def _controller_attempt(
        self, handoff_dir: Path, output_dir: Path, *, budget_sec: int, hard_timeout_sec: int
    ) -> dict[str, Any]:
        """Run the Controller subprocess, or report why it could not run at all."""
        from ..kernel.controller_submit import run_controller_subprocess

        if budget_sec <= 0 or hard_timeout_sec <= 0:
            return _controller_produced_nothing(
                "no_result", "no KERNEL phase budget remains for the rewrite controller", output_dir
            )
        try:
            return await asyncio.to_thread(
                run_controller_subprocess,
                handoff_dir=handoff_dir,
                output_dir=output_dir,
                budget_minutes=budget_sec / 60.0,
                hard_timeout_sec=hard_timeout_sec,
            )
        except Exception as error:
            log.exception("KERNEL entry: kernel rewrite controller failed")
            return _controller_produced_nothing("failed", f"controller invocation failed: {error}", output_dir)

    async def _integrate_controller_patches(self, result: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
        """Integrate what the Controller published, and report how far that got."""
        if int(result.get("patch_count") or 0) <= 0:
            return {
                "status": "not_run",
                "reason": "Controller published no patches",
                "kept_count": 0,
                "reverted_count": 0,
                "skipped_count": 0,
            }
        from ..kernel.controller_patch_integration import integrate_controller_patches

        try:
            integration = await integrate_controller_patches(
                patches_root=str(result.get("patches_root") or output_dir / "result" / "patches"),
                session_dir=self.session_dir,
                shared_state=self.shared_state,
                record_keep=self._coord.writeback.record_integrate_keep,
                admit_round=lambda rounds_run: controller_round_stop(self.shared_state, rounds_run),
            )
        except Exception as error:
            log.exception("KERNEL entry: Controller patch integration failed")
            return {"status": "failed", "reason": str(error), "kept_count": 0}
        return integration.to_dict()

    def _record_controller_patches_not_landed(self, result: Mapping[str, Any], *, cycle: int) -> None:
        """Record a Controller round whose patches were refused or skipped as an action failure."""
        integration = result.get("integration")
        if not isinstance(integration, dict):
            return
        skipped = [
            str(row.get("reason") or "")
            for row in (integration.get("results") or [])
            if isinstance(row, dict) and str(row.get("status") or "").startswith("skipped")
        ]
        status = str(integration.get("status") or "")
        if status not in {"failed", "no_patch_admitted"} and not skipped:
            return
        reasons = [str(integration.get("reason") or ""), *skipped]
        self.shared_state.record_action_failure(
            action="kernel_rewrite_controller",
            task_id=str(result.get("run_id") or f"forge-cycle-{cycle}"),
            result={
                "error_class": status or "patches_skipped",
                "error": "; ".join(reason for reason in reasons if reason)[:800],
            },
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
                return _as_int(self.shared_state.fusion_withheld_retries) < MAX_FUSION_WITHHELD_RETRIES
            return False
        if isinstance(last, dict) and last.get("infrastructure_abort"):
            # An abort judged nothing, so it must stay retryable -- but not forever.
            spent = _as_int(self.shared_state.fusion_infra_aborts)
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
        except Exception as exc:
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
        state = self.shared_state
        if not isinstance(result, dict):
            _save_last_fusion(state, self.session_dir, {"status": "failed"})
            await _announce_fusion_done(self.bus, result, status="failed")
            return
        _fail_unreadable_fusion_envelope(result)
        _charge_fusion_retry_caps(state, result)
        if not str(result.get("fusion_run_id") or "").strip():
            result["fusion_run_id"] = f"fusion-c{int(state.macro_cycle or 0)}-{time.time_ns():x}"
        _save_last_fusion(state, self.session_dir, result)
        self._record_fusion_timeline(result)
        await _announce_fusion_done(self.bus, result, status=str(result.get("status") or "unknown"))
        # A KEPT fusion is handed to integrate for the e2e re-baseline decision.
        if _fusion_awaits_e2e(result):
            await self._integrate_fusion(result)

    async def _integrate_fusion(self, result: dict) -> None:
        """Queue every KEPT forge-fusion sibling for the shared e2e integrate lane."""
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
        keep_pct = env_float("HYPERLOOM_FUSION_KEEP_PCT", default=1.0)
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
        except Exception as exc:
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
