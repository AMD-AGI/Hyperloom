# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared value types and helpers for the ``explore`` executor's grid runs.

A grid round of a session that froze a base launch (:func:`._suite.rounds.suite_route`) runs through
:func:`run_grid_suite_round`, which holds the server as the point's ``server_lifecycle`` block says
(:func:`._suite.rounds.point_server`) and hands the round back the way the grid reads a finished benchmark process: a
round the session deadline or a cancel stopped exits with the sentinel the grid reads as the run's own stop, any other
failed step exits ``1``, and the grid records it under that step's error class (:func:`failed_step_class`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom_common.coerce import to_str_list
from hyperloom.common.failure_signature import argparse_refusal
from hyperloom.common.perf_metric import GRADED_INTVTY_P50_CHECKPOINT
from hyperloom.common.env_safety import (
    BLOCKED_EXTERNAL_ENV_NAMES,
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
)
from hyperloom.common.overlay import is_safe_path_entry
from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID
from hyperloom.inference_optimizer.grid_server_args import (
    SGLANG_MOE_RUNNER_BACKEND_RE,
    moe_runner_requires_aiter,
    remove_moe_runner_backend_arg,
)

from inference_suite import stop
from inference_suite.contracts import FailedStep

from ._agentx_checkpoint import AgentxCheckpoint, read_checkpoint
from ._failed_step import failed_step_error_class
from ._subprocess_kill import (
    AGENTX_EARLY_STOP_ERROR_CLASS,
    AGENTX_EARLY_STOP_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    sync_agentx_phase_wait,
)
from ._ray_serving import ServingLease
from ._suite.rounds import SuiteRound, SuiteRoundRefused, SuiteRoute, deadline_unix, point_server, run_suite_round

log = logging.getLogger(__name__)


# Per-variant KEEP threshold (gain-pct + accuracy gate); the grid noise floor.
DEFAULT_KEEP_THRESHOLD_PCT = 1.0

# Terminal outcomes that ended a grid variant without a graded comparison, so
# they are neither a KEEP nor a REVERT.
TS_FAILED = "FAILED"
TS_SKIPPED_DEDUP = "SKIPPED_DEDUP"

#: The point YAML a grid variant's rounds run from, in the variant's slot beside the rounds' workspaces.
VARIANT_CONFIG_FILE = "config.yaml"


def variant_config_of(workspace: str) -> Path:
    """The point YAML the grid round whose workspace is ``workspace`` ran from."""
    return Path(workspace).parent / VARIANT_CONFIG_FILE


def sync_benchmark_timeout(config_path: Path, timeout_sec: float) -> None:
    """Hold the round of the point at ``config_path`` to the same cap as the enclosing benchmark process."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bench = cfg["benchmark"]
    bench["timeout_seconds"] = timeout_sec
    if bench.get("server_lifecycle"):
        bench["server_lifecycle"]["server_ready_timeout_s"] = timeout_sec
    envs = bench.setdefault("envs", {})
    envs["PYTHONUNBUFFERED"] = "1"
    sync_agentx_phase_wait(envs, timeout_sec)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True)
class GridRound:
    """One benchmark round of a grid: the config and directories it runs with, and its stop gates.

    ``server_already_ready`` is whether an earlier round left the round's server up. ``boot_only`` is whether the
    round only brings its server up for the rounds after it, which an inference_suite round does without a load.
    ``checkpoint`` stops an AgentX round that trails its reference at the checkpoint.
    """

    config_path: Path
    output_dir: Path
    timeout_sec: int
    cwd: str
    result_dir: str | None = None
    silence_timeout_sec: float | None = None
    server_already_ready: bool = False
    session_deadline_sec: float | None = None
    warmup: bool = False
    boot_only: bool = False
    checkpoint: AgentxCheckpoint | None = None


_FAILED_RETURNCODE = 1
_STOPPED_BY_THE_RUN: Mapping[FailedStep, int] = {
    FailedStep.DEADLINE_EXHAUSTED: SESSION_TIME_EXHAUSTED_RETURNCODE,
    FailedStep.CANCELLED: ORCHESTRATOR_CANCELLED_RETURNCODE,
}


@dataclass(frozen=True)
class SuiteRoundExit:
    """How a grid round on inference_suite ended: its exit code and, when it failed, the report's failure detail."""

    returncode: int
    detail: str = ""


def run_grid_suite_round(
    route: SuiteRoute,
    round_: GridRound,
    *,
    env_drop: frozenset[str],
    serving_lease: ServingLease | None,
    on_poll: Callable[[], None],
) -> SuiteRoundExit:
    """Run ``round_`` on ``route``; ``env_drop`` names host variables its processes do not inherit.

    ``serving_lease`` is the Ray lease whose actor runs the round, ``None`` to run it in this process.

    Raises:
        SuiteRoundRefused: When the round is refused before it runs.
    """
    sync_benchmark_timeout(round_.config_path, round_.timeout_sec)
    server = point_server(round_.config_path, finds_server=round_.server_already_ready)
    request = SuiteRound(
        config_path=round_.config_path,
        round_dir=round_.output_dir,
        lifecycle=server.lifecycle,
        warmup=round_.warmup,
        boot_only=round_.boot_only,
        deadline_unix=deadline_unix(round_.session_deadline_sec),
        env_drop=env_drop,
        serving_lease=serving_lease,
        checkpoint=round_.checkpoint,
    )
    try:
        report = run_suite_round(route, request, on_poll=on_poll).report
    except SuiteRoundRefused as exc:
        if exc.error_class != AGENTX_EARLY_STOP_ERROR_CLASS:
            raise
        return SuiteRoundExit(AGENTX_EARLY_STOP_RETURNCODE, exc.detail)
    finally:
        if server.stop_after is not None and (survivors := stop(server.stop_after).survivors):
            log.error("grid: pids %s outlived SIGKILL; %s keeps their record", survivors, server.stop_after)
    if report.failed_step is None:
        return SuiteRoundExit(0)
    # A leased round's client takes the checkpoint's SIGTERM with its CLI, so its report may name any step.
    if _stopped_by_checkpoint(round_):
        return SuiteRoundExit(AGENTX_EARLY_STOP_RETURNCODE, report.failure_detail or report.failed_step.value)
    returncode = _STOPPED_BY_THE_RUN.get(report.failed_step, _FAILED_RETURNCODE)
    return SuiteRoundExit(returncode, report.failure_detail or report.failed_step.value)


def _stopped_by_checkpoint(round_: GridRound) -> bool:
    """Whether the ruling the round's checkpoint wrote into its slot stopped it; the round runner clears a stale one."""
    return round_.checkpoint is not None and bool((read_checkpoint(round_.output_dir) or {}).get("stopped"))


def failed_step_class(measurement: Mapping[str, Any]) -> str | None:
    """The error class of the step a run_dir's round failed at, or ``None`` for a round that failed at no step."""
    step = measurement.get("failed_step")
    return None if step is None else failed_step_error_class(FailedStep(step))


@dataclass
class GridVariant:
    """One row of the grid we're going to test."""

    name: str
    extra_server_args: str = ""
    extra_envs: dict[str, str] | None = None
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = field(default=None, kw_only=True)
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = field(default=None, kw_only=True)
    args_mode: str = field(default="append", kw_only=True)
    note: str = ""

    def __post_init__(self) -> None:
        """Drop unsafe envs and normalize the removal and args-mode controls."""
        self.extra_envs, dropped_envs = filter_untrusted_env_mapping(
            self.extra_envs,
            allow_predicate=is_allowed_variant_env_key,
        )
        if dropped_envs:
            log.warning("Variant %s: dropping unsafe extra_envs %s", self.name, ", ".join(sorted(dropped_envs)))
        self.remove_args = to_str_list(self.remove_args)
        self.unset_envs = to_str_list(self.unset_envs)
        mode = str(self.args_mode or "append").strip().lower()
        self.args_mode = mode if mode in {"append", "replace"} else "append"
        # Optional runtime override; injected into materialized YAML benchmark.envs by _build_variant_yaml so the
        # server subprocess resolves the attempt runtime.
        self.runtime_override: dict[str, str] = {}
        self.source_patches: list[dict[str, str]] = []
        self.overlay_pythonpath: str = ""


@dataclass
class VariantResult:
    """One bench run's parsed result."""

    name: str
    extra_server_args: str
    extra_envs: dict[str, str]
    status: str
    valid_measurement: bool | None = None
    diagnostic_only: bool | None = None
    workload_identity: str | None = None
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    gpu_power_avg_w: float | None = None
    gpu_power_by_gpu_w: dict[str, float] | None = None
    tpot_mean_ms: float | None = None
    input_throughput: float | None = None
    tpot_p90_ms: float | None = None
    intvty_p90: float | None = None
    intvty_p50: float | None = None
    intvty_p50_checkpoint: float | None = None
    request_error_rate: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Short failure-classification tag; empty for successes.
    error_class: str = ""
    note: str = ""
    # Wall-clock seconds the round consumed.
    runtime_sec: float | None = None
    # True iff reaped by the overtime soft deadline.
    killed_overtime: bool = False
    # Rough output tok/s salvaged from server.log on the killed_overtime path; informational only, never feeds winner
    # selection.
    estimated_output_throughput: float | None = None
    server_log_path: str | None = None
    launch_evidence: dict[str, Any] = field(default_factory=dict)
    launch_evidence_path: str | None = None
    # Hard cap this run was held to, published by whoever granted it; None when no run was granted one.
    # Readers report this number rather than deriving their own, which would describe a different run.
    granted_cap_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain JSON-friendly dict."""
        return {
            "valid_measurement": self.valid_measurement,
            "diagnostic_only": self.diagnostic_only,
            "workload_identity": self.workload_identity,
            "name": self.name,
            "extra_server_args": self.extra_server_args,
            "extra_envs": self.extra_envs,
            "status": self.status,
            "output_throughput": self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds": self.duration_seconds,
            "ttft_mean_ms": self.ttft_mean_ms,
            "e2el_mean_ms": self.e2el_mean_ms,
            "gpu_power_avg_w": self.gpu_power_avg_w,
            "gpu_power_by_gpu_w": self.gpu_power_by_gpu_w,
            "tpot_mean_ms": self.tpot_mean_ms,
            "input_throughput": self.input_throughput,
            "tpot_p90_ms": self.tpot_p90_ms,
            "e2e_norm_intvty_p90": self.intvty_p90,
            "e2e_norm_intvty_p50": self.intvty_p50,
            GRADED_INTVTY_P50_CHECKPOINT: self.intvty_p50_checkpoint,
            "request_error_rate": self.request_error_rate,
            "workspace": self.workspace,
            "report_path": self.report_path,
            "raw_result_path": self.raw_result_path,
            "reported_success": self.reported_success,
            "returncode": self.returncode,
            "nonfatal_warnings": self.nonfatal_warnings,
            "error": self.error,
            "error_class": self.error_class,
            "note": self.note,
            "runtime_sec": self.runtime_sec,
            "killed_overtime": self.killed_overtime,
            "estimated_output_throughput": self.estimated_output_throughput,
            "server_log_path": self.server_log_path,
            "launch_evidence": self.launch_evidence,
            "launch_evidence_path": self.launch_evidence_path,
            "granted_cap_sec": self.granted_cap_sec,
        }


class RoundError(NamedTuple):
    """Why a grid round measured nothing, and under which class it is recorded."""

    summary: str
    error_class: str


def argv_refusal(server_log: str | None) -> RoundError | None:
    """The round as its server's argument parser refused it, when the server's log says it did."""
    if not server_log:
        return None
    try:
        refusal = argparse_refusal(Path(server_log).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return None if refusal is None else RoundError(refusal, ARGV_INVALID)


def apply_base_env(envs, base_extra_envs, base_unset_envs):
    """Apply the accepted environment edits to ``envs`` and return the base launch's environment."""
    for k in to_str_list(base_unset_envs):
        if k.strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            continue
        envs.pop(k, None)
    for k, v in (base_extra_envs or {}).items():
        envs[str(k)] = str(v)
    return dict(envs)


def apply_variant_env(envs, variant, base_extra_envs, base_unset_envs):
    """Apply accepted environment edits before the candidate's set/unset operations."""
    # Composed base-then-variant, so a variant unsetting a key the base sets
    # removes it: the last layer to name a key is the one that decides it.
    base_envs = apply_base_env(envs, base_extra_envs, base_unset_envs)
    for k in getattr(variant, "unset_envs", []) or []:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(k).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("grid: refusing to unset pinned env %s for variant %s", k, variant.name)
            continue
        envs.pop(str(k), None)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)
    return base_envs


def runnable_moe_backend_args(combined: str, *, args_env: str, model_path: str | None, variant: GridVariant) -> str:
    """``combined`` with a MoE runner backend this checkpoint cannot run dropped."""
    if not combined or not SGLANG_MOE_RUNNER_BACKEND_RE.search(combined):
        return combined
    if args_env != "EXTRA_SGLANG_ARGS" or not moe_runner_requires_aiter(combined, model_path):
        return combined
    log.warning(
        "grid: dropping inherited --moe-runner-backend for variant %s: "
        "this checkpoint's MoE quant scheme is only implemented on the "
        "aiter runner and would crash on the first forward pass.",
        variant.name,
    )
    return remove_moe_runner_backend_arg(combined)


def importable_overlay_dir(variant: GridVariant) -> str:
    """This variant's overlay directory, or ``""`` when it names none the server may import."""
    overlay = str(getattr(variant, "overlay_pythonpath", "") or "").strip()
    if not overlay:
        return ""
    if is_safe_path_entry(overlay) and Path(overlay).is_dir():
        return overlay
    log.warning(
        "grid: dropping unsafe overlay_pythonpath %r (not a single existing directory / contains separator or traversal)",
        overlay,
    )
    return ""


def pin_xdit_tool_path(envs: dict[str, Any], framework: Any) -> None:
    """Put the ROCm and venv tool directories ahead of a xDiT launch's ``PATH``."""
    if str(framework or "").strip().lower() != "xdit":
        return
    parts = [p for p in str(envs.get("PATH", "") or "").split(":") if p]
    for essential in ("/opt/rocm/bin", "/venv/bin"):
        if essential not in parts:
            parts.insert(0, essential)
    envs["PATH"] = ":".join(parts)


# Curated MTP-capable model class set (needs multi-token-prediction heads).
_ATOM_MTP_CAPABLE_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "moe_mla",
        "moe_mla_nsa",
    }
)


def _atom_default_grid(
    *,
    model_class: str,
    conc: int,
) -> list[GridVariant]:
    """Atom default explore grid, seeded from atom's known perf knobs."""
    mc_l = (model_class or "").strip().lower()
    is_moe = "moe" in mc_l
    is_mla = "mla" in mc_l
    is_fp8 = "fp8" in mc_l or mc_l.endswith("_fp8")
    is_mtp_capable = mc_l in _ATOM_MTP_CAPABLE_MODEL_CLASSES

    variants: list[GridVariant] = []

    def _add(name: str, args: str) -> None:
        """Append a ``default_grid``-provenance variant to the grid."""
        gv = GridVariant(
            name=name,
            extra_server_args=args,
            extra_envs={},
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # ``atom_level_3`` is atom's default, so use ``atom_level_2`` as the off-default contrast.
    _add("atom_level_2", "--level 2")
    _add("atom_prefix_cache", "--enable_prefix_caching")

    if is_fp8:
        _add("atom_kv_fp8", "--kv_cache_dtype fp8")

    if is_moe:
        _add("atom_ep", "--enable-expert-parallel")

    if is_mla:
        _add("atom_dp_attn", "--enable-dp-attention")

    if is_mtp_capable:
        _add(
            "atom_mtp_3",
            "--method mtp --num-speculative-tokens 3",
        )
        _add(
            "atom_mtp_1",
            "--method mtp --num-speculative-tokens 1",
        )

    if conc and conc > 0:
        # Bracket the live concurrency so cudagraph captures the actual decode batch sizes the workload spends most of
        # its time at.
        cg_sizes = sorted({1, 2, 4, 8, 16, int(conc)})
        cg_str = "[" + ",".join(str(s) for s in cg_sizes) + "]"
        _add(
            "atom_cudagraph_bracket",
            f"--cudagraph-capture-sizes {cg_str}",
        )

    return variants


def _xdit_default_grid() -> list[GridVariant]:
    """xDiT (diffusion) default explore grid, seeded from the empirical KB."""
    variants: list[GridVariant] = []

    def _add(name: str, *, envs: dict[str, str]) -> None:
        """Append a ``default_grid``-provenance env-only variant."""
        gv = GridVariant(
            name=name,
            extra_server_args="",
            extra_envs=dict(envs),
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # AMD buffer load/store instructions — directionally correct, BF16-safe.
    _add("xdit_buffer_ops", envs={"AMDGCN_USE_BUFFER_OPS": "1"})
    # torch.compile reduce-overhead: expose an explicit on/off contrast.
    _add("xdit_compile_reduce_overhead", envs={"XDIT_USE_TORCH_COMPILE": "1"})
    _add("xdit_no_compile", envs={"XDIT_USE_TORCH_COMPILE": "0"})
    # Confirm the safe attention backend (aiter).
    _add("xdit_attn_aiter", envs={"XDIT_ATTENTION_BACKEND": "aiter"})
    return variants


def _default_grid_for_framework(
    framework: str,
    *,
    model_class: str,
    conc: int = 0,
) -> list[GridVariant]:
    """Framework-keyed default grid dispatch."""
    fw = (framework or "").strip().lower()
    if fw == "atom":
        return _atom_default_grid(model_class=model_class, conc=conc)
    if fw == "xdit":
        return _xdit_default_grid()
    return []
