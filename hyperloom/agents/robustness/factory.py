"""High-level builders that turn a :class:`Config` into a running reactor.

Single entry point:

* :func:`build_reactor_components` returns a :class:`ReactorBundle` that
  bundles the reactor, the :class:`~robustness_agent.role.reactor.ReactorComponents`,
  the underlying :class:`~robustness_agent.findings.sink.FindingSink` and
  :class:`~robustness_agent.sources.server_client.RobustnessServerClient`
  so callers can manage their lifecycles (e.g. ``await bundle.aclose()``
  on shutdown).

Hosts (the Coordinator, the standalone ``main.py`` reactor loop, the
``robustness_agent.runtime.cli`` subprocess transport) all build a
bundle directly and drive ``bundle.reactor.tick(ctx)`` per tick. There
is no in-process Backend adapter anymore — the architecturally-blessed
transport is the subprocess CLI, mirroring the critic-agent layout.

The factory never blocks on remote services — :class:`Config.discover`
already probed reachability and recorded the URLs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


import os

from .config import Config
from .decision.action_ladder import ActionLadder, ActionLadderConfig
from .decision.policy_aware import PolicyAware
from .decision.rca_engine import (
    LlmRcaEngine,
    NoopRcaEngine,
    RcaEngine,
    RcaThrottle,
    RcaThrottleConfig,
)
from .finalize.postmortem import (
    PostmortemFinalizer,
    PostmortemFinalizerConfig,
)
from .findings.sink import FindingSink, FindingSinkConfig
from .role.reactor import Reactor, ReactorComponents
from .state_store import DetectorStateStore
from .signals import Classifier, SymptomSeverity
from .signals.aiter_jit import AiterJitConfig
from .signals.budget import BudgetConfig
from .signals.crash import CrashConfig
from .signals.decision_audit import DecisionAuditConfig
from .signals.event import EventConfig
from .signals.gpu_leak import GpuLeakConfig
from .signals.health import HealthConfig
from .signals.critic_health import CriticHealthConfig
from .signals.external_deps import ExternalDepsConfig
from .signals.kernel_pipeline import KernelPipelineConfig
from .signals.local_health import LocalHealthConfig
from .signals.preflight import (
    AmdahlCeilingConfig,
    ColdStartConfig,
    ModelGpuFitConfig,
)
from .signals.progress import ProgressConfig
from .signals.repeated_payload import RepeatedPayloadConfig
from .signals.state_integrity import StateIntegrityConfig
from .signals.stall import StallConfig
from .sources.base import DegradeRouter, Source, SourceData, SourceUnavailable
from .sources.local_probe import LocalProbeConfig, LocalProbeSource
from .sources.server_client import (
    RobustnessServerClient,
    RobustnessServerSource,
)


log = logging.getLogger(__name__)


@dataclass
class ReactorBundle:
    """Aggregate of reactor + lifecycle handles a caller needs to manage."""

    reactor: Reactor
    components: ReactorComponents
    server_client: RobustnessServerClient | None
    sink: FindingSink

    async def aclose(self) -> None:
        if self.server_client is not None:
            await self.server_client.aclose()


def build_reactor_components(
    config: Config,
    *,
    rca: RcaEngine | None = None,
    session_id: str | None = None,
) -> ReactorBundle:
    """Construct everything the reactor needs.

    Parameters
    ----------
    config:
        Discovered :class:`Config` — typically the result of
        ``await Config.discover()``.
    rca:
        Optional RCA engine override. Defaults to :class:`NoopRcaEngine`
        because M1 ships RCA disabled.
    session_id:
        Override for the FindingSink filename.  Defaults to
        ``config.session_dir.name`` so each per-session sandbox writes
        to a stable file.
    """
    # Primary source: robustness-server (omitted in local-only mode).
    server_client: RobustnessServerClient | None = None
    primary: Source
    if config.robustness_server_url:
        server_client = RobustnessServerClient(
            config.robustness_server_url,
            timeout_s=config.server_request_timeout_s,
        )
        primary = RobustnessServerSource(
            server_client,
            metrics_window_s=config.metrics_window_s,
        )
    else:
        primary = _NoServerSource(
            "robustness-server",
            "config.robustness_server_url is empty",
        )

    # Auto-include the local inference server health endpoint so an
    # SGLang/vLLM/Magpie SIGSTOP fires a symptom — without it, the
    # default ``health_probe_targets`` list is empty and operators
    # routinely forget to populate it.
    probe_targets = list(config.health_probe_targets)
    if (
        config.auto_probe_inference_server
        and config.inference_server_health_url
        and config.inference_server_health_url not in probe_targets
    ):
        probe_targets.append(config.inference_server_health_url)

    extra_log_globs: tuple[str, ...] = (
        "runs/*/*/server.log",
        "runs/*/*/server_log",
        "runs/*/server.log",
        # Mirrors ``LocalProbeConfig.extra_server_log_globs`` defaults —
        # see that field for the rationale (grid_runner variant layout).
        "runs/*/*/*/server.log",
        "runs/*/*/*/*/server.log",
    )
    if config.server_log_extra_globs:
        extra_log_globs = (
            *extra_log_globs,
            *(g.strip() for g in config.server_log_extra_globs.split(":") if g.strip()),
        )

    fallback = LocalProbeSource(
        LocalProbeConfig(
            session_dir=config.session_dir,
            process_patterns=tuple(
                config.server_process_patterns + config.benchmark_process_patterns
            ),
            health_probe_targets=tuple(probe_targets),
            health_probe_timeout_s=config.health_probe_timeout_s,
            ray_probe_enabled=config.ray_probe_enabled,
            ray_probe_timeout_s=config.ray_probe_timeout_s,
            fd_probe_enabled=config.fd_probe_enabled,
            fd_probe_pid=config.fd_probe_pid,
            decision_audit_enabled=config.decision_audit_enabled,
            decision_audit_max_integrate=config.decision_audit_max_integrate,
            decision_audit_max_oob_attempts=(
                config.decision_audit_max_oob_attempts
            ),
            preflight_enabled=config.preflight_enabled,
            critic_health_enabled=config.critic_health_enabled,
            max_critic_judge_bundles=config.critic_health_max_judge_bundles,
            extra_server_log_globs=extra_log_globs,
            max_extra_server_logs=config.server_log_max_extra,
            state_integrity_enabled=config.state_integrity_enabled,
            external_deps_enabled=config.external_deps_enabled,
            external_mount_stat_timeout_s=(
                config.external_mount_stat_timeout_s
            ),
            external_gateway_probe_url=config.external_gateway_probe_url,
        )
    )

    router = DegradeRouter(
        primary,
        fallback,
        fail_threshold=config.source_fail_threshold,
        recheck_interval_s=config.source_recheck_interval_s,
    )

    # Disk-backed state store — single source of truth for any
    # subsystem that needs to survive a subprocess restart (detectors,
    # ladder cooldown, RCA throttle). Built before the classifier so
    # we can pass it in once and Classifier wires it to all stateful
    # sub-detectors.
    state_store: DetectorStateStore | None = (
        DetectorStateStore(session_dir=config.session_dir)
        if config.state_store_enabled
        else None
    )

    classifier = Classifier(
        state_store=state_store,
        stall_config=StallConfig(
            stall_timeout_s=config.agent_stall_timeout_s,
        ),
        crash_config=CrashConfig(),
        event_config=EventConfig(
            idempotency_replay_threshold=config.idempotency_replay_threshold,
        ),
        health_config=HealthConfig(),
        local_health_config=LocalHealthConfig(
            gpu_temp_warn_c=config.gpu_temp_warn_c,
            disk_used_warn_pct=config.disk_used_warn_pct,
            disk_used_crit_pct=config.disk_used_crit_pct,
            shm_used_warn_pct=config.shm_used_warn_pct,
            shm_used_crit_pct=config.shm_used_crit_pct,
            fd_warn_used_pct=config.fd_warn_used_pct,
            fd_crit_used_pct=config.fd_crit_used_pct,
        ),
        gpu_leak_config=GpuLeakConfig(
            util_mem_pct_threshold=config.gpu_leak_util_mem_pct_threshold,
            free_mb_threshold=config.gpu_leak_free_mb_threshold,
            min_consecutive_ticks=config.gpu_leak_min_consecutive_ticks,
        ),
        budget_config=BudgetConfig(
            warn_pct=config.budget_warn_pct,
            imminent_pct=config.budget_imminent_pct,
            min_budget_minutes=config.budget_min_minutes,
            productive_gain_pct=config.budget_productive_gain_pct,
            strategy_drift_pct=config.budget_strategy_drift_pct,
            deadline_warning_minutes=config.budget_deadline_warning_minutes,
            deadline_hard_cutoff_minutes=(
                config.budget_deadline_hard_cutoff_minutes
            ),
        ),
        aiter_jit_config=AiterJitConfig(
            cold_so_count=config.aiter_jit_cold_so_count,
            regression_ratio=config.aiter_jit_regression_ratio,
            stale_build_threshold=config.aiter_jit_stale_build_threshold,
            stale_build_persist_ticks=config.aiter_jit_stale_build_persist_ticks,
        ),
        progress_config=ProgressConfig(
            gain_window_ticks=config.progress_gain_window_ticks,
            gain_epsilon_pct=config.progress_gain_epsilon_pct,
            no_levers_min_minutes=config.progress_no_levers_min_minutes,
            no_levers_min_ticks=config.progress_no_levers_min_ticks,
            productive_gain_pct=config.budget_productive_gain_pct,
        ),
        repeated_payload_config=RepeatedPayloadConfig(
            streak_threshold=config.repeated_payload_streak_threshold,
            lookback_events=config.repeated_payload_lookback_events,
        ),
        decision_audit_config=DecisionAuditConfig(
            min_keep_gain_pct=config.decision_audit_min_keep_gain_pct,
            dispatch_bypass_pre_post_epsilon_pct=(
                config.decision_audit_dispatch_bypass_epsilon_pct
            ),
        ),
        model_gpu_fit_config=ModelGpuFitConfig(
            min_headroom_pct=config.preflight_min_headroom_pct,
            activation_buf_gib=config.preflight_activation_buf_gib,
        ),
        amdahl_ceiling_config=AmdahlCeilingConfig(
            single_kernel_speedup=(
                config.preflight_amdahl_single_kernel_speedup
            ),
            min_e2e_ceiling_pct=config.preflight_amdahl_min_e2e_ceiling_pct,
        ),
        cold_start_config=ColdStartConfig(
            cold_so_count=config.preflight_cold_start_so_count,
            cold_start_minutes=config.preflight_cold_start_minutes,
        ),
        critic_health_config=CriticHealthConfig(
            min_outage_judges=config.critic_health_min_outage_judges,
            min_unavailable_verdicts=(
                config.critic_health_min_unavailable_verdicts
            ),
            max_workdir_count=config.critic_health_max_workdir_count,
        ),
        kernel_pipeline_config=KernelPipelineConfig(
            pending_count_threshold=(
                config.kernel_pipeline_pending_count_threshold
            ),
            min_pending_ticks=config.kernel_pipeline_min_pending_ticks,
            min_geak_sigterm_attempts=(
                config.kernel_pipeline_min_geak_sigterm_attempts
            ),
            min_cursor_401_hits=(
                config.kernel_pipeline_min_cursor_401_hits
            ),
            min_kernels_with_no_progress=(
                config.kernel_pipeline_min_kernels_with_no_progress
            ),
        ),
        state_integrity_config=StateIntegrityConfig(
            wal_bytes_warn_threshold=config.state_wal_bytes_warn_threshold,
            wal_bytes_critical_threshold=(
                config.state_wal_bytes_critical_threshold
            ),
            stale_lease_min_age_s=config.state_stale_lease_min_age_s,
            inbox_bloat_warn_bytes=config.state_inbox_bloat_warn_bytes,
            inbox_bloat_critical_bytes=(
                config.state_inbox_bloat_critical_bytes
            ),
        ),
        external_deps_config=ExternalDepsConfig(
            mount_latency_warn_ms=config.external_mount_latency_warn_ms,
            mount_latency_critical_ms=(
                config.external_mount_latency_critical_ms
            ),
        ),
    )

    ladder = ActionLadder(
        config=ActionLadderConfig(cooldown_ticks=config.cooldown_ticks),
        state_view=(
            state_store.view("action_ladder") if state_store else None
        ),
    )

    sink_session_id = session_id or config.session_dir.name or "default"
    sink = FindingSink(
        FindingSinkConfig(session_dir=config.session_dir, session_id=sink_session_id)
    )

    rca_engine: RcaEngine = rca if rca is not None else _build_rca_engine(
        config, state_store=state_store,
    )

    finalizer = (
        PostmortemFinalizer(
            session_dir=config.session_dir,
            session_id=sink_session_id,
            config=PostmortemFinalizerConfig(
                reports_subdir=config.finalize_reports_subdir,
                max_findings_in_report=config.finalize_max_findings_in_report,
                max_tasks_per_action=config.finalize_max_tasks_per_action,
            ),
        )
        if config.finalize_enabled
        else None
    )

    components = ReactorComponents(
        router=router,
        classifier=classifier,
        ladder=ladder,
        policy=PolicyAware(),
        sink=sink,
        rca=rca_engine,
        finalizer=finalizer,
        state_store=state_store,
    )
    return ReactorBundle(
        reactor=Reactor(components),
        components=components,
        server_client=server_client,
        sink=sink,
    )


def _build_rca_engine(
    config: Config,
    *,
    state_store: DetectorStateStore | None = None,
) -> RcaEngine:
    """Choose between Noop and Llm based on config + env override."""
    if os.environ.get("ROBUSTNESS_LLM_RCA_DISABLED", "").lower() in {"1", "true", "yes"}:
        log.info("LLM RCA disabled via ROBUSTNESS_LLM_RCA_DISABLED env override")
        return NoopRcaEngine()
    if config.llm_rca_enabled is False:
        return NoopRcaEngine()
    if not config.llm_base_url or not config.llm_api_key:
        if config.llm_rca_enabled is True:
            log.warning("llm_rca_enabled=True but base_url/api_key missing; using NoopRcaEngine")
        return NoopRcaEngine()

    severity_min = _parse_severity(config.llm_rca_severity_min)
    throttle = RcaThrottle(
        RcaThrottleConfig(
            severity_min=severity_min,
            cooldown_seconds=config.llm_rca_cooldown_s,
            max_calls_per_tick=config.llm_rca_max_calls_per_tick,
        ),
        state_view=(
            state_store.view("rca_throttle") if state_store else None
        ),
    )
    log.info(
        "LLM RCA enabled: model=%s severity_min=%s cooldown=%.1fs max_per_tick=%d",
        config.llm_model,
        severity_min.value,
        config.llm_rca_cooldown_s,
        config.llm_rca_max_calls_per_tick,
    )
    return LlmRcaEngine(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        timeout_s=config.llm_rca_timeout_s,
        max_chars=config.llm_rca_max_chars,
        throttle=throttle,
    )


def _parse_severity(value: str) -> SymptomSeverity:
    normalized = (value or "").strip().lower()
    if normalized in {"low", "info", "observe"}:
        return SymptomSeverity.LOW
    if normalized in {"medium", "warn", "warning"}:
        return SymptomSeverity.MEDIUM
    return SymptomSeverity.HIGH


@dataclass
class _NoServerSource:
    """Permanent stub used when no robustness-server URL is configured.

    DegradeRouter degrades it after ``fail_threshold`` ticks, after which
    the LocalProbe takes over without further probes.
    """

    name: str
    reason: str

    async def fetch(self, ctx: Any) -> SourceData:
        raise SourceUnavailable(self.reason)


__all__ = [
    "ReactorBundle",
    "build_reactor_components",
]
