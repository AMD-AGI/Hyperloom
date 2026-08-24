# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""High-level builders that turn a :class:`Config` into a running reactor.

:func:`build_reactor_components` returns a :class:`ReactorBundle` (reactor +
ReactorComponents + FindingSink) so callers can manage lifecycles via
``await bundle.aclose()``. All hosts drive ``bundle.reactor.tick(ctx)`` per
tick; the blessed transport is the subprocess CLI (mirrors critic-agent), no
in-process Backend adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


import os

from hyperloom.common import llm_config

from .config import Config
from .decision.action_ladder import ActionLadder, ActionLadderConfig
from .decision.policy_aware import PolicyAware
from .decision.rca_engine import (
    AnthropicRcaEngine,
    LlmRcaEngine,
    NoopRcaEngine,
    RcaEngine,
    RcaThrottle,
    RcaThrottleConfig,
)
from .role.findings import FindingSink, FindingSinkConfig
from .role.reactor import Reactor, ReactorComponents
from .state_store import DetectorStateStore
from .signals import Classifier, SymptomSeverity
from .signals.aiter_jit import AiterJitConfig
from .signals.budget import BudgetConfig
from .signals.crash import CrashConfig
from .signals.decision_audit import DecisionAuditConfig
from .signals.event import EventConfig
from .signals.gpu_leak import GpuLeakConfig
from .signals.critic_health import CriticHealthConfig
from .signals.external_deps import ExternalDepsConfig
from .signals.kernel_pipeline import KernelPipelineConfig
from .signals.local_health import LocalHealthConfig
from .signals.preflight import (
    AmdahlCeilingConfig,
    ColdStartConfig,
    ModelGpuFitConfig,
)
from .signals.conversation_progress import ConversationProgressConfig
from .signals.phase_budget import PhaseBudgetConfig
from .signals.progress import ProgressConfig
from .signals.repeated_payload import RepeatedPayloadConfig
from .signals.state_integrity import StateIntegrityConfig
from .signals.stall import StallConfig
from .sources.base import DegradeRouter, Source, SourceData
from .sources.local_probe import LocalProbeConfig, LocalProbeSource


log = logging.getLogger(__name__)


@dataclass
class ReactorBundle:
    """Aggregate of reactor + lifecycle handles a caller needs to manage."""

    reactor: Reactor
    components: ReactorComponents
    sink: FindingSink

    async def aclose(self) -> None:
        """Close the RCA engine's provider client.

        The engine only acts on a client it created, so an injected one is
        left to its owner.
        """
        await self.components.rca.aclose()


def _build_local_probe_config(config: Config) -> LocalProbeConfig:
    """Project the agent config onto the local probe's own configuration.

    ``server_process_patterns`` is passed through as both the server subset and
    part of the match list, so a framework an operator adds to that knob is
    matched *and* recognised as a server. Splitting those two decisions across
    separately-maintained lists is what let a configured framework show up as
    ``is_server=False`` and silently disable ``local_server_unreachable``.

    Args:
        config (Config): The discovered agent configuration.

    Returns:
        LocalProbeConfig: Configuration for :class:`LocalProbeSource`.
    """
    # Auto-include the local inference server health endpoint.
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
        "runs/*/*/*/server.log",
        "runs/*/*/*/*/server.log",
    )
    if config.server_log_extra_globs:
        extra_log_globs = (
            *extra_log_globs,
            *(g.strip() for g in config.server_log_extra_globs.split(":") if g.strip()),
        )

    return LocalProbeConfig(
        session_dir=config.session_dir,
        process_patterns=tuple(config.server_process_patterns + config.benchmark_process_patterns),
        server_process_patterns=tuple(config.server_process_patterns),
        health_probe_targets=tuple(probe_targets),
        health_probe_timeout_s=config.health_probe_timeout_s,
        ray_probe_enabled=config.ray_probe_enabled,
        ray_probe_timeout_s=config.ray_probe_timeout_s,
        fd_probe_enabled=config.fd_probe_enabled,
        fd_probe_pid=config.fd_probe_pid,
        decision_audit_enabled=config.decision_audit_enabled,
        decision_audit_max_integrate=config.decision_audit_max_integrate,
        decision_audit_max_oob_attempts=(config.decision_audit_max_oob_attempts),
        preflight_enabled=config.preflight_enabled,
        critic_health_enabled=config.critic_health_enabled,
        max_critic_judge_bundles=config.critic_health_max_judge_bundles,
        extra_server_log_globs=extra_log_globs,
        max_extra_server_logs=config.server_log_max_extra,
        state_integrity_enabled=config.state_integrity_enabled,
        external_deps_enabled=config.external_deps_enabled,
        external_gateway_probe_url=config.external_gateway_probe_url,
    )


def build_reactor_components(
    config: Config,
    *,
    rca: RcaEngine | None = None,
    session_id: str | None = None,
) -> ReactorBundle:
    """Construct everything the reactor needs.

    Wires the source, degrade router, detectors, state store, finding sink,
    and RCA engine into a single bundle.

    Args:
        config (Config): Discovered configuration — typically the result
            of ``Config.discover()``.
        rca (RcaEngine | None): Optional RCA engine override. Defaults to
            an auto-selected engine (Noop unless LLM RCA is enabled).
        session_id (str | None): Override for the FindingSink filename.
            Defaults to ``config.session_dir.name`` so each per-session
            sandbox writes to a stable file.

    Returns:
        ReactorBundle: The assembled reactor plus the components and sink
        the caller must manage.
    """
    # Multi-node guard: the probe only sees its own pod, so it is disabled
    # when config.disable_local_probe is set (ROBUSTNESS_DISABLE_LOCAL_PROBE
    # or --robustness-disable-local-probe). The router still ticks but skips
    # every fetch, returning blind empty data each time.
    if config.disable_local_probe:
        probe_source: Source | None = None
    else:
        probe_source = LocalProbeSource(_build_local_probe_config(config))

    router = DegradeRouter(
        _BlindSource() if probe_source is None else probe_source,
        fail_threshold=config.source_fail_threshold,
        recheck_interval_s=config.source_recheck_interval_s,
    )

    # Disk-backed state store surviving subprocess restarts; built before the classifier.
    state_store: DetectorStateStore | None = (
        DetectorStateStore(session_dir=config.session_dir) if config.state_store_enabled else None
    )

    # Config->SignalConfig map keyed by ``SignalSpec.config_attr``; omitted
    # slots are filled by the classifier from the registry ``config_factory``.
    signal_configs: dict[str, Any] = {
        "stall": StallConfig(
            stall_timeout_s=config.agent_stall_timeout_s,
            severity_high_after_s=config.agent_stall_high_after_s,
        ),
        "crash": CrashConfig(),
        "event": EventConfig(
            idempotency_replay_threshold=config.idempotency_replay_threshold,
        ),
        "local_health": LocalHealthConfig(
            gpu_temp_warn_c=config.gpu_temp_warn_c,
            disk_used_warn_pct=config.disk_used_warn_pct,
            disk_used_crit_pct=config.disk_used_crit_pct,
            shm_used_warn_pct=config.shm_used_warn_pct,
            shm_used_crit_pct=config.shm_used_crit_pct,
            fd_warn_used_pct=config.fd_warn_used_pct,
            fd_crit_used_pct=config.fd_crit_used_pct,
            session_dir=config.session_dir,
        ),
        "gpu_leak": GpuLeakConfig(
            util_mem_pct_threshold=config.gpu_leak_util_mem_pct_threshold,
            free_mb_threshold=config.gpu_leak_free_mb_threshold,
            min_consecutive_ticks=config.gpu_leak_min_consecutive_ticks,
        ),
        "budget": BudgetConfig(
            warn_pct=config.budget_warn_pct,
            imminent_pct=config.budget_imminent_pct,
            min_budget_minutes=config.budget_min_minutes,
            productive_gain_pct=config.budget_productive_gain_pct,
            strategy_drift_pct=config.budget_strategy_drift_pct,
            deadline_warning_minutes=config.budget_deadline_warning_minutes,
            deadline_hard_cutoff_minutes=(config.budget_deadline_hard_cutoff_minutes),
        ),
        "aiter_jit": AiterJitConfig(
            cold_so_count=config.aiter_jit_cold_so_count,
            regression_ratio=config.aiter_jit_regression_ratio,
            stale_build_threshold=config.aiter_jit_stale_build_threshold,
            stale_build_persist_ticks=config.aiter_jit_stale_build_persist_ticks,
        ),
        "progress": ProgressConfig(
            gain_window_ticks=config.progress_gain_window_ticks,
            gain_epsilon_pct=config.progress_gain_epsilon_pct,
            no_levers_min_minutes=config.progress_no_levers_min_minutes,
            no_levers_min_ticks=config.progress_no_levers_min_ticks,
            productive_gain_pct=config.budget_productive_gain_pct,
        ),
        "repeated_payload": RepeatedPayloadConfig(
            streak_threshold=config.repeated_payload_streak_threshold,
            lookback_events=config.repeated_payload_lookback_events,
        ),
        "decision_audit": DecisionAuditConfig(
            min_keep_gain_pct=config.decision_audit_min_keep_gain_pct,
            dispatch_bypass_pre_post_epsilon_pct=(config.decision_audit_dispatch_bypass_epsilon_pct),
        ),
        "model_gpu_fit": ModelGpuFitConfig(
            min_headroom_pct=config.preflight_min_headroom_pct,
            activation_buf_gib=config.preflight_activation_buf_gib,
        ),
        "amdahl_ceiling": AmdahlCeilingConfig(
            single_kernel_speedup=(config.preflight_amdahl_single_kernel_speedup),
            min_e2e_ceiling_pct=config.preflight_amdahl_min_e2e_ceiling_pct,
        ),
        "cold_start": ColdStartConfig(
            cold_so_count=config.preflight_cold_start_so_count,
            cold_start_minutes=config.preflight_cold_start_minutes,
        ),
        "critic_health": CriticHealthConfig(
            min_outage_judges=config.critic_health_min_outage_judges,
            min_unavailable_verdicts=(config.critic_health_min_unavailable_verdicts),
            max_workdir_count=config.critic_health_max_workdir_count,
        ),
        "kernel_pipeline": KernelPipelineConfig(
            pending_count_threshold=(config.kernel_pipeline_pending_count_threshold),
            min_pending_ticks=config.kernel_pipeline_min_pending_ticks,
            min_geak_sigterm_attempts=(config.kernel_pipeline_min_geak_sigterm_attempts),
            min_kernels_with_no_progress=(config.kernel_pipeline_min_kernels_with_no_progress),
        ),
        "state_integrity": StateIntegrityConfig(
            wal_bytes_warn_threshold=config.state_wal_bytes_warn_threshold,
            wal_bytes_critical_threshold=(config.state_wal_bytes_critical_threshold),
            inbox_bloat_warn_bytes=config.state_inbox_bloat_warn_bytes,
            inbox_bloat_critical_bytes=(config.state_inbox_bloat_critical_bytes),
        ),
        "external_deps": ExternalDepsConfig(
            mount_latency_warn_ms=config.external_mount_latency_warn_ms,
            mount_latency_critical_ms=(config.external_mount_latency_critical_ms),
        ),
        "phase_budget": PhaseBudgetConfig(
            warn_used_pct=config.phase_budget_warn_used_pct,
        ),
        "conversation_progress": ConversationProgressConfig(
            enabled=config.conversation_progress_enabled,
        ),
    }
    classifier = Classifier(configs=signal_configs, state_store=state_store)

    ladder = ActionLadder(
        config=ActionLadderConfig(cooldown_ticks=config.cooldown_ticks),
        state_view=(state_store.view("action_ladder") if state_store else None),
    )

    sink_session_id = session_id or config.session_dir.name or "default"
    sink = FindingSink(FindingSinkConfig(session_dir=config.session_dir, session_id=sink_session_id))

    rca_engine: RcaEngine = (
        rca
        if rca is not None
        else _build_rca_engine(
            config,
            state_store=state_store,
        )
    )

    components = ReactorComponents(
        router=router,
        classifier=classifier,
        ladder=ladder,
        policy=PolicyAware(),
        sink=sink,
        rca=rca_engine,
        state_store=state_store,
    )
    return ReactorBundle(
        reactor=Reactor(components),
        components=components,
        sink=sink,
    )


def _llm_credentials_ready(config: Config) -> bool:
    """Whether the discovered provider can authenticate an RCA call.

    The Anthropic side may hold no in-process key at all — a subscription token
    is spent by the Claude CLI, never by this process — so the key pair is the
    wrong question there. Ask the transport instead, which also covers the case
    where the token is present but the CLI that would spend it is not.

    Args:
        config (Config): The configuration carrying the discovered provider
            and credentials.

    Returns:
        bool: True when an RCA call can authenticate.
    """
    if config.llm_provider == "anthropic":
        return llm_config.anthropic_transport_ready()
    return bool(config.llm_base_url and config.llm_api_key)


def _build_rca_engine(
    config: Config,
    *,
    state_store: DetectorStateStore | None = None,
) -> RcaEngine:
    """Choose between Noop and Llm based on config + env override.

    Args:
        config (Config): Configuration carrying LLM RCA enablement,
            credentials, and throttle settings.
        state_store (DetectorStateStore | None): Optional store backing
            the RCA throttle's cross-tick cooldown state.

    Returns:
        RcaEngine: A :class:`LlmRcaEngine` when LLM RCA is enabled and
        credentials are present, otherwise a :class:`NoopRcaEngine`.
    """
    if os.environ.get("ROBUSTNESS_LLM_RCA_DISABLED", "").lower() in {"1", "true", "yes"}:
        log.info("LLM RCA disabled via ROBUSTNESS_LLM_RCA_DISABLED env override")
        return NoopRcaEngine()
    if config.llm_rca_enabled is False:
        return NoopRcaEngine()
    if not _llm_credentials_ready(config):
        if config.llm_rca_enabled is True:
            log.warning("llm_rca_enabled=True but no usable LLM credential; using NoopRcaEngine")
        return NoopRcaEngine()

    severity_min = _parse_severity(config.llm_rca_severity_min)
    throttle = RcaThrottle(
        RcaThrottleConfig(
            severity_min=severity_min,
            cooldown_seconds=config.llm_rca_cooldown_s,
            max_calls_per_tick=config.llm_rca_max_calls_per_tick,
        ),
        state_view=(state_store.view("rca_throttle") if state_store else None),
    )
    log.info(
        "LLM RCA enabled: model=%s severity_min=%s cooldown=%.1fs max_per_tick=%d",
        config.llm_model,
        severity_min.value,
        config.llm_rca_cooldown_s,
        config.llm_rca_max_calls_per_tick,
    )
    engine_cls = AnthropicRcaEngine if config.llm_provider == "anthropic" else LlmRcaEngine
    return engine_cls(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        timeout_s=config.llm_rca_timeout_s,
        max_chars=config.llm_rca_max_chars,
        throttle=throttle,
    )


def _parse_severity(value: str) -> SymptomSeverity:
    """Map a severity string to a :class:`SymptomSeverity`.

    Args:
        value (str): A severity label such as ``"low"``, ``"medium"``,
            or ``"high"`` (case-insensitive; synonyms accepted).

    Returns:
        SymptomSeverity: The matching severity, defaulting to ``HIGH``
        for unrecognised values.
    """
    normalized = (value or "").strip().lower()
    if normalized in {"low", "info", "observe"}:
        return SymptomSeverity.LOW
    if normalized in {"medium", "warn", "warning"}:
        return SymptomSeverity.MEDIUM
    return SymptomSeverity.HIGH


@dataclass
class _BlindSource:
    """Permanent no-op source used when the local probe is disabled.

    Returns empty ``SourceData(local_processes_known=False)`` every tick so
    probe-derived rules stay quiet rather than misfiring.  It never raises,
    which means the DegradeRouter stays HEALTHY and does not WARN about
    degradation on a deliberately disabled probe.
    """

    name: str = "local-probe-disabled"

    async def fetch(self, ctx: Any) -> SourceData:  # noqa: ARG002 - protocol
        """Return empty source data with process-visibility marked unknown.

        Args:
            ctx: Fetch context; unused because this source never collects data.

        Returns:
            A :class:`SourceData` with no signals and
            ``local_processes_known=False``.
        """
        return SourceData(local_processes_known=False)


__all__ = [
    "ReactorBundle",
    "build_reactor_components",
]
