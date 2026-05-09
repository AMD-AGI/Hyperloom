"""High-level builders that turn a :class:`Config` into a running reactor.

Two entry points:

* :func:`build_reactor_components` returns the M1
  :class:`~robustness_agent.role.reactor.ReactorComponents` plus the
  underlying :class:`~robustness_agent.findings.sink.FindingSink` and
  :class:`~robustness_agent.sources.server_client.RobustnessServerClient`
  so callers can manage their lifecycles (e.g. ``await client.aclose()``
  on shutdown).
* :func:`build_backend` wraps :func:`build_reactor_components` and
  hands back a ready-to-register
  :class:`~robustness_agent.role.backend_adapter.RobustnessAgentBackend`.

The factories never block on remote services — :class:`Config.discover`
already probed reachability and recorded the URLs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

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
from .findings.sink import FindingSink, FindingSinkConfig
from .role.backend_adapter import RobustnessAgentBackend
from .role.reactor import Reactor, ReactorComponents
from .signals import Classifier, SymptomSeverity
from .signals.crash import CrashConfig
from .signals.event import EventConfig
from .signals.health import HealthConfig
from .signals.local_health import LocalHealthConfig
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

    fallback = LocalProbeSource(
        LocalProbeConfig(
            session_dir=config.session_dir,
            process_patterns=tuple(
                config.server_process_patterns + config.benchmark_process_patterns
            ),
            health_probe_targets=tuple(config.health_probe_targets),
            health_probe_timeout_s=config.health_probe_timeout_s,
        )
    )

    router = DegradeRouter(
        primary,
        fallback,
        fail_threshold=config.source_fail_threshold,
        recheck_interval_s=config.source_recheck_interval_s,
    )

    classifier = Classifier(
        stall_config=StallConfig(
            stall_timeout_s=config.agent_stall_timeout_s,
        ),
        crash_config=CrashConfig(),
        event_config=EventConfig(),
        health_config=HealthConfig(),
        local_health_config=LocalHealthConfig(
            gpu_temp_warn_c=config.gpu_temp_warn_c,
        ),
    )

    ladder = ActionLadder(
        config=ActionLadderConfig(cooldown_ticks=config.cooldown_ticks),
    )

    sink_session_id = session_id or config.session_dir.name or "default"
    sink = FindingSink(
        FindingSinkConfig(session_dir=config.session_dir, session_id=sink_session_id)
    )

    rca_engine: RcaEngine = rca if rca is not None else _build_rca_engine(config)

    components = ReactorComponents(
        router=router,
        classifier=classifier,
        ladder=ladder,
        policy=PolicyAware(),
        sink=sink,
        rca=rca_engine,
    )
    return ReactorBundle(
        reactor=Reactor(components),
        components=components,
        server_client=server_client,
        sink=sink,
    )


def build_backend(
    config: Config,
    *,
    rca: RcaEngine | None = None,
    session_id: str | None = None,
) -> tuple[RobustnessAgentBackend, ReactorBundle]:
    """Return a Backend instance + the bundle to manage its lifecycle."""
    bundle = build_reactor_components(config, rca=rca, session_id=session_id)
    backend = RobustnessAgentBackend(reactor=bundle.reactor)
    return backend, bundle


def _build_rca_engine(config: Config) -> RcaEngine:
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
        )
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
    "build_backend",
    "build_reactor_components",
]
