# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Smoke tests for the M1 factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.robustness.config import Config
from hyperloom.agents.robustness.factory import build_reactor_components
from hyperloom.agents.robustness.role.envelope import IntentType
from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)


@pytest.mark.asyncio
async def test_build_reactor_components_local_only_mode_runs_a_tick(tmp_path: Path):
    config = Config(session_dir=tmp_path)
    config.robustness_server_url = ""
    bundle = build_reactor_components(config)
    try:
        ctx = ReactorContext(
            tick_index=0,
            shared_state=SharedStateSnapshot(session_id="sess-1", crash_count=2),
            inbox=[],
            now_unix=1000.0,
        )
        intents = await bundle.reactor.tick(ctx)
        assert intents
        assert any(i.type is IntentType.ALERT for i in intents)
        # No server URL means primary source degrades after fail_threshold.
        assert bundle.server_client is None
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_config_map_covers_all_registry_entries(tmp_path: Path):
    """The factory-built classifier must resolve a config for every registry
    slot: entries the factory omits (e.g. cluster_fault) fall back to the
    registry default, so nothing is left unconfigured."""
    from hyperloom.agents.robustness.signals import signal_registry_config_attrs

    config = Config(session_dir=tmp_path, robustness_server_url="")
    bundle = build_reactor_components(config)
    try:
        resolved = bundle.components.classifier.signal_configs
        assert set(resolved) == set(signal_registry_config_attrs())
        # Every distinct registry slot resolved to an instance (no None).
        assert all(cfg is not None for cfg in resolved.values())
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_build_reactor_components_uses_server_url_when_set(tmp_path: Path):
    config = Config(
        session_dir=tmp_path,
        robustness_server_url="http://example.invalid:8000",
    )
    bundle = build_reactor_components(config)
    try:
        # We cannot reach the URL; primary source should fail and the
        # fallback eventually serve. We just assert the bundle wires the
        # server_client when a URL is configured.
        assert bundle.server_client is not None
        assert bundle.server_client.base_url == "http://example.invalid:8000"
    finally:
        await bundle.aclose()


# ---------------------------------------------------------------------------
# M1.5 LLM RCA wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_uses_noop_engine_when_credentials_missing(tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import NoopRcaEngine

    config = Config(session_dir=tmp_path, robustness_server_url="")
    bundle = build_reactor_components(config)
    try:
        assert isinstance(bundle.components.rca, NoopRcaEngine)
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_uses_llm_engine_when_credentials_present(tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import LlmRcaEngine

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="http://chat-server.invalid/v1",
        llm_api_key="secret",
    )
    bundle = build_reactor_components(config)
    try:
        engine = bundle.components.rca
        assert isinstance(engine, LlmRcaEngine)
        assert engine.model == config.llm_model
        await engine.aclose()
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_respects_explicit_disable(tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import NoopRcaEngine

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="http://chat-server.invalid/v1",
        llm_api_key="secret",
        llm_rca_enabled=False,
    )
    bundle = build_reactor_components(config)
    try:
        assert isinstance(bundle.components.rca, NoopRcaEngine)
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_respects_env_disable(monkeypatch, tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import NoopRcaEngine

    monkeypatch.setenv("ROBUSTNESS_LLM_RCA_DISABLED", "1")
    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="http://chat-server.invalid/v1",
        llm_api_key="secret",
    )
    bundle = build_reactor_components(config)
    try:
        assert isinstance(bundle.components.rca, NoopRcaEngine)
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_propagates_severity_min_config(tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import LlmRcaEngine
    from hyperloom.agents.robustness.signals import SymptomSeverity

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="http://chat-server.invalid/v1",
        llm_api_key="secret",
        llm_rca_severity_min="medium",
    )
    bundle = build_reactor_components(config)
    try:
        engine = bundle.components.rca
        assert isinstance(engine, LlmRcaEngine)
        assert engine.throttle is not None
        assert engine.throttle.config.severity_min is SymptomSeverity.MEDIUM
        await engine.aclose()
    finally:
        await bundle.aclose()


# ---------------------------------------------------------------------------
# M2 multi-node policy: disable_local_probe + cluster fan-out wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_uses_quiet_fallback_when_local_probe_disabled(tmp_path: Path):
    """``disable_local_probe`` swaps the LocalProbe for a quiet stub that never yields high-severity local symptoms (Ray-worker policy)."""
    from hyperloom.agents.robustness.factory import _QuietFallback
    from hyperloom.agents.robustness.sources.local_probe import LocalProbeSource

    config = Config(session_dir=tmp_path, disable_local_probe=True)
    bundle = build_reactor_components(config)
    try:
        router = bundle.components.router
        fallback = router._fallback  # type: ignore[attr-defined]
        assert isinstance(fallback, _QuietFallback)
        assert not isinstance(fallback, LocalProbeSource)
        data = await fallback.fetch(None)
        assert data.local_processes == []
        assert data.local_server_health == []
        assert data.degraded_reason and "local-probe disabled" in data.degraded_reason
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_default_keeps_local_probe_fallback(tmp_path: Path):
    from hyperloom.agents.robustness.sources.local_probe import LocalProbeSource

    config = Config(session_dir=tmp_path)
    bundle = build_reactor_components(config)
    try:
        router = bundle.components.router
        fallback = router._fallback  # type: ignore[attr-defined]
        assert isinstance(fallback, LocalProbeSource)
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_default_auto_probes_inference_server(tmp_path: Path):
    """Default config appends the inference-server health URL to local probes."""
    from hyperloom.agents.robustness.sources.local_probe import LocalProbeSource

    config = Config(session_dir=tmp_path)
    bundle = build_reactor_components(config)
    try:
        fallback = bundle.components.router._fallback  # type: ignore[attr-defined]
        assert isinstance(fallback, LocalProbeSource)
        targets = fallback._config.health_probe_targets  # type: ignore[attr-defined]
        assert config.inference_server_health_url in targets
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_scriptable_skips_inference_server_probe(tmp_path: Path):
    """``auto_probe_inference_server=False`` (scriptable/server-less workloads)
    drops the 8888/health target while keeping the LocalProbe (gpu/disk/fd)."""
    from hyperloom.agents.robustness.sources.local_probe import LocalProbeSource

    config = Config(session_dir=tmp_path, auto_probe_inference_server=False)
    bundle = build_reactor_components(config)
    try:
        fallback = bundle.components.router._fallback  # type: ignore[attr-defined]
        # LocalProbe is retained — only the server health probe is skipped.
        assert isinstance(fallback, LocalProbeSource)
        targets = fallback._config.health_probe_targets  # type: ignore[attr-defined]
        assert config.inference_server_health_url not in targets
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_forwards_multi_node_options_to_server_source(tmp_path: Path):
    """``enable_cluster_pod_metrics`` / ``workload_uid`` reach the server source."""

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="http://example.invalid:8000",
        enable_cluster_pod_metrics=True,
        pod_metrics_categories=("gpu", "memory"),
        workload_uid="wl-42",
    )
    bundle = build_reactor_components(config)
    try:
        router = bundle.components.router
        primary = router._primary  # type: ignore[attr-defined]
        assert primary._enable_cluster_pod_metrics is True  # type: ignore[attr-defined]
        assert primary._pod_metrics_categories == ("gpu", "memory")  # type: ignore[attr-defined]
        assert primary._workload_uid == "wl-42"  # type: ignore[attr-defined]
    finally:
        await bundle.aclose()
