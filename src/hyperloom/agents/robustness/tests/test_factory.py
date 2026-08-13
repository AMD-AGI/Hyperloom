# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for the M1 factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.robustness import config as config_module
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
        assert bundle.server_client is None
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_config_map_covers_all_registry_entries(tmp_path: Path):
    """The factory-built classifier must resolve a config for every registry
    slot: entries the factory omits (e.g. cluster_fault) fall back to the
    registry default, so nothing is left unconfigured."""
    from hyperloom.agents.robustness.signals.classifier import _SIGNAL_REGISTRY

    expected_slots = {spec.config_attr for spec in _SIGNAL_REGISTRY if spec.config_attr}

    config = Config(session_dir=tmp_path, robustness_server_url="")
    bundle = build_reactor_components(config)
    try:
        resolved = bundle.components.classifier.signal_configs
        assert set(resolved) == expected_slots
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
        # Assert the bundle wires the server_client when a URL is configured.
        assert bundle.server_client is not None
        assert bundle.server_client.base_url == "http://example.invalid:8000"
    finally:
        await bundle.aclose()


# LLM RCA wiring


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
async def test_config_discover_normalizes_retired_deepseek_env(monkeypatch, tmp_path: Path):
    """Standalone robustness runs never reach CLI preflight, so it normalizes too.

    Without this the legacy sandbox would resolve no credentials and RCA would
    silently degrade to a no-op engine.
    """
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-token")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("_".join(("SAFE", "API", "KEY")), raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("ROBUSTNESS_LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    # The OpenAI side is filled too, and it is checked first.
    assert config.llm_provider == "openai"
    assert config.llm_base_url == "https://api.deepseek.com/v1"
    assert config.llm_api_key == "deepseek-token"
    assert config.llm_model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_config_discover_uses_dual_protocol_gateway_anthropic_side(monkeypatch, tmp_path: Path):
    """RCA reads the Anthropic side of a dual-protocol gateway like any other."""
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "deepseek-token")
    monkeypatch.setenv("CLAUDE_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("SAFE", "API", "KEY")), raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    assert config.llm_provider == "anthropic"
    assert config.llm_base_url == "https://api.deepseek.com/anthropic"
    assert config.llm_api_key == "deepseek-token"
    assert config.llm_model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_config_discover_selects_anthropic_for_a_subscription_token(monkeypatch, tmp_path: Path):
    """A subscription token selects the Anthropic side without becoming a key.

    Copying it into llm_api_key would hand an API-credits slot a credential the
    CLI must resolve itself, so discovery reports the provider and nothing else.
    """
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-token")
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("DEEPSEEK", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("SAFE", "API", "KEY")), raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    assert config.llm_provider == "anthropic"
    assert config.llm_api_key == ""
    assert "sk-ant-oat01-token" not in (config.llm_api_key, config.llm_base_url)


@pytest.mark.asyncio
async def test_config_discover_anthropic_model_follows_claude_model(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-token")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-6")
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("DEEPSEEK", "API", "KEY")), raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    assert config.llm_provider == "anthropic"
    assert config.llm_model == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_config_discover_openai_model_follows_codex_model(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-token")
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.5")
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("_".join(("DEEPSEEK", "API", "KEY")), raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-5.5"


@pytest.mark.asyncio
async def test_config_discover_does_not_treat_gateway_key_as_official_openai(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_GATEWAY_KEY", "gateway-token")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("_".join(("DEEPSEEK", "API", "KEY")), raising=False)
    monkeypatch.setattr("hyperloom.agents.robustness.config._probe_robustness_server", lambda: _async_value(""))

    config = await Config.discover()

    assert config.llm_base_url == ""
    assert config.llm_api_key == ""
    assert config.llm_provider == "openai"


async def _async_value(value):
    return value


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
async def test_factory_uses_anthropic_engine_for_provider(tmp_path: Path, monkeypatch):
    from hyperloom.agents.robustness.decision.rca_engine import AnthropicRcaEngine

    monkeypatch.setattr("hyperloom.common.llm_config.anthropic_transport_ready", lambda *_a, **_kw: True)
    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="https://api.deepseek.com/anthropic",
        llm_api_key="secret",
        llm_provider="anthropic",
        llm_model="deepseek-v4-pro",
    )
    bundle = build_reactor_components(config)
    try:
        engine = bundle.components.rca
        assert isinstance(engine, AnthropicRcaEngine)
        assert engine.model == "deepseek-v4-pro"
        await engine.aclose()
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_uses_anthropic_engine_for_a_subscription_token_host(tmp_path: Path, monkeypatch):
    """An oauth-only host resolves no key in-process, and must still get RCA.

    Driven through Config.discover so the empty base_url/api_key pair is the
    one the token actually produces, rather than a hand-written stand-in.
    """
    from hyperloom.agents.robustness.decision.rca_engine import AnthropicRcaEngine

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake")
    monkeypatch.setattr("hyperloom.common.llm_config.anthropic_transport_ready", lambda *_a, **_kw: True)

    base_url, api_key, provider = config_module._discover_llm_credentials()
    assert (base_url, api_key, provider) == ("", "", "anthropic")

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_provider=provider,
        llm_model="claude-opus-5",
    )
    bundle = build_reactor_components(config)
    try:
        engine = bundle.components.rca
        assert isinstance(engine, AnthropicRcaEngine)
        await engine.aclose()
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_falls_back_to_noop_when_the_anthropic_transport_is_unusable(
    tmp_path: Path, monkeypatch
):
    """A subscription token with no claude CLI, or no Anthropic credential at
    all, must degrade at build time instead of failing on every tick."""
    from hyperloom.agents.robustness.decision.rca_engine import NoopRcaEngine

    monkeypatch.setattr("hyperloom.common.llm_config.anthropic_transport_ready", lambda *_a, **_kw: False)
    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="",
        llm_api_key="",
        llm_provider="anthropic",
        llm_model="claude-opus-5",
    )
    bundle = build_reactor_components(config)
    try:
        assert isinstance(bundle.components.rca, NoopRcaEngine)
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_factory_still_noops_when_the_openai_side_has_no_key(tmp_path: Path):
    from hyperloom.agents.robustness.decision.rca_engine import NoopRcaEngine

    config = Config(
        session_dir=tmp_path,
        robustness_server_url="",
        llm_base_url="http://chat-server.invalid/v1",
        llm_api_key="",
        llm_provider="openai",
    )
    bundle = build_reactor_components(config)
    try:
        assert isinstance(bundle.components.rca, NoopRcaEngine)
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


# Multi-node policy: disable_local_probe + cluster fan-out wiring


@pytest.mark.asyncio
async def test_factory_uses_quiet_fallback_when_local_probe_disabled(tmp_path: Path):
    """``disable_local_probe`` swaps the LocalProbe for a quiet stub that never yields high-severity local symptoms."""
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
        assert isinstance(fallback, LocalProbeSource)
        targets = fallback._config.health_probe_targets  # type: ignore[attr-defined]
        assert config.inference_server_health_url not in targets
    finally:
        await bundle.aclose()


@pytest.mark.asyncio
async def test_a_framework_added_to_the_config_knob_is_recognised_as_a_server(tmp_path: Path, monkeypatch):
    """The documented knob decides ``is_server``, not a second copy of the list.

    A framework named only in ``server_process_patterns`` used to be matched as
    a process yet flagged ``is_server=False``, which silently disabled
    ``local_server_unreachable`` for the very deployment that configured it.
    """
    import subprocess

    from hyperloom.agents.robustness.sources import local_probe

    config = Config(session_dir=tmp_path)
    config.server_process_patterns.append("tinyserve.entrypoint")
    bundle = build_reactor_components(config)
    try:
        probe_cfg = bundle.components.router._fallback._config  # type: ignore[attr-defined]
        monkeypatch.setattr(
            local_probe.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 0, "  9 1048576 python -m tinyserve.entrypoint --port 8888\n", ""
            ),
        )

        found = local_probe._sample_processes(
            probe_cfg.process_patterns,
            probe_cfg.server_process_patterns,
        )

        assert found == [
            {
                "pid": 9,
                "rss_mb": 1024.0,
                "cmd": "python -m tinyserve.entrypoint --port 8888",
                "is_server": True,
            }
        ]
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
