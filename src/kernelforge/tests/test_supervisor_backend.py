# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for free-form Supervisor persistence and backend routing."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
)
from kernelforge.agent_backends.codex import resolve_codex_gateway
from kernelforge.config import Config
from kernelforge.orchestrator.supervisor import (
    make_supervisor_fn,
)

_GATEWAY_ENV = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "SAFE_API_KEY",
    "FORGE_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
)


def test_resolve_codex_gateway_uses_the_openai_line(monkeypatch):
    """Codex speaks the OpenAI-compatible protocol, so only OPENAI_* applies."""
    for k in _GATEWAY_ENV:
        monkeypatch.delenv(k, raising=False)

    # Nothing configured -> falsy (best-effort skip downstream).
    assert not resolve_codex_gateway().is_complete()

    # A complete Anthropic line belongs to Claude and does not enable this one.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example/api/v1/llm-proxy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: alice")
    assert not resolve_codex_gateway().is_complete()

    # Its own pair resolves verbatim, with only its own headers.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://direct/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "user: bob\nx-foo: bar")
    gw = resolve_codex_gateway()
    assert gw.base_url == "https://direct/v1"
    assert gw.key_env == "OPENAI_API_KEY"
    assert gw.headers == {"user": "bob", "x-foo": "bar"}


def test_resolve_codex_gateway_rejects_retired_keys(monkeypatch):
    """Neither SAFE_API_KEY nor FORGE_API_KEY configures the supervisor.

    Both used to satisfy the lookup, so a deployment carrying only one of them
    looked healthy while authenticating with a credential nobody configured.
    """
    for k in _GATEWAY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://direct/v1")
    monkeypatch.setenv("SAFE_API_KEY", "safe")
    monkeypatch.setenv("FORGE_API_KEY", "forge")
    assert not resolve_codex_gateway().is_complete()

    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    assert resolve_codex_gateway().key_env == "OPENAI_API_KEY"


@pytest.mark.parametrize("override", [None, {}, {"base_url": "https://partial/v1"}])
def test_empty_gateway_override_defers_to_the_environment(monkeypatch, override):
    """An override that resolves to nothing must not shadow the environment.

    LlmGateway has no truthiness, so `self.gateway or _resolve_gateway()` treated
    an empty override as configured and stopped reading OPENAI_*.
    """
    from kernelforge.agent_backends.base import AgentRuntimeConfig
    from kernelforge.agent_backends.codex import CodexBackend

    for k in _GATEWAY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://from-env/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    options = {} if override is None else {"gateway": override}
    backend = CodexBackend(
        codex_bin="/bin/true",
        runtime=AgentRuntimeConfig(provider="codex", model="gpt-5.6", options=options),
        bypass_sandbox=True,
    )
    gateway = backend._effective_gateway()
    assert gateway.base_url == "https://from-env/v1"
    assert gateway.key_env == "OPENAI_API_KEY"


def test_complete_gateway_override_wins(monkeypatch):
    from kernelforge.agent_backends.base import AgentRuntimeConfig
    from kernelforge.agent_backends.codex import CodexBackend

    monkeypatch.setenv("OPENAI_BASE_URL", "https://from-env/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    backend = CodexBackend(
        codex_bin="/bin/true",
        runtime=AgentRuntimeConfig(
            provider="codex",
            model="gpt-5.6",
            options={"gateway": {"base_url": "https://override/v1", "key_env": "OPENAI_API_KEY"}},
        ),
        bypass_sandbox=True,
    )
    assert backend._effective_gateway().base_url == "https://override/v1"


def test_provider_overrides_forwards_every_header(monkeypatch):
    """All of the provider's headers reach codex, not just the gateway NTID.

    An APIM subscription key is as mandatory as ``user``; forwarding only the
    latter silently dropped it and the gateway answered 401.
    """
    from kernelforge.agent_backends.codex import _provider_overrides

    for k in _GATEWAY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://direct/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "user: alice\nOcp-Apim-Subscription-Key: sub123",
    )

    overrides = _provider_overrides(resolve_codex_gateway())

    assert 'model_providers.forge.env_key="OPENAI_API_KEY"' in overrides
    assert 'model_providers.forge.http_headers.user="alice"' in overrides
    assert 'model_providers.forge.http_headers.Ocp-Apim-Subscription-Key="sub123"' in overrides
    # The secret itself is never copied into the config, only its variable name.
    assert not any("openai" in o for o in overrides)


def test_supervisor_backend_initialization_is_best_effort(
    tmp_path,
    monkeypatch,
) -> None:
    """Return an empty reply instead of failing loop setup without a provider."""
    calls = 0

    def unavailable_factory(runtime, **_kwargs):
        """Simulate unavailable primary and fallback providers."""
        nonlocal calls
        calls += 1
        raise AgentProviderUnavailableError("codex unavailable; fallback claude unavailable")

    monkeypatch.setattr(
        "kernelforge.agent_backends.registry.create_registered_backend",
        unavailable_factory,
    )
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-5.3-codex",
        agent_precheck=False,
        agent_fallback_provider="claude",
    )

    supervisor_fn = make_supervisor_fn(
        program_md="Optimize the kernel.",
        backend="codex",
        config=config,
    )

    assert calls == 0
    reply = asyncio.run(
        supervisor_fn(
            digest="iteration stalled",
            reason="plateau",
            workspace=str(tmp_path),
        )
    )
    assert reply == ""
    assert calls == 1
    assert getattr(supervisor_fn, "backend_name") == "codex"


def test_supervisor_api_failure_is_not_parsed_or_repaired(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    class ApiFailureBackend:
        name = "codex"

        async def run(self, spec, usage=None):
            nonlocal calls
            calls += 1
            return AgentRunResult(
                text="SDK error text is not a Supervisor Ruling.",
                end_reason="api_error",
                stderr_tail="gateway unavailable",
            )

    def fake_factory(runtime, **_kwargs):
        backend = ApiFailureBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(
        "kernelforge.agent_backends.registry.create_registered_backend",
        fake_factory,
    )
    supervisor_fn = make_supervisor_fn(
        config=Config(
            workspace=str(tmp_path),
            agent_backend="codex",
            agent_model="gpt-codex-test",
            agent_precheck=False,
        )
    )

    reply = asyncio.run(
        supervisor_fn(
            digest="iteration stalled",
            reason="plateau",
            workspace=str(tmp_path),
            iteration=4,
        )
    )

    assert reply == ""
    assert calls == 1
    persisted = (tmp_path / "forge_experiments" / "supervisor" / "intervention_iter_004.md").read_text()
    assert "SDK error text is not a Supervisor Ruling" not in persisted


def test_supervisor_provider_switch_clears_backend_specific_runtime(
    tmp_path,
    monkeypatch,
):
    """Do not pass a failed implementer's executable or options into its fallback."""
    captured = {}

    class FakeClaudeBackend:
        """Expose the resolved fallback runtime."""

        name = "claude"

        async def run(
            self,
            spec: AgentRunSpec,
            usage=None,
        ) -> AgentRunResult:
            """Return one empty best-effort Supervisor response."""
            return AgentRunResult()

    def fake_factory(runtime, **_kwargs):
        """Capture the runtime selected for the Supervisor."""
        captured["runtime"] = runtime
        backend = FakeClaudeBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(
        "kernelforge.agent_backends.registry.create_registered_backend",
        fake_factory,
    )
    config = Config(
        agent_backend="codex",
        agent_cli="/opt/bin/codex",
        agent_precheck=False,
        agent_fallback_provider="claude",
        agent_options={"codex_only": True},
    )

    supervisor_fn = make_supervisor_fn(backend="claude", config=config)
    asyncio.run(
        supervisor_fn(
            digest="iteration stalled",
            reason="plateau",
            workspace=str(tmp_path),
        )
    )

    runtime = captured["runtime"]
    assert runtime.provider == "claude"
    assert runtime.executable == ""
    assert runtime.options == {}


def test_codex_supervisor_uses_shared_backend_config_and_usage(
    tmp_path,
    monkeypatch,
):
    """Pass model, safety, timeout, effort, and usage through AgentBackend."""
    captured: dict[str, object] = {}

    class FakeCodexBackend:
        """Capture one normalized Supervisor run."""

        name = "codex"
        capabilities = AgentCapabilities()

        async def run(
            self,
            spec: AgentRunSpec,
            usage=None,
        ) -> AgentRunResult:
            """Record the spec and usage accumulator."""
            captured["spec"] = spec.resolved(self.runtime)
            captured["usage"] = usage
            return AgentRunResult(
                text=(
                    "# Current ruling\n\n"
                    "The fused merge remains untested. Revisit it with "
                    "race-free shared-memory staging."
                )
            )

    def fake_factory(runtime, **kwargs):
        """Capture one registered runtime and return the fake backend."""
        captured["runtime"] = runtime
        captured["factory_kwargs"] = kwargs
        backend = FakeCodexBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(
        "kernelforge.agent_backends.registry.create_registered_backend",
        fake_factory,
    )
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-supervisor-test",
        agent_sandbox_mode="bypass",
        agent_timeout_sec=77,
        agent_reasoning_effort="medium",
        agent_precheck=False,
        agent_fallback_provider="",
    )
    usage = object()
    supervisor_fn = make_supervisor_fn(
        program_md="Optimize the kernel.",
        backend="codex",
        config=config,
        usage=usage,
    )

    reply = asyncio.run(
        supervisor_fn(
            digest="iteration stalled",
            reason="plateau",
            workspace=str(tmp_path),
        )
    )

    spec = captured["spec"]
    runtime = captured["runtime"]
    assert isinstance(spec, AgentRunSpec)
    assert runtime.provider == "codex"
    assert runtime.sandbox_mode == "bypass"
    assert captured["factory_kwargs"] == {}
    assert captured["usage"] is usage
    assert spec.model == "gpt-codex-supervisor-test"
    assert spec.timeout_sec == 77
    assert spec.reasoning_effort == "max"
    assert spec.writable is False
    assert spec.protected_globs == ["*"]
    assert spec.provider_options == {}
    assert spec.tool_policy.read is True
    assert spec.tool_policy.write is False
    assert reply.startswith("# Current ruling")
    assert "race-free shared-memory staging" in reply
    assert getattr(supervisor_fn, "backend_name") == "codex"
