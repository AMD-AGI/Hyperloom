"""Tests for the pluggable Agent provider registry."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import kernelforge.agent_backends.codex as codex_backend
import kernelforge.agent_backends.registry as registry
from kernelforge.agent_backends import (
    AgentCapabilities,
    AgentProvider,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    create_registered_backend,
    get_agent_provider,
    list_agent_providers,
    register_agent_provider,
    resolve_agent_runtime,
)
from kernelforge.config import Config


@pytest.fixture(autouse=True)
def isolated_provider_registry(monkeypatch):
    """Give every test in this module its own copy of the provider registry.

    ``register_agent_provider`` writes into module-level state that outlives
    the test that called it, and the registry offers no way to unregister. Each
    fake registered below would therefore stay visible to every later test in
    the same worker process, which is how these tests came to depend on the
    order xdist happened to shard them in. Discovery runs first so the snapshot
    already holds the built-ins and any installed plugin; the module globals
    are then rebound to copies that monkeypatch drops during teardown.
    """
    registry.discover_agent_providers()
    monkeypatch.setattr(registry, "_providers", dict(registry._providers))
    monkeypatch.setattr(registry, "_plugin_errors", dict(registry._plugin_errors))


@pytest.fixture
def only_registered_providers(isolated_provider_registry, monkeypatch):
    """Empty the registry so a test's own registration order is the only order.

    ``select_default_agent_provider`` falls back to registration order when no
    available provider claims the model, so any provider the environment
    happens to make available wins that fallback. A test measuring the order it
    registers itself must therefore not inherit the built-ins: the answer has
    to be the same whether or not the optional claude/codex SDKs are installed.
    """
    monkeypatch.setattr(registry, "_providers", {})


@dataclass
class _FakeBackend:
    """Provide the minimum backend behavior required by registry tests."""

    name: str
    unavailable: bool = False

    def preflight(self) -> None:
        """Fail preflight when the test requests an unavailable provider."""
        if self.unavailable:
            raise AgentProviderUnavailableError(self.name)

    async def run(self, spec, usage=None) -> AgentRunResult:
        """Return a deterministic result for protocol compatibility."""
        return AgentRunResult(text=spec.user_prompt)


def _register_fake(
    name: str,
    *,
    unavailable: bool = False,
    model: str = "fake-model",
) -> AgentProvider:
    """Register and return one deterministic fake provider."""

    def factory(runtime):
        """Construct one fake backend from generic runtime config."""
        return _FakeBackend(name=runtime.provider, unavailable=unavailable)

    provider = AgentProvider(
        name=name,
        factory=factory,
        default_model=model,
        capabilities=AgentCapabilities(resumable=True),
    )
    register_agent_provider(provider)
    return provider


def _register_owning_fake(
    name: str,
    *,
    owns_prefix: str,
    unavailable: bool = False,
) -> AgentProvider:
    """Register a fake provider that claims one model-name prefix."""

    def factory(runtime):
        """Construct one fake backend from generic runtime config."""
        return _FakeBackend(name=runtime.provider, unavailable=unavailable)

    provider = AgentProvider(
        name=name,
        factory=factory,
        default_model=f"{name}-model",
        availability=(lambda: False) if unavailable else (lambda: True),
        owns_model=lambda model, prefix=owns_prefix: model.strip().lower().startswith(prefix),
    )
    register_agent_provider(provider)
    return provider


def test_builtin_providers_are_registered() -> None:
    """Expose built-in providers through the same public registry API."""
    assert {"claude", "codex"}.issubset(list_agent_providers())
    assert get_agent_provider("codex").capabilities.native_subagents


def test_builtin_providers_declare_the_session_environment_they_apply() -> None:
    """Both built-ins apply AgentRunSpec.env to the session they spawn.

    Claude hands it to the SDK as ClaudeAgentOptions.env and Codex merges it into
    the app server's child environment. The declaration is what the Implementer
    lane path reads before it agrees to run several sessions at once, because a
    provider that dropped the overlay would run them all out of one build cache.
    """
    assert get_agent_provider("claude").capabilities.session_env
    assert get_agent_provider("codex").capabilities.session_env


def test_only_a_hook_running_provider_declares_stop_hooks() -> None:
    """stop_hooks gates AgentRunSpec.hooks as a whole, and Codex runs none of it.

    Claude translates the PreToolUse, PostToolUse and Stop groups through one
    path keyed on ``spec.hooks is not None``; Codex has no equivalent, so a
    session it runs carries no protection hook however the caller builds one.
    """
    assert get_agent_provider("claude").capabilities.stop_hooks
    assert not get_agent_provider("codex").capabilities.stop_hooks


def test_builtin_model_ownership_predicates() -> None:
    """Recognize each built-in provider's model family without env coupling."""
    claude = get_agent_provider("claude")
    codex = get_agent_provider("codex")
    assert claude.default_model == "claude-opus-5"
    assert claude.fallback_model == "claude-opus-4-8"
    assert codex.default_model == "gpt-5.6"
    assert codex.fallback_model == "gpt-5.5"
    assert claude.owns_model("claude-opus-5")
    assert not claude.owns_model("gpt-5.6")
    assert codex.owns_model("gpt-5.6")
    assert codex.owns_model("o3-mini")
    assert codex.owns_model("internal-codex-preview")
    assert not codex.owns_model("olmo-7b")
    assert not codex.owns_model("orca-2")
    assert not codex.owns_model("openchat-3.5")
    assert not codex.owns_model("claude-opus-5")
    assert not codex.owns_model("")
    assert resolve_agent_runtime("claude").fallback_model == "claude-opus-4-8"
    assert (
        resolve_agent_runtime(
            "claude",
            model="claude-opus-4-8",
        ).fallback_model
        == ""
    )
    assert resolve_agent_runtime("codex").fallback_model == "gpt-5.5"


def test_default_runtime_uses_high_reasoning_effort() -> None:
    config = Config()
    assert config.agent_reasoning_effort == "high"


def test_provider_probe_falls_back_to_supported_model() -> None:
    attempted_models = []

    class Backend(_FakeBackend):
        def probe(self, *, cwd, usage=None):
            del cwd, usage
            attempted_models.append(self.runtime.model)
            if self.runtime.model == "future-model":
                raise AgentProviderUnavailableError("model not served")
            return AgentRunResult(text="OK")

    def factory(runtime):
        backend = Backend(name=runtime.provider)
        backend.runtime = runtime
        return backend

    register_agent_provider(
        AgentProvider(
            name="modelprobe",
            factory=factory,
            default_model="future-model",
            fallback_model="stable-model",
            capabilities=AgentCapabilities(probe=True),
        )
    )
    runtime = resolve_agent_runtime("modelprobe")
    backend = create_registered_backend(runtime, probe_cwd="/tmp")

    assert attempted_models == ["future-model", "stable-model"]
    assert backend.runtime.model == "stable-model"
    assert "future-model" in backend.model_fallback_reason


def test_select_prefers_model_owning_provider() -> None:
    """Route auto selection to the available provider that claims the model."""
    _register_owning_fake("alphacli", owns_prefix="alpha")
    _register_owning_fake("betacli", owns_prefix="beta")
    assert registry.select_default_agent_provider("beta-42").name == "betacli"
    assert registry.select_default_agent_provider("alpha-9").name == "alphacli"


def test_select_skips_unavailable_model_owner(only_registered_providers) -> None:
    """Fall back to registration order when the model owner is unavailable."""
    _register_owning_fake("gammacli", owns_prefix="gamma", unavailable=True)
    _register_owning_fake("deltacli", owns_prefix="delta")
    result = registry.select_default_agent_provider("gamma-1")
    assert result.name == "deltacli"
    assert result.availability() is True


def test_select_unknown_model_uses_registration_order(
    only_registered_providers,
) -> None:
    """Keep first-available behavior when no provider claims the model."""
    _register_owning_fake("epsiloncli", owns_prefix="epsilon")
    default = registry.select_default_agent_provider()
    assert default.name == "epsiloncli"
    assert registry.select_default_agent_provider("mystery-9").name == default.name


def test_custom_provider_uses_generic_runtime() -> None:
    """Construct a custom backend without changing any core dispatch code."""
    _register_fake("testcli")
    runtime = resolve_agent_runtime(
        "testcli",
        executable="/tmp/testcli",
        timeout_sec=77,
        options={"flag": "value"},
    )

    backend = create_registered_backend(runtime)

    assert backend.name == "testcli"
    assert backend.runtime == runtime
    assert backend.capabilities.resumable
    assert runtime.model == "fake-model"
    assert runtime.options == {"flag": "value"}


def test_generic_fallback_uses_registered_provider() -> None:
    """Fall back without hard-coding either provider name in dispatch."""
    _register_fake("offlinecli", unavailable=True)
    _register_fake("backupcli", model="backup-model")
    runtime = resolve_agent_runtime(
        "offlinecli",
        fallback_provider="backupcli",
    )

    backend = create_registered_backend(runtime)

    assert backend.name == "backupcli"
    assert backend.runtime.model == "backup-model"
    assert backend.fallback_reason == "offlinecli"


def test_external_entry_point_provider_is_discovered(monkeypatch) -> None:
    """Load a provider through the public Python entry-point contract."""

    def external_factory(runtime):
        """Construct one backend loaded from an external entry point."""
        return _FakeBackend(runtime.provider)

    provider = AgentProvider(
        name="externalcli",
        factory=external_factory,
        default_model="external-model",
    )

    class _EntryPoint:
        """Model the importlib metadata entry-point surface used by registry."""

        name = "externalcli"

        @staticmethod
        def load():
            """Return the external provider factory."""
            return lambda: provider

    class _EntryPoints:
        """Return only entries belonging to the requested provider group."""

        @staticmethod
        def select(*, group):
            """Filter fake entries by the public provider group.

            The loader also probes the deprecated ``kernel_agents.*`` group, so
            an unknown group must come back empty rather than raise.
            """
            if group == registry.PROVIDER_ENTRY_POINT_GROUP:
                return [_EntryPoint()]
            assert group == registry.LEGACY_PROVIDER_ENTRY_POINT_GROUP
            return []

    monkeypatch.setattr(registry.metadata, "entry_points", _EntryPoints)
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    assert get_agent_provider("externalcli") is provider


def test_legacy_entry_point_group_still_loads_and_warns(monkeypatch) -> None:
    """A provider published under the pre-rename group still loads, once, loudly."""

    provider = AgentProvider(
        name="legacycli",
        factory=lambda runtime: _FakeBackend(runtime.provider),
        default_model="legacy-model",
    )

    class _EntryPoint:
        name = "legacycli"

        @staticmethod
        def load():
            return lambda: provider

    class _EntryPoints:
        @staticmethod
        def select(*, group):
            if group == registry.LEGACY_PROVIDER_ENTRY_POINT_GROUP:
                return [_EntryPoint()]
            assert group == registry.PROVIDER_ENTRY_POINT_GROUP
            return []

    monkeypatch.setattr(registry.metadata, "entry_points", _EntryPoints)
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    with pytest.warns(DeprecationWarning, match=registry.LEGACY_PROVIDER_ENTRY_POINT_GROUP):
        assert get_agent_provider("legacycli") is provider


def test_current_entry_point_group_wins_over_the_legacy_one(monkeypatch) -> None:
    """A name published under both groups resolves to the current group's entry."""

    current = AgentProvider(name="dualcli", factory=lambda runtime: _FakeBackend(runtime.provider), default_model="m")
    legacy = AgentProvider(name="dualcli", factory=lambda runtime: _FakeBackend(runtime.provider), default_model="m")

    def _entry(target):
        class _EntryPoint:
            name = "dualcli"

            @staticmethod
            def load():
                return lambda: target

        return _EntryPoint()

    class _EntryPoints:
        @staticmethod
        def select(*, group):
            if group == registry.PROVIDER_ENTRY_POINT_GROUP:
                return [_entry(current)]
            return [_entry(legacy)]

    monkeypatch.setattr(registry.metadata, "entry_points", _EntryPoints)
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    assert get_agent_provider("dualcli") is current


def test_config_accepts_external_provider_without_core_choice_list() -> None:
    """Resolve custom providers through generic Config fields."""
    _register_fake("configcli", model="config-model")

    config = Config(
        agent_backend="configcli",
        agent_model="selected-model",
        agent_cli="/usr/bin/configcli",
        agent_timeout_sec=12,
        agent_fallback_provider="",
        agent_options={"temperature": 0},
    )
    runtime = config.agent_runtime()

    assert runtime.provider == "configcli"
    assert runtime.model == "selected-model"
    assert runtime.executable == "/usr/bin/configcli"
    assert runtime.timeout_sec == 12
    assert runtime.options == {"temperature": 0}


def test_explicit_agent_options_do_not_parse_environment(monkeypatch) -> None:
    """Let explicit provider options override malformed environment JSON."""
    monkeypatch.setenv("FORGE_AGENT_OPTIONS_JSON", "{invalid")

    config = Config.from_env(agent_options={"temperature": 0})

    assert config.agent_options == {"temperature": 0}


def test_builtin_codex_consumes_generic_runtime() -> None:
    """Configure the built-in Codex backend through provider-neutral fields."""
    runtime = resolve_agent_runtime(
        "codex",
        model="gpt-test",
        executable="/tmp/missing-codex",
        timeout_sec=91,
        reasoning_effort="medium",
        sandbox_mode="workspace-write",
        precheck=False,
        fallback_provider="",
    )

    backend = create_registered_backend(runtime)
    resolved = AgentRunSpec(
        system_prompt="system",
        user_prompt="user",
        cwd="/tmp",
    ).resolved(backend.runtime)

    assert backend.codex_bin == "/tmp/missing-codex"
    assert backend.bypass_sandbox is False
    assert backend.capabilities.resumable
    assert resolved.model == "gpt-test"
    assert resolved.timeout_sec == 91
    assert resolved.reasoning_effort == "medium"


def test_builtin_codex_uses_generic_fallback_on_preflight(monkeypatch) -> None:
    """Apply the same registry fallback path to a built-in CLI provider."""
    monkeypatch.setattr(codex_backend, "_load_codex_sdk", object)
    _register_fake("codexbackup", model="fallback-model")
    runtime = resolve_agent_runtime(
        "codex",
        executable="/tmp/definitely-missing-codex",
        fallback_provider="codexbackup",
    )

    backend = create_registered_backend(runtime)

    assert backend.name == "codexbackup"
    assert "not executable" in backend.fallback_reason
