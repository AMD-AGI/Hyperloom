# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry and entry-point discovery for pluggable Agent CLI providers."""

from __future__ import annotations

import logging
import os
import re
import threading
import warnings
from dataclasses import dataclass, replace
from importlib import metadata, util
from typing import Callable, Mapping

from hyperloom.common.reasoning_effort import DEFAULT_REASONING_EFFORT
from kernelforge.agent_backends.base import (
    AgentBackend,
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRuntimeConfig,
)

log = logging.getLogger(__name__)

# Keeps a package-style prefix even though this module now lives in ``kernelforge.llm``: the group name is the
# published contract third-party providers register against, and renaming it would drop every existing plugin without
# a word -- a plugin that fails to load is recorded as one log line, not raised.
PROVIDER_ENTRY_POINT_GROUP = "kernelforge.agent_providers"
LEGACY_PROVIDER_ENTRY_POINT_GROUP = "kernel_agents.agent_providers"
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


def _always_available() -> bool:
    """Defer unknown external provider availability to normal preflight."""
    return True


def _credential_unproven(env: Mapping[str, str]) -> bool:
    """Default credential answer: an external provider proves nothing here.

    The neutral value on a ranking key is the losing one. Answering ``True``
    would let any installed plugin outrank both built-in providers on every box
    whose first-party credential is absent, including the unconfigured box this
    selection deliberately still resolves to Claude.
    """
    return False


def _owns_no_model(model: str) -> bool:
    """Default model ownership: external providers claim no model family."""
    return False


@dataclass(frozen=True)
class AgentProvider:
    """Describe one registered Agent CLI implementation."""

    name: str
    factory: Callable[[AgentRuntimeConfig], AgentBackend]
    default_model: str
    capabilities: AgentCapabilities = AgentCapabilities()
    availability: Callable[[], bool] = _always_available
    credentialed: Callable[[Mapping[str, str]], bool] = _credential_unproven
    owns_model: Callable[[str], bool] = _owns_no_model

    def __post_init__(self) -> None:
        """Validate stable provider metadata at registration time."""
        normalized = normalize_provider_name(self.name)
        if normalized != self.name:
            raise ValueError(f"provider name must already be normalized: {self.name!r}")
        if not self.default_model.strip():
            raise ValueError(f"provider {self.name!r} requires a default model")


_providers: dict[str, AgentProvider] = {}
_plugin_errors: dict[str, str] = {}
_plugins_loaded = False
_registry_lock = threading.RLock()


def normalize_provider_name(name: str) -> str:
    """Normalize and validate one provider identifier."""
    normalized = (name or "").strip().lower()
    if not _PROVIDER_NAME.fullmatch(normalized):
        raise ValueError("provider names must match [a-z][a-z0-9_-]*")
    return normalized


def register_agent_provider(
    provider: AgentProvider,
    *,
    replace_existing: bool = False,
) -> None:
    """Register one provider without requiring core package modification."""
    with _registry_lock:
        if provider.name in _providers and not replace_existing:
            raise ValueError(f"agent provider {provider.name!r} is already registered")
        _providers[provider.name] = provider


def discover_agent_providers(*, force: bool = False) -> None:
    """Load external providers from the public Python entry-point group."""
    global _plugins_loaded
    with _registry_lock:
        if _plugins_loaded and not force:
            return
        _plugins_loaded = True
        try:
            discovered = metadata.entry_points()

            def _select(group: str):
                if hasattr(discovered, "select"):
                    return list(discovered.select(group=group))
                return list(discovered.get(group, []))

            entries = _select(PROVIDER_ENTRY_POINT_GROUP)
            legacy = [e for e in _select(LEGACY_PROVIDER_ENTRY_POINT_GROUP) if e.name not in {x.name for x in entries}]
            if legacy:
                warnings.warn(
                    f"Agent provider entry-point group {LEGACY_PROVIDER_ENTRY_POINT_GROUP!r} is deprecated; "
                    f"republish under {PROVIDER_ENTRY_POINT_GROUP!r}. Loading "
                    + ", ".join(sorted(e.name for e in legacy)),
                    DeprecationWarning,
                    stacklevel=2,
                )
                entries = entries + legacy
        except Exception as exc:  # noqa: BLE001 - plugin discovery is optional
            _plugin_errors["<discovery>"] = f"{type(exc).__name__}: {exc}"
            return

        for entry in entries:
            try:
                loaded = entry.load()
                provider = loaded() if callable(loaded) else loaded
                if not isinstance(provider, AgentProvider):
                    raise TypeError(
                        "entry point must resolve to AgentProvider or a zero-argument factory returning AgentProvider"
                    )
                entry_name = normalize_provider_name(entry.name)
                if provider.name != entry_name:
                    raise ValueError(f"entry-point name {entry_name!r} does not match provider name {provider.name!r}")
                register_agent_provider(provider)
            except Exception as exc:  # noqa: BLE001 - isolate broken plugins
                _plugin_errors[entry.name] = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "failed to load Agent provider entry point %s: %s",
                    entry.name,
                    exc,
                )


def get_agent_provider(name: str) -> AgentProvider:
    """Resolve one built-in or externally installed Agent provider."""
    discover_agent_providers()
    normalized = normalize_provider_name(name)
    provider = _providers.get(normalized)
    if provider is None:
        available = ", ".join(sorted(_providers)) or "(none)"
        detail = _plugin_errors.get(normalized)
        suffix = f"; plugin error: {detail}" if detail else ""
        raise ValueError(f"unknown agent provider {normalized!r}; available: {available}{suffix}")
    return provider


def list_agent_providers() -> tuple[str, ...]:
    """Return all built-in and successfully discovered provider names."""
    discover_agent_providers()
    return tuple(sorted(_providers))


def select_default_agent_provider(preferred_model: str = "") -> AgentProvider:
    """Select a provider by the one rule the whole repository shares.

    Two ranked keys, shared with
    :func:`hyperloom.common.llm_config.preferred_agent_backend`: a configured
    credential, then an installed SDK. Providers that tie on both keep
    registration order, which is what puts Claude ahead of Codex.

    Credentials lead because ranking on the installed SDK alone is what let an
    OpenAI-only box resolve to Claude whenever both extras happened to be
    installed, and then fail to authenticate.

    A named ``preferred_model`` narrows the candidates rather than joining the
    ranking: ownership says which provider the caller's model belongs to, and no
    credential shape should overrule that, while an owner that cannot run is
    worse than a fallback that can.

    A provider missing one of the two keys is still returned, so its own
    preflight reports the absent extra or the failed login. Only a provider
    missing both is refused.
    """
    discover_agent_providers()
    failures: list[str] = []
    model = (preferred_model or "").strip()

    def _holds(provider: AgentProvider, predicate: Callable[[], bool]) -> bool:
        """Answer one provider predicate, counting a raising provider as a "no"."""
        try:
            return bool(predicate())
        except Exception as exc:  # noqa: BLE001 - one provider must not decide the whole selection
            failures.append(f"{provider.name}: {type(exc).__name__}: {exc}")
            return False

    providers = list(_providers.values())
    if model:
        runnable_owners = [
            provider
            for provider in providers
            if _holds(provider, lambda: provider.owns_model(model)) and _holds(provider, provider.availability)
        ]
        providers = runnable_owners or providers
    ranks = {
        provider.name: (
            0 if _holds(provider, lambda: provider.credentialed(os.environ)) else 1,
            0 if _holds(provider, provider.availability) else 1,
        )
        for provider in providers
    }
    chosen = min(providers, key=lambda provider: ranks[provider.name], default=None)
    if chosen is not None and ranks[chosen.name] != (1, 1):
        return chosen
    detail = f"; checks: {'; '.join(failures)}" if failures else ""
    raise AgentProviderUnavailableError(
        "no Agent provider is configured or installed; install the 'claude' or "
        "'codex' extra of the distribution you installed (kernelforge provides "
        "both) and configure that provider's credentials, or configure an "
        "external provider"
        f"{detail}"
    )


def resolve_agent_runtime(
    provider: str,
    *,
    model: str = "",
    executable: str = "",
    timeout_sec: int = 1800,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    sandbox_mode: str = "bypass",
    precheck: bool = True,
    fallback_provider: str = "",
    options: dict | None = None,
) -> AgentRuntimeConfig:
    """Resolve provider defaults into one complete runtime configuration."""
    registration = get_agent_provider(provider)
    fallback = normalize_provider_name(fallback_provider) if fallback_provider else ""
    if fallback == registration.name:
        fallback = ""
    if fallback:
        get_agent_provider(fallback)
    selected = model.strip() or registration.default_model
    return AgentRuntimeConfig(
        provider=registration.name,
        model=selected,
        executable=executable.strip(),
        timeout_sec=timeout_sec,
        reasoning_effort=reasoning_effort.strip() or DEFAULT_REASONING_EFFORT,
        sandbox_mode=sandbox_mode.strip() or "bypass",
        precheck=precheck,
        fallback_provider=fallback,
        options=dict(options or {}),
    )


def create_registered_backend(
    runtime: AgentRuntimeConfig,
    *,
    preflight: bool | None = None,
    probe_cwd: str = "",
    usage=None,
) -> AgentBackend:
    """Construct and probe one provider backend, with optional provider fallback."""
    registration = get_agent_provider(runtime.provider)
    should_preflight = runtime.precheck if preflight is None else preflight
    try:
        return _prepare_backend(
            registration,
            runtime,
            preflight=should_preflight,
            probe_cwd=probe_cwd,
            usage=usage,
        )
    except AgentProviderUnavailableError as exc:
        log.warning(
            "agent provider unavailable provider=%s model=%s: %s",
            registration.name,
            runtime.model,
            exc,
        )
        if not runtime.fallback_provider:
            raise
        fallback_registration = get_agent_provider(runtime.fallback_provider)
        fallback_runtime = replace(
            runtime,
            provider=fallback_registration.name,
            model=fallback_registration.default_model,
            executable="",
            fallback_provider="",
            options={},
        )
        try:
            fallback = _prepare_backend(
                fallback_registration,
                fallback_runtime,
                preflight=should_preflight,
                probe_cwd=probe_cwd,
                usage=usage,
            )
        except AgentProviderUnavailableError as fallback_exc:
            raise AgentProviderUnavailableError(
                f"{runtime.provider} unavailable: {exc}; fallback "
                f"{fallback_registration.name} unavailable: {fallback_exc}"
            ) from fallback_exc
        setattr(fallback, "fallback_reason", str(exc))
        return fallback


def _prepare_backend(
    registration: AgentProvider,
    runtime: AgentRuntimeConfig,
    *,
    preflight: bool,
    probe_cwd: str,
    usage,
) -> AgentBackend:
    """Initialize one backend and run capabilities it explicitly declares."""
    backend = registration.factory(runtime)
    setattr(backend, "runtime", runtime)
    setattr(backend, "capabilities", registration.capabilities)
    if preflight and hasattr(backend, "preflight"):
        backend.preflight()
    if preflight and probe_cwd and registration.capabilities.probe and hasattr(backend, "probe"):
        backend.probe(cwd=probe_cwd, usage=usage)
    return backend


def _create_claude_backend(runtime: AgentRuntimeConfig) -> AgentBackend:
    """Construct the built-in Claude backend lazily."""
    from kernelforge.agent_backends.claude import ClaudeBackend

    return ClaudeBackend(runtime=runtime)


def _create_codex_backend(runtime: AgentRuntimeConfig) -> AgentBackend:
    """Construct the built-in Codex backend lazily."""
    from kernelforge.agent_backends.codex import CodexBackend

    return CodexBackend(runtime=runtime)


def _claude_available() -> bool:
    """Return whether the optional Claude SDK is installed."""
    return util.find_spec("claude_agent_sdk") is not None


def _codex_available() -> bool:
    """Return whether the optional Codex Python SDK is installed."""
    return util.find_spec("openai_codex") is not None


def _claude_credentialed(env: Mapping[str, str]) -> bool:
    """Return whether the Anthropic side can authenticate this CLI."""
    from hyperloom.common import llm_config

    return llm_config.anthropic_agent_credentialed(env)


def _codex_credentialed(env: Mapping[str, str]) -> bool:
    """Return whether the OpenAI side can authenticate this CLI."""
    from hyperloom.common import llm_config

    return llm_config.openai_agent_credentialed(env)


def _claude_owns_model(model: str) -> bool:
    """Recognize Anthropic Claude model identifiers."""
    return model.strip().lower().startswith("claude")


def _codex_owns_model(model: str) -> bool:
    """Recognize OpenAI/Codex gateway model identifiers."""
    normalized = model.strip().lower()
    if not normalized:
        return False
    if "codex" in normalized:
        return True
    if normalized.startswith("gpt"):
        return True
    return re.match(r"^o\d(?:$|[-._:/])", normalized) is not None


register_agent_provider(
    AgentProvider(
        name="claude",
        factory=_create_claude_backend,
        default_model="claude-opus-5",
        capabilities=AgentCapabilities(
            writable=True,
            resumable=True,
            stop_hooks=True,
            native_subagents=True,
            mcp=True,
            probe=True,
            # ClaudeBackend._provider_options folds spec.env into the SDK's env option, which the SDK applies over the
            # environment it spawns the CLI with.
            session_env=True,
            workspace_guard=True,
        ),
        availability=_claude_available,
        credentialed=_claude_credentialed,
        owns_model=_claude_owns_model,
    )
)
register_agent_provider(
    AgentProvider(
        name="codex",
        factory=_create_codex_backend,
        default_model="gpt-5.6-sol",
        capabilities=AgentCapabilities(
            writable=True,
            resumable=True,
            native_subagents=True,
            mcp=True,
            sandbox=True,
            probe=True,
            requires_workspace_cwd=True,
            # CodexBackend._sdk_config applies spec.env over the child environment of the app server that parents the
            # session.
            session_env=True,
            workspace_guard=True,
        ),
        availability=_codex_available,
        credentialed=_codex_credentialed,
        owns_model=_codex_owns_model,
    )
)


__all__ = [
    "AgentProvider",
    "PROVIDER_ENTRY_POINT_GROUP",
    "create_registered_backend",
    "discover_agent_providers",
    "get_agent_provider",
    "list_agent_providers",
    "normalize_provider_name",
    "register_agent_provider",
    "resolve_agent_runtime",
    "select_default_agent_provider",
]
