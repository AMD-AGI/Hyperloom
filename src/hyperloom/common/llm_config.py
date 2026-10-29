# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM gateway ownership: env resolution *and* client construction."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import AbstractContextManager, AsyncExitStack, ExitStack
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from hyperloom.common.env import is_truthy
from hyperloom.common.llm_attribution import call_headers as _attribution_headers
from hyperloom.common.llm_attribution import gateway_selected as _gateway_selected
from hyperloom.common.llm_attribution import inject_env as _inject_attribution_env
from hyperloom.common.llm_headers import expand_env_refs, parse_custom_headers
from hyperloom.common.llm_request_hooks import (
    AnthropicStream,
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_OPENAI_CHAT,
    RequestObservation,
    observed_request,
)
from hyperloom.common.reasoning_effort import gateway_reasoning_effort
from hyperloom.common.token_usage import anthropic_usage_counts, openai_usage_counts, sdk_field

log = logging.getLogger(__name__)

_OPENAI_SDK_MISSING = "openai SDK not installed; run `pip install openai>=1.50`"
_HTTPX_MISSING = "httpx not installed; run `pip install httpx>=0.27`"


class LLMConfigError(RuntimeError):
    """Raised when a requested LLM client cannot be configured from env."""


@dataclass(frozen=True)
class OpenAIClientConfig:
    """Resolved configuration for OpenAI-compatible SDK clients."""

    api_key: str
    base_url: str | None
    default_headers: dict[str, str]
    # The variable ``default_headers`` was parsed from.
    headers_env: str

    def as_kwargs(self) -> dict[str, object]:
        """Return kwargs accepted by ``openai.AsyncOpenAI``."""
        kwargs: dict[str, object] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs


# Subscription credential minted by ``claude setup-token``.
CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

# Single source of truth for "what counts as an Anthropic-side credential", highest precedence first.
ANTHROPIC_CREDENTIAL_ENV_ORDER: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    CLAUDE_OAUTH_TOKEN_ENV,
)

# The subset whose value may be copied into another credential var or persisted to ~/.claude/config.json.
ANTHROPIC_SYNTHESIZABLE_KEY_ENVS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)

# What may authenticate an OpenAI-protocol client, highest precedence first. The Anthropic-side keys come last
# because an Anthropic-only deployment fronts both protocols behind one gateway token.
_OPENAI_CLIENT_KEY_ENV_ORDER: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
)


def _first_set_value(names: Iterable[str], source: Mapping[str, str]) -> str:
    """First non-blank value among ``names``, in the order given."""
    for name in names:
        value = (source.get(name) or "").strip()
        if value:
            return value
    return ""


def has_anthropic_credential(env: Mapping[str, str] | None = None) -> bool:
    """True when any Anthropic-side credential form is set."""
    return bool(
        _first_set_value(
            ANTHROPIC_CREDENTIAL_ENV_ORDER,
            env if env is not None else os.environ,
        )
    )


def anthropic_synthesizable_key(env: Mapping[str, str] | None = None) -> str:
    """Highest-precedence credential that may be copied elsewhere."""
    return _first_set_value(
        ANTHROPIC_SYNTHESIZABLE_KEY_ENVS,
        env if env is not None else os.environ,
    )


def with_synthesized_anthropic_keys(env: Mapping[str, str]) -> dict[str, str]:
    """Copy of *env* whose missing Anthropic API key or auth token is filled from the other one.

    Only the synthesizable subset: an OAuth token copied into either API-key var would drop the CLI out of
    subscription mode and 401 the run.
    """
    source = dict(env)
    fallback_key = anthropic_synthesizable_key(source)
    if fallback_key:
        source.setdefault("ANTHROPIC_API_KEY", fallback_key)
        source.setdefault("ANTHROPIC_AUTH_TOKEN", fallback_key)
    return source


CLAUDE_GATEWAY_SIGNAL_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    CLAUDE_OAUTH_TOKEN_ENV,
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
)

# Retired provider-specific variables.
LEGACY_DEEPSEEK_ENV_KEYS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
)

_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
_DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_MODEL = "deepseek-v4-pro"

# Hosts known to serve the Anthropic protocol on ``/anthropic`` and OpenAI chat-completions on ``/v1``.
_DUAL_PROTOCOL_HOSTS: frozenset[str] = frozenset({"api.deepseek.com"})

# Gateways that serve only their own models.
_HOST_DEFAULT_MODELS: dict[str, str] = {"api.deepseek.com": _DEEPSEEK_MODEL}

# The retired variables are Anthropic-side shaped, so they are adopted only when that whole side is empty -- and then
# for BOTH protocols, since they describe one gateway.
_ANTHROPIC_SIDE_KEYS: tuple[str, ...] = ("ANTHROPIC_BASE_URL", *ANTHROPIC_CREDENTIAL_ENV_ORDER)
_OPENAI_SIDE_KEYS: tuple[str, ...] = ("OPENAI_BASE_URL", "OPENAI_API_KEY")

# A managed gateway carries the credential itself, so it can drive the Claude CLI without naming a key.
_ANTHROPIC_MANAGED_GATEWAY_ENVS: tuple[str, ...] = ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")

# The agent CLIs that can drive this repository's agentic roles.
AGENT_BACKEND_CLAUDE = "claude"
AGENT_BACKEND_CODEX = "codex"
AGENT_BACKEND_DEEPSEEK = "deepseek"

#: Names the agent CLI for the agentic roles, overriding the credential-ranked
#: rule between Claude and Codex. The DeepSeek Harness is reachable only this
#: way: it returns its intents in the reply body with no schema enforcing them,
#: so it is a route an operator asks for rather than one a credential shape can
#: select. The harness model comes from ``DSH_MODEL``. Quantization honours
#: only ``deepseek``: any other value leaves it on Claude.
AGENT_BACKEND_ENV = "INFERENCE_OPTIMIZER_AGENT_BACKEND"

AGENT_BACKEND_CHOICES: tuple[str, ...] = (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX, AGENT_BACKEND_DEEPSEEK)

# The model each backend runs when neither the operator nor a known gateway host names one. This is the last rung of
# every model ladder in both packages, so it lives beside the backend names it is keyed by.
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CODEX_MODEL = "gpt-6-astra"
# The harness route mounts DeepSeek's own models only, so this names one of
# those rather than a gateway deployment.
DEFAULT_DEEPSEEK_HARNESS_MODEL = "deepseek-v4-flash"


def has_anthropic_side(env: Mapping[str, str] | None = None) -> bool:
    """True when an Anthropic-side endpoint or key is configured."""
    source = env if env is not None else os.environ
    return any((source.get(name) or "").strip() for name in _ANTHROPIC_SIDE_KEYS)


def has_openai_side(env: Mapping[str, str] | None = None) -> bool:
    """True when an OpenAI-side endpoint or key is configured."""
    source = env if env is not None else os.environ
    return any((source.get(name) or "").strip() for name in _OPENAI_SIDE_KEYS)


def anthropic_agent_credentialed(env: Mapping[str, str] | None = None) -> bool:
    """True when the Anthropic side can authenticate an agent CLI run.

    Wider than :func:`has_anthropic_credential` by the managed gateways, which
    carry the credential themselves and so name no key. It is a predicate of its
    own rather than a widening of that one because the credential checks in
    ``cli/credentials.py`` hand a key to the Anthropic Messages API or persist it
    to ``~/.claude/config.json``, and a Bedrock or Vertex box satisfies neither.
    """
    source = env if env is not None else os.environ
    if has_anthropic_credential(source):
        return True
    return any(is_truthy(source.get(name)) for name in _ANTHROPIC_MANAGED_GATEWAY_ENVS)


def openai_agent_credentialed(env: Mapping[str, str] | None = None) -> bool:
    """True when the OpenAI side can authenticate an agent CLI run.

    ``OPENAI_API_KEY`` is the only name that authenticates one, which is the
    same name :mod:`hyperloom.common.codex_session` resolves and the same one
    ``_validate_credentials`` admits a run on. A bare ``OPENAI_BASE_URL`` with
    it unset is an endpoint hint, not a credential, and treating the URL alone
    as configured is what sends an unauthenticated Codex run.
    """
    source = env if env is not None else os.environ
    return bool((source.get("OPENAI_API_KEY") or "").strip())


def is_anthropic_only(env: Mapping[str, str] | None = None) -> bool:
    """True when the Anthropic side is the only configured provider."""
    return has_anthropic_side(env) and not has_openai_side(env)


def is_openai_only(env: Mapping[str, str] | None = None) -> bool:
    """True when the OpenAI side is the only configured provider."""
    return has_openai_side(env) and not has_anthropic_side(env)


def claude_agent_sdk_installed() -> bool:
    """Return whether the optional Claude Agent SDK is installed."""
    from importlib.util import find_spec

    return find_spec("claude_agent_sdk") is not None


def codex_agent_sdk_installed() -> bool:
    """Return whether the optional Codex Agent SDK is installed."""
    from importlib.util import find_spec

    return find_spec("openai_codex") is not None


def deepseek_harness_sdk_installed() -> bool:
    """Return whether the optional DeepSeek Harness SDK is installed."""
    from importlib.util import find_spec

    return find_spec("deepseek_harness") is not None


def agent_backend_rank(*, credentialed: bool, sdk_installed: bool) -> tuple[int, int]:
    """Sort key for one agent backend candidate: credential first, then SDK.

    The one ordering the whole repository shares, read by
    :func:`preferred_agent_backend` and by
    :func:`kernelforge.agent_backends.registry.select_default_agent_provider`.
    Lower sorts first and candidates that tie keep the order they were offered
    in, which is what puts Claude ahead of Codex on both sides.

    Credentials lead because ranking on the installed SDK alone is what let an
    OpenAI-only box resolve to Claude whenever both extras happened to be
    installed, and then fail to authenticate.
    """
    return (0 if credentialed else 1, 0 if sdk_installed else 1)


# The rank of a candidate that can neither authenticate nor import: no reason to prefer it over any other.
UNRUNNABLE_AGENT_RANK: tuple[int, int] = agent_backend_rank(credentialed=False, sdk_installed=False)


def explicit_agent_backend(env: Mapping[str, str] | None = None, *, env_name: str = AGENT_BACKEND_ENV) -> str:
    """Return the agent backend ``env_name`` names, or ``""`` when unset.

    Raises:
        ValueError: When it names no known backend, or the DeepSeek Harness without its SDK.
    """
    requested = ((env if env is not None else os.environ).get(env_name) or "").strip().lower()
    if requested and requested not in AGENT_BACKEND_CHOICES:
        raise ValueError(f"{env_name}={requested!r} not in {set(AGENT_BACKEND_CHOICES)}")
    if requested == AGENT_BACKEND_DEEPSEEK and not deepseek_harness_sdk_installed():
        raise ValueError(f"{env_name}=deepseek requires the DeepSeek Harness SDK: install the 'deepseek' extra")
    return requested


def resolve_agent_backend(env: Mapping[str, str] | None = None) -> str:
    """Return the harness backend every non-critic agent role should run.

    An explicit :data:`AGENT_BACKEND_ENV` wins over the credential-ranked rule
    between Claude and Codex. DeepSeek is reachable only through that explicit
    name: it is not a third candidate in :func:`preferred_agent_backend`.
    """
    return explicit_agent_backend(env) or preferred_agent_backend(env)


def preferred_agent_backend(env: Mapping[str, str] | None = None) -> str:
    """Return the agent backend this environment should run.

    Ranked by :func:`agent_backend_rank`. Claude wins whenever both backends
    tie, so an OpenAI-only credential is the only shape that selects Codex
    outright; a dual-configured deployment keeps Claude.

    With no credential on either side -- a runtime logged in by other means
    carries none this can see -- the installed SDK decides, and when neither
    extra is present Claude is still the default so preflight can report what is
    missing rather than silently picking the other CLI.
    """
    source = env if env is not None else os.environ
    ranks = {
        AGENT_BACKEND_CLAUDE: agent_backend_rank(
            credentialed=anthropic_agent_credentialed(source),
            sdk_installed=claude_agent_sdk_installed(),
        ),
        AGENT_BACKEND_CODEX: agent_backend_rank(
            credentialed=openai_agent_credentialed(source),
            sdk_installed=codex_agent_sdk_installed(),
        ),
    }
    return min(ranks, key=lambda backend: ranks[backend])


DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The Anthropic Messages API version, defined once for the whole repository.
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
_ANTHROPIC_MESSAGES_METHOD = "POST"
# The OpenAI path re-dials through the SDK (``max_retries``); on the Messages path this module owes the same.
ANTHROPIC_CONNECT_RETRIES = 2
# Waited before the second dial and doubled before the third, as the SDK's backoff does.
_ANTHROPIC_CONNECT_BACKOFF_S = 0.5


def _is_dual_protocol_host(url: str | None) -> bool:
    """True when ``url`` points at a known ``/anthropic`` + ``/v1`` gateway."""
    if not url:
        return False
    return urlsplit(str(url).strip()).hostname in _DUAL_PROTOCOL_HOSTS


def dual_protocol_endpoint_pair(base_url: str) -> tuple[str, str]:
    """Return ``(anthropic_base_url, openai_base_url)`` for a dual-protocol gateway."""
    base = base_url.strip().rstrip("/")
    if not base:
        return _DEEPSEEK_ANTHROPIC_BASE_URL, _DEEPSEEK_OPENAI_BASE_URL
    lowered = base.lower()
    # Swap the trailing path segment when a recognized protocol suffix is present.
    if lowered.endswith("/anthropic"):
        return base, f"{base.rsplit('/', 1)[0]}/v1"
    if lowered.endswith("/v1"):
        return f"{base.rsplit('/', 1)[0]}/anthropic", base
    # No recognized protocol suffix.
    if _is_dual_protocol_host(base):
        parsed = urlsplit(base)
        if not parsed.path or parsed.path == "/":
            return f"{base}/anthropic", f"{base}/v1"
    return base, f"{base}/v1"


def deepseek_compat_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Translate a legacy ``DEEPSEEK_*`` configuration into the standard variables."""
    source = env if env is not None else os.environ
    api_key = (source.get("DEEPSEEK_API_KEY") or "").strip()
    legacy_url = (source.get("DEEPSEEK_BASE_URL") or "").strip()
    if not api_key and not legacy_url:
        return {}
    if has_anthropic_side(source):
        return {}

    anthropic_url, openai_url = dual_protocol_endpoint_pair(legacy_url)
    model = (source.get("DEEPSEEK_MODEL") or "").strip() or _DEEPSEEK_MODEL
    candidates: dict[str, str] = {
        "ANTHROPIC_BASE_URL": anthropic_url,
        "CLAUDE_MODEL": model,
        # GEAKv4 drives Claude Code with its own model variable, so it follows whichever Claude model is actually in
        # effect -- an explicit CLAUDE_MODEL must not be contradicted by the gateway default.
        "GEAK_CLAUDE_MODEL": (source.get("CLAUDE_MODEL") or "").strip() or model,
    }
    if api_key:
        # Two spellings of one credential: x-api-key and bearer.
        candidates["ANTHROPIC_API_KEY"] = api_key
        candidates["ANTHROPIC_AUTH_TOKEN"] = api_key
    # The OpenAI side is adopted only when it is entirely free.
    if not has_openai_side(source):
        candidates["OPENAI_BASE_URL"] = openai_url
        candidates["CODEX_MODEL"] = model
        if api_key:
            candidates["OPENAI_API_KEY"] = api_key
    return {key: value for key, value in candidates.items() if value and not (source.get(key) or "").strip()}


def endpoint_default_model(base_url: str | None) -> str:
    """Return the model a known gateway host serves, or ``""`` for anything else."""
    if not base_url:
        return ""
    return _HOST_DEFAULT_MODELS.get(urlsplit(str(base_url).strip()).hostname or "", "")


def provider_model_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the model variables implied by the configured endpoints."""
    source = env if env is not None else os.environ
    resolved = dict(source)
    resolved.update(deepseek_compat_env(resolved))
    claude_model = (resolved.get("CLAUDE_MODEL") or "").strip() or endpoint_default_model(
        resolved.get("ANTHROPIC_BASE_URL")
    )
    codex_model = (resolved.get("CODEX_MODEL") or "").strip() or endpoint_default_model(resolved.get("OPENAI_BASE_URL"))
    candidates = {
        "CLAUDE_MODEL": claude_model,
        "CODEX_MODEL": codex_model,
        "GEAK_CLAUDE_MODEL": (resolved.get("GEAK_CLAUDE_MODEL") or "").strip() or claude_model,
    }
    return {key: value for key, value in candidates.items() if value and not (source.get(key) or "").strip()}


def resolve_forge_llm_model(
    agent_backend: str,
    *,
    env: Mapping[str, str] | None = None,
    explicit: str | None = None,
) -> str:
    """Resolve the Forge LLM model id for a chosen agent backend.

    The backend decides the last rung, so callers never have to: ``CLAUDE_MODEL``
    is unset on every run that authenticates by OAuth token, and a caller that
    forgot a default would post an empty model id.
    """
    source = env if env is not None else os.environ
    explicit_model = (explicit or "").strip()
    if explicit_model:
        return explicit_model
    backend = (agent_backend or "").strip().lower()
    if backend == AGENT_BACKEND_CODEX:
        return str(source.get("CODEX_MODEL") or "").strip() or DEFAULT_CODEX_MODEL
    if backend == AGENT_BACKEND_DEEPSEEK:
        return str(source.get("DSH_MODEL") or "").strip() or DEFAULT_DEEPSEEK_HARNESS_MODEL
    return str(source.get("CLAUDE_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL


def derive_openai_base_url(anthropic_base_url: str | None) -> str | None:
    """Derive an OpenAI-compatible base URL from an Anthropic endpoint."""
    if not anthropic_base_url:
        return None
    base = anthropic_base_url.strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    path = parts.path.rstrip("/")
    # Match case-insensitively: AMD's default endpoint uses "/Anthropic" (issue #929).
    path_lower = path.lower()
    if path_lower.endswith("/anthropic"):
        prefix = path[: -len("/anthropic")]
        return urlunsplit(parts._replace(path=f"{prefix}/Unified/v1"))
    if path_lower.endswith("/unified"):
        prefix = path[: -len("/unified")]
        return urlunsplit(parts._replace(path=f"{prefix}/Unified/v1"))
    return base


def resolve_openai_client_config(*, env: dict[str, str] | None = None) -> OpenAIClientConfig:
    """Resolve OpenAI-compatible client config from one or more LLM env sets."""
    source = env if env is not None else os.environ
    api_key = _first_set_value(_OPENAI_CLIENT_KEY_ENV_ORDER, source)
    if not api_key:
        raise LLMConfigError(
            f"{' / '.join(_OPENAI_CLIENT_KEY_ENV_ORDER)} not set in env (OpenAI-compatible client cannot auth)"
        )

    explicit_base_url = (source.get("OPENAI_BASE_URL") or "").strip()
    derived_base_url = (derive_openai_base_url(source.get("ANTHROPIC_BASE_URL")) or "").strip()
    base_url = explicit_base_url or derived_base_url or None

    # Gateway headers are operator-supplied, and they belong to an endpoint rather than to a protocol: AMD's gateway
    # rejects a call without its subscription header, and that header is only ever written to
    # ANTHROPIC_CUSTOM_HEADERS.
    headers_env = "OPENAI_CUSTOM_HEADERS"
    headers = parse_custom_headers(source.get(headers_env), env=source)
    if not headers and not explicit_base_url and derived_base_url:
        headers_env = "ANTHROPIC_CUSTOM_HEADERS"
        headers = parse_custom_headers(source.get(headers_env), env=source)
    return OpenAIClientConfig(api_key=api_key, base_url=base_url, default_headers=headers, headers_env=headers_env)


def build_http_timeout(
    *,
    connect: float,
    read: float,
    write: float | None = None,
    pool: float | None = None,
) -> object | None:
    """Build an ``httpx.Timeout`` to hand to :func:`get_openai_client`."""
    try:
        import httpx
    except ImportError:
        log.warning("llm_config: httpx unavailable; falling back to SDK default timeouts")
        return None
    return httpx.Timeout(
        connect=connect,
        read=read,
        write=read if write is None else write,
        pool=read if pool is None else pool,
    )


def _openai_sdk_client_class(class_name: str) -> object:
    """Resolve one OpenAI SDK client class, importing the SDK on demand."""
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LLMConfigError(_OPENAI_SDK_MISSING) from exc
    return getattr(openai, class_name)


def _openai_sdk_kwargs(*, env: dict[str, str] | None, timeout: object | None) -> dict[str, object]:
    """Resolve credentials and fold in an optional per-caller ``timeout``."""
    kwargs = resolve_openai_client_config(env=env).as_kwargs()
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def get_openai_client(
    *,
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the synchronous ``openai.OpenAI`` client."""
    factory = _openai_sdk_client_class("OpenAI")
    return factory(  # type: ignore[operator]
        **_openai_sdk_kwargs(
            env=env,
            timeout=timeout,
        )
    )


def get_async_openai_client(
    *,
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the asynchronous ``openai.AsyncOpenAI`` client."""
    factory = _openai_sdk_client_class("AsyncOpenAI")
    return factory(  # type: ignore[operator]
        **_openai_sdk_kwargs(
            env=env,
            timeout=timeout,
        )
    )


def _httpx_module() -> object:
    """Import ``httpx`` on demand, keeping this module cheap to import."""
    try:
        import httpx
    except ImportError as exc:
        raise LLMConfigError(_HTTPX_MISSING) from exc
    return httpx


def _anthropic_connect_failures() -> tuple[type[BaseException], ...]:
    """The ``httpx`` failures that mean nothing was sent, so dialling again cannot duplicate a completion."""
    httpx = _httpx_module()
    return (httpx.ConnectError, httpx.ConnectTimeout)  # type: ignore[attr-defined]


def _anthropic_client_kwargs(*, env: dict[str, str] | None, timeout: object | None) -> dict[str, object]:
    """Resolve base URL, auth and gateway headers for an Anthropic HTTP client."""
    source = env if env is not None else os.environ
    api_key = _first_set_value(ANTHROPIC_SYNTHESIZABLE_KEY_ENVS, source)
    if not api_key:
        raise LLMConfigError(
            f"{' / '.join(ANTHROPIC_SYNTHESIZABLE_KEY_ENVS)} not set in env (Anthropic client cannot auth)"
        )

    base_url = ((source.get("ANTHROPIC_BASE_URL") or "").strip() or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    # Merged last so an operator can override any default, the API version included.
    headers.update(parse_custom_headers(source.get("ANTHROPIC_CUSTOM_HEADERS"), env=source))
    kwargs: dict[str, object] = {"base_url": base_url, "headers": headers}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def get_anthropic_client(
    *,
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the synchronous ``httpx.Client`` for the Anthropic Messages API."""
    httpx = _httpx_module()
    return httpx.Client(  # type: ignore[attr-defined]
        **_anthropic_client_kwargs(
            env=env,
            timeout=timeout,
        )
    )


def get_async_anthropic_client(
    *,
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the asynchronous ``httpx.AsyncClient`` for the Messages API."""
    httpx = _httpx_module()
    return httpx.AsyncClient(  # type: ignore[attr-defined]
        **_anthropic_client_kwargs(
            env=env,
            timeout=timeout,
        )
    )


def claude_sdk_env_options(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    component: str = "",
    operation: str = "",
    **attribution: str,
) -> dict[str, object]:
    """Return Claude SDK options that isolate a run from global Claude config."""
    source = dict(env if env is not None else os.environ)
    # Callers that never reach CLI preflight (library-mode backends) may still carry a legacy DeepSeek configuration;
    # normalize before probing signals.
    source.update(deepseek_compat_env(source))
    if not any((source.get(key) or "").strip() for key in CLAUDE_GATEWAY_SIGNAL_KEYS):
        return {}

    source = with_synthesized_anthropic_keys(source)
    # Claude/Anthropic side reads only ANTHROPIC_CUSTOM_HEADERS.
    if source.get("ANTHROPIC_CUSTOM_HEADERS"):
        source["ANTHROPIC_CUSTOM_HEADERS"] = expand_env_refs(source["ANTHROPIC_CUSTOM_HEADERS"], source)
    # Disable the advisor-tool beta header by default since strict gateways reject it.
    source.setdefault("CLAUDE_CODE_DISABLE_ADVISOR_TOOL", "1")
    if model:
        source.setdefault("ANTHROPIC_MODEL", model)
        source.setdefault("ANTHROPIC_SMALL_FAST_MODEL", model)
    if component:
        _inject_attribution_env(source, component=component, operation=operation, **attribution)
    return {"env": source, "setting_sources": []}


def apply_reasoning_effort(
    params: dict[str, object],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Inject ``reasoning_effort`` into chat.completions params, env-gated.

    ``max`` is a Claude-only level and comes back a 400 here, so it is sent as ``xhigh``.
    """
    source = env if env is not None else os.environ
    raw = source.get("HYPERLOOM_REASONING_EFFORT") or source.get("OPENAI_REASONING_EFFORT") or ""
    val = gateway_reasoning_effort(raw)
    if val:
        params["reasoning_effort"] = val
    return params


@dataclass(frozen=True)
class ChatCompletionResult:
    """One non-streaming chat completion, flattened for Hyperloom callers."""

    text: str
    finish_reason: str | None
    usage: object | None


def _chat_completion_result(resp: object) -> ChatCompletionResult:
    """Flatten one chat-completions response onto :class:`ChatCompletionResult`."""
    choice = resp.choices[0]  # type: ignore[attr-defined]
    return ChatCompletionResult(
        text=choice.message.content or "",
        finish_reason=getattr(choice, "finish_reason", None),
        usage=getattr(resp, "usage", None),
    )


#: Call sites already warned about an untagged request, keyed by code location.
#: These helpers run once per LLM request, so without this an uninstrumented
#: caller would repeat the same line for the length of a run.
_UNTAGGED_SITES: set[str] = set()
_THIS_FILE = os.path.abspath(__file__)


def _headers_for(component: str, operation: str) -> dict[str, str]:
    """Render the attribution headers, saying so when a call site names nothing."""
    if component:
        return _attribution_headers(component=component, operation=operation)
    # Gated on a gateway actually being selected, not merely on the variable being set: a misspelled preset emits
    # nothing either, and blaming the call site for that would send someone to instrument code that is already fine.
    if _gateway_selected():
        site = _first_caller_outside_this_module()
        if site not in _UNTAGGED_SITES:
            _UNTAGGED_SITES.add(site)
            log.warning(
                "LLM call from %s names no attribution component, so its spend "
                "will roll up under none; pass component= at this call site",
                site,
            )
    return {}


def _first_caller_outside_this_module() -> str:
    """Locate the call site to report, skipping this module's own frames."""
    frame = sys._getframe(1)
    while frame is not None:
        if os.path.abspath(frame.f_code.co_filename) != _THIS_FILE:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "an unknown call site"


def _tag_request(params: dict[str, object], component: str, operation: str = "") -> dict[str, object]:
    """Merge the attribution header into an OpenAI-SDK request, in place."""
    headers = _headers_for(component, operation)
    if headers:
        existing = params.get("extra_headers") or {}
        params["extra_headers"] = {**existing, **headers}  # type: ignore[dict-item]
    return params


def _observe_chat(observation: RequestObservation, resp: object) -> ChatCompletionResult:
    """Fold one chat-completions response onto ``observation``, then flatten it."""
    result = _chat_completion_result(resp)
    observation.update(
        response_id=sdk_field(resp, "id"),
        model=sdk_field(resp, "model"),
        stop_reason=result.finish_reason,
        usage=openai_usage_counts(result.usage),
    )
    return result


def chat_completion(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> ChatCompletionResult:
    """Non-streaming chat completion; returns text, finish reason and usage."""
    with observed_request(
        protocol=PROTOCOL_OPENAI_CHAT, component=component, operation=operation, model=params.get("model")
    ) as observation:
        resp = client.chat.completions.create(**_tag_request(params, component, operation))  # type: ignore[union-attr]
        return _observe_chat(observation, resp)


async def achat_completion(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> ChatCompletionResult:
    """Async non-streaming chat completion; returns text, finish reason and usage."""
    with observed_request(
        protocol=PROTOCOL_OPENAI_CHAT, component=component, operation=operation, model=params.get("model")
    ) as observation:
        resp = await client.chat.completions.create(**_tag_request(params, component, operation))  # type: ignore[union-attr]
        return _observe_chat(observation, resp)


def _anthropic_text_from_content(content: object) -> str:
    """Concatenate the ``text`` blocks of an Anthropic Messages ``content`` array."""
    if not isinstance(content, list):
        return ""
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "".join(parts).strip()


@dataclass(frozen=True)
class AnthropicMessageResult:
    """One Anthropic Messages reply, flattened for Hyperloom callers."""

    text: str
    stop_reason: str | None
    usage: object | None


@dataclass(frozen=True)
class AnthropicCompletionRequest:
    """One single-shot Anthropic completion: the Messages body and the time budget it may take.

    ``timeout_s`` bounds the whole call on both transports. ``timeout`` is the HTTP client's (``timeout_s`` when unset):
    it bounds connecting and each wait for the next event of the streamed reply, not the reply as a whole.
    """

    model: str
    messages: Sequence[Mapping[str, object]]
    max_tokens: int
    system: str | None = None
    timeout: object | None = None
    timeout_s: float | None = None

    def http_params(self) -> dict[str, object]:
        """Assemble the Messages request body, omitting an absent system prompt."""
        params: dict[str, object] = {
            "model": self.model,
            "messages": list(self.messages),
            "max_tokens": int(self.max_tokens),
        }
        if self.system:
            params["system"] = self.system
        return params


def _attribution_tag_kwargs(component: str, operation: str = "") -> dict[str, object]:
    """Build the request keyword carrying the attribution header, if any."""
    headers = _headers_for(component, operation)
    return {"headers": headers} if headers else {}


def _observed_messages(
    component: str, operation: str, params: Mapping[str, object]
) -> AbstractContextManager[RequestObservation]:
    return observed_request(
        protocol=PROTOCOL_ANTHROPIC_MESSAGES,
        component=component,
        operation=operation,
        model=params.get("model"),
        streamed=True,
    )


def _anthropic_failure(status: int, body: bytes) -> RuntimeError:
    return RuntimeError(f"anthropic messages status={status} body={body.decode(errors='replace')[:200]}")


def _anthropic_body_result(body: Mapping[str, object]) -> AnthropicMessageResult:
    return AnthropicMessageResult(
        text=_anthropic_text_from_content(body.get("content")),
        stop_reason=body.get("stop_reason"),  # type: ignore[arg-type]
        usage=body.get("usage"),
    )


def _observe_anthropic(observation: RequestObservation, body: Mapping[str, object]) -> AnthropicMessageResult:
    """Fold one Messages reply body onto ``observation``, then flatten it."""
    result = _anthropic_body_result(body)
    observation.update(
        response_id=body.get("id"),
        model=body.get("model"),
        stop_reason=result.stop_reason,
        usage=anthropic_usage_counts(result.usage),
    )
    return result


def _dial_anthropic_messages(
    dial: ExitStack, client: object, params: dict[str, object], tag: dict[str, object], deadline: float | None = None
) -> object:
    """Open the Messages stream, dialling again while the connection never came up and ``deadline`` is still ahead."""
    # httpx re-dials only on its unproxied branch, so the re-dial lives here instead, where both routes pass.
    connect_failures = _anthropic_connect_failures()
    delay = _ANTHROPIC_CONNECT_BACKOFF_S
    body = {**params, "stream": True}
    for _ in range(ANTHROPIC_CONNECT_RETRIES):
        try:
            return dial.enter_context(
                client.stream(_ANTHROPIC_MESSAGES_METHOD, _ANTHROPIC_MESSAGES_PATH, json=body, **tag)  # type: ignore[union-attr]
            )
        except connect_failures:
            # A dial begun at or past the deadline would send a POST the reader is already bound to drop.
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise
            time.sleep(delay)
            delay *= 2
    # The last dial speaks for itself: its failure reaches the caller.
    return dial.enter_context(
        client.stream(_ANTHROPIC_MESSAGES_METHOD, _ANTHROPIC_MESSAGES_PATH, json=body, **tag)  # type: ignore[union-attr]
    )


async def _adial_anthropic_messages(
    dial: AsyncExitStack, client: object, params: dict[str, object], tag: dict[str, object]
) -> object:
    """Async twin of :func:`_dial_anthropic_messages`."""
    connect_failures = _anthropic_connect_failures()
    delay = _ANTHROPIC_CONNECT_BACKOFF_S
    body = {**params, "stream": True}
    for _ in range(ANTHROPIC_CONNECT_RETRIES):
        try:
            return await dial.enter_async_context(
                client.stream(_ANTHROPIC_MESSAGES_METHOD, _ANTHROPIC_MESSAGES_PATH, json=body, **tag)  # type: ignore[union-attr]
            )
        except connect_failures:
            await asyncio.sleep(delay)
            delay *= 2
    return await dial.enter_async_context(
        client.stream(_ANTHROPIC_MESSAGES_METHOD, _ANTHROPIC_MESSAGES_PATH, json=body, **tag)  # type: ignore[union-attr]
    )


def anthropic_messages(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    timeout_s: float | None = None,
    **params: object,
) -> AnthropicMessageResult:
    """POST one Anthropic Messages request and read its reply as server-sent events; returns text, stop_reason and
    usage. A streamed reply starts while it is written, so a gateway that cuts a request whose reply has not come does
    not cut a long one; ``timeout_s`` bounds the whole reply."""
    tag = _attribution_tag_kwargs(component, operation)
    # Inside the observation: the re-dials belong to the one request the ledger records.
    with _observed_messages(component, operation, params) as observation:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        stream = AnthropicStream(observation, deadline)
        with ExitStack() as dial:
            resp = _dial_anthropic_messages(dial, client, params, tag, deadline)
            if resp.status_code >= 400:
                raise _anthropic_failure(resp.status_code, resp.read())
            for chunk in resp.iter_bytes():
                stream.feed(chunk)
        return _observe_anthropic(observation, stream.finish())


async def aanthropic_messages(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    timeout_s: float | None = None,
    **params: object,
) -> AnthropicMessageResult:
    """Async twin of :func:`anthropic_messages`; see it for the full contract."""
    tag = _attribution_tag_kwargs(component, operation)

    async def read() -> AnthropicMessageResult:
        with _observed_messages(component, operation, params) as observation:
            stream = AnthropicStream(observation)
            async with AsyncExitStack() as dial:
                resp = await _adial_anthropic_messages(dial, client, params, tag)
                if resp.status_code >= 400:
                    raise _anthropic_failure(resp.status_code, await resp.aread())
                async for chunk in resp.aiter_bytes():
                    stream.feed(chunk)
            return _observe_anthropic(observation, stream.finish())

    return await asyncio.wait_for(read(), timeout_s)


# Single-shot Anthropic transports. "http" is the Messages API; "sdk" drives the Claude CLI, the only channel that
# accepts a Max/Pro subscription token.
ANTHROPIC_TRANSPORT_HTTP = "http"
ANTHROPIC_TRANSPORT_SDK = "sdk"

_ANTHROPIC_NO_CREDENTIAL = (
    f"no Anthropic credential configured: set one of {' / '.join(ANTHROPIC_CREDENTIAL_ENV_ORDER)}"
)


def anthropic_transport(env: Mapping[str, str] | None = None) -> str:
    """Pick the single-shot Anthropic transport the credential can authenticate."""
    source = env if env is not None else os.environ
    if anthropic_synthesizable_key(source):
        return ANTHROPIC_TRANSPORT_HTTP
    if (source.get(CLAUDE_OAUTH_TOKEN_ENV) or "").strip():
        return ANTHROPIC_TRANSPORT_SDK
    return ""


def anthropic_transport_ready(env: Mapping[str, str] | None = None) -> bool:
    """Whether a single-shot Anthropic call could actually be issued right now."""
    transport = anthropic_transport(env)
    if not transport:
        return False
    if transport == ANTHROPIC_TRANSPORT_SDK:
        from .claude_oneshot import ensure_available

        try:
            ensure_available()
        except RuntimeError:
            return False
    return True


def _one_shot_client(
    timeout_s: float | None,
    env: Mapping[str, str] | None,
    component: str = "",
    operation: str = "",
) -> object:
    """Build the Claude CLI one-shot client, imported late to avoid a cycle."""
    from .claude_oneshot import ClaudeOneShotClient

    if timeout_s is None:
        return ClaudeOneShotClient(env=env, component=component, operation=operation)
    return ClaudeOneShotClient(
        timeout_s=float(timeout_s),
        env=env,
        component=component,
        operation=operation,
    )


def anthropic_completion(
    *,
    request: AnthropicCompletionRequest,
    env: Mapping[str, str] | None = None,
    component: str = "",
    operation: str = "",
) -> AnthropicMessageResult:
    """Issue one single-shot Anthropic completion over whichever transport works."""
    transport = anthropic_transport(env)
    if transport == ANTHROPIC_TRANSPORT_SDK:
        client = _one_shot_client(request.timeout_s, env, component, operation)
        return client.messages(  # type: ignore[attr-defined]
            model=request.model,
            messages=request.messages,
            system=request.system,
            max_tokens=request.max_tokens,
        )
    if not transport:
        raise LLMConfigError(_ANTHROPIC_NO_CREDENTIAL)
    http_client = get_anthropic_client(
        env=dict(env) if env is not None else None, timeout=request.timeout or request.timeout_s
    )
    with http_client:  # type: ignore[attr-defined]
        return anthropic_messages(
            http_client,
            component=component,
            operation=operation,
            timeout_s=request.timeout_s,
            **request.http_params(),
        )


async def aanthropic_completion(
    *,
    request: AnthropicCompletionRequest,
    env: Mapping[str, str] | None = None,
    component: str = "",
    operation: str = "",
) -> AnthropicMessageResult:
    """Async twin of :func:`anthropic_completion`; see it for the full contract."""
    transport = anthropic_transport(env)
    if transport == ANTHROPIC_TRANSPORT_SDK:
        client = _one_shot_client(request.timeout_s, env, component, operation)
        return await client.amessages(  # type: ignore[attr-defined]
            model=request.model,
            messages=request.messages,
            system=request.system,
            max_tokens=request.max_tokens,
        )
    if not transport:
        raise LLMConfigError(_ANTHROPIC_NO_CREDENTIAL)
    http_client = get_async_anthropic_client(
        env=dict(env) if env is not None else None, timeout=request.timeout or request.timeout_s
    )
    async with http_client:  # type: ignore[attr-defined]
        return await aanthropic_messages(
            http_client,
            component=component,
            operation=operation,
            timeout_s=request.timeout_s,
            **request.http_params(),
        )


async def astream_chat_completion_text(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> "tuple[str, object | None]":
    """Async streamed chat completion; returns ``(text, usage)``."""
    params["stream"] = True
    params["stream_options"] = {"include_usage": True}
    parts: list[str] = []
    usage_obj: object | None = None
    with observed_request(
        protocol=PROTOCOL_OPENAI_CHAT,
        component=component,
        operation=operation,
        model=params.get("model"),
        streamed=True,
    ) as observation:
        stream = await client.chat.completions.create(**_tag_request(params, component, operation))  # type: ignore[union-attr]
        async for chunk in stream:
            _observe_stream_chunk(observation, chunk)
            if getattr(chunk, "usage", None) is not None:
                usage_obj = chunk.usage
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta is not None and delta.content:
                    parts.append(delta.content)
        observation.update(usage=openai_usage_counts(usage_obj))
    return "".join(parts), usage_obj


def _observe_stream_chunk(observation: RequestObservation, chunk: object) -> None:
    """Fold one streamed chunk's id, model, first content token and finish reason onto ``observation``."""
    choices = sdk_field(chunk, "choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else None
    delta = sdk_field(choice, "delta") if choice is not None else None
    if delta is not None and sdk_field(delta, "content"):
        observation.mark_first_token()
    observation.update(
        response_id=sdk_field(chunk, "id"),
        model=sdk_field(chunk, "model"),
        stop_reason=sdk_field(choice, "finish_reason") if choice is not None else None,
    )


__all__ = [
    "AGENT_BACKEND_CHOICES",
    "AGENT_BACKEND_CLAUDE",
    "AGENT_BACKEND_CODEX",
    "AGENT_BACKEND_DEEPSEEK",
    "AGENT_BACKEND_ENV",
    "ANTHROPIC_CONNECT_RETRIES",
    "ANTHROPIC_CREDENTIAL_ENV_ORDER",
    "ANTHROPIC_SYNTHESIZABLE_KEY_ENVS",
    "ANTHROPIC_TRANSPORT_HTTP",
    "ANTHROPIC_TRANSPORT_SDK",
    "AnthropicCompletionRequest",
    "AnthropicMessageResult",
    "CLAUDE_GATEWAY_SIGNAL_KEYS",
    "CLAUDE_OAUTH_TOKEN_ENV",
    "ChatCompletionResult",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_ANTHROPIC_VERSION",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_DEEPSEEK_HARNESS_MODEL",
    "LEGACY_DEEPSEEK_ENV_KEYS",
    "LLMConfigError",
    "OpenAIClientConfig",
    "UNRUNNABLE_AGENT_RANK",
    "aanthropic_completion",
    "aanthropic_messages",
    "achat_completion",
    "agent_backend_rank",
    "anthropic_completion",
    "anthropic_messages",
    "anthropic_agent_credentialed",
    "anthropic_synthesizable_key",
    "anthropic_transport",
    "anthropic_transport_ready",
    "apply_reasoning_effort",
    "astream_chat_completion_text",
    "build_http_timeout",
    "chat_completion",
    "claude_agent_sdk_installed",
    "claude_sdk_env_options",
    "codex_agent_sdk_installed",
    "deepseek_compat_env",
    "deepseek_harness_sdk_installed",
    "derive_openai_base_url",
    "dual_protocol_endpoint_pair",
    "endpoint_default_model",
    "get_anthropic_client",
    "get_async_anthropic_client",
    "get_async_openai_client",
    "get_openai_client",
    "has_anthropic_credential",
    "has_anthropic_side",
    "has_openai_side",
    "is_anthropic_only",
    "is_openai_only",
    "openai_agent_credentialed",
    "preferred_agent_backend",
    "provider_model_defaults",
    "explicit_agent_backend",
    "resolve_agent_backend",
    "resolve_forge_llm_model",
    "resolve_openai_client_config",
    "with_synthesized_anthropic_keys",
]
