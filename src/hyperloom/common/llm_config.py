# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM gateway ownership: env resolution *and* client construction.

:func:`get_openai_client` / :func:`get_async_openai_client` and their Anthropic
counterparts :func:`get_anthropic_client` / :func:`get_async_anthropic_client`
are the only sanctioned places in Hyperloom that instantiate an LLM transport.
Callers ask this module for a client instead of reading credentials themselves,
so provider selection stays one decision rather than one per call site. The
matching single-shot helpers (:func:`achat_completion`, :func:`aresponse`,
:func:`anthropic_messages` and their twins) own the request shapes, so no caller
hand-builds a payload or an endpoint path either.

Single-turn Anthropic work goes through :func:`anthropic_completion` /
:func:`aanthropic_completion`, which additionally own *transport* selection: the
Messages API rejects a Claude Max/Pro subscription token, so that credential is
served by a tool-free single-turn Claude CLI session instead. Both transports
return the same :class:`AnthropicMessageResult`, so a call site does not encode
which credential shape a deployment carries.

Hyperloom serves three deployment shapes — Anthropic-only, OpenAI-compatible
only, and both. Explicit configuration for a protocol always wins; when the
OpenAI side is absent, :func:`resolve_openai_client_config` falls back to the
Anthropic gateway so an Anthropic-only deployment can still drive the
OpenAI-protocol call sites.

Clients are deliberately **not** cached or shared. Both transports wrap an
``httpx.AsyncClient`` whose connection pool binds to the event loop that first
uses it, and Hyperloom starts a fresh loop per run / phase, so a process-wide
cache would hand a second loop a pool it cannot use. Credentials are also not
stable for the life of the process: preflight rewrites the provider env, and
callers may pass an explicit ``env`` mapping. Owning objects (``CodexBackend``,
``ProposalScorer``, ``CriticAgentBackend``) already hold one client each for
their own lifetime, which is the scope that actually matches an event loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from hyperloom.common.coerce import to_int as _to_int
from hyperloom.common.llm_attribution import call_headers as _attribution_headers
from hyperloom.common.llm_attribution import gateway_selected as _gateway_selected
from hyperloom.common.llm_attribution import inject_env as _inject_attribution_env

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

    def as_kwargs(self) -> dict[str, object]:
        """Return kwargs accepted by ``openai.AsyncOpenAI``."""
        kwargs: dict[str, object] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = dict(self.default_headers)
        return kwargs


# Subscription credential minted by ``claude setup-token``. Never synthesized
# into the API-key vars; see ``claude_sdk_env_options``.
CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

# Single source of truth for "what counts as an Anthropic-side credential",
# highest precedence first. Mirrors the Claude CLI's own resolution: an API key
# or gateway bearer token disables subscription (OAuth) mode entirely, so OAuth
# is only live when both are unset. Detection sites derive from this tuple via
# has_anthropic_credential(); adding a form here makes them all recognize it.
ANTHROPIC_CREDENTIAL_ENV_ORDER: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    CLAUDE_OAUTH_TOKEN_ENV,
)

# The subset whose value may be copied into another credential var or persisted
# to ~/.claude/config.json. Enumerated explicitly rather than derived by
# subtraction: a newly recognized credential form must be reviewed before it may
# be materialized, since anything that outranks a subscription token moves the
# run onto API-credits billing. A test asserts this stays a subset that excludes
# CLAUDE_OAUTH_TOKEN_ENV.
ANTHROPIC_SYNTHESIZABLE_KEY_ENVS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
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
    """Highest-precedence credential that may be copied elsewhere.

    Returns "" when a subscription token is the only credential available, which
    is what keeps OAuth out of API-key slots and out of config.json.
    """
    return _first_set_value(
        ANTHROPIC_SYNTHESIZABLE_KEY_ENVS,
        env if env is not None else os.environ,
    )


CLAUDE_GATEWAY_SIGNAL_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
    CLAUDE_OAUTH_TOKEN_ENV,
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "LLM_GATEWAY_KEY",
)

# Retired provider-specific variables. DeepSeek is a dual-protocol gateway, not
# a third provider, so it is expressed with the standard Anthropic + OpenAI
# variables; these are read only by ``deepseek_compat_env`` to migrate an
# existing configuration.
LEGACY_DEEPSEEK_ENV_KEYS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
)

_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
_DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_MODEL = "deepseek-v4-pro"

# Hosts known to serve the Anthropic protocol on ``/anthropic`` and OpenAI
# chat-completions on ``/v1``. Needed because the generic AMD gateway
# convention (``/Unified/v1``) does not exist on them.
_DUAL_PROTOCOL_HOSTS: frozenset[str] = frozenset({"api.deepseek.com"})

# Gateways that serve only their own models. Without a default the run would
# ask them for an AMD Claude / OpenAI id and fail on the first LLM call, since
# their catalog cannot be probed for a model the host does not have.
_HOST_DEFAULT_MODELS: dict[str, str] = {"api.deepseek.com": _DEEPSEEK_MODEL}

# The retired variables are Anthropic-side shaped, so they are adopted only
# when that whole side is empty -- and then for BOTH protocols, since they
# describe one gateway. Adopting them piecemeal next to an existing Anthropic
# credential would send that credential to DeepSeek's host.
# Derived from the credential registry so a newly registered form makes the
# side detectable everywhere at once, subscription tokens included.
_ANTHROPIC_SIDE_KEYS: tuple[str, ...] = ("ANTHROPIC_BASE_URL", *ANTHROPIC_CREDENTIAL_ENV_ORDER)
_OPENAI_SIDE_KEYS: tuple[str, ...] = ("OPENAI_BASE_URL", "OPENAI_API_KEY")


def has_anthropic_side(env: Mapping[str, str] | None = None) -> bool:
    """True when an Anthropic-side endpoint or key is configured."""
    source = env if env is not None else os.environ
    return any((source.get(name) or "").strip() for name in _ANTHROPIC_SIDE_KEYS)


def has_openai_side(env: Mapping[str, str] | None = None) -> bool:
    """True when an OpenAI-side endpoint or key is configured."""
    source = env if env is not None else os.environ
    return any((source.get(name) or "").strip() for name in _OPENAI_SIDE_KEYS)


def is_anthropic_only(env: Mapping[str, str] | None = None) -> bool:
    """True when the Anthropic side is the only configured provider.

    The canonical credential-shape test, so that backend selection, the TraceLens
    runner and the forge kernel backend cannot disagree about which shape they are in.
    Retired provider variables (``DEEPSEEK_*``) are deliberately not consulted:
    :func:`deepseek_compat_env` migrates those onto the standard pair first.
    """
    return has_anthropic_side(env) and not has_openai_side(env)


def is_openai_only(env: Mapping[str, str] | None = None) -> bool:
    """True when the OpenAI side is the only configured provider.

    The counterpart of :func:`is_anthropic_only`; see it for why the retired
    DeepSeek variables are out of scope.
    """
    return has_openai_side(env) and not has_anthropic_side(env)


DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The Anthropic Messages API version, defined once for the whole repository.
# Per-deployment overrides go through ANTHROPIC_CUSTOM_HEADERS, which is merged
# over the defaults when a client is built.
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _is_dual_protocol_host(url: str | None) -> bool:
    """True when ``url`` points at a known ``/anthropic`` + ``/v1`` gateway."""
    if not url:
        return False
    return urlsplit(str(url).strip()).hostname in _DUAL_PROTOCOL_HOSTS


def dual_protocol_endpoint_pair(base_url: str) -> tuple[str, str]:
    """Return ``(anthropic_base_url, openai_base_url)`` for a dual-protocol gateway.

    Segment swapping is only performed for hosts listed in
    ``_DUAL_PROTOCOL_HOSTS`` (currently DeepSeek).  For those hosts the
    ``/anthropic`` and ``/v1`` sibling paths are real routes; fabricating them
    for unknown hosts would point callers at endpoints that may not exist.

    Matching is case-insensitive because AMD's gateway spells the segment
    ``/Anthropic`` (issue #929).

    A bare known-dual-protocol host gets both segments appended. An unknown host
    with no recognized suffix gets its Anthropic URL as-is and an OpenAI URL
    with ``/v1`` appended. ``assets/install_baremetal.sh`` mirrors this for the
    pre-Python bootstrap.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        return _DEEPSEEK_ANTHROPIC_BASE_URL, _DEEPSEEK_OPENAI_BASE_URL
    lowered = base.lower()
    # Swap the trailing path segment when a recognized protocol suffix is present.
    # This works for any host: an operator naming one side implies the other exists.
    if lowered.endswith("/anthropic"):
        return base, f"{base.rsplit('/', 1)[0]}/v1"
    if lowered.endswith("/v1"):
        return f"{base.rsplit('/', 1)[0]}/anthropic", base
    # No recognized protocol suffix.  Known-dual-protocol hosts get both segments
    # appended; unknown hosts get /v1 appended to the OpenAI side.
    if _is_dual_protocol_host(base):
        parsed = urlsplit(base)
        if not parsed.path or parsed.path == "/":
            return f"{base}/anthropic", f"{base}/v1"
    return base, f"{base}/v1"


def deepseek_compat_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Translate a legacy ``DEEPSEEK_*`` configuration into the standard variables.

    DeepSeek serves both protocols from one gateway: ``/anthropic`` speaks the
    Anthropic API and ``/v1`` speaks OpenAI chat-completions, both authenticated
    with the same key. Expressing it as a third provider forced every caller to
    special-case it, so the runtime now only knows the Anthropic and OpenAI
    sides and this function is the single place that understands the retired
    variables.

    The gateway is adopted whole or not at all: when anything on the Anthropic
    side is already configured the retired variables are stale leftovers and
    are ignored completely. Half-adopting them would pair an explicit Anthropic
    credential with DeepSeek's endpoint, or invent an OpenAI side the operator
    never asked for. Beyond that, only keys absent from ``env`` are returned,
    so an explicit value always wins and repeated application is a no-op.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        The ``ANTHROPIC_*`` / ``OPENAI_*`` / model updates to apply, or an empty
        mapping when no legacy DeepSeek variable is set or the Anthropic side is
        already configured.
    """
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
        # GEAKv4 drives Claude Code with its own model variable, so it follows
        # whichever Claude model is actually in effect -- an explicit
        # CLAUDE_MODEL must not be contradicted by the gateway default.
        "GEAK_CLAUDE_MODEL": (source.get("CLAUDE_MODEL") or "").strip() or model,
    }
    if api_key:
        # Two spellings of one credential: x-api-key and bearer.
        candidates["ANTHROPIC_API_KEY"] = api_key
        candidates["ANTHROPIC_AUTH_TOKEN"] = api_key
    # The OpenAI side is adopted only when it is entirely free. Otherwise the
    # operator already runs some other gateway there, and neither its key nor
    # its model may be replaced with DeepSeek's.
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
    """Return the model variables implied by the configured endpoints.

    A gateway serving only its own models (DeepSeek) has to supply the model
    ids too: the AMD Claude / OpenAI defaults would be sent verbatim and fail on
    the first call, and its catalog cannot confirm a model it does not host.
    Retired ``DEEPSEEK_*`` config is normalized first so both spellings resolve
    identically.

    Only keys absent from ``env`` are returned, so an explicit operator model
    always wins. ``GEAK_CLAUDE_MODEL`` follows the Anthropic-side model, since
    GEAKv4 drives Claude Code over that side.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        The ``CLAUDE_MODEL`` / ``CODEX_MODEL`` / ``GEAK_CLAUDE_MODEL`` updates to
        apply, empty when the endpoints imply nothing.
    """
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
    default: str = "",
) -> str:
    """Resolve the Forge LLM model id for a chosen agent backend.

    Shared by forge-fusion, forge rewrite (``forge_submit``), and forge-collective
    so every Forge LLM surface honors the same precedence:

    1. ``explicit`` (request ``llm_model``);
    2. forge-specific env (``FORGE_CLAUDE_MODEL`` / ``FORGE_CODEX_MODEL``), the
       forge counterpart of ``GEAK_CLAUDE_MODEL``;
    3. orchestration-side ``CLAUDE_MODEL`` / ``CODEX_MODEL``;
    4. ``default`` (callers that always need a concrete id pass their built-in
       default; callers that defer to KernelForge omit it and skip ``--model``).

    Args:
        agent_backend: ``"claude"`` or ``"codex"`` (other values use the Claude
            env ladder, matching KernelForge's Claude default for ``auto``).
        env: Environment mapping to read; defaults to ``os.environ``.
        explicit: Request-level model override, typically ``payload["llm_model"]``.
        default: Fallback when no env model is set.

    Returns:
        The resolved model id, or ``default`` / ``""`` when nothing is configured.
    """
    source = env if env is not None else os.environ
    explicit_model = (explicit or "").strip()
    if explicit_model:
        return explicit_model
    backend = (agent_backend or "").strip().lower()
    if backend == "codex":
        return (
            str(source.get("FORGE_CODEX_MODEL") or "").strip()
            or str(source.get("CODEX_MODEL") or "").strip()
            or default
        )
    return (
        str(source.get("FORGE_CLAUDE_MODEL") or "").strip() or str(source.get("CLAUDE_MODEL") or "").strip() or default
    )


def _expand_env_refs(raw: str, env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ

    def repl(match: re.Match[str]) -> str:
        return str(source.get(match.group(1), ""))

    return _ENV_REF_RE.sub(repl, raw)


def parse_custom_headers(raw: str | None, *, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse custom LLM headers from env.

    ``ANTHROPIC_CUSTOM_HEADERS`` is newline-delimited ``Name: value`` in the
    Anthropic SDK. JSON object input is accepted as a convenience for launchers
    that already store structured environment values. Shell-style ``${VAR}``
    references are expanded from the supplied env mapping so .env files can keep
    one copy of a secret and derive gateway headers from it.
    """
    if not raw:
        return {}
    expanded = _expand_env_refs(raw, env)
    text = expanded.strip()
    if not text:
        return {}
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                return {str(k).strip(): str(v).strip() for k, v in parsed.items() if str(k).strip()}
            return {}

    headers: dict[str, str] = {}
    for line in expanded.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    return headers


def derive_openai_base_url(anthropic_base_url: str | None) -> str | None:
    """Derive an OpenAI-compatible base URL from an Anthropic endpoint.

    :func:`resolve_openai_client_config` calls this only after explicit
    OpenAI-side configuration comes up empty, which is what lets an
    Anthropic-only deployment reach an OpenAI-compatible endpoint at all. It
    handles AMD's split gateway convention, where Anthropic traffic uses
    ``/anthropic`` and OpenAI-compatible chat completions use ``/Unified/v1``.
    The trailing path segment is matched case-insensitively so a capitalized
    ``/Anthropic`` (AMD's default) is still recognized. Unknown Anthropic URLs
    fall back to their original value.
    """
    if not anthropic_base_url:
        return None
    base = anthropic_base_url.strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    path = parts.path.rstrip("/")
    # Match case-insensitively: AMD's default endpoint uses "/Anthropic" (issue #929).
    # Keep the original path for slicing so any prefix casing is preserved, and
    # always emit the canonical "/Unified/v1" segment.
    path_lower = path.lower()
    if path_lower.endswith("/anthropic"):
        prefix = path[: -len("/anthropic")]
        return urlunsplit(parts._replace(path=f"{prefix}/Unified/v1"))
    if path_lower.endswith("/unified"):
        prefix = path[: -len("/unified")]
        return urlunsplit(parts._replace(path=f"{prefix}/Unified/v1"))
    return base


def resolve_openai_client_config(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> OpenAIClientConfig:
    """Resolve OpenAI-compatible client config from one or more LLM env sets.

    Explicit OpenAI-side configuration always wins, so codex-only and
    dual-configured deployments resolve exactly as they did before the Anthropic
    fallback existed. Only once the OpenAI side comes up empty is the Anthropic
    side consulted: the base URL is derived with :func:`derive_openai_base_url`,
    the key falls back to the Anthropic gateway tokens, and -- because that derived
    URL is the same gateway host -- so do its custom headers. That is what lets the
    single-shot OpenAI-protocol call sites (proposal scorer, framework ranker,
    audit refinement) work in an Anthropic-only deployment instead of failing to
    configure.
    """
    source = env if env is not None else os.environ
    api_key = (
        (source.get(api_key_env) or "").strip()
        or (source.get("OPENAI_API_KEY") or "").strip()
        or (source.get("LLM_GATEWAY_KEY") or "").strip()
        # Anthropic-only deployments: one gateway token authenticates both protocols.
        or (source.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (source.get("ANTHROPIC_API_KEY") or "").strip()
    )
    if not api_key:
        key_names = " / ".join(
            dict.fromkeys(
                [
                    api_key_env,
                    "OPENAI_API_KEY",
                    "LLM_GATEWAY_KEY",
                    "ANTHROPIC_AUTH_TOKEN",
                    "ANTHROPIC_API_KEY",
                ]
            )
        )
        raise LLMConfigError(f"{key_names} not set in env (OpenAI-compatible client cannot auth)")

    explicit_base_url = (source.get(base_url_env) or "").strip() or (source.get("OPENAI_BASE_URL") or "").strip()
    derived_base_url = (derive_openai_base_url(source.get("ANTHROPIC_BASE_URL")) or "").strip()
    base_url = explicit_base_url or derived_base_url or None

    # Gateway headers are operator-supplied, and they belong to an endpoint rather
    # than to a protocol: AMD's gateway rejects a call without its subscription
    # header, and that header is only ever written to ANTHROPIC_CUSTOM_HEADERS.
    # So the Anthropic side is consulted exactly when the base URL was derived
    # from it -- an explicit OPENAI_BASE_URL may well be a different host, whose
    # headers we must not guess at.
    headers = parse_custom_headers(source.get("OPENAI_CUSTOM_HEADERS"), env=source)
    if not headers and not explicit_base_url and derived_base_url:
        headers = parse_custom_headers(source.get("ANTHROPIC_CUSTOM_HEADERS"), env=source)
    return OpenAIClientConfig(api_key=api_key, base_url=base_url, default_headers=headers)


def openai_client_kwargs(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return kwargs accepted by ``openai.AsyncOpenAI``."""
    return resolve_openai_client_config(api_key_env=api_key_env, base_url_env=base_url_env, env=env).as_kwargs()


def build_http_timeout(
    *,
    connect: float,
    read: float,
    write: float | None = None,
    pool: float | None = None,
) -> object | None:
    """Build an ``httpx.Timeout`` to hand to :func:`get_openai_client`.

    ``write`` and ``pool`` default to ``read``. Returns ``None`` — logging a
    warning — when ``httpx`` is not importable, so the caller still gets a
    working client on the SDK's default timeouts instead of failing to construct
    one at all. A timeout ``httpx`` itself rejects is a caller bug and raises:
    silently dropping it would turn a bad knob into an unbounded stall.

    Args:
        connect: Connect-phase timeout in seconds.
        read: Read-phase timeout in seconds.
        write: Write-phase timeout in seconds; defaults to ``read``.
        pool: Pool-acquire timeout in seconds; defaults to ``read``.

    Returns:
        An ``httpx.Timeout``, or ``None`` when ``httpx`` is unavailable.
    """
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
    """Resolve one OpenAI SDK client class, importing the SDK on demand.

    ``openai`` is imported lazily so importing this module (a dependency of the
    whole orchestrator) does not pay the SDK import cost.

    Args:
        class_name: SDK attribute to read, ``"OpenAI"`` or ``"AsyncOpenAI"``.

    Returns:
        The requested SDK client class.

    Raises:
        LLMConfigError: If the ``openai`` package is not installed.
    """
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LLMConfigError(_OPENAI_SDK_MISSING) from exc
    return getattr(openai, class_name)


def _openai_sdk_kwargs(
    *,
    api_key_env: str,
    base_url_env: str,
    env: dict[str, str] | None,
    timeout: object | None,
) -> dict[str, object]:
    """Resolve credentials and fold in an optional per-caller ``timeout``."""
    kwargs = openai_client_kwargs(api_key_env=api_key_env, base_url_env=base_url_env, env=env)
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def get_openai_client(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the synchronous ``openai.OpenAI`` client.

    One of the two sanctioned client constructors (see the module docstring):
    credentials, base URL and gateway headers all come from
    :func:`resolve_openai_client_config`, so no caller reads provider env itself.

    Args:
        api_key_env: Env var consulted first for the API key.
        base_url_env: Env var consulted first for the base URL.
        env: Env mapping to read instead of ``os.environ``.
        timeout: Transport timeout forwarded to the SDK verbatim; build one with
            :func:`build_http_timeout`. ``None`` keeps the SDK default.

    Returns:
        A new ``openai.OpenAI``; clients are never cached (module docstring).

    Raises:
        LLMConfigError: If the ``openai`` package is missing, or no OpenAI-side
            API key is configured.
    """
    factory = _openai_sdk_client_class("OpenAI")
    return factory(  # type: ignore[operator]
        **_openai_sdk_kwargs(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            env=env,
            timeout=timeout,
        )
    )


def get_async_openai_client(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the asynchronous ``openai.AsyncOpenAI`` client.

    The async twin of :func:`get_openai_client`; see it for the argument
    contract and the module docstring for why clients are not shared.

    Args:
        api_key_env: Env var consulted first for the API key.
        base_url_env: Env var consulted first for the base URL.
        env: Env mapping to read instead of ``os.environ``.
        timeout: Transport timeout forwarded to the SDK verbatim; build one with
            :func:`build_http_timeout`. ``None`` keeps the SDK default.

    Returns:
        A new ``openai.AsyncOpenAI``; clients are never cached.

    Raises:
        LLMConfigError: If the ``openai`` package is missing, or no OpenAI-side
            API key is configured.
    """
    factory = _openai_sdk_client_class("AsyncOpenAI")
    return factory(  # type: ignore[operator]
        **_openai_sdk_kwargs(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            env=env,
            timeout=timeout,
        )
    )


def _httpx_module() -> object:
    """Import ``httpx`` on demand, keeping this module cheap to import.

    Raises:
        LLMConfigError: If ``httpx`` is not installed.
    """
    try:
        import httpx
    except ImportError as exc:
        raise LLMConfigError(_HTTPX_MISSING) from exc
    return httpx


def _anthropic_client_kwargs(
    *,
    api_key_env: str,
    base_url_env: str,
    env: dict[str, str] | None,
    timeout: object | None,
) -> dict[str, object]:
    """Resolve base URL, auth and gateway headers for an Anthropic HTTP client.

    Raises:
        LLMConfigError: If no Anthropic-side API key is configured.
    """
    source = env if env is not None else os.environ
    api_key = (
        (source.get(api_key_env) or "").strip()
        or (source.get("ANTHROPIC_API_KEY") or "").strip()
        or (source.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    )
    if not api_key:
        key_names = " / ".join(dict.fromkeys([api_key_env, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]))
        raise LLMConfigError(f"{key_names} not set in env (Anthropic client cannot auth)")

    base_url = (
        (source.get(base_url_env) or "").strip()
        or (source.get("ANTHROPIC_BASE_URL") or "").strip()
        or DEFAULT_ANTHROPIC_BASE_URL
    ).rstrip("/")
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
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url_env: str = "ANTHROPIC_BASE_URL",
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the synchronous ``httpx.Client`` for the Anthropic Messages API.

    The Anthropic counterpart of :func:`get_openai_client`. Anthropic ships no
    Hyperloom-facing SDK, so the transport is a plain HTTP client preloaded with
    the base URL, ``x-api-key``, ``anthropic-version`` and any
    ``ANTHROPIC_CUSTOM_HEADERS`` the gateway requires; pair it with
    :func:`anthropic_messages` rather than posting by hand.

    Args:
        api_key_env: Env var consulted first for the API key.
        base_url_env: Env var consulted first for the base URL.
        env: Env mapping to read instead of ``os.environ``.
        timeout: Transport timeout forwarded to ``httpx`` verbatim; build one
            with :func:`build_http_timeout`. ``None`` keeps the httpx default.

    Returns:
        A new ``httpx.Client``; clients are never cached (module docstring).

    Raises:
        LLMConfigError: If ``httpx`` is missing, or no Anthropic-side API key is
            configured.
    """
    httpx = _httpx_module()
    return httpx.Client(  # type: ignore[attr-defined]
        **_anthropic_client_kwargs(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            env=env,
            timeout=timeout,
        )
    )


def get_async_anthropic_client(
    *,
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url_env: str = "ANTHROPIC_BASE_URL",
    env: dict[str, str] | None = None,
    timeout: object | None = None,
) -> object:
    """Construct the asynchronous ``httpx.AsyncClient`` for the Messages API.

    The async twin of :func:`get_anthropic_client`; see it for the argument
    contract and the module docstring for why clients are not shared.

    Args:
        api_key_env: Env var consulted first for the API key.
        base_url_env: Env var consulted first for the base URL.
        env: Env mapping to read instead of ``os.environ``.
        timeout: Transport timeout forwarded to ``httpx`` verbatim; build one
            with :func:`build_http_timeout`. ``None`` keeps the httpx default.

    Returns:
        A new ``httpx.AsyncClient``; clients are never cached.

    Raises:
        LLMConfigError: If ``httpx`` is missing, or no Anthropic-side API key is
            configured.
    """
    httpx = _httpx_module()
    return httpx.AsyncClient(  # type: ignore[attr-defined]
        **_anthropic_client_kwargs(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
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
    """Return Claude SDK options that isolate a run from global Claude config.

    Claude Code and ``claude_agent_sdk`` can read ``~/.claude/settings.json``.
    Hyperloom runs, however, may intentionally point at a per-run provider or
    gateway. When any LLM-provider signal is present in the current environment,
    pass an explicit child environment and disable settings sources so global
    developer-machine configuration cannot override the run contract.

    Naming a ``component`` also tags the child's calls for the gateway. That
    only applies on the gateway-signal path: without a signal the child inherits
    this process's environment untouched, and a deployment with no gateway has
    nothing to attribute the spend to anyway.

    Args:
        model: Model to pin for the child, or ``None`` to leave it unset.
        env: Environment to derive the child environment from.
        component: Producer label for the child's calls; ``""`` disables
            attribution tagging.
        operation: What the child is being spawned to do.
        **attribution: Additional attribution fields (e.g. ``task_id``).
    """
    source = dict(env if env is not None else os.environ)
    # Callers that never reach CLI preflight (library-mode backends) may still
    # carry a legacy DeepSeek configuration; normalize before probing signals.
    source.update(deepseek_compat_env(source))
    if not any((source.get(key) or "").strip() for key in CLAUDE_GATEWAY_SIGNAL_KEYS):
        return {}

    # Anthropic-side credentials only, and only the synthesizable subset: an
    # OAuth token copied into either API-key var would drop the CLI out of
    # subscription mode and 401 the run.
    fallback_key = anthropic_synthesizable_key(source)
    if fallback_key:
        source.setdefault("ANTHROPIC_API_KEY", fallback_key)
        source.setdefault("ANTHROPIC_AUTH_TOKEN", fallback_key)
    # Claude/Anthropic side reads only ANTHROPIC_CUSTOM_HEADERS.
    if source.get("ANTHROPIC_CUSTOM_HEADERS"):
        source["ANTHROPIC_CUSTOM_HEADERS"] = _expand_env_refs(source["ANTHROPIC_CUSTOM_HEADERS"], source)
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

    Sets ``params["reasoning_effort"]`` only when ``HYPERLOOM_REASONING_EFFORT``
    (or ``OPENAI_REASONING_EFFORT``) is a recognized value. No-op otherwise, so
    non-reasoning models and gateways that reject the field are unaffected.
    Mutates and returns ``params``.
    """
    source = env if env is not None else os.environ
    val = (source.get("HYPERLOOM_REASONING_EFFORT") or source.get("OPENAI_REASONING_EFFORT") or "").strip().lower()
    if val in {"minimal", "low", "medium", "high"}:
        params["reasoning_effort"] = val
    return params


@dataclass(frozen=True)
class ChatCompletionResult:
    """One non-streaming chat completion, flattened for Hyperloom callers.

    Attributes:
        text: The assistant message content, ``""`` when the model sent none.
        finish_reason: The chosen completion's ``finish_reason``, when present.
        usage: The SDK ``usage`` object verbatim — providers disagree on which
            counters they report, so mapping it stays with the caller that owns
            the token accounting.
    """

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
    """Render the attribution headers, saying so when a call site names nothing.

    An empty ``component`` is how a call site that was never instrumented looks
    from here, and skipping it quietly is what let several of them reach
    production: the spend simply arrives under no component and nothing in the
    logs connects it to the code that made it. Reported once per call site, and
    only when a gateway is actually selected -- a deployment that emits no
    attribution at all has nothing to fix.

    Args:
        component: Producer label for the call; ``""`` yields no headers.
        operation: What this particular call does.

    Returns:
        Header name to value; empty when unattributed for any reason.
    """
    if component:
        return _attribution_headers(component=component, operation=operation)
    # Gated on a gateway actually being selected, not merely on the variable
    # being set: a misspelled preset emits nothing either, and blaming the call
    # site for that would send someone to instrument code that is already fine.
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
    """Locate the call site to report, skipping this module's own frames.

    The helpers between a caller and here vary in depth, so a fixed offset would
    name ``chat_completion`` rather than whoever called it -- which is the one
    thing the message needs to get right to be actionable.

    Walks the live frames rather than building a formatted traceback: this runs
    on the request path of every untagged call, and only the first one at each
    site produces any output, so materializing the whole stack to then discard
    it would charge every later call for a line it will not print.

    Returns:
        ``path:line`` of the nearest frame outside this file.
    """
    frame = sys._getframe(1)
    while frame is not None:
        if os.path.abspath(frame.f_code.co_filename) != _THIS_FILE:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "an unknown call site"


def _tag_request(params: dict[str, object], component: str, operation: str = "") -> dict[str, object]:
    """Merge the attribution header into an OpenAI-SDK request, in place.

    Tagging the request rather than the client is deliberate: a caller builds
    its client once and keeps it across phases, so a header fixed at
    construction would report whichever phase happened to be current then.

    Args:
        params: ``create`` parameters, mutated in place.
        component: Producer label for the call; ``""`` skips tagging.
        operation: What this particular call does.

    Returns:
        The same ``params`` mapping, for call-site brevity.
    """
    headers = _headers_for(component, operation)
    if headers:
        existing = params.get("extra_headers") or {}
        params["extra_headers"] = {**existing, **headers}  # type: ignore[dict-item]
    return params


def chat_completion(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> ChatCompletionResult:
    """Non-streaming chat completion; returns text, finish reason and usage.

    Transport and API errors propagate: each caller tags them with its own role
    context, and absorbing them here would hide an unreachable gateway.

    Args:
        client: A client from :func:`get_openai_client`.
        component: Producer label used to tag the call for the gateway.
        operation: What this particular call does, e.g. ``rank_candidates``.
        **params: ``chat.completions.create`` parameters.

    Returns:
        The flattened :class:`ChatCompletionResult` for the first choice.
    """
    return _chat_completion_result(client.chat.completions.create(**_tag_request(params, component, operation)))  # type: ignore[union-attr]


async def achat_completion(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> ChatCompletionResult:
    """Async non-streaming chat completion; returns text, finish reason and usage.

    Transport and API errors propagate: each caller tags them with its own role
    context, and absorbing them here would hide an unreachable gateway.

    Args:
        client: A client from :func:`get_async_openai_client`.
        component: Producer label used to tag the call for the gateway.
        operation: What this particular call does, e.g. ``rank_candidates``.
        **params: ``chat.completions.create`` parameters.

    Returns:
        The flattened :class:`ChatCompletionResult` for the first choice.
    """
    return _chat_completion_result(await client.chat.completions.create(**_tag_request(params, component, operation)))  # type: ignore[union-attr]


def _sdk_field(obj: object, key: str) -> object:
    """Read ``key`` off a dict or an attribute-carrying object.

    Responses-API results arrive as pydantic models from the SDK and as plain
    dicts from fixtures and pass-through gateways; both shapes reach here.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _sdk_token_count(usage: object, key: str) -> int:
    """Read one ``usage`` counter as a non-negative int; ``0`` when absent, not numeric, or negative."""
    raw = _sdk_field(usage, key)
    parsed = _to_int(raw, default=0)
    return max(0, parsed)  # type: ignore[type-var]


@dataclass(frozen=True)
class ResponsesResult:
    """One Responses-API result, flattened for Hyperloom callers.

    Token counters are flattened here, unlike :class:`ChatCompletionResult`
    which hands ``usage`` back untouched: the Responses API fixes their names,
    whereas chat-completions callers still map provider-specific spellings.

    Attributes:
        text: The ``output_text`` blocks of every ``message`` item, newline-joined.
        citations: Ordered ``url_citation`` URLs annotated on those blocks, which
            is where server-side tools such as ``web_search`` report sources.
        status: The response ``status``, standing in for the ``finish_reason`` a
            chat completion would carry.
        input_tokens: Prompt tokens from ``usage``.
        output_tokens: Generated tokens from ``usage``.
    """

    text: str
    citations: list[str]
    status: str | None
    input_tokens: int
    output_tokens: int


async def aresponse(
    client: object,
    **params: object,
) -> ResponsesResult:
    """Async Responses-API call; returns text, citations, status and token counts.

    Server-side tools resolve within this single call, so there is no
    client-side tool loop to drive. Transport and API errors propagate: each
    caller tags them with its own role context, and absorbing them here would
    hide an unreachable gateway.

    Args:
        client: A client from :func:`get_async_openai_client`.
        **params: ``responses.create`` parameters.

    Returns:
        The flattened :class:`ResponsesResult`.
    """
    resp = await client.responses.create(**params)  # type: ignore[union-attr]
    texts: list[str] = []
    citations: list[str] = []
    for item in _sdk_field(resp, "output") or []:  # type: ignore[union-attr]
        if _sdk_field(item, "type") != "message":
            continue
        for block in _sdk_field(item, "content") or []:  # type: ignore[union-attr]
            if _sdk_field(block, "type") != "output_text":
                continue
            chunk = _sdk_field(block, "text") or ""
            if chunk:
                texts.append(str(chunk))
            for ann in _sdk_field(block, "annotations") or []:  # type: ignore[union-attr]
                url = _sdk_field(ann, "url")
                if isinstance(url, str) and url:
                    citations.append(url)
    usage = _sdk_field(resp, "usage")
    return ResponsesResult(
        text="\n".join(texts),
        citations=citations,
        status=_sdk_field(resp, "status"),  # type: ignore[arg-type]
        input_tokens=_sdk_token_count(usage, "input_tokens"),
        output_tokens=_sdk_token_count(usage, "output_tokens"),
    )


def _anthropic_text_from_content(content: object) -> str:
    """Concatenate the ``text`` blocks of an Anthropic Messages ``content`` array.

    Non-text blocks (``tool_use``, ``thinking``) and malformed shapes contribute
    nothing, so a reply Hyperloom cannot read comes back empty rather than
    raising: the caller decides whether empty text is fatal.
    """
    if not isinstance(content, list):
        return ""
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "".join(parts).strip()


@dataclass(frozen=True)
class AnthropicMessageResult:
    """One Anthropic Messages reply, flattened for Hyperloom callers.

    Attributes:
        text: The reply's ``text`` content blocks, concatenated and stripped.
        stop_reason: Why generation ended — Anthropic's analogue of a
            chat completion's ``finish_reason``.
        usage: The ``usage`` block verbatim. Anthropic reports ``input_tokens`` /
            ``output_tokens`` alongside provider-specific cache counters, so
            mapping stays with the caller that owns the token accounting.
    """

    text: str
    stop_reason: str | None
    usage: object | None


def _attribution_tag_kwargs(component: str, operation: str = "") -> dict[str, object]:
    """Build the ``post`` keyword carrying the attribution header, if any.

    The keyword is omitted rather than passed as ``None`` so an unconfigured
    deployment calls the transport with exactly the arguments it always did.

    Args:
        component: Producer label for the call; ``""`` yields no keyword.
        operation: What this particular call does.

    Returns:
        ``{"headers": {...}}``, or ``{}`` when there is nothing to tag with.
    """
    headers = _headers_for(component, operation)
    return {"headers": headers} if headers else {}


def _anthropic_message_result(resp: object) -> AnthropicMessageResult:
    """Check one Messages response for failure, then flatten it.

    Raises:
        RuntimeError: On a >= 400 status, or a body that is not JSON. Both mean
            the request produced no usable reply, so they surface rather than
            degrading to empty text.
    """
    status = int(getattr(resp, "status_code", 200) or 200)
    if status >= 400:
        detail = str(getattr(resp, "text", ""))[:200]
        raise RuntimeError(f"anthropic messages status={status} body={detail}")
    try:
        body = resp.json()  # type: ignore[attr-defined]
    except ValueError as exc:
        raise RuntimeError(f"anthropic messages returned a non-JSON body: {exc!r}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"anthropic messages returned a non-object JSON body: {type(body).__name__}")
    payload = body
    return AnthropicMessageResult(
        text=_anthropic_text_from_content(payload.get("content")),
        stop_reason=payload.get("stop_reason"),
        usage=payload.get("usage"),
    )


def anthropic_messages(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> AnthropicMessageResult:
    """POST one Anthropic Messages request; returns text, stop_reason and usage.

    Args:
        client: A client from :func:`get_anthropic_client`, which carries the
            base URL and auth headers.
        component: Producer label used to tag the call for the gateway. This
            transport sends ``params`` as the request body, so the tag travels
            as a header rather than as a parameter.
        operation: What this particular call does, e.g. ``review_commit``.
        **params: The Messages request body — ``model``, ``messages``,
            ``max_tokens``, and ``system`` when there is a system prompt.

    Returns:
        The flattened :class:`AnthropicMessageResult`.

    Raises:
        RuntimeError: On a >= 400 status or a non-JSON body. Transport errors
            from the client propagate untouched so each caller can tag them with
            its own role context.
    """
    tag = _attribution_tag_kwargs(component, operation)
    return _anthropic_message_result(client.post(_ANTHROPIC_MESSAGES_PATH, json=params, **tag))  # type: ignore[union-attr]


async def aanthropic_messages(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> AnthropicMessageResult:
    """Async twin of :func:`anthropic_messages`; see it for the full contract.

    Args:
        client: A client from :func:`get_async_anthropic_client`.
        component: Producer label used to tag the call for the gateway.
        operation: What this particular call does, e.g. ``review_commit``.
        **params: The Messages request body.

    Returns:
        The flattened :class:`AnthropicMessageResult`.
    """
    tag = _attribution_tag_kwargs(component, operation)
    resp = await client.post(_ANTHROPIC_MESSAGES_PATH, json=params, **tag)  # type: ignore[union-attr]
    return _anthropic_message_result(resp)


# Single-shot Anthropic transports. "http" is the Messages API; "sdk" drives the
# Claude CLI, the only channel that accepts a Max/Pro subscription token.
ANTHROPIC_TRANSPORT_HTTP = "http"
ANTHROPIC_TRANSPORT_SDK = "sdk"

_ANTHROPIC_NO_CREDENTIAL = (
    f"no Anthropic credential configured: set one of {' / '.join(ANTHROPIC_CREDENTIAL_ENV_ORDER)}"
)


def anthropic_transport(env: Mapping[str, str] | None = None) -> str:
    """Pick the single-shot Anthropic transport the credential can authenticate.

    An API key or gateway bearer token authenticates the Messages API directly.
    A subscription token cannot: ``/v1/messages`` rejects it with "OAuth
    authentication is currently not supported", and the Claude CLI is the only
    sanctioned channel that accepts it. Key beats token, matching the CLI's own
    precedence, so a host carrying both resolves the same way everywhere.

    Args:
        env: Env mapping to read instead of ``os.environ``.

    Returns:
        :data:`ANTHROPIC_TRANSPORT_HTTP`, :data:`ANTHROPIC_TRANSPORT_SDK`, or
        ``""`` when no Anthropic-side credential is configured.
    """
    source = env if env is not None else os.environ
    if anthropic_synthesizable_key(source):
        return ANTHROPIC_TRANSPORT_HTTP
    if (source.get(CLAUDE_OAUTH_TOKEN_ENV) or "").strip():
        return ANTHROPIC_TRANSPORT_SDK
    return ""


def anthropic_transport_ready(env: Mapping[str, str] | None = None) -> bool:
    """Whether a single-shot Anthropic call could actually be issued right now.

    Distinct from :func:`anthropic_transport`, which only reads the credential:
    the CLI transport additionally needs the agent SDK installed. Callers that
    degrade gracefully rather than failing the run probe with this first.

    Args:
        env: Env mapping to read instead of ``os.environ``.

    Returns:
        True when a credential is configured and its transport is available.
    """
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


def _anthropic_http_params(
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: str | None,
    max_tokens: int,
    temperature: float | None = None,
) -> dict[str, object]:
    """Assemble the Messages request body, omitting an absent system prompt."""
    params: dict[str, object] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": int(max_tokens),
    }
    if system:
        params["system"] = system
    if temperature is not None:
        params["temperature"] = float(temperature)
    return params


def _one_shot_client(
    timeout_s: float | None,
    env: Mapping[str, str] | None,
    component: str = "",
    operation: str = "",
) -> object:
    """Build the Claude CLI one-shot client, imported late to avoid a cycle.

    ``env`` is forwarded rather than dropped: the CLI resolves its own
    credential from the environment it is handed, so a caller that passed an
    explicit mapping would otherwise silently get the ambient one.

    ``component`` and ``operation`` travel with it so the SDK transport tags its
    spend the same way the HTTP one does, instead of going out anonymous.
    """
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
    model: str,
    messages: Sequence[Mapping[str, object]],
    max_tokens: int,
    system: str | None = None,
    temperature: float | None = None,
    env: Mapping[str, str] | None = None,
    timeout: object | None = None,
    timeout_s: float | None = None,
    component: str = "",
    operation: str = "",
) -> AnthropicMessageResult:
    """Issue one single-shot Anthropic completion over whichever transport works.

    The single sanctioned entry point for single-turn Anthropic inference. It
    owns transport selection so no caller has to know that a subscription token
    cannot reach the Messages API; both transports return the same
    :class:`AnthropicMessageResult`.

    Transports differ in three observable ways. ``max_tokens`` is a request
    field on the HTTP path but rides the CLI environment on the SDK path,
    ``stop_reason`` is always present on the HTTP path while the SDK reports it
    only when the model supplies one, and ``temperature`` reaches the model only
    on the HTTP path -- the CLI exposes no such knob.

    Args:
        model: The Claude model id.
        messages: Anthropic-style turns.
        max_tokens: Output-token cap; required by the Messages API.
        system: The system instruction, or ``None``.
        temperature: Sampling temperature, or ``None`` for the model default.
            Sent on the HTTP path and silently ignored on the SDK one, which
            has nowhere to put it.
        env: Env mapping to read instead of ``os.environ``.
        timeout: HTTP-path transport timeout; build one with
            :func:`build_http_timeout`.
        timeout_s: SDK-path wall-clock budget in seconds, CLI startup included.
        component: Producer label used to tag the call for the gateway. The
            HTTP path sends it as a header; the SDK path passes it through the
            CLI child's environment.
        operation: What this particular call does, e.g. ``review_commit``.

    Returns:
        The flattened :class:`AnthropicMessageResult`.

    Raises:
        LLMConfigError: If no Anthropic-side credential is configured.
        RuntimeError: On an HTTP failure status, a non-JSON body, or an
            unavailable Claude CLI.
    """
    transport = anthropic_transport(env)
    if transport == ANTHROPIC_TRANSPORT_SDK:
        client = _one_shot_client(timeout_s, env, component, operation)
        return client.messages(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )
    if not transport:
        raise LLMConfigError(_ANTHROPIC_NO_CREDENTIAL)
    http_client = get_anthropic_client(env=dict(env) if env is not None else None, timeout=timeout)
    with http_client:  # type: ignore[attr-defined]
        return anthropic_messages(
            http_client,
            component=component,
            operation=operation,
            **_anthropic_http_params(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )


async def aanthropic_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    max_tokens: int,
    system: str | None = None,
    temperature: float | None = None,
    env: Mapping[str, str] | None = None,
    timeout: object | None = None,
    timeout_s: float | None = None,
    component: str = "",
    operation: str = "",
) -> AnthropicMessageResult:
    """Async twin of :func:`anthropic_completion`; see it for the full contract.

    Args:
        model: The Claude model id.
        messages: Anthropic-style turns.
        max_tokens: Output-token cap; required by the Messages API.
        system: The system instruction, or ``None``.
        temperature: Sampling temperature, or ``None`` for the model default.
            Sent on the HTTP path and silently ignored on the SDK one, which
            has nowhere to put it.
        env: Env mapping to read instead of ``os.environ``.
        timeout: HTTP-path transport timeout.
        timeout_s: SDK-path wall-clock budget in seconds.
        component: Producer label used to tag the call for the gateway. The
            HTTP path sends it as a header; the SDK path passes it through the
            CLI child's environment.
        operation: What this particular call does, e.g. ``review_commit``.

    Returns:
        The flattened :class:`AnthropicMessageResult`.

    Raises:
        LLMConfigError: If no Anthropic-side credential is configured.
        RuntimeError: On an HTTP failure status, a non-JSON body, or an
            unavailable Claude CLI.
    """
    transport = anthropic_transport(env)
    if transport == ANTHROPIC_TRANSPORT_SDK:
        client = _one_shot_client(timeout_s, env, component, operation)
        return await client.amessages(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )
    if not transport:
        raise LLMConfigError(_ANTHROPIC_NO_CREDENTIAL)
    http_client = get_async_anthropic_client(env=dict(env) if env is not None else None, timeout=timeout)
    async with http_client:  # type: ignore[attr-defined]
        return await aanthropic_messages(
            http_client,
            component=component,
            operation=operation,
            **_anthropic_http_params(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )


def stream_chat_completion_text(
    client: object,
    *,
    component: str = "",
    operation: str = "",
    **params: object,
) -> "tuple[str, object | None]":
    """Streamed chat completion; returns ``(text, usage)``."""
    params["stream"] = True
    params["stream_options"] = {"include_usage": True}
    parts: list[str] = []
    usage_obj: object | None = None
    stream = client.chat.completions.create(**_tag_request(params, component, operation))  # type: ignore[union-attr]
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
    return "".join(parts), usage_obj


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
    stream = await client.chat.completions.create(**_tag_request(params, component, operation))  # type: ignore[union-attr]
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
    return "".join(parts), usage_obj


__all__ = [
    "ANTHROPIC_CREDENTIAL_ENV_ORDER",
    "ANTHROPIC_SYNTHESIZABLE_KEY_ENVS",
    "ANTHROPIC_TRANSPORT_HTTP",
    "ANTHROPIC_TRANSPORT_SDK",
    "AnthropicMessageResult",
    "CLAUDE_GATEWAY_SIGNAL_KEYS",
    "CLAUDE_OAUTH_TOKEN_ENV",
    "ChatCompletionResult",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_ANTHROPIC_VERSION",
    "LEGACY_DEEPSEEK_ENV_KEYS",
    "LLMConfigError",
    "OpenAIClientConfig",
    "ResponsesResult",
    "aanthropic_completion",
    "aanthropic_messages",
    "achat_completion",
    "anthropic_completion",
    "anthropic_messages",
    "anthropic_synthesizable_key",
    "anthropic_transport",
    "anthropic_transport_ready",
    "apply_reasoning_effort",
    "aresponse",
    "astream_chat_completion_text",
    "build_http_timeout",
    "chat_completion",
    "claude_sdk_env_options",
    "deepseek_compat_env",
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
    "openai_client_kwargs",
    "parse_custom_headers",
    "provider_model_defaults",
    "resolve_forge_llm_model",
    "resolve_openai_client_config",
    "stream_chat_completion_text",
]
