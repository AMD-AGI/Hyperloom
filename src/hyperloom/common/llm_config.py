# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM gateway environment resolution shared by LLM SDK callers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


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


CLAUDE_GATEWAY_SIGNAL_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CUSTOM_HEADERS",
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
_ANTHROPIC_SIDE_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)
_OPENAI_SIDE_KEYS: tuple[str, ...] = ("OPENAI_BASE_URL", "OPENAI_API_KEY")

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _is_dual_protocol_host(url: str | None) -> bool:
    """True when ``url`` points at a known ``/anthropic`` + ``/v1`` gateway."""
    if not url:
        return False
    return urlsplit(str(url).strip()).hostname in _DUAL_PROTOCOL_HOSTS


def dual_protocol_endpoint_pair(base_url: str) -> tuple[str, str]:
    """Return ``(anthropic_base_url, openai_base_url)`` for a dual-protocol gateway.

    The input may name either side (or neither). The sibling is derived by
    swapping the trailing path segment, so both protocols resolve to a route
    that exists. Matching is case-insensitive because AMD's own gateway spells
    the segment ``/Anthropic`` (issue #929).

    A bare host on a known dual-protocol gateway gets both segments appended;
    for an unknown host the value is kept as the Anthropic side, since that is
    what the caller named it.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        return _DEEPSEEK_ANTHROPIC_BASE_URL, _DEEPSEEK_OPENAI_BASE_URL
    lowered = base.lower()
    # rsplit rather than a cased suffix strip: it drops the final segment
    # whatever its casing, which keeps this identical to the shell installers.
    if lowered.endswith("/anthropic"):
        return base, f"{base.rsplit('/', 1)[0]}/v1"
    if lowered.endswith("/v1"):
        return f"{base.rsplit('/', 1)[0]}/anthropic", base
    if _is_dual_protocol_host(base):
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
    if any((source.get(name) or "").strip() for name in _ANTHROPIC_SIDE_KEYS):
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
    if not any((source.get(name) or "").strip() for name in _OPENAI_SIDE_KEYS):
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
    codex_model = (resolved.get("CODEX_MODEL") or "").strip() or endpoint_default_model(
        resolved.get("OPENAI_BASE_URL")
    )
    candidates = {
        "CLAUDE_MODEL": claude_model,
        "CODEX_MODEL": codex_model,
        "GEAK_CLAUDE_MODEL": (resolved.get("GEAK_CLAUDE_MODEL") or "").strip() or claude_model,
    }
    return {key: value for key, value in candidates.items() if value and not (source.get(key) or "").strip()}


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
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {str(k).strip(): str(v).strip() for k, v in parsed.items() if str(k).strip()}

    headers: dict[str, str] = {}
    for line in expanded.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    return headers


def derive_openai_base_url(anthropic_base_url: str | None) -> str | None:
    """Derive an OpenAI-compatible base URL from an Anthropic endpoint.

    Explicit ``OPENAI_BASE_URL`` always wins in callers. This fallback handles
    AMD's split gateway convention, where Anthropic traffic uses
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
    """Resolve OpenAI-compatible client config from one or more LLM env sets."""
    source = env if env is not None else os.environ
    # OpenAI-side credentials only.
    api_key = (
        (source.get(api_key_env) or "").strip()
        or (source.get("OPENAI_API_KEY") or "").strip()
        or (source.get("LLM_GATEWAY_KEY") or "").strip()
    )
    if not api_key:
        key_names = " / ".join(dict.fromkeys([api_key_env, "OPENAI_API_KEY", "LLM_GATEWAY_KEY"]))
        raise LLMConfigError(f"{key_names} not set in env (OpenAI-compatible client cannot auth)")

    base_url = (source.get(base_url_env) or "").strip() or (source.get("OPENAI_BASE_URL") or "").strip()
    base_url = base_url or None

    # OpenAI/Codex side reads only OPENAI_CUSTOM_HEADERS; gateway headers are operator-supplied.
    headers = parse_custom_headers(source.get("OPENAI_CUSTOM_HEADERS"), env=source)
    return OpenAIClientConfig(api_key=api_key, base_url=base_url, default_headers=headers)


def openai_client_kwargs(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return kwargs accepted by ``openai.AsyncOpenAI``."""
    return resolve_openai_client_config(api_key_env=api_key_env, base_url_env=base_url_env, env=env).as_kwargs()


def claude_sdk_env_options(
    *,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return Claude SDK options that isolate a run from global Claude config.

    Claude Code and ``claude_agent_sdk`` can read ``~/.claude/settings.json``.
    Hyperloom runs, however, may intentionally point at a per-run provider or
    gateway. When any LLM-provider signal is present in the current environment,
    pass an explicit child environment and disable settings sources so global
    developer-machine configuration cannot override the run contract.
    """
    source = dict(env if env is not None else os.environ)
    # Callers that never reach CLI preflight (library-mode backends) may still
    # carry a legacy DeepSeek configuration; normalize before probing signals.
    source.update(deepseek_compat_env(source))
    if not any((source.get(key) or "").strip() for key in CLAUDE_GATEWAY_SIGNAL_KEYS):
        return {}

    # Anthropic-side credentials only.
    fallback_key = source.get("ANTHROPIC_AUTH_TOKEN") or source.get("ANTHROPIC_API_KEY") or ""
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


def stream_chat_completion_text(
    client: object,
    **params: object,
) -> "tuple[str, object | None]":
    """Streamed chat completion; returns ``(text, usage)``."""
    params["stream"] = True
    params["stream_options"] = {"include_usage": True}
    parts: list[str] = []
    usage_obj: object | None = None
    stream = client.chat.completions.create(**params)  # type: ignore[union-attr]
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
    **params: object,
) -> "tuple[str, object | None]":
    """Async streamed chat completion; returns ``(text, usage)``."""
    params["stream"] = True
    params["stream_options"] = {"include_usage": True}
    parts: list[str] = []
    usage_obj: object | None = None
    stream = await client.chat.completions.create(**params)  # type: ignore[union-attr]
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
    return "".join(parts), usage_obj


__all__ = [
    "CLAUDE_GATEWAY_SIGNAL_KEYS",
    "LEGACY_DEEPSEEK_ENV_KEYS",
    "LLMConfigError",
    "OpenAIClientConfig",
    "apply_reasoning_effort",
    "astream_chat_completion_text",
    "claude_sdk_env_options",
    "deepseek_compat_env",
    "derive_openai_base_url",
    "dual_protocol_endpoint_pair",
    "endpoint_default_model",
    "provider_model_defaults",
    "openai_client_kwargs",
    "parse_custom_headers",
    "resolve_openai_client_config",
    "stream_chat_completion_text",
]
