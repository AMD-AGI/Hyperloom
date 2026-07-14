# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""LLM gateway environment resolution shared by LLM SDK callers."""

from __future__ import annotations

import json
import os
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
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "SAFE_API_KEY",
    "LLM_GATEWAY_KEY",
)

DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def parse_custom_headers(raw: str | None) -> dict[str, str]:
    """Parse custom LLM headers from env.

    ``ANTHROPIC_CUSTOM_HEADERS`` is newline-delimited ``Name: value`` in the
    Anthropic SDK. JSON object input is accepted as a convenience for launchers
    that already store structured environment values.
    """
    if not raw:
        return {}
    text = raw.strip()
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
    for line in raw.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    return headers


def derive_openai_base_url(anthropic_base_url: str | None) -> str | None:
    """Derive an OpenAI-compatible base URL from an Anthropic endpoint.

    Explicit ``OPENAI_BASE_URL`` always wins in callers. This fallback handles
    AMD's split gateway convention, where Anthropic traffic uses
    ``/anthropic`` and OpenAI-compatible chat completions use ``/Unified/v1``.
    Unknown Anthropic URLs fall back to their original value to preserve the
    previous single-gateway behavior.
    """
    if not anthropic_base_url:
        return None
    base = anthropic_base_url.strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    path = parts.path.rstrip("/")
    if path.endswith("/anthropic") or path == "/anthropic":
        prefix = path[: -len("/anthropic")] if path.endswith("/anthropic") else ""
        return urlunsplit(parts._replace(path=f"{prefix}/Unified/v1"))
    if path.endswith("/Unified"):
        return urlunsplit(parts._replace(path=f"{path}/v1"))
    return base


def resolve_openai_client_config(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> OpenAIClientConfig:
    """Resolve OpenAI-compatible client config from one or more LLM env sets."""
    source = env if env is not None else os.environ
    api_key = (
        (source.get(api_key_env) or "").strip()
        or (source.get("OPENAI_API_KEY") or "").strip()
        or (source.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (source.get("ANTHROPIC_API_KEY") or "").strip()
        or (source.get("LLM_GATEWAY_KEY") or "").strip()
        or (source.get("SAFE_API_KEY") or "").strip()
    )
    if not api_key:
        key_names = " / ".join(
            dict.fromkeys(
                [
                    api_key_env,
                    "OPENAI_API_KEY",
                    "ANTHROPIC_AUTH_TOKEN",
                    "ANTHROPIC_API_KEY",
                    "LLM_GATEWAY_KEY",
                    "SAFE_API_KEY",
                ]
            )
        )
        raise LLMConfigError(f"{key_names} not set in env (OpenAI-compatible client cannot auth)")

    base_url = (
        (source.get(base_url_env) or "").strip()
        or (source.get("OPENAI_BASE_URL") or "").strip()
        or derive_openai_base_url(source.get("ANTHROPIC_BASE_URL"))
    )
    base_url = base_url or None

    headers = parse_custom_headers(source.get("OPENAI_CUSTOM_HEADERS")) or parse_custom_headers(
        source.get("ANTHROPIC_CUSTOM_HEADERS")
    )
    if base_url and _should_add_amd_subscription_header(base_url, headers):
        headers = {**headers, "Ocp-Apim-Subscription-Key": api_key}
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
    if not any((source.get(key) or "").strip() for key in CLAUDE_GATEWAY_SIGNAL_KEYS):
        return {}

    if "ANTHROPIC_BASE_URL" not in source and source.get("DEEPSEEK_API_KEY"):
        source["ANTHROPIC_BASE_URL"] = source.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL

    fallback_key = (
        source.get("ANTHROPIC_AUTH_TOKEN")
        or source.get("ANTHROPIC_API_KEY")
        or source.get("DEEPSEEK_API_KEY")
        or source.get("OPENAI_API_KEY")
        or source.get("SAFE_API_KEY")
        or source.get("LLM_GATEWAY_KEY")
        or ""
    )
    if fallback_key:
        source.setdefault("ANTHROPIC_API_KEY", fallback_key)
        source.setdefault("ANTHROPIC_AUTH_TOKEN", fallback_key)
    if "ANTHROPIC_CUSTOM_HEADERS" not in source and source.get("OPENAI_CUSTOM_HEADERS"):
        source["ANTHROPIC_CUSTOM_HEADERS"] = source["OPENAI_CUSTOM_HEADERS"]
    # Inject the AMD gateway subscription header on the Claude CLI subprocess
    # path too. The OpenAI client (resolve_openai_client_config) and the
    # preflight catalog probe both auto-add ``Ocp-Apim-Subscription-Key`` by
    # host, but the Claude subprocess previously only forwarded
    # ANTHROPIC_CUSTOM_HEADERS verbatim — so a bare gateway config (no explicit
    # header) 401'd for orchestration / GEAK. Reuse the shared host rule and add
    # it only when absent (an operator-supplied header always wins).
    anthropic_base_url = source.get("ANTHROPIC_BASE_URL", "")
    custom_headers = parse_custom_headers(source.get("ANTHROPIC_CUSTOM_HEADERS"))
    if anthropic_base_url and fallback_key and _should_add_amd_subscription_header(anthropic_base_url, custom_headers):
        custom_headers["Ocp-Apim-Subscription-Key"] = fallback_key
        source["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(f"{name}: {value}" for name, value in custom_headers.items())
    # Claude Code >= 2.1.x injects ``anthropic-beta: advisor-tool-*``, which
    # strict gateways reject with HTTP 400 — stalling orchestration (is_error
    # every tick, 0 intents). Disable it by default; an operator can re-enable
    # by presetting CLAUDE_CODE_DISABLE_ADVISOR_TOOL in the environment.
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


def _should_add_amd_subscription_header(base_url: str, headers: dict[str, str]) -> bool:
    if any(name.lower() == "ocp-apim-subscription-key" for name in headers):
        return False
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    return host == "llm-api.amd.com" or parts.path.rstrip("/").endswith("/Unified/v1")


__all__ = [
    "CLAUDE_GATEWAY_SIGNAL_KEYS",
    "DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "LLMConfigError",
    "OpenAIClientConfig",
    "apply_reasoning_effort",
    "claude_sdk_env_options",
    "derive_openai_base_url",
    "openai_client_kwargs",
    "parse_custom_headers",
    "resolve_openai_client_config",
]
