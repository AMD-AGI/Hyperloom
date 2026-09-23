# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Paired resolution of the LLM gateway endpoint, credential, and headers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from hyperloom.common.llm_headers import parse_custom_headers

__all__ = [
    "LlmGateway",
    "normalize_anthropic_base_url",
    "resolve_anthropic_gateway",
    "resolve_openai_gateway",
]

# Anthropic protocol, so the native x-api-key form leads and the gateway bearer token follows -- matching Hyperloom's
# Claude paths, which order it the same way.
_ANTHROPIC_KEY_ENVS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

log = logging.getLogger("kernelforge.llm")


@dataclass
class LlmGateway:
    """One provider's endpoint, credential variable name, and headers."""

    base_url: str = ""
    key_env: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    # No __bool__: "complete" means different things per line.
    @property
    def has_endpoint(self) -> bool:
        """True when an explicit base URL was configured."""
        return bool(self.base_url)

    @property
    def has_key(self) -> bool:
        """True when a credential variable was configured."""
        return bool(self.key_env)

    def is_complete(self) -> bool:
        """True when both halves are present, which the OpenAI line requires."""
        return self.has_endpoint and self.has_key

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> LlmGateway:
        """Build one from a config mapping, ignoring unknown keys."""
        raw_headers = mapping.get("headers")
        headers = (
            {str(k).strip(): str(v).strip() for k, v in raw_headers.items()} if isinstance(raw_headers, Mapping) else {}
        )
        return cls(
            base_url=str(mapping.get("base_url") or "").strip(),
            key_env=str(mapping.get("key_env") or "").strip(),
            headers=headers,
        )


def normalize_anthropic_base_url(base_url: str) -> str:
    """Strip the path suffix every Anthropic client appends for itself."""
    base = base_url.strip().rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def resolve_anthropic_gateway() -> LlmGateway:
    """Resolve the Anthropic line from ``ANTHROPIC_*`` and nothing else."""
    key_env = next(
        (env for env in _ANTHROPIC_KEY_ENVS if os.environ.get(env, "").strip()),
        "",
    )
    return LlmGateway(
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "").strip(),
        key_env=key_env,
        headers=parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS")),
    )


#: Default ports, so ``https://gw`` and ``https://gw:443`` are one origin.
_DEFAULT_PORTS = {"http": 80, "https": 443}

#: Headers the OpenAI line authenticates itself with. Borrowing these from the
#: Anthropic line would replace the bearer the SDK builds from OPENAI_API_KEY --
#: a 401 on every call, blamed on a variable that is set correctly.
_LINE_OWN_AUTH_HEADERS = frozenset({"authorization", "x-api-key"})


def _origin(url: str) -> tuple[str, str, int] | None:
    """Reduce a base URL to the origin that decides who may see its headers."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
    if not host:
        return None
    resolved = port if port is not None else _DEFAULT_PORTS.get(scheme)
    if resolved is None:
        return None
    return scheme, host, resolved


def _same_host(first: str, second: str) -> bool:
    """Whether two base URLs address the same origin."""
    left = _origin(first)
    return left is not None and left == _origin(second)


def _resolve_openai_gateway_headers(base_url: str) -> dict[str, str]:
    """Headers for the OpenAI-compatible line."""
    headers = parse_custom_headers(os.environ.get("OPENAI_CUSTOM_HEADERS"))
    if headers:
        return headers
    if not _same_host(base_url, os.environ.get("ANTHROPIC_BASE_URL", "").strip()):
        return {}
    borrowed = parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS"))
    return {name: value for name, value in borrowed.items() if name.lower() not in _LINE_OWN_AUTH_HEADERS}


def resolve_openai_gateway() -> LlmGateway:
    """Resolve the OpenAI-compatible endpoint from ``OPENAI_*`` env vars."""
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url or not os.environ.get("OPENAI_API_KEY", "").strip():
        return LlmGateway()
    return LlmGateway(
        base_url=base_url,
        key_env="OPENAI_API_KEY",
        headers=_resolve_openai_gateway_headers(base_url),
    )
