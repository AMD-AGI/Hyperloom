# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Paired resolution of the LLM gateway endpoint, credential, and headers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

__all__ = [
    "LlmGateway",
    "expand_env_refs",
    "format_custom_headers",
    "normalize_anthropic_base_url",
    "parse_custom_headers",
    "resolve_anthropic_gateway",
    "resolve_openai_gateway",
]

# Anthropic protocol, so the native x-api-key form leads and the gateway bearer
# token follows -- matching Hyperloom's Claude paths, which order it the same way.
_ANTHROPIC_KEY_ENVS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

log = logging.getLogger("kernelforge.llm")

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# A value carrying ", Some-Name:" almost certainly meant to be two headers.
_PACKED_PAIR_RE = re.compile(r",\s*[A-Za-z0-9][A-Za-z0-9_-]*\s*:")


@dataclass
class LlmGateway:
    """One provider's endpoint, credential variable name, and headers.

    Every field comes from the same provider. ``key_env`` is the variable NAME
    so callers can pass the credential by reference instead of copying it.
    """

    base_url: str = ""
    key_env: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    # No __bool__: "complete" means different things per line. The OpenAI line
    # needs both halves, while a Claude CLI on a Max login needs neither, so a
    # single truthiness rule would silently mislabel one of them.
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


def expand_env_refs(raw: str) -> str:
    """Substitute shell-style ``${VAR}`` references from the environment.

    Lets a ``.env`` keep one copy of a secret and derive gateway headers from it.
    Both provider lines get this, even though only one of them has its headers
    parsed here — the Claude CLI reads its own variable and would otherwise send
    the reference text verbatim.
    """
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), ""), raw)


def parse_custom_headers(raw: str | None) -> dict[str, str]:
    """Parse custom LLM headers (JSON object OR newline-delimited ``Name: value``).

    Only corporate gateways need these; a direct provider endpoint does not.
    APIM deployments (AMD's among them) require an ``Ocp-Apim-Subscription-Key``
    that neither SDK sends from ``api_key`` alone, and answer 401 "missing
    subscription key" without it. ``${VAR}`` references expand from
    ``os.environ`` so a ``.env`` can keep one copy of a secret.

    Newline-delimited is the Anthropic SDK's own format; the JSON object form is
    accepted for launchers that already store structured environment values.
    """
    if not raw:
        return {}
    expanded = expand_env_refs(raw).strip()
    if not expanded:
        return {}
    headers: dict[str, str] = {}
    parsed_json = False
    if expanded.startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            obj = json.loads(expanded)
            if isinstance(obj, dict):
                headers = {str(k).strip(): str(v).strip() for k, v in obj.items() if str(k).strip()}
                parsed_json = True
    if not parsed_json:
        for line in expanded.splitlines():
            name, sep, value = line.partition(":")
            if sep and name.strip():
                headers[name.strip()] = value.strip()
    # An empty value usually means an unresolved ${VAR}; a blank subscription key
    # still 401s at the gateway, so surface it rather than fail silently.
    for name, value in headers.items():
        if not value:
            log.warning("custom header %r has an empty value (unresolved ${VAR}?)", name)
    if not parsed_json:
        dropped = sum(1 for line in expanded.splitlines() if line.strip() and ":" not in line)
        if dropped:
            log.warning("ignored %d custom header line(s) without a 'Name: value' colon", dropped)
    # Comma-separated pairs on one line are not supported: a header value may
    # legitimately contain commas, so splitting on them would corrupt real values.
    for name, value in headers.items():
        if _PACKED_PAIR_RE.search(value):
            log.warning(
                "custom header %r value %r looks like it packs more headers on one "
                "line; put each on its own line (comma-separated is not split)",
                name,
                value,
            )
    return headers


def format_custom_headers(headers: Mapping[str, str]) -> str:
    """Render headers as the newline-delimited form both SDKs understand."""
    return "\n".join(f"{name}: {value}" for name, value in headers.items())


def normalize_anthropic_base_url(base_url: str) -> str:
    """Strip the path suffix every Anthropic client appends for itself.

    The SDK and the Claude CLI both post to ``{base_url}/v1/messages``, so a
    base that already carries that tail produces ``/v1/v1/messages`` and 404s.
    Measured against a LiteLLM proxy, which publishes its base as ``.../v1``:
    left as configured the CLI reports "There's an issue with the selected
    model ... it may not exist or you may not have access to it", which sends
    the reader after a model and a permission that were never the problem.

    Only the two tails a client would duplicate are removed: ``/v1`` and
    ``/v1/messages``. A base ending in a bare ``/messages`` is left alone --
    that is not a duplicate of what gets appended, and a gateway really serving
    ``{base}/messages`` would be made *more* wrong by stripping it.

    This is the one place either provider line rewrites what the operator
    configured, and it is narrow on purpose: the OpenAI line still passes its
    base through untouched, because there no client appends a path of its own,
    so a mismatch there is a real typo worth surfacing rather than absorbing.
    """
    base = base_url.strip().rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base


def resolve_anthropic_gateway() -> LlmGateway:
    """Resolve the Anthropic line from ``ANTHROPIC_*`` and nothing else.

    This line serves the Claude CLI and SDK, which read the variables themselves.
    Nothing here is handed to them, so this exists to normalize and to report what
    the operator configured — not to gate the backend. Do not require
    :meth:`LlmGateway.is_complete` of the result: an absent ``base_url`` means the
    CLI applies its own default endpoint, and a CLI logged in with Claude Code Max
    needs no credential at all, so both halves are legitimately optional here.

    Returns:
        An :class:`LlmGateway` whose ``base_url`` and ``key_env`` may each be
        empty; use :attr:`LlmGateway.has_endpoint` / :attr:`LlmGateway.has_key`
        when a caller genuinely needs to know.
    """
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
    """Reduce a base URL to the origin that decides who may see its headers.

    Compares scheme, host and effective port rather than the raw authority:
    the scheme is what separates a TLS endpoint from a plaintext one carrying
    the same name, the default port must compare equal to its explicit form,
    and any userinfo is not part of the identity of the host.

    Args:
        url: A base URL, possibly empty or unparseable.

    Returns:
        ``(scheme, host, port)``, or ``None`` when the URL names no host or the
        port is unusable -- neither of which may be treated as a match.
    """
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
    """Whether two base URLs address the same origin.

    Args:
        first: A base URL, possibly empty.
        second: The base URL to compare against, possibly empty.

    Returns:
        True only when both resolve to an origin and the two match.
    """
    left = _origin(first)
    return left is not None and left == _origin(second)


def _resolve_openai_gateway_headers(base_url: str) -> dict[str, str]:
    """Headers for the OpenAI-compatible line.

    ``OPENAI_CUSTOM_HEADERS`` wins when set. When it is empty, the Anthropic
    headers are reused only if both lines address the same host, which is the
    single-gateway shape a Hyperloom claw sandbox runs: one proxy serving both
    protocols, with the LiteLLM spend tag and the APIM subscription key written
    to the Anthropic side alone.

    Two things bound the borrowing. The origin check keeps it from becoming a
    credential leak: these headers routinely carry a gateway secret, so reusing
    them elsewhere would hand it to a machine the operator never pointed at.
    Hyperloom's ``resolve_openai_client_config`` reaches the same place from the
    other direction -- it borrows only when the OpenAI base URL was *derived*
    from ``ANTHROPIC_BASE_URL``, which is same-origin by construction -- while
    KernelForge always requires an explicit ``OPENAI_BASE_URL`` and so has to
    compare the two it was given.

    The second bound is that authentication is never borrowed even within one
    origin. The two lines authenticate separately, so an ``Authorization`` from
    the Anthropic side would replace the bearer the OpenAI SDK builds from
    ``OPENAI_API_KEY`` and 401 every call. What is worth carrying across is
    everything else the endpoint requires of any caller: the subscription key
    the gateway rejects a call without, and the spend tag.

    Args:
        base_url: The OpenAI-compatible base URL already resolved for this line.

    Returns:
        Header name to value; empty when neither line supplies usable headers.
    """
    headers = parse_custom_headers(os.environ.get("OPENAI_CUSTOM_HEADERS"))
    if headers:
        return headers
    if not _same_host(base_url, os.environ.get("ANTHROPIC_BASE_URL", "").strip()):
        return {}
    borrowed = parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS"))
    return {name: value for name, value in borrowed.items() if name.lower() not in _LINE_OWN_AUTH_HEADERS}


def resolve_openai_gateway() -> LlmGateway:
    """Resolve the OpenAI-compatible endpoint from ``OPENAI_*`` env vars.

    KernelForge has two independent provider lines. ``ANTHROPIC_BASE_URL`` plus an
    Anthropic credential serves the Claude CLI and SDK, which read those variables
    themselves. ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY`` serves the callers that
    speak the OpenAI-compatible protocol — fusion discovery and the Codex backend
    — and that is the only line this function looks at for endpoint and credential.

    Endpoint and credential never cross lines. Custom headers may, but only
    within one host: when ``OPENAI_CUSTOM_HEADERS`` is unset and both lines
    point at the same gateway, :func:`_resolve_openai_gateway_headers` reuses
    ``ANTHROPIC_CUSTOM_HEADERS`` so LiteLLM spend tags injected there (typical in
    Hyperloom claw sandboxes) also reach OpenAI-protocol call sites.

    The base URL is used exactly as configured: rewriting it would guess at a
    layout the operator already knows, and hide their typos behind ours.

    Returns:
        A populated :class:`LlmGateway`, or an empty one when either half of the
        pair is missing. Check :meth:`LlmGateway.is_complete`; the dataclass has
        no truthiness of its own, so an empty instance is not falsy.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url or not os.environ.get("OPENAI_API_KEY", "").strip():
        return LlmGateway()
    return LlmGateway(
        base_url=base_url,
        key_env="OPENAI_API_KEY",
        headers=_resolve_openai_gateway_headers(base_url),
    )
