# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic-agent web tools — pluggable, off-by-default ``web_search`` and
``web_fetch`` capability for the critic LLM reasoning step.

Public surface:

* :class:`WebToolsConfig` — env-driven configuration; build once per process.
* :class:`WebSearchClient` / :class:`WebFetchClient` — facades that return
  the formatted string to feed back as an OpenAI ``tool`` message.
* :func:`build_tool_schemas` — list of OpenAI tool schemas, gated by config.
* :func:`build_clients` — convenience factory that wires providers and the
  default ``httpx.Client`` together; tests call the underlying classes
  directly with their own transports.

See ``Claw/docs/builtin-tools-design.md`` (sections 5.1 / 5.2) for the
reference design these clients mirror.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import IMPLEMENTED_SEARCH_PROVIDERS, KNOWN_PROVIDERS, ProviderName, WebToolsConfig
from .fetch_client import FetchError, WebFetchClient, _new_default_http_client
from .providers import (
    ProviderError,
    SearchHit,
    SearchOptions,
    SerperProvider,
    TavilyProvider,
    WebSearchProvider,
)
from .search_client import WebSearchClient, WebSearchInput
from .tool_schemas import build_tool_schemas


log = logging.getLogger(__name__)


@dataclass
class WebToolClients:
    """Bundle of the optionally-built clients returned by :func:`build_clients`."""

    search: WebSearchClient | None
    fetch: WebFetchClient | None


def build_clients(
    config: WebToolsConfig,
    *,
    http_client: httpx.Client | None = None,
) -> WebToolClients:
    """Build :class:`WebSearchClient` / :class:`WebFetchClient` from config.

    Returns instances with ``None`` slots when the corresponding feature
    is disabled. Caller owns the returned ``http_client``; pass ``None``
    to let this function create one (default for production code).

    Tests should construct ``WebSearchClient`` / ``WebFetchClient``
    directly with their own provider list and transport instead of going
    through this factory.

    Args:
        config (WebToolsConfig): Resolved web-tools configuration.
        http_client (httpx.Client | None): Transport to reuse; a default
            client is created when ``None``.

    Returns:
        WebToolClients: Bundle whose ``search`` / ``fetch`` slots are
        ``None`` when the corresponding feature is disabled or unusable.
    """
    if not config.critic_web_tools_enabled:
        return WebToolClients(search=None, fetch=None)

    http = http_client or _new_default_http_client()

    providers: list[WebSearchProvider] = []
    for name in config.search_provider_chain():
        if name not in IMPLEMENTED_SEARCH_PROVIDERS:
            log.info(
                "skipping web search provider %s — not implemented yet", name,
            )
            continue
        if not config.has_search_api_key(name):
            log.info(
                "skipping web search provider %s — no API key configured", name,
            )
            continue
        try:
            providers.append(_provider_factory(name, config, http))
        except ValueError as exc:
            log.warning("failed to construct provider %s: %s", name, exc)

    search = WebSearchClient(config=config, providers=tuple(providers)) if providers else None
    fetch = WebFetchClient(config=config, http_client=http) if config.fetch_enabled else None

    if search is None and fetch is None:
        log.info(
            "web tools enabled but neither search nor fetch is usable "
            "(provider chain=%s, fetch_enabled=%s)",
            config.search_provider_chain(), config.fetch_enabled,
        )

    return WebToolClients(search=search, fetch=fetch)


def _provider_factory(
    name: str, config: WebToolsConfig, http: httpx.Client,
) -> WebSearchProvider:
    """Construct a search provider by name.

    Args:
        name (str): Implemented provider name (``tavily`` or ``serper``).
        config (WebToolsConfig): Configuration holding the API keys.
        http (httpx.Client): Shared HTTP transport to inject.

    Returns:
        WebSearchProvider: The constructed provider.

    Raises:
        ValueError: If ``name`` is not a known provider.
    """
    if name == "tavily":
        return TavilyProvider(api_key=config.tavily_api_key, http_client=http)
    if name == "serper":
        return SerperProvider(api_key=config.serper_api_key, http_client=http)
    raise ValueError(f"unknown provider {name!r}")


__all__ = [
    "IMPLEMENTED_SEARCH_PROVIDERS",
    "KNOWN_PROVIDERS",
    "FetchError",
    "ProviderError",
    "ProviderName",
    "SearchHit",
    "SearchOptions",
    "SerperProvider",
    "TavilyProvider",
    "WebFetchClient",
    "WebSearchClient",
    "WebSearchInput",
    "WebSearchProvider",
    "WebToolClients",
    "WebToolsConfig",
    "build_clients",
    "build_tool_schemas",
]
