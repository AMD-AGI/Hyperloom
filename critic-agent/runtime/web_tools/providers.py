# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pluggable search backends — Tavily and Serper.

Each implements :class:`WebSearchProvider` returning normalized
:class:`SearchHit`s over an injected synchronous ``httpx.Client`` (so tests
swap in ``httpx.MockTransport``). Field mappings mirror Primus-Claw's
``web-tools/search.ts``; the Claude-only "sub-LLM wrapper" provider is not
ported since critic-agent talks to Codex / OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx


# ── Public types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchHit:
    """One normalized search result entry."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class SearchOptions:
    """Per-call search options. All fields optional."""

    max_results: int | None = None
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    freshness: str | None = None


class WebSearchProvider(Protocol):
    """Each backend implements this interface."""

    name: str

    def search(self, query: str, opts: SearchOptions) -> list[SearchHit]:
        pass


class ProviderError(RuntimeError):
    """Raised when a provider call fails non-recoverably for this provider.

    The :class:`WebSearchClient` catches this and tries the next provider
    in the fallback chain. Non-:class:`ProviderError` exceptions bubble up
    untouched so genuine bugs are visible.
    """


# ── Tavily ──────────────────────────────────────────────────────────────

_TAVILY_FRESHNESS_DAYS: dict[str, int] = {
    "day": 1, "week": 7, "month": 30, "year": 365,
}


class TavilyProvider:
    """Tavily Search API — ``POST https://api.tavily.com/search``.

    Field mapping (mirrors Primus-Claw):

    * ``allowed_domains`` -> ``include_domains``
    * ``blocked_domains`` -> ``exclude_domains``
    * ``freshness`` -> ``days`` (via _TAVILY_FRESHNESS_DAYS)
    * ``max_results`` -> ``max_results`` (default 5)
    """

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, http_client: httpx.Client) -> None:
        if not api_key:
            raise ValueError("TavilyProvider requires non-empty api_key")
        self._api_key = api_key
        self._http = http_client

    def search(self, query: str, opts: SearchOptions) -> list[SearchHit]:
        body: dict[str, object] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": opts.max_results or 5,
        }
        if opts.allowed_domains:
            body["include_domains"] = list(opts.allowed_domains)
        if opts.blocked_domains:
            body["exclude_domains"] = list(opts.blocked_domains)
        if opts.freshness and opts.freshness in _TAVILY_FRESHNESS_DAYS:
            body["days"] = _TAVILY_FRESHNESS_DAYS[opts.freshness]

        try:
            resp = self._http.post(self._ENDPOINT, json=body, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"tavily request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"tavily returned non-JSON body: {exc}") from exc

        results = data.get("results")
        if not isinstance(results, list):
            return []
        return [
            SearchHit(
                title=str(r.get("title") or ""),
                url=str(r.get("url") or ""),
                snippet=str(r.get("content") or ""),
            )
            for r in results
            if isinstance(r, dict)
        ]


# ── Serper ──────────────────────────────────────────────────────────────

_SERPER_FRESHNESS_TBS: dict[str, str] = {
    "day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y",
}


class SerperProvider:
    """Serper (Google search proxy) — ``POST https://google.serper.dev/search``.

    Field mapping:

    * ``allowed_domains`` -> appended to ``q`` as ``site:<d> OR site:<d>``
    * ``blocked_domains`` -> post-filter on the returned hits' hostname
    * ``freshness`` -> ``tbs`` (qdr:d|w|m|y)
    * ``max_results`` -> ``num`` (default 5)
    """

    name = "serper"
    _ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str, http_client: httpx.Client) -> None:
        if not api_key:
            raise ValueError("SerperProvider requires non-empty api_key")
        self._api_key = api_key
        self._http = http_client

    def search(self, query: str, opts: SearchOptions) -> list[SearchHit]:
        q = query
        if opts.allowed_domains:
            q = q + " " + " OR ".join(f"site:{d}" for d in opts.allowed_domains)

        body: dict[str, object] = {"q": q, "num": opts.max_results or 5}
        if opts.freshness and opts.freshness in _SERPER_FRESHNESS_TBS:
            body["tbs"] = _SERPER_FRESHNESS_TBS[opts.freshness]

        try:
            resp = self._http.post(
                self._ENDPOINT,
                json=body,
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"serper request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"serper returned non-JSON body: {exc}") from exc

        organic = data.get("organic")
        if not isinstance(organic, list):
            return []
        hits = [
            SearchHit(
                title=str(r.get("title") or ""),
                url=str(r.get("link") or ""),
                snippet=str(r.get("snippet") or ""),
            )
            for r in organic
            if isinstance(r, dict)
        ]
        if opts.blocked_domains:
            hits = [h for h in hits if not _hostname_in(h.url, opts.blocked_domains)]
        return hits


# ── helpers ─────────────────────────────────────────────────────────────

def _hostname_in(url: str, denylist: tuple[str, ...]) -> bool:
    if not denylist:
        return False
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, AttributeError):
        return False
    return any(host == d or host.endswith("." + d) for d in denylist)


__all__ = [
    "ProviderError",
    "SearchHit",
    "SearchOptions",
    "SerperProvider",
    "TavilyProvider",
    "WebSearchProvider",
]
