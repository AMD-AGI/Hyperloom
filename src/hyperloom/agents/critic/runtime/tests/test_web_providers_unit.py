# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for web_tools.providers search backends."""

from __future__ import annotations

import httpx
import pytest

from hyperloom.agents.critic.runtime.web_tools.providers import (
    ProviderError,
    SearchOptions,
    SerperProvider,
    TavilyProvider,
    _hostname_in,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hostname_in() -> None:
    assert _hostname_in("http://x.com", ()) is False  # line 270 (empty denylist)
    assert _hostname_in("https://www.evil.com/p", ("evil.com",)) is True
    assert _hostname_in("https://good.com", ("evil.com",)) is False


def test_provider_requires_api_key() -> None:
    with pytest.raises(ValueError):
        TavilyProvider("", _client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ValueError):
        SerperProvider("", _client(lambda r: httpx.Response(200, json={})))


def test_tavily_search_normalizes() -> None:
    def handler(request):
        return httpx.Response(200, json={"results": [{"title": "T", "url": "u", "content": "c"}, "skip"]})

    prov = TavilyProvider("k", _client(handler))
    hits = prov.search("q", SearchOptions(max_results=3, allowed_domains=("a.com",), freshness="week"))
    assert len(hits) == 1
    assert hits[0].title == "T"


def test_tavily_http_error_raises_provider_error() -> None:
    def handler(request):
        return httpx.Response(500, text="boom")

    prov = TavilyProvider("k", _client(handler))
    with pytest.raises(ProviderError):
        prov.search("q", SearchOptions())


def test_serper_search_with_block_filter() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "organic": [
                    {"title": "ok", "link": "https://good.com/x", "snippet": "s"},
                    {"title": "bad", "link": "https://evil.com/y", "snippet": "s"},
                ]
            },
        )

    prov = SerperProvider("k", _client(handler))
    hits = prov.search(
        "q",
        SearchOptions(blocked_domains=("evil.com",), allowed_domains=("good.com",), freshness="day"),
    )
    assert [h.url for h in hits] == ["https://good.com/x"]


def test_serper_non_json_raises(monkeypatch) -> None:
    def handler(request):
        return httpx.Response(200, text="not json at all")

    prov = SerperProvider("k", _client(handler))
    with pytest.raises(ProviderError):  # lines 235-236
        prov.search("q", SearchOptions())
