# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :class:`runtime.web_tools.providers.TavilyProvider` and
:class:`runtime.web_tools.providers.SerperProvider`.

We use ``httpx.MockTransport`` to capture outgoing requests and return
canned bodies; no network IO. Each test asserts the request mapping
(URL, body, headers) and the response normalization.
"""

from __future__ import annotations

import json

import httpx
import pytest

from runtime.web_tools.providers import (
    ProviderError,
    SearchOptions,
    SerperProvider,
    TavilyProvider,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── Tavily ──────────────────────────────────────────────────────────────

class TestTavily:
    def test_happy_path_maps_results(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"results": [
                    {"title": "T1", "url": "https://a/b", "content": "snip1"},
                    {"title": "T2", "url": "https://c/d", "content": ""},
                ]},
            )

        provider = TavilyProvider(api_key="tk", http_client=_client(handler))
        hits = provider.search(
            "sglang fp8 quant",
            SearchOptions(
                max_results=7,
                allowed_domains=("docs.sglang.ai",),
                freshness="week",
            ),
        )

        assert captured["url"] == "https://api.tavily.com/search"
        assert captured["body"] == {
            "api_key": "tk",
            "query": "sglang fp8 quant",
            "max_results": 7,
            "include_domains": ["docs.sglang.ai"],
            "days": 7,
        }
        assert [(h.title, h.url, h.snippet) for h in hits] == [
            ("T1", "https://a/b", "snip1"),
            ("T2", "https://c/d", ""),
        ]

    def test_blocked_domains_passed_through(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"results": []})

        TavilyProvider(api_key="tk", http_client=_client(handler)).search(
            "x", SearchOptions(blocked_domains=("spam.io",)),
        )
        assert captured["body"]["exclude_domains"] == ["spam.io"]
        assert "include_domains" not in captured["body"]

    def test_http_error_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="dead")

        provider = TavilyProvider(api_key="tk", http_client=_client(handler))
        with pytest.raises(ProviderError):
            provider.search("x", SearchOptions())

    def test_non_json_body_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>")

        provider = TavilyProvider(api_key="tk", http_client=_client(handler))
        with pytest.raises(ProviderError):
            provider.search("x", SearchOptions())

    def test_missing_results_key_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"foo": "bar"})

        provider = TavilyProvider(api_key="tk", http_client=_client(handler))
        assert provider.search("x", SearchOptions()) == []

    def test_freshness_ignored_when_unknown(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"results": []})

        TavilyProvider(api_key="tk", http_client=_client(handler)).search(
            "x", SearchOptions(freshness="anything"),
        )
        assert "days" not in captured["body"]

    def test_empty_key_rejected(self):
        with pytest.raises(ValueError):
            TavilyProvider(api_key="", http_client=_client(lambda r: httpx.Response(200)))


# ── Serper ──────────────────────────────────────────────────────────────

class TestSerper:
    def test_happy_path_with_site_filter(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"organic": [
                {"title": "T1", "link": "https://github.com/foo", "snippet": "s1"},
            ]})

        provider = SerperProvider(api_key="sk", http_client=_client(handler))
        hits = provider.search(
            "vllm bug",
            SearchOptions(
                max_results=3,
                allowed_domains=("github.com", "docs.vllm.ai"),
                freshness="month",
            ),
        )

        assert captured["url"] == "https://google.serper.dev/search"
        assert captured["headers"]["x-api-key"] == "sk"
        assert captured["body"] == {
            "q": "vllm bug site:github.com OR site:docs.vllm.ai",
            "num": 3,
            "tbs": "qdr:m",
        }
        assert [(h.url, h.snippet) for h in hits] == [("https://github.com/foo", "s1")]

    def test_blocked_domains_filtered_post_fetch(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"organic": [
                {"title": "ok", "link": "https://github.com/x"},
                {"title": "bad", "link": "https://spam.io/y"},
                {"title": "sub", "link": "https://api.spam.io/z"},
            ]})

        provider = SerperProvider(api_key="sk", http_client=_client(handler))
        hits = provider.search(
            "x", SearchOptions(blocked_domains=("spam.io",)),
        )
        assert [h.url for h in hits] == ["https://github.com/x"]

    def test_blocked_does_not_modify_query(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"organic": []})

        SerperProvider(api_key="sk", http_client=_client(handler)).search(
            "x", SearchOptions(blocked_domains=("spam.io",)),
        )
        assert captured["body"]["q"] == "x"  # no site: appended for blocked

    def test_http_failure_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="oops")

        provider = SerperProvider(api_key="sk", http_client=_client(handler))
        with pytest.raises(ProviderError):
            provider.search("x", SearchOptions())

    def test_missing_organic_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        provider = SerperProvider(api_key="sk", http_client=_client(handler))
        assert provider.search("x", SearchOptions()) == []

    def test_empty_key_rejected(self):
        with pytest.raises(ValueError):
            SerperProvider(api_key="", http_client=_client(lambda r: httpx.Response(200)))
