"""Tests for :class:`runtime.web_tools.search_client.WebSearchClient`.

Providers are simulated with tiny in-process classes that record their
inputs and return canned hits, so tests stay deterministic and never go
near httpx.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from hyperloom.agents.critic.runtime.web_tools.config import WebToolsConfig
from hyperloom.agents.critic.runtime.web_tools.providers import ProviderError, SearchHit, SearchOptions
from hyperloom.agents.critic.runtime.web_tools.search_client import (
    WebSearchClient,
    WebSearchInput,
    _LeakyBucket,
)


@dataclass
class _RecordingProvider:
    name: str
    hits: list[SearchHit]
    calls: list[tuple[str, SearchOptions]]

    def search(self, query: str, opts: SearchOptions) -> list[SearchHit]:
        self.calls.append((query, opts))
        return list(self.hits)


def _provider(name: str, *hits: SearchHit) -> _RecordingProvider:
    return _RecordingProvider(name=name, hits=list(hits), calls=[])


def _cfg(**overrides) -> WebToolsConfig:
    base = dict(
        critic_web_tools_enabled=True,
        search_provider="tavily",
        search_rate_limit_per_min=30,
        search_max_results_cap=10,
    )
    base.update(overrides)
    return WebToolsConfig(**base)


# ── WebSearchInput.from_payload ─────────────────────────────────────────

class TestInputValidation:
    def test_rejects_short_query(self):
        with pytest.raises(ValueError, match="at least 2"):
            WebSearchInput.from_payload({"query": "x"}, 10)

    def test_rejects_blank_query(self):
        with pytest.raises(ValueError):
            WebSearchInput.from_payload({"query": "   "}, 10)

    def test_max_results_clamped(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "max_results": 999}, 5,
        )
        assert out.max_results == 5
        out = WebSearchInput.from_payload(
            {"query": "abc", "max_results": 0}, 10,
        )
        assert out.max_results == 1

    def test_max_results_default_5(self):
        out = WebSearchInput.from_payload({"query": "abc"}, 10)
        assert out.max_results == 5

    def test_invalid_max_results_falls_back(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "max_results": "no"}, 10,
        )
        assert out.max_results == 5

    def test_site_promoted_to_allowed_domains(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "site": "Docs.SgLang.AI"}, 10,
        )
        assert out.allowed_domains == ("docs.sglang.ai",)

    def test_site_ignored_when_allowed_set(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "site": "a.com", "allowed_domains": ["b.com"]}, 10,
        )
        assert out.allowed_domains == ("b.com",)

    def test_allowed_and_blocked_both_rejected(self):
        with pytest.raises(ValueError, match="cannot both"):
            WebSearchInput.from_payload(
                {"query": "abc", "allowed_domains": ["a"], "blocked_domains": ["b"]},
                10,
            )

    def test_freshness_any_becomes_none(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "freshness": "any"}, 10,
        )
        assert out.freshness is None

    def test_freshness_unknown_becomes_none(self):
        out = WebSearchInput.from_payload(
            {"query": "abc", "freshness": "decade"}, 10,
        )
        assert out.freshness is None


# ── WebSearchClient.execute ─────────────────────────────────────────────

class TestExecute:
    def test_returns_disabled_error_when_no_provider(self):
        client = WebSearchClient(config=_cfg(), providers=())
        assert "web search disabled" in client.execute({"query": "abc"})

    def test_returns_input_error_intact(self):
        client = WebSearchClient(config=_cfg(), providers=(_provider("t"),))
        assert client.execute({"query": "x"}).startswith("Error: query must be at least")

    def test_happy_path_format_and_cite_reminder(self):
        p = _provider("t",
                      SearchHit(title="T1", url="https://a/b", snippet="snip1"),
                      SearchHit(title="T2", url="https://c/d", snippet=""))
        client = WebSearchClient(config=_cfg(), providers=(p,))
        out = client.execute({"query": "abc"})
        assert 'Web search results for query: "abc"' in out
        assert "Links: " in out
        # Snippet rendered only when non-empty
        assert "[T1](https://a/b): snip1" in out
        assert "[T2](https://c/d):" not in out
        # Cite reminder must be present
        assert "REMINDER" in out
        # Links payload uses JSON
        line = next(line for line in out.splitlines() if line.startswith("Links:"))
        parsed = json.loads(line[len("Links: "):])
        assert parsed == [
            {"title": "T1", "url": "https://a/b"},
            {"title": "T2", "url": "https://c/d"},
        ]

    def test_no_links_message_when_empty(self):
        p = _provider("t")
        client = WebSearchClient(config=_cfg(), providers=(p,))
        out = client.execute({"query": "abc"})
        assert "No links found." in out
        assert "REMINDER" in out

    def test_fallback_chain_on_provider_error(self):
        class FailingProvider:
            name = "tavily"
            def search(self, query, opts):
                raise ProviderError("503")
        p2 = _provider("serper", SearchHit("T", "https://x/y"))
        client = WebSearchClient(config=_cfg(), providers=(FailingProvider(), p2))
        out = client.execute({"query": "abc"})
        assert "[T](https://x/y)" in out or "https://x/y" in out

    def test_all_providers_fail_returns_aggregated_error(self):
        class Bad1:
            name = "tavily"
            def search(self, q, o):
                raise ProviderError("a")
        class Bad2:
            name = "serper"
            def search(self, q, o):
                raise ProviderError("b")
        client = WebSearchClient(config=_cfg(), providers=(Bad1(), Bad2()))
        out = client.execute({"query": "abc"})
        assert out.startswith("Error: all web search providers failed")
        assert "serper: b" in out

    def test_rate_limit_blocks_extra_calls(self):
        p = _provider("t", SearchHit("T", "https://a/b"))
        client = WebSearchClient(
            config=_cfg(search_rate_limit_per_min=1), providers=(p,),
        )
        out1 = client.execute({"query": "abc"})
        out2 = client.execute({"query": "abc"})
        assert "REMINDER" in out1
        assert out2.startswith("Error: web search rate limit exceeded")

    def test_global_denylist_strips_hits(self):
        cfg = _cfg(search_domain_denylist=("spam.io",))
        p = _provider("t",
                      SearchHit("ok", "https://a.com/x"),
                      SearchHit("bad", "https://api.spam.io/y"))
        client = WebSearchClient(config=cfg, providers=(p,))
        out = client.execute({"query": "abc"})
        assert "https://a.com/x" in out
        assert "https://api.spam.io/y" not in out

    def test_global_denylist_merged_into_provider_blocked(self):
        cfg = _cfg(search_domain_denylist=("spam.io",))
        p = _provider("t")
        client = WebSearchClient(config=cfg, providers=(p,))
        client.execute({"query": "abc", "blocked_domains": ["evil.com"]})
        _, opts = p.calls[-1]
        assert set(opts.blocked_domains) == {"spam.io", "evil.com"}

    def test_global_denylist_not_sent_when_allowed_domains_set(self):
        cfg = _cfg(search_domain_denylist=("spam.io",))
        p = _provider("t")
        client = WebSearchClient(config=cfg, providers=(p,))
        client.execute({"query": "abc", "allowed_domains": ["docs.sglang.ai"]})
        _, opts = p.calls[-1]
        # Anthropic-style mutual-exclusion rule: don't send both at once
        assert opts.blocked_domains == ()
        assert opts.allowed_domains == ("docs.sglang.ai",)

    def test_allowed_domain_post_filter(self):
        p = _provider("t",
                      SearchHit("in", "https://docs.sglang.ai/api"),
                      SearchHit("out", "https://random.example/api"))
        client = WebSearchClient(config=_cfg(), providers=(p,))
        out = client.execute({"query": "abc", "allowed_domains": ["docs.sglang.ai"]})
        assert "docs.sglang.ai" in out
        assert "random.example" not in out


# ── _LeakyBucket ────────────────────────────────────────────────────────

class TestLeakyBucket:
    def test_drains_capacity_then_blocks(self):
        now = [0.0]
        bucket = _LeakyBucket(2, time_fn=lambda: now[0])
        assert bucket.try_consume()
        assert bucket.try_consume()
        assert bucket.try_consume() is False

    def test_refills_after_one_minute(self):
        now = [0.0]
        bucket = _LeakyBucket(2, time_fn=lambda: now[0])
        assert bucket.try_consume()
        assert bucket.try_consume()
        assert bucket.try_consume() is False
        now[0] = 60.1
        assert bucket.try_consume()
        assert bucket.try_consume()

    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            _LeakyBucket(0)
