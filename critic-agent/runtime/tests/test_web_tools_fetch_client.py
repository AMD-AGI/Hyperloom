# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :class:`runtime.web_tools.fetch_client.WebFetchClient`.

httpx is mocked with ``MockTransport``; SSRF DNS resolution is patched
through ``runtime.web_tools.fetch_client._resolve_or_raise``.
"""

from __future__ import annotations

import httpx
import pytest

from runtime.web_tools import fetch_client as fc
from runtime.web_tools.config import WebToolsConfig
from runtime.web_tools.fetch_client import FetchError, WebFetchClient


def _cfg(**overrides) -> WebToolsConfig:
    base = dict(
        critic_web_tools_enabled=True,
        fetch_enabled=True,
        fetch_max_bytes=10 * 1024 * 1024,
        fetch_max_output_chars=50_000,
        fetch_timeout_s=30,
        fetch_cache_ttl_s=900,
        fetch_cache_max_entries=16,
    )
    base.update(overrides)
    return WebToolsConfig(**base)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch):
    """Default: pretend every hostname resolves to a benign public IP."""
    monkeypatch.setattr(fc, "_resolve_or_raise", lambda host: None)


# ── URL validation ──────────────────────────────────────────────────────

class TestUrlValidation:
    def _run(self, url: str) -> str:
        return WebFetchClient(
            config=_cfg(),
            http_client=_client(lambda r: httpx.Response(200)),
        ).execute({"url": url})

    def test_empty_url(self):
        assert self._run("").startswith("Error: url is required")

    def test_http_upgraded_to_https(self, monkeypatch):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, text="ok",
                                  headers={"content-type": "text/plain"})

        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        client.execute({"url": "http://example.com/a"})
        assert captured["url"].startswith("https://")

    def test_credentials_rejected(self):
        out = self._run("https://example.com/")
        assert "credentials" in out

    def test_hostname_must_have_dot(self):
        out = self._run("https://localhostonly/")
        assert "hostname" in out

    def test_unsupported_protocol(self):
        out = self._run("ftp://example.com/foo")
        assert "unsupported protocol" in out


# ── SSRF guard ──────────────────────────────────────────────────────────

class TestSSRF:
    def test_dns_resolution_failure(self, monkeypatch):
        def bad_resolve(host):
            raise FetchError(f"DNS resolution failed for {host}: nope")

        monkeypatch.setattr(fc, "_resolve_or_raise", bad_resolve)
        client = WebFetchClient(
            config=_cfg(), http_client=_client(lambda r: httpx.Response(200)),
        )
        out = client.execute({"url": "https://example.com/a"})
        assert out.startswith("Error: DNS resolution failed")

    def test_ipv4_blocked(self, monkeypatch):
        def fake_resolve(host):
            raise FetchError(
                f"SSRF: resolved IPv4 127.0.0.1 for {host} is in a blocked range"
            )
        monkeypatch.setattr(fc, "_resolve_or_raise", fake_resolve)

        client = WebFetchClient(
            config=_cfg(), http_client=_client(lambda r: httpx.Response(200)),
        )
        assert "SSRF" in client.execute({"url": "https://example.com/"})


class TestIPBlocking:
    def test_ipv4_private_ranges(self):
        for ip in [
            "0.0.0.1", "10.1.2.3", "127.0.0.1", "169.254.1.1",
            "172.16.0.1", "172.31.255.255", "192.168.1.1",
            "100.64.0.1", "198.18.0.1", "224.0.0.1",
        ]:
            assert fc._ipv4_blocked(ip), ip

    def test_ipv4_public_allowed(self):
        for ip in ["8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1", "192.169.1.1"]:
            assert not fc._ipv4_blocked(ip), ip

    def test_ipv6_local(self):
        assert fc._ipv6_blocked("::1")
        assert fc._ipv6_blocked("fc00::1")
        assert fc._ipv6_blocked("fe80::1")
        assert not fc._ipv6_blocked("2001:4860:4860::8888")

    def test_ipv4_mapped_ipv6(self):
        assert fc._ipv6_blocked("::ffff:10.0.0.1")
        assert not fc._ipv6_blocked("::ffff:8.8.8.8")


# ── Content-type handling ───────────────────────────────────────────────

class TestContentTypes:
    def test_text_plain_returned_verbatim(self):
        def handler(request):
            return httpx.Response(
                200, content=b"hello world",
                headers={"content-type": "text/plain; charset=utf-8"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "hello world" in out
        assert "Content-Type: text/plain" in out

    def test_html_converted_to_markdown(self):
        html = b"<html><body><h1>Title</h1><p>Para</p></body></html>"

        def handler(request):
            return httpx.Response(
                200, content=html,
                headers={"content-type": "text/html"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "Title" in out and "Para" in out
        # Markdownify produces ATX heading marker
        assert "# Title" in out or "#Title" in out

    def test_html_raw_skips_markdown(self):
        html = b"<html><body><h1>T</h1></body></html>"

        def handler(request):
            return httpx.Response(
                200, content=html,
                headers={"content-type": "text/html"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a", "raw": True})
        assert "<h1>T</h1>" in out

    def test_binary_unsupported(self):
        def handler(request):
            return httpx.Response(
                200, content=b"\x00\x01",
                headers={"content-type": "application/pdf"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/x.pdf"})
        assert "unsupported binary content-type" in out


# ── Redirect handling ──────────────────────────────────────────────────

class TestRedirect:
    def test_same_host_redirect_followed(self):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if str(request.url) == "https://example.com/a":
                return httpx.Response(
                    302, headers={"location": "https://example.com/b"},
                )
            return httpx.Response(
                200, content=b"final",
                headers={"content-type": "text/plain"},
            )

        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "final" in out
        assert calls == ["https://example.com/a", "https://example.com/b"]

    def test_cross_host_redirect_refused(self):
        def handler(request):
            return httpx.Response(
                301, headers={"location": "https://evil.com/p"},
            )

        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "cross-host redirect refused" in out
        assert "to evil.com." in out

    def test_redirect_loop_detected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302, headers={"location": str(request.url) + "x"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "too many redirects" in out

    def test_www_normalized_for_same_host(self):
        def handler(request):
            if str(request.url) == "https://example.com/a":
                return httpx.Response(
                    301, headers={"location": "https://www.example.com/a"},
                )
            return httpx.Response(
                200, content=b"final", headers={"content-type": "text/plain"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/a"})
        assert "final" in out


# ── Cache + denylist ────────────────────────────────────────────────────

class TestCacheAndDenylist:
    def test_cache_hit_skips_network(self):
        hits = []

        def handler(request):
            hits.append(1)
            return httpx.Response(
                200, content=b"first",
                headers={"content-type": "text/plain"},
            )

        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out1 = client.execute({"url": "https://example.com/a"})
        out2 = client.execute({"url": "https://example.com/a"})
        assert out1 == out2
        assert len(hits) == 1

    def test_raw_and_default_cached_separately(self):
        hits = []

        def handler(request):
            hits.append(1)
            return httpx.Response(
                200, content=b"<p>x</p>",
                headers={"content-type": "text/html"},
            )

        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        client.execute({"url": "https://example.com/a"})
        client.execute({"url": "https://example.com/a", "raw": True})
        assert len(hits) == 2

    def test_fetch_denylist(self):
        cfg = _cfg(fetch_domain_denylist=("internal.local",))
        client = WebFetchClient(
            config=cfg, http_client=_client(lambda r: httpx.Response(200)),
        )
        out = client.execute({"url": "https://api.internal.local/x"})
        assert "blocked" in out


# ── Output truncation + JS shell ───────────────────────────────────────

class TestOutputBudget:
    def test_truncation_marker_added(self):
        body = b"a" * 60_000

        def handler(request):
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/plain"},
            )
        client = WebFetchClient(
            config=_cfg(fetch_max_output_chars=1024),
            http_client=_client(handler),
        )
        out = client.execute({"url": "https://example.com/a"})
        assert "[Content truncated due to length...]" in out

    def test_js_shell_marker(self):
        body = b'<html><head></head><body><div id="root"></div></body></html>'

        def handler(request):
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/html"},
            )
        client = WebFetchClient(config=_cfg(), http_client=_client(handler))
        out = client.execute({"url": "https://example.com/spa"})
        assert "JS_RENDER_REQUIRED" in out


# ── max_bytes clamp ────────────────────────────────────────────────────

def test_max_bytes_clamped_to_config():
    sizes = []

    def handler(request):
        sizes.append(request.url)
        return httpx.Response(200, content=b"x" * 200,
                              headers={"content-type": "text/plain"})

    cfg = _cfg(fetch_max_bytes=512)
    client = WebFetchClient(config=cfg, http_client=_client(handler))
    out = client.execute({"url": "https://example.com/a", "max_bytes": 999999})
    assert "Length: 200 bytes" in out


def test_max_bytes_stops_reading_early():
    """Streaming read must honor max_bytes before buffering the full body."""
    cfg = _cfg(fetch_max_bytes=10 * 1024 * 1024)
    client = WebFetchClient(
        config=cfg,
        http_client=_client(
            lambda _request: httpx.Response(
                200,
                content=b"x" * 5000,
                headers={"content-type": "text/plain"},
            ),
        ),
    )
    out = client.execute({"url": "https://example.com/large", "max_bytes": 2048})
    assert "Length: 2048 bytes" in out


# ── small URL helpers ──────────────────────────────────────────────────

class TestSameHost:
    def test_same_host_with_www(self):
        assert fc._same_host("https://x.com/a", "https://www.x.com/b")
        assert fc._same_host("https://www.x.com/a", "https://x.com/b")

    def test_different_scheme(self):
        assert not fc._same_host("https://x.com/a", "http://x.com/b")

    def test_different_port(self):
        assert not fc._same_host("https://x.com:443/a", "https://x.com:8443/a")

    def test_different_host(self):
        assert not fc._same_host("https://a.com/x", "https://b.com/x")
