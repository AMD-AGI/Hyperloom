"""WebFetchClient — single-URL fetcher with SSRF guard, HTML→Markdown
conversion, redirect handling and a TTL+LRU cache.

Mirrors the safety posture documented in Primus-Claw's
``Claw/docs/builtin-tools-design.md`` §5.2 — in particular the DNS-based
SSRF rejection (stronger than claude-code) and the same-host-only
redirect policy. Summarization (Haiku) is **not** ported because critic-
agent is already inside a reasoning loop; double-summarization wastes
tokens and obscures attribution.

Synchronous on purpose: the CriticAgentBackend wraps ``execute`` in
``asyncio.to_thread``. Keeping the client sync avoids leaking the asyncio
loop into provider mocks and matches how Primus-Claw's TS version is
structured per-request.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from cachetools import TTLCache
from markdownify import markdownify as html_to_markdown

from .config import WebToolsConfig


log = logging.getLogger(__name__)


_USER_AGENT = "Hyperloom-Critic (critic-agent; +https://github.com/AMD-AGI/Hyperloom)"
_MAX_REDIRECTS = 10
_JS_SHELL_MARKERS = (
    '<div id="root">',
    '<div id="app">',
    "<noscript>",
    "enable JavaScript",
    'id="__next"',
)


# ── SSRF guard ──────────────────────────────────────────────────────────

def _ipv4_blocked(addr: str) -> bool:
    parts = addr.split(".")
    if len(parts) != 4:
        return True
    try:
        a, b, *_ = (int(p) for p in parts)
    except ValueError:
        return True
    if a == 0:
        return True
    if a == 10:
        return True
    if a == 127:
        return True
    if a == 169 and b == 254:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    if a == 100 and 64 <= b <= 127:
        return True
    if a == 198 and b in (18, 19):
        return True
    if a >= 224:
        return True
    return False


def _ipv6_blocked(addr: str) -> bool:
    a = addr.lower()
    if a == "::1":
        return True
    if a.startswith(("fc", "fd")):
        return True
    if a.startswith(("fe8", "fe9", "fea", "feb")):
        return True
    if a.startswith("::ffff:"):
        mapped = a.split(":")[-1]
        if "." in mapped and _ipv4_blocked(mapped):
            return True
    return False


def _resolve_or_raise(hostname: str) -> None:
    """Resolve ``hostname`` and raise :class:`FetchError` for blocked IPs."""
    try:
        results = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise FetchError(f"DNS resolution failed for {hostname}: {exc}") from exc

    for family, _type, _proto, _canon, sockaddr in results:
        addr = sockaddr[0]
        if family == socket.AF_INET and _ipv4_blocked(addr):
            raise FetchError(
                f"SSRF: resolved IPv4 {addr} for {hostname} is in a blocked range",
            )
        if family == socket.AF_INET6 and _ipv6_blocked(addr):
            raise FetchError(
                f"SSRF: resolved IPv6 {addr} for {hostname} is in a blocked range",
            )


# ── URL validation ──────────────────────────────────────────────────────

class FetchError(RuntimeError):
    """Surface-able fetch failure. Message is returned to the LLM verbatim."""


def _validate_url(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise FetchError("url is required")
    candidate = raw.strip()
    if candidate.startswith("http://"):
        candidate = "https://" + candidate[len("http://") :]
    elif not candidate.startswith("https://"):
        if "://" in candidate:
            raise FetchError(
                f"unsupported protocol: only http(s) allowed (got {candidate.split('://', 1)[0]})",
            )
        candidate = "https://" + candidate

    parsed = urlparse(candidate)
    if parsed.username or parsed.password:
        raise FetchError("url must not contain credentials")
    if not parsed.hostname or "." not in parsed.hostname:
        raise FetchError("hostname must contain at least one dot")
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"unsupported protocol: {parsed.scheme}")
    return candidate


def _domain_blocked(hostname: str, denylist: tuple[str, ...]) -> bool:
    if not denylist:
        return False
    h = hostname.lower()
    return any(h == d or h.endswith("." + d) for d in denylist)


def _same_host(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)

    def _norm(host: str | None) -> str:
        return (host or "").lower().removeprefix("www.")

    return (
        _norm(pa.hostname) == _norm(pb.hostname)
        and pa.port == pb.port
        and pa.scheme == pb.scheme
    )


# ── Fetch result ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FetchResult:
    body: bytes
    content_type: str
    status_code: int
    final_url: str


def _looks_like_js_shell(text: str) -> bool:
    stripped_len = len(text) - text.count("<")
    if stripped_len > 4000:
        return False
    return any(marker in text for marker in _JS_SHELL_MARKERS)


# ── Cache key ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CacheEntry:
    body: str
    content_type: str
    status_code: int
    final_url: str
    byte_len: int


def _read_response_body_limited(resp: httpx.Response, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from a streaming response body."""
    parts: list[bytes] = []
    nbytes = 0
    for chunk in resp.iter_bytes():
        if nbytes >= max_bytes:
            break
        remaining = max_bytes - nbytes
        if len(chunk) <= remaining:
            parts.append(chunk)
            nbytes += len(chunk)
        else:
            parts.append(chunk[:remaining])
            break
    return b"".join(parts)



@dataclass
class WebFetchClient:
    """Single-URL fetcher.

    Reuses one ``httpx.Client`` across calls so connection pooling kicks
    in. The cache is per-instance (one critic process == one cache); we
    intentionally avoid a process-wide singleton so concurrent sessions
    do not leak fetched pages across security boundaries.
    """

    config: WebToolsConfig
    http_client: httpx.Client
    _cache: TTLCache = field(init=False, repr=False)
    call_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._cache = TTLCache(
            maxsize=self.config.fetch_cache_max_entries,
            ttl=max(1, self.config.fetch_cache_ttl_s),
        )

    def execute(self, payload: dict) -> str:
        """Run a single ``web_fetch`` call. Always returns a string."""
        self.call_count += 1
        url_raw = payload.get("url")
        raw_param = bool(payload.get("raw"))
        max_bytes_raw = payload.get("max_bytes")
        try:
            max_bytes = int(max_bytes_raw) if max_bytes_raw is not None else self.config.fetch_max_bytes
        except (TypeError, ValueError):
            max_bytes = self.config.fetch_max_bytes
        max_bytes = max(1024, min(max_bytes, self.config.fetch_max_bytes))

        try:
            url = _validate_url(str(url_raw or ""))
        except FetchError as exc:
            return f"Error: {exc}"

        parsed = urlparse(url)
        if _domain_blocked(parsed.hostname or "", self.config.fetch_domain_denylist):
            return (
                f"Error: domain {parsed.hostname} is blocked by "
                f"WEB_FETCH_DOMAIN_DENYLIST"
            )

        cache_key = (url, raw_param)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return _format_output(cached, truncate_at=self.config.fetch_max_output_chars)

        try:
            result = self._fetch_with_redirects(url, max_bytes)
        except FetchError as exc:
            return f"Error: {exc}"
        except httpx.HTTPError as exc:
            log.warning("web_fetch transport failure url=%s err=%s", url, exc)
            return f"Error: fetch failed — {exc}"

        ct = result.content_type
        if not (
            ct.startswith("text/")
            or ct in {"application/json", "application/xml"}
        ):
            return (
                f"Error: unsupported binary content-type {ct or '<unknown>'}. "
                "Use a browser MCP tool for this content."
            )

        try:
            body_str = result.body.decode("utf-8", errors="replace")
        except (UnicodeDecodeError, AttributeError) as exc:  # pragma: no cover
            return f"Error: failed to decode body — {exc}"

        if raw_param:
            content = body_str
        elif ct == "text/html":
            try:
                content = html_to_markdown(body_str, heading_style="ATX")
            except (RuntimeError, ValueError, AttributeError) as exc:
                log.info("html_to_markdown failed url=%s err=%s", url, exc)
                content = body_str
        else:
            content = body_str

        entry = _CacheEntry(
            body=content,
            content_type=ct,
            status_code=result.status_code,
            final_url=result.final_url,
            byte_len=len(result.body),
        )
        self._cache[cache_key] = entry

        out = _format_output(entry, truncate_at=self.config.fetch_max_output_chars)
        if not raw_param and ct == "text/html" and _looks_like_js_shell(body_str):
            out += (
                "\n\nJS_RENDER_REQUIRED: This page appears to require "
                "JavaScript rendering. Use a browser MCP tool to inspect it."
            )
        return out

    # ──────────────────────────────────────────────────────────────────
    def _fetch_with_redirects(self, start_url: str, max_bytes: int) -> _FetchResult:
        current = start_url
        for _hop in range(_MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            _resolve_or_raise(parsed.hostname or "")

            with self.http_client.stream(
                "GET",
                current,
                follow_redirects=False,
                timeout=self.config.fetch_timeout_s,
                headers={"User-Agent": _USER_AGENT},
            ) as resp:
                if 300 <= resp.status_code < 400:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(
                            f"Redirect {resp.status_code} without Location header",
                        )
                    next_url = str(httpx.URL(current).join(location))
                    next_parsed = urlparse(next_url)
                    if next_parsed.scheme not in {"http", "https"}:
                        raise FetchError(
                            f"redirect to unsupported protocol: {next_parsed.scheme}",
                        )
                    if next_parsed.username or next_parsed.password:
                        raise FetchError("redirect url must not contain credentials")
                    if not _same_host(current, next_url):
                        raise FetchError(
                            f"cross-host redirect refused — "
                            f"from {parsed.hostname} to {next_parsed.hostname}. "
                            f"Re-call web_fetch with url={next_url} if you still "
                            f"want this content."
                        )
                    current = next_url
                    continue

                body = _read_response_body_limited(resp, max_bytes)
                ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                return _FetchResult(
                    body=body,
                    content_type=ct,
                    status_code=resp.status_code,
                    final_url=current,
                )

        raise FetchError(f"too many redirects (>{_MAX_REDIRECTS})")


def _format_output(entry: _CacheEntry, *, truncate_at: int) -> str:
    header = (
        f"URL: {entry.final_url}\n"
        f"Status: {entry.status_code}\n"
        f"Content-Type: {entry.content_type or '<unknown>'}\n"
        f"Length: {entry.byte_len} bytes\n"
        f"---\n"
    )
    body = entry.body
    if len(body) > truncate_at:
        body = body[:truncate_at] + "\n\n[Content truncated due to length...]"
    return header + body


def _new_default_http_client() -> httpx.Client:
    """Helper used by the public factory; tests usually supply their own."""
    return httpx.Client(http2=False, trust_env=True)


__all__ = [
    "FetchError",
    "WebFetchClient",
    "_new_default_http_client",
]
