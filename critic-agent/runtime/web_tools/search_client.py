# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""WebSearchClient — facade over multiple :mod:`providers`.

Rate-limits (leaky bucket, 30 req/min default), validates
:class:`WebSearchInput`, tries the provider chain (falling back on
:class:`ProviderError`), applies the global ``WEB_SEARCH_DOMAIN_DENYLIST``,
and formats one provider-agnostic string (always with the cite-source
reminder). Output mirrors Primus-Claw's ``formatThirdPartyResults``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence
from urllib.parse import urlparse

from .config import WebToolsConfig
from .providers import (
    ProviderError,
    SearchHit,
    SearchOptions,
    WebSearchProvider,
)


log = logging.getLogger(__name__)


_MAX_RESULT_OUTPUT_CHARS = 100_000
_MIN_QUERY_LEN = 2
_CITE_REMINDER = (
    "\nREMINDER: You MUST include the sources above when referencing them in "
    "your response, using markdown hyperlinks [Title](URL)."
)


# ── Input validation ────────────────────────────────────────────────────

@dataclass(frozen=True)
class WebSearchInput:
    """Normalized, validated input for one ``web_search`` call.

    Tool callers pass a free-form ``dict`` (from the LLM's JSON arguments);
    :meth:`from_payload` does shape + sanity checks once so providers can
    assume invariants. The class is intentionally not a pydantic model to
    keep the runtime dependency surface tiny.
    """

    query: str
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    max_results: int = 5
    freshness: str | None = None

    @classmethod
    def from_payload(cls, raw: dict, max_results_cap: int) -> "WebSearchInput":
        query = str(raw.get("query") or "").strip()
        if len(query) < _MIN_QUERY_LEN:
            raise ValueError(
                f"query must be at least {_MIN_QUERY_LEN} characters",
            )

        allowed = _normalize_str_list(raw.get("allowed_domains"))
        blocked = _normalize_str_list(raw.get("blocked_domains"))
        site = raw.get("site")
        if site and not allowed:
            allowed = (str(site).strip().lower(),)
        if allowed and blocked:
            raise ValueError(
                "allowed_domains and blocked_domains cannot both be non-empty",
            )

        max_results_raw = raw.get("max_results")
        try:
            max_results = int(max_results_raw) if max_results_raw is not None else 5
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results_cap, max_results))

        freshness = raw.get("freshness")
        if freshness is not None and freshness not in {
            "day", "week", "month", "year", "any",
        }:
            freshness = None
        elif freshness == "any":
            freshness = None

        return cls(
            query=query,
            allowed_domains=allowed,
            blocked_domains=blocked,
            max_results=max_results,
            freshness=freshness,
        )


def _normalize_str_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip().lower()
            if s:
                out.append(s)
    return tuple(out)


# ── Rate limiter ────────────────────────────────────────────────────────

class _LeakyBucket:
    """Per-client leaky bucket.

    ``capacity`` tokens refilled linearly over one minute. ``try_consume``
    returns False when the bucket is empty (request should be rejected),
    True otherwise (consumes 1 token). Not thread-safe; the caller is
    expected to be the engine loop which is single-threaded per session.
    """

    def __init__(self, capacity: int, *, time_fn=time.monotonic) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_per_s = capacity / 60.0
        self._time = time_fn
        self._last = time_fn()

    def try_consume(self) -> bool:
        now = self._time()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_s)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ── Client ──────────────────────────────────────────────────────────────

@dataclass
class WebSearchClient:
    """Stateful, per-session web search facade.

    Construct via :meth:`build`; the engine loop reuses one instance per
    critic session. ``execute`` returns the final formatted string fed
    straight into the ``tool`` message — never raises for ordinary
    provider/transport failures (they degrade to "Error: ..." text the
    LLM can react to).
    """

    config: WebToolsConfig
    providers: Sequence[WebSearchProvider] = field(default_factory=tuple)
    _rate_limiter: _LeakyBucket = field(init=False, repr=False)
    call_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.providers:
            log.info("WebSearchClient constructed with empty provider chain")
        self._rate_limiter = _LeakyBucket(self.config.search_rate_limit_per_min)

    def execute(self, payload: dict) -> str:
        """Run a single ``web_search`` call.

        ``payload`` is the raw JSON object the LLM emitted as the tool
        arguments. Returns the formatted result string to feed back to
        the model.
        """
        self.call_count += 1
        try:
            request = WebSearchInput.from_payload(
                payload, self.config.search_max_results_cap,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        if not self.providers:
            return "Error: web search disabled (no provider configured)"

        if not self._rate_limiter.try_consume():
            return (
                f"Error: web search rate limit exceeded "
                f"({self.config.search_rate_limit_per_min} req/min). "
                f"Wait a moment before searching again."
            )

        global_deny = self.config.search_domain_denylist
        if request.allowed_domains:
            blocked_merged = ()
        else:
            blocked_merged = tuple(
                dict.fromkeys(list(request.blocked_domains) + list(global_deny))
            )
        opts = SearchOptions(
            max_results=request.max_results,
            allowed_domains=request.allowed_domains,
            blocked_domains=blocked_merged,
            freshness=request.freshness,
        )

        last_error: str | None = None
        for provider in self.providers:
            try:
                hits = provider.search(request.query, opts)
            except ProviderError as exc:
                last_error = f"{provider.name}: {exc}"
                log.info(
                    "web_search provider failed provider=%s err=%s",
                    provider.name, exc,
                )
                continue
            if global_deny:
                hits = [h for h in hits if not _host_in(h.url, global_deny)]
            if request.allowed_domains:
                hits = [
                    h for h in hits
                    if _host_in(h.url, request.allowed_domains)
                ]
            return _format_results(request.query, hits)

        if last_error:
            return f"Error: all web search providers failed; last={last_error}"
        return "Error: no web search provider available"


def _host_in(url: str, domains: Sequence[str]) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, AttributeError):
        return False
    return any(host == d or host.endswith("." + d) for d in domains)


def _format_results(query: str, hits: Sequence[SearchHit]) -> str:
    parts: list[str] = [f'Web search results for query: "{query}"', ""]
    if not hits:
        parts.append("No links found.")
    else:
        links = [{"title": h.title, "url": h.url} for h in hits]
        parts.append(f"Links: {json.dumps(links, ensure_ascii=False)}")
        parts.append("")
        for h in hits:
            if h.snippet:
                parts.append(f"[{h.title}]({h.url}): {h.snippet}")
                parts.append("")

    out = "\n".join(parts).rstrip() + _CITE_REMINDER
    if len(out) > _MAX_RESULT_OUTPUT_CHARS:
        out = (
            out[: _MAX_RESULT_OUTPUT_CHARS]
            + "\n[Results truncated due to length...]"
        )
    return out


__all__ = [
    "WebSearchClient",
    "WebSearchInput",
]
