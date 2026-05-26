"""Environment-driven configuration for critic-agent web tools.

All names mirror Primus-Claw's ``Claw/packages/brain/src/config.ts`` so an
operator who already configured a Claw deployment can reuse the same env
file. Defaults are intentionally conservative: web tools stay **off** unless
``CRITIC_WEB_TOOLS_ENABLED=true`` AND a search provider is configured.

Reading is done lazily through :class:`WebToolsConfig.from_env` so unit tests
can build a config object directly without touching ``os.environ``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

ProviderName = Literal["disabled", "tavily", "serper", "brave"]
KNOWN_PROVIDERS: frozenset[str] = frozenset({"disabled", "tavily", "serper", "brave"})
# Providers with a working backend in ``runtime.web_tools`` today. ``brave``
# is recognized in env/config for forward-compat but not yet implemented.
IMPLEMENTED_SEARCH_PROVIDERS: frozenset[str] = frozenset({"tavily", "serper"})


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return val if val is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    )


def _normalize_provider(raw: str) -> ProviderName:
    p = raw.strip().lower()
    if p in KNOWN_PROVIDERS:
        return p  # type: ignore[return-value]
    return "disabled"


@dataclass(frozen=True)
class WebToolsConfig:
    """Resolved configuration for one critic-agent process.

    Built once at backend construction time. Immutable so it can be passed
    safely across async boundaries and into multi-thread tool execution.
    """

    # ── master switches ───────────────────────────────────────────────
    critic_web_tools_enabled: bool = False
    critic_web_max_tool_turns: int = 4

    # ── web_search ────────────────────────────────────────────────────
    search_provider: ProviderName = "disabled"
    search_fallback: tuple[str, ...] = field(default_factory=tuple)
    search_domain_denylist: tuple[str, ...] = field(default_factory=tuple)
    search_max_results_cap: int = 10
    search_rate_limit_per_min: int = 30

    tavily_api_key: str = ""
    serper_api_key: str = ""
    brave_api_key: str = ""

    # ── web_fetch ─────────────────────────────────────────────────────
    fetch_enabled: bool = False
    fetch_max_bytes: int = 10 * 1024 * 1024
    fetch_max_output_chars: int = 50_000
    fetch_timeout_s: int = 60
    fetch_domain_denylist: tuple[str, ...] = field(default_factory=tuple)
    fetch_cache_ttl_s: int = 15 * 60
    fetch_cache_max_entries: int = 256

    @classmethod
    def from_env(cls) -> "WebToolsConfig":
        provider = _normalize_provider(_env("WEB_SEARCH_PROVIDER", "disabled"))
        fallback_raw = _env("WEB_SEARCH_FALLBACK", "")
        fallback = tuple(
            p for p in (s.strip().lower() for s in fallback_raw.split(","))
            if p in KNOWN_PROVIDERS and p not in {"disabled", provider}
        )
        return cls(
            critic_web_tools_enabled=_env_bool("CRITIC_WEB_TOOLS_ENABLED", False),
            critic_web_max_tool_turns=max(1, _env_int("CRITIC_WEB_MAX_TOOL_TURNS", 4)),
            search_provider=provider,
            search_fallback=fallback,
            search_domain_denylist=_parse_csv(_env("WEB_SEARCH_DOMAIN_DENYLIST", "")),
            search_max_results_cap=max(1, _env_int("WEB_SEARCH_MAX_RESULTS_CAP", 10)),
            search_rate_limit_per_min=max(
                1, _env_int("WEB_SEARCH_RATE_LIMIT_PER_MIN", 30),
            ),
            tavily_api_key=_env("TAVILY_API_KEY", ""),
            serper_api_key=_env("SERPER_API_KEY", ""),
            brave_api_key=_env("BRAVE_API_KEY", ""),
            fetch_enabled=_env_bool("WEB_FETCH_ENABLED", False),
            fetch_max_bytes=max(1024, _env_int("WEB_FETCH_MAX_BYTES", 10 * 1024 * 1024)),
            fetch_max_output_chars=max(
                1024, _env_int("WEB_FETCH_MAX_OUTPUT_CHARS", 50_000),
            ),
            fetch_timeout_s=max(1, _env_int("WEB_FETCH_TIMEOUT_S", 60)),
            fetch_domain_denylist=_parse_csv(_env("WEB_FETCH_DOMAIN_DENYLIST", "")),
            fetch_cache_ttl_s=max(0, _env_int("WEB_FETCH_CACHE_TTL_S", 15 * 60)),
            fetch_cache_max_entries=max(
                1, _env_int("WEB_FETCH_CACHE_MAX_ENTRIES", 256),
            ),
        )

    def search_provider_chain(self) -> tuple[str, ...]:
        """Resolved provider try-order, excluding ``disabled``."""
        if self.search_provider == "disabled":
            return tuple(p for p in self.search_fallback if p != "disabled")
        return (self.search_provider, *self.search_fallback)

    def has_search_api_key(self, provider: str) -> bool:
        return {
            "tavily": bool(self.tavily_api_key),
            "serper": bool(self.serper_api_key),
            "brave": bool(self.brave_api_key),
        }.get(provider, False)


__all__ = [
    "IMPLEMENTED_SEARCH_PROVIDERS",
    "KNOWN_PROVIDERS",
    "ProviderName",
    "WebToolsConfig",
]
