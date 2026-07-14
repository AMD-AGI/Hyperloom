# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Environment-driven configuration for critic-agent web tools.

All names mirror Primus-Claw's ``Claw/packages/brain/src/config.ts`` so an
operator who already configured a Claw deployment can reuse the same env
file. Defaults are intentionally conservative: web tools stay **off** unless
``CRITIC_WEB_TOOLS_ENABLED=true`` AND a search provider is configured.

Reading is done lazily through :class:`WebToolsConfig.from_env` so unit tests
can build a config object directly without touching ``os.environ``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hyperloom.common.env import env_bool, env_int, env_str

ProviderName = Literal["disabled", "tavily", "serper", "brave"]
KNOWN_PROVIDERS: frozenset[str] = frozenset({"disabled", "tavily", "serper", "brave"})
# Providers with a working backend in ``runtime.web_tools`` today. ``brave``
# is recognized in env/config for forward-compat but not yet implemented.
IMPLEMENTED_SEARCH_PROVIDERS: frozenset[str] = frozenset({"tavily", "serper"})


def _parse_csv(raw: str) -> tuple[str, ...]:
    """Split a comma-separated string into trimmed, lowercased items.

    Args:
        raw (str): The comma-separated source string.

    Returns:
        tuple[str, ...]: Non-empty, trimmed, lowercased items.
    """
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _normalize_provider(raw: str) -> ProviderName:
    """Normalise a provider name, defaulting unknown values to ``disabled``.

    Args:
        raw (str): Raw provider name from config/env.

    Returns:
        ProviderName: A known provider name, or ``"disabled"``.
    """
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
        """Build a config instance from process environment variables.

        Reads the ``CRITIC_WEB_*``, ``WEB_SEARCH_*``, ``WEB_FETCH_*`` and
        provider API-key variables, applying conservative clamps/defaults.

        Returns:
            WebToolsConfig: The resolved, immutable configuration.
        """
        provider = _normalize_provider(env_str("WEB_SEARCH_PROVIDER", "disabled"))
        fallback_raw = env_str("WEB_SEARCH_FALLBACK", "")
        fallback = tuple(
            p
            for p in (s.strip().lower() for s in fallback_raw.split(","))
            if p in KNOWN_PROVIDERS and p not in {"disabled", provider}
        )
        return cls(
            critic_web_tools_enabled=env_bool("CRITIC_WEB_TOOLS_ENABLED", False),
            critic_web_max_tool_turns=max(1, env_int("CRITIC_WEB_MAX_TOOL_TURNS", 4)),
            search_provider=provider,
            search_fallback=fallback,
            search_domain_denylist=_parse_csv(env_str("WEB_SEARCH_DOMAIN_DENYLIST", "")),
            search_max_results_cap=max(1, env_int("WEB_SEARCH_MAX_RESULTS_CAP", 10)),
            search_rate_limit_per_min=max(
                1,
                env_int("WEB_SEARCH_RATE_LIMIT_PER_MIN", 30),
            ),
            tavily_api_key=env_str("TAVILY_API_KEY", ""),
            serper_api_key=env_str("SERPER_API_KEY", ""),
            brave_api_key=env_str("BRAVE_API_KEY", ""),
            fetch_enabled=env_bool("WEB_FETCH_ENABLED", False),
            fetch_max_bytes=max(1024, env_int("WEB_FETCH_MAX_BYTES", 10 * 1024 * 1024)),
            fetch_max_output_chars=max(
                1024,
                env_int("WEB_FETCH_MAX_OUTPUT_CHARS", 50_000),
            ),
            fetch_timeout_s=max(1, env_int("WEB_FETCH_TIMEOUT_S", 60)),
            fetch_domain_denylist=_parse_csv(env_str("WEB_FETCH_DOMAIN_DENYLIST", "")),
            fetch_cache_ttl_s=max(0, env_int("WEB_FETCH_CACHE_TTL_S", 15 * 60)),
            fetch_cache_max_entries=max(
                1,
                env_int("WEB_FETCH_CACHE_MAX_ENTRIES", 256),
            ),
        )

    def search_provider_chain(self) -> tuple[str, ...]:
        """Resolved provider try-order, excluding ``disabled``.

        Returns:
            tuple[str, ...]: The primary provider followed by fallbacks, or
            just the fallbacks when the primary is ``disabled``.
        """
        if self.search_provider == "disabled":
            return tuple(p for p in self.search_fallback if p != "disabled")
        return (self.search_provider, *self.search_fallback)

    def has_search_api_key(self, provider: str) -> bool:
        """Report whether an API key is configured for a provider.

        Args:
            provider (str): Provider name (e.g. ``tavily``).

        Returns:
            bool: True when a non-empty key is set for ``provider``.
        """
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
