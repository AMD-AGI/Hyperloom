# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KnowledgePlane facade for advisory knowledge inputs.

Coordinator-side glue exposing the PR Monitor (wraps
:class:`PRMonitorClient`) for specialist prompt assembly. Stateless.
KB_design §3.6 §4.3: specialists use direct MCP access, not this facade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import asset_actions_dir
from .pr_monitor import (
    DEFAULT_PR_FEED_PER_REPO_LIMIT,
    DEFAULT_PR_FEED_TOTAL_BUDGET_SEC,
    DEFAULT_PR_FEED_WINDOW_DAYS,
    DEFAULT_PR_MONITOR_MCP_URL,
    PRMonitorClient,
    PRSummary,
)
from .specialist_domains import SPECIALIST_DOMAIN_KEYS


log = logging.getLogger(__name__)


# domain → repos config
@dataclass(frozen=True)
class DomainRepos:
    """Resolved domain-repos entry for one specialist domain.

    A wildcard ``"*"`` becomes :data:`DOMAIN_REPOS_WILDCARD`, expanded to
    the PR Monitor's full repo list at call time.
    """

    domain: str
    repos: tuple[str, ...]
    default_keywords: tuple[str, ...]
    is_wildcard: bool = False


DOMAIN_REPOS_WILDCARD: str = "*"


_DOMAIN_REPOS_FILENAME: str = "_domain_repos.yaml"


def _domain_repos_path() -> Path:
    """Where ``_domain_repos.yaml`` lives (centralised for test monkeypatching)."""
    return asset_actions_dir() / "_meta" / _DOMAIN_REPOS_FILENAME


def load_domain_repos(path: Path | None = None) -> dict[str, DomainRepos]:
    """Load the domain → repos config from yaml (re-parses each call).

    Returns ``{domain_key: DomainRepos}``; missing/malformed yaml returns an
    empty dict + warning log ("no PR feed for any domain").
    """
    target = path or _domain_repos_path()
    if not target.exists():
        log.info(
            "domain_repos: %s missing; PR feed will be empty for all domains",
            target,
        )
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        log.warning("domain_repos: PyYAML not installed; cannot load %s", target)
        return {}
    try:
        with target.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("domain_repos: failed to parse %s: %s", target, exc)
        return {}
    out: dict[str, DomainRepos] = {}
    for domain_key, entry in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(entry, dict):
            continue
        if domain_key not in SPECIALIST_DOMAIN_KEYS:
            log.info(
                "domain_repos: yaml lists unknown domain=%r; ignoring",
                domain_key,
            )
            continue
        repos_field = entry.get("repos")
        is_wildcard = False
        repos: tuple[str, ...]
        if isinstance(repos_field, str) and repos_field.strip() == DOMAIN_REPOS_WILDCARD:
            repos = ()
            is_wildcard = True
        elif isinstance(repos_field, list):
            repos = tuple(
                str(r).strip() for r in repos_field
                if isinstance(r, str) and r.strip()
            )
        else:
            repos = ()
        kw_raw = entry.get("default_keywords") or []
        if isinstance(kw_raw, list):
            keywords = tuple(
                str(k).strip() for k in kw_raw
                if isinstance(k, str) and k.strip()
            )
        else:
            keywords = ()
        out[str(domain_key)] = DomainRepos(
            domain=str(domain_key),
            repos=repos,
            default_keywords=keywords,
            is_wildcard=is_wildcard,
        )
    return out


# Facade
@dataclass
class KnowledgePlane:
    """Single facade for the knowledge sources.

    Lives for one Coordinator run; PR Monitor caches are cleared between
    EXPLORE rounds by :meth:`reset_round_caches`.
    """

    pr_monitor: PRMonitorClient | None = None
    domain_repos: dict[str, DomainRepos] = field(default_factory=dict)
    pr_feed_window_days: int = DEFAULT_PR_FEED_WINDOW_DAYS
    pr_feed_per_repo_limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT
    pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL
    # Aggregated warnings from the latest pr_feed_warm call.
    last_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_clients(
        cls,
        *,
        cortex_kb: Any = None,
        pr_monitor: PRMonitorClient | None = None,
        domain_repos: dict[str, DomainRepos] | None = None,
        pr_feed_window_days: int = DEFAULT_PR_FEED_WINDOW_DAYS,
        pr_feed_per_repo_limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT,
        pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL,
    ) -> "KnowledgePlane":
        return cls(
            pr_monitor=pr_monitor,
            domain_repos=domain_repos if domain_repos is not None else load_domain_repos(),
            pr_feed_window_days=int(pr_feed_window_days),
            pr_feed_per_repo_limit=int(pr_feed_per_repo_limit),
            pr_monitor_mcp_url=pr_monitor_mcp_url,
        )

    def reset_round_caches(self) -> None:
        """Drop per-round caches at EXPLORE round boundaries."""
        if self.pr_monitor is not None:
            self.pr_monitor.reset_cache()
        self.last_warnings = []

    # Read surface
    @property
    def pr_monitor_enabled(self) -> bool:
        return self.pr_monitor is not None and self.pr_monitor.enabled

    @property
    def cortex_enabled(self) -> bool:
        # Backwards-compatible shim; KnowledgePlane no longer wires Cortex.
        return False

    def resolve_domain_repos(self, domain: str) -> DomainRepos | None:
        """Look up domain config; returns None for unknown domains."""
        return self.domain_repos.get(domain)

    def specialist_mcp_url(self) -> str:
        """MCP URL to advertise in the specialist tool whitelist.

        Returns ``""`` when PR Monitor is disabled so the runner can elide
        the ``mcp__pr_monitor__*`` tool block.
        """
        if not self.pr_monitor_enabled:
            return ""
        return self.pr_monitor_mcp_url

    def pr_feed_warm_all_domains(
        self,
        *,
        window_days: int | None = None,
        per_repo_limit: int | None = None,
        total_budget_sec: float = DEFAULT_PR_FEED_TOTAL_BUDGET_SEC,
    ) -> dict[str, tuple[list[PRSummary], list[str]]]:
        """Batch-warm the PR feed for every known specialist domain.

        Called once per EXPLORE phase entry (KB_design §3.6 §5.2). Returns a
        ``{domain: (prs, warnings)}`` map; aggregated warnings stashed on
        :attr:`last_warnings`. Fail-soft.
        """
        out: dict[str, tuple[list[PRSummary], list[str]]] = {}
        all_warnings: list[str] = []
        # SPECIALIST_DOMAIN_KEYS is authoritative; the yaml may be a subset,
        # so we still call pr_feed_warm for a consistent warning shape.
        for domain in SPECIALIST_DOMAIN_KEYS:
            try:
                prs, warnings = self.pr_feed_warm(
                    domain,
                    window_days=window_days,
                    per_repo_limit=per_repo_limit,
                    total_budget_sec=total_budget_sec,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                prs, warnings = [], [f"pr_feed_warm:{domain}:exc:{exc!r}"[:240]]
            out[domain] = (prs, warnings)
            all_warnings.extend(warnings)
        self.last_warnings = all_warnings
        return out

    def pr_feed_warm(
        self,
        domain: str,
        *,
        extra_keywords: list[str] | None = None,
        window_days: int | None = None,
        per_repo_limit: int | None = None,
        total_budget_sec: float = DEFAULT_PR_FEED_TOTAL_BUDGET_SEC,
    ) -> tuple[list[PRSummary], list[str]]:
        """Warm the PR feed for one specialist domain.

        Returns ``(prs, warnings)`` (also stashed on :attr:`last_warnings`).
        Failure semantics: disabled → ``["pr_monitor:disabled"]``; unknown
        domain → ``["pr_monitor:unknown_domain:<d>"]``; per-repo failures
        fold into ``warnings`` while reachable repos still surface.
        """
        warnings: list[str] = []
        if not self.pr_monitor_enabled:
            warnings.append("pr_monitor:disabled")
            self.last_warnings = warnings
            return [], warnings
        cfg = self.resolve_domain_repos(domain)
        if cfg is None:
            warnings.append(f"pr_monitor:unknown_domain:{domain}")
            self.last_warnings = warnings
            return [], warnings
        # Wildcard: ask PR Monitor what it monitors.
        if cfg.is_wildcard:
            try:
                repos = self.pr_monitor.list_repos()  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"pr_monitor:list_repos_failed:{exc!r}"[:240])
                self.last_warnings = warnings
                return [], warnings
            if not repos:
                warnings.append("pr_monitor:wildcard_empty")
        else:
            repos = list(cfg.repos)
        if not repos:
            self.last_warnings = warnings
            return [], warnings
        keywords = list(cfg.default_keywords)
        if extra_keywords:
            keywords.extend(extra_keywords)
        prs, fetch_warnings = self.pr_monitor.pr_feed_warm(  # type: ignore[union-attr]
            repos,
            keywords=keywords,
            window_days=window_days or self.pr_feed_window_days,
            per_repo_limit=per_repo_limit or self.pr_feed_per_repo_limit,
            total_budget_sec=total_budget_sec,
        )
        warnings.extend(fetch_warnings)
        self.last_warnings = warnings
        return prs, warnings

__all__ = [
    "DOMAIN_REPOS_WILDCARD",
    "DomainRepos",
    "KnowledgePlane",
    "load_domain_repos",
]
