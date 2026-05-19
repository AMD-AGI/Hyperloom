"""KnowledgePlane facade — v0.8 M1+M4 (KB_design §3.6).

Single point of contact between the Coordinator and the three
external knowledge sources:

1. **Cortex KB** — cross-session optimization memory + hypothesize /
   verify / commit protocol. Wraps :class:`CortexKBClient` (M1).
2. **PR Monitor** — recent PRs across repos the LLM specialists care
   about. Wraps :class:`PRMonitorClient` (M4).
3. **Local framework source** — read-only access to
   ``/sgl-workspace/{aiter,sglang,vllm}/`` (and operator-supplied
   roots). Exposed via the existing PolicyGate-gated tool whitelist;
   the facade only surfaces the *roots* so specialist prompt
   assembly can advertise them in section 7.

The facade is **stateless** (it holds the two clients + the
domain-repos config) and offers two flavours of methods:

* ``read_*`` — used everywhere (Coordinator prompt assembly, breakdown
  collectors). Reads never write to SharedState directly.
* ``write_*`` — used only by the Coordinator's T0/T2/T3/T4 hooks.
  Implements Inv-6.2 ("writes must go through the Coordinator"); the
  facade itself doesn't enforce ACLs but isolates the write surface so
  the PolicyGate gate sits one call up.

KB_design §3.6 §4.3 says specialists DON'T use this facade — they get
direct MCP tool access. The facade is purely Coordinator-side glue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..cortex_kb_client import CortexKBClient
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


# ---------------------------------------------------------------------------
# domain → repos config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainRepos:
    """Resolved domain-repos entry for one specialist domain.

    ``repos`` is a tuple of canonical repo names (``owner/name``);
    a wildcard ``"*"`` in the yaml becomes the special sentinel
    :data:`DOMAIN_REPOS_WILDCARD` that the facade later expands to the
    PR Monitor's full repo list at call time.
    """

    domain: str
    repos: tuple[str, ...]
    default_keywords: tuple[str, ...]
    is_wildcard: bool = False


DOMAIN_REPOS_WILDCARD: str = "*"


_DOMAIN_REPOS_FILENAME: str = "_domain_repos.yaml"


def _domain_repos_path() -> Path:
    """Where ``_domain_repos.yaml`` lives. Centralised so tests can
    monkeypatch the resolution if they ever need to."""
    return asset_actions_dir() / "_meta" / _DOMAIN_REPOS_FILENAME


def load_domain_repos(path: Path | None = None) -> dict[str, DomainRepos]:
    """Load the domain → repos config from yaml. Idempotent; safe to
    call repeatedly (no caching layered on; each call re-parses).

    Returns ``{domain_key: DomainRepos}`` with the six M5/M6
    specialist domains pre-populated. Missing / malformed yaml falls
    back to an empty dict + a warning log; callers should treat that
    as "no PR feed for any domain".
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


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
@dataclass
class KnowledgePlane:
    """Single facade for the three knowledge sources.

    Construction is cheap (no I/O); the clients are injected. The
    facade is meant to live for the lifetime of one Coordinator run;
    PR Monitor in-memory caches are cleared between EXPLORE rounds
    by :meth:`reset_round_caches`.
    """

    cortex_kb: CortexKBClient | None = None
    pr_monitor: PRMonitorClient | None = None
    domain_repos: dict[str, DomainRepos] = field(default_factory=dict)
    pr_feed_window_days: int = DEFAULT_PR_FEED_WINDOW_DAYS
    pr_feed_per_repo_limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT
    pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL
    # Aggregated warnings from the latest pr_feed_warm call. Exposed
    # for the breakdown collector + prompt assembly to surface
    # "pr_monitor_unreachable" lines.
    last_warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_clients(
        cls,
        *,
        cortex_kb: CortexKBClient | None,
        pr_monitor: PRMonitorClient | None,
        domain_repos: dict[str, DomainRepos] | None = None,
        pr_feed_window_days: int = DEFAULT_PR_FEED_WINDOW_DAYS,
        pr_feed_per_repo_limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT,
        pr_monitor_mcp_url: str = DEFAULT_PR_MONITOR_MCP_URL,
    ) -> "KnowledgePlane":
        return cls(
            cortex_kb=cortex_kb,
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

    # ------------------------------------------------------------------
    # Read surface (KB_design §3.6 §4.1)
    # ------------------------------------------------------------------
    @property
    def pr_monitor_enabled(self) -> bool:
        return self.pr_monitor is not None and self.pr_monitor.enabled

    @property
    def cortex_enabled(self) -> bool:
        return self.cortex_kb is not None and self.cortex_kb.enabled

    def resolve_domain_repos(self, domain: str) -> DomainRepos | None:
        """Look up domain config; returns None for unknown domains."""
        return self.domain_repos.get(domain)

    def specialist_mcp_url(self) -> str:
        """MCP URL to advertise in the specialist tool whitelist.

        Returns ``""`` when PR Monitor is disabled so the specialist
        runner can elide the ``mcp__pr_monitor__*`` tool block instead
        of dangling a broken endpoint.
        """
        if not self.pr_monitor_enabled:
            return ""
        return self.pr_monitor_mcp_url

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

        Returns ``(prs, warnings)``. ``warnings`` is also stashed on
        :attr:`last_warnings` so the breakdown collector can surface
        them in the ``warnings`` section.

        Failure semantics (KB_design §3.13 M4 §10):

        - PR Monitor disabled (``--no-pr-monitor``) → ``([],
          ["pr_monitor:disabled"])``.
        - Unknown domain → ``([], ["pr_monitor:unknown_domain:<d>"])``.
        - Per-repo unreachability is folded into ``warnings`` (PRs
          from reachable repos still surface).
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
        # The PR Monitor client returns warnings inline; merge them.
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

    # ------------------------------------------------------------------
    # Write surface (KB_design §3.6 §4.2) — all writes are
    # Coordinator-only; PolicyGate sits one call up.
    # ------------------------------------------------------------------
    def cortex_propose_point(
        self,
        *,
        canonical_id: str,
        kind: str,
        attrs: Mapping[str, Any] | None = None,
        authority: str = "HYPOTHESIZED",
        evidence: list[str] | None = None,
        source: str = "agent_observation",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Forward to CortexKBClient.propose_point.

        Returns ``{"status": "skip_disabled"}`` when Cortex is
        disabled so callers can branchlessly handle the
        ``--no-cortex`` case.
        """
        if not self.cortex_enabled:
            return {"status": "skip_disabled"}
        return self.cortex_kb.propose_point(  # type: ignore[union-attr]
            canonical_id=canonical_id,
            kind=kind,
            attrs=attrs,
            authority=authority,
            evidence=evidence,
            source=source,
            idempotency_key=idempotency_key,
        )

    def cortex_hypothesize(
        self,
        *,
        sid: str,
        from_canonical: str,
        to_canonical: str,
        edge_type: str = "hypothetical",
        reason: str = "",
        attrs: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.cortex_enabled:
            return {"status": "skip_disabled", "tentative_edge_id": ""}
        return self.cortex_kb.hypothesize(  # type: ignore[union-attr]
            sid=sid,
            from_canonical=from_canonical,
            to_canonical=to_canonical,
            edge_type=edge_type,
            reason=reason,
            attrs=attrs,
            evidence=evidence,
            idempotency_key=idempotency_key,
        )

    def mint_pr_node(
        self,
        *,
        repo: str,
        number: int,
        url: str = "",
        sha: str = "",
        attrs: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mint a Cortex ``pr_node`` for a PR referenced by a specialist.

        KB_design §3.6 §5.2: when a specialist puts a PR reference in
        ``specialist_done.proposal_set[i].pr_evidence``, the Coordinator
        mint it as a Cortex point so future sessions can cite the
        same canonical_id (``pr.<repo>#<number>`` or
        ``pr.<repo>@<sha>``).

        ``url`` defaults to the GitHub PR URL derived from
        ``(repo, number)`` so a minimal call site only needs
        ``(repo, number)``.

        Returns the CortexKBClient.propose_point response, or
        ``{"status": "skip_disabled"}`` when Cortex isn't wired.
        """
        canonical = pr_node_canonical_id(repo, number=number, sha=sha)
        derived_url = url or (
            f"https://github.com/{repo}/pull/{int(number)}"
            if number and number > 0 else ""
        )
        merged_attrs: dict[str, Any] = {
            "repo":   repo,
            "number": int(number) if number else 0,
            "sha":    sha or "",
            "url":    derived_url,
        }
        if attrs:
            merged_attrs.update(attrs)
        merged_evidence: list[str] = list(evidence or [])
        if derived_url and not any("url:" in e for e in merged_evidence):
            merged_evidence.append(f"url:{derived_url}")
        return self.cortex_propose_point(
            canonical_id=canonical,
            kind="pr_node",
            attrs=merged_attrs,
            authority="EXPERIENTIAL",
            evidence=merged_evidence,
            source="pr_monitor",
            idempotency_key=f"mint_pr_node:{canonical}",
        )


def pr_node_canonical_id(repo: str, *, number: int = 0, sha: str = "") -> str:
    """Derive the Cortex canonical_id for one PR (KB_design §3.6 §5.2).

    Priority: ``pr.<repo>#<number>`` when ``number`` is positive;
    otherwise ``pr.<repo>@<sha>``. Falls back to
    ``pr.<repo>.unknown`` (defensive — should never happen since the
    caller always knows at least one of the two).
    """
    repo_clean = str(repo or "").strip()
    if not repo_clean:
        repo_clean = "unknown_repo"
    if number and int(number) > 0:
        return f"pr.{repo_clean}#{int(number)}"
    if sha and str(sha).strip():
        return f"pr.{repo_clean}@{str(sha).strip()}"
    return f"pr.{repo_clean}.unknown"


__all__ = [
    "DOMAIN_REPOS_WILDCARD",
    "DomainRepos",
    "KnowledgePlane",
    "load_domain_repos",
    "pr_node_canonical_id",
]
