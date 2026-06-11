# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR Monitor REST client — v0.8 M4.

Stdlib-only client for the ``primus-cortex-pr-api`` REST surface. Fail-soft
(Inv-6.3), read-only, cross-cluster aware (KB_design §3.14 R-02).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


log = logging.getLogger(__name__)


# Default in-cluster service URL; operator overrides via ``--pr-monitor-url``.
DEFAULT_PR_MONITOR_URL: str = (
    "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1"
)
# MCP URL passed to specialist LLM backend; trailing slash mandatory.
DEFAULT_PR_MONITOR_MCP_URL: str = (
    "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/mcp/"
)

DEFAULT_PR_FEED_WINDOW_DAYS: int = 30

# Per-repo request limit (REST max is 200; kept smaller for compact prompts).
DEFAULT_PR_FEED_PER_REPO_LIMIT: int = 25

# Single HTTP call timeout — fail fast rather than hold the Orchestration tick.
DEFAULT_PR_MONITOR_TIMEOUT_SEC: float = 5.0


# Total wall-clock budget for one ``pr_feed_warm`` invocation.
DEFAULT_PR_FEED_TOTAL_BUDGET_SEC: float = 15.0


@dataclass
class PRSummary:
    """One PR row returned by ``pr_feed_warm`` / ``list_prs``."""

    repo: str
    number: int
    title: str
    url: str
    state: str
    labels: tuple[str, ...] = ()
    author: str = ""
    merged_at: str = ""
    updated_at: str = ""
    body_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the PR summary to a JSON-friendly dict.

        Returns:
            dict[str, Any]: All fields, with ``labels`` rendered as a
            list.
        """
        return {
            "repo":         self.repo,
            "number":       self.number,
            "title":        self.title,
            "url":          self.url,
            "state":        self.state,
            "labels":       list(self.labels),
            "author":       self.author,
            "merged_at":    self.merged_at,
            "updated_at":   self.updated_at,
            "body_snippet": self.body_snippet,
        }


class PRMonitorError(RuntimeError):
    """Raised for unrecoverable PR Monitor interactions.

    Most callers treat it as "PR feed unavailable" (KB_design §3.14 R-02).
    """


@dataclass
class PRMonitorClient:
    """Stdlib-only REST client for the PR Monitor surface.

    ``enabled=False`` (``--degraded-pr``) turns every call into a no-op
    returning empty data (KB_design §3.13 M4).
    """

    base_url: str = DEFAULT_PR_MONITOR_URL
    enabled: bool = True
    timeout_sec: float = DEFAULT_PR_MONITOR_TIMEOUT_SEC
    user_agent: str = "inference-optimizer/v0.8 (PRMonitorClient)"
    # Per-tick URL-keyed cache; cleared via ``reset_cache``.
    _cache: dict[str, list[PRSummary]] = field(default_factory=dict)

    @classmethod
    def from_args(
        cls,
        *,
        url: str | None = None,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ) -> "PRMonitorClient":
        """Build a client, resolving config from args then env vars.

        Args:
            url (str | None): Explicit base URL; falls back to
                ``PR_MONITOR_URL`` / ``PRIMUS_CORTEX_PR_URL`` env vars
                then :data:`DEFAULT_PR_MONITOR_URL`.
            enabled (bool): Whether the client issues real requests.
            timeout_sec (float | None): Per-request timeout; falls back
                to the ``PR_MONITOR_TIMEOUT_SEC`` env var then the
                default.

        Returns:
            PRMonitorClient: The configured client instance.
        """
        resolved_url = (
            url
            or os.environ.get("PR_MONITOR_URL", "").strip()
            or os.environ.get("PRIMUS_CORTEX_PR_URL", "").strip()
            or DEFAULT_PR_MONITOR_URL
        ).rstrip("/")
        if timeout_sec is None:
            try:
                timeout_sec = float(
                    os.environ.get(
                        "PR_MONITOR_TIMEOUT_SEC",
                        str(DEFAULT_PR_MONITOR_TIMEOUT_SEC),
                    )
                )
            except (TypeError, ValueError):
                timeout_sec = DEFAULT_PR_MONITOR_TIMEOUT_SEC
        return cls(
            base_url=resolved_url,
            enabled=bool(enabled),
            timeout_sec=float(timeout_sec or DEFAULT_PR_MONITOR_TIMEOUT_SEC),
        )

    def reset_cache(self) -> None:
        """Drop the per-tick cache (call between EXPLORE rounds)."""
        self._cache.clear()

    # Low-level HTTP helper
    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``base_url + path`` with ``params``; parsed JSON or raises.

        Response size capped at 4 MiB as defense-in-depth.

        Args:
            path: The endpoint path appended to ``base_url``.
            params: Optional query parameters (empty/``None`` values dropped).

        Returns:
            The parsed JSON response.

        Raises:
            PRMonitorError: If the client is disabled, the request fails, or
                the response is not valid JSON.
        """
        if not self.enabled:
            raise PRMonitorError("PR Monitor client disabled (--degraded-pr)")
        query = ""
        if params:
            cleaned = {
                k: str(v) for k, v in params.items()
                if v is not None and str(v) != ""
            }
            if cleaned:
                query = "?" + urllib.parse.urlencode(cleaned)
        url = f"{self.base_url}{path}{query}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                payload = resp.read(4 * 1024 * 1024)
        except urllib.error.HTTPError as exc:
            raise PRMonitorError(
                f"PR Monitor HTTP {exc.code} for {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PRMonitorError(
                f"PR Monitor unreachable at {url}: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise PRMonitorError(
                f"PR Monitor timeout/IO error at {url}: {exc}"
            ) from exc
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PRMonitorError(
                f"PR Monitor non-JSON response at {url}: {exc}"
            ) from exc

    # REST endpoint wrappers
    def healthz(self) -> bool:
        """Probe the PR Monitor health endpoint.

        Returns:
            bool: ``True`` when ``/healthz`` responds successfully;
            ``False`` on any :class:`PRMonitorError`.
        """
        try:
            self._get_json("/healthz")
            return True
        except PRMonitorError as exc:
            log.info("pr_monitor.healthz failed: %s", exc)
            return False

    def list_repos(self) -> list[str]:
        """Return the list of active repo names PR Monitor knows about.

        Accepts either the array or ``{"items": [...]}`` shape and either
        ``repo_name``/``name`` field; skips ``is_active=False`` entries.

        Returns:
            The active repo names, or ``[]`` on failure.
        """
        try:
            data = self._get_json("/repos")
        except PRMonitorError as exc:
            log.info("pr_monitor.list_repos failed: %s", exc)
            return []
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        names: list[str] = []
        for entry in items:
            if isinstance(entry, dict):
                if entry.get("is_active") is False:
                    continue
                name = str(
                    entry.get("repo_name") or entry.get("name") or ""
                ).strip()
            else:
                name = str(entry).strip()
            if name:
                names.append(name)
        return names

    def list_prs(
        self,
        repo: str,
        *,
        state: str = "all",
        since: str | None = None,
        limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT,
    ) -> list[PRSummary]:
        """List recent PRs for ``repo``.

        Returns :class:`PRSummary` list (empty on failure). Cached by
        rendered URL to avoid re-hitting the network within a tick.

        Args:
            repo: Canonical ``owner/name`` repo identifier.
            state: PR state filter (``all`` / ``open`` / ``closed``).
            since: Optional ISO lower bound on ``updated_at``.
            limit: Maximum number of PRs to fetch.

        Returns:
            The PR summaries, or ``[]`` on failure.
        """
        params = {
            "state": state,
            "limit": int(limit) if limit and limit > 0 else
                     DEFAULT_PR_FEED_PER_REPO_LIMIT,
        }
        if since:
            params["since"] = since
        cache_key = (
            f"/repos/{repo}/prs"
            + (("?" + urllib.parse.urlencode(params)) if params else "")
        )
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        try:
            data = self._get_json(f"/repos/{repo}/prs", params=params)
        except PRMonitorError as exc:
            log.info("pr_monitor.list_prs(%s) failed: %s", repo, exc)
            self._cache[cache_key] = []
            return []
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            self._cache[cache_key] = []
            return []
        prs: list[PRSummary] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            number = entry.get("number") or entry.get("pr_number")
            try:
                number_i = int(number) if number is not None else 0
            except (TypeError, ValueError):
                continue
            if number_i <= 0:
                continue
            labels_raw = entry.get("labels") or []
            if isinstance(labels_raw, list):
                labels = tuple(
                    (l.get("name") if isinstance(l, dict) else str(l))
                    for l in labels_raw if l
                )
            else:
                labels = ()
            url = str(
                entry.get("html_url")
                or entry.get("url")
                or f"https://github.com/{repo}/pull/{number_i}"
            )
            body = str(entry.get("body") or "")
            body_snippet = (body[:280] + "…") if len(body) > 280 else body
            prs.append(PRSummary(
                repo=repo,
                number=number_i,
                title=str(entry.get("title") or "").strip(),
                url=url,
                state=str(entry.get("state") or "").strip(),
                labels=tuple(str(l).strip() for l in labels if l),
                author=str(
                    (entry.get("author") or {}).get("login")
                    if isinstance(entry.get("author"), dict)
                    else (entry.get("author") or entry.get("user") or "")
                ).strip(),
                merged_at=str(entry.get("merged_at") or "").strip(),
                updated_at=str(entry.get("updated_at") or "").strip(),
                body_snippet=body_snippet,
            ))
        self._cache[cache_key] = list(prs)
        return prs

    def get_pr(self, repo: str, number: int) -> dict[str, Any] | None:
        """Fetch the full detail payload for one PR.

        Args:
            repo (str): Canonical ``owner/name`` repo identifier.
            number (int): PR number.

        Returns:
            dict[str, Any] | None: The PR detail dict, or ``None`` on
            failure.
        """
        try:
            return self._get_json(f"/repos/{repo}/prs/{int(number)}")
        except PRMonitorError as exc:
            log.info("pr_monitor.get_pr(%s#%s) failed: %s", repo, number, exc)
            return None

    # High-level helper used by KnowledgePlane.pr_feed_warm
    def pr_feed_warm(
        self,
        repos: list[str],
        *,
        keywords: list[str] | None = None,
        window_days: int = DEFAULT_PR_FEED_WINDOW_DAYS,
        per_repo_limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT,
        total_budget_sec: float = DEFAULT_PR_FEED_TOTAL_BUDGET_SEC,
        now: datetime | None = None,
    ) -> tuple[list[PRSummary], list[str]]:
        """Return ``(prs, warnings)`` for the union of ``repos``.

        Args:
            repos: The repos to fetch and union.
            keywords: Optional keyword filter applied to each PR.
            window_days: Lookback window in days for the ``since`` bound.
            per_repo_limit: Maximum PRs fetched per repo.
            total_budget_sec: Total wall-clock budget across all repos.
            now: Optional reference time (for tests); defaults to current UTC.

        Returns:
            A ``(prs, warnings)`` tuple; PRs are sorted newest-first.
        """
        out: list[PRSummary] = []
        warns: list[str] = []
        if not self.enabled:
            warns.append("pr_monitor:disabled")
            return out, warns
        if not repos:
            return out, warns
        now = now or datetime.now(timezone.utc)
        since_dt = now - timedelta(days=max(1, int(window_days)))
        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        import time as _time
        deadline = _time.monotonic() + float(total_budget_sec or 0.0)
        kw_lower = [k.lower() for k in (keywords or []) if k]
        for repo in repos:
            if total_budget_sec and _time.monotonic() > deadline:
                warns.append(f"pr_monitor:budget_exhausted:{repo}")
                continue
            try:
                rows = self._list_prs_raising(
                    repo, state="all", since=since_iso, limit=per_repo_limit,
                )
            except PRMonitorError as exc:
                warns.append(f"pr_monitor:fetch_failed:{repo}:{exc}"[:240])
                continue
            except Exception as exc:  # noqa: BLE001 — defensive
                warns.append(f"pr_monitor:exception:{repo}:{exc!r}"[:240])
                continue
            if not rows:
                continue
            if kw_lower:
                rows = [
                    r for r in rows
                    if _matches_keywords(r, kw_lower)
                ]
            out.extend(rows)
        out.sort(
            key=lambda r: (r.updated_at or r.merged_at or ""),
            reverse=True,
        )
        return out, warns

    def _list_prs_raising(
        self,
        repo: str,
        *,
        state: str = "all",
        since: str | None = None,
        limit: int = DEFAULT_PR_FEED_PER_REPO_LIMIT,
    ) -> list[PRSummary]:
        """Same as :meth:`list_prs` but re-raises :class:`PRMonitorError`.

        Lets :meth:`pr_feed_warm` distinguish empty-window from fetch-failed.

        Args:
            repo: Canonical ``owner/name`` repo identifier.
            state: PR state filter (``all`` / ``open`` / ``closed``).
            since: Optional ISO lower bound on ``updated_at``.
            limit: Maximum number of PRs to fetch.

        Returns:
            The PR summaries for the repo.

        Raises:
            PRMonitorError: If the underlying HTTP fetch fails.
        """
        params: dict[str, Any] = {
            "state": state,
            "limit": int(limit) if limit and limit > 0 else
                     DEFAULT_PR_FEED_PER_REPO_LIMIT,
        }
        if since:
            params["since"] = since
        data = self._get_json(f"/repos/{repo}/prs", params=params)
        items = data.get("items") if isinstance(data, dict) else data
        prs: list[PRSummary] = []
        if not isinstance(items, list):
            return prs
        for entry in items:
            if not isinstance(entry, dict):
                continue
            number_raw = entry.get("number") or entry.get("pr_number")
            try:
                number_i = int(number_raw) if number_raw is not None else 0
            except (TypeError, ValueError):
                continue
            if number_i <= 0:
                continue
            labels_raw = entry.get("labels") or []
            if isinstance(labels_raw, list):
                labels = tuple(
                    (l.get("name") if isinstance(l, dict) else str(l))
                    for l in labels_raw if l
                )
            else:
                labels = ()
            url = str(
                entry.get("html_url")
                or entry.get("url")
                or f"https://github.com/{repo}/pull/{number_i}"
            )
            body = str(entry.get("body") or "")
            body_snippet = (body[:280] + "…") if len(body) > 280 else body
            prs.append(PRSummary(
                repo=repo,
                number=number_i,
                title=str(entry.get("title") or "").strip(),
                url=url,
                state=str(entry.get("state") or "").strip(),
                labels=tuple(str(l).strip() for l in labels if l),
                author=str(
                    (entry.get("author") or {}).get("login")
                    if isinstance(entry.get("author"), dict)
                    else (entry.get("author") or entry.get("user") or "")
                ).strip(),
                merged_at=str(entry.get("merged_at") or "").strip(),
                updated_at=str(entry.get("updated_at") or "").strip(),
                body_snippet=body_snippet,
            ))
        return prs


def _matches_keywords(pr: PRSummary, keywords_lower: list[str]) -> bool:
    """Return True iff ``pr`` (title + labels + body snippet) mentions any keyword.

    Args:
        pr: The PR summary to test.
        keywords_lower: Lowercased keywords to match (empty matches all).

    Returns:
        ``True`` when any keyword appears in the PR's title, labels, or body
        snippet (or when no keywords are given).
    """
    if not keywords_lower:
        return True
    haystack = " ".join([
        pr.title.lower(),
        " ".join(l.lower() for l in pr.labels),
        pr.body_snippet.lower(),
    ])
    return any(kw in haystack for kw in keywords_lower)


__all__ = [
    "DEFAULT_PR_FEED_PER_REPO_LIMIT",
    "DEFAULT_PR_FEED_TOTAL_BUDGET_SEC",
    "DEFAULT_PR_FEED_WINDOW_DAYS",
    "DEFAULT_PR_MONITOR_MCP_URL",
    "DEFAULT_PR_MONITOR_TIMEOUT_SEC",
    "DEFAULT_PR_MONITOR_URL",
    "PRMonitorClient",
    "PRMonitorError",
    "PRSummary",
]
