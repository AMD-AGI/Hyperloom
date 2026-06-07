# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR Monitor REST client — v0.8 M4.

Stdlib-only (``urllib``) client for the
``primus-cortex-pr-api`` REST surface (see
``primus-cortex-pr-monitor-access.md``). Used by the
:class:`KnowledgePlane` facade to warm specialist prompts with recent PR
summaries from the repos a given domain cares about.

Design priorities:

1. **Stdlib-only** so the client works in stripped sandboxes that
   don't have ``requests`` or ``httpx``. ``urllib`` is enough for the
   small GET surface this milestone needs.
2. **Fail-soft**: any HTTP / network / parse failure surfaces as an
   empty result + a warning entry, never an exception in the
   prompt-assembly path. The whole point of M4 is "PR feed is a
   bonus, not a critical path"; Inv-6.3 covers this.
3. **No mutation**: PR Monitor itself is read-only; this client
   reflects that — no POST / PUT / DELETE methods are exposed.
4. **Cross-cluster aware**: production deploys put the
   ``primus-cortex-pr-api`` service in a different cluster from the
   optimizer pod. The default URL hits the in-cluster DNS name; the
   ``--pr-monitor-url`` CLI flag overrides for ad-hoc port-forwarded
   debug. KB_design §3.14 R-02 acknowledges the unreachable case is
   the dominant failure mode.

The PR Monitor's full tool surface is documented in
``primus-cortex-pr-monitor-access.md``; this client only ships the
endpoints M4 needs (list-repos, list-PRs). The MCP surface for
specialists is configured separately (a URL + tool whitelist passed to
the LLM backend, not Python code).
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


# Default service URL — primus-cortex-pr-api in the primus-cortex
# namespace (see primus-cortex-pr-monitor-access.md §"服务地址"). When
# the optimizer pod runs outside the primus-cortex cluster, the operator
# must override via ``--pr-monitor-url`` / env var with a port-forward.
DEFAULT_PR_MONITOR_URL: str = (
    "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1"
)
# MCP URL (passed to specialist LLM backend; the runner doesn't need
# a Python client). Note the trailing slash is mandatory per the
# upstream doc.
DEFAULT_PR_MONITOR_MCP_URL: str = (
    "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/mcp/"
)

# Default look-back window for ``pr_feed_warm`` (recent PRs, 30 days).
DEFAULT_PR_FEED_WINDOW_DAYS: int = 30

# Per-repo request limit (REST max is 200 per spec; we ask for less so
# the prompt block stays compact).
DEFAULT_PR_FEED_PER_REPO_LIMIT: int = 25

# How long to wait on a single HTTP call before giving up. PR feed
# warming is on the critical path for specialist dispatch — we'd
# rather time out fast and surface a warning than hold the
# Orchestration tick.
DEFAULT_PR_MONITOR_TIMEOUT_SEC: float = 5.0


# Total wall-clock budget for one ``pr_feed_warm`` invocation. With
# 5 repos × 5 s timeout per request, the worst case is 25 s; cap at
# 15 s and let unreachable repos fall to the warning list rather than
# blocking specialist dispatch indefinitely.
DEFAULT_PR_FEED_TOTAL_BUDGET_SEC: float = 15.0


@dataclass
class PRSummary:
    """One PR row returned by ``pr_feed_warm`` / ``list_prs``.

    Mirrors the subset of the REST PR list response we put in
    specialist prompts. Extra fields the upstream returns
    (``head_sha``, ``base_sha``, …) are dropped — specialists can dig
    them up via the MCP if they need.
    """

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


# ---------------------------------------------------------------------------
class PRMonitorError(RuntimeError):
    """Raised for unrecoverable PR Monitor interactions.

    Most callers catch this and treat it as "PR feed unavailable" —
    the prompt builder renders a ``(unavailable: pr_monitor)`` line and
    specialists continue without PR context (KB_design §3.5 §6
    section 6 / §3.14 R-02).
    """


# ---------------------------------------------------------------------------
@dataclass
class PRMonitorClient:
    """Stdlib-only REST client for the PR Monitor surface.

    Instantiated once by the CLI / Coordinator. ``enabled=False``
    (``--degraded-pr``) turns every call into a no-op returning
    empty data — KB_design §3.13 M4 §9 verification 5.
    """

    base_url: str = DEFAULT_PR_MONITOR_URL
    enabled: bool = True
    timeout_sec: float = DEFAULT_PR_MONITOR_TIMEOUT_SEC
    user_agent: str = "inference-optimizer/v0.8 (PRMonitorClient)"
    # In-memory cache of (repo, params_hash) → list[PRSummary] keyed by
    # the full URL. Specialist warmups in the same Orchestration tick
    # tend to hit the same repos; we cache for one tick lifetime via
    # ``reset_cache``.
    _cache: dict[str, list[PRSummary]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_args(
        cls,
        *,
        url: str | None = None,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ) -> "PRMonitorClient":
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

    # ------------------------------------------------------------------
    # Low-level HTTP helper
    # ------------------------------------------------------------------
    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``base_url + path`` with ``params`` query string. Returns
        parsed JSON or raises :class:`PRMonitorError`.

        Defense-in-depth: limits response size to 4 MiB so a misbehaving
        endpoint can't hang the optimizer pod by streaming gigabytes.
        """
        if not self.enabled:
            raise PRMonitorError("PR Monitor client disabled (--degraded-pr)")
        query = ""
        if params:
            # Drop None / empty values so we don't send ``?state=&label=``.
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
                # Cap at 4 MiB to defend against runaway responses; PR
                # Monitor's documented list endpoints are bounded.
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

    # ------------------------------------------------------------------
    # REST endpoint wrappers
    # ------------------------------------------------------------------
    def healthz(self) -> bool:
        """Return True if the PR Monitor health endpoint responds 2xx."""
        try:
            self._get_json("/healthz")
            return True
        except PRMonitorError as exc:
            log.info("pr_monitor.healthz failed: %s", exc)
            return False

    def list_repos(self) -> list[str]:
        """Return the list of active repo names PR Monitor knows about.

        The REST ``/repos`` endpoint returns a top-level JSON array
        whose entries carry ``repo_name`` (canonical ``owner/name``)
        and ``is_active``; older / test fixtures may wrap the array in
        ``{"items": [...]}`` and use the legacy ``name`` field, so we
        accept either shape. Entries with ``is_active=False`` are
        skipped because PR Monitor stops polling them and they return
        empty PR lists.
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

        Mirrors REST ``/repos/{repo}/prs``. Returns a list of
        :class:`PRSummary` (empty on failure). Uses the in-memory cache
        keyed by the rendered URL so multiple specialist dispatches
        within the same tick don't re-hit the network.
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
        """Return the full PR detail dict, or None on failure."""
        try:
            return self._get_json(f"/repos/{repo}/prs/{int(number)}")
        except PRMonitorError as exc:
            log.info("pr_monitor.get_pr(%s#%s) failed: %s", repo, number, exc)
            return None

    # ------------------------------------------------------------------
    # High-level helper used by KnowledgePlane.pr_feed_warm
    # ------------------------------------------------------------------
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

        Always returns 2-tuples; warnings list is populated when a
        repo couldn't be fetched (unreachable / 4xx). The caller is
        expected to forward both into the specialist prompt builder
        (``pr_feed`` section + ``warnings`` log).

        Filtering rules:

        - ``state=all`` (open + merged + closed merged together).
        - ``since = now - window_days`` (ISO ``YYYY-MM-DDTHH:MM:SSZ``).
        - When ``keywords`` is non-empty, drop PRs whose title + label
          set contain none of the keywords (case-insensitive).
        - Sort the merged list by ``updated_at`` desc (newest first).
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
                # Surface per-repo failures so the caller can render a
                # "pr_monitor_unreachable:<repo>" line in the
                # breakdown.warnings + specialist prompt.
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

        :meth:`pr_feed_warm` uses this so it can distinguish ``[]`` =
        "successful but no PRs in window" from ``[]`` = "fetch failed".
        External callers should stick with the fail-soft
        :meth:`list_prs`.
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


# ---------------------------------------------------------------------------
def _matches_keywords(pr: PRSummary, keywords_lower: list[str]) -> bool:
    """Return True iff ``pr`` mentions any of ``keywords_lower``.

    The PR text we check is the concatenation of title + labels +
    short body snippet (which is what specialists actually see in the
    prompt). Empty keyword list → match everything (caller should
    not call this in that case).
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
