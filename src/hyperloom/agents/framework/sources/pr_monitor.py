# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR Monitor candidate source client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hyperloom.common.github_urls import repo_slug
from hyperloom.common.url_safety import require_http_url as _base_require_http_url

from ._shared import GitHubPr


class PRMonitorError(RuntimeError):
    """Raised when a pr_monitor request cannot be completed (CLI exit code 2)."""


def _require_http_url(url: str) -> None:
    _base_require_http_url(url, error=PRMonitorError, context="PR Monitor URL")


def _normalise_base_url(base_url: str) -> str:
    """Trim trailing slash and optional API-version suffix on the base URL."""
    if not base_url:
        raise PRMonitorError("pr_monitor.base_url is empty")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _build_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    """Compose a full URL, urlencoding the query (skipping empty values)."""
    base = _normalise_base_url(base_url)
    if not path.startswith("/"):
        path = "/" + path
    qs = ""
    if query:
        items: list[tuple[str, str]] = []
        for key, value in query.items():
            if value is None or value == "":
                continue
            items.append((str(key), str(value)))
        if items:
            qs = "?" + urllib.parse.urlencode(items)
    return base + path + qs


def _http_get(url: str, *, timeout_sec: float) -> tuple[int, bytes, str]:
    """Return ``(status, body_bytes, content_type)``; raise on transport errors."""
    _require_http_url(url)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, text/plain;q=0.9, */*;q=0.5",
            "User-Agent": "framework-agent-pr-monitor/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - URL scheme checked above.
            status = int(getattr(resp, "status", 200) or 200)
            body = resp.read()
            content_type = resp.headers.get("Content-Type", "") if resp.headers else ""
            return status, body, content_type
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:512]
        except Exception:  # noqa: BLE001 - read can raise OSError on closed body
            err_body = ""
        raise PRMonitorError(f"pr_monitor HTTP {exc.code} at {url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise PRMonitorError(f"pr_monitor unreachable at {url}: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise PRMonitorError(f"pr_monitor transport error at {url}: {exc}") from exc


def _http_get_json(url: str, *, timeout_sec: float) -> Any:
    """GET and parse JSON body; raise PRMonitorError on >=400 or bad JSON."""
    status, body, _ = _http_get(url, timeout_sec=timeout_sec)
    if status >= 400:
        raise PRMonitorError(f"pr_monitor HTTP {status} at {url}")
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PRMonitorError(f"pr_monitor returned non-JSON at {url}: {exc}; body[:200]={text[:200]!r}") from exc


def _coerce_pr_item(item: Any, *, source_url: str) -> GitHubPr:
    """Coerce a pr_monitor list item into the shared GitHubPr record."""
    if not isinstance(item, dict):
        raise PRMonitorError(f"pr_monitor item at {source_url} is not a JSON object: {type(item).__name__}")
    # ``/v1/search/prs`` returns match records shaped as ``{"summary": {...pr fields...}, "matched_field": ...,
    # "snippet": ...}``.
    summary = item.get("summary")
    if isinstance(summary, dict):
        item = summary
    number = item.get("number")
    if not isinstance(number, int):
        raise PRMonitorError(f"pr_monitor item at {source_url} has non-int 'number': {number!r}")
    html_url = str(item.get("html_url") or item.get("url") or "")
    repo_name = str(item.get("repo_name") or item.get("repository") or "").strip()
    if not html_url and repo_name:
        html_url = f"https://github.com/{repo_name}/pull/{number}"
    return GitHubPr(
        number=number,
        title=str(item.get("title") or ""),
        html_url=html_url,
    )


def _extract_pr_list(payload: Any, *, source_url: str) -> list[dict[str, Any]]:
    """Normalise a pr_monitor list response into ``list[dict]``."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("items", "prs", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            raise PRMonitorError(
                f"pr_monitor response at {source_url} is a dict but has no list "
                f"field (tried items/prs/data/results); keys={list(payload.keys())!r}"
            )
    else:
        raise PRMonitorError(f"pr_monitor response at {source_url} is not list or dict: {type(payload).__name__}")
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def list_perf_prs(
    repo_url: str,
    *,
    base_url: str,
    limit: int = 5,
    state: str = "open",
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """List PRs from pr_monitor."""
    try:
        slug = repo_slug(repo_url)
    except ValueError as exc:
        raise PRMonitorError(f"cannot derive repo slug from repo_url={repo_url!r}: {exc}") from exc

    query: dict[str, Any] = {"state": state, "limit": limit}
    url = _build_url(base_url, f"/v1/repos/{slug}/prs", query)
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    items = _extract_pr_list(payload, source_url=url)
    out: list[GitHubPr] = []
    for item in items[:limit]:
        out.append(_coerce_pr_item(item, source_url=url))
    return out


def search_perf_prs_via_pr_monitor_search(
    repo_url: str,
    *,
    base_url: str,
    query: str,
    limit: int = 5,
    state: str = "open",
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Free-text search via ``/v1/search/prs``; alternate to ``list_perf_prs``."""
    try:
        slug = repo_slug(repo_url)
    except ValueError as exc:
        raise PRMonitorError(f"cannot derive repo slug from repo_url={repo_url!r}: {exc}") from exc

    url = _build_url(
        base_url,
        "/v1/search/prs",
        {"q": query, "repo": slug, "state": state, "limit": limit},
    )
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    items = _extract_pr_list(payload, source_url=url)
    return [_coerce_pr_item(item, source_url=url) for item in items[:limit]]


__all__ = [
    "PRMonitorError",
    "list_perf_prs",
    "search_perf_prs_via_pr_monitor_search",
]
