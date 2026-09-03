# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR Monitor candidate source client.

Optional replacement for anonymous GitHub Search; talks to the
``pr_monitor`` REST service. Stdlib-only (``urllib.request``). Hard-fails
on errors (network / non-200 / bad JSON) so misconfigured nodes don't silently
fall back to an empty list (CLI surfaces exit code 2). Returns
:class:`GitHubPr` records shared with the GitHub backend.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hyperloom.common.url_safety import require_http_url as _base_require_http_url

from ._shared import GitHubPr, _repo_slug


class PRMonitorError(RuntimeError):
    """Raised when a pr_monitor request cannot be completed (CLI exit code 2)."""


def _require_http_url(url: str) -> None:
    _base_require_http_url(url, error=PRMonitorError, context="PR Monitor URL")


def _normalise_base_url(base_url: str) -> str:
    """Trim trailing slash and optional API-version suffix on the base URL.

    Args:
        base_url (str): The configured pr_monitor base URL.

    Returns:
        str: The service root URL with any trailing slash or trailing ``/v1`` removed.

    Raises:
        PRMonitorError: If ``base_url`` is empty.
    """
    if not base_url:
        raise PRMonitorError("pr_monitor.base_url is empty")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _build_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    """Compose a full URL, urlencoding the query (skipping empty values).

    Args:
        base_url (str): The pr_monitor base URL.
        path (str): Request path; a leading slash is added when missing.
        query (dict[str, Any] | None): Query parameters; ``None``/empty values
            are skipped.

    Returns:
        str: The fully composed URL with an encoded query string.
    """
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
    """Return ``(status, body_bytes, content_type)``; raise on transport errors.

    Args:
        url (str): Fully composed URL to GET.
        timeout_sec (float): Per-request timeout in seconds.

    Returns:
        tuple[int, bytes, str]: The HTTP status code, raw body bytes, and the
            ``Content-Type`` header value.

    Raises:
        PRMonitorError: On HTTP errors, unreachable hosts, timeouts, or other
            transport failures.
    """
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
    """GET and parse JSON body; raise PRMonitorError on >=400 or bad JSON.

    Args:
        url (str): Fully composed URL to GET.
        timeout_sec (float): Per-request timeout in seconds.

    Returns:
        Any: The parsed JSON payload.

    Raises:
        PRMonitorError: On a >=400 status or a non-JSON body.
    """
    status, body, _ = _http_get(url, timeout_sec=timeout_sec)
    if status >= 400:
        raise PRMonitorError(f"pr_monitor HTTP {status} at {url}")
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PRMonitorError(f"pr_monitor returned non-JSON at {url}: {exc}; body[:200]={text[:200]!r}") from exc


def _coerce_pr_item(item: Any, *, source_url: str) -> GitHubPr:
    """Coerce a pr_monitor list item into the shared GitHubPr record.

    Args:
        item (Any): A single PR list item, expected to be a JSON object.
        source_url (str): URL the item came from, used in error messages.

    Returns:
        GitHubPr: The normalised PR record.

    Raises:
        PRMonitorError: If ``item`` is not an object or lacks an int
            ``number``.
    """
    if not isinstance(item, dict):
        raise PRMonitorError(f"pr_monitor item at {source_url} is not a JSON object: {type(item).__name__}")
    # ``/v1/search/prs`` returns match records shaped as
    # ``{"summary": {...pr fields...}, "matched_field": ..., "snippet": ...}``.
    # Normalise them to the same payload shape as the list endpoint.
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
    """Normalise a pr_monitor list response into ``list[dict]``.

    Args:
        payload (Any): The decoded response; a list, or a dict carrying a list
            under ``items``/``prs``/``data``/``results``.
        source_url (str): URL the payload came from, used in error messages.

    Returns:
        list[dict[str, Any]]: The extracted PR objects (non-dict entries
            dropped).

    Raises:
        PRMonitorError: If the payload is neither a list nor a dict with a
            recognised list field.
    """
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
    label: str | None = None,
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """List PRs from pr_monitor.

    Returns :class:`GitHubPr` (same as the GitHub backend) so the dispatcher
    can union both sources without per-source branching.

    Args:
        repo_url: Repository URL to list PRs for.
        base_url: PR Monitor base URL.
        limit: Maximum number of PRs to return.
        state: PR state filter (e.g. ``"open"``).
        label: Optional label filter.
        timeout_sec: Per-request timeout.

    Returns:
        A list of :class:`GitHubPr` records.

    Raises:
        PRMonitorError: On bad repo URL or transport/parse errors.
    """
    try:
        repo_slug = _repo_slug(repo_url)
    except ValueError as exc:
        raise PRMonitorError(f"cannot derive repo slug from repo_url={repo_url!r}: {exc}") from exc

    query: dict[str, Any] = {"state": state, "limit": limit}
    if label:
        query["label"] = label
    url = _build_url(base_url, f"/v1/repos/{repo_slug}/prs", query)
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    items = _extract_pr_list(payload, source_url=url)
    out: list[GitHubPr] = []
    for item in items[:limit]:
        out.append(_coerce_pr_item(item, source_url=url))
    return out


def pr_get(
    repo_slug: str,
    number: int,
    *,
    base_url: str,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """GET ``/v1/repos/{repo}/prs/{number}`` returning the PR detail object.

    Args:
        repo_slug (str): Repository slug in ``owner/name`` form.
        number (int): PR number to fetch.
        base_url (str): pr_monitor service base URL.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        dict[str, Any]: The PR detail object.

    Raises:
        PRMonitorError: On transport/parse errors or a non-object response.
    """
    url = _build_url(base_url, f"/v1/repos/{repo_slug}/prs/{number}")
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    if not isinstance(payload, dict):
        raise PRMonitorError(f"pr_monitor pr_get at {url} did not return an object: {type(payload).__name__}")
    return payload


def pr_files(
    repo_slug: str,
    number: int,
    *,
    base_url: str,
    timeout_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """GET ``/v1/repos/{repo}/prs/{number}/files`` returning the file list.

    Args:
        repo_slug (str): Repository slug in ``owner/name`` form.
        number (int): PR number to fetch files for.
        base_url (str): pr_monitor service base URL.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        list[dict[str, Any]]: The changed-file objects.

    Raises:
        PRMonitorError: On transport/parse errors or an unexpected response
            shape.
    """
    url = _build_url(base_url, f"/v1/repos/{repo_slug}/prs/{number}/files")
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("files", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            raise PRMonitorError(
                f"pr_monitor pr_files at {url} returned dict without list field "
                f"(tried files/items/data); keys={list(payload.keys())!r}"
            )
    else:
        raise PRMonitorError(f"pr_monitor pr_files at {url} returned non-list/dict: {type(payload).__name__}")
    return [item for item in items if isinstance(item, dict)]


def pr_patches(
    repo_slug: str,
    number: int,
    *,
    base_url: str,
    timeout_sec: float = 30.0,
) -> str:
    """GET ``/v1/repos/{repo}/prs/{number}/patches`` and render as unified diff.

    The service returns a JSON patch array, not raw diff text; this synthesises
    ``diff --git`` / ``--- a/`` / ``+++ b/`` headers per file so ``git apply``
    can consume it. Honours ``previous_path`` / ``status`` for renames+deletes;
    items missing ``patch`` (binary) emit only the file header.

    Args:
        repo_slug: ``owner/name`` repository slug.
        number: PR number to fetch patches for.
        base_url: PR Monitor base URL.
        timeout_sec: Per-request timeout.

    Returns:
        A unified-diff string suitable for ``git apply``.

    Raises:
        PRMonitorError: On unexpected payload shapes or transport errors.
    """
    url = _build_url(base_url, f"/v1/repos/{repo_slug}/prs/{number}/patches")
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("patches", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            raise PRMonitorError(
                f"pr_monitor pr_patches at {url} returned dict without list field "
                f"(tried patches/items/data); keys={list(payload.keys())!r}"
            )
    elif isinstance(payload, str):
        return payload
    else:
        raise PRMonitorError(f"pr_monitor pr_patches at {url} returned non-list/dict/str: {type(payload).__name__}")

    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        file_meta = item.get("file") if isinstance(item.get("file"), dict) else item
        path = file_meta.get("file_path") or file_meta.get("filename") or file_meta.get("path") or ""
        if not isinstance(path, str) or not path:
            continue
        previous_path = file_meta.get("previous_path") or path
        status = (file_meta.get("status") or "").lower()
        old_path = previous_path if isinstance(previous_path, str) and previous_path else path
        if status == "added":
            old_label = "/dev/null"
        else:
            old_label = f"a/{old_path}"
        if status == "deleted" or status == "removed":
            new_label = "/dev/null"
        else:
            new_label = f"b/{path}"
        chunks.append(f"diff --git a/{old_path} b/{path}")
        chunks.append(f"--- {old_label}")
        chunks.append(f"+++ {new_label}")
        patch_body = item.get("patch")
        if isinstance(patch_body, str) and patch_body:
            chunks.append(patch_body.rstrip("\n"))
    if not chunks:
        return ""
    return "\n".join(chunks) + "\n"


def search_perf_prs_via_pr_monitor_search(
    repo_url: str,
    *,
    base_url: str,
    query: str,
    limit: int = 5,
    state: str = "open",
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Free-text search via ``/v1/search/prs``; alternate to ``list_perf_prs``.

    Args:
        repo_url (str): Git URL of the repo; parsed to an ``owner/name`` slug.
        base_url (str): pr_monitor service base URL.
        query (str): Free-text search query.
        limit (int): Maximum number of PRs to return. Defaults to 5.
        state (str): PR state filter. Defaults to ``"open"``.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        list[GitHubPr]: The matching PRs (at most ``limit``).

    Raises:
        PRMonitorError: On an unparseable repo URL or any transport/parse
            error.
    """
    try:
        repo_slug = _repo_slug(repo_url)
    except ValueError as exc:
        raise PRMonitorError(f"cannot derive repo slug from repo_url={repo_url!r}: {exc}") from exc

    url = _build_url(
        base_url,
        "/v1/search/prs",
        {"q": query, "repo": repo_slug, "state": state, "limit": limit},
    )
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    items = _extract_pr_list(payload, source_url=url)
    return [_coerce_pr_item(item, source_url=url) for item in items[:limit]]


__all__ = [
    "PRMonitorError",
    "list_perf_prs",
    "pr_files",
    "pr_get",
    "pr_patches",
    "search_perf_prs_via_pr_monitor_search",
]
