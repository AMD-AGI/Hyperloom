"""Primus Cortex PR Monitor client.

Internal-network replacement for the anonymous GitHub Search path. Talks
to the ``primus-cortex-pr-monitor`` REST service.

Policy:

* Zero external dependencies (stdlib ``urllib.request`` only).
* **Hard-fail on errors** when ``base_url`` is configured (network errors,
  non-200 status, malformed JSON). The CLI surfaces this as exit code 2
  so misconfigured nodes do not silently fall back to an empty candidate
  list.
* Returns :class:`GitHubPr` records (shared with the GitHub backend), so
  call sites can swap data sources without touching downstream
  :class:`Candidate` plumbing.

Ported from zhenggong/framework-agent with one local change: the shared
``GitHubPr`` / ``_repo_slug`` helpers live in ``sources/_shared.py``
instead of a sibling ``github_search`` module, so this file is usable
before the github backend lands in PR-B.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ._shared import GitHubPr, _repo_slug


class PrimusCortexError(RuntimeError):
    """Raised when a primus-cortex request cannot be completed.

    Inherits from :class:`RuntimeError` so the CLI's blanket
    ``except Exception`` translates it into exit code 2 with a clear
    message.
    """


def _normalise_base_url(base_url: str) -> str:
    """Trim trailing slash on the configured base URL.

    Args:
        base_url (str): The configured primus_cortex base URL.

    Returns:
        str: The base URL with any trailing slash removed.

    Raises:
        PrimusCortexError: If ``base_url`` is empty.
    """
    if not base_url:
        raise PrimusCortexError("primus_cortex.base_url is empty")
    return base_url.rstrip("/")


def _build_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    """Compose a full URL, urlencoding the query (skipping empty values).

    Args:
        base_url (str): The primus_cortex base URL.
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
        PrimusCortexError: On HTTP errors, unreachable hosts, timeouts, or other
            transport failures.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, text/plain;q=0.9, */*;q=0.5",
            "User-Agent": "framework-agent-primus-cortex/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            body = resp.read()
            content_type = (
                resp.headers.get("Content-Type", "") if resp.headers else ""
            )
            return status, body, content_type
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:512]
        except Exception:  # noqa: BLE001 - read can raise OSError on closed body
            err_body = ""
        raise PrimusCortexError(
            f"primus_cortex HTTP {exc.code} at {url}: {err_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PrimusCortexError(
            f"primus_cortex unreachable at {url}: {exc.reason}"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise PrimusCortexError(
            f"primus_cortex transport error at {url}: {exc}"
        ) from exc


def _http_get_json(url: str, *, timeout_sec: float) -> Any:
    """GET and parse JSON body; raise PrimusCortexError on >=400 or bad JSON.

    Args:
        url (str): Fully composed URL to GET.
        timeout_sec (float): Per-request timeout in seconds.

    Returns:
        Any: The parsed JSON payload.

    Raises:
        PrimusCortexError: On a >=400 status or a non-JSON body.
    """
    status, body, _ = _http_get(url, timeout_sec=timeout_sec)
    if status >= 400:
        raise PrimusCortexError(f"primus_cortex HTTP {status} at {url}")
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PrimusCortexError(
            f"primus_cortex returned non-JSON at {url}: {exc}; body[:200]={text[:200]!r}"
        ) from exc


def _http_get_text(url: str, *, timeout_sec: float) -> str:
    """GET and return raw text body; raise PrimusCortexError on >=400.

    Args:
        url (str): Fully composed URL to GET.
        timeout_sec (float): Per-request timeout in seconds.

    Returns:
        str: The decoded response body.

    Raises:
        PrimusCortexError: On a >=400 status.
    """
    status, body, _ = _http_get(url, timeout_sec=timeout_sec)
    if status >= 400:
        raise PrimusCortexError(f"primus_cortex HTTP {status} at {url}")
    return body.decode("utf-8", errors="replace")


def _coerce_pr_item(item: Any, *, source_url: str) -> GitHubPr:
    """Coerce a primus-cortex PR list item into the shared GitHubPr record.

    Args:
        item (Any): A single PR list item, expected to be a JSON object.
        source_url (str): URL the item came from, used in error messages.

    Returns:
        GitHubPr: The normalised PR record.

    Raises:
        PrimusCortexError: If ``item`` is not an object or lacks an int
            ``number``.
    """
    if not isinstance(item, dict):
        raise PrimusCortexError(
            f"primus_cortex item at {source_url} is not a JSON object: {type(item).__name__}"
        )
    number = item.get("number")
    if not isinstance(number, int):
        raise PrimusCortexError(
            f"primus_cortex item at {source_url} has non-int 'number': {number!r}"
        )
    return GitHubPr(
        number=number,
        title=str(item.get("title") or ""),
        html_url=str(item.get("html_url") or item.get("url") or ""),
    )


def _extract_pr_list(payload: Any, *, source_url: str) -> list[dict[str, Any]]:
    """Normalise a primus-cortex PR list response into ``list[dict]``.

    Args:
        payload (Any): The decoded response; a list, or a dict carrying a list
            under ``items``/``prs``/``data``/``results``.
        source_url (str): URL the payload came from, used in error messages.

    Returns:
        list[dict[str, Any]]: The extracted PR objects (non-dict entries
            dropped).

    Raises:
        PrimusCortexError: If the payload is neither a list nor a dict with a
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
            raise PrimusCortexError(
                f"primus_cortex response at {source_url} is a dict but has no list "
                f"field (tried items/prs/data/results); keys={list(payload.keys())!r}"
            )
    else:
        raise PrimusCortexError(
            f"primus_cortex response at {source_url} is not list or dict: "
            f"{type(payload).__name__}"
        )
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
    """List PRs from primus-cortex; hard-fails on transport/parse errors.

    The result type :class:`GitHubPr` is identical to what the public
    GitHub Search backend returns, so the dispatcher can union both
    sources without per-source branching.

    Args:
        repo_url (str): Git URL of the repo; parsed to an ``owner/name`` slug.
        base_url (str): primus_cortex service base URL.
        limit (int): Maximum number of PRs to return. Defaults to 5.
        state (str): PR state filter. Defaults to ``"open"``.
        label (str | None): Optional label filter.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        list[GitHubPr]: The matching PRs (at most ``limit``).

    Raises:
        PrimusCortexError: On an unparseable repo URL or any transport/parse
            error.
    """
    try:
        repo_slug = _repo_slug(repo_url)
    except ValueError as exc:
        raise PrimusCortexError(
            f"cannot derive repo slug from repo_url={repo_url!r}: {exc}"
        ) from exc

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
        base_url (str): primus_cortex service base URL.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        dict[str, Any]: The PR detail object.

    Raises:
        PrimusCortexError: On transport/parse errors or a non-object response.
    """
    url = _build_url(base_url, f"/v1/repos/{repo_slug}/prs/{number}")
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    if not isinstance(payload, dict):
        raise PrimusCortexError(
            f"primus_cortex pr_get at {url} did not return an object: {type(payload).__name__}"
        )
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
        base_url (str): primus_cortex service base URL.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        list[dict[str, Any]]: The changed-file objects.

    Raises:
        PrimusCortexError: On transport/parse errors or an unexpected response
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
            raise PrimusCortexError(
                f"primus_cortex pr_files at {url} returned dict without list field "
                f"(tried files/items/data); keys={list(payload.keys())!r}"
            )
    else:
        raise PrimusCortexError(
            f"primus_cortex pr_files at {url} returned non-list/dict: {type(payload).__name__}"
        )
    return [item for item in items if isinstance(item, dict)]


def pr_patches(
    repo_slug: str,
    number: int,
    *,
    base_url: str,
    timeout_sec: float = 30.0,
) -> str:
    """GET ``/v1/repos/{repo}/prs/{number}/patches`` and render as unified diff.

    The primus-cortex service returns a JSON array
    ``[{"file": {...}, "patch": "@@ ...", "patch_truncated": bool}]``
    instead of raw unified-diff text. This helper synthesises ``diff --git`` /
    ``--- a/<path>`` / ``+++ b/<path>`` headers per file so the result is a
    valid unified patch that ``git apply`` can consume. Renamed and deleted
    files honour ``previous_path`` and ``status``. Items missing the
    ``patch`` field (e.g. binary diffs) emit only the file header and skip
    the hunk body.

    Args:
        repo_slug (str): Repository slug in ``owner/name`` form.
        number (int): PR number to fetch patches for.
        base_url (str): primus_cortex service base URL.
        timeout_sec (float): Per-request timeout. Defaults to 30.0.

    Returns:
        str: A synthesised unified diff (empty string when no patch content is
            present); a raw string payload is returned verbatim.

    Raises:
        PrimusCortexError: On transport/parse errors or an unexpected response
            shape.
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
            raise PrimusCortexError(
                f"primus_cortex pr_patches at {url} returned dict without list field "
                f"(tried patches/items/data); keys={list(payload.keys())!r}"
            )
    elif isinstance(payload, str):
        return payload
    else:
        raise PrimusCortexError(
            f"primus_cortex pr_patches at {url} returned non-list/dict/str: {type(payload).__name__}"
        )

    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        file_meta = item.get("file") if isinstance(item.get("file"), dict) else item
        path = (
            file_meta.get("file_path")
            or file_meta.get("filename")
            or file_meta.get("path")
            or ""
        )
        if not isinstance(path, str) or not path:
            continue
        previous_path = file_meta.get("previous_path") or path
        status = (file_meta.get("status") or "").lower()
        old_path = (
            previous_path
            if isinstance(previous_path, str) and previous_path
            else path
        )
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


def search_perf_prs_via_primus_search(
    repo_url: str,
    *,
    base_url: str,
    query: str,
    limit: int = 5,
    state: str = "all",
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Free-text search via ``/v1/search/prs``; alternate to ``list_perf_prs``.

    Args:
        repo_url (str): Git URL of the repo; parsed to an ``owner/name`` slug.
        base_url (str): primus_cortex service base URL.
        query (str): Free-text search query.
        limit (int): Maximum number of PRs to return. Defaults to 5.
        state (str): PR state filter. Defaults to ``"all"``.
        timeout_sec (float): Per-request timeout. Defaults to 10.0.

    Returns:
        list[GitHubPr]: The matching PRs (at most ``limit``).

    Raises:
        PrimusCortexError: On an unparseable repo URL or any transport/parse
            error.
    """
    try:
        repo_slug = _repo_slug(repo_url)
    except ValueError as exc:
        raise PrimusCortexError(
            f"cannot derive repo slug from repo_url={repo_url!r}: {exc}"
        ) from exc

    url = _build_url(
        base_url,
        "/v1/search/prs",
        {"q": query, "repo": repo_slug, "state": state, "limit": limit},
    )
    payload = _http_get_json(url, timeout_sec=timeout_sec)
    items = _extract_pr_list(payload, source_url=url)
    return [_coerce_pr_item(item, source_url=url) for item in items[:limit]]


__all__ = [
    "PrimusCortexError",
    "list_perf_prs",
    "pr_files",
    "pr_get",
    "pr_patches",
    "search_perf_prs_via_primus_search",
]
