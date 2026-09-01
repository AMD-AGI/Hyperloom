"""REST client for PR Monitor.

404 means absence; contract and transport failures remain distinct. Pagination
is disabled because the service cursor skips rows sharing its timestamp.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from hyperloom.common.pr_monitor_urls import pr_monitor_base_url

log = logging.getLogger(__name__)

# Self-imposed ceiling: no query may ask for more than one bounded first page.
BOUNDED_PAGE_LIMIT = 50

_MAX_WORKERS = 8


class PRMonitorError(Exception):
    """Base error for PR Monitor transport failures."""


class PRTransportError(PRMonitorError):
    """Network, timeout, or unexpected server-side failure; retryable."""


class PRContractError(PRMonitorError):
    """400/422/non-JSON: the server contract changed. Must be alerted on."""


@dataclass(frozen=True)
class FetchOutcome:
    """One completed request in a concurrent batch."""

    path: str
    payload: Any | None = None
    error: Exception | None = None


def normalize_base_url(raw: str = "") -> str:
    """Return the service root without a trailing ``/v1``."""
    base = (raw or pr_monitor_base_url()).strip()
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return base


def clamp_limit(limit: int) -> int:
    """Clamp a caller's limit to the bounded first page this client allows."""
    return max(1, min(int(limit), BOUNDED_PAGE_LIMIT))


def extract_items(payload: Any) -> list[dict]:
    """Read rows from a bare array or the service's ``items`` envelope."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    else:
        raise PRContractError("expected an array or an object with an items array")
    if not all(isinstance(item, dict) for item in items):
        raise PRContractError("expected every response item to be an object")
    return items


class PRMonitorClient:
    """Stdlib REST client for the PR Monitor service (no auth required)."""

    def __init__(
        self,
        base_url: str = "",
        *,
        timeout_sec: float = 0.0,
        budget_sec: float = 0.0,
    ) -> None:
        """Configure endpoint and budgets, using ``PR_KB_*`` env defaults."""
        self._base = normalize_base_url(base_url)
        self._timeout = timeout_sec or float(os.environ.get("PR_KB_TIMEOUT_SEC", "10") or 10)
        self._budget = budget_sec or float(os.environ.get("PR_KB_BUDGET_SEC", "30") or 30)

    @property
    def base_url(self) -> str:
        """Service root, guaranteed free of a trailing ``/v1``."""
        return self._base

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        """Build one absolute ``/v1`` URL, dropping parameters left as None."""
        if not self._base:
            raise PRMonitorError("KB_STORE_URL is required when PR Monitor knowledge is enabled")
        url = f"{self._base}/v1{path}"
        if params:
            query = {k: v for k, v in params.items() if v is not None}
            if query:
                url += "?" + urllib.parse.urlencode(query)
        return url

    def _request_timeout(self, remaining: float | None) -> float:
        """A single request may never outlive the caller's remaining budget."""
        if remaining is None:
            return self._timeout
        return max(0.0, min(self._timeout, remaining))

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_sec: float | None = None,
    ) -> Any | None:
        """GET one endpoint; return None for a normal 404 absence.

        Raises PRContractError on 400/422/non-JSON and PRTransportError on
        timeouts, connection failures, and 5xx.
        """
        if params and "before" in params:
            raise PRMonitorError("pagination is disabled: the server cursor drops same-timestamp rows")
        url = self._url(path, params)
        try:
            with urllib.request.urlopen(url, timeout=self._request_timeout(timeout_sec)) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code in (400, 422):
                raise PRContractError(f"HTTP {error.code} on {path}") from error
            raise PRTransportError(f"HTTP {error.code} on {path}") from error
        except (
            OSError,
            urllib.error.URLError,
            http.client.HTTPException,
        ) as error:
            raise PRTransportError(f"{type(error).__name__} on {path}") from error
        try:
            return json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PRContractError(f"non-JSON body from {path}") from error

    def get_many(
        self,
        requests: list[tuple[str, dict[str, Any] | None]],
        *,
        budget_sec: float | None = None,
    ) -> list[FetchOutcome]:
        """Fetch concurrently within one budget and preserve request order.

        Every request that answered inside the budget is kept: waiting on the
        batch as a whole stops one slow request from discarding the results
        already sitting next to it.
        """
        if not requests:
            return []
        budget = self._budget if budget_sec is None else max(0.0, budget_sec)
        if budget <= 0:
            return [FetchOutcome(path, error=PRTransportError("budget exhausted")) for path, _ in requests]
        deadline = time.monotonic() + budget
        outcomes: list[FetchOutcome] = []
        pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

        def fetch(path: str, params: dict[str, Any] | None) -> Any | None:
            """Start one request only while the shared batch deadline permits."""
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PRTransportError("budget exhausted")
            return self.get(path, params, timeout_sec=remaining)

        try:
            futures = [pool.submit(fetch, path, params) for path, params in requests]
            wait(futures, timeout=max(0.0, deadline - time.monotonic()))
            for (path, _), future in zip(requests, futures):
                if not future.done():
                    future.cancel()
                    outcomes.append(FetchOutcome(path, error=PRTransportError("budget exhausted")))
                    continue
                try:
                    outcomes.append(FetchOutcome(path, payload=future.result()))
                except PRMonitorError as error:
                    outcomes.append(FetchOutcome(path, error=error))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return outcomes

    def healthz(self, *, timeout_sec: float | None = None) -> bool:
        """Return True only when ``/healthz`` returns a payload."""
        try:
            return self.get("/healthz", timeout_sec=timeout_sec) is not None
        except PRMonitorError as error:
            log.warning("pr-monitor preflight failed: %s", error)
            return False

    def list_repos(self, *, timeout_sec: float | None = None) -> list[dict]:
        """List tracked repositories with their polling state."""
        payload = self.get("/repos", timeout_sec=timeout_sec)
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise PRContractError("repository list must be an array of objects")
        return payload

    def list_recent_prs(
        self,
        repo: str,
        *,
        state: str = "merged",
        limit: int = 5,
        timeout_sec: float | None = None,
    ) -> list[dict]:
        """List the most recently updated PRs as a low-precision fallback."""
        payload = self.get(
            f"/repos/{repo}/prs",
            {"state": state, "limit": clamp_limit(limit)},
            timeout_sec=timeout_sec,
        )
        if payload is None:
            return []
        return extract_items(payload)

    def get_pr(self, repo: str, number: int) -> dict | None:
        """Fetch one PR: summary, body, files, commits and distill in one hop."""
        payload = self.get(f"/repos/{repo}/prs/{number}")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise PRContractError("PR detail must be an object")
        return payload

    def get_file_patch(self, repo: str, number: int, file_path: str) -> dict | None:
        """Fetch the diff of one changed file.

        Filters on the PR's current head while ``?file_path=`` reverse lookup
        does not, so after a force-push a path that matched the PR can 404 here.
        """
        payload = self.get(f"/repos/{repo}/prs/{number}/files/by-path", {"path": file_path})
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise PRContractError("file patch must be an object")
        return payload

    def pr_request(self, repo: str, number: int) -> tuple[str, None]:
        """Return a get_many() request tuple for enriching one PR."""
        return (f"/repos/{repo}/prs/{number}", None)
