# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""HTTP client for the InferenceX public benchmarks API.

Endpoint shape verified against
``https://inferencex.semianalysis.com/api/v1/benchmarks?model=<name>``
(see the end-to-end Go test in
``Primus-SaFE/SaFE/apiserver/pkg/handlers/inferencex/handler_e2e_test.go``).

Two design rules driven by the call-site (target_analysis executor):

* **Never raise on network / parsing problems.** Returns ``None`` (or
  an empty list, depending on the call) plus a structured warning the
  caller can persist into ``BaselineSummary.warning``. The orchestration
  loop must keep running even if InferenceX is down.
* **Bounded timeout + small retry budget.** Default total wall-time
  budget is ~8 seconds (2 attempts * 3s connect/read + a touch of
  jitter). target_analysis advertises ``cost_minutes_p50 = 0.1`` and we
  want to honour that even on flaky upstreams.

Optional environment overrides:

* ``INFERENCEX_BASE_URL``     — defaults to upstream public URL; tests
  point this at httptest.
* ``INFERENCEX_TIMEOUT_SEC``  — per-request timeout (default 5).
* ``INFERENCEX_MAX_ATTEMPTS`` — default 2.
* ``INFERENCEX_INSECURE``     — accept ``"1"``/``"true"`` to skip TLS
  verification, mirroring the SaFE handler's ``InsecureSkipVerify``
  (some pod images ship outdated CA bundles).
"""

from __future__ import annotations

import gzip
import logging
import os
import socket
import ssl
import urllib.request
from urllib.error import HTTPError, URLError


log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://inferencex.semianalysis.com/api/v1"
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_MAX_ATTEMPTS = 2


class InferenceXFetchError(Exception):
    """Internal — raised by ``_fetch_raw`` on any fetch failure."""

    pass


def _base_url() -> str:
    """Resolve the API base URL from the environment.

    Returns:
        str: ``INFERENCEX_BASE_URL`` when set and non-empty, otherwise
            :data:`DEFAULT_BASE_URL`.
    """
    return os.environ.get("INFERENCEX_BASE_URL", "").strip() or DEFAULT_BASE_URL


def _timeout_sec() -> float:
    """Resolve the per-request timeout from the environment.

    Returns:
        float: ``INFERENCEX_TIMEOUT_SEC`` clamped to a 0.5s floor, or
            :data:`DEFAULT_TIMEOUT_SEC` when unset or unparseable.
    """
    raw = os.environ.get("INFERENCEX_TIMEOUT_SEC", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SEC
    try:
        return max(0.5, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


def _max_attempts() -> int:
    """Resolve the retry attempt budget from the environment.

    Returns:
        int: ``INFERENCEX_MAX_ATTEMPTS`` clamped to a minimum of 1, or
            :data:`DEFAULT_MAX_ATTEMPTS` when unset or unparseable.
    """
    raw = os.environ.get("INFERENCEX_MAX_ATTEMPTS", "").strip()
    if not raw:
        return DEFAULT_MAX_ATTEMPTS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS


def _insecure() -> bool:
    """Report whether TLS verification should be skipped.

    Returns:
        bool: ``True`` when ``INFERENCEX_INSECURE`` is one of
            ``"1"``/``"true"``/``"yes"``, otherwise ``False``.
    """
    raw = os.environ.get("INFERENCEX_INSECURE", "").strip().lower()
    return raw in ("1", "true", "yes")


def _build_ssl_context() -> ssl.SSLContext | None:
    """Build an SSL context honouring the insecure override.

    Returns:
        ssl.SSLContext | None: ``None`` for normal (verified) TLS, or a
            context with hostname checking and certificate verification
            disabled when :func:`_insecure` is set.
    """
    if not _insecure():
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_raw(url: str) -> bytes:
    """Single HTTP GET with gzip support.

    Args:
        url (str): The fully-formed request URL to fetch.

    Returns:
        bytes: The (gzip-decoded if needed) response body.

    Raises:
        InferenceXFetchError: On any non-200 status, network failure, or
            transport-level decode error.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "src/hyperloom/inference_optimizer/baseline_comparison",
        },
    )
    ctx = _build_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=_timeout_sec(), context=ctx) as resp:
            status = resp.getcode()
            if status != 200:
                raise InferenceXFetchError(f"HTTP {status}")
            body = resp.read()
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                body = gzip.decompress(body)
            return body
    except HTTPError as exc:
        raise InferenceXFetchError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise InferenceXFetchError(f"URL error: {exc.reason}") from exc
    except socket.timeout as exc:
        raise InferenceXFetchError("socket timeout") from exc
    except (OSError, ssl.SSLError) as exc:
        raise InferenceXFetchError(f"transport error: {exc}") from exc


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_MAX_ATTEMPTS",
    "InferenceXFetchError",
]
