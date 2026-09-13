# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HTTP client for the InferenceX public benchmarks API."""

from __future__ import annotations

import gzip
import json
import logging
import os
import socket
import ssl
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from hyperloom.common.url_safety import require_http_url as _base_require_http_url

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://inferencex.semianalysis.com/api/v1"
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_MAX_ATTEMPTS = 2


class InferenceXFetchError(Exception):
    """Raised on any InferenceX fetch failure (unsupported URL scheme, non-200 status, network or transport error)."""

    pass


def _require_http_url(url: str) -> None:
    _base_require_http_url(url, error=InferenceXFetchError)


def _base_url() -> str:
    """Resolve the API base URL from the environment."""
    return os.environ.get("INFERENCEX_BASE_URL", "").strip() or DEFAULT_BASE_URL


def _timeout_sec() -> float:
    """Resolve the per-request timeout from the environment."""
    raw = os.environ.get("INFERENCEX_TIMEOUT_SEC", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SEC
    try:
        return max(0.5, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


def _max_attempts() -> int:
    """Resolve the retry attempt budget from the environment."""
    raw = os.environ.get("INFERENCEX_MAX_ATTEMPTS", "").strip()
    if not raw:
        return DEFAULT_MAX_ATTEMPTS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS


def _fetch_raw(url: str) -> bytes:
    """Single HTTP GET with gzip support."""
    _require_http_url(url)
    req = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "src/hyperloom/inference_optimizer/baseline_comparison",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout_sec()) as resp:  # nosec B310 - URL scheme checked above.
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


def base_url() -> str:
    """Public accessor for the resolved API base URL (honours env override)."""
    return _base_url()


def _to_int(value: object) -> int | None:
    """Best-effort integer coercion used by dimension filtering."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_rows(model_api_name: str) -> list[dict] | None:
    """Fetch InferenceX benchmark rows for a model. Never raises."""
    name = str(model_api_name or "").strip()
    if not name:
        return None
    url = f"{_base_url()}/benchmarks?model={quote(name)}"
    attempts = _max_attempts()
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            body = _fetch_raw(url)
        except InferenceXFetchError as exc:
            last_exc = exc
            continue
        try:
            if body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            log.warning("InferenceX: JSON parse failed for %s: %s", name, exc)
            return None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "error" in data:
                log.warning("InferenceX API error for %s: %s", name, data.get("error"))
                return []
            for key in ("data", "benchmarks", "results", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []
    if last_exc is not None:
        log.warning(
            "InferenceX: fetch failed for %s after %d attempt(s): %s",
            name,
            attempts,
            last_exc,
        )
    return None


def find_reference_rows(
    rows: list[dict],
    *,
    hardware: str,
    isl: int,
    osl: int,
    precision: str = "",
) -> list[dict]:
    """Filter InferenceX rows down to those aligned with our run. Never raises."""
    hw = str(hardware or "").strip().casefold()
    matched = [
        r
        for r in rows
        if isinstance(r, dict)
        and str(r.get("hardware") or "").strip().casefold() == hw
        and _to_int(r.get("isl")) == int(isl)
        and _to_int(r.get("osl")) == int(osl)
        and not bool(r.get("is_multinode"))
        and not bool(r.get("disagg"))
    ]
    prec = str(precision or "").strip().casefold()
    if prec:
        matched = [r for r in matched if str(r.get("precision") or "").strip().casefold() == prec]
    return matched


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_MAX_ATTEMPTS",
    "InferenceXFetchError",
    "base_url",
    "fetch_rows",
    "find_reference_rows",
]
