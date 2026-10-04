# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HTTP client for the InferenceX public benchmarks API."""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import socket
import ssl
import urllib.request
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from hyperloom.common.url_safety import require_http_url as _base_require_http_url

from .types import BenchmarkMode

log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://inferencex.semianalysis.com/api/v1"
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_MAX_ATTEMPTS = 2
_DERIVED_BATCH_SIZE = 200


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
    except (OSError, ssl.SSLError, EOFError, zlib.error) as exc:
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


def normalize_benchmark_id(value: object) -> str:
    """Normalize API IDs without accepting floats or query fragments."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("benchmark ID must be a positive integer")
    text = str(int(value) if isinstance(value, int) else value).strip()
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ValueError("benchmark ID must be a positive integer")
    return str(int(text))


def fetch_agentic_interactivity(benchmark_ids: list[str | int]) -> dict[str, float | None] | None:
    """Fetch exact P90 by benchmark ID; None means fetch/schema failure, not missing data."""
    ids = list(dict.fromkeys(normalize_benchmark_id(value) for value in benchmark_ids))
    result: dict[str, float | None] = {}
    attempts = _max_attempts()
    for start in range(0, len(ids), _DERIVED_BATCH_SIZE):
        batch = ids[start : start + _DERIVED_BATCH_SIZE]
        url = f"{_base_url()}/derived-agentic-metrics?ids={','.join(batch)}"
        for attempt in range(attempts):
            try:
                body = _fetch_raw(url)
                break
            except InferenceXFetchError as exc:
                if attempt == attempts - 1:
                    log.warning("InferenceX: derived interactivity fetch failed: %s", exc)
                    return None
        try:
            if body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict) or "error" in data:
                raise ValueError("expected a benchmark-ID mapping")
            for key in batch:
                if key not in data:
                    continue
                row = data[key]
                if not isinstance(row, dict) or normalize_benchmark_id(row.get("id")) != key:
                    raise ValueError("derived metric benchmark ID does not match its key")
                value = row.get("p90_e2e_norm_intvty")
                result[key] = (
                    float(value)
                    if not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                    else None
                )
        except (ValueError, OSError, EOFError, OverflowError, zlib.error) as exc:
            log.warning("InferenceX: invalid derived interactivity response: %s", exc)
            return None
    return result


def find_reference_rows(
    rows: list[dict],
    *,
    hardware: str,
    isl: int | None,
    osl: int | None,
    precision: str = "",
    benchmark_mode: BenchmarkMode = "synthetic",
) -> list[dict]:
    """Filter matching single-node rows without treating agentic lengths as fixed shapes."""
    hw = str(hardware or "").strip().casefold()
    matched = [
        r
        for r in rows
        if isinstance(r, dict)
        and str(r.get("hardware") or "").strip().casefold() == hw
        and not bool(r.get("is_multinode"))
        and not bool(r.get("disagg"))
    ]
    if benchmark_mode == "agentx":
        matched = [r for r in matched if r.get("benchmark_type") == "agentic_traces"]
    else:
        matched = [
            r
            for r in matched
            if isl is not None
            and osl is not None
            and _to_int(r.get("isl")) == int(isl)
            and _to_int(r.get("osl")) == int(osl)
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
    "fetch_agentic_interactivity",
    "find_reference_rows",
    "normalize_benchmark_id",
]
