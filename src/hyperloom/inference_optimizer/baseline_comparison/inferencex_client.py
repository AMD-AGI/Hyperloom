# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""HTTP client for the InferenceX public benchmarks API.

Endpoint shape:
``https://inferencex.semianalysis.com/api/v1/benchmarks?model=<name>``.

Two design rules driven by the call-site (target_analysis executor):

* **Never raise on network / parsing problems.** Returns ``None`` (or
  an empty list, depending on the call) plus a structured warning the
  caller can persist into ``BaselineSummary.warning``.
* **Bounded timeout + small retry budget.**

Optional environment overrides:

* ``INFERENCEX_BASE_URL``     — defaults to upstream public URL.
* ``INFERENCEX_TIMEOUT_SEC``  — per-request timeout (default 5).
* ``INFERENCEX_MAX_ATTEMPTS`` — default 2.
* ``INFERENCEX_INSECURE``     — accept ``"1"``/``"true"`` to skip TLS
  verification.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import socket
import ssl
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import quote


log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://inferencex.semianalysis.com/api/v1"
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_MAX_ATTEMPTS = 2




def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        raise InferenceXFetchError(f"unsupported URL scheme: {scheme!r}")


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
    _require_http_url(url)
    req = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "src/hyperloom/inference_optimizer/baseline_comparison",
        },
    )
    ctx = _build_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=_timeout_sec(), context=ctx) as resp:  # nosec B310 - URL scheme checked above.
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
    """Public accessor for the resolved API base URL (honours env override).

    Returns:
        str: The base URL used for benchmark queries, suitable for recording
            as the provenance ``source`` on a persisted comparison artefact.
    """
    return _base_url()


def _to_int(value: object) -> int | None:
    """Best-effort integer coercion used by dimension filtering.

    Args:
        value: Arbitrary value to coerce.

    Returns:
        int | None: The integer value, or ``None`` when it cannot be parsed.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_rows(model_api_name: str) -> list[dict] | None:
    """Fetch InferenceX benchmark rows for a model. Never raises.

    Builds ``<base>/benchmarks?model=<name>``, performs a bounded-retry GET via
    :func:`_fetch_raw`, transparently gunzips, and JSON-parses the response.

    Args:
        model_api_name (str): InferenceX API model identifier (e.g.
            ``DeepSeek-R1-0528``) — the value returned by
            ``target_analyzer.to_inferencex_name``.

    Returns:
        list[dict] | None: A list of benchmark record dicts on success, an
            empty list when the model has no rows or the API reports a
            structured error, or ``None`` on any network / parse failure.
    """
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
            name, attempts, last_exc,
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
    """Filter InferenceX rows down to those aligned with our run. Never raises.

    Alignment is **strict** on ``hardware``, ``isl`` and ``osl`` — the whole
    point of the comparison is that the shapes match. Disaggregated and
    multinode rows are dropped as well: their per-GPU throughput is not
    comparable to a single-node colocated run (they use a different serving
    topology). ``precision`` is **also strict when supplied**: rows of a
    different precision are dropped rather than substituted, so an fp4 run is
    never compared against fp8 numbers. When ``precision`` is empty the filter
    is skipped (precision unconstrained).

    Args:
        rows (list[dict]): Raw benchmark records from :func:`fetch_rows`.
        hardware (str): Target GPU id to match against each row's
            ``hardware`` field (case-insensitive).
        isl (int): Required input sequence length.
        osl (int): Required output sequence length.
        precision (str): Optional precision label (e.g. ``fp8`` / ``fp4``);
            a hard filter when supplied.

    Returns:
        list[dict]: The subset of ``rows`` matching the required dimensions
            (possibly empty).
    """
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
        matched = [
            r for r in matched
            if str(r.get("precision") or "").strip().casefold() == prec
        ]
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
