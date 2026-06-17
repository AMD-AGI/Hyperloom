# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read a session's ``session_breakdown`` back out of a Langfuse trace.

The write side (:mod:`langfuse_emitter`) attaches the full
``session_breakdown.json`` as the **root output** of each session's trace via
``record_session_breakdown``. This module is the symmetric **read** side: given
a trace id (or a correlation seed such as the ``claw_session_id``), it fetches
the trace and returns that breakdown dict — which the upload path then feeds to
:func:`inference_optimizer.breakdown.agent_timeline.build_agent_timeline` to
produce the ``agent_timeline`` section.

Design notes:

* **Read-only, best-effort.** Every failure degrades to ``None`` (with a
  logged reason); nothing here raises into the upload path.
* **No new dependency.** The Langfuse public REST API is called with the
  stdlib (``urllib``) and HTTP Basic auth (``public_key`` : ``secret_key``).
  When the ``langfuse`` SDK happens to be installed it is used as a fallback.
* **Trace id derivation** reuses :func:`langfuse_mapping.derive_trace_id` so a
  reader and the writer always agree on the id for a given session.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .langfuse_mapping import derive_trace_id
from .trace_env import langfuse_credentials

log = logging.getLogger(__name__)

_TRACE_API_PATH = "/api/public/traces/"
_OBSERVATIONS_API_PATH = "/api/public/observations"


def resolve_trace_id(*, trace_id: str | None = None, seed: str | None = None) -> str | None:
    """Resolve the Langfuse trace id from an explicit id or a correlation seed.

    Args:
        trace_id: An explicit 32-char trace id (returned as-is when given).
        seed: A correlation seed (typically ``claw_session_id`` or the internal
            session id); hashed via :func:`derive_trace_id`.

    Returns:
        The resolved trace id, or ``None`` when neither input was usable.
    """
    if trace_id and trace_id.strip():
        return trace_id.strip()
    if seed and str(seed).strip():
        return derive_trace_id(str(seed).strip())
    return None


def _credentials(creds: dict[str, str] | None) -> dict[str, str] | None:
    """Return a complete ``{host, public_key, secret_key}`` set or ``None``.

    Args:
        creds: Optional explicit override; falls back to env-derived creds.

    Returns:
        A dict with all three keys present, or ``None`` when incomplete.
    """
    src = creds or langfuse_credentials()
    host = (src.get("LANGFUSE_HOST") or src.get("host") or "").strip().rstrip("/")
    public_key = (src.get("LANGFUSE_PUBLIC_KEY") or src.get("public_key") or "").strip()
    secret_key = (src.get("LANGFUSE_SECRET_KEY") or src.get("secret_key") or "").strip()
    if not (host and public_key and secret_key):
        return None
    return {"host": host, "public_key": public_key, "secret_key": secret_key}


def _get_json(url: str, creds: dict[str, str], timeout: float) -> Any | None:
    """GET a Langfuse public-API URL with Basic auth and parse the JSON body.

    Args:
        url: Fully-qualified request URL.
        creds: A complete ``{host, public_key, secret_key}`` set.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed JSON (dict/list), or ``None`` on any failure.
    """
    token = base64.b64encode(f"{creds['public_key']}:{creds['secret_key']}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.warning("langfuse_reader: HTTP %s GET %s", exc.code, url)
    except (urllib.error.URLError, TimeoutError) as exc:
        log.warning("langfuse_reader: network error GET %s: %s", url, exc)
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("langfuse_reader: unparseable body GET %s: %s", url, exc)
    return None


def fetch_trace(
    trace_id: str,
    *,
    credentials: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """Fetch one trace (with its details) from the Langfuse public REST API.

    Args:
        trace_id: The trace id to fetch.
        credentials: Optional explicit ``{host, public_key, secret_key}`` (or
            the ``LANGFUSE_*`` env-key form); defaults to env-derived creds.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed trace JSON dict, or ``None`` on any failure (missing creds,
        network error, non-2xx, unparseable body).
    """
    creds = _credentials(credentials)
    if creds is None:
        log.warning("langfuse_reader: credentials incomplete; cannot fetch trace %s", trace_id)
        return None
    result = _get_json(f"{creds['host']}{_TRACE_API_PATH}{trace_id}", creds, timeout)
    return result if isinstance(result, dict) else None


def fetch_observations(
    trace_id: str,
    *,
    name_prefix: str | None = None,
    credentials: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Fetch a trace's observations (spans/generations) from the public API.

    Paginates ``GET /api/public/observations?traceId=...`` and returns the flat
    list. The per-event KB granularity (recipe-snapshot reads, per-iter
    ``kb_assess`` / ``kb_priors``) lives in **observations**, not the trace
    ``output``, so the KB-timeline langfuse path needs this.

    Args:
        trace_id: The trace id whose observations to fetch.
        name_prefix: When set, keep only observations whose ``name`` starts with
            this prefix (client-side filter, e.g. ``"kb:recipe_snapshot"``).
        credentials: Optional explicit credentials override.
        timeout: Per-request timeout in seconds.
        max_pages: Safety cap on pagination.

    Returns:
        A list of observation dicts (possibly empty); never raises.
    """
    creds = _credentials(credentials)
    if creds is None:
        log.warning("langfuse_reader: credentials incomplete; cannot fetch observations for %s", trace_id)
        return []

    limit = 100
    out: list[dict[str, Any]] = []
    page = 1
    truncated = False
    while True:
        if page > max_pages:
            truncated = True
            break
        url = f"{creds['host']}{_OBSERVATIONS_API_PATH}?traceId={trace_id}&page={page}&limit={limit}"
        payload = _get_json(url, creds, timeout)
        if not isinstance(payload, dict):
            break
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            break
        out.extend(r for r in rows if isinstance(r, dict))
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        total_pages = meta.get("totalPages")
        if isinstance(total_pages, int):
            if page >= total_pages:
                break
        elif len(rows) < limit:
            # No usable totalPages: a short page means we've reached the end.
            break
        page += 1

    if truncated:
        log.warning(
            "langfuse_reader: observation pagination for %s hit max_pages=%d (>%d rows); "
            "results may be truncated",
            trace_id,
            max_pages,
            max_pages * limit,
        )

    if name_prefix:
        out = [o for o in out if str(o.get("name") or "").startswith(name_prefix)]
    return out


def _coerce_output(output: Any) -> dict[str, Any] | None:
    """Coerce an ``output`` value (dict or JSON-encoded string) to a dict.

    Args:
        output: The raw ``output`` value from a trace or observation.

    Returns:
        The dict form, or ``None`` when absent / not an object.
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (ValueError, json.JSONDecodeError):
            return None
    return output if isinstance(output, dict) else None


def _breakdown_from_observations(observations: list[Any]) -> dict[str, Any] | None:
    """Find the ``session_breakdown`` observation and return its ``output``.

    ``record_session_breakdown`` attaches the full breakdown as the ``output``
    of a ``session_breakdown`` span observation (not the trace root output), so
    that is the authoritative place to recover it from.

    Args:
        observations: A list of observation dicts.

    Returns:
        The breakdown dict, or ``None`` when no usable observation is found.
    """
    for obs in observations:
        if isinstance(obs, dict) and obs.get("name") == "session_breakdown":
            bd = _coerce_output(obs.get("output"))
            if bd is not None:
                return bd
    return None


def _extract_breakdown(
    trace: dict[str, Any],
    trace_id: str,
    *,
    credentials: dict[str, str] | None,
    timeout: float,
) -> dict[str, Any] | None:
    """Recover the breakdown from a fetched trace, trying every known location.

    Order: the ``session_breakdown`` span among the trace's embedded
    ``observations`` (where the writer puts it) -> the trace root ``output``
    (fast path, in case a future writer sets it) -> a dedicated observations
    fetch for the ``session_breakdown`` span.

    Args:
        trace: The fetched trace JSON dict.
        trace_id: The trace id (for the fallback observations fetch).
        credentials: Optional explicit credentials override.
        timeout: Per-request timeout in seconds.

    Returns:
        The breakdown dict, or ``None`` when not recoverable.
    """
    embedded = trace.get("observations")
    if isinstance(embedded, list):
        bd = _breakdown_from_observations(embedded)
        if bd is not None:
            return bd

    bd = _coerce_output(trace.get("output"))
    if bd is not None:
        return bd

    # Embedded observations were absent or only IDs: fetch them explicitly.
    obs = fetch_observations(
        trace_id, name_prefix="session_breakdown", credentials=credentials, timeout=timeout
    )
    return _breakdown_from_observations(obs)


def fetch_session_breakdown(
    *,
    trace_id: str | None = None,
    seed: str | None = None,
    credentials: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """Recover the ``session_breakdown`` dict attached to a session's trace.

    Resolves the trace id (explicit or derived from ``seed``), fetches the
    trace, and returns its root ``output`` (the breakdown the writer attached).

    Args:
        trace_id: Explicit trace id; takes precedence over ``seed``.
        seed: Correlation seed (e.g. ``claw_session_id``) when ``trace_id`` is
            unknown.
        credentials: Optional explicit credentials override.
        timeout: Per-request timeout in seconds.

    Returns:
        The recovered breakdown dict, or ``None`` on any failure.
    """
    resolved = resolve_trace_id(trace_id=trace_id, seed=seed)
    if not resolved:
        log.warning("langfuse_reader: no trace_id and no usable seed; cannot fetch breakdown")
        return None
    trace = fetch_trace(resolved, credentials=credentials, timeout=timeout)
    if trace is None:
        return None
    breakdown = _extract_breakdown(trace, resolved, credentials=credentials, timeout=timeout)
    if breakdown is None:
        log.warning("langfuse_reader: trace %s has no recoverable session_breakdown", resolved)
    return breakdown


__all__ = [
    "fetch_observations",
    "fetch_session_breakdown",
    "fetch_trace",
    "resolve_trace_id",
]
