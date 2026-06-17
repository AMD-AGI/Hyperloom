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

    url = f"{creds['host']}{_TRACE_API_PATH}{trace_id}"
    token = base64.b64encode(f"{creds['public_key']}:{creds['secret_key']}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        log.warning("langfuse_reader: HTTP %s fetching trace %s", exc.code, trace_id)
    except (urllib.error.URLError, TimeoutError) as exc:
        log.warning("langfuse_reader: network error fetching trace %s: %s", trace_id, exc)
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("langfuse_reader: unparseable trace body for %s: %s", trace_id, exc)
    return None


def _extract_output(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the breakdown dict out of a fetched trace's ``output``.

    The Langfuse SDK / API can store ``output`` either as a dict or as a
    JSON-encoded string; both are handled.

    Args:
        trace: A fetched trace JSON dict.

    Returns:
        The breakdown dict, or ``None`` when ``output`` is absent / not an object.
    """
    output = trace.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (ValueError, json.JSONDecodeError):
            return None
    if isinstance(output, dict):
        return output
    return None


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
    breakdown = _extract_output(trace)
    if breakdown is None:
        log.warning("langfuse_reader: trace %s has no usable output (breakdown)", resolved)
    return breakdown


__all__ = [
    "fetch_session_breakdown",
    "fetch_trace",
    "resolve_trace_id",
]
