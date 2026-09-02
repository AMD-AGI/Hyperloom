# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Call the predictor service and normalise its answer.

Stdlib only, one request, no retry. A retry would be the wrong trade here: the
caller is inside the tick loop, the answer is advisory, and a chain that stops
one step early costs nothing measurable. Every failure — transport, status,
malformed body — collapses to the same "no action" result the model returns
when it declines to answer, so the pump has one path to handle rather than a
taxonomy of errors.

The service owns prompt rendering, generation, parsing and flag repair. What
comes back is already an action; this module only checks its shape.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.url_safety import require_http_url

log = logging.getLogger(__name__)

#: Response envelope this client understands.
RESPONSE_SCHEMA_PREFIX = "primatune.predictor_response."

_PREDICT_PATH = "/v1/predict"


@dataclass(frozen=True)
class Prediction:
    """One predictor answer, already parsed and flag-repaired by the service.

    ``parsed=False`` is the single failure representation: it covers a declined
    answer, a transport error and a malformed body alike, because the pump does
    the same thing in all three cases.
    """

    parsed: bool = False
    server_args: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, Any] = field(default_factory=dict)
    source_change: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def has_config(self) -> bool:
        """Whether there is a launch-configuration change to benchmark."""
        return bool(self.server_args or self.envs)

    @property
    def has_source_change(self) -> bool:
        """Whether there is prose describing a source edit."""
        return bool(self.source_change.strip())

    @property
    def is_empty(self) -> bool:
        """Whether the answer carries no actionable content at all."""
        return not (self.has_config or self.has_source_change)


def _failed(reason: str) -> Prediction:
    log.info("predictor_client: no action (%s)", reason)
    return Prediction(parsed=False, error=reason)


def _as_str_map(value: Any) -> dict[str, Any]:
    """Coerce a JSON object into a string-keyed map, dropping anything else."""
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items() if str(k).strip()}


def predict(request: dict[str, Any], *, endpoint: str, timeout_sec: float) -> Prediction:
    """POST ``request`` to the predictor and return its answer.

    Args:
        request (dict[str, Any]): Body from
            :func:`~hyperloom.orchestrator.predictor.payload.build_request`.
        endpoint (str): Service base URL.
        timeout_sec (float): Per-request timeout. Exceeding it ends the chain.

    Returns:
        Prediction: ``parsed=False`` on any failure; the predictor is never
            allowed to fail a session.
    """
    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        return _failed("no endpoint configured")
    url = f"{base}{_PREDICT_PATH}"
    try:
        require_http_url(url, context="predictor endpoint")
    except ValueError as exc:
        return _failed(f"unsafe endpoint {url!r}: {exc}")

    try:
        data = json.dumps(request).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return _failed(f"request is not JSON-serialisable: {exc}")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - scheme checked above.
            status = int(getattr(resp, "status", 0) or 0)
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return _failed(f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _failed(f"transport error: {exc}")

    if status and not 200 <= status < 300:
        return _failed(f"HTTP {status}")

    try:
        payload = json.loads(body or "{}")
    except (json.JSONDecodeError, ValueError) as exc:
        return _failed(f"malformed response body: {exc}")
    if not isinstance(payload, dict):
        return _failed(f"response is {type(payload).__name__}, expected object")

    schema = str(payload.get("schema") or "")
    if schema and not schema.startswith(RESPONSE_SCHEMA_PREFIX):
        return _failed(f"unexpected response schema {schema!r}")

    if not payload.get("parsed"):
        return Prediction(parsed=False, meta=_as_str_map(payload.get("meta")), error="predictor declined")

    action = payload.get("action")
    action = action if isinstance(action, dict) else {}
    return Prediction(
        parsed=True,
        server_args=_as_str_map(action.get("server_args")),
        envs=_as_str_map(action.get("envs")),
        source_change=str(action.get("source_change") or "").strip(),
        meta=_as_str_map(payload.get("meta")),
    )
