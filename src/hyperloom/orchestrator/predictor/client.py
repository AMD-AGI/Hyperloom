# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Call the predictor service and read its answer.

One request, no retry: the answer is advisory. Every failure -- transport,
HTTP status, malformed body -- becomes the same unparsed answer the service
returns when it declines, so the pump handles one shape. The service owns
prompt rendering, generation, parsing and flag repair.
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.url_safety import require_http_url

log = logging.getLogger(__name__)

PREDICT_PATH = "/v1/predict"
RESPONSE_SCHEMA_PREFIX = "primatune.predictor_response."


@dataclass(frozen=True)
class Action:
    """One proposal: launch flags and env vars, source-change prose, or both."""

    server_args: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, Any] = field(default_factory=dict)
    source_change: str = ""

    @property
    def has_config(self) -> bool:
        """Whether there is a launch-configuration change to benchmark."""
        return bool(self.server_args or self.envs)


@dataclass(frozen=True)
class Prediction:
    """One answer, best-first; ``parsed=False`` covers a decline, a transport error and a malformed body alike."""

    parsed: bool = False
    actions: tuple[Action, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def config_actions(self) -> tuple[Action, ...]:
        """Every proposal with a launch-configuration change."""
        return tuple(a for a in self.actions if a.has_config)

    @property
    def source_change(self) -> str:
        """The first proposal's source-change prose, or ``""``."""
        return next((a.source_change for a in self.actions if a.source_change), "")


def _str_map(value: Any) -> dict[str, Any]:
    return {str(k): v for k, v in value.items() if str(k).strip()} if isinstance(value, dict) else {}


def _action(value: Any) -> Action:
    raw = value if isinstance(value, dict) else {}
    return Action(
        server_args=_str_map(raw.get("server_args")),
        envs=_str_map(raw.get("envs")),
        source_change=str(raw.get("source_change") or "").strip(),
    )


def _failed(reason: str) -> Prediction:
    # A warning: an unreachable or broken service otherwise turns the feature off with no visible sign.
    log.warning("predictor: no answer (%s)", reason)
    return Prediction(error=reason)


def predict(request: dict[str, Any], *, endpoint: str, timeout_sec: float) -> Prediction:
    """POST ``request`` to ``{endpoint}/v1/predict`` and read the answer; never raises on a service failure."""
    url = f"{endpoint.strip().rstrip('/')}{PREDICT_PATH}"
    try:
        require_http_url(url, context="predictor endpoint")
    except ValueError as exc:
        return _failed(str(exc))
    req = urllib.request.Request(
        url,
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - scheme checked above
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as exc:
        return _failed(f"HTTP {exc.code}")
    # HTTPException is not an OSError: a bad port (InvalidURL), a non-HTTP listener (BadStatusLine) or a cut-off body
    # (IncompleteRead) would otherwise escape.
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        return _failed(f"transport error: {exc!r}")
    except json.JSONDecodeError as exc:
        return _failed(f"malformed response body: {exc}")
    if not isinstance(payload, dict):
        return _failed(f"response is {type(payload).__name__}, expected an object")
    schema = str(payload.get("schema") or "")
    if schema and not schema.startswith(RESPONSE_SCHEMA_PREFIX):
        return _failed(f"unexpected response schema {schema!r}")
    meta = _str_map(payload.get("meta"))
    if not payload.get("parsed"):
        return Prediction(meta=meta, error="predictor declined")
    # A sampling service sends ``actions``; every service sends ``action``.
    rows = payload.get("actions")
    rows = rows if isinstance(rows, list) and rows else [payload.get("action")]
    actions = tuple(a for a in map(_action, rows) if a.has_config or a.source_change)
    return Prediction(parsed=True, actions=actions, meta=meta)
