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
class Action:
    """One proposed change: launch configuration, prose, or both."""

    server_args: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, Any] = field(default_factory=dict)
    source_change: str = ""

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
        """Whether this action carries no actionable content at all."""
        return not (self.has_config or self.has_source_change)


@dataclass(frozen=True)
class Prediction:
    """One predictor answer, already parsed and flag-repaired by the service.

    ``parsed=False`` is the single failure representation: it covers a declined
    answer, a transport error and a malformed body alike, because the pump does
    the same thing in all three cases.

    The service may sample more than once and return every distinct proposal.
    ``actions`` is that list, best-first; the single-action accessors below read
    its head, which is what keeps a caller that wants one answer -- and every
    caller written before sampling existed -- unchanged.
    """

    parsed: bool = False
    actions: tuple[Action, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def first(self) -> Action:
        """The best proposal, or an empty one when there is none."""
        return self.actions[0] if self.actions else Action()

    @property
    def config_actions(self) -> tuple[Action, ...]:
        """Every proposal with a launch-configuration change to benchmark."""
        return tuple(a for a in self.actions if a.has_config)

    @property
    def source_actions(self) -> tuple[Action, ...]:
        """Every proposal that describes a source edit."""
        return tuple(a for a in self.actions if a.has_source_change)

    # Each single-action accessor below reads the first action carrying the
    # content its predicate reports, not the head of the list. Reading the head
    # instead would let ``has_source_change`` be true while ``source_change``
    # came back empty -- the head being a config-only sample -- and the pump
    # would dispatch a specialist with an empty mandate. With one sample every
    # one of these is that sample.

    @property
    def server_args(self) -> dict[str, Any]:
        """The best configuration proposal's launch flags."""
        return (self.config_actions or (Action(),))[0].server_args

    @property
    def envs(self) -> dict[str, Any]:
        """The best configuration proposal's environment variables."""
        return (self.config_actions or (Action(),))[0].envs

    @property
    def source_change(self) -> str:
        """The best source proposal's prose."""
        return (self.source_actions or (Action(),))[0].source_change

    @property
    def has_config(self) -> bool:
        """Whether any proposal has a launch-configuration change."""
        return bool(self.config_actions)

    @property
    def has_source_change(self) -> bool:
        """Whether any proposal describes a source edit."""
        return bool(self.source_actions)

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


def _as_action(value: Any) -> Action:
    """Read one action object, tolerating missing or mistyped members."""
    raw = value if isinstance(value, dict) else {}
    return Action(
        server_args=_as_str_map(raw.get("server_args")),
        envs=_as_str_map(raw.get("envs")),
        source_change=str(raw.get("source_change") or "").strip(),
    )


def _actions_of(payload: dict[str, Any]) -> tuple[Action, ...]:
    """Every proposal in a response body, best-first.

    ``actions`` is what a sampling service sends and ``action`` is what every
    service sends, so reading the list and falling back to the single field
    covers both without a schema version to branch on. Empty actions are
    dropped: the service already reports how many samples produced nothing, and
    an empty one here would become a variant with no flags to benchmark.
    """
    rows = payload.get("actions")
    if isinstance(rows, list) and rows:
        return tuple(a for a in (_as_action(row) for row in rows) if not a.is_empty)
    single = _as_action(payload.get("action"))
    return () if single.is_empty else (single,)


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

    return Prediction(
        parsed=True,
        actions=_actions_of(payload),
        meta=_as_str_map(payload.get("meta")),
    )
