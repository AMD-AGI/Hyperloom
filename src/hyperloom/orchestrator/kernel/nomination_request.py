# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Hyperloom hands forge when forge picks the kernels itself.

In `--auto` mode Hyperloom names no kernel, so there is no candidate to hang
this on: it travels as its own JSON file, mirroring the `--input-json` shape the
fusion and gemm lanes already use. The consumer half lives in
``kernelforge.nomination`` and mirrors these field names; ``PROTOCOL_VERSION``
is the only thing both sides must agree on before reading anything else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Bump on any field removal or meaning change. The consumer refuses a version
#: it does not know rather than guessing at a partially understood request.
PROTOCOL_VERSION = 1

#: Canonical filename, written beside the other per-attempt forge inputs.
REQUEST_FILENAME = "forge_nomination_input.json"

LANE_REWRITE = "rewrite"
LANE_FUSION = "fusion"
LANE_GEMM = "gemm"

KNOWN_LANES = frozenset({LANE_REWRITE, LANE_FUSION, LANE_GEMM})


class NominationRequestError(ValueError):
    """A request that cannot be built or read is a contract violation, not a skip."""


@dataclass(frozen=True)
class NominationRequest:
    """One lane's self-nomination brief for a single macro cycle."""

    lane: str
    trace_path: str
    candidates_path: str
    lane_budget_sec: int
    max_kernels: int
    trace_captured_after: str = ""
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize every field, including the protocol version the reader gates on."""
        return asdict(self)


def build_request(
    *,
    lane: str,
    trace_path: str,
    candidates_path: str,
    lane_budget_sec: int,
    max_kernels: int,
    trace_captured_after: str = "",
) -> NominationRequest:
    """Validate and build a request; raises rather than emitting a half-formed one.

    Args:
        lane: One of :data:`KNOWN_LANES`.
        trace_path: Absolute path to the raw profiler trace file, not a directory.
        candidates_path: Absolute path to the hot-kernel candidate list.
        lane_budget_sec: Wall-clock seconds this lane may spend this cycle.
        max_kernels: Ceiling on how many targets forge may pick.
        trace_captured_after: Code-state marker for the trace, for triage.

    Returns:
        The validated request.

    Raises:
        NominationRequestError: On an unknown lane, a missing input file, or a
            non-positive budget or ceiling.
    """
    if lane not in KNOWN_LANES:
        raise NominationRequestError(f"unknown lane {lane!r}; expected one of {sorted(KNOWN_LANES)}")
    trace = _require_file(trace_path, field="trace_path")
    candidates = _require_file(candidates_path, field="candidates_path")
    budget = _require_positive_int(lane_budget_sec, field="lane_budget_sec")
    ceiling = _require_positive_int(max_kernels, field="max_kernels")
    return NominationRequest(
        lane=lane,
        trace_path=trace,
        candidates_path=candidates,
        lane_budget_sec=budget,
        max_kernels=ceiling,
        trace_captured_after=str(trace_captured_after or ""),
    )


def write_request(directory: Path, request: NominationRequest) -> Path:
    """Persist a request as UTF-8 JSON and return its path."""
    from hyperloom.common.io import atomic_write_json

    path = Path(directory) / REQUEST_FILENAME
    atomic_write_json(path, request.to_dict(), ensure_ascii=False)
    return path


def _require_file(value: Any, *, field: str) -> str:
    """Resolve a path that must already exist; an absent input is a hard error."""
    raw = str(value or "").strip()
    if not raw:
        raise NominationRequestError(f"{field} is required")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise NominationRequestError(f"{field} is not a file: {raw}")
    return str(path.resolve())


def _require_positive_int(value: Any, *, field: str) -> int:
    """Coerce to a positive int; zero means the caller should not have called."""
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise NominationRequestError(f"{field} must be an integer, got {value!r}") from error
    if number <= 0:
        raise NominationRequestError(f"{field} must be positive, got {number}")
    return number


def read_request(path: Path) -> NominationRequest:
    """Read a request written by :func:`write_request`, for tests and triage.

    The production consumer is ``kernelforge.nomination``; this exists so the
    producer side can round-trip its own output without importing forge.

    Args:
        path: Path to the request JSON.

    Returns:
        The parsed request.

    Raises:
        NominationRequestError: On unreadable JSON or an unknown protocol version.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NominationRequestError(f"could not read nomination request {path}: {error}") from error
    if not isinstance(payload, dict):
        raise NominationRequestError(f"nomination request must be a JSON object: {path}")
    version = payload.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise NominationRequestError(
            f"unsupported nomination protocol {version!r}; this build speaks {PROTOCOL_VERSION}"
        )
    return build_request(
        lane=str(payload.get("lane") or ""),
        trace_path=str(payload.get("trace_path") or ""),
        candidates_path=str(payload.get("candidates_path") or ""),
        lane_budget_sec=payload.get("lane_budget_sec"),
        max_kernels=payload.get("max_kernels"),
        trace_captured_after=str(payload.get("trace_captured_after") or ""),
    )
