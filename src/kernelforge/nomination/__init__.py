# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer half of the nomination contract.

Mirrors ``hyperloom.orchestrator.kernel.nomination_request`` field for field.
The two halves are deliberately separate modules rather than a shared import:
forge stays independently importable, and ``PROTOCOL_VERSION`` is what keeps
them honest. Bump both together.

The seam a real implementation replaces is :func:`nominate`; everything else
here is shell -- read, validate, delegate, assemble.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Must equal the producer's value. A mismatch stops the run rather than
#: risking a partially understood request.
PROTOCOL_VERSION = 1

LANE_REWRITE = "rewrite"
LANE_FUSION = "fusion"
LANE_GEMM = "gemm"

KNOWN_LANES = frozenset({LANE_REWRITE, LANE_FUSION, LANE_GEMM})

#: The complete set of keys a request may carry. Anything outside this set is a
#: field Hyperloom wrote that this build does not understand -- silently dropping
#: it hides a version skew (a producer that thinks a field is honoured when it is
#: not), so an unknown key stops the run rather than being swallowed.
_KNOWN_REQUEST_KEYS = frozenset(
    {
        "protocol_version",
        "lane",
        "trace_path",
        "candidates_path",
        "lane_budget_sec",
        "max_kernels",
        "trace_captured_after",
    }
)


class NominationError(RuntimeError):
    """A malformed request or an unreadable candidate list stops the run."""


@dataclass(frozen=True)
class NominationRequest:
    """One lane's brief, as read off disk."""

    lane: str
    trace_path: str
    candidates_path: str
    lane_budget_sec: int
    max_kernels: int
    trace_captured_after: str = ""


@dataclass(frozen=True)
class Target:
    """One kernel a nominator picked, plus the budget it was given."""

    kernel_name: str
    source_file: str
    budget_sec: int
    gpu_pct: float = 0.0
    reason: str = ""


@dataclass
class NominationSummary:
    """Counts that make "how many hot kernels did we rescue" answerable.

    Deliberately counts only -- no per-kernel detail, so this stays inside the
    "forge does not report what it looked at and skipped" decision.
    """

    candidates_seen: int = 0
    resolved: int = 0
    selected: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Candidate:
    """One row of the hot-kernel list, reduced to what a nominator needs."""

    kernel_name: str
    source_file: str = ""
    gpu_pct: float = 0.0
    reason_class: str = ""
    attempts: int = 0
    rejected: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """A row without a source file cannot be handed to a campaign as-is."""
        return bool(self.source_file)


def read_request(path: str | Path) -> NominationRequest:
    """Read and validate a nomination request written by Hyperloom.

    Args:
        path: Path to the request JSON.

    Returns:
        The parsed request.

    Raises:
        NominationError: On unreadable JSON, an unknown protocol version, an
            unknown lane, an unknown request field, or a non-positive budget or
            ceiling.
    """
    payload = _load_json(path, what="nomination request")
    if not isinstance(payload, dict):
        raise NominationError(f"nomination request must be a JSON object: {path}")
    unknown = set(payload) - _KNOWN_REQUEST_KEYS
    if unknown:
        raise NominationError(f"unknown nomination request field(s): {sorted(unknown)}")
    version = payload.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise NominationError(f"unsupported nomination protocol {version!r}; this build speaks {PROTOCOL_VERSION}")
    lane = str(payload.get("lane") or "")
    if lane not in KNOWN_LANES:
        raise NominationError(f"unknown lane {lane!r}; expected one of {sorted(KNOWN_LANES)}")
    return NominationRequest(
        lane=lane,
        trace_path=str(payload.get("trace_path") or ""),
        candidates_path=str(payload.get("candidates_path") or ""),
        lane_budget_sec=_positive_int(payload.get("lane_budget_sec"), field_name="lane_budget_sec"),
        max_kernels=_positive_int(payload.get("max_kernels"), field_name="max_kernels"),
        trace_captured_after=str(payload.get("trace_captured_after") or ""),
    )


def read_candidates(path: str | Path) -> list[Candidate]:
    """Read the hot-kernel list, keeping unresolved rows.

    Unresolved rows are the whole point of handing the full list over: a
    nominator that never sees them cannot rescue them.

    Args:
        path: Path to the candidate list JSON.

    Returns:
        Candidates in file order; ranking is the nominator's business.

    Raises:
        NominationError: On unreadable JSON or a missing ``hot_kernels`` array.
    """
    payload = _load_json(path, what="candidate list")
    rows = payload.get("hot_kernels") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise NominationError(f"candidate list has no hot_kernels array: {path}")
    candidates: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("kernel_name") or row.get("name") or "").strip()
        if not name:
            continue
        candidates.append(
            Candidate(
                kernel_name=name,
                source_file=str(row.get("source_file") or "").strip(),
                gpu_pct=_finite_float(row.get("gpu_pct")),
                reason_class=str(row.get("reason_class") or "").strip(),
                attempts=max(0, _int_or_zero(row.get("attempts"))),
                rejected=bool(row.get("rejected")),
                raw=row,
            )
        )
    return candidates


def nominate(request: NominationRequest, candidates: list[Candidate]) -> list[Target]:
    """Pick the kernels to optimize and split the budget across them.

    **This is the seam.** The shipped implementation is a placeholder that only
    reads the candidate list; it does not parse the trace and does not attempt
    source resolution, which is what a real nominator adds.

    Args:
        request: The lane brief, including budget and ceiling.
        candidates: Rows from :func:`read_candidates`.

    Returns:
        At most ``request.max_kernels`` targets, strongest first.
    """
    from kernelforge.nomination.stub import nominate_from_candidates

    return nominate_from_candidates(request, candidates)


def summarize(candidates: list[Candidate], targets: list[Target]) -> NominationSummary:
    """Count what was seen, what was resolvable, and what was picked."""
    return NominationSummary(
        candidates_seen=len(candidates),
        resolved=sum(1 for candidate in candidates if candidate.is_resolved),
        selected=len(targets),
    )


@dataclass(frozen=True)
class Resolution:
    """Everything one nomination pass decided, ready for the CLI to act on."""

    request: NominationRequest
    targets: tuple[Target, ...]
    summary: NominationSummary


def resolve(nomination_input: str | Path) -> Resolution:
    """Read the brief, rank the candidates, and pick targets.

    Args:
        nomination_input: Path to the nomination request JSON.

    Returns:
        The request, the chosen targets, and the counts to report.

    Raises:
        NominationError: On a malformed request or candidate list.
    """
    request = read_request(nomination_input)
    candidates = read_candidates(request.candidates_path)
    targets = nominate(request, candidates)
    return Resolution(
        request=request,
        targets=tuple(targets),
        summary=summarize(candidates, targets),
    )


def patch_entry(
    target: Target,
    *,
    patch_path: str,
    base_commit: str = "",
    micro_speedup: float = 0.0,
    kernel_repo: str = "",
    snapshot_dir: str = "",
) -> dict[str, Any]:
    """Build one result-envelope entry for a target that produced a patch.

    Field names mirror ``hyperloom.orchestrator.kernel.nomination_result``; the
    three the consumer requires are the kernel name, the patch path, and the
    target file.

    Args:
        target: The nominated target this patch came from.
        patch_path: Path to the published patch.
        base_commit: Commit the patch is diffed against.
        micro_speedup: Forge's own measurement, used only as a queue tiebreaker.
        kernel_repo: Repo root, required for a multi-file patch.
        snapshot_dir: Snapshot dir, which enables atomic multi-file apply.

    Returns:
        The entry mapping.
    """
    entry: dict[str, Any] = {
        "kernel_name": target.kernel_name,
        "patch_path": str(patch_path),
        "target_file": target.source_file,
        "micro_speedup": float(micro_speedup or 0.0),
    }
    for key, value in (
        ("base_commit", base_commit),
        ("kernel_repo", kernel_repo),
        ("snapshot_dir", snapshot_dir),
    ):
        if str(value or "").strip():
            entry[key] = str(value)
    return entry


def _load_json(path: str | Path, *, what: str) -> Any:
    """Read JSON, turning every failure into one contract error."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NominationError(f"could not read {what} {path}: {error}") from error


def _positive_int(value: Any, *, field_name: str) -> int:
    """Coerce to a positive int; anything else is a contract violation."""
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise NominationError(f"{field_name} must be an integer, got {value!r}") from error
    if number <= 0:
        raise NominationError(f"{field_name} must be positive, got {number}")
    return number


def _finite_float(value: Any) -> float:
    """Non-finite or non-numeric shares rank as zero rather than crashing."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _int_or_zero(value: Any) -> int:
    """Missing counters mean zero; a bad counter must not stop nomination."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
