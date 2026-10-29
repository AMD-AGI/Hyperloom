# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether this session already holds a verdict for an explore variant on the stack it would launch on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from hyperloom_common.coerce import to_str_list
from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
from ...policy.gate import PolicyDenied

from ._grid_base import DEFAULT_KEEP_THRESHOLD_PCT
from ._proposal_identity import (
    BaseControls,
    content_fingerprint,
    controls_of,
    effective_fingerprint,
    normalize_proposal,
)

__all__ = [
    "GAIN_BELOW_THRESHOLD",
    "MeasuredOn",
    "explore_grid_already_covered",
    "grid_fingerprints",
    "measured_verdict",
    "stack_of",
    "stack_of_measured",
]

# ``reason`` of a REVERT the throughput gate ruled: the variant's gain missed the KEEP bar. Only the throughput gate
# moves with the bar, so only these rows can go stale when it falls.
GAIN_BELOW_THRESHOLD = "gain_below_threshold"

# Outcomes that carry a measured verdict. A FAILED or overtime-killed row says nothing about the launch's merit, so it
# may run again.
_MEASURED_OUTCOMES = frozenset({"KEEP", "REVERT"})


@dataclass(frozen=True)
class MeasuredOn:
    """What a ledger verdict must still hold against to answer a variant: the stack, the workload and the KEEP bar."""

    stack: Mapping[str, Any]
    workload_signature: str
    keep_threshold_pct: float


def stack_of(params: Mapping[str, Any]) -> dict[str, Any]:
    """The stack an explore task's variants launch on, in the shape a ledger row's ``measured_against`` records."""
    return {
        "throughput": float(params.get("base_tput") or 0.0),
        "extra_server_args": str(params.get("base_extra_args") or "").strip(),
        "extra_envs": {str(k): str(v) for k, v in dict(params.get("base_extra_envs") or {}).items()},
        "remove_args": list(dict.fromkeys(to_str_list(params.get("base_remove_args")))),
        "unset_envs": list(dict.fromkeys(to_str_list(params.get("base_unset_envs")))),
        "args_mode": str(params.get("base_args_mode") or "append").strip().lower(),
    }


def grid_fingerprints(params: Mapping[str, Any]) -> list[str]:
    """Fingerprint each variant of an explore task's grid against its own base controls, as the executor does.

    A variant carrying a kernel overlay gets an empty fingerprint: the overlay changes the launch but not the identity,
    so two such variants cannot be told apart from their args and envs.
    """
    out: list[str] = []
    for raw in params.get("grid") or ():
        if not isinstance(raw, Mapping) or not raw.get("name") or raw.get("overlay_pythonpath"):
            out.append("")
            continue
        fields = normalize_proposal(raw)
        out.append(
            effective_fingerprint(
                fields["extra_args"],
                fields["extra_envs"],
                controls=controls_of(fields),
                base=BaseControls.of_params(params),
            )
        )
    return out


def measured_verdict(
    ledger: Any, fingerprint: str, *, variant: Mapping[str, Any], now: MeasuredOn
) -> dict[str, Any] | None:
    """The ledger row that already measured ``variant`` (keyed ``fingerprint``) under ``now``, if any.

    The stack includes the anchor throughput, so any change to the running server since that measurement -- a KEEP, a
    kernel overlay, a re-measured baseline -- makes the variant worth measuring again. The row's own change must match
    too: its key was fingerprinted against the controls its round opened with, which a KEEP in that round can move.
    The KEEP bar decays per macro-cycle, so a REVERT the throughput gate ruled stands only while its gain is still
    below ``now.keep_threshold_pct``; accuracy and tail-latency REVERTs do not depend on the bar.
    """
    tested = (ledger or {}).get("tested") if isinstance(ledger, Mapping) else None
    row = tested.get(fingerprint) if isinstance(tested, Mapping) and fingerprint else None
    if not isinstance(row, Mapping) or str(row.get("outcome") or "") not in _MEASURED_OUTCOMES:
        return None
    if str(row.get("workload_signature") or "") != now.workload_signature:
        return None
    against = row.get("measured_against")
    if not isinstance(against, Mapping) or stack_of_measured(against) != dict(now.stack):
        return None
    if content_fingerprint(row) != content_fingerprint(variant):
        return None
    if _cleared_by_lower_bar(row, now.keep_threshold_pct):
        return None
    return dict(row)


def _cleared_by_lower_bar(row: Mapping[str, Any], keep_threshold_pct: float) -> bool:
    """Whether a throughput-gate REVERT's recorded gain now meets the bar it missed."""
    if row.get("outcome") != "REVERT" or row.get("reason") != GAIN_BELOW_THRESHOLD:
        return False
    gain = row.get("graded_gain_pct")
    return isinstance(gain, (int, float)) and not isinstance(gain, bool) and gain >= keep_threshold_pct


def stack_of_measured(against: Mapping[str, Any]) -> dict[str, Any]:
    """Read a ``measured_against`` record back through :func:`stack_of` so both sides normalise alike."""
    return stack_of(
        {
            "base_tput": against.get("throughput"),
            "base_extra_args": against.get("extra_server_args"),
            "base_extra_envs": against.get("extra_envs"),
            "base_remove_args": against.get("remove_args"),
            "base_unset_envs": against.get("unset_envs"),
            "base_args_mode": against.get("args_mode"),
        }
    )


def explore_grid_already_covered(params: Mapping[str, Any], active: Iterable[Any], ledger: Any) -> PolicyDenied | None:
    """Deny an explore grid every variant of which is already measured, queued or running on this same stack.

    ``active`` is the queued and running tasks. The delegator names its own idempotency key, so two grids with the
    same content under different keys would otherwise both run. A grid with one genuinely new variant is admitted; the
    executor skips the rest.
    """
    keys = _launch_keys(params)
    if not keys or not all(fp for fp, _ in keys):
        return None
    now = MeasuredOn(
        stack_of(params),
        workload_signature(),
        float(params.get("keep_threshold_pct", DEFAULT_KEEP_THRESHOLD_PCT)),
    )
    pending = {
        key
        for task in active
        if task.kind == "explore" and stack_of(task.params or {}) == now.stack
        for key in _launch_keys(task.params or {})
    }
    grid = list(params.get("grid") or ())
    if not all(
        key in pending or measured_verdict(ledger, key[0], variant=raw, now=now) for key, raw in zip(keys, grid)
    ):
        return None
    names = [str(v.get("name") or "") for v in grid]
    return PolicyDenied(
        f"delegate{{action_name='explore'}} adds nothing new: {', '.join(names)} already measured, queued or "
        "running on the current stack",
        rule="explore_grid_already_covered",
        hint="Read the measured verdicts in explore_search, or propose a variant that changes the launch.",
    )


def _launch_keys(params: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each grid variant's stack fingerprint paired with the fingerprint of its own change.

    The stack fingerprint alone can collide: on a replacing base, a variant's own removal and the base's removal fold
    into one identity although the runner applies only the variant's. On the same stack, equal pairs launch alike.
    """
    grid = list(params.get("grid") or ())
    return [(fp, content_fingerprint(raw) if fp else "") for fp, raw in zip(grid_fingerprints(params), grid)]
