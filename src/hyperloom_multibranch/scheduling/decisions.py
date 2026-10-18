# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision logs: one JSON line per decision, with its reason, so a post-mortem can replay why the session went the
way it did.

The scheduler's log, ``MB-scheduling/MB-decisions.jsonl``, gets every phase change, branch opened, forked, repaired,
re-sized or stopped, judge call, event the judge is shown, profile of the integration head, final choice and session
stop. The integrator keeps its merges in a log of its own, ``MB-integration/MB-merges.jsonl``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.scheduling.clock import Clock


class DecisionKind(str, Enum):
    PHASE = "phase"
    PRELUDE = "prelude"
    OPEN = "open"
    FORK = "fork"
    REPAIR = "repair"
    SET_K = "set_k"
    STOP = "stop"
    JUDGE = "judge"
    JUDGE_FAILED = "judge_failed"
    MERGE = "merge"
    EVENT = "event"
    PROFILE = "profile"
    FINAL = "final"
    SESSION_STOP = "session_stop"


@dataclass(frozen=True)
class Decision:
    """``branch`` is empty for a session-wide decision; ``detail`` holds JSON-encodable facts behind it."""

    kind: DecisionKind
    reason: str
    branch: str = ""
    detail: Mapping[str, object] = field(default_factory=dict)


class DecisionLog:
    def __init__(self, path: Path, clock: Clock) -> None:
        self._path = path
        self._clock = clock

    def record(self, decision: Decision) -> None:
        append_jsonl(
            self._path,
            {
                "at": self._clock.now(),
                "kind": decision.kind.value,
                "reason": decision.reason,
                "branch": decision.branch,
                "detail": encode(dict(decision.detail)),
            },
            make_parents=True,
        )


__all__ = ["Decision", "DecisionKind", "DecisionLog"]
