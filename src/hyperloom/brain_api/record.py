# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A brain call as it is recorded: one JSON line holding the state, the ask and the answer.

The record is the replay unit. An evaluator reads a session's calls back, hands the same state and ask to another
brain, and scores both answers against a dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from hyperloom.brain_api.calls import CallSite, Proposal, ProposeAsk, Review, ReviewAsk
from hyperloom.brain_api.state import SearchState
from hyperloom.brain_api.wire import WireError, from_wire, json_value, to_wire

#: Call site -> (ask type, answer type).
CALL_TYPES: Mapping[CallSite, tuple[type, type]] = {
    CallSite.PROPOSE: (ProposeAsk, Proposal),
    CallSite.REVIEW: (ReviewAsk, Review),
}
#: Call site -> the eval point of the brain method the body calls there.
MAIN_POINTS: Mapping[CallSite, str] = {CallSite.PROPOSE: "main.propose", CallSite.REVIEW: "main.review"}

_POINT_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def valid_point_name(name: str) -> bool:
    """Whether ``name`` is a dotted lower_snake_case eval point name with at least two parts."""
    return bool(_POINT_NAME.match(name))


@dataclass(frozen=True)
class BrainCall:
    """One call of an eval point: what it was given, what it answered (or why it failed), when and for how long.

    ``point`` is the eval point called; it defaults to the main point of ``call_site``.
    """

    call_site: CallSite
    contract_version: str
    brain: str
    state: SearchState
    ask: ProposeAsk | ReviewAsk
    answer: Proposal | Review | None = None
    error: str = ""
    point: str = ""
    point_version: str = "0.1"
    started_unix: float = 0.0
    duration_sec: float = 0.0

    def __post_init__(self) -> None:
        if not self.point:
            object.__setattr__(self, "point", MAIN_POINTS[self.call_site])
        if not valid_point_name(self.point):
            raise WireError(f"eval point name {self.point!r} is not dotted lower_snake_case")
        ask_type, answer_type = CALL_TYPES[self.call_site]
        if not isinstance(self.ask, ask_type):
            raise WireError(f"a {self.call_site.value} call takes a {ask_type.__name__}")
        if self.answer is not None and not isinstance(self.answer, answer_type):
            raise WireError(f"a {self.call_site.value} call answers a {answer_type.__name__}")
        if (self.answer is None) == (not self.error):
            raise WireError("a call holds exactly one of an answer and an error")

    def to_wire(self) -> dict[str, Any]:
        """The call as a JSON object."""
        return {
            "call_site": self.call_site.value,
            "contract_version": self.contract_version,
            "brain": self.brain,
            "state": to_wire(self.state),
            "ask": to_wire(self.ask),
            "answer": None if self.answer is None else to_wire(self.answer),
            "error": self.error,
            "point": self.point,
            "point_version": self.point_version,
            "started_unix": self.started_unix,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> BrainCall:
        """The inverse of :meth:`to_wire`."""
        keys = {
            "call_site",
            "contract_version",
            "brain",
            "state",
            "ask",
            "answer",
            "error",
            "point",
            "point_version",
            "started_unix",
            "duration_sec",
        }
        if set(data) != keys:
            raise WireError(f"BrainCall keys {sorted(data)} are not {sorted(keys)}")
        try:
            site = CallSite(data["call_site"])
        except ValueError:
            raise WireError(f"unknown call_site {data['call_site']!r}") from None
        ask_type, answer_type = CALL_TYPES[site]
        answer = data["answer"]
        return cls(
            call_site=site,
            contract_version=str(data["contract_version"]),
            brain=str(data["brain"]),
            state=from_wire(SearchState, data["state"]),
            ask=from_wire(ask_type, data["ask"]),
            answer=None if answer is None else from_wire(answer_type, answer),
            error=str(data["error"]),
            point=str(data["point"]),
            point_version=str(data["point_version"]),
            started_unix=float(data["started_unix"]),
            duration_sec=float(data["duration_sec"]),
        )


def append_call(path: Path, call: BrainCall) -> None:
    """Append ``call`` to the JSON-lines file at ``path``."""
    line = json.dumps(json_value(call.to_wire(), "BrainCall"), allow_nan=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_calls(path: Path) -> Iterator[BrainCall]:
    """Every call recorded at ``path``, in order; raises :class:`WireError` naming the first bad line."""
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield BrainCall.from_wire(json.loads(line))
            except (ValueError, WireError) as exc:
                raise WireError(f"{path}:{number}: {exc}") from None
