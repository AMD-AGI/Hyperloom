# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A device-contention finding that outlives the iteration that found it.

The reaper answers "is anything of ours still running in this directory" and,
where the answer is yes, the iteration that asked skips its canonical
measurement. That skip was the whole response, and it was scoped to one
iteration -- which is exactly as long as the hazard is NOT scoped to. By the
ownership model the reaper works from, what it could not clear is quite likely
nothing the campaign may kill at all: a parallel campaign, a human's shell, a
previous run's leftovers. None of those leaves because an iteration ended, so
the next measurement ran anyway and was contaminated in the way the skip existed
to prevent.

So the finding is recorded here and re-checked before every measurement, and
two things decide when it stops blocking:

* **What clears it is the device, not the clock.** The re-check asks
  :func:`~kernelforge.llm.process_reaping.still_holding_device` whether the processes
  that made the finding still have a device node open, keyed on the identity
  they were recorded under so a recycled pid cannot answer for them. That is a
  narrower question than "did the reap succeed", and deliberately so: re-running
  the reaper would only re-establish what it may kill, which is not what is
  blocking the measurement.
* **A hazard nothing can clear ends the campaign.** A foreign process may hold
  the device forever. A loop that retries until its budget runs out has spent a
  whole run producing nothing while reporting nothing wrong, which is no better
  than the bad measurement -- so after :data:`MAX_BLOCKED_ITERATIONS` refusals
  the run stops under a termination reason of its own, loudly and terminally.

The record is a small JSON file beside the loop's other durable control state,
written the way the lane queue is: what it holds is worthless to a process that
did not survive to read it, and a campaign ending between iterations is the
ordinary case, not the crash case.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from kernelforge.llm.process_reaping import device_holders, still_holding_device

log = logging.getLogger(__name__)

# How many iterations one hazard may refuse before the campaign stops. Three
# rather than one because a leftover benchmark of our own that survived SIGKILL
# does normally finish and exit, and rather than "many" because every refused
# iteration is a whole iteration of the budget spent measuring nothing.
MAX_BLOCKED_ITERATIONS = 3


@dataclass(frozen=True)
class DeviceHazard:
    """A device the campaign may not measure on, and who is holding it."""

    # The reaper's own words, carried so the refusal can be reported in the
    # terms a reader of ``REVERT_CONTENDED`` already knows.
    detail: str = ""
    # pid -> start time of the processes that were holding a device node when
    # the hazard was found. The pair is the identity: a bare pid stops naming
    # the same process the moment it exits.
    holders: dict[int, int] = field(default_factory=dict)
    found_iteration: int = 0
    # How many iterations this hazard has refused, the one that found it
    # included. Compared against MAX_BLOCKED_ITERATIONS.
    blocked_iterations: int = 0
    # The last iteration it refused, so a re-check within one iteration is
    # idempotent: the loop consults the hazard both before and after its
    # fan-out round, and the second look must not count as a second refusal.
    last_blocked_iteration: int = 0
    # Which holders the most recent re-check still found on the device. Kept
    # beside ``detail`` rather than folded into it, so a hazard that refuses
    # several iterations does not accumulate a sentence per refusal.
    still_held_by: tuple[int, ...] = ()

    @property
    def exhausted(self) -> bool:
        """Whether this hazard has blocked as long as the campaign allows."""
        return self.blocked_iterations >= MAX_BLOCKED_ITERATIONS

    def describe(self) -> str:
        """One line naming what is holding the device and what found it."""
        if not self.still_held_by:
            return self.detail
        return f"{self.detail}; pid(s) {list(self.still_held_by)} still hold a device node"

    def to_dict(self) -> dict:
        return {
            "detail": self.detail,
            "holders": {str(pid): start for pid, start in self.holders.items()},
            "found_iteration": self.found_iteration,
            "blocked_iterations": self.blocked_iterations,
            "last_blocked_iteration": self.last_blocked_iteration,
            "still_held_by": list(self.still_held_by),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "DeviceHazard":
        return cls(
            detail=str(record["detail"]),
            holders={int(pid): int(start) for pid, start in dict(record["holders"]).items()},
            found_iteration=int(record["found_iteration"]),
            blocked_iterations=int(record["blocked_iterations"]),
            last_blocked_iteration=int(record["last_blocked_iteration"]),
            still_held_by=tuple(int(pid) for pid in record["still_held_by"]),
        )


class DeviceHazardLog:
    """Where a contention finding waits for the device to become free again.

    One file per campaign, read once at construction so a resumed process
    inherits what the previous one was refused by. A hazard that cannot be
    written costs this campaign the ability to carry the refusal across a
    restart and nothing else: the in-memory record still blocks every
    measurement this process would have taken.
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir).resolve() / "forge_experiments" / "device_hazard.json"
        self._hazard = self._load()

    @property
    def live(self) -> DeviceHazard | None:
        """The hazard currently refusing measurements, without re-checking.

        Read after :meth:`recheck` has already ruled on this iteration, and
        after anything that may have recorded a new one.
        """
        return self._hazard

    def _load(self) -> DeviceHazard | None:
        try:
            return DeviceHazard.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("no readable device hazard at %s", self.path)
            return None

    def _save(self, hazard: DeviceHazard) -> None:
        from kernelforge.loop.recovery import atomic_write_json

        self._hazard = hazard
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, hazard.to_dict())
        except OSError as error:
            log.warning(
                "device hazard not durable (%s); a restart will measure without knowing the device was held",
                error,
            )

    def clear(self) -> None:
        """Forget the hazard, in memory and on disk."""
        self._hazard = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not remove %s", self.path, exc_info=True)

    def record(self, *, iteration: int, detail: str, pids: Iterable[int]) -> DeviceHazard:
        """Record that ``pids`` left the device unsafe to measure on.

        Which of them actually hold a device node is resolved now, while they
        are still identifiable, because that is the question every later
        re-check asks. A finding with no device holder in it is still a refusal
        for the iteration that made it -- the reaper said the directory could
        not be cleared -- but it has nothing for a later iteration to wait on,
        so it clears at the next re-check rather than blocking on a process that
        demonstrably is not on the device.
        """
        holders = device_holders(pids)
        hazard = DeviceHazard(
            detail=detail,
            holders=holders,
            found_iteration=iteration,
            blocked_iterations=1,
            last_blocked_iteration=iteration,
            still_held_by=tuple(sorted(holders)),
        )
        self._save(hazard)
        return hazard

    def recheck(self, iteration: int) -> DeviceHazard | None:
        """Rule on whether a recorded hazard still blocks this iteration.

        Called once per iteration, before anything is spent on it. Answers the
        live hazard, or None once the device is free again. Idempotent within
        one iteration, so a caller that consults it twice does not count one
        refusal as two.
        """
        hazard = self._hazard
        if hazard is None:
            return None
        if iteration in (hazard.found_iteration, hazard.last_blocked_iteration):
            return hazard
        still = still_holding_device(hazard.holders)
        if not still:
            log.info("device hazard cleared: %s", hazard.detail)
            self.clear()
            return None
        blocked = replace(
            hazard,
            blocked_iterations=hazard.blocked_iterations + 1,
            last_blocked_iteration=iteration,
            still_held_by=still,
        )
        self._save(blocked)
        return blocked


__all__ = [
    "MAX_BLOCKED_ITERATIONS",
    "DeviceHazard",
    "DeviceHazardLog",
]
