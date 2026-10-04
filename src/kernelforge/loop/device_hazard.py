# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A device-contention finding that outlives the iteration that found it."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from kernelforge.llm.process_reaping import device_holders, still_holding_device

log = logging.getLogger(__name__)

# How many iterations one hazard may refuse before the campaign stops.
MAX_BLOCKED_ITERATIONS = 3


@dataclass(frozen=True)
class DeviceHazard:
    """A device the campaign may not measure on, and who is holding it."""

    # The reaper's own words, carried so the refusal can be reported in the terms a reader of ``REVERT_CONTENDED``
    # already knows.
    detail: str = ""
    # pid -> start time of the processes that were holding a device node when the hazard was found.
    holders: dict[int, int] = field(default_factory=dict)
    found_iteration: int = 0
    # How many iterations this hazard has refused, the one that found it included.
    blocked_iterations: int = 0
    # The last iteration it refused, so a re-check within one iteration is idempotent: the loop consults the hazard
    # both before and after its fan-out round, and the second look must not count as a second refusal.
    last_blocked_iteration: int = 0
    # Which holders the most recent re-check still found on the device.
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
    """Where a contention finding waits for the device to become free again."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.path = Path(workspace_dir).resolve() / "forge_experiments" / "device_hazard.json"
        self._hazard = self._load()

    @property
    def live(self) -> DeviceHazard | None:
        """The hazard currently refusing measurements, without re-checking."""
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
        """Record that ``pids`` left the device unsafe to measure on."""
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
        """Rule on whether a recorded hazard still blocks this iteration."""
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
