# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the remaining campaign budget can pay for another round."""

from __future__ import annotations

from dataclasses import dataclass

# What a round's sessions are priced at: the median production session (12.3 min). A round's lanes run side by side,
# so its width does not change what it costs on the clock.
DISPATCH_SESSION_SEC = 738.0

# The floor under the requirement: the p90 production session (34.6 min) less the 15-minute grace a 10.75 h budget
# left before an 11 h external kill.
DISPATCH_FLOOR_SEC = 1176.0

# What the canonical validation and benchmark of a round's candidate are priced at.
MEASUREMENT_SEC = 600.0


@dataclass(frozen=True)
class DispatchAdmission:
    """The decision on starting a round, and every number it was made from."""

    admitted: bool
    remaining_sec: float
    required_sec: float

    def summary(self) -> str:
        """One line naming the decision's cost breakdown, in minutes."""
        return (
            f"{self.required_sec / 60:.0f} min needed (session {DISPATCH_SESSION_SEC / 60:.0f}, measurement "
            f"{MEASUREMENT_SEC / 60:.0f}); {self.remaining_sec / 60:.0f} min remain"
        )


def admit_dispatch(*, remaining_sec: float) -> DispatchAdmission:
    """Decide whether a round may start its sessions, which the loop cannot interrupt once they run."""
    required_sec = max(DISPATCH_FLOOR_SEC, DISPATCH_SESSION_SEC + MEASUREMENT_SEC)
    return DispatchAdmission(
        admitted=remaining_sec >= required_sec, remaining_sec=remaining_sec, required_sec=required_sec
    )
