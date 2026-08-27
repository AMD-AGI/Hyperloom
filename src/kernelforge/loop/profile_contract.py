# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared contract for driver-owned kernel profiling."""

import subprocess


PROFILE_RUN_FLAG = "--profile-run"

# Optional companion to PROFILE_RUN_FLAG, narrowing the profile to one case.
# Only meaningful for drivers whose suite runs the same kernel at several
# shapes -- a collective sweep, for instance, where every case dispatches the
# same all-reduce. Profiling all of them at once averages distinct shapes into
# one set of counters, which is not a valid profile of any of them. Drivers
# with a single case have nothing to narrow, so this stays optional rather than
# becoming another argument every driver must implement.
PROFILE_CASE_FLAG = "--profile-case"


def driver_supports_profile_case(driver_script: str, timeout_sec: float = 30.0) -> bool:
    """Whether the driver accepts PROFILE_CASE_FLAG.

    Asks the driver rather than assuming, so drivers written before the flag
    existed keep working unchanged. Any failure to ask is read as "no": passing
    an unknown argument would make argparse exit non-zero and cost the profile
    entirely, while skipping it only leaves the profile as wide as it was.
    """
    try:
        proc = subprocess.run(
            ["python3", driver_script, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception:  # noqa: BLE001 - probing must never break profiling
        return False
    return PROFILE_CASE_FLAG in (proc.stdout or "") + (proc.stderr or "")


__all__ = [
    "PROFILE_RUN_FLAG",
    "PROFILE_CASE_FLAG",
    "driver_supports_profile_case",
]
