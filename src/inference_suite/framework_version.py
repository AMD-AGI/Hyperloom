# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hold the installed framework to the recipe's version (D6).

Only the release segment is compared, so ``0.5.23.dev4`` and ``0.5.23+rocm700`` both reach ``0.5.23``: a nightly
and the release of one version number are not told apart. An installed version that cannot be read is refused,
never let through.
"""

from __future__ import annotations

from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

from inference_suite.contracts.launch import VersionCheck
from inference_suite.contracts.report import FailedStep


@dataclass(frozen=True)
class VersionRefusal:
    """Why the installed framework may not run the recipe's launch."""

    step: FailedStep
    detail: str


def _release(version: str) -> tuple[int, ...]:
    """The release segment of ``version`` without trailing zeros, so ``0.5`` and ``0.5.0`` compare equal."""
    release = list(Version(version).release)
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    return tuple(release)


def check(installed: str | None, recipe: str | None) -> VersionCheck | VersionRefusal:
    """Whether the ``installed`` framework version reaches the ``recipe`` one; ``recipe`` is ``None`` when the base
    names no recipe image, and ``installed`` is ``None`` when the install carries no version."""
    if recipe is None:
        return VersionCheck.NOT_APPLICABLE
    try:
        held = _release(installed) if installed is not None else None
    except InvalidVersion:
        held = None
    if held is None:
        detail = f"the installed framework version {installed!r} cannot be read, so it cannot be held to {recipe}"
        return VersionRefusal(FailedStep.FRAMEWORK_VERSION_UNKNOWN, detail)
    if held < _release(recipe):
        detail = f"the installed framework version {installed} is older than the recipe's {recipe}"
        return VersionRefusal(FailedStep.FRAMEWORK_VERSION_TOO_OLD, detail)
    return VersionCheck.PASSED


__all__ = ["VersionRefusal", "check"]
