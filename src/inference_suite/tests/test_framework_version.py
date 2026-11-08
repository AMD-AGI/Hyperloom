# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The D6 check: the installed framework's release reaches the recipe's, and an unreadable one is refused."""

from __future__ import annotations

import pytest

from inference_suite.contracts.launch import VersionCheck
from inference_suite.contracts.report import FailedStep
from inference_suite.framework_version import VersionRefusal, check


@pytest.mark.parametrize(
    "installed", ["0.5.23", "0.5.23.dev4", "0.5.23rc1", "0.5.23.post1", "0.5.23+rocm700", "0.5.24", "0.6", "1.0.0"]
)
def test_an_install_of_the_recipe_release_or_later_passes(installed: str) -> None:
    assert check(installed, "0.5.23") is VersionCheck.PASSED


def test_trailing_zero_release_segments_compare_equal() -> None:
    assert check("0.5", "0.5.0") is VersionCheck.PASSED
    assert check("0.5.0.0", "0.5") is VersionCheck.PASSED


@pytest.mark.parametrize("installed", ["0.5.22", "0.5.22.post3", "0.4.99", "0.5.22+rocm700"])
def test_an_older_release_is_refused_as_too_old(installed: str) -> None:
    verdict = check(installed, "0.5.23")
    assert isinstance(verdict, VersionRefusal)
    assert verdict.step is FailedStep.FRAMEWORK_VERSION_TOO_OLD
    assert installed in verdict.detail and "0.5.23" in verdict.detail


@pytest.mark.parametrize("installed", [None, "unknown", ""])
def test_an_unreadable_install_version_is_refused_as_unknown(installed: str | None) -> None:
    verdict = check(installed, "0.5.23")
    assert isinstance(verdict, VersionRefusal)
    assert verdict.step is FailedStep.FRAMEWORK_VERSION_UNKNOWN


@pytest.mark.parametrize("installed", [None, "0.1"])
def test_a_base_with_no_recipe_version_is_not_checked(installed: str | None) -> None:
    assert check(installed, None) is VersionCheck.NOT_APPLICABLE
