# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the fork admission answer the scheduling side gives the rule gate."""

from __future__ import annotations

import pytest

from hyperloom_multibranch.contracts.scheduling import AdmissionStatus, ForkAdmission


def test_only_an_accepted_fork_names_its_branch() -> None:
    assert ForkAdmission(AdmissionStatus.ACCEPTED, "slot free", "b5").branch == "b5"
    assert ForkAdmission(AdmissionStatus.REJECTED, "at the branch cap").branch == ""
    with pytest.raises(ValueError):
        ForkAdmission(AdmissionStatus.ACCEPTED, "slot free")
    with pytest.raises(ValueError):
        ForkAdmission(AdmissionStatus.PENDING, "at the branch cap", "b5")
