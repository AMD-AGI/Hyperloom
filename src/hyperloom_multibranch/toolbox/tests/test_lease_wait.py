# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``lease_branch_sandbox(..., wait_sec=)`` on every implementation: one keyword with one meaning, and a toolbox with no
cards to lease refuses with its own refusal whatever the wait."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLeases
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolbox
from hyperloom_multibranch.toolbox.leased_toolbox import LeasedToolbox, LeaseBook
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.tests.helpers import build
from hyperloom_multibranch.toolbox.toolbox import RealToolbox

_LEASES = (
    RealToolbox.lease_branch_sandbox,
    LeasedToolbox.lease_branch_sandbox,
    LocalHostedToolbox.lease_branch_sandbox,
    HostedToolbox.lease_branch_sandbox,
    FakeToolbox.lease_branch_sandbox,
    LeaseBook.lease,
    BranchLeases.lease,
    HostedBranchLeases.lease,
)


@pytest.mark.parametrize("lease", _LEASES, ids=lambda lease: lease.__qualname__)
def test_every_lease_takes_wait_sec_as_a_keyword_that_defaults_to_the_bounded_wait(lease: Callable[..., Any]) -> None:
    parameter = inspect.signature(lease).parameters["wait_sec"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is None


def _real_without_cards(tmp_path: Path) -> RealToolbox:
    return build(tmp_path).toolbox


@pytest.mark.parametrize("make", [_real_without_cards, lambda _tmp: FakeToolbox()], ids=["real", "fake"])
@pytest.mark.parametrize("wait", [None, 0, 5.0])
async def test_a_toolbox_with_no_cards_to_lease_refuses_whatever_the_wait(
    tmp_path: Path, make: Callable[[Path], RealToolbox], wait: float | None
) -> None:
    with pytest.raises(RequestRejected) as refused:
        await make(tmp_path).lease_branch_sandbox("b02", 1, wait_sec=wait)
    assert refused.value.code == "unsupported_kind"
