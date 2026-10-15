# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A queue policy that lists the branches allowed to submit: a stopped branch's late request is refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import build, req


async def test_a_branch_taken_out_of_the_policy_is_refused_and_nothing_is_queued(tmp_path: Path) -> None:
    listed = QueuePolicy(global_concurrency=9, branch_k={"b1": 1, "b2": 1}, listed_branches_only=True)
    rig = build(tmp_path, policy=listed)
    await rig.toolbox.submit_test(req("b1"))
    await rig.toolbox.set_policy(QueuePolicy(global_concurrency=9, branch_k={"b2": 1}, listed_branches_only=True))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req("b1", server_args="--late"))
    assert refused.value.code == "branch_stopped"
    state = await rig.toolbox.queue_state()
    assert state.queued_by_branch.get("b1", 0) + state.running_by_branch.get("b1", 0) <= 1
    await rig.toolbox.submit_test(req("b2"))
