# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A later leg of a hosted run may be another account of the run's group: it builds the real peers of the session an
earlier leg of one account left on shared storage, which reads back what that leg froze in ``MB-toolbox``."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import hyperloom_multibranch
from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.runner import suite_publish
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.local_peers import local_peers, workload_environment
from hyperloom_multibranch.scheduling.peers import PeerSetup
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.tests.test_local_peers import AGENTX, HOST, WORKLOAD, _Publishes
from hyperloom_multibranch.specialist.model_sessions import ModelSessions
from hyperloom_multibranch.tests.fakes import RefusingCritic

MODEL = "meta-llama/Llama-3-8B"


def _workload(agentx: bool) -> object:
    return replace(AGENTX, model=MODEL) if agentx else WORKLOAD


def build_peers(workspace: str, root: str | None, agentx: bool) -> dict[str, str]:
    """Build and close the real peers of the session in ``root`` as the account this process is, as a leg of a run
    does; a new session in ``workspace`` that froze its base launch when ``root`` is ``None``. Its directory."""

    workload = _workload(agentx)
    for name in (*workload_environment(workload), BENCHMARK_CONFIG_ENV, "GPU_TYPE", "HYPERLOOM_KB_URL"):
        os.environ.pop(name, None)
    os.environ["HL_MODEL_BASE"] = str(Path(workspace) / "models")
    suite_publish.publish_own_inference_suite = _Publishes()  # no staging is fetched here
    clock = Clock()
    if root is None:
        layout = create_session(Path(workspace), workload.model_basename, clock)
        freeze_base(layout.toolbox, base_launch())
    else:
        layout = SessionLayout(Path(root))
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock), clock)
    setup = PeerSetup(
        layout,
        workload,
        desk,
        ProcessRegistry(layout.pids, clock),
        LocalBlobStore(layout.blobs),
        RefusingCritic(),
        {"geak": layout.reference / "geak"},
        0,
        (),
        "",
        agent_launch="/launches/arbor.json",
    )

    def sessions(_wiring: object) -> ModelSessions:
        def no_specialist(*_: object) -> object:
            raise AssertionError("no specialist runs here")

        return ModelSessions(no_specialist, no_specialist)

    async def build() -> None:
        peers = local_peers(HOST, sessions, setup)
        peers.mcp.stop()
        await peers.toolbox.close()

    asyncio.run(build())
    return {"root": str(layout.root)}


def _as_account(accounts: Accounts, uid: int, call: str) -> dict[str, str]:
    source = Path(hyperloom_multibranch.__file__).parents[1]
    code = (
        f"import json, sys\nsys.path.insert(0, {str(source)!r})\n"
        "from hyperloom_multibranch.scheduling.tests import test_peers_other_account as here\n"
        f"print(json.dumps(here.{call}))"
    )
    return json.loads(accounts.run(uid, code).splitlines()[-1])


@pytest.mark.parametrize("agentx", [False, True], ids=["inferencex", "agentx"])
def test_another_account_of_the_group_builds_the_peers_of_a_session_one_account_left(agentx: bool) -> None:
    with other_accounts() as accounts:
        source = Path(hyperloom_multibranch.__file__).parents[1]
        if any(not directory.stat().st_mode & stat.S_IXOTH for directory in (source, *source.parents)):
            pytest.skip(f"another account cannot reach this checkout's {source}")
        if accounts.python != sys.executable:
            pytest.skip(f"another account cannot run {sys.executable}, whose environment holds the project's packages")
        workspace = accounts.shared_dir("workspace")
        (workspace / "models" / Path(MODEL).name).mkdir(parents=True)
        left = _as_account(accounts, accounts.first, f"build_peers({str(workspace)!r}, None, {agentx})")
        toolbox = SessionLayout(Path(left["root"])).toolbox
        state = SharedState.state_path(toolbox)
        assert (state.stat().st_uid, stat.S_IMODE(state.stat().st_mode)) == (accounts.first, 0o664)

        _as_account(accounts, accounts.second, f"build_peers({str(workspace)!r}, {left['root']!r}, {agentx})")

        unreadable = [
            path
            for path in toolbox.rglob("*")
            if path.is_file() and not path.is_symlink() and not stat.S_IMODE(path.stat().st_mode) & stat.S_IRGRP
        ]
        assert unreadable == [], "every file the first account left in MB-toolbox is the group's to read"
