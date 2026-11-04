# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX session's planned points carry the launch overrides, the server's step timeout included."""

from __future__ import annotations

from pathlib import Path

import yaml

from hyperloom.measure.tests._recipes import recipe_with
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.tests import test_planner
from hyperloom_multibranch.toolbox.tests.helpers import FakeRepo, build, req

#: The planner tests' process environment, cleaned the same way for this test.
clean_process_env = test_planner.clean_process_env


async def test_an_agentx_session_plans_points_through_the_agentx_launch_overrides(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        repo=FakeRepo(recipe_with()),
        planner=test_planner._agentx_planner(tmp_path),
        workload_kind=WorkloadKind.AGENTX,
    )

    await rig.toolbox.submit_test(req(server_args="--max-running-requests 77", concurrency=(8, 64)))
    await rig.toolbox.drain()

    points = [yaml.safe_load(path.read_text()) for path in rig.executor.jobs[0].environment.points]
    assert [point["benchmark"]["envs"]["CONC"] for point in points] == [8, 64]
    for point in points:
        bench = point["benchmark"]
        assert bench["agentx"]["launch_overrides"]["append_args"] == [
            "--max-running-requests",
            "77",
            "--watchdog-timeout",
            "1800",
        ]
        assert bench["envs"]["RUN_EVAL"] == "false"
        assert not {"benchmark_script", "workload_spec"} & bench.keys()
        assert not {"NUM_PROMPTS", "NUM_WARMUPS", "EXTRA_SGLANG_ARGS"} & bench["envs"].keys()
