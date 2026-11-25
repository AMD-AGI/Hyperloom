# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded test of a server built from its checkout carries the last lines its server logs time requests with.

The executor is the real one over the hosted backend and the local sandbox provider; the server log the sandbox's rounds
leave in the landed output holds llama-server's timing lines. What is read from the result is what the blackboard stored
and what ``test_result`` shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Validity
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import CompiledEnvironment
from hyperloom.measure.tests.hosted_fixtures import point_yaml
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import (  # noqa: F401  (rig is a fixture)
    OK,
    PointPlanner,
    ScriptedBackend,
    rig,
    stored_and_logged,
)
from hyperloom_multibranch.toolbox.timing_lines import read_test_timing_lines

ROUND_LOG = "variant_00_c1/benchmark_llamacpp_0/server.log"
_SLOT = "6.23.810.779 I slot print_timing: id  0 | task 4161 |"
LOG = "\n".join(
    [
        f"{_SLOT} prompt eval time =     562.90 ms /   220 tokens (    2.56 ms per token,   390.83 tokens per second)",
        f"{_SLOT}        eval time =   12621.18 ms /   512 tokens (   24.70 ms per token,    40.49 tokens per second)",
        f"{_SLOT} draft acceptance = 0.47872 (  360 accepted /   752 generated), mean len =  3.38",
    ]
)
EVAL = (
    "6.23.810.779 I slot print_timing: id 0 | task 4161 | eval time = 12621.18 ms / 512 tokens ( 24.70 ms per token, "
    "40.49 tokens per second)"
)
DRAFT = (
    "6.23.810.779 I slot print_timing: id 0 | task 4161 | draft acceptance = 0.47872 ( 360 accepted / 752 generated), "
    "mean len = 3.38"
)


def _point(tmp_path: Path, framework: str) -> Path:
    return _named(point_yaml(tmp_path, "planned.yaml"), framework)


def _named(point: Path, framework: str) -> Path:
    point.write_text(point.read_text().replace("benchmark:\n", f"benchmark:\n  framework: {framework}\n"))
    return point


@dataclass
class FrameworkBackend(ScriptedBackend):
    """The scripted backend, whose point names ``framework`` as a generated point does: the script rewrites the point
    before each attempt, and the result is read after it."""

    framework: str = ""

    def run_test(self, call):  # type: ignore[no-untyped-def]
        outcome = super().run_test(call)
        _named(self.point, self.framework)
        return outcome


async def _run(tmp_path: Path, r, mode: str):
    point = _point(tmp_path, "llamacpp")
    scripted = FrameworkBackend(r.backend, point, [mode], stamps={ROUND_LOG: LOG}, framework="llamacpp")
    planner = PointPlanner(point, CompiledEnvironment(Recipe(IMAGE), IMAGE, (), (), {}, tmp_path / "blobs"))
    return await stored_and_logged(tmp_path, scripted, planner)


async def test_a_test_of_a_checkout_built_server_carries_the_last_timing_lines_of_its_logs(tmp_path: Path, rig) -> None:
    stored, outcome = await _run(tmp_path, rig(), OK)

    assert outcome.validity is Validity.SUCCEEDED
    for result in (stored, outcome):
        assert result.timing_lines == (f"{ROUND_LOG}: {EVAL}", f"{ROUND_LOG}: {DRAFT}")
    assert jsonable(stored)["timing_lines"] == [f"{ROUND_LOG}: {EVAL}", f"{ROUND_LOG}: {DRAFT}"], "test_result shows"


def test_each_log_s_lines_are_led_by_its_path_and_a_line_another_log_copied_is_shown_once(tmp_path: Path) -> None:
    output = tmp_path / "out"
    later = LOG.replace("task 4161", "task 4200")
    logs = {"round_1/server.log": LOG, "round_2/server.log": later, "server/server.log": "\n".join([LOG, later])}
    for name, text in logs.items():
        (output / name).parent.mkdir(parents=True)
        (output / name).write_text(text + "\n", encoding="utf-8")
    (output / "round_3" / "server.log").mkdir(parents=True)  # a path that matches and cannot be read

    lines = read_test_timing_lines((_point(tmp_path, "llamacpp"),), output)

    assert lines == (
        f"round_1/server.log: {EVAL}",
        f"round_1/server.log: {DRAFT}",
        f"round_2/server.log: {EVAL.replace('4161', '4200')}",
        f"round_2/server.log: {DRAFT.replace('4161', '4200')}",
        "round_3/server.log: not read (IsADirectoryError)",
    )
    assert read_test_timing_lines((), output) == ()


@pytest.mark.parametrize("point", ["missing.yaml", "benchmark: [1]\n", "benchmark: {\n"], ids=["gone", "shape", "yaml"])
def test_a_point_that_cannot_be_read_gives_no_lines(tmp_path: Path, point: str) -> None:
    path = tmp_path / "point.yaml"
    if point != "missing.yaml":
        path.write_text(point, encoding="utf-8")

    assert read_test_timing_lines((path,), tmp_path) == ()
