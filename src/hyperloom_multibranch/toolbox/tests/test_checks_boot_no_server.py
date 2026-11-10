# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A merge test and a FINAL test each boot exactly one server: the one their rounds run on, which then answers the check.

Acceptance plans each test's short-prompt check, and the real local backend runs the test with ``run_grid`` replaced by a
stand-in that brings a real HTTP server up on the port the round's lifecycle names whenever a round holds a server and
none is up. Every such start is a server boot; the check may add none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.contracts.measurement import Purpose
from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.short_prompts.outputs import OK, read_outputs
from hyperloom.measure.short_prompts.tests.support import FakeServer, free_port, staged_plan
from hyperloom.measure.socket import TestCall
from hyperloom.orchestrator.actions.executors import VariantResult
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup
from hyperloom_multibranch.toolbox.tests.test_short_prompts import _accepted, _prepare, _purposed


class _Booting:
    """``run_grid`` and the lifecycle's teardown: a round that holds a server starts one when none is up."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds: list[tuple[str, bool]] = []
        self.boots: list[str] = []
        self.teardowns = 0
        self._live: FakeServer | None = None

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        name, launch = kwargs["grid"][0].name, kwargs["launch"]
        self.rounds.append((name, launch.lifecycle_boot_only))
        if launch.server_lifecycle and self._live is None:
            self._live = FakeServer(port=launch.server_lifecycle["port"]).__enter__()
            self.boots.append(name)
        workspace = self.root / "ws" / name
        workspace.mkdir(parents=True, exist_ok=True)
        return [VariantResult(name, "", {}, "succeeded", returncode=0, workspace=str(workspace))]

    def teardown(self, **kwargs: Any) -> None:
        self.teardowns += 1
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None


def _booting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Booting:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
    booting = _Booting(tmp_path)
    port = free_port()
    monkeypatch.setattr(
        local, "resolve_lifecycle_params", lambda path: {"eligible": True, "port": port, "framework": "sglang"}
    )
    monkeypatch.setattr(local, "teardown_lifecycle_server", booting.teardown)
    monkeypatch.setattr(local, "run_grid", booting)
    return booting


@pytest.mark.parametrize(
    ("purpose", "fields"),
    [(Purpose.INTEGRATION, {"consistency_check": True}), (Purpose.FINAL, {})],
    ids=["merge", "final"],
)
def test_a_merge_test_and_a_final_test_each_boot_exactly_one_server_which_answers_the_check(
    tmp_path, monkeypatch, purpose, fields
):
    cache = tmp_path / "cache"
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(cache))
    staged = staged_plan(cache)
    setup = ShortPromptSetup(staged.prompt_set, staged.prompts_sha256)
    plan = _prepare(
        _purposed(purpose, **fields), "", setup=setup, accepted=_accepted(tmp_path)
    ).environment.short_prompts
    booting = _booting(tmp_path, monkeypatch)
    point = tmp_path / "point.yaml"
    point.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}}})
    )
    call = TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=900,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t-1",
        short_prompts=plan,
    )

    outcome = LocalMeasurementBackend().run_test(call)

    assert booting.boots == ["test"] and booting.teardowns == 1, "one server, the test's own, torn down once"
    assert booting.rounds == [("test", False)], "no boot-only round for the check"
    assert [(e.index, e.status) for e in read_outputs(tmp_path / "out").processes] == [(0, OK)]
    assert outcome.returncode == 0
