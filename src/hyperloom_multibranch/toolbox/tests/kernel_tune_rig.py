# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tests stand on: a landed test with a server log, a stand-in for the Forge command, and the
real local toolbox (pool, leases, tool jobs, blackboard) built over them.

The server log, its launch line and hit log, the tuned rows and the landed test are :mod:`.kernel_tune_data`'s, which
the tool's own reader tests use without the toolbox; this module re-exports them for the tests that run the tool.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import runner_template, spec_context
from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.orchestrator.actions.executors import RoundWatch
from hyperloom.sandbox.card_occupancy import CardGuard, CardReading, read_holders
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox, LocalOptions, LocalSession
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req, lineage_blackboard
from hyperloom_multibranch.toolbox.tests.kernel_rows import HEADER, install
from hyperloom_multibranch.toolbox.tests.kernel_tune_data import (
    ENV,
    GRAPH_SIZES,
    HIT_LOG,
    HIT_SHAPES,
    LOG_SHAPES,
    REL_NORM_HEADER,
    SERVER_ARGS,
    SERVER_LOG,
    TABLE,
    TEST_ID,
    TUNER,
    forge_dropped,
    landed_test,
    profile_row,
    table_text,
    tuned_rows,
)
from hyperloom_multibranch.toolbox.tests.test_card_placement import _PointPerBranch
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import kernel_tune

SECRET_ENV = "HF_TOKEN"


async def record_test(
    blackboard: LocalBlackboard, branch: str = "b1", output_dir: str = f"tests/{TEST_ID}/attempt-1"
) -> str:
    """The blackboard's record of a test of ``branch`` whose output landed in ``output_dir`` of the session."""
    await blackboard.write_request(
        replace(req(branch, purpose=Purpose.EXPLORE, kind=TestKind.TEST), request_id=TEST_ID)
    )
    await blackboard.write_result(TestResult(request_id=TEST_ID, validity=Validity.SUCCEEDED, output_dir=output_dir))
    return TEST_ID


def flags(cmd: Sequence[str]) -> dict[str, str]:
    """The ``--flag value`` pairs of a Forge command line; a flag without a value reads ``""``."""
    found: dict[str, str] = {}
    for index, word in enumerate(cmd):
        if word.startswith("--"):
            follows = cmd[index + 1] if index + 1 < len(cmd) else "--"
            found[word] = "" if follows.startswith("--") else follows
    return found


@dataclass
class FakeForge:
    """Answers a Forge command as the tuner does: its ``result.json`` and the table it names under ``tuners/<tuner>/``.

    ``rows`` are the lines of the table; ``dropped`` the rows Forge says it removed as wrong, and ``profile`` the lines of
    the profile of every candidate the tuner measured (written beside the table, as the tuner does, under ``profile_file``
    when the tuner names it its own way); ``hold`` makes the run wait,
    as a tuner on a card does, until ``release`` is set or the job's cancel scope fires (which ends a real child's process
    group); ``fail`` makes it leave no result.
    """

    rows: Sequence[str] = field(default_factory=tuned_rows)
    header: str = HEADER
    dropped: Sequence[Mapping[str, Any]] = ()
    profile: Sequence[str] = ()
    profile_file: str = ""
    hold: bool = False
    fail: bool = False
    release: threading.Event = field(default_factory=threading.Event)
    calls: list[tuple[list[str], dict[str, str]]] = field(default_factory=list)
    demands: list[dict[str, Any]] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)

    def __call__(self, cmd: list[str], *, env: dict[str, str], watch: RoundWatch) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(cmd), dict(env)))
        wanted = flags(cmd)
        self.demands.append(json.loads(Path(wanted["--demand"]).read_text(encoding="utf-8")))
        self.started.set()
        scope = current_cancel_scope()
        while self.hold and not self.release.is_set():
            if scope is not None and scope.cancelled:
                return subprocess.CompletedProcess(cmd, -15, "", "")
            self.release.wait(0.01)
        if self.fail:
            return subprocess.CompletedProcess(cmd, 2, "", "Traceback: the tuner crashed")
        out, tuner = Path(wanted["--output-dir"]), wanted["--tuner"]
        artifact = out / "tuners" / tuner / f"candidate_{tuner}.csv"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(table_text(self.header, self.rows), encoding="utf-8")
        if self.profile:
            name = self.profile_file or f"profile_{tuner}.csv"
            (artifact.parent / name).write_text(table_text(self.header, self.profile), encoding="utf-8")
        run = {"tuner": tuner, "status": "ok", "dropped_inaccurate": list(self.dropped), "shape_results": []}
        report = {
            "status": "ok",
            "micro_decision": "candidate",
            "tuners_run": [run],
            "artifacts": {tuner: str(artifact)},
            "total_elapsed_s": 1.0,
        }
        (out / "result.json").write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@dataclass
class Rig:
    box: LocalHostedToolbox
    blackboard: LocalBlackboard
    session: Path
    model: Path
    configs: Path
    forge: FakeForge

    def landed(self, **kw: Any) -> Path:
        return landed_test(self.session, self.model, **kw)

    async def submit(self, branch: str = "b1", **params: Any) -> str:
        return await self.box.submit_analysis(branch, kernel_tune.KERNEL_TUNE_TOOL, params)

    async def finished(self, job: str) -> Any:
        """The status of ``job`` once it ended and its pointer was pushed."""
        return await self.box._tools.wait(job)  # type: ignore[union-attr]

    def result(self, job: str) -> dict[str, Any]:
        """The ``result.json`` the job wrote for its branch's inbox."""
        return json.loads((self.session / "tool-results" / job / "result.json").read_text(encoding="utf-8"))


Make = Callable[..., Awaitable[Rig]]


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def build_rig(tmp_path: Path, monkeypatch: Any, made: list[LocalHostedToolbox], forge: FakeForge | None = None) -> Make:
    """``make(cards=2, tp=1, **branches)``: a local toolbox with the real tools on that many cards, Forge faked."""
    forge = forge or FakeForge()
    monkeypatch.setattr(kernel_tune, "run_with_session_kill", forge)

    def fake_runner(name: str, *, gpus: int, python: str, envs: Any) -> Any:
        return replace(runner_template(name, gpus=gpus), envs=dict(envs or {}))

    monkeypatch.setattr(local_hosted, "local_runner_template", fake_runner)

    async def make(
        cards: int = 2,
        tp: int = 1,
        points: Mapping[str, Path] | None = None,
        card_holders: Callable[[Sequence[int]], CardReading] = read_holders,
        card_guard: CardGuard | None = None,
    ) -> Rig:
        session = tmp_path / f"session-{len(made)}"
        session.mkdir()
        configs = install(tmp_path / f"image-{len(made)}")
        blackboard = lineage_blackboard(session / "blackboard")
        box = local_hosted.build_local_toolbox(
            LocalSession(session, f"run-{len(made)}", range(cards), tp=tp),
            blackboard,
            AcceptanceSetup(FakeRepo(), _PointPerBranch(dict(points or {})), IMAGE, WorkloadKind.AGENTX),
            spec_context(session, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")),
            options=LocalOptions(
                poll_interval_sec=0.05,
                framework_configs=str(configs),
                card_holders=card_holders,
                card_guard=card_guard,
            ),
        )
        made.append(box)
        await box.start()
        return Rig(box, blackboard, session, tmp_path / "model", configs, forge)

    return make


__all__ = [
    "ENV",
    "GRAPH_SIZES",
    "HIT_LOG",
    "HIT_SHAPES",
    "LOG_SHAPES",
    "REL_NORM_HEADER",
    "SECRET_ENV",
    "SERVER_ARGS",
    "SERVER_LOG",
    "TABLE",
    "TEST_ID",
    "TUNER",
    "FakeForge",
    "Rig",
    "build_rig",
    "csv_rows",
    "flags",
    "forge_dropped",
    "landed_test",
    "profile_row",
    "record_test",
    "tuned_rows",
]
