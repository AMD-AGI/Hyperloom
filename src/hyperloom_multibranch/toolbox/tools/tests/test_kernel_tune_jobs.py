# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two ``kernel_tune`` jobs that tune at once each get the rows of their own shapes.

Forge reads aiter's comparison files back from ``<TMPDIR>/aiter_compare`` and takes the newest candidate a tuner wrote
there since its chunk started, whoever wrote it. The stand-in below does the same: it writes its own candidate there,
waits until the other job has written one too, and hands over the newest. Each job's tuner therefore has to be given a
temporary directory of its own, and a row of a shape the job did not ask for is refused.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import RoundWatch
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge
from hyperloom_multibranch.toolbox.tests.kernel_rows import ENV, HEADER, SHARED, install
from hyperloom_multibranch.toolbox.tests.kernel_tune_data import TEST_ID, landed_test, table_text
from hyperloom_multibranch.toolbox.tools import kernel_tune
from hyperloom_multibranch.toolbox.tools.base import ToolRequest
from hyperloom_multibranch.toolbox.tools.kernel_tune import (
    KERNEL_TUNE_TOOL,
    LANDED_PARAM,
    SHAPES_PARAM,
    TEST_PARAM,
    KernelTuneTool,
)

_BARRIER_SEC = 30.0
_COLUMNS = HEADER.split(",")
#: The (N, K) each branch asks for; Forge tunes them at a padded M.
_ASKED = {"alpha": (512, 7168), "beta": (4096, 1280)}


class OneCardEach:
    """Leases one card per job, numbered from 1; what the tool needs of the branch leases."""

    def __init__(self) -> None:
        self.leased = 0
        self.released: list[str] = []

    def check(self, cards: int) -> None:
        """Refuses no lease: every job has a card of its own."""

    async def lease(self, holder: str, cards: int, root: Path, *, charge: LeaseCharge) -> BranchLease:
        del holder, cards
        self.leased += 1
        return BranchLease(f"lease-{self.leased}", charge.branch, (self.leased,), root, {}, 0)

    def release(self, lease_id: str) -> None:
        self.released.append(lease_id)


def _flags(cmd: list[str]) -> dict[str, str]:
    return {word: cmd[index + 1] for index, word in enumerate(cmd[:-1]) if word.startswith("--")}


def _row(m: int, n: int, k: int) -> str:
    cells = dict(zip(_COLUMNS, SHARED[0].split(","), strict=True)) | {"M": str(m), "N": str(n), "K": str(k)}
    return ",".join(cells[column] for column in _COLUMNS)


class ComparingForge:
    """A tuner run as Forge drives it: it writes the candidate of its demand below ``<TMPDIR>/aiter_compare``, waits until
    every job of the test has written one, and makes its table of the newest candidate there, as Forge's chunk reader
    does. It records the ``TMPDIR`` each run was given; ``foreign`` is an (N, K) it tunes as well, unasked."""

    def __init__(self, jobs: int, foreign: tuple[int, int] | None = None) -> None:
        self.barrier = threading.Barrier(jobs, timeout=_BARRIER_SEC)
        self.foreign = foreign
        self.temp_dirs: list[str] = []

    def __call__(self, cmd: list[str], *, env: dict[str, str], watch: RoundWatch) -> subprocess.CompletedProcess[str]:
        del watch
        wanted = _flags(cmd)
        out, tuner = Path(wanted["--output-dir"]), wanted["--tuner"]
        (demand,) = json.loads(Path(wanted["--demand"]).read_text(encoding="utf-8"))["demands"]
        rows = [_row(16, int(key["N"]), int(key["K"])) for key in demand["keys"]]
        rows += [_row(16, *self.foreign)] if self.foreign else []
        temp = env.get("TMPDIR", "")
        self.temp_dirs.append(temp)
        compare = Path(temp or tempfile.gettempdir()) / "aiter_compare"
        compare.mkdir(parents=True, exist_ok=True)
        mine = compare / f"tuned_{tuner}_0.{threading.get_ident()}.candidate.csv"
        mine.write_text(table_text(HEADER, rows), encoding="utf-8")
        self.barrier.wait()
        newest = max(compare.glob(f"tuned_{tuner}_0.*.candidate.csv"), key=lambda path: path.stat().st_mtime_ns)
        self.barrier.wait()
        artifact = out / "tuners" / tuner / f"candidate_{tuner}.csv"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(newest.read_text(encoding="utf-8"), encoding="utf-8")
        run = {"tuner": tuner, "status": "ok", "dropped_inaccurate": [], "shape_results": []}
        report = {"status": "ok", "tuners_run": [run], "artifacts": {tuner: str(artifact)}}
        (out / "result.json").write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _request(branch: str, landed: Path) -> ToolRequest:
    n, k = _ASKED[branch]
    shape = {"env": ENV, "M": 16, "N": n, "K": k}
    return ToolRequest(
        KERNEL_TUNE_TOOL, branch, {TEST_PARAM: TEST_ID, SHAPES_PARAM: [shape], LANDED_PARAM: str(landed)}
    )


def _handed_over(result: dict[str, Any]) -> set[tuple[int, int]]:
    (table,) = result["tables"]
    return {(int(row["N"]), int(row["K"])) for row in csv.DictReader(io.StringIO(table["rows_csv"]))}


def _tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forge: ComparingForge) -> tuple[KernelTuneTool, Path]:
    """The tool on an image install, Forge stood in for, and the landed test both jobs read; the toolbox's own TMPDIR is
    ``inherited-tmp``."""
    inherited = tmp_path / "inherited-tmp"
    inherited.mkdir()
    monkeypatch.setenv("TMPDIR", str(inherited))
    monkeypatch.setattr(tempfile, "tempdir", str(inherited))
    monkeypatch.setattr(kernel_tune, "run_with_session_kill", forge)
    tool = KernelTuneTool(OneCardEach(), install(tmp_path / "image"))  # type: ignore[arg-type]
    return tool, landed_test(tmp_path / "session", tmp_path / "model")


async def test_two_jobs_tuning_at_once_each_get_the_rows_of_their_own_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forge = ComparingForge(jobs=2)
    tool, landed = _tool(tmp_path, monkeypatch, forge)
    inherited = str(tmp_path / "inherited-tmp")

    alpha, beta = await asyncio.gather(
        *(tool.run(_request(branch, landed), tmp_path / "results" / branch) for branch in _ASKED)
    )

    assert _handed_over(dict(alpha.result)) == {_ASKED["alpha"]}
    assert _handed_over(dict(beta.result)) == {_ASKED["beta"]}
    assert len(set(forge.temp_dirs)) == 2 and inherited not in forge.temp_dirs, "a TMPDIR of each job's own"
    assert not any(Path(temp).exists() for temp in forge.temp_dirs), "removed when the job's tuning ended"
    assert os.environ["TMPDIR"] == inherited, "the toolbox's own TMPDIR is left as it was"


async def test_a_tuned_row_of_a_shape_the_job_did_not_ask_for_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool, landed = _tool(tmp_path, monkeypatch, ComparingForge(jobs=1, foreign=_ASKED["beta"]))

    alpha = await tool.run(_request("alpha", landed), tmp_path / "results" / "alpha")

    assert _handed_over(dict(alpha.result)) == {_ASKED["alpha"]}
    (table,) = alpha.result["tables"]
    assert [(int(row["N"]), int(row["K"])) for row in table["rejected_rows"]] == [_ASKED["beta"]]
