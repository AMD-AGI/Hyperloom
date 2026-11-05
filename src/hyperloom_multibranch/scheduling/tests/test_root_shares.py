# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from urllib.parse import quote

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.root_shares import (
    ShareSources,
    KERNEL_SHARES_SOURCE,
    RESULT_FILE,
    TOOL_RESULTS_DIR,
    kernel_shares,
    root_shares,
)
from hyperloom_multibranch.specialist.agent_onboarding import ROOT_PROFILE_SOURCE
from hyperloom_multibranch.toolbox.tools.base import JobStatus

ROOT, JOB = "c0ffee", "prelude:p1"
OPS = OpShares(ROOT, JOB, "", {"gemm": 48.0, "attention": 31.0})
#: The kernel table of a degraded report, as the report route writes it: a kernel in two rows, a row with no number.
TABLE = """name,Percentage (%),decode_percent,prefill_percent
paged_attention_ll4mi_QKV_mfma4_kernel,32.48,35.56,2.59
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs,27.28,30.09,0.0
aiter::add_rmsnorm_quant_kernel,7.16,7.31,5.68
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs,6.21,0.0,66.53
Memcpy HtoD,n/a,0,0
"""


class Toolbox:
    """A toolbox whose job reports ``statuses`` in turn, then the last one for good; no job at all with none."""

    def __init__(self, *statuses: JobStatus) -> None:
        self.statuses, self.asked = list(statuses), 0

    def tool_status(self, job_id: str) -> JobStatus:
        self.asked += 1
        if not self.statuses:
            raise LookupError(f"no job {job_id}")
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


class Records:
    def __init__(self, shares: OpShares | None) -> None:
        self.shares = shares

    async def op_shares(self, commit_id: str) -> OpShares | None:
        return self.shares if commit_id == ROOT else None


def _clock() -> Clock:
    now = [1000.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    return Clock(now=lambda: now[0], sleep=sleep)


def _degraded(toolbox_dir: Path, table: str | None = TABLE) -> None:
    """The job's result as a degraded report leaves it: no op shares, its kernel table beside it."""

    workdir = toolbox_dir / TOOL_RESULTS_DIR / quote(JOB, safe="")
    workdir.mkdir(parents=True)
    path = ""
    if table is not None:
        (workdir / "kernel_shares.csv").write_text(table, encoding="utf-8")
        path = str(workdir / "kernel_shares.csv")
    result = {"job_id": JOB, "status": "succeeded", "result": {"kernel_shares_path": path}}
    (workdir / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")


async def test_the_root_s_op_shares_are_read_once_its_profile_job_ended(tmp_path: Path) -> None:
    toolbox, clock = Toolbox(JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.SUCCEEDED), _clock()

    read = await root_shares(ShareSources(Records(OPS), toolbox, clock, tmp_path), ROOT, JOB)  # type: ignore[arg-type]

    assert read.shares is not None and (read.shares.shares, read.shares.source) == (OPS, ROOT_PROFILE_SOURCE)
    assert (read.detail["job"], read.detail["taken"], toolbox.asked) == ("succeeded", "op shares", 3)
    assert clock.now() == 1000.0 + 3 * 5.0  # two polls while it ran, one for its result to land


async def test_a_degraded_profile_s_kernel_shares_stand_in_for_arbor_alone(tmp_path: Path) -> None:
    _degraded(tmp_path)

    read = await root_shares(ShareSources(Records(None), Toolbox(JobStatus.SUCCEEDED), _clock(), tmp_path), ROOT, JOB)  # type: ignore[arg-type]

    assert read.shares is not None and read.shares.source == KERNEL_SHARES_SOURCE
    assert read.shares.shares == OpShares(
        ROOT,
        JOB,
        "",
        {
            "paged_attention_ll4mi_QKV_mfma4_kernel": 32.48,
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs": 27.28 + 6.21,
            "aiter::add_rmsnorm_quant_kernel": 7.16,
        },
    )
    assert (read.detail["taken"], read.detail["kernels"]) == ("kernel shares", 3)


async def test_without_a_job_or_a_table_or_past_the_wait_there_is_nothing_to_map_from(tmp_path: Path) -> None:
    none = await root_shares(ShareSources(Records(None), Toolbox(), _clock(), tmp_path), ROOT, "")  # type: ignore[arg-type]
    assert none.shares is None and none.detail["job"] == "no root profile was started"

    unknown = await root_shares(ShareSources(Records(None), Toolbox(), _clock(), tmp_path), ROOT, JOB)  # type: ignore[arg-type]
    assert unknown.detail["job"] == "the toolbox knows no such job"

    running = Toolbox(JobStatus.RUNNING)
    late = await root_shares(ShareSources(Records(None), running, _clock(), tmp_path), ROOT, JOB, wait_seconds=30)  # type: ignore[arg-type]
    assert (late.shares, late.detail["job"]) == (None, "still running after 30 s")
    assert late.detail["taken"] == "none: no op shares and no kernel table"


def test_kernel_shares_are_the_largest_of_the_table_and_none_without_one(tmp_path: Path) -> None:
    _degraded(tmp_path)
    result = tmp_path / TOOL_RESULTS_DIR / quote(JOB, safe="") / RESULT_FILE

    assert list(kernel_shares(result, limit=2)) == [
        "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs",
        "paged_attention_ll4mi_QKV_mfma4_kernel",
    ]
    assert kernel_shares(tmp_path / "missing" / RESULT_FILE) == {}
    _degraded(tmp_path / "empty", table=None)
    assert kernel_shares(tmp_path / "empty" / TOOL_RESULTS_DIR / quote(JOB, safe="") / RESULT_FILE) == {}
