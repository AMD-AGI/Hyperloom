# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernel_tune`` runs Forge's real tuner on one real card for shapes a real server log reports untuned (W117).

The toolbox is the real local one; the test is landed output with the lines of a real server log (Qwen3-14B-FP8 on gfx950:
two dense blockscale GEMMs that missed their tuned table), the tool leases its card from the toolbox's pool, runs
``kernelforge gemm-tune run`` on it for the shapes' padded M and the decode band of the point's concurrency, and answers
with the rows-only CSV. What must hold: the job succeeds and held exactly one leased card while it ran (``leased_cards``
1 during, 0 after), on the card named; every row handed over has a finite ``errRatio`` of at most Forge's limit (an asm row at a
splitK above 1 may have its relative-norm error within the limit instead) and ``verified`` true; the CSV is accepted by ``put_kernel_artifact(rows_only=true)`` and lays over the image's own default
table without a repeated shape. What was measured is printed (``PROBE``).

Needs a GPU machine with the image's aiter at ``/sgl-workspace`` and ``kernelforge`` importable, ``HL_W117_MODEL`` (a
directory with the model's ``config.json``, fp8 blockscale), ``HL_W117_OUT`` (an empty directory) and ``HL_W117_CARD`` (the
one card the tuner may use; the pool also holds one card that does not exist, which no job ever reaches). Anywhere else
the module is skipped.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import KernelKind
from hyperloom.measure.framework.base import DEFAULT_CONFIGS
from hyperloom.measure.framework.rows import lay_rows_over
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import LocalOptions, LocalSession, build_local_toolbox
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo
from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import (
    ENV,
    SERVER_LOG,
    TEST_ID,
    csv_rows,
    landed_test,
    record_test,
)
from hyperloom_multibranch.toolbox.tests.test_card_placement import _PointPerBranch
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import JobStatus
from hyperloom_multibranch.toolbox.tools.tune_rows import CHECK_REL_NORM_ERR, ERR_RATIO_LIMIT

MODEL = os.environ.get("HL_W117_MODEL", "")
OUT = os.environ.get("HL_W117_OUT", "")
CARD = int(os.environ.get("HL_W117_CARD", "-1"))
DEFAULT_TABLE = Path(DEFAULT_CONFIGS) / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
TIMEOUT_SEC = 3600
#: The two untuned shapes: the (N, K) of the model's gate_up and o projections, as the real log reports them at M=496.
UNTUNED = ("M:496, N:5120, K:5120", "M:496, N:34816, K:5120")
pytestmark = pytest.mark.skipif(
    not (Path(MODEL, "config.json").is_file() and Path(OUT).is_dir() and CARD >= 0 and DEFAULT_TABLE.is_file()),
    reason="needs HL_W117_MODEL, HL_W117_OUT, HL_W117_CARD and the image's aiter tables",
)


def probe(label: str, **fields: object) -> None:
    print(f"PROBE {label} " + json.dumps(fields, default=str), flush=True)


def real_log() -> str:
    """The lines of the real server log that say these two shapes missed the blockscale table."""
    return "".join(line + "\n" for line in SERVER_LOG.splitlines() if any(shape in line for shape in UNTUNED))


async def test_the_real_tuner_tunes_two_untuned_shapes_on_the_leased_card_and_the_rows_are_checked() -> None:
    session = Path(OUT) / "session"
    session.mkdir()
    notebook = LocalNotebook(session / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    box = build_local_toolbox(
        LocalSession(session, "w117", [CARD, CARD + 1000], tp=1),
        notebook,
        AcceptanceSetup(FakeRepo(), _PointPerBranch({}), IMAGE, WorkloadKind.AGENTX),
        spec_context(session, blob_root=str(session / "hosted" / "blobs")),
        options=LocalOptions(),
    )
    await box.start()
    try:
        assert "kernel_tune" in box.analysis_tools
        landed_test(session, Path(MODEL), log=real_log(), concs=(1,))
        await record_test(notebook)
        job = await box.submit_analysis("b1", "kernel_tune", {"test_request_id": TEST_ID, "timeout_sec": TIMEOUT_SEC})
        await asyncio.sleep(5)
        during = await box.measurement_capacity()
        assert box.tool_status(job) is JobStatus.RUNNING and during == MeasurementCapacity(2, 2, leased_cards=1)

        status = await box._tools.wait(job)  # type: ignore[union-attr]

        record = json.loads((session / "tool-results" / job / "result.json").read_text(encoding="utf-8"))
        assert status is JobStatus.SUCCEEDED, record["summary"]
        result = record["result"]
        (table,) = result["tables"]
        probe(
            "tuned",
            summary=record["summary"],
            cards=result["cards"],
            leased_during=during.leased_cards,
            status=table["status"],
            rows=[
                {k: row[k] for k in ("M", "N", "K", "libtype", "kernelName", "us", "errRatio", "verified")}
                for row in table["verified_rows"]
            ],
            rejected=table["rejected_rows"],
            forge={k: table["forge"][k] for k in ("status", "micro_decision", "total_elapsed_s")},
        )
        assert result["cards"] == [CARD] and table["env"] == ENV and table["status"] == "tuned" and table["rows"] >= 1
        assert all(
            row["verified"] is True and (0 <= row["err_ratio"] <= ERR_RATIO_LIMIT or row["check"] == CHECK_REL_NORM_ERR)
            for row in table["verified_rows"]
        )
        assert {row["gfx"] for row in table["verified_rows"]} and csv_rows(table["rows_csv"])
        artifact = await box.put_kernel_artifact(
            KernelKind.GEMM, table["target"], table["env"], table["rows_csv"].encode(), rows_only=True
        )
        over = lay_rows_over(
            [DEFAULT_TABLE], [Path(table["rows_file"])], Path(DEFAULT_CONFIGS), "a8w8_blockscale_bpreshuffle_tuned_gemm"
        )
        probe("put", artifact=artifact.ref, rows=over.rows, overridden=over.overridden, key=over.key)
        assert await box.measurement_capacity() == MeasurementCapacity(2, 2)
    finally:
        await box.close()
