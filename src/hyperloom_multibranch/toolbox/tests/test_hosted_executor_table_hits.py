# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A succeeded test carries how much of each kernel table its recipe loads the test's server hit (#109 item 4).

The executor is the real one over the hosted backend and the local sandbox provider; the compiled environment is a real
one that loads a table of the rows of run ``cc2bf0ef``, and the server log the sandbox's rounds leave in the landed output
holds aiter's hit lines. What is read from the result is what the notebook stored.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.contracts.measurement import TABLE_EXERCISED, TABLE_NOT_OBSERVABLE, TestResult, Validity
from hyperloom.contracts.recipe import KernelKind, Recipe
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore
from hyperloom.measure.tests.hosted_fixtures import point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.mcp_tools import jsonable
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, build, req
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import B01_NEW, ENV, csv_text
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import (  # noqa: F401  (rig is a fixture)
    CANDIDATE,
    OK,
    POLICY,
    PointPlanner,
    ScriptedBackend,
    rig,
)
from hyperloom_multibranch.toolbox.tests.test_table_hits import ROW_1_34816, ROW_16_5120, MERGED, a8w8_hit

SERVER_LOG = "variant_00_c128/benchmark_sglang_0/server.log"


def _environment(root: Path, *, with_table: bool) -> CompiledEnvironment:
    blobs = LocalBlobStore(root / "blobs")
    table = csv_text(B01_NEW)
    kernels = (mint_kernel_artifact(KernelKind.GEMM, "gemm", ENV, table),) if with_table else ()
    blobs.put(table)
    return CompiledEnvironment(Recipe(IMAGE, kernels=kernels), IMAGE, (), (), {}, blobs.root)


async def _run(tmp_path: Path, r, mode: str, *, with_table: bool, log: str | None):
    point = point_yaml(tmp_path, "planned.yaml")
    scripted = ScriptedBackend(r.backend, point, [mode])
    if log is not None:
        scripted.stamps = {SERVER_LOG: log}
    planner = PointPlanner(point, _environment(tmp_path, with_table=with_table))
    tb = build(
        tmp_path,
        executor=HostedExecutor(scripted, tmp_path / "session"),  # type: ignore[arg-type]
        planner=planner,
        policy=POLICY,
        workload_kind=WorkloadKind.AGENTX,
    )
    await tb.toolbox.start()
    request_id = await tb.toolbox.submit_test(req())
    await asyncio.wait_for(tb.toolbox.drain(), 60)
    record = await tb.notebook.record(request_id)
    assert record.result is not None
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    return record.result, TestResult.from_document(entry.outcome["result"])


async def test_a_test_whose_server_hit_rows_of_its_table_carries_the_count_and_the_finding(tmp_path: Path, rig) -> None:
    log = "\n".join([a8w8_hit(ROW_1_34816, MERGED), a8w8_hit(ROW_16_5120, MERGED), a8w8_hit(ROW_16_5120, MERGED)])
    stored, outcome = await _run(tmp_path, rig(), OK, with_table=True, log=log)
    assert outcome.validity is Validity.SUCCEEDED
    for result in (stored, outcome):
        (hits,) = result.table_hits
        assert (hits.env, hits.target, hits.status) == (ENV, "gemm", TABLE_EXERCISED)
        assert (hits.rows, hits.rows_hit, hits.hit_lines) == (4, 2, 3)
        assert result.runtime_findings == (
            f"kernel table gemm ({ENV}): exercised, 2 of 4 rows hit in 3 log lines; 2 never hit",
        )
    shown = jsonable(stored)
    assert shown["table_hits"][0]["status"] == TABLE_EXERCISED and shown["runtime_findings"], "what test_result shows"


async def test_a_test_whose_server_logged_no_hit_says_its_table_is_not_observable_not_flat(tmp_path: Path, rig) -> None:
    # The stand-in runner's own server.log says nothing and the point does not ask for hits: a count of none is no answer.
    _, outcome = await _run(tmp_path, rig(), OK, with_table=True, log=None)
    (hits,) = outcome.table_hits
    assert (hits.status, hits.rows, hits.rows_hit) == (TABLE_NOT_OBSERVABLE, 4, None)
    (finding,) = outcome.runtime_findings
    assert finding.startswith(
        f"kernel table gemm ({ENV}): not observable, not every server log of the test holds a hit line of any table"
    )
    assert "AITER_LOG_TUNED_CONFIG" not in finding, "a finding does not ask the specialist to set the variable"
    assert finding.endswith("whether its rows were hit is unknown")


async def test_a_test_that_loads_no_table_has_no_hits_and_no_findings(tmp_path: Path, rig) -> None:
    stored, outcome = await _run(tmp_path, rig(), OK, with_table=False, log=a8w8_hit(ROW_1_34816, MERGED))
    for result in (stored, outcome):
        assert result.table_hits == () and result.runtime_findings == ()


async def test_a_test_that_did_not_succeed_makes_no_claim_about_its_tables(tmp_path: Path, rig) -> None:
    _, outcome = await _run(tmp_path, rig(), CANDIDATE, with_table=True, log=a8w8_hit(ROW_1_34816, MERGED))
    assert outcome.validity is not Validity.SUCCEEDED
    assert outcome.table_hits == () and outcome.runtime_findings == ()
