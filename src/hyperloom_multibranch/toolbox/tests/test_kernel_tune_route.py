# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``start_analysis`` for ``kernel_tune`` over real HTTP: advertised only where the toolbox serves it, started on the output a
test landed (never a directory of the caller's), and refused by name before anything is started or any card is held."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import (
    McpSpecialistTokens,
    SpecialistToolOptions,
    register_specialist_tools,
)
from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import ENV, TEST_ID, Make, Rig, build_rig, record_test
from hyperloom_multibranch.toolbox.tests.test_analysis_route import Stack
from hyperloom_multibranch.toolbox.tools import JobStatus
from hyperloom_multibranch.toolbox.tools.kernel_tune import MAX_SHAPES, MAX_TIMEOUT_SEC, MIN_TUNER_SEC

SHAPE = {"env": ENV, "M": 64, "N": 5120, "K": 5120}


@dataclass
class Served:
    rig: Rig
    stack: Stack


@pytest_asyncio.fixture
async def make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Make]:
    made: list[LocalHostedToolbox] = []
    servers: list[McpServer] = []
    build = build_rig(tmp_path, monkeypatch, made)

    async def serve(cards: int = 2, tp: int = 1, landed: bool = True) -> Served:
        rig = await build(cards=cards, tp=tp)
        if landed:
            rig.landed()
            await record_test(rig.blackboard)
        server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w117")
        register_specialist_tools(
            server,
            toolbox=rig.box,
            blackboards=lambda b: LocalBlackboard(rig.session / "blackboard", Writer(Role.SPECIALIST_RUNTIME, b, b)),
            options=SpecialistToolOptions(analysis=rig.box, toolbox_loop=asyncio.get_running_loop()),
        )
        server.start()
        servers.append(server)
        return Served(rig, Stack(None, rig.box._tools, None, {}, server, McpSpecialistTokens(server)))  # type: ignore[arg-type]

    yield serve  # type: ignore[misc]
    for server in servers:
        server.stop()
    for box in made:
        if not box._closed:
            await box.withdraw_all()
            await box.close()


def untouched(rig: Rig) -> bool:
    """Nothing was started: no Forge run, no job directory, no card held."""
    results = rig.session / "tool-results"
    return not rig.forge.calls and not (results.exists() and list(results.iterdir())) and not rig.box.leases._granted


async def test_the_local_toolbox_advertises_the_tuner_where_it_can_spare_a_card_and_not_where_it_cannot(
    make: Any,
) -> None:
    served = await make(cards=2)
    entry = served.stack.advertised()
    assert entry["inputSchema"]["properties"]["tool"]["enum"] == [
        "tracelens",
        "roofline",
        "kernel_tune",
        "upstream_pr_search",
    ]
    assert "kernel_tune" in entry["description"] and "lease_leaves_no_measurement" in entry["description"]
    assert "verified=false" in entry["description"] and "rows_only=true" in entry["description"]

    one = await make(cards=1)
    bare = one.stack.advertised()
    assert bare["inputSchema"]["properties"]["tool"]["enum"] == ["tracelens", "roofline", "upstream_pr_search"]
    assert "kernel_tune" not in bare["description"] and "kernel_tune" not in json.dumps(bare["inputSchema"])
    assert "must be one of ['tracelens', 'roofline', 'upstream_pr_search']" in await one.stack.refused(
        tool="kernel_tune", params={"test_request_id": TEST_ID}
    )


async def test_the_advertisement_says_a_test_request_id_is_required_and_the_server_refuses_shapes_alone(
    make: Any,
) -> None:
    """A specialist read the old text as "shapes alone will do" and got test_required: both places now say otherwise."""
    served = await make(cards=2)
    entry = served.stack.advertised()
    description = entry["description"]
    assert "params.test_request_id is required" in description, "the tool's description"
    assert "names a test whose landed output the tool reads" in description
    assert "model, framework and precision" in description, "what the test is for"
    assert "params.shapes" in description and "is optional" in description
    assert "replaces the shapes of that log" in description and "does not stand in for test_request_id" in description
    params = entry["inputSchema"]["properties"]["params"]["description"]
    assert "test_request_id is required" in params and "test_required" in params, "the schema's own description"

    text = await served.stack.refused(tool="kernel_tune", params={"shapes": [SHAPE]})  # what the text warns of
    assert text.startswith("rejected (test_required):") and "name it with test_request_id" in text
    assert untouched(served.rig)


async def test_the_advertisement_says_that_the_relaxed_rows_are_the_specialists_choice_and_an_accuracy_check_decides(
    make: Any,
) -> None:
    description = (await make(cards=2)).stack.advertised()["description"]

    assert "params.max_err_ratio (up to 0.05)" in description and "rows_csv_relaxed" in description
    assert "relaxed_rows" in description and "err_ratio_above_default" in description
    assert "never in rows_csv" in description and "your explicit choice" in description
    assert "end-to-end accuracy check" in description and "a mismatch ratio of at most 0.01" in description
    assert "params.retune=true" in description and "CUDA-graph batch sizes" in description


async def test_a_specialist_starts_the_tuner_on_a_test_and_its_rows_reach_its_inbox(make: Any) -> None:
    served = await make(cards=2)
    reply = await served.stack.start(tool="kernel_tune", params={"test_request_id": TEST_ID})
    assert reply["result"]["isError"] is False, reply
    job = json.loads(reply["result"]["content"][0]["text"])["job_id"]

    assert await served.rig.finished(job) is JobStatus.SUCCEEDED

    page = await LocalBlackboard(served.rig.blackboard.root, Writer(Role.SPECIALIST_RUNTIME, "b1", "b1")).poll("b1")
    (pointer,) = [item for item in page.items if item.pointer == job]
    assert pointer.kind.value == "tool_result"
    (table,) = served.rig.result(job)["result"]["tables"]
    assert table["rows"] == 2 and table["target"] == "gemm_a8w8_blockscale_bpreshuffle"


async def test_a_job_that_would_starve_measurement_is_refused_to_the_specialist_by_name(make: Any) -> None:
    served = await make(cards=2)
    await served.rig.box.lease_branch_sandbox("agent", 1)

    text = await served.stack.refused(tool="kernel_tune", params={"test_request_id": TEST_ID})

    assert "rejected (lease_leaves_no_measurement)" in text
    assert not served.rig.forge.calls and await served.rig.box.measurement_capacity() == MeasurementCapacity(
        2, 2, leased_cards=1
    )


@pytest.mark.parametrize(
    ("params", "code", "reason"),
    [
        ({}, "test_required", "name it with test_request_id"),
        ({"test_request_id": "nope"}, "test_unknown", "holds no test 'nope'"),
        (
            {"test_request_id": TEST_ID, "landed_dir": "/etc"},
            "server_owned_param",
            "landed_dir in params is set by the server",
        ),
        ({"test_request_id": TEST_ID, "shape": []}, "invalid_request", "not ['shape']"),
        ({"test_request_id": TEST_ID, "shapes": []}, "invalid_request", "a list of 1 to"),
        ({"test_request_id": TEST_ID, "shapes": "M=1"}, "invalid_request", "a list of 1 to"),
        ({"test_request_id": TEST_ID, "shapes": [SHAPE] * (MAX_SHAPES + 1)}, "invalid_request", "a list of 1 to"),
        ({"test_request_id": TEST_ID, "shapes": [3]}, "invalid_request", "shape 0 is not an object"),
        (
            {"test_request_id": TEST_ID, "shapes": [{**SHAPE, "env": "AITER_CONFIG_FMOE"}]},
            "invalid_request",
            "not one of",
        ),
        ({"test_request_id": TEST_ID, "shapes": [{"env": ENV, "M": 1, "N": 2}]}, "invalid_request", "lacks ['K']"),
        ({"test_request_id": TEST_ID, "shapes": [{**SHAPE, "cu_num": 256}]}, "invalid_request", "unknown column"),
        ({"test_request_id": TEST_ID, "shapes": [{**SHAPE, "M": 0}]}, "invalid_request", "positive integer"),
        ({"test_request_id": TEST_ID, "shapes": [{**SHAPE, "N": True}]}, "invalid_request", "positive integer"),
        ({"test_request_id": TEST_ID, "timeout_sec": MIN_TUNER_SEC - 1}, "invalid_request", "whole seconds from"),
        ({"test_request_id": TEST_ID, "timeout_sec": MAX_TIMEOUT_SEC + 1}, "invalid_request", "whole seconds from"),
        ({"test_request_id": TEST_ID, "timeout_sec": "long"}, "invalid_request", "whole seconds from"),
        (
            {"test_request_id": TEST_ID, "max_err_ratio": 0.0501},
            "invalid_request",
            "mismatch ratio from 0.01 (the default) to 0.05",
        ),
        (
            {"test_request_id": TEST_ID, "max_err_ratio": 0.0099},
            "invalid_request",
            "mismatch ratio from 0.01 (the default) to 0.05",
        ),
        ({"test_request_id": TEST_ID, "max_err_ratio": "0.02"}, "invalid_request", "mismatch ratio from 0.01"),
        ({"test_request_id": TEST_ID, "max_err_ratio": True}, "invalid_request", "mismatch ratio from 0.01"),
        ({"test_request_id": TEST_ID, "max_err_ratio": float("nan")}, "invalid_request", "mismatch ratio from 0.01"),
        ({"test_request_id": TEST_ID, "retune": "yes"}, "invalid_request", "retune is true or false"),
        ({"test_request_id": TEST_ID, "retune": 1}, "invalid_request", "retune is true or false"),
        ({"test_request_id": TEST_ID, "retune": True, "shapes": [SHAPE]}, "invalid_request", "give one or the other"),
    ],
)
async def test_a_request_no_run_can_use_is_refused_by_name_before_anything_starts(
    make: Any, params: dict[str, Any], code: str, reason: str
) -> None:
    served = await make(cards=2)

    with pytest.raises(RequestRejected) as refused:
        await served.rig.submit(**params)

    assert (refused.value.code, reason in refused.value.reason) == (code, True), refused.value
    assert untouched(served.rig) and await served.rig.box.measurement_capacity() == MeasurementCapacity(2, 2)


@pytest.mark.parametrize("where", ["", "tests/t1/attempt-1", "tests/t1/link"])
async def test_a_test_that_landed_no_output_inside_the_session_is_refused_by_name(
    make: Any, tmp_path: Path, where: str
) -> None:
    """No output directory recorded, one that is not there, and one that is a link out of the session."""
    served = await make(cards=2, landed=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "server.log").write_text(
        "[aiter] shape is M:1, N:2, K:3, not found tuned config in /x.csv, will", encoding="utf-8"
    )
    (served.rig.session / "tests" / "t1").mkdir(parents=True)
    (served.rig.session / "tests" / "t1" / "link").symlink_to(outside)
    await record_test(served.rig.blackboard, output_dir=where)

    with pytest.raises(RequestRejected) as refused:
        await served.rig.submit(test_request_id=TEST_ID)

    assert refused.value.code == "test_without_output" and "has no landed output to read" in refused.value.reason
    assert untouched(served.rig)


async def test_a_test_of_another_branch_may_be_tuned_for_as_the_blackboards_records_are_shared(make: Any) -> None:
    served = await make(cards=2)

    job = await served.rig.submit(branch="b2", test_request_id=TEST_ID)

    assert await served.rig.finished(job) is JobStatus.SUCCEEDED
    assert served.rig.result(job)["branch"] == "b2"
