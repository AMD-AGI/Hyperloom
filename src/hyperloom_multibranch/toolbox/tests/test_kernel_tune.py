# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernel_tune`` on the real local toolbox: its card is a lease from the pool of the measurements and the branches'
sandboxes, counted in ``leased_cards`` and given back on every way the job can end, and its rows are the rows-only CSV that
``put_kernel_artifact`` takes. The pool, the leases, the tool jobs, the notebook and the MCP service are the real ones;
Forge is a stand-in that answers as the tuner does (``kernel_tune_rig``), except where a test says it runs a real child.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hyperloom.contracts.recipe import KernelKind
from hyperloom.measure.framework_rows import lay_rows_over
from hyperloom.measure.tests.hosted_fixtures import point_yaml
from hyperloom.orchestrator.actions.executors import run_with_session_kill
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, MeasurementCapacity, RequestRejected
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import (
    ENV,
    SECRET_ENV,
    TEST_ID,
    Make,
    Rig,
    build_rig,
    csv_rows,
    flags,
    record_test,
    tuned_rows,
)
from hyperloom_multibranch.toolbox.tests.test_card_placement import HANG, _behaviour, _until
from hyperloom_multibranch.toolbox.tools import JobStatus, kernel_tune
from hyperloom_multibranch.toolbox.tools.kernel_tune import KERNEL_TUNE_TOOL, MIN_TUNER_SEC

SHARED_ROW = tuned_rows()[0]


@pytest_asyncio.fixture
async def make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Make]:
    made: list[LocalHostedToolbox] = []
    yield build_rig(tmp_path, monkeypatch, made)
    for box in made:
        if not box._closed:
            await box.withdraw_all()
            await box.close()


async def ready(rig: Rig, **kw: Any) -> Rig:
    """The rig with a test landed (a server log of untuned shapes, two points) and recorded in its notebook."""
    rig.landed(**kw)
    await record_test(rig.notebook)
    return rig


async def started(rig: Rig) -> None:
    """Wait until the stand-in tuner is running, that is until the job holds its card."""
    assert await asyncio.to_thread(rig.forge.started.wait, 20), "the tuner never started"


def pool(rig: Rig) -> tuple[int, ...]:
    return rig.box.provider.pool.free_cards()


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


# --- the card is a lease -----------------------------------------------------------------------------------------


async def test_a_running_job_holds_its_card_as_a_lease_that_leased_cards_counts(make: Make) -> None:
    rig = await ready(await make(cards=2))
    rig.forge.hold = True
    job = await rig.submit(test_request_id=TEST_ID)
    await started(rig)

    assert await rig.box.measurement_capacity() == MeasurementCapacity(2, 2, leased_cards=1)
    (held,) = rig.box.leases._granted.values()
    assert held.branch_id == f"tool:{job}" and pool(rig) == (1 - held.gpu_ids[0],)
    assert (await rig.box.queue_state()).running == 0, "a tool job is not a test: it is in no queue figure"
    assert [env["ROCR_VISIBLE_DEVICES"] for _, env in rig.forge.calls] == [str(held.gpu_ids[0])]

    rig.forge.release.set()
    assert await rig.finished(job) is JobStatus.SUCCEEDED
    assert await rig.box.measurement_capacity() == MeasurementCapacity(2, 2) and pool(rig) == (0, 1)


async def test_the_card_goes_back_when_the_tuner_fails(make: Make) -> None:
    rig = await ready(await make(cards=2))
    rig.forge.fail = True
    job = await rig.submit(test_request_id=TEST_ID)

    assert await rig.finished(job) is JobStatus.FAILED
    assert "exited 2 and left no result: Traceback: the tuner crashed" in rig.result(job)["summary"]
    assert await rig.box.measurement_capacity() == MeasurementCapacity(2, 2) and pool(rig) == (0, 1)


async def test_the_card_goes_back_when_the_run_raises(make: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = await ready(await make(cards=2))

    def boom(*_: Any, **__: Any) -> None:
        raise OSError("the subprocess could not start")

    monkeypatch.setattr(kernel_tune, "run_with_session_kill", boom)
    job = await rig.submit(test_request_id=TEST_ID)

    assert await rig.finished(job) is JobStatus.FAILED and "OSError" in rig.result(job)["error"]
    assert await rig.box.measurement_capacity() == MeasurementCapacity(2, 2) and pool(rig) == (0, 1)


async def test_a_cancelled_job_ends_its_child_before_its_card_goes_back(
    make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = await ready(await make(cards=2))
    script = "echo $$ > {out}/child.pid; sleep 40 & echo $! > {out}/grandchild.pid; wait"
    monkeypatch.setattr(kernel_tune, "run_with_session_kill", run_with_session_kill)
    monkeypatch.setattr(
        kernel_tune.forge_gemm_tuning, "build_command", lambda args: ["sh", "-c", script.format(out=args["output_dir"])]
    )
    seen: list[bool] = []
    give_back = rig.box.provider.pool.release
    job = await rig.submit(test_request_id=TEST_ID)
    workdir = rig.session / "tool-results" / job
    child, grandchild = (
        workdir / "a8w8_blockscale_bpreshuffle" / "child.pid",
        workdir / "a8w8_blockscale_bpreshuffle" / "grandchild.pid",
    )
    await _until(lambda: child.is_file() and grandchild.is_file() and grandchild.read_text().strip() != "", "the child")
    pids = int(child.read_text()), int(grandchild.read_text())

    def checked(cards: Any) -> None:
        seen.append(any(alive(pid) for pid in pids))
        give_back(cards)

    rig.box.provider.pool.release = checked  # type: ignore[method-assign]
    assert await rig.box.cancel_branch("b1") == 1

    assert await rig.finished(job) is JobStatus.CANCELLED
    assert seen == [False], "the card went back to the pool while the tuner's processes were alive"
    assert not any(alive(pid) for pid in pids) and pool(rig) == (0, 1)
    assert await rig.box.measurement_capacity() == MeasurementCapacity(2, 2)


async def test_closing_the_toolbox_ends_the_job_and_gives_its_card_back(make: Make) -> None:
    rig = await ready(await make(cards=2))
    rig.forge.hold = True
    job = await rig.submit(test_request_id=TEST_ID)
    await started(rig)

    await rig.box.close()

    assert rig.box.tool_status(job) is JobStatus.CANCELLED
    assert rig.box.leases.capacity() == MeasurementCapacity(2, 2) and pool(rig) == (0, 1)


async def test_a_job_that_waits_for_a_busy_card_is_already_counted_and_gives_its_promise_back_when_stopped(
    make: Make, tmp_path: Path
) -> None:
    points = {b: point_yaml(tmp_path, f"{b}.yaml", **_behaviour(HANG)) for b in ("a", "b")}
    rig = await ready(await make(cards=2, points=points))
    await rig.box.submit_test(req("a"))
    await rig.box.submit_test(req("b"))
    await _until(lambda: pool(rig) == (), "both cards measuring")

    job = await rig.submit(test_request_id=TEST_ID)

    await _until(lambda: rig.box.leases.capacity().leased_cards == 1, "the job promised its card")
    assert rig.forge.calls == [] and rig.box.tool_status(job) is JobStatus.RUNNING
    assert await rig.box.cancel_branch("b1") == 1
    assert await rig.finished(job) is JobStatus.CANCELLED
    assert (await rig.box.measurement_capacity()).leased_cards == 0 and rig.forge.calls == []
    await rig.box.withdraw_all()


async def test_the_tests_a_job_kept_waiting_start_when_its_card_goes_back(make: Make, tmp_path: Path) -> None:
    """Four cards, two to a test: with the job's card two tests no longer run at once, and the second starts when the card
    returns although no test has ended."""
    points = {b: point_yaml(tmp_path, f"{b}.yaml", tp=2, **_behaviour(HANG)) for b in ("a", "b")}
    rig = await ready(await make(cards=4, tp=2, points=points))
    rig.forge.hold = True
    job = await rig.submit(test_request_id=TEST_ID)
    await started(rig)
    await rig.box.submit_test(req("a"))
    await rig.box.submit_test(req("b"))
    await _until(lambda: len(rig.box.provider.find({}, role="measurement")) == 1, "one test on the cards left")
    await asyncio.sleep(0.3)
    assert (await rig.box.queue_state()).queued == 1

    rig.forge.release.set()
    await rig.finished(job)

    await _until(
        lambda: len(rig.box.provider.find({}, role="measurement")) == 2, "the second test on the returned card"
    )
    await rig.box.withdraw_all()


# --- a tool job never starves measurement -------------------------------------------------------------------------


@pytest.mark.parametrize("held", [1, 2])
async def test_a_job_that_would_leave_less_than_one_test_needs_is_refused_by_name_and_starts_nothing(
    make: Make, held: int
) -> None:
    rig = await ready(await make(cards=2 + held - 1, tp=1))
    await rig.box.lease_branch_sandbox("agent", held)
    before = await rig.box.measurement_capacity()

    with pytest.raises(RequestRejected) as refused:
        await rig.submit(test_request_id=TEST_ID)

    assert refused.value.code == "lease_leaves_no_measurement" and "lease_leaves_no_measurement" in REJECTION_CODES
    assert rig.forge.calls == [] and await rig.box.measurement_capacity() == before
    assert not (rig.session / "tool-results").exists() or not list((rig.session / "tool-results").iterdir())


@pytest.mark.parametrize(("cards", "tp"), [(1, 1), (2, 2), (3, 3)])
async def test_a_pool_that_cannot_spare_a_card_does_not_serve_the_tool_at_all(make: Make, cards: int, tp: int) -> None:
    rig = await ready(await make(cards=cards, tp=tp))

    assert rig.box.analysis_tools == ("tracelens",) and KERNEL_TUNE_TOOL not in rig.box._tools.tool_names  # type: ignore[union-attr]
    with pytest.raises(RequestRejected) as refused:
        await rig.submit(test_request_id=TEST_ID)
    assert refused.value.code == "tool_unavailable_on_backend"


async def test_a_branch_that_holds_a_sandbox_lease_may_run_the_tool_too_and_both_are_counted(make: Make) -> None:
    rig = await ready(await make(cards=4))
    await rig.box.lease_branch_sandbox("b1", 1)
    rig.forge.hold = True
    job = await rig.submit(branch="b1", test_request_id=TEST_ID)
    await started(rig)

    assert await rig.box.measurement_capacity() == MeasurementCapacity(4, 4, leased_cards=2)

    rig.forge.release.set()
    await rig.finished(job)
    assert await rig.box.measurement_capacity() == MeasurementCapacity(4, 4, leased_cards=1)


# --- what the tuner is asked, and what it answers -----------------------------------------------------------------


async def test_the_tuner_is_asked_for_the_shapes_of_the_log_on_the_leased_card_alone(
    make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SECRET_ENV, "hf_do_not_pass_on")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    rig = await ready(await make(cards=2))
    job = await rig.submit(test_request_id=TEST_ID, timeout_sec=1200)
    await rig.finished(job)

    ((cmd, env),) = rig.forge.calls
    wanted = flags(cmd)
    assert cmd[2:5] == ["kernelforge.cli", "gemm-tune", "run"]
    assert (wanted["--tuner"], wanted["--framework"], wanted["--precision"]) == (
        "a8w8_blockscale_bpreshuffle",
        "sglang",
        "fp8",
    )
    assert (wanted["--tp"], wanted["--conc"], wanted["--mp"]) == ("1", "128", "1")
    assert wanted["--model-path"] == str(rig.model) and wanted["--timeout"] == wanted["--global-timeout"] == "1200"
    assert {(k["M"], k["N"], k["K"]) for k in rig.forge.demands[0]["demands"][0]["keys"]} == {
        ("496", "7168", "5120"),
        ("496", "5120", "5120"),
        ("496", "34816", "5120"),
        ("496", "5120", "17408"),
    }
    assert env["ROCR_VISIBLE_DEVICES"] in {"0", "1"} and "HIP_VISIBLE_DEVICES" not in env and SECRET_ENV not in env
    assert wanted["--output-dir"].startswith(str(rig.session / "tool-results" / job))


async def test_the_answer_carries_each_tables_rows_as_the_csv_put_kernel_artifact_takes(make: Make) -> None:
    rig = await ready(await make(cards=2))
    rig.forge.rows = tuned_rows()[::-1]  # the table in the wrong order
    job = await rig.submit(test_request_id=TEST_ID)
    assert await rig.finished(job) is JobStatus.SUCCEEDED

    result = rig.result(job)["result"]
    (table,) = result["tables"]
    assert (table["env"], table["kind"], table["rows_only"], table["status"], table["rows"]) == (
        ENV,
        "gemm",
        True,
        "tuned",
        2,
    )
    assert [(r["M"], r["verified"]) for r in table["verified_rows"]] == [("16", True), ("20480", True)]
    assert [r["M"] for r in csv_rows(table["rows_csv"])] == ["16", "20480"], "in the order of the key"
    assert Path(table["rows_file"]).read_text() == table["rows_csv"]
    assert "advice" in result["advice"] and "not a measurement the gate reads" in result["advice"]
    assert result["source"] == {"shapes": "server_log", "test_request_id": TEST_ID} and result["cards"] in ([0], [1])

    artifact = await rig.box.put_kernel_artifact(
        KernelKind.GEMM, table["target"], table["env"], table["rows_csv"].encode(), rows_only=True
    )
    assert artifact.metadata["load"]["rows_only"] is True


async def test_the_rows_override_the_default_table_laid_under_them_with_one_row_to_a_shape(make: Make) -> None:
    """The rows are keyed as ``table_key`` keys the variable, so laying them over the image's default table takes the row of
    a shape both have out of the default, and the rows are accepted without a repeated shape."""
    rig = await ready(await make(cards=2))
    job = await rig.submit(test_request_id=TEST_ID)
    await rig.finished(job)
    (table,) = rig.result(job)["result"]["tables"]
    default = rig.configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    header = table["rows_csv"].splitlines()[0]
    default.write_text(f"{header}\n{SHARED_ROW}\n", encoding="utf-8")

    over = lay_rows_over([default], [Path(table["rows_file"])], rig.configs, "a8w8_blockscale_bpreshuffle_tuned_gemm")

    assert over.rows == 2 and over.overridden == 1 and set(over.key or ()) == {"gfx", "cu_num", "M", "N", "K"}


async def test_a_row_that_fails_the_reference_or_cannot_be_checked_is_listed_as_not_verified_and_is_in_no_csv(
    make: Make,
) -> None:
    rig = await ready(await make(cards=2))
    wrong = tuned_rows(us=(1.0, 1.0), err=(0.69, 0.0))[0].replace(",16,", ",64,", 1)
    unchecked = tuned_rows()[1].replace(",20480,", ",1024,", 1).rsplit(",", 1)[0] + ",Null"
    forge_dropped = {"M": 2048, "N": 512, "K": 7168, "libtype": "ck", "splitK": 0, "us": 2.0, "err_ratio": 0.4}
    rig.forge.rows = [*tuned_rows(), wrong, unchecked]
    rig.forge.dropped = [forge_dropped]
    job = await rig.submit(test_request_id=TEST_ID)
    assert await rig.finished(job) is JobStatus.SUCCEEDED

    result = rig.result(job)
    (table,) = result["result"]["tables"]
    assert [r["M"] for r in csv_rows(table["rows_csv"])] == ["16", "20480"]
    refused = {str(r["M"]): r for r in table["rejected_rows"]}
    assert set(refused) == {"64", "1024", "2048"} and all(r["verified"] is False for r in refused.values())
    assert "outside the reference's tolerance" in refused["64"]["reason"] and refused["64"]["err_ratio"] == 0.69
    assert "was not checked" in refused["1024"]["reason"] and "Forge removed it" in refused["2048"]["reason"]
    assert "3 rejected" in result["summary"] and "advice only" in result["summary"]
    assert "rejected rows" not in table["rows_csv"] and not {"64", "1024", "2048"} & {
        r["M"] for r in csv_rows(table["rows_csv"])
    }


async def test_a_table_whose_every_row_was_refused_has_no_csv_at_all(make: Make) -> None:
    rig = await ready(await make(cards=2))
    rig.forge.rows = tuned_rows(err=(0.69, 0.5))
    job = await rig.submit(test_request_id=TEST_ID)

    assert await rig.finished(job) is JobStatus.SUCCEEDED
    (table,) = rig.result(job)["result"]["tables"]
    assert (table["status"], table["rows"], table["rows_csv"], table["rows_file"]) == ("no_rows", 0, "", "")
    assert len(table["rejected_rows"]) == 2 and "none of the 2 row(s) the tuner measured passed" in table["reason"]


async def test_explicit_shapes_replace_the_log_and_the_test_only_names_the_workload(make: Make) -> None:
    rig = await ready(await make(cards=2), log=None)
    shapes = [{"env": ENV, "M": 64, "N": 5120, "K": 5120}, {"env": ENV, "M": 128, "N": 5120, "K": 17408}]
    job = await rig.submit(test_request_id=TEST_ID, shapes=shapes)

    assert await rig.finished(job) is JobStatus.SUCCEEDED
    (entry,) = rig.forge.demands[0]["demands"]
    assert {(k["M"], k["N"], k["K"]) for k in entry["keys"]} == {("64", "5120", "5120"), ("128", "5120", "17408")}
    assert rig.result(job)["result"]["source"]["shapes"] == "shapes"


async def test_a_log_with_nothing_untuned_fails_the_job_before_it_takes_a_card(make: Make) -> None:
    rig = await ready(await make(cards=2), log="[aiter] import [module_aiter_core] under /sgl-workspace/aiter\n")
    job = await rig.submit(test_request_id=TEST_ID)

    assert await rig.finished(job) is JobStatus.FAILED
    assert "report no untuned dense GEMM shape" in rig.result(job)["summary"]
    assert rig.forge.calls == [] and await rig.box.measurement_capacity() == MeasurementCapacity(2, 2)


async def test_the_time_is_shared_among_the_tables_and_a_table_it_does_not_reach_is_skipped(make: Make) -> None:
    rig = await ready(await make(cards=2), log=None)
    shapes = [
        {"env": ENV, "M": 64, "N": 5120, "K": 5120},
        {"env": "AITER_CONFIG_GEMM_BF16", "M": 8, "N": 256, "K": 5120},
    ]
    shared = await rig.submit(test_request_id=TEST_ID, shapes=shapes, timeout_sec=2 * MIN_TUNER_SEC)
    await rig.finished(shared)
    short = await rig.submit(test_request_id=TEST_ID, shapes=shapes, timeout_sec=MIN_TUNER_SEC)
    await rig.finished(short)

    first, second = (flags(cmd) for cmd, _ in rig.forge.calls[:2])
    assert first["--timeout"] == second["--timeout"] == str(MIN_TUNER_SEC)
    statuses = {t["env"]: t["status"] for t in rig.result(short)["result"]["tables"]}
    assert statuses == {ENV: "tuned", "AITER_CONFIG_GEMM_BF16": "skipped"} and len(rig.forge.calls) == 3
    assert (
        "no time left" in next(t for t in rig.result(short)["result"]["tables"] if t["status"] == "skipped")["reason"]
    )
    both = {t["env"]: t for t in rig.result(shared)["result"]["tables"]}
    assert (
        both["AITER_CONFIG_GEMM_BF16"]["status"] == "failed"
        and "declares no shape key" in both["AITER_CONFIG_GEMM_BF16"]["reason"]
    )


async def test_a_job_whose_one_attempted_table_failed_and_the_others_were_skipped_is_a_failed_job(make: Make) -> None:
    rig = await ready(await make(cards=2), log=None)
    rig.forge.fail = True
    shapes = [
        {"env": ENV, "M": 64, "N": 5120, "K": 5120},
        {"env": "AITER_CONFIG_GEMM_BF16", "M": 8, "N": 256, "K": 5120},
    ]
    job = await rig.submit(test_request_id=TEST_ID, shapes=shapes, timeout_sec=MIN_TUNER_SEC)

    assert await rig.finished(job) is JobStatus.FAILED
    statuses = {t["env"]: t["status"] for t in rig.result(job)["result"]["tables"]}
    assert statuses == {ENV: "failed", "AITER_CONFIG_GEMM_BF16": "skipped"} and len(rig.forge.calls) == 1


async def test_a_job_whose_every_table_failed_is_a_failed_job(make: Make) -> None:
    rig = await ready(await make(cards=2), log=None)
    job = await rig.submit(
        test_request_id=TEST_ID, shapes=[{"env": "AITER_CONFIG_GEMM_BF16", "M": 8, "N": 256, "K": 5120}]
    )

    assert await rig.finished(job) is JobStatus.FAILED
    assert "declares no shape key for AITER_CONFIG_GEMM_BF16" in rig.result(job)["summary"]
    assert (await rig.box.measurement_capacity()).leased_cards == 0
