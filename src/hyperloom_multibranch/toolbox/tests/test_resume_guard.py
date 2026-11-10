# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The baseline-reuse guard and the infra-ended keeps a resume reads, over session directories the toolbox wrote.

Every directory here is written by the toolbox itself: the base launch frozen as a session freezes it, the record of what
the baseline is measured on written when the toolbox accepts the first baseline test, and the submit log by the toolbox's
own submits, outcomes and withdrawals.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, Validity
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.measure.pins import IMAGE_DIGEST_VAR, IMAGE_REF_VAR
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.build import CodeVersion, code_version
from hyperloom_multibranch.toolbox import resume_guard
from hyperloom_multibranch.toolbox.resume_guard import (
    BASELINE_ENVIRONMENT_FILE,
    BaselineNotReusable,
    MeasuredOn,
    check_baseline_reusable,
    infra_ended_keeps,
)
from hyperloom_multibranch.toolbox.submission_log import SUBMISSIONS_FILE
from hyperloom_multibranch.toolbox.tests.helpers import (
    IMAGE,
    OTHER_IMAGE,
    PARENT,
    Rig,
    ScriptedExecutor,
    build,
    failed,
    ok,
    req,
)

NODE = "node-a"
CARDS = (0, 1, 2, 3, 4, 5, 6, 7)


@pytest.fixture(autouse=True)
def _this_node_and_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """This process runs on :data:`NODE` in the image :data:`IMAGE`, as the session's first leg did."""
    monkeypatch.setenv("NODE_NAME", NODE)
    monkeypatch.setenv(IMAGE_REF_VAR, "registry.example/vllm:session")
    monkeypatch.setenv(IMAGE_DIGEST_VAR, IMAGE)


def _measured_toolbox(root: Path, executor: ScriptedExecutor | None = None) -> Rig:
    """A toolbox whose session directory is ``root``, its base launch frozen there and its tests measured on
    :data:`NODE`'s :data:`CARDS` in :data:`IMAGE`, as the local toolbox is built."""
    freeze_base(root, base_launch())
    rig = build(root, executor=executor)
    rig.toolbox.measured_on = MeasuredOn(root, NODE, IMAGE, len(CARDS))
    return rig


async def _baselined(root: Path) -> Rig:
    """A session directory whose toolbox measured the root's baseline."""
    rig = _measured_toolbox(root)
    await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.BASELINE))
    await rig.toolbox.drain()
    return rig


def _refusal(root: Path, cards: tuple[int, ...] = CARDS) -> str:
    with pytest.raises(BaselineNotReusable) as refused:
        check_baseline_reusable(root, cards)
    assert "\n" not in refused.value.reason
    return refused.value.reason


async def test_a_session_resumed_on_what_its_baseline_was_measured_on_reuses_it(tmp_path: Path) -> None:
    await _baselined(tmp_path)

    assert check_baseline_reusable(tmp_path, CARDS) is None
    recorded = json.loads((tmp_path / BASELINE_ENVIRONMENT_FILE).read_text())
    assert recorded == {
        "build": dataclasses.asdict(code_version()),
        "measurement_epoch": bootstrap.MEASUREMENT_EPOCH,
        "node": NODE,
        "image_digest": IMAGE,
        "cards": len(CARDS),
        "base_launch_digest": base_launch().digest(),
    }


def _tree(root: Path) -> Iterator[tuple[str, int, str]]:
    """Every file below ``root`` with its mode and content digest."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield str(path.relative_to(root)), path.stat().st_mode, hashlib.sha256(path.read_bytes()).hexdigest()


async def test_the_guard_writes_nothing_whether_it_passes_or_refuses(tmp_path: Path, monkeypatch) -> None:
    await _baselined(tmp_path)
    before, mtime = list(_tree(tmp_path)), os.stat(tmp_path).st_mtime_ns

    check_baseline_reusable(tmp_path, CARDS)
    monkeypatch.setenv("NODE_NAME", "node-b")
    _refusal(tmp_path)

    assert list(_tree(tmp_path)) == before and os.stat(tmp_path).st_mtime_ns == mtime


async def test_another_build_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _baselined(tmp_path)
    ran = code_version()
    monkeypatch.setattr(resume_guard, "code_version", lambda: CodeVersion(ran.package, "f" * 40))

    assert _refusal(tmp_path) == (
        f"the baseline was measured with build '{ran.package}@{ran.git_sha}' and this run has"
        f" '{ran.package}@{'f' * 40}'"
    )


async def test_a_measurement_epoch_change_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _baselined(tmp_path)
    monkeypatch.setattr(bootstrap, "MEASUREMENT_EPOCH", bootstrap.MEASUREMENT_EPOCH + 1)

    assert _refusal(tmp_path) == (
        f"the baseline was measured with measurement_epoch {bootstrap.MEASUREMENT_EPOCH - 1} and this run has"
        f" {bootstrap.MEASUREMENT_EPOCH}"
    )


async def test_another_node_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _baselined(tmp_path)
    monkeypatch.setenv("NODE_NAME", "node-b")

    assert _refusal(tmp_path) == "the baseline was measured with node 'node-a' and this run has 'node-b'"


async def test_another_image_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _baselined(tmp_path)
    monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)

    assert _refusal(tmp_path) == (
        f"the baseline was measured with image_digest {IMAGE!r} and this run has {OTHER_IMAGE!r}"
    )


async def test_an_image_this_run_cannot_name_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _baselined(tmp_path)
    monkeypatch.delenv(IMAGE_DIGEST_VAR)

    assert _refusal(tmp_path).startswith("the image this run measures in cannot be named: ")


async def test_another_card_count_is_refused(tmp_path: Path) -> None:
    await _baselined(tmp_path)

    assert _refusal(tmp_path, CARDS[:4]) == "the baseline was measured with cards 8 and this run has 4"


async def test_another_base_launch_is_refused(tmp_path: Path) -> None:
    await _baselined(tmp_path)
    base = base_launch()
    other = dataclasses.replace(base, args=(*base.args, "--enable-chunked-prefill"))
    freeze_base(tmp_path, other)

    assert _refusal(tmp_path) == (
        f"the baseline was measured with base_launch_digest {base.digest()!r} and this run has {other.digest()!r}"
    )


async def test_a_base_launch_changed_behind_the_session_s_back_is_refused(tmp_path: Path) -> None:
    await _baselined(tmp_path)
    launch = tmp_path / "base_launch.json"
    launch.write_text(launch.read_text().replace("0.8", "0.9"))

    assert _refusal(tmp_path).startswith("the session's frozen base launch cannot be read: ")


async def test_a_session_whose_toolbox_recorded_nothing_is_refused(tmp_path: Path) -> None:
    """A session from before the record: its toolbox measured the baseline and recorded nothing."""
    freeze_base(tmp_path, base_launch())
    rig = build(tmp_path)
    await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.BASELINE))
    await rig.toolbox.drain()

    reason = _refusal(tmp_path)

    assert reason.startswith(f"{tmp_path / BASELINE_ENVIRONMENT_FILE} does not exist: ")


@pytest.mark.parametrize("dropped", [("node",), ("cards", "measurement_epoch")])
async def test_a_record_an_older_build_wrote_without_a_field_is_refused_naming_it(
    tmp_path: Path, dropped: tuple[str, ...]
) -> None:
    """The record as a build that recorded fewer fields wrote it: the toolbox's record less ``dropped``."""
    await _baselined(tmp_path)
    path = tmp_path / BASELINE_ENVIRONMENT_FILE
    document = json.loads(path.read_text())
    path.write_text(json.dumps({name: value for name, value in document.items() if name not in dropped}))

    names = ", ".join(name for name in resume_guard.RECORDED_FIELDS if name in dropped)
    assert _refusal(tmp_path) == f"{path} does not record {names}, so it cannot be compared"


async def test_the_record_is_written_once_by_the_first_baseline_and_kept(tmp_path: Path) -> None:
    rig = _measured_toolbox(tmp_path)
    await rig.toolbox.submit_test(req("b1"))
    assert not (tmp_path / BASELINE_ENVIRONMENT_FILE).exists(), "only a baseline test records"

    await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.BASELINE))
    first = (tmp_path / BASELINE_ENVIRONMENT_FILE).read_bytes()
    rig.toolbox.measured_on = MeasuredOn(tmp_path, "node-b", OTHER_IMAGE, 2)
    await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.BASELINE, server_args="--again"))
    await rig.toolbox.drain()

    assert (tmp_path / BASELINE_ENVIRONMENT_FILE).read_bytes() == first
    assert check_baseline_reusable(tmp_path, CARDS) is None


async def test_a_record_that_cannot_be_written_leaves_the_baseline_test_accepted(tmp_path: Path, caplog) -> None:
    rig = _measured_toolbox(tmp_path)
    rig.toolbox.measured_on = MeasuredOn(tmp_path / "base_launch.json", NODE, IMAGE, len(CARDS))

    request_id = await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.BASELINE))
    await rig.toolbox.drain()

    assert rig.toolbox._log.get(request_id).settled  # type: ignore[union-attr]
    assert "cannot be recorded, so a resume will refuse" in caplog.text
    assert _refusal(tmp_path).endswith(
        "the session recorded nothing of what its baseline was measured on"
        " (it predates the record, or its tests did not run on this machine's cards)"
    )


# --- the keeps infra ended ----------------------------------------------------------------------------------------


def _script(behaviours: dict[str, object]) -> ScriptedExecutor:
    """An executor that runs each test as ``behaviours`` says by its server arguments, else measures it."""

    def run(job):  # type: ignore[no-untyped-def]
        marker = job.request.delta.config.server_args
        behaviour = behaviours.get(marker)
        return behaviour(job) if callable(behaviour) else ok(job)

    return ScriptedExecutor([run])


async def test_the_keeps_infra_ended_before_measuring_are_read_back_with_their_branch_commit_and_reason(
    tmp_path: Path,
) -> None:
    infra = lambda job: failed(job, FailureClass.INFRA, reason="image pull failed")  # noqa: E731
    candidate = lambda job: failed(job, FailureClass.CANDIDATE, reason="server crashed")  # noqa: E731
    executor = _script({"--k1": infra, "--k2": ok, "--k3": candidate, "--k4": infra, "": infra})
    rig = build(tmp_path, executor=executor)
    first = await rig.toolbox.submit_test(req("b1", server_args="--k1"))
    await rig.toolbox.submit_test(req("b1", server_args="--k2"))
    await rig.toolbox.submit_test(req("b2", server_args="--k3"))
    integration = await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, server_args="--k4"))
    health = await rig.toolbox.submit_test(req("b2", purpose=Purpose.HEALTH))
    await rig.toolbox.drain()
    log = (tmp_path / SUBMISSIONS_FILE).read_bytes()

    keeps = infra_ended_keeps(tmp_path)

    assert [(keep.branch_id, keep.commit, keep.request_id, keep.reason) for keep in keeps] == [
        ("b1", PARENT, first, "infra: 4 attempts failed for infrastructure reasons: image pull failed")
    ]
    assert keeps[0].request.delta.config.server_args == "--k1" and keeps[0].request.request_id == first
    assert integration not in {keep.request_id for keep in keeps} and health not in {k.request_id for k in keeps}
    assert (tmp_path / SUBMISSIONS_FILE).read_bytes() == log, "read only"


async def test_a_keep_submitted_again_is_judged_by_its_last_test(tmp_path: Path) -> None:
    tries: dict[str, int] = {}

    def infra_once(job):  # type: ignore[no-untyped-def]
        marker = job.request.delta.config.server_args
        tries[marker] = tries.get(marker, 0) + 1
        return failed(job, FailureClass.INFRA, reason="node lost") if tries[marker] <= 4 else ok(job)

    infra = lambda job: failed(job, FailureClass.INFRA, reason="node lost")  # noqa: E731
    rig = build(tmp_path, executor=_script({"--a": infra_once, "--b": infra}))
    await rig.toolbox.submit_test(req("b1", server_args="--a"))
    await rig.toolbox.submit_test(req("b1", server_args="--b"))
    await rig.toolbox.drain()
    await rig.toolbox.submit_test(req("b1", server_args="--a"))
    again = await rig.toolbox.submit_test(req("b1", server_args="--b"))
    await rig.toolbox.drain()

    assert [keep.request_id for keep in infra_ended_keeps(tmp_path)] == [again]


async def test_a_keep_still_waiting_or_withdrawn_is_no_keep_that_infra_ended(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    rig = build(tmp_path, executor=executor)
    waiting = await rig.toolbox.submit_test(req("b1", server_args="--waits"))
    withdrawn = await rig.toolbox.submit_test(req("b2", server_args="--withdrawn"))
    await rig.toolbox.cancel(withdrawn)

    assert infra_ended_keeps(tmp_path) == []
    assert rig.toolbox._log.get(waiting).waiting  # type: ignore[union-attr]
    executor.gate.set()
    await rig.toolbox.drain()


async def test_a_torn_last_line_is_skipped_and_left_on_the_log(tmp_path: Path) -> None:
    infra = lambda job: failed(job, FailureClass.INFRA, reason="pull failed")  # noqa: E731
    rig = build(tmp_path, executor=_script({"--a": infra}))
    keep = await rig.toolbox.submit_test(req("b1", server_args="--a"))
    await rig.toolbox.drain()
    path = tmp_path / SUBMISSIONS_FILE
    with path.open("a") as log:
        log.write('{"type": "accepted", "request_id": "torn')
    torn = path.read_bytes()

    assert [ref.request_id for ref in infra_ended_keeps(tmp_path)] == [keep]
    assert path.read_bytes() == torn


def test_a_session_whose_toolbox_logged_nothing_has_no_keeps(tmp_path: Path) -> None:
    assert infra_ended_keeps(tmp_path) == []
    assert not (tmp_path / SUBMISSIONS_FILE).exists()


async def test_a_measured_result_is_read_as_its_validity_says(tmp_path: Path) -> None:
    """The reader takes the validity the toolbox logged: a result it logged as measured is no infra end."""
    rig = build(tmp_path, executor=_script({}))
    await rig.toolbox.submit_test(req("b1", server_args="--measured"))
    await rig.toolbox.drain()
    (entry,) = rig.toolbox._log.entries()
    assert entry.outcome is not None and entry.outcome["result"]["validity"] == Validity.SUCCEEDED.value

    assert infra_ended_keeps(tmp_path) == []
