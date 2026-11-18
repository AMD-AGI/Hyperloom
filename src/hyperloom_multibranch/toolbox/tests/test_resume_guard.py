# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The baseline-reuse guard of a hosted restart, over session directories the toolbox wrote.

Every directory here is written by the toolbox itself: the base launch frozen as a session freezes it, and the record of
what the baseline is measured on written when the toolbox accepts the first baseline test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
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
    check_hosted_baseline_reusable,
)
from hyperloom_multibranch.toolbox.tests.helpers import (
    IMAGE,
    OTHER_IMAGE,
    Rig,
    ScriptedExecutor,
    build,
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
        check_hosted_baseline_reusable(root, len(cards))
    assert "\n" not in refused.value.reason
    return refused.value.reason


async def test_a_session_resumed_on_what_its_baseline_was_measured_on_reuses_it(tmp_path: Path) -> None:
    await _baselined(tmp_path)

    assert check_hosted_baseline_reusable(tmp_path, len(CARDS)) is None
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

    check_hosted_baseline_reusable(tmp_path, len(CARDS))
    monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)
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


def _hosted_refusal(root: Path, cards: int = len(CARDS)) -> str:
    with pytest.raises(BaselineNotReusable) as refused:
        check_hosted_baseline_reusable(root, cards)
    assert "\n" not in refused.value.reason
    return refused.value.reason


async def test_a_hosted_restart_on_another_node_reuses_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rescheduled application runs on another node by design; its tests run in the grant, not there."""
    await _baselined(tmp_path)
    before = list(_tree(tmp_path))
    monkeypatch.setenv("NODE_NAME", "node-b")

    assert check_hosted_baseline_reusable(tmp_path, len(CARDS)) is None
    assert list(_tree(tmp_path)) == before, "the guard writes nothing"


@pytest.mark.parametrize("changed", ["build", "measurement_epoch", "image_digest", "cards", "base_launch_digest"])
async def test_a_hosted_restart_onto_anything_else_its_baseline_was_measured_on_is_refused_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    await _baselined(tmp_path)
    monkeypatch.setenv("NODE_NAME", "node-b")
    cards = len(CARDS)
    if changed == "build":
        ran = code_version()
        monkeypatch.setattr(resume_guard, "code_version", lambda: CodeVersion(ran.package, "f" * 40))
    elif changed == "measurement_epoch":
        monkeypatch.setattr(bootstrap, "MEASUREMENT_EPOCH", bootstrap.MEASUREMENT_EPOCH + 1)
    elif changed == "image_digest":
        monkeypatch.setenv(IMAGE_DIGEST_VAR, OTHER_IMAGE)
    elif changed == "cards":
        cards = 4
    else:
        base = base_launch()
        freeze_base(tmp_path, dataclasses.replace(base, args=(*base.args, "--enable-chunked-prefill")))

    assert _hosted_refusal(tmp_path, cards).startswith(f"the baseline was measured with {changed} ")


async def test_a_hosted_restart_of_a_session_that_recorded_nothing_is_refused(tmp_path: Path) -> None:
    freeze_base(tmp_path, base_launch())

    assert _hosted_refusal(tmp_path).startswith(f"{tmp_path / BASELINE_ENVIRONMENT_FILE} does not exist: ")


async def test_a_hosted_restart_that_may_not_read_the_frozen_base_launch_is_refused_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another account of the group that the earlier leg's base launch is refused to: the restart finishes instead."""
    await _baselined(tmp_path)

    def refused(directory: Path) -> object:
        raise PermissionError(13, "Permission denied", str(directory / "base_launch.json"))

    monkeypatch.setattr("hyperloom.orchestrator.state.base_launch.frozen_base", refused)

    assert _hosted_refusal(tmp_path).startswith("the session's frozen base launch cannot be read: ")


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
    assert check_hosted_baseline_reusable(tmp_path, len(CARDS)) is None


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
