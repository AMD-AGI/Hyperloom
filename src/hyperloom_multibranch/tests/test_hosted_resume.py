# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A restart of a hosted run over a session whose baseline landed: it resumes that session up to the deadline the first
start set when the baseline still measures what a test measured now would, and finishes it otherwise.

The host restarts a preempted run on whichever node has room, and the application may then run as another account of
the run's group: neither is compared, as the session's tests run in the grant's sandboxes. The build, the measurement
epoch, the image, the base launch and the runner template's cards are, as the toolbox recorded them when it accepted
the session's first baseline test.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import hyperloom_multibranch
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import freeze_base
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
from hyperloom_multibranch import cli
from hyperloom_multibranch.scheduling import hosted_run
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS, FinalWork
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.tests.test_cli import (  # noqa: F401 - fixtures of the hosted entry
    ANOTHER_POD,
    EXPLORE_CAPS,
    GRANT,
    IMAGE,
    IMAGE_VARS,
    RESUMED_TO,
    Host,
    _ended_before_final,
    _killed,
    _manifest,
    _settings,
    agent_binaries,
    agent_cli,
    eight_cards,
    host,
    ready_environment,
)
from hyperloom_multibranch.toolbox.resume_guard import BASELINE_ENVIRONMENT_FILE, MeasuredOn

#: The node the earlier start ran on; the restart runs on whatever node this is.
EARLIER_NODE = "preempted-node"
#: How long the session the earlier start began has left.
LEFT_SEC = 7200.0


def preempted_session(
    root: str, image_digest: str, left_sec: float = LEFT_SEC, in_grant: bool = True
) -> dict[str, float]:
    """A session measured in a grant's sandboxes (one that cloned no runner source unless ``in_grant``), as the start
    it ran in leaves it when the host preempts that start after its baseline landed: the toolbox recorded what
    the baseline was measured on (on :data:`EARLIER_NODE`, in ``image_digest``), its manifest has ``left_sec`` of its
    budget left, and it wrote no report. The deadline."""

    layout = _ended_before_final(Path(root))
    deadline = time.time() + left_sec
    if in_grant:
        layout.runner_source.mkdir()
    _manifest(layout, started_at=deadline - 2 * LEFT_SEC, deadline=deadline)
    _killed(layout, ANOTHER_POD)
    freeze_base(layout.toolbox, base_launch())
    tp = _settings().workload.tp
    assert MeasuredOn(layout.toolbox, EARLIER_NODE, image_digest, tp).record()
    return {"deadline": deadline}


def _session_of(host: Host, tmp_path: Path, **preempted: Any) -> tuple[SessionLayout, float]:
    root = tmp_path / "MB-preempted-session"
    deadline = preempted_session(str(root), IMAGE[1], **preempted)["deadline"]
    (host.state_dir / "session.json").write_text(json.dumps({"session_dir": str(root)}), encoding="utf-8")
    return SessionLayout(root), deadline


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """What the hosted entry warned of, the storage checks' setgid warnings about the test's own directories aside."""
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.getMessage().startswith("hosted run: ")
    ]


@pytest.fixture
def rescheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The restart runs on another node than the earlier start did."""
    monkeypatch.setenv("NODE_NAME", "rescheduled-node")


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_on_another_node_resumes_the_session_up_to_the_first_starts_deadline(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, deadline = _session_of(host, tmp_path)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()], "carried on with its manifest's settings, not the spec's"
    assert RESUMED_TO == [deadline], "the budget counts from the original start: no downtime is given back"
    assert _warnings(caplog) == []
    assert len(host.stops) == 1, "the grant stop is armed for the resume as for every start"


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_resumes_the_session_with_explore_capped_by_the_grant_the_first_start_ran_in(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path
) -> None:
    _session_of(host, tmp_path)
    created = datetime.fromisoformat(os.environ["HYPERLOOM_RUN_CREATED_AT"]).timestamp()

    assert cli.main(["hosted", "run"]) == 0

    [cap] = EXPLORE_CAPS
    stop = created + GRANT["max_runtime_sec"] - MB_CLOSE_GRACE_CAP_SECONDS
    assert cap is not None
    work = FinalWork(6000.0, 4500.0, 1500.0, 1)
    assert cap(work) == pytest.approx(
        stop - hosted_run.GRANT_MARGIN_SECONDS - hosted_run.final_reserve_seconds(work), abs=1.0
    ), "the grant counts from the run's creation, which the restart reads as the first start did"


def _changed_image(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(IMAGE_VARS[1], "sha256:" + "3" * 64)
    return "image_digest"


def _changed_epoch(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(bootstrap, "MEASUREMENT_EPOCH", bootstrap.MEASUREMENT_EPOCH + 1)
    return "measurement_epoch"


@pytest.mark.usefixtures("ready_environment", "rescheduled")
@pytest.mark.parametrize("change", [_changed_image, _changed_epoch], ids=["image", "epoch"])
def test_a_restart_onto_another_image_or_epoch_finishes_the_session_with_one_warning_naming_it(
    host: Host,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    change,  # type: ignore[no-untyped-def]
) -> None:
    layout, _ = _session_of(host, tmp_path)
    field = change(monkeypatch)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()] and RESUMED_TO == [], "finished as before, not resumed"
    [warning] = _warnings(caplog)
    assert warning.getMessage().startswith(f"hosted run: finishing {layout.root} instead of resuming it: ")
    assert f"the baseline was measured with {field} " in warning.getMessage()
    assert host.exit_document()["final"] is True


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_past_the_first_starts_deadline_finishes_the_session_with_one_warning(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout, _ = _session_of(host, tmp_path, left_sec=-60.0)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()] and RESUMED_TO == []
    [warning] = _warnings(caplog)
    assert "used its budget, which ended" in warning.getMessage()


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_of_a_session_its_toolbox_recorded_nothing_for_finishes_it(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A session an older build began, before the toolbox recorded what its baseline was measured on."""
    layout, _ = _session_of(host, tmp_path)
    (layout.toolbox / BASELINE_ENVIRONMENT_FILE).unlink()

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()] and RESUMED_TO == []
    [warning] = _warnings(caplog)
    assert "recorded nothing of what its baseline was measured on" in warning.getMessage()


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_of_a_session_not_measured_in_the_grant_finishes_it(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _session_of(host, tmp_path, in_grant=False)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()] and RESUMED_TO == []
    [warning] = _warnings(caplog)
    assert "whose tests did not run in the grant's sandboxes" in warning.getMessage()


@pytest.mark.usefixtures("ready_environment", "rescheduled")
def test_a_restart_that_cannot_read_what_the_resume_checks_finishes_the_session_rather_than_crash(
    host: Host,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier leg's file this account may not read, as another account of the group meets one an older build left
    owner-only: the finish runs and says so itself, as before, instead of the start crashing with no ``exit.json``,
    which would have the host start the run again and again."""
    layout, _ = _session_of(host, tmp_path)

    def unreadable(persisted: object) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(layout.blackboard))

    monkeypatch.setattr(hosted_run, "require_resumable", unreadable)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()] and RESUMED_TO == [], "finished, not resumed"
    [warning] = _warnings(caplog)
    assert warning.getMessage().startswith(f"hosted run: finishing {layout.root} instead of resuming it: ")
    assert str(layout.blackboard) in warning.getMessage()
    assert host.exit_document()["final"] is True


# -- a reschedule onto another account of the run's group ------------------------------------------------------------


def restart_leg(state_dir: str, image: list[str]) -> dict[str, object]:
    """What the restart of the run whose state is in ``state_dir`` carries the session on with, run as the account this
    process is in ``image``: the leg and the deadline it resumes up to."""

    os.environ.update(zip(IMAGE_VARS, image))
    os.environ["NODE_NAME"] = "rescheduled-node"
    persisted = hosted_run.take_over_earlier_session(Path(state_dir)).finish
    assert persisted is not None
    leg = hosted_run.hosted_restart_leg(persisted, cli.final_only, cli.resume)
    if leg is cli.final_only:
        return {"leg": "final_only", "deadline": None}
    assert isinstance(leg, partial) and leg.func is cli.resume
    return {"leg": "resume", "deadline": leg.keywords["deadline"]}


def _as_account(accounts: Accounts, uid: int, call: str) -> dict[str, object]:
    source = Path(hyperloom_multibranch.__file__).parents[1]
    code = (
        f"import json, sys, tempfile\nsys.path.insert(0, {str(source)!r})\n"
        "os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()  # this account has no home of its own\n"
        "from hyperloom_multibranch.tests import test_hosted_resume as here\n"
        f"print(json.dumps(here.{call}))"
    )
    return json.loads(accounts.run(uid, code, umask=0o002).splitlines()[-1])


@pytest.mark.parametrize("image", [IMAGE[1], "sha256:" + "3" * 64], ids=["same-image", "another-image"])
def test_a_restart_as_another_account_of_the_group_resumes_what_one_account_preempted(image: str) -> None:
    with other_accounts() as accounts:
        source = Path(hyperloom_multibranch.__file__).parents[1]
        if any(not directory.stat().st_mode & stat.S_IXOTH for directory in (source, *source.parents)):
            pytest.skip(f"another account cannot reach this checkout's {source}")
        if accounts.python != sys.executable:
            pytest.skip(f"another account cannot run {sys.executable}, whose environment holds the project's packages")
        workspace, state_dir = accounts.shared_dir("workspace"), accounts.shared_dir("state")
        root = workspace / "MB-preempted-session"
        left = _as_account(accounts, accounts.first, f"preempted_session({str(root)!r}, {IMAGE[1]!r})")
        (state_dir / "session.json").write_text(json.dumps({"session_dir": str(root)}), encoding="utf-8")
        record = root / "MB-toolbox" / BASELINE_ENVIRONMENT_FILE
        assert record.stat().st_uid == accounts.first

        leg = _as_account(accounts, accounts.second, f"restart_leg({str(state_dir)!r}, {[IMAGE[0], image]!r})")

        if image == IMAGE[1]:
            assert leg == {"leg": "resume", "deadline": left["deadline"]}, "resumed, up to the first start's deadline"
        else:
            assert leg == {"leg": "final_only", "deadline": None}
