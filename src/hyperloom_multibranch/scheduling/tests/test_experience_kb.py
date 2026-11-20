# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session starts the workspace's Experience KB service outside its own tree and reads the schemas its tests and
learnings are published under."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hyperloom_kb import RemoteExperienceKB
from hyperloom_kb.collect import load_mapping

from hyperloom_multibranch.blackboard.knowledge import experience, learnings
from hyperloom_multibranch.blackboard.knowledge.recall import SchemaReaders
from hyperloom_multibranch.blackboard.rsi.session import CloseTargets
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.scheduling import experience_kb
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.experience_kb import (
    ENSURE_COMMAND,
    ExperienceReaders,
    close_targets,
    experience_readers,
    start_experience_kb,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, adopting_orphans

LOOPBACK = {"HYPERLOOM_KB_URL": "http://127.0.0.1:24680", "HYPERLOOM_KB_TOKEN": "workspace-token"}
LOGGER = "hyperloom_multibranch.scheduling.experience_kb"


def _schema(mapping: Any) -> str:
    return load_mapping(mapping).declaration.schema_ref


class _Runs:
    """Stands in for ``subprocess.run``: records each argv and answers with ``answer``."""

    def __init__(self, answer: subprocess.CompletedProcess[str] | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _ran(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.mark.parametrize("env", [{}, {"HYPERLOOM_KB_URL": "https://kb.example.com"}])
def test_no_service_is_started_without_a_loopback_url(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _Runs(_ran(0))
    monkeypatch.setattr(experience_kb.subprocess, "run", runs)

    start_experience_kb(env)

    assert runs.calls == []


def test_a_loopback_service_is_started_by_a_process_of_its_own_that_restarts_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = _Runs(_ran(0, "Experience KB service started at http://127.0.0.1:24680: 3 Experiences"))
    monkeypatch.setattr(experience_kb.subprocess, "run", runs)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        start_experience_kb(LOOPBACK)

    ((argv, kwargs),) = runs.calls
    assert argv == [sys.executable, *ENSURE_COMMAND] and argv[-1] == "--no-restart"
    assert kwargs["env"] == LOOPBACK and kwargs["check"] is False
    assert "3 Experiences" in caplog.text


@pytest.mark.parametrize(
    "answer",
    [
        _ran(1, stderr="Experience KB service failed: exited"),
        OSError("no interpreter"),
        subprocess.TimeoutExpired("x", 1),
    ],
)
def test_a_service_that_does_not_start_is_logged_and_the_session_goes_on(
    answer: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(experience_kb.subprocess, "run", _Runs(answer))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start_experience_kb(LOOPBACK)

    assert "the session runs without it" in caplog.text


def test_a_service_left_serving_with_other_settings_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stale = "Experience KB service left serving with other settings: it was started with other settings"
    monkeypatch.setattr(experience_kb.subprocess, "run", _Runs(_ran(0, "already running", stale)))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        start_experience_kb(LOOPBACK)

    assert stale in caplog.text


#: Stands in for ``ensure``: starts a long-lived process in a session of its own and off its pipes, as the KB service
#: is, and exits.
_SPAWNS_A_SERVICE = (
    "-c",
    "import subprocess, sys; "
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True, "
    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
    "print(f'service pid {p.pid}')",
)


async def test_a_started_service_is_no_process_the_session_stop_would_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(experience_kb, "ENSURE_COMMAND", _SPAWNS_A_SERVICE)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        start_experience_kb(LOOPBACK)
    match = re.search(r"service pid (\d+)", caplog.text)
    assert match is not None
    service = int(match.group(1))
    child = None
    try:
        with adopting_orphans():
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            leftovers = {record.pid for record in await ProcessRegistry(tmp_path / "pids.jsonl", Clock()).leftovers()}
        # A child the scheduler started itself is one; the service started outside its tree is not.
        assert child.pid in leftovers and service not in leftovers
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(service, signal.SIGKILL)
        if child is not None:
            child.kill()
            child.wait()


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionLayout:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
    monkeypatch.delenv("HYPERLOOM_KB_HOME", raising=False)
    return create_session(tmp_path / "workspace", "model", Clock())


def test_without_an_experience_kb_close_writes_to_none(layout: SessionLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)

    targets = close_targets(layout)

    assert targets == CloseTargets(receipts=layout.reports, judge_transcript=layout.judge, blobs=layout.blobs)


def test_close_writes_every_schema_through_one_client_spooled_beside_the_workspaces_service(
    layout: SessionLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in LOOPBACK.items():
        monkeypatch.setenv(name, value)

    targets = close_targets(layout)

    writers = (targets.experience_kb, targets.learning_kb, targets.multibranch())
    assert all(isinstance(writer, RemoteExperienceKB) for writer in writers)
    assert [writer.schema_ref for writer in writers] == [
        _schema(experience.MAPPING_PATH),
        _schema(learnings.MAPPING_PATH),
        _schema(experience.MULTIBRANCH_MAPPING_PATH),
    ]
    assert len({id(writer.client) for writer in writers}) == 1
    config = targets.experience_kb.client.config
    assert (config.spool_root, config.timeout_seconds) == (tmp_path / "workspace" / "experience-kb" / "spool", 30.0)
    assert (targets.receipts, targets.judge_transcript, targets.blobs) == (layout.reports, layout.judge, layout.blobs)


def test_close_with_a_url_but_no_token_is_logged_and_leaves_the_steps_to_fail(
    layout: SessionLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", LOOPBACK["HYPERLOOM_KB_URL"])
    monkeypatch.delenv("HYPERLOOM_KB_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        targets = close_targets(layout)

    assert targets.experience_kb is None and targets.multibranch() is None
    assert "HYPERLOOM_KB_TOKEN" in caplog.text


def test_without_an_experience_kb_there_are_no_readers(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        unset = experience_readers(WorkloadKind.INFERENCEX, {})
        tokenless = experience_readers(WorkloadKind.AGENTX, {"HYPERLOOM_KB_URL": LOOPBACK["HYPERLOOM_KB_URL"]})

    assert unset == tokenless == ExperienceReaders()
    assert "HYPERLOOM_KB_TOKEN" in caplog.text


def test_an_agentx_session_reads_its_tests_under_its_own_schema() -> None:
    readers = experience_readers(WorkloadKind.AGENTX, LOOPBACK)

    assert isinstance(readers.tests, RemoteExperienceKB)
    assert readers.tests.schema_ref == _schema(experience.MULTIBRANCH_MAPPING_PATH)
    assert readers.learnings is not None and readers.learnings.schema_ref == _schema(learnings.MAPPING_PATH)


def test_an_inferencex_session_reads_its_tests_under_both_schemas_kernel_changes_included() -> None:
    readers = experience_readers(WorkloadKind.INFERENCEX, LOOPBACK)

    assert isinstance(readers.tests, SchemaReaders)
    assert [reader.schema_ref for reader in readers.tests.readers] == [
        _schema(experience.MAPPING_PATH),
        _schema(experience.MULTIBRANCH_MAPPING_PATH),
    ]
    assert readers.learnings is not None and readers.learnings.schema_ref == _schema(learnings.MAPPING_PATH)
