# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session starts the workspace's Experience KB service outside its own tree and reads the schemas its tests and
learnings are published under."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

import pytest
from hyperloom_kb import RemoteExperienceKB
from hyperloom_kb.collect import load_mapping

from hyperloom_multibranch.blackboard.knowledge import experience, learnings
from hyperloom_multibranch.blackboard.knowledge.recall import SchemaReaders
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.scheduling import experience_kb
from hyperloom_multibranch.scheduling.experience_kb import (
    ENSURE_COMMAND,
    ExperienceReaders,
    experience_readers,
    start_experience_kb,
)

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
