# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record controller-only pytest events for the full shell comparison."""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Activate only when an external, previously unused event path is supplied."""
    destination = os.environ.get("CI_SHELL_FULL_EVENTS")
    if not destination or hasattr(config, "workerinput"):
        return
    path = Path(destination)
    if not path.is_absolute():
        raise pytest.UsageError("CI_SHELL_FULL_EVENTS must be an absolute path")
    path = path.resolve()
    if path.is_relative_to(config.rootpath.resolve()):
        raise pytest.UsageError("CI_SHELL_FULL_EVENTS must be outside the subject root")
    recorder = EventRecorder(config, path)
    config.add_cleanup(recorder.close)
    config.pluginmanager.register(recorder, "ci-shell-full-events")


class EventRecorder:
    """Append one flushed JSON line per event, never sharing a file with workers."""

    def __init__(self, config: pytest.Config, path: Path) -> None:
        self.config = config
        self.stream = path.open("x", encoding="utf-8", buffering=1, newline="\n")

    def close(self) -> None:
        self.stream.close()

    def emit(self, event: str, **fields: Any) -> None:
        self.stream.write(json.dumps({"event": event, **fields}, ensure_ascii=True, separators=(",", ":")) + "\n")

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.emit(
            "session_start",
            schema_version=1,
            collectonly=bool(self.config.getoption("collectonly")),
            rootpath=str(self.config.rootpath.resolve()),
            workers=self.config.getoption("numprocesses", default=0) or 0,
        )

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        if not self.config.getoption("numprocesses", default=0):
            self.emit("collection", worker_id=None, nodeids=[item.nodeid for item in session.items])

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: Sequence[str]) -> None:
        # Workers have already applied pytest-split before publishing these ids.
        self.emit("collection", worker_id=node.gateway.id, nodeids=list(ids))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        wasxfail = getattr(report, "wasxfail", None)
        skipreason = None
        if report.skipped:
            if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
                skipreason = str(report.longrepr[2])
            elif wasxfail is not None:
                skipreason = str(wasxfail)
            else:
                skipreason = _detail(report.longrepr)
        self.emit(
            "report",
            nodeid=report.nodeid,
            when=report.when,
            outcome=report.outcome,
            wasxfail=None if wasxfail is None else str(wasxfail),
            skipreason=skipreason,
            longrepr=_detail(report.longrepr),
            duration=report.duration,
        )

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.emit("collection_error", nodeid=report.nodeid, longrepr=_detail(report.longrepr))

    def pytest_internalerror(self, excrepr: Any, excinfo: Any) -> None:
        if not self.stream.closed:
            self.emit("internal_error", longrepr=_detail(excrepr))

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        self.emit("worker_down", worker_id=node.gateway.id, error=_detail(error))

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> Generator[None, Any, None]:
        outcome = yield
        try:
            # Run after coverage and other finish hooks have updated exitstatus.
            if outcome.excinfo is None:
                self.emit("session_finish", exitstatus=int(session.exitstatus), testscollected=session.testscollected)
        finally:
            self.close()


def _detail(value: Any) -> str | None:
    return None if value is None else str(value)[:1000]
