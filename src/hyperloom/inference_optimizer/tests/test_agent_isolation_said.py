# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local run that cannot isolate its agents says so: one WARNING when the optimize command starts, before any agent
runs, and a line in the session's final report (the full one and the safety net), with the launches that ran
unconfined."""

from __future__ import annotations

import logging

import pytest

from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import NoIsolation, unconfined_launch
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.orchestrator.actions.executors import report as rp

_ABSENT = "bubblewrap: not found on PATH"


def test_the_optimize_command_announces_how_agents_are_isolated_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said: list[str] = []

    def announce() -> None:
        said.append("announced")
        raise SystemExit(98)

    monkeypatch.setattr(agent_isolation, "announce_agent_isolation", announce)

    with pytest.raises(SystemExit) as stopped:
        cli.main(["optimize"])

    assert stopped.value.code == 98 and said == ["announced"]


def test_a_local_run_without_bubblewrap_warns_once_at_start_up(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        report = agent_isolation.announce_agent_isolation()
        unconfined_launch(NoIsolation(report.reasons[:2]))

    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        "agents run unconfined on this host, as before"
    ]
    assert _ABSENT in caplog.records[0].getMessage()


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch):
    from hyperloom.orchestrator.state.shared_state import SharedState

    monkeypatch.setattr(rp, "_platform_fingerprint", lambda *_: {"status": "not_recorded"})
    found = SharedState()
    found.framework = "vllm"
    return found


def _said(count: int) -> tuple[str, ...]:
    report = agent_isolation.announce_agent_isolation()
    for _ in range(count):
        unconfined_launch(NoIsolation(report.reasons[:2]))
    lines = agent_isolation.session_isolation_lines()
    assert lines[0] == report.line() and len(lines) == (2 if count else 1)
    return lines


def test_the_final_report_carries_the_isolation_line_and_the_unconfined_launches(state) -> None:
    lines = _said(2)

    summary = rp._build_summary_dict(state, {}, [])
    rendered = rp._format_md(summary)

    assert summary["agent_isolation"] == list(lines)
    assert lines[1].startswith("agent isolation: 2 agent launches ran unconfined: ")
    assert all(f"- {line}\n" in rendered for line in lines)


def test_a_run_that_said_nothing_reports_no_isolation_line(state) -> None:
    summary = rp._build_summary_dict(state, {}, [])
    assert summary["agent_isolation"] == []
    assert "agent isolation" not in rp._format_md(summary)


def test_the_safety_net_final_report_carries_the_isolation_line_too(tmp_path) -> None:
    lines = _said(0)

    text = exporter.write_minimal_final_report(tmp_path).read_text(encoding="utf-8")

    assert f"- {lines[0]}\n" in text and "specialists none, sandboxes none" in text
