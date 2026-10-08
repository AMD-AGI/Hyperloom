# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement setup-command replay."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors import _setup_replay
from hyperloom.orchestrator.actions.executors._setup_replay import (
    SETUP_CMD_MAX,
    resolve_setup_commands,
    run_setup_commands,
    setup_report_fields,
)
from hyperloom.orchestrator.enablement.recipe.setup_ledger import command_digest


def _proposed(*commands: str) -> list[tuple[str, str]]:
    return [(cmd, "proposed") for cmd in commands]


def _run(commands: list[tuple[str, str]], tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": tmp_path,
        "log_dir": tmp_path / "logs",
        "round_task_id": "r1",
        "seq_start": 0,
        "on_execution": lambda _row: None,
    }
    kwargs.update(overrides)
    return run_setup_commands(commands, **kwargs)


def _succeed(monkeypatch, ran: list[str] | None = None) -> None:
    def _fake_run(cmd, *args, **kwargs):
        if ran is not None:
            ran.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)


def _forbid_execution(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: pytest.fail("a rejected command was executed"))


def test_resolve_dedups_base_then_done_and_keeps_the_inherited_source():
    got = resolve_setup_commands(
        params={"enablement_setup_commands": ["pip install a", " pip install b ", ""]},
        done_payload={"setup_commands": ["pip install b", "pip install c", None]},
    )
    assert got == [("pip install a", "inherited"), ("pip install b", "inherited"), ("pip install c", "proposed")]


def test_resolve_caps_the_command_count():
    got = resolve_setup_commands(
        params={"enablement_setup_commands": [f"pip install p{i}" for i in range(SETUP_CMD_MAX + 5)]},
        done_payload={"setup_commands": ["pip install late"]},
    )
    assert len(got) == SETUP_CMD_MAX
    assert ("pip install late", "proposed") not in got


def test_resolve_ignores_a_source_that_is_not_a_list():
    got = resolve_setup_commands(
        params={"enablement_setup_commands": "pip install a"},
        done_payload={"setup_commands": ["pip install b"]},
    )
    assert got == [("pip install b", "proposed")]


def test_run_setup_commands_skips_non_allowlisted(tmp_path: Path, monkeypatch):
    """A non-allowlisted command is skipped (never executed); allowlisted runs."""
    ran: list[str] = []
    _succeed(monkeypatch, ran)

    out = _run(_proposed("pip install -U transformers", "rm -rf /tmp/x"), tmp_path)

    assert out["applied"] == ["pip install -U transformers"]
    assert out["skipped"] == ["rm -rf /tmp/x"]
    assert ran == ["pip install -U transformers"]
    assert (tmp_path / "logs" / "enablement_setup.log").exists()


def test_run_setup_commands_stops_between_commands_on_cancel(tmp_path: Path, monkeypatch):
    """Cancel is cooperative between commands; an in-flight subprocess.run is not killed."""
    ran: list[str] = []
    scope = CancelScope()

    def _fake_run(cmd, *args, **kwargs):
        ran.append(cmd)
        scope.cancel(reason="test")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with use_cancel_scope(scope):
        out = _run(_proposed("pip install -U transformers", "pip install -U torch"), tmp_path)
    assert ran == ["pip install -U transformers"]
    assert out["applied"] == ["pip install -U transformers"]
    assert out["failed"] == []


def test_run_setup_commands_records_one_row_per_attempted_command(tmp_path: Path, monkeypatch):
    """Occurrence identity needs every attempt, not just the ones that worked.

    ``setup_commands`` dedupes to one string per command, so the ledger is the
    only place a failed or skipped execution is recorded at all.
    """
    outcomes = {"pip install good": 0, "pip install bad": 1}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            args=cmd, returncode=outcomes.get(cmd, 0), stdout="", stderr=""
        ),
    )
    persisted: list[dict[str, Any]] = []

    out = _run(
        [("pip install good", "inherited"), ("pip install bad", "proposed"), ("rm -rf /tmp/x", "proposed")],
        tmp_path,
        seq_start=4,
        on_execution=persisted.append,
    )

    rows = out["executions"]
    assert persisted == rows
    assert [row["outcome"] for row in rows] == ["applied", "failed", "skipped"]
    assert [row["seq"] for row in rows] == [5, 6, 7]
    assert [row["source"] for row in rows] == ["inherited", "proposed", "proposed"]
    assert {row["round_task_id"] for row in rows} == {"r1"}


def test_a_rejected_command_never_carries_its_credentialed_url(tmp_path: Path, monkeypatch, caplog):
    _forbid_execution(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=_setup_replay.__name__):
        out = _run(_proposed("pip install --index-url https://alice:pass@host/simple foo && x"), tmp_path)
    reason = setup_report_fields("boot failed", out)["reason"]

    assert out["skipped"] == ["pip install --index-url <index_url> foo && x"]
    for surface in (" ".join(out["skipped"]), reason, caplog.text):
        assert "alice" not in surface
        assert "host/simple" not in surface


def test_the_skipped_list_is_stored_sanitised_and_bounded(tmp_path, monkeypatch):
    """The list itself must be safe, not just the sentence built from it.

    ``setup_commands_skipped`` is copied verbatim into every result payload and
    from there into the journal, the report and the KB, and is read back into the
    next round's mandate -- so a credential in one must not survive, and a full
    list of long ones must not bury the reason it is appended to.
    """
    _forbid_execution(monkeypatch)
    commands = [f"rm -rf /tmp/{i}/ghp_notarealtoken " + "y" * 900 for i in range(SETUP_CMD_MAX)]

    out = _run(_proposed(*commands), tmp_path)
    reason = setup_report_fields("boot failed", out)["reason"]

    assert "ghp_notarealtoken" not in " ".join(out["skipped"]), "a credential was stored verbatim"
    assert all(len(c) <= 200 for c in out["skipped"]), "an unbounded command was stored"
    assert "ghp_notarealtoken" not in reason
    assert reason.startswith("boot failed")
    assert len(reason) < 4000, f"one rejection list grew to {len(reason)} chars"


def test_applied_commands_stay_runnable_but_their_ledger_row_is_redacted(tmp_path, monkeypatch):
    """``applied`` is the replay channel, so it stays verbatim; the durable row does not.

    ``lane.py`` stacks ``setup_commands_applied`` into
    ``state.enablement.setup_commands``, and the next round EXECUTES what it
    finds there, so redacting where the list is built would hand pip a masked
    URL. The ledger row is the durable record of the execution, and it carries
    only the sanitized form.
    """
    cmd = "pip install --extra-index-url http://pkgs.internal/simple foo ghp_notarealtoken"
    _succeed(monkeypatch)

    out = _run(_proposed(cmd), tmp_path)

    assert out["applied"] == [cmd]
    assert "ghp_notarealtoken" not in out["executions"][0]["cmd_sanitized"]


def test_a_log_write_failure_is_reported_and_does_not_fail_the_install(tmp_path, monkeypatch, caplog):
    _succeed(monkeypatch)
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=_setup_replay.__name__):
        out = _run(_proposed("pip install foo"), tmp_path, log_dir=blocked / "logs")

    assert out["applied"] == ["pip install foo"]
    assert "enablement setup output not written" in caplog.text


def test_rejections_are_named_in_the_round_reason():
    """A rejected command must reach the conclusion, not just a log line.

    It used to be a lone ``log.warning``. Downstream saw the round's outcome
    with no link to the cause, so the same proposal was re-authored and
    re-dropped until the budget ran out -- the fix was never the problem, and
    nothing in the result said so.
    """
    fields = setup_report_fields(
        "authored patch produced no gain",
        {"applied": ["pip install a"], "skipped": ["/opt/x/uv venv /opt/v", "ln -sf a b"], "failed": []},
    )
    assert fields["reason"].startswith("authored patch produced no gain (2 setup command(s) were REJECTED")
    assert "ln -sf a b" in fields["reason"]
    assert fields["setup_commands_applied"] == ["pip install a"]
    assert fields["setup_commands_skipped"] == ["/opt/x/uv venv /opt/v", "ln -sf a b"]


def test_reason_is_untouched_when_nothing_was_rejected():
    base = "authored patch produced no gain"
    fields = setup_report_fields(base, {"applied": ["pip install x"], "skipped": [], "failed": []})
    assert fields == {"reason": base, "setup_commands_applied": ["pip install x"], "setup_commands_skipped": []}


@pytest.mark.asyncio
async def test_a_completed_setup_row_is_durable_even_when_the_await_is_cancelled(tmp_path: Path, monkeypatch):
    """Cancelling the await unwinds the caller; the worker keeps running.

    ``asyncio.to_thread`` cannot kill the thread, so a command already inside
    ``subprocess.run`` runs to completion and installs into the shared venv.
    A row handed back through the return value never arrives -- the await
    raised -- so the only record that the round installed anything at all is
    lost, and a later reader sees a round that never ran setup.
    """
    reached_second = threading.Event()
    release = threading.Event()
    durable: list[dict] = []

    def _executor(cmd, *, cwd, env, log_path):
        if cmd.endswith("two"):
            reached_second.set()
            release.wait(timeout=30)
        return True

    monkeypatch.setattr(_setup_replay, "execute_setup_command", _executor)

    pending = asyncio.ensure_future(
        asyncio.to_thread(
            run_setup_commands,
            _proposed("pip install one", "pip install two"),
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            round_task_id="r-cancel",
            seq_start=0,
            on_execution=durable.append,
        )
    )
    # The first command has finished and the second is in flight.
    await asyncio.to_thread(reached_second.wait, 30)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    release.set()

    assert durable, "the row for the command that completed was discarded"
    assert durable[0]["cmd_digest"] == command_digest("pip install one")
    assert durable[0]["round_task_id"] == "r-cancel"
    assert durable[0]["outcome"] == "applied"
    assert durable[0]["cmd_index"] == 0
