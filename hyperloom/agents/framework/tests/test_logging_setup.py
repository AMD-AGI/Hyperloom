"""Tests for framework_agent.logging_setup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hyperloom.agents.framework.logging_setup import (
    configure_logging,
    get_logger,
    stage_log,
)


@pytest.fixture(autouse=True)
def _reset_root() -> None:
    """Reset the framework_agent root after every test so handlers do not leak."""
    yield
    root = logging.getLogger("framework_agent")
    for h in list(root.handlers):
        root.removeHandler(h)


def test_get_logger_returns_child_under_root() -> None:
    """get_logger(name) attaches under the framework_agent root."""
    log = get_logger("explorer")
    assert log.name == "framework_agent.explorer"
    log2 = get_logger("framework_agent.sub.deep")
    assert log2.name == "framework_agent.sub.deep"


def test_configure_logging_idempotent(caplog: pytest.LogCaptureFixture) -> None:
    """Re-calling configure_logging clears old handlers (no duplicate emit)."""
    configure_logging(level="DEBUG")
    configure_logging(level="DEBUG")
    root = logging.getLogger("framework_agent")
    assert sum(
        1 for h in root.handlers if isinstance(h, logging.StreamHandler)
    ) == 1


def test_configure_logging_resolves_env_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """FRAMEWORK_EXPLORER_LOG_LEVEL is honoured when no explicit level passed."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("FRAMEWORK_AGENT_LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger("framework_agent").level == logging.DEBUG


def test_configure_logging_writes_file(tmp_path: Path) -> None:
    """A --log-file path receives records appended to the same sink."""
    log_path = tmp_path / "fa.log"
    configure_logging(level="INFO", log_file=log_path)
    log = get_logger("test")
    log.info("hello %s", "world")
    contents = log_path.read_text(encoding="utf-8")
    assert "hello world" in contents


def test_configure_logging_json_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """json_output=True emits one JSON object per record (machine-parseable)."""
    log_path = tmp_path / "fa.jsonl"
    configure_logging(level="INFO", json_output=True, log_file=log_path)
    log = get_logger("test")
    log.info("hi", extra={"extra_candidate": "PR:42"})
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "hi"
    assert payload["candidate"] == "PR:42"
    assert payload["level"] == "INFO"


def test_stage_log_emits_start_and_done(tmp_path: Path) -> None:
    """stage_log brackets the block with start + done envelopes."""
    log_path = tmp_path / "fa.log"
    configure_logging(level="DEBUG", log_file=log_path)
    log = get_logger("test")
    with stage_log(log, "build", candidate="PR:1") as ctx:
        ctx["throughput"] = 123.4
    body = log_path.read_text(encoding="utf-8")
    assert "stage.start build" in body
    assert "stage.done build" in body


def test_stage_log_emits_failed_on_exception(tmp_path: Path) -> None:
    """An exception inside stage_log produces a 'stage.failed' envelope + re-raises."""
    log_path = tmp_path / "fa.log"
    configure_logging(level="DEBUG", log_file=log_path)
    log = get_logger("test")

    def _raise_inside_stage_log() -> None:
        with stage_log(log, "bench"):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _raise_inside_stage_log()
    body = log_path.read_text(encoding="utf-8")
    assert "stage.failed bench" in body
