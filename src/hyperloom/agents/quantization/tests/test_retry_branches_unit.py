"""Branch coverage for quantization_agent.driver.retry helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from hyperloom.agents.quantization.driver import retry
from hyperloom.agents.quantization.driver.outcomes import ASK_RETRYABLE, OutcomeId


def test_read_counter_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / retry._COUNTER_FILE).write_text("not-an-int", encoding="utf-8")
    # ValueError swallowed -> 0 (lines 100-101).
    assert retry._read_counter(tmp_path) == 0


def test_resolve_interactive_explicit_and_error(monkeypatch) -> None:
    assert retry._resolve_interactive(True) is True
    assert retry._resolve_interactive(False) is False

    # isatty raising -> auto-detect returns False (lines 140-141).
    def _boom():
        raise OSError("no tty")

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=_boom))
    assert retry._resolve_interactive(None) is False


def test_ask_operator_yes_and_eof(monkeypatch) -> None:
    # Affirmative answer.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=lambda: "yes\n"))
    assert retry._ask_operator("ok?") is True

    # EOF / interrupt -> False (lines 154-159).
    def _raise():
        raise EOFError

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=_raise))
    assert retry._ask_operator("ok?") is False


def test_decide_next_step_operator_declines_retry(tmp_path, monkeypatch) -> None:
    outcome = next(iter(ASK_RETRYABLE))
    # Fix hypothesis present so we reach the operator prompt.
    (tmp_path / "fix_hypothesis_attempt_2.md").write_text("plan", encoding="utf-8")
    monkeypatch.setattr(retry, "_ask_operator", lambda _msg: False)
    decision = retry._decide_next_step(
        outcome,
        workspace=tmp_path,
        attempt_number=1,
        interactive=True,
        max_requantize_attempts=3,
        counter=0,
    )
    # Operator declined -> no retry (line 270).
    assert decision.retry is False
    assert decision.note == "operator_declined_retry"


def test_decide_next_step_no_fix_hypothesis(tmp_path) -> None:
    outcome = next(iter(ASK_RETRYABLE))
    decision = retry._decide_next_step(
        outcome,
        workspace=tmp_path,
        attempt_number=1,
        interactive=False,
        max_requantize_attempts=3,
        counter=0,
    )
    assert decision.retry is False
    assert decision.note == "no_fix_hypothesis"


def test_decide_next_step_checkpoint_aborted(tmp_path) -> None:
    decision = retry._decide_next_step(
        OutcomeId.checkpoint_aborted,
        workspace=tmp_path,
        attempt_number=1,
        interactive=False,
        max_requantize_attempts=1,
        counter=0,
    )
    assert decision.retry is False
    assert "checkpoint_aborted" in decision.note
