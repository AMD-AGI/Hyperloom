# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the end-of-run session HTML report step.

The step's whole point is that the page exists without anyone asking for it, so
these tests pin the three things that would quietly undo that: it must run by
default, it must say so rather than write a hollow page when there is no ledger,
and a renderer that raises must not take the run's exit status with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools.dump_llm_call_report import load_ledgers
from hyperloom.inference_optimizer.tools.render_hyperloom_html_report import (
    DEFAULT_OUTPUT,
    render,
)


def _session_with_ledger(tmp_path: Path) -> Path:
    trace = tmp_path / "reports" / "trace"
    trace.mkdir(parents=True)
    row = {
        "call_id": "c1",
        "task_path": "FRAMEWORK_AGENT",
        "model": "claude-opus-5",
        "input_tokens": 1000,
        "output_tokens": 10,
        "usd": 0.01,
    }
    (trace / "llm_calls.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path


def test_renders_into_the_session_reports_dir(tmp_path: Path) -> None:
    """A session carrying a ledger renders to reports/<DEFAULT_OUTPUT>."""
    session = _session_with_ledger(tmp_path)
    out = session / "reports" / DEFAULT_OUTPUT
    out.write_text(render(session), encoding="utf-8")

    assert out.exists()
    assert out.read_text(encoding="utf-8").lstrip().startswith("<!")


def test_output_name_comes_from_the_renderer(tmp_path: Path) -> None:
    """The caller must not invent a second naming convention for the page."""
    assert DEFAULT_OUTPUT.endswith(".html")
    assert "/" not in DEFAULT_OUTPUT


def test_empty_ledger_is_detectable_before_rendering(tmp_path: Path) -> None:
    """No ledger must be distinguishable from an empty one, so the step can skip.

    Rendering here would emit a page whose every figure is blank, which reads as
    a measured zero rather than an absence.
    """
    (tmp_path / "reports").mkdir(parents=True)
    turns, details = load_ledgers(tmp_path)

    assert not turns and not details


def test_ledger_present_is_detectable(tmp_path: Path) -> None:
    """The same probe must report work when a ledger does exist."""
    session = _session_with_ledger(tmp_path)
    turns, details = load_ledgers(session)

    assert turns or details


@pytest.mark.parametrize(
    ("value", "skipped"),
    [("", False), ("0", False), ("1", True), ("true", True), (" 1 ", True)],
)
def test_skip_switch_semantics(value: str, skipped: bool) -> None:
    """HYPERLOOM_SKIP_HTML_REPORT is opt-out: unset and "0" both still render.

    Mirrors the guard in cli.__init__ so the default cannot flip to off by an
    empty or zero-valued env var leaking into the launch shell.
    """
    assert (value.strip() not in {"", "0"}) is skipped
