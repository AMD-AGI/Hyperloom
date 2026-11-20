# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for the CLI front door, run in-process via :func:`runtime.cli.main`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.agents.critic.runtime.cli import main


def _write(path: Path, body) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_session_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SESSION_MEMORY_DIR", str(tmp_path / "sm"))
    yield


def test_prepare_and_commit_review_for_coordinator_inbox(tmp_path, capsys):
    request = _write(
        tmp_path / "req.json",
        {
            "session_id": "sess_cli_e2e",
            "raw_prompt": (
                "=== Shared session state ===\n"
                "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
                "=== Inbox for critic ===\n"
                "  seq=1 msg_id=cli01 from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
            ),
        },
    )
    rc = main(["prepare-review", "--request", str(request), "--out", str(tmp_path / "judge.json")])
    assert rc == 0
    judge = json.loads((tmp_path / "judge.json").read_text("utf-8"))
    assert judge["proposals"][0]["msg_id"] == "cli01"

    review = _write(
        tmp_path / "review.json",
        {
            "review_verdicts": [
                {
                    "target_proposal_msg_id": "cli01",
                    "verdict": "approve",
                    "reasoning": "ok",
                }
            ]
        },
    )
    rc = main(
        [
            "commit-review",
            "--request",
            str(request),
            "--review",
            str(review),
            "--out",
            str(tmp_path / "emit.json"),
        ]
    )
    assert rc == 0
    emit = json.loads((tmp_path / "emit.json").read_text("utf-8"))
    assert emit["intent_envelope"]["intents"][0]["payload"]["verdict"] == "approve"


def test_invalid_request_returns_exit_code_2(tmp_path):
    bad = _write(tmp_path / "bad.json", {"session_id": "s"})
    rc = main(["prepare-review", "--request", str(bad)])
    assert rc == 2


def test_commit_review_rejects_non_object_review(tmp_path):
    request = _write(tmp_path / "req.json", {"raw_prompt": ""})
    review = _write(tmp_path / "review.json", ["not", "a", "dict"])
    rc = main(["commit-review", "--request", str(request), "--review", str(review)])
    assert rc == 2
