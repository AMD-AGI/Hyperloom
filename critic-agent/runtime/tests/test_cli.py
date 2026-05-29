"""Smoke tests for the CLI front door.

We run the parser/dispatch in-process via :func:`runtime.cli.main` so the
behaviour exercised matches what ``python -m runtime.cli ...`` would do.
The tests focus on the new commands; the legacy ones (``write-verdict``,
``write-kb-drafts``, ...) are covered transitively by ``test_kb_writer``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.cli import main


def _write(path: Path, body) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_session_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SESSION_MEMORY_DIR", str(tmp_path / "sm"))
    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "inmemory")
    monkeypatch.setenv("KB_DEAD_LETTER_DIR", str(tmp_path / "dlq"))
    yield


def test_init_session_writes_context(tmp_path, capsys):
    request = _write(tmp_path / "req.json", {
        "kind": "critic_decision_request",
        "session_id": "sess_cli_init",
        "context": {"model": "qwen3-14b", "framework": "sglang"},
        "messages": [],
    })
    rc = main(["init-session", "--request", str(request)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert json.loads(captured)["session_id"] == "sess_cli_init"


def test_prepare_and_commit_review_for_coordinator_inbox(tmp_path, capsys):
    request = _write(tmp_path / "req.json", {
        "kind": "coordinator_inbox",
        "session_id": "sess_cli_e2e",
        "raw_prompt": (
            "=== Shared session state ===\n"
            "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
            "=== Inbox for critic ===\n"
            "  seq=1 msg_id=cli01 from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
        ),
    })
    rc = main(["prepare-review", "--request", str(request), "--out", str(tmp_path / "judge.json")])
    assert rc == 0
    judge = json.loads((tmp_path / "judge.json").read_text("utf-8"))
    assert judge["proposals"][0]["msg_id"] == "cli01"

    review = _write(tmp_path / "review.json", {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "cli01",
                "verdict": "approve",
                "reasoning": "ok",
            }
        ]
    })
    rc = main([
        "commit-review",
        "--request", str(request),
        "--review", str(review),
        "--out", str(tmp_path / "emit.json"),
    ])
    assert rc == 0
    emit = json.loads((tmp_path / "emit.json").read_text("utf-8"))
    assert emit["intent_envelope"]["intents"][0]["payload"]["verdict"] == "approve"


def test_close_session_emits_summary(tmp_path):
    request = _write(tmp_path / "req.json", {
        "kind": "critic_decision_request",
        "session_id": "sess_cli_close",
        "context": {
            "model": "qwen3-14b", "framework": "sglang",
            "model_family": "qwen", "workload": "decode", "precision": "fp8",
        },
    })
    main(["init-session", "--request", str(request)])
    rc = main([
        "close-session",
        "--request", str(request),
        "--out", str(tmp_path / "close.json"),
    ])
    assert rc == 0
    out = json.loads((tmp_path / "close.json").read_text("utf-8"))
    assert out["session_id"] == "sess_cli_close"


def test_invalid_request_returns_exit_code_2(tmp_path):
    bad = _write(tmp_path / "bad.json", {"kind": "wat"})
    rc = main(["init-session", "--request", str(bad)])
    assert rc == 2
