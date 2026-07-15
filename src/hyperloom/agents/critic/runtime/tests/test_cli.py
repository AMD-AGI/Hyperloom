# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "inmemory")
    monkeypatch.setenv("KB_DEAD_LETTER_DIR", str(tmp_path / "dlq"))
    yield


def test_init_session_writes_context(tmp_path, capsys):
    request = _write(
        tmp_path / "req.json",
        {
            "kind": "critic_decision_request",
            "session_id": "sess_cli_init",
            "context": {"model": "qwen3-14b", "framework": "sglang"},
            "messages": [],
        },
    )
    rc = main(["init-session", "--request", str(request)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert json.loads(captured)["session_id"] == "sess_cli_init"


def test_prepare_and_commit_review_for_coordinator_inbox(tmp_path, capsys):
    request = _write(
        tmp_path / "req.json",
        {
            "kind": "coordinator_inbox",
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


def test_close_session_emits_summary(tmp_path):
    request = _write(
        tmp_path / "req.json",
        {
            "kind": "critic_decision_request",
            "session_id": "sess_cli_close",
            "context": {
                "model": "qwen3-14b",
                "framework": "sglang",
                "model_family": "qwen",
                "workload": "decode",
                "precision": "fp8",
            },
        },
    )
    main(["init-session", "--request", str(request)])
    rc = main(
        [
            "close-session",
            "--request",
            str(request),
            "--out",
            str(tmp_path / "close.json"),
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "close.json").read_text("utf-8"))
    assert out["session_id"] == "sess_cli_close"


def test_invalid_request_returns_exit_code_2(tmp_path):
    bad = _write(tmp_path / "bad.json", {"kind": "wat"})
    rc = main(["init-session", "--request", str(bad)])
    assert rc == 2


# Low-level KB commands (inmemory client).

_PACKET = {
    "context": {
        "framework": "sglang",
        "model": "deepseek-r1-0528-fp8",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }
}


def test_list_priors_emits_scope_cache_key(tmp_path):
    packet = _write(tmp_path / "packet.json", _PACKET)
    out_path = tmp_path / "priors.json"
    rc = main(["list-priors", "--packet", str(packet), "--out", str(out_path)])
    assert rc == 0
    payload = json.loads(out_path.read_text("utf-8"))
    assert payload["cache"] in {"miss", "hit", "disabled"}
    assert isinstance(payload["scope_cache_key"], str)
    assert payload["scope_cache_key"]


def test_write_verdict_persists_reject_lesson(tmp_path):
    packet = _write(tmp_path / "packet.json", _PACKET)
    verdict = _write(
        tmp_path / "verdict.json",
        {
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
            "confidence": "high",
        },
    )
    ctx = _write(
        tmp_path / "ctx.json",
        {"session_id": "sess_wv", "review_id": "rev_1", "topic": "active dispatch path unproven"},
    )
    out_path = tmp_path / "wv.json"
    rc = main(
        [
            "write-verdict",
            "--packet",
            str(packet),
            "--verdict",
            str(verdict),
            "--ctx",
            str(ctx),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    result = json.loads(out_path.read_text("utf-8"))
    assert result["status"] in {"ok", "skipped", "dead_lettered", "disabled"}


def test_write_kb_drafts_batch(tmp_path):
    packet = _write(tmp_path / "packet.json", _PACKET)
    kb_draft = _write(
        tmp_path / "draft.json",
        {
            "kb_drafts": [
                {
                    "category": "kernel_optimization",
                    "action": "Patch fused attention kernel for the decode path.",
                    "lesson": "Active dispatch path must be updated jointly.",
                    "tags": ["attention"],
                    "result": {"status": "KEEP", "gain_pct": 4.2},
                }
            ]
        },
    )
    ctx = _write(tmp_path / "ctx.json", {"session_id": "sess_wd", "review_id": "rev_wd"})
    out_path = tmp_path / "wd.json"
    rc = main(
        [
            "write-kb-drafts",
            "--packet",
            str(packet),
            "--kb-draft",
            str(kb_draft),
            "--ctx",
            str(ctx),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    result = json.loads(out_path.read_text("utf-8"))
    assert result["status"] in {"ok", "skipped", "dead_lettered", "disabled"}


def test_add_contradiction_command(tmp_path):
    ctx = _write(tmp_path / "ctx.json", {"session_id": "sess_ac", "review_id": "rev_ac"})
    out_path = tmp_path / "ac.json"
    rc = main(
        [
            "add-contradiction",
            "--new-id",
            "kb_new_1",
            "--old-ids",
            "kb_old_1, kb_old_2 ,",
            "--ctx",
            str(ctx),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    result = json.loads(out_path.read_text("utf-8"))
    assert "status" in result


def test_replay_dead_letter_empty_queue(tmp_path):
    out_path = tmp_path / "replay.json"
    rc = main(
        [
            "replay-dead-letter",
            "--dir",
            str(tmp_path / "dlq_replay"),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text("utf-8"))
    assert isinstance(summary, dict)


def test_replay_dead_letter_dispatches_queued_records(tmp_path):
    from hyperloom.agents.critic.runtime.cli import _replay_dispatch
    from hyperloom.agents.critic.runtime.dead_letter import DeadLetter
    from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError
    from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient

    dlq_dir = tmp_path / "dlq_dispatch"
    dlq = DeadLetter(root=dlq_dir)
    common = {"attempts": 1, "last_error": "boom"}
    # Endpoints stored in the DLQ use filesystem-safe names (no "/").
    dlq.append("upsert", {"id": "kb_x", "kind": "pitfall", "scope": {}}, **common)
    dlq.append("batch_insert", {"items": [{"id": "kb_y", "kind": "recipe", "scope": {}}]}, **common)
    dlq.append("list", {"scope_filter": {}, "limit": 5}, **common)

    out_path = tmp_path / "replay2.json"
    rc = main(
        [
            "replay-dead-letter",
            "--dir",
            str(dlq_dir),
            "--keep-on-success",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text("utf-8"))
    assert isinstance(summary, dict)

    client = InMemoryKBClient()
    _replay_dispatch(client, "edges/add", {"edges": []})
    with pytest.raises(RuntimeAdapterError):
        _replay_dispatch(client, "nope", {})


def test_commit_review_rejects_non_object_review(tmp_path):
    request = _write(tmp_path / "req.json", {"kind": "coordinator_inbox", "raw_prompt": ""})
    review = _write(tmp_path / "review.json", ["not", "a", "dict"])
    rc = main(["commit-review", "--request", str(request), "--review", str(review)])
    assert rc == 2


def test_resolve_kb_client_live_requires_url(monkeypatch):
    from hyperloom.agents.critic.runtime.cli import _resolve_kb_client
    from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError

    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "live")
    monkeypatch.delenv("KB_BASE_URL", raising=False)
    with pytest.raises(RuntimeAdapterError, match="KB_BASE_URL"):
        _resolve_kb_client()

    monkeypatch.setenv("KB_BASE_URL", "http://kb.invalid")
    client = _resolve_kb_client()
    assert client is not None
