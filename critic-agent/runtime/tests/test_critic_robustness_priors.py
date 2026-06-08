# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""L4: Critic injects recent Robustness findings as priors.

Covers :func:`runtime.decision_reviewer._discover_robustness_findings_path`
and the bundle field population from the ``prepare_review`` tail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.decision_reviewer import (
    DecisionReviewer,
    _discover_robustness_findings_path,
    _load_robustness_priors,
)
from runtime.in_memory_kb_client import InMemoryKBClient
from runtime.kb_writer import KBWriter
from runtime.session_memory import SessionMemory


_PROMPT = (
    "=== Shared session state ===\n"
    "session_id=sess_a model=Qwen3-14B framework=sglang baseline_tput=1200\n"
    "=== Inbox for critic (newest last) ===\n"
    "  seq=1 msg_id=aaa1 from=orchestration topic=proposal "
    "payload={'action_name': 'baseline'}\n"
)


@pytest.fixture()
def reviewer(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = InMemoryKBClient()
    writer = KBWriter(kb, session_memory=sm)
    return DecisionReviewer(session_memory=sm, kb_writer=writer)


def _seed_findings(
    findings_path: Path, entries: list[dict],
) -> None:
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    with findings_path.open("w", encoding="utf-8") as fh:
        for row in entries:
            fh.write(json.dumps(row) + "\n")


def test_discover_path_via_explicit_dir(
    tmp_path: Path, monkeypatch
) -> None:
    findings_path = tmp_path / "f" / "sess_a.jsonl"
    findings_path.parent.mkdir(parents=True)
    findings_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("CRITIC_ROBUSTNESS_FINDINGS_DIR", str(tmp_path / "f"))
    monkeypatch.delenv("ROBUSTNESS_AGENT_SESSION_DIR", raising=False)
    out = _discover_robustness_findings_path("sess_a")
    assert out == findings_path


def test_discover_path_via_session_dir(
    tmp_path: Path, monkeypatch
) -> None:
    sd = tmp_path / "sd"
    findings_path = sd / "agents" / "robustness" / "findings" / "sess_x.jsonl"
    findings_path.parent.mkdir(parents=True)
    findings_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("CRITIC_ROBUSTNESS_FINDINGS_DIR", raising=False)
    monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))
    out = _discover_robustness_findings_path("sess_x")
    assert out == findings_path


def test_discover_returns_none_when_unset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CRITIC_ROBUSTNESS_FINDINGS_DIR", raising=False)
    monkeypatch.delenv("ROBUSTNESS_AGENT_SESSION_DIR", raising=False)
    assert _discover_robustness_findings_path("sess") is None


def test_load_priors_filters_severity_and_caps(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    rows = [
        {
            "tick_index": 1, "severity": "low",
            "symptom_name": "low1", "summary": "skip me",
            "intents": [], "evidence": {}, "rca_text": "",
        },
        {
            "tick_index": 2, "severity": "medium",
            "symptom_name": "med1", "summary": "med",
            "intents": [], "evidence": {}, "rca_text": "",
        },
        {
            "tick_index": 3, "severity": "high",
            "symptom_name": "high1", "summary": "h1",
            "intents": [{"intent_type": "alert", "payload": {}}],
            "evidence": {"k": 1}, "rca_text": "rca1",
        },
        {
            "tick_index": 4, "severity": "high",
            "symptom_name": "high2", "summary": "h2",
            "intents": [], "evidence": {}, "rca_text": "",
        },
        {
            "tick_index": 5, "severity": "high",
            "symptom_name": "high3", "summary": "h3",
            "intents": [], "evidence": {}, "rca_text": "",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    # HIGH only: 3 rows; cap to 2 → last two (high2, high3).
    priors = _load_robustness_priors(path, limit=2, min_severity="high")
    names = [p["symptom_name"] for p in priors]
    assert names == ["high2", "high3"]
    assert priors[0]["tick_index"] == 4
    # MEDIUM-or-above: still capped to 2; last two of {med1, high1..3}
    priors_med = _load_robustness_priors(
        path, limit=10, min_severity="medium",
    )
    assert [p["symptom_name"] for p in priors_med] == [
        "med1", "high1", "high2", "high3",
    ]


def test_load_priors_handles_missing_file(tmp_path: Path) -> None:
    # Nonexistent path → empty list (no crash).
    out = _load_robustness_priors(
        tmp_path / "missing.jsonl", limit=5, min_severity="high",
    )
    assert out == []


def test_prepare_review_injects_robustness_priors(
    reviewer, tmp_path: Path, monkeypatch
):
    findings_dir = tmp_path / "sd" / "agents" / "robustness" / "findings"
    findings_path = findings_dir / "sess_a.jsonl"
    _seed_findings(findings_path, [
        {
            "tick_index": 1, "timestamp_unix": 1.0,
            "symptom_name": "gpu_leak_persistent",
            "severity": "high", "summary": "GPU leak repeated",
            "intents": [{
                "intent_type": "alert",
                "payload": {"severity": "high"},
            }],
            "evidence": {"used_mb": 70000},
            "rca_text": "reboot mitigated last 3 times",
        },
        {
            "tick_index": 2, "timestamp_unix": 2.0,
            "symptom_name": "quota_low_hit",
            "severity": "medium", "summary": "low quota",
            "intents": [], "evidence": {}, "rca_text": "",
        },
    ])
    monkeypatch.setenv(
        "ROBUSTNESS_AGENT_SESSION_DIR", str(tmp_path / "sd"),
    )
    monkeypatch.delenv("CRITIC_ROBUSTNESS_FINDINGS_DIR", raising=False)
    monkeypatch.delenv("CRITIC_ROBUSTNESS_PRIORS_DISABLED", raising=False)

    bundle = reviewer.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_a",
        "raw_prompt": _PROMPT,
    })
    assert len(bundle.robustness_priors) == 1
    p = bundle.robustness_priors[0]
    assert p["symptom_name"] == "gpu_leak_persistent"
    assert p["severity"] == "high"
    assert p["rca_text"] == "reboot mitigated last 3 times"
    assert any(
        "robustness_priors_injected" in n for n in bundle.notes
    )
    # to_dict() must include the new field.
    serialised = bundle.to_dict()
    assert "robustness_priors" in serialised
    assert serialised["robustness_priors"][0]["symptom_name"] == (
        "gpu_leak_persistent"
    )


def test_prepare_review_disables_via_env(
    reviewer, tmp_path: Path, monkeypatch
):
    findings_dir = tmp_path / "sd" / "agents" / "robustness" / "findings"
    findings_path = findings_dir / "sess_a.jsonl"
    _seed_findings(findings_path, [
        {
            "tick_index": 1, "timestamp_unix": 1.0,
            "symptom_name": "x", "severity": "high",
            "summary": "s", "intents": [], "evidence": {},
            "rca_text": "",
        },
    ])
    monkeypatch.setenv(
        "ROBUSTNESS_AGENT_SESSION_DIR", str(tmp_path / "sd"),
    )
    monkeypatch.setenv("CRITIC_ROBUSTNESS_PRIORS_DISABLED", "1")
    bundle = reviewer.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_a",
        "raw_prompt": _PROMPT,
    })
    assert bundle.robustness_priors == []


def test_prepare_review_no_priors_when_no_findings(
    reviewer, monkeypatch, tmp_path: Path,
):
    # ROBUSTNESS_AGENT_SESSION_DIR points to an empty dir → no file.
    monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(tmp_path))
    monkeypatch.delenv("CRITIC_ROBUSTNESS_FINDINGS_DIR", raising=False)
    bundle = reviewer.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_a",
        "raw_prompt": _PROMPT,
    })
    assert bundle.robustness_priors == []
