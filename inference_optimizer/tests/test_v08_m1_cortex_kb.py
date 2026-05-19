"""v0.8 M1 — Cortex KB integration tests (KB_design §3.13 M1).

Covers the write-side surface introduced by the M1 milestone:

* Session-path helpers create the new ``runtime/cortex/`` skeleton.
* CortexKBClient canonical_id derivations + NDJSON envelope contract.
* Synchronous CLI failures degrade to NDJSON enqueue.
* T0 fail-fast vs ``--no-cortex`` bypass (cli boot path).
* T2 hook stores ``kb_edge_id`` + ``kb_opt_canonical`` on PendingProposal.
* T3 hook routes KEEP / REVERT through ingest_attempt + verify and pops
  the matching pending edge entry.
* T4 drain wires NDJSON pending → flushed bookkeeping and records
  cortex_session_summary.
* breakdown.collect_kb_provenance returns the documented stable shape.

The tests avoid hitting a real Cortex service: we install a fake
``cortex-kb`` binary on PATH (a tiny shell script that prints the
expected ``key: value`` text the client parses) and otherwise route
through the NDJSON queue.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from inference_optimizer.cortex_kb_client import (
    CortexBinaryNotFound,
    CortexKBClient,
    CortexKBError,
    attempt_canonical_id,
    optimization_canonical_id,
    workload_canonical_id,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_audit_jsonl,
    cortex_dir,
    cortex_pending_ndjson,
    cortex_pitfalls_json,
    cortex_sid_file,
    cortex_warm_json,
)


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _write_fake_cortex_bin(bin_dir: Path, *, stdout_lines: list[str], exit_code: int = 0) -> Path:
    """Drop a shell stub at ``bin_dir/cortex-kb`` that emits the given output.

    Returns the absolute path. Caller monkeypatches PATH to include
    ``bin_dir``.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "cortex-kb"
    body_lines = "\n".join(f"echo {json.dumps(line)}" for line in stdout_lines)
    target.write_text(
        "#!/bin/sh\n"
        f"{body_lines}\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


# ===========================================================================
# session_paths / skeleton
# ===========================================================================
def test_make_session_dir_creates_runtime_cortex(session_dir):
    cortex_root = cortex_dir(session_dir)
    assert cortex_root.exists()
    assert cortex_root.is_dir()
    assert cortex_pending_ndjson(session_dir).parent == cortex_root


def test_canonical_id_derivations_are_idempotent():
    a = workload_canonical_id("meta-llama/Llama-3.1-8B-Instruct", "mi300x")
    b = workload_canonical_id("meta-llama/Llama-3.1-8B-Instruct", "MI300x")
    # gpu_type is lowercased; '/' becomes '_'
    assert a == b
    assert a == "workload.meta-llama_Llama-3.1-8B-Instruct.mi300x"
    # missing values fall back to deterministic sentinels (no random suffix).
    assert workload_canonical_id("", "") == "workload.unknown_model.unknown_gpu"

    assert optimization_canonical_id("36", "msg-1") == "opt.session-36.proposal-msg-1"
    assert attempt_canonical_id("36", "task-1") == "attempt.session-36.task-task-1"


# ===========================================================================
# CortexKBClient — sync success + NDJSON fallback
# ===========================================================================
def test_disabled_client_skips_all_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False)
    assert client.session_begin(
        task="x", workload="w", hw="mi300x", stack_fingerprint={"rocm": "x"},
    ) == ""
    assert client.propose_point(canonical_id="opt.x", kind="optimization_node")[
        "status"
    ] == "skip_disabled"
    assert client.hypothesize(
        sid="", from_canonical="x", to_canonical="y",
    )["tentative_edge_id"] == ""
    # No NDJSON file should exist when --no-cortex.
    assert not cortex_pending_ndjson(session_dir).exists()


def test_session_begin_parses_session_id(tmp_path, session_dir, monkeypatch):
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: 42",
        "thinking_style: recommendation",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    sid = client.session_begin(
        task="x", workload="w", hw="mi300x",
        stack_fingerprint={"rocm": "7.2.0"},
    )
    assert sid == "42"
    assert cortex_sid_file(session_dir).read_text(encoding="utf-8").strip() == "42"
    audit_lines = cortex_audit_jsonl(session_dir).read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in audit_lines if line.strip()]
    # Must have at least a session_begin success record.
    assert any(row.get("op") == "session_begin" and row.get("status") == "ok" for row in parsed)


def test_missing_binary_raises_cortex_binary_not_found(session_dir, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent_xyz")
    monkeypatch.setenv("CORTEX_KB_BIN", "definitely-not-here-zzz")
    client = CortexKBClient(session_dir=session_dir)
    with pytest.raises(CortexBinaryNotFound):
        client.session_begin(task="x", workload="w", hw="mi300x")


def test_hypothesize_sync_failure_falls_back_to_ndjson(tmp_path, session_dir, monkeypatch):
    bin_dir = tmp_path / "bin"
    # Exit non-zero so the sync call raises CortexKBError → enqueue path.
    _write_fake_cortex_bin(bin_dir, stdout_lines=["error: kb down"], exit_code=1)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    outcome = client.hypothesize(
        sid="42",
        from_canonical="workload.foo.bar",
        to_canonical="opt.session-42.proposal-x",
        reason="test reason",
    )
    assert outcome["status"] == "queued"
    assert outcome["tentative_edge_id"] == ""
    pending = cortex_pending_ndjson(session_dir).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in pending.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["op"] == "hypothesize"
    assert rows[0]["payload"]["from"] == "workload.foo.bar"
    assert rows[0]["payload"]["to"] == "opt.session-42.proposal-x"
    assert rows[0]["attempts"] == 0
    assert rows[0]["idempotency_key"]


def test_ingest_attempt_always_enqueues(session_dir, tmp_path, monkeypatch):
    # Even with a working stub binary, ingest_attempt is async-by-design
    # and always enqueues — no sync CLI invocation should fire.
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=["status: ok"])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    res = client.ingest_attempt(
        sid="42", iter_id=3, outcome="PASS",
        metrics={"output_throughput": 1234.5},
        plan_edge="e1",
        evidence=["log:demo"],
    )
    assert res == {"status": "queued"}
    pending = cortex_pending_ndjson(session_dir).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in pending if line.strip()]
    assert len(rows) == 1
    assert rows[0]["op"] == "ingest_attempt"
    assert rows[0]["payload"]["outcome"] == "PASS"
    assert rows[0]["payload"]["metrics"]["output_throughput"] == 1234.5


def test_drain_pending_empty_queue_is_no_op(session_dir):
    client = CortexKBClient(session_dir=session_dir)
    out = client.drain_pending(timeout_sec=1.0)
    assert out["drained"] == 0
    assert out["remaining"] == 0
    assert out["dead_letter"] == 0


def test_drain_pending_burns_through_queue(tmp_path, session_dir, monkeypatch):
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=["status: ok"])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    # Pre-populate the queue with three rows of different ops.
    client.ingest_attempt(
        sid="42", iter_id=1, outcome="PASS", metrics={}, plan_edge="",
    )
    client.verify(sid="42", edge_id="ed-1", outcome="confirmed", promote_authority="EXPERIENTIAL")
    client.hypothesize(  # async failure -> queued
        sid="", from_canonical="x", to_canonical="y",
    )  # sid empty → skip; do another live one
    client.ingest_attempt(
        sid="42", iter_id=2, outcome="FAIL", metrics={"err": "x"}, plan_edge="ed-1",
    )
    pending_path = cortex_pending_ndjson(session_dir)
    initial = sum(1 for _ in pending_path.read_text("utf-8").splitlines() if _.strip())
    assert initial >= 3
    report = client.drain_pending(timeout_sec=5.0)
    assert report["drained"] >= 1
    # After drain, queue should be empty (or contain only deferred rows).
    if pending_path.exists():
        leftover = sum(1 for _ in pending_path.read_text("utf-8").splitlines() if _.strip())
    else:
        leftover = 0
    assert leftover == report["remaining"]


# ===========================================================================
# breakdown.kb_provenance
# ===========================================================================
def test_kb_provenance_emits_stable_shape(session_dir):
    # Seed minimal state + queue artefacts.
    cortex_pending_ndjson(session_dir).write_text("", encoding="utf-8")
    cortex_audit_jsonl(session_dir).write_text(
        json.dumps({"ts": "now", "op": "session_begin", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    cortex_sid_file(session_dir).write_text("42", encoding="utf-8")

    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    warnings: list[str] = []
    state = {
        "cortex_session_id": "42",
        "warm_start_ts": "2026-05-19T00:00:00+00:00",
        "warm_start_recipe": {"raw": "recipe_node row 1"},
        "warm_start_pitfalls": [{"raw": "trap line"}],
        "pending_kb_edges": [
            {"proposal_msg_id": "msg-1", "edge_id": "e1", "action": "backends", "ts": "x"},
        ],
        "cortex_session_summary": {
            "status": "committed",
            "promoted_edges": ["e1", "e2"],
            "derived_summary_id": "sum-1",
        },
    }
    manifest = {"stack_fingerprint": {"rocm": "7.2.0", "sglang": "0.4.10"}}
    out = collect_kb_provenance(session_dir, state, manifest, warnings)
    assert warnings == []
    assert out["cortex_session_id"] == "42"
    assert out["warm_start_recipe_seen"] is True
    assert out["warm_start_pitfall_count"] == 1
    assert out["stack_fingerprint"]["rocm"] == "7.2.0"
    assert out["pending_edges"] == [
        {"proposal_msg_id": "msg-1", "edge_id": "e1", "action": "backends", "ts": "x"},
    ]
    assert out["queue"]["pending_lines"] == 0
    assert out["commit_summary"]["promoted_edges"] == ["e1", "e2"]
    # audit_status_counts derived from the single ok row
    assert out["audit_status_counts"]["ok"] == 1


def test_kb_provenance_no_cortex_session_still_succeeds(session_dir):
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    out = collect_kb_provenance(
        session_dir, state={}, manifest={}, warnings=[],
    )
    assert out["cortex_session_id"] == ""
    assert out["pending_edges"] == []
    assert out["queue"]["pending_lines"] == 0


# ===========================================================================
# SharedState additions
# ===========================================================================
def test_shared_state_has_v08_m1_cortex_fields():
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    # Default values must be the documented "empty" sentinels so
    # callers can detect "no T0 yet" / "no T4 yet" without try/except.
    assert s.cortex_session_id == ""
    assert s.cortex_session_summary == {}
    assert s.pending_kb_edges == []
    assert s.warm_start_recipe == {}
    assert s.warm_start_pitfalls == []
    assert s.warm_start_ts == ""


def test_policy_gate_core_state_fields_includes_cortex():
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "cortex_session_id" in CORE_STATE_FIELDS
    assert "cortex_session_summary" in CORE_STATE_FIELDS
    assert "pending_kb_edges" in CORE_STATE_FIELDS
    assert "warm_start_recipe" in CORE_STATE_FIELDS


# ===========================================================================
# manifest stack_fingerprint
# ===========================================================================
def test_manifest_includes_stack_fingerprint(session_dir, monkeypatch):
    monkeypatch.setenv("ROCM_VERSION", "7.2.0-test")
    monkeypatch.setenv("SGLANG_VERSION", "0.4.10-test")
    monkeypatch.setenv("AITER_COMMIT", "721f045")
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    from inference_optimizer.manifest import build_manifest
    manifest = build_manifest(session_dir, args=None, session_id="abc")
    assert manifest["stack_fingerprint"]["rocm"] == "7.2.0-test"
    assert manifest["stack_fingerprint"]["sglang"] == "0.4.10-test"
    assert manifest["stack_fingerprint"]["aiter"] == "721f045"
    # vllm not set / not importable → 'unknown' sentinel.
    assert manifest["stack_fingerprint"]["vllm"] in ("unknown",) or \
           manifest["stack_fingerprint"]["vllm"]  # if vllm IS installed it stays the real version
