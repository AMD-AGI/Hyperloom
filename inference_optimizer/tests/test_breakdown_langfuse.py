# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``session_breakdown.json`` ``langfuse`` section coverage.

Pins the receipt-vs-live two-tier source in
:func:`collectors.collect_langfuse`, the redaction contract, and the
``patch_breakdown_langfuse`` post-flush splice -- without importing the CLI
(which drags in fcntl-only modules on non-POSIX dev machines).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import (
    BREAKDOWN_FILENAME,
    build,
    patch_breakdown_langfuse,
    write_breakdown_json,
)
from inference_optimizer.breakdown import collectors
from inference_optimizer.orchestrator.trace import langfuse_emitter as lfe


def _seed_session(tmp_path: Path, **manifest_extra) -> Path:
    sd = tmp_path / "SID"
    (sd / "reports" / "trace").mkdir(parents=True)
    manifest = {
        "schema_version": 3,
        "session_id": "Model_20260101T000000Z_abcd1234",
        "model_name": "Model",
        "claw_session_id": "claw-abc-123",
    }
    manifest.update(manifest_extra)
    (sd / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sd


@pytest.fixture(autouse=True)
def _clear_registry():
    lfe._REGISTRY.clear()
    yield
    lfe._REGISTRY.clear()


# ---------------------------------------------------------------------------
# collect_langfuse: two-tier source
# ---------------------------------------------------------------------------
def test_collect_prefers_receipt_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = _seed_session(tmp_path)
    # A persisted receipt (as flush_session would leave it) wins.
    receipt = {
        "enabled": True,
        "disabled_reason": None,
        "config": {"enable_flag": True, "host": "https://lf.test",
                   "public_key_set": True, "secret_key_set": True,
                   "sdk_available": True},
        "trace_id": "deadbeef", "session_id": "claw-abc-123",
        "correlated_on": "claw_session_id",
        "counts": {"generations_sent": 7, "scores_sent": 3},
        "counts_final": True,
    }
    (sd / "reports" / "trace" / "langfuse_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    warnings: list[str] = []
    section = collectors.collect_langfuse(sd, {"claw_session_id": "claw-abc-123"}, warnings)
    assert section["receipt_source"] == "receipt_file"
    assert section["counts_final"] is True
    assert section["counts"]["generations_sent"] == 7
    assert warnings == []


def test_collect_live_emitter_when_no_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = _seed_session(tmp_path)
    warnings: list[str] = []
    section = collectors.collect_langfuse(sd, {"claw_session_id": "claw-abc-123"}, warnings)
    # No receipt on disk -> live emitter read (disabled here).
    assert section["receipt_source"] == "live_emitter"
    assert section["enabled"] is False
    assert section["disabled_reason"] == "disabled"
    assert section["correlated_on"] == "claw_session_id"


def test_collect_redacts_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_LANGFUSE_ENABLE", "1")
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-secret-xyz")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-secret-xyz")
    sd = _seed_session(tmp_path)
    warnings: list[str] = []
    section = collectors.collect_langfuse(sd, {}, warnings)
    blob = json.dumps(section)
    assert "pk-secret-xyz" not in blob
    assert "sk-secret-xyz" not in blob
    # Host (a URL) is fine to record; presence booleans only for keys.
    assert section["config"]["host"] == "https://lf.test"
    assert section["config"]["public_key_set"] is True


# ---------------------------------------------------------------------------
# build() integration + patch splice
# ---------------------------------------------------------------------------
def test_build_includes_langfuse_section(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = _seed_session(tmp_path)
    breakdown = build(sd)
    assert "langfuse" in breakdown
    assert breakdown["langfuse"]["enabled"] is False


def test_patch_splices_post_flush_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = _seed_session(tmp_path)
    # Breakdown written first (pre-flush): live_emitter source, counts not final.
    write_breakdown_json(sd)
    before = json.loads((sd / BREAKDOWN_FILENAME).read_text(encoding="utf-8"))
    assert before["langfuse"]["counts_final"] is False

    # Now a receipt lands (as flush_session would write it) with final counts.
    receipt = {
        "enabled": True, "disabled_reason": None,
        "config": {"enable_flag": True, "host": "https://lf.test",
                   "public_key_set": True, "secret_key_set": True,
                   "sdk_available": True},
        "trace_id": "abc", "session_id": "claw-abc-123",
        "correlated_on": "claw_session_id",
        "counts": {"generations_sent": 5}, "counts_final": True,
    }
    (sd / "reports" / "trace" / "langfuse_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    patched = patch_breakdown_langfuse(sd)
    assert patched is True
    after = json.loads((sd / BREAKDOWN_FILENAME).read_text(encoding="utf-8"))
    assert after["langfuse"]["counts_final"] is True
    assert after["langfuse"]["counts"]["generations_sent"] == 5
    assert after["langfuse"]["receipt_source"] == "receipt_file"
    # Other sections untouched by the splice.
    assert after["session"] == before["session"]


def test_patch_noop_without_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_LANGFUSE_ENABLE", raising=False)
    sd = _seed_session(tmp_path)
    write_breakdown_json(sd)
    assert patch_breakdown_langfuse(sd) is False
