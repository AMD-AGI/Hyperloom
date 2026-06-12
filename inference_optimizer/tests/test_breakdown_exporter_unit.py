# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the session_breakdown.json exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import exporter as ex


# ---- _load_state / _load_manifest ----

def test_load_state_missing(tmp_path):
    warnings = []
    assert ex._load_state(tmp_path, warnings) == {}
    assert any("state.json missing" in w for w in warnings)


def test_load_state_valid(tmp_path):
    (tmp_path / "state.json").write_text('{"session_id": "s"}', encoding="utf-8")
    assert ex._load_state(tmp_path, [])["session_id"] == "s"


def test_load_state_parse_error(tmp_path):
    (tmp_path / "state.json").write_text("{bad", encoding="utf-8")
    warnings = []
    assert ex._load_state(tmp_path, warnings) == {}
    assert any("failed to parse state.json" in w for w in warnings)


def test_load_manifest_missing(tmp_path):
    warnings = []
    assert ex._load_manifest(tmp_path, warnings) == {}
    assert any("manifest.json missing" in w for w in warnings)


def test_load_manifest_parse_error(tmp_path):
    (tmp_path / "manifest.json").write_text("{bad", encoding="utf-8")
    warnings = []
    assert ex._load_manifest(tmp_path, warnings) == {}
    assert any("failed to parse manifest.json" in w for w in warnings)


# ---- _safe_collect ----

def test_safe_collect_success():
    assert ex._safe_collect("x", lambda: 42, []) == 42


def test_safe_collect_exception_default_dict():
    warnings = []
    out = ex._safe_collect("x", lambda: (_ for _ in ()).throw(ValueError("e")), warnings)
    assert out == {}
    assert any("collector:x failed" in w for w in warnings)


def test_safe_collect_exception_with_default():
    def boom():
        raise RuntimeError("e")

    assert ex._safe_collect("x", boom, [], default=[]) == []


# ---- _json_default ----

def test_json_default_path():
    assert ex._json_default(Path("/x")) == "/x"


def test_json_default_set():
    assert ex._json_default({3, 1, 2}) == [1, 2, 3]


def test_json_default_typeerror():
    with pytest.raises(TypeError):
        ex._json_default(object())


# ---- build ----

def test_build_empty_session(tmp_path):
    out = ex.build(tmp_path)
    assert out["exporter_version"] == ex.EXPORTER_VERSION
    assert "warnings" in out
    assert "session" in out
    # Missing state/manifest produce warnings.
    assert any("missing" in w for w in out["warnings"])


def test_build_include_transcripts_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "1")
    out = ex.build(tmp_path)
    assert out["schema_version"] is not None


# ---- write_breakdown_json ----

def test_write_breakdown_json(tmp_path):
    target = ex.write_breakdown_json(tmp_path)
    assert target.name == ex.BREAKDOWN_FILENAME
    assert target.is_file()
    data = json.loads(target.read_text())
    assert data["exporter_version"] == ex.EXPORTER_VERSION


def test_write_breakdown_json_custom_output(tmp_path):
    out = tmp_path / "sub" / "bd.json"
    target = ex.write_breakdown_json(tmp_path, output_path=out)
    assert target == out.resolve()
    assert out.is_file()


# ---- patch_breakdown_langfuse ----

def test_patch_breakdown_langfuse_no_breakdown(tmp_path):
    # No receipt and no breakdown file -> False.
    assert ex.patch_breakdown_langfuse(tmp_path) is False


# ---- write_minimal_final_report ----

def test_write_minimal_final_report_creates(tmp_path):
    target = ex.write_minimal_final_report(tmp_path)
    assert target.name == "final.md"
    assert target.is_file()
    text = target.read_text()
    assert "emergency final report" in text


def test_write_minimal_final_report_idempotent(tmp_path):
    target = ex.write_minimal_final_report(tmp_path)
    target.write_text("PRESERVED", encoding="utf-8")
    again = ex.write_minimal_final_report(tmp_path)
    assert again.read_text() == "PRESERVED"


def test_write_minimal_final_report_with_attempts(tmp_path):
    # Populate SharedState so the last_* attempt + sweep branches render.
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(tmp_path)
    state.last_sweep = {"grid_size": 3, "best_overall": {"output_throughput": 99.5}, "ts": "t0"}
    state.last_baseline = {"tput": 50.0, "ts": "t0"}
    state.save(tmp_path)

    target = ex.write_minimal_final_report(tmp_path)
    text = target.read_text()
    assert "grid_size=3" in text
    assert "last_baseline" in text


def test_patch_breakdown_langfuse_success(tmp_path):
    from inference_optimizer.orchestrator.trace.langfuse_emitter import _receipt_path

    # Write a breakdown first, then a post-flush receipt to splice in.
    ex.write_breakdown_json(tmp_path)
    receipt_path = _receipt_path(tmp_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({"enabled": True, "counts_final": True}), encoding="utf-8")

    assert ex.patch_breakdown_langfuse(tmp_path) is True
    bd = json.loads((tmp_path / ex.BREAKDOWN_FILENAME).read_text())
    assert bd["langfuse"]["enabled"] is True
    # Second call is a no-op (already current).
    assert ex.patch_breakdown_langfuse(tmp_path) is False
