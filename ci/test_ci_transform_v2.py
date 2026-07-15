# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/transform_to_session_summary_v2.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import transform_to_session_summary_v2 as tx  # noqa: E402


# ── safe_get ──


def test_safe_get_nested():
    assert tx.safe_get({"a": {"b": 1}}, "a", "b") == 1


def test_safe_get_missing_returns_default():
    assert tx.safe_get({"a": {}}, "a", "b", default="x") == "x"


def test_safe_get_non_dict_returns_default():
    assert tx.safe_get({"a": 5}, "a", "b", default=None) is None


def test_safe_get_none_value_returns_default():
    assert tx.safe_get({"a": None}, "a", default="d") == "d"


# ── _patch_baseline ──


def test_patch_baseline_empty_default():
    data = {"baseline": {}}
    tx._patch_baseline(data)
    assert data["baseline"]["extra_server_args"] == ""
    assert data["baseline"]["extra_envs"] == {}


def test_patch_baseline_envs_from_invocation():
    data = {"baseline": {"extra_server_args": "x", "invocation": {"extra_envs": {"A": "1"}}}}
    tx._patch_baseline(data)
    assert data["baseline"]["extra_envs"] == {"A": "1"}


def test_patch_baseline_no_baseline():
    assert tx._patch_baseline({}) == []


# ── _best_gain_for_phase ──


def test_best_gain_from_param_search():
    data = {"param_search": {"params": {"top_by_gain": [{"gain_pct": 12.5}]}}}
    assert tx._best_gain_for_phase(data, "params") == 12.5


def test_best_gain_from_phase_timeline():
    data = {
        "phase_timeline": [
            {"action": "sweep", "extras": {"best_gain_pct_vs_base": 3.0}},
            {"action": "sweep", "extras": {"best_gain_pct_vs_base": 7.0}},
        ]
    }
    assert tx._best_gain_for_phase(data, "sweep") == 7.0


def test_best_gain_none():
    assert tx._best_gain_for_phase({}, "params") is None


# ── _patch_capability_summary ──


def test_patch_capability_summary_fills_phases():
    data = {
        "capability_summary": {"params": {}, "validate_stack": {"last_validated_gain_pct": 9.0}},
        "param_search": {"params": {"top_by_gain": [{"gain_pct": 4.0}]}},
    }
    tx._patch_capability_summary(data)
    assert data["capability_summary"]["params"]["best_gain_pct"] == 4.0
    assert data["capability_summary"]["validate_stack"]["best_gain_pct"] == 9.0


def test_patch_capability_summary_no_cs():
    assert tx._patch_capability_summary({}) == []


# ── _patch_phase_timeline ──


def test_patch_phase_timeline_alias_added():
    data = {"phase_timeline": [{"extras": {"candidate_extra_server_args": "--x"}}]}
    tx._patch_phase_timeline(data)
    assert data["phase_timeline"][0]["extras"]["best_extra_server_args"] == "--x"


def test_patch_phase_timeline_no_pt():
    assert tx._patch_phase_timeline({}) == []


def test_patch_baseline_migrates_legacy_sglang_args():
    """Legacy baseline.extra_sglang_args seeds extra_server_args (no data loss)."""
    data = {"baseline": {"extra_sglang_args": "--enable-foo"}}
    tx._patch_baseline(data)
    assert data["baseline"]["extra_server_args"] == "--enable-foo"


def test_patch_baseline_modern_key_takes_precedence_over_legacy():
    """An explicit extra_server_args is never overwritten by the legacy key."""
    data = {"baseline": {"extra_server_args": "--modern", "extra_sglang_args": "--legacy"}}
    tx._patch_baseline(data)
    assert data["baseline"]["extra_server_args"] == "--modern"


def test_patch_phase_timeline_migrates_legacy_candidate_sglang_args():
    """Legacy candidate_extra_sglang_args maps to best_extra_server_args."""
    data = {"phase_timeline": [{"extras": {"candidate_extra_sglang_args": "--candidate-foo"}}]}
    tx._patch_phase_timeline(data)
    assert data["phase_timeline"][0]["extras"]["best_extra_server_args"] == "--candidate-foo"


def test_patch_phase_timeline_modern_candidate_wins_over_legacy():
    """When both candidate keys exist, the modern one is chosen."""
    data = {
        "phase_timeline": [
            {
                "extras": {
                    "candidate_extra_server_args": "--modern",
                    "candidate_extra_sglang_args": "--legacy",
                }
            }
        ]
    }
    tx._patch_phase_timeline(data)
    assert data["phase_timeline"][0]["extras"]["best_extra_server_args"] == "--modern"


def test_is_already_v2_false_for_legacy_candidate_sglang_args():
    """A phase entry with only the legacy candidate key still needs patching."""
    doc = _v2_doc()
    doc["phase_timeline"] = [{"extras": {"candidate_extra_sglang_args": "--x"}}]
    assert tx.is_already_v2(doc) is False


def test_transform_end_to_end_preserves_legacy_args():
    """Full transform() keeps both legacy baseline + phase-timeline server args."""
    legacy = {
        "baseline": {"extra_sglang_args": "--enable-foo"},
        "capability_summary": {
            p: {"best_gain_pct": 1.0} for p in ("params", "backends", "sweep", "geak", "oob")
        },
        "phase_timeline": [{"extras": {"candidate_extra_sglang_args": "--candidate-foo"}}],
        "kernel_lifecycle": {"detected": [{"geak": {}, "oob": {}}]},
    }
    out = tx.transform(legacy)["data"]
    assert out["baseline"]["extra_server_args"] == "--enable-foo"
    assert out["phase_timeline"][0]["extras"]["best_extra_server_args"] == "--candidate-foo"


# ── _aggregate_backend ──


def test_aggregate_backend_picks_best():
    data = {
        "kernel_decision_path": [
            {
                "kid": "k1",
                "steps": [
                    {"backend": "geak", "speedup": 1.2, "outcome": "PARTIAL"},
                    {"backend": "geak", "speedup": 1.5, "decision": "KEEP"},
                    {"backend": "oob", "speedup": 9.0},
                ],
            }
        ]
    }
    agg = tx._aggregate_backend(data, "k1", "geak")
    assert agg == {"decision": "KEEP", "best_speedup": 1.5}


def test_aggregate_backend_no_path():
    assert tx._aggregate_backend({}, "k1", "geak") == {"decision": None, "best_speedup": None}


# ── _patch_detected_kernels ──


def test_patch_detected_kernels_fills():
    data = {
        "kernel_lifecycle": {"detected": [{"kernel_id": "k1"}]},
        "kernel_decision_path": [
            {
                "kid": "k1",
                "steps": [
                    {"backend": "geak", "speedup": 2.0, "decision": "KEEP"},
                ],
            }
        ],
    }
    tx._patch_detected_kernels(data)
    k = data["kernel_lifecycle"]["detected"][0]
    assert k["geak"]["decision"] == "KEEP"
    assert k["oob"] == {"decision": None, "best_speedup": None}


def test_patch_detected_kernels_no_detected():
    assert tx._patch_detected_kernels({}) == []


# ── is_already_v2 / transform ──


def _v2_doc() -> dict:
    return {
        "baseline": {"extra_server_args": "", "extra_envs": {}},
        "capability_summary": {p: {"best_gain_pct": 1.0} for p in ("params", "backends", "sweep", "geak", "oob")},
        "phase_timeline": [],
        "kernel_lifecycle": {"detected": []},
    }


def test_is_already_v2_true():
    assert tx.is_already_v2(_v2_doc()) is True


def test_is_already_v2_false_for_legacy():
    assert tx.is_already_v2({"baseline": {}}) is False


def test_transform_already_v2_passthrough():
    out = tx.transform(_v2_doc())
    assert out["source"] == "hyperloom_v2"
    assert out["error"] is None


def test_transform_legacy_applies_patches():
    legacy = {
        "baseline": {},
        "capability_summary": {"params": {}},
        "phase_timeline": [{"extras": {"candidate_extra_server_args": "--z"}}],
        "kernel_lifecycle": {"detected": [{"kernel_id": "k"}]},
    }
    out = tx.transform(legacy)
    assert out["source"] == "claw_legacy_phased"
    assert "_v2_patches" in out["data"]
    assert out["data"]["_v2_patches"]  # non-empty


def test_transform_does_not_mutate_input():
    legacy = {"baseline": {}}
    tx.transform(legacy)
    assert legacy == {"baseline": {}}


# ── transform_file / _output_name_for / main ──


def test_transform_file_default_suffix(tmp_path: Path):
    p = tmp_path / "session_breakdown.json"
    p.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    out = tx.transform_file(p, None)
    assert out == p.with_suffix(".v2.json")
    assert out.exists()


def test_transform_file_explicit_out(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    target = tmp_path / "sub" / "out.json"
    out = tx.transform_file(p, target)
    assert out == target
    assert json.loads(target.read_text())["source"] == "claw_legacy_phased"


def test_transform_file_stdout(tmp_path: Path, capsys):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    out = tx.transform_file(p, Path("-"))
    assert str(out) == "-"
    assert "claw_legacy_phased" in capsys.readouterr().out


def test_transform_file_non_object_raises(tmp_path: Path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        tx.transform_file(p, None)


def test_output_name_for_session_id(tmp_path: Path):
    f = tmp_path / "session_breakdown.json"
    f.write_text(json.dumps({"session": {"session_id": "abc123"}}), encoding="utf-8")
    assert tx._output_name_for(f, tmp_path) == Path("abc123.json")


def test_output_name_for_fallback(tmp_path: Path):
    f = tmp_path / "nested" / "session_breakdown.json"
    f.parent.mkdir()
    f.write_text(json.dumps({}), encoding="utf-8")
    assert tx._output_name_for(f, tmp_path) == Path("nested/session_breakdown.json")


def test_main_batch_mode(tmp_path: Path, monkeypatch, capsys):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "session_breakdown.json").write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["t.py", "--in-dir", str(in_dir), "--out-dir", str(out_dir)])
    tx.main()
    produced = list(out_dir.rglob("*.json"))
    assert produced


def test_main_batch_no_files(tmp_path: Path, monkeypatch, capsys):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["t.py", "--in-dir", str(in_dir), "--out-dir", str(out_dir)])
    tx.main()
    assert "no input" in capsys.readouterr().err


def test_main_single_file(tmp_path: Path, monkeypatch, capsys):
    p = tmp_path / "session_breakdown.json"
    p.write_text(json.dumps({"baseline": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["t.py", str(p)])
    tx.main()
    assert p.with_suffix(".v2.json").exists()


def test_main_single_file_error_reported(tmp_path: Path, monkeypatch, capsys):
    p = tmp_path / "session_breakdown.json"
    p.write_text("[1,2]", encoding="utf-8")  # not an object
    monkeypatch.setattr(sys, "argv", ["t.py", str(p)])
    tx.main()
    assert "[err]" in capsys.readouterr().err
