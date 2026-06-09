"""Tests for ``collect_artifacts`` — disk scan only, no classification."""

from __future__ import annotations

import pytest

from quantization_agent.driver.result_collector import (
    STRICT_VALIDATION_ENV,
    collect_artifacts,
)


def test_clean_workspace_returns_zero_artifacts(tmp_path):
    art = collect_artifacts(tmp_path)
    assert art.manifest_present is False
    assert art.quantized_model_dir is None
    assert art.quantized_dir_exists is False
    assert art.validation_report_present is False
    assert art.eval_report_data is None
    assert art.fix_hypothesis_attempts == ()
    assert art.last_phase is None


def test_full_artifact_set(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    assert art.manifest_present is True
    assert art.manifest_parse_error is None
    assert art.quantized_model_dir is not None and art.quantized_model_dir.exists()
    assert art.has_config_json
    assert art.has_weights
    assert art.has_tokenizer
    assert art.validation_report_present
    assert art.validation_steps.md5 == "ok"
    assert art.validation_steps.config == "ok"
    assert art.validation_steps.fuzzy == "ok"
    assert art.validation_steps.auxiliary == "ok"
    assert art.eval_report_data is not None
    assert art.eval_report_data["relative_gap"] == 0.0


def test_validation_step_fail_parsed(build_workspace):
    ws = build_workspace(validation_tag="md5_fail")
    art = collect_artifacts(ws)
    assert art.validation_steps.md5 == "FAIL"
    assert art.validation_steps.config == "ok"


def test_validation_step_skipped_parsed(build_workspace):
    ws = build_workspace(validation_tag="must_validate_skipped")
    art = collect_artifacts(ws)
    assert art.validation_steps.md5 == "skipped"
    assert art.validation_steps.config == "skipped"


def test_missing_validation_step_returns_none(tmp_path):
    # Report present but without all 4 step lines.
    (tmp_path / "validation_report.md").write_text(
        "**Step 4 — fuzzy tensor names**: ok\n", encoding="utf-8"
    )
    art = collect_artifacts(tmp_path)
    assert art.validation_steps.fuzzy == "ok"
    assert art.validation_steps.md5 is None
    assert art.validation_steps.config is None
    assert art.validation_steps.auxiliary is None


def test_manifest_missing_outputs_key(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "run_manifest.yaml").write_text(
        "version: '1'\n", encoding="utf-8"
    )
    art = collect_artifacts(tmp_path)
    assert art.manifest_present is True
    assert art.manifest_parse_error == "missing_outputs_quantized_model_dir"
    assert art.quantized_model_dir is None


def test_manifest_yaml_malformed(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "run_manifest.yaml").write_text("not: [valid: yaml", encoding="utf-8")
    art = collect_artifacts(tmp_path)
    assert art.manifest_present is True
    assert art.manifest_parse_error is not None
    assert "yaml_error" in art.manifest_parse_error


def test_eval_report_json_malformed(tmp_path):
    (tmp_path / "eval_report.json").write_text("{not json", encoding="utf-8")
    art = collect_artifacts(tmp_path)
    assert art.eval_report_present is True
    assert art.eval_report_data is None


def test_fix_hypothesis_attempts_sorted(build_workspace):
    ws = build_workspace(fix_hypotheses=(3, 1, 2))
    art = collect_artifacts(ws)
    assert art.fix_hypothesis_attempts == (1, 2, 3)


def test_last_phase_stripped(tmp_path):
    (tmp_path / "last_phase.txt").write_text("  exec  \n", encoding="utf-8")
    art = collect_artifacts(tmp_path)
    assert art.last_phase == "exec"


def test_blocked_md_carried_through(build_workspace):
    ws = build_workspace(blocked_md="outcome_id: exec_oom\n")
    art = collect_artifacts(ws)
    assert art.blocked_reason is not None
    assert "exec_oom" in art.blocked_reason


def test_strict_validation_default_true(tmp_path, monkeypatch):
    monkeypatch.delenv(STRICT_VALIDATION_ENV, raising=False)
    art = collect_artifacts(tmp_path)
    assert art.strict_validation is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", ""])
def test_strict_validation_disabled_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv(STRICT_VALIDATION_ENV, value)
    art = collect_artifacts(tmp_path)
    assert art.strict_validation is False


def test_quantized_dir_resolved_relative(tmp_path):
    pytest.importorskip("yaml")
    qdir = tmp_path / "out"
    qdir.mkdir()
    (qdir / "config.json").write_text("{}", encoding="utf-8")
    (qdir / "model.safetensors").write_bytes(b"\x00")
    (qdir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_manifest.yaml").write_text(
        "outputs:\n  quantized_model_dir: out\n", encoding="utf-8"
    )
    art = collect_artifacts(tmp_path)
    assert art.quantized_model_dir is not None
    assert art.quantized_model_dir.name == "out"
    assert art.has_weights
