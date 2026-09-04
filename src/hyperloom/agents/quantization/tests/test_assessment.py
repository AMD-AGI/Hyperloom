"""Branch coverage for hyperloom.agents.quantization.driver.assessment helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.agents.quantization.driver import assessment as A
from hyperloom.agents.quantization.driver.assessment import (
    Assessment,
    build_assessment,
    derive_status,
)
from hyperloom.agents.quantization.driver.outcomes import OutcomeId
from hyperloom.agents.quantization.driver.result_collector import collect_artifacts


def _base_artifacts(tmp_path: Path):
    return collect_artifacts(tmp_path)


def test_parse_blocked_outcome() -> None:
    assert A._parse_blocked_outcome(None) is None
    assert A._parse_blocked_outcome("nothing here") is None
    assert A._parse_blocked_outcome("outcome_id: eval_oom") == OutcomeId.eval_oom
    assert A._parse_blocked_outcome("outcome_id: not_a_real_outcome") is None


def test_classify_sdk_phase_error() -> None:
    cls = A._classify_sdk_phase_error
    assert cls("cuda out of memory", "exec") == OutcomeId.exec_oom
    assert cls("safetensorerror loading", "exec") == OutcomeId.exec_model_load_failed
    assert cls("calibration dataloader broke", "exec") == OutcomeId.exec_calibration_data_missing
    assert cls("save_pretrained ioerror", "export") == OutcomeId.export_crashed
    assert cls("out of memory", "eval") == OutcomeId.eval_oom
    assert cls("vllm engine.start failed", "eval") == OutcomeId.quantized_load_failed
    assert cls("some generic eval error", "eval") == OutcomeId.eval_env_unavailable
    assert cls("config.json: no such file", "intake") == OutcomeId.model_path_unreachable
    assert cls("nothing matches", "unknown_phase") is None


def test_eval_backend_unavailable_is_not_quantized_load_failure(tmp_path) -> None:
    art = dataclasses.replace(
        _base_artifacts(tmp_path),
        eval_skipped_reason="offline host has no Docker daemon, vllm, or sglang installed",
    )
    assert A._classify_eval_outcome(art, acceptable_eval_gap=0.03) == OutcomeId.eval_env_unavailable


def test_classify_phase_artifact_gap_pyyaml(tmp_path) -> None:
    art = dataclasses.replace(
        _base_artifacts(tmp_path),
        manifest_present=True,
        manifest_parse_error="pyyaml_missing",
    )
    # pyyaml missing -> nice_to_have_skipped.
    assert A._classify_phase_artifact_gap(art, "exec") == OutcomeId.nice_to_have_skipped

    art2 = dataclasses.replace(
        _base_artifacts(tmp_path),
        manifest_present=True,
        manifest_parse_error="yaml_error: boom",
    )
    assert A._classify_phase_artifact_gap(art2, "exec") == OutcomeId.manifest_artifact_invalid_or_missing


def test_build_assessment_empty_and_bad_gap(tmp_path) -> None:
    with pytest.raises(ValueError, match="attempts must be non-empty"):
        build_assessment([], workspace=tmp_path)

    art = dataclasses.replace(
        _base_artifacts(tmp_path),
        eval_report_data={"relative_gap": "not-a-float"},
    )
    a2 = build_assessment([OutcomeId.eval_gap_accepted], workspace=tmp_path, artifacts=art)
    # Bad relative_gap swallowed.
    assert a2.eval_gap is None


def test_derive_status_must_have_recover_fails(tmp_path) -> None:
    art = dataclasses.replace(
        _base_artifacts(tmp_path),
        has_config_json=False,
        has_tokenizer=False,
        strict_validation=True,
    )
    assessment = Assessment(
        final=OutcomeId.must_have_config_missing_or_invalid,
        attempts=(OutcomeId.must_have_config_missing_or_invalid,),
        recovered=False,
        eval_gap=None,
        notes=(),
    )
    # MUST-have recover without artifact on disk -> failed.
    assert derive_status(assessment, art) == "failed"


def test_derive_status_success_and_partial(tmp_path) -> None:
    art = _base_artifacts(tmp_path)
    ok = Assessment(final=None, attempts=(None,), recovered=False, eval_gap=None, notes=())
    assert derive_status(ok, art) == "success"
    partial = Assessment(
        final=OutcomeId.eval_gap_exceeded,
        attempts=(OutcomeId.eval_gap_exceeded,),
        recovered=False,
        eval_gap=0.1,
        notes=(),
    )
    assert derive_status(partial, art) == "partial"
