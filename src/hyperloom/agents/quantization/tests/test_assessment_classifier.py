"""classify_attempt — one fixture per outcome row (where artifact-driven)."""

from __future__ import annotations

import pytest

from hyperloom.agents.quantization.driver.assessment import (
    _parse_blocked_outcome,
    build_assessment,
    classify_attempt,
    derive_status,
)
from hyperloom.agents.quantization.driver.outcomes import SUCCESS_TAGS, OutcomeId
from hyperloom.agents.quantization.driver.result_collector import collect_artifacts


# ───────────────────────────────────────────────────────────────────────────── bootstrap / sdk_error path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        ("RuntimeError: QUARK_ROOT not set; quark_root missing", OutcomeId.quark_root_missing),
        ("PermissionError: [Errno 13] cannot write to workspace /tmp/foo", OutcomeId.workspace_unwritable),
        ("Quark skill.md missing at .claude/skills/quark-torch-ptq/SKILL.md", OutcomeId.quark_skill_unavailable),
        ("anthropic.RateLimitError: 429 rate limit exceeded", OutcomeId.sdk_runtime_error),
        ("Mysterious upstream error nobody has seen before", OutcomeId.unclassified_failure),
    ],
    ids=["quark_root_missing", "workspace_unwritable", "quark_skill_unavailable", "rate_limit", "unknown_error"],
)
def test_an_sdk_error_is_classified_by_its_text(build_workspace, sdk_error, expected):
    ws = build_workspace()
    assert classify_attempt(collect_artifacts(ws), sdk_error=sdk_error) == expected


# ───────────────────────────────────────────────────────────────────────────── explicit blocked.md outcome marker
# ─────────────────────────────────────────────────────────────────────────────


def test_blocked_md_outcome_marker_wins(build_workspace):
    # An explicit blocked.md marker overrides successful artifacts.
    ws = build_workspace(blocked_md="outcome_id: upstream_change_required\nreason: needs quark patch")
    assert classify_attempt(collect_artifacts(ws)) == OutcomeId.upstream_change_required


def test_blocked_md_invalid_outcome_id_ignored(build_workspace):
    # Unknown outcome string in blocked.md → fall through to disk scan.
    ws = build_workspace(blocked_md="outcome_id: not_a_real_outcome\n")
    assert classify_attempt(collect_artifacts(ws)) is None  # successful artifacts


def test_blocked_md_eval_gap_accepted_does_not_bypass(build_workspace):
    ws = build_workspace(
        blocked_md="outcome_id: eval_gap_accepted\n",
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
    )
    assert classify_attempt(collect_artifacts(ws)) != OutcomeId.eval_gap_accepted


def test_blocked_md_accepts_every_known_non_success_outcome():
    """New known failures are blocked unless explicitly classified as success."""
    for outcome in list(OutcomeId):
        parsed = _parse_blocked_outcome(f"outcome_id: {outcome.value}\n")
        assert parsed is (None if outcome in SUCCESS_TAGS else outcome)


# ───────────────────────────────────────────────────────────────────────────── phase-aware artifact gaps
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"model_analysis": False, "quant_plan": False, "last_phase": "intake"},
            OutcomeId.analysis_artifact_invalid_or_missing,
        ),
        ({"quant_plan": False, "last_phase": "plan"}, OutcomeId.plan_artifact_invalid_or_missing),
        ({"last_phase": "manifest"}, OutcomeId.manifest_artifact_invalid_or_missing),
    ],
    ids=["analysis_under_intake", "plan_under_plan", "manifest_under_manifest"],
)
def test_a_missing_phase_artifact_is_named_by_the_last_phase(build_workspace, overrides, expected):
    ws = build_workspace(
        include_manifest=False,
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        **overrides,
    )
    assert classify_attempt(collect_artifacts(ws)) == expected


# ───────────────────────────────────────────────────────────────────────────── MUST-have on quantized dir
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("include_weights", OutcomeId.must_have_weights_missing),
        ("include_config", OutcomeId.must_have_config_missing_or_invalid),
        ("include_tokenizer", OutcomeId.must_have_tokenizer_missing),
    ],
    ids=["weights", "config", "tokenizer"],
)
def test_a_missing_must_have_file_is_named(build_workspace, missing, expected):
    ws = build_workspace(**{missing: False}, include_validation_report=False)
    assert classify_attempt(collect_artifacts(ws)) == expected


# ───────────────────────────────────────────────────────────────────────────── validator
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"include_validation_report": False}, OutcomeId.validation_report_absent),
        ({"validation_tag": "md5_fail"}, OutcomeId.must_validate_md5_mismatch),
        ({"validation_tag": "config_fail"}, OutcomeId.must_validate_config_mismatch),
        ({"validation_tag": "fuzzy_fail"}, OutcomeId.fuzzy_check_failed),
        ({"validation_tag": "aux_fail"}, OutcomeId.should_have_aux_missing),
        ({"validation_tag": "must_validate_skipped"}, OutcomeId.must_validate_skipped),
    ],
    ids=["report_absent", "md5_fail", "config_fail", "fuzzy_fail", "aux_fail", "must_validate_skipped"],
)
def test_the_validation_report_decides_the_outcome(build_workspace, overrides, expected):
    ws = build_workspace(**overrides)
    assert classify_attempt(collect_artifacts(ws)) == expected


# ───────────────────────────────────────────────────────────────────────────── eval
# ─────────────────────────────────────────────────────────────────────────────


def test_eval_gap_within_threshold_no_narrative_when_zero(build_workspace):
    # Default eval_report has gap=0.0 — clean success, no narrative tag.
    ws = build_workspace()
    assert classify_attempt(collect_artifacts(ws)) is None


def test_eval_gap_within_threshold_narrative_tag(build_workspace):
    ws = build_workspace(
        eval_report={
            "metric_name": "gsm8k",
            "dataset": "gsm8k",
            "backend": "vllm",
            "source_score": 0.5,
            "quantized_score": 0.49,
            "relative_gap": 0.02,
        }
    )
    assert classify_attempt(collect_artifacts(ws), acceptable_eval_gap=0.03) == OutcomeId.eval_gap_accepted


def test_eval_gap_exceeded(build_workspace):
    ws = build_workspace(
        eval_report={
            "metric_name": "gsm8k",
            "dataset": "gsm8k",
            "backend": "vllm",
            "source_score": 0.5,
            "quantized_score": 0.45,
            "relative_gap": 0.10,
        }
    )
    assert classify_attempt(collect_artifacts(ws), acceptable_eval_gap=0.03) == OutcomeId.eval_gap_exceeded


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("docker not available", OutcomeId.eval_env_unavailable),
        ("quantized engine OOM on MI300X", OutcomeId.eval_oom),
        ("vLLM engine.start failed: quantized model can't load", OutcomeId.quantized_load_failed),
    ],
    ids=["env_unavailable", "oom", "quantized_load_failed"],
)
def test_the_eval_skipped_reason_is_classified(build_workspace, reason, expected):
    ws = build_workspace(include_eval_report=False, eval_skipped_reason=reason)
    assert classify_attempt(collect_artifacts(ws)) == expected


def test_eval_report_malformed_treated_as_env_unavailable(build_workspace):
    ws = build_workspace(
        eval_report={"metric_name": "gsm8k"},  # missing required keys
    )
    assert classify_attempt(collect_artifacts(ws)) == OutcomeId.eval_env_unavailable


# ───────────────────────────────────────────────────────────────────────────── phase-tagged sdk errors
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("last_phase", "sdk_error", "expected"),
    [
        ("exec", "torch.OutOfMemoryError: CUDA out of memory", OutcomeId.exec_oom),
        ("export", "OSError: [Errno 28] No space left on device during save_pretrained", OutcomeId.export_crashed),
        ("exec", "RuntimeError: calibration dataloader returned 0 samples", OutcomeId.exec_calibration_data_missing),
    ],
    ids=["exec_oom", "export_crashed", "exec_calibration_data_missing"],
)
def test_a_phase_tagged_sdk_error_is_classified(build_workspace, last_phase, sdk_error, expected):
    ws = build_workspace(
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase=last_phase,
    )
    assert classify_attempt(collect_artifacts(ws), sdk_error=sdk_error) == expected


# ───────────────────────────────────────────────────────────────────────────── Assessment assembly
# ─────────────────────────────────────────────────────────────────────────────


def test_build_assessment_single_clean_attempt(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([None], artifacts=art)
    assert a.final is None
    assert a.attempts == (None,)
    assert a.recovered is False
    assert a.eval_gap == 0.0


def test_build_assessment_recovered_after_failure(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.exec_oom, None], artifacts=art)
    assert a.final is None
    assert a.recovered is True


def test_build_assessment_failed_final(build_workspace):
    ws = build_workspace(validation_tag="md5_fail")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_md5_mismatch], artifacts=art)
    assert a.final == OutcomeId.must_validate_md5_mismatch
    assert a.recovered is False


def test_assessment_to_dict_roundtrip(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.exec_oom, OutcomeId.eval_gap_accepted], artifacts=art)
    d = a.to_dict()
    assert d["final"] == "eval_gap_accepted"
    assert d["attempts"] == ["exec_oom", "eval_gap_accepted"]
    assert d["recovered"] is True


# ───────────────────────────────────────────────────────────────────────────── derive_status
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "final", "expected"),
    [
        ({}, None, "success"),
        ({}, OutcomeId.eval_gap_accepted, "success"),
        ({"validation_tag": "md5_fail"}, OutcomeId.must_validate_md5_mismatch, "failed"),
        ({}, OutcomeId.eval_gap_exceeded, "partial"),
        (
            {"include_eval_report": False, "eval_skipped_reason": "docker missing"},
            OutcomeId.eval_env_unavailable,
            "partial",
        ),
        ({}, OutcomeId.unclassified_failure, "failed"),
    ],
    ids=[
        "clean_success",
        "eval_gap_accepted_is_success",
        "md5_mismatch_failed",
        "eval_gap_exceeded_partial",
        "eval_env_unavailable_is_partial",
        "unclassified_is_failed",
    ],
)
def test_derive_status_follows_the_final_outcome(build_workspace, overrides, final, expected):
    ws = build_workspace(**overrides)
    art = collect_artifacts(ws)
    a = build_assessment([final], artifacts=art)
    assert derive_status(a, art) == expected


@pytest.mark.parametrize(("strict", "expected"), [("1", "failed"), ("0", "partial")], ids=["strict", "lenient"])
def test_derive_status_must_validate_skipped(build_workspace, monkeypatch, strict, expected):
    monkeypatch.setenv("HYPERLOOM_QUANT_STRICT_VALIDATION", strict)
    ws = build_workspace(validation_tag="must_validate_skipped")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_skipped], artifacts=art)
    assert derive_status(a, art) == expected
