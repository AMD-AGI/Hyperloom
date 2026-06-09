"""classify_attempt — one fixture per outcome row (where artifact-driven).

A few outcomes (e.g. quark_skill_unavailable, validator_self_test_failed) are
only visible to the runner via sdk_error or the LLM's blocked.md; the
classifier covers them via the sdk_error pattern path or the
blocked.md outcome-id parser.
"""

from __future__ import annotations

from quantization_agent.driver.assessment import build_assessment, classify_attempt, derive_status
from quantization_agent.driver.outcomes import OutcomeId
from quantization_agent.driver.result_collector import collect_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap / sdk_error path
# ─────────────────────────────────────────────────────────────────────────────

def test_quark_root_missing_from_sdk_error(build_workspace):
    ws = build_workspace()
    assert classify_attempt(
        ws, sdk_error="RuntimeError: QUARK_ROOT not set; quark_root missing"
    ) == OutcomeId.quark_root_missing


def test_workspace_unwritable_from_sdk_error(build_workspace):
    ws = build_workspace()
    assert classify_attempt(
        ws, sdk_error="PermissionError: [Errno 13] cannot write to workspace /tmp/foo"
    ) == OutcomeId.workspace_unwritable


def test_quark_skill_unavailable_from_sdk_error(build_workspace):
    ws = build_workspace()
    assert classify_attempt(
        ws, sdk_error="Quark skill.md missing at .claude/skills/quark-torch-ptq/SKILL.md"
    ) == OutcomeId.quark_skill_unavailable


def test_sdk_runtime_error_rate_limit(build_workspace):
    ws = build_workspace()
    assert classify_attempt(
        ws, sdk_error="anthropic.RateLimitError: 429 rate limit exceeded"
    ) == OutcomeId.sdk_runtime_error


# ─────────────────────────────────────────────────────────────────────────────
# explicit blocked.md outcome marker
# ─────────────────────────────────────────────────────────────────────────────

def test_blocked_md_outcome_marker_wins(build_workspace):
    # Even with a fully-successful artifact set, an explicit blocked.md
    # marker overrides — SKILL.md knows something disk evidence doesn't.
    ws = build_workspace(blocked_md="outcome_id: upstream_change_required\nreason: needs quark patch")
    assert classify_attempt(ws) == OutcomeId.upstream_change_required


def test_blocked_md_invalid_outcome_id_ignored(build_workspace):
    # Unknown outcome string in blocked.md → fall through to disk scan.
    ws = build_workspace(blocked_md="outcome_id: not_a_real_outcome\n")
    assert classify_attempt(ws) is None  # successful artifacts


# ─────────────────────────────────────────────────────────────────────────────
# phase-aware artifact gaps
# ─────────────────────────────────────────────────────────────────────────────

def test_analysis_missing_under_intake_phase(build_workspace):
    ws = build_workspace(
        include_manifest=False,
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        model_analysis=False,
        quant_plan=False,
        last_phase="intake",
    )
    assert classify_attempt(ws) == OutcomeId.analysis_artifact_invalid_or_missing


def test_plan_missing_under_plan_phase(build_workspace):
    ws = build_workspace(
        include_manifest=False,
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        quant_plan=False,
        last_phase="plan",
    )
    assert classify_attempt(ws) == OutcomeId.plan_artifact_invalid_or_missing


def test_manifest_missing_under_manifest_phase(build_workspace):
    ws = build_workspace(
        include_manifest=False,
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase="manifest",
    )
    assert classify_attempt(ws) == OutcomeId.manifest_artifact_invalid_or_missing


# ─────────────────────────────────────────────────────────────────────────────
# MUST-have on quantized dir
# ─────────────────────────────────────────────────────────────────────────────

def test_must_have_weights_missing(build_workspace):
    ws = build_workspace(include_weights=False, include_validation_report=False)
    assert classify_attempt(ws) == OutcomeId.must_have_weights_missing


def test_must_have_config_missing(build_workspace):
    ws = build_workspace(include_config=False, include_validation_report=False)
    assert classify_attempt(ws) == OutcomeId.must_have_config_missing_or_invalid


def test_must_have_tokenizer_missing(build_workspace):
    ws = build_workspace(include_tokenizer=False, include_validation_report=False)
    assert classify_attempt(ws) == OutcomeId.must_have_tokenizer_missing


# ─────────────────────────────────────────────────────────────────────────────
# validator
# ─────────────────────────────────────────────────────────────────────────────

def test_validation_report_absent(build_workspace):
    ws = build_workspace(include_validation_report=False)
    assert classify_attempt(ws) == OutcomeId.validation_report_absent


def test_md5_fail(build_workspace):
    ws = build_workspace(validation_tag="md5_fail")
    assert classify_attempt(ws) == OutcomeId.must_validate_md5_mismatch


def test_config_fail(build_workspace):
    ws = build_workspace(validation_tag="config_fail")
    assert classify_attempt(ws) == OutcomeId.must_validate_config_mismatch


def test_fuzzy_fail(build_workspace):
    ws = build_workspace(validation_tag="fuzzy_fail")
    assert classify_attempt(ws) == OutcomeId.fuzzy_check_failed


def test_aux_fail(build_workspace):
    ws = build_workspace(validation_tag="aux_fail")
    assert classify_attempt(ws) == OutcomeId.should_have_aux_missing


def test_must_validate_skipped(build_workspace):
    ws = build_workspace(validation_tag="must_validate_skipped")
    assert classify_attempt(ws) == OutcomeId.must_validate_skipped


# ─────────────────────────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_gap_within_threshold_no_narrative_when_zero(build_workspace):
    # Default eval_report has gap=0.0 — clean success, no narrative tag.
    ws = build_workspace()
    assert classify_attempt(ws) is None


def test_eval_gap_within_threshold_narrative_tag(build_workspace):
    ws = build_workspace(
        eval_report={
            "metric_name": "gsm8k", "dataset": "gsm8k", "backend": "vllm",
            "source_score": 0.5, "quantized_score": 0.49, "relative_gap": 0.02,
        }
    )
    assert classify_attempt(ws, acceptable_eval_gap=0.03) == OutcomeId.eval_gap_accepted


def test_eval_gap_exceeded(build_workspace):
    ws = build_workspace(
        eval_report={
            "metric_name": "gsm8k", "dataset": "gsm8k", "backend": "vllm",
            "source_score": 0.5, "quantized_score": 0.45, "relative_gap": 0.10,
        }
    )
    assert classify_attempt(ws, acceptable_eval_gap=0.03) == OutcomeId.eval_gap_exceeded


def test_eval_env_unavailable(build_workspace):
    ws = build_workspace(
        include_eval_report=False,
        eval_skipped_reason="docker not available",
    )
    assert classify_attempt(ws) == OutcomeId.eval_env_unavailable


def test_eval_oom(build_workspace):
    ws = build_workspace(
        include_eval_report=False,
        eval_skipped_reason="quantized engine OOM on MI300X",
    )
    assert classify_attempt(ws) == OutcomeId.eval_oom


def test_quantized_load_failed_from_eval_skipped(build_workspace):
    ws = build_workspace(
        include_eval_report=False,
        eval_skipped_reason="vLLM engine.start failed: quantized model can't load",
    )
    assert classify_attempt(ws) == OutcomeId.quantized_load_failed


def test_eval_report_malformed_treated_as_env_unavailable(build_workspace):
    ws = build_workspace(
        eval_report={"metric_name": "gsm8k"},  # missing required keys
    )
    assert classify_attempt(ws) == OutcomeId.eval_env_unavailable


# ─────────────────────────────────────────────────────────────────────────────
# phase-tagged sdk errors
# ─────────────────────────────────────────────────────────────────────────────

def test_exec_oom_via_sdk_error_under_exec_phase(build_workspace):
    ws = build_workspace(
        include_manifest=False,
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase="exec",
    )
    # Need to add manifest so we get past §3 phase-gap check; instead pass
    # last_phase via sdk path. With no manifest under exec phase, the
    # phase-aware check fires first and returns manifest_artifact_*.
    # So test phase-tagged sdk_error path with manifest present but quantized
    # dir absent + sdk error.
    ws2 = build_workspace(
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase="exec",
    )
    assert classify_attempt(
        ws2, sdk_error="torch.OutOfMemoryError: CUDA out of memory"
    ) == OutcomeId.exec_oom


def test_export_crashed_via_sdk_error(build_workspace):
    ws = build_workspace(
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase="export",
    )
    assert classify_attempt(
        ws, sdk_error="OSError: [Errno 28] No space left on device during save_pretrained"
    ) == OutcomeId.export_crashed


def test_exec_calibration_data_missing_via_sdk_error(build_workspace):
    ws = build_workspace(
        include_quantized_dir=False,
        include_validation_report=False,
        include_eval_report=False,
        last_phase="exec",
    )
    assert classify_attempt(
        ws, sdk_error="RuntimeError: calibration dataloader returned 0 samples"
    ) == OutcomeId.exec_calibration_data_missing


# ─────────────────────────────────────────────────────────────────────────────
# fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_unclassified_failure_when_unknown_sdk_error(build_workspace):
    ws = build_workspace()
    assert classify_attempt(
        ws, sdk_error="Mysterious upstream error nobody has seen before"
    ) == OutcomeId.unclassified_failure


def test_clean_success_returns_none(build_workspace):
    ws = build_workspace()
    assert classify_attempt(ws) is None


# ─────────────────────────────────────────────────────────────────────────────
# Assessment assembly
# ─────────────────────────────────────────────────────────────────────────────

def test_build_assessment_single_clean_attempt(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([None], workspace=ws, artifacts=art)
    assert a.final is None
    assert a.attempts == (None,)
    assert a.recovered is False
    assert a.eval_gap == 0.0


def test_build_assessment_recovered_after_failure(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.exec_oom, None], workspace=ws, artifacts=art)
    assert a.final is None
    assert a.recovered is True


def test_build_assessment_failed_final(build_workspace):
    ws = build_workspace(validation_tag="md5_fail")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_md5_mismatch], workspace=ws, artifacts=art)
    assert a.final == OutcomeId.must_validate_md5_mismatch
    assert a.recovered is False


def test_assessment_to_dict_roundtrip(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.exec_oom, OutcomeId.eval_gap_accepted], workspace=ws, artifacts=art)
    d = a.to_dict()
    assert d["final"] == "eval_gap_accepted"
    assert d["attempts"] == ["exec_oom", "eval_gap_accepted"]
    assert d["recovered"] is True


# ─────────────────────────────────────────────────────────────────────────────
# derive_status
# ─────────────────────────────────────────────────────────────────────────────

def test_derive_status_clean_success(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([None], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "success"


def test_derive_status_eval_gap_accepted_is_success(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.eval_gap_accepted], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "success"


def test_derive_status_md5_mismatch_failed(build_workspace):
    ws = build_workspace(validation_tag="md5_fail")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_md5_mismatch], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "failed"


def test_derive_status_eval_gap_exceeded_partial(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.eval_gap_exceeded], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "partial"


def test_derive_status_must_validate_skipped_strict(build_workspace, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_QUANT_STRICT_VALIDATION", "1")
    ws = build_workspace(validation_tag="must_validate_skipped")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_skipped], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "failed"


def test_derive_status_must_validate_skipped_lenient(build_workspace, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_QUANT_STRICT_VALIDATION", "0")
    ws = build_workspace(validation_tag="must_validate_skipped")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.must_validate_skipped], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "partial"


def test_derive_status_eval_env_unavailable_is_partial(build_workspace):
    ws = build_workspace(include_eval_report=False, eval_skipped_reason="docker missing")
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.eval_env_unavailable], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "partial"


def test_derive_status_unclassified_is_failed(build_workspace):
    ws = build_workspace()
    art = collect_artifacts(ws)
    a = build_assessment([OutcomeId.unclassified_failure], workspace=ws, artifacts=art)
    assert derive_status(a, art) == "failed"
