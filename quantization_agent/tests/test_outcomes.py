"""Tests for the 30-row outcome taxonomy."""

from __future__ import annotations

from quantization_agent.driver.outcomes import (
    ASK,
    ASK_RETRYABLE,
    AUTO_FAIL,
    AUTO_RECOVER,
    OutcomeId,
    SUCCESS_TAGS,
    UNCLASSIFIED_FAILURE,
)


# The 30 rows from Appendix §A, grouped by category. Numbers are the table
# IDs; the enum stores them by name only (the row number is implicit in the
# §A documentation).
_AUTO_RECOVER_NAMES = {
    "intent_parse_failed",                        # #8
    "analysis_artifact_invalid_or_missing",       # #10
    "plan_artifact_invalid_or_missing",           # #11
    "manifest_artifact_invalid_or_missing",       # #12
    "must_have_config_missing_or_invalid",        # #14
    "must_have_tokenizer_missing",                # #15
    "must_validate_config_mismatch",              # #17
    "should_have_aux_missing",                    # #19
    "nice_to_have_skipped",                       # #20
    "eval_env_unavailable",                       # #22
    "validation_report_absent",                   # #25
    "must_validate_skipped",                      # #27
    "eval_oom",                                   # #29
}

_AUTO_FAIL_NAMES = {
    "quark_root_missing",                # #1
    "exec_model_load_failed",            # #4
    "exec_calibration_data_missing",     # #5
    "quark_skill_unavailable",           # #7
    "model_path_unreachable",            # #9
    "validator_self_test_failed",        # #13
    "must_validate_md5_mismatch",        # #18
    "workspace_unwritable",              # #23
    "sdk_runtime_error",                 # #24
    "quantized_load_failed",             # #28
    "upstream_change_required",          # derived from #30
}

_ASK_NAMES = {
    "checkpoint_aborted",                # #2
    "exec_oom",                          # #3
    "export_crashed",                    # #6
    "must_have_weights_missing",         # #16
    "eval_gap_exceeded",                 # #21
    "fuzzy_check_failed",                # #26
}


def test_category_sizes_match_design_appendix_a():
    assert len(AUTO_RECOVER) == 13
    assert len(AUTO_FAIL) == 11  # 10 from #1..#28 + upstream_change_required
    assert len(ASK) == 6


def test_partition_is_disjoint():
    assert AUTO_RECOVER.isdisjoint(AUTO_FAIL)
    assert AUTO_RECOVER.isdisjoint(ASK)
    assert AUTO_FAIL.isdisjoint(ASK)
    assert UNCLASSIFIED_FAILURE not in AUTO_RECOVER
    assert UNCLASSIFIED_FAILURE not in AUTO_FAIL
    assert UNCLASSIFIED_FAILURE not in ASK


def test_category_names_match_design():
    assert {o.value for o in AUTO_RECOVER} == _AUTO_RECOVER_NAMES
    assert {o.value for o in AUTO_FAIL} == _AUTO_FAIL_NAMES
    assert {o.value for o in ASK} == _ASK_NAMES


def test_unclassified_failure_is_sentinel():
    assert UNCLASSIFIED_FAILURE == OutcomeId.unclassified_failure
    assert UNCLASSIFIED_FAILURE.value == "unclassified_failure"


def test_success_tags_include_none_and_eval_accepted():
    assert None in SUCCESS_TAGS
    assert OutcomeId.eval_gap_accepted in SUCCESS_TAGS
    # eval_gap_exceeded is NOT a success tag — it's an Ask-class failure.
    assert OutcomeId.eval_gap_exceeded not in SUCCESS_TAGS


def test_ask_retryable_subset_of_ask():
    assert ASK_RETRYABLE.issubset(ASK)
    # The four CI-auto-retry rows per design §A bottom of table.
    assert {o.value for o in ASK_RETRYABLE} == {
        "exec_oom", "export_crashed", "must_have_weights_missing", "fuzzy_check_failed",
    }
    # checkpoint_aborted and eval_gap_exceeded are Ask but NOT retryable —
    # retrying them won't synthesize missing info / shrink the gap.
    assert OutcomeId.checkpoint_aborted not in ASK_RETRYABLE
    assert OutcomeId.eval_gap_exceeded not in ASK_RETRYABLE


def test_enum_is_string_valued():
    # StrEnum members compare equal to their string values — important for
    # blocked.md parsing and JSON round-trips through Assessment.to_dict().
    assert OutcomeId.exec_oom == "exec_oom"
    assert str(OutcomeId.exec_oom) == "exec_oom"


def test_enum_size_is_32_total():
    # 30 rows from §A + eval_gap_accepted narrative tag + upstream_change_required.
    assert len(list(OutcomeId)) == 32
