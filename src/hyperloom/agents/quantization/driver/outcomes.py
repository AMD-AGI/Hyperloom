"""Outcome taxonomy for quantization-agent.

Enumerates the 30 failure rows handled by the SKILL.md auto-recover /
auto-fail / ask catalogs, plus
the narrative success tag ``eval_gap_accepted`` and the upstream-mutation
sentinel ``upstream_change_required`` (the resolution #30 takes when the LLM
diagnoses that the fix would require editing files under ``quark_root``).

The IDs here are the **vocabulary** that ``_assessment.classify_attempt``
emits and that ``Assessment.final`` / ``Assessment.attempts`` carry to the
caller. Category sets (``AUTO_RECOVER`` / ``AUTO_FAIL`` / ``ASK``) drive the
retry-loop branching in ``_retry.run_with_retries``.
"""

from __future__ import annotations

try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # Python 3.10 fallback
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            """Return the enum's string value (mirrors 3.11 ``StrEnum``).

            Returns:
                The enum member's string value.
            """
            return str(self.value)


class OutcomeId(StrEnum):
    # Bootstrap / environment
    quark_root_missing = "quark_root_missing"  # #1   Auto-fail
    quark_skill_unavailable = "quark_skill_unavailable"  # #7   Auto-fail
    intent_parse_failed = "intent_parse_failed"  # #8   Auto-recover
    workspace_unwritable = "workspace_unwritable"  # #23  Auto-fail

    # Intake artifacts
    model_path_unreachable = "model_path_unreachable"  # #9   Auto-fail
    analysis_artifact_invalid_or_missing = "analysis_artifact_invalid_or_missing"  # #10  Auto-recover

    # Planning artifacts
    checkpoint_aborted = "checkpoint_aborted"  # #2   Ask
    plan_artifact_invalid_or_missing = "plan_artifact_invalid_or_missing"  # #11  Auto-recover

    # Manifest artifacts
    manifest_artifact_invalid_or_missing = "manifest_artifact_invalid_or_missing"  # #12  Auto-recover

    # Execution outcomes
    exec_oom = "exec_oom"  # #3   Ask
    exec_model_load_failed = "exec_model_load_failed"  # #4   Auto-fail
    exec_calibration_data_missing = "exec_calibration_data_missing"  # #5   Auto-fail

    # Export outcomes
    export_crashed = "export_crashed"  # #6   Ask

    # Validation outcomes
    validator_self_test_failed = "validator_self_test_failed"  # #13  Auto-fail
    must_have_config_missing_or_invalid = "must_have_config_missing_or_invalid"  # #14  Auto-recover
    must_have_tokenizer_missing = "must_have_tokenizer_missing"  # #15  Auto-recover
    must_have_weights_missing = "must_have_weights_missing"  # #16  Ask
    must_validate_config_mismatch = "must_validate_config_mismatch"  # #17  Auto-recover
    must_validate_md5_mismatch = "must_validate_md5_mismatch"  # #18  Auto-fail
    should_have_aux_missing = "should_have_aux_missing"  # #19  Auto-recover
    nice_to_have_skipped = "nice_to_have_skipped"  # #20  Auto-recover
    validation_report_absent = "validation_report_absent"  # #25  Auto-recover
    fuzzy_check_failed = "fuzzy_check_failed"  # #26  Ask
    must_validate_skipped = "must_validate_skipped"  # #27  Auto-recover

    # Eval outcomes
    eval_gap_exceeded = "eval_gap_exceeded"  # #21  Ask
    eval_env_unavailable = "eval_env_unavailable"  # #22  Auto-recover
    quantized_load_failed = "quantized_load_failed"  # #28  Auto-fail
    eval_oom = "eval_oom"  # #29  Auto-recover

    # SDK / catch-all
    sdk_runtime_error = "sdk_runtime_error"  # #24  Auto-fail
    unclassified_failure = "unclassified_failure"  # #30  Auto-recover* (runtime-classified)

    # Narrative / derived tags (not in the 30-row table):
    #   eval_gap_accepted — success with gap within threshold
    #   upstream_change_required — #30 diagnosed as needing a quark_root edit
    eval_gap_accepted = "eval_gap_accepted"
    upstream_change_required = "upstream_change_required"


# Partition of the outcome IDs into retry classes.
AUTO_RECOVER: frozenset[OutcomeId] = frozenset(
    {
        OutcomeId.intent_parse_failed,  # #8
        OutcomeId.analysis_artifact_invalid_or_missing,  # #10
        OutcomeId.plan_artifact_invalid_or_missing,  # #11
        OutcomeId.manifest_artifact_invalid_or_missing,  # #12
        OutcomeId.must_have_config_missing_or_invalid,  # #14
        OutcomeId.must_have_tokenizer_missing,  # #15
        OutcomeId.must_validate_config_mismatch,  # #17
        OutcomeId.should_have_aux_missing,  # #19
        OutcomeId.nice_to_have_skipped,  # #20
        OutcomeId.eval_env_unavailable,  # #22
        OutcomeId.validation_report_absent,  # #25
        OutcomeId.must_validate_skipped,  # #27
        OutcomeId.eval_oom,  # #29
    }
)

AUTO_FAIL: frozenset[OutcomeId] = frozenset(
    {
        OutcomeId.quark_root_missing,  # #1
        OutcomeId.exec_model_load_failed,  # #4
        OutcomeId.exec_calibration_data_missing,  # #5
        OutcomeId.quark_skill_unavailable,  # #7
        OutcomeId.model_path_unreachable,  # #9
        OutcomeId.validator_self_test_failed,  # #13
        OutcomeId.must_validate_md5_mismatch,  # #18
        OutcomeId.workspace_unwritable,  # #23
        OutcomeId.sdk_runtime_error,  # #24
        OutcomeId.quantized_load_failed,  # #28
        OutcomeId.upstream_change_required,  # derived from #30
    }
)

ASK: frozenset[OutcomeId] = frozenset(
    {
        OutcomeId.checkpoint_aborted,  # #2
        OutcomeId.exec_oom,  # #3
        OutcomeId.export_crashed,  # #6
        OutcomeId.must_have_weights_missing,  # #16
        OutcomeId.eval_gap_exceeded,  # #21
        OutcomeId.fuzzy_check_failed,  # #26
    }
)

UNCLASSIFIED_FAILURE: OutcomeId = OutcomeId.unclassified_failure  # #30 sentinel

# Outcomes that count as ``recovered=True`` when a multi-attempt trail ends on
# them. ``None`` is clean success; ``eval_gap_accepted`` marks a gap accepted
# after earlier attempts exceeded it.
SUCCESS_TAGS: frozenset[OutcomeId | None] = frozenset({None, OutcomeId.eval_gap_accepted})


# Ask-class IDs that participate in Python-driven retry (the only outcomes that
# increment ``requantize_attempts.txt``).
ASK_RETRYABLE: frozenset[OutcomeId] = frozenset(
    {
        OutcomeId.exec_oom,  # #3
        OutcomeId.export_crashed,  # #6
        OutcomeId.must_have_weights_missing,  # #16
        OutcomeId.fuzzy_check_failed,  # #26
    }
)


# Auto-recover outcomes that still demote to "failed" (not "partial") when the
# MUST-have artifact is missing on the final attempt (model unusable).
MUST_HAVE_RECOVERS_THAT_FAIL_WITHOUT_ARTIFACT: frozenset[OutcomeId] = frozenset(
    {
        OutcomeId.must_have_config_missing_or_invalid,
        OutcomeId.must_have_tokenizer_missing,
        OutcomeId.must_validate_config_mismatch,
    }
)


__all__ = [
    "OutcomeId",
    "AUTO_RECOVER",
    "AUTO_FAIL",
    "ASK",
    "ASK_RETRYABLE",
    "MUST_HAVE_RECOVERS_THAT_FAIL_WITHOUT_ARTIFACT",
    "UNCLASSIFIED_FAILURE",
    "SUCCESS_TAGS",
]
