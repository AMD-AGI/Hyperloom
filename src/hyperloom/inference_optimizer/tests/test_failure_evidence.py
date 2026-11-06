# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the failure evidence pure helpers."""

from __future__ import annotations

from hyperloom.orchestrator.state.failure_evidence import (
    FAILURE_ATTRIBUTION_CANDIDATE,
    FAILURE_ATTRIBUTION_ENVIRONMENT,
    FAILURE_ATTRIBUTION_HARNESS,
    FAILURE_ATTRIBUTION_UNKNOWN,
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    classify_failure_attribution,
    failure_from_variant_outcome,
    make_failure_id,
    render_failure_line,
    tail_excerpt,
)


class TestClassifyFailureAttribution:
    def test_known_invalid_measurement_is_harness_failure(self):
        assert (
            classify_failure_attribution(error_class="magpie_nonzero_invalid_measurement")
            == FAILURE_ATTRIBUTION_HARNESS
        )

    def test_explicit_candidate_class_is_publishable(self):
        assert classify_failure_attribution(error_class="capability_unsupported") == FAILURE_ATTRIBUTION_CANDIDATE

    def test_candidate_specific_cli_error_is_publishable(self):
        assert (
            classify_failure_attribution(
                error_class="server_init_dead",
                error_excerpt="error: unrecognized argument --unsupported-fast-path",
            )
            == FAILURE_ATTRIBUTION_CANDIDATE
        )

    def test_environment_evidence_overrides_generic_failure_class(self):
        assert (
            classify_failure_attribution(
                error_class="server_init_dead",
                error_excerpt="memory reservation failed because a concurrent GPU holder remained",
            )
            == FAILURE_ATTRIBUTION_ENVIRONMENT
        )

    def test_ambiguous_failure_remains_unknown(self):
        assert classify_failure_attribution(error_class="server_init_dead") == FAILURE_ATTRIBUTION_UNKNOWN

    def test_valid_explicit_attribution_wins(self):
        assert (
            classify_failure_attribution(
                error_class="magpie_timeout",
                explicit=FAILURE_ATTRIBUTION_CANDIDATE,
            )
            == FAILURE_ATTRIBUTION_CANDIDATE
        )


class TestTailExcerpt:
    def test_returns_none_for_falsy(self):
        assert tail_excerpt(None) is None
        assert tail_excerpt("") is None

    def test_returns_full_text_when_short(self):
        assert tail_excerpt("hello") == "hello"

    def test_takes_tail_not_head(self):
        text = "HEADER" + "x" * 1300
        result = tail_excerpt(text, limit=1200)
        assert result is not None
        assert result.startswith("x")
        assert "HEADER" not in result

    def test_does_not_truncate_when_within_limit(self):
        text = "a" * 100
        assert tail_excerpt(text, limit=200) == text

    def test_redacts_credential_patterns(self):
        # The bearer-token pattern should be masked.
        result = tail_excerpt("Authorization: Bearer abcdef12345678")
        assert result is not None
        assert "abcdef12345678" not in result


class TestMakeFailureId:
    def test_stable_across_calls(self):
        fid = make_failure_id(task_id="t1", fingerprint="abcdef123456")
        assert fid == make_failure_id(task_id="t1", fingerprint="abcdef123456")

    def test_uses_first_12_chars_of_fingerprint(self):
        fid = make_failure_id(task_id="t1", fingerprint="abcdef123456789")
        assert fid == "fail.t1.abcdef123456"

    def test_falls_back_to_variant_name_slug_when_no_fingerprint(self):
        fid = make_failure_id(task_id="t2", fingerprint="", variant_name="my variant!")
        assert fid.startswith("fail.t2.")
        assert " " not in fid
        assert "!" not in fid

    def test_different_fingerprints_produce_different_ids(self):
        a = make_failure_id(task_id="t1", fingerprint="aaa")
        b = make_failure_id(task_id="t1", fingerprint="bbb")
        assert a != b

    def test_different_tasks_produce_different_ids(self):
        a = make_failure_id(task_id="t1", fingerprint="abc")
        b = make_failure_id(task_id="t2", fingerprint="abc")
        assert a != b


class TestFailureFromVariantOutcome:
    def _make_vo(self, **kw):
        return {
            "variant_name": "fp8_kv",
            "outcome": "FAILED",
            "fingerprint": "abc123",
            "stage": FAILURE_STAGE_WARMUP,
            "error_class": "server_init_dead",
            "error_excerpt": "AssertionError: batch_size == 1",
            "reason": "warmup_failed",
            "failure_attribution": FAILURE_ATTRIBUTION_UNKNOWN,
            "server_log_path": "/runs/v00/server.log",
            "workspace": "/runs/v00",
            "raw_result_path": None,
            "variant": {"extra_server_args": "--kv-cache-dtype fp8", "extra_envs": {}, "note": ""},
            **kw,
        }

    def test_failure_id_matches_make_failure_id(self):
        vo = self._make_vo()
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        assert fe["failure_id"] == make_failure_id(task_id="t1", fingerprint="abc123")

    def test_all_fields_present(self):
        vo = self._make_vo()
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        for key in (
            "failure_id",
            "task_id",
            "round_id",
            "variant_name",
            "fingerprint",
            "stage",
            "outcome",
            "error_class",
            "attribution",
            "error_excerpt",
            "reason",
            "server_log_path",
            "workspace",
            "variant",
        ):
            assert key in fe, f"missing key: {key}"

    def test_attribution_is_the_one_the_producer_classified(self):
        vo = self._make_vo(error_class="capability_unsupported", failure_attribution=FAILURE_ATTRIBUTION_ENVIRONMENT)
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        assert fe["attribution"] == FAILURE_ATTRIBUTION_ENVIRONMENT

    def test_stage_defaults_to_decision_when_absent(self):
        vo = self._make_vo()
        del vo["stage"]
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        assert fe["stage"] == FAILURE_STAGE_DECISION


class TestRenderFailureLine:
    def test_contains_key_fields(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "fp8_kv",
            "stage": FAILURE_STAGE_WARMUP,
            "error_class": "server_init_dead",
            "error_excerpt": "AssertionError: batch == 1",
        }
        line = render_failure_line(fe)
        assert "fail.t1.abc" in line
        assert "fp8_kv" in line
        assert "server_init_dead" in line

    def test_falls_back_to_reason_when_no_excerpt(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "v",
            "stage": FAILURE_STAGE_DECISION,
            "error_class": "",
            "error_excerpt": "",
            "reason": "warmup_failed",
        }
        line = render_failure_line(fe)
        assert "warmup_failed" in line

    def test_keeps_the_tail_of_a_long_body(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "v",
            "stage": FAILURE_STAGE_WARMUP,
            "error_excerpt": "banner " * 200 + "AssertionError: batch == 1",
        }
        line = render_failure_line(fe, excerpt_chars=40)
        assert "AssertionError: batch == 1" in line


class TestFailedStepClasses:
    def test_every_failed_step_has_a_class(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        classes = {step: failed_step_error_class(step) for step in FailedStep}
        assert all(classes.values())
        assert len(set(classes.values())) == len(FailedStep)

    def test_action_owned_steps_are_not_mapped_twice(self):
        from hyperloom.orchestrator.actions.executors import _failed_step
        from hyperloom.orchestrator.state.failure_evidence import FAILED_STEP_ERROR_CLASSES

        assert not set(_failed_step._ACTION_OWNED_STEP_CLASSES) & set(FAILED_STEP_ERROR_CLASSES)

    def test_replaced_failures_keep_their_class(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID
        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
        from hyperloom.orchestrator.actions.executors._spec_decode_pin import SPECULATIVE_DECODING_PINNED
        from hyperloom.orchestrator.actions.stop_attribution import STOPPED_BY_THE_RUN

        assert failed_step_error_class(FailedStep.FLAG_UNKNOWN) == ARGV_INVALID
        assert failed_step_error_class(FailedStep.GOLDEN_OWNED_TOUCHED) == SPECULATIVE_DECODING_PINNED
        assert failed_step_error_class(FailedStep.SERVER_EXITED_BEFORE_READY) == "server_init_dead"
        assert failed_step_error_class(FailedStep.SERVER_STALLED) == "detokenizer_stall"
        assert failed_step_error_class(FailedStep.CLIENT_FAILED) == "magpie_nonzero_invalid_measurement"
        assert failed_step_error_class(FailedStep.CLIENT_OUTPUT_MISSING) == "benchmark_report_missing"
        assert failed_step_error_class(FailedStep.ROUND_TIMEOUT) == "magpie_timeout"
        assert failed_step_error_class(FailedStep.CONFIG_INVALID) == "yaml_build_error"
        for step in (FailedStep.DEADLINE_EXHAUSTED, FailedStep.CANCELLED):
            assert failed_step_error_class(step) in STOPPED_BY_THE_RUN

    def test_new_step_classes_carry_their_attribution(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        environment = {
            FailedStep.DISK_FULL,
            FailedStep.FRAMEWORK_VERSION_TOO_OLD,
            FailedStep.FRAMEWORK_VERSION_UNKNOWN,
            FailedStep.PORT_IN_USE,
            FailedStep.REUSE_NOT_OWNED,
            FailedStep.SERVER_SILENT_BEFORE_READY,
            FailedStep.TEARDOWN_INCOMPLETE,
        }
        harness = {
            FailedStep.BASE_DIGEST_MISMATCH,
            FailedStep.EVAL_CONTEXT_TOO_SHORT,
            FailedStep.SOURCE_FILES_CHANGED,
            FailedStep.STAGING_MISSING,
        }
        for step in environment:
            assert classify_failure_attribution(error_class=failed_step_error_class(step)) == (
                FAILURE_ATTRIBUTION_ENVIRONMENT
            )
        for step in harness:
            assert (
                classify_failure_attribution(error_class=failed_step_error_class(step)) == FAILURE_ATTRIBUTION_HARNESS
            )
