# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the failure evidence pure helpers."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID
from hyperloom.orchestrator.state.failure_evidence import (
    FAILURE_ATTRIBUTION_CANDIDATE,
    FAILURE_ATTRIBUTION_ENVIRONMENT,
    FAILURE_ATTRIBUTION_HARNESS,
    FAILURE_ATTRIBUTION_UNKNOWN,
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    classify_failure_attribution,
    engine_capacity_cause,
    failure_from_variant_outcome,
    make_failure_id,
    render_failure_line,
    tail_excerpt,
)

_MAMBA_CAPACITY_LINE = (
    "(EngineCore pid=7) ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (237). "
    "Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed. "
    "Please lower max_num_seqs to at most 237 or increase gpu_memory_utilization."
)
_KV_MEMORY_LINE = (
    "ValueError: To serve at least one request with the model's max seq len (262144), (24.5 GiB KV cache is "
    "needed, which is larger than the available KV cache memory (9.1 GiB). Try increasing "
    "`gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine."
)
_KV_EMPTY_LINE = (
    "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` "
    "when initializing the engine."
)


def test_engine_capacity_cause_parses_the_numbers():
    assert engine_capacity_cause(_MAMBA_CAPACITY_LINE) == {
        "kind": "mamba_cache_blocks",
        "requested": 256,
        "available": 237,
    }
    assert engine_capacity_cause(_KV_MEMORY_LINE) == {
        "kind": "kv_cache_memory",
        "max_model_len": 262144,
        "needed_gib": 24.5,
        "available_gib": 9.1,
    }
    assert engine_capacity_cause(_KV_EMPTY_LINE) == {"kind": "kv_cache_empty"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "RuntimeError: HIP out of memory. Tried to allocate 10.54 GiB",
        "hipErrorNoBinaryForGpu: no kernel image is available for execution on the device",
        "Transformers does not recognize this architecture",
        "INFO max_num_seqs=256 gpu_memory_utilization=0.9",
    ],
)
def test_engine_capacity_cause_ignores_other_failures(text):
    assert engine_capacity_cause(text) is None


class TestClassifyFailureAttribution:
    def test_known_invalid_measurement_is_harness_failure(self):
        assert (
            classify_failure_attribution(framework_changed=False, error_class="round_nonzero_invalid_measurement")
            == FAILURE_ATTRIBUTION_HARNESS
        )

    def test_explicit_candidate_class_is_publishable(self):
        assert (
            classify_failure_attribution(framework_changed=False, error_class="capability_unsupported")
            == FAILURE_ATTRIBUTION_CANDIDATE
        )

    def test_an_argv_the_server_refused_is_the_candidates(self):
        assert (
            classify_failure_attribution(framework_changed=False, error_class=ARGV_INVALID)
            == FAILURE_ATTRIBUTION_CANDIDATE
        )

    def test_candidate_specific_cli_error_is_publishable(self):
        assert (
            classify_failure_attribution(
                framework_changed=False,
                error_class="server_init_dead",
                error_excerpt="error: unrecognized argument --unsupported-fast-path",
            )
            == FAILURE_ATTRIBUTION_CANDIDATE
        )

    def test_environment_evidence_overrides_generic_failure_class(self):
        assert (
            classify_failure_attribution(
                framework_changed=False,
                error_class="server_init_dead",
                error_excerpt="memory reservation failed because a concurrent GPU holder remained",
            )
            == FAILURE_ATTRIBUTION_ENVIRONMENT
        )

    def test_a_taken_server_port_is_the_environments(self):
        assert (
            classify_failure_attribution(
                framework_changed=False,
                error_class="round_nonzero_invalid_measurement",
                error_excerpt="AgentX server port 8888 is unavailable; existing servers are not reused",
            )
            == FAILURE_ATTRIBUTION_ENVIRONMENT
        )

    @pytest.mark.parametrize(
        "excerpt",
        [
            "AgentX server exited before becoming healthy (-9)",
            "AgentX server exited during its health check (1)",
            "Memory access fault by GPU node-2 (Agent handle: 0x5d0) on address 0x7f00. Reason: Unknown.",
            "ray_actor_error: The actor died unexpectedly before finishing this task.",
        ],
    )
    def test_a_server_that_died_is_not_the_harnesss(self, excerpt):
        assert (
            classify_failure_attribution(
                framework_changed=False, error_class="round_nonzero_invalid_measurement", error_excerpt=excerpt
            )
            == FAILURE_ATTRIBUTION_UNKNOWN
        )

    def test_ambiguous_failure_remains_unknown(self):
        assert (
            classify_failure_attribution(framework_changed=False, error_class="server_init_dead")
            == FAILURE_ATTRIBUTION_UNKNOWN
        )

    def test_engine_capacity_is_not_attributed_to_the_candidate_or_the_host(self):
        assert (
            classify_failure_attribution(framework_changed=False, error_class="engine_capacity")
            == FAILURE_ATTRIBUTION_UNKNOWN
        )

    def test_valid_explicit_attribution_wins(self):
        assert (
            classify_failure_attribution(
                framework_changed=False,
                error_class="round_timeout",
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
        from hyperloom.common.failure_signature import SERVE_FLAG
        from hyperloom.orchestrator.actions.stop_attribution import STOPPED_BY_THE_RUN

        assert failed_step_error_class(FailedStep.FLAG_UNKNOWN) == ARGV_INVALID
        assert failed_step_error_class(FailedStep.GOLDEN_OWNED_TOUCHED) == SERVE_FLAG
        assert failed_step_error_class(FailedStep.SERVER_EXITED_BEFORE_READY) == "server_init_dead"
        assert failed_step_error_class(FailedStep.SERVER_STALLED) == "detokenizer_stall"
        assert failed_step_error_class(FailedStep.CLIENT_OUTPUT_MISSING) == "benchmark_report_missing"
        assert failed_step_error_class(FailedStep.CONFIG_INVALID) == "yaml_build_error"
        assert failed_step_error_class(FailedStep.GPUS_UNAVAILABLE) == "host_gpus_unavailable"
        for step in (FailedStep.DEADLINE_EXHAUSTED, FailedStep.CANCELLED):
            assert failed_step_error_class(step) in STOPPED_BY_THE_RUN

    def test_a_capacity_error_in_the_detail_of_a_server_that_exited_before_ready_is_engine_capacity(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        detail = "ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (237)."

        assert failed_step_error_class(FailedStep.SERVER_EXITED_BEFORE_READY, detail) == "engine_capacity"
        assert (
            failed_step_error_class(FailedStep.SERVER_EXITED_BEFORE_READY, "KeyError: 'qwen3_moe'")
            == "server_init_dead"
        )
        assert failed_step_error_class(FailedStep.SERVER_DIED, detail) == FailedStep.SERVER_DIED.value

    def test_new_step_classes_carry_their_attribution(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        environment = {
            FailedStep.GPUS_UNAVAILABLE,
            FailedStep.FRAMEWORK_VERSION_TOO_OLD,
            FailedStep.FRAMEWORK_VERSION_UNKNOWN,
            FailedStep.PORT_IN_USE,
            FailedStep.REUSE_NOT_OWNED,
            FailedStep.SERVER_SILENT_BEFORE_READY,
            FailedStep.TEARDOWN_INCOMPLETE,
        }
        harness = {
            FailedStep.BASE_DIGEST_MISMATCH,
            FailedStep.CLIENT_FAILED,
            FailedStep.EVAL_CONTEXT_TOO_SHORT,
            FailedStep.ROUND_TIMEOUT,
            FailedStep.SOURCE_FILES_CHANGED,
            FailedStep.STAGING_MISSING,
        }
        for step in environment:
            assert classify_failure_attribution(framework_changed=False, error_class=failed_step_error_class(step)) == (
                FAILURE_ATTRIBUTION_ENVIRONMENT
            )
        for step in harness:
            assert (
                classify_failure_attribution(framework_changed=False, error_class=failed_step_error_class(step))
                == FAILURE_ATTRIBUTION_HARNESS
            )

    def test_a_golden_owned_touch_is_a_candidate_failure(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        error_class = failed_step_error_class(FailedStep.GOLDEN_OWNED_TOUCHED)
        assert (
            classify_failure_attribution(framework_changed=False, error_class=error_class)
            == FAILURE_ATTRIBUTION_CANDIDATE
        )


class TestTheSuitePathClasses:
    def test_a_client_or_timeout_step_is_its_own_class(self):
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        for step in (FailedStep.CLIENT_FAILED, FailedStep.ROUND_TIMEOUT):
            assert failed_step_error_class(step) == step.value

    def test_a_nonzero_round_is_attributed_to_the_harness(self):
        for error_class in ("round_nonzero_invalid_measurement", "round_nonzero_after_valid_measurement"):
            assert (
                classify_failure_attribution(framework_changed=False, error_class=error_class)
                == FAILURE_ATTRIBUTION_HARNESS
            )

    def test_a_failed_client_is_signed_by_its_output_like_a_nonzero_round(self):
        from hyperloom.orchestrator.state.failure_evidence import nonzero_output_signature

        client = nonzero_output_signature("client_failed", "", "Traceback: boom at port 8001")
        nonzero = nonzero_output_signature("round_nonzero_invalid_measurement", "", "Traceback: boom at port 8001")
        assert client.startswith("client_failed|") and client.split("|")[1] == nonzero.split("|")[1]


class TestAnUnknownFrameworkVersion:
    """A round the suite stopped at ``framework_version_unknown`` is the candidate's only when the attempt changed the
    framework and the probe says the framework cannot be imported; an unchanged install that cannot be imported, and
    a probe that says nothing of the framework, are the environment's. The details are the real probe's."""

    @staticmethod
    def _bare_interpreter(tmp_path) -> str:
        """An interpreter that sees the standard library only, so no framework package is importable from it."""
        import shlex
        import sys

        script = tmp_path / "bare-python"
        script.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -S "$@"\n')
        script.chmod(0o755)
        return str(script)

    def _details(self, tmp_path) -> dict[str, str]:
        from inference_suite import framework_version
        from inference_suite.framework_probe import FrameworkProbeError, probe_install, run_probe
        from inference_suite.frameworks import Framework
        from inference_suite.recipe.image_version import recipe_image_facts

        bare = self._bare_interpreter(tmp_path)
        tagged = recipe_image_facts("lmsysorg/sglang:v0.5.0-rocm700", None)
        details = {"unversioned": framework_version.check(None, tagged).detail}  # type: ignore[union-attr]
        probes = {
            "not-installed": lambda: probe_install(bare, Framework.SGLANG, {}, cwd=tmp_path),
            "import-raised": lambda: run_probe(bare, "import sglang_patch", (), {}, cwd=tmp_path),
            "no-interpreter": lambda: run_probe(str(tmp_path / "no-such-python"), "pass", (), {}, cwd=tmp_path),
            "other-error": lambda: run_probe(bare, "1/0", (), {}, cwd=tmp_path),
        }
        for name, probe in probes.items():
            with pytest.raises(FrameworkProbeError) as refused:
                probe()
            details[name] = str(refused.value)
        return details

    @staticmethod
    def _attributed(detail: str, framework_changed: bool) -> str:
        from inference_suite.contracts import FailedStep

        from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class

        return classify_failure_attribution(
            error_class=failed_step_error_class(FailedStep.FRAMEWORK_VERSION_UNKNOWN),
            framework_changed=framework_changed,
            error_excerpt=detail,
        )

    @staticmethod
    def _settled(detail: str, framework_changed: bool):
        from inference_suite.contracts import FailedStep

        from hyperloom.measure.failures import FailureEvidence, StepFailure, classify_test_failure

        evidence = FailureEvidence(
            step_failure=StepFailure(FailedStep.FRAMEWORK_VERSION_UNKNOWN, detail), framework_changed=framework_changed
        )
        return classify_test_failure(returncode=1, evidence=evidence)

    def test_an_unimportable_framework_is_the_candidates_only_when_the_attempt_changed_it(self, tmp_path):
        details = self._details(tmp_path)
        attributed = {
            (name, changed): self._attributed(detail, changed)
            for name, detail in details.items()
            for changed in (False, True)
        }

        environment, candidate = FAILURE_ATTRIBUTION_ENVIRONMENT, FAILURE_ATTRIBUTION_CANDIDATE
        assert attributed == {
            ("unversioned", False): environment,
            ("unversioned", True): environment,
            ("not-installed", False): environment,
            ("not-installed", True): candidate,
            ("import-raised", False): environment,
            ("import-raised", True): candidate,
            ("no-interpreter", False): environment,
            ("no-interpreter", True): environment,
            ("other-error", False): environment,
            ("other-error", True): environment,
        }

    @pytest.mark.parametrize("framework_changed", [False, True], ids=["image", "changed"])
    def test_the_ledger_and_the_toolbox_agree_on_the_same_detail(self, tmp_path, framework_changed):
        from hyperloom.contracts.measurement import FailureClass

        for name, detail in self._details(tmp_path).items():
            settled = self._settled(detail, framework_changed).failure_class is FailureClass.CANDIDATE
            attributed = self._attributed(detail, framework_changed) == FAILURE_ATTRIBUTION_CANDIDATE
            assert attributed == settled, name

    @pytest.mark.parametrize(
        ("detail", "unimportable"),
        [
            pytest.param("the probe ran\n/opt/py/bin/python cannot import sglang", True, id="last-line"),
            pytest.param("/opt/py/bin/python cannot import sglang\n", True, id="trailing-newline"),
            pytest.param("/opt/py/bin/python cannot import sglang\nthen the probe ran past 60 s", False, id="mid-line"),
        ],
    )
    def test_the_probe_answer_counts_only_at_the_end_of_the_detail(self, detail, unimportable):
        """The probe says ``<python> cannot import <package>`` as the whole detail, so a line inside it that ends so is
        no answer of the probe."""
        expected = FAILURE_ATTRIBUTION_CANDIDATE if unimportable else FAILURE_ATTRIBUTION_ENVIRONMENT
        assert self._attributed(detail, framework_changed=True) == expected

    def test_with_no_evidence_it_stays_the_environments(self):
        assert (
            classify_failure_attribution(error_class="framework_version_unknown", framework_changed=True)
            == FAILURE_ATTRIBUTION_ENVIRONMENT
        )
