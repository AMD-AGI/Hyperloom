"""Classifying an eval-rooted baseline failure, and who may redefine the gate."""

from __future__ import annotations

import hyperloom.orchestrator.actions.executors.baseline as baseline_mod


def _bare_executor() -> baseline_mod.BaselineExecutor:
    """A ctx-less BaselineExecutor for exercising its pure helpers."""
    return object.__new__(baseline_mod.BaselineExecutor)


def test_error_text_carrying_a_run_eval_marker_is_eval_rooted():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({"error": "...\nERROR: run_eval failed with exit code 1\n"}) is True


def test_the_rejected_flag_itself_counts_as_eval_rooted():
    """The flag message is the shape that killed real runs; it must classify."""
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({"error": "Unknown parameter: --concurrent-requests"}) is True


def test_marker_in_a_nonfatal_warning_still_classifies():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({"error": "", "nonfatal_warnings": ["run_eval failed with exit code 1"]}) is True


def test_an_ordinary_benchmark_failure_is_not_eval_rooted():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({"error": "CUDA out of memory", "nonfatal_warnings": ["slow start"]}) is False


def test_empty_result_is_not_eval_rooted_and_does_not_raise():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({}) is False


def test_a_refused_connection_marks_the_eval_failure_as_an_unreachable_server():
    """The shape that killed the real run: the server went away mid-eval, so the client's next request was refused."""
    ex = _bare_executor()
    err = (
        "aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host 0.0.0.0:41099 "
        "ssl:default [Connect call failed ('0.0.0.0', 41099)]\n"
        "ERROR: run_eval failed with exit code 1\n"
    )
    # Still eval-rooted -- the run_eval marker is genuinely there.
    assert ex._is_eval_rooted_failure({"error": err}) is True
    # But the cause is the measurement apparatus, not the model or the framework.
    assert ex._is_server_unreachable_eval_failure({"error": err}) is True


def test_an_eval_that_ran_to_a_verdict_is_not_an_unreachable_server():
    """A reachable server that simply scored badly must keep routing to enablement."""
    ex = _bare_executor()
    err = "accuracy 0.21 below floor 0.5\nERROR: run_eval failed with exit code 1\n"
    assert ex._is_eval_rooted_failure({"error": err}) is True
    assert ex._is_server_unreachable_eval_failure({"error": err}) is False


def test_the_rejected_eval_flag_is_not_an_unreachable_server():
    """``--concurrent-requests`` is an argument-contract break, not a vanished server."""
    ex = _bare_executor()
    assert ex._is_server_unreachable_eval_failure({"error": "Unknown parameter: --concurrent-requests"}) is False


def test_unreachable_server_evidence_is_read_from_warnings_too():
    ex = _bare_executor()
    result = {"error": "", "nonfatal_warnings": ["Cannot connect to host 0.0.0.0:8888"]}
    assert ex._is_server_unreachable_eval_failure(result) is True


def test_empty_result_is_not_an_unreachable_server_and_does_not_raise():
    ex = _bare_executor()
    assert ex._is_server_unreachable_eval_failure({}) is False


def test_only_a_genuine_baseline_may_establish_the_quality_reference():
    assert baseline_mod._should_establish_quality_ref("baseline") is True
    # replay_warm_recipe reuses this executor but is a candidate: letting it redefine the reference would mask its own
    # deviation from the baseline.
    assert baseline_mod._should_establish_quality_ref("replay_warm_recipe") is False
    assert baseline_mod._should_establish_quality_ref("") is False
    assert baseline_mod._should_establish_quality_ref(None) is False


def test_measure_round_config_disables_eval(tmp_path):
    """Round 2 must not re-measure accuracy."""
    import yaml

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"RUN_EVAL": "true", "CONC": 64}}}),
        encoding="utf-8",
    )
    ex = _bare_executor()

    warm = ex._write_lifecycle_config(base, tmp_path / "warmup", cleanup=False, pid_dir=tmp_path, port=41713)
    meas = ex._write_lifecycle_config(
        base, tmp_path / "measure", cleanup=True, pid_dir=tmp_path, port=41713, run_eval=False
    )

    warm_envs = yaml.safe_load(warm.read_text())["benchmark"]["envs"]
    meas_envs = yaml.safe_load(meas.read_text())["benchmark"]["envs"]

    # Round 1 stays the accuracy source; round 2 is throughput-only.
    assert str(warm_envs["RUN_EVAL"]).lower() == "true"
    assert str(meas_envs["RUN_EVAL"]).lower() == "false"
    # Everything else must survive the injection.
    assert meas_envs["CONC"] == 64
