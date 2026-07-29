"""Classifying an eval-rooted baseline failure, and who may redefine the gate.

When InferenceX's ``run_eval`` aborts the benchmark, the executor has to tell
that apart from an ordinary benchmark failure: the retry-with-``RUN_EVAL=false``
path is only correct for tasks that are *not* establishing the accuracy
reference. Getting this wrong either burns a second full benchmark for nothing
or silently redefines the quality reference from a throughput-only run.
"""

from __future__ import annotations

import hyperloom.orchestrator.actions.executors.baseline as baseline_mod


def _bare_executor() -> baseline_mod.BaselineExecutor:
    """A ctx-less BaselineExecutor for exercising its pure helpers.

    These tests only touch classification/config helpers that never read
    instance state, so we build the instance with ``object.__new__`` and skip
    the real (ctx-hungry) ``__init__`` -- rather than subclassing with a no-op
    constructor, which trips CodeQL's missing-super-init check.
    """
    return object.__new__(baseline_mod.BaselineExecutor)


def test_error_text_carrying_a_run_eval_marker_is_eval_rooted():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure(
        {"error": "...\nERROR: run_eval failed with exit code 1\n"}
    ) is True


def test_the_rejected_flag_itself_counts_as_eval_rooted():
    """The flag message is the shape that killed real runs; it must classify."""
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure(
        {"error": "Unknown parameter: --concurrent-requests"}
    ) is True


def test_marker_in_a_nonfatal_warning_still_classifies():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure(
        {"error": "", "nonfatal_warnings": ["run_eval failed with exit code 1"]}
    ) is True


def test_an_ordinary_benchmark_failure_is_not_eval_rooted():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure(
        {"error": "CUDA out of memory", "nonfatal_warnings": ["slow start"]}
    ) is False


def test_empty_result_is_not_eval_rooted_and_does_not_raise():
    ex = _bare_executor()
    assert ex._is_eval_rooted_failure({}) is False


def test_only_a_genuine_baseline_may_establish_the_quality_reference():
    assert baseline_mod._should_establish_quality_ref("baseline") is True
    # replay_warm_recipe reuses this executor but is a candidate: letting it
    # redefine the reference would mask its own deviation from the baseline.
    assert baseline_mod._should_establish_quality_ref("replay_warm_recipe") is False
    assert baseline_mod._should_establish_quality_ref("") is False
    assert baseline_mod._should_establish_quality_ref(None) is False


def test_measure_round_config_disables_eval(tmp_path):
    """Round 2 must not re-measure accuracy.

    Accuracy does not depend on cold-vs-hot timing, so a second eval only costs
    minutes and doubles the window in which a server death can take it down --
    which is how a real run was lost.
    """
    import yaml

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"RUN_EVAL": "true", "CONC": 64}}}),
        encoding="utf-8",
    )
    ex = _bare_executor()

    warm = ex._write_lifecycle_config(
        base, tmp_path / "warmup", cleanup=False, pid_dir=tmp_path, port=41713
    )
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
