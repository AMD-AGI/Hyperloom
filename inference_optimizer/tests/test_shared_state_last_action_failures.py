"""Global ``last_action_failures`` rolling-log tests for :class:`SharedState`.

Covers:

* ``record_action_failure`` appends to the rolling log and truncates the
  error excerpt to 800 chars.
* ``stderr_tail`` is populated only for subprocess-style failures (the
  last 1000 chars).
* The list caps at ``_DEFAULT_LAST_FAILURES``.
* ``to_prompt_summary`` renders only the last 3 entries plus the
  "[+N earlier]" suffix.
* Round-trips through ``save``/``load_or_init``.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import (
    _DEFAULT_LAST_FAILURES,
    SharedState,
)


def test_record_action_failure_basic_fields():
    s = SharedState()
    s.record_action_failure(
        action="baseline",
        task_id="t-1",
        result={
            "error_class": "no_report",
            "error": "benchmark_report.json missing under runs/baseline/...",
            "workspace": "/runs/baseline/t-1/benchmark_sglang_xyz",
            "reported_success": False,
            "raw_result_path": None,
        },
    )
    assert len(s.last_action_failures) == 1
    entry = s.last_action_failures[0]
    assert entry["action"] == "baseline"
    assert entry["task_id"] == "t-1"
    assert entry["error_class"] == "no_report"
    assert entry["error_excerpt"].startswith("benchmark_report.json missing")
    # no_report is not a subprocess failure → no stderr_tail.
    assert entry["stderr_tail"] is None
    assert entry["workspace"] == "/runs/baseline/t-1/benchmark_sglang_xyz"
    assert entry["reported_success"] is False


def test_record_action_failure_truncates_excerpt_and_tails_subprocess():
    s = SharedState()
    long_err = "x" * 2500
    s.record_action_failure(
        action="backends",
        task_id="t-2",
        result={
            "error_class": "subprocess_nonzero",
            "error": long_err,
        },
    )
    entry = s.last_action_failures[0]
    assert len(entry["error_excerpt"]) == 800
    assert entry["stderr_tail"] is not None
    assert len(entry["stderr_tail"]) == 1000


def test_record_action_failure_caps_at_default():
    s = SharedState()
    for i in range(_DEFAULT_LAST_FAILURES + 3):
        s.record_action_failure(
            action="baseline",
            task_id=f"t-{i}",
            result={"error_class": "no_report", "error": f"err{i}"},
        )
    assert len(s.last_action_failures) == _DEFAULT_LAST_FAILURES
    # newest survives, oldest dropped
    assert s.last_action_failures[-1]["task_id"] == (
        f"t-{_DEFAULT_LAST_FAILURES + 2}"
    )
    assert s.last_action_failures[0]["task_id"] == "t-3"


def test_to_prompt_summary_shows_last_three_with_suffix():
    s = SharedState()
    for i in range(5):
        s.record_action_failure(
            action="baseline" if i % 2 == 0 else "backends",
            task_id=f"t-{i}",
            result={"error_class": "no_report", "error": f"err{i}"},
        )
    txt = s.to_prompt_summary()
    assert "last_action_failures=" in txt
    # Last 3 rendered: i=2 (baseline), i=3 (backends), i=4 (baseline).
    line = next(
        l for l in txt.splitlines() if l.startswith("last_action_failures=")
    )
    assert "[baseline/no_report" in line
    assert "[backends/no_report" in line
    # 5 - 3 = 2 earlier
    assert "[+2 earlier]" in line


def test_save_load_round_trips_failure_log(tmp_path):
    s = SharedState()
    s.record_action_failure(
        action="params",
        task_id="p-1",
        result={"error_class": "subprocess_nonzero", "error": "rc=1\nboom"},
    )
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert len(s2.last_action_failures) == 1
    assert s2.last_action_failures[0]["action"] == "params"
    assert s2.last_action_failures[0]["error_class"] == "subprocess_nonzero"
