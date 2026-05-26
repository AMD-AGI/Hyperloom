"""Per-action audit-trail unit tests for :class:`SharedState`.

Covers the kernel-parity-for-non-kernel-actions plan:

* ``record_action_attempt`` populates ``last_<action>`` snapshot for each
  of the 6 audit kinds.
* ``<action>_attempts`` rolling history caps at ``_DEFAULT_ATTEMPTS_HISTORY``.
* Failure attempts surface ``error_excerpt`` truncated to 800 chars.
* Non-audit kinds (kernel-owned) silently no-op.
* ``to_prompt_summary`` renders the new lines without raising.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import (
    _DEFAULT_ATTEMPTS_HISTORY,
    SharedState,
)


@pytest.mark.parametrize(
    "action,metric_key,metric_kind",
    [
        ("baseline", "output_throughput", "output_throughput"),
        ("profile",  "output_throughput", "output_throughput"),
        ("sweep",    "output_throughput", "output_throughput"),
        ("explore",  "best_gain_pct",     "gain_pct"),
    ],
)
def test_record_action_attempt_succeeded_populates_last_and_history(
    action, metric_key, metric_kind,
):
    s = SharedState()
    entry = s.record_action_attempt(
        action=action,
        task_id="t-1",
        status="succeeded",
        decision="promoted",
        result={metric_key: 1234.5, "workspace": "/runs/" + action},
        extras={"variant_name": "vA"},
    )
    assert entry is not None
    last = getattr(s, f"last_{action}")
    history = getattr(s, f"{action}_attempts")
    assert last["task_id"] == "t-1"
    assert last["status"] == "succeeded"
    assert last["decision"] == "promoted"
    assert last["key_metric"] == pytest.approx(1234.5)
    assert last["key_metric_kind"] == metric_kind
    assert last["workspace"] == "/runs/" + action
    assert last["extras"] == {"variant_name": "vA"}
    assert history[-1] == last


def test_record_action_attempt_failed_truncates_error_excerpt():
    s = SharedState()
    long_err = "boom! " * 400  # > 800 chars
    s.record_action_attempt(
        action="baseline",
        task_id="t-2",
        status="failed",
        decision="no_promote",
        result={
            "error_class": "no_report",
            "error": long_err,
            "workspace": "/ws/baseline-2",
            "reported_success": False,
        },
    )
    last = s.last_baseline
    assert last["status"] == "failed"
    assert last["decision"] == "no_promote"
    assert last["error_class"] == "no_report"
    assert last["error_excerpt"] is not None
    assert len(last["error_excerpt"]) == 800
    assert last["error_excerpt"].startswith("boom!")
    assert last["reported_success"] is False
    assert last["key_metric"] is None


def test_attempts_history_caps_at_default():
    s = SharedState()
    for i in range(_DEFAULT_ATTEMPTS_HISTORY + 5):
        s.record_action_attempt(
            action="explore",
            task_id=f"t-{i}",
            status="succeeded",
            decision="discarded",
            result={"best_gain_pct": float(i)},
        )
    history = s.explore_attempts
    assert len(history) == _DEFAULT_ATTEMPTS_HISTORY
    # newest survives, oldest dropped
    assert history[-1]["task_id"] == f"t-{_DEFAULT_ATTEMPTS_HISTORY + 4}"
    assert history[0]["task_id"] == f"t-5"


def test_record_action_attempt_skips_non_audit_kinds():
    s = SharedState()
    # kernel-owned kinds are intentionally outside _AUDIT_ACTIONS
    out = s.record_action_attempt(
        action="kernel_opt",
        task_id="t-99",
        status="succeeded",
        decision="promoted",
        result={"output_throughput": 1.0},
    )
    assert out is None
    # No corresponding last_kernel_opt churn from this method (the
    # dedicated record_kernel_opt is the only writer there).
    assert not hasattr(s, "kernel_opt_attempts_list")


def test_to_prompt_summary_renders_new_lines():
    s = SharedState()
    s.record_action_attempt(
        action="baseline",
        task_id="t-1",
        status="succeeded",
        decision="promoted",
        result={"output_throughput": 1761.6, "workspace": "/ws"},
    )
    s.record_action_attempt(
        action="explore",
        task_id="t-2",
        status="failed",
        decision="no_promote",
        result={"error_class": "subprocess_nonzero", "error": "rc=1"},
    )
    txt = s.to_prompt_summary()
    assert "last_baseline=" in txt
    assert "last_profile=" in txt
    assert "last_explore=" in txt
    assert "last_sweep=" in txt
    assert "attempts_history=" in txt
    assert "baseline:1(s1,f0)" in txt
    assert "explore:1(s0,f1)" in txt


def test_save_load_round_trips_new_fields(tmp_path):
    s = SharedState()
    s.record_action_attempt(
        action="profile",
        task_id="p-1",
        status="succeeded",
        decision="promoted",
        result={"output_throughput": 100.0},
        extras={"trace_path": "/tmp/trace.json"},
    )
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert s2.last_profile["task_id"] == "p-1"
    assert s2.profile_attempts[-1]["extras"]["trace_path"] == "/tmp/trace.json"
