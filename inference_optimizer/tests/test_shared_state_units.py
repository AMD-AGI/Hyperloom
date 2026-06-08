# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Focused unit tests for ``SharedState`` helpers / audit trails.

Covers pure-data helpers that only had integration coverage (policy-denial
bookkeeping, kernel-patch identity resolution, prune-family mutators) plus
the per-action attempt audit trail and the global ``last_action_failures``
rolling log.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_LAST_FAILURES,
    SharedState,
)


# ---------------------------------------------------------------------------
# pruned families + policy denial book-keeping
# ---------------------------------------------------------------------------

class TestPolicyDenialAndPruned:
    def test_add_pruned_family_is_idempotent(self):
        s = SharedState()
        assert s.add_pruned_family("kernel_opt") is True
        assert s.is_pruned("kernel_opt") is True
        # Second add returns False.
        assert s.add_pruned_family("kernel_opt") is False
        # prune_family is an alias.
        assert s.prune_family("kernel_opt") is False

    def test_record_policy_denial_tracks_streak(self):
        s = SharedState()
        first = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=1,
        )
        second = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=2,
        )
        assert first == 1 and second == 2
        # Resetting drops the matching streak rows.
        s.reset_policy_denial_streak("kernel_opt")
        again = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=3,
        )
        assert again == 1

    def test_record_policy_denial_keeps_intent_keys_when_present(self):
        s = SharedState()
        s.record_policy_denial(
            action_name="profile",
            rule="missing-prereq",
            hint="needs baseline",
            intent_type="propose_action",
            tick=1,
            intent_payload={"action": "profile", "params": {}},
        )
        assert s.policy_denial_history[-1]["intent_payload_keys"] == ["action", "params"]

    def test_reset_policy_denial_streak_ignores_blank_name(self):
        s = SharedState()
        s.policy_denial_streak["foo:bar"] = 3
        s.reset_policy_denial_streak("")
        assert s.policy_denial_streak == {"foo:bar": 3}

    def test_policy_denial_history_caps_to_50(self):
        s = SharedState()
        for tick in range(60):
            s.record_policy_denial(
                action_name="x", rule="r",
                hint="h", intent_type="propose_action", tick=tick,
            )
        assert len(s.policy_denial_history) == 50

    def test_policy_denial_summary_returns_empty_when_history_empty(self):
        assert SharedState().to_policy_denial_summary() == ""

    def test_policy_denial_summary_includes_recent_rows(self):
        s = SharedState()
        for tick in range(4):
            s.record_policy_denial(
                action_name=f"a{tick}", rule="rule",
                hint=f"hint-{tick}", intent_type="propose_action", tick=tick,
            )
        summary = s.to_policy_denial_summary(top_k=2)
        # Newest two rows surface in the summary.
        assert "a2" in summary
        assert "a3" in summary


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------

class TestApplyChanges:
    def test_empty_changes_returns_empty(self):
        assert SharedState().apply_changes({}, allow_core=True) == {}

    def test_unknown_keys_are_skipped(self):
        s = SharedState()
        applied = s.apply_changes({"unknown_field": 1}, allow_core=True)
        assert applied == {}

    def test_known_field_set(self):
        s = SharedState()
        applied = s.apply_changes({"model_name": "foo"}, allow_core=True)
        assert applied == {"model_name": "foo"}
        assert s.model_name == "foo"


# ---------------------------------------------------------------------------
# kernel-patch identity helpers
# ---------------------------------------------------------------------------

class TestKernelPatchIdentity:
    def test_resolves_explicit_payload(self):
        s = SharedState()
        kid, patch, target, args = s._resolve_kernel_patch_identity({
            "kernel_id": "k1",
            "patch_path": "/tmp/k1.py",
            "target_file": "/srv/k1.py",
            "extra_server_args": " --foo 1 ",
        })
        assert (kid, patch, target, args) == (
            "k1", "/tmp/k1.py", "/srv/k1.py", "--foo 1",
        )

    def test_falls_back_to_last_kernel_opt_patch(self):
        s = SharedState()
        s.last_kernel_opt = {"kernel_id": "k1", "best_artifact_path": "/srv/best.py"}
        kid, patch, target, args = s._resolve_kernel_patch_identity({
            "kernel_id": "k1",
        })
        assert patch == "/srv/best.py"
        assert kid == "k1"
        assert target == ""
        assert args == ""

    def test_kernel_patch_key_empty_when_payload_incomplete(self):
        s = SharedState()
        assert s.kernel_patch_key(None) == ""
        assert s.kernel_patch_key({"kernel_id": "k1"}) == ""

    def test_kernel_patch_key_concatenates_fields(self):
        s = SharedState()
        key = s.kernel_patch_key({
            "kernel_id": "k1",
            "patch_path": "/srv/k1.py",
            "extra_server_args": "--a 1",
        })
        assert key == "k1|/srv/k1.py|--a 1"

    def test_find_rejected_kernel_patch_lookup(self):
        s = SharedState()
        s.rejected_kernel_patches.append({
            "key": "k1|/srv/k1.py|--a 1",
            "reason": "no_e2e_gain",
        })
        hit = s.find_rejected_kernel_patch({
            "kernel_id": "k1",
            "patch_path": "/srv/k1.py",
            "extra_server_args": "--a 1",
        })
        assert hit and hit["reason"] == "no_e2e_gain"

    def test_find_rejected_kernel_patch_missing_returns_none(self):
        assert SharedState().find_rejected_kernel_patch({"kernel_id": "x"}) is None


# ---------------------------------------------------------------------------
# load_or_init / save round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_load_or_init_returns_default_when_missing(self, tmp_path):
        s = SharedState.load_or_init(tmp_path)
        assert s.session_id == ""

    def test_save_then_load_round_trip(self, tmp_path):
        s = SharedState(session_id="abc", baseline_tput=42.0)
        s.save(tmp_path)
        loaded = SharedState.load_or_init(tmp_path)
        assert loaded.session_id == "abc"
        assert loaded.baseline_tput == 42.0

    def test_save_atomic_path_uses_state_json(self, tmp_path):
        s = SharedState()
        s.save(tmp_path)
        assert (tmp_path / "state.json").is_file()

    def test_from_dict_drops_unknown_keys(self):
        raw = {"session_id": "abc", "unknown_field": "ignored"}
        s = SharedState.from_dict(raw)
        assert s.session_id == "abc"
        assert not hasattr(s, "unknown_field")


# ---------------------------------------------------------------------------
# Per-action attempt audit trail (record_action_attempt + <action>_attempts)
# ---------------------------------------------------------------------------

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
    # no_report is not a subprocess failure -> no stderr_tail.
    assert last["stderr_tail"] is None


def test_record_action_attempt_subprocess_failure_captures_stderr_tail():
    """A subprocess_nonzero baseline attempt records stderr_tail into the
    attempts history so the breakdown exporter can surface the raw crash
    (regression: the field was only filled on the last_action_failures
    path, leaving baseline_attempts[].stderr_tail always None)."""
    s = SharedState()
    big_err = "x" * 2000 + "torch.OutOfMemoryError: HIP out of memory"
    s.record_action_attempt(
        action="baseline",
        task_id="t-oom",
        status="failed",
        decision="no_promote",
        result={
            "error_class": "subprocess_nonzero",
            "error": big_err,
            "reported_success": False,
            "stderr_log_path": "/runs/baseline/t-oom/baseline_stderr.log",
        },
    )
    attempt = s.baseline_attempts[-1]
    assert attempt["error_class"] == "subprocess_nonzero"
    assert attempt["stderr_tail"] is not None
    assert len(attempt["stderr_tail"]) == 1000
    assert attempt["stderr_tail"].endswith("HIP out of memory")
    assert attempt["stderr_log_path"] == "/runs/baseline/t-oom/baseline_stderr.log"


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
    assert history[-1]["task_id"] == f"t-{_DEFAULT_ATTEMPTS_HISTORY + 4}"
    assert history[0]["task_id"] == "t-5"


def test_record_action_attempt_skips_non_audit_kinds():
    s = SharedState()
    out = s.record_action_attempt(
        action="kernel_opt",
        task_id="t-99",
        status="succeeded",
        decision="promoted",
        result={"output_throughput": 1.0},
    )
    assert out is None
    assert not hasattr(s, "kernel_opt_attempts_list")


def test_to_prompt_summary_renders_attempt_lines():
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


def test_save_load_round_trips_attempt_fields(tmp_path):
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


# ---------------------------------------------------------------------------
# Global last_action_failures rolling log
# ---------------------------------------------------------------------------

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
    assert entry["stderr_tail"] is None
    assert entry["workspace"] == "/runs/baseline/t-1/benchmark_sglang_xyz"
    assert entry["reported_success"] is False


def test_record_action_failure_truncates_excerpt_and_tails_subprocess():
    s = SharedState()
    long_err = "x" * 2500
    s.record_action_failure(
        action="explore",
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
    assert s.last_action_failures[-1]["task_id"] == (
        f"t-{_DEFAULT_LAST_FAILURES + 2}"
    )
    assert s.last_action_failures[0]["task_id"] == "t-3"


def test_to_prompt_summary_shows_last_three_failures_with_suffix():
    s = SharedState()
    for i in range(5):
        s.record_action_failure(
            action="baseline" if i % 2 == 0 else "explore",
            task_id=f"t-{i}",
            result={"error_class": "no_report", "error": f"err{i}"},
        )
    txt = s.to_prompt_summary()
    assert "last_action_failures=" in txt
    line = next(
        l for l in txt.splitlines() if l.startswith("last_action_failures=")
    )
    assert "[baseline/no_report" in line
    assert "[explore/no_report" in line
    assert "[+2 earlier]" in line


def test_save_load_round_trips_failure_log(tmp_path):
    s = SharedState()
    s.record_action_failure(
        action="sweep",
        task_id="p-1",
        result={"error_class": "subprocess_nonzero", "error": "rc=1\nboom"},
    )
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert len(s2.last_action_failures) == 1
    assert s2.last_action_failures[0]["action"] == "sweep"
    assert s2.last_action_failures[0]["error_class"] == "subprocess_nonzero"
