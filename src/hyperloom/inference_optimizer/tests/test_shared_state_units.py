# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``SharedState`` helpers / audit trails."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.state.shared_state import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_LAST_FAILURES,
    SharedState,
    inject_stack_base_params,
    resolve_grading_anchor_tput,
)


class TestResolveGradingAnchorTput:
    def test_prefers_current_best_over_baseline(self):
        s = SharedState()
        s.baseline_tput = 2195.86
        s.current_best = {"action": "replay_warm_recipe", "tput": 2358.80}
        assert resolve_grading_anchor_tput(s) == 2358.80

    def test_falls_back_to_baseline_before_any_validated_layer(self):
        s = SharedState()
        s.baseline_tput = 2195.86
        assert resolve_grading_anchor_tput(s) == 2195.86

    def test_reads_output_throughput_when_tput_absent(self):
        s = SharedState()
        s.baseline_tput = 800.0
        s.current_best = {"action": "explore", "output_throughput": 900.0}
        assert resolve_grading_anchor_tput(s) == 900.0

    @pytest.mark.parametrize("state", [None, object()])
    def test_tolerates_missing_state(self, state):
        assert resolve_grading_anchor_tput(state) == 0.0

    def test_zero_when_nothing_established(self):
        assert resolve_grading_anchor_tput(SharedState()) == 0.0


class TestInjectStackBaseParams:
    @staticmethod
    def _state():
        s = SharedState()
        s.baseline_tput = 800.0
        s.current_best = {
            "tput": 1000.0,
            "extra_server_args": "--live-layer 1",
            "extra_envs": {"LIVE_ENV": "1"},
            "remove_args": "--dropped",
            "args_mode": "replace",
        }
        return s

    def test_seeds_the_anchor_together_with_its_config(self):
        params: dict = {}
        inject_stack_base_params(params, self._state(), anchor=True)
        assert params == {
            "base_tput": 1000.0,
            "base_extra_args": "--live-layer 1",
            "base_extra_envs": {"LIVE_ENV": "1"},
            "base_remove_args": ["--dropped"],
            "base_args_mode": "replace",
        }

    def test_omits_the_anchor_unless_asked(self):
        params: dict = {}
        inject_stack_base_params(params, self._state())
        assert "base_tput" not in params
        assert params["base_extra_args"] == "--live-layer 1"

    def test_keeps_a_caller_supplied_value(self):
        params = {"base_extra_args": "--operator-pinned"}
        inject_stack_base_params(params, self._state(), anchor=True)
        assert params["base_extra_args"] == "--operator-pinned"

    def test_overwrite_replaces_a_superseded_layer(self):
        params = {"base_extra_args": "--stale-layer 1", "base_tput": 800.0}
        state = self._state()
        state.current_best = {"tput": 1000.0, "extra_server_args": "", "extra_envs": {}}
        inject_stack_base_params(params, state, anchor=True, overwrite=True)
        assert params["base_tput"] == 1000.0
        assert params["base_extra_args"] == ""
        assert params["base_extra_envs"] == {}

    def test_skips_fields_current_best_does_not_carry(self):
        params: dict = {}
        state = SharedState()
        state.baseline_tput = 800.0
        state.current_best = {"action": "baseline", "tput": 800.0}
        inject_stack_base_params(params, state, anchor=True)
        assert params == {"base_tput": 800.0}

    def test_falls_back_to_the_baseline_anchor_with_no_stack(self):
        params: dict = {}
        state = SharedState()
        state.baseline_tput = 800.0
        inject_stack_base_params(params, state, anchor=True)
        assert params == {"base_tput": 800.0}

    @pytest.mark.parametrize("state", [None, object()])
    def test_tolerates_missing_state(self, state):
        params: dict = {}
        inject_stack_base_params(params, state, anchor=True)
        assert params == {}


class TestGridSessionDeadline:
    def test_returns_none_when_budget_unbounded(self):
        s = SharedState()
        s.max_minutes = None
        assert s.grid_session_deadline_sec() is None

    def test_future_deadline_when_ample_budget(self, monkeypatch):
        import hyperloom.orchestrator.state.shared_state as ss_mod

        s = SharedState()
        s.max_minutes = 60.0
        monkeypatch.setattr(type(s), "remaining_minutes", lambda self, **_: 10.0)
        monkeypatch.setattr(ss_mod.time, "monotonic", lambda: 1000.0)
        # 10 min remaining, 72s closing reserve -> now + (600 - 72).
        assert s.grid_session_deadline_sec() == pytest.approx(1000.0 + 528.0)

    def test_deadline_is_now_when_under_reserve(self, monkeypatch):
        import hyperloom.orchestrator.state.shared_state as ss_mod

        s = SharedState()
        s.max_minutes = 60.0
        monkeypatch.setattr(type(s), "remaining_minutes", lambda self, **_: 1.0)
        monkeypatch.setattr(ss_mod.time, "monotonic", lambda: 500.0)
        # 60s remaining < 72s closing reserve -> deadline == now (already exhausted).
        assert s.grid_session_deadline_sec() == pytest.approx(500.0)


class TestDeadlineUnix:
    """The persisted unix deadline is the one remaining-time check."""

    def test_an_unbounded_session_does_not_stamp_a_deadline(self):
        state = SharedState(session_id="s")
        assert state.stamp_deadline_unix() == 0.0
        assert state.deadline_unix == 0.0
        assert state.remaining_minutes() is None

    def test_the_first_stamp_is_start_plus_the_budget(self):
        from hyperloom.common.coerce import to_unix

        state = SharedState(session_id="s", max_minutes=60)
        stamped = state.stamp_deadline_unix()
        start = to_unix(state.start_ts)
        assert stamped == pytest.approx(start + 3600.0, abs=1.0)
        assert state.remaining_minutes() == pytest.approx(60.0, abs=0.1)

    def test_a_second_stamp_does_not_reissue_the_budget(self):
        state = SharedState(session_id="s", max_minutes=60)
        first = state.stamp_deadline_unix(now_unix=1_000.0)
        state.start_ts = "2099-01-01T00:00:00+00:00"
        assert state.stamp_deadline_unix(now_unix=9_000.0) == first
        assert state.deadline_unix == first

    def test_remaining_minutes_reads_the_stamp_not_elapsed(self):
        from datetime import datetime, timezone

        state = SharedState(session_id="s", max_minutes=60)
        state.deadline_unix = 2_000.0
        now = datetime.fromtimestamp(1_400.0, tz=timezone.utc)
        assert state.remaining_minutes(now=now) == pytest.approx(10.0)

    def test_a_spent_stamp_reads_as_zero_not_negative(self):
        from datetime import datetime, timezone

        state = SharedState(session_id="s", max_minutes=60)
        state.deadline_unix = 100.0
        now = datetime.fromtimestamp(500.0, tz=timezone.utc)
        assert state.remaining_minutes(now=now) == 0.0

    def test_a_stamp_survives_a_max_minutes_truncated_to_zero(self):
        from datetime import datetime, timezone

        state = SharedState(session_id="s")
        now_unix = 1_000.0
        state.start_ts = datetime.fromtimestamp(now_unix, tz=timezone.utc).isoformat()
        stamped = state.stamp_deadline_unix(
            budget_minutes=0.0001,
            now_unix=now_unix,
        )
        now = datetime.fromtimestamp(now_unix, tz=timezone.utc)
        assert stamped == pytest.approx(now_unix + 0.006)
        assert state.remaining_minutes(now=now) == pytest.approx(0.0001)
        state.max_minutes = 0
        now = datetime.fromtimestamp(stamped - 6.0, tz=timezone.utc)
        assert state.remaining_minutes(now=now) == pytest.approx(0.1)

    def test_remaining_minutes_reads_a_stamp_when_max_minutes_is_zero(self):
        from datetime import datetime, timezone

        state = SharedState(session_id="s", max_minutes=0)
        state.deadline_unix = 2_000.0
        now = datetime.fromtimestamp(1_400.0, tz=timezone.utc)
        assert state.remaining_minutes(now=now) == pytest.approx(10.0)

    def test_teardown_timings_accumulate_and_keep_a_total(self):
        state = SharedState(session_id="s")
        state.record_teardown_timing("final_json", 1.25)
        state.record_teardown_timing("langfuse", 0.5)
        assert state.teardown_timings_sec["final_json"] == 1.25
        assert state.teardown_timings_sec["langfuse"] == 0.5
        assert state.teardown_timings_sec["total"] == pytest.approx(1.75)

    def test_timed_teardown_step_records_the_elapsed_wall(self, monkeypatch):
        from hyperloom.orchestrator.state import shared_state as ss_mod
        from hyperloom.orchestrator.state.shared_state import timed_teardown_step

        clock = {"t": 0.0}
        monkeypatch.setattr(ss_mod.time, "monotonic", lambda: clock["t"])
        state = SharedState(session_id="s")
        with timed_teardown_step(state, "final_md"):
            clock["t"] = 2.5
        assert state.teardown_timings_sec["final_md"] == 2.5


class TestProfileWorkloadContext:
    def test_normalizes_state_and_payload_overrides(self):
        state = SharedState(
            framework="VLLM",
            precision="FP8",
            model_path="/models/old",
            tp=1,
            conc=64,
            isl=1024,
            osl=512,
            max_model_len=4096,
        )

        assert state.profile_workload_context(
            {
                "framework": " SGLang ",
                "model_path": "/models/new",
                "tp": "2",
                "conc": "128",
            }
        ) == {
            "framework": "sglang",
            "precision": "fp8",
            "model_path": "/models/new",
            "tp": 2,
            "conc": 128,
            "isl": 1024,
            "osl": 512,
            "max_model_len": 4096,
            # The context also carries the runtime identity. This payload sets
            # none of it, so these normalize to their empty forms -- asserted
            # explicitly so a silent change to the identity shape fails here.
            "server_args": "",
            "extra_envs": {},
            "remove_args": [],
            "unset_envs": [],
            "args_mode": "append",
        }

    def test_last_profile_workload_round_trips(self, tmp_path):
        state = SharedState()
        state.last_profile_workload = {"framework": "vllm", "conc": 64}
        state.save(tmp_path)

        assert SharedState.load_or_init(tmp_path).last_profile_workload == {
            "framework": "vllm",
            "conc": 64,
        }

    def test_profile_workload_context_tracks_serving_config(self):
        state = SharedState(
            framework="vllm",
            precision="fp8",
            current_best={
                "extra_server_args": " --attention-backend AITER ",
                "extra_envs": {"B": 2, "A": 1},
            },
        )

        assert state.profile_workload_context()["serving_config"] == {
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": {"A": "1", "B": "2"},
        }


class TestPolicyDenialAndPruned:
    def test_add_pruned_family_is_idempotent(self):
        s = SharedState()
        assert s.add_pruned_family("kernel_opt") is True
        assert s.is_pruned("kernel_opt") is True
        # Second add returns False.
        assert s.add_pruned_family("kernel_opt") is False

    def test_record_policy_denial_tracks_streak(self):
        s = SharedState()
        first = s.record_policy_denial(
            action_name="kernel_opt",
            rule="cooldown",
            hint="hint",
            intent_type="propose_action",
            tick=1,
        )
        second = s.record_policy_denial(
            action_name="kernel_opt",
            rule="cooldown",
            hint="hint",
            intent_type="propose_action",
            tick=2,
        )
        assert first == 1 and second == 2
        # Resetting drops the matching streak rows.
        s.reset_policy_denial_streak("kernel_opt")
        again = s.record_policy_denial(
            action_name="kernel_opt",
            rule="cooldown",
            hint="hint",
            intent_type="propose_action",
            tick=3,
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
                action_name="x",
                rule="r",
                hint="h",
                intent_type="propose_action",
                tick=tick,
            )
        assert len(s.policy_denial_history) == 50

    def test_policy_denial_summary_returns_empty_when_history_empty(self):
        assert SharedState().to_policy_denial_summary() == ""

    def test_policy_denial_summary_includes_recent_rows(self):
        s = SharedState()
        for tick in range(4):
            s.record_policy_denial(
                action_name=f"a{tick}",
                rule="rule",
                hint=f"hint-{tick}",
                intent_type="propose_action",
                tick=tick,
            )
        summary = s.to_policy_denial_summary(top_k=2)
        assert "a2" in summary
        assert "a3" in summary


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

    def test_core_field_dropped_when_allow_core_false(self):
        # A non-privileged (allow_core=False) changes dict must not write a core field.
        s = SharedState()
        before = s.cumulative_gain_validated  # cumulative_gain_validated is a core field
        applied = s.apply_changes(
            {"current_action": "baseline", "cumulative_gain_validated": 999.0},
            allow_core=False,
        )
        assert applied == {"current_action": "baseline"}
        assert s.current_action == "baseline"
        assert s.cumulative_gain_validated == before  # core write dropped

    def test_a_stop_time_cannot_be_written_apart_from_its_reason(self):
        # stop_reason is a core field, so a changes dict that carries both must
        # not land the timestamp half either: the pair is what the export reads
        # as "the session ended then, for this reason".
        s = SharedState()
        s.set_stop_reason("time_exhausted")
        pinned = s.stop_ts
        applied = s.apply_changes(
            {"stop_reason": "target_reached", "stop_ts": "2026-01-01T00:01:00+00:00"},
            allow_core=False,
        )
        assert applied == {}
        assert s.stop_reason == "time_exhausted"
        assert s.stop_ts == pinned

    def test_core_field_written_when_allow_core_true(self):
        s = SharedState()
        applied = s.apply_changes({"cumulative_gain_validated": 999.0}, allow_core=True)
        assert applied == {"cumulative_gain_validated": 999.0}
        assert s.cumulative_gain_validated == 999.0


class TestKernelPatchIdentity:
    def test_resolves_explicit_payload(self):
        s = SharedState()
        kid, patch, target, args = s._resolve_kernel_patch_identity(
            {
                "kernel_id": "k1",
                "patch_path": "/tmp/k1.py",
                "target_file": "/srv/k1.py",
                "extra_server_args": " --foo 1 ",
            }
        )
        assert (kid, patch, target, args) == (
            "k1",
            "/tmp/k1.py",
            "/srv/k1.py",
            "--foo 1",
        )

    def test_falls_back_to_last_kernel_opt_patch(self):
        s = SharedState()
        s.last_kernel_opt = {"kernel_id": "k1", "best_artifact_path": "/srv/best.py"}
        kid, patch, target, args = s._resolve_kernel_patch_identity(
            {
                "kernel_id": "k1",
            }
        )
        assert patch == "/srv/best.py"
        assert kid == "k1"
        assert target == ""
        assert args == ""

    def test_find_rejected_kernel_patch_lookup(self):
        s = SharedState()
        s.rejected_kernel_patches.append(
            {
                "key": "k1|/srv/k1.py|--a 1",
                "reason": "no_e2e_gain",
            }
        )
        hit = s.find_rejected_kernel_patch(
            {
                "kernel_id": "k1",
                "patch_path": "/srv/k1.py",
                "extra_server_args": "--a 1",
            }
        )
        assert hit and hit["reason"] == "no_e2e_gain"

    def test_find_rejected_kernel_patch_missing_returns_none(self):
        assert SharedState().find_rejected_kernel_patch({"kernel_id": "x"}) is None


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


@pytest.mark.parametrize(
    "action,metric_key,metric_kind",
    [
        ("baseline", "output_throughput", "output_throughput"),
        ("profile", "output_throughput", "output_throughput"),
        ("sweep", "output_throughput", "output_throughput"),
        ("explore", "best_gain_pct", "gain_pct"),
    ],
)
def test_record_action_attempt_succeeded_populates_last_and_history(
    action,
    metric_key,
    metric_kind,
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
    long_err = "boom! " * 400
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
    assert len(last["error_excerpt"]) == 1200
    assert last["error_excerpt"].startswith("boom!")
    assert last["reported_success"] is False
    assert last["key_metric"] is None
    # stderr_tail is now captured for EVERY failure carrying an error blob
    # (no error_class whitelist), so orchestration/RCA see the actionable tail.
    assert last["stderr_tail"] is not None
    assert len(last["stderr_tail"]) == 1000
    assert "boom!" in last["stderr_tail"]


def test_record_action_attempt_subprocess_failure_captures_stderr_tail():
    """A subprocess_nonzero baseline attempt records stderr_tail into the
    attempts history so the breakdown exporter can surface the raw crash."""
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


def test_record_action_attempt_redacts_secrets_from_persisted_errors():
    s = SharedState()
    # Named for what it is -- a value planted to be found missing -- rather
    # than for what it imitates. A test-local holding a credential-shaped
    # literal reads to the clear-text-logging analysis as a live credential,
    # and it then reports every diagnostic path this value could reach as a
    # leak of it.
    planted = "ak-sensitive-value"
    s.record_action_attempt(
        action="baseline",
        task_id="t-secret",
        status="failed",
        decision="no_promote",
        result={
            "error_class": "subprocess_nonzero",
            "error": f"OPENAI_API_KEY={planted} Authorization: Bearer {planted}",
        },
    )

    attempt = s.baseline_attempts[-1]
    assert planted not in attempt["error_excerpt"]
    assert planted not in attempt["stderr_tail"]
    assert "[REDACTED]" in attempt["error_excerpt"]
    assert "[REDACTED]" in attempt["stderr_tail"]


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


def test_profile_workload_context_normalizes_path_and_runtime_controls(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)
    state = SharedState(
        framework="vllm",
        precision="fp8",
        model_path=str(model),
        tp=1,
        conc=64,
        isl=1024,
        osl=1024,
        max_model_len=4096,
        current_best={
            "extra_server_args": "  --foo   bar ",
            "extra_envs": {"B": 2, "A": 1},
            "remove_args": ["--old-b", "--old-a"],
            "unset_envs": ["OLD_B", "OLD_A"],
            "args_mode": "replace",
        },
    )

    recorded = state.profile_workload_context(
        {
            "model_path": str(alias),
            "base_extra_args": "--foo bar",
            "base_extra_envs": {"A": "1", "B": "2"},
            "base_remove_args": ["--old-a", "--old-b"],
            "base_unset_envs": ["OLD_A", "OLD_B"],
            "base_args_mode": "REPLACE",
        }
    )

    assert recorded == state.current_profile_workload_context()
    assert recorded["model_path"] == str(model.resolve())
    assert recorded["server_args"] == "--foo bar"
    assert recorded["extra_envs"] == {"A": "1", "B": "2"}
    assert recorded["remove_args"] == ["--old-a", "--old-b"]
    assert recorded["unset_envs"] == ["OLD_A", "OLD_B"]
    assert recorded["args_mode"] == "replace"


def test_profile_workload_context_prefers_effective_profile_params():
    state = SharedState(
        current_best={
            "extra_server_args": "--enable-torch-compile --attention-backend TRITON",
        }
    )

    context = state.profile_workload_context(
        {
            "base_extra_args": "--enable-torch-compile --attention-backend TRITON",
            "extra_server_args": "--attention-backend TRITON",
            "base_extra_envs": {"BACKEND": "base"},
            "extra_envs": {"BACKEND": "effective"},
            "base_remove_args": ["--base-remove"],
            "remove_args": ["--effective-remove"],
            "base_unset_envs": ["BASE_ENV"],
            "unset_envs": ["EFFECTIVE_ENV"],
            "base_args_mode": "append",
            "args_mode": "replace",
        }
    )

    assert context["server_args"] == "--attention-backend TRITON"
    assert context["extra_envs"] == {"BACKEND": "effective"}
    assert context["remove_args"] == ["--effective-remove"]
    assert context["unset_envs"] == ["EFFECTIVE_ENV"]
    assert context["args_mode"] == "replace"
    assert state.current_profile_workload_context()["server_args"] == ("--attention-backend TRITON")


def test_baseline_current_best_reuses_recorded_profile_runtime():
    """A bare baseline current_best inherits the runtime its profile measured."""
    state = SharedState(
        framework="vllm",
        precision="fp8",
        model_path="/models/qwen",
        current_best={"action": "baseline", "tput": 100.0},
    )
    state.record_profile_workload(
        {
            "base_extra_args": "--attention-backend AITER",
            "base_extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
        }
    )

    assert state.last_profile_workload_action == "baseline"
    assert state.current_profile_workload_context() == state.last_profile_workload


def test_profile_trace_matches_workload_with_server_args():
    """Regression (H1): profile_trace_matches_workload() with no explicit target
    must compare the recorded profile against the *current-best* runtime identity,
    not the bare profile_workload_context() (which reports server_args="" and
    skips the current_best backfill). Otherwise any workload carrying server
    args/extra envs reads its own fresh profile as stale on every run and the
    forge shape resolvers discard a perfectly good TraceLens profile."""
    state = SharedState(
        framework="vllm",
        precision="fp8",
        model_path="/models/qwen",
        last_profile_status="succeeded",
        current_best={
            "action": "gemm_tuning",
            "tput": 120.0,
            "extra_server_args": "--tp 8 --attention-backend TRITON",
            "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
        },
    )
    state.record_profile_workload(
        {
            "base_extra_args": "--tp 8 --attention-backend TRITON",
            "base_extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
        }
    )

    # The recorded profile matches the active current-best runtime, so freshness
    # with no explicit target must hold.
    assert state.current_profile_workload_context() == state.last_profile_workload
    assert state.profile_trace_matches_workload() is True
    # The bare context really does disagree (server_args=""), which is exactly why
    # defaulting to it would falsely flag this fresh profile as stale.
    assert state.last_profile_workload != state.profile_workload_context()


def test_baseline_current_best_ignores_tuned_arm_profile_runtime():
    """A runtime measured off a tuned arm must not read as still active."""
    state = SharedState(
        framework="vllm",
        precision="fp8",
        model_path="/models/qwen",
        current_best={
            "action": "gemm_tuning",
            "tput": 120.0,
            "extra_envs": {"AITER_CONFIG_FMOE": "1"},
        },
    )
    state.record_profile_workload(
        {
            "base_extra_args": "--attention-backend AITER",
            "base_extra_envs": {"AITER_CONFIG_FMOE": "1"},
        }
    )
    assert state.last_profile_workload_action == "gemm_tuning"

    # Reverting to a bare baseline drops the tuned runtime, so the fingerprint
    # must no longer claim the tuned arm's args are in effect.
    state.current_best = {"action": "baseline", "tput": 100.0}
    context = state.current_profile_workload_context()

    assert context != state.last_profile_workload
    assert context["server_args"] == ""
    assert context["extra_envs"] == {}


def test_legacy_session_without_recorded_arm_still_reuses_profile_runtime():
    """Sessions predating the recorded arm keep reusing instead of reprofiling."""
    state = SharedState(
        framework="vllm",
        precision="fp8",
        model_path="/models/qwen",
        current_best={"action": "baseline", "tput": 100.0},
    )
    state.last_profile_workload = state.profile_workload_context({"base_extra_args": "--attention-backend AITER"})

    assert state.last_profile_workload_action == ""
    assert state.current_profile_workload_context() == state.last_profile_workload


def test_prelude_roofline_records_baseline_arm_regardless_of_current_best():
    """A PRELUDE profile measures the baseline arm even if current_best is tuned."""
    state = SharedState(
        framework="vllm",
        current_best={"action": "gemm_tuning", "tput": 120.0},
    )
    state.record_profile_workload({"base_extra_args": "--tp 8"}, arm="baseline")

    assert state.last_profile_workload_action == "baseline"


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
    # stderr_tail is captured for all failures now (no error_class whitelist).
    assert entry["stderr_tail"] is not None
    assert entry["stderr_tail"].startswith("benchmark_report.json missing")
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
    assert len(entry["error_excerpt"]) == 1200
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
    assert s.last_action_failures[-1]["task_id"] == (f"t-{_DEFAULT_LAST_FAILURES + 2}")
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
    # All 5 fit within the 10-entry window; each renders on its own lines.
    failures_block = txt[txt.index("last_action_failures=") :]
    assert "[baseline/no_report" in failures_block
    assert "[explore/no_report" in failures_block
    # No suffix when all entries fit.
    assert "[+2 earlier]" not in failures_block


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


def test_record_action_failure_captures_stderr_tail_for_kv_cache_oom():
    # kv_cache_oom is a subprocess-style failure, so its stderr tail is captured.
    s = SharedState()
    entry = s.record_action_failure(
        action="explore",
        task_id="t-kvoom",
        result={
            "error_class": "kv_cache_oom",
            "error": "Loaded weights leave no GPU memory for the KV cache",
        },
    )
    assert entry["stderr_tail"] is not None
    assert "no GPU memory for the KV cache" in entry["stderr_tail"]


def test_record_action_attempt_kv_cache_oom_captures_stderr_tail():
    # record_action_attempt captures kv_cache_oom stderr_tail into <action>_attempts.
    s = SharedState()
    s.record_action_attempt(
        action="baseline",
        task_id="t-kvoom",
        status="failed",
        decision="no_promote",
        result={
            "error_class": "kv_cache_oom",
            "error": "Loaded weights leave no GPU memory for the KV cache",
            "reported_success": False,
        },
    )
    attempt = s.baseline_attempts[-1]
    assert attempt["error_class"] == "kv_cache_oom"
    assert attempt["stderr_tail"] is not None
    assert "no GPU memory for the KV cache" in attempt["stderr_tail"]


def test_record_action_failure_stores_variant_name():
    s = SharedState()
    entry = s.record_action_failure(
        action="explore",
        task_id="t-v",
        result={
            "error_class": "server_init_dead",
            "error": "mla_gluon requires batch_size=1",
            "variant_name": "fp8_kv",
        },
    )
    assert entry["variant_name"] == "fp8_kv"


def test_record_action_failure_caps_at_new_default():
    """_DEFAULT_LAST_FAILURES is now 30."""
    s = SharedState()
    for i in range(35):
        s.record_action_failure(
            action="explore",
            task_id=f"t-{i}",
            result={"error_class": "server_init_dead", "error": f"crash {i}"},
        )
    assert len(s.last_action_failures) == 30
    assert s.last_action_failures[-1]["task_id"] == "t-34"


# ---- failure evidence ledger ----


def test_record_failure_evidence_stores_packet():
    s = SharedState()
    fe = {"failure_id": "fail.t1.abc", "task_id": "t1", "variant_name": "v"}
    s.record_failure_evidence(fe)
    assert len(s.failures) == 1
    assert s.failures[0]["failure_id"] == "fail.t1.abc"


def test_record_failure_evidence_is_idempotent_last_wins():
    s = SharedState()
    fe1 = {"failure_id": "fail.t1.abc", "task_id": "t1", "error_class": "a"}
    fe2 = {"failure_id": "fail.t1.abc", "task_id": "t1", "error_class": "b"}
    s.record_failure_evidence(fe1)
    s.record_failure_evidence(fe2)
    assert len(s.failures) == 1
    assert s.failures[0]["error_class"] == "b"


def test_record_failure_evidence_caps_at_default():
    from hyperloom.orchestrator.state.shared_state import _DEFAULT_LAST_FAILURES

    s = SharedState()
    for i in range(_DEFAULT_LAST_FAILURES + 5):
        s.record_failure_evidence({"failure_id": f"fail.t.{i:04d}", "task_id": "t"})
    assert len(s.failures) == _DEFAULT_LAST_FAILURES


def test_record_failure_evidence_does_not_raise_without_session_dir():
    s = SharedState()
    assert not hasattr(s, "_session_dir") or getattr(s, "_session_dir", None) is None
    fe = {"failure_id": "fail.t1.abc", "task_id": "t1"}
    s.record_failure_evidence(fe)
    assert s.failures[0]["failure_id"] == "fail.t1.abc"


def test_find_failure_returns_matching_entry():
    s = SharedState()
    s.record_failure_evidence({"failure_id": "fail.t1.abc", "task_id": "t1"})
    s.record_failure_evidence({"failure_id": "fail.t1.def", "task_id": "t1"})
    result = s.find_failure("fail.t1.abc")
    assert result is not None
    assert result["failure_id"] == "fail.t1.abc"


def test_find_failure_returns_none_when_missing():
    s = SharedState()
    assert s.find_failure("fail.t1.xyz") is None


def test_failures_for_task_returns_correct_entries():
    s = SharedState()
    s.record_failure_evidence({"failure_id": "fail.t1.a", "task_id": "t1"})
    s.record_failure_evidence({"failure_id": "fail.t2.a", "task_id": "t2"})
    s.record_failure_evidence({"failure_id": "fail.t1.b", "task_id": "t1"})
    result = s.failures_for_task("t1")
    assert len(result) == 2
    ids = {e["failure_id"] for e in result}
    assert ids == {"fail.t1.a", "fail.t1.b"}


def test_record_failure_evidence_writes_json(tmp_path):
    s = SharedState()
    s._session_dir = tmp_path
    fe = {"failure_id": "fail.t1.abc123456789", "task_id": "t1"}
    s.record_failure_evidence(fe)
    from hyperloom.inference_optimizer.session.session_paths import failure_evidence_path
    import json

    path = failure_evidence_path(tmp_path, "fail.t1.abc123456789")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["failure_id"] == "fail.t1.abc123456789"


def test_common_result_fields_includes_failure_id():
    s = SharedState()
    fields = s._common_result_fields({"failure_id": "fail.t1.abc"})
    assert fields["failure_id"] == "fail.t1.abc"


def test_common_result_fields_failure_id_none_when_absent():
    s = SharedState()
    fields = s._common_result_fields({})
    assert "failure_id" in fields
    assert fields["failure_id"] is None
