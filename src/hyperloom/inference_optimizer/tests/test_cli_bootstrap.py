# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Focused coverage for CLI bootstrap helpers."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.coerce import to_unix
from hyperloom.inference_optimizer.cli import bootstrap as cb
from hyperloom.orchestrator.state.shared_state import SharedState


def _args(**overrides):
    base = dict(
        model="/models/moonshotai-Kimi-K2.6",
        model_class="KimiK2ForCausalLM",
        gpu_type="mi300x",
        precision="int4",
        framework="vllm",
        framework_version="",
        tp="8",
        ep="0",
        conc=64,
        isl=8192,
        osl=1024,
        max_model_len=13312,
        no_kernel=False,
        auto_kernel_opt=True,
        target_summary="",
        target_gain=60.0,
        target_tput=None,
        max_hours=30,
        research_lane_capacity=999,
        gpu_specialist_capacity="bad",
        plateau_explore_keep_gain=1.5,
        plateau_explore_empty_streak=2,
        plateau_explore_lookback=4,
        plateau_kernel_revert_streak=3,
        plateau_kernel_keep_gain=2.5,
        plateau_kernel_lookback=5,
        explore_overtime_kill_ratio="bad",
        explore_variant_timeout_sec="bad",
        explore_variant_timeout_safety_margin="bad",
        enable_roofline=False,
        no_framework_agent=True,
        research_scout=False,
        research_scout_interval=0,
        target_advisory=False,
        recipe_sediment=False,
        enable_conc_sweep=True,
        conc_sweep_concs="1, bad, 4,,8",
        conc_sweep_total_budget_sec=120,
        conc_sweep_timeout_sec=30,
        reference_script="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _state_with_stamp(*, elapsed_h: float, remaining_h: float) -> SharedState:
    now = time.time()
    start = datetime.fromtimestamp(now - elapsed_h * 3600.0, tz=timezone.utc).isoformat()
    state = SharedState(session_id="s", start_ts=start, max_minutes=int((elapsed_h + remaining_h) * 60))
    state.deadline_unix = now + remaining_h * 3600.0
    return state


def test_seed_shared_state_populates_geak_and_cli_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge, geak")
    monkeypatch.setenv("CLAW_SESSION_ID", "claw-1")
    monkeypatch.setenv("SANDBOX_USER_ID", "user-1")
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("FRAMEWORK_VERSION", "0.21.0")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setattr(
        cb,
        "_load_model_config_tags",
        lambda _p: {
            "architectures": ["KimiK2ForCausalLM"],
            "model_type": "kimi_k25",
        },
    )
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {"layers": 61})
    monkeypatch.setattr(
        cb,
        "_resolve_reference_recipe",
        lambda _args: (
            "--block-size 64",
            {"ENV_A": "1"},
            "Kimi-K2.6",
            "/recipes/kimi.sh",
        ),
    )

    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    state = cb._seed_shared_state(tmp_path, _args(), session_id="session-1")

    assert state.session_id == "session-1"
    assert state.claw_session_id == "claw-1"
    assert state.sandbox_user_id == "user-1"
    assert state.model_name == "moonshotai-Kimi-K2.6"
    assert state.model_arch == {"layers": 61}
    assert state.model_architectures == ["KimiK2ForCausalLM"]
    assert state.model_type == "kimi_k25"
    assert state.framework == "vllm"
    assert state.framework_version == "0.21.0"
    assert state.tp == 8
    assert state.conc == 64
    assert state.isl == 8192
    assert state.osl == 1024
    assert state.max_model_len == 13312
    assert state.kernel_optimizer == "geak"
    assert state.research_lane_capacity == 16
    assert state.gpu_specialist_capacity == 8
    assert state.plateau_overrides["explore_keep_gain_pct"] == 1.5
    assert state.plateau_overrides["kernel_keep_gain_pct"] == 2.5
    assert state.explore_overtime_kill_ratio == 2.0
    assert state.explore_variant_timeout_sec_override == 0
    assert state.explore_variant_timeout_safety_margin == 0.5
    # One switch for the one phase.
    assert state.framework_agent_phase_enabled is False
    assert state.conc_sweep_concs == [1, 4, 8]
    assert state.conc_sweep_total_budget_sec == 120
    assert state.conc_sweep_variant_timeout_sec == 30
    assert state.reference_server_args == "--block-size 64"
    assert json.loads((tmp_path / "state.json").read_text())["session_id"] == "session-1"


def test_seed_shared_state_records_custom_workload_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FRAMEWORK", "custom")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "/scripts")
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "/fw")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: ("", {}, "", ""))
    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 1)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 1)

    state = cb._seed_shared_state(tmp_path, _args(framework="custom"), session_id="s-custom")
    assert state.bypass_scripts_dir == "/scripts"
    assert state.framework_repo_path == "/fw"
    assert state.benchmark_backend == "bypass"


def _neutralize_seed_io(monkeypatch):
    """Stub the model/recipe reads so a seed can be asserted on one field."""
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: ("", {}, "", ""))
    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 1)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 1)


def test_seed_records_the_launch_verdict_for_the_partition_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The verdict carries provenance the published env cannot express.

    ``published_shape()`` reads back mode, count, CU and streams, but nothing
    that says the CU count was probed from the device rather than derived from
    the board table. Re-reading the env therefore reported a fresh launch's
    probed count as a table guess, which is the one thing the section is for.
    """
    _neutralize_seed_io(monkeypatch)
    monkeypatch.setattr(cb, "published_shape", lambda: {"mode": "CPX", "cu_per_partition": 32})

    verdict = {"mode": "CPX", "partitions": 8, "cu_per_partition": 32, "cu_probed": True}
    state = cb._seed_shared_state(tmp_path, _args(), session_id="s-shape", compute_partition=verdict)

    assert state.compute_partition == verdict


def test_seed_falls_back_to_the_published_shape_when_handed_no_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _neutralize_seed_io(monkeypatch)
    monkeypatch.setattr(cb, "published_shape", lambda: {"mode": "DPX"})

    state = cb._seed_shared_state(tmp_path, _args(), session_id="s-fallback")
    assert state.compute_partition == {"mode": "DPX"}


def test_seed_shared_state_exact_forge_records_native_kernel_optimizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: ("", {}, "", ""))

    state = cb._seed_shared_state(tmp_path, _args(), session_id="session-forge")

    assert state.kernel_optimizer == "native"


def test_seed_shared_state_loads_model_arch_from_session_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """model_arch.json is per-session, not shared across USER_DATA_PATH."""
    workspace_root = tmp_path / "workspace"
    session_dir = workspace_root / "Model-A" / "20260721T031500Z"
    session_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})

    def _load_arch(root: Path, model_name: str, launched_model: str = "") -> dict:
        captured["root"] = root
        captured["model_name"] = model_name
        captured["launched_model"] = launched_model
        return {"source": "session-local"}

    monkeypatch.setattr(cb, "_load_model_arch", _load_arch)
    monkeypatch.setattr(
        cb,
        "_resolve_reference_recipe",
        lambda _args: ("", {}, "", ""),
    )

    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 1)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 1)

    state = cb._seed_shared_state(session_dir, _args(model="/models/Model-A"), session_id="session-arch")

    assert captured["root"] == session_dir
    assert state.model_arch == {"source": "session-local"}


def test_seed_shared_state_preserves_quantized_model_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression: after the quantize prelude pins ``args.model_display_name``,
    ``SharedState.model_name`` must use it rather than the collapsed
    ``<...>/quantized`` path basename."""
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(
        cb,
        "_resolve_reference_recipe",
        lambda _args: ("", {}, "", ""),
    )

    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    quant_dir = tmp_path / "quantization" / "google-gemma-4-26B-A4B-it" / "quantized"
    quant_dir.mkdir(parents=True)
    args = _args(
        model=str(quant_dir),
        model_display_name="google-gemma-4-26B-A4B-it-quantized",
    )
    state = cb._seed_shared_state(tmp_path, args, session_id="s-q")

    assert state.model_name == "google-gemma-4-26B-A4B-it-quantized"
    assert state.model_name != "quantized"


def test_seed_shared_state_falls_back_to_path_basename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Without a pinned display name (the common, non-quantized path) the model
    name is still the plain model-path basename."""
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(
        cb,
        "_resolve_reference_recipe",
        lambda _args: ("", {}, "", ""),
    )

    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    state = cb._seed_shared_state(
        tmp_path,
        _args(model="/models/Qwen3-32B"),
        session_id="s-plain",
    )
    assert state.model_name == "Qwen3-32B"


def test_manifest_preserves_quantized_model_identity(tmp_path: Path) -> None:
    """Regression: ``manifest.json`` ``model_name`` must honor the pinned
    display name from the quantize prelude, not the collapsed path basename."""
    from hyperloom.inference_optimizer.session import manifest as m

    quant_dir = tmp_path / "quantization" / "google-gemma-4-26B-A4B-it" / "quantized"
    quant_dir.mkdir(parents=True)
    args = _args(
        model=str(quant_dir),
        model_display_name="google-gemma-4-26B-A4B-it-quantized",
    )
    built = m.build_manifest(tmp_path, args=args, session_id="s-q")
    assert built["model_name"] == "google-gemma-4-26B-A4B-it-quantized"
    assert built["model_name"] != "quantized"
    assert built["model_path"].endswith("/quantized")


def test_resolve_model_display_name_helper() -> None:
    """Unit cover for the identity resolver: pinned override wins, else basename."""
    plain = SimpleNamespace(model="/models/Qwen3-32B")
    assert cb.resolve_model_display_name(plain) == "Qwen3-32B"

    pinned = SimpleNamespace(
        model="/tmp/quantization/x/quantized",
        model_display_name="x-quantized",
    )
    assert cb.resolve_model_display_name(pinned) == "x-quantized"

    empty_override = SimpleNamespace(model="/models/Foo", model_display_name="")
    assert cb.resolve_model_display_name(empty_override) == "Foo"


def test_auto_kernel_opt_defaults_to_dispatching() -> None:
    """Neither spelling passed: the entry dispatches on its own."""
    unset = SimpleNamespace(auto_kernel_opt=None, continue_kernel_after_gemm=None)
    assert cb.resolve_auto_kernel_opt(unset) is True
    # A caller that predates both dests must not crash on the lookup.
    assert cb.resolve_auto_kernel_opt(SimpleNamespace()) is True


def test_auto_kernel_opt_honours_the_current_flag() -> None:
    """``--no-auto-kernel-opt`` alone opts out."""
    opted_out = SimpleNamespace(auto_kernel_opt=False, continue_kernel_after_gemm=None)
    assert cb.resolve_auto_kernel_opt(opted_out) is False


def test_deprecated_gemm_spelling_still_opts_out_and_warns() -> None:
    """The old flag keeps working, because dropping it would silently opt back in."""
    legacy = SimpleNamespace(auto_kernel_opt=None, continue_kernel_after_gemm=False)
    with pytest.warns(DeprecationWarning, match="--auto-kernel-opt"):
        assert cb.resolve_auto_kernel_opt(legacy) is False


def test_current_flag_outranks_the_deprecated_spelling() -> None:
    """Both passed and disagreeing: the current flag decides."""
    both = SimpleNamespace(auto_kernel_opt=True, continue_kernel_after_gemm=False)
    with pytest.warns(DeprecationWarning):
        assert cb.resolve_auto_kernel_opt(both) is True


def test_seed_shared_state_records_the_auto_kernel_opt_optout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The resolved switch lands on the field the KERNEL phase reads."""
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {})
    monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: ("", {}, "", ""))

    state = cb._seed_shared_state(
        tmp_path,
        _args(auto_kernel_opt=False),
        session_id="session-auto-kernel-opt",
    )

    assert state.auto_kernel_opt_enabled is False


def test_target_summary_and_conc_sweep_parser(caplog) -> None:
    assert ">= 12.5%" in cb._default_target_summary(
        _args(model="/m/foo", target_gain=12.5, target_tput=None, max_hours=4)
    )
    assert "123.0 tok/s/GPU" in cb._default_target_summary(
        _args(model="/m/foo", target_gain=None, target_tput=123.0, max_hours=4)
    )
    assert "no target" in cb._default_target_summary(
        _args(model="/m/foo", target_gain=None, target_tput=None, max_hours=4)
    )
    assert cb._parse_conc_sweep_concs(_args(conc_sweep_concs="")) == [
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
    ]
    assert cb._parse_conc_sweep_concs(_args(conc_sweep_concs="bad,")) == [
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
    ]
    assert "ignoring non-integer CONC token" in caplog.text


def test_read_failure_summary_and_final_summary_output(tmp_path: Path, capsys) -> None:
    assert cb._read_failure_summary(tmp_path) is None
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final.json").write_text(
        json.dumps(
            {
                "failure_summary": {
                    "root_cause_type": "config",
                    "root_cause": "bad flag",
                    "server_log": "/tmp/server.log",
                },
            }
        ),
        encoding="utf-8",
    )
    assert cb._read_failure_summary(tmp_path)["root_cause"] == "bad flag"

    state = SharedState(
        session_id="s",
        model_name="m",
        baseline_tput=10.0,
        cumulative_gain_validated=1.0,
        cumulative_gain_validated_ts="2026-01-01T00:00:00Z",
        cumulative_gain_validated_stack_len=0,
        current_best={"action": "x"},
        pruned_families=["a"],
        crash_count=2,
    )
    state.optimization_stack = [{"action": "geak_e2e"}]

    cb._print_final_summary(state, "baseline_failed", tmp_path)

    out = capsys.readouterr().out
    assert "root_cause" in out
    assert "bad flag" in out
    assert "stack changed since validation" in out

    cb._print_final_summary(
        SharedState(session_id="s2", model_name="m2", baseline_tput=0.0),
        "done",
        None,
    )
    assert "never validated" in capsys.readouterr().out


def test_resolve_reference_recipe_branches_and_final_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    import pytest
    from hyperloom.inference_optimizer import reference_script

    assert cb._resolve_reference_recipe(_args(reference_script="")) == ("", {}, "", "")

    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: SimpleNamespace(
            server_args="--tp 8",
            envs={"A": "1"},
            model="kimi",
        ),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="usable.sh")) == (
        "--tp 8",
        {"A": "1"},
        "kimi",
        "usable.sh",
    )

    # Unreadable path raises SystemExit(2) instead of falling back to discovery.
    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: (_ for _ in ()).throw(OSError("not found")),
    )
    with pytest.raises(SystemExit) as exc_info:
        cb._resolve_reference_recipe(_args(reference_script="empty.sh"))
    assert exc_info.value.code == 2

    cb._print_final_summary(
        SharedState(session_id="s2", model_name="m2", baseline_tput=0.0),
        "done",
        None,
    )
    assert "never validated" in capsys.readouterr().out


def test_snapshot_skeleton_and_session_dir_helpers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cb._snapshot_system_prompts(tmp_path, prompts={"orch": "hello", "robustness": ""})
    assert (tmp_path / "agents" / "orch" / "system_prompt.snapshot.md").read_text(
        encoding="utf-8",
    ) == "hello"
    assert (tmp_path / "agents" / "robustness" / "system_prompt.snapshot.md").read_text(
        encoding="utf-8",
    ) == "(empty)"

    for sub in cb._SESSION_SKELETON[:2]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    cb._print_session_skeleton(tmp_path)
    out = capsys.readouterr().out
    assert "Session layout under" in out
    assert "manifest.json" in out

    monkeypatch.setenv("HYPERLOOM_SESSION_DIR", str(tmp_path))
    assert cb._resolve_session_dir_for_summary(None) == tmp_path
    monkeypatch.setenv("HYPERLOOM_SESSION_DIR", str(tmp_path / "missing"))
    assert cb._resolve_session_dir_for_summary(None) is None


def test_a_clean_stop_resume_records_where_the_new_leg_began() -> None:
    """start_ts stays the budget anchor, so the resume timestamp is the only leg boundary."""
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00", crash_count=1)
    state.deadline_unix = 1_700_000_000.0
    state.teardown_timings_sec = {"total": 1.5}

    cb._begin_resume_leg(state, reanchor_budget=False)

    assert state.start_ts == "2026-08-01T00:00:00+00:00"
    assert state.resumed_ts > state.start_ts
    assert state.crash_count == 1
    assert state.deadline_unix == 1_700_000_000.0
    assert state.teardown_timings_sec == {"total": 1.5}


def test_clean_stop_resume_notes_follow_the_stamp_not_a_larger_cli_budget() -> None:
    """Raising --max-hours on a clean-stop resume must not be reported as time left."""
    state = _state_with_stamp(elapsed_h=2.0, remaining_h=1.0)
    lines = cb._clean_stop_resume_budget_lines(state, max_hours=8.0)
    text = "\n".join(lines)
    match = re.search(r"budget: ([0-9.]+)h elapsed, ([0-9.]+)h left on the persisted stamp", text)
    assert match is not None
    assert abs(float(match.group(1)) - 2.0) < 0.05
    assert abs(float(match.group(2)) - 1.0) < 0.05
    assert "this invocation's --max-hours 8.00 does not extend or shrink that stamp" in text
    assert "raise --max-hours or start a fresh session" not in text


def test_clean_stop_resume_notes_do_not_tell_the_operator_to_raise_max_hours() -> None:
    state = _state_with_stamp(elapsed_h=3.5, remaining_h=-0.5)
    lines = cb._clean_stop_resume_budget_lines(state, max_hours=8.0)
    text = "\n".join(lines)
    match = re.search(r"budget: ([0-9.]+)h elapsed, ([0-9.]+)h left on the persisted stamp", text)
    assert match is not None
    assert abs(float(match.group(1)) - 3.5) < 0.05
    assert abs(float(match.group(2)) - 0.0) < 0.05
    assert "WARNING: the stamped deadline is already spent" in text
    assert "start a fresh session" in text
    assert "does not extend the stamp" in text
    assert "raise --max-hours or start a fresh session" not in text


def test_clean_stop_resume_notes_omit_the_cli_mismatch_when_hours_match_the_stamp() -> None:
    state = _state_with_stamp(elapsed_h=1.0, remaining_h=2.0)
    lines = cb._clean_stop_resume_budget_lines(state, max_hours=3.0)
    text = "\n".join(lines)
    assert "does not extend or shrink that stamp" not in text
    assert "WARNING:" not in text


def test_a_resume_after_a_stop_re_anchors_the_budget_on_the_new_leg() -> None:
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00", crash_count=4)
    state.set_stop_reason("time_exhausted")
    state.closing_phase = True
    state.deadline_unix = 1_700_000_000.0
    state.teardown_timings_sec = {"close_backends": 0.2, "total": 0.2}

    cb._begin_resume_leg(state, reanchor_budget=True)

    assert state.start_ts == state.resumed_ts
    assert state.stop_reason == ""
    assert state.stop_ts == ""
    assert state.closing_phase is False
    assert state.crash_count == 0
    assert state.deadline_unix == 0.0
    assert state.teardown_timings_sec == {}
    stamped = state.stamp_deadline_unix(budget_minutes=60)
    start = to_unix(state.start_ts)
    assert abs(stamped - (start + 3600.0)) < 2.0


def test_a_resume_banks_what_the_stopped_leg_spent_in_its_phase() -> None:
    """A phase segment is only durable once banked, and stopping never banks it."""
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00")
    state.phase = "PRELUDE"
    state.phase_started_ts = "2026-08-01T00:00:00+00:00"
    state.phase_started_unix = 1785_542_400.0
    state.set_stop_reason("time_exhausted")
    # Pin where the leg ended so the banked segment is a checkable number.
    state.stop_ts = "2026-08-01T00:30:00+00:00"

    cb._begin_resume_leg(state, reanchor_budget=True)

    assert state.phase_elapsed_totals["PRELUDE"] == 1800.0
    assert state.stop_ts == ""


def test_a_second_resume_banks_only_the_leg_that_just_stopped() -> None:
    """The first leg's segment is already durable; re-banking it would double-charge the phase."""
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00")
    state.phase = "PRELUDE"
    state.phase_started_ts = "2026-08-01T00:00:00+00:00"
    state.phase_started_unix = 1785_542_400.0
    state.set_stop_reason("time_exhausted")
    state.stop_ts = "2026-08-01T00:30:00+00:00"
    cb._begin_resume_leg(state, reanchor_budget=True)

    # A second leg picked up a day later and ran an hour, still in PRELUDE.
    state.resumed_ts = "2026-08-02T00:00:00+00:00"
    state.set_stop_reason("time_exhausted")
    state.stop_ts = "2026-08-02T01:00:00+00:00"
    cb._begin_resume_leg(state, reanchor_budget=True)

    assert state.phase_elapsed_totals["PRELUDE"] == 1800.0 + 3600.0


def test_a_resume_does_not_bank_a_stop_stamped_after_the_present() -> None:
    """Banking past now charges the phase for time no leg ran, which ends it early."""
    started = time.time() - 60.0
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00")
    state.phase = "PRELUDE"
    state.phase_started_ts = datetime.fromtimestamp(started, tz=timezone.utc).isoformat()
    state.phase_started_unix = started
    state.set_stop_reason("time_exhausted")
    state.stop_ts = datetime.fromtimestamp(started + 10 * 86400.0, tz=timezone.utc).isoformat()

    cb._begin_resume_leg(state, reanchor_budget=True)

    assert 60.0 <= state.phase_elapsed_totals["PRELUDE"] < 120.0


def test_a_resume_with_no_recorded_stop_leaves_the_segment_unbanked() -> None:
    """A clean stop records no end time; under-charge the phase rather than guess one."""
    state = SharedState(session_id="s", start_ts="2026-08-01T00:00:00+00:00")
    state.phase = "PRELUDE"
    state.phase_started_ts = "2026-08-01T00:00:00+00:00"
    state.phase_started_unix = 1785_542_400.0

    cb._begin_resume_leg(state, reanchor_budget=False)

    assert state.phase_elapsed_totals == {}


def test_reconcile_crash_count_updates_state_and_final_json(tmp_path: Path) -> None:
    state = SharedState(session_id="s", crash_count=5)
    SharedState(session_id="s", crash_count=1).save(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final.json").write_text(
        json.dumps({"crash_count": 2, "other": True}),
        encoding="utf-8",
    )

    cb._reconcile_crash_count(state, tmp_path)

    assert SharedState.load_or_init(tmp_path).crash_count == 5
    patched = json.loads((reports / "final.json").read_text(encoding="utf-8"))
    assert patched["crash_count"] == 5
    assert patched["other"] is True


def test_kernel_opt_summary_line_prints_totals(tmp_path: Path, monkeypatch, capsys) -> None:
    from hyperloom.orchestrator.kernel import attempt_summary as kernel_attempt_summary

    monkeypatch.setenv("HYPERLOOM_SESSION_DIR", str(tmp_path))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "kernel_optimization_summary.json").write_text("{}", encoding="utf-8")

    def _summary(_state, _session_dir):
        return {
            "totals": {"attempted": 3, "integrated": 1, "rejected": 1, "unattempted": 2},
            "top_takeaways": ["headline", "root cause"],
        }

    monkeypatch.setattr(kernel_attempt_summary, "build_kernel_optimization_summary", _summary)

    cb._print_kernel_opt_summary_line(SharedState(session_id="s"))

    out = capsys.readouterr().out
    assert "3 attempted" in out
    assert "root cause" in out
    assert "kernel_optimization_summary.json" in out


def test_resolve_reference_recipe_branches(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from hyperloom.inference_optimizer import reference_script

    args = _args(model="/models/kimi", reference_script="")
    assert cb._resolve_reference_recipe(args) == ("", {}, "", "")

    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: SimpleNamespace(
            server_args="--tp 8",
            envs={"A": "1"},
            model="kimi",
        ),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="usable.sh")) == (
        "--tp 8",
        {"A": "1"},
        "kimi",
        "usable.sh",
    )

    # Unreadable path → SystemExit(2), no discovery fallback.
    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: (_ for _ in ()).throw(OSError("not found")),
    )
    with pytest.raises(SystemExit) as exc_info:
        cb._resolve_reference_recipe(_args(reference_script="empty.sh"))
    assert exc_info.value.code == 2
