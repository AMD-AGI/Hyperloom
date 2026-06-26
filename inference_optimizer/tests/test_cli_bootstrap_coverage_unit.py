# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Focused coverage for CLI bootstrap helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace

from inference_optimizer import cli_bootstrap as cb
from inference_optimizer.orchestrator.shared_state import SharedState


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
        continue_kernel_after_gemm=True,
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
        explore_force_exit_hours_remaining=1.25,
        explore_force_exit_budget_pct=0.2,
        explore_overtime_kill_ratio="bad",
        explore_variant_timeout_sec="bad",
        explore_variant_timeout_safety_margin="bad",
        enable_roofline=False,
        no_framework=True,
        no_explore=True,
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


def test_seed_shared_state_populates_perfskills_and_cli_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge, perfskills")
    monkeypatch.setenv("CLAW_SESSION_ID", "claw-1")
    monkeypatch.setenv("SANDBOX_USER_ID", "user-1")
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("FRAMEWORK_VERSION", "0.21.0")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {
        "architectures": ["KimiK2ForCausalLM"],
        "model_type": "kimi_k25",
    })
    monkeypatch.setattr(cb, "_load_model_arch", lambda *_a, **_k: {"layers": 61})
    monkeypatch.setattr(cb, "_workspace_root_resolve", lambda: tmp_path)
    monkeypatch.setattr(cb, "_resolve_reference_recipe", lambda _args: (
        "--block-size 64",
        {"ENV_A": "1"},
        "Kimi-K2.6",
        "/recipes/kimi.sh",
    ))

    from inference_optimizer.orchestrator import policy

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
    assert state.kernel_optimizer == "perfskills"
    assert state.research_lane_capacity == 16
    assert state.gpu_specialist_capacity == 8
    assert state.plateau_overrides["explore_keep_gain_pct"] == 1.5
    assert state.plateau_overrides["kernel_keep_gain_pct"] == 2.5
    assert state.explore_overtime_kill_ratio == 2.0
    assert state.explore_variant_timeout_sec_override == 0
    assert state.explore_variant_timeout_safety_margin == 0.5
    assert state.framework_phase_enabled is False
    assert state.explore_enabled is False
    assert state.conc_sweep_concs == [1, 4, 8]
    assert state.conc_sweep_total_budget_sec == 120
    assert state.conc_sweep_variant_timeout_sec == 30
    assert state.reference_server_args == "--block-size 64"
    assert json.loads((tmp_path / "state.json").read_text())["session_id"] == "session-1"


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
        1, 2, 4, 8, 16, 32, 64, 128,
    ]
    assert cb._parse_conc_sweep_concs(_args(conc_sweep_concs="bad,")) == [
        1, 2, 4, 8, 16, 32, 64, 128,
    ]
    assert "ignoring non-integer CONC token" in caplog.text


def test_read_failure_summary_and_final_summary_output(tmp_path: Path, capsys) -> None:
    assert cb._read_failure_summary(tmp_path) is None
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final.json").write_text(
        json.dumps({
            "failure_summary": {
                "root_cause_type": "config",
                "root_cause": "bad flag",
                "server_log": "/tmp/server.log",
            },
        }),
        encoding="utf-8",
    )
    assert cb._read_failure_summary(tmp_path)["root_cause"] == "bad flag"

    state = SharedState(
        session_id="s",
        model_name="m",
        baseline_tput=10.0,
        cumulative_gain=1.25,
        cumulative_gain_validated=1.0,
        cumulative_gain_validated_ts="2026-01-01T00:00:00Z",
        cumulative_gain_validated_stack_len=0,
        current_best={"action": "x"},
        pruned_families=["a"],
        crash_count=2,
    )
    state.optimization_stack = [{"action": "perfskills_e2e"}]

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


def test_snapshot_skeleton_and_session_dir_helpers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cb._snapshot_system_prompts(tmp_path, prompts={"orch": "hello", "kernel": ""})
    assert (tmp_path / "agents" / "orch" / "system_prompt.snapshot.md").read_text(
        encoding="utf-8",
    ) == "hello"
    assert (tmp_path / "agents" / "kernel" / "system_prompt.snapshot.md").read_text(
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
    from inference_optimizer.orchestrator import kernel_attempt_summary

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
    from inference_optimizer import reference_script

    assert cb._resolve_reference_recipe(_args(reference_script="")) == ("", {}, "", "")

    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: SimpleNamespace(
            server_args="--tp 8" if source == "usable.sh" else "",
            envs={"A": "1"} if source == "usable.sh" else {},
            model="kimi" if source == "usable.sh" else "",
        ),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="usable.sh")) == (
        "--tp 8",
        {"A": "1"},
        "kimi",
        "usable.sh",
    )

    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == ("", {}, "", "")

    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.setattr(
        reference_script,
        "discover_reference_script",
        lambda *_a, **_k: ("auto.sh", "exact"),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == (
        "",
        {},
        "",
        "auto.sh",
    )

    monkeypatch.setattr(
        reference_script,
        "discover_reference_script",
        lambda *_a, **_k: ("fuzzy.sh", "fuzzy"),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == ("", {}, "", "")

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
    cb._snapshot_system_prompts(tmp_path, prompts={"orch": "hello", "kernel": ""})
    assert (tmp_path / "agents" / "orch" / "system_prompt.snapshot.md").read_text(
        encoding="utf-8",
    ) == "hello"
    assert (tmp_path / "agents" / "kernel" / "system_prompt.snapshot.md").read_text(
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
    from inference_optimizer.orchestrator import kernel_attempt_summary

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
    from inference_optimizer import reference_script

    args = _args(model="/models/kimi", reference_script="")
    assert cb._resolve_reference_recipe(args) == ("", {}, "", "")

    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: SimpleNamespace(
            server_args="--tp 8" if source == "usable.sh" else "",
            envs={"A": "1"} if source == "usable.sh" else {},
            model="kimi" if source == "usable.sh" else "",
        ),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="usable.sh")) == (
        "--tp 8",
        {"A": "1"},
        "kimi",
        "usable.sh",
    )

    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == ("", {}, "", "")

    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.setattr(
        reference_script,
        "discover_reference_script",
        lambda *_a, **_k: ("auto.sh", "exact"),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == (
        "",
        {},
        "",
        "auto.sh",
    )

    monkeypatch.setattr(
        reference_script,
        "discover_reference_script",
        lambda *_a, **_k: ("fuzzy.sh", "fuzzy"),
    )
    assert cb._resolve_reference_recipe(_args(reference_script="empty.sh")) == ("", {}, "", "")
