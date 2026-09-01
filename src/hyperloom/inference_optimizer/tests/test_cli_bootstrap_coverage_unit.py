# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Focused coverage for CLI bootstrap helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

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


def _stub_seed_deps(monkeypatch, tmp_path):
    """Common monkeypatches so ``_seed_shared_state`` runs without real I/O."""
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


def test_seed_uses_explicit_workload_flags_over_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #903: explicit workload flags must reach SharedState.

    When the operator passes ``--isl/--osl/--conc``, those values (not the
    fallback defaults) must seed SharedState, regardless of any inherited env.
    """
    _stub_seed_deps(monkeypatch, tmp_path)
    monkeypatch.setenv("ISL", "256")
    monkeypatch.setenv("OSL", "256")
    monkeypatch.setenv("CONC", "8")

    state = cb._seed_shared_state(
        tmp_path,
        _args(model="/models/Qwen2.5-7B", isl=512, osl=512, conc=64, tp=1, ep=1),
        session_id="s-explicit",
    )

    assert state.isl == 512
    assert state.osl == 512
    assert state.conc == 64
    assert state.tp == 1
    assert state.ep == 1


def test_seed_applies_new_workload_defaults_when_unset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #903: when workload flags are unset (argparse default
    ``None``) and no inherited env is present, SharedState must seed the new
    fallback defaults (ISL/OSL=1024, CONC=64, TP/EP=1)."""
    _stub_seed_deps(monkeypatch, tmp_path)
    for key in ("ISL", "OSL", "CONC", "TP", "EP"):
        monkeypatch.delenv(key, raising=False)

    state = cb._seed_shared_state(
        tmp_path,
        _args(
            model="/models/Qwen2.5-7B",
            isl=None,
            osl=None,
            conc=None,
            tp=None,
            ep=None,
        ),
        session_id="s-default",
    )

    assert state.isl == 1024
    assert state.osl == 1024
    assert state.conc == 64
    assert state.tp == 1
    assert state.ep == 1


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


def test_seed_passes_raw_model_path_to_model_arch_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression for #930: the stale guard needs the raw ``--model`` path so an
    HF cache ``models--org--repo/snapshots/<hash>`` launch can recover the repo
    name; passing only the collapsed identity loses it."""
    captured: dict[str, tuple] = {}

    def _spy(*args, **_kwargs):
        captured["args"] = args
        return {}

    monkeypatch.setattr(cb, "_load_model_config_tags", lambda _p: {})
    monkeypatch.setattr(cb, "_load_model_arch", _spy)
    monkeypatch.setattr(
        cb,
        "_resolve_reference_recipe",
        lambda _args: ("", {}, "", ""),
    )
    from hyperloom.orchestrator.policy import gate as policy

    monkeypatch.setattr(policy, "detect_gpu_count", lambda: 8)
    monkeypatch.setattr(policy, "research_lane_ceiling", lambda: 16)

    raw = (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    )
    cb._seed_shared_state(tmp_path, _args(model=raw), session_id="s-raw")

    assert captured["args"][2] == raw


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

    # Unreadable path raises SystemExit(2).
    monkeypatch.setattr(
        reference_script,
        "parse_reference_script",
        lambda source, framework: (_ for _ in ()).throw(OSError("not found")),
    )
    with pytest.raises(SystemExit) as exc_info:
        cb._resolve_reference_recipe(_args(reference_script="empty.sh"))
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# _detect_checkpoint_precision tests
# ---------------------------------------------------------------------------


def test_detect_checkpoint_precision_bf16_from_torch_dtype(tmp_path):
    from hyperloom.inference_optimizer.cli import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "bf16"


def test_detect_checkpoint_precision_fp8_from_quant_method(tmp_path):
    from hyperloom.inference_optimizer.cli import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "fp8"


def test_detect_checkpoint_precision_fallback_on_missing():
    from hyperloom.inference_optimizer.cli import _detect_checkpoint_precision

    assert _detect_checkpoint_precision("/nonexistent/model/path") == ""
    assert _detect_checkpoint_precision(None) == ""


def test_detect_checkpoint_precision_float8_e4m3fn(tmp_path):
    from hyperloom.inference_optimizer.cli import _detect_checkpoint_precision

    (tmp_path / "config.json").write_text('{"torch_dtype": "float8_e4m3fn"}', encoding="utf-8")
    assert _detect_checkpoint_precision(str(tmp_path)) == "fp8"


def test_resolve_workload_knobs_precision_auto_detected(tmp_path):
    import argparse

    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    args = argparse.Namespace(precision=None, model=str(tmp_path), tp=None, ep=None, conc=None)
    _resolve_workload_knobs(args)
    assert args.precision == "bf16"


def test_resolve_workload_knobs_explicit_precision_kept_with_warning(tmp_path, capsys):
    import argparse

    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    args = argparse.Namespace(precision="fp8", model=str(tmp_path), tp=None, ep=None, conc=None)
    _resolve_workload_knobs(args)
    assert args.precision == "fp8"
    captured = capsys.readouterr()
    assert "WARN" in captured.err and "bf16" in captured.err


def test_resolve_workload_knobs_resume_precision_wins(tmp_path):
    import argparse
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    (tmp_path / "config.json").write_text('{"torch_dtype": "bfloat16"}', encoding="utf-8")
    args = argparse.Namespace(precision=None, model=str(tmp_path), tp=None, ep=None, conc=None)
    state = SimpleNamespace(precision="fp8", tp=2, ep=1, conc=64)
    _resolve_workload_knobs(args, state)
    assert args.precision == "fp8"
