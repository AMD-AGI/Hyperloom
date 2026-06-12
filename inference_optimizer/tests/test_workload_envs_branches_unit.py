# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch-coverage tests for shared workload-env materialization: GPU-count
detection, profile-window math, per-model work-arounds, and NUM_PROMPTS
sizing."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import _workload_envs as we


def _clear_env(monkeypatch):
    for k in ("TP", "ISL", "OSL", "CONC", "MAX_MODEL_LEN", "PRECISION",
              "RANDOM_RANGE_RATIO", "ROCR_VISIBLE_DEVICES", "RUN_EVAL",
              "PROFILE", "MODEL_PATH", "INFERENCEX_PATH",
              "HYPERLOOM_PROFILE_MAX_ITERS", "HYPERLOOM_PROFILE_DELAY_ITERS",
              "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT",
              "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP"):
        monkeypatch.delenv(k, raising=False)


def _write(path, **bench_extra):
    bench = {"framework": "sglang", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _materialize(src, out, **kw):
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]


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
    monkeypatch.setattr(we.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=out))
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


# ---- default_baseline_config ----------------------------------------------
def test_default_baseline_config(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "atom")
    assert we.default_baseline_config().name == "baseline_atom.yaml"
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert we.default_baseline_config().name == "baseline_vllm.yaml"
    monkeypatch.setenv("FRAMEWORK", "weird")
    assert we.default_baseline_config().name == "baseline_sglang.yaml"


# ---- precision + gpu_type without framework -------------------------------
def test_precision_and_gpu_type_no_framework(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PRECISION", "fp8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = tmp_path / "cfg.yaml"
    # framework empty -> gpu_type branch pops benchmark_script
    src.write_text(yaml.safe_dump({"benchmark": {
        "model": "/m", "envs": {}, "benchmark_script": "old.sh"}}),
        encoding="utf-8")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x")
    assert bench["precision"] == "fp8"
    assert "benchmark_script" not in bench


# ---- ROCR-derived TP ------------------------------------------------------
def test_rocr_derives_tp(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["TP"] == 3


# ---- NUM_PROMPTS factor branches (non-profile) ----------------------------
@pytest.mark.parametrize("isl,osl,conc,factor", [
    (4000, 2000, 8, 3),     # 1024 < seq <= 16384 region (6000) -> factor 3
    (3000, 1000, 8, 5),     # 1024 < seq <= 4096 (4000) -> factor 5
    (20000, 5000, 8, 2),    # > 16384 -> factor 2
])
def test_num_prompts_factor(monkeypatch, tmp_path, isl, osl, conc, factor):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ISL", str(isl))
    monkeypatch.setenv("OSL", str(osl))
    monkeypatch.setenv("CONC", str(conc))
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["NUM_PROMPTS"] == conc * factor


# ---- server_args merge into existing --------------------------------------
def test_server_args_merge_existing(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml",
                 envs={"EXTRA_SGLANG_ARGS": "--mem-fraction-static 0.9"})
    bench = _materialize(src, tmp_path / "out",
                         extra_server_args="--chunked-prefill-size 2048")
    merged = bench["envs"]["EXTRA_SGLANG_ARGS"]
    assert "mem-fraction-static" in merged
    assert "chunked-prefill-size" in merged


# ---- per-model work-around: mimo-v2 ---------------------------------------
def test_mimo_v2_injects_triton_attention(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out",
                         model_path="/wekafs/models/MiMo-V2-7B")
    assert "attention-backend triton" in bench["envs"]["EXTRA_SGLANG_ARGS"]


# ---- RUN_EVAL from env ----------------------------------------------------
def test_run_eval_from_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("RUN_EVAL", "true")
    src = _write(tmp_path / "cfg.yaml")
    bench = _materialize(src, tmp_path / "out")
    assert bench["envs"]["RUN_EVAL"] == "true"


# ---- profile-window math --------------------------------------------------
def test_profile_negative_delay_clamped(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("ISL", "1")
    monkeypatch.setenv("OSL", "1")
    monkeypatch.setenv("CONC", "8")
    # PROFILE in yaml envs triggers is_profile; bad RANDOM_RANGE_RATIO -> except
    src = _write(tmp_path / "cfg.yaml",
                 envs={"PROFILE": "1", "RANDOM_RANGE_RATIO": "not-a-float"})
    bench = _materialize(src, tmp_path / "out")
    # profile path forces NUM_PROMPTS; small OSL clamps delay to 0
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
    src.write_text(yaml.safe_dump({"benchmark": {
        "framework": "atom", "model": "/m", "envs": {"PROFILE": "1"}}}),
        encoding="utf-8")
    bench = _materialize(src, tmp_path / "out")
    # atom defers NUM_PROMPTS to Magpie (profile_num_prompts None) -> factor path
    assert "NUM_PROMPTS" in bench["envs"]


def test_profile_sglang_bad_extra_body(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    src = _write(tmp_path / "cfg.yaml",
                 envs={"PROFILE": "1", "PROFILE_EXTRA_BODY": "{bad json"})
    bench = _materialize(src, tmp_path / "out")
    body = bench["envs"]["PROFILE_EXTRA_BODY"]
    assert "start_step" in body and "num_steps" in body
