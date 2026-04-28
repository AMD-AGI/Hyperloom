"""Tests for ``env_probe`` — GPU / framework auto-detection."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import pytest

from inference_optimizer.orchestrator import env_probe as ep


# ---------------------------------------------------------------------------
@dataclass
class _FakeProc:
    returncode: int
    stdout: str
    stderr: str = ""


def _runner_returning(text: str, *, rc: int = 0):
    def runner(cmd):
        return _FakeProc(returncode=rc, stdout=text)
    return runner


# ---------------------------------------------------------------------------
# detect_gpu_count
# ---------------------------------------------------------------------------
def test_detect_gpu_count_explicit_env_wins():
    assert ep.detect_gpu_count({"GPU_COUNT": "8"}, _runner_returning("")) == 8


def test_detect_gpu_count_hip_visible_devices():
    assert ep.detect_gpu_count(
        {"HIP_VISIBLE_DEVICES": "0,1,2,3"},
        _runner_returning(""),
    ) == 4


def test_detect_gpu_count_rocr_visible_devices():
    assert ep.detect_gpu_count(
        {"ROCR_VISIBLE_DEVICES": "0,1"},
        _runner_returning(""),
    ) == 2


def test_detect_gpu_count_cuda_visible_devices():
    assert ep.detect_gpu_count(
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        _runner_returning(""),
    ) == 8


def test_detect_gpu_count_amd_smi(monkeypatch):
    """``amd-smi list`` produces lines starting with ``GPU:``."""
    monkeypatch.setattr(ep.shutil, "which",
                        lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None)
    text = "GPU: 0001:00\nMEM: 64GB\nGPU: 0002:00\nMEM: 64GB\nGPU: 0003:00\n"
    assert ep.detect_gpu_count({}, _runner_returning(text)) == 3


def test_detect_gpu_count_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(ep.shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi"
                        if name == "nvidia-smi" else None)
    text = "GPU 0: NVIDIA H100\nGPU 1: NVIDIA H100\n"
    assert ep.detect_gpu_count({}, _runner_returning(text)) == 2


def test_detect_gpu_count_returns_none_when_nothing_works(monkeypatch):
    monkeypatch.setattr(ep.shutil, "which", lambda _name: None)
    assert ep.detect_gpu_count({}, _runner_returning("")) is None


# ---------------------------------------------------------------------------
# detect_gpu_type
# ---------------------------------------------------------------------------
def test_detect_gpu_type_explicit_env_wins():
    assert ep.detect_gpu_type({"GPU_TYPE": "MI300X"}, _runner_returning("")) \
        == "MI300X"


def test_detect_gpu_type_rocm_smi(monkeypatch):
    monkeypatch.setattr(ep.shutil, "which",
                        lambda name: "/usr/bin/rocm-smi"
                        if name == "rocm-smi" else None)
    text = (
        "============================ ROCm System Management Interface ============================\n"
        "GPU[0]                  : GFX Version: gfx942\n"
        "GPU[1]                  : GFX Version: gfx942\n"
    )
    assert ep.detect_gpu_type({}, _runner_returning(text)) == "gfx942"


def test_detect_gpu_type_returns_none_when_no_smi(monkeypatch):
    monkeypatch.setattr(ep.shutil, "which", lambda _n: None)
    assert ep.detect_gpu_type({}, _runner_returning("")) is None


# ---------------------------------------------------------------------------
# detect_framework
# ---------------------------------------------------------------------------
def test_detect_framework_explicit_env_wins(monkeypatch):
    """``FRAMEWORK=vllm`` overrides any auto-detection."""
    name, ver = ep.detect_framework({"FRAMEWORK": "vllm"})
    assert name == "vllm"  # version may be None if vllm isn't installed


def test_detect_framework_rejects_garbage_env(monkeypatch):
    """Unknown ``FRAMEWORK=`` value is ignored, falls through to auto."""
    name, _ver = ep.detect_framework({"FRAMEWORK": "tinygrad"})
    # auto-detect can return either sglang/vllm/None depending on test env;
    # the important assertion is that we did NOT return "tinygrad"
    assert name in (None, "sglang", "vllm")


def test_import_version_returns_none_on_missing_module():
    assert ep._import_version("definitely_not_a_real_module_xyz") is None


# ---------------------------------------------------------------------------
# fill_default_env
# ---------------------------------------------------------------------------
def test_fill_default_env_does_not_clobber_operator_settings():
    base = {"TP": "8", "CONC": "999", "ISL": "555", "OSL": "777", "PORT": "9000"}
    probe = ep.EnvProbe(gpu_count=4, gpu_type="gfx942",
                        framework="sglang", framework_version="0.5")
    result = ep.fill_default_env(base, probe)
    assert result["TP"] == "8"     # operator wins
    assert result["CONC"] == "999"
    assert result["ISL"] == "555"
    assert result["OSL"] == "777"
    assert result["PORT"] == "9000"
    # auto-detected fields ARE added
    assert result["GPU_COUNT"] == "4"
    assert result["GPU_TYPE"] == "gfx942"
    assert result["FRAMEWORK"] == "sglang"


def test_fill_default_env_derives_tp_from_gpu_count():
    probe = ep.EnvProbe(gpu_count=4)
    result = ep.fill_default_env({}, probe)
    assert result["TP"] == "4"
    assert result["CONC"] == "32"   # tp <= 4 → 32
    assert result["ISL"] == "1024"
    assert result["OSL"] == "256"
    assert result["PORT"] == "8888"


def test_fill_default_env_conc_buckets():
    """The CONC default depends on TP (sister script convention)."""
    cases = [(1, "4"), (4, "32"), (8, "64")]
    for tp, expected in cases:
        result = ep.fill_default_env({}, ep.EnvProbe(gpu_count=tp))
        assert result["CONC"] == expected, f"tp={tp}"


def test_fill_default_env_no_gpu_no_tp():
    """When no GPU is detected and operator hasn't set TP, we must not
    invent one — the executor will detect the missing env later and
    fall back to LLM."""
    probe = ep.EnvProbe()  # all None
    result = ep.fill_default_env({}, probe)
    assert "TP" not in result
    # Defaults that don't depend on TP still apply:
    assert result["ISL"] == "1024"
    assert result["OSL"] == "256"
    assert result["PORT"] == "8888"


# ---------------------------------------------------------------------------
# probe_environment (top-level smoke)
# ---------------------------------------------------------------------------
def test_probe_environment_gracefully_handles_no_tools(monkeypatch):
    monkeypatch.setattr(ep.shutil, "which", lambda _n: None)
    probe = ep.probe_environment(env={}, runner=_runner_returning(""))
    assert probe.gpu_count is None
    assert probe.gpu_type is None
    assert probe.rocm_smi is None
    assert probe.amd_smi is None
    assert probe.nvidia_smi is None


def test_probe_environment_prefers_explicit_env_overrides():
    probe = ep.probe_environment(
        env={"GPU_COUNT": "16", "GPU_TYPE": "MI355X", "FRAMEWORK": "sglang"},
        runner=_runner_returning(""),
    )
    assert probe.gpu_count == 16
    assert probe.gpu_type == "MI355X"
    assert probe.framework == "sglang"


def test_envprobe_to_env_excludes_none_fields():
    probe = ep.EnvProbe(gpu_count=4, gpu_type=None, framework=None)
    e = probe.to_env()
    assert e == {"GPU_COUNT": "4"}
