# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_common_io_bytes_and_safe_mtime_edges(monkeypatch, tmp_path) -> None:
    from common import io

    path = tmp_path / "nested" / "payload.bin"
    io.atomic_write_bytes(path, b"abc", options=io.WriteOptions(make_parents=True, fsync=True, mode=0o777))
    assert path.read_bytes() == b"abc"
    assert path.stat().st_mode & 0o777 == 0o700

    assert io.safe_mtime(tmp_path / "missing") == 0.0

    def _boom_replace(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", _boom_replace)
    with pytest.raises(OSError, match="replace failed"):
        io.atomic_write_bytes(tmp_path / "will_fail.bin", b"x")
    assert not list(tmp_path.glob(".will_fail.bin.*.tmp"))


def test_llm_config_parse_and_derive_edges() -> None:
    from common.llm_config import (
        claude_sdk_env_options,
        derive_openai_base_url,
    )
    from common.llm_headers import parse_custom_headers

    assert parse_custom_headers(None) == {}
    assert parse_custom_headers("   ") == {}
    assert parse_custom_headers('{"X-Team": " hyperloom ", "": "drop"}') == {"X-Team": "hyperloom"}
    assert parse_custom_headers("{not json}\nX-Fallback: yes") == {"X-Fallback": "yes"}

    assert derive_openai_base_url(None) is None
    assert derive_openai_base_url("   ") is None
    assert derive_openai_base_url("https://gw.example/Unified") == "https://gw.example/Unified/v1"
    assert derive_openai_base_url("https://gw.example/custom") == "https://gw.example/custom"

    assert claude_sdk_env_options(env={}) == {}


def test_retry_policy_env_and_on_retry_error(monkeypatch) -> None:
    from hyperloom.orchestrator.roles import base
    from hyperloom.orchestrator.roles.base import RetryPolicy

    monkeypatch.setenv("HL_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("HL_RETRY_BASE_S", "bad")
    monkeypatch.setenv("HL_RETRY_MAX_S", "inf")
    monkeypatch.setenv("HL_RETRY_MULT", "-1")
    monkeypatch.setenv("HL_RETRY_JITTER_S", "2")
    policy = RetryPolicy.from_env("HL_RETRY")
    assert policy.max_attempts == 1
    assert policy.base_delay_s == 1.0
    assert policy.max_delay_s == 30.0
    assert policy.multiplier == 2.0
    assert policy.jitter_s == 2.0

    monkeypatch.setattr(base.random, "uniform", lambda _lo, _hi: 0.25)
    assert RetryPolicy(base_delay_s=2.0, max_delay_s=3.0, multiplier=10.0, jitter_s=0.5).delay_for(2) == 3.25


@pytest.mark.asyncio
async def test_retry_with_backoff_propagates_on_retry_callback_error() -> None:
    from hyperloom.orchestrator.roles.base import RetryPolicy, retry_with_backoff

    async def _flaky():
        raise ConnectionError("transient")

    async def _sleep(delay: float) -> None:
        raise AssertionError("must not sleep after the callback raised")

    def _bad_on_retry(*_args):
        raise RuntimeError("telemetry failed")

    with pytest.raises(RuntimeError, match="telemetry failed"):
        await retry_with_backoff(
            _flaky,
            policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, jitter_s=0.0),
            retry_on=(ConnectionError,),
            sleep=_sleep,
            on_retry=_bad_on_retry,
        )


def test_gpu_type_resolution_env_probe_and_runner(monkeypatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    assert gpu_types.gpu_runner_type("MI325X") == "mi300x"
    assert gpu_types.gpu_runner_type("mi355x") == "mi355x"
    assert gpu_types.gpu_runner_type("r9700") == "gfx12"
    assert gpu_types._runner_framework_error("r9700", "vllm") is None
    assert "no validated gfx12 runner" in gpu_types._runner_framework_error("r9700", "sglang")

    resolved, warnings = gpu_types._resolve_gpu_type("mi300x", "mi355x")
    assert resolved == "mi355x"
    assert "disagrees" in warnings[0]
    assert gpu_types._resolve_gpu_type("mi300x", "") == ("mi300x", [])

    monkeypatch.delenv("TARGET_GPU_TYPE", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    assert gpu_types._resolve_amd_gpu_type("mi308x") == "mi308x"
    assert gpu_types._resolve_amd_gpu_type("nvidia") is None

    monkeypatch.setenv("GPU_TYPE", " MI325X ")
    assert gpu_types._resolve_amd_gpu_type() == "mi325x"
    monkeypatch.setenv("TARGET_GPU_TYPE", "MI325X")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    assert gpu_types._resolve_amd_gpu_type() == "mi325x"
    monkeypatch.delenv("TARGET_GPU_TYPE")
    monkeypatch.setenv("GPU_TYPE", "unknown")
    assert gpu_types._resolve_amd_gpu_type() is None

    monkeypatch.setenv("TARGET_GPU_TYPE", "r9700")
    monkeypatch.setenv("GPU_TYPE", "gfx12")
    assert gpu_types._resolve_amd_gpu_type() == "r9700"

    monkeypatch.delenv("TARGET_GPU_TYPE", raising=False)
    monkeypatch.setenv("GPU_TYPE", "gfx1201")
    assert gpu_types._resolve_amd_gpu_type() is None

    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda: "mi355x")
    assert gpu_types._resolve_amd_gpu_type() == "mi355x"
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda: "gfx000")
    assert gpu_types._resolve_amd_gpu_type() is None


def test_gpu_type_autodetect_rocm_and_torch_fallback(monkeypatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    class _Completed:
        stdout = "GPU[0] : Card series: AMD Instinct MI325X"

    def _rocm_ok(cmd, capture_output, text, timeout):
        assert cmd == ["rocm-smi", "--showproductname"]
        assert capture_output is True
        assert text is True
        assert timeout == 5
        return _Completed()

    monkeypatch.setattr(subprocess, "run", _rocm_ok)
    assert gpu_types._autodetect_gpu_type() == "mi325x"

    def _rocm_missing(*_args, **_kwargs):
        raise FileNotFoundError("rocm-smi")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"))
    )
    monkeypatch.setattr(subprocess, "run", _rocm_missing)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert gpu_types._autodetect_gpu_type() == "mi355x"

    fake_torch.cuda.get_device_properties = lambda _idx: (_ for _ in ()).throw(RuntimeError("no gpu"))
    assert gpu_types._autodetect_gpu_type() is None

    fake_torch.cuda.get_device_properties = lambda _idx: (_ for _ in ()).throw(KeyError("device"))
    with pytest.raises(KeyError, match="device"):
        gpu_types._autodetect_gpu_type()


def test_r9700_autodetect_requires_exact_product_not_gfx1201(monkeypatch) -> None:
    from hyperloom.inference_optimizer import gpu_types

    class _Completed:
        stdout = "GPU[0] : Card series: AMD Radeon AI PRO R9700"

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: _Completed())
    assert gpu_types._autodetect_gpu_type() == "r9700"

    _Completed.stdout = "GPU[0] : Card series: AMD Radeon AI PRO R9700S"
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx1201"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert gpu_types._autodetect_gpu_type() is None


def test_multi_node_state_paths_resolution_and_binding(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths
    from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    with pytest.raises(RuntimeError, match="cannot resolve"):
        state_paths.resolve_state_file()

    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(explicit))
    assert state_paths.resolve_state_file() == explicit

    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    session = tmp_path / "session"
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(session))
    assert state_paths.resolve_state_file() == session / "runtime" / "multi_node_state.json"

    missing = tmp_path / "missing.json"
    assert state_paths.state_file_safe_to_read(missing) is False
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text("{}", encoding="utf-8")
    unsafe.chmod(0o666)
    assert state_paths.state_file_safe_to_read(unsafe) is False
    unsafe.chmod(0o600)
    assert state_paths.state_file_safe_to_read(unsafe) is True

    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "source_state.json"))
    bound = state_paths.bind_state_file_to_session(session)
    assert bound == session / "runtime" / "multi_node_state.json"
    assert not bound.exists()
    assert state_paths.resolve_state_file() == bound
    assert bound.parent.stat().st_mode & 0o777 == 0o700


def test_multi_node_state_paths_warn_on_permission_failures(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer.multi_node import state_paths

    messages: list[str] = []
    monkeypatch.setattr(state_paths, "warn", messages.append)

    runtime_dir = tmp_path / "runtime"
    original_chmod = type(runtime_dir).chmod

    def _bad_chmod(self, mode):
        if self == runtime_dir:
            raise OSError("runtime chmod denied")
        return original_chmod(self, mode)

    monkeypatch.setattr(type(runtime_dir), "chmod", _bad_chmod)
    state_paths._ensure_runtime_dir(runtime_dir)
    assert runtime_dir.is_dir()
    assert "could not chmod runtime dir" in messages[-1]
