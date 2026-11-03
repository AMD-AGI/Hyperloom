# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest


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
