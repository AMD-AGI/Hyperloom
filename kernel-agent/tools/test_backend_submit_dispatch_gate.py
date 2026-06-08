# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the Ray-cluster pre-flight gate on the backend
submit paths.

Defect 2 from /home/sapmajum/hyperloom_work/logs/geak_dispatch_audit.md:
``geak_submit.submit`` and ``oob_submit.submit`` used to call
``quiet_ray_init()`` directly inside their ``run_via_ray`` wrapper, which
spends 30 seconds retrying ``global_state_accessor.cc:500`` GCS handshake
attempts when the raylet is wedged. These tests pin down two behaviours:

  1. ``submit`` calls ``ensure_ray_cluster()`` *before* attempting to use Ray
     so a wedged cluster gets restarted (or the failure surfaces fast).
  2. When ``ensure_ray_cluster()`` raises, ``submit`` returns the
     dispatch-failure envelope with the diagnostic hint string in
     ``stderr_tail`` (so downstream
     ``kernel_optimization.build_verification`` can record
     ``artifact_error`` and ``make_proposal`` can surface it).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import geak_submit  # noqa: E402
import oob_submit  # noqa: E402


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    return out


def test_geak_submit_calls_ensure_ray_cluster_before_run_via_ray(tmp_output_dir, tmp_path):
    """``ensure_ray_cluster`` must run before ``run_via_ray`` so we don't
    burn the 30 s ray.init retry budget on a wedged cluster."""
    call_order: list[str] = []

    def _fake_ensure(num_gpus=None, log_path=None):
        call_order.append("ensure_ray_cluster")
        return False  # cluster already healthy

    def _fake_run_via_ray(*args, **kwargs):
        call_order.append("run_via_ray")
        return {"returncode": 0, "stdout_tail": "ok", "stderr_tail": "",
                "gpu_ids": "0", "elapsed_s": 0.1, "cmd": []}

    with mock.patch.object(geak_submit, "ensure_ray_cluster", _fake_ensure), \
         mock.patch.object(geak_submit, "run_via_ray", _fake_run_via_ray):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("noop", encoding="utf-8")
        result = geak_submit.submit(
            prompt_file=prompt, output_dir=tmp_output_dir,
            kernel_path="", cost_limit=0.0, num_gpus=1,
            prefer_ray=True,
        )

    assert call_order == ["ensure_ray_cluster", "run_via_ray"], call_order
    assert result["returncode"] == 0


def test_geak_submit_returns_dispatch_failure_envelope_on_ensure_failure(
    tmp_output_dir, tmp_path,
):
    """When ``ensure_ray_cluster`` fails (raylet wedged, ray start cannot
    recover), ``submit`` must return a dispatch-failure envelope whose
    ``stderr_tail`` carries the diagnostic hint so kernel_optimization can
    record ``artifact_error`` and ``make_proposal`` can surface it."""
    def _boom(num_gpus=None, log_path=None):
        raise RuntimeError("failed to start Ray; see ray_lifecycle.log")

    run_via_ray_called = {"hit": False}

    def _fake_run_via_ray(*args, **kwargs):
        run_via_ray_called["hit"] = True
        return {"returncode": 0}

    with mock.patch.object(geak_submit, "ensure_ray_cluster", _boom), \
         mock.patch.object(geak_submit, "run_via_ray", _fake_run_via_ray):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("noop", encoding="utf-8")
        result = geak_submit.submit(
            prompt_file=prompt, output_dir=tmp_output_dir,
            kernel_path="", cost_limit=0.0, num_gpus=1,
            prefer_ray=True,
        )

    assert result["returncode"] == 1
    assert result["elapsed_s"] == 0.0
    assert "ray submission failed" in result["stderr_tail"]
    assert "RuntimeError" in result["stderr_tail"]
    assert "ray status" in result["stderr_tail"]
    assert "raylet zombie" in result["stderr_tail"]
    assert run_via_ray_called["hit"] is False, "ensure_ray_cluster failure must short-circuit before run_via_ray"


def test_oob_submit_calls_ensure_ray_cluster_before_run_via_ray(tmp_output_dir, tmp_path):
    call_order: list[str] = []

    def _fake_ensure(num_gpus=None, log_path=None):
        call_order.append("ensure_ray_cluster")
        return False

    def _fake_run_via_ray(*args, **kwargs):
        call_order.append("run_via_ray")
        return {"returncode": 0, "stdout_tail": "ok", "stderr_tail": "",
                "stdout": "ok", "gpu_ids": "0", "elapsed_s": 0.1, "cmd": []}

    with mock.patch.object(oob_submit, "ensure_ray_cluster", _fake_ensure), \
         mock.patch.object(oob_submit, "run_via_ray", _fake_run_via_ray):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("noop", encoding="utf-8")
        result = oob_submit.submit(
            agent="claude", prompt_file=prompt, output_dir=tmp_output_dir,
            source_file="", num_gpus=1, prefer_ray=True,
        )

    assert call_order == ["ensure_ray_cluster", "run_via_ray"], call_order
    assert result["returncode"] == 0


def test_oob_submit_returns_dispatch_failure_envelope_on_ensure_failure(
    tmp_output_dir, tmp_path,
):
    def _boom(num_gpus=None, log_path=None):
        raise RuntimeError("failed to start Ray; see ray_lifecycle.log")

    run_via_ray_called = {"hit": False}

    def _fake_run_via_ray(*args, **kwargs):
        run_via_ray_called["hit"] = True
        return {"returncode": 0}

    with mock.patch.object(oob_submit, "ensure_ray_cluster", _boom), \
         mock.patch.object(oob_submit, "run_via_ray", _fake_run_via_ray):
        prompt = tmp_path / "prompt.md"
        prompt.write_text("noop", encoding="utf-8")
        result = oob_submit.submit(
            agent="claude", prompt_file=prompt, output_dir=tmp_output_dir,
            source_file="", num_gpus=1, prefer_ray=True,
        )

    assert result["returncode"] == 1
    assert result["elapsed_s"] == 0.0
    assert "ray submission failed" in result["stderr_tail"]
    assert "RuntimeError" in result["stderr_tail"]
    assert "ray status" in result["stderr_tail"]
    assert "raylet zombie" in result["stderr_tail"]
    assert run_via_ray_called["hit"] is False
