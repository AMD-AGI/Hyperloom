"""Tests for ``orchestrator.kernel_opt_constants`` — IMPL-CHECKLIST §1.11‒1.20.

The constants live at module top-level as ``Final``; importing them at
runtime should always produce the same object identity. We also verify
the env-driven helpers raise cleanly when the env is missing.
"""
from __future__ import annotations

import importlib

import pytest

from inference_optimizer.orchestrator import kernel_opt_constants as koc


def test_kernel_opt_backends_value():
    assert koc.KERNEL_OPT_BACKENDS == "geak,codex"


def test_oob_round_iterations():
    assert koc.OOB_ROUND_ITERATIONS == 3


def test_kernel_opt_workspace():
    assert koc.KERNEL_OPT_WORKSPACE == "control-plane-moe"


def test_geak_budget_defaults():
    assert koc.GEAK_STEP_LIMIT == 100
    assert koc.GEAK_MAX_RETRIES == 3
    assert koc.GEAK_MAX_SUBMISSIONS == 15
    assert koc.GEAK_TOP_CANDIDATES == 5
    assert koc.GEAK_CONSECUTIVE_DISCARDS == 5
    assert koc.GEAK_WALL_CLOCK_MIN == 120
    assert koc.GEAK_POLL_INTERVAL_S == 60
    assert koc.GEAK_POLL_TIMEOUT_MIN == 15


def test_filtered_trace_name():
    assert koc.FILTERED_TRACE_NAME == "filtered-TP-0.trace.json.gz"


def test_min_gpu_pct_and_kill_wait():
    assert koc.MIN_GPU_PCT == 3
    assert koc.SERVER_KILL_WAIT_S == 10


def test_kernel_opt_image_reads_env(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_IMAGE", "ghcr.io/example/inferx:latest")
    assert koc.kernel_opt_image() == "ghcr.io/example/inferx:latest"


def test_kernel_opt_image_raises_when_missing(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_IMAGE", raising=False)
    with pytest.raises(RuntimeError):
        koc.kernel_opt_image()


def test_venv_bin_path_default(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_VENV_BIN", raising=False)
    assert koc.venv_bin_path() == "/opt/venv/bin"


def test_venv_bin_path_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VENV_BIN", "/srv/.venv/bin")
    assert koc.venv_bin_path() == "/srv/.venv/bin"


def test_constants_are_module_level_final():
    """Sanity check: the public names exist as module attributes."""
    expected = {
        "KERNEL_OPT_BACKENDS",
        "OOB_ROUND_ITERATIONS",
        "KERNEL_OPT_WORKSPACE",
        "GEAK_STEP_LIMIT",
        "GEAK_MAX_RETRIES",
        "GEAK_MAX_SUBMISSIONS",
        "GEAK_TOP_CANDIDATES",
        "GEAK_CONSECUTIVE_DISCARDS",
        "GEAK_WALL_CLOCK_MIN",
        "GEAK_POLL_INTERVAL_S",
        "GEAK_POLL_TIMEOUT_MIN",
        "MIN_GPU_PCT",
        "SERVER_KILL_WAIT_S",
        "FILTERED_TRACE_NAME",
    }
    actual = {n for n in dir(koc) if n.isupper()}
    assert expected <= actual


def test_module_reload_preserves_values():
    importlib.reload(koc)
    assert koc.OOB_ROUND_ITERATIONS == 3
