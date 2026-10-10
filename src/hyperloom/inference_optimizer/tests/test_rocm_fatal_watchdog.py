# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm fatal faults use the shared watchdog during boot and generation."""

from __future__ import annotations

import sys

import pytest

from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk


FAULTS = (
    "Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT",
    ":0:rocdevice.cpp :3905: Memory Fault Error [host: test-host, GPU index: 0, kernel: example_kernel]",
)


@pytest.fixture(autouse=True)
def disable_metrics_scraping(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KV_METRICS", "0")


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("nested", (False, True))
def test_rocm_fault_detection_and_excerpt(tmp_path, fault, nested):
    watched = tmp_path / "server.log"
    actual = tmp_path / "benchmark_sglang_20260915_214837" / "server.log" if nested else watched
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_text("compilation complete\n" + fault + "\n")
    assert sk._server_log_shows_death(str(watched)) is not None
    assert fault in sk.server_log_death_excerpt(str(watched))


@pytest.mark.parametrize(
    "text",
    (
        "Successfully allocated memory; compilation completed",
        "Warning: memory usage is high",
        "Memory Fault Error rate: 0",
        "HSA_STATUS_SUCCESS",
    ),
)
def test_rocm_nonfatal_output_is_not_terminal(tmp_path, text):
    watched = tmp_path / "server.log"
    watched.write_text(text + "\n")
    assert sk._server_log_shows_death(str(watched)) is None
    assert sk.server_log_death_excerpt(str(watched)) is None


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("ready", (False, True))
def test_rocm_fault_reaps_hung_child_despite_continued_output(tmp_path, monkeypatch, fault, nested, ready):
    monkeypatch.setattr(sk, "STOP_GATE_POLL_SECONDS", 0.02)
    watched = tmp_path / "server.log"
    actual = tmp_path / "benchmark_sglang_test" / "server.log" if nested else watched
    payload = ("Application startup complete\n" if ready else "loading weights\n") + fault + "\n"
    script = (
        "import pathlib, sys, time\n"
        "p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(sys.argv[2])\n"
        "for _ in range(3000):\n"
        "    with p.open('a') as f: f.write('still logging\\n')\n"
        "    time.sleep(0.02)\n"
    )
    result = sk.run_with_session_kill(
        [sys.executable, "-c", script, str(actual), payload],
        timeout=5,
        server_log_path=str(watched),
        server_dead_grace_sec=0.05,
        detok_stall_grace_sec=30,
    )
    assert result.returncode == sk.SERVER_DEAD_RETURNCODE


@pytest.mark.parametrize("fault", FAULTS)
def test_rocm_fault_preserves_grace_for_clean_exit(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(sk, "STOP_GATE_POLL_SECONDS", 0.02)
    watched = tmp_path / "server.log"
    script = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); time.sleep(0.1)"
    result = sk.run_with_session_kill(
        [sys.executable, "-c", script, str(watched), fault],
        timeout=5,
        server_log_path=str(watched),
        server_dead_grace_sec=1,
    )
    assert result.returncode == 0
