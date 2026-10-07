# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured-phase GPU power: the reading ``--max-power-w`` is graded on."""

from __future__ import annotations

import json
import time

import pytest

from hyperloom.orchestrator.actions.executors import _subprocess_kill
from hyperloom.orchestrator.actions.executors._gpu_power import (
    GPU_POWER_ARTIFACT_NAME,
    GpuPowerRecorder,
    build_gpu_power_recorder,
    parse_power_sample,
    read_measured_gpu_power,
    read_measured_gpu_power_by_gpu,
)
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

_TOTAL_MB = 294896


def _row(gpu: int, watts: float | None, used_mb: float) -> dict:
    power = {"socket_power": {"value": watts, "unit": "W"}} if watts is not None else {"socket_power": "N/A"}
    return {
        "gpu": gpu,
        "power": power,
        "mem_usage": {"total_vram": {"value": _TOTAL_MB, "unit": "MB"}, "used_vram": {"value": used_mb, "unit": "MB"}},
    }


def _tp4_on_eight(load_w: float) -> list[dict]:
    """TP4 on cards 0-3 of an unpinned eight-card host: 4 serving at load, 4 idle."""
    return [_row(g, load_w, 250_000) for g in range(4)] + [_row(g, 225.0, 284) for g in range(4, 8)]


class _Replay:
    """Feeds ``amd-smi`` payloads in order, repeating the last one."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return payload


def _run_measured(recorder: GpuPowerRecorder, query: _Replay, samples: int) -> None:
    recorder.note_phase("measured", time.monotonic())
    deadline = time.monotonic() + 5.0
    while query.calls < samples and time.monotonic() < deadline:
        time.sleep(0.01)


def test_parse_power_sample_reads_power_and_vram_fraction():
    parsed = parse_power_sample({"gpu_data": [_row(0, 812.0, 250_000), _row(1, None, 284)]})
    assert parsed[0] == (812.0, pytest.approx(250_000 / _TOTAL_MB))
    assert parsed[1][0] is None


def test_averages_only_the_serving_cards(tmp_path):
    query = _Replay(_tp4_on_eight(800.0), _tp4_on_eight(900.0))
    recorder = GpuPowerRecorder(output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=4)
    out = recorder.close()
    assert out["serving_gpus"] == [0, 1, 2, 3]
    assert 800.0 <= out["avg_power_w"] <= 900.0
    assert out["max_power_w"] == 900.0
    assert json.loads((tmp_path / GPU_POWER_ARTIFACT_NAME).read_text())["serving_gpus"] == [0, 1, 2, 3]


def test_the_by_gpu_reading_names_each_serving_card_and_no_idle_one(tmp_path):
    query = _Replay(_tp4_on_eight(800.0))
    recorder = GpuPowerRecorder(output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=3)
    recorder.close()
    found, by_gpu = read_measured_gpu_power_by_gpu(tmp_path)
    assert found is True
    assert by_gpu == {str(g): pytest.approx(800.0) for g in range(4)}


def test_samples_nothing_outside_the_measured_phase(tmp_path):
    query = _Replay(_tp4_on_eight(800.0))
    recorder = GpuPowerRecorder(output_path=None, query=query, interval_sec=0.5)
    recorder._interval = 0.01
    recorder.note_phase("warmup", time.monotonic())
    time.sleep(0.05)
    assert query.calls == 0
    _run_measured(recorder, query, samples=2)
    recorder.note_phase("eval", time.monotonic())
    time.sleep(0.05)
    frozen = query.calls
    time.sleep(0.05)
    assert query.calls == frozen
    out = recorder.close()
    assert out["avg_power_w"] == 800.0


def test_visible_mask_limits_the_cards_considered(tmp_path):
    query = _Replay(_tp4_on_eight(800.0))
    recorder = GpuPowerRecorder(output_path=None, gpus={2, 3}, query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=2)
    out = recorder.close()
    assert out["serving_gpus"] == [2, 3]


def test_query_failures_leave_the_round_unmeasured_not_zero(tmp_path):
    query = _Replay(RuntimeError("amd-smi exited 1"))
    recorder = GpuPowerRecorder(output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=2)
    out = recorder.close()
    assert out["avg_power_w"] is None
    assert out["query_errors"] >= 1
    assert read_measured_gpu_power(tmp_path) == (True, None)


def test_close_is_idempotent_and_never_started_writes_an_empty_reading(tmp_path):
    recorder = GpuPowerRecorder(output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=_Replay([]))
    first = recorder.close(aborted=True)
    assert first["samples"] == 0 and first["aborted"] is True
    recorder.close()
    assert json.loads((tmp_path / GPU_POWER_ARTIFACT_NAME).read_text())["aborted"] is True


def _write_artifact(directory, *, avg, started_unix):
    (directory / GPU_POWER_ARTIFACT_NAME).write_text(json.dumps({"avg_power_w": avg, "started_unix": started_unix}))


def test_reader_finds_the_artifact_beside_server_log_one_level_up(tmp_path):
    workspace = tmp_path / "benchmark_vllm_20260929"
    workspace.mkdir()
    _write_artifact(tmp_path, avg=812.5, started_unix=1000.0)
    assert read_measured_gpu_power(workspace, subprocess_started_unix=999.0) == (True, 812.5)


def test_reader_ignores_an_artifact_from_an_earlier_round(tmp_path):
    _write_artifact(tmp_path, avg=812.5, started_unix=1000.0)
    assert read_measured_gpu_power(tmp_path, subprocess_started_unix=2000.0) == (False, None)


_MAGPIE_REPORT = {"gpu_monitor": {"sample_count": 157, "power_watts": {"min": 225.0, "max": 790.0, "avg": 313.7}}}


def test_measurement_prefers_the_measured_phase_over_magpies_whole_process_reading(tmp_path):
    _write_artifact(tmp_path, avg=780.0, started_unix=time.time())
    measurement = extract_benchmark_measurement(_MAGPIE_REPORT, workspace=tmp_path)
    assert measurement["gpu_power_avg_w"] == 780.0


def test_measurement_keeps_a_sampled_but_unmeasured_round_unmeasured(tmp_path):
    _write_artifact(tmp_path, avg=None, started_unix=time.time())
    assert extract_benchmark_measurement(_MAGPIE_REPORT, workspace=tmp_path)["gpu_power_avg_w"] is None


def test_measurement_falls_back_to_the_report_when_no_recorder_ran(tmp_path):
    measurement = extract_benchmark_measurement(_MAGPIE_REPORT, workspace=tmp_path)
    assert measurement["gpu_power_avg_w"] == 313.7
    assert measurement["gpu_power_by_gpu_w"] is None, "Magpie's host-wide mean cannot be split per card"


def test_a_sampled_round_with_no_serving_card_has_no_by_gpu_reading(tmp_path):
    _write_artifact(tmp_path, avg=None, started_unix=time.time())
    assert read_measured_gpu_power_by_gpu(tmp_path) == (True, None)


def test_builder_follows_amd_smi_and_the_switch(tmp_path, monkeypatch):
    """On wherever amd-smi exists, unless the operator turns it off; nothing to sample without amd-smi."""
    log_path = str(tmp_path / "server.log")
    monkeypatch.delenv("HYPERLOOM_GPU_POWER_SAMPLING", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert build_gpu_power_recorder(log_path, {}) is None
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
    recorder = build_gpu_power_recorder(log_path, {"HIP_VISIBLE_DEVICES": "4,5"})
    assert recorder is not None and recorder._gpus == {4, 5}
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "0")
    assert build_gpu_power_recorder(log_path, {}) is None
    assert build_gpu_power_recorder(None, {}) is None


class _Recorder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def note_phase(self, phase, mono):
        self.calls.append(f"phase:{phase}")
        if self.fail:
            raise RuntimeError("boom")

    def tick(self, mono):
        self.calls.append("tick")

    def close(self, *, aborted=False):
        self.calls.append("close")
        if self.fail:
            raise RuntimeError("boom")


def test_fanout_isolates_a_failing_recorder():
    broken, healthy = _Recorder(fail=True), _Recorder()
    fanout = _subprocess_kill._RecorderFanout([broken, healthy])
    fanout.note_phase("measured", 0.0)
    fanout.tick(1.0)
    fanout.close(aborted=False)
    assert healthy.calls == ["phase:measured", "tick", "close"]


def test_round_recorders_combine_kv_and_power(monkeypatch, tmp_path):
    kv, power = _Recorder(), _Recorder()
    monkeypatch.setattr(_subprocess_kill, "_build_kv_recorder", lambda *a: kv)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._gpu_power.build_gpu_power_recorder", lambda *a: power
    )
    combined = _subprocess_kill._build_round_recorders(str(tmp_path / "server.log"), {})
    combined.note_phase("measured", 0.0)
    assert kv.calls == power.calls == ["phase:measured"]
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._gpu_power.build_gpu_power_recorder", lambda *a: None)
    assert _subprocess_kill._build_round_recorders(str(tmp_path / "server.log"), {}) is kv


class _PhaseLog:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def note_phase(self, phase, mono):
        self.phases.append(phase)


def _ready_line() -> str:
    from hyperloom.orchestrator.actions.executors._subprocess_kill import _SERVER_READY_MARKERS

    return _SERVER_READY_MARKERS[0] + "\n"


def test_replay_logs_drive_boot_and_measured_per_replica(tmp_path):
    """Each replica boots its own server: a new log is a boot, its ready marker opens the measured phase."""
    from hyperloom.orchestrator.actions.executors._gpu_power import ServerLogPhaseDriver
    from hyperloom.orchestrator.actions.executors._geak_sweep import _replay_server_logs

    phases = _PhaseLog()
    from hyperloom.orchestrator.actions.executors._subprocess_kill import _scan_logs_increment

    driver = ServerLogPhaseDriver(phases, lambda: _replay_server_logs(tmp_path), scan=_scan_logs_increment)
    first = tmp_path / "replica_0" / "attempt_0" / "server.log"
    first.parent.mkdir(parents=True)
    first.write_text("loading weights\n")
    driver.poll()
    assert phases.phases == ["boot"]
    with first.open("a") as fh:
        fh.write(_ready_line())
    driver.poll()
    assert phases.phases[-1] == "measured"
    second = tmp_path / "replica_1" / "attempt_0" / "server.log"
    second.parent.mkdir(parents=True)
    second.write_text("loading weights\n")
    driver.poll()
    assert phases.phases[-1] == "boot"
    with second.open("a") as fh:
        fh.write(_ready_line())
    driver.stop()
    assert phases.phases[-1] == "measured"


def test_a_geak_replay_is_sampled_over_its_measured_phase(tmp_path, monkeypatch):
    """The replay subprocess runs under the recorder; gpu_power.json lands in its output directory."""
    import subprocess as sp

    from hyperloom.orchestrator.actions.executors import _geak_sweep, _gpu_power

    monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "1")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
    real = _gpu_power.GpuPowerRecorder

    def _recorder(**kwargs):
        rec = real(**{**kwargs, "query": lambda: _tp4_on_eight(780.0)})
        rec._interval = 0.01
        return rec

    monkeypatch.setattr(_gpu_power, "GpuPowerRecorder", _recorder)

    def _replay() -> sp.CompletedProcess:
        log_path = tmp_path / "server.log"
        log_path.write_text("loading weights\n")
        time.sleep(2.2)
        with log_path.open("a") as fh:
            fh.write(_ready_line())
        time.sleep(2.5)
        return sp.CompletedProcess(["bash"], 0, "", "")

    proc = _geak_sweep._run_with_power_sampling(_replay, tmp_path, {})
    assert proc.returncode == 0
    found, watts = read_measured_gpu_power(tmp_path)
    assert found is True
    assert watts == 780.0
