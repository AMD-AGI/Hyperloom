# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured-phase GPU power: the reading ``--max-power-w`` is graded on."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _gpu_power, _subprocess_kill
from hyperloom.orchestrator.actions.executors._gpu_power import (
    GPU_POWER_ARTIFACT_NAME,
    GpuPowerRecorder,
    _read_node_power,
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
    """Feeds ``amd-smi`` payloads in order, repeating the last one; a ``--violation`` read finds no counters."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def __call__(self, *metrics: str) -> object:
        if "--violation" in metrics:
            return {}
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


@pytest.fixture(autouse=True)
def _no_node_power_sysfs(monkeypatch):
    """No recorder here reads the host's NPM files: the node reads as one without NPM.

    A test of the node reading patches ``_read_node_power`` again, or calls the reader with its own board glob.
    """
    monkeypatch.setattr(_gpu_power, "_read_node_power", lambda: None)


def _board(root, card: int, **files: str) -> None:
    board = root / f"card{card}" / "device" / "board"
    board.mkdir(parents=True)
    for name, text in {"gpuboard_temp": "", **files}.items():
        (board / name).write_text(text)


def test_the_node_power_is_read_from_the_npm_master_cards_board(tmp_path, monkeypatch):
    """Every card has a board directory; only the cards that publish the node's files have them in it."""
    monkeypatch.setattr(_gpu_power, "_BOARD_GLOB", str(tmp_path / "card*" / "device" / "board"))
    no_boards = _read_node_power()
    _board(tmp_path, 1)
    _board(tmp_path, 57, npm_status="enabled\n", node_power="3492\n", cur_node_power_limit="2600\n")
    enabled = _read_node_power()
    (tmp_path / "card57" / "device" / "board" / "npm_status").write_text("disabled\n")
    disabled = _read_node_power()
    _board(tmp_path, 9, npm_status="enabled\n", node_power="5200\n", cur_node_power_limit="8000\n")
    behind_a_disabled_copy = _read_node_power()
    assert (no_boards, enabled, disabled) == (None, (3492.0, 2600.0), None)
    assert behind_a_disabled_copy == (5200.0, 8000.0)


def _node_summary(monkeypatch, read_node) -> dict:
    monkeypatch.setattr(_gpu_power, "_read_node_power", read_node)
    query = _Replay(_tp4_on_eight(800.0))
    recorder = GpuPowerRecorder(output_path=None, query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=4)
    return recorder.close()


def test_a_node_held_at_its_npm_limit_is_reported_though_no_card_counts_it(monkeypatch, caplog):
    """Measured on MI350X: under a 2600 W node limit every card ran near 430 W against a 1000 W cap, the node read
    above the limit, and the cards' own PPT counters moved for under 1% of the phase."""
    out = _node_summary(monkeypatch, lambda: (3492.0, 2600.0))
    assert out["power_throttled_gpus"] == []
    assert out["node"]["limit_w"] == 2600.0
    assert out["node"]["limited_fraction"] == 1.0
    assert out["node"]["avg_power_w"] == 3492.0
    assert "held at its NPM power limit (2600.0 W) for 100% of the measured phase" in caplog.text


@pytest.mark.parametrize(("power", "held"), [(2587.0, 1.0), (2573.0, 0.0)], ids=["99.5%", "99.0%-"])
def test_a_sample_counts_as_held_from_99_percent_of_the_limit(monkeypatch, caplog, power, held):
    out = _node_summary(monkeypatch, lambda: (power, 2600.0))
    warned = "NPM power limit" in caplog.text
    assert (out["node"]["limited_fraction"], warned) == (held, held > 0)


def test_a_limit_that_changed_during_the_phase_is_unread(monkeypatch):
    readings = iter([(2600.0, 2600.0), (2600.0, 8000.0)] * 8)
    out = _node_summary(monkeypatch, lambda: next(readings))
    assert out["node"]["limit_w"] is None


def test_npm_enabled_with_no_limit_is_a_failed_read(tmp_path, monkeypatch):
    monkeypatch.setattr(_gpu_power, "_BOARD_GLOB", str(tmp_path / "card*" / "device" / "board"))
    _board(tmp_path, 57, npm_status="enabled\n", node_power="900\n", cur_node_power_limit="0\n")
    out = _node_summary(monkeypatch, _read_node_power)
    assert out["node"] is None and out["node_query_errors"] > 0


def test_a_node_without_npm_has_no_node_reading(monkeypatch):
    out = _node_summary(monkeypatch, lambda: None)
    assert (out["node"], out["node_query_errors"]) == (None, 0)


def test_a_node_read_that_fails_is_counted_and_costs_the_cards_nothing(monkeypatch):
    def _unreadable():
        raise OSError("permission denied")

    out = _node_summary(monkeypatch, _unreadable)
    assert out["node"] is None and out["node_query_errors"] > 0
    assert out["avg_power_w"] == 800.0


def _summary_of(tmp_path, payload, *, gpus=None) -> dict:
    query = _Replay(payload)
    recorder = GpuPowerRecorder(
        output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=query, interval_sec=0.5, gpus=gpus
    )
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=3)
    return recorder.close()


def test_a_serving_card_with_no_power_reading_leaves_the_round_unmeasured(tmp_path):
    """TP4 on cards 4-7 with card 7 at N/A power: the other three must not stand in for the total."""
    rows = [_row(g, 500.0, 250_000) for g in (4, 5, 6)] + [_row(7, None, 250_000)]
    out = _summary_of(tmp_path, rows, gpus={4, 5, 6, 7})
    assert out["unread_gpus"] == [7]
    assert out["avg_power_w"] is None
    assert read_measured_gpu_power_by_gpu(tmp_path) == (True, None)


@pytest.mark.parametrize("bad_w", [0.0, -5.0])
def test_a_serving_card_reading_non_positive_power_leaves_the_round_unmeasured(tmp_path, bad_w):
    """A sensor reporting 0 W instead of N/A is the same failure: card 7 must not count as measured at 0 W."""
    rows = [_row(g, 500.0, 250_000) for g in (4, 5, 6)] + [_row(7, bad_w, 250_000)]
    out = _summary_of(tmp_path, rows, gpus={4, 5, 6, 7})
    assert out["unread_gpus"] == [7]
    assert out["avg_power_w"] is None
    assert read_measured_gpu_power_by_gpu(tmp_path) == (True, None)


def test_a_masked_card_missing_from_amd_smi_leaves_the_round_unmeasured(tmp_path):
    out = _summary_of(tmp_path, [_row(g, 500.0, 250_000) for g in (4, 5, 6)], gpus={4, 5, 6, 7})
    assert out["unread_gpus"] == [7]
    assert read_measured_gpu_power_by_gpu(tmp_path) == (True, None)


def test_a_card_whose_vram_cannot_be_read_cannot_be_ruled_out(tmp_path):
    rows = [_row(g, 500.0, 250_000) for g in (4, 5, 6)] + [_row(7, 500.0, 250_000)]
    rows[3]["mem_usage"] = "N/A"
    out = _summary_of(tmp_path, rows, gpus={4, 5, 6, 7})
    assert out["unread_gpus"] == [7]


def test_an_idle_card_with_no_power_reading_does_not_block(tmp_path):
    """An unpinned host: four cards serve, an idle one reports N/A power; the idle one holds no model."""
    rows = [_row(g, 800.0, 250_000) for g in range(4)] + [_row(4, None, 284)]
    out = _summary_of(tmp_path, rows)
    assert out["unread_gpus"] == []
    assert read_measured_gpu_power_by_gpu(tmp_path)[1] == {str(g): pytest.approx(800.0) for g in range(4)}


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


def test_native_power_uses_client_boundaries_and_waits_for_responses(tmp_path, monkeypatch):
    from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
    from hyperloom.orchestrator.actions.executors import _gpu_power

    config = tmp_path / "native.yaml"
    config.write_text("benchmark:\n  agentx: true\n")
    server = tmp_path / "server.log"
    server.write_text("Application startup complete\n")
    client = tmp_path / "aiperf_artifacts" / "logs" / "aiperf.log"
    client.parent.mkdir(parents=True)
    query = _Replay(_tp4_on_eight(800.0))
    monkeypatch.setattr(_gpu_power, "gpu_power_sampling_unavailable", lambda: "")
    monkeypatch.setattr(_gpu_power, "_query_amd_smi", lambda run, *metrics: query(*metrics))
    monkeypatch.setattr(_gpu_power, "_MIN_INTERVAL_SEC", 0.01)
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_INTERVAL_S", "0.01")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KV_METRICS", "0")
    recorders = _subprocess_kill._build_round_recorders(str(server), {BENCHMARK_CONFIG_ENV: str(config)})
    cursor = _subprocess_kill._LogCursor()

    def advance(line):
        with client.open("a") as stream:
            stream.write(line + "\n")
        scan = cursor.advance(str(server))
        if scan.saw_ready:
            recorders.note_phase("ready", time.monotonic())
        recorders.note_client_phases(scan, time.monotonic())

    try:
        advance("Phase warmup (warmup) started")
        time.sleep(0.03)
        assert query.calls == 0
        advance("Phase profiling (profiling) started")
        deadline = time.monotonic() + 2.0
        while query.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert query.calls >= 2
        advance("Phase profiling (profiling) sending complete")
        before = query.calls
        time.sleep(0.05)
        assert query.calls > before, "outstanding responses still belong to the measured window"
        advance("Phase profiling (profiling) complete")
        time.sleep(0.03)
        after = query.calls
        time.sleep(0.03)
        assert query.calls == after
    finally:
        recorders.close()
    assert read_measured_gpu_power_by_gpu(tmp_path) == (True, {str(g): 800.0 for g in range(4)})


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


def test_fanout_propagates_a_failing_recorder():
    broken, healthy = _Recorder(fail=True), _Recorder()
    fanout = _subprocess_kill._RecorderFanout([broken, healthy])
    with pytest.raises(RuntimeError, match="^boom$"):
        fanout.note_phase("measured", 0.0)
    assert broken.calls == ["phase:measured"]
    assert healthy.calls == []


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
    lone = _subprocess_kill._build_round_recorders(str(tmp_path / "server.log"), {})
    lone.note_phase("measured", 0.0)
    assert kv.calls == ["phase:measured", "phase:measured"]


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
    from hyperloom.orchestrator.actions.executors._subprocess_kill import _LogCursor

    driver = ServerLogPhaseDriver(phases, lambda: _replay_server_logs(tmp_path), cursor=_LogCursor())
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
        rec = real(**{**kwargs, "query": _Replay(_tp4_on_eight(780.0))})
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

    entry: dict = {}
    proc = _geak_sweep._run_with_power_sampling(_replay, tmp_path, {}, entry)
    assert proc.returncode == 0
    found, watts = read_measured_gpu_power(tmp_path)
    assert found is True
    assert watts == 780.0
    assert entry["gpu_power_avg_w"] == 780.0


def test_a_geak_replay_that_raises_records_no_power(tmp_path):
    from hyperloom.orchestrator.actions.executors import _geak_sweep

    def _replay():
        raise TimeoutError("replay timed out")

    entry: dict = {}
    with pytest.raises(TimeoutError):
        _geak_sweep._run_with_power_sampling(_replay, tmp_path, {}, entry)
    assert entry == {"gpu_power_avg_w": None, "gpu_power_by_gpu_w": None}


def _unlistable_logs() -> list[str]:
    raise PermissionError("replay output is not readable")


def test_an_unreadable_replay_log_is_warned_once_and_polling_continues(caplog):
    from hyperloom.orchestrator.actions.executors._gpu_power import ServerLogPhaseDriver

    driver = ServerLogPhaseDriver(_PhaseLog(), _unlistable_logs, cursor=_subprocess_kill._LogCursor())
    with caplog.at_level("DEBUG", logger="hyperloom.orchestrator.actions.executors._gpu_power"):
        driver._poll_logs()
        driver._poll_logs()
    levels = [record.levelname for record in caplog.records if "cannot read the replay" in record.getMessage()]
    assert levels == ["WARNING", "DEBUG"]


def test_a_phase_driver_bug_is_not_swallowed():
    from hyperloom.orchestrator.actions.executors._gpu_power import ServerLogPhaseDriver

    class _BrokenCursor(_subprocess_kill._LogCursor):
        def advance(self, server_log_path):
            raise KeyError("scan bug")

    driver = ServerLogPhaseDriver(_PhaseLog(), lambda: ["server.log"], cursor=_BrokenCursor())
    with pytest.raises(KeyError):
        driver.stop()


def test_the_replay_recorder_is_closed_even_when_the_driver_fails(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _geak_sweep

    closed: list[bool] = []

    class _Recorder:
        def close(self):
            closed.append(True)

    class _FailingDriver:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            raise KeyError("driver bug")

    monkeypatch.setattr(_geak_sweep, "build_gpu_power_recorder", lambda path, env: _Recorder())
    monkeypatch.setattr(_geak_sweep, "ServerLogPhaseDriver", _FailingDriver)
    with pytest.raises(KeyError):
        _geak_sweep._run_sampled(lambda: None, tmp_path, {})
    assert closed == [True]


# --- clock and power-throttle share over the measured phase -----------------------------------------------------------


def _clocked(gpu: int, watts: float, used_mb: float, sclk: list[float], limit: float = 2400.0) -> dict:
    """A power/mem row with the ``metric --clock`` block MI355X reports: one ``gfx_<n>`` per XCD."""
    row = _row(gpu, watts, used_mb)
    row["clock"] = {
        **{
            f"gfx_{n}": {"clk": {"value": mhz}, "min_clk": {"value": 500}, "max_clk": {"value": limit}}
            for n, mhz in enumerate(sclk)
        },
        "mem_0": {"clk": {"value": 2000}, "max_clk": {"value": 2000}},
    }
    return row


def _with_counters(power: _Replay, counters) -> Callable[..., object]:
    """One ``amd-smi`` stand-in: *counters* answers the ``--violation`` reads, *power* every other."""
    return lambda *metrics: counters() if "--violation" in metrics else power(*metrics)


def _throttle(ticks: float, ppt: float, gpus=(4, 5)) -> dict:
    return {"gpu_data": [{"gpu": g, "throttle": {"accumulation_counter": ticks, "ppt_accumulated": ppt}} for g in gpus]}


def test_parse_clock_sample_averages_the_xcds_and_reads_the_limit():
    from hyperloom.orchestrator.actions.executors._gpu_power import parse_clock_sample

    parsed = parse_clock_sample({"gpu_data": [_clocked(4, 700.0, 250_000, [1700, 1800], limit=2000)]})
    assert parsed[4] == (1750.0, frozenset({2000.0}))


def _partitioned(gpu: int, watts: float, filled: int, mhz: float = 1700.0) -> dict:
    """amd-smi's real shape: all eight ``gfx_<n>`` slots printed, only the first *filled* backed by an XCD."""
    row = _clocked(gpu, watts, 250_000, [mhz] * filled)
    na = {"clk": "N/A", "min_clk": "N/A", "max_clk": "N/A", "clk_locked": "N/A", "deep_sleep": "N/A"}
    row["clock"].update({f"gfx_{n}": dict(na) for n in range(filled, 8)})
    return row


def test_the_unfilled_xcd_slots_amd_smi_prints_leave_the_limit_read():
    from hyperloom.orchestrator.actions.executors._gpu_power import parse_clock_sample

    assert parse_clock_sample({"gpu_data": [_partitioned(4, 700.0, filled=1)]})[4] == (1700.0, frozenset({2400.0}))
    assert parse_clock_sample({"gpu_data": [_partitioned(4, 700.0, filled=0)]})[4] == (None, frozenset())


def test_a_sample_with_an_unread_xcd_does_not_unread_the_rounds_clock_limit(tmp_path):
    full = [_partitioned(g, 700.0, filled=8) for g in (4, 5)]
    one_unread = [_partitioned(g, 700.0, filled=7) for g in (4, 5)]
    none_read = [_partitioned(g, 700.0, filled=0) for g in (4, 5)]
    query = _Replay(full, one_unread, none_read, full)
    recorder = GpuPowerRecorder(output_path=None, query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=4)
    out = recorder.close()
    assert out["per_gpu"]["4"]["sclk_max_mhz"] == 2400.0


def test_a_clock_limit_that_changed_during_the_round_is_left_unread(tmp_path):
    """amd-smi copies one limit into every filled slot, so a second value means the limit itself moved mid-round."""
    steady = [_clocked(g, 700.0, 250_000, [1700.0] * 8) for g in (4, 5)]
    changed = [_clocked(4, 700.0, 250_000, [1700.0] * 8, limit=2100), _clocked(5, 700.0, 250_000, [1700.0] * 8)]
    query = _Replay(steady, changed, steady)
    recorder = GpuPowerRecorder(output_path=None, query=query, interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=3)
    per_gpu = recorder.close()["per_gpu"]
    assert (per_gpu["4"]["sclk_max_mhz"], per_gpu["5"]["sclk_max_mhz"]) == (None, 2400.0)


def test_parse_throttle_sample_needs_both_counters():
    from hyperloom.orchestrator.actions.executors._gpu_power import parse_throttle_sample

    assert parse_throttle_sample(_throttle(1000, 40, gpus=(4,))) == {4: (1000.0, 40.0)}
    assert parse_throttle_sample({"gpu_data": [{"gpu": 4, "throttle": {"accumulation_counter": "N/A"}}]}) == {}


def _throttled_round(tmp_path, *, end_ppt: float, sclk: float = 1730.0) -> dict:
    rows = [_clocked(g, 700.0, 250_000, [sclk] * 8) for g in (4, 5)] + [_clocked(6, 225.0, 284, [38.0] * 8)]
    query = _Replay(rows)
    counters = _Replay(_throttle(10_000, 1_000), _throttle(20_000, end_ppt))
    recorder = GpuPowerRecorder(
        output_path=str(tmp_path / GPU_POWER_ARTIFACT_NAME), query=_with_counters(query, counters), interval_sec=0.5
    )
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=3)
    return recorder.close()


def test_a_power_throttled_round_names_the_cards_and_their_clock(tmp_path, caplog):
    """30% of the measured ticks power limited: the round names the cards, their share and the clock they held."""
    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.actions.executors._gpu_power"):
        out = _throttled_round(tmp_path, end_ppt=4_000)
    assert out["per_gpu"]["4"]["power_throttle_fraction"] == pytest.approx(0.3)
    assert out["per_gpu"]["4"]["sclk_mean_mhz"] == pytest.approx(1730.0)
    assert out["per_gpu"]["4"]["sclk_max_mhz"] == 2400.0
    assert out["power_throttled_gpus"] == [4, 5]
    assert "suggested_sclk_max_mhz" not in out
    assert (
        "GPU(s) 4, 5 were power throttled for 30%, 30% of the measured phase, at a mean clock of 1730.0 MHz"
        in caplog.text
    )
    written = json.loads((tmp_path / GPU_POWER_ARTIFACT_NAME).read_text())
    assert written["power_throttled_gpus"] == [4, 5]


def test_a_throttled_card_whose_clock_was_never_read_awake_is_named_unread(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.actions.executors._gpu_power"):
        out = _throttled_round(tmp_path, end_ppt=4_000, sclk=38.0)
    assert out["per_gpu"]["4"]["sclk_mean_mhz"] is None
    assert "at a mean clock of unread, unread" in caplog.text


def test_an_unthrottled_round_reports_no_throttling(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.actions.executors._gpu_power"):
        out = _throttled_round(tmp_path, end_ppt=1_100)
    assert out["per_gpu"]["4"]["power_throttle_fraction"] == pytest.approx(0.01)
    assert out["power_throttled_gpus"] == []
    assert "power throttled" not in caplog.text


def test_a_failed_throttle_read_leaves_the_power_reading_and_its_error_count_alone(tmp_path):
    """An amd-smi without ``--violation`` must not make a round whose power was read look like a failed power read."""

    def _no_violation():
        raise subprocess.CalledProcessError(2, ["amd-smi", "metric", "--violation"])

    query = _Replay([_clocked(g, 700.0, 250_000, [1730.0] * 8) for g in (4, 5)])
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(query, _no_violation), interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=3)
    out = recorder.close()
    assert out["avg_power_w"] == pytest.approx(700.0)
    assert (out["query_errors"], out["last_error"]) == (0, None)
    assert out["throttle_query_errors"] >= 1 and "--violation" in out["throttle_last_error"]
    assert out["per_gpu"]["4"]["power_throttle_fraction"] is None
    assert out["power_throttled_gpus"] == []


def _two_windows(*reads: object) -> dict:
    """A recorder taken through two measured windows whose four counter reads are *reads*, in order."""

    def _next():
        read = next(sequence)
        if isinstance(read, Exception):
            raise read
        return read

    sequence = iter(reads)
    rows = [_clocked(g, 700.0, 250_000, [1000.0] * 8) for g in (4, 5)]
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(_Replay(rows), _next))
    recorder._phase = "measured"
    recorder._sample_once()
    for _ in range(2):
        recorder._window_open = recorder._read_throttle()
        recorder._close_window()
    return recorder.summary()


def test_a_failed_counter_read_in_any_window_leaves_every_cards_share_unread():
    """Summing only the windows that were read would publish an unthrottled share for a throttled round."""
    read = _two_windows(_throttle(1_000, 0), _throttle(1_100, 0), _throttle(1_100, 0), _throttle(11_100, 9_900))
    assert read["per_gpu"]["4"]["power_throttle_fraction"] == pytest.approx(9_900 / 10_100, abs=1e-4)
    out = _two_windows(
        _throttle(1_000, 0), _throttle(1_100, 0), RuntimeError("amd-smi exited 1"), _throttle(11_100, 9_900)
    )
    assert out["throttle_query_errors"] == 1
    assert out["per_gpu"]["4"]["power_throttle_fraction"] is None
    assert out["power_throttled_gpus"] == []
    unread = RuntimeError("amd-smi exited 1")
    both_ends = _two_windows(_throttle(1_000, 0), _throttle(1_100, 0), unread, unread)
    assert both_ends["per_gpu"]["4"]["power_throttle_fraction"] is None, "a window unread at both ends still counts"


def test_a_card_unread_at_both_ends_of_a_window_is_left_unread():
    """amd-smi prints a card's whole throttle block as N/A when it cannot read it; that window is still unread."""
    out = _two_windows(
        _throttle(1_000, 0),
        _throttle(2_000, 900),
        _throttle(2_000, 900, gpus=(5,)),
        _throttle(100_000, 1_000, gpus=(5,)),
    )
    assert out["per_gpu"]["4"]["power_throttle_fraction"] is None, "not the first window's 90% alone"
    assert out["per_gpu"]["5"]["power_throttle_fraction"] == pytest.approx(1_000 / 99_000, abs=1e-4)
    assert out["power_throttled_gpus"] == []


def test_a_sampler_still_blocked_in_amd_smi_at_close_leaves_every_share_unread():
    reads = iter([_throttle(1_000, 0), _throttle(2_000, 900)])
    rows = [_clocked(g, 700.0, 250_000, [1000.0] * 8) for g in (4, 5)]
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(_Replay(rows), lambda: next(reads)))
    recorder._phase = "measured"
    recorder._sample_once()
    recorder._window_open = recorder._read_throttle()
    recorder._close_window()
    assert recorder.summary()["per_gpu"]["4"]["power_throttle_fraction"] == pytest.approx(0.9)
    recorder._thread = SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: True)
    out = recorder.close()
    assert out["per_gpu"]["4"]["power_throttle_fraction"] is None


def test_a_card_unread_at_one_end_of_a_window_or_whose_counters_went_back_is_left_unread():
    from hyperloom.orchestrator.actions.executors import _gpu_power

    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(_Replay({}), dict))
    recorder._window_open = {4: (1_000.0, 0.0), 5: (1_000.0, 0.0), 6: (5_000.0, 100.0), 7: (5_000.0, 100.0)}
    closed = {4: (2_000.0, 300.0), 6: (500.0, 0.0), 7: (6_000.0, 50.0)}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_gpu_power, "parse_throttle_sample", lambda payload: closed)
        recorder._close_window()
    assert recorder._complete_throttle_totals() == {4: (1_000.0, 300.0)}


def test_throttle_counters_sum_over_every_measured_window(tmp_path):
    """A GEAK replay re-enters boot per replica; the windows between are not the measured load and must not count."""
    reads = iter([_throttle(1_000, 0), _throttle(2_000, 500), _throttle(10_000, 500), _throttle(11_000, 1_000)])
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(_Replay({}), lambda: next(reads)))
    for _ in range(2):
        recorder._window_open = recorder._read_throttle()
        recorder._close_window()
    assert recorder._throttle_totals[4] == [2_000.0, 1_000.0, 2], "8,000 ticks of boot between the windows are excluded"


def test_deep_sleep_clocks_do_not_pull_the_sustained_clock_down():
    """Idle stretches inside the measured phase read ~38 MHz; averaging them in would understate the loaded clock."""
    from hyperloom.orchestrator.actions.executors._gpu_power import parse_clock_sample

    asleep = parse_clock_sample({"gpu_data": [_clocked(4, 230.0, 250_000, [38.0] * 8)]})
    assert asleep[4] == (None, frozenset({2400.0}))
    mixed = parse_clock_sample({"gpu_data": [_clocked(4, 900.0, 250_000, [1700.0] * 6 + [38.0] * 2)]})
    assert mixed[4][0] == pytest.approx(1700.0)


def test_a_round_with_idle_gaps_reports_the_loaded_clock(tmp_path):
    busy = [_clocked(g, 1390.0, 250_000, [1700.0] * 8) for g in (4, 5)]
    idle = [_clocked(g, 230.0, 250_000, [38.0] * 8) for g in (4, 5)]
    query = _Replay(busy, idle, busy, idle, busy)
    counters = _Replay(_throttle(10_000, 1_000), _throttle(20_000, 4_000))
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(query, counters), interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=5)
    out = recorder.close()
    assert out["per_gpu"]["4"]["sclk_mean_mhz"] == pytest.approx(1700.0)


def test_the_sampler_closes_a_window_when_the_round_leaves_the_measured_phase(tmp_path):
    """The window is closed by the sampler thread at the phase change, not only by close()."""
    query = _Replay([_clocked(g, 700.0, 250_000, [1730.0] * 8) for g in (4, 5)])
    counters = _Replay(_throttle(10_000, 1_000), _throttle(20_000, 4_000))
    recorder = GpuPowerRecorder(output_path=None, query=_with_counters(query, counters), interval_sec=0.5)
    recorder._interval = 0.01
    _run_measured(recorder, query, samples=2)
    recorder.note_phase("eval", time.monotonic())
    deadline = time.monotonic() + 5.0
    while counters.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder._window_open is None and recorder._throttle_totals[4] == [10_000.0, 3_000.0, 1]
    recorder.close()
