# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The power recorder against recorded MI355X ``amd-smi`` output and a sysfs tree built from the same node's files."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import pytest

from hyperloom_common import gpu_telemetry
from hyperloom_common.gpu_telemetry import (
    PowerRecorder,
    RecorderScope,
    RecorderSources,
    RecorderTiming,
    TokenCounts,
    gpu_monitor,
    publish_power,
)
from hyperloom_common.power_sources import FAST_COLUMNS, SysfsNode

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "mi355x_power.json").read_text())
_RECORDED = len(_FIXTURE["static"]["gpu_data"])


def _rows(section: str, cards: int) -> dict[str, Any]:
    """The recorded node's ``section`` for ``cards`` cards; past the four recorded, each repeats one under a new
    index and PCI address."""
    rows = []
    for gpu in range(cards):
        row = copy.deepcopy(_FIXTURE[section]["gpu_data"][gpu % _RECORDED])
        row["gpu"] = gpu
        if "bus" in row:
            row["bus"]["bdf"] = _bdf(gpu)
        rows.append(row)
    return {"gpu_data": rows}


def _bdf(gpu: int) -> str:
    return _FIXTURE["static"]["gpu_data"][gpu]["bus"]["bdf"] if gpu < _RECORDED else f"0003:00:0{gpu - 3}.0"


_VRAM_TOTAL = 309_220_868_096
_SERVING_VRAM = 280_000_000_000
_COUNTS = TokenCounts(successful_queries=50, input_tokens=20_000, output_tokens=5_000)
_FAST = RecorderTiming(fast_sec=0.02, full_sec=0.05, write_sec=3600.0)


@pytest.fixture(autouse=True)
def _fast_sampling(monkeypatch):
    monkeypatch.setattr(gpu_telemetry, "MIN_INTERVAL_SEC", 0.01)


def _write(path: Path, text: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n")


def _card(root: Path, index: int, bdf: str, hwmon: dict[str, Any], vram_used: int) -> None:
    device = root / "bus" / "pci" / "devices" / bdf
    mon = device / "hwmon" / f"hwmon{8 + index}"
    channels = {
        "power1": ("PPT", int(hwmon["power_w"] * 1e6)),
        "temp2": ("junction", 61_000),
        "temp3": ("mem", 42_000),
        "freq1": ("sclk", 2_100_000_000),
        "freq2": ("mclk", 2_000_000_000),
        "in2": ("vddboard", int(hwmon["vddboard_mv"])),
    }
    for name, (label, value) in channels.items():
        _write(mon / f"{name}_label", label)
        _write(mon / f"{name}_input", value)
    for name, key in (("cap", "power_cap_w"), ("cap_default", "power_cap_default_w"), ("cap_max", "power_cap_max_w")):
        _write(mon / f"power1_{name}", int(hwmon[key] * 1e6))
    _write(mon / "power1_cap_min", 0)
    files = {
        "gpu_busy_percent": 97,
        "mem_busy_percent": 31,
        "mem_info_vram_used": vram_used,
        "mem_info_vram_total": _VRAM_TOTAL,
        "mem_info_gtt_used": 25_055_232,
        "power_dpm_force_performance_level": "auto",
        "pp_dpm_sclk": "S: 94Mhz *\n0: 500Mhz\n1: 2400Mhz",
    }
    for name, value in files.items():
        _write(device / name, value)


def _sysfs(
    root: Path,
    *,
    cards: int = _RECORDED,
    serving: tuple[int, ...] = (0, 1, 2, 3),
    npm: str | None = None,
    rapl: bool = False,
) -> Path:
    """A sysfs tree with ``cards`` cards, each reading as its recorded card did."""
    for gpu in range(cards):
        hwmon = _FIXTURE["hwmon"][_bdf(gpu % _RECORDED)]
        _card(root, gpu, _bdf(gpu), hwmon, _SERVING_VRAM if gpu in serving else 297_766_912)
    if npm is not None:
        board = root / "class" / "drm" / "card0" / "device" / "board"
        for name, value in (("npm_status", npm), ("node_power", 3_100), ("cur_node_power_limit", 3_200)):
            _write(board / name, value)
    if rapl:
        zone = root / "class" / "powercap" / "intel-rapl:0"
        for name, value in (("name", "package-0"), ("energy_uj", 5_000_000), ("max_energy_range_uj", 2**32)):
            _write(zone / name, value)
        _write(zone.with_name("intel-rapl:0:0") / "energy_uj", 1_000)
    return root


class _AmdSmi:
    """Answers each ``amd-smi`` read with the recorded node's output; the energy accumulator advances 255 W."""

    def __init__(self, failing: tuple[str, ...] = (), cards: int = _RECORDED) -> None:
        self.failing = failing
        self.cards = cards
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> Any:
        self.calls.append(args)
        if args[0] in self.failing or " ".join(args) in self.failing:
            raise RuntimeError(f"amd-smi {' '.join(args)} exited 1")
        if args == ("static",):
            return _rows("static", self.cards)
        if args == ("version",):
            return [{"tool": "AMDSMI Tool", "version": "27.0.0+6b0e43f3"}]
        if args == ("metric", "-E"):
            energy = _rows("energy", self.cards)
            for row in energy["gpu_data"]:
                row["energy"]["total_energy_consumption"]["value"] += 255.0 * time.monotonic()
            return energy
        metric = _rows("metric", self.cards)
        ticks = int(1e6 * time.monotonic())
        for row in metric["gpu_data"]:
            row["throttle"]["accumulation_counter"] += ticks
            row["throttle"]["total_gfx_clk_below_host_limit_accumulated"]["xcp_0"][0] += ticks // 4
        return metric


def _recorder(tmp_path: Path, query: _AmdSmi | None = None, **sysfs: Any) -> PowerRecorder:
    sources = RecorderSources(query=query or _AmdSmi(), sysfs=SysfsNode(_sysfs(tmp_path / "sys", **sysfs)))
    scope = RecorderScope(gpus=(0, 1, 2, 3))
    return PowerRecorder(output_path=str(tmp_path / "power.json"), scope=scope, timing=_FAST, sources=sources)


def _round(recorder: PowerRecorder, *, measured_sec: float = 0.4) -> float:
    """Drive a round through its phases; the measured mark's instant, given ahead of when it is heard."""
    recorder.start()
    recorder.note_phase("boot", time.monotonic())
    time.sleep(0.1)
    measured = time.monotonic()
    time.sleep(0.05)
    recorder.note_phase("measured", measured)
    time.sleep(measured_sec)
    recorder.note_phase("finished", time.monotonic())
    time.sleep(0.1)
    return measured


def test_one_file_holds_static_phases_energy_samples_and_errors(tmp_path):
    recorder = _recorder(tmp_path)
    _round(recorder)
    document = recorder.close().document(_COUNTS)

    assert document["schema_version"] == 2
    assert set(document) == {
        "schema_version", "host", "started_unix", "final", "aborted", "recorder", "scope", "static",
        "phase_marks", "phases", "energy", "samples", "errors",
    }  # fmt: skip
    gpu0 = document["static"]["start"]["gpus"]["0"]
    assert gpu0["bdf"] == "0002:00:01.0" and gpu0["caps_w"] == {
        "cap": 1400.0,
        "default": 1400.0,
        "min": 0.0,
        "max": 1400.0,
    }
    assert gpu0["amd_smi_static"]["limit"]["ppt0"]["socket_power_limit"]["value"] == 1400
    assert gpu0["perf_level"] == "auto" and gpu0["pp_dpm"]["pp_dpm_sclk"].startswith("S: 94Mhz")
    assert set(document["static"]) == {"start", "end"}
    assert document["recorder"]["amd_smi_version"] == "27.0.0+6b0e43f3"
    assert [mark["phase"] for mark in document["phase_marks"]] == ["boot", "measured", "finished"]
    assert set(document["phases"]) == {"boot", "measured", "finished"}

    measured = document["phases"]["measured"]["per_gpu"]["1"]
    assert measured["power_w"]["avg"] == 249.0 and measured["junction_c"] == {"avg": 61.0, "max": 61.0}
    assert measured["sclk_mhz"]["max"] == 2100.0 and measured["vram_peak_frac"] == pytest.approx(0.9055, abs=1e-4)
    assert measured["throttle"]["ppt"] == 0.0
    assert measured["throttle"]["total_gfx_clk_below_host_limit.xcp_0.0"] == pytest.approx(0.25, abs=0.01)

    energy = document["energy"]
    assert energy["power_valid"] == 1, energy["reasons"]
    assert energy["reasons"] == [] and energy["warnings"] == []
    # The fast loop's samples land a scheduler tick off the marks on a loaded runner; the boundary interpolation
    # then trims a sliver of the window, so the integral is held to 1% of power times duration, not exactly.
    assert energy["total_gpu_energy_j"] == pytest.approx(
        (256 + 249 + 255 + 254) * energy["window"]["duration_s"], rel=1e-2
    )
    assert energy["joules_per_output_token"] == pytest.approx(energy["total_gpu_energy_j"] / 5_000, abs=1e-6)
    assert energy["accumulator_check"]["within_tolerance"] is True

    samples = document["samples"]
    assert samples["fast"]["columns"] == ["mono", "unix", "gpu", *FAST_COLUMNS]
    assert samples["energy"]["columns"] == ["mono", "unix", "gpu", "energy_j", "read_s"]
    full = samples["full"]
    assert full["columns"][:3] == ["mono", "unix", "gpu"] and "throttle.ppt_accumulated" in full["columns"]
    assert full["units"]["power.socket_power"] == "W" and len(full["columns"]) > 250
    assert document["errors"] == {}


def test_a_card_amd_smi_reads_as_na_is_measured_from_sysfs(tmp_path):
    recorder = _recorder(tmp_path)
    _round(recorder)
    document = recorder.close().document(_COUNTS)
    full = document["samples"]["full"]
    socket = full["columns"].index("power.socket_power")
    assert {row[2]: row[socket] for row in full["rows"]}[2] is None
    assert document["phases"]["measured"]["per_gpu"]["2"]["power_w"]["avg"] == 255.0
    assert document["energy"]["per_gpu"]["2"]["integral_j"] > 0
    assert document["scope"]["unread_gpus"] == []


def test_the_window_opens_at_the_marks_instant_not_when_it_was_heard(tmp_path):
    recorder = _recorder(tmp_path)
    measured = _round(recorder)
    recording = recorder.close()
    window = recording.window()
    assert window is not None and window.intervals[0][0] == measured
    assert recording.document(_COUNTS)["energy"]["window"]["start_unix"] == round(measured + recording.unix_offset, 3)


def test_the_node_and_cpu_blocks_are_read_when_the_node_has_them(tmp_path):
    recorder = _recorder(tmp_path, npm="enabled", rapl=True)
    _round(recorder)
    document = recorder.close().document(_COUNTS)
    sources = document["recorder"]["sources"]
    assert sources["node"] == {"source": "sysfs_npm", "status": "enabled", "reason": None}
    assert sources["cpu"] == {"source": "rapl", "reason": None}
    assert document["static"]["start"]["npm"]["limit_w"] == 3200.0
    assert document["phases"]["measured"]["node"]["avg_w"] == 3100.0
    assert document["energy"]["node_energy_j"] == pytest.approx(
        3100.0 * document["energy"]["window"]["duration_s"], rel=1e-2
    )
    assert document["energy"]["cpu_energy_j"] == 0.0
    domains = document["static"]["start"]["cpu"]["domains"]
    assert [(domain["name"], domain["counted"]) for domain in domains] == [
        ("rapl:intel-rapl:0:package-0", True),
        ("rapl:intel-rapl:0:0:", False),
    ]


def test_absent_node_and_cpu_sources_are_null_with_a_reason(tmp_path):
    recorder = _recorder(tmp_path)
    _round(recorder)
    document = recorder.close().document(_COUNTS)
    sources = document["recorder"]["sources"]
    assert sources["node"]["source"] is None and sources["node"]["reason"]
    assert sources["cpu"]["source"] is None and sources["cpu"]["reason"]
    assert document["energy"]["node_energy_j"] is None and document["energy"]["cpu_energy_j"] is None
    assert document["phases"]["measured"]["node"] is None and document["phases"]["measured"]["cpu"] is None


def test_npm_reporting_disabled_reads_no_node_power(tmp_path):
    recorder = _recorder(tmp_path, npm="disabled")
    _round(recorder, measured_sec=0.1)
    document = recorder.close().document(_COUNTS)
    assert document["recorder"]["sources"]["node"]["reason"] == "npm_status is disabled: firmware reports no node power"


def test_a_configured_card_without_a_model_invalidates_the_energy(tmp_path):
    recorder = _recorder(tmp_path, serving=(0, 1))
    _round(recorder, measured_sec=0.1)
    document = recorder.close().document(_COUNTS)
    assert document["scope"]["serving_gpus"] == [0, 1]
    assert document["energy"]["reasons"] == ["expected_gpu_count_mismatch"]


def _scoped_recorder(tmp_path: Path, scope: tuple[int, ...], serving: tuple[int, ...]) -> PowerRecorder:
    sysfs = SysfsNode(_sysfs(tmp_path / "sys", cards=8, serving=serving))
    return PowerRecorder(
        output_path=None,
        scope=RecorderScope(gpus=scope),
        timing=_FAST,
        sources=RecorderSources(query=_AmdSmi(cards=8), sysfs=sysfs),
    )


def test_another_server_on_cards_outside_the_scope_leaves_the_energy_valid(tmp_path):
    recorder = _scoped_recorder(tmp_path, scope=(0, 1, 2, 3), serving=tuple(range(8)))
    _round(recorder, measured_sec=0.1)
    document = recorder.close().document(_COUNTS)
    assert document["scope"]["gpus_sampled"] == list(range(8))
    assert document["scope"]["serving_gpus"] == [0, 1, 2, 3]
    assert document["energy"]["power_valid"] == 1, document["energy"]["reasons"]


def test_a_scope_card_without_a_model_is_flagged_whatever_serves_outside_the_scope(tmp_path):
    recorder = _scoped_recorder(tmp_path, scope=(0, 1, 2, 3), serving=(1, 2, 3, 4))
    _round(recorder, measured_sec=0.1)
    document = recorder.close().document(_COUNTS)
    assert document["scope"]["serving_gpus"] == [1, 2, 3]
    assert document["energy"]["reasons"] == ["expected_gpu_count_mismatch"]


def test_failed_reads_are_recorded_and_never_raise(tmp_path):
    query = _AmdSmi(failing=("static", "metric", "version"))
    recorder = PowerRecorder(
        output_path=None,
        scope=RecorderScope(gpus=(0,)),
        timing=_FAST,
        sources=RecorderSources(query=query, sysfs=SysfsNode(tmp_path / "empty")),
    )
    _round(recorder, measured_sec=0.1)
    document = recorder.close(aborted=True).document(None)
    assert set(document["errors"]) == {"amd_smi_static", "amd_smi_version", "amd_smi_metric", "amd_smi_energy"}
    assert document["errors"]["amd_smi_metric"]["last"] == "amd-smi metric exited 1"
    assert document["energy"]["power_valid"] == 0
    assert {"aborted", "expected_gpu_count_mismatch"} <= set(document["energy"]["reasons"])
    assert document["aborted"] is True


def test_a_running_round_is_written_before_close(tmp_path):
    recorder = PowerRecorder(
        output_path=str(tmp_path / "power.json"),
        scope=RecorderScope(gpus=(0, 1, 2, 3)),
        timing=RecorderTiming(fast_sec=0.02, full_sec=0.02, write_sec=0.0),
        sources=RecorderSources(query=_AmdSmi(), sysfs=SysfsNode(_sysfs(tmp_path / "sys"))),
    )
    recorder.start()
    deadline, written = time.monotonic() + 10.0, {}
    while not written.get("samples", {}).get("fast", {}).get("rows") and time.monotonic() < deadline:
        time.sleep(0.01)
        if (tmp_path / "power.json").exists():
            written = json.loads((tmp_path / "power.json").read_text())
    recorder.close()
    assert written["final"] is False and written["samples"]["fast"]["rows"]


def test_close_is_idempotent_and_an_unknown_phase_is_ignored(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.note_phase("profile_open", time.monotonic())
    first = recorder.close()
    assert recorder.close() is first and first.marks == []
    recorder.note_phase("measured", time.monotonic())
    assert recorder.close().marks == []


def test_the_gpu_monitor_blocks_are_the_energy_cards_over_the_measured_phase(tmp_path):
    recorder = _recorder(tmp_path)
    _round(recorder)
    blocks = gpu_monitor(recorder.close())
    assert [block.gpu for block in blocks] == [0, 1, 2, 3]
    assert blocks[2].power_watts.avg == 255.0 and blocks[2].vram_pct.max == pytest.approx(90.55, abs=0.01)
    assert set(blocks[0].as_dict()) == {
        "gpu", "sample_count", "power_watts", "temperature_c", "gpu_clock_mhz", "mem_clock_mhz", "gpu_util_pct",
        "vram_pct",
    }  # fmt: skip


def test_publish_logs_a_power_throttled_card(tmp_path, caplog):
    document = {
        "scope": {"energy_gpus": [0]},
        "phases": {"measured": {"per_gpu": {"0": {"throttle": {"ppt": 0.4}}}, "node": {"limited_fraction": 0.5}}},
    }
    publish_power(tmp_path / "power.json", document)
    assert "power throttled for 40%" in caplog.text and "NPM power limit for 50%" in caplog.text
    assert json.loads((tmp_path / "power.json").read_text()) == document
