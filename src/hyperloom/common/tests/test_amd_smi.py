# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared amd-smi probes.

The payload fixtures are trimmed captures of real output from two amd-smi majors on one gfx1151: tool 26.5.0
(ROCm 7.15) and tool 27.1.0 (ROCm 10.2). Hyperloom supports ROCm 7.2.x and 10.0 at once, so every probe is asserted
against both shapes rather than against one release's spelling.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from hyperloom.common import amd_smi
from hyperloom.common.amd_smi import (
    AmdSmiError,
    GpuVram,
    amd_smi_json,
    gpu_asic_info,
    gpu_count,
    gpu_status,
    gpu_vram_usage,
    lookup,
    scalar,
    vram_busy,
)


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _binary_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_smi.shutil, "which", lambda _n: "/opt/rocm/bin/amd-smi")


def _stub_output(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: _FakeProc(0, text))


def _mem(total: int = 2048, used: int = 179) -> dict[str, object]:
    """``metric --mem-usage``: VRAM alongside the visible-VRAM and GTT aliases a loose matcher would catch."""
    return {
        "total_vram": {"value": total, "unit": "MB"},
        "used_vram": {"value": used, "unit": "MB"},
        "free_vram": {"value": total - used, "unit": "MB"},
        "total_visible_vram": {"value": total, "unit": "MB"},
        "used_visible_vram": {"value": used, "unit": "MB"},
        "total_gtt": {"value": 49152, "unit": "MB"},
        "used_gtt": {"value": 17, "unit": "MB"},
    }


#: ``metric --usage --mem-usage --temperature --power`` as amd-smi 26.5.0 prints it: no ``hotspot``/``mem``
#: temperature keys and no ``gfx_voltage``-family power keys at all.
_METRIC_26 = {
    "gpu_data": [
        {
            "gpu": 0,
            "usage": {
                "gfx_activity": {"value": 80, "unit": "%"},
                "apu_average_gfx_activity": {"value": 0, "unit": "%"},
            },
            "power": {
                "socket_power": {"value": 300, "unit": "W"},
                "apu_average_socket_power": {"value": 11, "unit": "W"},
            },
            "temperature": {"edge": {"value": 45, "unit": "C"}},
            "mem_usage": _mem(total=2000, used=1000),
        }
    ]
}

#: The same query as amd-smi 27.1.0 prints it: the extra keys are present but ``"N/A"`` on this part.
_METRIC_27 = {
    "gpu_data": [
        {
            "gpu": 0,
            "usage": {"gfx_activity": {"value": 80, "unit": "%"}, "umc_activity": "N/A", "mm_activity": "N/A"},
            "power": {
                "socket_power": {"value": 300, "unit": "W"},
                "gfx_voltage": "N/A",
                "throttle_status": "N/A",
                "ubb_power": "N/A",
            },
            "temperature": {"edge": {"value": 45, "unit": "C"}, "hotspot": "N/A", "mem": "N/A"},
            "mem_usage": _mem(total=2000, used=1000),
        }
    ]
}


# -- amd_smi_json -----------------------------------------------------------


def test_amd_smi_json_appends_the_json_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller names the subcommand; the helper owns ``amd-smi`` and ``--json``."""
    seen: list[list[str]] = []

    def _run(cmd, **_kwargs):
        seen.append(cmd)
        return _FakeProc(0, "[]")

    monkeypatch.setattr(amd_smi.subprocess, "run", _run)
    assert amd_smi_json(["static", "--asic"]) == []
    assert seen == [["amd-smi", "static", "--asic", "--json"]]


def test_amd_smi_json_raises_on_every_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing binary, exec error, non-zero exit and non-JSON stdout all raise one error type."""
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("amd-smi")))
    with pytest.raises(AmdSmiError):
        amd_smi_json(["list"])

    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    with pytest.raises(AmdSmiError):
        amd_smi_json(["list"])

    monkeypatch.setattr(
        amd_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("amd-smi", 1))
    )
    with pytest.raises(AmdSmiError):
        amd_smi_json(["list"])

    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: _FakeProc(1, "", "boom"))
    with pytest.raises(AmdSmiError):
        amd_smi_json(["list"])

    # The no-driver case: amd-smi logs to stderr, prints nothing on stdout, and still exits 0.
    _stub_output(monkeypatch, "")
    with pytest.raises(AmdSmiError):
        amd_smi_json(["list"])


# -- scalar / lookup --------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"value": 12, "unit": "W"}, 12.0),  # unit-wrapped, as from ROCm 6.1 onwards
        (0, 0.0),  # bare scalar: amd-smi 26.5.0 still prints apu_average_ipu_reads this way
        (11.75, 11.75),
        ("N/A", None),
        (None, None),
        ({"value": "N/A"}, None),
        (True, None),  # a bool is not a reading
        ({}, None),
    ],
)
def test_scalar_reads_both_wrapped_and_bare_values(block: object, expected: float | None) -> None:
    """A metric may be a number or a ``{value, unit}`` object, and ``"N/A"`` means not reported."""
    assert scalar(block) == expected


def test_lookup_skips_absent_and_not_available_keys() -> None:
    """The first key present with a real value wins; ``"N/A"`` counts as absent."""
    assert lookup({"hotspot": "N/A", "edge": 45}, "hotspot", "edge") == 45
    assert lookup({"edge": 45}, "hotspot", "edge") == 45
    assert lookup({"hotspot": "N/A"}, "hotspot", "edge") is None
    assert lookup({"edge": ""}, "edge") is None
    assert lookup("N/A", "edge") is None


def test_lookup_skips_a_non_string_key_rather_than_raising() -> None:
    """An unhashable name is skipped: subscripting with one would raise TypeError, not simply fail to match."""
    assert lookup({"edge": 45}, {"unhashable": 1}, "edge") == 45
    assert lookup({"edge": 45}, ["unhashable"]) is None


# -- gpu_vram_usage ---------------------------------------------------------


def test_vram_usage_reads_vram_not_gtt_or_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``used_vram``/``total_vram`` count: GTT is 24x VRAM on an APU, so a loose match is silently wrong."""
    _stub_output(monkeypatch, {"gpu_data": [{"gpu": 0, "mem_usage": _mem()}, {"gpu": 1, "mem_usage": _mem(used=250)}]})
    assert gpu_vram_usage() == [GpuVram(179.0, 2048.0), GpuVram(250.0, 2048.0)]


def test_vram_usage_converts_whatever_unit_is_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GiB-labelled total and a byte-labelled used figure both land in MiB."""
    _stub_output(
        monkeypatch,
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "mem_usage": {
                        "total_vram": {"value": 288, "unit": "GB"},
                        "used_vram": {"value": 100 * 1024**2, "unit": "B"},
                    },
                }
            ]
        },
    )
    assert gpu_vram_usage() == [GpuVram(100.0, 288 * 1024.0)]


def test_vram_usage_none_when_a_card_omits_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Used without total cannot answer a ratio gate, so the probe is unknown rather than partial."""
    _stub_output(monkeypatch, {"gpu_data": [{"gpu": 0, "mem_usage": {"used_vram": {"value": 512, "unit": "MB"}}}]})
    assert gpu_vram_usage() is None


def test_vram_usage_none_when_total_is_not_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero total would make the ratio undefined."""
    _stub_output(
        monkeypatch,
        {"gpu_data": [{"gpu": 0, "mem_usage": {"total_vram": {"value": 0}, "used_vram": {"value": 0}}}]},
    )
    assert gpu_vram_usage() is None


def test_vram_usage_none_on_unusable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing binary, non-zero exit, bad JSON and an empty device list all degrade to None."""
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert gpu_vram_usage() is None

    _stub_output(monkeypatch, "{not json")
    assert gpu_vram_usage() is None

    _stub_output(monkeypatch, {"gpu_data": []})
    assert gpu_vram_usage() is None


def test_vram_busy_is_a_fraction_of_total() -> None:
    """A card holding more than ``VRAM_BUSY_FRACTION`` of its VRAM is busy."""
    assert vram_busy(GpuVram(100.0, 2048.0)) is True
    assert vram_busy(GpuVram(1.0, 2048.0)) is False


# -- gpu_count --------------------------------------------------------------


def test_gpu_count_is_the_length_of_the_device_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """``amd-smi list`` gives one row per device, so no dedup of repeated ``GPU[n]`` lines is needed."""
    _stub_output(
        monkeypatch,
        [
            {"gpu": 0, "bdf": "0000:c5:00.0", "uuid": "00001586-0000-1000-8000-000000000000"},
            {"gpu": 1, "bdf": "0000:c6:00.0", "uuid": "00001587-0000-1000-8000-000000000000"},
        ],
    )
    assert gpu_count() == 2


def test_gpu_count_zero_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No binary, a failed probe, or no devices all read as zero."""
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("amd-smi")))
    assert gpu_count() == 0

    _stub_output(monkeypatch, "[]")
    assert gpu_count() == 0


def test_gpu_count_on_a_cpx_partitioned_card_is_unverified_against_real_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UNVERIFIED ASSUMPTION, not an observed fact (rocm-smi-to-amd-smi plan, risk R5).

    The plan records that on an MI300 in CPX (compute-partitioned) mode, ``amd-smi list`` is expected to enumerate
    one row per *partition* rather than one row per physical card -- which would make ``gpu_count()`` (and every
    caller that treats it as a physical-card count) report up to 8x the real card count. No phase, including this
    one, has run this against actual CPX-partitioned hardware: every host available during this migration is a
    non-partitioned APU. The payload below is HAND-WRITTEN to encode that expectation, not a capture -- it documents
    today's ``len(gpu_rows(...))`` behavior given such a payload, so a future change to ``gpu_count()`` that claims
    to fix CPX-awareness has a test to update, not a regression to discover blind. It must not be read as evidence the
    inflation (or this row shape) has been seen on real silicon.
    """
    one_physical_card_in_cpx_mode = [
        {"gpu": i, "bdf": "0000:c5:00.0", "uuid": "00001586-0000-1000-8000-000000000000"} for i in range(8)
    ]
    _stub_output(monkeypatch, one_physical_card_in_cpx_mode)
    assert gpu_count() == 8  # documents today's inflation given this shape; not validated on hardware


# -- gpu_status -------------------------------------------------------------


@pytest.mark.parametrize("payload", [_METRIC_26, _METRIC_27], ids=["amd-smi-26.5.0", "amd-smi-27.1.0"])
def test_gpu_status_reads_both_amd_smi_majors(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """The same readings come out of either release, whose power and temperature blocks differ in which keys exist."""
    _stub_output(monkeypatch, payload)
    (status,) = gpu_status()
    assert (status.gpu_id, status.temperature_c, status.power_w) == (0, 45.0, 300.0)
    assert (status.utilization_pct, status.memory_used_mib, status.memory_total_mib) == (80.0, 1000.0, 2000.0)
    assert status.busy is True


def test_gpu_status_keeps_unsupported_readings_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A part that reports ``"N/A"`` for usage and power is unknown, not zero -- and unknown is not busy."""
    _stub_output(
        monkeypatch,
        {"gpu_data": [{"gpu": 0, "usage": "N/A", "power": {"socket_power": "N/A"}, "temperature": "N/A"}]},
    )
    (status,) = gpu_status()
    assert (status.utilization_pct, status.power_w, status.temperature_c) == (None, None, None)
    assert status.busy is False


def test_gpu_status_falls_back_to_the_apu_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    """An APU that publishes only the windowed averages still reports a utilization and a power."""
    _stub_output(
        monkeypatch,
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "usage": {"gfx_activity": "N/A", "apu_average_gfx_activity": {"value": 7, "unit": "%"}},
                    "power": {"socket_power": "N/A", "apu_average_socket_power": {"value": 11.75, "unit": "W"}},
                    "temperature": {"edge": "N/A", "apu_temperature_gfx": {"value": 31.88, "unit": "C"}},
                }
            ]
        },
    )
    (status,) = gpu_status()
    assert (status.utilization_pct, status.power_w, status.temperature_c) == (7.0, 11.75, 31.88)


def test_gpu_status_empty_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe yields no cards rather than a card with no readings."""
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert gpu_status() == []


# -- gpu_asic_info ----------------------------------------------------------


def test_asic_info_reads_name_and_gfx_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call supplies both the marketing name and the gfx target, replacing the rocm-smi plus torch pair."""
    _stub_output(
        monkeypatch,
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "asic": {
                        "market_name": "AMD Radeon 8060S Graphics",
                        "target_graphics_version": "gfx1151",
                        "num_compute_units": 40,
                    },
                }
            ]
        },
    )
    (asic,) = gpu_asic_info()
    assert (asic.gpu_id, asic.market_name) == (0, "AMD Radeon 8060S Graphics")
    assert (asic.target_graphics_version, asic.num_compute_units) == ("gfx1151", 40)


def test_asic_info_keeps_the_gfx_target_when_the_name_is_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a part whose marketing name the driver cannot resolve, the gfx target is still exact."""
    _stub_output(
        monkeypatch,
        {"gpu_data": [{"gpu": 0, "asic": {"market_name": "0x1586", "target_graphics_version": "gfx1151"}}]},
    )
    (asic,) = gpu_asic_info()
    assert (asic.market_name, asic.target_graphics_version, asic.num_compute_units) == ("0x1586", "gfx1151", None)


def test_asic_info_none_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe is unknown, not an empty roster."""
    monkeypatch.setattr(amd_smi.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert gpu_asic_info() is None
