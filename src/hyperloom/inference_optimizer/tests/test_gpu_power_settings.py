# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU power settings are read, asserted and recorded; nothing in a session sets them."""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from hyperloom.common.gpu_power_settings import (
    GpuPowerSettingsError,
    declared_setting_problems,
    read_gpu_power_settings,
)

# --- reading and asserting the settings -----------------------------------------------------------------------------

_LIMIT = {
    "gpu_data": [
        {"gpu": 0, "limit": {"ppt0": {"socket_power_limit": {"value": 1400, "unit": "W"}}}},
        {"gpu": 1, "limit": {"ppt0": {"socket_power_limit": {"value": 1000, "unit": "W"}}}},
    ]
}
_PERF = {
    "gpu_data": [
        {"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"},
        {"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_DETERMINISM"},
    ]
}


def _gfx(*max_mhz: float) -> dict:
    """``metric --clock`` for one card: one ``gfx_<n>`` block per XCD, as MI355X reports it."""
    blocks = {
        f"gfx_{n}": {"clk": {"value": 1700, "unit": "MHz"}, "max_clk": {"value": mhz, "unit": "MHz"}}
        for n, mhz in enumerate(max_mhz)
    }
    return {**blocks, "mem_0": {"max_clk": {"value": 2000, "unit": "MHz"}}}


_CLOCK = {"gpu_data": [{"gpu": 0, "clock": _gfx(*[2400] * 8)}, {"gpu": 1, "clock": _gfx(*[2000] * 8)}]}


def _fake_run(cmd, **_kwargs):
    payload = _LIMIT if "--limit" in cmd else _CLOCK if "--clock" in cmd else _PERF
    return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")


def test_the_settings_are_read_from_amd_smi_json():
    settings = read_gpu_power_settings(run=_fake_run)
    assert settings == {
        0: {"power_cap_w": 1400.0, "perf_level": "auto", "sclk_max_mhz": 2400.0},
        1: {"power_cap_w": 1000.0, "perf_level": "determinism", "sclk_max_mhz": 2000.0},
    }


def test_a_failing_clock_read_leaves_the_cap_and_level_checkable():
    def _no_clock(cmd, **kwargs):
        if "--clock" in cmd:
            return subprocess.CompletedProcess(cmd, 2, "", "unsupported")
        return _fake_run(cmd, **kwargs)

    settings = read_gpu_power_settings(run=_no_clock)
    assert settings[0] == {"power_cap_w": 1400.0, "perf_level": "auto", "sclk_max_mhz": None}
    assert declared_setting_problems(settings, power_cap_w=1400.0, gpus={0}) == []
    assert declared_setting_problems(settings, sclk_max_mhz=2400.0, gpus={0}) == [
        "GPU 0 sclk max is unreadable, declared 2400 MHz"
    ]


def test_an_sclk_limit_the_xcds_disagree_on_is_unread():
    """A partial set leaves XCDs at different limits; reporting either one would assert a cap that does not hold."""
    from hyperloom.common.gpu_power_settings import _sclk_max_mhz

    assert _sclk_max_mhz(_gfx(2000, 2000, 2400)) is None
    assert _sclk_max_mhz({"mem_0": {"max_clk": {"value": 2000}}}) is None


def test_a_declared_sclk_max_is_asserted_like_the_cap():
    settings = read_gpu_power_settings(run=_fake_run)
    assert declared_setting_problems(settings, sclk_max_mhz=2000.0, gpus={1}) == []
    assert declared_setting_problems(settings, sclk_max_mhz=2000.0, gpus={0}) == [
        "GPU 0 sclk max is 2400 MHz, declared 2000 MHz"
    ]


@pytest.mark.parametrize(
    ("level", "refused"), [("auto", True), ("determinism", True), ("manual", False), (None, False)]
)
def test_an_sclk_max_holds_only_at_the_manual_perf_level(level, refused):
    from hyperloom.common.gpu_power_settings import declared_combination_error

    error = declared_combination_error(perf_level=level, sclk_max_mhz=2000.0)
    assert bool(error) is refused
    assert declared_combination_error(perf_level=level, sclk_max_mhz=None) == ""


def test_a_failing_amd_smi_is_an_error_not_an_empty_reading():
    def _fail(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 2, "", "permission denied")

    with pytest.raises(GpuPowerSettingsError):
        read_gpu_power_settings(run=_fail)


def test_declared_settings_are_checked_per_card():
    settings = read_gpu_power_settings(run=_fake_run)
    assert declared_setting_problems(settings, power_cap_w=1000.0, gpus={1}) == []
    problems = declared_setting_problems(settings, power_cap_w=1000.0, perf_level="determinism")
    assert problems == [
        "GPU 0 power cap is 1400.0 W, declared 1000 W",
        "GPU 0 perf level is 'auto', declared 'determinism'",
    ]


class TestResolve:
    def _resolve(self, monkeypatch, **kw):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        kw.setdefault("nodes", 1)
        kw.setdefault("power_cap_w", None)
        kw.setdefault("perf_level", None)
        kw.setdefault("read", lambda: read_gpu_power_settings(run=_fake_run))
        return resolve_gpu_power_settings(**kw)

    def test_nothing_declared_records_what_the_cards_report(self, monkeypatch):
        record, error = self._resolve(monkeypatch)
        assert error == ""
        assert record["declared"] == {}
        assert record["observed"]["1"] == {"power_cap_w": 1000.0, "perf_level": "determinism", "sclk_max_mhz": 2000.0}

    def test_a_declared_cap_the_cards_are_not_at_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, power_cap_w=1000.0)
        assert "GPU 0 power cap is 1400.0 W" in error

    def test_only_the_cards_the_session_uses_are_checked(self, monkeypatch):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
        record, error = resolve_gpu_power_settings(
            power_cap_w=1000.0, perf_level="determinism", nodes=1, read=lambda: read_gpu_power_settings(run=_fake_run)
        )
        assert error == ""
        assert list(record["observed"]) == ["1"]

    def test_a_declaration_that_cannot_be_read_refuses(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, read=_unreadable)
        assert "cannot be checked" in error

    def test_no_declaration_and_no_amd_smi_records_nothing(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        assert self._resolve(monkeypatch, read=_unreadable) == ({}, "")

    def test_a_multi_node_declaration_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, nodes=2)
        assert "multi-node" in error


class TestCli:
    def _parse(self, *argv: str) -> argparse.Namespace:
        from hyperloom.inference_optimizer.cli.parser import _build_parser

        return _build_parser().parse_args(["optimize", "--model", "/m", *argv])

    def test_the_flags_parse(self):
        args = self._parse("--gpu-power-cap-w", "1000", "--gpu-perf-level", "determinism")
        assert (args.gpu_power_cap_w, args.gpu_perf_level) == (1000.0, "determinism")

    def test_both_are_off_by_default(self):
        args = self._parse()
        assert (args.gpu_power_cap_w, args.gpu_perf_level) == (None, None)

    @pytest.mark.parametrize("bad", ["1000W", "0", "-5", "nan"])
    def test_an_unusable_cap_stops_the_launch(self, bad):
        with pytest.raises(SystemExit) as exc:
            self._parse("--gpu-power-cap-w", bad)
        assert exc.value.code == 2


def test_the_platform_fingerprint_records_the_settings(monkeypatch):
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV, platform_fingerprint

    monkeypatch.setenv(GPU_POWER_SETTINGS_ENV, json.dumps({"observed": {"0": {"power_cap_w": 1000.0}}}))
    record = platform_fingerprint("mi355x")
    if record.get("status") != "ok":
        pytest.skip("no host sysfs")
    assert record["gpu"]["power_settings"]["observed"]["0"]["power_cap_w"] == 1000.0
