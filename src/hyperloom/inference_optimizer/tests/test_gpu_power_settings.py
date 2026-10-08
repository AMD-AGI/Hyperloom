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


def _fake_run(cmd, **_kwargs):
    payload = _LIMIT if "--limit" in cmd else _PERF
    return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")


def test_the_settings_are_read_from_amd_smi_json():
    settings = read_gpu_power_settings(run=_fake_run)
    assert settings == {
        0: {"power_cap_w": 1400.0, "perf_level": "auto"},
        1: {"power_cap_w": 1000.0, "perf_level": "determinism"},
    }


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
        assert record["observed"]["1"] == {"power_cap_w": 1000.0, "perf_level": "determinism"}

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
