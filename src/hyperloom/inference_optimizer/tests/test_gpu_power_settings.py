# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU power settings are read, asserted and recorded; nothing in a session sets them."""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from hyperloom.common import gpu_power_settings
from hyperloom.common.gpu_power_settings import (
    GpuPowerSettingsError,
    _read_gpu_kfd_nodes as _real_read_gpu_kfd_nodes,
    declared_setting_problems,
    read_gpu_power_settings,
    visible_gpu_indices,
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

    def test_a_setting_that_did_not_take_after_an_apply_says_so(self, monkeypatch):
        """The operator did not set these; telling them to set them by hand would send them the wrong way."""
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        read = lambda: read_gpu_power_settings(run=_fake_run)  # noqa: E731
        _, error = resolve_gpu_power_settings(
            power_cap_w=1000.0, perf_level=None, nodes=1, read=read, just_applied=True
        )
        assert error.startswith("the GPUs did not take the power settings just applied: GPU 0 power cap is 1400.0 W")
        _, error = resolve_gpu_power_settings(power_cap_w=1000.0, perf_level=None, nodes=1, read=read)
        assert "(set them with amd-smi before launch)" in error

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


#: An MI355X board as amd-smi lists it: ROCm numbers cards in KFD node order, amd-smi in PCI order, and they differ.
_MI355X_NODES = {0: 9, 1: 11, 2: 10, 3: 8, 4: 13, 5: 15, 6: 14, 7: 12}


class TestVisibleCards:
    """A mask number is a ROCm number, never an amd-smi index: it must be translated before it reaches amd-smi."""

    @pytest.fixture(autouse=True)
    def _mi355x(self, monkeypatch):
        monkeypatch.setattr(gpu_power_settings, "_read_gpu_kfd_nodes", lambda: dict(_MI355X_NODES))
        for name in ("ROCR_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize(
        ("env", "cards"),
        [
            ({"ROCR_VISIBLE_DEVICES": "0"}, {3}),
            ({"ROCR_VISIBLE_DEVICES": "0,1,2,3"}, {0, 1, 2, 3}),
            ({"ROCR_VISIBLE_DEVICES": "4,5,6,7"}, {4, 5, 6, 7}),
            ({"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "0"}, {7}),
            ({"ROCR_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "0"}, {3}),
            ({"HIP_VISIBLE_DEVICES": "1"}, {0}),
        ],
        ids=["rocr-0", "rocr-0-3", "rocr-4-7", "hip-within-rocr-4-7", "hip-within-rocr-0-3", "hip-alone"],
    )
    def test_a_mask_names_the_physical_cards(self, env, cards):
        assert visible_gpu_indices(env) == cards

    def test_no_mask_is_every_card_and_asks_amd_smi_nothing(self, monkeypatch):
        monkeypatch.setattr(gpu_power_settings, "_read_gpu_kfd_nodes", lambda: pytest.fail("no mask, no amd-smi"))
        assert visible_gpu_indices({}) is None

    @pytest.mark.parametrize(
        "env",
        [
            {"ROCR_VISIBLE_DEVICES": "GPU-745a0e162570d044"},
            {"ROCR_VISIBLE_DEVICES": "8"},
            {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "2"},
        ],
        ids=["uuid", "beyond-the-board", "hip-beyond-rocr"],
    )
    def test_a_mask_whose_cards_cannot_be_named_refuses(self, env):
        with pytest.raises(GpuPowerSettingsError):
            visible_gpu_indices(env)

    def test_the_kfd_nodes_come_from_amd_smi_list(self, monkeypatch):
        listed = {
            "gpu_data": [{"gpu": gpu, "bdf": "0000:00:00.0", "node_id": node} for gpu, node in _MI355X_NODES.items()]
        }
        monkeypatch.setattr(gpu_power_settings.shutil, "which", lambda name: "/usr/bin/amd-smi")
        monkeypatch.setattr(
            gpu_power_settings.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, json.dumps(listed), ""),
        )
        assert _real_read_gpu_kfd_nodes() == _MI355X_NODES
        listed["gpu_data"][0].pop("node_id")
        with pytest.raises(GpuPowerSettingsError, match="without its KFD node"):
            _real_read_gpu_kfd_nodes()

    def test_the_assertion_checks_the_card_the_mask_names(self, monkeypatch):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        cards = {gpu: {"power_cap_w": 1000.0 if gpu == 3 else 1400.0, "perf_level": "auto"} for gpu in range(8)}
        record, error = resolve_gpu_power_settings(power_cap_w=1000.0, perf_level=None, nodes=1, read=lambda: cards)
        assert (error, list(record["observed"])) == ("", ["3"])

    def test_a_declaration_whose_cards_cannot_be_named_refuses(self, monkeypatch):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-745a0e162570d044")
        cards = {0: {"power_cap_w": 1000.0, "perf_level": "auto"}}
        _, error = resolve_gpu_power_settings(power_cap_w=1000.0, perf_level=None, nodes=1, read=lambda: cards)
        assert "cannot be checked" in error and "device numbers" in error

    def test_the_sampler_samples_the_cards_the_mask_names(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors._gpu_power import build_gpu_power_recorder

        monkeypatch.delenv("HYPERLOOM_GPU_POWER_SAMPLING", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
        log_path = str(tmp_path / "server.log")
        assert build_gpu_power_recorder(log_path, {"ROCR_VISIBLE_DEVICES": "4,5"})._gpus == {4, 7}
        assert build_gpu_power_recorder(log_path, {"ROCR_VISIBLE_DEVICES": "GPU-x"})._gpus is None, "every card"

    def test_a_per_gpu_budget_whose_cards_cannot_be_named_refuses_the_launch(self, monkeypatch, capsys):
        from types import SimpleNamespace

        from hyperloom.inference_optimizer.cli import bootstrap

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-745a0e162570d044")
        monkeypatch.setattr(bootstrap, "power_budget_unmeasurable_error", lambda **kw: "")
        args = SimpleNamespace(max_per_gpu_power_w={"3": 700.0}, apply_gpu_power_settings=False)
        with pytest.raises(SystemExit) as exc:
            bootstrap.resolve_launch_gpu_power(args, nodes=1, owner="s")
        assert exc.value.code == 2
        assert "--max-per-gpu-power-w cannot be checked against the visible cards" in capsys.readouterr().err


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
