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
    DeclaredPowerSettings,
    GpuPowerSettingsError,
    _parse_node_power,
    _read_gpu_kfd_nodes as _real_read_gpu_kfd_nodes,
    declared_node_problems,
    declared_setting_problems,
    read_gpu_power_settings,
    read_node_power_settings,
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


def _node_payload(status="ENABLED", limit=8000):
    return [{"node": {"power_management": {"limit": {"value": limit, "unit": "W"}, "status": status}}}]


def _fake_run(cmd, **_kwargs):
    payload = _LIMIT if "--limit" in cmd else _PERF
    return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")


def _node_reader(status="ENABLED", limit=8000):
    return lambda: _parse_node_power(_node_payload(status, limit))


def _no_node_command():
    raise GpuPowerSettingsError("amd-smi node --power-management exited 2: invalid choice: 'node'")


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


def test_the_node_settings_are_read_from_amd_smi_json():
    assert _parse_node_power(_node_payload()) == {"npm": "enabled", "node_power_limit_w": 8000.0}
    assert _parse_node_power(_node_payload("DISABLED", 0)) == {"npm": "disabled", "node_power_limit_w": 0.0}
    assert _parse_node_power(_node_payload()[0]) == {"npm": "enabled", "node_power_limit_w": 8000.0}


def test_an_amd_smi_without_the_node_command_is_an_error_not_a_disabled_node(monkeypatch):
    from hyperloom.common import gpu_power_settings

    def _old_amd_smi(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 2, "", "invalid choice: 'node'")

    monkeypatch.setattr(gpu_power_settings.shutil, "which", lambda _name: "/usr/bin/amd-smi")
    monkeypatch.setattr(gpu_power_settings.subprocess, "run", _old_amd_smi)
    with pytest.raises(GpuPowerSettingsError, match="invalid choice"):
        read_node_power_settings()


def test_declared_node_settings_are_checked():
    node = {"npm": "enabled", "node_power_limit_w": 8000.0}
    assert declared_node_problems(node, DeclaredPowerSettings(npm="enabled", node_power_limit_w=8000.0)) == []
    assert declared_node_problems(node, DeclaredPowerSettings()) == []
    assert declared_node_problems(node, DeclaredPowerSettings(npm="disabled", node_power_limit_w=6000.0)) == [
        "node power management is 'enabled', declared 'disabled'",
        "node power limit is 8000.0 W, declared 6000 W",
    ]
    unread = {"npm": "", "node_power_limit_w": None}
    assert declared_node_problems(unread, DeclaredPowerSettings(npm="enabled", node_power_limit_w=1.0)) == [
        "node power management is 'unreadable', declared 'enabled'",
        "node power limit is unreadable W, declared 1 W",
    ]


class TestResolve:
    def _resolve(self, monkeypatch, **kw):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        kw.setdefault("nodes", 1)
        kw.setdefault("read", lambda: read_gpu_power_settings(run=_fake_run))
        from hyperloom.common import gpu_power_settings

        monkeypatch.setattr(gpu_power_settings, "read_node_power_settings", kw.pop("read_node", None) or _node_reader())
        fields = ("power_cap_w", "perf_level", "npm", "node_power_limit_w")
        declared = DeclaredPowerSettings(**{key: value for key in fields if (value := kw.pop(key, None)) is not None})
        return resolve_gpu_power_settings(declared, **kw)

    def test_the_node_npm_state_is_recorded_without_a_declaration(self, monkeypatch):
        record, error = self._resolve(monkeypatch)
        assert error == ""
        assert record["node"] == {"npm": "enabled", "node_power_limit_w": 8000.0}

    def test_a_node_without_npm_support_records_no_node_and_does_not_refuse(self, monkeypatch):
        record, error = self._resolve(monkeypatch, read_node=_no_node_command)
        assert error == ""
        assert "node" not in record

    def test_a_declared_npm_state_the_node_is_at_holds(self, monkeypatch):
        record, error = self._resolve(monkeypatch, npm="enabled", node_power_limit_w=8000.0)
        assert error == ""
        assert record["declared"] == {"npm": "enabled", "node_power_limit_w": 8000.0}

    def test_a_declared_npm_state_the_node_is_not_at_refuses_and_points_at_the_bmc(self, monkeypatch):
        _, error = self._resolve(monkeypatch, npm="enabled", read_node=_node_reader("DISABLED", 0))
        assert error == (
            "the node is not at the declared power management settings (NPM is set on the BMC): "
            "node power management is 'disabled', declared 'enabled'"
        )

    def test_a_declared_node_limit_firmware_clamped_refuses(self, monkeypatch):
        """The BMC may be asked for 8400 W while firmware enforces 8000 W; the enforced limit is the contract."""
        _, error = self._resolve(monkeypatch, node_power_limit_w=8400.0)
        assert "node power limit is 8000.0 W, declared 8400 W" in error

    def test_a_declared_npm_state_that_cannot_be_read_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, npm="disabled", read_node=_no_node_command)
        assert error == (
            "the declared node power management settings cannot be checked: amd-smi node --power-management exited 2: "
            "invalid choice: 'node'"
        )

    def test_card_and_node_mismatches_are_reported_together(self, monkeypatch):
        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, npm="disabled")
        assert error == (
            "the GPUs are not at the declared power settings (set them with amd-smi before launch): GPU 0 power cap is "
            "1400.0 W, declared 1000 W; the node is not at the declared power management settings (NPM is set on the "
            "BMC): node power management is 'enabled', declared 'disabled'"
        )

    def test_a_node_declaration_needs_neither_the_card_reading_nor_the_mask(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi static --limit exited 1")

        record, error = self._resolve(monkeypatch, npm="enabled", read=_unreadable)
        assert (error, record["node"]["npm"]) == ("", "enabled")
        assert self._resolve(monkeypatch, read=_unreadable) == (
            {"node": {"npm": "enabled", "node_power_limit_w": 8000.0}},
            "",
        )
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-745a0e162570d044")
        _, error = self._resolve(monkeypatch, npm="disabled")
        assert "node power management is 'enabled', declared 'disabled'" in error

    def test_a_multi_node_npm_declaration_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, npm="enabled", nodes=2)
        assert "multi-node" in error

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
            DeclaredPowerSettings(power_cap_w=1000.0, perf_level="determinism"),
            nodes=1,
            read=lambda: read_gpu_power_settings(run=_fake_run),
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
            DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, read=read, just_applied=True
        )
        assert error.startswith("the GPUs did not take the power settings just applied: GPU 0 power cap is 1400.0 W")
        _, error = resolve_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, read=read)
        assert "(set them with amd-smi before launch)" in error

    def test_a_declaration_that_cannot_be_read_refuses(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, read=_unreadable)
        assert "cannot be checked" in error

    def test_no_declaration_and_no_amd_smi_records_nothing(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        assert self._resolve(monkeypatch, read=_unreadable, read_node=_no_node_command) == ({}, "")

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
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HSA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        ):
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
            ({"CUDA_VISIBLE_DEVICES": "1"}, {0}),
            ({"HSA_VISIBLE_DEVICES": "1"}, None),
            ({"GPU_DEVICE_ORDINAL": "1"}, None),
        ],
        ids=[
            "rocr-0",
            "rocr-0-3",
            "rocr-4-7",
            "hip-within-rocr-4-7",
            "hip-within-rocr-0-3",
            "hip-alone",
            "cuda-alone",
            "hsa-ignored-by-rocm",
            "gpu-device-ordinal-ignored-by-rocm",
        ],
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
            {"ROCR_VISIBLE_DEVICES": "3,3,2"},
            {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "4"},
            {"HIP_VISIBLE_DEVICES": "1, 2"},
            {"HIP_VISIBLE_DEVICES": "1,,2"},
            {"HIP_VISIBLE_DEVICES": "1;2"},
            {"CUDA_VISIBLE_DEVICES": "01"},
            {"ROCR_VISIBLE_DEVICES": " 2"},
        ],
        ids=[
            "uuid",
            "beyond-the-board",
            "hip-beyond-rocr",
            "repeated-card",
            "set-but-empty",
            "space-after-comma",
            "empty-token",
            "semicolon",
            "leading-zero",
            "padded",
        ],
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
        record, error = resolve_gpu_power_settings(
            DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, read=lambda: cards
        )
        assert (error, list(record["observed"])) == ("", ["3"])

    def test_a_declaration_whose_cards_cannot_be_named_refuses(self, monkeypatch):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-745a0e162570d044")
        cards = {0: {"power_cap_w": 1000.0, "perf_level": "auto"}}
        _, error = resolve_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, read=lambda: cards)
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

    def test_the_node_flags_parse(self):
        args = self._parse("--node-power-management", "ENABLED", "--node-power-limit-w", "6000")
        assert (args.node_power_management, args.node_power_limit_w) == ("enabled", 6000.0)

    def test_the_node_flags_are_off_by_default(self):
        args = self._parse()
        assert (args.node_power_management, args.node_power_limit_w) == (None, None)

    @pytest.mark.parametrize("argv", [("--node-power-management", "on"), ("--node-power-limit-w", "0")])
    def test_an_unusable_node_declaration_stops_the_launch(self, argv):
        with pytest.raises(SystemExit) as exc:
            self._parse(*argv)
        assert exc.value.code == 2


def test_a_resume_re_asserts_the_archived_npm_declaration(monkeypatch):
    """A resume that omits the flag still holds the node to the state the session was measured under."""
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.cli import gpu_power_apply
    from hyperloom.inference_optimizer.cli.bootstrap import resume_gpu_power_settings

    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_power_apply, "warn_about_orphaned_cards", lambda: None)
    monkeypatch.setattr(gpu_power_settings, "read_gpu_power_settings", lambda: read_gpu_power_settings(run=_fake_run))
    monkeypatch.setattr(gpu_power_settings, "read_node_power_settings", _node_reader("DISABLED", 0))
    state = SimpleNamespace(gpu_power_settings={"declared": {"npm": "enabled"}}, nodes=1, session_id="s")
    args = SimpleNamespace(gpu_power_cap_w=None, gpu_perf_level=None, node_power_management=None, nodes=1)
    record, error = resume_gpu_power_settings(args, state)
    assert record["declared"] == {"npm": "enabled"}
    assert "node power management is 'disabled', declared 'enabled'" in error


def test_the_platform_fingerprint_records_the_settings(monkeypatch):
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV, platform_fingerprint

    monkeypatch.setenv(GPU_POWER_SETTINGS_ENV, json.dumps({"observed": {"0": {"power_cap_w": 1000.0}}}))
    record = platform_fingerprint("mi355x")
    if record.get("status") != "ok":
        pytest.skip("no host sysfs")
    assert record["gpu"]["power_settings"]["observed"]["0"]["power_cap_w"] == 1000.0
