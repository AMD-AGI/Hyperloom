# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--apply-gpu-power-settings``: set once for the session, recorded first, restored at exit, recovered after a crash."""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.common.gpu_power_settings import (
    GpuPowerSettingsError,
    PowerSettingsLease,
    _set_commands,
    apply_gpu_power_settings,
    orphaned_power_records,
    restore_gpu_power_settings,
)
from hyperloom.inference_optimizer.cli import _establish_gpu_power_settings
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.inference_optimizer.cli.bootstrap import (
    apply_declared_gpu_power_settings,
    orphaned_power_settings_warning,
)
from hyperloom.inference_optimizer.cli.parser import _build_parser


def _cards(cap: float = 1400.0, level: str = "auto", gpus=range(8)) -> dict[int, dict]:
    return {gpu: {"power_cap_w": cap, "perf_level": level} for gpu in gpus}


class _AmdSmi:
    """Records ``amd-smi`` argv; fails the calls whose argv contains ``fail_on``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        rc = 1 if self.fail_on and self.fail_on in argv else 0
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="permission denied" if rc else "")


class TestSetCommands:
    def test_cards_sharing_a_value_are_set_in_one_call(self):
        commands = _set_commands({4: {"power_cap_w": 1000, "perf_level": "auto"}, 5: {"power_cap_w": 1000}})
        assert ["set", "-g", "4", "5", "-o", "ppt0", "1000"] in commands
        assert ["set", "-g", "4", "-l", "AUTO"] in commands

    def test_apply_raises_on_a_failed_set(self):
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            apply_gpu_power_settings({4}, power_cap_w=1000, perf_level=None, run=_AmdSmi(fail_on="ppt0"))

    def test_restore_attempts_every_card_even_after_one_fails(self):
        smi = _AmdSmi(fail_on="1400")
        problems = restore_gpu_power_settings({4: {"power_cap_w": 1400}, 5: {"power_cap_w": 1300}}, run=smi)
        assert len(problems) == 1
        assert len(smi.calls) == 2


class TestLease:
    def test_a_held_card_cannot_be_leased_twice(self, tmp_path):
        first = PowerSettingsLease(tmp_path, {4, 5})
        with pytest.raises(GpuPowerSettingsError, match="GPU 5"):
            PowerSettingsLease(tmp_path, {5, 6})
        # All-or-nothing: the failed lease must not have kept card 6.
        PowerSettingsLease(tmp_path, {6}).release()
        first.release()

    def test_a_record_left_by_a_dead_holder_is_an_orphan(self, tmp_path):
        lease = PowerSettingsLease(tmp_path, {4})
        lease.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 1000}, owner="s1")
        assert orphaned_power_records(tmp_path) == {}, "a held record is not an orphan"
        lease.release()
        assert orphaned_power_records(tmp_path)[4]["original"]["power_cap_w"] == 1400
        again = PowerSettingsLease(tmp_path, {4})
        assert again.orphaned()[4]["owner"] == "s1"
        again.clear()
        again.release()
        assert orphaned_power_records(tmp_path) == {}


def _apply(tmp_path, monkeypatch, *, observed=None, vram=None, mask="4,5,6,7", smi=None, **kw):
    if mask:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", mask)
    else:
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    smi = smi or _AmdSmi()
    reads = iter(observed or [_cards(), _cards()])
    kw.setdefault("power_cap_w", 1000.0)
    kw.setdefault("perf_level", None)
    out = apply_declared_gpu_power_settings(
        nodes=kw.pop("nodes", 1),
        owner="sess",
        ledger_dir=tmp_path,
        read=lambda: next(reads),
        resident_vram=lambda: vram or {gpu: 284.0 for gpu in range(8)},
        apply=lambda gpus, **a: apply_gpu_power_settings(gpus, run=smi, **a),
        restore=lambda originals: restore_gpu_power_settings(originals, run=smi),
        **kw,
    )
    return (*out, smi)


class TestApplyDeclared:
    def test_sets_only_the_masked_cards_and_restores_them(self, tmp_path, monkeypatch):
        restore, applied, error, smi = _apply(tmp_path, monkeypatch)
        assert error == ""
        assert applied["gpus"] == [4, 5, 6, 7]
        assert applied["originals"]["4"] == {"power_cap_w": 1400.0}, "only what is changed is recorded"
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1000"]]
        assert orphaned_power_records(tmp_path) == {}, "held while the session lives"
        assert restore() == []
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1400"]
        assert orphaned_power_records(tmp_path) == {}, "cleared once restored"
        assert restore() == [] and len(smi.calls) == 2, "restore is idempotent"

    def test_the_record_is_written_before_anything_is_set(self, tmp_path, monkeypatch):
        seen: list = []

        def _apply_and_look(gpus, **_):
            seen.append(orphaned_power_records(tmp_path))
            raise GpuPowerSettingsError("boom")

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4")
        restore, applied, error = apply_declared_gpu_power_settings(
            power_cap_w=1000.0,
            perf_level=None,
            nodes=1,
            owner="s",
            ledger_dir=tmp_path,
            read=_cards,
            resident_vram=dict,
            apply=_apply_and_look,
            restore=lambda originals: [],
        )
        assert seen == [{}], "held, so not visible as an orphan, but the file exists"
        assert (tmp_path / "gpu4.json").exists()
        assert restore is None and "boom" in error

    def test_a_failed_set_restores_and_refuses(self, tmp_path, monkeypatch):
        restore, applied, error, smi = _apply(tmp_path, monkeypatch, smi=_AmdSmi(fail_on="1000"))
        assert restore is None and "permission denied" in error
        assert smi.calls[-1][-1] == "1400", "the originals are put back"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_card_with_someone_elses_model_is_refused_untouched(self, tmp_path, monkeypatch):
        vram = {gpu: 284.0 for gpu in range(8)} | {6: 260_000.0}
        restore, _, error, smi = _apply(tmp_path, monkeypatch, vram=vram)
        assert restore is None and "GPU(s) 6" in error
        assert smi.calls == []

    def test_an_orphan_is_restored_first_and_its_originals_are_the_true_ones(self, tmp_path, monkeypatch):
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        left_changed = _cards(cap=900.0)
        restore, applied, error, smi = _apply(tmp_path, monkeypatch, observed=[left_changed, _cards()])
        assert error == ""
        assert applied["recovered_from_orphan"] == [4]
        assert smi.calls[0] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]
        assert applied["originals"]["4"]["power_cap_w"] == 1400.0
        restore()

    def test_a_card_held_by_a_live_session_is_refused(self, tmp_path, monkeypatch):
        live = PowerSettingsLease(tmp_path, {5})
        restore, _, error, smi = _apply(tmp_path, monkeypatch)
        assert restore is None and "another running Hyperloom session" in error
        assert smi.calls == []
        live.release()

    @pytest.mark.parametrize(
        ("kw", "needle"),
        [({"nodes": 2}, "multi-node"), ({"power_cap_w": None, "perf_level": None}, "needs --gpu-power-cap-w")],
    )
    def test_refusals_before_touching_anything(self, tmp_path, monkeypatch, kw, needle):
        restore, _, error, smi = _apply(tmp_path, monkeypatch, **kw)
        assert restore is None and needle in error and smi.calls == []


def test_orphan_warning_names_the_card_and_its_original(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    lease = PowerSettingsLease(tmp_path, {4})
    lease.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 900}, owner="dead")
    lease.release()
    warning = orphaned_power_settings_warning(tmp_path)
    assert "GPU 4 at cap 900 W from session dead, originally cap 1400 W, perf level auto" in warning
    assert "--apply-gpu-power-settings" in warning
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    assert orphaned_power_settings_warning(tmp_path) == ""


def test_parser_flag_is_off_by_default():
    parser = _build_parser()
    base = ["optimize", "--model", "m", "--framework", "vllm"]
    assert parser.parse_args(base).apply_gpu_power_settings is False
    assert parser.parse_args([*base, "--apply-gpu-power-settings"]).apply_gpu_power_settings is True


class TestEstablish:
    def test_apply_registers_the_restore_and_records_what_was_applied(self, monkeypatch):
        registered: list = []
        monkeypatch.setattr("atexit.register", lambda fn, *a: registered.append((fn, a)))
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.cli.apply_declared_gpu_power_settings",
            lambda **kw: (lambda: [], {"by": "hyperloom", "gpus": [4]}, ""),
        )
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.cli.resolve_gpu_power_settings",
            lambda **kw: ({"declared": {"power_cap_w": 1000.0}, "observed": {}}, ""),
        )
        record, error = _establish_gpu_power_settings(
            power_cap_w=1000.0, perf_level=None, nodes=1, apply=True, owner="s"
        )
        assert error == "" and record["applied"]["by"] == "hyperloom"
        assert len(registered) == 1

    def test_without_permission_nothing_is_applied_and_orphans_are_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.cli.apply_declared_gpu_power_settings",
            lambda **kw: pytest.fail("must not apply without --apply-gpu-power-settings"),
        )
        monkeypatch.setattr("hyperloom.inference_optimizer.cli.orphaned_power_settings_warning", lambda: "GPU 4 left")
        monkeypatch.setattr("hyperloom.inference_optimizer.cli.resolve_gpu_power_settings", lambda **kw: ({}, ""))
        assert _establish_gpu_power_settings(power_cap_w=None, perf_level=None, nodes=1, apply=False, owner="s") == (
            {},
            "",
        )
        assert "GPU 4 left" in capsys.readouterr().err


def test_ledger_dir_follows_the_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HYPERLOOM_RUNTIME_DIR", str(tmp_path / "rt"))
    assert bootstrap.gpu_power_ledger_dir() == tmp_path / "rt" / "gpu_power_settings"
    monkeypatch.delenv("HYPERLOOM_RUNTIME_DIR")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "ws"))
    assert bootstrap.gpu_power_ledger_dir() == tmp_path / "ws" / "runtime" / "gpu_power_settings"


def test_both_declared_are_both_restored(tmp_path, monkeypatch):
    restore, applied, error, smi = _apply(tmp_path, monkeypatch, perf_level="determinism")
    assert error == ""
    assert applied["originals"]["4"] == {"power_cap_w": 1400.0, "perf_level": "auto"}
    assert ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "DETERMINISM"] in smi.calls
    restore()
    assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "AUTO"]
