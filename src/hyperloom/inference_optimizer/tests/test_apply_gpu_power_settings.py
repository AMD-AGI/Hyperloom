# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--apply-gpu-power-settings``: set once on the masked cards, recorded first, restored however the session ends."""

from __future__ import annotations

import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

from hyperloom.common import gpu_power_control
from hyperloom.common.gpu_power_control import (
    GPU_POWER_LEDGER_DIR,
    PowerSettingsLease,
    _set_commands,
    apply_gpu_power_settings,
    orphaned_power_records,
    restore_gpu_power_settings,
)
from hyperloom.common.gpu_power_settings import DeclaredPowerSettings, GpuPowerSettingsError
from hyperloom.inference_optimizer.cli import bootstrap, gpu_power_apply
from hyperloom.inference_optimizer.cli.gpu_power_apply import (
    _Hooks,
    _RestoreGuard,
    apply_declared_gpu_power_settings,
    orphaned_power_settings_warning,
)
from hyperloom.inference_optimizer.cli.parser import _build_parser

_MASK = {4, 5, 6, 7}


def _cards(cap: float = 1400.0, level: str = "auto", gpus=range(8), sclk: float = 2400.0) -> dict[int, dict]:
    return {gpu: {"power_cap_w": cap, "perf_level": level, "sclk_max_mhz": sclk} for gpu in gpus}


class _AmdSmi:
    """Records ``amd-smi`` argv; fails the calls whose argv contains ``fail_on``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        rc = 1 if self.fail_on and self.fail_on in argv else 0
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr="permission denied" if rc else "")


def _hooks(*, observed=None, vram=None, mask=_MASK, smi=None) -> tuple[_Hooks, _AmdSmi]:
    smi = smi or _AmdSmi()
    reads = list(observed or [_cards(), _cards()])
    hooks = _Hooks(
        read=lambda: reads.pop(0) if len(reads) > 1 else reads[0],
        resident_vram=lambda: dict.fromkeys(range(8), 284.0) if vram is None else vram,
        apply=lambda gpus, declared: apply_gpu_power_settings(gpus, declared, run=smi),
        restore=lambda originals: restore_gpu_power_settings(originals, run=smi),
        mask=lambda: mask,
        arm=lambda restore: None,
        disarm=lambda: None,
    )
    return hooks, smi


def _apply(tmp_path, *, nodes=1, power_cap_w=1000.0, perf_level=None, sclk_max_mhz=None, **hook_kw):
    hooks, smi = _hooks(**hook_kw)
    applied = apply_declared_gpu_power_settings(
        DeclaredPowerSettings(power_cap_w=power_cap_w, perf_level=perf_level or "", sclk_max_mhz=sclk_max_mhz),
        nodes=nodes,
        owner="sess",
        ledger_dir=tmp_path,
        hooks=hooks,
    )
    return applied, smi


class TestSetCommands:
    def test_cards_sharing_a_value_are_set_in_one_call(self):
        commands = _set_commands({4: {"power_cap_w": 1000, "perf_level": "auto"}, 5: {"power_cap_w": 1000}})
        assert ["set", "-g", "4", "5", "-o", "ppt0", "1000"] in commands
        assert ["set", "-g", "4", "-l", "AUTO"] in commands

    def test_apply_raises_on_a_failed_set(self):
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            apply_gpu_power_settings({4}, DeclaredPowerSettings(power_cap_w=1000), run=_AmdSmi(fail_on="ppt0"))

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

    def test_a_record_from_an_earlier_boot_is_not_an_orphan(self, tmp_path, monkeypatch):
        """A reboot reset the card, so restoring the record would set a value nobody asked for."""
        lease = PowerSettingsLease(tmp_path, {4})
        lease.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 1000}, owner="s1")
        lease.release()
        assert orphaned_power_records(tmp_path)[4]["boot_id"] == gpu_power_control.current_boot_id()
        monkeypatch.setattr(gpu_power_control, "current_boot_id", lambda: "another-boot")
        assert orphaned_power_records(tmp_path) == {}

    def test_records_are_readable_by_all_and_writable_by_their_owner_in_a_sticky_directory(self, tmp_path):
        ledger = tmp_path / "ledger"
        PowerSettingsLease(ledger, {4}).release()
        assert ledger.stat().st_mode & 0o7777 == 0o1777
        assert (ledger / "gpu4.json").stat().st_mode & 0o777 == 0o644

    def test_a_directory_other_accounts_can_plant_records_in_is_refused(self, tmp_path):
        """A privileged restore sets whatever a record says, so a planted record must never be read."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        os.chmod(ledger, 0o777)
        with pytest.raises(GpuPowerSettingsError, match="writable by other accounts"):
            PowerSettingsLease(ledger, {4})
        assert orphaned_power_records(ledger) == {}

    def test_a_record_other_accounts_can_write_is_refused(self, tmp_path):
        planted = PowerSettingsLease(tmp_path, {4})
        planted.record({4: {"power_cap_w": 1.0}}, applied={"power_cap_w": 900}, owner="x")
        planted.release()
        os.chmod(tmp_path / "gpu4.json", 0o666)
        assert orphaned_power_records(tmp_path) == {}
        with pytest.raises(GpuPowerSettingsError, match="writable by other accounts"):
            PowerSettingsLease(tmp_path, {4})

    def test_a_symlinked_record_is_refused(self, tmp_path):
        target = tmp_path / "elsewhere.json"
        target.write_text("{}")
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        (ledger / "gpu4.json").symlink_to(target)
        with pytest.raises(GpuPowerSettingsError):
            PowerSettingsLease(ledger, {4})

    def test_the_ledger_is_host_local_and_ignores_the_workspace(self):
        assert str(GPU_POWER_LEDGER_DIR) == "/var/tmp/hyperloom/gpu_power_settings"
        assert apply_declared_gpu_power_settings.__kwdefaults__["ledger_dir"] == GPU_POWER_LEDGER_DIR


class TestApplyDeclared:
    def test_sets_only_the_masked_cards_and_restores_them(self, tmp_path):
        applied, smi = _apply(tmp_path)
        assert applied.record["gpus"] == [4, 5, 6, 7]
        assert applied.record["originals"]["4"] == {"power_cap_w": 1400.0}, "only what is changed is recorded"
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1000"]]
        assert orphaned_power_records(tmp_path) == {}, "held while the session lives"
        assert applied.restore() == []
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1400"]
        assert orphaned_power_records(tmp_path) == {}, "cleared once restored"
        assert applied.restore() == [] and len(smi.calls) == 2, "restore is idempotent"

    @pytest.mark.parametrize("mask", [None], ids=["no-integer-mask"])
    def test_without_an_integer_mask_nothing_is_set_or_recorded(self, tmp_path, mask):
        with pytest.raises(GpuPowerSettingsError, match="integer ROCR_VISIBLE_DEVICES"):
            _apply(tmp_path, mask=mask)
        assert list(tmp_path.iterdir()) == []

    def test_a_masked_card_amd_smi_does_not_report_refuses(self, tmp_path):
        with pytest.raises(GpuPowerSettingsError, match=r"masked GPU\(s\) 9"):
            _apply(tmp_path, mask={4, 9})

    def test_the_record_is_written_before_anything_is_set(self, tmp_path):
        seen: list = []
        hooks, _ = _hooks(mask={4})

        def _apply_and_look(gpus, _declared):
            seen.append(orphaned_power_records(tmp_path))
            raise GpuPowerSettingsError("boom")

        hooks.apply = _apply_and_look
        with pytest.raises(GpuPowerSettingsError, match="boom"):
            apply_declared_gpu_power_settings(
                DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s", ledger_dir=tmp_path, hooks=hooks
            )
        assert seen == [{}], "held, so not visible as an orphan, but the file exists"
        assert (tmp_path / "gpu4.json").exists()

    def test_a_failed_set_restores_and_refuses(self, tmp_path):
        smi = _AmdSmi(fail_on="1000")
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            _apply(tmp_path, smi=smi)
        assert smi.calls[-1][-1] == "1400", "the originals are put back"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_card_with_someone_elses_model_is_refused_untouched(self, tmp_path):
        vram = dict.fromkeys(range(8), 284.0) | {6: 260_000.0}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"GPU\(s\) 6 already hold"):
            _apply(tmp_path, vram=vram, smi=smi)
        assert smi.calls == []

    def test_a_card_whose_vram_cannot_be_read_is_refused_untouched(self, tmp_path):
        vram = {gpu: 284.0 for gpu in range(8) if gpu != 5}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"VRAM use of GPU\(s\) 5 could not be read"):
            _apply(tmp_path, vram=vram, smi=smi)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_an_orphan_is_restored_first_and_its_originals_are_the_true_ones(self, tmp_path):
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        applied, smi = _apply(tmp_path, observed=[_cards(cap=900.0), _cards()])
        assert applied.record["recovered_from_orphan"] == [4]
        assert smi.calls[0] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]
        assert applied.record["originals"]["4"]["power_cap_w"] == 1400.0
        applied.restore()

    def test_a_card_held_by_a_live_session_is_refused(self, tmp_path):
        live = PowerSettingsLease(tmp_path, {5})
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match="another running Hyperloom session"):
            _apply(tmp_path, smi=smi)
        assert smi.calls == []
        live.release()

    @pytest.mark.parametrize(
        ("kw", "needle"),
        [({"nodes": 2}, "multi-node"), ({"power_cap_w": None, "perf_level": None}, "needs at least one of")],
    )
    def test_refusals_before_touching_anything(self, tmp_path, kw, needle):
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=needle):
            _apply(tmp_path, smi=smi, **kw)
        assert smi.calls == []

    def test_both_declared_are_both_restored(self, tmp_path):
        applied, smi = _apply(tmp_path, perf_level="determinism")
        assert applied.record["originals"]["4"] == {"power_cap_w": 1400.0, "perf_level": "auto", "sclk_max_mhz": 2400.0}
        assert ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "DETERMINISM"] in smi.calls
        applied.restore()
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "AUTO"]

    def test_an_sclk_cap_alone_restores_the_perf_level_it_changed(self, tmp_path):
        """A clock limit puts the card in manual; from auto, restoring auto also resets the limit, so no sclk set."""
        applied, smi = _apply(tmp_path, power_cap_w=None, sclk_max_mhz=2000.0)
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-L", "sclk", "max", "2000"]]
        assert applied.record["originals"]["4"] == {"sclk_max_mhz": 2400.0, "perf_level": "auto"}
        applied.restore()
        assert smi.calls[1:] == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "AUTO"]]

    def test_an_sclk_cap_from_manual_restores_level_then_limit(self, tmp_path):
        manual = _cards(level="manual", sclk=2100.0)
        applied, smi = _apply(
            tmp_path, power_cap_w=None, perf_level="manual", sclk_max_mhz=2000.0, observed=[manual, manual]
        )
        assert smi.calls == [
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "MANUAL"],
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-L", "sclk", "max", "2000"],
        ], "the level first: setting a level resets the clock limit"
        applied.restore()
        assert smi.calls[2:] == [
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "MANUAL"],
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-L", "sclk", "max", "2100"],
        ]

    def test_an_sclk_cap_with_another_perf_level_is_refused_untouched(self, tmp_path):
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match="holds only at the manual perf level"):
            _apply(tmp_path, perf_level="auto", sclk_max_mhz=2000.0, smi=smi)
        assert smi.calls == []
        assert list(tmp_path.iterdir()) == []

    def test_an_unreadable_sclk_limit_refuses(self, tmp_path):
        split = {**_cards(), 5: {"power_cap_w": 1400.0, "perf_level": "auto", "sclk_max_mhz": None}}
        with pytest.raises(GpuPowerSettingsError, match=r"sclk max .*GPU\(s\) 5"):
            _apply(tmp_path, power_cap_w=None, sclk_max_mhz=2000.0, observed=[split, split])

    @pytest.mark.parametrize(
        ("level", "kw", "needle"),
        [
            ("determinism", {"power_cap_w": None, "sclk_max_mhz": 2000.0}, "determinism perf level"),
            ("determinism", {"power_cap_w": None, "perf_level": "manual"}, "determinism perf level"),
            ("high", {"power_cap_w": None, "sclk_max_mhz": 2000.0}, "are at the high perf level"),
        ],
    )
    def test_a_starting_level_a_restore_could_not_bring_back_is_refused_untouched(self, tmp_path, level, kw, needle):
        cards = _cards(level=level, sclk=1300.0)
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=needle):
            _apply(tmp_path, observed=[cards, cards], smi=smi, **kw)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_a_cap_alone_on_a_determinism_card_is_allowed(self, tmp_path):
        """The perf level is not touched, so the determinism soft-max is never at risk."""
        cards = _cards(level="determinism", sclk=1300.0)
        applied, smi = _apply(tmp_path, observed=[cards, cards])
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1000"]]
        assert applied.restore() == []

    def test_a_perf_level_apply_from_manual_puts_back_the_operators_sclk_limit(self, tmp_path):
        """Setting a level resets the clock limit, so a level-only apply must record and restore the limit too."""
        manual = _cards(level="manual", sclk=2100.0)
        applied, smi = _apply(tmp_path, power_cap_w=None, perf_level="auto", observed=[manual, manual])
        assert applied.record["originals"]["4"] == {"perf_level": "manual", "sclk_max_mhz": 2100.0}
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "AUTO"]]
        applied.restore()
        assert smi.calls[1:] == [
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "MANUAL"],
            ["amd-smi", "set", "-g", "4", "5", "6", "7", "-L", "sclk", "max", "2100"],
        ]

    def test_a_restore_the_card_did_not_take_is_reported_and_its_record_kept(self, tmp_path):
        """amd-smi can exit 0 without the card taking the value; the record must survive for the next launch."""
        stuck_manual = _cards(level="manual", sclk=2000.0)
        applied, _ = _apply(
            tmp_path, power_cap_w=None, sclk_max_mhz=2000.0, observed=[_cards(), _cards(), stuck_manual]
        )
        problems = applied.restore()
        assert any("perf level reads 'manual' after the restore, recorded 'auto'" in p for p in problems)
        assert orphaned_power_records(tmp_path)[4]["original"]["perf_level"] == "auto"

    @pytest.mark.parametrize(
        ("observed", "perf_level", "unreadable_gpu"),
        [
            pytest.param({**_cards(), 5: {"power_cap_w": None, "perf_level": "auto"}}, None, "5", id="cap-unreadable"),
            pytest.param(
                {**_cards(), 6: {"power_cap_w": 1400.0, "perf_level": ""}}, "determinism", "6", id="level-unreadable"
            ),
        ],
    )
    def test_an_original_that_cannot_be_read_refuses_before_anything_is_set(
        self, tmp_path, observed, perf_level, unreadable_gpu
    ):
        """Restore skips a value it never read, so applying over one would leave the card changed after the session."""
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=rf"GPU\(s\) {unreadable_gpu} .*nothing was applied"):
            _apply(tmp_path, observed=[observed, observed], perf_level=perf_level, smi=smi)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_a_card_missing_from_the_reread_after_orphan_recovery_refuses(self, tmp_path):
        """The re-read after restoring an orphan can drop a card; nothing restores what was not read."""
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        reread = {gpu: row for gpu, row in _cards().items() if gpu != 7}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"GPU\(s\) 7 .*nothing was applied"):
            _apply(tmp_path, observed=[_cards(cap=900.0), reread], smi=smi)
        assert smi.calls == [["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]], "only the orphan was restored"


def test_orphan_warning_names_the_card_and_its_original(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
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
    args = parser.parse_args([*base, "--apply-gpu-power-settings", "--gpu-power-cap-w", "1000"])
    assert args.apply_gpu_power_settings is True and args.gpu_power_cap_w == 1000.0


class TestRestoreGuard:
    """The restore runs on the way out however the session leaves, short of SIGKILL."""

    @pytest.fixture
    def guard(self, monkeypatch):
        monkeypatch.setattr("atexit.register", lambda fn, *a: None)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        guard = _RestoreGuard()
        monkeypatch.setattr(gpu_power_apply, "_GUARD", guard)
        yield guard
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
    def test_a_stop_signal_unwinds_through_the_restore(self, guard, sig):
        restored: list[bool] = []
        before = signal.getsignal(sig)
        with pytest.raises(SystemExit) as exc:
            with gpu_power_apply.gpu_power_restore_guard():
                guard.arm(lambda: restored.append(True) or [])
                os.kill(os.getpid(), sig)
        assert exc.value.code == 128 + sig
        assert restored == [True]
        assert signal.getsignal(sig) == before, "the handler is put back once the cards are"

    @pytest.mark.parametrize("interrupt", ["sigterm", "keyboard"])
    def test_an_interrupt_during_the_set_restores(self, guard, tmp_path, interrupt):
        """A card is changed as soon as its set returns, before the next set; the restore must already be armed."""
        hooks, smi = _hooks(mask={4})
        hooks.arm, hooks.disarm = guard.arm, guard.disarm

        def _set_then_interrupt(gpus, declared):
            apply_gpu_power_settings(gpus, declared, run=smi)
            if interrupt == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            raise KeyboardInterrupt

        hooks.apply = _set_then_interrupt
        with pytest.raises((SystemExit, KeyboardInterrupt)):
            with gpu_power_apply.gpu_power_restore_guard():
                apply_declared_gpu_power_settings(
                    DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s", ledger_dir=tmp_path, hooks=hooks
                )
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"], "the original is put back"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_failed_set_restores_once_and_disarms(self, guard, tmp_path):
        smi = _AmdSmi(fail_on="1000")
        hooks, _ = _hooks(mask={4}, smi=smi)
        hooks.arm, hooks.disarm = guard.arm, guard.disarm
        with pytest.raises(GpuPowerSettingsError):
            apply_declared_gpu_power_settings(
                DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s", ledger_dir=tmp_path, hooks=hooks
            )
        guard.run()
        assert [call[-1] for call in smi.calls] == ["1000", "1400"], "restored once, by the failed apply"
        assert signal.getsignal(signal.SIGTERM) is not _RestoreGuard._exit_on_signal

    def test_an_exit_after_applying_restores(self, guard):
        restored: list[bool] = []
        with pytest.raises(SystemExit):
            with gpu_power_apply.gpu_power_restore_guard():
                guard.arm(lambda: restored.append(True) or [])
                raise SystemExit(2)
        assert restored == [True]

    def test_a_stop_signal_during_the_restore_cannot_cut_it_short(self, guard):
        steps: list[str] = []

        def _restore():
            os.kill(os.getpid(), signal.SIGTERM)
            steps.append("finished")
            return []

        guard.arm(_restore)
        guard.run()
        assert steps == ["finished"]

    def test_the_restore_runs_once_and_a_failure_is_loud(self, guard, capsys):
        calls: list[int] = []
        guard.arm(lambda: calls.append(1) or ["amd-smi set exited 1"])
        guard.run()
        guard.run()
        assert calls == [1]
        assert "could not restore the GPU power settings" in capsys.readouterr().err


class TestSessionWiring:
    def test_without_permission_nothing_is_applied_and_orphans_are_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(
            gpu_power_apply,
            "apply_for_session",
            lambda *a, **kw: pytest.fail("must not apply without --apply-gpu-power-settings"),
        )
        monkeypatch.setattr(gpu_power_apply, "orphaned_power_settings_warning", lambda: "GPU 4 left")
        monkeypatch.setattr(bootstrap, "resolve_gpu_power_settings", lambda **kw: ({"declared": {}}, ""))
        args = SimpleNamespace(apply_gpu_power_settings=False)
        record = bootstrap.resolve_launch_gpu_power(args, nodes=1, owner="s")
        assert "applied" not in record
        assert "GPU 4 left" in capsys.readouterr().err

    def test_a_launch_refused_on_its_budget_sets_nothing(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "power_budget_unmeasurable_error", lambda **kw: "multi-node")
        monkeypatch.setattr(
            gpu_power_apply,
            "apply_for_session",
            lambda *a, **kw: pytest.fail("must not apply when the budget is refused"),
        )
        args = SimpleNamespace(max_power_w=2000.0, apply_gpu_power_settings=True, gpu_power_cap_w=1000.0)
        with pytest.raises(SystemExit) as exc:
            bootstrap.resolve_launch_gpu_power(args, nodes=2, owner="s")
        assert exc.value.code == 2

    def test_a_launch_refused_on_its_per_gpu_budget_sets_nothing(self, monkeypatch, capsys):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(bootstrap, "power_budget_unmeasurable_error", lambda **kw: "")
        monkeypatch.setattr(
            gpu_power_apply,
            "apply_for_session",
            lambda *a, **kw: pytest.fail("must not apply when the budget is refused"),
        )
        args = SimpleNamespace(max_per_gpu_power_w={"7": 700.0}, apply_gpu_power_settings=True, gpu_power_cap_w=1000.0)
        with pytest.raises(SystemExit):
            bootstrap.resolve_launch_gpu_power(args, nodes=1, owner="s")
        assert "names GPU(s) 7" in capsys.readouterr().err

    def test_a_launch_records_what_it_applied(self, monkeypatch):
        monkeypatch.setattr(gpu_power_apply, "apply_for_session", lambda *a, **kw: {"by": "hyperloom", "gpus": [4]})
        monkeypatch.setattr(
            bootstrap, "resolve_gpu_power_settings", lambda **kw: ({"declared": {"power_cap_w": 1000.0}}, "")
        )
        args = SimpleNamespace(apply_gpu_power_settings=True, gpu_power_cap_w=1000.0)
        record = bootstrap.resolve_launch_gpu_power(args, nodes=1, owner="s")
        assert record["applied"] == {"by": "hyperloom", "gpus": [4]}

    def test_a_resume_of_an_applied_session_says_to_pass_the_flag_again(self, monkeypatch):
        monkeypatch.setattr(gpu_power_apply, "apply_for_session", lambda *a, **kw: {})
        monkeypatch.setattr(
            bootstrap, "resolve_gpu_power_settings", lambda **kw: ({}, "GPU 4 power cap is 1400 W, declared 1000 W")
        )
        state = SimpleNamespace(
            gpu_power_settings={"declared": {"power_cap_w": 1000.0}, "applied": {"by": "hyperloom"}},
            nodes=1,
            session_id="s",
        )
        _, error = bootstrap.resume_gpu_power_settings(SimpleNamespace(apply_gpu_power_settings=False), state)
        assert "pass --apply-gpu-power-settings to apply them again" in error

    def test_an_apply_refusal_reaches_the_launch_error(self, monkeypatch, capsys):
        def _refuse(*a, **kw):
            raise GpuPowerSettingsError("needs an integer ROCR_VISIBLE_DEVICES mask")

        monkeypatch.setattr(gpu_power_apply, "apply_for_session", _refuse)
        args = SimpleNamespace(apply_gpu_power_settings=True, gpu_power_cap_w=1000.0)
        with pytest.raises(SystemExit):
            bootstrap.resolve_launch_gpu_power(args, nodes=1, owner="s")
        assert "cannot apply the declared GPU power settings: needs an integer" in capsys.readouterr().err
