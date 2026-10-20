# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--apply-gpu-power-settings``: set once on the masked cards, recorded first, restored however the session ends."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
import threading
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
    _RestoreGuard,
    apply_declared_gpu_power_settings,
    orphaned_power_settings_warning,
)
from hyperloom.inference_optimizer.cli.parser import _build_parser

_MASK = {4, 5, 6, 7}


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


@pytest.fixture(autouse=True)
def guard(monkeypatch):
    """Every test gets its own restore guard, so an apply arms nothing real: no atexit hook, no signal handler kept."""
    monkeypatch.setattr("atexit.register", lambda fn, *a: None)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    guard = _RestoreGuard()
    monkeypatch.setattr(gpu_power_apply, "_GUARD", guard)
    yield guard
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _install(monkeypatch, ledger, *, observed=None, vram=None, mask=_MASK, smi=None, modes=None) -> _AmdSmi:
    """What the apply reads (cards, VRAM, mask, partition modes, ledger); *smi* answers every ``amd-smi`` command it runs."""
    smi = smi or _AmdSmi()
    monkeypatch.setattr(gpu_power_apply, "read_partition_modes", lambda: dict.fromkeys(range(8), "SPX") | (modes or {}))
    reads = list(observed or [_cards(), _cards()])
    resident = dict.fromkeys(range(8), 284.0) if vram is None else vram
    monkeypatch.setattr(
        gpu_power_apply, "read_gpu_power_settings", lambda: reads.pop(0) if len(reads) > 1 else reads[0]
    )
    monkeypatch.setattr(gpu_power_apply, "read_resident_vram_mb", lambda: resident)
    monkeypatch.setattr(gpu_power_apply, "visible_gpu_indices", lambda: mask)
    monkeypatch.setattr(gpu_power_apply, "GPU_POWER_LEDGER_DIR", ledger)
    monkeypatch.setattr(gpu_power_control.subprocess, "run", smi)
    return smi


def _apply(monkeypatch, tmp_path, *, nodes=1, power_cap_w=1000.0, perf_level=None, **install_kw):
    smi = _install(monkeypatch, tmp_path, **install_kw)
    record = apply_declared_gpu_power_settings(
        DeclaredPowerSettings(power_cap_w=power_cap_w, perf_level=perf_level or ""), nodes=nodes, owner="sess"
    )
    return record, smi


def _warning(monkeypatch, ledger, cards, mask=_MASK) -> str:
    monkeypatch.setattr(gpu_power_apply, "GPU_POWER_LEDGER_DIR", ledger)
    monkeypatch.setattr(gpu_power_apply, "read_gpu_power_settings", lambda: cards)
    monkeypatch.setattr(gpu_power_apply, "visible_gpu_indices", lambda: mask)
    return orphaned_power_settings_warning()


class TestSetCommands:
    def test_cards_sharing_a_value_are_set_in_one_call(self):
        commands = _set_commands({4: {"power_cap_w": 1000, "perf_level": "auto"}, 5: {"power_cap_w": 1000}})
        assert ["set", "-g", "4", "5", "-o", "ppt0", "1000"] in commands
        assert ["set", "-g", "4", "-l", "AUTO"] in commands

    def test_apply_raises_on_a_failed_set(self, monkeypatch):
        monkeypatch.setattr(gpu_power_control.subprocess, "run", _AmdSmi(fail_on="ppt0"))
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            apply_gpu_power_settings({4: {"power_cap_w": 1000}})

    def test_a_failure_names_amd_smis_reason_not_its_traceback(self, monkeypatch):
        def _traceback(argv, **kwargs):
            err = "Traceback (most recent call last):\n  ...\nPermissionError: Command requires elevation\n"
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

        monkeypatch.setattr(gpu_power_control.subprocess, "run", _traceback)
        with pytest.raises(GpuPowerSettingsError, match=r"exited 1: PermissionError: Command requires elevation$"):
            apply_gpu_power_settings({4: {"power_cap_w": 1000}})

    def test_restore_attempts_every_card_even_after_one_fails(self, monkeypatch):
        smi = _AmdSmi(fail_on="1400")
        monkeypatch.setattr(gpu_power_control.subprocess, "run", smi)
        problems = restore_gpu_power_settings({4: {"power_cap_w": 1400}, 5: {"power_cap_w": 1300}})
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

    def test_a_lease_waits_out_the_brief_shared_lock_of_a_warning(self, tmp_path):
        """A launch without the flag reads the record under a shared lock; a concurrent flagged launch must not refuse."""
        PowerSettingsLease(tmp_path, {4}).release()
        with open(tmp_path / "gpu4.json", encoding="utf-8") as reader:
            fcntl.flock(reader, fcntl.LOCK_SH)
            timer = threading.Timer(0.3, lambda: fcntl.flock(reader, fcntl.LOCK_UN))
            timer.start()
            try:
                PowerSettingsLease(tmp_path, {4}).release()
            finally:
                timer.join()

    def test_a_lock_error_other_than_held_closes_the_record(self, tmp_path, monkeypatch):
        def _no_locks(fd, op):
            raise OSError(37, "no locks")

        monkeypatch.setattr(gpu_power_control.fcntl, "flock", _no_locks)
        before = set(os.listdir("/proc/self/fd"))
        with pytest.raises(GpuPowerSettingsError, match="no locks"):
            PowerSettingsLease(tmp_path, {4})
        assert set(os.listdir("/proc/self/fd")) <= before, "the record's descriptor is closed"

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

    def test_records_are_readable_by_all_and_writable_only_by_the_account_that_applies(self, tmp_path):
        ledger = tmp_path / "ledger"
        PowerSettingsLease(ledger, {4}).release()
        assert ledger.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_ISVTX) == 0
        assert (ledger / "gpu4.json").stat().st_mode & 0o777 == 0o644

    def test_the_ledger_stays_readable_by_every_account_under_a_strict_umask(self, tmp_path):
        """A root launch under umask 077 must not leave a ledger the warning of a non-root launch cannot read."""
        previous = os.umask(0o077)
        try:
            PowerSettingsLease(tmp_path / "var" / "ledger", {4}).release()
        finally:
            os.umask(previous)
        for level in (tmp_path / "var", tmp_path / "var" / "ledger"):
            assert level.stat().st_mode & 0o777 == 0o755

    def test_a_ledger_made_by_hand_too_narrow_is_opened_up_for_the_warning(self, tmp_path):
        """A directory an operator made for the host mount under a strict umask must not hide records from others."""
        ledger = tmp_path / "var" / "ledger"
        ledger.mkdir(parents=True, mode=0o700)
        os.chmod(tmp_path / "var", 0o700)
        PowerSettingsLease(ledger, {4}).release()
        for level in (tmp_path / "var", ledger):
            assert level.stat().st_mode & 0o777 == 0o755

    def test_a_ledger_this_account_cannot_reach_reads_as_no_records(self, tmp_path):
        """The warning of a launch without the flag must not crash on a ledger it may not search."""
        ledger = tmp_path / "var" / "ledger"
        PowerSettingsLease(ledger, {4}).release()
        os.chmod(tmp_path / "var", 0o000)
        try:
            assert orphaned_power_records(ledger) == {}
        finally:
            os.chmod(tmp_path / "var", 0o700)

    @pytest.mark.parametrize("mode", [0o777, 0o1777], ids=["world-writable", "sticky"])
    def test_a_directory_other_accounts_can_plant_records_in_is_refused(self, tmp_path, mode):
        """A privileged restore sets whatever a record says, so a planted record must never be read."""
        ledger = tmp_path / "ledger"
        ledger.mkdir()
        os.chmod(ledger, mode)
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

    def test_a_hard_linked_record_is_refused_and_its_target_left_alone(self, tmp_path):
        """A root restore writing through a planted hard link would overwrite whatever file it points at."""
        target = tmp_path / "elsewhere"
        target.write_text("keep")
        ledger = tmp_path / "ledger"
        ledger.mkdir(mode=0o755)
        os.link(target, ledger / "gpu4.json")
        with pytest.raises(GpuPowerSettingsError, match="not a plain file"):
            PowerSettingsLease(ledger, {4})
        assert target.read_text() == "keep"

    def test_the_ledger_is_host_local_and_ignores_the_workspace(self):
        assert str(GPU_POWER_LEDGER_DIR) == "/var/tmp/hyperloom/gpu_power_settings"


class TestApplyDeclared:
    def test_sets_only_the_masked_cards_and_restores_them(self, tmp_path, monkeypatch, guard):
        record, smi = _apply(monkeypatch, tmp_path)
        assert record["gpus"] == [4, 5, 6, 7]
        assert record["originals"]["4"] == {"power_cap_w": 1400.0}, "only what is changed is recorded"
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1000"]]
        assert orphaned_power_records(tmp_path) == {}, "held while the session lives"
        assert guard.run() == []
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1400"]
        assert orphaned_power_records(tmp_path) == {}, "cleared once restored"
        assert guard.run() == [] and len(smi.calls) == 2, "restore is idempotent"

    @pytest.mark.parametrize("mask", [None], ids=["no-integer-mask"])
    def test_without_an_integer_mask_nothing_is_set_or_recorded(self, tmp_path, mask, monkeypatch):
        with pytest.raises(GpuPowerSettingsError, match="integer ROCR_VISIBLE_DEVICES"):
            _apply(monkeypatch, tmp_path, mask=mask)
        assert list(tmp_path.iterdir()) == []

    def test_a_masked_card_amd_smi_does_not_report_refuses(self, tmp_path, monkeypatch):
        with pytest.raises(GpuPowerSettingsError, match=r"masked GPU\(s\) 9"):
            _apply(monkeypatch, tmp_path, mask={4, 9})

    def test_the_record_is_written_before_anything_is_set(self, tmp_path, monkeypatch):
        seen: list = []
        _install(monkeypatch, tmp_path, mask={4})

        def _apply_and_look(_settings):
            seen.append(orphaned_power_records(tmp_path))
            seen.append(json.loads((tmp_path / "gpu4.json").read_text()))
            raise GpuPowerSettingsError("boom")

        monkeypatch.setattr(gpu_power_apply, "apply_gpu_power_settings", _apply_and_look)
        with pytest.raises(GpuPowerSettingsError, match="boom"):
            apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert seen[0] == {}, "held, so not visible as an orphan"
        assert seen[1]["original"] == {"power_cap_w": 1400.0}, "the originals are on disk before the first set"

    def test_a_failed_set_restores_and_refuses(self, tmp_path, monkeypatch):
        smi = _AmdSmi(fail_on="1000")
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            _apply(monkeypatch, tmp_path, smi=smi)
        assert smi.calls[-1][-1] == "1400", "the originals are put back"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_card_with_someone_elses_model_is_refused_untouched(self, tmp_path, monkeypatch):
        vram = dict.fromkeys(range(8), 284.0) | {6: 260_000.0}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"GPU\(s\) 6 already hold"):
            _apply(monkeypatch, tmp_path, vram=vram, smi=smi)
        assert smi.calls == []

    @pytest.mark.parametrize("modes", [{6: "CPX"}, {6: "DPX"}, None], ids=["cpx", "dpx", "unreadable"])
    def test_a_partitioned_card_is_refused_untouched(self, tmp_path, monkeypatch, modes):
        """A cap is per socket: setting one partition's card would cap the sibling partitions other sessions use."""
        smi = _install(monkeypatch, tmp_path, modes=modes)
        if modes is None:

            def _unreadable():
                raise gpu_power_apply.PartitionError("amd-smi partition reported no compute-partition state")

            monkeypatch.setattr(gpu_power_apply, "read_partition_modes", _unreadable)
        with pytest.raises(GpuPowerSettingsError, match="compute-partition"):
            apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert smi.calls == [] and list(tmp_path.iterdir()) == []

    def test_a_card_the_partition_reading_does_not_list_is_refused(self, tmp_path, monkeypatch):
        smi = _install(monkeypatch, tmp_path)
        monkeypatch.setattr(gpu_power_apply, "read_partition_modes", lambda: {4: "SPX", 5: "SPX", 7: "SPX"})
        with pytest.raises(GpuPowerSettingsError, match=r"GPU\(s\) 6 are not in SPX"):
            apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert smi.calls == []

    def test_a_fractional_cap_is_refused_before_anything_is_leased(self, tmp_path, monkeypatch):
        """amd-smi set takes whole watts; finding that out after the lease and records would be too late."""
        with pytest.raises(GpuPowerSettingsError, match="whole number of watts; 1000.5 W"):
            _apply(monkeypatch, tmp_path, power_cap_w=1000.5)
        assert list(tmp_path.iterdir()) == []

    def test_a_card_whose_vram_cannot_be_read_is_refused_untouched(self, tmp_path, monkeypatch):
        vram = {gpu: 284.0 for gpu in range(8) if gpu != 5}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"VRAM use of GPU\(s\) 5 could not be read"):
            _apply(monkeypatch, tmp_path, vram=vram, smi=smi)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_an_orphan_is_restored_first_and_its_originals_are_the_true_ones(self, tmp_path, monkeypatch, guard):
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        record, smi = _apply(monkeypatch, tmp_path, observed=[_cards(cap=900.0), _cards(cap=900.0), _cards()])
        assert record["recovered_from_orphan"] == [4]
        assert smi.calls[0] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]
        assert record["originals"]["4"]["power_cap_w"] == 1400.0
        guard.run()

    def test_a_stop_signal_cannot_cut_an_orphans_restore_short(self, tmp_path, monkeypatch):
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        signal.signal(signal.SIGTERM, _RestoreGuard._exit_on_signal)

        def _signal_then_restore(originals):
            os.kill(os.getpid(), signal.SIGTERM)
            return restore_gpu_power_settings(originals)

        smi = _install(monkeypatch, tmp_path, observed=[_cards(cap=900.0), _cards(cap=900.0), _cards()])
        monkeypatch.setattr(gpu_power_apply, "restore_gpu_power_settings", _signal_then_restore)
        record = apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert record["recovered_from_orphan"] == [4]
        assert smi.calls[0] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]

    def test_a_card_held_by_a_live_session_is_refused(self, tmp_path, monkeypatch):
        live = PowerSettingsLease(tmp_path, {5})
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match="another running Hyperloom session"):
            _apply(monkeypatch, tmp_path, smi=smi)
        assert smi.calls == []
        live.release()

    @pytest.mark.parametrize(
        ("kw", "needle"),
        [({"nodes": 2}, "multi-node"), ({"power_cap_w": None, "perf_level": None}, "needs --gpu-power-cap-w")],
    )
    def test_refusals_before_touching_anything(self, tmp_path, kw, needle, monkeypatch):
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=needle):
            _apply(monkeypatch, tmp_path, smi=smi, **kw)
        assert smi.calls == []

    def test_both_declared_are_both_restored(self, tmp_path, monkeypatch, guard):
        record, smi = _apply(monkeypatch, tmp_path, perf_level="determinism")
        assert record["originals"]["4"] == {"power_cap_w": 1400.0, "perf_level": "auto"}, "auto resets the limit"
        assert ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "DETERMINISM"] in smi.calls
        guard.run()
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "5", "6", "7", "-l", "AUTO"]

    @pytest.mark.parametrize(
        ("level", "kw", "needle"),
        [
            ("determinism", {"power_cap_w": None, "perf_level": "manual"}, "determinism perf level"),
            ("manual", {"power_cap_w": None, "perf_level": "auto"}, "hand-tuned clock limits"),
        ],
    )
    def test_a_starting_level_a_restore_could_not_bring_back_is_refused_untouched(
        self, tmp_path, level, kw, needle, monkeypatch
    ):
        cards = _cards(level=level)
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=needle):
            _apply(monkeypatch, tmp_path, observed=[cards, cards], smi=smi, **kw)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_a_cap_alone_on_a_determinism_card_is_allowed(self, tmp_path, monkeypatch, guard):
        """The perf level is not touched, so the determinism soft-max is never at risk."""
        cards = _cards(level="determinism")
        record, smi = _apply(monkeypatch, tmp_path, observed=[cards, cards])
        assert smi.calls == [["amd-smi", "set", "-g", "4", "5", "6", "7", "-o", "ppt0", "1000"]]
        assert guard.run() == []

    def test_a_restore_whose_commands_fail_but_whose_cards_are_back_clears_the_record(self, tmp_path, monkeypatch):
        """Without set privilege the apply changes nothing and the restore's sets fail too; the cards decide."""
        smi = _AmdSmi(fail_on="ppt0")
        with pytest.raises(GpuPowerSettingsError, match="permission denied"):
            _apply(monkeypatch, tmp_path, smi=smi)
        assert orphaned_power_records(tmp_path) == {}, "the cards read back at their originals, so nothing to recover"

    def test_only_the_card_that_did_not_come_back_keeps_its_record(self, tmp_path, monkeypatch, guard):
        """Clearing every record over one stuck card would leave false warnings about the cards that are back."""
        restored = _cards()
        restored[4] = {**restored[4], "perf_level": "manual"}
        record, _ = _apply(
            monkeypatch, tmp_path, power_cap_w=None, perf_level="manual", observed=[_cards(), _cards(), restored]
        )
        problems = guard.run()
        assert problems and all("GPU 4" in p for p in problems)
        assert sorted(orphaned_power_records(tmp_path)) == [4]

    def test_a_stale_record_is_cleared_not_restored(self, tmp_path, monkeypatch, guard):
        """A card an operator has reset since is no longer the dead session's: restoring its original would undo them."""
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        operator_set = _cards()
        operator_set[4] = {**operator_set[4], "power_cap_w": 1200.0}
        record, smi = _apply(monkeypatch, tmp_path, observed=[operator_set, operator_set])
        assert "recovered_from_orphan" not in record
        assert ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"] not in smi.calls
        assert record["originals"]["4"] == {"power_cap_w": 1200.0}, "the operator's value is what comes back"
        guard.run()

    def test_a_partly_restored_card_gets_the_rest_put_back(self, tmp_path, monkeypatch, guard):
        """A restore that put back the cap but not the level leaves the level still the dead session's to restore."""
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record(
            {4: {"power_cap_w": 1400, "perf_level": "auto"}},
            applied={"power_cap_w": 1000, "perf_level": "high"},
            owner="dead",
        )
        dead.release()
        half_back = _cards()
        half_back[4] = {"power_cap_w": 1400.0, "perf_level": "high"}
        warning = _warning(monkeypatch, tmp_path, half_back)
        assert "GPU 4 at perf level high from session dead, originally perf level auto" in warning, "the cap is back"
        record, smi = _apply(monkeypatch, tmp_path, observed=[half_back, half_back, _cards()])
        assert record["recovered_from_orphan"] == [4]
        assert smi.calls[0] == ["amd-smi", "set", "-g", "4", "-l", "AUTO"], "only the setting still applied is restored"
        guard.run()

    def test_a_card_already_at_the_declared_level_is_left_alone(self, tmp_path, monkeypatch, guard):
        """Asking a manual card to stay manual changes nothing, so it needs no refusal, set or restore."""
        manual = _cards(level="manual")
        record, smi = _apply(monkeypatch, tmp_path, perf_level="manual", observed=[manual, manual])
        assert record["originals"]["4"] == {"power_cap_w": 1400.0}
        assert all("-l" not in call for call in smi.calls)
        guard.run()
        assert all("-l" not in call for call in smi.calls)

    def test_an_older_record_key_does_not_break_recovery_messages(self, tmp_path, monkeypatch, guard):
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400, "sclk_max_mhz": 2400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        record, _ = _apply(monkeypatch, tmp_path, observed=[_cards(cap=900.0), _cards(cap=900.0), _cards()])
        assert record["recovered_from_orphan"] == [4]
        guard.run()

    def test_a_record_whose_card_cannot_be_read_is_kept_and_reported(self, tmp_path, monkeypatch):
        """Not being able to tell is not proof the card moved: clearing the record could lose the only way back."""
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        unread = _cards()
        unread[4] = {**unread[4], "power_cap_w": None}
        assert "GPU 4 may still be at cap 900 W" in _warning(monkeypatch, tmp_path, unread)
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"the record is kept. Check those cards .*gpu4\.json"):
            _apply(monkeypatch, tmp_path, observed=[unread, unread], smi=smi)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path)[4]["applied"] == {"power_cap_w": 900}

    def test_a_setting_a_card_already_has_is_neither_set_nor_recorded(self, tmp_path, monkeypatch):
        """Otherwise a SIGKILL would leave a record whose applied value equals its original, warning forever."""
        at_cap = _cards(cap=1000.0)
        record, smi = _apply(monkeypatch, tmp_path, observed=[at_cap, at_cap])
        assert smi.calls == []
        assert record["originals"]["4"] == {}

    def test_a_record_that_cannot_be_written_refuses_cleanly(self, tmp_path, monkeypatch):
        def _full_disk(*args):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(gpu_power_control.os, "pwrite", _full_disk)
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match="No space left"):
            _apply(monkeypatch, tmp_path, smi=smi)
        assert smi.calls == [], "nothing is set without a record"

    def test_a_record_that_cannot_be_cleared_is_reported_and_kept(self, tmp_path, monkeypatch, guard):
        record, _ = _apply(monkeypatch, tmp_path)

        def _read_only(*args):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(gpu_power_control.os, "ftruncate", _read_only)
        problems = guard.run()
        assert any("Read-only file system" in p for p in problems)
        assert sorted(orphaned_power_records(tmp_path)) == [4, 5, 6, 7], "the next launch can still recover them"

    def test_a_restore_the_card_did_not_take_is_reported_and_its_record_kept(self, tmp_path, monkeypatch, guard):
        """amd-smi can exit 0 without the card taking the value; the record must survive for the next launch."""
        stuck_manual = _cards(level="manual")
        record, _ = _apply(
            monkeypatch, tmp_path, power_cap_w=None, perf_level="manual", observed=[_cards(), _cards(), stuck_manual]
        )
        problems = guard.run()
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
        self, tmp_path, observed, perf_level, unreadable_gpu, monkeypatch
    ):
        """Restore skips a value it never read, so applying over one would leave the card changed after the session."""
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=rf"GPU\(s\) {unreadable_gpu} .*nothing was applied"):
            _apply(monkeypatch, tmp_path, observed=[observed, observed], perf_level=perf_level, smi=smi)
        assert smi.calls == []
        assert orphaned_power_records(tmp_path) == {}

    def test_a_card_missing_from_the_reread_after_orphan_recovery_refuses(self, tmp_path, monkeypatch):
        """The re-read after restoring an orphan can drop a card; nothing restores what was not read."""
        dead = PowerSettingsLease(tmp_path, {4})
        dead.record({4: {"power_cap_w": 1400}}, applied={"power_cap_w": 900}, owner="dead")
        dead.release()
        reread = {gpu: row for gpu, row in _cards().items() if gpu != 7}
        smi = _AmdSmi()
        with pytest.raises(GpuPowerSettingsError, match=r"GPU\(s\) 7 .*nothing was applied"):
            _apply(monkeypatch, tmp_path, observed=[_cards(cap=900.0), _cards(cap=900.0), reread], smi=smi)
        assert smi.calls == [["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"]], "only the orphan was restored"


def test_orphan_warning_names_the_card_and_its_original(tmp_path, monkeypatch):
    lease = PowerSettingsLease(tmp_path, {4})
    lease.record({4: {"power_cap_w": 1400, "perf_level": "auto"}}, applied={"power_cap_w": 900}, owner="dead")
    lease.release()
    warning = _warning(monkeypatch, tmp_path, _cards(cap=900.0), mask={4, 5})
    assert "GPU 4 at cap 900 W from session dead, originally cap 1400 W." in warning, "only what is still applied"
    assert "--apply-gpu-power-settings" in warning
    assert _warning(monkeypatch, tmp_path, _cards(), mask={4, 5}) == "", "put back since: nothing to warn about"
    assert _warning(monkeypatch, tmp_path, _cards(cap=900.0), mask={5}) == "", "not this session's card"


def test_the_perf_level_is_one_amd_smi_accepts(capsys):
    parser = _build_parser()
    base = ["optimize", "--model", "m", "--framework", "vllm"]
    assert parser.parse_args([*base, "--gpu-perf-level", "DETERMINISM"]).gpu_perf_level == "determinism"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--gpu-perf-level", "turbo"])
    assert "invalid choice: 'turbo'" in capsys.readouterr().err


def test_parser_flag_is_off_by_default():
    parser = _build_parser()
    base = ["optimize", "--model", "m", "--framework", "vllm"]
    assert parser.parse_args(base).apply_gpu_power_settings is False
    args = parser.parse_args([*base, "--apply-gpu-power-settings", "--gpu-power-cap-w", "1000"])
    assert args.apply_gpu_power_settings is True and args.gpu_power_cap_w == 1000.0


class TestRestoreGuard:
    """The restore runs on the way out however the session leaves, short of SIGKILL."""

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
    def test_an_interrupt_during_the_set_restores(self, guard, tmp_path, interrupt, monkeypatch):
        """A card is changed as soon as its set returns, before the next set; the restore must already be armed."""
        smi = _install(monkeypatch, tmp_path, mask={4})

        def _set_then_interrupt(settings):
            apply_gpu_power_settings(settings)
            if interrupt == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            raise KeyboardInterrupt

        monkeypatch.setattr(gpu_power_apply, "apply_gpu_power_settings", _set_then_interrupt)
        with pytest.raises((SystemExit, KeyboardInterrupt)):
            with gpu_power_apply.gpu_power_restore_guard():
                apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert smi.calls[-1] == ["amd-smi", "set", "-g", "4", "-o", "ppt0", "1400"], "the original is put back"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_stop_signal_during_the_restore_after_a_failed_set_cannot_cut_it_short(
        self, guard, tmp_path, monkeypatch
    ):
        """The restore after a failed set runs inside apply, before the session's finally; it holds the signals too."""
        smi = _install(monkeypatch, tmp_path, mask={4}, smi=_AmdSmi(fail_on="1000"))

        def _signal_then_restore(originals):
            os.kill(os.getpid(), signal.SIGTERM)
            return restore_gpu_power_settings(originals)

        monkeypatch.setattr(gpu_power_apply, "restore_gpu_power_settings", _signal_then_restore)
        with pytest.raises(GpuPowerSettingsError):
            apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert smi.calls[-1][-1] == "1400", "the original was put back despite the signal"
        assert orphaned_power_records(tmp_path) == {}

    def test_a_failed_set_restores_once_and_disarms(self, guard, tmp_path, monkeypatch):
        smi = _install(monkeypatch, tmp_path, mask={4}, smi=_AmdSmi(fail_on="1000"))
        with pytest.raises(GpuPowerSettingsError):
            apply_declared_gpu_power_settings(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")
        assert signal.getsignal(signal.SIGTERM) is not _RestoreGuard._exit_on_signal, "the stop signals are given back"
        assert guard.run() == []
        assert [call[-1] for call in smi.calls] == ["1000", "1400"], "restored once, by the failed apply"

    def test_an_exit_after_applying_restores(self, guard):
        restored: list[bool] = []

        def _exit_after_applying():
            with gpu_power_apply.gpu_power_restore_guard():
                guard.arm(lambda: restored.append(True) or [])
                sys.exit(2)

        with pytest.raises(SystemExit) as exited:
            _exit_after_applying()
        assert exited.value.code == 2
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

    def test_the_cli_runs_the_session_inside_the_guard(self, guard, monkeypatch):
        from hyperloom.inference_optimizer import cli

        restored: list[bool] = []

        async def _session(args):
            gpu_power_apply._GUARD.arm(lambda: restored.append(True) or [])
            raise SystemExit(3)

        monkeypatch.setattr(cli, "_run_optimize", _session)
        saved = dict(os.environ)
        try:
            with pytest.raises(SystemExit) as exc:
                cli.main(["optimize"])
        finally:
            os.environ.clear()
            os.environ.update(saved)
        assert (exc.value.code, restored) == (3, [True])

    def test_the_signals_are_held_before_the_restore_is_taken(self, guard, monkeypatch):
        """A SIGTERM between taking the restore and holding the signals would drop it with nothing left to run it."""
        guard.arm(list)
        held_with_restore: list[bool] = []
        real = signal.signal

        def _spy(sig, handler):
            held_with_restore.append(guard._restore is not None)
            return real(sig, handler)

        monkeypatch.setattr(gpu_power_apply.signal, "signal", _spy)
        guard.run()
        assert held_with_restore[0] is True

    def test_the_restore_runs_once_and_a_failure_is_loud(self, guard, capsys):
        calls: list[int] = []
        guard.arm(lambda: calls.append(1) or ["amd-smi set exited 1"])
        guard.run()
        guard.run()
        assert calls == [1]
        assert "could not restore the GPU power settings" in capsys.readouterr().err


class TestSessionWiring:
    def test_a_resume_stops_its_own_left_over_servers_before_it_applies(self, monkeypatch, tmp_path):
        """After a SIGKILL the session's server outlives it; the apply must not take it for another tenant's model."""
        from hyperloom.inference_optimizer import cli
        from hyperloom.orchestrator.actions.executors import _server_lifecycle

        steps: list[str] = []
        monkeypatch.setattr(cli, "_restore_partition_shape_from_state", lambda args, state: None)
        monkeypatch.setattr(_server_lifecycle, "reap_orphaned_servers", lambda d: steps.append(f"reap {d}") or [4242])

        def _resume(args, state):
            steps.append("apply")
            raise SystemExit(9)

        monkeypatch.setattr(cli, "resume_gpu_power_settings", _resume)
        lock = SimpleNamespace(session_dir=tmp_path, release=lambda: None)
        with pytest.raises(SystemExit):
            cli._restore_platform_state(SimpleNamespace(apply_gpu_power_settings=True), SimpleNamespace(), lock)
        assert steps == [f"reap {tmp_path}", "apply"]

    def test_a_launch_without_root_is_refused_before_any_record_exists(self, monkeypatch):
        """amd-smi set needs root; records an unprivileged attempt left would make a later root launch refuse."""
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        monkeypatch.setattr(
            gpu_power_apply,
            "apply_declared_gpu_power_settings",
            lambda *a, **kw: pytest.fail("must not take the lease without root"),
        )
        with pytest.raises(GpuPowerSettingsError, match="needs root"):
            gpu_power_apply.apply_for_session(DeclaredPowerSettings(power_cap_w=1000.0), nodes=1, owner="s")

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
