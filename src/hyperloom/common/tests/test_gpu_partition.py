# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the compute-partition lever.

``amd-smi`` is faked throughout, so these run on a CPU host and on a card that
must not be repartitioned by a test suite. The cases that matter are the two
silent-wrong-answer paths: a set that reports success without taking effect, and
a feasibility check that passes one stream where two will run.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from hyperloom.common import gpu_partition
from hyperloom.common.gpu_partition import (
    DEFAULT_MODE,
    PartitionError,
    fits_in_partition,
    layout_for,
    parse_modes,
    partition_device_predicate,
    partitioned,
    read_partition_mode,
    read_partition_modes,
    set_partition_mode,
)


class _FakeSmi:
    """An ``amd-smi`` whose partition state a test can drive.

    ``stage_only`` reproduces the mode change that is accepted and then does not
    take effect, which is the failure the read-back exists to catch.
    """

    def __init__(self, modes: dict[int, str], *, stage_only: bool = False, set_rc: int = 0):
        self.modes = dict(modes)
        self.stage_only = stage_only
        self.set_rc = set_rc
        self.set_calls: list[tuple[int, str]] = []
        self.argv: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        if "set" in cmd and "--compute-partition" in cmd:
            self.argv.append(list(cmd))
            gpu_id = int(cmd[cmd.index("-g") + 1])
            mode = cmd[cmd.index("--compute-partition") + 1]
            self.set_calls.append((gpu_id, mode))
            if self.set_rc == 0 and not self.stage_only:
                self.modes[gpu_id] = mode
            return subprocess.CompletedProcess(cmd, self.set_rc, "", "busy" if self.set_rc else "")
        rows = [
            {"gpu_id": gid, "memory": "NPS1", "accelerator_type": mode, "partition_id": "0"}
            for gid, mode in sorted(self.modes.items())
        ]
        payload = json.dumps({"current_partition": rows})
        return subprocess.CompletedProcess(cmd, 0, payload, "")


@pytest.fixture
def smi(monkeypatch):
    """Install a fake amd-smi with all eight cards in SPX."""
    fake = _FakeSmi({i: "SPX" for i in range(8)})
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def test_parse_modes_canonicalizes_dedupes_and_keeps_order():
    assert parse_modes("spx, dpx ,cpx,dpx") == ("SPX", "DPX", "CPX")
    assert parse_modes(["CPX", "spx"]) == ("CPX", "SPX")


def test_parse_modes_off_by_default():
    assert parse_modes(None) == ()
    assert parse_modes("") == ()
    assert parse_modes(" , ") == ()


def test_parse_modes_refuses_unknown_mode_at_parse_time():
    with pytest.raises(PartitionError, match="unknown compute-partition mode 'NPS2'"):
        parse_modes("spx,nps2")


@pytest.mark.parametrize(
    ("mode", "partitions", "cu"),
    [("SPX", 1, 256), ("DPX", 2, 128), ("QPX", 4, 64), ("CPX", 8, 32)],
)
def test_layout_matches_mi355x_hardware(mode, partitions, cu):
    layout = layout_for("mi355x", mode, hbm_gib=288.0)
    assert (layout.partitions, layout.cu_per_partition) == (partitions, cu)
    assert layout.gib_per_partition == pytest.approx(288.0 / partitions)


def test_layout_derives_cu_from_the_board_not_a_second_table():
    # mi300x carries 304 CU, so its CPX partitions are 38 CU, not mi355x's 32.
    assert layout_for("mi300x", "CPX").cu_per_partition == 38


def test_layout_refuses_unknown_board_and_mode():
    with pytest.raises(PartitionError, match="unknown gpu_type"):
        layout_for("mi999x", "SPX")
    with pytest.raises(PartitionError, match="unknown compute-partition mode"):
        layout_for("mi355x", "OPX")


def test_fits_in_partition_gates_on_the_streams_that_will_run():
    cpx = layout_for("mi355x", "CPX", hbm_gib=288.0)
    # 20 GiB fits a 36 GiB partition alone and exhausts it in pairs. Gating on
    # the single-stream figure is what lets a doomed config be declared feasible.
    assert fits_in_partition(20.0, cpx, streams_per_partition=1)
    assert not fits_in_partition(20.0, cpx, streams_per_partition=2)
    # The 6-view footprint that did run at two streams per CPX partition.
    assert fits_in_partition(8.3, cpx, streams_per_partition=2)


def test_fits_in_partition_is_permissive_when_capacity_is_unknown():
    assert fits_in_partition(999.0, layout_for("mi355x", "CPX"), streams_per_partition=8)


def test_partition_device_predicate_selects_by_cu_not_index():
    is_dpx = partition_device_predicate(layout_for("mi355x", "DPX").cu_per_partition)
    # Under DPX on one card of eight, devices 0-6 are whole 256-CU cards and the
    # partitions are the two 128-CU devices enumerated after them.
    enumerated = [256] * 7 + [128, 128]
    assert [i for i, cu in enumerate(enumerated) if is_dpx(cu)] == [7, 8]


def test_read_partition_modes_parses_amd_smi_json(smi):
    assert read_partition_modes() == {i: "SPX" for i in range(8)}
    assert read_partition_mode(3) == "SPX"


def test_read_partition_mode_refuses_absent_gpu(smi):
    with pytest.raises(PartitionError, match="no state for GPU 99"):
        read_partition_mode(99)


def test_set_partition_mode_verifies_the_read_back(smi):
    assert set_partition_mode(0, "cpx") == "CPX"
    assert smi.set_calls == [(0, "CPX")]


def test_set_partition_mode_refuses_a_change_that_did_not_take_effect(monkeypatch):
    # The NPS2 failure shape: exit code 0, mode unchanged. Trusting the exit
    # code here attributes a whole measurement to a mode that never applied.
    fake = _FakeSmi({0: "SPX"}, stage_only=True)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(PartitionError, match="reports SPX after being set to CPX"):
        set_partition_mode(0, "CPX")


#: Verbatim stderr from a card that refused to repartition while a container was
#: still shutting down. The retry matches on the status code because the prose
#: half of this message is not stable across amd-smi builds.
BUSY_STDERR = (
    "amdsmi.amdsmi_exception.AmdSmiLibraryException: Error code:\n"
    "\t30 | AMDSMI_STATUS_BUSY - Device busy\n\n"
    "The above exception was the direct cause of the following exception:\n\n"
    "ValueError: Unable to set accelerator partition to CPX on GPU ID: 0 BDF:0000:09:00.0."
)


def _busy_until(succeed_on: int, fake: _FakeSmi):
    """An amd-smi that refuses ``succeed_on - 1`` sets, then behaves."""
    attempts = {"n": 0}

    def run(cmd, **kwargs):
        if "set" in cmd and "--compute-partition" in cmd:
            attempts["n"] += 1
            if attempts["n"] < succeed_on:
                return subprocess.CompletedProcess(cmd, 1, "", BUSY_STDERR)
        return fake(cmd, **kwargs)

    return run, attempts


def test_set_partition_mode_waits_for_a_busy_card_to_drain(monkeypatch):
    # docker rm -f returns before the runtime has released the device, so the
    # repartition that follows it races teardown. The refusal clears on its own.
    fake = _FakeSmi({0: "SPX"})
    run, attempts = _busy_until(3, fake)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(gpu_partition.time, "sleep", lambda _s: None)

    assert set_partition_mode(0, "CPX") == "CPX"
    assert attempts["n"] == 3


def test_set_partition_mode_gives_up_on_a_card_that_never_drains(monkeypatch):
    fake = _FakeSmi({0: "SPX"})
    run, attempts = _busy_until(10**6, fake)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(PartitionError, match="still held processes"):
        set_partition_mode(0, "CPX", drain_timeout_s=0.0)
    assert attempts["n"] == 1


def test_set_partition_mode_does_not_retry_a_permanent_failure(monkeypatch):
    # Retrying an unknown flag or a denied permission for two minutes turns a
    # clear error into a hang, so only a busy card is waited out.
    fake = _FakeSmi({0: "SPX"})
    attempts = {"n": 0}

    def denied(cmd, **kwargs):
        if "set" in cmd and "--compute-partition" in cmd:
            attempts["n"] += 1
            return subprocess.CompletedProcess(cmd, 1, "", "permission denied")
        return fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", denied)
    with pytest.raises(PartitionError, match="permission denied"):
        set_partition_mode(0, "CPX", drain_timeout_s=60.0)
    assert attempts["n"] == 1


def test_restore_also_waits_out_a_busy_card(monkeypatch):
    # The restore is the path that matters most: a failure there leaves a shared
    # card in a mode its next tenant did not ask for.
    fake = _FakeSmi({0: "SPX"})
    attempts = {"n": 0}

    def busy_on_restore(cmd, **kwargs):
        if "set" in cmd and "--compute-partition" in cmd:
            attempts["n"] += 1
            # Refuse the first restore attempt only.
            if attempts["n"] == 2:
                return subprocess.CompletedProcess(cmd, 1, "", BUSY_STDERR)
        return fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", busy_on_restore)
    monkeypatch.setattr(gpu_partition.time, "sleep", lambda _s: None)
    with partitioned(0, "QPX"):
        pass
    assert fake.modes[0] == "SPX"


def test_set_partition_mode_is_a_no_op_when_already_there(smi):
    assert set_partition_mode(0, "SPX") == "SPX"
    assert smi.set_calls == []


def test_set_partition_mode_refuses_unknown_mode_without_touching_hardware(smi):
    with pytest.raises(PartitionError):
        set_partition_mode(0, "NPS2")
    assert smi.set_calls == []


def test_partitioned_restores_the_entry_mode(smi):
    with partitioned(0, "CPX") as mode:
        assert mode == "CPX"
        assert smi.modes[0] == "CPX"
    assert smi.modes[0] == "SPX"


def test_partitioned_restores_after_a_failure_inside_the_block(smi):
    with pytest.raises(ZeroDivisionError):
        with partitioned(0, "QPX"):
            assert smi.modes[0] == "QPX"
            raise ZeroDivisionError
    assert smi.modes[0] == "SPX"


def test_partitioned_restore_failure_does_not_mask_the_real_error(monkeypatch, caplog):
    fake = _FakeSmi({0: "SPX"})

    def flaky(cmd, **kwargs):
        # Let the entry set through, then refuse every later set.
        if cmd[:2] == ["amd-smi", "set"] and fake.modes.get(0) != "SPX":
            # A permanent refusal, so the restore fails immediately instead of
            # waiting out the drain timeout.
            return subprocess.CompletedProcess(cmd, 1, "", "permission denied")
        return fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", flaky)
    with pytest.raises(ValueError, match="workload blew up"):
        with partitioned(0, "DPX"):
            raise ValueError("workload blew up")
    assert "left in DPX" in caplog.text


def test_partitioned_honours_an_explicit_restore_target(smi):
    with partitioned(0, "CPX", restore_to=DEFAULT_MODE):
        pass
    assert smi.modes[0] == DEFAULT_MODE


def test_set_is_unprivileged_by_default(smi, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PARTITION_SUDO", raising=False)
    set_partition_mode(0, "CPX")
    assert smi.argv[0][:1] == ["amd-smi"]


def test_set_routes_through_sudo_when_opted_in(smi, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PARTITION_SUDO", "1")
    set_partition_mode(0, "CPX")
    # -n so an automated loop fails fast instead of waiting on a password prompt.
    assert smi.argv[0][:3] == ["sudo", "-n", "amd-smi"]


def test_permission_failure_that_exits_zero_is_still_caught(monkeypatch):
    # Observed on ROCm 7.x: amd-smi prints AmdSmiPermissionDeniedException and
    # exits 0. Only the read-back distinguishes this from a real change.
    fake = _FakeSmi({0: "SPX"}, stage_only=True)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(PartitionError, match="returned success but the mode did not change"):
        set_partition_mode(0, "CPX")
