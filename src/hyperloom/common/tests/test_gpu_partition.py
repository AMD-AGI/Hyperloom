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


#: Verbatim shape of ``amd-smi partition -a --json`` on an MI355X, trimmed to
#: two profiles. The sparse continuation rows are the point: a profile's first
#: row names it and carries its XCC count, and the rows after it describe the
#: same profile's other resources under blank identity fields. A parser that
#: treats every row as a profile invents four per mode.
PROFILE_PAYLOAD = {
    "partition_profiles": [
        {
            "gpu_id": 0,
            "profile_index": 0,
            "memory_partition_caps": "NPS1",
            # amd-smi marks the live profile with a trailing asterisk.
            "accelerator_type": "SPX*",
            "num_partitions": 1,
            "resource_index": 0,
            "resource_type": "XCC",
            "resource_instances": 8,
        },
        {
            "gpu_id": "",
            "profile_index": "",
            "memory_partition_caps": "",
            "accelerator_type": "",
            "num_partitions": "",
            "resource_index": 1,
            "resource_type": "DECODER",
            "resource_instances": 4,
        },
        {
            "gpu_id": "",
            "profile_index": 3,
            "memory_partition_caps": "NPS1,NPS2",
            "accelerator_type": "CPX",
            "num_partitions": 8,
            "resource_index": 4,
            "resource_type": "XCC",
            "resource_instances": 1,
        },
        {
            "gpu_id": "",
            "profile_index": "",
            "memory_partition_caps": "",
            "accelerator_type": "",
            "num_partitions": "",
            "resource_index": 5,
            "resource_type": "JPEG",
            "resource_instances": 20,
        },
    ]
}

#: What the same command returns without privilege: the shape is there and every
#: value is gone. This is the reason the query reports "unknown" rather than
#: raising -- and the reason it must not read as "supports nothing".
PROFILE_PAYLOAD_UNPRIVILEGED = {
    "partition_profiles": [
        {
            "gpu_id": 0,
            "profile_index": "N/A",
            "memory_partition_caps": "N/A",
            "accelerator_type": "N/A",
            "num_partitions": "N/A",
            "resource_type": "N/A",
            "resource_instances": "N/A",
        }
    ]
}


def _profiles(monkeypatch, payload):
    """Install a fake amd-smi answering the profile query with ``payload``."""

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", run)


class TestCapabilityQuery:
    def test_reads_the_profiles_the_card_reports(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD)
        profiles = gpu_partition.read_partition_profiles(0)
        assert [p.mode for p in profiles] == ["SPX", "CPX"]
        assert [p.partitions for p in profiles] == [1, 8]
        assert [p.xcc_per_partition for p in profiles] == [8, 1]
        # The pairing constraint is captured even though nothing acts on it yet:
        # SPX is NPS1-only here while CPX accepts NPS2.
        assert profiles[0].memory_modes == ("NPS1",)
        assert profiles[1].memory_modes == ("NPS1", "NPS2")

    def test_strips_the_current_profile_marker(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD)
        assert gpu_partition.supported_modes(0) == ("SPX", "CPX")

    def test_unprivileged_query_is_unknown_not_empty_support(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD_UNPRIVILEGED)
        assert gpu_partition.supported_modes(0) == ()
        # The distinction that matters: an unanswerable query must not reject
        # every mode, or an unprivileged session cannot ask for anything.
        assert gpu_partition.unsupported_modes(["DPX", "CPX"]) == ()

    def test_a_missing_amd_smi_is_unknown_rather_than_fatal(self, monkeypatch):
        def missing(cmd, **kwargs):
            raise FileNotFoundError("amd-smi")

        monkeypatch.setattr(subprocess, "run", missing)
        assert gpu_partition.read_partition_profiles(0) == ()
        assert gpu_partition.supported_modes(0) == ()

    def test_names_the_modes_the_card_does_not_offer(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD)
        assert gpu_partition.unsupported_modes(["SPX", "DPX", "QPX"]) == ("DPX", "QPX")
        assert gpu_partition.unsupported_modes(["spx", "cpx"]) == ()

    def test_profile_query_is_routed_through_sudo_when_opted_in(self, monkeypatch):
        seen: list[list[str]] = []

        def run(cmd, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, json.dumps(PROFILE_PAYLOAD), "")

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setenv("HYPERLOOM_PARTITION_SUDO", "1")
        gpu_partition.read_partition_profiles(0)
        assert seen[0][:2] == ["sudo", "-n"]
        assert "amd-smi" in seen[0]

        seen.clear()
        monkeypatch.delenv("HYPERLOOM_PARTITION_SUDO")
        gpu_partition.read_partition_profiles(0)
        assert seen[0][0] == "amd-smi"


class TestStaticTableIsVerifiedNotTrusted:
    def test_no_conflict_when_the_card_agrees(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD)
        assert gpu_partition.partition_count_conflicts(0) == ()

    def test_conflict_is_reported_when_the_ladder_differs(self, monkeypatch):
        payload = {
            "partition_profiles": [
                {
                    "gpu_id": 0,
                    "profile_index": 3,
                    "accelerator_type": "CPX",
                    # A board with six XCDs would land here.
                    "num_partitions": 6,
                    "memory_partition_caps": "NPS1",
                    "resource_type": "XCC",
                    "resource_instances": 1,
                }
            ]
        }
        _profiles(monkeypatch, payload)
        conflicts = gpu_partition.partition_count_conflicts(0)
        assert len(conflicts) == 1
        assert "card reports 6" in conflicts[0] and "table says 8" in conflicts[0]

    def test_unqueryable_card_reports_no_conflict(self, monkeypatch):
        _profiles(monkeypatch, PROFILE_PAYLOAD_UNPRIVILEGED)
        assert gpu_partition.partition_count_conflicts(0) == ()


def test_layout_refuses_a_cu_count_that_does_not_divide(monkeypatch):
    # Flooring would be silent, and then fatal much later and for an apparently
    # unrelated reason: device selection matches the per-partition CU count
    # exactly, so a floored value matches no device at all.
    monkeypatch.setitem(gpu_partition.AMD_GPU_DISPATCH_IDENTITIES, "oddboard", ("gfx950", 300, "x"))
    with pytest.raises(PartitionError, match="does not divide"):
        layout_for("oddboard", "CPX")
    # The same board is fine in a mode its CU count does divide by.
    assert layout_for("oddboard", "QPX").cu_per_partition == 75


def test_every_shipped_board_divides_across_every_mode():
    """Guards the assumption the divisibility check exists to catch."""
    for board in gpu_partition.AMD_GPU_DISPATCH_IDENTITIES:
        for mode in gpu_partition.MODE_PARTITION_COUNTS:
            layout = layout_for(board, mode)
            assert layout.cu_per_partition > 0
