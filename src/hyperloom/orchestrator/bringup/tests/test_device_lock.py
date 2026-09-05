# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who holds the claim on a card, and for exactly how long.

The property under test is not that a lock file exists but that the claim
outlives the process that took it whenever a serving child inherited it. A
claim released when the coordinator dies is released at the one moment it was
needed -- an orphaned server still on the cards.
"""

from __future__ import annotations

import fcntl
import os
import subprocess  # nosec B404 - spawns a child that must inherit the claim
import sys
import time

import pytest

from hyperloom.orchestrator.bringup import device_lock
from hyperloom.orchestrator.bringup.device_lock import (
    DevicesBusy,
    claim_devices,
    visible_devices,
)

_HOLD_FD = "import sys, time; time.sleep(300)"

_NODES = ("renderD128", "renderD129", "renderD130")


@pytest.fixture(autouse=True)
def _no_device_nodes(monkeypatch):
    """Read no device nodes unless a test says which ones are there.

    A token that names no node stands for itself, so the lock-file tests name
    their own cards instead of the ones this host happens to have.
    """
    monkeypatch.setattr(device_lock, "present_devices", lambda: ())


def _is_locked(path) -> bool:
    """Whether some process holds an exclusive lock on ``path``."""
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def test_the_claimed_devices_come_from_what_the_launch_will_see(monkeypatch):
    """The claim has to name the cards the child gets, not the cards the host has.

    A mask indexes the cards the process can see, so the claim is named by the
    device node behind the index: an index means a different card in a container
    that was given a different set, and two of those must not exclude each other
    over the same name.
    """
    monkeypatch.setattr(device_lock, "present_devices", lambda: _NODES)

    assert visible_devices({"HIP_VISIBLE_DEVICES": "2,0"}) == ("renderD128", "renderD130")
    assert visible_devices({"CUDA_VISIBLE_DEVICES": "1"}) == ("renderD129",)


def test_the_rocm_variable_decides_when_more_than_one_mask_is_set(monkeypatch):
    """ROCR is the canonical pinning, and the launch honours it first."""
    monkeypatch.setattr(device_lock, "present_devices", lambda: _NODES)

    env = {"ROCR_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "0"}
    assert visible_devices(env) == ("renderD130",)


def test_an_unrestricted_launch_claims_every_card_it_can_open(monkeypatch):
    """Two containers holding disjoint cards must not collide over one name."""
    monkeypatch.setattr(device_lock, "present_devices", lambda: _NODES)

    assert visible_devices({}) == _NODES


def test_a_launch_masked_to_no_card_claims_nothing(monkeypatch):
    """An empty mask is set, and what it says is that the launch sees no card."""
    monkeypatch.setattr(device_lock, "present_devices", lambda: _NODES)

    assert visible_devices({"HIP_VISIBLE_DEVICES": ""}) == ()


def test_a_second_claim_on_the_same_card_is_refused(tmp_path):
    """This is the exclusion the per-session lease cannot express."""
    env = {"HIP_VISIBLE_DEVICES": "0"}
    first = claim_devices(env, lock_dir=tmp_path)
    try:
        child = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", "import time; time.sleep(300)"],
            pass_fds=first.fds,
        )
        try:
            # A different process, so the in-process reference count does not
            # apply: it must be told no.
            probe = subprocess.run(  # nosec B603
                [
                    sys.executable,
                    "-c",
                    "import fcntl, os, sys;"
                    f"fd = os.open({str(tmp_path / 'gpu-0.lock')!r}, os.O_RDWR);"
                    "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
                ],
                capture_output=True,
                check=False,
            )
            assert probe.returncode != 0
        finally:
            child.kill()
            child.wait()
    finally:
        first.release()


def test_the_claim_survives_the_process_that_took_it(tmp_path):
    """The descriptor belongs to the serving tree, which is the whole mechanism."""
    env = {"HIP_VISIBLE_DEVICES": "7"}
    claim = claim_devices(env, lock_dir=tmp_path)
    child = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", _HOLD_FD],
        pass_fds=claim.fds,
    )
    try:
        # The taker lets go, standing in for a coordinator that was killed.
        claim.release()
        assert _is_locked(tmp_path / "gpu-7.lock"), "an orphaned server must keep the cards claimed"
    finally:
        child.kill()
        child.wait()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _is_locked(tmp_path / "gpu-7.lock"):
        time.sleep(0.05)
    assert not _is_locked(tmp_path / "gpu-7.lock"), "and release them when it finally exits"


def test_a_second_launch_in_this_process_does_not_queue_behind_itself(tmp_path):
    """One session serving twice on cards it already holds is not a conflict."""
    env = {"HIP_VISIBLE_DEVICES": "1"}
    first = claim_devices(env, lock_dir=tmp_path)
    second = claim_devices(env, lock_dir=tmp_path)
    try:
        assert first.fds == second.fds
        second.release()
        assert _is_locked(tmp_path / "gpu-1.lock"), "the first holder is still holding"
    finally:
        first.release()
    assert not _is_locked(tmp_path / "gpu-1.lock")


def test_the_claim_can_be_switched_off_entirely(tmp_path):
    """A host that cannot give the guarantee must still be able to run."""
    claim = claim_devices({"HIP_VISIBLE_DEVICES": "0", "HYPERLOOM_DEVICE_LOCK": "0"}, lock_dir=tmp_path)
    assert claim.fds == ()
    assert not (tmp_path / "gpu-0.lock").exists()


def test_a_partial_claim_is_given_back_before_it_is_reported_busy(tmp_path):
    """A launch refused on its second card must not keep holding its first."""
    blocker = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-c",
            "import fcntl, os, time;"
            f"fd = os.open({str(tmp_path / 'gpu-b.lock')!r}, os.O_CREAT | os.O_RDWR, 0o666);"
            "fcntl.flock(fd, fcntl.LOCK_EX);"
            "print('held', flush=True);"
            "time.sleep(300)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        blocker.stdout.readline()
        with pytest.raises(DevicesBusy):
            claim_devices({"HIP_VISIBLE_DEVICES": "a,b"}, lock_dir=tmp_path)
        assert not _is_locked(tmp_path / "gpu-a.lock")
    finally:
        blocker.kill()
        blocker.wait()
