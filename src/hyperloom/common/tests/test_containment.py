# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Containment capability detection: what the host can actually enforce."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.common import containment


def test_a_host_without_cgroup_v2_reports_why(monkeypatch: pytest.MonkeyPatch):
    """The gate must name the missing requirement, not just say no.

    An operator reading "containment unavailable" has to know whether to change
    the kernel, the mount, or the pod's capabilities.
    """
    monkeypatch.setattr(containment, "cgroup_v2_root", lambda: None)

    support = containment.probe_containment_support()

    assert support.available is False
    assert "cgroup v2" in support.reason


def test_an_undelegated_subtree_is_unavailable_not_assumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A subtree we cannot write is a hard stop.

    Reclaiming a lane requires creating a domain per task. If that fails there is
    no domain to attest empty, and the honest answer is to keep the lane rather
    than fall back to one of the process identities this design exists to retire.
    """
    monkeypatch.setattr(containment, "cgroup_v2_root", lambda: tmp_path)

    # Permission bits would not express this under a root-run test, and the
    # condition being checked is "the subtree is not delegated to us", which the
    # kernel reports as a failed mkdir however it arises.
    def _denied(self, *a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "mkdir", _denied)

    support = containment.probe_containment_support(parent=tmp_path)

    assert support.available is False
    assert support.delegated_root is None
    assert "delegated" in support.reason


def test_a_kernel_that_refuses_clone_into_cgroup_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without an atomic placement there is a window, and a window is the whole bug.

    Adding the cgroup from a preexec hook, or moving the pid after Popen returns,
    both let the workload run first -- and one fork in that interval escapes the
    domain for good.
    """
    monkeypatch.setattr(containment, "cgroup_v2_root", lambda: tmp_path)
    monkeypatch.setattr(containment, "_probe_clone_into_cgroup", lambda cgroup: False)

    support = containment.probe_containment_support(parent=tmp_path)

    assert support.available is False
    assert "clone3" in support.reason


def test_a_workload_that_keeps_cap_sys_admin_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Structural containment needs the workload unable to leave its domain.

    A process holding CAP_SYS_ADMIN can write cgroupfs and migrate itself or a
    descendant out, which turns the cgroup back into the kind of proxy this
    replaces. Startup treats that as unsupported instead of degrading quietly.
    """
    monkeypatch.setattr(containment, "cgroup_v2_root", lambda: tmp_path)
    monkeypatch.setattr(containment, "_probe_clone_into_cgroup", lambda cgroup: True)
    monkeypatch.setattr(containment, "_effective_caps", lambda: 0)

    # The control files live in the probe cgroup the kernel would have created,
    # so they are planted as mkdir returns rather than beside it.
    real_mkdir = Path.mkdir

    def _with_control_files(self, *a, **kw):
        real_mkdir(self, *a, **kw)
        for name in ("cgroup.kill", "cgroup.events"):
            (self / name).write_text("", encoding="utf-8")

    monkeypatch.setattr(Path, "mkdir", _with_control_files)

    support = containment.probe_containment_support(parent=tmp_path)

    assert support.available is False
    assert "CAP_SYS_ADMIN" in support.reason


def test_the_probe_leaves_no_cgroup_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Detection runs on every start; a leaked probe directory would accumulate."""
    monkeypatch.setattr(containment, "cgroup_v2_root", lambda: tmp_path)
    monkeypatch.setattr(containment, "_probe_clone_into_cgroup", lambda cgroup: True)

    containment.probe_containment_support(parent=tmp_path)

    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == []


@pytest.mark.skipif(
    containment.cgroup_v2_root() is None or os.geteuid() != 0,
    reason="needs a writable cgroup v2 hierarchy",
)
def test_a_setsid_descendant_cannot_leave_its_domain():
    """The property the whole design rests on, exercised against the real kernel.

    The survivor here reproduces every escape that defeated the earlier proofs:
    its root exits, it is reparented, and it calls setsid twice, so it shares no
    process group or session with anything recorded at spawn. It is still in the
    cgroup, and cgroup.events answers with a fact rather than a failure to find.
    """
    root = containment.cgroup_v2_root()
    assert root is not None
    domain = root / f"hyperloom_test_{os.getpid()}"
    domain.mkdir(exist_ok=True)
    try:
        fd = os.open(str(domain), os.O_RDONLY | os.O_DIRECTORY)
        try:
            import ctypes
            import ctypes.util
            import signal as _signal
            import time

            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            args = containment._CloneArgs(flags=containment.CLONE_INTO_CGROUP, exit_signal=_signal.SIGCHLD, cgroup=fd)
            pid = libc.syscall(containment._SYS_clone3, ctypes.byref(args), ctypes.sizeof(args))
            if pid == 0:
                os.setsid()
                if os.fork() != 0:
                    os._exit(0)
                os.setsid()
                time.sleep(60)
                os._exit(0)
            assert pid > 0
            os.waitpid(pid, 0)
            time.sleep(0.5)

            members = {int(x) for x in (domain / "cgroup.procs").read_text().split()}
            assert members, "a setsid'd survivor escaped its cgroup"
            survivor = next(iter(members))
            # It left every pid identity the refuted designs relied on.
            assert os.getsid(survivor) != os.getsid(os.getpid())

            (domain / "cgroup.kill").write_text("1")
            for _ in range(50):
                if not {int(x) for x in (domain / "cgroup.procs").read_text().split()}:
                    break
                time.sleep(0.1)
            assert "populated 0" in (domain / "cgroup.events").read_text()
        finally:
            os.close(fd)
    finally:
        try:
            domain.rmdir()
        except OSError:
            pass
