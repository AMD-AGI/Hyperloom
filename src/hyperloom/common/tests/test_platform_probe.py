# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared host CPU platform probe.

Every case builds a fake ``/sys`` + ``/proc`` tree under ``tmp_path``, so these
run identically on any host and assert the degradation rules the three call
sites depend on.
"""

from __future__ import annotations

import socket
from pathlib import Path

from hyperloom.common import platform_probe as platform_probe_mod
from hyperloom.common.platform_probe import (
    CpuPlatform,
    amdgpu_device_count,
    boost_state,
    cpu_model,
    cpufreq_governor,
    nodes_per_socket,
    numa_node_count,
    platform_fingerprint,
    probe_cpu_platform,
    read_kernel_file,
    smt_state,
    socket_count,
    sysfs_available,
)


def _mk(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _host(
    root: Path,
    *,
    cpus: int = 4,
    sockets: int = 2,
    nodes: int = 8,
    smt: str = "1",
    governor: str = "performance",
    boost: str = "1",
) -> Path:
    """A plausible two-socket EPYC host."""
    _mk(root, "sys/devices/system/cpu/smt/active", smt)
    for i in range(cpus):
        _mk(root, f"sys/devices/system/cpu/cpu{i}/topology/physical_package_id", str(i % sockets))
    for i in range(nodes):
        (root / f"sys/devices/system/node/node{i}").mkdir(parents=True, exist_ok=True)
    _mk(root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", governor)
    _mk(root, "sys/devices/system/cpu/cpufreq/boost", boost)
    _mk(root, "proc/cpuinfo", "processor\t: 0\nmodel name\t: AMD EPYC 9575F 64-Core Processor\n")
    _mk(root, "proc/sys/kernel/osrelease", "5.15.0-70-generic")
    return root


def test_reads_a_tuned_host(tmp_path):
    root = _host(tmp_path)
    plat = probe_cpu_platform(root=root)
    assert plat is not None
    assert plat.smt == "on"
    assert plat.sockets == 2
    assert plat.numa_nodes == 8
    assert plat.nps == "NPS4"
    assert plat.governor == "performance"
    assert plat.boost == "on"
    assert plat.cpu == "AMD EPYC 9575F 64-Core Processor"
    assert plat.kernel == "5.15.0-70-generic"


def test_absent_sysfs_is_not_a_failure(tmp_path):
    """No host CPU sysfs must be distinguishable from a broken probe."""
    assert sysfs_available(root=tmp_path) is False
    assert probe_cpu_platform(root=tmp_path) is None


def test_missing_numa_tree_does_not_produce_nps0(tmp_path):
    """A container seeing CPUs but no node tree must not report a fake NPS0.

    NPS0 is not a value any BIOS can hold, so a reader comparing the record
    against a target cannot tell it apart from a real misconfiguration. None
    says the question was unanswerable on this host, which is the truth.
    """
    root = _host(tmp_path, nodes=0)
    assert numa_node_count(root=root) is None
    assert nodes_per_socket(root=root) is None
    plat = probe_cpu_platform(root=root)
    assert plat is not None and plat.nps is None
    # The rest of the record survives the one missing input.
    assert plat.sockets == 2 and plat.governor == "performance"


def test_one_unreadable_field_does_not_discard_the_record(tmp_path):
    """Per-field degradation: a missing governor must not drop CPU/SMT/NUMA."""
    root = _host(tmp_path)
    (root / "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").unlink()
    plat = probe_cpu_platform(root=root)
    assert plat is not None
    assert plat.governor == "unknown"
    assert plat.smt == "on" and plat.sockets == 2 and plat.nps == "NPS4"


def test_smt_off_and_boost_off(tmp_path):
    root = _host(tmp_path, smt="0", boost="0")
    assert smt_state(root=root) == "off"
    assert boost_state(root=root) == "off"


def test_unreadable_values_degrade_to_placeholders(tmp_path):
    root = _host(tmp_path)
    (root / "sys/devices/system/cpu/cpufreq/boost").unlink()
    (root / "proc/cpuinfo").unlink()
    assert boost_state(root=root) == "unknown"
    assert cpu_model(root=root) == "unknown"
    assert cpufreq_governor(root=root) == "performance"


def test_blank_package_ids_are_not_counted(tmp_path):
    """An empty topology file must not inflate the socket count."""
    root = _host(tmp_path, cpus=2, sockets=1)
    _mk(root, "sys/devices/system/cpu/cpu9/topology/physical_package_id", "")
    assert socket_count(root=root) == 1


def test_read_kernel_file_never_raises(tmp_path):
    assert read_kernel_file("proc/nope", root=tmp_path) == ""
    d = tmp_path / "adir"
    d.mkdir()
    assert read_kernel_file("adir", root=tmp_path) == ""


def _amdgpu(root: Path, *addresses: str) -> Path:
    """A driver directory holding the given devices next to its own entries."""
    d = root / "sys/bus/pci/drivers/amdgpu"
    d.mkdir(parents=True, exist_ok=True)
    for address in addresses:
        (d / address).mkdir()
    for entry in ("bind", "unbind", "uevent", "new_id", "remove_id"):
        (d / entry).write_text("")
    (d / "module").mkdir()
    return root


def test_gpus_outside_pci_domain_zero_are_counted(tmp_path):
    """A host large enough to need several PCI domains puts its GPUs outside
    ``0000:``, and that is the hardware this record exists to describe. Matching
    ``0000:`` alone reported no accelerators on exactly those machines."""
    root = _amdgpu(
        tmp_path,
        "0002:00:01.0",
        "0002:00:02.0",
        "0002:00:03.0",
        "0002:00:04.0",
        "0003:00:01.0",
        "0003:00:02.0",
        "0003:00:03.0",
        "0003:00:04.0",
    )
    assert amdgpu_device_count(root=root) == 8


def test_the_ordinary_single_domain_host_still_counts(tmp_path):
    root = _amdgpu(tmp_path, "0000:63:00.0", "0000:a3:00.0")
    assert amdgpu_device_count(root=root) == 2


def test_driver_control_entries_are_not_devices(tmp_path):
    """``bind``, ``module`` and friends share the directory with the devices."""
    assert amdgpu_device_count(root=_amdgpu(tmp_path)) is None


def test_a_host_without_the_amdgpu_driver_reports_nothing(tmp_path):
    """None, not zero: the driver dir is absent, which is not a count of zero."""
    assert amdgpu_device_count(root=tmp_path) is None


def test_platform_fingerprint_unavailable_when_probe_returns_none(monkeypatch):
    monkeypatch.setattr(platform_probe_mod, "probe_cpu_platform", lambda: None)
    got = platform_fingerprint(gpu_type="mi300x", multi_node=False)
    assert got == {"status": "unavailable", "reason": "no host CPU sysfs on this machine"}


def test_platform_fingerprint_outer_failure_is_status_error(monkeypatch):
    # Use a non-RuntimeError to pin the broad `except Exception` net; narrowing
    # it to `except RuntimeError` would let OSError/AttributeError escape.
    def _boom():
        raise OSError("probe exploded")

    monkeypatch.setattr(platform_probe_mod, "probe_cpu_platform", _boom)
    got = platform_fingerprint()
    assert got["status"] == "error"
    assert "probe exploded" in got["reason"]


def _make_plat():
    return CpuPlatform(
        cpu="AMD EPYC",
        smt="on",
        sockets=2,
        numa_nodes=8,
        nps="NPS4",
        governor="performance",
        boost="on",
        kernel="5.15.0",
    )


def test_platform_fingerprint_gpu_block_degrades_stack_survives(monkeypatch):
    """GPU block error must not corrupt the stack block (per-block independence)."""
    # platform_fingerprint takes no injectable `root`; these error tiers are
    # unreachable through a fake sysfs tree, so we monkeypatch module globals.
    monkeypatch.setattr(platform_probe_mod, "probe_cpu_platform", _make_plat)

    def _gpu_boom():
        raise OSError("gpu sysfs unreadable")

    fake_stack = {"rocm": "6.0.0", "driver": "amdgpu"}
    monkeypatch.setattr(platform_probe_mod, "amdgpu_device_count", _gpu_boom)
    monkeypatch.setattr(platform_probe_mod, "detect_stack_fingerprint", lambda _env: fake_stack)
    got = platform_fingerprint(gpu_type="mi300x", multi_node=False)
    assert got["status"] == "ok"
    assert got["cpu"] == "AMD EPYC"
    assert got["gpu"] == {"status": "error"}
    # Stack block must carry its real content, not the error sentinel.
    assert got["stack"] == fake_stack


def test_platform_fingerprint_stack_block_degrades_gpu_survives(monkeypatch):
    """Stack block error must not corrupt the GPU block (per-block independence)."""
    monkeypatch.setattr(platform_probe_mod, "probe_cpu_platform", _make_plat)

    def _stack_boom(_env):
        raise RuntimeError("stack probe failed")

    monkeypatch.setattr(platform_probe_mod, "amdgpu_device_count", lambda: 8)
    monkeypatch.setattr(platform_probe_mod, "detect_stack_fingerprint", _stack_boom)
    got = platform_fingerprint(gpu_type="mi300x", multi_node=True)
    assert got["status"] == "ok"
    assert got["multi_node_session"] is True
    assert got["stack"] == {"status": "error"}
    # GPU block must carry real content — presence of gfx_arch confirms the
    # table lookup ran rather than producing the error sentinel.
    assert got["gpu"] != {"status": "error"}
    assert "gfx_arch" in got["gpu"]


def test_platform_fingerprint_ok_record_shape_and_multi_node_none(monkeypatch):
    """All-healthy ok record: gpu sub-dict keys, host, and None-vs-False multi_node."""
    # Same justification as the degrade cases: no injectable root on this entry.
    fake_stack = {"rocm": "6.0.0", "driver": "amdgpu"}
    monkeypatch.setattr(platform_probe_mod, "probe_cpu_platform", _make_plat)
    monkeypatch.setattr(platform_probe_mod, "amdgpu_device_count", lambda: 8)
    monkeypatch.setattr(platform_probe_mod, "detect_stack_fingerprint", lambda _env: fake_stack)
    monkeypatch.setattr(platform_probe_mod, "read_kernel_file", lambda *_a, **_k: None)
    monkeypatch.delenv("HYPERLOOM_GFX_ARCH", raising=False)
    monkeypatch.delenv("GFX_ARCH", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)

    got = platform_fingerprint(gpu_type="mi300x")
    assert got["status"] == "ok"
    assert got["host"] == socket.gethostname()
    assert got["cpu"] == "AMD EPYC"
    # Default multi_node is None (unset), not an unearned False.
    assert got["multi_node_session"] is None
    assert got["gpu"]["host_count"] == 8
    assert got["gpu"]["gfx_arch"] == "gfx942"
    assert got["gpu"]["amdgpu_driver"] == "unknown"
    assert got["stack"] == fake_stack

    got_false = platform_fingerprint(gpu_type="mi300x", multi_node=False)
    assert got_false["multi_node_session"] is False

    # gpu_type unset + probe=False → gfx_arch falls back to "unknown".
    got_unknown = platform_fingerprint()
    assert got_unknown["gpu"]["gfx_arch"] == "unknown"
