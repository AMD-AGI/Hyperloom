# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared host CPU platform probe.

Every case builds a fake ``/sys`` + ``/proc`` tree under ``tmp_path``, so these
run identically on any host and assert the degradation rules the three call
sites depend on.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.platform_probe import (
    boost_state,
    cpu_model,
    cpufreq_governor,
    nodes_per_socket,
    numa_node_count,
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


def _host(root: Path, *, cpus: int = 4, sockets: int = 2, nodes: int = 8,
          smt: str = "1", governor: str = "performance", boost: str = "1") -> Path:
    """A plausible two-socket EPYC host."""
    _mk(root, "sys/devices/system/cpu/smt/active", smt)
    for i in range(cpus):
        _mk(root, f"sys/devices/system/cpu/cpu{i}/topology/physical_package_id",
            str(i % sockets))
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

    NPS0 is not a value any BIOS can hold, and it previously read downstream as
    a real misconfiguration rather than as an unanswerable question.
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
