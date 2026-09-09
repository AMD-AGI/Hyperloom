# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host CPU platform probe, shared by every caller that records tuning state."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.gpu_partition import published_shape
from hyperloom.common.provenance import detect_gfx_arch, detect_stack_fingerprint

log = logging.getLogger(__name__)

#: Filesystem root the probe reads under. Overridden in tests.
DEFAULT_ROOT = Path("/")

_CPU_ROOT = "sys/devices/system/cpu"
_NODE_ROOT = "sys/devices/system/node"
_AMDGPU_DRIVER_ROOT = "sys/bus/pci/drivers/amdgpu"


def read_kernel_file(path: Path | str, *, root: Path = DEFAULT_ROOT) -> str:
    """Read a ``/sys`` or ``/proc`` file, returning ``\"\"`` when unreadable."""
    p = Path(path)
    target = p if p.is_absolute() else root / p
    try:
        return target.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def sysfs_available(*, root: Path = DEFAULT_ROOT) -> bool:
    """Whether host CPU sysfs is visible, i.e. whether a probe is meaningful."""
    return bool(read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)) or (root / _CPU_ROOT / "cpu0").exists()


def smt_state(*, root: Path = DEFAULT_ROOT) -> str | None:
    """``"on"``/``"off"``, or ``None`` when the kernel does not expose SMT."""
    raw = read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)
    if not raw:
        return None
    return "on" if raw == "1" else "off"


def socket_count(*, root: Path = DEFAULT_ROOT) -> int | None:
    """Distinct physical package IDs, or ``None`` when none are readable."""
    try:
        ids = {read_kernel_file(p) for p in (root / _CPU_ROOT).glob("cpu*/topology/physical_package_id")}
    except OSError:
        return None
    return len(ids - {""}) or None


def numa_node_count(*, root: Path = DEFAULT_ROOT) -> int | None:
    """Count of NUMA nodes, or ``None`` when the node tree is absent."""
    try:
        return len(list((root / _NODE_ROOT).glob("node[0-9]*"))) or None
    except OSError:
        return None


def nodes_per_socket(*, root: Path = DEFAULT_ROOT) -> str | None:
    """``\"NPS1\"``-style label, or ``None`` when it cannot be derived."""
    sockets = socket_count(root=root)
    nodes = numa_node_count(root=root)
    if not sockets or not nodes:
        return None
    return f"NPS{nodes // sockets}"


def cpufreq_governor(*, root: Path = DEFAULT_ROOT) -> str:
    """Scaling governor of cpu0, or ``"unknown"``."""
    return read_kernel_file(f"{_CPU_ROOT}/cpu0/cpufreq/scaling_governor", root=root) or "unknown"


def boost_state(*, root: Path = DEFAULT_ROOT) -> str:
    """Core Performance Boost as ``"on"``/``"off"``/``"unknown"``."""
    raw = read_kernel_file(f"{_CPU_ROOT}/cpufreq/boost", root=root)
    return {"1": "on", "0": "off"}.get(raw, "unknown")


def cpu_model(*, root: Path = DEFAULT_ROOT) -> str:
    """First ``model name`` line from ``/proc/cpuinfo``, or ``"unknown"``."""
    for line in read_kernel_file("proc/cpuinfo", root=root).splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def kernel_release(*, root: Path = DEFAULT_ROOT) -> str:
    """Running kernel release, or ``"unknown"``."""
    return read_kernel_file("proc/sys/kernel/osrelease", root=root) or "unknown"


def amdgpu_device_count(*, root: Path = DEFAULT_ROOT) -> int | None:
    """PCI devices bound to ``amdgpu``, or ``None`` when none are readable."""
    try:
        return len(list((root / _AMDGPU_DRIVER_ROOT).glob("*:*:*.*"))) or None
    except OSError:
        return None


@dataclass(frozen=True)
class CpuPlatform:
    """Host CPU tuning state. Every field degrades independently."""

    cpu: str
    smt: str | None
    sockets: int | None
    numa_nodes: int | None
    nps: str | None
    governor: str
    boost: str
    kernel: str

    def as_dict(self) -> dict:
        return asdict(self)


def probe_cpu_platform(*, root: Path = DEFAULT_ROOT) -> CpuPlatform | None:
    """Read host CPU tuning state, or ``None`` when not on Linux sysfs."""
    if not sysfs_available(root=root):
        return None
    return CpuPlatform(
        cpu=cpu_model(root=root),
        smt=smt_state(root=root),
        sockets=socket_count(root=root),
        numa_nodes=numa_node_count(root=root),
        nps=nodes_per_socket(root=root),
        governor=cpufreq_governor(root=root),
        boost=boost_state(root=root),
        kernel=kernel_release(root=root),
    )


def platform_fingerprint(
    gpu_type: str | None = None,
    *,
    multi_node: bool | None = None,
) -> dict[str, Any]:
    """Full host record -- CPU tuning, GPUs and software stack -- for provenance."""
    try:
        plat = probe_cpu_platform()
        if plat is None:
            return {"status": "unavailable", "reason": "no host CPU sysfs on this machine"}

        record: dict[str, Any] = {
            "status": "ok",
            "host": socket.gethostname(),
            "multi_node_session": multi_node,
            **plat.as_dict(),
        }
        # Each block below degrades on its own: one unreadable file must not take the whole platform record with it.
        try:
            record["gpu"] = {
                # PCI devices bound to amdgpu: what the host has, not what the run could see. *_VISIBLE_DEVICES
                # masking does not change this number, so it is named for the host to keep it from being read as the
                # run's device count.
                "host_count": amdgpu_device_count(),
                # probe=False: gpu_type already answers this, and report generation runs in-process under unit tests
                # that must not spawn rocminfo.
                "gfx_arch": detect_gfx_arch(os.environ, gpu_type=gpu_type, probe=False) or "unknown",
                "amdgpu_driver": read_kernel_file("/sys/module/amdgpu/version") or "unknown",
            }
            # The card's compute-partition shape, when this session established one.
            partition = published_shape()
            if partition:
                record["gpu"]["compute_partition"] = partition
        except Exception:  # noqa: BLE001 - one degraded field, not a dropped record
            log.warning("platform fingerprint: GPU block unreadable", exc_info=True)
            record["gpu"] = {"status": "error"}
        try:
            record["stack"] = detect_stack_fingerprint(os.environ)
        except Exception:  # noqa: BLE001
            log.warning("platform fingerprint: stack block unreadable", exc_info=True)
            record["stack"] = {"status": "error"}
        return record
    except Exception as exc:  # noqa: BLE001 - never break the caller
        # Warning, not debug: this record is provenance, and a silent hole in it is only discovered when someone needs
        # it and it is too late to re-run.
        log.warning("platform fingerprint failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc)}


__all__ = [
    "CpuPlatform",
    "DEFAULT_ROOT",
    "amdgpu_device_count",
    "boost_state",
    "cpu_model",
    "cpufreq_governor",
    "kernel_release",
    "nodes_per_socket",
    "numa_node_count",
    "platform_fingerprint",
    "probe_cpu_platform",
    "read_kernel_file",
    "smt_state",
    "socket_count",
    "sysfs_available",
]
