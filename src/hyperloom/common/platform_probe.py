# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host CPU platform probe, shared by every caller that records tuning state.

Single source of truth for reading the handful of ``/sys`` and ``/proc`` values
that describe how a host is tuned: SMT, socket and NUMA counts, nodes-per-socket,
the cpufreq governor, and Core Performance Boost. Three call sites grew their own
copy of this logic (CLI preflight, the run report, and the standalone auditor) and
they drifted -- one dropped the empty-string filter, one omitted the zero guard
that turns a container without ``/sys/devices/system/node`` into a bogus ``NPS0``.

Design:

* **per-field degradation**: every field is read independently and falls back to
  ``None``/``"unknown"`` on its own. One unreadable file must not take the rest of
  the record with it, because the record exists to explain a result after the fact.
* **absence is not failure**: :func:`sysfs_available` separates "this host is not
  Linux sysfs" from "the probe broke", which a reader of an archived report cannot
  otherwise distinguish.
* **injectable root**: every function takes ``root`` so tests can build a fake
  ``/sys`` tree instead of monkeypatching module internals.
* **stdlib-only, never raises**: importable from any layer without a cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

#: Filesystem root the probe reads under. Overridden in tests.
DEFAULT_ROOT = Path("/")

_CPU_ROOT = "sys/devices/system/cpu"
_NODE_ROOT = "sys/devices/system/node"


def read_kernel_file(path: Path | str, *, root: Path = DEFAULT_ROOT) -> str:
    """Read a ``/sys`` or ``/proc`` file, returning ``""`` when unreadable.

    Named for what it reads rather than for sysfs alone: callers also use it for
    ``/proc/cpuinfo`` and ``/proc/sys/kernel/osrelease``.
    """
    p = Path(path)
    target = p if p.is_absolute() else root / p
    try:
        return target.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def sysfs_available(*, root: Path = DEFAULT_ROOT) -> bool:
    """Whether host CPU sysfs is visible, i.e. whether a probe is meaningful."""
    return bool(read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)) or (
        root / _CPU_ROOT / "cpu0"
    ).exists()


def smt_state(*, root: Path = DEFAULT_ROOT) -> str | None:
    """``"on"``/``"off"``, or ``None`` when the kernel does not expose SMT."""
    raw = read_kernel_file(f"{_CPU_ROOT}/smt/active", root=root)
    if not raw:
        return None
    return "on" if raw == "1" else "off"


def socket_count(*, root: Path = DEFAULT_ROOT) -> int | None:
    """Distinct physical package IDs, or ``None`` when none are readable."""
    try:
        ids = {
            read_kernel_file(p)
            for p in (root / _CPU_ROOT).glob("cpu*/topology/physical_package_id")
        }
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
    """``"NPS1"``-style label, or ``None`` when it cannot be derived.

    Both counts are required. A container that sees CPU topology but no node
    tree would otherwise divide into zero and report ``NPS0`` -- a value no BIOS
    can be set to, which then reads as a real misconfiguration downstream.
    """
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
    """Read host CPU tuning state, or ``None`` when not on Linux sysfs.

    ``None`` means only one thing -- there is no host CPU sysfs to read, so the
    question is not meaningful here. A field that could not be read on a host
    that *does* have sysfs comes back as ``None``/``"unknown"`` in that field
    alone, so a single unreadable file never discards the rest of the record.
    """
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


__all__ = [
    "CpuPlatform",
    "DEFAULT_ROOT",
    "boost_state",
    "cpu_model",
    "cpufreq_governor",
    "kernel_release",
    "nodes_per_socket",
    "numa_node_count",
    "probe_cpu_platform",
    "read_kernel_file",
    "smt_state",
    "socket_count",
    "sysfs_available",
]
