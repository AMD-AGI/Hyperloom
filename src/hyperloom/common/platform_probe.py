# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host CPU platform probe, shared by every caller that records tuning state.

Single source of truth for reading the handful of ``/sys`` and ``/proc`` values
that describe how a host is tuned: SMT, socket and NUMA counts, nodes-per-socket,
the cpufreq governor, and Core Performance Boost. It is shared rather than
reimplemented per caller because these reads have edge cases -- an empty-string
socket id, a container with no NUMA tree -- that every copy has to get right, and
a copy that gets one wrong reports a plausible wrong number rather than failing.

``scripts/platform_audit.py`` is the deliberate exception: it must run on hosts
with no Hyperloom install, so it carries its own copy and says so.

Design:

* **per-field degradation**: every field is read independently and falls back to
  ``None``/``"unknown"`` on its own. One unreadable file must not take the rest of
  the record with it, because the record exists to explain a result after the fact.
* **absence is not failure**: :func:`sysfs_available` separates "this host is not
  Linux sysfs" from "the probe broke", which a reader of an archived report cannot
  otherwise distinguish.
* **injectable root**: every function takes ``root`` so tests can build a fake
  ``/sys`` tree instead of monkeypatching module internals.
* **light and never raises**: stdlib plus ``hyperloom.common.provenance``, which
  is itself stdlib-only. Importable from any layer without a cycle, and cheap
  enough for the crash-safe writer to use after a run has already died.
"""

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


def amdgpu_device_count(*, root: Path = DEFAULT_ROOT) -> int | None:
    """PCI devices bound to ``amdgpu``, or ``None`` when none are readable.

    Counts every PCI domain rather than ``0000`` alone. A host with more
    devices than one domain can address puts its GPUs under ``0002:``,
    ``0003:`` and so on -- which is the normal layout on an MI300/MI350-class
    node -- and a ``0000:``-only match counts zero of them there, i.e. it fails
    on exactly the hardware this record is written to describe.

    The ``*:*:*.*`` shape is what separates a device entry from the driver's
    own ``bind``/``unbind``/``module`` siblings, which share the directory and
    are not PCI addresses.
    """
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


def platform_fingerprint(
    gpu_type: str | None = None,
    *,
    multi_node: bool | None = None,
) -> dict[str, Any]:
    """Full host record -- CPU tuning, GPUs and software stack -- for provenance.

    Lives here rather than beside the report renderer because both callers that
    need it, the run report and the crash-safe ``final.json`` writer, sit above
    ``hyperloom.common``. Reaching the other way round would make the crash path
    import a private symbol out of the orchestrator, and with it the message bus
    and a SQLite connection layer -- roughly 350 modules -- to fill in six fields
    at the exact moment those subsystems may be the thing that just failed.

    Always returns a dict carrying ``status``. An archived report must be able
    to distinguish "this host had no CPU sysfs" from "the probe broke", which a
    bare ``null`` cannot express, and that distinction matters precisely when
    someone is trying to explain a delta long after the run.

    Scope: this samples the calling process's own node. In a multi-node session
    that is usually not the benchmark node, so the record would describe a
    machine the numbers did not come from -- worse than no record, because it
    reads as fact. ``host`` names the sampled machine and ``multi_node_session``
    marks when the record is known to be partial.

    Args:
        gpu_type: Session ``--gpu-type``. Resolves the gfx arch from the board
            table, so building this record never spawns a probe subprocess.
        multi_node: Whether this is a >=2-node session, when the caller knows.
            ``None`` records that nobody established it, which is the honest
            answer on the crash path rather than an unearned ``False``.

    Returns:
        dict[str, Any]: ``status="ok"`` with the platform facts, or
        ``status="unavailable"``/``"error"`` with a human-readable ``reason``.
    """
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
        # Each block below degrades on its own: one unreadable file must not
        # take the whole platform record with it.
        try:
            record["gpu"] = {
                # PCI devices bound to amdgpu: what the host has, not what the
                # run could see. *_VISIBLE_DEVICES masking does not change this
                # number, so it is named for the host to keep it from being read
                # as the run's device count.
                "host_count": amdgpu_device_count(),
                # probe=False: gpu_type already answers this, and report
                # generation runs in-process under unit tests that must not
                # spawn rocminfo.
                "gfx_arch": detect_gfx_arch(os.environ, gpu_type=gpu_type, probe=False) or "unknown",
                "amdgpu_driver": read_kernel_file("/sys/module/amdgpu/version") or "unknown",
            }
            # The card's compute-partition shape, when this session established
            # one. Recorded for the same reason NPS is: it changes what the
            # numbers mean, and without it two runs of the same configuration on
            # the same card in SPX and in CPX are indistinguishable in the
            # history. Read from the env the launch published rather than probed,
            # so this stays subprocess-free on the crash path.
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
        # Warning, not debug: this record is provenance, and a silent hole in it
        # is only discovered when someone needs it and it is too late to re-run.
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
