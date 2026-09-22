# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware roofs, measured once and read thereafter.

Peaks are a property of the machine, not of the operator being analysed. The
first cut re-ran ``--roof-only`` for every ceiling because the cache was keyed
by operator, so ten operators on one box measured the same roofs ten times.
Worse, a host without ``rocprof-compute`` could never measure them at all and
fell straight to vendor datasheet figures, which are around twice what the chip
sustains.

Both follow from storing the roofs in the wrong place. Here they get their own
artifact, keyed by the machine:

1. A **local profile** under the writable state root, written the first time
   this box measures itself and read by every run after.
2. A **reference profile** shipped with the package, matched on architecture,
   device name and partition mode. Measured on a real card, stamped with when,
   with which tool versions, and under which partition and power cap. It is a
   better answer than a datasheet for a host that cannot measure, and an honest
   one because a reader can see it came from somewhere else.
3. The datasheet, last, as before.

A reference profile is not a measurement of *your* box and never claims to be:
``peak_source`` distinguishes the three, and the report says which applied.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.durable_io import atomic_write_text
from kernelforge.fusion.gpu_arch import canon_arch, detect_arch
from kernelforge.resources import default_project_root, resource_path

log = logging.getLogger("kernelforge.roofline_ceiling")

PROFILE_SCHEMA_VERSION = 1
REFERENCE_DIRNAME = "device_profiles"

_MARKETING_RE = re.compile(r"^\s*Marketing Name:\s*(AMD Instinct\s+\S+)", re.MULTILINE)
_COMPUTE_PARTITION_RE = re.compile(r"Compute Partition:\s*(\S+)")
_MEMORY_PARTITION_RE = re.compile(r"Memory Partition:\s*(\S+)")


@dataclass(frozen=True)
class DeviceIdentity:
    """What distinguishes one measurable machine configuration from another.

    Partition mode is part of the identity, not metadata. Splitting an MI355X
    into CPX slices changes the bandwidth one slice can reach, so a profile
    measured under SPX describes a different machine than the same card under
    CPX even though the arch and marketing name are identical.
    """

    arch: str
    device_name: str = ""
    compute_partition: str = ""
    memory_partition: str = ""

    def slug(self) -> str:
        """A filename-safe key for this configuration."""
        parts = [
            self.arch or "unknown",
            (self.device_name or "").lower().replace("amd instinct", "").strip().replace(" ", "-"),
            (self.compute_partition or "").lower(),
            (self.memory_partition or "").lower(),
        ]
        return "-".join(part for part in parts if part) or "unknown"

    def matches(self, criteria: dict[str, Any]) -> bool:
        """Whether a shipped profile's ``match`` block describes this machine.

        A criterion the profile does not state is not checked, so a profile may
        deliberately claim a whole architecture. A criterion this box could not
        determine does not match a profile that states one: guessing that an
        undetected partition is the profile's would be the silent substitution
        this module exists to avoid.
        """
        for field_name in ("arch", "device_name", "compute_partition", "memory_partition"):
            wanted = str(criteria.get(field_name) or "").strip().lower()
            if not wanted:
                continue
            if str(getattr(self, field_name) or "").strip().lower() != wanted:
                return False
        return True


@dataclass(frozen=True)
class DeviceProfile:
    """One machine's measured roofs, with the provenance to judge them by."""

    peak_flops: dict[str, float]
    bandwidth: dict[str, float]
    dispatch_floor_s: float
    source_by_figure: dict[str, str] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    origin: str = ""

    @property
    def usable(self) -> bool:
        """Whether this profile carries enough to stand in for a measurement."""
        return bool(self.peak_flops) and float(self.bandwidth.get("hbm", 0.0)) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize, in the shape the shipped profiles use."""
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "peak_flops": dict(self.peak_flops),
            "bandwidth": dict(self.bandwidth),
            "dispatch_floor_s": self.dispatch_floor_s,
            "source_by_figure": dict(self.source_by_figure),
            "measurement": dict(self.measurement),
            "notes": list(self.notes),
        }


def _run_text(argv: list[str], timeout_sec: float = 20.0) -> str:
    """Run a probe binary and return its output, or ``""`` on any failure."""
    if not shutil.which(argv[0]):
        return ""
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def describe_device(arch: str = "") -> DeviceIdentity:
    """Identify this machine's measurable configuration.

    Every field degrades independently to an empty string, which narrows what a
    shipped profile may claim rather than widening it.
    """
    resolved = canon_arch(arch) or detect_arch()
    marketing = _MARKETING_RE.search(_run_text(["rocminfo"]))
    partitions = _run_text(["rocm-smi", "--showcomputepartition", "--showmemorypartition"])
    compute = _COMPUTE_PARTITION_RE.search(partitions)
    memory = _MEMORY_PARTITION_RE.search(partitions)
    return DeviceIdentity(
        arch=resolved,
        device_name=(marketing.group(1).strip() if marketing else ""),
        compute_partition=(compute.group(1).strip() if compute else ""),
        memory_partition=(memory.group(1).strip() if memory else ""),
    )


def _parse_profile(payload: dict[str, Any], origin: str) -> DeviceProfile | None:
    """Read one profile document, or ``None`` when it cannot be trusted."""
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        log.warning("ignoring device profile %s: schema %r", origin, payload.get("schema_version"))
        return None
    try:
        profile = DeviceProfile(
            peak_flops={str(k): float(v) for k, v in (payload.get("peak_flops") or {}).items()},
            bandwidth={str(k): float(v) for k, v in (payload.get("bandwidth") or {}).items()},
            dispatch_floor_s=float(payload.get("dispatch_floor_s") or 0.0),
            source_by_figure={str(k): str(v) for k, v in (payload.get("source_by_figure") or {}).items()},
            measurement=dict(payload.get("measurement") or {}),
            notes=tuple(str(note) for note in (payload.get("notes") or ())),
            origin=origin,
        )
    except (TypeError, ValueError) as exc:
        log.warning("ignoring malformed device profile %s: %s", origin, exc)
        return None
    return profile if profile.usable else None


def local_path(identity: DeviceIdentity, project_root: str | Path | None = None) -> Path:
    """Where this box's own measurement is cached."""
    root = Path(project_root) if project_root is not None else default_project_root()
    return root / "device_profiles" / f"{identity.slug()}.json"


def load_local(identity: DeviceIdentity, project_root: str | Path | None = None) -> DeviceProfile | None:
    """Return this box's previously measured profile, if it has one."""
    path = local_path(identity, project_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable device profile %s: %s", path, exc)
        return None
    return _parse_profile(payload, origin=str(path))


def store_local(
    profile: DeviceProfile,
    identity: DeviceIdentity,
    project_root: str | Path | None = None,
) -> Path:
    """Cache a fresh measurement so later runs read it instead of remeasuring."""
    path = local_path(identity, project_root)
    atomic_write_text(path, json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_reference(identity: DeviceIdentity) -> DeviceProfile | None:
    """Return the shipped profile that claims this machine, if one does."""
    try:
        directory = resource_path(f"roofline_ceiling/{REFERENCE_DIRNAME}")
    except FileNotFoundError:
        return None
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("ignoring malformed shipped profile %s: %s", path, exc)
            continue
        if not identity.matches(payload.get("match") or {}):
            continue
        profile = _parse_profile(payload, origin=f"shipped:{path.name}")
        if profile is not None:
            return profile
    return None


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "REFERENCE_DIRNAME",
    "DeviceIdentity",
    "DeviceProfile",
    "describe_device",
    "load_local",
    "load_reference",
    "local_path",
    "store_local",
]
