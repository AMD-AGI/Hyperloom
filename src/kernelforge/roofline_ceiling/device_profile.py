# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware roofs, measured once per machine configuration and committed.

Peaks are a property of the machine, not of the operator being analysed, and
not of the day. They are looked up here, never measured at run time: two
ceilings taken a month apart divide by the same numbers, a campaign needs no
profiler installed, and every figure a ceiling rests on has been through review
rather than being whatever the box reported that morning.

A profile is matched on architecture, device name and partition mode, and
records when, with which tool versions, and under which partition and power cap
it was measured. Partition mode is part of the identity rather than metadata on
it: splitting an MI355X into CPX changes the bandwidth one slice can reach, so
the same card under SPX is a different machine.

Adding a machine means measuring it once and committing the result. Nothing
cross-checks a committed profile against a fresh measurement, because there is
no longer a fresh measurement. A roof that reads low is the dangerous
direction: the ceiling derived from it is too loose, so the kernel reads as
closer to done than it is, and a campaign with an attainment target stops with
the work half finished. Hence :func:`validate_profile`, and hence a profile
that fails it is refused rather than used.

The self-check catches only what the datasheet contradicts. Two transcription
errors it cannot see are worth knowing about when authoring an entry. A roof
left unscaled -- ``roofline.csv`` reports GFLOP/s and GB/s, so every figure
needs a factor of 1e9 -- produces attainment above one on every case, which
excludes them all and leaves the target unable to fire; loud, but only once a
campaign runs. A dispatch floor timed eagerly rather than from a captured graph
comes out around three times too high, because eager timing also pays the
framework's per-op host submission that a graph-timed driver never pays, and
nothing downstream can tell.
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

from kernelforge.fusion.gpu_arch import canon_arch, detect_arch
from kernelforge.resources import resource_path
from kernelforge.roofline_ceiling.specs import arch_spec, equal_rate_paths

log = logging.getLogger("kernelforge.roofline_ceiling")

PROFILE_SCHEMA_VERSION = 1
REFERENCE_DIRNAME = "device_profiles"

#: How far a committed figure may sit above its datasheet peak and still be
#: read as rounding rather than a transcription error. A measurement cannot
#: beat the vendor peak, so anything past this is wrong by construction.
_OVER_DATASHEET_TOLERANCE = 1.02

#: How far two instruction paths the datasheet rates identically may drift in a
#: committed profile. The failure this catches is rocprofiler-compute reporting
#: bf16 MFMA at exactly half fp16 on a chip that runs both at one rate; left in,
#: every bf16 ceiling comes out twice as loose as it should.
_EQUAL_RATE_TOLERANCE = 0.05

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


def validate_profile(profile: DeviceProfile, arch: str) -> list[str]:
    """Return everything wrong with a committed profile, judged against the datasheet.

    This is the only check left on a hand-authored figure, so it is deliberately
    narrow: it reports what the vendor's own numbers contradict and nothing it
    would have to guess at. Two rules cover the transcription errors that
    actually happen.

    A figure above its datasheet peak cannot be a measurement. It is a column
    read off by the wrong name, a unit left unscaled, or a row taken for the
    wrong device.

    Two instruction paths the vendor documents as a single rate must stay
    together. The case this exists for is rocprofiler-compute reporting bf16
    MFMA at exactly half fp16 on a chip that runs both at one rate: a roof wrong
    in that direction makes the ceiling too loose, and a campaign with an
    attainment target stops early believing the kernel is done.

    The pairs come from :data:`~kernelforge.roofline_ceiling.specs.EQUAL_RATE_PATHS`
    rather than from paths that happen to share a datasheet peak. Measured
    throughput within one datasheet group legitimately differs -- gfx950 rates
    fp4 and fp6 together and a real card measures them 17% apart -- so inferring
    the rule from equal peaks flags the chip's own behaviour as a mistake.
    """
    spec = arch_spec(arch)
    if spec is None:
        return []

    problems: list[str] = []
    for path, value in sorted(profile.peak_flops.items()):
        peak = float(spec.peak_flops.get(path) or 0.0)
        if peak > 0 and value > peak * _OVER_DATASHEET_TOLERANCE:
            problems.append(
                f"{path} is {value:.6g} FLOP/s, above the {arch} datasheet peak of {peak:.6g}; "
                "no measurement beats the vendor peak, so this figure was read off wrongly"
            )
    hbm = float(profile.bandwidth.get("hbm") or 0.0)
    if spec.hbm_bw_bytes_per_s > 0 and hbm > spec.hbm_bw_bytes_per_s * _OVER_DATASHEET_TOLERANCE:
        problems.append(
            f"hbm bandwidth is {hbm:.6g} B/s, above the {arch} datasheet peak of "
            f"{spec.hbm_bw_bytes_per_s:.6g}; no measurement beats the vendor peak"
        )

    for left, right in equal_rate_paths(arch):
        a, b = float(profile.peak_flops.get(left) or 0.0), float(profile.peak_flops.get(right) or 0.0)
        if a <= 0 or b <= 0:
            continue
        if abs(a / b - 1.0) > _EQUAL_RATE_TOLERANCE:
            problems.append(
                f"{left} is {a:.6g} and {right} is {b:.6g}, but {arch} runs both at one rate; "
                "a profiler that halves one of them was transcribed without the correction"
            )
    return problems


def _parse_profile(payload: dict[str, Any], origin: str, arch: str = "") -> DeviceProfile | None:
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
    if not profile.usable:
        return None

    problems = validate_profile(profile, arch or str(payload.get("match", {}).get("arch") or ""))
    for problem in problems:
        log.error("device profile %s is not self-consistent: %s", origin, problem)
    if problems:
        log.error(
            "refusing device profile %s; the campaign falls back to datasheet peaks, which read "
            "attainment far too low rather than far too high",
            origin,
        )
        return None
    return profile


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
        profile = _parse_profile(payload, origin=f"shipped:{path.name}", arch=identity.arch)
        if profile is not None:
            return profile
    return None


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "REFERENCE_DIRNAME",
    "DeviceIdentity",
    "DeviceProfile",
    "describe_device",
    "load_reference",
    "validate_profile",
]
