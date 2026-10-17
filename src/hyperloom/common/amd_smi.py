# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared read-only ``amd-smi`` probes: VRAM usage, device count, telemetry, ASIC identity.

``rocm-smi`` is deprecated from ROCm 7.0 and is no longer built as part of the standard ROCm 10.1 build, so a stack
Hyperloom supports can ship ``amd-smi`` and nothing else -- the ROCm 10 pip/TheRock wheels already do. There is
deliberately no ``rocm-smi`` fallback here: every ROCm Hyperloom declares support for ships ``amd-smi`` with every
subcommand this module uses, so a second parser would serve no supported configuration.

The one real compatibility burden is that Hyperloom supports ROCm 7.2.x and 10.0 *at once*, i.e. two amd-smi majors
whose JSON has moved between them: a metric scalar may be a bare number or a ``{"value", "unit"}`` object, a field may
be the string ``"N/A"`` instead of a number, and whole keys appear and disappear between releases. Everything here
reads through :func:`scalar` and :func:`lookup` for that reason, the way ``gpu_power_settings.py`` already does for the
socket power cap.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # nosec B404 - fixed read-only amd-smi invocations.
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

log = logging.getLogger(__name__)

_TIMEOUT_SEC = 20.0

#: Share of a card's VRAM above which something is resident on it.
VRAM_BUSY_FRACTION = 0.01

#: Utilization above which a card counts as in use by someone else.
GPU_BUSY_UTILIZATION_PCT = 50.0

#: MiB per unit amd-smi may label a memory size with. Its "MB" is mebibytes -- an MI355X with 288 GiB reports 294912
#: of them -- so the binary divisor is the right one, and "GB" is read the same way for consistency with it. Same
#: convention as ``gpu_partition._VRAM_UNIT_GIB``.
_MIB_PER_UNIT: dict[str, float] = {
    "B": 1.0 / 1024**2,
    "KB": 1.0 / 1024,
    "KIB": 1.0 / 1024,
    "MB": 1.0,
    "MIB": 1.0,
    "GB": 1024.0,
    "GIB": 1024.0,
    "TB": 1024.0**2,
    "TIB": 1024.0**2,
}

#: Keys ``metric --usage`` has carried the graphics-busy percentage under. The APU average is last because it is a
#: windowed mean the Instinct parts do not publish at all.
_GFX_ACTIVITY_KEYS: tuple[str, ...] = ("gfx_activity", "graphics_activity", "gfx_usage", "apu_average_gfx_activity")

#: Keys ``metric --power`` has carried the socket draw under, mirroring ``gpu_power_settings._power_cap_w``.
_SOCKET_POWER_KEYS: tuple[str, ...] = ("socket_power", "average_socket_power", "apu_average_socket_power")

#: Keys ``metric --temperature`` has carried a representative die temperature under, hottest-first.
_TEMPERATURE_KEYS: tuple[str, ...] = ("hotspot", "junction", "edge", "apu_temperature_gfx")


class AmdSmiError(RuntimeError):
    """``amd-smi`` is absent, failed, or printed something that is not its JSON."""


class GpuVram(NamedTuple):
    """Per-GPU VRAM usage in MiB."""

    used_mib: float
    total_mib: float


@dataclass(frozen=True)
class GpuStatus:
    """One card's telemetry. Every reading is optional: amd-smi reports ``"N/A"`` for what a part does not expose."""

    gpu_id: int
    temperature_c: float | None
    power_w: float | None
    utilization_pct: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None

    @property
    def busy(self) -> bool:
        """Whether the card is under load by this probe's threshold; unknown utilization reads as not busy."""
        return self.utilization_pct is not None and self.utilization_pct > GPU_BUSY_UTILIZATION_PCT


@dataclass(frozen=True)
class GpuAsic:
    """One card's identity, as ``amd-smi static --asic`` reports it."""

    gpu_id: int
    market_name: str
    target_graphics_version: str
    num_compute_units: int | None


def amd_smi_available() -> bool:
    """Whether ``amd-smi`` is on PATH."""
    return shutil.which("amd-smi") is not None


def amd_smi_json(args: Sequence[str], *, timeout_s: float = _TIMEOUT_SEC) -> Any:
    """Run a read-only ``amd-smi`` subcommand with ``--json`` and parse its stdout.

    Args:
        args: Subcommand and flags, without ``amd-smi`` and without ``--json``.
        timeout_s: Seconds to wait before giving up on the probe.

    Returns:
        Any: The parsed payload, whose top level is a list for ``list``/``firmware``/``process`` and a dict with a
            ``gpu_data`` list for ``static``/``metric``.

    Raises:
        AmdSmiError: ``amd-smi`` is not on PATH, could not be run, exited non-zero, or did not print its JSON.
    """
    cmd = ["amd-smi", *args, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)  # nosec B603 B607
    except FileNotFoundError as exc:
        raise AmdSmiError("amd-smi is not on PATH") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise AmdSmiError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
    if proc.returncode != 0:
        raise AmdSmiError(f"amd-smi {' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        # Also the no-driver case: amd-smi logs its "Unable to detect any GPU devices" to stderr and still exits 0,
        # so an absent GPU arrives here as empty stdout rather than as a failed command.
        raise AmdSmiError(f"amd-smi {' '.join(args)} printed no parseable JSON") from exc


def _json_or_none(args: Sequence[str], *, timeout_s: float = _TIMEOUT_SEC) -> Any | None:
    """Best-effort :func:`amd_smi_json`: ``None`` instead of a raise, for the probes that degrade silently."""
    try:
        return amd_smi_json(args, timeout_s=timeout_s)
    except AmdSmiError as exc:
        log.debug("amd-smi %s unavailable: %s", " ".join(args), exc)
        return None


def gpu_rows(payload: Any) -> list[dict[str, Any]]:
    """Return the per-GPU dicts in an amd-smi payload, whether it wraps them in ``gpu_data`` or is a bare list.

    Args:
        payload: A parsed amd-smi payload.

    Returns:
        list[dict[str, Any]]: One dict per card, in amd-smi's enumeration order.
    """
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict)]


def scalar(block: Any) -> float | None:
    """Read a metric that may be a bare number or a ``{"value", "unit"}`` object.

    Args:
        block: The raw field, which amd-smi renders as a number, as a unit-wrapped object (since ROCm 6.1), or as the
            string ``"N/A"`` when the part does not expose it.

    Returns:
        float | None: The numeric value, or ``None`` when it is absent or not reported.
    """
    if isinstance(block, dict):
        block = block.get("value")
    if isinstance(block, bool) or not isinstance(block, (int, float)):
        return None
    return float(block)


def lookup(mapping: Any, *names: str) -> Any:
    """Return the first of ``names`` present in ``mapping`` with a reportable value.

    amd-smi renames and drops keys between releases -- ``metric --power`` on ROCm 10.2 carries ``gfx_voltage`` and
    friends that ROCm 7.15 does not have at all -- so a caller names every spelling it will accept rather than one.

    Args:
        mapping: The block to read, which need not be a dict.
        *names: Candidate keys, most-preferred first.

    Returns:
        Any: The first value found that is neither missing nor the ``"N/A"`` amd-smi prints for an unsupported field;
            ``None`` when none of them qualifies.
    """
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        if value is None or (isinstance(value, str) and value.strip().upper() in ("", "N/A")):
            continue
        return value
    return None


def _mib(block: Any) -> float | None:
    """Convert an amd-smi memory field to MiB, honouring whichever unit it labels itself with."""
    value = scalar(block)
    if value is None:
        return None
    unit = str(block.get("unit") or "").strip().upper() if isinstance(block, dict) else "MB"
    scale = _MIB_PER_UNIT.get(unit)
    if scale is None:
        return None
    return value * scale


def vram_busy(gpu: GpuVram) -> bool:
    """Whether something holds VRAM on ``gpu``."""
    return gpu.used_mib > gpu.total_mib * VRAM_BUSY_FRACTION


def gpu_vram_usage() -> list[GpuVram] | None:
    """Return per-GPU VRAM usage, or ``None`` when it cannot be read.

    Only ``used_vram`` and ``total_vram`` are read, by exact name. The neighbouring ``*_visible_vram`` and ``*_gtt``
    fields are deliberately not matched: on an APU, GTT is 48 GiB where VRAM is 2 GiB, so a loose key match would
    report a number 24x too large rather than fail.

    Returns:
        list[GpuVram] | None: One entry per card in amd-smi's order, or ``None`` when amd-smi is absent, failed, or
            reported no card with a positive total.
    """
    payload = _json_or_none(["metric", "--mem-usage"])
    if payload is None:
        return None
    result: list[GpuVram] = []
    for row in gpu_rows(payload):
        mem = row.get("mem_usage")
        if not isinstance(mem, dict):
            continue
        used = _mib(lookup(mem, "used_vram"))
        total = _mib(lookup(mem, "total_vram"))
        if used is None or total is None:
            # A card that answers with a used figure and no total cannot settle the ratio gate every caller applies,
            # so the whole probe is unknown rather than partially trusted.
            return None
        if total <= 0.0:
            return None
        result.append(GpuVram(used, total))
    return result or None


def gpu_count() -> int:
    """Return how many GPUs ``amd-smi`` enumerates, or 0 when it cannot be asked.

    This is the device probe only; it applies no ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` mask, because
    amd-smi does not honour them either. Callers that scope to an operator's mask read the mask themselves first.

    Returns:
        int: The number of enumerated cards; 0 when amd-smi is absent or failed.
    """
    payload = _json_or_none(["list"], timeout_s=5.0)
    if payload is None:
        return 0
    return len(gpu_rows(payload))


def gpu_status() -> list[GpuStatus]:
    """Return per-GPU temperature, power, utilization and VRAM; empty when amd-smi cannot be read.

    Returns:
        list[GpuStatus]: One entry per card, with ``None`` for any reading the part does not expose.
    """
    payload = _json_or_none(["metric", "--usage", "--mem-usage", "--temperature", "--power"], timeout_s=15.0)
    if payload is None:
        return []
    out: list[GpuStatus] = []
    for index, row in enumerate(gpu_rows(payload)):
        gpu_id = row.get("gpu")
        mem = row.get("mem_usage")
        out.append(
            GpuStatus(
                gpu_id=gpu_id if isinstance(gpu_id, int) and not isinstance(gpu_id, bool) else index,
                temperature_c=scalar(lookup(row.get("temperature"), *_TEMPERATURE_KEYS)),
                power_w=scalar(lookup(row.get("power"), *_SOCKET_POWER_KEYS)),
                utilization_pct=scalar(lookup(row.get("usage"), *_GFX_ACTIVITY_KEYS)),
                memory_used_mib=_mib(lookup(mem, "used_vram")),
                memory_total_mib=_mib(lookup(mem, "total_vram")),
            )
        )
    return out


def gpu_asic_info() -> list[GpuAsic] | None:
    """Return each card's marketing name and gfx target, or ``None`` when amd-smi cannot be read.

    ``market_name`` and ``target_graphics_version`` come from the same call, which is why this replaces the
    ``rocm-smi --showproductname`` plus torch-``gcnArchName`` pair: on a part whose marketing name the driver cannot
    resolve, ``market_name`` degrades to the raw device id (``"0x1586"``) while the gfx target stays exact.

    Returns:
        list[GpuAsic] | None: One entry per card, or ``None`` when amd-smi is absent or failed.
    """
    payload = _json_or_none(["static", "--asic"])
    if payload is None:
        return None
    out: list[GpuAsic] = []
    for index, row in enumerate(gpu_rows(payload)):
        asic = row.get("asic") if isinstance(row.get("asic"), dict) else row
        gpu_id = row.get("gpu")
        cu = lookup(asic, "num_compute_units", "compute_units", "num_cu", "cu_count")
        out.append(
            GpuAsic(
                gpu_id=gpu_id if isinstance(gpu_id, int) and not isinstance(gpu_id, bool) else index,
                market_name=str(lookup(asic, "market_name", "product_name") or ""),
                target_graphics_version=str(lookup(asic, "target_graphics_version", "gfx_version") or ""),
                num_compute_units=int(cu) if isinstance(cu, int) and not isinstance(cu, bool) else None,
            )
        )
    return out or None


__all__ = [
    "GPU_BUSY_UTILIZATION_PCT",
    "VRAM_BUSY_FRACTION",
    "AmdSmiError",
    "GpuAsic",
    "GpuStatus",
    "GpuVram",
    "amd_smi_available",
    "amd_smi_json",
    "gpu_asic_info",
    "gpu_count",
    "gpu_rows",
    "gpu_status",
    "gpu_vram_usage",
    "lookup",
    "scalar",
    "vram_busy",
]
