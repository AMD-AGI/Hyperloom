# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every power-related reading a node exposes, read once: sysfs files and the parsing of ``amd-smi`` JSON.

The 1 s stream reads sysfs only: a read of every card's files costs microseconds, where one ``amd-smi`` call costs a
Python start-up (about 0.12 s of CPU) and ``amd-smi metric`` with every field about a second of wall time. sysfs also
reads a card amd-smi reports as ``N/A``: on MI355X nodes amd-smi can lose one card's ``socket_power`` while its hwmon
``power1_input`` still reads.

A card's amd-smi index is tied to its sysfs files through the PCI address ``amd-smi static`` reports for it, since
amd-smi numbers cards in PCI order and the DRM card numbers follow probe order. The node's NPM files sit in the
``board`` directory of one card per XGMI hive; the CPU's energy counters are RAPL powercap zones or the
``amd_energy`` hwmon, whichever the kernel exposes. A source a node does not have is absent, never zero.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed read-only amd-smi invocations.
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.amd_smi import NPM_ENABLED, gpu_rows, metric_value

__all__ = [
    "FAST_COLUMNS",
    "CpuDomain",
    "GpuFiles",
    "NodeFiles",
    "SysfsNode",
    "amd_smi_query",
    "bdf_by_gpu",
    "energy_by_gpu",
    "flatten_metric",
]

_QUERY_TIMEOUT_SEC = 30.0
_MICRO = 1e6
_MILLI = 1e3

#: The columns of one card's 1 s reading, in ``power.json``'s ``samples.fast`` order after ``mono, unix, gpu``.
FAST_COLUMNS = (
    "power_w",
    "junction_c",
    "mem_c",
    "sclk_mhz",
    "mclk_mhz",
    "vddboard_mv",
    "gfx_busy_pct",
    "umc_busy_pct",
    "vram_used_b",
    "gtt_used_b",
)

#: hwmon channels by the label amdgpu gives them, with the divisor that turns the raw value into the column's unit.
_HWMON_CHANNELS = {
    ("power", "PPT"): ("power_w", _MICRO),
    ("temp", "junction"): ("junction_c", _MILLI),
    ("temp", "mem"): ("mem_c", _MILLI),
    ("freq", "sclk"): ("sclk_mhz", _MICRO),
    ("freq", "mclk"): ("mclk_mhz", _MICRO),
    ("in", "vddboard"): ("vddboard_mv", 1.0),
}
#: Device files read with the card's hwmon, by column.
_DEVICE_FILES = {
    "gfx_busy_pct": "gpu_busy_percent",
    "umc_busy_pct": "mem_busy_percent",
    "vram_used_b": "mem_info_vram_used",
    "gtt_used_b": "mem_info_gtt_used",
}
_CAP_FILES = {"cap": "power1_cap", "default": "power1_cap_default", "min": "power1_cap_min", "max": "power1_cap_max"}
_DPM_TABLES = ("pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "pp_dpm_socclk", "pp_dpm_dclk", "pp_dpm_vclk")
_PERF_LEVEL = "power_dpm_force_performance_level"
_VRAM_TOTAL = "mem_info_vram_total"
_AMD_ENERGY = "amd_energy"
#: amd_energy publishes per-core and per-socket counters; the socket ones cover the cores, so only they are summed.
_AMD_ENERGY_SOCKET = "Esocket"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_number(path: Path, divisor: float = 1.0) -> float | None:
    text = _read_text(path)
    try:
        return None if text is None else float(text) / divisor
    except ValueError:
        return None


def amd_smi_query(*args: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Any:
    """The parsed ``amd-smi <args> --json`` output.

    Raises:
        OSError, subprocess.SubprocessError, RuntimeError, ValueError: amd-smi could not run, failed, or printed no JSON.
    """
    done = run(
        ["amd-smi", *args, "--json"],  # nosec B603 B607
        capture_output=True,
        text=True,
        timeout=_QUERY_TIMEOUT_SEC,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"amd-smi {' '.join(args)} exited {done.returncode}: {(done.stderr or '').strip()[:200]}")
    return json.loads(done.stdout)


def bdf_by_gpu(static: Any) -> dict[int, str]:
    """``{amd-smi index: PCI address}`` from ``amd-smi static --json``."""
    out: dict[int, str] = {}
    for row in gpu_rows(static):
        bus = row.get("bus")
        if isinstance(bus, dict) and isinstance(bus.get("bdf"), str):
            out[row["gpu"]] = bus["bdf"].lower()
    return out


def energy_by_gpu(payload: Any) -> dict[int, float]:
    """``{gpu: accumulated energy J}`` from ``amd-smi metric -E --json``; a card without a reading is left out."""
    out: dict[int, float] = {}
    for row in gpu_rows(payload):
        energy = row.get("energy")
        value = metric_value(energy.get("total_energy_consumption")) if isinstance(energy, dict) else None
        if value is not None:
            out[row["gpu"]] = value
    return out


def _flatten(value: Any, path: str, values: dict[str, Any], units: dict[str, str]) -> None:
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        values[path] = value["value"]
        units[path] = str(value["unit"])
    elif isinstance(value, dict):
        for key, inner in value.items():
            _flatten(inner, f"{path}.{key}" if path else str(key), values, units)
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            _flatten(inner, f"{path}.{index}", values, units)
    else:
        values[path] = None if value == "N/A" else value


def flatten_metric(payload: Any) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    """Each card's ``amd-smi metric --json`` row as ``{dotted key path: value}``, and the unit of each path.

    ``{"value": v, "unit": u}`` readings become ``v`` with ``u`` in the units; ``N/A`` becomes ``None``; status strings
    such as ``ACTIVE`` are kept.
    """
    rows: dict[int, dict[str, Any]] = {}
    units: dict[str, str] = {}
    for row in gpu_rows(payload):
        values: dict[str, Any] = {}
        _flatten({key: value for key, value in row.items() if key != "gpu"}, "", values, units)
        rows[row["gpu"]] = values
    return rows, units


@dataclass(frozen=True)
class GpuFiles:
    """One card's sysfs files: the hwmon channels and device files of :data:`FAST_COLUMNS`, by column."""

    gpu: int
    bdf: str
    device: Path
    hwmon: Path | None
    columns: Mapping[str, tuple[Path, float]]

    @classmethod
    def find(cls, root: Path, gpu: int, bdf: str) -> GpuFiles:
        device = root / "bus" / "pci" / "devices" / bdf
        hwmon = next(iter(sorted((device / "hwmon").glob("hwmon*"))), None)
        columns: dict[str, tuple[Path, float]] = {}
        for label_file in sorted(hwmon.glob("*_label")) if hwmon is not None else ():
            stem = label_file.name[: -len("_label")]
            channel = _HWMON_CHANNELS.get((stem.rstrip("0123456789"), _read_text(label_file) or ""))
            if channel is not None:
                columns[channel[0]] = (label_file.with_name(f"{stem}_input"), channel[1])
        for column, name in _DEVICE_FILES.items():
            if (device / name).exists():
                columns[column] = (device / name, 1.0)
        return cls(gpu, bdf, device, hwmon, columns)

    def read(self) -> tuple[float | None, ...]:
        """One reading of every :data:`FAST_COLUMNS` column; a file that is absent or fails to read is ``None``."""
        values = {column: _read_number(path, divisor) for column, (path, divisor) in self.columns.items()}
        power = values.get("power_w")
        # No card draws 0 W: a non-positive socket power is a sensor that failed to read.
        if power is not None and power <= 0:
            values["power_w"] = None
        return tuple(values.get(column) for column in FAST_COLUMNS)

    def static(self) -> dict[str, Any]:
        """The card's power caps, performance level, DPM tables and VRAM size."""
        caps = {
            name: _read_number(self.hwmon / file, _MICRO) if self.hwmon is not None else None
            for name, file in _CAP_FILES.items()
        }
        return {
            "bdf": self.bdf,
            "hwmon": None if self.hwmon is None else self.hwmon.name,
            "caps_w": caps,
            "perf_level": _read_text(self.device / _PERF_LEVEL),
            "pp_dpm": {name: _read_text(self.device / name) for name in _DPM_TABLES},
            "vram_total_b": _read_number(self.device / _VRAM_TOTAL),
        }


@dataclass(frozen=True)
class NodeFiles:
    """The node's Node Power Management files, in the ``board`` directory of the card that publishes them."""

    board: Path

    def status(self) -> str | None:
        return _read_text(self.board / "npm_status")

    def read(self) -> tuple[float | None, float | None]:
        """``(node power W, enforced node limit W)``; either is ``None`` when its file does not read.

        With NPM disabled firmware reports both as 0, which is no reading: no node draws 0 W or is held to it.
        """
        values = (_read_number(self.board / "node_power"), _read_number(self.board / "cur_node_power_limit"))
        power, limit = (value if value is not None and value > 0 else None for value in values)
        return power, limit

    def static(self) -> dict[str, Any]:
        """Every file of the board directory, verbatim, with the NPM status and limit read out."""
        files = {path.name: _read_text(path) for path in sorted(self.board.iterdir()) if path.is_file()}
        return {"board": str(self.board), "status": self.status(), "limit_w": self.read()[1], "files": files}


@dataclass(frozen=True)
class CpuDomain:
    """One CPU energy counter: a RAPL zone or an ``amd_energy`` channel, in microjoules.

    ``counted`` marks the domains whose sum is the CPU's energy: top-level RAPL zones, or the per-socket
    ``amd_energy`` channels; a subzone or a core counter is inside one of those.
    """

    name: str
    path: Path
    wrap_uj: float | None
    counted: bool

    def read(self) -> float | None:
        return _read_number(self.path)


@dataclass(frozen=True)
class SysfsNode:
    """The sysfs tree power is read from; the root is ``/sys`` on a host."""

    root: Path = Path("/sys")

    def gpu_files(self, bdfs: Mapping[int, str]) -> dict[int, GpuFiles]:
        return {gpu: GpuFiles.find(self.root, gpu, bdf) for gpu, bdf in sorted(bdfs.items())}

    def node_files(self) -> NodeFiles | None:
        """The NPM files of the first card that publishes them, enabled first; ``None`` on a node without NPM.

        Without an XGMI hive every card publishes its own copy; one reading disabled says nothing of the others.
        """
        boards = [NodeFiles(path) for path in sorted(self.root.glob("class/drm/card*/device/board"))]
        published = [board for board in boards if board.status() is not None]
        enabled = [board for board in published if board.status() == NPM_ENABLED]
        return next(iter(enabled or published), None)

    def cpu_domains(self) -> tuple[str | None, tuple[CpuDomain, ...]]:
        """``(source, domains)``: RAPL zones when the kernel exposes any, else ``amd_energy`` channels, else none."""
        rapl = self._rapl_domains()
        if rapl:
            return "rapl", rapl
        amd_energy = self._amd_energy_domains()
        return ("amd_energy", amd_energy) if amd_energy else (None, ())

    def _rapl_domains(self) -> tuple[CpuDomain, ...]:
        domains = []
        for zone in sorted(self.root.glob("class/powercap/*")):
            if not (zone / "energy_uj").exists():
                continue
            name = f"rapl:{zone.name}:{_read_text(zone / 'name') or ''}"
            top_level = zone.name.count(":") == 1
            domains.append(CpuDomain(name, zone / "energy_uj", _read_number(zone / "max_energy_range_uj"), top_level))
        return tuple(domains)

    def _amd_energy_domains(self) -> tuple[CpuDomain, ...]:
        domains = []
        for hwmon in sorted(self.root.glob("class/hwmon/hwmon*")):
            if _read_text(hwmon / "name") != _AMD_ENERGY:
                continue
            for label_file in sorted(hwmon.glob("energy*_label")):
                label = _read_text(label_file) or label_file.name
                path = label_file.with_name(f"{label_file.name[: -len('_label')]}_input")
                domains.append(
                    CpuDomain(f"amd_energy:{hwmon.name}:{label}", path, None, label.startswith(_AMD_ENERGY_SOCKET))
                )
        return tuple(domains)
