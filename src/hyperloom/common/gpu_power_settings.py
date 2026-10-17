# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only view of the GPU power settings a session is measured under.

The power cap, the DPM performance level and the graphics clock's maximum (sclk) decide how much of a card's
throughput is available and at what power, so they are part of the measurement contract. Nothing here changes them:
setting any of them is privileged and card-wide, and belongs to the operator before launch (``amd-smi set
--power-cap`` / ``--perf-level`` / ``--clk-limit sclk max``) or to ``--apply-gpu-power-settings``, the same way
``--compute-partition-mode`` asserts a mode rather than setting one. This module reads what the cards are at, so a
session can record it and refuse to start when an operator's declared value did not take effect.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - fixed read-only amd-smi invocations.
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from hyperloom.common.env import env_flag

__all__ = [
    "GPU_POWER_SAMPLING_ENV",
    "GpuPowerSettingsError",
    "DeclaredPowerSettings",
    "PERF_LEVEL_MANUAL",
    "declared_combination_error",
    "declared_setting_problems",
    "gpu_power_sampling_unavailable",
    "normalize_perf_level",
    "read_gpu_power_settings",
    "visible_gpu_indices",
]

_PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"
#: The perf level a clock limit puts a card in, and the only one at which the limit holds.
PERF_LEVEL_MANUAL = "manual"
_UNREADABLE = "unreadable"

#: Escape hatch for a host where running ``amd-smi`` every couple of seconds is not acceptable.
GPU_POWER_SAMPLING_ENV = "HYPERLOOM_GPU_POWER_SAMPLING"


def gpu_power_sampling_unavailable() -> str:
    """Why no round on this host can be sampled, or ``""``."""
    if not env_flag(GPU_POWER_SAMPLING_ENV, default=True):
        return f"{GPU_POWER_SAMPLING_ENV}=0 turns the GPU power sampler off"
    if shutil.which("amd-smi") is None:
        return "amd-smi is not on PATH, so the GPU power sampler has nothing to read"
    return ""


class GpuPowerSettingsError(RuntimeError):
    """``amd-smi`` could not be run or its output could not be read."""


def normalize_perf_level(value: Any) -> str:
    """``AMDSMI_DEV_PERF_LEVEL_AUTO`` and ``auto`` both read as ``"auto"``; ``""`` when absent."""
    text = str(value or "").strip()
    if text.upper().startswith(_PERF_LEVEL_PREFIX):
        text = text[len(_PERF_LEVEL_PREFIX) :]
    return text.lower()


def visible_gpu_indices(env: Mapping[str, str] | None = None) -> set[int] | None:
    """Card indices the session may use, from ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``; ``None`` for all."""
    source = os.environ if env is None else env
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw = str(source.get(name) or "").strip()
        if not raw:
            continue
        try:
            return {int(part) for part in raw.split(",") if part.strip()}
        except ValueError:
            return None
    return None


def _gpu_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("gpu"), int)]


def _watts(block: Any) -> float | None:
    if isinstance(block, dict):
        block = block.get("value")
    return float(block) if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def _power_cap_w(limit: Any) -> float | None:
    """The socket power limit, whichever key this amd-smi release uses for it."""
    if not isinstance(limit, dict):
        return None
    for key in ("ppt0", "ppt"):
        section = limit.get(key)
        if isinstance(section, dict) and (cap := _watts(section.get("socket_power_limit"))) is not None:
            return cap
    for key in ("socket_power_limit", "power_cap"):
        if (cap := _watts(limit.get(key))) is not None:
            return cap
    return None


@dataclass(frozen=True)
class DeclaredPowerSettings:
    """The GPU power settings a launch declares; ``None`` (or ``""`` for the level) for one it leaves alone."""

    power_cap_w: float | None = None
    perf_level: str = ""
    sclk_max_mhz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "perf_level", normalize_perf_level(self.perf_level) if self.perf_level else "")

    def as_row(self) -> dict[str, Any]:
        """The settings keyed as a card's reading is."""
        return {"power_cap_w": self.power_cap_w, "perf_level": self.perf_level, "sclk_max_mhz": self.sclk_max_mhz}


def _sclk_max_mhz(clock: Any) -> float | None:
    """The graphics clock's maximum limit, read on every XCD; ``None`` when unread or the XCDs disagree.

    ``amd-smi metric --clock`` reports one ``gfx_<n>`` block per XCD, and ``max_clk`` is the limit in force: an
    ``amd-smi set -L sclk max`` shows there, not in ``static --clock``.
    """
    if not isinstance(clock, dict):
        return None
    limits = {
        _watts(block.get("max_clk"))
        for name, block in clock.items()
        if str(name).startswith("gfx_") and isinstance(block, dict)
    }
    return limits.pop() if len(limits) == 1 else None


def declared_combination_error(*, perf_level: str | None, sclk_max_mhz: float | None) -> str:
    """Why the declared settings cannot hold together, or ``""``.

    Setting a clock limit puts a card in the MANUAL perf level, and setting any other level resets the limit, so a
    declared sclk cap holds only at MANUAL.
    """
    level = normalize_perf_level(perf_level) if perf_level else ""
    if sclk_max_mhz is not None and level and level != PERF_LEVEL_MANUAL:
        return (
            f"--gpu-sclk-max-mhz holds only at the manual perf level, and --gpu-perf-level declares {level!r}: "
            "amd-smi puts a card in manual when a clock limit is set, and any other level resets the limit"
        )
    return ""


def read_gpu_power_settings(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[int, dict[str, Any]]:
    """``{gpu: {"power_cap_w": float | None, "perf_level": str, "sclk_max_mhz": float | None}}`` per listed card.

    Raises:
        GpuPowerSettingsError: ``amd-smi`` is absent, failed, or printed something that is not its JSON.
    """
    if shutil.which("amd-smi") is None and run is subprocess.run:
        raise GpuPowerSettingsError("amd-smi is not on PATH")

    def _query(*args: str) -> Any:
        try:
            done = run(["amd-smi", *args, "--json"], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
        if done.returncode != 0:
            raise GpuPowerSettingsError(
                f"amd-smi {' '.join(args)} exited {done.returncode}: {done.stderr.strip()[:200]}"
            )
        try:
            return json.loads(done.stdout)
        except ValueError as exc:
            raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} printed no JSON") from exc

    settings: dict[int, dict[str, Any]] = {}
    for row in _gpu_rows(_query("static", "--limit")):
        settings.setdefault(row["gpu"], {})["power_cap_w"] = _power_cap_w(row.get("limit"))
    for row in _gpu_rows(_query("metric", "--perf-level")):
        settings.setdefault(row["gpu"], {})["perf_level"] = normalize_perf_level(row.get("perf_level"))
    try:
        clocks = _gpu_rows(_query("metric", "--clock"))
    except GpuPowerSettingsError:
        # Where only the clock read fails, the sclk limit reads as unread: that refuses a declared --gpu-sclk-max-mhz,
        # not a cap or perf level that can still be checked.
        clocks = []
    for row in clocks:
        if row["gpu"] in settings:
            settings[row["gpu"]]["sclk_max_mhz"] = _sclk_max_mhz(row.get("clock"))
    for entry in settings.values():
        entry.setdefault("power_cap_w", None)
        entry.setdefault("perf_level", "")
        entry.setdefault("sclk_max_mhz", None)
    return settings


def declared_setting_problems(
    settings: Mapping[int, Mapping[str, Any]],
    *,
    power_cap_w: float | None = None,
    perf_level: str | None = None,
    sclk_max_mhz: float | None = None,
    gpus: set[int] | None = None,
) -> list[str]:
    """Why the cards are not at the declared settings; empty when they are, or when nothing is declared.

    A declared value that cannot be read counts as a mismatch: an assertion nobody verified is not a satisfied one.
    """
    wanted_level = normalize_perf_level(perf_level) if perf_level else ""
    checked = {gpu: row for gpu, row in settings.items() if gpus is None or gpu in gpus}
    if (power_cap_w is not None or wanted_level or sclk_max_mhz is not None) and not checked:
        return ["no GPU the session can use was reported by amd-smi"]
    problems: list[str] = []
    for gpu, row in sorted(checked.items()):
        if power_cap_w is not None:
            cap = row.get("power_cap_w")
            if not isinstance(cap, (int, float)) or abs(float(cap) - float(power_cap_w)) > 0.5:
                problems.append(
                    f"GPU {gpu} power cap is {cap if cap is not None else _UNREADABLE} W, declared {power_cap_w:g} W"
                )
        if wanted_level and row.get("perf_level") != wanted_level:
            problems.append(
                f"GPU {gpu} perf level is {row.get('perf_level') or _UNREADABLE!r}, declared {wanted_level!r}"
            )
        if sclk_max_mhz is not None:
            sclk = row.get("sclk_max_mhz")
            if not isinstance(sclk, (int, float)) or abs(float(sclk) - float(sclk_max_mhz)) > 0.5:
                problems.append(
                    f"GPU {gpu} sclk max is {f'{sclk:g} MHz' if sclk is not None else _UNREADABLE}, "
                    f"declared {sclk_max_mhz:g} MHz"
                )
    return problems
