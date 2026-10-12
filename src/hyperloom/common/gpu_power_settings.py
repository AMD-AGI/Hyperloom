# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only view of the GPU power settings a session is measured under.

The power cap and the DPM performance level decide how much of a card's throughput is available and at what power,
so they are part of the measurement contract. Nothing here changes them: setting either is privileged and card-wide,
and belongs to the operator before launch (``amd-smi set --power-cap`` / ``--perf-level``), the same way
``--compute-partition-mode`` asserts a mode rather than setting one. This module reads what the cards are at, so a
session can record it and refuse to start when an operator's declared value did not take effect.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - fixed read-only amd-smi invocations.
from typing import Any, Callable, Mapping

__all__ = [
    "GpuPowerSettingsError",
    "declared_setting_problems",
    "normalize_perf_level",
    "read_gpu_power_settings",
    "visible_gpu_indices",
]

_PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"


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


def read_gpu_power_settings(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[int, dict[str, Any]]:
    """``{gpu: {"power_cap_w": float | None, "perf_level": str}}`` for every card ``amd-smi`` lists.

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
    for entry in settings.values():
        entry.setdefault("power_cap_w", None)
        entry.setdefault("perf_level", "")
    return settings


def declared_setting_problems(
    settings: Mapping[int, Mapping[str, Any]],
    *,
    power_cap_w: float | None = None,
    perf_level: str | None = None,
    gpus: set[int] | None = None,
) -> list[str]:
    """Why the cards are not at the declared settings; empty when they are, or when nothing is declared.

    A declared value that cannot be read counts as a mismatch: an assertion nobody verified is not a satisfied one.
    """
    wanted_level = normalize_perf_level(perf_level) if perf_level else ""
    checked = {gpu: row for gpu, row in settings.items() if gpus is None or gpu in gpus}
    if (power_cap_w is not None or wanted_level) and not checked:
        return ["no GPU the session can use was reported by amd-smi"]
    problems: list[str] = []
    for gpu, row in sorted(checked.items()):
        if power_cap_w is not None:
            cap = row.get("power_cap_w")
            if not isinstance(cap, (int, float)) or abs(float(cap) - float(power_cap_w)) > 0.5:
                problems.append(
                    f"GPU {gpu} power cap is {cap if cap is not None else 'unreadable'} W, declared {power_cap_w:g} W"
                )
        if wanted_level and row.get("perf_level") != wanted_level:
            problems.append(
                f"GPU {gpu} perf level is {row.get('perf_level') or 'unreadable'!r}, declared {wanted_level!r}"
            )
    return problems
