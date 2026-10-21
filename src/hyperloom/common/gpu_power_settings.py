# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only view of the GPU power settings a session is measured under.

The power cap and the DPM performance level decide how much of a card's throughput is available and at what power, so
they are part of the measurement contract. Nothing here changes them: setting either is privileged and card-wide, and
belongs to the operator before launch (``amd-smi set --power-cap`` / ``--perf-level``) or to
``--apply-gpu-power-settings``, the same way
``--compute-partition-mode`` asserts a mode rather than setting one. This module reads what the cards are at, so a
session can record it and refuse to start when an operator's declared value did not take effect.

Node Power Management (NPM, MI350 series) is the node-wide counterpart: firmware holds the sum of every card's power to
a node limit and moves power between cards, so whether it is on and at what limit is part of the same contract. It is
read and asserted here alongside the per-card settings; it is enabled out of band (the BMC), never by a session.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed read-only amd-smi invocations.
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from hyperloom.common.env import env_flag

__all__ = [
    "GPU_POWER_SAMPLING_ENV",
    "GpuPowerSettingsError",
    "DeclaredPowerSettings",
    "NODE_POWER_LIMIT_W",
    "NPM",
    "NPM_STATES",
    "PERF_LEVEL",
    "PERF_LEVELS",
    "PERF_LEVEL_MANUAL",
    "POWER_CAP_W",
    "declared_node_problems",
    "declared_setting_problems",
    "gpu_power_sampling_unavailable",
    "normalize_perf_level",
    "read_gpu_power_settings",
    "read_node_power_settings",
    "visible_gpu_indices",
]

_PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"
#: The setting keys a card's power settings, a declaration and a record share.
POWER_CAP_W = "power_cap_w"
PERF_LEVEL = "perf_level"
#: The DPM performance levels ``amd-smi set --perf-level`` accepts, lowercased.
PERF_LEVELS = (
    "auto",
    "low",
    "high",
    "manual",
    "stable_std",
    "stable_peak",
    "stable_min_mclk",
    "stable_min_sclk",
    "determinism",
)
#: The perf level an operator sets to tune a card's clocks by hand.
PERF_LEVEL_MANUAL = "manual"
#: The node-wide setting keys a node's reading, a declaration and a record share.
NPM = "npm"
NODE_POWER_LIMIT_W = "node_power_limit_w"
#: The NPM states ``amd-smi node -p`` reports, lowercased.
NPM_STATES = ("enabled", "disabled")
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


#: The masks the ROCm runtime honours, ROCr level then HIP level. HSA_VISIBLE_DEVICES and GPU_DEVICE_ORDINAL are left
#: out: ROCm ignores them, so a card they name is not one the session's servers are limited to.
_ROCR_MASKS = ("ROCR_VISIBLE_DEVICES",)
_HIP_MASKS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


#: The only spelling the HIP runtime reads as written: it stops at a space, an empty or ``;``-joined token or a leading
#: zero, so a looser parse would name cards the session's servers never see.
_MASK_SPELLING = re.compile(r"(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*")


def _first_mask(source: Mapping[str, str], names: tuple[str, ...]) -> list[int] | None:
    """The visible-device ordinals of the first of *names* that is set; ``None`` when none is."""
    for name in names:
        if name not in source:
            continue
        raw = str(source.get(name) or "")
        if not raw.strip():
            raise GpuPowerSettingsError(f"{name} is set but empty, so the session sees no cards")
        if not _MASK_SPELLING.fullmatch(raw):
            raise GpuPowerSettingsError(
                f"{name}={raw!r} is not a plain comma-separated list of device numbers, so the cards the runtime "
                "exposes for it cannot be named"
            )
        ordinals = [int(token) for token in raw.split(",")]
        if len(set(ordinals)) != len(ordinals):
            raise GpuPowerSettingsError(f"{name}={raw} repeats a card, which the runtime does not expose as written")
        return ordinals
    return None


def visible_gpu_indices(env: Mapping[str, str] | None = None) -> set[int] | None:
    """The amd-smi indices of the cards the session's visible-device masks expose; ``None`` when no mask is set.

    ROCm numbers GPUs in KFD node order and amd-smi in PCI order, and the two differ on MI300X/MI355X boards, so a mask
    number is never an amd-smi index. ``ROCR_VISIBLE_DEVICES`` picks from the KFD-ordered cards and
    ``HIP_VISIBLE_DEVICES`` (or ``CUDA_VISIBLE_DEVICES``) then indexes into what it left; each resulting card is matched
    to amd-smi through its KFD node id.

    Raises:
        GpuPowerSettingsError: a mask is set but empty, not numeric, repeats a card or names one that does not exist,
            or amd-smi cannot say which KFD node each card is.
    """
    source = os.environ if env is None else env
    rocr = _first_mask(source, _ROCR_MASKS)
    hip = _first_mask(source, _HIP_MASKS)
    if rocr is None and hip is None:
        return None
    by_node = [gpu for gpu, _ in sorted(_read_gpu_kfd_nodes().items(), key=lambda item: item[1])]
    visible = list(range(len(by_node))) if rocr is None else rocr
    if hip is not None:
        if any(index >= len(visible) for index in hip):
            raise GpuPowerSettingsError(f"HIP_VISIBLE_DEVICES names a card beyond the {len(visible)} visible")
        visible = [visible[index] for index in hip]
    if any(ordinal >= len(by_node) for ordinal in visible):
        raise GpuPowerSettingsError(f"the visible-device mask names a card beyond the {len(by_node)} amd-smi lists")
    return {by_node[ordinal] for ordinal in visible}


def _gpu_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("gpu"), int)]


def _value(block: Any) -> float | None:
    if isinstance(block, dict):
        block = block.get("value")
    return float(block) if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def _power_cap_w(limit: Any) -> float | None:
    """The socket power limit, whichever key this amd-smi release uses for it."""
    if not isinstance(limit, dict):
        return None
    for key in ("ppt0", "ppt"):
        section = limit.get(key)
        if isinstance(section, dict) and (cap := _value(section.get("socket_power_limit"))) is not None:
            return cap
    for key in ("socket_power_limit", "power_cap"):
        if (cap := _value(limit.get(key))) is not None:
            return cap
    return None


@dataclass(frozen=True)
class DeclaredPowerSettings:
    """The GPU power settings a launch declares; ``None`` (or ``""`` for a state) for one it leaves alone."""

    power_cap_w: float | None = None
    perf_level: str = ""
    npm: str = ""
    node_power_limit_w: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "perf_level", normalize_perf_level(self.perf_level) if self.perf_level else "")
        object.__setattr__(self, "npm", str(self.npm or "").lower())

    def as_row(self) -> dict[str, Any]:
        """The per-card settings keyed as a card's reading is; the only ones a session may apply."""
        return {"power_cap_w": self.power_cap_w, "perf_level": self.perf_level}

    def node_row(self) -> dict[str, Any]:
        """The node-wide settings keyed as the node's reading is."""
        return {NPM: self.npm, NODE_POWER_LIMIT_W: self.node_power_limit_w}


def _amd_smi_json(run: Callable[..., subprocess.CompletedProcess], timeout: float, *args: str) -> Any:
    if shutil.which("amd-smi") is None and run is subprocess.run:
        raise GpuPowerSettingsError("amd-smi is not on PATH")
    try:
        done = run(["amd-smi", *args, "--json"], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
    if done.returncode != 0:
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} exited {done.returncode}: {done.stderr.strip()[:200]}")
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise GpuPowerSettingsError(f"amd-smi {' '.join(args)} printed no JSON") from exc


def _read_gpu_kfd_nodes() -> dict[int, int]:
    """``{amd-smi index: KFD node id}`` for every card ``amd-smi list`` reports.

    Raises:
        GpuPowerSettingsError: ``amd-smi`` failed, or reported a card without its KFD node.
    """
    rows = _gpu_rows(_amd_smi_json(subprocess.run, 30.0, "list"))
    nodes = {row["gpu"]: row["node_id"] for row in rows if isinstance(row.get("node_id"), int)}
    if len(nodes) != len(rows):
        raise GpuPowerSettingsError("amd-smi list reports a card without its KFD node, so its ROCm number is unknown")
    return nodes


def read_gpu_power_settings(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[int, dict[str, Any]]:
    """``{gpu: {"power_cap_w": float | None, "perf_level": str}}`` for every card ``amd-smi`` lists.

    Raises:
        GpuPowerSettingsError: ``amd-smi`` is absent, failed, or printed something that is not its JSON.
    """
    settings: dict[int, dict[str, Any]] = {}
    for row in _gpu_rows(_amd_smi_json(run, timeout, "static", "--limit")):
        settings.setdefault(row["gpu"], {})["power_cap_w"] = _power_cap_w(row.get("limit"))
    for row in _gpu_rows(_amd_smi_json(run, timeout, "metric", "--perf-level")):
        settings.setdefault(row["gpu"], {})["perf_level"] = normalize_perf_level(row.get("perf_level"))
    for entry in settings.values():
        entry.setdefault("power_cap_w", None)
        entry.setdefault("perf_level", "")
    return settings


def read_node_power_settings(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """``{"npm": "enabled" | "disabled" | "", "node_power_limit_w": float | None}`` from ``amd-smi node -p``.

    The limit is the one firmware enforces, which can sit below the one the BMC was asked for.

    Raises:
        GpuPowerSettingsError: ``amd-smi`` is absent, failed (a release before ROCm 7.2 has no ``node`` command), or
            printed something that is not its JSON.
    """
    payload = _amd_smi_json(run, timeout, "node", "--power-management")
    rows = payload if isinstance(payload, list) else [payload]
    block = next((row["node"] for row in rows if isinstance(row, dict) and isinstance(row.get("node"), dict)), {})
    management = block.get("power_management") if isinstance(block.get("power_management"), dict) else {}
    status = str(management.get("status") or "").lower()
    return {
        NPM: status if status in NPM_STATES else "",
        NODE_POWER_LIMIT_W: _value(management.get("limit")),
    }


def declared_node_problems(
    node: Mapping[str, Any], *, npm: str | None = None, node_power_limit_w: float | None = None
) -> list[str]:
    """Why the node is not at the declared NPM settings; empty when it is, or when nothing is declared."""
    problems: list[str] = []
    if npm and node.get(NPM) != npm:
        problems.append(f"node power management is {node.get(NPM) or _UNREADABLE!r}, declared {npm!r}")
    if node_power_limit_w is not None:
        limit = node.get(NODE_POWER_LIMIT_W)
        if not isinstance(limit, (int, float)) or abs(float(limit) - float(node_power_limit_w)) > 0.5:
            problems.append(
                f"node power limit is {limit if limit is not None else _UNREADABLE} W, "
                f"declared {node_power_limit_w:g} W"
            )
    return problems


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
                    f"GPU {gpu} power cap is {cap if cap is not None else _UNREADABLE} W, declared {power_cap_w:g} W"
                )
        if wanted_level and row.get("perf_level") != wanted_level:
            problems.append(
                f"GPU {gpu} perf level is {row.get('perf_level') or _UNREADABLE!r}, declared {wanted_level!r}"
            )
    return problems
