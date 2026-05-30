"""GPU hardware detection and specs.

Auto-detects GPU type from rocminfo/nvidia-smi and provides hardware
constants (compute units, architecture, memory) that launch scripts
and plugins may need.

Supports AMD MI300X, MI308X, MI325X, MI350X, MI355X and NVIDIA
H100, H200, B200, B300, GB200, GB300.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class GpuSpec:
    """Hardware specification for a GPU family."""

    name: str           # e.g. "MI300X", "H200"
    arch: str           # e.g. "gfx942", "sm_90a"
    compute_units: int  # CUs (AMD) or SMs (NVIDIA)
    memory_gb: int      # HBM capacity
    vendor: str         # "amd" or "nvidia"

    @property
    def env_vars(self) -> dict[str, str]:
        """Environment variables useful for launch scripts on this GPU."""
        env: dict[str, str] = {}
        if self.vendor == "amd":
            env["GPU_ARCHS"] = self.arch
            env["CU_NUM"] = str(self.compute_units)
        return env


# Known GPU specifications
_GPU_SPECS: dict[str, GpuSpec] = {
    "MI300X": GpuSpec("MI300X", "gfx942", 304, 192, "amd"),
    "MI308X": GpuSpec("MI308X", "gfx942", 80, 128, "amd"),
    "MI325X": GpuSpec("MI325X", "gfx942", 304, 256, "amd"),
    "MI350X": GpuSpec("MI350X", "gfx950", 320, 288, "amd"),
    "MI355X": GpuSpec("MI355X", "gfx950", 320, 288, "amd"),
    "H100": GpuSpec("H100", "sm_90a", 132, 80, "nvidia"),
    "H200": GpuSpec("H200", "sm_90a", 132, 141, "nvidia"),
    "B200": GpuSpec("B200", "sm_100a", 192, 192, "nvidia"),
    "B300": GpuSpec("B300", "sm_100a", 192, 288, "nvidia"),
    "GB200": GpuSpec("GB200", "sm_100a", 192, 192, "nvidia"),
    "GB300": GpuSpec("GB300", "sm_100a", 192, 288, "nvidia"),
}


def get_spec(gpu_name: str) -> GpuSpec | None:
    """Look up GPU spec by name (case-insensitive, fuzzy)."""
    key = gpu_name.upper().replace("-", "").replace(" ", "")
    for name, spec in _GPU_SPECS.items():
        if name.replace("-", "") in key or key in name.replace("-", ""):
            return spec
    return None


def get_spec_by_arch(arch: str) -> GpuSpec | None:
    """Look up GPU spec by architecture string."""
    for spec in _GPU_SPECS.values():
        if spec.arch == arch:
            return spec
    return None


@lru_cache(maxsize=1)
def detect_gpu() -> GpuSpec | None:
    """Auto-detect the GPU type from the system.

    Tries rocminfo (AMD) then nvidia-smi (NVIDIA).
    Returns None if detection fails.
    """
    override = os.environ.get("HYPERLOOM_GPU_TYPE", "")
    if override:
        return get_spec(override)

    spec = _detect_amd()
    if spec:
        return spec
    return _detect_nvidia()


def _detect_amd() -> GpuSpec | None:
    """Detect AMD GPU via rocminfo."""
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        return None
    try:
        result = subprocess.run(
            [rocminfo], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        import re
        agents = re.split(r"Agent\s+\d+", result.stdout)
        for agent in agents:
            lines = agent.splitlines()
            is_gpu = any("Device Type:" in l and "GPU" in l for l in lines)
            if not is_gpu:
                continue

            arch = ""
            cu_count = 0
            for line in lines:
                stripped = line.strip()
                if not arch and stripped.startswith("Name:") and "gfx" in stripped:
                    val = stripped.split(":", 1)[-1].strip()
                    match = re.search(r"(gfx\d+)", val)
                    if match:
                        arch = match.group(1)
                if "Compute Unit:" in stripped:
                    try:
                        cu_count = int(stripped.split(":")[-1].strip())
                    except ValueError:
                        pass

            if not arch:
                continue

            for s in _GPU_SPECS.values():
                if s.arch == arch and s.compute_units == cu_count:
                    return s

            return get_spec_by_arch(arch)

        return None

    except (subprocess.TimeoutExpired, OSError):
        return None


def _detect_nvidia() -> GpuSpec | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not name:
            return None

        return get_spec(name)

    except (subprocess.TimeoutExpired, OSError):
        return None
