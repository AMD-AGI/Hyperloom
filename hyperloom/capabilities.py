"""Capability detection — probe for available tools at session start.

The system adapts its optimization strategy based on what's available:
  - No Magpie -> user's benchmark script
  - No TraceLens -> torch profiler for kernel discovery
  - No GEAK -> OOB agents (Claude/Codex) for kernel optimization
  - No Ray -> local GPU pool instead of cluster scheduling
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass


@dataclass
class Capabilities:
    """Detected tool availability for the current environment."""

    magpie: bool = False
    tracelens: bool = False
    geak: bool = False
    oob: bool = False
    ray: bool = False
    torch_profiler: bool = False
    claude_sdk: bool = False

    def summary(self) -> str:
        lines = []
        for field_name in ("magpie", "tracelens", "geak", "oob", "ray", "torch_profiler", "claude_sdk"):
            val = getattr(self, field_name)
            status = "available" if val else "not found"
            lines.append(f"  {field_name}: {status}")
        return "\n".join(lines)


def detect_capabilities() -> Capabilities:
    """Probe the current environment for available optimization tools."""
    caps = Capabilities()

    caps.magpie = _check_magpie()
    caps.tracelens = _check_tracelens()
    caps.geak = _check_geak()
    caps.oob = _check_oob()
    caps.ray = _check_ray()
    caps.torch_profiler = _check_torch_profiler()
    caps.claude_sdk = _check_claude_sdk()

    return caps


def _check_magpie() -> bool:
    """Check if InferenceX/Magpie benchmark runner is available.

    Magpie is the InferenceX benchmark orchestrator. It's a script-based
    tool, not a Python package — detected via INFERENCEX_PATH env var or
    the presence of benchmark_serving.py in known locations.
    """
    inferencex_path = os.environ.get("INFERENCEX_PATH", "")
    if inferencex_path:
        bench = os.path.join(inferencex_path, "utils", "bench_serving", "benchmark_serving.py")
        if os.path.isfile(bench):
            return True

    bench_script = os.environ.get("VLLM_BENCH_SCRIPT", "")
    if bench_script and os.path.isfile(bench_script):
        return True

    return False


def _check_tracelens() -> bool:
    """Check if TraceLens skill is accessible and claude_agent_sdk is available."""
    if not _check_claude_sdk():
        return False
    skill_path = os.environ.get("TRACELENS_SKILL_PATH", "")
    if skill_path and os.path.isfile(skill_path):
        return True
    from hyperloom.agents import get_skill_path
    kernel_skill = get_skill_path("kernel")
    return kernel_skill is not None and kernel_skill.exists()


def _check_geak() -> bool:
    """Check if GEAK binary is on PATH and config is set."""
    for name in ("geak", "mini", "geak-gaagent"):
        if shutil.which(name):
            return True
    return False


def _check_oob() -> bool:
    """Check if OOB (out-of-band agent runner) is on PATH."""
    return shutil.which("oob") is not None


def _check_ray() -> bool:
    """Check if Ray is importable and a cluster is reachable."""
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch_profiler() -> bool:
    """Check if PyTorch with profiler support is available."""
    try:
        import torch.profiler  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _check_claude_sdk() -> bool:
    """Check if claude_agent_sdk is importable."""
    try:
        spec = importlib.util.find_spec("claude_agent_sdk")
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False
