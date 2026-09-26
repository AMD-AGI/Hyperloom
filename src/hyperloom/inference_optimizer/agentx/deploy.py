# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime deployment of AgentX assets into the InferenceX benchmarks dir."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

# The client the AgentX switch pins as ``benchmark_script``. Downstream consumers
# that must tell "this recipe drives a client" from "this recipe launches a
# server" match on this name, so it lives here beside the deployment that
# publishes it rather than being spelled out again at each reader.
AIPERF_CLIENT_SCRIPT = "aiperf_client.sh"
MLPERF_CLIENT_SCRIPT = "mlperf_agentic_client.sh"
AGENTX_CLIENT_SCRIPT = AIPERF_CLIENT_SCRIPT
AGENTX_CLIENT_SCRIPTS = frozenset({AIPERF_CLIENT_SCRIPT, MLPERF_CLIENT_SCRIPT})

_ASSET_FILES = (
    AIPERF_CLIENT_SCRIPT,
    MLPERF_CLIENT_SCRIPT,
    "map_aiperf.py",
    "map_mlperf.py",
    "aiperf_phase_gate.py",
)

# map_aiperf.py / map_mlperf.py import the mapping from their own directory under
# this name; the prefix keeps it from clobbering an InferenceX file in the shared
# benchmarks dir.
_MAPPING_MODULE = "agentx_mapping.py"


def agentic_backend(env: Mapping[str, str] | None = None) -> str:
    """Return ``mlperf`` or ``aiperf`` from ``HYPERLOOM_AGENTIC_BACKEND``."""
    runtime = env or os.environ
    raw = str(runtime.get("HYPERLOOM_AGENTIC_BACKEND") or "").strip().lower()
    if raw in {"mlperf", "mlperf-agentic"}:
        return "mlperf"
    return "aiperf"


def is_mlperf_backend(env: Mapping[str, str] | None = None) -> bool:
    return agentic_backend(env) == "mlperf"


def agentx_client_script(env: Mapping[str, str] | None = None) -> str:
    """The Magpie ``benchmark_script`` this AgentX backend pins."""
    return MLPERF_CLIENT_SCRIPT if is_mlperf_backend(env) else AIPERF_CLIENT_SCRIPT


def is_agentx_client_script(name: str | None) -> bool:
    return Path(str(name or "")).name in AGENTX_CLIENT_SCRIPTS


# Wall-clock a single agentic trajectory costs, measured on 8xMI355X Kimi-K3 at
# concurrency 16: 25 trajectories in 629-666s and 150 in 3849-4046s, i.e. ~26s
# either way, so the workload scales linearly in trajectory count.
MLPERF_SECONDS_PER_TRAJECTORY = 26

# Server boot, aiter JIT rebuild and drain sit outside the measured window.
MLPERF_BOOT_ALLOWANCE_SEC = 1800

# A first, cold round runs materially slower than the steady state (measured:
# 15 req/min against 48 once warm), so the cap is sized for the cold case.
MLPERF_TIMEOUT_SAFETY_FACTOR = 2.0

MLPERF_CANONICAL_TRAJECTORIES = 613
MLPERF_SMOKE_TRAJECTORIES = 150


def mlperf_trajectories(env: Mapping[str, str] | None = None) -> int:
    """Trajectory count this run will issue, from the flow and its override."""
    runtime = env or os.environ
    raw = str(runtime.get("AGENTIC_NUM_TRAJECTORIES") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    flow = str(runtime.get("MLPERF_AGENTIC_FLOW") or "smoke_test").strip()
    return MLPERF_SMOKE_TRAJECTORIES if flow == "smoke_test" else MLPERF_CANONICAL_TRAJECTORIES


def mlperf_benchmark_timeout_sec(env: Mapping[str, str] | None = None, *, floor: float = 0.0) -> float:
    """Benchmark cap sized for this run's trajectory count.

    The stock cap is sized for the aiperf 3600s measurement window; a full 613
    trajectory run needs roughly four hours and is killed by it well short of a
    result. Never returns less than ``floor`` so the derivation can only raise
    the cap, never tighten one the operator or the default already set.
    """
    trajectories = mlperf_trajectories(env)
    derived = trajectories * MLPERF_SECONDS_PER_TRAJECTORY * MLPERF_TIMEOUT_SAFETY_FACTOR
    return max(float(floor), derived + MLPERF_BOOT_ALLOWANCE_SEC)


def agentx_asset_dir() -> Path:
    """Return the packaged ``assets/agentx`` directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "agentx"


def deploy_agentx_assets(benchmarks_dir: str | Path) -> list[Path]:
    """Copy AgentX assets and the ``mapping`` module they import into ``benchmarks_dir`` (idempotent)."""
    src_dir = agentx_asset_dir()
    sources = {name: src_dir / name for name in _ASSET_FILES}
    sources[_MAPPING_MODULE] = Path(__file__).resolve().with_name("mapping.py")
    dst_dir = Path(benchmarks_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(f"AgentX asset missing from package: {src}")
        dst = dst_dir / name
        # Atomic publish: copy to a temp file in the same dir, set mode, then os.replace() (atomic rename).
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=str(dst_dir))
        os.close(fd)
        try:
            shutil.copyfile(src, tmp)
            os.chmod(tmp, 0o700 if name.endswith(".sh") else 0o600)
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        written.append(dst)
    return written
