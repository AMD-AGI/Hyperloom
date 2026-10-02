# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime deployment of AgentX assets into the InferenceX benchmarks dir."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hyperloom.common.agentx_workload import AIPERF_CLIENT_SCRIPT, MLPERF_CLIENT_SCRIPT

_ASSET_FILES = (
    AIPERF_CLIENT_SCRIPT,
    MLPERF_CLIENT_SCRIPT,
    "map_aiperf.py",
    "map_mlperf.py",
    "aiperf_phase_gate.py",
    "agentx_launch_capture.py",
)

# map_aiperf.py / map_mlperf.py import the mapping from their own directory under
# this name; the prefix keeps it from clobbering an InferenceX file in the shared
# benchmarks dir.
_MAPPING_MODULE = "agentx_mapping.py"


def agentx_asset_dir() -> Path:
    """Return the packaged ``assets/agentx`` directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "agentx"


def deploy_agentx_assets(benchmarks_dir: str | Path) -> list[Path]:
    """Copy AgentX assets and the ``mapping`` module they import into ``benchmarks_dir`` (idempotent)."""
    src_dir = agentx_asset_dir()
    sources = {name: src_dir / name for name in _ASSET_FILES}
    sources[_MAPPING_MODULE] = Path(__file__).resolve().with_name("mapping.py")
    common = Path(__file__).resolve().parents[2] / "common"
    for name in ("__init__.py", "serving_launch.py", "env_safety.py", "visible_devices.py", "proctree.py"):
        sources[f"_hyperloom_launch/{name}"] = common / name
    dst_dir = Path(benchmarks_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(f"AgentX asset missing from package: {src}")
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish: copy to a temp file in the same dir, set mode, then os.replace() (atomic rename).
        fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=str(dst.parent))
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
