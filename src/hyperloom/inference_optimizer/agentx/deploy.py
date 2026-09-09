# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime deployment of AgentX assets into the InferenceX benchmarks dir."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_ASSET_FILES = ("aiperf_client.sh", "map_aiperf.py", "aiperf_phase_gate.py")


def agentx_asset_dir() -> Path:
    """Return the packaged ``assets/agentx`` directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "agentx"


def deploy_agentx_assets(benchmarks_dir: str | Path) -> list[Path]:
    """Copy AgentX assets into ``benchmarks_dir`` (idempotent)."""
    src_dir = agentx_asset_dir()
    dst_dir = Path(benchmarks_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in _ASSET_FILES:
        src = src_dir / name
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
