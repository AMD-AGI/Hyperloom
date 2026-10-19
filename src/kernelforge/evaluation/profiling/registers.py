# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Register analysis tool — extract VGPR/AGPR/spill from compiled kernels."""

from __future__ import annotations

import asyncio
import glob
from pathlib import Path

from kernelforge.evaluation.profiling.compiler_output import parse_register_info
from kernelforge.tools.process_group import communicate_process_group


async def check_registers(build_dir: str | None = None) -> dict:
    """Analyze register usage from the first compiled kernel binary in a build directory."""
    if build_dir is None:
        return {"success": False, "message": "No build_dir provided"}

    so_files = glob.glob(str(Path(build_dir) / "*.so"))
    if not so_files:
        return {
            "success": False,
            "message": f"No .so files found in {build_dir}",
        }
    binary_path = so_files[0]

    # Disassemble
    cmd = ["llvm-objdump", "-d", "--mcpu=gfx950", binary_path]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await communicate_process_group(proc, timeout=60)
    except asyncio.TimeoutError:
        return {"success": False, "message": "llvm-objdump timed out"}

    output = stdout.decode(errors="replace")

    if proc.returncode != 0:
        # Try alternative: readelf for metadata
        cmd2 = ["readelf", "-n", binary_path]
        proc2 = await asyncio.create_subprocess_exec(
            *cmd2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout2, _ = await communicate_process_group(proc2, timeout=30)
        output = stdout2.decode(errors="replace")

    info = parse_register_info(output)

    return {
        "success": True,
        "vgpr": info.vgpr,
        "agpr": info.agpr,
        "sgpr": info.sgpr,
        "lds_bytes": info.lds_bytes,
        "spill_bytes": info.spill_bytes,
        "has_spill": info.has_spill,
        "occupancy": info.occupancy,
        "message": f"REGISTERS: {info.summary()}",
    }
