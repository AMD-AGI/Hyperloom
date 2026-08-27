# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Register analysis tool — extract VGPR/AGPR/spill from compiled kernels."""

from __future__ import annotations

import asyncio
import glob
from pathlib import Path

from kernelforge.mcp_server.parsers.compiler_output import parse_register_info
from kernelforge.mcp_server.tools._subprocess import communicate_process_group


async def check_registers(
    binary_path: str | None = None,
    build_dir: str | None = None,
    kernel_name: str | None = None,
    gpu_target: str = "gfx950",
) -> dict:
    """Analyze register usage from a compiled kernel binary.

    Uses llvm-objdump to disassemble and extract register metadata.

    Args:
        binary_path: Direct path to .so or .hsaco file.
        build_dir: Directory to search for .so files.
        kernel_name: If provided, filter output to this kernel.
        gpu_target: GPU target for disassembly (default gfx950).

    Returns:
        Dict with: register_info, occupancy_analysis, message.
    """
    # Find binary
    if binary_path is None and build_dir is not None:
        so_files = glob.glob(str(Path(build_dir) / "*.so"))
        if not so_files:
            return {
                "success": False,
                "message": f"No .so files found in {build_dir}",
            }
        binary_path = so_files[0]

    if binary_path is None:
        return {"success": False, "message": "No binary_path or build_dir provided"}

    if not Path(binary_path).exists():
        return {"success": False, "message": f"Binary not found: {binary_path}"}

    # Disassemble
    cmd = ["llvm-objdump", "-d", f"--mcpu={gpu_target}", binary_path]
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

    # Filter to specific kernel if requested
    if kernel_name and kernel_name in output:
        # Extract the section for this kernel
        sections = output.split("\n\n")
        filtered = [s for s in sections if kernel_name in s]
        if filtered:
            output = "\n\n".join(filtered)

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
