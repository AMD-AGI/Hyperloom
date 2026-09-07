# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assemble complete AMDHSA source into a loadable code object with ROCm LLVM."""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

_TARGET = re.compile(r"gfx[0-9][0-9a-f]{2,3}(?::(?:xnack|sramecc)[+-])*")
_TARGET_DIRECTIVE = re.compile(r'^\s*\.amdgcn_target\s+"amdgcn-amd-amdhsa--([^"\n]+)"', re.MULTILINE)


class AssemblyError(RuntimeError):
    """Invalid assembly input or a failed ROCm assembler/linker invocation."""


def _target_parts(target: str) -> tuple[str, dict[str, str]]:
    if not _TARGET.fullmatch(target):
        raise AssemblyError(
            f"Invalid GPU target {target!r}; expected a gfx target such as gfx950 or gfx90a:sramecc+:xnack-. "
            "Only xnack and sramecc target features are supported."
        )
    cpu, *feature_parts = target.split(":")
    features: dict[str, str] = {}
    for part in feature_parts:
        name, setting = part[:-1], part[-1]
        if name in features:
            raise AssemblyError(f"Duplicate target feature {name!r} in {target!r}")
        features[name] = setting
    return cpu, features


def _validate_source(source_text: str, gpu_target: str) -> tuple[str, dict[str, str]]:
    expected = _target_parts(gpu_target)
    for directive in ("amdhsa_kernel", "end_amdhsa_kernel", "amdgpu_metadata", "end_amdgpu_metadata"):
        if not re.search(rf"^\s*\.{directive}\b", source_text, re.MULTILINE):
            raise AssemblyError(
                f"Missing .{directive}: supply complete compiler-generated AMDHSA assembly, "
                "including kernel descriptors and metadata; instruction-only disassembly cannot be reassembled."
            )
    targets = _TARGET_DIRECTIVE.findall(source_text)
    if len(targets) != 1:
        raise AssemblyError("Expected exactly one .amdgcn_target directive for amdgcn-amd-amdhsa")
    if _target_parts(targets[0]) != expected:
        raise AssemblyError(
            f"Assembly target {targets[0]!r} does not match requested GPU target {gpu_target!r}; "
            "include the same xnack/sramecc features."
        )
    return expected


def _run_tool(command: list[str], *, cwd: Path, deadline: float, stage: str) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssemblyError(f"Assembly timeout expired before {stage}")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssemblyError(f"{stage} timed out: {command[0]}") from exc
    except OSError as exc:
        raise AssemblyError(f"Could not execute {stage} tool {command[0]}: {exc}") from exc
    if result.returncode:
        diagnostics = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        raise AssemblyError(f"{stage} failed with exit code {result.returncode}: {command[0]}\n{diagnostics}".rstrip())


def _require_artifact(path: Path, stage: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssemblyError(f"{stage} reported success but did not produce a nonempty artifact: {path}")


def assemble(
    source: Path,
    output: Path,
    *,
    gpu_target: str,
    toolchain_dir: Path,
    timeout_sec: float = 60,
) -> Path:
    """Build an HSACO from full assembly and return its absolute path.

    The toolchain directory must contain ROCm's llvm-mc and ld.lld. The
    requested target, including xnack/sramecc features, must match the source's
    .amdgcn_target. Descriptors and metadata are passed through byte-for-byte.
    Assembly and linking share one timeout and always run against fresh staged
    files; no cache is used. On failure this raises without returning an output.
    A previous output is preserved until a new code object is atomically ready;
    callers must propagate failures rather than load that previous output.

    This does not convert instruction-only disassembly or launch a GPU kernel.
    """
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise AssemblyError("timeout_sec must be positive and finite")
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source == output:
        raise AssemblyError("The output must differ from the assembly source")
    source_bytes = source.read_bytes()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"Assembly source is not UTF-8 text: {source}") from exc
    cpu, features = _validate_source(source_text, gpu_target)

    toolchain_dir = Path(toolchain_dir).expanduser().resolve()
    assembler = toolchain_dir / "llvm-mc"
    linker = toolchain_dir / "ld.lld"
    for tool in (assembler, linker):
        if not tool.is_file():
            raise AssemblyError(f"Missing ROCm LLVM tool: {tool}; set toolchain_dir to the ROCm LLVM bin directory")

    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    # Staging beside the destination keeps os.replace atomic across filesystems.
    with tempfile.TemporaryDirectory(prefix=".forge-assembly-", dir=output.parent) as temporary:
        staging = Path(temporary)
        staged_source = staging / "kernel.s"
        relocatable = staging / "kernel.o"
        code_object = staging / "kernel.hsaco"
        staged_source.write_bytes(source_bytes)
        command = [str(assembler), "-triple=amdgcn-amd-amdhsa", f"-mcpu={cpu}", "-filetype=obj"]
        if features:
            command.append("-mattr=" + ",".join(setting + name for name, setting in sorted(features.items())))
        command.extend([str(staged_source), "-o", str(relocatable)])
        _run_tool(command, cwd=staging, deadline=deadline, stage="AMDGPU assembly")
        _require_artifact(relocatable, "AMDGPU assembly")
        _run_tool(
            [str(linker), "-shared", "--no-undefined", str(relocatable), "-o", str(code_object)],
            cwd=staging,
            deadline=deadline,
            stage="AMDGPU linking",
        )
        _require_artifact(code_object, "AMDGPU linking")
        os.replace(code_object, output)
    return output
