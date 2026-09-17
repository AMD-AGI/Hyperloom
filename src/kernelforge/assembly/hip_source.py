# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile an explicit HIP device translation unit and bind its compiler ISA."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

from kernelforge.assembly.compiler import (
    AssemblyError,
    _require_artifact,
    _run_tool,
    _target_parts,
    _validate_source,
    assemble,
)


def _compile(
    source: Path, gpu_target: str, flags: tuple[str, ...], rocm: Path, timeout_sec: float
) -> tuple[bytes, str]:
    _target_parts(gpu_target)
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise AssemblyError("HIP compilation timeout must be positive and finite")
    # Clang's default CUID hashes the random temporary output path. Use a stable
    # unit ID so static-symbol names survive fresh builds and clean patch replay.
    cuid = hashlib.sha256(source.read_bytes() + json.dumps([gpu_target, flags]).encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory(prefix="forge-hip-source-") as scratch:
        directory = Path(scratch)
        binary = directory / "kernel.hsaco"
        _run_tool(
            [
                str(rocm / "bin/hipcc"),
                *flags,
                "--genco",
                "-save-temps=obj",
                "-g0",
                "-cuid=" + cuid,
                "--offload-arch=" + gpu_target,
                str(source.resolve()),
                "-o",
                str(binary),
            ],
            cwd=Path.cwd(),
            deadline=time.monotonic() + timeout_sec,
            stage="HIP device compilation",
        )
        _require_artifact(binary, "HIP device compilation")
        sources = list(directory.glob("*.s"))
        if len(sources) != 1:
            raise AssemblyError("HIP capture requires exactly one device ISA file; extract one device translation unit")
        isa = sources[0].read_text(encoding="utf-8")
        _validate_source(isa, gpu_target)
        return binary.read_bytes(), isa


def compile_hip(
    source: Path | str,
    output: Path | str,
    *,
    gpu_target: str,
    flags: tuple[str, ...] = ("-O3",),
    timeout_sec: float = 120,
) -> Path:
    """Build standalone HIP device code for an existing, explicit HIP launcher.

    Supply the original build's defines/include paths in ``flags``. The caller
    owns the kernel symbol, argument ABI, launch geometry and stream. This does
    not extract or replace kernels inside an already-linked Python extension.
    """
    binary, _ = _compile(Path(source), gpu_target, flags, Path(os.environ.get("ROCM_PATH", "/opt/rocm")), timeout_sec)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    return output


class HIPAssembly:
    """Replace one explicit compile_hip boundary while retaining its caller's ABI."""

    def __init__(self, module: str, assembly: str, target: str, *, export: bool = False):
        self.source = Path(module).resolve().parent / assembly
        self.manifest = self.source.with_suffix(self.source.suffix + ".json")
        self.target = target
        self.export = export

    def __call__(
        self,
        source: Path | str,
        output: Path | str,
        *,
        gpu_target: str,
        flags: tuple[str, ...] = ("-O3",),
        timeout_sec: float = 120,
    ) -> Path:
        if _target_parts(gpu_target) != _target_parts(self.target):
            raise AssemblyError("HIP build target does not match the assembly campaign")
        rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        binary, isa = _compile(Path(source), gpu_target, flags, rocm, timeout_sec)
        normalized = re.sub(r"^\s*\.(?:file|loc)\s+.*$", "", isa, flags=re.MULTILINE)
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        identity = dict(schema_version=1, frontend="hip", gpu_target=self.target, source_ir_sha256=fingerprint)
        if self.export and not self.manifest.exists():
            self.source.write_text(isa, encoding="utf-8")
            self.manifest.write_text(
                json.dumps(dict(identity, compiler_assembly_sha256=hashlib.sha256(isa.encode()).hexdigest()), indent=2)
                + "\n",
                encoding="utf-8",
            )
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise AssemblyError("HIP compiler output differs from the captured source/build contract")
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.export:
            output.write_bytes(binary)
            return output
        return assemble(
            self.source, output, gpu_target=gpu_target, toolchain_dir=rocm / "llvm/bin", timeout_sec=timeout_sec
        )
