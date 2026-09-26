# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture Triton/Gluon AMDGPU ISA and rebuild it through the original launcher."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from kernelforge.assembly.compiler import AssemblyError, _validate_source, assemble

if TYPE_CHECKING:
    from triton.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction


def _launch_contract(source: str) -> tuple[dict, dict]:
    descriptor = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)\s*\n(.*?)^\s*\.end_amdhsa_kernel", source, re.M | re.S)
    metadata = re.findall(r"^\s*\.amdgpu_metadata\s*\n(.*?)^\s*\.end_amdgpu_metadata", source, re.M | re.S)
    if len(descriptor) != 1 or len(metadata) != 1:
        raise AssemblyError("Triton assembly requires exactly one AMDHSA kernel and metadata block")
    resource_fields = {"amdhsa_next_free_vgpr", "amdhsa_next_free_sgpr", "amdhsa_accum_offset"}
    fields = dict(re.findall(r"^\s*\.(\w+)\s+([^\n/]+)", descriptor[0][1], re.M))
    fields = {key: value.strip() for key, value in fields.items() if key not in resource_fields}
    fields["symbol"] = descriptor[0][0]
    kernels = yaml.safe_load(metadata[0])["amdhsa.kernels"]
    if len(kernels) != 1:
        raise AssemblyError("Triton assembly requires single-kernel metadata")
    kernel = dict(kernels[0])
    for key in (".vgpr_count", ".sgpr_count", ".vgpr_spill_count", ".sgpr_spill_count", ".agpr_count"):
        kernel.pop(key, None)
    return fields, kernel


def _compiler_source(compiled: CompiledKernel, target: str) -> str:
    from triton.compiler import CompiledKernel

    if not isinstance(compiled, CompiledKernel):
        raise AssemblyError("expected a Triton CompiledKernel; freeze autotuning before the ASM campaign")
    if compiled.metadata.target.backend != "hip":
        raise AssemblyError("Triton ASM currently requires the AMD HIP backend")
    source = compiled.asm.get("amdgcn")
    if not isinstance(source, str):
        raise AssemblyError("Triton compiler did not retain complete amdgcn assembly")
    _validate_source(source, target)
    return source


def with_assembly(
    compiled: CompiledKernel,
    source: Path | str,
    *,
    gpu_target: str,
    toolchain_dir: Path | str,
    timeout_sec: float = 60,
) -> CompiledKernel:
    """Build an independent CompiledKernel without modifying Triton's JIT cache.

    The original signature, constexprs, launch metadata and argument marshaller
    remain intact. Shared memory and ABI declarations are frozen; register counts
    may change. Warm this callable before timing or graph capture.
    """
    original = _compiler_source(compiled, gpu_target)
    source = Path(source)
    candidate = source.read_text(encoding="utf-8")
    _validate_source(candidate, gpu_target)
    if _launch_contract(candidate) != _launch_contract(original):
        raise AssemblyError("assembly changed the Triton launch ABI or shared-memory contract")
    from triton.compiler import CompiledKernel

    build = tempfile.TemporaryDirectory(prefix="forge-triton-assembly-")
    directory = Path(build.name)
    try:
        binary = assemble(
            source,
            directory / "kernel.hsaco",
            gpu_target=gpu_target,
            toolchain_dir=Path(toolchain_dir),
            timeout_sec=timeout_sec,
        )
        digest = hashlib.sha256((compiled.hash + hashlib.sha256(binary.read_bytes()).hexdigest()).encode()).hexdigest()
        metadata = dict(compiled.metadata._asdict(), target=dataclasses.asdict(compiled.metadata.target), hash=digest)
        (directory / "kernel.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "kernel.amdgcn").write_text(candidate, encoding="utf-8")
        group = {path.name: str(path) for path in directory.iterdir()}
        variant = CompiledKernel(compiled.src, group, digest)
    except (OSError, ValueError, TypeError, AssemblyError):
        build.cleanup()
        raise
    variant._forge_build = build
    return variant


class TritonAssembly:
    """Bind one JIT kernel and specialization at an existing bracket launch site."""

    def __init__(self, module: str, assembly: str, target: str, *, export: bool = False):
        self.source = Path(module).resolve().parent / assembly
        self.manifest = self.source.with_suffix(self.source.suffix + ".json")
        self.target = target
        self.export = export
        self._binding: _BoundKernel | None = None

    def __call__(self, kernel: JITFunction) -> _BoundKernel:
        from triton.runtime.jit import JITFunction

        if not isinstance(kernel, JITFunction):
            raise AssemblyError(
                "ASM capture requires a direct Triton/Gluon JITFunction; freeze autotune/heuristics first"
            )
        if self._binding is None:
            self._binding = _BoundKernel(self, kernel)
        elif self._binding.kernel is not kernel:
            raise AssemblyError("one assembly launch site cannot select multiple JIT kernels")
        return self._binding

    def _bind(self, compiled: CompiledKernel) -> CompiledKernel:
        import triton

        original = _compiler_source(compiled, self.target)
        # Triton's compilation hash includes source, specialization, target and options,
        # but not tensor addresses or runtime grid dimensions.
        fingerprint = hashlib.sha256((triton.__version__ + ":" + compiled.hash).encode()).hexdigest()
        frontend = "gluon" if compiled.src.fn.is_gluon() else "triton"
        identity = dict(schema_version=1, frontend=frontend, gpu_target=self.target, source_ir_sha256=fingerprint)
        if self.export and not self.manifest.exists():
            self.source.write_text(original, encoding="utf-8")
            self.manifest.write_text(
                json.dumps(
                    dict(identity, compiler_assembly_sha256=hashlib.sha256(original.encode()).hexdigest()), indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise AssemblyError("assembly binding does not match this Triton specialization; start a separate campaign")
        if self.export:
            return compiled
        return with_assembly(
            compiled,
            self.source,
            gpu_target=self.target,
            toolchain_dir=Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin",
        )


class _BoundKernel:
    def __init__(self, owner: TritonAssembly, kernel: JITFunction):
        self.owner = owner
        self.kernel = kernel
        self.variants: dict[tuple[int, str], CompiledKernel] = {}

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            from triton.runtime import driver

            compiled = self.kernel.warmup(*args, grid=grid, **kwargs)
            key = (driver.active.get_current_device(), compiled.hash)
            if key not in self.variants:
                self.variants[key] = self.owner._bind(compiled)
            candidate = self.variants[key]
            arguments = self.kernel.signature.bind(
                *args, **{name: value for name, value in kwargs.items() if name in self.kernel.arg_names}
            )
            arguments.apply_defaults()
            dimensions = grid(arguments.arguments) if callable(grid) else grid
            if not 1 <= len(dimensions) <= 3:
                raise AssemblyError("Triton launch grid must have one to three dimensions")
            dimensions = tuple(dimensions) + (1,) * (3 - len(dimensions))
            candidate[dimensions](*arguments.arguments.values())
            return candidate

        return launch
