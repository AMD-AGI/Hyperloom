# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build an assembly variant with the original FlyDSL host launcher and ABI."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from kernelforge.assembly.compiler import AssemblyError, _target_parts, assemble

if TYPE_CHECKING:
    from flydsl.compiler.jit_function import CompiledFunction


def _replace_binary(module, code_object: bytes, *, gpu_target: str, binary_name: str | None) -> None:
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu

    binaries = [op for op in module.body.operations if op.operation.name == "gpu.binary"]
    if binary_name is not None:
        binaries = [op for op in binaries if ir.StringAttr(op.sym_name).value == binary_name]
    if len(binaries) != 1:
        raise AssemblyError("select exactly one gpu.binary with binary_name; no implicit multi-module replacement")
    binary = binaries[0]
    if len(binary.objects) != 1:
        raise AssemblyError("assembly replacement requires a single-target gpu.binary")
    original = gpu.ObjectAttr(binary.objects[0])
    target = str(original.target)
    chip = re.search(r'\bchip\s*=\s*"(gfx[0-9a-f]+)"', target)
    triple = re.search(r'\btriple\s*=\s*"([^"]*)"', target)
    if (
        not target.startswith("#rocdl.target<")
        or chip is None
        or (triple is not None and triple[1] != "amdgcn-amd-amdhsa")
    ):
        raise AssemblyError(f"assembly target {gpu_target!r} does not match the FlyDSL target {target}")
    features = re.search(r'\bfeatures\s*=\s*"([^"]*)"', target)
    feature_parts = [part.strip() for part in features[1].split(",")] if features is not None else []
    abi_features = [
        part[1:] + part[0] for part in feature_parts if part in {"+xnack", "-xnack", "+sramecc", "-sramecc"}
    ]
    if _target_parts(":".join([chip[1], *abi_features])) != _target_parts(gpu_target):
        raise AssemblyError(f"assembly target {gpu_target!r} does not match the FlyDSL target {target}")
    if not bytes(original.object).startswith(b"\x7fELF"):
        raise AssemblyError("the original FlyDSL object must be an ELF code object, not ISA or an offload bundle")
    replacement = gpu.ObjectAttr.get(
        original.target,
        original.format,
        code_object,
        original.properties,
        original.kernels,
    )
    binary.objects = ir.ArrayAttr.get([replacement])
    module.operation.verify()


def with_assembly(
    compiled: CompiledFunction,
    source: Path | str,
    *,
    gpu_target: str,
    toolchain_dir: Path | str,
    binary_name: str | None = None,
    timeout_sec: float = 60,
) -> CompiledFunction:
    """Return an independently owned callable using an edited AMDHSA assembly file.

    ``compiled`` is the result of ``flydsl.compiler.compile(launcher, *args)``.
    The returned callable accepts the same positional arguments, including the
    stream. Create it once before graph capture or timing, and run the complete
    correctness driver before treating it as a candidate.

    This adapter uses FlyDSL's CompiledFunction/CompiledArtifact interface as
    shipped in 0.2.4. It never patches the compiler or changes a JIT cache entry.
    Every call assembles the current source and creates a separate execution
    engine, so both the reference and earlier variants remain usable.
    """
    try:
        from flydsl._mlir import ir
        from flydsl.compiler.jit_executor import CallState, CompiledArtifact
        from flydsl.compiler.jit_function import CompiledFunction, _create_mlir_context
    except ImportError as exc:
        raise AssemblyError(
            "FlyDSL with the CompiledFunction/CompiledArtifact API is required (tested with 0.2.4)"
        ) from exc

    if not isinstance(compiled, CompiledFunction):
        raise TypeError("compiled must be returned by flydsl.compiler.compile(launcher, *args)")
    try:
        reference = compiled._keepalive
        reference_ir = reference.ir
        source_ir = reference.source_ir
        entry = reference._entry
        call_spec = compiled._call_state._spec
        has_dependencies = reference._link_libs or reference._post_load_processors or reference._uses_explicit_module
    except AttributeError as exc:
        raise AssemblyError(
            f"Unsupported FlyDSL CompiledFunction/CompiledArtifact API (tested with 0.2.4): {exc}"
        ) from exc
    if has_dependencies:
        raise AssemblyError(
            "assembly replacement currently requires a self-contained FlyDSL kernel without extern links"
        )

    with tempfile.TemporaryDirectory(prefix="forge-assembly-") as scratch:
        code_object = assemble(
            Path(source),
            Path(scratch) / "kernel.hsaco",
            gpu_target=gpu_target,
            toolchain_dir=Path(toolchain_dir),
            timeout_sec=timeout_sec,
        ).read_bytes()

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(reference_ir)
        _replace_binary(module, code_object, gpu_target=gpu_target, binary_name=binary_name)
        artifact = CompiledArtifact(module, entry, source_ir)
    state = CallState(call_spec, artifact._get_func_exe())
    return CompiledFunction(state, artifact)
