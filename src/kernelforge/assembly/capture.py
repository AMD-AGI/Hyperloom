# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind one explicit FlyDSL compile call to its compiler-emitted assembly."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from kernelforge.assembly.compiler import AssemblyError, _validate_source
from kernelforge.assembly.flydsl import with_assembly


@contextmanager
def _capture_environment(directory: Path):
    values = {
        "FLYDSL_DUMP_IR": "1",
        "FLYDSL_DUMP_DIR": str(directory / "dump"),
        "FLYDSL_RUNTIME_CACHE_DIR": str(directory / "cache"),
    }
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bind_compile(source: str, assembly: str, target: str, *, export: bool) -> str:
    """Mechanically wrap one direct compile call, preserving its arguments and launcher."""
    tree = ast.parse(source)
    aliases = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(
                (alias.asname or alias.name) + ".compile" for alias in node.names if alias.name == "flydsl.compiler"
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "flydsl.compiler":
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "compile")
            elif node.module == "flydsl":
                aliases.update(
                    (alias.asname or alias.name) + ".compile" for alias in node.names if alias.name == "compiler"
                )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) in aliases]
    if len(calls) != 1:
        raise AssemblyError(
            "automatic assembly capture requires one direct flydsl.compiler.compile(...) call; "
            "extract a single compiled specialization first. Triton/HIP/library extraction is not implemented"
        )
    if "_forge_assembly" in source:
        raise AssemblyError("reserved assembly binding name already exists; use a fresh source workspace")
    function = calls[0].func
    lines = source.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[: function.lineno - 1])) + function.col_offset
    end = sum(map(len, lines[: function.end_lineno - 1])) + function.end_col_offset
    data = source.encode("utf-8")
    data = data[:start] + b"_forge_assembly" + data[end:]
    # Preserve the module docstring and future imports, including their line positions.
    insertion = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ) or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            insertion = node.end_lineno
        else:
            break
    lines = data.decode("utf-8").splitlines(keepends=True)
    binding = (
        "\nfrom kernelforge.assembly.capture import FlyDSLAssembly as _forge_assembly_type\n"
        f"_forge_assembly = _forge_assembly_type(__file__, {assembly!r}, {target!r}, export={export!r})\n\n"
    )
    return "".join(lines[:insertion]) + binding + "".join(lines[insertion:])


def _fingerprint(compiled) -> str:
    from flydsl._mlir import ir
    from flydsl.compiler.jit_function import _create_mlir_context

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(compiled._keepalive.source_ir)
        normalized = module.operation.get_asm(enable_debug_info=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


class FlyDSLAssembly:
    """Export or rebuild one specialization without changing its host call specification.

    Constructed by the preparation host. Export is temporary; the committed binding
    always rebuilds the selected .s and never falls back to frontend execution.
    Call at the original compilation boundary, outside timing and graph capture.
    """

    def __init__(self, module: str, assembly: str, target: str, *, export: bool = False):
        self.source = Path(module).resolve().parent / assembly
        self.manifest = self.source.with_suffix(self.source.suffix + ".json")
        self.target = target
        self.export = export

    def __call__(self, *args, **kwargs):
        import flydsl.compiler as flyc

        if self.export:
            with tempfile.TemporaryDirectory(prefix="forge-flydsl-capture-") as scratch:
                directory = Path(scratch)
                with _capture_environment(directory):
                    compiled = flyc.compile(*args, **kwargs)
                fingerprint = _fingerprint(compiled)
                sources = list((directory / "dump").rglob("*_final_isa.s"))
                if self.manifest.exists():
                    self._check(fingerprint)
                    return compiled
                if len(sources) != 1:
                    raise AssemblyError("capture requires exactly one fresh compiler ISA dump; no cached/guessed seed")
                source = sources[0].read_text(encoding="utf-8")
                _validate_source(source, self.target)
                self.source.write_text(source, encoding="utf-8")
                self.manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "frontend": "flydsl",
                            "gpu_target": self.target,
                            "source_ir_sha256": fingerprint,
                            "compiler_assembly_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return compiled

        compiled = flyc.compile(*args, **kwargs)
        self._check(_fingerprint(compiled))
        return with_assembly(
            compiled,
            self.source,
            gpu_target=self.target,
            toolchain_dir=Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin",
        )

    def _check(self, fingerprint: str) -> None:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("gpu_target") != self.target
            or manifest.get("source_ir_sha256") != fingerprint
        ):
            raise AssemblyError(
                "assembly binding does not match this compiled specialization; start a separate campaign"
            )
