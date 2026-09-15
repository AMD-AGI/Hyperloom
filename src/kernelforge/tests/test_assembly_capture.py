# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler capture provenance, specialization checks and mechanical source binding."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from kernelforge.assembly import capture
from kernelforge.assembly.compiler import AssemblyError
from kernelforge.tests.test_assembly_preparation import ASM


@pytest.mark.parametrize(
    "import_line,call",
    [
        ("import flydsl.compiler as flyc", "flyc.compile"),
        ("from flydsl import compiler as fc", "fc.compile"),
        ("from flydsl.compiler import compile as build", "build"),
    ],
)
def test_binding_preserves_call_arguments_and_frontend(import_line, call):
    original = f'"""module"""\nfrom __future__ import annotations\n{import_line}\ndef launch(x):\n    return {call}(kernel, x, stream=stream)\n'
    bound = capture.bind_compile(original, "kernel.s", "gfx950", export=False)
    tree = ast.parse(bound)
    compile(tree, "kernel.py", "exec")
    function = tree.body[-1]
    assert ast.unparse(function) == "def launch(x):\n    return _forge_assembly(kernel, x, stream=stream)"
    assert import_line in bound
    assert '"""module"""' in bound
    assert "export=False" in bound


@pytest.mark.parametrize(
    "source",
    [
        "import triton\nf = kernel[(1,)](x)\n",
        "import flydsl.compiler as f\na=f.compile(k,x)\nb=f.compile(k,y)\n",
        "import flydsl.compiler as f\na=f.compile[{}](k,x)\n",
    ],
)
def test_unsupported_capture_is_explicit(source):
    with pytest.raises(AssemblyError, match="one direct"):
        capture.bind_compile(source, "kernel.s", "gfx950", export=True)


@pytest.fixture
def frontend(monkeypatch):
    module = ModuleType("flydsl")
    compiler = ModuleType("flydsl.compiler")
    module.compiler = compiler
    monkeypatch.setitem(sys.modules, "flydsl", module)
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler)
    monkeypatch.setattr(capture, "_fingerprint", lambda compiled: compiled)

    def compile_kernel(identity):
        if os.environ.get("FLYDSL_DUMP_DIR"):
            dump = Path(os.environ["FLYDSL_DUMP_DIR"]) / "kernel"
            dump.mkdir(parents=True, exist_ok=True)
            (dump / "15_final_isa.s").write_text(ASM)
        return identity

    compiler.compile = compile_kernel
    return compiler


def test_capture_rebuild_and_failure_never_substitute_original(tmp_path, monkeypatch, frontend):
    monkeypatch.setenv("FLYDSL_DUMP_DIR", "previous-directory")
    source = tmp_path / "kernel.s"
    exporter = capture.FlyDSLAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    assert exporter("specialization-1") == "specialization-1"
    assert source.read_text() == ASM
    manifest = json.loads(source.with_suffix(".s.json").read_text())
    assert manifest["source_ir_sha256"] == "specialization-1"
    assert os.environ["FLYDSL_DUMP_DIR"] == "previous-directory"
    monkeypatch.delenv("FLYDSL_DUMP_DIR")

    calls = []

    def rebuild(compiled, path, **kwargs):
        calls.append(path.read_text())
        if "broken" in path.read_text():
            raise AssemblyError("assembler rejected candidate")
        return "assembly-variant"

    monkeypatch.setattr(capture, "with_assembly", rebuild)
    loader = capture.FlyDSLAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950")
    assert loader("specialization-1") == "assembly-variant"
    with pytest.raises(AssemblyError, match="specialization"):
        loader("specialization-2")
    assert len(calls) == 1
    source.write_text(ASM + "// broken\n")
    with pytest.raises(AssemblyError, match="assembler rejected"):
        loader("specialization-1")
    source.unlink()
    with pytest.raises(FileNotFoundError):
        loader("specialization-1")


def test_capture_rejects_multiple_specializations(tmp_path, frontend):
    exporter = capture.FlyDSLAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    exporter("first")
    with pytest.raises(AssemblyError, match="specialization"):
        exporter("second")
    assert json.loads((tmp_path / "kernel.s.json").read_text())["source_ir_sha256"] == "first"


def test_capture_refuses_missing_dump_and_restores_environment(tmp_path, monkeypatch, frontend):
    monkeypatch.setattr(frontend, "compile", lambda identity: identity)
    monkeypatch.delenv("FLYDSL_DUMP_DIR", raising=False)
    exporter = capture.FlyDSLAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    with pytest.raises(AssemblyError, match="fresh compiler ISA dump"):
        exporter("first")
    assert "FLYDSL_DUMP_DIR" not in os.environ
    assert not (tmp_path / "kernel.s").exists()
