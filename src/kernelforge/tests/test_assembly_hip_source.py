# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HIP compiler capture and source/build identity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.assembly import hip_source
from kernelforge.assembly.capture import bind_compile
from kernelforge.assembly.compiler import AssemblyError
from kernelforge.tests.test_assembly_preparation import ASM


def test_hip_binding_keeps_original_build_arguments():
    original = 'from kernelforge.assembly.hip_source import compile_hip as build\ndef compile(source, out):\n    return build(source, out, gpu_target="gfx950", flags=("-O3", "-DSIZE=256"))\n'
    bound = bind_compile(original, "kernel.s", "gfx950", export=True)
    assert "HIPAssembly" in bound
    assert 'return _forge_assembly(source, out, gpu_target="gfx950", flags=("-O3", "-DSIZE=256"))' in bound


def test_hip_compilation_uses_explicit_flags_and_single_fresh_isa(tmp_path, monkeypatch):
    calls = []

    def run(command, *, cwd, **kwargs):
        calls.append(command)
        directory = Path(command[-1]).parent
        (directory / "kernel.hsaco").write_bytes(b"native-hip")
        (directory / "kernel-device.s").write_text(ASM)
        assert cwd == Path.cwd()

    monkeypatch.setattr(hip_source, "_run_tool", run)
    source = tmp_path / "kernel.hip"
    source.write_text("extern int example;")
    binary, isa = hip_source._compile(source, "gfx950", ("-O3", "-DN=256"), tmp_path, 10)
    assert binary == b"native-hip" and isa == ASM
    assert "-DN=256" in calls[0] and "--offload-arch=gfx950" in calls[0]
    assert "--genco" in calls[0] and "-save-temps=obj" in calls[0]
    hip_source._compile(source, "gfx950", ("-O3", "-DN=256"), tmp_path, 10)
    assert next(x for x in calls[0] if x.startswith("-cuid=")) == next(x for x in calls[1] if x.startswith("-cuid="))


def test_hip_candidate_rebuild_and_changed_source_rejection(tmp_path, monkeypatch):
    isa = [ASM]
    monkeypatch.setattr(hip_source, "_compile", lambda *args: (b"native-hip", isa[0]))
    output = tmp_path / "build/kernel.hsaco"
    arguments = (tmp_path / "kernel.hip", output)
    assert hip_source.compile_hip(*arguments, gpu_target="gfx950").read_bytes() == b"native-hip"
    export = hip_source.HIPAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    export(*arguments, gpu_target="gfx950")
    assert json.loads((tmp_path / "kernel.s.json").read_text())["frontend"] == "hip"

    def assemble(source, destination, **kwargs):
        if ".error" in source.read_text():
            raise AssemblyError("deliberate build failure")
        destination.write_bytes(b"candidate-hip")
        return destination

    monkeypatch.setattr(hip_source, "assemble", assemble)
    bind = hip_source.HIPAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950")
    assert bind(*arguments, gpu_target="gfx950").read_bytes() == b"candidate-hip"
    (tmp_path / "kernel.s").write_text(ASM + '\n.error "probe"\n')
    with pytest.raises(AssemblyError, match="deliberate"):
        bind(*arguments, gpu_target="gfx950")
    isa[0] = ASM.replace("s_endpgm", "s_nop 0\n    s_endpgm")
    with pytest.raises(AssemblyError, match="source/build contract"):
        bind(*arguments, gpu_target="gfx950")
    with pytest.raises(AssemblyError, match="build target"):
        bind(*arguments, gpu_target="gfx942")
