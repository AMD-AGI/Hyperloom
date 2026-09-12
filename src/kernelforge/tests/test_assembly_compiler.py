# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermetic assembly toolchain contract tests; no ROCm installation or GPU needed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kernelforge.assembly import AssemblyError, assemble
from kernelforge.assembly import __main__ as cli
from kernelforge.assembly import compiler

_SOURCE = """\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
\t.text
\t.globl kernel
\t.p2align 8
\t.type kernel,@function
kernel:
\ts_endpgm
\t.size kernel, .-kernel
\t.section .rodata,"a",@progbits
\t.p2align 6
\t.amdhsa_kernel kernel
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 0
\t\t.amdhsa_next_free_vgpr 0
\t\t.amdhsa_next_free_sgpr 0
\t.end_amdhsa_kernel
\t.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: kernel
    .symbol: kernel.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 0
    .vgpr_count: 0
    .max_flat_workgroup_size: 256
...
\t.end_amdgpu_metadata
"""


@pytest.fixture
def build_paths(tmp_path):
    source_dir = tmp_path / "source with spaces"
    source_dir.mkdir()
    source = source_dir / "candidate kernel.s"
    source.write_bytes(_SOURCE.replace("\n", "\r\n").encode())
    toolchain = tmp_path / "ROCm toolchain" / "bin"
    toolchain.mkdir(parents=True)
    (toolchain / "llvm-mc").touch()
    (toolchain / "ld.lld").touch()
    output = tmp_path / "build outputs" / "candidate kernel.hsaco"
    return source, output, toolchain


@pytest.fixture
def fake_tools(monkeypatch):
    calls = []

    def run(command, **kwargs):
        artifact = Path(command[command.index("-o") + 1])
        input_path = Path(command[command.index("-o") - 1])
        contents = input_path.read_bytes()
        calls.append((command, kwargs, contents))
        artifact.write_bytes(contents)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    return calls


def test_assembles_staged_bytes_and_atomically_publishes_only_the_requested_output(
    build_paths, fake_tools, monkeypatch
):
    source, output, toolchain = build_paths
    original = source.read_bytes()
    output.parent.mkdir()
    output.write_bytes(b"previous code object")
    real_replace = compiler.os.replace
    publications = []

    def replace(staged, destination):
        assert output.read_bytes() == b"previous code object"
        assert Path(staged).parent.parent == output.parent
        publications.append((staged, destination))
        return real_replace(staged, destination)

    monkeypatch.setattr(compiler.os, "replace", replace)
    assert assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain) == output.resolve()

    assert len(fake_tools) == 2
    assembler, linker = fake_tools
    assert assembler[0][0] == str(toolchain / "llvm-mc")
    assert "-triple=amdgcn-amd-amdhsa" in assembler[0]
    assert "-mcpu=gfx950" in assembler[0]
    assert "-filetype=obj" in assembler[0]
    assert "-shared" in linker[0] and "--no-undefined" in linker[0]
    assert linker[0][0] == str(toolchain / "ld.lld")
    assert all(call[2] == original for call in fake_tools)
    assert all(call[1]["cwd"] != source.parent for call in fake_tools)
    assert all(call[1]["check"] is False and 0 < call[1]["timeout"] <= 60 for call in fake_tools)
    assert 0 < linker[1]["timeout"] <= assembler[1]["timeout"]
    assert len(publications) == 1
    assert source.read_bytes() == original
    assert output.read_bytes() == original
    assert list(source.parent.iterdir()) == [source]
    assert list(output.parent.iterdir()) == [output]


def test_source_edits_always_trigger_fresh_assembly_even_when_mtime_is_unchanged(build_paths, fake_tools):
    source, output, toolchain = build_paths
    assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)
    timestamp = source.stat().st_mtime_ns
    changed = source.read_bytes().replace(b"\ts_endpgm", b"\ts_nop 0\r\n\ts_endpgm")
    source.write_bytes(changed)
    compiler.os.utime(source, ns=(timestamp, timestamp))

    assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert len(fake_tools) == 4
    assert fake_tools[0][2] != fake_tools[2][2]
    assert fake_tools[2][2] == changed == output.read_bytes()


def test_target_features_are_matched_independent_of_order_and_passed_to_llvm(build_paths, fake_tools):
    source, output, toolchain = build_paths
    source.write_text(_SOURCE.replace("gfx950", "gfx90a:xnack-:sramecc+"))

    assemble(source, output, gpu_target="gfx90a:sramecc+:xnack-", toolchain_dir=toolchain)

    assert "-mcpu=gfx90a" in fake_tools[0][0]
    assert "-mattr=+sramecc,-xnack" in fake_tools[0][0]


@pytest.mark.parametrize("target", ["gfx942", "gfx950:xnack-", "gfx950:sramecc+"])
def test_mismatched_target_is_rejected_before_invoking_tools(build_paths, fake_tools, target):
    source, output, toolchain = build_paths

    with pytest.raises(AssemblyError, match="does not match requested GPU target"):
        assemble(source, output, gpu_target=target, toolchain_dir=toolchain)

    assert fake_tools == []
    assert not output.exists()


@pytest.mark.parametrize(
    "target",
    [
        "",
        "native",
        "gfxbad",
        "gfx95",
        "gfx950;touch bad",
        "gfx950:xnack",
        "gfx950:wavefrontsize32+",
        "gfx950:xnack+:xnack-",
    ],
)
def test_invalid_or_unsupported_target_is_rejected(build_paths, fake_tools, target):
    source, output, toolchain = build_paths

    with pytest.raises(AssemblyError, match="GPU target|Duplicate target feature"):
        assemble(source, output, gpu_target=target, toolchain_dir=toolchain)

    assert fake_tools == []


@pytest.mark.parametrize("directive", ["amdhsa_kernel", "end_amdhsa_kernel", "amdgpu_metadata", "end_amdgpu_metadata"])
def test_incomplete_assembly_is_rejected_without_regenerating_metadata(build_paths, fake_tools, directive):
    source, output, toolchain = build_paths
    source.write_text(_SOURCE.replace("." + directive, "; omitted_" + directive))

    with pytest.raises(AssemblyError, match="Missing ." + directive):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert fake_tools == []


def test_instruction_only_disassembly_is_rejected(build_paths, fake_tools):
    source, output, toolchain = build_paths
    source.write_text("0000000000001000 <kernel>:\n\ts_endpgm // 000000001000: BF810000\n")

    with pytest.raises(AssemblyError, match="instruction-only disassembly cannot be reassembled"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert fake_tools == []


@pytest.mark.parametrize(
    "replacement",
    ["", '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n.amdgcn_target "amdgcn-amd-amdhsa--gfx950"'],
)
def test_source_must_identify_one_hsa_target(build_paths, fake_tools, replacement):
    source, output, toolchain = build_paths
    source.write_text(_SOURCE.replace('.amdgcn_target "amdgcn-amd-amdhsa--gfx950"', replacement))

    with pytest.raises(AssemblyError, match="exactly one .amdgcn_target"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert fake_tools == []


@pytest.mark.parametrize("tool", ["llvm-mc", "ld.lld"])
def test_missing_explicit_toolchain_never_falls_back_to_path(build_paths, fake_tools, tool):
    source, output, toolchain = build_paths
    (toolchain / tool).unlink()

    with pytest.raises(AssemblyError, match="Missing ROCm LLVM tool") as error:
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert tool in str(error.value)
    assert fake_tools == []


@pytest.mark.parametrize("fail_stage", ["llvm-mc", "ld.lld"])
def test_failed_build_raises_with_diagnostics_without_publishing_partial_or_stale_output(
    build_paths, monkeypatch, fail_stage
):
    source, output, toolchain = build_paths
    output.parent.mkdir()
    output.write_bytes(b"previous valid code object")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"partial output")
        failed = Path(command[0]).name == fail_stage
        return subprocess.CompletedProcess(
            command, 17 if failed else 0, stdout="assembler context", stderr="invalid operand"
        )

    monkeypatch.setattr(compiler.subprocess, "run", run)

    with pytest.raises(AssemblyError, match="failed with exit code 17") as error:
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert "assembler context" in str(error.value) and "invalid operand" in str(error.value)
    assert len(calls) == (1 if fail_stage == "llvm-mc" else 2)
    assert output.read_bytes() == b"previous valid code object"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(["llvm-mc"], 1), PermissionError("not executable")])
def test_execution_failures_are_actionable_and_never_publish_output(build_paths, monkeypatch, failure):
    source, output, toolchain = build_paths

    def run(command, **kwargs):
        raise failure

    monkeypatch.setattr(compiler.subprocess, "run", run)

    with pytest.raises(AssemblyError, match="timed out|Could not execute") as error:
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert "llvm-mc" in str(error.value)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(build_paths, fake_tools, timeout):
    source, output, toolchain = build_paths

    with pytest.raises(AssemblyError, match="positive and finite"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain, timeout_sec=timeout)

    assert fake_tools == []


def test_deadline_covers_both_assembly_and_linking(build_paths, fake_tools, monkeypatch):
    source, output, toolchain = build_paths
    ticks = iter([10, 10, 12])
    monkeypatch.setattr(compiler.time, "monotonic", lambda: next(ticks))

    with pytest.raises(AssemblyError, match="timeout expired before AMDGPU linking"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain, timeout_sec=1)

    assert len(fake_tools) == 1
    assert not output.exists()


@pytest.mark.parametrize("missing_stage", ["llvm-mc", "ld.lld"])
def test_tool_success_without_an_artifact_is_a_build_failure(build_paths, monkeypatch, missing_stage):
    source, output, toolchain = build_paths

    def run(command, **kwargs):
        if Path(command[0]).name != missing_stage:
            Path(command[command.index("-o") + 1]).write_bytes(b"object file")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)

    with pytest.raises(AssemblyError, match="did not produce a nonempty artifact"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert not output.exists()


def test_output_cannot_overwrite_the_source(build_paths, fake_tools):
    source, _, toolchain = build_paths

    with pytest.raises(AssemblyError, match="output must differ"):
        assemble(source, source, gpu_target="gfx950", toolchain_dir=toolchain)

    assert fake_tools == []


def test_binary_input_is_rejected(build_paths, fake_tools):
    source, output, toolchain = build_paths
    source.write_bytes(b"\x7fELF\xff")

    with pytest.raises(AssemblyError, match="not UTF-8 text"):
        assemble(source, output, gpu_target="gfx950", toolchain_dir=toolchain)

    assert fake_tools == []


def test_cli_reports_success_only_after_a_completed_build(build_paths, fake_tools, capsys):
    source, output, toolchain = build_paths

    status = cli.main(
        [
            "assemble",
            "--source",
            str(source),
            "--output",
            str(output),
            "--gpu-target",
            "gfx950",
            "--toolchain-dir",
            str(toolchain),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out.strip() == str(output.resolve())
    assert captured.err == ""
    assert output.exists()


def test_cli_returns_nonzero_and_no_result_for_failed_candidate(build_paths, fake_tools, capsys):
    source, output, toolchain = build_paths

    status = cli.main(
        [
            "assemble",
            "--source",
            str(source),
            "--output",
            str(output),
            "--gpu-target",
            "gfx942",
            "--toolchain-dir",
            str(toolchain),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "does not match requested GPU target" in captured.err
    assert fake_tools == []


def test_module_entry_point_is_available_without_rocm():
    result = subprocess.run(
        [sys.executable, "-m", "kernelforge.assembly", "assemble", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--gpu-target" in result.stdout and "--toolchain-dir" in result.stdout
