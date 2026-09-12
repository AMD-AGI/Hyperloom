# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional ROCm test proving an assembly edit executes through the FlyDSL ABI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
flyc = pytest.importorskip("flydsl.compiler")
fx = pytest.importorskip("flydsl.expr")

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires an AMD GPU and ROCm PyTorch", allow_module_level=True)

from kernelforge.assembly.flydsl import with_assembly


@flyc.kernel
def _vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr):
    index = fx.block_idx.x * 256 + fx.thread_idx.x
    if index < N:
        a = fx.logical_divide(A, fx.make_layout(1, 1))
        b = fx.logical_divide(B, fx.make_layout(1, 1))
        c = fx.logical_divide(C, fx.make_layout(1, 1))
        copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ra = fx.make_rmem_tensor(1, fx.Float32)
        rb = fx.make_rmem_tensor(1, fx.Float32)
        rc = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy_atom_call(copy, fx.slice(a, (None, index)), ra)
        fx.copy_atom_call(copy, fx.slice(b, (None, index)), rb)
        rc.store(ra.load() + rb.load())
        fx.copy_atom_call(copy, rc, fx.slice(c, (None, index)))


@flyc.jit
def _vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr, stream: fx.Stream):
    _vector_add_kernel(A, B, C, N).launch(grid=((N + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


def test_roundtrip_executes_edits_without_mutating_reference(tmp_path, monkeypatch):
    toolchain = Path("/opt/rocm/llvm/bin")
    if not all((toolchain / name).is_file() for name in ("llvm-mc", "ld.lld")):
        pytest.skip("requires ROCm LLVM tools under /opt/rocm/llvm/bin")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FLYDSL_COMPILE_LLVM_DIR", raising=False)
    n = 4103  # The final block exercises bounds and the original kernarg layout.
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    out = torch.full_like(a, float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    reference = flyc.compile(_vector_add, a, b, out, n, stream)
    stream.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)

    sources = list((tmp_path / "dumps").rglob("*_final_isa.s"))
    assert len(sources) == 1
    source = sources[0]
    assembly = source.read_text()
    target = re.search(r'\.amdgcn_target\s+"amdgcn-amd-amdhsa--([^\"]+)"', assembly)
    assert target is not None
    options = {"gpu_target": target[1], "toolchain_dir": toolchain}
    candidate = with_assembly(reference, source, **options)

    def check(function, left=a, right=b):
        result = torch.full_like(left, float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        function(left, right, result, n, stream)
        stream.synchronize()
        torch.testing.assert_close(result, left + right, rtol=0, atol=0)

    check(candidate)
    edited, count = re.subn(r"\bv_add_f32(?P<encoding>_e32|_e64)?\b", r"v_sub_f32\g<encoding>", assembly)
    assert count == 1
    source.write_text(edited)
    negative = with_assembly(reference, source, **options)
    with pytest.raises(AssertionError):
        check(negative)
    check(reference)
    check(candidate)
    check(candidate, torch.randn_like(a), torch.randn_like(b))

    source.write_text(assembly)
    restored = with_assembly(reference, source, **options)
    check(restored)
    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        restored(a, b, out, n, stream)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        restored(a, b, out, n, stream)
    out.fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)
