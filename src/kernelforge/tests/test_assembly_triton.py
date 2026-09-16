# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Triton candidate ownership, argument binding and specialization boundaries."""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from kernelforge.assembly import triton as adapter
from kernelforge.assembly.compiler import AssemblyError

ASM = """.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
add:
  s_endpgm
.amdhsa_kernel add
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: add
    .symbol: add.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .wavefront_size: 64
    .vgpr_count: 8
    .args:
      - { .name: X, .offset: 0, .size: 8, .value_kind: global_buffer }
...
.end_amdgpu_metadata
"""


@pytest.fixture
def runtime(monkeypatch):
    @dataclasses.dataclass
    class Target:
        backend: str = "hip"
        arch: str = "gfx950"
        warp_size: int = 64

    launches = []
    state = SimpleNamespace(device=0, stream=17, warmups=[])

    class CompiledKernel:
        def __init__(self, src, group, hash):
            self.src, self.hash = src, hash
            self.asm = {Path(p).suffix[1:]: Path(p).read_bytes() for p in group.values() if not p.endswith(".json")}
            self.asm["amdgcn"] = self.asm["amdgcn"].decode()
            self.kernel = self.asm["hsaco"]
            data = json.loads(Path(group["kernel.json"]).read_text())
            data["target"] = Target(**data["target"])
            self.metadata = namedtuple("Metadata", data)(**data)
            self.module = self.function = None
            self.metadata_group = group

        def __getitem__(self, grid):
            def launch(*args):
                launches.append((self, grid, args, state.stream))

            return launch

    class JITFunction:
        def __init__(self, gluon=False):
            self.signature = inspect.signature(lambda X, N, BLOCK=256: None)
            self.arg_names = list(self.signature.parameters)
            self.gluon = gluon
            self.compiled = CompiledKernel.__new__(CompiledKernel)
            self.compiled.src = SimpleNamespace(fn=self)
            self.compiled.hash = "source-specialization"
            self.compiled.asm = {"amdgcn": ASM, "hsaco": b"original"}
            self.compiled.metadata = namedtuple("Metadata", "target shared name")(Target(), 0, "add")
            self.compiled.kernel = b"original"
            self.compiled.module = "original-module"
            self.compiled.function = "original-function"

        def is_gluon(self):
            return self.gluon

        def warmup(self, *args, grid, **kwargs):
            state.warmups.append((args, grid, kwargs))
            return self.compiled

    modules = {name: ModuleType(name) for name in ["triton", "triton.compiler", "triton.runtime", "triton.runtime.jit"]}
    modules["triton"].__version__ = "3.4.0"
    modules["triton.compiler"].CompiledKernel = CompiledKernel
    modules["triton.runtime.jit"].JITFunction = JITFunction
    modules["triton.runtime"].driver = SimpleNamespace(active=SimpleNamespace(get_current_device=lambda: state.device))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    builds = []

    def assemble(source, output, **kwargs):
        builds.append(source.read_bytes())
        if ".error" in source.read_text():
            raise AssemblyError("assembler rejected candidate")
        output.write_bytes(b"candidate-" + source.read_bytes())
        return output

    monkeypatch.setattr(adapter, "assemble", assemble)
    return SimpleNamespace(JITFunction=JITFunction, launches=launches, state=state, builds=builds)


@pytest.mark.parametrize("gluon", [False, True])
def test_capture_launch_and_independent_binary(tmp_path, runtime, gluon):
    kernel = runtime.JITFunction(gluon=gluon)
    exporter = adapter.TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    tensor = object()

    def grid(args):
        return ((args["N"] + args["BLOCK"] - 1) // args["BLOCK"],)

    exporter(kernel)[grid](tensor, N=513, num_warps=4)
    assert runtime.launches == [(kernel.compiled, (3, 1, 1), (tensor, 513, 256), 17)]
    assert json.loads((tmp_path / "kernel.s.json").read_text())["frontend"] == ("gluon" if gluon else "triton")

    loader = adapter.TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950")
    runtime.state.stream = 23
    candidate = loader(kernel)[grid](tensor, N=513, num_warps=4)
    assert runtime.launches[-1] == (candidate, (3, 1, 1), (tensor, 513, 256), 23)
    assert candidate is not kernel.compiled
    assert candidate.src is kernel.compiled.src
    assert candidate.module is None and candidate.function is None
    assert candidate.kernel.startswith(b"candidate-")
    assert kernel.compiled.kernel == b"original"
    assert kernel.compiled.module == "original-module"
    assert all(Path(p).exists() for p in candidate.metadata_group.values())
    loader(kernel)[(1,)](tensor, N=7)
    assert len(runtime.builds) == 1
    assert len(runtime.launches) == 3, "warmup must not execute source before the candidate (atomic outputs)"


def test_reject_changed_specialization_without_executing_source(tmp_path, runtime):
    kernel = runtime.JITFunction()
    exporter = adapter.TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950", export=True)
    exporter(kernel)[(1,)](object(), 256)
    kernel.compiled.hash = "different-constexpr-or-options"
    loader = adapter.TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950")
    with pytest.raises(AssemblyError, match="specialization"):
        loader(kernel)[(1,)](object(), 512)
    assert len(runtime.launches) == 1
    assert not runtime.builds


@pytest.mark.parametrize(
    "before,after",
    [
        (".kernarg_segment_size: 16", ".kernarg_segment_size: 24"),
        (".offset: 0", ".offset: 8"),
        (".wavefront_size: 64", ".wavefront_size: 32"),
        (".amdhsa_group_segment_fixed_size 0", ".amdhsa_group_segment_fixed_size 256"),
    ],
)
def test_changed_launch_contract_is_rejected(tmp_path, runtime, before, after):
    source = tmp_path / "kernel.s"
    source.write_text(ASM.replace(before, after))
    with pytest.raises(AssemblyError, match="launch ABI"):
        adapter.with_assembly(runtime.JITFunction().compiled, source, gpu_target="gfx950", toolchain_dir="unused")
    assert not runtime.builds


def test_register_edits_allowed_and_assembler_errors_propagate(tmp_path, runtime):
    source = tmp_path / "kernel.s"
    source.write_text(ASM.replace(".amdhsa_next_free_vgpr 8", ".amdhsa_next_free_vgpr 12"))
    compiled = runtime.JITFunction().compiled
    adapter.with_assembly(compiled, source, gpu_target="gfx950", toolchain_dir="unused")
    source.write_text(ASM + '\n.error "FORGE_ASSEMBLY_BUILD_PROBE"\n')
    with pytest.raises(AssemblyError, match="assembler rejected"):
        adapter.with_assembly(compiled, source, gpu_target="gfx950", toolchain_dir="unused")
    assert compiled.kernel == b"original"


def test_reject_non_jit_and_nvidia_targets(tmp_path, runtime):
    loader = adapter.TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", "gfx950")
    with pytest.raises(AssemblyError, match="direct Triton/Gluon"):
        loader(object())
    kernel = runtime.JITFunction()
    kernel.compiled.metadata.target.backend = "cuda"
    with pytest.raises(AssemblyError, match="AMD HIP"):
        adapter._compiler_source(kernel.compiled, "gfx950")
