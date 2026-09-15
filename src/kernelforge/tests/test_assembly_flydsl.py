# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU contracts for replacing a FlyDSL device binary without changing its launcher."""

from __future__ import annotations

import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from kernelforge.assembly.compiler import AssemblyError
from kernelforge.assembly.flydsl import _replace_binary, with_assembly


def _binary(name="kernels", *, target='#rocdl.target<chip = "gfx950">', code=b"\x7fELF-reference"):
    device_object = SimpleNamespace(
        target=target,
        format="bin",
        object=code,
        properties={"optimization": 3},
        kernels={"gemm": {"arguments": ["pointer", "i32"]}},
    )
    return SimpleNamespace(operation=SimpleNamespace(name="gpu.binary"), sym_name=name, objects=[device_object])


def _module(*binaries):
    launcher = SimpleNamespace(operation=SimpleNamespace(name="llvm.func"), name="launch", body="original ABI")
    return SimpleNamespace(
        body=SimpleNamespace(operations=[launcher, *binaries]),
        operation=SimpleNamespace(verify=Mock(return_value=True)),
    )


@pytest.fixture(params=["0.2.0", "0.2.4"])
def flydsl_api(monkeypatch, request):
    modules = {}
    for name in (
        "flydsl",
        "flydsl._mlir",
        "flydsl._mlir.ir",
        "flydsl._mlir.dialects",
        "flydsl._mlir.dialects.gpu",
        "flydsl.compiler",
        "flydsl.compiler.jit_executor",
        "flydsl.compiler.jit_function",
    ):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(modules[parent], leaf, module)

    ir = modules["flydsl._mlir.ir"]
    gpu = modules["flydsl._mlir.dialects.gpu"]
    serialized_modules = {}
    ir.Context = nullcontext
    ir.Location = SimpleNamespace(unknown=nullcontext)
    ir.StringAttr = lambda value: SimpleNamespace(value=value)
    ir.ArrayAttr = SimpleNamespace(get=list)
    ir.Module = SimpleNamespace(parse=lambda text: copy.deepcopy(serialized_modules[text]))

    def object_attr(value):
        return value

    object_attr.get = lambda target, format, code, properties, kernels: SimpleNamespace(
        target=target, format=format, object=code, properties=properties, kernels=kernels
    )
    gpu.ObjectAttr = object_attr

    class CompiledArtifact:
        def __init__(self, module, entry="launch", source_ir="source MLIR"):
            self.module = module
            self.ir = f"serialized module {len(serialized_modules)}"
            serialized_modules[self.ir] = module
            self._entry = entry
            self.source_ir = source_ir
            self._link_libs = []
            self._post_load_processors = []
            self._uses_explicit_module = False

        def _get_func_exe(self):
            def invoke(*args):
                binaries = [op for op in self.module.body.operations if op.operation.name == "gpu.binary"]
                return tuple(obj.object for op in binaries for obj in op.objects), args

            return invoke

    class CallState:
        def __init__(self, spec, executor):
            self._spec = spec
            self.executor = executor

    class CompiledFunction:
        def __init__(self, state, artifact):
            self._call_state = state
            self._keepalive = artifact

        def __call__(self, *args):
            return self._call_state.executor(*args)

    if request.param == "0.2.4":
        modules["flydsl.compiler.jit_executor"].CallState = CallState
    modules["flydsl.compiler.jit_function"].CallState = CallState
    modules["flydsl.compiler.jit_executor"].CompiledArtifact = CompiledArtifact
    modules["flydsl.compiler.jit_function"].CompiledFunction = CompiledFunction
    modules["flydsl.compiler.jit_function"]._create_mlir_context = nullcontext

    def compiled(module):
        artifact = CompiledArtifact(module)
        return CompiledFunction(CallState(object(), artifact._get_func_exe()), artifact)

    return SimpleNamespace(compiled=compiled, ir=ir)


@pytest.fixture()
def source_assembler(tmp_path, monkeypatch):
    from kernelforge.assembly import flydsl

    source = tmp_path / "kernel.s"
    source.write_bytes(b"first assembly")
    outputs = []

    def assemble(source_path, output_path, **kwargs):
        assert kwargs["gpu_target"] == "gfx950"
        assert kwargs["toolchain_dir"] == Path("toolchain")
        assert kwargs["timeout_sec"] == 60
        output_path.write_bytes(b"\x7fELF-" + source_path.read_bytes())
        outputs.append(output_path)
        return output_path

    monkeypatch.setattr(flydsl, "assemble", assemble)
    return SimpleNamespace(source=source, outputs=outputs)


def test_variant_preserves_reference_launcher_metadata_and_call_arguments(flydsl_api, source_assembler):
    original = _module(_binary())
    reference = flydsl_api.compiled(original)
    arguments = (object(), 37, object())
    reference_result = reference(*arguments)

    variant = with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")

    assert variant is not reference
    assert variant._keepalive is not reference._keepalive
    assert variant._call_state._spec is reference._call_state._spec
    assert variant._keepalive._entry == reference._keepalive._entry
    assert variant._keepalive.source_ir == reference._keepalive.source_ir
    assert variant(*arguments) == ((b"\x7fELF-first assembly",), arguments)
    assert reference(*arguments) == reference_result
    before = original.body.operations
    after = variant._keepalive.module.body.operations
    assert after[0] == before[0]
    for attribute in ("target", "format", "properties", "kernels"):
        assert getattr(after[1].objects[0], attribute) == getattr(before[1].objects[0], attribute)
    original.operation.verify.assert_not_called()
    variant._keepalive.module.operation.verify.assert_called_once_with()
    assert all(not output.exists() for output in source_assembler.outputs)


def test_edited_source_creates_fresh_variant_without_replacing_earlier_variant(flydsl_api, source_assembler):
    reference = flydsl_api.compiled(_module(_binary()))
    first = with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
    source_assembler.source.write_bytes(b"second assembly")

    second = with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")

    assert reference()[0] == (b"\x7fELF-reference",)
    assert first()[0] == (b"\x7fELF-first assembly",)
    assert second()[0] == (b"\x7fELF-second assembly",)
    assert len(source_assembler.outputs) == 2
    assert source_assembler.outputs[0] != source_assembler.outputs[1]


@pytest.mark.parametrize("names", [[], ["first", "second"]])
def test_rejects_missing_or_ambiguous_device_binary(flydsl_api, names):
    module = _module(*(_binary(name) for name in names))

    with pytest.raises(AssemblyError, match="select exactly one gpu.binary"):
        _replace_binary(module, b"\x7fELF-new", gpu_target="gfx950", binary_name=None)
    module.operation.verify.assert_not_called()


def test_named_selection_replaces_only_the_selected_device_module(flydsl_api):
    module = _module(_binary("first"), _binary("second"))

    _replace_binary(module, b"\x7fELF-new", gpu_target="gfx950", binary_name="second")

    assert module.body.operations[1].objects[0].object == b"\x7fELF-reference"
    assert module.body.operations[2].objects[0].object == b"\x7fELF-new"


def test_unknown_binary_name_is_rejected(flydsl_api):
    module = _module(_binary("kernels"))

    with pytest.raises(AssemblyError, match="select exactly one gpu.binary"):
        _replace_binary(module, b"\x7fELF-new", gpu_target="gfx950", binary_name="missing")


@pytest.mark.parametrize(
    "target",
    [
        '#rocdl.target<chip = "gfx942">',
        '#nvvm.target<chip = "sm_90">',
        "#rocdl.target<>",
        '#rocdl.target<chip = "gfx950", triple = "amdgcn-amd-amdpal">',
    ],
)
def test_rejects_incompatible_compiled_target_without_mutating_reference(flydsl_api, source_assembler, target):
    original = _module(_binary(target=target))
    reference = flydsl_api.compiled(original)

    with pytest.raises(AssemblyError, match="does not match the FlyDSL target"):
        with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")

    assert reference()[0] == (b"\x7fELF-reference",)
    original.operation.verify.assert_not_called()


@pytest.mark.parametrize(
    "features,gpu_target",
    [
        ("+sramecc,-xnack,+wavefrontsize64", "gfx950:xnack-:sramecc+"),
        ("+wavefrontsize64", "gfx950"),
        ("", "gfx950"),
    ],
)
def test_matches_abi_features_and_preserves_other_llvm_features(flydsl_api, features, gpu_target):
    target = f'#rocdl.target<chip = "gfx950", triple = "amdgcn-amd-amdhsa", features = "{features}">'
    module = _module(_binary(target=target))

    _replace_binary(module, b"\x7fELF-new", gpu_target=gpu_target, binary_name=None)

    assert module.body.operations[1].objects[0].target == target
    assert module.body.operations[1].objects[0].object == b"\x7fELF-new"


@pytest.mark.parametrize(
    "features,gpu_target",
    [("+xnack", "gfx950"), ("+xnack", "gfx950:xnack-"), ("", "gfx950:xnack+"), ("-sramecc", "gfx950:sramecc+")],
)
def test_rejects_abi_feature_mismatch_without_replacing_binary(flydsl_api, features, gpu_target):
    module = _module(_binary(target=f'#rocdl.target<chip = "gfx950", features = "{features}">'))

    with pytest.raises(AssemblyError, match="does not match the FlyDSL target"):
        _replace_binary(module, b"\x7fELF-new", gpu_target=gpu_target, binary_name=None)
    assert module.body.operations[1].objects[0].object == b"\x7fELF-reference"


@pytest.mark.parametrize("features", ["+xnack,-xnack", "+sramecc,+sramecc"])
def test_rejects_duplicate_abi_features_with_shared_target_parser(flydsl_api, features):
    module = _module(_binary(target=f'#rocdl.target<chip = "gfx950", features = "{features}">'))

    with pytest.raises(AssemblyError, match="Duplicate target feature"):
        _replace_binary(module, b"\x7fELF-new", gpu_target="gfx950", binary_name=None)


def test_rejects_multiple_target_objects(flydsl_api):
    binary = _binary()
    binary.objects.append(copy.deepcopy(binary.objects[0]))

    with pytest.raises(AssemblyError, match="single-target"):
        _replace_binary(_module(binary), b"\x7fELF-new", gpu_target="gfx950", binary_name=None)


@pytest.mark.parametrize("code", [b"s_endpgm", b"__CLANG_OFFLOAD_BUNDLE__"])
def test_rejects_non_elf_reference_object(flydsl_api, code):
    with pytest.raises(AssemblyError, match="ELF code object"):
        _replace_binary(_module(_binary(code=code)), b"\x7fELF-new", gpu_target="gfx950", binary_name=None)


@pytest.mark.parametrize(
    "field,value",
    [("_link_libs", ["external.so"]), ("_post_load_processors", [object()]), ("_uses_explicit_module", True)],
)
def test_rejects_external_runtime_dependencies_before_assembly(flydsl_api, source_assembler, field, value):
    reference = flydsl_api.compiled(_module(_binary()))
    setattr(reference._keepalive, field, value)

    with pytest.raises(AssemblyError, match="self-contained"):
        with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
    assert source_assembler.outputs == []


def test_rejects_uncompiled_callable(flydsl_api, source_assembler):
    with pytest.raises(TypeError, match="flydsl.compiler.compile"):
        with_assembly(lambda: None, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
    assert source_assembler.outputs == []


def test_failed_assembly_does_not_change_reference(flydsl_api, source_assembler, monkeypatch):
    from kernelforge.assembly import flydsl

    reference = flydsl_api.compiled(_module(_binary()))
    monkeypatch.setattr(flydsl, "assemble", Mock(side_effect=AssemblyError("invalid instruction")))

    with pytest.raises(AssemblyError, match="invalid instruction"):
        with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
    assert reference()[0] == (b"\x7fELF-reference",)


def test_optional_flydsl_dependency_failure_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "flydsl._mlir", None)

    with pytest.raises(AssemblyError, match="FlyDSL with the CompiledFunction/CompiledArtifact API is required"):
        with_assembly(object(), tmp_path / "kernel.s", gpu_target="gfx950", toolchain_dir="toolchain")


@pytest.mark.parametrize(
    "owner,field",
    [
        ("compiled", "_keepalive"),
        ("artifact", "ir"),
        ("artifact", "_entry"),
        ("artifact", "_link_libs"),
        ("state", "_spec"),
    ],
)
def test_incompatible_artifact_api_fails_before_assembly(flydsl_api, source_assembler, owner, field):
    reference = flydsl_api.compiled(_module(_binary()))
    owners = {"compiled": reference, "artifact": reference._keepalive, "state": reference._call_state}
    delattr(owners[owner], field)

    with pytest.raises(AssemblyError, match="Unsupported FlyDSL CompiledFunction/CompiledArtifact API") as error:
        with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
    assert field in str(error.value)
    assert isinstance(error.value.__cause__, AttributeError)
    assert source_assembler.outputs == []


def test_internal_parser_attribute_error_is_not_misreported_as_api_mismatch(flydsl_api, source_assembler, monkeypatch):
    reference = flydsl_api.compiled(_module(_binary()))
    monkeypatch.setattr(flydsl_api.ir.Module, "parse", Mock(side_effect=AttributeError("internal parser failure")))

    with pytest.raises(AttributeError, match="internal parser failure"):
        with_assembly(reference, source_assembler.source, gpu_target="gfx950", toolchain_dir="toolchain")
