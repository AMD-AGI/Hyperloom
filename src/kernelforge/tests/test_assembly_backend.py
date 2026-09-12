# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly backend selection and cross-language knowledge routing."""

from __future__ import annotations

import pytest

from kernelforge.config import Config
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.kernel_backends.constants import KERNEL_BACKENDS, resolve_language_dirs
from kernelforge.loop.campaign_config import infer_kernel_backend, resolve_kernel_backend_override


@pytest.fixture(autouse=True)
def no_backend_override(monkeypatch):
    monkeypatch.delenv("FORGE_KERNEL_BACKEND", raising=False)


def test_assembly_override_selects_target_expertise_for_flydsl_source(tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text("import flydsl.compiler as flyc\n", encoding="utf-8")
    assert infer_kernel_backend([source]) == "flydsl"

    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "assembly")
    assert infer_kernel_backend([source]) == "assembly"
    assert resolve_kernel_backend_override("assembly") == "assembly"
    assert "assembly" in KERNEL_BACKENDS


@pytest.mark.parametrize("suffix", [".s", ".S", ".asm"])
@pytest.mark.parametrize(
    "directive",
    [
        '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"',
        "  .amdhsa_kernel gemm",
        ".amdgpu_hsa_kernel gemm",
    ],
)
def test_infers_amdgpu_assembly_from_target_directive(tmp_path, suffix, directive):
    source = tmp_path / f"kernel{suffix}"
    source.write_text(f".text\n{directive}\ngemm:\n    s_endpgm\n", encoding="utf-8")

    assert infer_kernel_backend([source]) == "assembly"


def test_amdgpu_source_directive_outranks_framework_path_and_generator_comment(tmp_path):
    source = tmp_path / "aiter" / "kernels" / "gemm.s"
    source.parent.mkdir(parents=True)
    source.write_text("// Generated from FlyDSL\n.amdhsa_kernel gemm\n", encoding="utf-8")

    assert infer_kernel_backend([source]) == "assembly"


@pytest.mark.parametrize(
    "source_text",
    [
        ".text\n.globl add\nadd:\n    addl %edi, %eax\n    ret\n",
        ".version 8.0\n.target sm_90\n.entry add() { ret; }\n",
        "# .amdhsa_kernel is an AMD directive\n.text\n    ret\n",
        ".text\nadd:\n    s_endpgm\n",
    ],
)
def test_assembly_extension_without_amdgpu_directive_does_not_select_backend(tmp_path, source_text):
    source = tmp_path / "kernel.s"
    source.write_text(source_text, encoding="utf-8")

    with pytest.raises(ValueError, match="could not infer"):
        infer_kernel_backend([source])


def test_amdgpu_directive_mention_does_not_reclassify_python_kernel(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        'import flydsl.compiler as flyc\nASM_EXAMPLE = """\n.amdhsa_kernel gemm\n"""\n',
        encoding="utf-8",
    )

    assert infer_kernel_backend([source]) == "flydsl"


@pytest.mark.parametrize("defer_maps", [False, True])
def test_assembly_prompt_loads_frontend_and_assembly_knowledge(tmp_path, defer_maps):
    languages = ("assembly", "flydsl", "triton", "gluon", "hip")
    for language in languages:
        directory = tmp_path / "languages" / language
        directory.mkdir(parents=True)
        (directory / "INDEX.md").write_text(f"{language} test knowledge map\n", encoding="utf-8")
    config = Config(gpu_target="gfx950", local_knowledge_dir=tmp_path, defer_knowledge_maps=defer_maps)

    assert resolve_language_dirs("assembly", tmp_path) == languages
    prompt = build_single_kernel_backend_prompt(config, "assembly")
    for language in languages:
        if defer_maps or language != "assembly":
            assert str(tmp_path / "languages" / language / "INDEX.md") in prompt
            assert f"{language} test knowledge map" not in prompt
        else:
            assert f"{language} test knowledge map" in prompt
    assert "gfx950" in prompt


def test_missing_optional_frontend_knowledge_is_not_advertised(tmp_path):
    (tmp_path / "languages" / "assembly").mkdir(parents=True)

    assert resolve_language_dirs("assembly", tmp_path) == ("assembly",)


def test_flydsl_prompt_can_reach_assembly_workflow():
    config = Config(gpu_target="gfx950")

    assert resolve_language_dirs("flydsl", config.local_knowledge_dir) == ("flydsl", "assembly")
    prompt = build_single_kernel_backend_prompt(config, "flydsl")
    assert "## languages/assembly/" in prompt
    assert "## languages/flydsl/" in prompt


def test_opportunity_prompt_uses_backend_registry(monkeypatch):
    from kernelforge.kernel_rewrite_controller import opportunity_agent

    monkeypatch.setattr(opportunity_agent, "KERNEL_BACKENDS", ["assembly", "test_frontend"])
    prompt = opportunity_agent._system_prompt()

    assert "one of `assembly`, `test_frontend`" in prompt
    assert "{kernel_backends}" not in prompt
