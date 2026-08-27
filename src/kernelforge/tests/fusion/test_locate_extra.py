# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover locate resolution branches: package dir, missing model_type, read errors."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from kernelforge.fusion import locate
from kernelforge.fusion.locate import (
    _first_source_file,
    _package_dir,
    _read_source,
    resolve_framework_source_file,
)


def test_resolve_empty_model_type_returns_empty(monkeypatch):
    # model_type unresolvable from config -> ("", ...)
    monkeypatch.setattr(locate, "load_model_config", lambda p: {})
    path, note = resolve_framework_source_file("/m", "sglang")
    assert path == ""


def test_first_source_file_uses_package_dir(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "sglang"
    (pkg_dir / "srt" / "models").mkdir(parents=True)
    (pkg_dir / "srt" / "models" / "lfm2.py").write_text("# m")
    monkeypatch.setattr(locate, "_package_dir", lambda pkg: str(pkg_dir))
    got = _first_source_file("lfm2", "", ("srt/models",), pkg="sglang", pkg_models=("srt", "models"))
    assert got.endswith("srt/models/lfm2.py")


def test_first_source_file_none_found(monkeypatch):
    monkeypatch.setattr(locate, "_package_dir", lambda pkg: "")
    got = _first_source_file("nope", "", ("srt/models",), pkg="sglang", pkg_models=("srt", "models"))
    assert got == ""


def test_package_dir_import_error(monkeypatch):
    def boom(pkg):
        raise ImportError("no pkg")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert _package_dir("nonexistent_pkg_xyz") == ""


def test_package_dir_spec_none(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda pkg: None)
    assert _package_dir("whatever") == ""


def test_package_dir_real_stdlib():
    # A real installed package (json) resolves to its own directory.
    d = _package_dir("json")
    assert d and d.endswith("json")


class TestVllmRegisteredSource:
    """The registry, not the path convention, decides which file vLLM runs.

    The entries below are shaped like real ``ModelRegistry.models`` values read
    off vllm 0.1.dev19253+g5f76ae224, which is what the code has to survive:

    ==========================  ===========================================  =========================
    architecture                module_name                                  class_name
    ==========================  ===========================================  =========================
    ``LlamaForCausalLM``        ``vllm.model_executor.models.llama``         ``LlamaForCausalLM``
    ``DeepseekV4ForCausalLM``   ``vllm.models.deepseek_v4``                  ``DeepseekV4ForCausalLM``
    ``DeepseekV32ForCausalLM``  ``vllm.model_executor.models.deepseek_v2``   ``DeepseekV3ForCausalLM``
    ==========================  ===========================================  =========================

    The first two differ in layout, and the third is one of the 93 entries whose
    class is not named after the architecture.
    """

    def _registry(self, monkeypatch, models: dict):
        """Publish ``models`` as ``ModelRegistry.models`` in a fake vllm package."""
        registry = types.ModuleType("vllm.model_executor.models.registry")
        registry.ModelRegistry = types.SimpleNamespace(models=models)
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.models",
            "vllm.model_executor.models.registry",
        ):
            monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "vllm.model_executor.models.registry", registry)

    def _module(self, tmp_path, monkeypatch, module_name: str, class_name: str):
        """Import a one-class module from disk under ``module_name``."""
        impl = tmp_path / "model.py"
        impl.write_text(f"class {class_name}:\n    pass\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location(module_name, impl)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, module_name, module)
        return impl, module

    def _config(self, monkeypatch, model_type: str, arch: str):
        monkeypatch.setattr(
            locate,
            "load_model_config",
            lambda p: {"model_type": model_type, "architectures": [arch]},
        )

    @pytest.mark.parametrize(
        "arch, module_name, class_name, model_type",
        [
            ("LlamaForCausalLM", "vllm.model_executor.models.llama", "LlamaForCausalLM", "llama"),
            ("DeepseekV4ForCausalLM", "vllm.models.deepseek_v4", "DeepseekV4ForCausalLM", "deepseek_v4"),
            (
                "DeepseekV32ForCausalLM",
                "vllm.model_executor.models.deepseek_v2",
                "DeepseekV3ForCausalLM",
                "deepseek_v2",
            ),
        ],
    )
    def test_follows_the_registered_entry(self, tmp_path, monkeypatch, arch, module_name, class_name, model_type):
        impl, _ = self._module(tmp_path, monkeypatch, module_name, class_name)
        self._registry(
            monkeypatch,
            {
                arch: types.SimpleNamespace(module_name=module_name, class_name=class_name),
            },
        )
        self._config(monkeypatch, model_type, arch)

        assert resolve_framework_source_file("/m", "vllm") == (str(impl), "vllm registry")

    def test_follows_an_out_of_tree_class(self, tmp_path, monkeypatch):
        """``register_model(arch, cls)`` stores the class, not a module name."""
        impl, module = self._module(tmp_path, monkeypatch, "my_pkg.my_mod", "MyCustomModel")
        self._registry(
            monkeypatch,
            {
                "MyCustomModel": types.SimpleNamespace(model_cls=module.MyCustomModel),
            },
        )
        self._config(monkeypatch, "custom", "MyCustomModel")

        assert resolve_framework_source_file("/m", "vllm") == (str(impl), "vllm registry")

    def test_falls_back_to_the_path_convention(self, tmp_path, monkeypatch):
        """An architecture the registry does not know must not lose the legacy lookup."""
        models = tmp_path / "vllm" / "model_executor" / "models"
        models.mkdir(parents=True)
        (models / "llama.py").write_text("# m", encoding="utf-8")
        self._registry(monkeypatch, {})
        monkeypatch.setattr(locate, "_package_dir", lambda pkg: str(tmp_path / "vllm"))
        self._config(monkeypatch, "llama", "LlamaForCausalLM")

        assert resolve_framework_source_file("/m", "vllm") == (
            str(models / "llama.py"),
            "path convention (registry missed)",
        )

    def test_names_a_registered_arch_it_could_not_follow(self, tmp_path, monkeypatch, caplog):
        """The silent debug fallback is what hid this resolution being broken."""
        self._registry(
            monkeypatch,
            {
                "SomeArch": types.SimpleNamespace(
                    module_name="vllm.model_executor.models.nonexistent",
                    class_name="SomeClass",
                ),
            },
        )
        self._config(monkeypatch, "some", "SomeArch")
        monkeypatch.setattr(locate, "_package_dir", lambda pkg: "")

        with caplog.at_level(logging.WARNING, logger="kernelforge.fusion.locate"):
            resolve_framework_source_file("/m", "vllm")

        assert [r for r in caplog.records if "SomeArch" in r.getMessage()]


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vllm not installed")
def test_the_real_registry_resolves_to_a_file_in_the_vllm_package(tmp_path):
    """Run the resolution against the installed vLLM, which CI cannot do.

    The hermetic tests above assert against entries transcribed by hand, so they
    agree with the code even if both are wrong about vLLM. This one does not.
    """
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )

    resolved = locate._vllm_registered_source(str(tmp_path))

    vllm_dir = Path(importlib.util.find_spec("vllm").origin).parent
    assert Path(resolved).is_file()
    assert Path(resolved).is_relative_to(vllm_dir)


def test_read_source_empty_path():
    assert _read_source("") == ""


def test_read_source_missing_file(tmp_path):
    assert _read_source(str(tmp_path / "nope.py")) == ""


def test_read_source_ok(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("# content\n")
    assert _read_source(str(f)) == "# content\n"
