# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source resolution must follow ``architectures``, not the model_type filename.

Every model family added after the filename convention broke down is covered
here: an implementation named after another family, a multimodal wrapper that
delegates to a decoder in a sibling file, and a per-vendor fork.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.fusion.locate import resolve_framework_source_file

DECODER = "class {prefix}Attention(nn.Module):\n    pass\n\n\nclass {prefix}DecoderLayer(nn.Module):\n    pass\n"


def _model_dir(tmp_path: Path, *, model_type: str, architectures: list[str], text_config: dict | None = None) -> str:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    config: dict = {"model_type": model_type, "architectures": architectures}
    if text_config is not None:
        config["text_config"] = text_config
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model)


def _models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "fw" / "model_executor" / "models"
    models.mkdir(parents=True, exist_ok=True)
    return models


def _root(tmp_path: Path) -> str:
    return str(tmp_path / "fw")


def test_resolves_via_architecture_when_no_file_matches_model_type(tmp_path: Path) -> None:
    """GLM-5.2 shape: served by another family's file, so the name never matches."""
    models = _models_dir(tmp_path)
    (models / "deepseek_v2.py").write_text(
        DECODER.format(prefix="DeepseekV2") + "\n\nclass GlmMoeDsaForCausalLM(nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    model_path = _model_dir(tmp_path, model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"])

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == str(models / "deepseek_v2.py")


def test_prefers_decoder_over_multimodal_wrapper(tmp_path: Path) -> None:
    """gemma-4 shape: the registered class lives in a wrapper with nothing fusible."""
    models = _models_dir(tmp_path)
    (models / "gemma4_mm.py").write_text(
        "class Gemma4MultiModalProcessor:\n    pass\n\n\nclass Gemma4ForConditionalGeneration(nn.Module):\n    pass\n",
        encoding="utf-8",
    )
    (models / "gemma4.py").write_text(DECODER.format(prefix="Gemma4"), encoding="utf-8")
    model_path = _model_dir(tmp_path, model_type="gemma4", architectures=["Gemma4ForConditionalGeneration"])

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == str(models / "gemma4.py")


def test_prefers_vendor_fork_for_the_running_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """deepseek_v4 shape: amd/, nvidia/ and xpu/ all define the same class."""
    plugin = tmp_path / "fw" / "models" / "deepseek_v4"
    for vendor in ("amd", "nvidia", "xpu"):
        vendor_dir = plugin / vendor
        vendor_dir.mkdir(parents=True, exist_ok=True)
        padding = "# pad\n" * (100 if vendor == "nvidia" else 1)
        (vendor_dir / "model.py").write_text(
            DECODER.format(prefix="DeepseekV4") + padding + "\nclass DeepseekV4ForCausalLM(nn.Module):\n    pass\n",
            encoding="utf-8",
        )
    model_path = _model_dir(tmp_path, model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"])
    monkeypatch.setenv("FORGE_FUSION_VENDOR", "amd")

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    # nvidia is the larger file; the vendor hint has to outrank size.
    assert got == str(plugin / "amd" / "model.py")


def test_text_config_architecture_wins_over_wrapper(tmp_path: Path) -> None:
    """MiniMax shape: the text tower is named only inside ``text_config``."""
    models = _models_dir(tmp_path)
    (models / "wrapper.py").write_text("class FooForConditionalGeneration:\n    pass\n", encoding="utf-8")
    (models / "tower.py").write_text(
        DECODER.format(prefix="Foo") + "\nclass FooForCausalLM(nn.Module):\n    pass\n", encoding="utf-8"
    )
    model_path = _model_dir(
        tmp_path,
        model_type="foo_vl",
        architectures=["FooForConditionalGeneration"],
        text_config={"architectures": ["FooForCausalLM"]},
    )

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == str(models / "tower.py")


def test_config_suffix_class_is_not_mistaken_for_the_model(tmp_path: Path) -> None:
    """``BarForCausalLMConfig`` must not satisfy a search for ``BarForCausalLM``."""
    models = _models_dir(tmp_path)
    (models / "other.py").write_text("class BarForCausalLMConfig:\n    pass\n", encoding="utf-8")
    model_path = _model_dir(tmp_path, model_type="bar", architectures=["BarForCausalLM"])

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == ""


def test_legacy_model_type_still_resolves(tmp_path: Path) -> None:
    """qwen3/llama shape: the historical guess keeps working unchanged."""
    models = _models_dir(tmp_path)
    (models / "qwen3.py").write_text(DECODER.format(prefix="Qwen3"), encoding="utf-8")
    model_path = _model_dir(tmp_path, model_type="qwen3", architectures=[])

    got, _how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == str(models / "qwen3.py")


def test_unknown_framework_returns_empty(tmp_path: Path) -> None:
    model_path = _model_dir(tmp_path, model_type="x", architectures=["XForCausalLM"])

    path, how = resolve_framework_source_file(model_path, "tensorrt", framework_root=_root(tmp_path))

    assert path == ""
    assert "unsupported" in how


def test_a_pinned_root_outranks_the_installed_vllm_the_registry_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry answers for the importable vLLM, not the pinned one.

    The author stage patches ``--framework-root``, so resolving outside it hands
    the campaign a file no patch of it can reach.
    """
    models = _models_dir(tmp_path)
    (models / "qwen3.py").write_text(DECODER.format(prefix="Qwen3"), encoding="utf-8")
    model_path = _model_dir(tmp_path, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    monkeypatch.setattr(
        "kernelforge.fusion.locate._vllm_registered_source",
        lambda _p: "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py",
    )

    got, how = resolve_framework_source_file(model_path, "vllm", framework_root=_root(tmp_path))

    assert got == str(models / "qwen3.py")
    assert how != "vllm registry"


def test_the_registry_still_answers_when_no_root_is_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is pinned, so the importable vLLM IS the tree being optimized."""
    installed = tmp_path / "site-packages" / "vllm" / "model_executor" / "models"
    installed.mkdir(parents=True)
    (installed / "qwen3.py").write_text(DECODER.format(prefix="Qwen3"), encoding="utf-8")
    model_path = _model_dir(tmp_path, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    monkeypatch.setattr(
        "kernelforge.fusion.locate._vllm_registered_source",
        lambda _p: str(installed / "qwen3.py"),
    )

    got, how = resolve_framework_source_file(model_path, "vllm")

    assert got == str(installed / "qwen3.py")
    assert how == "vllm registry"
