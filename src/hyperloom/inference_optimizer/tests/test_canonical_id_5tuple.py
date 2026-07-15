# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the 5-tuple recipe-snapshot canonical_id derivation."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_FRAMEWORK_SLUG,
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    DEFAULT_HARDWARE_SLUG,
    DEFAULT_MODEL_SLUG,
    DEFAULT_PRECISION_SLUG,
    F_LABEL_FRAMEWORK_NAME,
    F_LABEL_FRAMEWORK_VERSION,
    F_LABEL_HARDWARE,
    F_LABEL_MODEL,
    F_LABEL_PRECISION,
    canonical_labels,
    detect_framework_version,
    recipe_canonical_id,
)


def test_canonical_id_shape_is_mhfvp_with_inference_prefix() -> None:
    """``inference:{model}:{hw}:{fw}:{fwver}:{precision}:{model_type}:{arch}`` — eight segments."""
    cid = recipe_canonical_id(
        model="Qwen3-30B-A3B",
        hardware="MI355X",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    assert cid == "inference:qwen3-30b-a3b:mi355x:sglang:unknown_model_type:unknown_arch:0.4.5:fp8"
    assert cid.split(":") == [
        "inference",
        "qwen3-30b-a3b",
        "mi355x",
        "sglang",
        "unknown_model_type",
        "unknown_arch",
        "0.4.5",
        "fp8",
    ]


def test_canonical_id_lowercases_every_component() -> None:
    cid = recipe_canonical_id(
        model="DEEPSEEK-R1",
        hardware="MI300X",
        framework_name="VLLM",
        framework_version="0.6.0+ROCM",
        precision="BF16",
    )
    assert cid.startswith("inference:")
    parts = cid.split(":")[1:]
    for part in parts:
        assert part == part.lower(), part


def test_canonical_id_basenames_path_style_model() -> None:
    """A path-style ``--model`` must converge on the same canonical_id as the bare name."""
    cid_path = recipe_canonical_id(
        model="/wekafs/models/Qwen3-30B-A3B",
        hardware="mi355x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    cid_bare = recipe_canonical_id(
        model="Qwen3-30B-A3B",
        hardware="mi355x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    assert cid_path == cid_bare


def test_canonical_id_is_keyword_only() -> None:
    """Positional args must raise ``TypeError`` — prevents accidental re-ordering."""
    # Splat a runtime-built arg list to keep the positional drift a runtime check.
    bad_positional_args = ["m", "h", "f", "v", "p"]
    with pytest.raises(TypeError):
        recipe_canonical_id(*bad_positional_args)  # type: ignore[misc]


def test_canonical_id_normalises_whitespace_and_slashes() -> None:
    """Whitespace and embedded ``/`` inside any slug become ``_``."""
    cid = recipe_canonical_id(
        model="A B",
        hardware="mi 355x",
        framework_name="sg lang",
        framework_version="0.4 5",
        precision="fp 8",
    )
    assert cid == "inference:a_b:mi_355x:sg_lang:unknown_model_type:unknown_arch:0.4_5:fp_8"


def test_canonical_id_substitutes_defaults_for_each_empty_component() -> None:
    from hyperloom.inference_optimizer.recipe_snapshot_constants import (
        DEFAULT_MODEL_TYPE_SLUG,
        DEFAULT_ARCHITECTURES_SLUG,
    )

    cid = recipe_canonical_id(
        model="",
        hardware="",
        framework_name="",
        framework_version="",
        precision="",
    )
    assert cid == (
        f"inference:"
        f"{DEFAULT_MODEL_SLUG}:"
        f"{DEFAULT_HARDWARE_SLUG}:"
        f"{DEFAULT_FRAMEWORK_SLUG}:"
        f"{DEFAULT_MODEL_TYPE_SLUG}:"
        f"{DEFAULT_ARCHITECTURES_SLUG}:"
        f"{DEFAULT_FRAMEWORK_VERSION_SLUG}:"
        f"{DEFAULT_PRECISION_SLUG}"
    )


def test_canonical_id_treats_whitespace_only_as_empty() -> None:
    """Whitespace-only is logically empty — must fall back to the default slug."""
    cid = recipe_canonical_id(
        model="   ",
        hardware="\t",
        framework_name="",
        framework_version="\n",
        precision="",
    )
    assert DEFAULT_MODEL_SLUG in cid
    assert DEFAULT_HARDWARE_SLUG in cid
    assert DEFAULT_FRAMEWORK_VERSION_SLUG in cid


def test_canonical_id_partial_defaults_keep_explicit_components() -> None:
    """Explicit components survive verbatim; only missing slots get defaults."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import (
        DEFAULT_MODEL_TYPE_SLUG,
        DEFAULT_ARCHITECTURES_SLUG,
    )

    cid = recipe_canonical_id(
        model="m",
        hardware="",
        framework_name="sglang",
        framework_version="",
        precision="fp8",
    )
    parts = cid.split(":")
    assert parts == [
        "inference",
        "m",
        DEFAULT_HARDWARE_SLUG,
        "sglang",
        DEFAULT_MODEL_TYPE_SLUG,
        DEFAULT_ARCHITECTURES_SLUG,
        DEFAULT_FRAMEWORK_VERSION_SLUG,
        "fp8",
    ]


def test_canonical_id_defaults_empty_architecture_list() -> None:
    from hyperloom.inference_optimizer.recipe_snapshot_constants import DEFAULT_ARCHITECTURES_SLUG

    cid = recipe_canonical_id(
        model="m",
        hardware="h",
        framework_name="sglang",
        architectures=[],
        framework_version="v",
        precision="p",
    )
    assert cid.split(":")[5] == DEFAULT_ARCHITECTURES_SLUG


def test_canonical_labels_keys_match_documented_constants() -> None:
    from hyperloom.inference_optimizer.recipe_snapshot_constants import (
        F_LABEL_MODEL_TYPE,
        F_LABEL_ARCHITECTURES,
    )

    labels = canonical_labels(
        model="m",
        hardware="h",
        framework_name="f",
        framework_version="v",
        precision="p",
    )
    assert set(labels.keys()) == {
        F_LABEL_MODEL,
        F_LABEL_HARDWARE,
        F_LABEL_FRAMEWORK_NAME,
        F_LABEL_FRAMEWORK_VERSION,
        F_LABEL_PRECISION,
        F_LABEL_MODEL_TYPE,
        F_LABEL_ARCHITECTURES,
    }


def test_canonical_labels_values_match_canonical_id_components() -> None:
    """A label's value MUST equal the corresponding canonical_id slug."""
    cid = recipe_canonical_id(
        model="DeepSeek-R1",
        hardware="MI300X",
        framework_name="vLLM",
        framework_version="0.6.0",
        precision="bf16",
    )
    labels = canonical_labels(
        model="DeepSeek-R1",
        hardware="MI300X",
        framework_name="vLLM",
        framework_version="0.6.0",
        precision="bf16",
    )
    parts = cid.split(":")[1:]
    assert labels[F_LABEL_MODEL] == parts[0]
    assert labels[F_LABEL_HARDWARE] == parts[1]
    assert labels[F_LABEL_FRAMEWORK_NAME] == parts[2]
    assert labels[F_LABEL_FRAMEWORK_VERSION] == parts[5]
    assert labels[F_LABEL_PRECISION] == parts[6]


def test_canonical_labels_substitutes_defaults_for_empty() -> None:
    from hyperloom.inference_optimizer.recipe_snapshot_constants import (
        DEFAULT_MODEL_TYPE_SLUG,
        DEFAULT_ARCHITECTURES_SLUG,
        F_LABEL_MODEL_TYPE,
        F_LABEL_ARCHITECTURES,
    )

    labels = canonical_labels(
        model="",
        hardware="",
        framework_name="",
        framework_version="",
        precision="",
    )
    assert labels == {
        F_LABEL_MODEL: DEFAULT_MODEL_SLUG,
        F_LABEL_HARDWARE: DEFAULT_HARDWARE_SLUG,
        F_LABEL_FRAMEWORK_NAME: DEFAULT_FRAMEWORK_SLUG,
        F_LABEL_FRAMEWORK_VERSION: DEFAULT_FRAMEWORK_VERSION_SLUG,
        F_LABEL_PRECISION: DEFAULT_PRECISION_SLUG,
        F_LABEL_MODEL_TYPE: DEFAULT_MODEL_TYPE_SLUG,
        F_LABEL_ARCHITECTURES: DEFAULT_ARCHITECTURES_SLUG,
    }


def test_detect_framework_version_returns_default_for_blank_framework_name() -> None:
    assert detect_framework_version("") == DEFAULT_FRAMEWORK_VERSION_SLUG
    assert detect_framework_version("   ") == DEFAULT_FRAMEWORK_VERSION_SLUG


def test_detect_framework_version_returns_default_for_unknown_framework_name() -> None:
    """Frameworks not in the allow-list never trigger an import."""
    assert detect_framework_version("not-a-real-framework_name") == (DEFAULT_FRAMEWORK_VERSION_SLUG)


def test_detect_framework_version_reads_dunder_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesise a fake package and confirm ``__version__`` is read + slugged."""
    fake = types.ModuleType("vllm")
    fake.__version__ = "0.6.3+rocm"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    assert importlib.import_module("vllm") is fake
    detected = detect_framework_version("vllm")
    assert detected == "0.6.3+rocm"


def test_detect_framework_version_falls_back_to_VERSION_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falls back to ``VERSION`` when ``__version__`` is absent (PEP 396)."""
    fake = types.ModuleType("sglang")
    # No __version__ — only VERSION.
    fake.VERSION = "0.4.5"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake)
    assert detect_framework_version("sglang") == "0.4.5"


def test_detect_framework_version_returns_default_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failure must not raise — the helper is best-effort."""
    monkeypatch.delitem(sys.modules, "atom", raising=False)

    real_import_module = importlib.import_module

    def _fake_import_module(name: str, *a, **kw):
        if name == "atom":
            raise ImportError("simulated: atom not installed")
        return real_import_module(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)
    assert detect_framework_version("atom") == DEFAULT_FRAMEWORK_VERSION_SLUG


def test_detect_framework_version_strips_dunder_version_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray-whitespace ``__version__`` must produce a well-formed slug."""
    fake = types.ModuleType("vllm")
    fake.__version__ = "  0.6.0\n"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    assert detect_framework_version("vllm") == "0.6.0"
