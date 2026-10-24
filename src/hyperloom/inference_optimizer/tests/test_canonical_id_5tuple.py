# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the 7-tuple recipe-snapshot canonical_id derivation."""

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
    RecipeIdentity,
    detect_framework_version,
    kb_hardware_slug,
    recipe_canonical_id,
)


def test_canonical_id_shape_is_mhfvp_with_inference_prefix() -> None:
    """``inference:{model}:{hw}:{fw}:{fwver}:{precision}:{model_type}:{arch}`` — eight segments."""
    cid = recipe_canonical_id(
        RecipeIdentity(
            model="Qwen3-30B-A3B",
            hardware="MI355X",
            framework_name="sglang",
            framework_version="0.4.5",
            precision="fp8",
        )
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
        RecipeIdentity(
            model="DEEPSEEK-R1",
            hardware="MI300X",
            framework_name="VLLM",
            framework_version="0.6.0+ROCM",
            precision="BF16",
        )
    )
    assert cid.startswith("inference:")
    parts = cid.split(":")[1:]
    for part in parts:
        assert part == part.lower(), part


def test_canonical_id_basenames_path_style_model() -> None:
    """A path-style ``--model`` must converge on the same canonical_id as the bare name."""
    cid_path = recipe_canonical_id(
        RecipeIdentity(
            model="/path/models/Qwen3-30B-A3B",
            hardware="mi355x",
            framework_name="sglang",
            framework_version="0.4.5",
            precision="fp8",
        )
    )
    cid_bare = recipe_canonical_id(
        RecipeIdentity(
            model="Qwen3-30B-A3B",
            hardware="mi355x",
            framework_name="sglang",
            framework_version="0.4.5",
            precision="fp8",
        )
    )
    assert cid_path == cid_bare


def test_canonical_id_is_keyword_only() -> None:
    """Positional args must raise ``TypeError`` — prevents accidental re-ordering."""
    # Splat a runtime-built arg list to keep the positional drift a runtime check.
    bad_positional_args = ["m", "h", "f", "v", "p"]
    with pytest.raises(TypeError):
        RecipeIdentity(*bad_positional_args)  # type: ignore[misc]


def test_canonical_id_normalises_whitespace_and_slashes() -> None:
    """Whitespace and embedded ``/`` inside any slug become ``_``."""
    cid = recipe_canonical_id(
        RecipeIdentity(
            model="A B",
            hardware="mi 355x",
            framework_name="sg lang",
            framework_version="0.4 5",
            precision="fp 8",
        )
    )
    assert cid == "inference:a_b:mi_355x:sg_lang:unknown_model_type:unknown_arch:0.4_5:fp_8"


def test_canonical_id_substitutes_defaults_for_each_empty_component() -> None:
    from hyperloom.inference_optimizer.recipe_snapshot_constants import (
        DEFAULT_MODEL_TYPE_SLUG,
        DEFAULT_ARCHITECTURES_SLUG,
    )

    cid = recipe_canonical_id(
        RecipeIdentity(
            model="",
            hardware="",
            framework_name="",
            framework_version="",
            precision="",
        )
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
        RecipeIdentity(
            model="   ",
            hardware="\t",
            framework_name="",
            framework_version="\n",
            precision="",
        )
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
        RecipeIdentity(
            model="m",
            hardware="",
            framework_name="sglang",
            framework_version="",
            precision="fp8",
        )
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
        RecipeIdentity(
            model="m",
            hardware="h",
            framework_name="sglang",
            architectures=[],
            framework_version="v",
            precision="p",
        )
    )
    assert cid.split(":")[5] == DEFAULT_ARCHITECTURES_SLUG


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


def test_detect_framework_version_reads_the_resolved_serving_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An isolated vLLM venv is not importable here; the KB key must not degrade to ``unknown_version``."""
    site = tmp_path / "vllm-venv" / "lib" / "python3.12" / "site-packages"
    info = site / "vllm-0.29.0+rocm100.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.4\nName: vllm\nVersion: 0.29.0+rocm100\n")
    bin_dir = tmp_path / "vllm-venv" / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    monkeypatch.setenv("HYPERLOOM_RESOLVED_FRAMEWORK", "vllm")
    monkeypatch.setenv("HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON", str(bin_dir / "python"))
    monkeypatch.setitem(sys.modules, "vllm", None)

    assert detect_framework_version("vllm") == "0.29.0+rocm100"


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


def test_kb_hardware_slug_single_node_unchanged() -> None:
    """Single-node KB keys must stay byte-for-byte identical to pre-infera runs."""
    gpu = "MI300X"
    assert kb_hardware_slug(gpu, nodes=1) == gpu
    # Cluster kwargs are all ignored when nodes < 2, and the default shape (dense, whole card) adds nothing.
    assert (
        kb_hardware_slug(
            gpu,
            nodes=1,
            gpus_per_node=8,
            pd_mode="disaggregated",
            pd_prefill_nodes=2,
            pd_decode_nodes=1,
            tp=8,
            ep=1,
            partition_mode="SPX",
            backend="rayjob",
        )
        == gpu
    )


def test_kb_hardware_slug_separates_a_divided_card_from_a_whole_one() -> None:
    """SPX and CPX share a GPU type but are different machines; the key has to say so."""
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, partition_mode="CPX") == "MI300X_cpx"
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, partition_mode="DPX") == "MI300X_dpx"
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, partition_mode="QPX") == "MI300X_qpx"
    # The mode also suffixes inside a cluster key, after the formation and before the backend.
    assert (
        kb_hardware_slug("MI300X", nodes=2, gpus_per_node=8, tp=8, ep=8, partition_mode="CPX", backend="rayjob")
        == "MI300X_ws16_tp8_ep8_cpx_rayjob"
    )


def test_kb_hardware_slug_treats_the_default_shape_as_no_shape() -> None:
    """Anything that would re-key a historical row has to be omitted, not encoded."""
    # One partition IS the whole card, so SPX must be indistinguishable from a mode nobody published.
    assert kb_hardware_slug("MI300X", nodes=1, partition_mode="SPX") == "MI300X"
    assert kb_hardware_slug("MI300X", nodes=1, partition_mode="") == "MI300X"
    # A mode this build does not know is not a licence to invent a key for it.
    assert kb_hardware_slug("MI300X", nodes=1, partition_mode="TPX") == "MI300X"
    # tp is deliberately NOT encoded single-node: almost every run sets it, so it would move the whole corpus.
    assert kb_hardware_slug("MI300X", nodes=1, tp=8) == "MI300X"


def test_kb_hardware_slug_encodes_single_node_expert_parallelism() -> None:
    """``ep`` is fixed at launch and invalidates a config across splits, whatever the node count."""
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, ep=8) == "MI300X_ep8"
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, ep=8, partition_mode="CPX") == "MI300X_ep8_cpx"
    # Dense is the default, so ep<=1 stays silent and keeps the historical key.
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, ep=1) == "MI300X"
    assert kb_hardware_slug("MI300X", nodes=1, tp=8, ep=0) == "MI300X"


def test_kb_hardware_slug_multi_node_adds_topology_suffix() -> None:
    """Contrast: multi-node encodes world_size (and PD split when disaggregated)."""
    assert kb_hardware_slug("MI300X", nodes=2, gpus_per_node=8) == "MI300X_ws16"
    assert (
        kb_hardware_slug(
            "MI300X",
            nodes=2,
            gpus_per_node=8,
            pd_mode="disaggregated",
            pd_prefill_nodes=1,
            pd_decode_nodes=1,
        )
        == "MI300X_ws16_pd1p1d"
    )


def test_kb_hardware_slug_encodes_formation_and_backend() -> None:
    """tp/ep/backend enter the key for BOTH aggregated and disaggregated."""
    # Aggregated (no _pd): world_size + tp + ep + backend.
    assert (
        kb_hardware_slug(
            "MI325X",
            nodes=2,
            gpus_per_node=2,
            tp=4,
            ep=4,
            backend="rayjob",
        )
        == "MI325X_ws4_tp4_ep4_rayjob"
    )
    # Disaggregated: _pd split precedes the formation dims.
    assert (
        kb_hardware_slug(
            "MI300X",
            nodes=2,
            gpus_per_node=8,
            pd_mode="disaggregated",
            pd_prefill_nodes=1,
            pd_decode_nodes=1,
            tp=8,
            ep=8,
            backend="infera",
        )
        == "MI300X_ws16_pd1p1d_tp8_ep8_infera"
    )
    # ep==1 (no expert parallelism) is omitted; tp still encoded.
    assert (
        kb_hardware_slug("MI300X", nodes=2, gpus_per_node=8, tp=8, ep=1, backend="rayjob") == "MI300X_ws16_tp8_rayjob"
    )
