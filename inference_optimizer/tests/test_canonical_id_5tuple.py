# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the 5-tuple recipe-snapshot canonical_id derivation.

Covers the contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md`` and
the in-repo helper :func:`recipe_canonical_id`:

* Shape       — ``inference:{model}:{hw}:{fw}:{fwver}:{precision}``
                (mhfvp ordering: identity-strength strongest → weakest).
* Slug rules  — lowercase, basename-of-path, whitespace/colons/slashes
                normalised to ``_``.
* Defaults    — every empty / blank component falls back to its
                documented ``unknown_*`` slug; the canonical_id is
                always 6 colon-separated segments so the path
                converter never sees an empty component.
* Keyword-only — positional args raise ``TypeError`` to prevent
                 accidental component re-ordering.
* Labels mirror — :func:`canonical_labels` produces the five-key
                  dict that ``/recipes/search`` ``label_match`` keys
                  off of, slug-stable with the canonical_id.
* Auto-detect — :func:`detect_framework_version` reads
                ``framework.__version__`` for the supported
                frameworks and degrades to the default slug on import
                failure (or unsupported framework names).

These tests intentionally do NOT touch the HTTP client — see
``test_recipe_snapshot_client.py`` for that.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_FRAMEWORK_SLUG,
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    DEFAULT_HARDWARE_SLUG,
    DEFAULT_MODEL_SLUG,
    DEFAULT_PRECISION_SLUG,
    F_LABEL_FRAMEWORK,
    F_LABEL_FRAMEWORK_VERSION,
    F_LABEL_HARDWARE,
    F_LABEL_MODEL,
    F_LABEL_PRECISION,
    canonical_labels,
    detect_framework_version,
    recipe_canonical_id,
)


# ===========================================================================
# Shape — mhfvp ordering, six segments, ``inference:`` prefix
# ===========================================================================
def test_canonical_id_shape_is_mhfvp_with_inference_prefix() -> None:
    """``inference:{model}:{hw}:{fw}:{fwver}:{precision}`` — exactly
    six colon-separated segments, prefix ``inference``."""
    cid = recipe_canonical_id(
        model="Qwen3-30B-A3B",
        hardware="MI355X",
        framework="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    assert cid == "inference:qwen3-30b-a3b:mi355x:sglang:0.4.5:fp8"
    assert cid.split(":") == [
        "inference", "qwen3-30b-a3b", "mi355x", "sglang", "0.4.5", "fp8",
    ]


def test_canonical_id_lowercases_every_component() -> None:
    cid = recipe_canonical_id(
        model="DEEPSEEK-R1",
        hardware="MI300X",
        framework="VLLM",
        framework_version="0.6.0+ROCM",
        precision="BF16",
    )
    # Every dynamic component is lower-cased; only the literal prefix
    # keeps the reader's casing.
    assert cid.startswith("inference:")
    parts = cid.split(":")[1:]  # strip prefix
    for part in parts:
        assert part == part.lower(), part


def test_canonical_id_basenames_path_style_model() -> None:
    """A CLI ``--model /wekafs/models/Qwen3-30B-A3B`` must converge on
    the same canonical_id as the bare model name. Identity must NOT
    depend on where the model weights happen to be mounted."""
    cid_path = recipe_canonical_id(
        model="/wekafs/models/Qwen3-30B-A3B",
        hardware="mi355x",
        framework="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    cid_bare = recipe_canonical_id(
        model="Qwen3-30B-A3B",
        hardware="mi355x",
        framework="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )
    assert cid_path == cid_bare


def test_canonical_id_is_keyword_only() -> None:
    """Positional args must raise ``TypeError`` — prevents accidental
    re-ordering (``model`` swapped with ``hardware`` etc.) which would
    silently route a row to a different canonical_id."""
    with pytest.raises(TypeError):
        recipe_canonical_id(  # type: ignore[misc]
            "m", "h", "f", "v", "p",
        )


def test_canonical_id_normalises_whitespace_and_slashes() -> None:
    """Whitespace inside any slug becomes ``_``; embedded ``/`` after
    basename collapse also becomes ``_`` (defence-in-depth for the
    local filesystem store)."""
    cid = recipe_canonical_id(
        model="A B",
        hardware="mi 355x",
        framework="sg lang",
        framework_version="0.4 5",
        precision="fp 8",
    )
    assert cid == "inference:a_b:mi_355x:sg_lang:0.4_5:fp_8"


# ===========================================================================
# Defaults — every empty component substitutes the documented slug
# ===========================================================================
def test_canonical_id_substitutes_defaults_for_each_empty_component() -> None:
    cid = recipe_canonical_id(
        model="",
        hardware="",
        framework="",
        framework_version="",
        precision="",
    )
    assert cid == (
        f"inference:"
        f"{DEFAULT_MODEL_SLUG}:"
        f"{DEFAULT_HARDWARE_SLUG}:"
        f"{DEFAULT_FRAMEWORK_SLUG}:"
        f"{DEFAULT_FRAMEWORK_VERSION_SLUG}:"
        f"{DEFAULT_PRECISION_SLUG}"
    )


def test_canonical_id_treats_whitespace_only_as_empty() -> None:
    """``"   "`` is logically empty for identity purposes — must
    fall back to the default slug, not produce an underscore-only
    component."""
    cid = recipe_canonical_id(
        model="   ",
        hardware="\t",
        framework="",
        framework_version="\n",
        precision="",
    )
    assert DEFAULT_MODEL_SLUG in cid
    assert DEFAULT_HARDWARE_SLUG in cid
    assert DEFAULT_FRAMEWORK_VERSION_SLUG in cid


def test_canonical_id_partial_defaults_keep_explicit_components() -> None:
    """When only some components are unspecified, the explicit ones
    survive verbatim and only the missing slots get defaults."""
    cid = recipe_canonical_id(
        model="m",
        hardware="",
        framework="sglang",
        framework_version="",
        precision="fp8",
    )
    parts = cid.split(":")
    assert parts == [
        "inference", "m", DEFAULT_HARDWARE_SLUG, "sglang",
        DEFAULT_FRAMEWORK_VERSION_SLUG, "fp8",
    ]


# ===========================================================================
# canonical_labels — 5-key dict mirroring the canonical_id
# ===========================================================================
def test_canonical_labels_keys_match_documented_constants() -> None:
    labels = canonical_labels(
        model="m", hardware="h", framework="f",
        framework_version="v", precision="p",
    )
    assert set(labels.keys()) == {
        F_LABEL_MODEL,
        F_LABEL_HARDWARE,
        F_LABEL_FRAMEWORK,
        F_LABEL_FRAMEWORK_VERSION,
        F_LABEL_PRECISION,
    }


def test_canonical_labels_values_match_canonical_id_components() -> None:
    """A label's value MUST equal the corresponding canonical_id slug,
    so ``label_match`` round-trips without surprise."""
    cid = recipe_canonical_id(
        model="DeepSeek-R1",
        hardware="MI300X",
        framework="vLLM",
        framework_version="0.6.0",
        precision="bf16",
    )
    labels = canonical_labels(
        model="DeepSeek-R1",
        hardware="MI300X",
        framework="vLLM",
        framework_version="0.6.0",
        precision="bf16",
    )
    parts = cid.split(":")[1:]  # drop ``inference``
    assert labels[F_LABEL_MODEL]             == parts[0]
    assert labels[F_LABEL_HARDWARE]          == parts[1]
    assert labels[F_LABEL_FRAMEWORK]         == parts[2]
    assert labels[F_LABEL_FRAMEWORK_VERSION] == parts[3]
    assert labels[F_LABEL_PRECISION]         == parts[4]


def test_canonical_labels_substitutes_defaults_for_empty() -> None:
    labels = canonical_labels(
        model="", hardware="", framework="",
        framework_version="", precision="",
    )
    assert labels == {
        F_LABEL_MODEL:             DEFAULT_MODEL_SLUG,
        F_LABEL_HARDWARE:          DEFAULT_HARDWARE_SLUG,
        F_LABEL_FRAMEWORK:         DEFAULT_FRAMEWORK_SLUG,
        F_LABEL_FRAMEWORK_VERSION: DEFAULT_FRAMEWORK_VERSION_SLUG,
        F_LABEL_PRECISION:         DEFAULT_PRECISION_SLUG,
    }


# ===========================================================================
# detect_framework_version — auto-detect via importing __version__
# ===========================================================================
def test_detect_framework_version_returns_default_for_blank_framework() -> None:
    assert detect_framework_version("") == DEFAULT_FRAMEWORK_VERSION_SLUG
    assert detect_framework_version("   ") == DEFAULT_FRAMEWORK_VERSION_SLUG


def test_detect_framework_version_returns_default_for_unknown_framework() -> None:
    """Frameworks not in the allow-list never trigger an import — the
    helper degrades cleanly so a typo can't crash the boot."""
    assert detect_framework_version("not-a-real-framework") == (
        DEFAULT_FRAMEWORK_VERSION_SLUG
    )


def test_detect_framework_version_reads_dunder_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesise a fake top-level package and confirm
    ``__version__`` is read + slugged."""
    fake = types.ModuleType("vllm")
    fake.__version__ = "0.6.3+rocm"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    # ``importlib.import_module`` consults ``sys.modules`` first.
    assert importlib.import_module("vllm") is fake
    detected = detect_framework_version("vllm")
    # Slugged: lower-case (already lower), '+' preserved, dots
    # preserved, no spaces.
    assert detected == "0.6.3+rocm"


def test_detect_framework_version_falls_back_to_VERSION_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some frameworks export ``VERSION`` instead of ``__version__``;
    helper tries the standard PEP 396 spelling first, then falls back
    to ``VERSION`` so we don't lose info on non-conforming packages."""
    fake = types.ModuleType("sglang")
    # Intentionally no __version__ — only VERSION.
    fake.VERSION = "0.4.5"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake)
    assert detect_framework_version("sglang") == "0.4.5"


def test_detect_framework_version_returns_default_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failure must not raise — the helper is best-effort and
    the optimizer should still boot when the framework's not in the
    venv (CI / dry-runs)."""
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
    """A framework that exposes a stray-whitespace version (some
    nightly builds do ``__version__ = '0.4.5\\n'``) must produce a
    well-formed slug that doesn't break filesystem-backed local KB
    lookups in Commit 2."""
    fake = types.ModuleType("vllm")
    fake.__version__ = "  0.6.0\n"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake)
    assert detect_framework_version("vllm") == "0.6.0"
