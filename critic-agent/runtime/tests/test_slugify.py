# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Slugify spec tests against the contract Appendix D.4 golden fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from runtime.errors import SlugifyError
from runtime.slugify import slugify, slugify_safe


_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "slugify-golden.json"


def _golden_cases():
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_cases_all_pass():
    for case in _golden_cases():
        if "throws" in case:
            with pytest.raises(SlugifyError) as exc:
                slugify(case["input"])
            assert case["reason"] in str(exc.value)
            continue
        out = slugify(case["input"])
        if "output_pattern" in case:
            assert re.match(case["output_pattern"], out), (
                f"input={case['input']!r} got={out!r} expected pattern "
                f"{case['output_pattern']!r}"
            )
        else:
            assert out == case["output"], (
                f"input={case['input']!r} got={out!r} expected {case['output']!r}"
            )


def test_slugify_rejects_non_ascii():
    with pytest.raises(SlugifyError, match="non_ascii"):
        slugify("FA3 在 H100 上崩溃")


def test_slugify_rejects_too_short_after_normalisation():
    with pytest.raises(SlugifyError, match="too_short"):
        slugify("a-b")


def test_slugify_rejects_only_separators():
    with pytest.raises(SlugifyError, match="empty"):
        slugify("---   ___")


def test_slugify_safe_pure_ascii_passes_through():
    assert slugify_safe("MLA FP8 torch.compile incompat") == "mla-fp8-torch-compile-incompat"


def test_slugify_safe_uses_translate_fn_for_non_ascii():
    out = slugify_safe(
        "FA3 在 H100 上崩溃",
        translate_fn=lambda t: "fa3 h100 crash",
    )
    assert out == "fa3-h100-crash"


def test_slugify_safe_fallback_prefix_when_no_translator():
    out = slugify_safe("FA3 在 H100 上崩溃")
    assert out.startswith("auto-")
    assert len(out.split("-", 1)[1]) == 8


def test_slugify_safe_fallback_when_translator_raises():
    def bad_translate(_):
        raise RuntimeError("translation service unavailable")

    out = slugify_safe(
        "FA3 在 H100 上崩溃",
        translate_fn=bad_translate,
        fallback_prefix="critic",
    )
    assert out.startswith("critic-")


def test_slugify_safe_idempotent_for_same_non_ascii_input():
    a = slugify_safe("FA3 在 H100 上崩溃")
    b = slugify_safe("FA3 在 H100 上崩溃")
    assert a == b


def test_slugify_long_input_truncated_with_hash():
    long = "a" * 200
    out = slugify(long)
    assert len(out) == 72 + 1 + 7
    assert out.startswith("a" * 72 + "-")
