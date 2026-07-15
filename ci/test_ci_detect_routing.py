# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the CI auto-detection policy in optimize_submit.

Guards three policy decisions:

* ``detect_framework`` routes the new sglang-supported architectures to sglang.
* ``detect_tp`` uses the MI300X thresholds (80 / 128 / 256).
* ``detect_concurrency`` is a fixed 64 across frameworks and TP sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from optimize_submit import (  # noqa: E402
    detect_concurrency,
    detect_framework,
    detect_image,
    detect_tp,
)

_GEMMA4_VLLM_IMAGE = "harbor.core42.example-internal-host.invalid/sync/vllm-openai-rocm:gemma4"

# Architectures that must route to sglang, not the vLLM fallback image.
NEW_SGLANG_ARCHS = [
    "Gemma4ForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "NemotronHForCausalLM",
    "Glm4ForCausalLM",
    "Glm4MoeForCausalLM",
]


@pytest.mark.parametrize("arch", NEW_SGLANG_ARCHS)
def test_new_architectures_route_to_sglang(arch: str) -> None:
    assert detect_framework({"architectures": [arch]}) == "sglang"


def test_vllm_required_arch_still_routes_to_vllm() -> None:
    assert detect_framework({"architectures": ["MiniMaxText01ForCausalLM"]}) == "vllm"


def test_unknown_arch_falls_back_to_vllm() -> None:
    assert detect_framework({"architectures": ["TotallyUnknownForCausalLM"]}) == "vllm"


@pytest.mark.parametrize("arch", NEW_SGLANG_ARCHS)
def test_quantized_new_arch_still_routes_to_vllm(arch: str) -> None:
    """The quant guard runs before the sglang allowlist, so a quantized
    variant of a new arch still goes to vLLM."""
    config = {
        "architectures": [arch],
        "quantization_config": {"quant_method": "awq"},
    }
    assert detect_framework(config) == "vllm"


@pytest.mark.parametrize(
    "params_b,expected_tp",
    [
        (0, 1),
        (7, 1),
        (79, 1),
        (80, 1),
        (80.1, 2),
        (128, 2),
        (128.1, 4),
        (256, 4),
        (256.1, 8),
        (671, 8),
    ],
)
def test_detect_tp_mi300x_thresholds(params_b: float, expected_tp: int) -> None:
    assert detect_tp(params_b, gpu_type="mi300x") == expected_tp


@pytest.mark.parametrize(
    "tp,framework",
    [
        (1, "sglang"),
        (4, "sglang"),
        (8, "sglang"),
        (1, "vllm"),
        (4, "vllm"),
        (8, "vllm"),
    ],
)
def test_detect_concurrency_is_fixed_64(tp: int, framework: str) -> None:
    assert detect_concurrency(tp, framework) == 64


# ── detect_image: gemma-4 dedicated vLLM image ──


@pytest.mark.parametrize(
    "repo_id",
    [
        "google/gemma-4-26B-A4B-it",
        "google-gemma-4-26B-A4B-it",   # slug form
        "someorg/Gemma4-mini",          # 'gemma4' (no hyphen) matches
    ],
)
def test_gemma4_vllm_uses_dedicated_image(repo_id: str) -> None:
    assert detect_image("vllm", repo_id) == _GEMMA4_VLLM_IMAGE


def test_gemma4_sglang_unaffected() -> None:
    # The gemma4 override is vLLM-only.
    img = detect_image("sglang", "google/gemma-4-26B-A4B-it")
    assert img != _GEMMA4_VLLM_IMAGE
    assert "sglang" in img


def test_non_gemma4_vllm_uses_default_image() -> None:
    img = detect_image("vllm", "Qwen/Qwen3-32B")
    assert img != _GEMMA4_VLLM_IMAGE
    assert "vllm-openai-rocm" in img
