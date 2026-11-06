# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch keys a golden acceptance length owns, so no layer above the base may set them.

A golden acceptance length is only valid for the draft it was recorded for (InferenceX keys its
curve by model, method and draft length), so what is drafted and how acceptance is simulated
belong to the golden layer; how fast the draft runs stays open.
"""

from __future__ import annotations

GOLDEN_OWNED_FLAGS = frozenset(
    {
        "speculative-num-steps",
        "speculative-num-draft-tokens",
        "speculative-eagle-topk",
        "speculative-algorithm",
        "speculative-draft-model-path",
        "speculative-dspark-block-size",
        "num-speculative-tokens",
        "method",
        "draft-model",
    }
)
GOLDEN_OWNED_FLAG_PREFIXES = ("spec-decode-acceptance", "speculative-acceptance")
# InferenceX does not simulate acceptance once vLLM's adaptive verification is on, so turning it on
# silently drops the golden acceptance length.
GOLDEN_OWNED_SPEC_CONFIG_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
        "rejection_sample_method",
        "synthetic_acceptance_length",
        "draft_sample_method",
        "enable_adaptive_verification",
    }
)
GOLDEN_OWNED_ENV_PREFIXES = ("SGLANG_SIMULATE_ACC_",)

__all__ = [
    "GOLDEN_OWNED_ENV_PREFIXES",
    "GOLDEN_OWNED_FLAGS",
    "GOLDEN_OWNED_FLAG_PREFIXES",
    "GOLDEN_OWNED_SPEC_CONFIG_KEYS",
]
