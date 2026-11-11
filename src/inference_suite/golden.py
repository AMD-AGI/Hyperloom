# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch keys a golden acceptance length owns, so no layer above the base may set them.

A golden acceptance length is only valid for the draft it was recorded for (InferenceX keys its
curve by model, method and draft length), so what is drafted and how acceptance is simulated
belong to the golden layer; how fast the draft runs stays open.
"""

from __future__ import annotations

from collections.abc import Mapping

from inference_suite.argv import SPECULATIVE_CONFIG_FLAG, speculative_config

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


def is_golden_owned_flag(name: str) -> bool:
    """Whether the flag ``name`` (canonical, as :func:`~inference_suite.argv.flag_occurrences` spells it) is golden's."""
    return name in GOLDEN_OWNED_FLAGS or name.startswith(GOLDEN_OWNED_FLAG_PREFIXES)


def golden_owned_settings(occurrences: list[tuple[str, str]], envs: Mapping[str, object]) -> dict[str, object]:
    """The golden-owned settings a launch carries, keyed by ``--flag``, ``--speculative-config.<key>`` or env name.

    A repeated flag keeps its last value, as the server parses it. vLLM also takes a ``--speculative-config.<key>``
    per key, merged over the JSON object by :func:`~inference_suite.argv.speculative_config`, so both spellings land
    in the same key.
    """
    settings: dict[str, object] = {str(k): v for k, v in envs.items() if str(k).startswith(GOLDEN_OWNED_ENV_PREFIXES)}
    settings.update({f"--{name}": value for name, value in dict(occurrences).items() if is_golden_owned_flag(name)})
    config = speculative_config(occurrences)
    settings.update(
        {f"--{SPECULATIVE_CONFIG_FLAG}.{key}": config[key] for key in GOLDEN_OWNED_SPEC_CONFIG_KEYS & config.keys()}
    )
    return settings


__all__ = [
    "GOLDEN_OWNED_ENV_PREFIXES",
    "GOLDEN_OWNED_FLAGS",
    "GOLDEN_OWNED_FLAG_PREFIXES",
    "GOLDEN_OWNED_SPEC_CONFIG_KEYS",
    "golden_owned_settings",
    "is_golden_owned_flag",
]
