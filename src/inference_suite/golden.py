# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch keys a golden acceptance length owns, so no layer above the base may set them.

A golden acceptance length is only valid for the draft it was recorded for (InferenceX keys its
curve by model, method and draft length), so what is drafted and how acceptance is simulated
belong to the golden layer; how fast the draft runs stays open.

:func:`real_acceptance` is the golden layer with only the simulation taken out (sglang's ``SGLANG_SIMULATE_ACC_*``
environment, vLLM's synthetic rejection sampling in ``--speculative-config``): the same draft, accepting what the
target model really accepts, so the server answers with the model's own text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from inference_suite.argv import (
    SPECULATIVE_CONFIG_FLAG,
    SPECULATIVE_CONFIG_KEY_PREFIX,
    canonical_flag_name,
    is_flag,
    json_object,
    normalize_argv,
    speculative_config,
)

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
_REJECTION_SAMPLER = "rejection_sample_method"
_SYNTHETIC_LENGTH = "synthetic_acceptance_length"
GOLDEN_OWNED_SPEC_CONFIG_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
        _REJECTION_SAMPLER,
        _SYNTHETIC_LENGTH,
        "draft_sample_method",
        "enable_adaptive_verification",
    }
)
GOLDEN_OWNED_ENV_PREFIXES = ("SGLANG_SIMULATE_ACC_",)
#: vLLM's ``--speculative-config`` keys that simulate acceptance: the synthetic sampler and the length it forces.
#: ``rejection_sample_method`` simulates only when it names the synthetic sampler.
SIMULATED_SPEC_CONFIG_KEYS = (_REJECTION_SAMPLER, _SYNTHETIC_LENGTH)
_SYNTHETIC_SAMPLER = "synthetic"


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


def _simulates(key: str, value: object) -> bool:
    """Whether the speculative-config ``key`` set to ``value`` simulates acceptance."""
    if key == _REJECTION_SAMPLER:
        return value == _SYNTHETIC_SAMPLER
    return key in SIMULATED_SPEC_CONFIG_KEYS


def _groups(args: Sequence[str]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for token in args:
        if is_flag(token) or not groups:
            groups.append((token, []))
        else:
            groups[-1][1].append(token)
    return groups


def _without_simulation(flag: str, values: list[str]) -> list[str] | None:
    """The values of the golden flag group ``flag`` with its simulation taken out; ``None`` drops the whole group."""
    name = canonical_flag_name(flag)
    if name.startswith(SPECULATIVE_CONFIG_KEY_PREFIX):
        key = name.removeprefix(SPECULATIVE_CONFIG_KEY_PREFIX).replace("-", "_")
        return None if _simulates(key, values[0] if values else "") else values
    if name != SPECULATIVE_CONFIG_FLAG or len(values) != 1 or not (config := json_object(values[0])):
        return values
    return [json.dumps({key: value for key, value in config.items() if not _simulates(key, value)})]


def real_acceptance(args: Sequence[str], env: Mapping[str, str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """The golden layer ``args`` and ``env`` less its acceptance simulation, every other key kept: no
    ``SGLANG_SIMULATE_ACC_*`` variable, and in vLLM's ``--speculative-config`` (the JSON object, or its per-key flags)
    no synthetic ``rejection_sample_method`` and no ``synthetic_acceptance_length``. The draft, its length and its
    method stay as the golden layer names them."""
    kept: list[str] = []
    for flag, values in _groups(args):
        if not is_flag(flag):
            kept.append(flag)
            continue
        left = _without_simulation(flag, values)
        if left is not None:
            kept.extend((flag, *left))
    owned_env = {name: value for name, value in env.items() if not name.startswith(GOLDEN_OWNED_ENV_PREFIXES)}
    return normalize_argv(kept), owned_env


__all__ = [
    "GOLDEN_OWNED_ENV_PREFIXES",
    "GOLDEN_OWNED_FLAGS",
    "GOLDEN_OWNED_FLAG_PREFIXES",
    "GOLDEN_OWNED_SPEC_CONFIG_KEYS",
    "SIMULATED_SPEC_CONFIG_KEYS",
    "golden_owned_settings",
    "is_golden_owned_flag",
    "real_acceptance",
]
