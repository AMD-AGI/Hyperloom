# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a benchmarked candidate may change about a launch that simulates speculative-decoding acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.inference_optimizer.grid_server_args import (
    SPECULATIVE_CONFIG_FLAG,
    flag_occurrences,
    speculative_config,
)

SPECULATIVE_DECODING_PINNED = "speculative_decoding_pinned"

# A simulated acceptance length is only valid for the draft it was set for (InferenceX's golden AL is keyed by model,
# method and draft length), so what is drafted and how acceptance is simulated stay pinned; how fast the draft runs
# stays open.
_PINNED_FLAGS = frozenset(
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
_ACCEPTANCE_FLAG_PREFIXES = ("spec-decode-acceptance", "speculative-acceptance")
_PINNED_SPEC_CONFIG_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
        "rejection_sample_method",
        "synthetic_acceptance_length",
        "draft_sample_method",
    }
)
_ACCEPTANCE_ENV_PREFIX = "SGLANG_SIMULATE_ACC_"
_SIMULATED_LENGTH_ENV = "SGLANG_SIMULATE_ACC_LEN"
_SYNTHETIC_REJECTION = f"--{SPECULATIVE_CONFIG_FLAG}.rejection_sample_method"
_SYNTHETIC_LENGTH = f"--{SPECULATIVE_CONFIG_FLAG}.synthetic_acceptance_length"


class SpeculativeDecodingPinnedError(ValueError):
    """A candidate changes the draft or the simulated acceptance its base launch pins."""

    error_class = SPECULATIVE_DECODING_PINNED


def _canonical(value: Any) -> str:
    """``value`` as text, numbers spelled one way so ``3``, ``3.0`` and ``"3"`` (or ``3.61`` and ``3.610``) agree."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _speculative_identity(server_args: str, envs: Mapping[str, Any]) -> dict[str, str]:
    """The pinned settings a launch carries, keyed by flag, ``--speculative-config`` key or env name.

    vLLM also takes a ``--speculative-config.<key>`` per key, merged over the JSON object by
    :func:`speculative_config`, so both spellings land in the same keys.
    """
    identity = {str(k): _canonical(v) for k, v in envs.items() if str(k).startswith(_ACCEPTANCE_ENV_PREFIX)}
    occurrences = flag_occurrences(server_args)
    config = speculative_config(occurrences)
    for flag, value in dict(occurrences).items():
        if flag in _PINNED_FLAGS or flag.startswith(_ACCEPTANCE_FLAG_PREFIXES):
            identity[f"--{flag}"] = _canonical(value)
    identity.update(
        {
            f"--{SPECULATIVE_CONFIG_FLAG}.{key}": _canonical(config[key])
            for key in _PINNED_SPEC_CONFIG_KEYS & config.keys()
        }
    )
    return identity


def _simulates_acceptance(identity: Mapping[str, Any]) -> bool:
    return (
        bool(identity.get(_SIMULATED_LENGTH_ENV))
        or identity.get(_SYNTHETIC_REJECTION) == "synthetic"
        or _SYNTHETIC_LENGTH in identity
        or any(key.lstrip("-").startswith(_ACCEPTANCE_FLAG_PREFIXES) for key in identity)
    )


def _configures_a_draft(identity: Mapping[str, Any]) -> bool:
    return any(not key.startswith(_ACCEPTANCE_ENV_PREFIX) for key in identity)


def refuse_unpinned_speculative_change(
    base_args: str,
    base_envs: Mapping[str, Any],
    candidate_args: str,
    candidate_envs: Mapping[str, Any],
) -> None:
    """Refuse a candidate whose acceptance length is not comparable with its base launch's.

    A base that simulates acceptance pins its draft and simulated acceptance: both launches are judged as composed,
    so adding, changing, removing or unsetting a pinned setting, and a replacing argument string that drops one, all
    count as a change. A base that verifies a real draft leaves the draft open, but its acceptance is measured, so a
    candidate may not switch to a simulated length. A base with no speculative decoding has no acceptance to compare
    against, so a candidate may enable it either way.

    Raises:
        SpeculativeDecodingPinnedError: When the base simulates acceptance and the candidate's pinned settings differ,
            or the base verifies a real draft and the candidate simulates acceptance.
    """
    base = _speculative_identity(base_args, base_envs)
    candidate = _speculative_identity(candidate_args, candidate_envs)
    if not _simulates_acceptance(base):
        if _configures_a_draft(base) and _simulates_acceptance(candidate):
            raise SpeculativeDecodingPinnedError(
                f"{SPECULATIVE_DECODING_PINNED}: the base launch verifies a real draft, so its acceptance is measured; "
                "this candidate switches to a simulated acceptance length, which is declared, so its gain would not be "
                "comparable. Keep real draft verification and vary the draft or other knobs."
            )
        return
    changed = sorted(key for key in base.keys() | candidate.keys() if base.get(key) != candidate.get(key))
    if changed:
        changes = ", ".join(f"{key} ({base.get(key)!r} -> {candidate.get(key)!r})" for key in changed)
        raise SpeculativeDecodingPinnedError(
            f"{SPECULATIVE_DECODING_PINNED}: the base launch simulates speculative-decoding acceptance, which is "
            f"only valid for the draft it was set for, so the draft and its simulated acceptance are pinned; this "
            f"candidate changes {changes}. Keep them as the base sets them (a --{SPECULATIVE_CONFIG_FLAG} must restate "
            "its pinned keys) and vary other knobs."
        )


__all__ = [
    "SPECULATIVE_DECODING_PINNED",
    "SpeculativeDecodingPinnedError",
    "refuse_unpinned_speculative_change",
]
