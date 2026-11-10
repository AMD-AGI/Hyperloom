# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a benchmarked candidate may change about a launch that simulates speculative-decoding acceptance.

The simulated acceptance is the golden layer of the launch (D10): a candidate that touches a key golden owns is a
candidate ``serve_flag`` failure naming that key, as a flag the parser refuses is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_suite.argv import SPECULATIVE_CONFIG_FLAG, flag_occurrences
from inference_suite.golden import (
    GOLDEN_OWNED_ENV_PREFIXES,
    GOLDEN_OWNED_FLAG_PREFIXES,
    golden_owned_settings,
)

from hyperloom.common.failure_signature import SERVE_FLAG

_SIMULATED_LENGTH_ENV = "SGLANG_SIMULATE_ACC_LEN"
_SYNTHETIC_REJECTION = f"--{SPECULATIVE_CONFIG_FLAG}.rejection_sample_method"
_SYNTHETIC_LENGTH = f"--{SPECULATIVE_CONFIG_FLAG}.synthetic_acceptance_length"


class SpeculativeDecodingPinnedError(ValueError):
    """A candidate changes the draft or the simulated acceptance its base launch pins."""

    error_class = SERVE_FLAG


def _canonical(value: Any) -> str:
    """``value`` as text, numbers spelled one way so ``3``, ``3.0`` and ``"3"`` (or ``3.61`` and ``3.610``) agree."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else repr(number)


def _speculative_identity(server_args: str, envs: Mapping[str, Any]) -> dict[str, str]:
    """The pinned settings a launch carries, canonicalized so equal values compare equal."""
    settings = golden_owned_settings(flag_occurrences(server_args), envs)
    return {key: _canonical(value) for key, value in settings.items()}


def _simulates_acceptance(identity: Mapping[str, Any]) -> bool:
    return (
        bool(identity.get(_SIMULATED_LENGTH_ENV))
        or identity.get(_SYNTHETIC_REJECTION) == "synthetic"
        or _SYNTHETIC_LENGTH in identity
        or any(key.lstrip("-").startswith(GOLDEN_OWNED_FLAG_PREFIXES) for key in identity)
    )


def _configures_a_draft(identity: Mapping[str, Any]) -> bool:
    return any(not key.startswith(GOLDEN_OWNED_ENV_PREFIXES) for key in identity)


def speculative_pin_reason(
    base_args: str,
    base_envs: Mapping[str, Any],
    candidate_args: str,
    candidate_envs: Mapping[str, Any],
) -> str | None:
    """Why this candidate's acceptance length is not comparable with its base launch's; ``None`` when it is.

    A base that simulates acceptance pins its draft and simulated acceptance: both launches are judged as composed,
    so adding, changing, removing or unsetting a pinned setting, and a replacing argument string that drops one, all
    count as a change. A base that verifies a real draft leaves the draft open, but its acceptance is measured, so a
    candidate may not switch to a simulated length. A base with no speculative decoding has no acceptance to compare
    against, so a candidate may enable it either way.

    This is the rule itself, so a layer that admits candidates can ask it of a proposal before a round spends a slot
    on it; :func:`refuse_unpinned_speculative_change` is the same rule stated as a refusal, for a launch being built.
    """
    base = _speculative_identity(base_args, base_envs)
    candidate = _speculative_identity(candidate_args, candidate_envs)
    changed = sorted(key for key in base.keys() | candidate.keys() if base.get(key) != candidate.get(key))
    if not _simulates_acceptance(base):
        if _configures_a_draft(base) and _simulates_acceptance(candidate):
            return (
                f"{SERVE_FLAG}: golden-owned {', '.join(changed)}: the base launch verifies a real draft, so its "
                "acceptance is measured; this candidate switches to a simulated acceptance length, which is declared, "
                "so its gain would not be comparable. Keep real draft verification and vary the draft or other knobs."
            )
        return None
    if not changed:
        return None
    changes = ", ".join(f"{key} ({base.get(key)!r} -> {candidate.get(key)!r})" for key in changed)
    return (
        f"{SERVE_FLAG}: golden-owned {', '.join(changed)}: the base launch simulates speculative-decoding "
        "acceptance, which is only valid for the draft it was set for, so the draft and its simulated acceptance "
        f"are pinned; this candidate changes {changes}. Keep them as the base sets them "
        f"(a --{SPECULATIVE_CONFIG_FLAG} must restate its pinned keys) and vary other knobs."
    )


def refuse_unpinned_speculative_change(
    base_args: str,
    base_envs: Mapping[str, Any],
    candidate_args: str,
    candidate_envs: Mapping[str, Any],
) -> None:
    """Refuse a candidate whose acceptance length is not comparable with its base launch's.

    Raises:
        SpeculativeDecodingPinnedError: When the base simulates acceptance and the candidate's pinned settings differ,
            or the base verifies a real draft and the candidate simulates acceptance.
    """
    reason = speculative_pin_reason(base_args, base_envs, candidate_args, candidate_envs)
    if reason:
        raise SpeculativeDecodingPinnedError(reason)


__all__ = [
    "SpeculativeDecodingPinnedError",
    "refuse_unpinned_speculative_change",
    "speculative_pin_reason",
]
