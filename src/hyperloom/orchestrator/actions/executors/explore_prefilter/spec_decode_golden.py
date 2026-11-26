# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Grade speculative decoding a candidate turns on at the acceptance InferenceX records for its draft (synthetic only).

The synthetic benchmark sends random-token prompts at temperature 0 and holds every request to its full output length
(``ignore_eos``). Most outputs become repetition loops there, and a model-based draft (MTP, EAGLE, a draft model)
predicts them almost as well as a copying one: Qwen3.8-27B with 3 MTP draft tokens accepted 89-90% of its draft tokens
on that benchmark against 63-65% under the model's own sampling, and a gain reported at +82.9% was about +40-45%.
InferenceX grades speculative decoding at an acceptance length measured beforehand on real prompts with the model's
sampling, forced on the engine; this holds a candidate to the same.

A candidate that turns speculative decoding on over a base that runs none is refused before it runs unless it forces
the acceptance length InferenceX records for its own draft (:func:`inference_suite.candidate_golden`): a real draft is
told the value to force, a forced value that differs from it is told the recorded one, and a draft InferenceX records
nothing for is refused, since its gain cannot be graded on this benchmark until the model's acceptance is measured.
A context-copying draft is refused by :mod:`.._spec_decode_context_copy`, a base that simulates acceptance pins its draft
(:mod:`.._spec_decode_pin`), and a base that verifies a real draft is the operator's choice. AgentX sessions benchmark
real traces and are out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_suite import GoldenLookupError, SourceGolden, candidate_golden
from inference_suite.argv import tokenize_server_args_preserving_json

from ....state.base_launch import frozen_base
from ..framework_change._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_method, context_copy_refusal
from ..framework_change._spec_decode_pin import (
    canonical_value,
    configures_a_draft,
    simulates_acceptance,
    speculative_identity,
)
from .._suite.rounds import staged_tool_cache

__all__ = [
    "GOLDEN_ACCEPTANCE_REFUSAL",
    "GoldenLookup",
    "golden_acceptance_refusal",
    "session_golden_lookup",
    "synthetic_draft_refusal",
]

#: Reason a refused candidate is recorded with.
GOLDEN_ACCEPTANCE_REFUSAL = "speculative_acceptance_not_golden_on_synthetic"


@dataclass(frozen=True)
class GoldenLookup:
    """The framework a session serves on, and the golden acceptance InferenceX records for the draft a launch's argv
    tokens configure on it (``None`` when they draft nothing), the model bound by the caller."""

    framework: str
    answer: Callable[[Sequence[str]], SourceGolden | None]


#: Each engine's forced acceptance length, and what else must hold for it to be forced as InferenceX forces it: vLLM
#: reads ``synthetic_acceptance_length`` only under its synthetic rejection sampler, and ignores SGLang's environment;
#: SGLang holds the mean length only with its ``match-expected`` method (``multinomial`` rounds a normal draw), and
#: InferenceX streams the drafted tokens (``real-draft-token``) rather than SGLang's default ``fixed`` one.
_FORCED_LENGTH: Mapping[str, tuple[str, Mapping[str, str]]] = {
    "vllm": (
        "--speculative-config.synthetic_acceptance_length",
        {"--speculative-config.rejection_sample_method": "synthetic"},
    ),
    "sglang": (
        "SGLANG_SIMULATE_ACC_LEN",
        {"SGLANG_SIMULATE_ACC_METHOD": "match-expected", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
    ),
    "atom": ("--spec-decode-acceptance-length", {}),
}
#: What an engine reads a setting as when a launch leaves it unset.
_UNSET_READS_AS = {"SGLANG_SIMULATE_ACC_METHOD": "match-expected"}
_WHY = (
    "on the synthetic benchmark (random prompts at temperature 0, held to their full output length) the model falls "
    "into repetition loops that a draft predicts almost perfectly, so a measured acceptance does not carry over to "
    "real text"
)


def _forced_length(framework: str, identity: Mapping[str, str]) -> str | None:
    """The acceptance length ``identity`` forces on ``framework``, ``None`` when that engine forces none."""
    if framework not in _FORCED_LENGTH:
        return None
    key, required = _FORCED_LENGTH[framework]
    held = all(identity.get(name, _UNSET_READS_AS.get(name)) == value for name, value in required.items())
    return identity.get(key) if held else None


def _how_to_force(value: float) -> str:
    return (
        f'force it: on vLLM, "rejection_sample_method": "synthetic" and "synthetic_acceptance_length": {value:g} in '
        f"--speculative-config; on SGLang, SGLANG_SIMULATE_ACC_LEN={value:g} with "
        "SGLANG_SIMULATE_ACC_METHOD=match-expected and SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token; on ATOM, "
        f"--spec-decode-acceptance-length {value:g}"
    )


def _graded(golden: SourceGolden, forced: str | None) -> str:
    """Why a draft forcing ``forced`` (``None``: forcing no length as InferenceX does) is not graded at ``golden``, or
    ``""``."""
    if golden.value is None:
        return (
            f"InferenceX records no acceptance length for this draft ({golden.reason}), and {_WHY}; its gain cannot "
            "be graded on this benchmark until the model's acceptance is measured (InferenceX's SPEED-Bench "
            "collection). Leave speculative decoding off."
        )
    source = golden.curve.path if golden.curve is not None else "InferenceX"
    if forced is None:
        return (
            f"this candidate does not force its draft's acceptance as InferenceX does (it verifies a real draft, or "
            f"simulates acceptance in another form), and {_WHY}. InferenceX records {golden.value:g} for it "
            f"({source}); {_how_to_force(golden.value)}."
        )
    if forced != canonical_value(golden.value):
        return (
            f"this candidate forces an acceptance length of {forced}, but InferenceX records {golden.value:g} for its "
            f"draft ({source}); the acceptance length is not a knob, so {_how_to_force(golden.value)}."
        )
    return ""


def golden_acceptance_refusal(
    base_argv: str, base_envs: Mapping[str, Any], argv: str, envs: Mapping[str, Any], lookup: GoldenLookup
) -> str:
    """Why a candidate's launch is refused on a synthetic benchmark, or ``""`` when it is not.

    Judged only when the base runs no speculative decoding and the candidate turns it on with a draft that does not
    copy from the context; both launches are the materialized ones.
    """
    base = speculative_identity(base_argv, base_envs)
    candidate = speculative_identity(argv, envs)
    if configures_a_draft(base) or simulates_acceptance(base) or not configures_a_draft(candidate):
        return ""
    if context_copy_method(argv, envs):
        return ""
    parsed = tokenize_server_args_preserving_json(argv)
    try:
        golden = lookup.answer(parsed[1] if parsed is not None else argv.split())
    except GoldenLookupError as exc:
        return f"the acceptance InferenceX records for this draft could not be read ({exc}), and {_WHY}."
    if golden is None:
        return ""
    return _graded(golden, _forced_length(lookup.framework, candidate))


def session_golden_lookup(session_dir: Path | None, framework: str) -> GoldenLookup:
    """The lookup a candidate of the session at ``session_dir`` is graded with: the model its frozen base was resolved
    for, on ``framework``, from the staged InferenceX. The base is read when a draft is first looked up."""

    def answer(argv: Sequence[str]) -> SourceGolden | None:
        frozen = None if session_dir is None else frozen_base(Path(session_dir))
        if frozen is None:
            raise GoldenLookupError(f"the session {session_dir} froze no base launch to name its model")
        return candidate_golden(staged_tool_cache(), framework, frozen.launch.query.model_id, argv)

    return GoldenLookup(framework, answer)


def synthetic_draft_refusal(
    base_argv: str, base_envs: Mapping[str, Any], argv: str, envs: Mapping[str, Any], lookup: GoldenLookup
) -> tuple[str, str] | None:
    """The reason code and detail a synthetic benchmark refuses a candidate's speculative decoding with, or ``None``."""
    copied = context_copy_refusal(base_argv, base_envs, argv, envs)
    if copied:
        return CONTEXT_COPY_REFUSAL, copied
    graded = golden_acceptance_refusal(base_argv, base_envs, argv, envs, lookup)
    return (GOLDEN_ACCEPTANCE_REFUSAL, graded) if graded else None
