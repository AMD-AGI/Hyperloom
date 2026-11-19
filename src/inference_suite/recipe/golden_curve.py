# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The golden acceptance of a cookbook AgentX launch, read from InferenceX's curves as recorded data.

InferenceX pairs a golden acceptance length with its own AgentX points; a cookbook launch gets one from the same
curves. Which curve a launch's draft selects is InferenceX's code (``golden_length``), run once when the snapshot is
taken: :func:`golden_requests` names what it is asked and :class:`GoldenTable` records its answers, so pairing a
launch here runs no external code. The model prefix the curves are keyed by is the one InferenceX's master config
gives the model, else its HF base model (one level, no further). A launch with no curve, or whose vLLM adaptive
verification turns simulation off, runs with no golden acceptance and says why.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


from hyperloom_common.contract_fields import ContractFields
from inference_suite.argv import (
    SPECULATIVE_CONFIG_FLAG,
    argv_flag_occurrences,
    last_alias_value,
    speculative_config,
    without_speculative_config,
)
from inference_suite.contracts.base_launch import DEFAULT_THINKING_MODE, GoldenCurve, LaunchQuery
from inference_suite.golden import SIMULATED_FLAG_PREFIXES
from inference_suite.recipe.inferencex import GOLDEN_DIR, model_prefixes
from inference_suite.recipe.model import (
    RecipeSourceError,
    RepoBundle,
    SourceGolden,
    SourceLaunch,
    adaptive_verification_reason,
)

_FIELDS = ContractFields(RecipeSourceError)

# The values below are InferenceX's infx/srt_slurm/synthetic_acceptance.py: how it reads a draft from a launch
# (spec_parameters, verifier) and how it forces the golden length on one (build_overrides).
_SGLANG_ALGORITHM = frozenset({"speculative-algorithm", "speculative-algo"})
_SGLANG_SIMULATION = ("SGLANG_SIMULATE_ACC_LEN", "SGLANG_SIMULATE_ACC_METHOD", "SGLANG_SIMULATE_ACC_TOKEN_MODE")
_SIMULATION_METHOD = "match-expected"
_SIMULATION_TOKEN_MODE = "real-draft-token"
_SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class GoldenRequest:
    """One question to InferenceX's ``golden_length``: a model prefix, a draft config, a thinking mode, a verifier."""

    model_prefix: str
    spec: Mapping[str, Any]
    thinking: str
    verify: str | None

    def __post_init__(self) -> None:
        _FIELDS.text(self.model_prefix, "a golden request's model prefix")
        _FIELDS.mapping(self.spec, "a golden request's draft config")
        _FIELDS.text(self.thinking, "a golden request's thinking mode")
        _FIELDS.optional_text(self.verify, "a golden request's verifier")

    @property
    def key(self) -> str:
        """The request as canonical JSON, which :class:`GoldenTable` records its answer under."""
        return json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))

    def to_document(self) -> dict[str, object]:
        return {
            "model_prefix": self.model_prefix,
            "spec": dict(self.spec),
            "thinking": self.thinking,
            "verify": self.verify,
        }


@dataclass(frozen=True)
class GoldenLength:
    """``golden_length``'s answer: the length and the curve (file stem) it was read from, or the error it raised."""

    value: float | None
    curve: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.error is None):
            raise RecipeSourceError("a golden length is a value or the error that replaced it, not both")
        if (self.value is None) != (self.curve is None):
            raise RecipeSourceError("a golden length names the curve it was read from, and only then")
        if self.value is not None and _FIELDS.number(self.value, "a golden length") <= 0:
            raise RecipeSourceError(f"a golden length must be positive, got {self.value!r}")
        _FIELDS.optional_text(self.curve, "a golden length's curve")
        _FIELDS.optional_text(self.error, "a golden length's error")

    def to_document(self) -> dict[str, object]:
        return {"value": self.value, "curve": self.curve, "error": self.error}

    @classmethod
    def from_document(cls, document: object) -> GoldenLength:
        fields = _FIELDS.known(document, ("value", "curve", "error"), "a golden length")
        return cls(fields.get("value"), fields.get("curve"), fields.get("error"))


@dataclass(frozen=True)
class GoldenTable:
    """InferenceX at one commit and ``golden_length``'s answers there, keyed by :attr:`GoldenRequest.key`."""

    bundle: RepoBundle
    lengths: Mapping[str, GoldenLength]

    def __post_init__(self) -> None:
        _FIELDS.mapping(self.lengths, "the recorded golden lengths")

    def to_document(self) -> dict[str, object]:
        return {
            "bundle": self.bundle.to_document(),
            "lengths": {key: self.lengths[key].to_document() for key in sorted(self.lengths)},
        }

    @classmethod
    def from_document(cls, document: object) -> GoldenTable:
        fields = _FIELDS.known(document, ("bundle", "lengths"), "a golden table")
        lengths = _FIELDS.mapping(fields.get("lengths", {}), "the recorded golden lengths")
        return cls(
            RepoBundle.from_document(fields.get("bundle")),
            {key: GoldenLength.from_document(value) for key, value in lengths.items()},
        )


def _model_prefix(bundle: RepoBundle, model_id: str, base_model: str | None) -> str | SourceGolden:
    for model in dict.fromkeys(name for name in (model_id, base_model) if name):
        prefixes = model_prefixes(bundle, model)
        if len(prefixes) > 1:
            return SourceGolden(None, reason=f"InferenceX gives {model} several model prefixes: {prefixes}")
        if prefixes:
            return prefixes[0]
    named = f"{model_id} or its base model {base_model}" if base_model else model_id
    return SourceGolden(None, reason=f"InferenceX names no model prefix for {named}, so no golden curve is keyed to it")


def _count(value: str | None) -> int | str | None:
    return int(value) if value is not None and value.isdigit() else value


@dataclass(frozen=True)
class LaunchDraft:
    """A launch's draft as InferenceX keys a curve by it (``spec``, ``verify``), and the acceptance simulation the
    launch runs (``simulation``, ``token_mode``; ``None`` while it forces none)."""

    spec: Mapping[str, Any]
    verify: str | None
    simulation: str | None = None
    token_mode: str | None = None


def _sglang_draft(occurrences: list[tuple[str, str]], env: Mapping[str, str]) -> LaunchDraft:
    values = dict(occurrences)
    algorithm = (last_alias_value(occurrences, _SGLANG_ALGORITHM) or "").lower()
    if not algorithm:
        return LaunchDraft({}, "")
    model = values.get("speculative-draft-model-path", "")
    if algorithm == "dflash":
        width = _count(values.get("speculative-num-draft-tokens"))
        tokens = width - 1 if isinstance(width, int) else None
    else:
        steps = "speculative-dspark-block-size" if algorithm == "dspark" else "speculative-num-steps"
        tokens = _count(values.get(steps))
    verify = "block" if "speculative-use-block-verification" in values else "standard"
    spec = {"method": algorithm, "num_speculative_tokens": tokens, "model": model}
    return LaunchDraft(spec, verify, env.get(_SGLANG_SIMULATION[1]), env.get(_SGLANG_SIMULATION[2]))


def _vllm_draft(occurrences: list[tuple[str, str]], env: Mapping[str, str]) -> LaunchDraft:
    spec = speculative_config(occurrences)
    method = spec.get("rejection_sample_method")
    verify = "block" if method == _SYNTHETIC else str(method or "standard")
    # vLLM's synthetic sampler streams the drafted tokens it accepts, as sglang's real-draft-token mode does.
    return LaunchDraft(spec, verify, method, _SIMULATION_TOKEN_MODE)


def _atom_draft(occurrences: list[tuple[str, str]], env: Mapping[str, str]) -> LaunchDraft:
    values = dict(occurrences)
    method = values.get("method", "")
    if not method:
        return LaunchDraft({}, None)
    # ATOM drafts one token when the launch names no count. InferenceX keys no ATOM curve by its verifier.
    tokens = _count(values.get("num-speculative-tokens")) or 1
    spec = {"method": method, "num_speculative_tokens": tokens, "model": values.get("draft-model", "")}
    # ATOM forces acceptance with one flag, by length or by rate, and the flag that does is the simulation. Its
    # accepted draft tokens are streamed, as sglang's real-draft-token mode streams them.
    forced = next((name for name, _ in reversed(occurrences) if name.startswith(SIMULATED_FLAG_PREFIXES)), None)
    return LaunchDraft(spec, None, forced, _SIMULATION_TOKEN_MODE)


def _force_sglang(launch: SourceLaunch, request: GoldenRequest, value: float) -> SourceLaunch:
    settings = (f"{value:g}", _SIMULATION_METHOD, _SIMULATION_TOKEN_MODE)
    return dataclasses.replace(launch, env={**launch.env, **dict(zip(_SGLANG_SIMULATION, settings, strict=True))})


def _force_vllm(launch: SourceLaunch, request: GoldenRequest, value: float) -> SourceLaunch:
    spec = {**request.spec, "rejection_sample_method": _SYNTHETIC, "synthetic_acceptance_length": value}
    args = (*without_speculative_config(launch.args), f"--{SPECULATIVE_CONFIG_FLAG}", json.dumps(spec))
    return dataclasses.replace(launch, args=args)


@dataclass(frozen=True)
class _GoldenSpelling:
    """How one framework's launch reads its draft, and how a cookbook launch of it is made to force its golden
    acceptance length as InferenceX's ``build_overrides`` forces it (``None``: its cookbook serves no AgentX)."""

    read: Callable[[list[tuple[str, str]], Mapping[str, str]], LaunchDraft]
    force: Callable[[SourceLaunch, GoldenRequest, float], SourceLaunch] | None = None


#: ATOM's catalog serves fixed workloads only, so no ATOM cookbook launch is forced.
_SPELLINGS: Mapping[str, _GoldenSpelling] = MappingProxyType(
    {
        "sglang": _GoldenSpelling(_sglang_draft, _force_sglang),
        "vllm": _GoldenSpelling(_vllm_draft, _force_vllm),
        "atom": _GoldenSpelling(_atom_draft),
    }
)


def _spelling(framework: str) -> _GoldenSpelling:
    spelling = _SPELLINGS.get(framework)
    if spelling is None:
        raise RecipeSourceError(f"a {framework} launch drafts nothing InferenceX keys a golden curve by")
    return spelling


def launch_draft(framework: str, args: Sequence[str], env: Mapping[str, str]) -> LaunchDraft:
    """The draft a ``framework`` launch of ``args`` and ``env`` runs, and the acceptance simulation it forces."""
    return _spelling(framework).read(argv_flag_occurrences(args), env)


def _force(framework: str) -> Callable[[SourceLaunch, GoldenRequest, float], SourceLaunch]:
    force = _spelling(framework).force
    if force is None:
        raise RecipeSourceError(f"no golden acceptance is forced on a {framework} cookbook launch")
    return force


def _golden_plan(
    launch: SourceLaunch, query: LaunchQuery, bundle: RepoBundle, base_model: str | None
) -> GoldenRequest | SourceGolden | None:
    """What ``golden_length`` is asked for ``launch``; ``None`` when it drafts nothing, else why it is not asked."""
    _force(launch.framework)
    draft = launch_draft(launch.framework, launch.args, launch.env)
    if not draft.spec:
        return None
    adaptive = adaptive_verification_reason(draft.spec)
    if adaptive is not None:
        return SourceGolden(None, reason=adaptive)
    prefix = _model_prefix(bundle, query.model_id, base_model)
    if isinstance(prefix, SourceGolden):
        return prefix
    return GoldenRequest(prefix, draft.spec, DEFAULT_THINKING_MODE, draft.verify)


def golden_requests(
    query: LaunchQuery, launches: Iterable[SourceLaunch], bundle: RepoBundle, base_model: str | None
) -> tuple[GoldenRequest, ...]:
    """The questions to ask ``golden_length`` at ``bundle`` so :func:`with_golden` can pair each of ``launches``.

    ``base_model`` is the model's HF ``cardData.base_model``, if it names one.
    """
    plans = (_golden_plan(launch, query, bundle, base_model) for launch in launches)
    requests = {plan.key: plan for plan in plans if isinstance(plan, GoldenRequest)}
    return tuple(requests.values())


def with_golden(launch: SourceLaunch, query: LaunchQuery, table: GoldenTable, base_model: str | None) -> SourceLaunch:
    """A cookbook AgentX ``launch`` paired with its golden acceptance and forcing it, or with why it has none.

    A launch that drafts nothing is returned as it is. ``base_model`` is as for :func:`golden_requests`.
    """
    plan = _golden_plan(launch, query, table.bundle, base_model)
    if plan is None:
        return launch
    if isinstance(plan, SourceGolden):
        return dataclasses.replace(launch, golden=plan)
    answer = table.lengths.get(plan.key)
    if answer is None:
        raise RecipeSourceError(f"the golden table at {table.bundle.commit} records no answer to {plan.key}")
    if answer.value is None or answer.curve is None:
        return dataclasses.replace(
            launch, golden=SourceGolden(None, reason=f"InferenceX has no golden curve for it: {answer.error}")
        )
    path = f"{GOLDEN_DIR}/{answer.curve}.yaml"
    curve = GoldenCurve(repo=table.bundle.repo, commit=table.bundle.commit, path=path, sha256=table.bundle.sha256(path))
    forced = _force(launch.framework)(launch, plan, answer.value)
    return dataclasses.replace(forced, golden=SourceGolden(answer.value, curve=curve))


__all__ = [
    "GoldenLength",
    "GoldenRequest",
    "GoldenTable",
    "LaunchDraft",
    "golden_requests",
    "launch_draft",
    "with_golden",
]
