# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What recipe resolution reads and what each source answers with.

A source is read from a snapshot taken once at PRELUDE: a :class:`RepoBundle` holds the files of one repository at
one commit, an :class:`ApiResponse` one HTTP response of a site that has no commit, and whatever needed upstream code
to compute (InferenceX's variant selection and golden acceptance) is recorded beside them as data, so resolving runs
no external code. A source answers a query with the launches it holds
for it (:class:`SourceLaunch`), or with why it holds none (:class:`SourceMiss`).

The per-framework facts every source and the resolver share live here as well: the serving entrypoint, the flags the
runner owns, and how the framework spells its topology (:class:`TopologySpelling`), read and written in one place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyperloom_common.contract_fields import ContractFields
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value
from inference_suite.contracts.base_launch import GoldenCurve, LaunchSource, SourceApi, SourceKind, Topology


class RecipeSourceError(ValueError):
    """A snapshot breaks its own format or does not cover the query it is resolved for."""


_FIELDS = ContractFields(RecipeSourceError)


def _size(value: str | None, what: str) -> int:
    try:
        return int(value or 1)
    except ValueError as error:
        raise RecipeSourceError(f"its {what} {value!r} is not a size") from error


@dataclass(frozen=True)
class TopologySpelling:
    """How a framework spells its topology: ``tp`` and a sized ``ep`` flag, or an ``ep_switch`` that spreads the
    experts over every rank (``tp`` times the ``dp`` replicas). The first spelling of each is the one written."""

    tp: tuple[str, ...]
    ep: tuple[str, ...] = ()
    ep_switch: tuple[str, ...] = ()
    dp: tuple[str, ...] = ()

    def read(self, args: Sequence[str]) -> Topology:
        """The topology ``args`` run, an unset size being 1."""
        occurrences = argv_flag_occurrences(args)

        def value(spellings: tuple[str, ...]) -> str | None:
            return last_alias_value(occurrences, frozenset(map(canonical_flag_name, spellings)))

        tp = _size(value(self.tp), "tp")
        if self.ep:
            return Topology(tp=tp, ep=_size(value(self.ep), "ep"))
        if value(self.ep_switch) is None:
            return Topology(tp=tp)
        return Topology(tp=tp, ep=tp * _size(value(self.dp), "dp"))

    def argv(self, topology: Topology) -> tuple[str, ...] | str:
        """The flags that run ``topology`` on one replica, or why the framework cannot."""
        args = (self.tp[0], str(topology.tp))
        if topology.ep == 1:
            return args
        if self.ep:
            return (*args, self.ep[0], str(topology.ep))
        if topology.ep != topology.tp:
            return f"its experts span all {topology.tp} TP ranks, so it runs no ep {topology.ep}"
        return (*args, self.ep_switch[0])


SGLANG_ENTRYPOINT = ("python3", "-m", "sglang.launch_server")
#: The sglang server flags the runner sets or allocates for every round (canonical names), as srtctl owns them.
SGLANG_RUNNER_OWNED = frozenset(
    {"model-path", "served-model-name", "host", "port", "nccl-port", "grpc-http-sidecar-port", "smg-http-sidecar-port"}
)
SGLANG_TOPOLOGY = TopologySpelling(
    tp=("--tensor-parallel-size", "--tp-size", "--tp"), ep=("--ep-size", "--expert-parallel-size", "--ep")
)

VLLM_ENTRYPOINT = ("vllm", "serve")
#: The vLLM server flags the runner sets or allocates for every round (canonical names), as srtctl owns them.
VLLM_RUNNER_OWNED = frozenset(
    {"model", "served-model-name", "host", "port", "headless", "master-addr", "nnodes", "node-rank"}
)
VLLM_TOPOLOGY = TopologySpelling(
    tp=("--tensor-parallel-size", "-tp"),
    ep_switch=("--enable-expert-parallel", "-ep"),
    dp=("--data-parallel-size", "-dp"),
)


ATOM_ENTRYPOINT = ("python3", "-m", "atom.entrypoints.openai_server")
#: The ATOM server flags the runner sets or allocates for every round (canonical names), as srtctl owns them.
ATOM_RUNNER_OWNED = frozenset({"model", "host", "server-port"})
ATOM_TOPOLOGY = TopologySpelling(tp=("-tp", "--tensor-parallel-size"), ep_switch=("--enable-expert-parallel",))


def boards_interchangeable(board: str, other: str) -> bool:
    """Whether a launch tuned on ``other`` runs as tuned on ``board``: the same architecture and HBM capacity."""
    first, second = AMD_GPU_BOARDS.get(board), AMD_GPU_BOARDS.get(other)
    if first is None or second is None:
        return board == other
    return (first.gfx, first.hbm_gb) == (second.gfx, second.hbm_gb)


@dataclass(frozen=True)
class RepoBundle:
    """The files of one repository at one commit that a source reads, keyed by their path in the repository."""

    repo: str
    commit: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        _FIELDS.text(self.repo, "a bundle's repository")
        _FIELDS.text(self.commit, "a bundle's commit")
        _FIELDS.string_map(self.files, "a bundle's files")

    def text(self, path: str) -> str:
        if path not in self.files:
            raise RecipeSourceError(f"the {self.repo} snapshot at {self.commit} does not carry {path}")
        return self.files[path]

    def sha256(self, path: str) -> str:
        return hashlib.sha256(self.text(path).encode()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return {"repo": self.repo, "commit": self.commit, "files": dict(sorted(self.files.items()))}

    @classmethod
    def from_document(cls, document: object) -> RepoBundle:
        fields = _FIELDS.known(document, ("repo", "commit", "files"), "a repository bundle")
        return cls(fields.get("repo"), fields.get("commit"), fields.get("files"))


@dataclass(frozen=True)
class ApiResponse:
    """One HTTP response a source reads: its status and body as fetched from ``url`` at ``fetched_unix``."""

    url: str
    status: int
    body: str
    fetched_unix: int

    def __post_init__(self) -> None:
        _FIELDS.text(self.url, "a response's url")
        _FIELDS.integer(self.status, "a response's status", minimum=100)
        if not isinstance(self.body, str):
            raise RecipeSourceError(f"a response's body must be text, got {type(self.body).__name__}")
        _FIELDS.integer(self.fetched_unix, "a response's fetch time", minimum=0)

    def provenance(self) -> SourceApi:
        return SourceApi(self.url, hashlib.sha256(self.body.encode()).hexdigest(), self.fetched_unix)

    def json(self) -> Mapping[str, Any]:
        """The body as a JSON object."""
        try:
            document = json.loads(self.body)
        except ValueError as error:
            raise RecipeSourceError(f"{self.url} answered with no JSON: {error}") from error
        return _FIELDS.mapping(document, f"the response of {self.url}")

    def to_document(self) -> dict[str, object]:
        return {"url": self.url, "status": self.status, "body": self.body, "fetched_unix": self.fetched_unix}

    @classmethod
    def from_document(cls, document: object) -> ApiResponse:
        fields = _FIELDS.known(document, ("url", "status", "body", "fetched_unix"), "an API response")
        return cls(fields.get("url"), fields.get("status"), fields.get("body"), fields.get("fetched_unix"))


@dataclass(frozen=True)
class InferencexSelection:
    """What InferenceX's own code answered for one matrix point, recorded so resolving need not run it.

    ``variant`` and ``recipe`` are ``select_recipe``'s answer: the one variant the point binds and that variant bound
    to the point. ``files`` holds the sha256 of each recipe file the variant was composed from.
    ``acceptance_length`` and ``golden_overrides`` are ``acceptance_length`` and ``build_overrides`` on that recipe,
    and ``golden_curve`` the committed curve the length was read from.
    """

    variant: str
    recipe: Mapping[str, Any]
    files: Mapping[str, str]
    acceptance_length: float | None = None
    golden_overrides: tuple[str, ...] = ()
    golden_curve: str | None = None

    def __post_init__(self) -> None:
        _FIELDS.text(self.variant, "a selection's variant")
        _FIELDS.mapping(self.recipe, "a selection's recipe")
        for path, sha256 in _FIELDS.string_map(self.files, "a selection's files").items():
            _FIELDS.sha256(sha256, f"the sha256 of recipe file {path}")
        _FIELDS.optional_number(self.acceptance_length, "a selection's acceptance length")
        _FIELDS.string_tuple(self.golden_overrides, "a selection's golden overrides")
        _FIELDS.optional_text(self.golden_curve, "a selection's golden curve")
        if (self.acceptance_length is None) != (self.golden_curve is None):
            raise RecipeSourceError("a recorded acceptance length names the curve it was read from, and only then")

    def to_document(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "recipe": self.recipe,
            "files": dict(sorted(self.files.items())),
            "acceptance_length": self.acceptance_length,
            "golden_overrides": list(self.golden_overrides),
            "golden_curve": self.golden_curve,
        }

    @classmethod
    def from_document(cls, document: object) -> InferencexSelection:
        keys = ("variant", "recipe", "files", "acceptance_length", "golden_overrides", "golden_curve")
        fields = _FIELDS.known(document, keys, "an InferenceX selection")
        return cls(
            variant=fields.get("variant"),
            recipe=fields.get("recipe"),
            files=fields.get("files"),
            acceptance_length=fields.get("acceptance_length"),
            golden_overrides=_FIELDS.strings(fields.get("golden_overrides", ()), "a selection's golden overrides"),
            golden_curve=fields.get("golden_curve"),
        )


@dataclass(frozen=True)
class InferencexSnapshot:
    """InferenceX at one commit: its master and runner configs, and the selections recorded for matrix points.

    ``selections`` and ``rejections`` are keyed by :attr:`~inference_suite.recipe.inferencex.InferencexPoint.key`; a
    rejection is the error InferenceX's selection raised for its point. ``srt_slurm_commit`` pins the srtctl whose
    argv rules turn a selected recipe into a command line.
    """

    bundle: RepoBundle
    srt_slurm_commit: str
    selections: Mapping[str, InferencexSelection] = field(default_factory=dict)
    rejections: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _FIELDS.text(self.srt_slurm_commit, "the srt-slurm commit")
        _FIELDS.string_map(self.rejections, "the InferenceX rejections")
        recorded_twice = sorted(set(self.selections) & set(self.rejections))
        if recorded_twice:
            raise RecipeSourceError(f"points both selected and rejected: {', '.join(recorded_twice)}")

    def to_document(self) -> dict[str, object]:
        return {
            "bundle": self.bundle.to_document(),
            "srt_slurm_commit": self.srt_slurm_commit,
            "selections": {key: self.selections[key].to_document() for key in sorted(self.selections)},
            "rejections": dict(sorted(self.rejections.items())),
        }

    @classmethod
    def from_document(cls, document: object) -> InferencexSnapshot:
        keys = ("bundle", "srt_slurm_commit", "selections", "rejections")
        fields = _FIELDS.known(document, keys, "an InferenceX snapshot")
        selections = _FIELDS.mapping(fields.get("selections", {}), "the InferenceX selections")
        return cls(
            bundle=RepoBundle.from_document(fields.get("bundle")),
            srt_slurm_commit=fields.get("srt_slurm_commit"),
            selections={key: InferencexSelection.from_document(value) for key, value in selections.items()},
            rejections=fields.get("rejections", {}),
        )


@dataclass(frozen=True)
class SourceGolden:
    """The golden acceptance a source pairs with an AgentX launch, or why it has none (``value`` is ``None``)."""

    value: float | None
    curve: GoldenCurve | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise RecipeSourceError("a golden acceptance carries a value or the reason it has none, not both")
        if (self.value is None) != (self.curve is None):
            raise RecipeSourceError("a golden acceptance value names the curve it was read from, and only then")


@dataclass(frozen=True)
class SourceLaunch:
    """One launch a source holds for a query, as the source would run it.

    ``conc`` is the source point's concurrency and ``exact`` whether that is the one the query asked for. ``args``
    and ``env`` carry no runner-owned keys; ``stripped`` names what was taken out for that reason. ``client_env`` is
    the load generator's environment the source pairs with the launch, the point's own workload settings left out.
    """

    source: LaunchSource
    framework: str
    entrypoint: tuple[str, ...]
    topology: Topology
    conc: int
    exact: bool
    image: str | None
    args: tuple[str, ...]
    env: Mapping[str, str]
    client_env: Mapping[str, str] = field(default_factory=dict)
    golden: SourceGolden | None = None
    kv_offloading: str = "none"
    stripped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceMiss:
    """Why a source holds no launch for a query."""

    kind: SourceKind
    reason: str


def adaptive_verification_reason(spec: Mapping[str, Any]) -> str | None:
    """Why a vLLM draft ``spec`` gets no golden acceptance when its adaptive verification is on, else ``None``.

    A per-key flag carries the value as text and the JSON object as a bool; vLLM parses both to the same bool.
    """
    if str(spec.get("enable_adaptive_verification")).lower() != "true":
        return None
    return "vLLM adaptive verification is on, and InferenceX simulates no acceptance under it"


__all__ = [
    "ATOM_ENTRYPOINT",
    "ATOM_RUNNER_OWNED",
    "ATOM_TOPOLOGY",
    "ApiResponse",
    "InferencexSelection",
    "InferencexSnapshot",
    "RecipeSourceError",
    "RepoBundle",
    "SGLANG_ENTRYPOINT",
    "SGLANG_RUNNER_OWNED",
    "SGLANG_TOPOLOGY",
    "SourceGolden",
    "SourceLaunch",
    "SourceMiss",
    "TopologySpelling",
    "VLLM_ENTRYPOINT",
    "VLLM_RUNNER_OWNED",
    "VLLM_TOPOLOGY",
    "adaptive_verification_reason",
    "boards_interchangeable",
]
