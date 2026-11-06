# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What recipe resolution reads and what each source answers with.

A source is read from a snapshot taken once at PRELUDE: a :class:`RepoBundle` holds the files of one repository at
one commit, and whatever needed upstream code to compute (InferenceX's variant selection and golden acceptance) is
recorded beside it as data, so resolving runs no external code. A source answers a query with the launches it holds
for it (:class:`SourceLaunch`), or with why it holds none (:class:`SourceMiss`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hyperloom_common.contract_fields import ContractFields
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from inference_suite.contracts.base_launch import GoldenCurve, LaunchSource, SourceKind, Topology


class RecipeSourceError(ValueError):
    """A snapshot breaks its own format or does not cover the query it is resolved for."""


_FIELDS = ContractFields(RecipeSourceError)

SGLANG_ENTRYPOINT = ("python3", "-m", "sglang.launch_server")
#: The sglang server flags the runner sets or allocates for every round (canonical names), as srtctl owns them.
SGLANG_RUNNER_OWNED = frozenset(
    {"model-path", "served-model-name", "host", "port", "nccl-port", "grpc-http-sidecar-port", "smg-http-sidecar-port"}
)


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


__all__ = [
    "InferencexSelection",
    "InferencexSnapshot",
    "RecipeSourceError",
    "RepoBundle",
    "SGLANG_ENTRYPOINT",
    "SGLANG_RUNNER_OWNED",
    "SourceGolden",
    "SourceLaunch",
    "SourceMiss",
    "boards_interchangeable",
]
