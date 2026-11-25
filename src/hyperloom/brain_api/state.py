# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The search state a brain decides on, and the candidates it decides about.

A :class:`SearchState` is everything a call may read: the workload, where the search stands, what was tried and
what the evidence says. It is plain data, so a call recorded in production replays offline against a dataset that
knows what each :class:`Candidate` is worth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping

from hyperloom.brain_api.wire import WireError, json_mapping

ARGS_MODES = ("append", "replace")


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


@dataclass(frozen=True)
class Workload:
    """What is served, on what, under which load."""

    model_name: str
    framework: str
    framework_version: str = ""
    gpu_type: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    benchmark_mode: str = ""
    model_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.model_name and self.framework):
            raise WireError("Workload needs model_name and framework")
        _set(self, "model_info", json_mapping(self.model_info, "Workload.model_info"))


@dataclass(frozen=True)
class ConfigDelta:
    """A launch-config change: server args and env vars added, args and envs removed."""

    extra_server_args: str = ""
    extra_envs: Mapping[str, str] = field(default_factory=dict)
    remove_args: tuple[str, ...] = ()
    unset_envs: tuple[str, ...] = ()
    args_mode: str = "append"

    def __post_init__(self) -> None:
        envs = json_mapping(self.extra_envs, "ConfigDelta.extra_envs")
        if not all(isinstance(v, str) for v in envs.values()):
            raise WireError("ConfigDelta.extra_envs values must be strings")
        if self.args_mode not in ARGS_MODES:
            raise WireError(f"ConfigDelta.args_mode {self.args_mode!r} is not one of {ARGS_MODES}")
        _set(self, "extra_envs", envs)
        _set(self, "remove_args", tuple(self.remove_args))
        _set(self, "unset_envs", tuple(self.unset_envs))

    @property
    def empty(self) -> bool:
        """Whether it changes nothing."""
        return not (self.extra_server_args.strip() or self.extra_envs or self.remove_args or self.unset_envs)


@dataclass(frozen=True)
class PatchFile:
    """One source file a candidate changes, as a unified diff against the framework source root."""

    path: str
    diff: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise WireError(f"PatchFile.path {self.path!r} must be relative to the source root")
        if not self.diff.strip():
            raise WireError(f"PatchFile {self.path!r} has an empty diff")


@dataclass(frozen=True)
class Candidate:
    """One thing to measure: a config change, source patches, or both."""

    name: str
    config: ConfigDelta = field(default_factory=ConfigDelta)
    patches: tuple[PatchFile, ...] = ()
    reason: str = ""
    predicted_gain_pct: float | None = None
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _set(self, "patches", tuple(self.patches))
        _set(self, "citations", tuple(self.citations))
        if not self.name:
            raise WireError("Candidate needs a name")
        if self.config.empty and not self.patches:
            raise WireError(f"Candidate {self.name!r} changes nothing")


class Outcome(str, enum.Enum):
    """What the body made of a measured candidate."""

    KEEP = "keep"
    REVERT = "revert"
    FAILED = "failed"


@dataclass(frozen=True)
class Attempt:
    """A candidate the search measured, and the gain over the stack it was measured on."""

    candidate: Candidate
    outcome: Outcome
    gain_pct: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Gap:
    """A shortfall the search knows of, named by its canonical id."""

    canonical_id: str
    domain: str = ""
    severity: str = ""
    layer: str = ""
    symptom: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_id:
            raise WireError("Gap needs a canonical_id")
        _set(self, "evidence", json_mapping(self.evidence, "Gap.evidence"))


@dataclass(frozen=True)
class Evidence:
    """What the search has learned beyond measurements: the profile, the server log, prior knowledge."""

    roofline: Mapping[str, Any] = field(default_factory=dict)
    runtime_findings: str = ""
    experience: str = ""
    lessons: tuple[Mapping[str, Any], ...] = ()
    pitfalls: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _set(self, "roofline", json_mapping(self.roofline, "Evidence.roofline"))
        _set(self, "lessons", tuple(json_mapping(v, "Evidence.lessons") for v in self.lessons))
        _set(self, "pitfalls", tuple(json_mapping(v, "Evidence.pitfalls") for v in self.pitfalls))


@dataclass(frozen=True)
class SearchState:
    """Where the search stands when the body calls the brain.

    ``launch`` is the config the current best runs with; ``stack`` is what the search adopted on top of the
    baseline, in order; ``attempts`` is every measured candidate, adopted or not, oldest first.
    """

    workload: Workload
    baseline_tput: float
    current_tput: float
    keep_threshold_pct: float
    launch: ConfigDelta = field(default_factory=ConfigDelta)
    stack: tuple[Candidate, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    gaps: tuple[Gap, ...] = ()
    evidence: Evidence = field(default_factory=Evidence)

    def __post_init__(self) -> None:
        _set(self, "stack", tuple(self.stack))
        _set(self, "attempts", tuple(self.attempts))
        _set(self, "gaps", tuple(self.gaps))
        if self.baseline_tput < 0 or self.current_tput < 0:
            raise WireError("SearchState throughputs must not be negative")
