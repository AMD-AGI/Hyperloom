# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Request and result models for framework/ref exploration."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.pr_monitor_urls import pr_monitor_base_url, pr_monitor_enabled


@dataclass(frozen=True)
class Baseline:
    """Baseline throughput/accuracy that thresholds compare against."""

    throughput: float
    accuracy: float | None = None
    completed: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Baseline":
        """Parse raw dict into Baseline; throughput must be > 0."""
        throughput = float(raw.get("throughput") or raw.get("output_throughput") or 0.0)
        if throughput <= 0:
            raise ValueError("baseline.throughput must be > 0")
        accuracy_raw = raw.get("accuracy")
        accuracy = float(accuracy_raw) if isinstance(accuracy_raw, (int, float)) else None
        return cls(
            throughput=throughput,
            accuracy=accuracy,
            completed=str(raw.get("completed") or ""),
        )


@dataclass(frozen=True)
class Thresholds:
    """Winner-gate thresholds: throughput ratio + accuracy drop."""

    min_throughput_ratio: float = 1.05
    max_accuracy_drop: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Thresholds":
        """Parse raw dict; missing keys fall back to defaults."""
        raw = raw or {}
        return cls(
            min_throughput_ratio=float(raw.get("min_throughput_ratio", 1.05)),
            max_accuracy_drop=float(raw.get("max_accuracy_drop", 0.05)),
        )


@dataclass(frozen=True)
class PRMonitorConfig:
    """Configuration for the pr_monitor service."""

    base_url: str
    timeout_sec: float = 10.0
    default_label: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PRMonitorConfig":
        """Parse pr_monitor block; base_url is mandatory."""
        base_url = str(raw.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("pr_monitor.base_url is required when pr_monitor block is set")
        timeout_raw = raw.get("timeout_sec", 10.0)
        label_raw = raw.get("default_label")
        return cls(
            base_url=base_url,
            timeout_sec=float(timeout_raw),
            default_label=(str(label_raw).strip() if isinstance(label_raw, str) and label_raw.strip() else None),
        )


@dataclass(frozen=True)
class CommandSpec:
    """A single build/benchmark/accuracy command for a candidate run."""

    command: str
    timeout_sec: int = 3600
    required: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CommandSpec":
        """Parse a command block; command text is mandatory."""
        command = str(raw.get("command") or "").strip()
        if not command:
            raise ValueError("command spec requires a non-empty command")
        return cls(
            command=command,
            timeout_sec=int(raw.get("timeout_sec", 3600)),
            required=bool(raw.get("required", True)),
        )


@dataclass(frozen=True)
class Candidate:
    """A single PR or git ref candidate (explicit, pr_monitor, or GitHub)."""

    ref: str
    repo: str
    source: str = "explicit"
    head_sha: str = ""
    title: str = ""
    labels: tuple[str, ...] = ()
    author: str = ""
    changed_files: tuple[str, ...] = ()
    updated_at: str = ""
    html_url: str = ""
    score: float = 0.0
    framework: str = ""
    model_class: str = ""
    gpu_type: str = ""
    precision: str = ""
    gap_canonical_id: str = ""
    gap_description: str = ""
    gap_keywords: tuple[str, ...] = ()
    prior_score: float = 0.0
    prior_rank: int = 0
    pr_kb_files_slug: str = ""

    @property
    def slug(self) -> str:
        """Filesystem-safe slug derived from ref (used for candidate_dir name)."""
        out = []
        for ch in self.ref.lower():
            if ch.isalnum():
                out.append(ch)
            elif ch in (".", "-", "_"):
                out.append(ch)
            else:
                out.append("-")
        slug = "".join(out).strip("-")
        return slug or "candidate"

    @property
    def pr_number(self) -> int | None:
        """Return the PR number when ref starts with ``PR:``; else None."""
        if not self.ref.startswith("PR:"):
            return None
        try:
            return int(self.ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return None


@dataclass(frozen=True)
class PrFilter:
    """Client-side filter applied to enumerated PR candidates."""

    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()
    require_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    since: str = ""
    until: str = ""
    max_changed_files: int = 0
    min_changed_files: int = 0

    @staticmethod
    def _as_tuple(raw: Any) -> tuple[str, ...]:
        """Coerce string/list/None into a clean tuple of non-empty strings."""
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw.strip(),) if raw.strip() else ()
        if isinstance(raw, (list, tuple)):
            return tuple(str(v).strip() for v in raw if str(v).strip())
        raise ValueError(f"pr_filter list field must be string or list, got {type(raw).__name__}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PrFilter":
        """Parse pr_filter block; missing keys fall back to empty defaults."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("pr_filter must be an object when present")
        return cls(
            include_paths=cls._as_tuple(raw.get("include_paths")),
            exclude_paths=cls._as_tuple(raw.get("exclude_paths")),
            require_labels=cls._as_tuple(raw.get("require_labels")),
            exclude_labels=cls._as_tuple(raw.get("exclude_labels")),
            authors=cls._as_tuple(raw.get("authors")),
            since=str(raw.get("since") or "").strip(),
            until=str(raw.get("until") or "").strip(),
            max_changed_files=int(raw.get("max_changed_files", 0) or 0),
            min_changed_files=int(raw.get("min_changed_files", 0) or 0),
        )

    @property
    def is_empty(self) -> bool:
        """True when no constraint is set (filter is a no-op)."""
        return (
            not self.include_paths
            and not self.exclude_paths
            and not self.require_labels
            and not self.exclude_labels
            and not self.authors
            and not self.since
            and not self.until
            and self.max_changed_files == 0
            and self.min_changed_files == 0
        )


_VALID_SEARCH_MODES = frozenset({"gbrain_pr_kb", "pr_monitor", "github"})
_VALID_PR_STATES = frozenset({"open", "merged", "closed", "all"})


def _parse_pr_states(raw: Any) -> tuple[str, ...]:
    """Coerce an optional ``pr_states`` field into a validated tuple."""
    if raw is None or raw == "":
        return ("open",)
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        items = [str(v).strip() for v in raw if str(v).strip()]
    else:
        raise ValueError(f"pr_states must be string or list, got {type(raw).__name__}")
    for item in items:
        if item not in _VALID_PR_STATES:
            raise ValueError(
                f"pr_states contains unknown state {item!r}; valid values are {sorted(_VALID_PR_STATES)!r}"
            )
    return tuple(items) or ("open",)


def _parse_keywords(raw: Any) -> tuple[str, ...]:
    """Coerce an optional ``keywords`` field into a tuple of trimmed strings."""
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        items = [tok.strip() for tok in raw.replace(",", " ").split() if tok.strip()]
        return tuple(items)
    if isinstance(raw, (list, tuple)):
        items = [str(v).strip() for v in raw if str(v).strip()]
        return tuple(items)
    raise ValueError(f"keywords must be list/tuple/str when present, got {type(raw).__name__}")


def _parse_search_modes(raw: Any) -> tuple[str, ...]:
    """Coerce a list of mode names; default to pr_monitor + GitHub."""
    if raw is None or raw == "":
        return ("pr_monitor", "github")
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        items = [str(v).strip() for v in raw if str(v).strip()]
    else:
        raise ValueError(f"search_modes must be string or list, got {type(raw).__name__}")
    for item in items:
        if item not in _VALID_SEARCH_MODES:
            raise ValueError(
                f"search_modes contains unknown source {item!r}; valid values are {sorted(_VALID_SEARCH_MODES)!r}"
            )
    return tuple(items)


@dataclass(frozen=True)
class ExploreRequest:
    """Top-level request for `fa explore` / `fa candidates`."""

    framework: str
    repo_url: str
    work_dir: Path
    baseline: Baseline
    thresholds: Thresholds = field(default_factory=Thresholds)
    candidate_refs: tuple[str, ...] = ()
    search_perf_prs: bool = False
    max_search_candidates: int = 5
    prepare_candidate_env: bool = True
    commands: dict[str, CommandSpec] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    # Explicit config wins; otherwise local mode derives the default KB endpoint.
    pr_monitor: PRMonitorConfig | None = None
    pr_filter: PrFilter = field(default_factory=PrFilter)
    gap_description: str = ""
    gap_canonical_id: str = ""
    model_class: str = ""
    gpu_type: str = ""
    precision: str = ""
    # Explicit keyword override; non-empty bypasses extract_keywords().
    keywords: tuple[str, ...] = ()
    search_modes: tuple[str, ...] = ("pr_monitor", "github")
    # PR states to include in discovery.
    pr_states: tuple[str, ...] = ("open",)
    # Empty string disables the KB contribute hook.
    kb_domain: str = ""
    # True: return all candidates sorted by score; False: short-circuit on first winner.
    ranking_mode: bool = False
    # True: remove worktree+venv of every non-winner at end of run.
    keep_winner_only: bool = False
    # > 0: build multiple candidates concurrently; <=1 => fully serial.
    build_concurrency: int = 1
    # Disk preflight threshold (GB); None -> env FRAMEWORK_EXPLORER_DISK_MIN_GB. Set 0 to bypass.
    disk_min_free_gb: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExploreRequest":
        """Parse a JSON request payload into ExploreRequest, validating required fields."""
        framework = str(raw.get("framework") or "").strip().lower()
        if not framework:
            raise ValueError("framework is required")
        repo_url = str(raw.get("repo_url") or "").strip()
        if not repo_url:
            raise ValueError("repo_url is required")
        work_dir = Path(str(raw.get("work_dir") or (Path(tempfile.gettempdir()) / "framework-agent"))).expanduser()
        baseline_raw = raw.get("baseline")
        if not isinstance(baseline_raw, dict):
            raise ValueError("baseline object is required")
        commands_raw = raw.get("commands") or {}
        if not isinstance(commands_raw, dict):
            raise ValueError("commands must be an object")
        commands = {
            str(name): CommandSpec.from_dict(spec) for name, spec in commands_raw.items() if isinstance(spec, dict)
        }
        outputs = raw.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise ValueError("outputs must be an object when present")
        pr_monitor_raw = raw.get("pr_monitor")
        if isinstance(pr_monitor_raw, dict):
            pr_monitor = PRMonitorConfig.from_dict(pr_monitor_raw)
        elif pr_monitor_raw is None:
            env_base_url = pr_monitor_base_url() if pr_monitor_enabled() else ""
            pr_monitor = PRMonitorConfig(base_url=env_base_url) if env_base_url else None
        else:
            raise ValueError("pr_monitor must be an object when present")
        return cls(
            framework=framework,
            repo_url=repo_url,
            work_dir=work_dir,
            baseline=Baseline.from_dict(baseline_raw),
            thresholds=Thresholds.from_dict(raw.get("thresholds")),
            candidate_refs=tuple(str(r).strip() for r in raw.get("candidate_refs") or () if str(r).strip()),
            search_perf_prs=bool(raw.get("search_perf_prs", False)),
            max_search_candidates=int(raw.get("max_search_candidates", 5)),
            prepare_candidate_env=bool(raw.get("prepare_candidate_env", True)),
            commands=commands,
            outputs={str(k): str(v) for k, v in outputs.items()},
            pr_monitor=pr_monitor,
            pr_filter=PrFilter.from_dict(raw.get("pr_filter")),
            gap_description=str(raw.get("gap_description") or "").strip(),
            gap_canonical_id=str(raw.get("gap_canonical_id") or "").strip(),
            model_class=str(raw.get("model_class") or raw.get("model") or "").strip(),
            gpu_type=str(raw.get("gpu_type") or "").strip(),
            precision=str(raw.get("precision") or "").strip(),
            keywords=_parse_keywords(raw.get("keywords")),
            search_modes=_parse_search_modes(raw.get("search_modes")),
            pr_states=_parse_pr_states(raw.get("pr_states")),
            kb_domain=str(raw.get("kb_domain") or "").strip(),
            ranking_mode=bool(raw.get("ranking_mode", False)),
            keep_winner_only=bool(raw.get("keep_winner_only", False)),
            build_concurrency=max(1, int(raw.get("build_concurrency", 1) or 1)),
            disk_min_free_gb=(None if raw.get("disk_min_free_gb") is None else float(raw.get("disk_min_free_gb"))),
        )


@dataclass
class CommandResult:
    """Result of a single shell command (build/bench/accuracy)."""

    name: str
    command: str
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True iff returncode == 0 and command did not time out."""
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    """A single distilled observation suitable for KB contribution."""

    title: str
    body: str = ""
    source: str = ""
    session_id: str = ""
    candidate_ref: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateResult:
    """Per-candidate run summary used in explore_summary.json."""

    candidate: Candidate
    candidate_dir: str
    worktree_dir: str
    venv_dir: str
    status: str
    throughput: float | None = None
    accuracy: float | None = None
    completed: str = ""
    winner: bool = False
    reason: str = ""
    commands: list[CommandResult] = field(default_factory=list)
    patches_path: str = ""
    files_json_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output, expanding nested fields."""
        data = asdict(self)
        data["candidate"] = asdict(self.candidate)
        data["commands"] = [c.to_dict() for c in self.commands]
        return data
