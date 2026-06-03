"""Request and result models for framework PR/ref exploration.

Ported from zhenggong/framework-agent with two fusion-plan additions:

* :attr:`ExploreRequest.gap_description` - free-form bottleneck description
  consumed by :mod:`framework_agent.keywords` to extract perf keywords
  before hitting GitHub Search.
* :attr:`ExploreRequest.search_modes` - tuple of enabled candidate
  sources. Order matters; ``("primus_cortex", "github")`` means we union
  results from both, with primus-cortex hard-failing on transport errors
  and GitHub falling back to best-effort.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PRIMUS_CORTEX_ENV_VAR = "PRIMUS_CORTEX_PR_API"


@dataclass(frozen=True)
class Baseline:
    """Baseline throughput/accuracy that thresholds compare against."""

    throughput: float
    accuracy: float | None = None
    completed: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Baseline":
        """Parse raw dict into Baseline; throughput must be > 0.

        Args:
            raw (dict[str, Any]): Mapping with ``throughput`` (or
                ``output_throughput``), optional ``accuracy``, and optional
                ``completed``.

        Returns:
            Baseline: The parsed baseline.

        Raises:
            ValueError: If the resolved throughput is not greater than 0.
        """
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
        """Parse raw dict; missing keys fall back to defaults.

        Args:
            raw (dict[str, Any] | None): Mapping with optional
                ``min_throughput_ratio`` and ``max_accuracy_drop``.

        Returns:
            Thresholds: The parsed thresholds, with defaults for missing keys.
        """
        raw = raw or {}
        return cls(
            min_throughput_ratio=float(raw.get("min_throughput_ratio", 1.05)),
            max_accuracy_drop=float(raw.get("max_accuracy_drop", 0.05)),
        )


@dataclass(frozen=True)
class PrimusCortexConfig:
    """Configuration for the internal primus-cortex-pr-monitor service.

    When present on :class:`ExploreRequest`, the agent routes PR candidate
    enumeration through this service. Errors are hard-failed by callers
    (see :mod:`framework_agent.sources.primus_cortex`).
    """

    base_url: str
    timeout_sec: float = 10.0
    default_label: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrimusCortexConfig":
        """Parse primus_cortex block; base_url is mandatory.

        Args:
            raw (dict[str, Any]): Mapping with ``base_url`` and optional
                ``timeout_sec`` / ``default_label``.

        Returns:
            PrimusCortexConfig: The parsed config.

        Raises:
            ValueError: If ``base_url`` is missing or empty.
        """
        base_url = str(raw.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("primus_cortex.base_url is required when primus_cortex block is set")
        timeout_raw = raw.get("timeout_sec", 10.0)
        label_raw = raw.get("default_label")
        return cls(
            base_url=base_url,
            timeout_sec=float(timeout_raw),
            default_label=(
                str(label_raw).strip()
                if isinstance(label_raw, str) and label_raw.strip()
                else None
            ),
        )


@dataclass(frozen=True)
class CommandSpec:
    """A single build/benchmark/accuracy command for a candidate run."""

    command: str
    timeout_sec: int = 3600
    required: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CommandSpec":
        """Parse a command block; command text is mandatory.

        Args:
            raw (dict[str, Any]): Mapping with ``command`` and optional
                ``timeout_sec`` / ``required``.

        Returns:
            CommandSpec: The parsed command spec.

        Raises:
            ValueError: If ``command`` is missing or empty.
        """
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
    """A single PR or git ref candidate (explicit, primus_cortex, or github).

    ``score`` carries the gap-relevance score produced by the dispatcher's
    anti-aware reranker (:func:`framework_agent.keywords.score_title_with_anti_signal`).
    It is 0.0 when no gap-driven ranking happened (e.g. ``source='explicit'``
    or label-only listing). Downstream consumers (notably the IO
    ``framework_pr`` arm) use it to log why a candidate won and to drive
    a "best vs second" gap in their bandit history; the field is non-load-
    bearing in fa itself (sort order is preserved by the dataclass list).
    """

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

    @property
    def slug(self) -> str:
        """Filesystem-safe slug derived from ref (used for candidate_dir name).

        Returns:
            str: A lowercased slug with non-alphanumeric characters (except
                ``.-_``) replaced by hyphens, defaulting to ``"candidate"``.
        """
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
        """Return the PR number when ref starts with ``PR:``; else None.

        Returns:
            int | None: The parsed PR number, or ``None`` when the ref is not a
                ``PR:`` ref or the number is unparseable.
        """
        if not self.ref.startswith("PR:"):
            return None
        try:
            return int(self.ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return None


@dataclass(frozen=True)
class PrFilter:
    """Server-side and client-side filter applied to enumerated PR candidates.

    Path filters require Stage 2 enrichment (changed_files populated).
    Labels are case-insensitive. Dates flow through to primus-cortex's
    REST query when supported.
    """

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
        """Coerce string/list/None into a clean tuple of non-empty strings.

        Args:
            raw (Any): ``None``, a string, or a list/tuple of values.

        Returns:
            tuple[str, ...]: Trimmed, non-empty string values.

        Raises:
            ValueError: If ``raw`` is neither None, a string, nor a list/tuple.
        """
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw.strip(),) if raw.strip() else ()
        if isinstance(raw, (list, tuple)):
            return tuple(str(v).strip() for v in raw if str(v).strip())
        raise ValueError(
            f"pr_filter list field must be string or list, got {type(raw).__name__}"
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PrFilter":
        """Parse pr_filter block; missing keys fall back to empty defaults.

        Args:
            raw (dict[str, Any] | None): The ``pr_filter`` object, or ``None``.

        Returns:
            PrFilter: The parsed filter (empty when ``raw`` is ``None``).

        Raises:
            ValueError: If ``raw`` is present but not an object.
        """
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
        """True when no constraint is set (filter is a no-op).

        Returns:
            bool: ``True`` if every filter field is unset/zero, else ``False``.
        """
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


_VALID_SEARCH_MODES = frozenset({"primus_cortex", "github"})


def _parse_keywords(raw: Any) -> tuple[str, ...]:
    """Coerce an optional ``keywords`` field into a tuple of trimmed strings.

    Accepts:
      * ``None`` / missing  -> ``()``  (auto-extract from gap_description)
      * empty string        -> ``()``
      * non-empty string    -> split on comma/whitespace
      * list / tuple        -> coerce items to str + trim + drop empties

    Anything else raises ``ValueError`` (matches the strictness of the
    other ``_parse_*`` helpers in this module).

    Args:
        raw (Any): ``None``, a string (split on comma/whitespace), or a
            list/tuple of values.

    Returns:
        tuple[str, ...]: Trimmed, non-empty keyword tokens.

    Raises:
        ValueError: If ``raw`` is neither None/empty, a string, nor a
            list/tuple.
    """
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        items = [tok.strip() for tok in raw.replace(",", " ").split() if tok.strip()]
        return tuple(items)
    if isinstance(raw, (list, tuple)):
        items = [str(v).strip() for v in raw if str(v).strip()]
        return tuple(items)
    raise ValueError(
        f"keywords must be list/tuple/str when present, got {type(raw).__name__}"
    )


def _parse_search_modes(raw: Any) -> tuple[str, ...]:
    """Coerce a list of mode names; default to primus_cortex + github.

    Args:
        raw (Any): ``None``/empty (defaults applied), a single mode string, or a
            list/tuple of mode names.

    Returns:
        tuple[str, ...]: The validated search-mode names.

    Raises:
        ValueError: If ``raw`` is not a string/list, or contains an unknown
            mode name.
    """
    if raw is None or raw == "":
        return ("primus_cortex", "github")
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        items = [str(v).strip() for v in raw if str(v).strip()]
    else:
        raise ValueError(
            f"search_modes must be string or list, got {type(raw).__name__}"
        )
    for item in items:
        if item not in _VALID_SEARCH_MODES:
            raise ValueError(
                f"search_modes contains unknown source {item!r}; "
                f"valid values are {sorted(_VALID_SEARCH_MODES)!r}"
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
    primus_cortex: PrimusCortexConfig | None = None
    pr_filter: PrFilter = field(default_factory=PrFilter)
    # Fusion-plan additions (not present in zhenggong v0.2)
    gap_description: str = ""
    # C: explicit keyword override for the primus_cortex search query +
    # client-side rerank. Non-empty value bypasses extract_keywords() and
    # uses these tokens verbatim. See ``sources._resolve_keywords``.
    keywords: tuple[str, ...] = ()
    search_modes: tuple[str, ...] = ("primus_cortex", "github")
    # KB integration (PR4); empty string disables the contribute hook.
    kb_domain: str = ""
    # Merged-design §4.4.1 additions: ranking + cleanup + disk preflight.
    # When True, ``explore()`` keeps going after the first winner and
    # returns the full list sorted by ``candidate_score`` descending.
    # When False (default — matches zhenggong v0.2), it short-circuits
    # on the first winner.
    ranking_mode: bool = False
    # When True, ``explorer`` removes the worktree+venv directories of
    # every non-winner candidate at the end of the run. The
    # ``candidate_dir`` itself and the audit material inside it stay so
    # reviewers can still diff the PRs that lost.
    keep_winner_only: bool = False
    # When > 0, ``explore()`` runs the build step of multiple candidates
    # concurrently via ``asyncio.gather`` (bench/accuracy stay strictly
    # serial to avoid GPU contention). 0 / 1 / negative => fully serial.
    build_concurrency: int = 1
    # Disk preflight threshold (GB). ``None`` falls back to the env var
    # ``FRAMEWORK_EXPLORER_DISK_MIN_GB`` (default 20). Set 0 to bypass.
    disk_min_free_gb: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExploreRequest":
        """Parse a JSON request payload into ExploreRequest, validating required fields.

        Args:
            raw (dict[str, Any]): The decoded JSON request payload.

        Returns:
            ExploreRequest: The fully parsed and validated request.

        Raises:
            ValueError: If a required field (``framework``, ``repo_url``,
                ``baseline``) is missing/invalid, or a nested block has the
                wrong type.
        """
        framework = str(raw.get("framework") or "").strip().lower()
        if not framework:
            raise ValueError("framework is required")
        repo_url = str(raw.get("repo_url") or "").strip()
        if not repo_url:
            raise ValueError("repo_url is required")
        work_dir = Path(str(raw.get("work_dir") or "/tmp/framework-agent")).expanduser()
        baseline_raw = raw.get("baseline")
        if not isinstance(baseline_raw, dict):
            raise ValueError("baseline object is required")
        commands_raw = raw.get("commands") or {}
        if not isinstance(commands_raw, dict):
            raise ValueError("commands must be an object")
        commands = {
            str(name): CommandSpec.from_dict(spec)
            for name, spec in commands_raw.items()
            if isinstance(spec, dict)
        }
        outputs = raw.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise ValueError("outputs must be an object when present")
        primus_raw = raw.get("primus_cortex")
        if isinstance(primus_raw, dict):
            primus_cortex = PrimusCortexConfig.from_dict(primus_raw)
        elif primus_raw is None:
            env_base_url = os.environ.get(PRIMUS_CORTEX_ENV_VAR, "").strip()
            primus_cortex = (
                PrimusCortexConfig(base_url=env_base_url) if env_base_url else None
            )
        else:
            raise ValueError("primus_cortex must be an object when present")
        return cls(
            framework=framework,
            repo_url=repo_url,
            work_dir=work_dir,
            baseline=Baseline.from_dict(baseline_raw),
            thresholds=Thresholds.from_dict(raw.get("thresholds")),
            candidate_refs=tuple(
                str(r).strip() for r in raw.get("candidate_refs") or () if str(r).strip()
            ),
            search_perf_prs=bool(raw.get("search_perf_prs", False)),
            max_search_candidates=int(raw.get("max_search_candidates", 5)),
            prepare_candidate_env=bool(raw.get("prepare_candidate_env", True)),
            commands=commands,
            outputs={str(k): str(v) for k, v in outputs.items()},
            primus_cortex=primus_cortex,
            pr_filter=PrFilter.from_dict(raw.get("pr_filter")),
            gap_description=str(raw.get("gap_description") or "").strip(),
            keywords=_parse_keywords(raw.get("keywords")),
            search_modes=_parse_search_modes(raw.get("search_modes")),
            kb_domain=str(raw.get("kb_domain") or "").strip(),
            ranking_mode=bool(raw.get("ranking_mode", False)),
            keep_winner_only=bool(raw.get("keep_winner_only", False)),
            build_concurrency=max(1, int(raw.get("build_concurrency", 1) or 1)),
            disk_min_free_gb=(
                None
                if raw.get("disk_min_free_gb") is None
                else float(raw.get("disk_min_free_gb"))
            ),
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
        """True iff returncode == 0 and command did not time out.

        Returns:
            bool: ``True`` when the command succeeded and did not time out.
        """
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output.

        Returns:
            dict[str, Any]: A dataclass-derived dict of all fields.
        """
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    """A single distilled observation suitable for KB contribution.

    Used by :mod:`framework_agent.kb.synthesize_findings`. Keeping the
    record frozen + flat keeps the markdown rendering deterministic.
    """

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
        """Serialize to a plain dict for JSON output, expanding nested fields.

        Returns:
            dict[str, Any]: A dict with the nested ``candidate`` and
                ``commands`` fields expanded to plain dicts.
        """
        data = asdict(self)
        data["candidate"] = asdict(self.candidate)
        data["commands"] = [c.to_dict() for c in self.commands]
        return data
