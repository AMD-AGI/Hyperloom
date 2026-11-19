# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 2 (LLM-autonomous discovery): find fusible op chains from trace + source."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.config.resources import resource_path

from .diagnose import LAUNCH_BOUND_CATEGORIES, _load_trace_events, categorize_kernel_name
from .discovery_prompt import _DEFAULT_MAX_FUSIONS, DiscoveryEvidence, DiscoveryTarget, build_discovery_prompt
from .discovery_response import _extract_json_array, parse_discovered_recipes
from .llm_failure import (
    API_ERROR,
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_DEADLINE_SEC,
    DEFAULT_MAX_DELAY_SEC,
    RETRYABLE_KINDS,
    LlmUnavailableError,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    retry_delay,
)
from .models import Diagnosis, Recipe

log = logging.getLogger("forge_fusion")

LlmFn = Callable[[str], str]  # prompt -> raw model text (expected to contain JSON)


@dataclass(frozen=True)
class DiscoveryPrompting:
    """How many fusions discovery asks for, the operator-knowledge corpus its hints come from, and profiler hints."""

    max_fusions: Optional[int] = None
    knowledge_root: str | Path | None = None
    fusion_hints: Sequence[dict[str, Any]] = ()


@dataclass(frozen=True)
class DiscoverySession:
    """One read-only discovery session: model, timeout, transcript, working dir, guarded files, usage accumulator."""

    model: str = ""
    timeout_s: int = 900
    log_path: str = ""
    workdir: str = "."
    protected_files: Sequence[str] = ()
    usage: Any = None


@dataclass(frozen=True)
class DiscoveryRetry:
    """Overrides for how many attempts a discovery call makes and how it backs off between them."""

    attempts: Optional[int] = None
    base_delay_sec: Optional[float] = None
    max_delay_sec: Optional[float] = None
    sleep: Optional[Callable[[float], Any]] = None


# Hot kernels quoted in the discovery prompt.
_TOP_KERNELS = 15
# Operator hints offered, and the share of the best score a hint must reach.
_OPERATOR_HINT_LIMIT = 12
_OPERATOR_HINT_MIN_SCORE_RATIO = 0.25

# Discovery is handed read and search tools, and the first tool call ends the turn.
DEFAULT_DISCOVERY_TURNS = 60

# Heavy kernels a fusible chain runs between; everything else is launch-bound tail.
COMPUTE_CATEGORIES: frozenset[str] = frozenset({"gemm", "attention", "conv", "moe"})


def _resolve_max_fusions(value: Optional[int] = None) -> int:
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get("FORGE_MAX_FUSIONS", "").strip()
    if raw:
        with contextlib.suppress(ValueError):
            return max(1, int(raw))
    return _DEFAULT_MAX_FUSIONS


def hot_kernels_from_trace(
    trace_path: str | Path, *, top_n: int = 15, launch_bound_only: bool = True
) -> list[dict[str, Any]]:
    """Top GPU kernels from a kineto trace by total-duration share."""
    agg: dict[str, list[float]] = {}
    total = 0.0
    for ev in _load_trace_events(trace_path):
        if ev.get("cat") != "kernel":
            continue
        try:
            dur = float(ev.get("dur"))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        name = str(ev.get("name") or "")
        d = agg.setdefault(name, [0.0, 0])
        d[0] += dur
        d[1] += 1
        total += dur
    if total <= 0:
        return []

    rows: list[dict[str, Any]] = []
    for name, (dur, count) in agg.items():
        cat = categorize_kernel_name(name)
        if launch_bound_only and (cat in COMPUTE_CATEGORIES or cat == "collective"):
            continue
        rows.append(
            {
                "name": name,
                "category": cat,
                "share": dur / total,
                "count": count,
                "avg_us": dur / count,
            }
        )
    rows.sort(key=lambda r: r["share"], reverse=True)
    return rows[:top_n]


def kernel_names_from_trace(trace_path: str | Path, *, top_n: int = 40) -> list[str]:
    """Distinct kernel names ranked by total duration, compute kernels included."""
    totals: dict[str, float] = defaultdict(float)
    for event in _load_trace_events(trace_path):
        if event.get("cat") != "kernel":
            continue
        try:
            duration = float(event.get("dur"))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        name = str(event.get("name") or "")
        if name:
            totals[name] += duration
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_n]]


def stream_ordered_kernels(
    trace_path: str | Path,
) -> tuple[dict[tuple[Any, Any], list[dict[str, Any]]], float]:
    """Kernel events bucketed by GPU stream, each bucket ordered by timestamp.

    Returns the buckets and the summed kernel duration. Execution order only holds
    within one stream, so every adjacency question has to be asked per bucket.
    """
    streams: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    total_kernel_us = 0.0
    for event in _load_trace_events(trace_path):
        if event.get("cat") != "kernel":
            continue
        try:
            timestamp = float(event.get("ts"))
            duration = float(event.get("dur"))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        name = str(event.get("name") or "")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        stream_key = (
            args.get("device", event.get("pid", 0)),
            args.get("stream", event.get("tid", 0)),
        )
        streams[stream_key].append(
            {
                "name": name,
                "category": categorize_kernel_name(name),
                "ts": timestamp,
                "dur": duration,
            }
        )
        total_kernel_us += duration
    for bucket in streams.values():
        bucket.sort(key=lambda item: item["ts"])
    return streams, total_kernel_us


def _normalized_kernel_name(name: str) -> str:
    """``name`` lower-cased with hex addresses, integers and whitespace runs collapsed."""
    value = re.sub(r"0x[0-9a-f]+", "0x*", name.lower())
    value = re.sub(r"\b\d+\b", "N", value)
    return re.sub(r"\s+", " ", value).strip()[:160]


def _record_boundary_segment(
    aggregated: dict[tuple[tuple[str, str], ...], dict[str, Any]],
    segment: list[dict[str, Any]],
    max_chain_len: int,
) -> None:
    """Count one compute-anchored ``segment`` into ``aggregated`` when it holds a launch-bound kernel."""
    if len(segment) < 2 or len(segment) > max_chain_len:
        return
    categories = [str(item["category"]) for item in segment]
    if not any(category in LAUNCH_BOUND_CATEGORIES for category in categories):
        return
    key = tuple((str(item["category"]), _normalized_kernel_name(str(item["name"]))) for item in segment)
    interior_count = max(0, len(segment) - 2)
    if len(segment) == 3 and categories[0] == "gemm" and categories[1] in LAUNCH_BOUND_CATEGORIES:
        boundary_kind = "epilogue"
    elif categories[0] in COMPUTE_CATEGORIES:
        boundary_kind = "compute_boundary"
    else:
        boundary_kind = "vertical"
    # The trailing kernel is the NEXT compute anchor.
    terminal_compute = categories[-1] if categories[-1] in COMPUTE_CATEGORIES else ""
    fusable_categories = categories[1:-1] if terminal_compute else categories[1:]
    row = aggregated.setdefault(
        key,
        {
            "signature": " -> ".join(categories),
            "categories": categories,
            "fusable_categories": fusable_categories,
            "terminal_compute": terminal_compute,
            "kernels": [str(item["name"])[:200] for item in segment],
            "count": 0,
            "total_us": 0.0,
            "boundary_kind": boundary_kind,
            "launches_removed_upper_bound": max(1, interior_count),
        },
    )
    row["count"] += 1
    row["total_us"] += sum(float(item["dur"]) for item in segment)


def ordered_fusion_boundaries_from_trace(
    trace_path: str | Path,
    *,
    top_n: int = 16,
    max_chain_len: int = 8,
    min_repeats: int = 2,
) -> list[dict[str, Any]]:
    """Recover repeated compute-to-compute fusion boundaries from stream order."""
    streams, total_kernel_us = stream_ordered_kernels(trace_path)
    aggregated: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for ordered in streams.values():
        start_index: Optional[int] = None
        for index, event in enumerate(ordered):
            if event["category"] not in COMPUTE_CATEGORIES:
                continue
            if start_index is not None:
                _record_boundary_segment(aggregated, ordered[start_index : index + 1], max_chain_len)
            start_index = index
        if start_index is not None:
            _record_boundary_segment(aggregated, ordered[start_index:], max_chain_len)

    rows: list[dict[str, Any]] = []
    for row in aggregated.values():
        if int(row["count"]) < min_repeats:
            continue
        row["avg_chain_us"] = row["total_us"] / row["count"]
        # Ranking heuristic only, NOT a true fraction of GPU time: a kernel that sits between two compute anchors
        # belongs to two overlapping segments, so its duration is counted once per segment and shares can sum above 1.
        row["share_heuristic"] = row["total_us"] / total_kernel_us if total_kernel_us > 0 else 0.0
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["count"]) * int(row["launches_removed_upper_bound"]),
            float(row["total_us"]),
        ),
        reverse=True,
    )
    return rows[:top_n]


def _semantic_terms(boundary: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for value in boundary.get("categories", []):
        terms.update(re.findall(r"[a-z0-9]+", str(value).lower()))
    for value in boundary.get("kernels", []):
        terms.update(re.findall(r"[a-z0-9]+", str(value).lower()))

    aliases = {
        "activation": {"act", "gelu", "relu", "silu", "swiglu"},
        "copy": {"cache", "copy", "store", "write"},
        "elementwise": {"copy", "elementwise", "materialize"},
        "gemm": {"gate", "gemm", "linear", "matmul", "projection", "up"},
        "rmsnorm": {"norm", "rms", "rmsnorm"},
        "rope": {"rope", "rotary"},
        "attention": {"attention", "attn"},
    }
    for term in tuple(terms):
        if term in aliases:
            terms.update(aliases[term])
        if "cache" in term:
            terms.update({"cache", "store", "write"})
        if term in {"act", "activation"}:
            terms.update(aliases["activation"])
    return {term for term in terms if len(term) >= 3}


def _default_knowledge_root() -> Path:
    configured = os.environ.get("FORGE_LOCAL_KNOWLEDGE", "").strip()
    if configured:
        return Path(configured)
    return resource_path("local_knowledge", missing_ok=True)


def _tokens(text: str) -> set[str]:
    """Whole-word tokens of ``text``."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_OPERATOR_MARKERS = (
    "fused",
    "gemm",
    "rope",
    "cache",
    "silu",
    "swiglu",
    "rmsnorm",
    "attention",
)
_OPERATOR_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_.:]*)`")
_DECLARED_OPERATOR_PATTERN = re.compile(r"^operator:\s*([A-Za-z_][A-Za-z0-9_.:]*)\s*$", re.MULTILINE)

# Parsed knowledge documents, keyed by (path, mtime_ns, size) so an unchanged knowledge base is not re-read and
# re-parsed on every discovery run.
_KNOWLEDGE_CACHE: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def _operator_terms(operator: str) -> set[str]:
    terms = _tokens(operator)
    if any(term.startswith("gate") for term in terms):
        terms.update({"gate", "activation"})
    if "kv" in terms:
        terms.update({"cache", "store"})
    if "qk" in terms:
        terms.update({"norm", "rope"})
    return {term for term in terms if len(term) >= 3}


def _parse_knowledge_document(path: Path) -> list[dict[str, Any]]:
    """Extract every candidate operator mention from one knowledge document."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    declared_match = _DECLARED_OPERATOR_PATTERN.search(text)
    declared_operator = declared_match.group(1) if declared_match else ""
    entries: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph_tokens = _tokens(paragraph)
        for operator in _OPERATOR_PATTERN.findall(paragraph):
            lowered_operator = operator.lower()
            if not any(marker in lowered_operator for marker in _OPERATOR_MARKERS):
                continue
            if "_" not in operator and "." not in operator:
                continue
            entries.append(
                {
                    "operator": operator,
                    "operator_terms": _operator_terms(operator),
                    "paragraph_tokens": paragraph_tokens,
                    "paragraph_length": len(paragraph),
                    "evidence": re.sub(r"\s+", " ", paragraph).strip()[:300],
                    "is_declared": operator == declared_operator,
                }
            )
    return entries


def _knowledge_entries(root: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    documents: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in root.rglob("*.md"):
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            continue
        entries = _KNOWLEDGE_CACHE.get(key)
        if entries is None:
            entries = _parse_knowledge_document(path)
            _KNOWLEDGE_CACHE[key] = entries
        documents.append((path, entries))
    return documents


def existing_operator_hints_from_knowledge(
    knowledge_root: str | Path | None,
    boundaries: list[dict[str, Any]],
    *,
    fallback_categories: Optional[list[str]] = None,
    fallback_kernel_names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Retrieve existing ROCm operator names using observed runtime semantics."""
    root = Path(knowledge_root) if knowledge_root else _default_knowledge_root()
    if not root.is_dir():
        return []

    evidence_sources = list(boundaries)
    if fallback_categories or fallback_kernel_names:
        evidence_sources.append(
            {
                "categories": list(fallback_categories or []),
                "kernels": list(fallback_kernel_names or []),
            }
        )
    boundary_terms = [_semantic_terms(source) for source in evidence_sources]
    if not any(boundary_terms):
        return []

    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for path, entries in _knowledge_entries(root):
        for entry in entries:
            overlap = max(
                len(entry["operator_terms"] & terms) * 100.0 + len(terms & entry["paragraph_tokens"]) * 10.0
                for terms in boundary_terms
            )
            score = overlap - entry["paragraph_length"] / 500.0
            if entry["is_declared"]:
                score += 500.0
            if score < 20.0:
                continue
            row = {
                "operator": entry["operator"],
                "path": path.relative_to(root).as_posix(),
                "evidence": entry["evidence"],
                "score": round(score, 1),
            }
            previous = best.get(entry["operator"])
            if previous is None or score > previous[0]:
                best[entry["operator"]] = (score, row)

    ranked = sorted(
        best.values(),
        key=lambda item: (item[0], item[1]["operator"]),
        reverse=True,
    )
    if not ranked:
        return []
    # Pre-trim marginal matches relative to the best one, so a long tail of weak hints cannot pad the prompt and
    # inflate downstream authoring attempts.
    cutoff = ranked[0][0] * _OPERATOR_HINT_MIN_SCORE_RATIO
    return [row for score, row in ranked[:_OPERATOR_HINT_LIMIT] if score >= cutoff]


class DiscoverySafetyError(RuntimeError):
    """Report any source mutation made by a discovery-only Agent session."""


_DISCOVERY_SYSTEM_PROMPT = """\
You are the read-only discovery stage of KernelForge forge-fuse.
Analyze the evidence in the user prompt and return only the requested final text.
Do not edit, create, delete, or rename files. Do not run commands that modify the
workspace. An OS-level full-access preset is valid only when an explicit external
sandbox is authoritative; it does not grant logical write or shell permission.
"""


def _protected_file_snapshot(
    protected_files: list[str],
) -> dict[Path, tuple[bool, bytes, int]]:
    """Capture exact bytes so discovery can detect and undo source mutations."""
    snapshot: dict[Path, tuple[bool, bytes, int]] = {}
    for value in protected_files:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        try:
            exists = path.is_file()
            snapshot[path] = (
                exists,
                path.read_bytes() if exists else b"",
                path.stat().st_mode & 0o777 if exists else 0,
            )
        except OSError as exc:
            raise DiscoverySafetyError(f"cannot snapshot protected discovery source {path}: {exc}") from exc
    return snapshot


def _restore_changed_protected_files(
    snapshot: dict[Path, tuple[bool, bytes, int]],
) -> list[str]:
    """Restore changed protected files and return their paths."""
    changed: list[str] = []
    for path, (existed, content, mode) in snapshot.items():
        try:
            exists = path.is_file()
            differs = exists != existed
            if exists and existed:
                differs = path.read_bytes() != content or (path.stat().st_mode & 0o777) != mode
            if not differs:
                continue
            changed.append(str(path))
            if existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(mode)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError as exc:
            raise DiscoverySafetyError(f"discovery modified protected source {path} and restore failed: {exc}") from exc
    return changed


def _run_agent_discovery_once(
    backend: Any,
    spec: AgentRunSpec,
    *,
    timeout_s: int,
    protected_files: list[str],
    usage: Any = None,
) -> Any:
    """Run one SDK turn and enforce the provider-neutral source invariant."""
    snapshot = _protected_file_snapshot(protected_files)

    async def _run() -> Any:
        return await asyncio.wait_for(
            backend.run(spec, usage=usage),
            timeout=watchdog_timeout_sec(max(1, int(timeout_s))),
        )

    try:
        result = asyncio.run(_run())
    except BaseException as exc:
        changed = _restore_changed_protected_files(snapshot)
        if changed:
            raise DiscoverySafetyError("discovery agent modified protected source: " + ", ".join(changed)) from exc
        raise
    changed = _restore_changed_protected_files(snapshot)
    if changed:
        raise DiscoverySafetyError("discovery agent modified protected source: " + ", ".join(changed))
    return result


def _override_or_env(override: Any, name: str, default: float, cast: Callable[[Any], Any]) -> Any:
    """An explicit override, else the environment setting ``name`` with its default."""
    return cast(override) if override is not None else cast(env_setting(name, default, cast=cast))


def _discovery_run_spec(
    prompt: str, session: DiscoverySession, model: str, max_turns: int, progress: list[str]
) -> AgentRunSpec:
    """One read-only discovery turn in the session's working directory, logging into ``progress``."""
    return AgentRunSpec(
        system_prompt=_DISCOVERY_SYSTEM_PROMPT,
        user_prompt=prompt,
        role="fusion discovery",
        cwd=session.workdir,
        model=model,
        writable=False,
        timeout_sec=max(1, int(session.timeout_s)),
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=max_turns,
        ),
        protected_globs=["*"],
        # Not read_only_resume: discovery only needs the "tolerate a dirty worktree" half of that flag, and
        # claiming the resume contract disqualifies this session from the guard's read-only fast path
        # (workspace_guard.is_read_only_session), forcing a git-worktree requirement on a cwd that is
        # routinely a pip install root.
        allow_dirty_baseline=True,
        progress_log=progress,
    )


def _discovery_answer(result: Any, backend: Any) -> tuple[str, str]:
    """``(text, "")`` when one session's answer is usable, else ``("", why it is not)``."""
    text = str(getattr(result, "text", "") or "").strip()
    end_reason = result.end_reason
    cut_short = end_reason in {"turn_cap", "timeout"}
    # A cut-short session still answered if it got its proposals out first, and discovery spends turns by
    # design -- it is handed read and search tools precisely so it explores.
    usable = text and (not cut_short or _extract_json_array(text))
    if usable and not is_api_failure(result):
        if cut_short:
            log.warning(
                "discovery Agent ended with %s but its proposals parsed; using them",
                end_reason,
            )
        return text, ""
    return "", f"{backend.name} returned no final text" if not text else f"{backend.name} ended with {end_reason}"


def registered_agent_llm_fn(
    backend: Any,
    session: DiscoverySession = DiscoverySession(),
    retry: DiscoveryRetry = DiscoveryRetry(),
) -> LlmFn:
    """Adapt one registered Agent backend into discovery's text interface."""
    import time as _time

    timeout_s, log_path = session.timeout_s, session.log_path
    selected_model = session.model.strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    resolved_attempts = _override_or_env(retry.attempts, "FORGE_FUSION_LLM_ATTEMPTS", DEFAULT_ATTEMPTS, int)
    resolved_turns = max(1, int(env_setting("FORGE_FUSION_DISCOVERY_TURNS", DEFAULT_DISCOVERY_TURNS, cast=int)))
    resolved_base_delay = _override_or_env(
        retry.base_delay_sec, "FORGE_FUSION_LLM_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, float
    )
    resolved_max_delay = _override_or_env(
        retry.max_delay_sec, "FORGE_FUSION_LLM_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, float
    )
    resolved_deadline = float(env_setting("FORGE_LLM_RETRY_DEADLINE_SEC", DEFAULT_DEADLINE_SEC, cast=float))
    pause = retry.sleep or _time.sleep
    clock = _time.monotonic
    protected = list(session.protected_files or [])

    def _record_transcript(progress: list[str], text: str) -> None:
        """Persist what the session did, whatever the outcome."""
        if not log_path:
            return
        with contextlib.suppress(OSError):
            Path(log_path).write_text("\n".join([*progress, text]).strip() + "\n", encoding="utf-8")

    def _back_off(
        attempt: int, started_at: float, progress: list[str], last_error: str, last_exc: Exception | None
    ) -> None:
        """Sleep before the next attempt, or give up once the retry deadline has passed."""
        elapsed = clock() - started_at
        if resolved_deadline > 0 and elapsed >= resolved_deadline:
            _record_transcript(progress, last_error)
            raise LlmUnavailableError(
                f"discovery Agent produced no usable answer in {attempt} "
                f"attempt(s) and {elapsed:.0f}s "
                f"(deadline {resolved_deadline:.0f}s): {last_error}",
                kind=API_ERROR if last_exc is None else classify_llm_error(last_exc),
                attempts=attempt,
            ) from last_exc
        pause(
            retry_delay(
                attempt,
                base_sec=resolved_base_delay,
                max_sec=resolved_max_delay,
            )
        )

    def _fn(prompt: str) -> str:
        started_at = clock()
        last_error = ""
        last_kind = API_ERROR
        last_exc: Exception | None = None
        progress: list[str] = []
        for attempt in range(1, max(1, resolved_attempts) + 1):
            spec = _discovery_run_spec(prompt, session, selected_model, resolved_turns, progress)
            try:
                result = _run_agent_discovery_once(
                    backend,
                    spec,
                    timeout_s=timeout_s,
                    protected_files=protected,
                    usage=session.usage,
                )
                text, failure = _discovery_answer(result, backend)
                if text:
                    _record_transcript(progress, text)
                    return text
                last_error = failure
                last_kind, last_exc = API_ERROR, None
            except DiscoverySafetyError:
                raise
            except Exception as exc:
                if is_agent_safety_error(exc):
                    raise DiscoverySafetyError("discovery Agent safety violation: " + str(exc)) from exc
                last_kind, last_exc = classify_llm_error(exc), exc
                last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                if last_kind not in RETRYABLE_KINDS:
                    raise LlmUnavailableError(
                        f"discovery Agent call failed ({last_kind}): {last_error}",
                        kind=last_kind,
                        attempts=attempt,
                    ) from exc
            log.warning(
                "discovery Agent attempt %d/%d failed: %s",
                attempt,
                max(1, resolved_attempts),
                last_error,
                exc_info=last_exc,
            )
            if attempt >= max(1, resolved_attempts):
                break
            _back_off(attempt, started_at, progress, last_error, last_exc)
        _record_transcript(progress, last_error)
        raise LlmUnavailableError(
            f"discovery Agent produced no usable answer in {max(1, resolved_attempts)} attempt(s): {last_error}",
            kind=last_kind,
            attempts=max(1, resolved_attempts),
        ) from last_exc

    return _fn


def discover_recipes(
    diagnosis: Diagnosis,
    target: DiscoveryTarget,
    trace_path: str,
    llm_fn: LlmFn,
    prompting: DiscoveryPrompting = DiscoveryPrompting(),
) -> list[Recipe]:
    """LLM-autonomous discovery: propose fusible chains from the trace + source, citing advisory profiler hints."""
    source_file, repo_scope = target.source_file, target.repo_scope
    if not diagnosis.is_candidate:
        return []
    try:
        source_text = Path(source_file).read_text(encoding="utf-8") if source_file else ""
    except OSError:
        source_text = ""
    if not source_text and not repo_scope:
        log.warning("discovery: model source unreadable (%s); cannot self-discover", source_file)
        return []
    hot = hot_kernels_from_trace(trace_path, top_n=_TOP_KERNELS)
    ordered_boundaries = ordered_fusion_boundaries_from_trace(trace_path)
    # Hot kernels and the diagnosis categories are folded in as a second evidence source: ordered boundaries need
    # repeats to exist, so a short trace would otherwise leave retrieval with nothing to match against.
    existing_operator_hints = existing_operator_hints_from_knowledge(
        prompting.knowledge_root,
        ordered_boundaries,
        fallback_categories=list(diagnosis.dominant_categories),
        fallback_kernel_names=kernel_names_from_trace(trace_path),
    )
    prompt = build_discovery_prompt(
        target,
        source_text,
        DiscoveryEvidence(
            diagnosis=diagnosis,
            hot_kernels=hot,
            ordered_boundaries=ordered_boundaries,
            existing_operator_hints=existing_operator_hints,
            fusion_hints=prompting.fusion_hints,
        ),
        _resolve_max_fusions(prompting.max_fusions),
    )
    raw = llm_fn(prompt)
    recipes = parse_discovered_recipes(raw, target, category_shares=diagnosis.category_shares)
    log.info(
        "discovery proposed %d fusion(s): %s",
        len(recipes),
        ", ".join(r.pattern_id for r in recipes),
    )
    return recipes
