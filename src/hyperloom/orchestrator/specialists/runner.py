# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SpecialistRunner."""

from __future__ import annotations

import enum
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common import io as _common_io
from hyperloom.common.env_safety import BENCHMARK_SECRET_ENV_NAMES, redact_secret_values
from hyperloom.common.timeutil import now_iso

from hyperloom.inference_optimizer.session.session_paths import runs_dir, specialist_intel_path
from ..roles.base import BackendError, LLMCallFailed
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from ..trace.conversation_trace import ConversationRecord, append_conversation
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from .domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    FREEFORM_DOMAIN,
    SPECIALIST_DOMAIN_KEYS,
    SpecialistDomain,
    domain_for_tag,
    normalize_dispatch_tags,
)
from .subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    SpecialistSubprocessResult,
    _pick_worktree_base,
    _setup_worktree,
)
from . import patch_safety as _patch_safety
from .profile import MODE_PATCH, SpecialistProfile, resolve_specialist_profile
from ..framework.paths import resolve_framework_tree
from ..loop.sub_agent_runner import RunnerContext
from ..prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


log = logging.getLogger(__name__)


def resolve_specialist_max_turns(raw: Any, *, default: int) -> int:
    """Resolve the specialist turn budget from dispatch params."""
    if raw in (None, ""):
        return int(default)
    max_turns = int(raw)
    if max_turns == 0:
        return int(default)
    return max_turns


def _extra_focus_tags(
    params: dict[str, Any],
    domain: "SpecialistDomain",
) -> tuple[str, ...]:
    """Knowledge-domain tags beyond the primary domain's anchor."""
    tags = normalize_dispatch_tags(params)
    primary_anchor = (domain.kb_anchor or "").strip()
    return tuple(t for t in tags if t and t != primary_anchor)


SPECIALIST_TOOL_DENYLIST: frozenset[str] = frozenset(
    {
        "KillShell",
        "SlashCommand",
    }
)


_now_iso = now_iso


_SECRET_ENV_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *BENCHMARK_SECRET_ENV_NAMES,
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_CUSTOM_HEADERS",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAW_API_KEY",
            "GITHUB_TOKEN",
            "HF_TOKEN",
            "HF_TOKEN_2",
            "HYPERLOOM_GIT_TOKEN",
            "HYPERLOOM_PR_CI_GH_TOKEN",
            "LLM_API_KEY",
            "LLM_GATEWAY_KEY",
            "OPENAI_API_KEY",
            "OPENAI_CUSTOM_HEADERS",
            "SAFE_API_KEY",
        }
    )
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\b(?:"
    + "|".join(re.escape(name) for name in _SECRET_ENV_NAMES)
    + r")\b)(?P<sep>\s*(?:=|:)\s*)(?P<quote>['\"]?)(?!\[REDACTED\])"
    + r"(?P<value>[^\s,'\"\]}]+)(?P=quote)",
    re.IGNORECASE,
)


def _sibling_checkouts(roots: tuple[str, ...], base: Path | None) -> tuple[Path, ...]:
    """Return the allowlisted source trees other than ``base``."""
    base_resolved = base.resolve() if base else None
    out: list[Path] = []
    for raw in roots:
        root = Path(raw)
        if not root.is_dir() or root.resolve() == base_resolved:
            continue
        out.append(root)
    return tuple(out)


def _grounding_explicit_root(
    *,
    declared: str,
    patches: list[str],
    patch_roots: dict[str, str],
) -> Path | None:
    """Return the root to ground the whole patch set against, or ``None``."""
    if declared:
        return Path(declared)
    if not patches or any(patch not in patch_roots for patch in patches):
        return None
    roots = set(patch_roots.values())
    return Path(roots.pop()) if len(roots) == 1 else None


def _patch_path_within_bases(path: Path, bases: list[Path]) -> bool:
    """True when ``path`` resolves inside one of the specialist sandbox bases."""
    try:
        rp = path.resolve()
    except OSError:
        return False
    for base in bases:
        try:
            br = base.resolve()
        except OSError:
            continue
        try:
            if rp == br or rp.is_relative_to(br):
                return True
        except AttributeError:  # pragma: no cover - Python <3.9
            try:
                rp.relative_to(br)
                return True
            except ValueError:
                continue
    return False


def _safe_redact(s: str) -> str:
    """Redact obvious secrets from a transcript line before writing to disk."""
    out = redact_secret_values(s)
    out = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group('key')}{m.group('sep')}{m.group('quote')}[REDACTED]{m.group('quote')}",
        out,
    )
    return out


def _redact_transcript_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_redact(value)
    if isinstance(value, dict):
        return {str(k): _redact_transcript_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_transcript_value(v) for v in value]
    if isinstance(value, tuple):
        return [_redact_transcript_value(v) for v in value]
    return value


@dataclass
class _PreparedRun:
    """Shared setup-phase output threaded into both execution paths."""

    domain: SpecialistDomain | None = None
    gap: str = ""
    max_turns: int = 0
    # Resolved dispatch profile (scope / mode / bench / lane).
    profile: "SpecialistProfile" = field(default_factory=SpecialistProfile)
    workspace: Path | None = None
    worktree: Path | None = None
    worktree_base: Path | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    notes: list[str] = field(default_factory=list)
    # When set, the caller returns this verbatim and skips execute.
    early_return: "SpecialistRunResult | None" = None


@dataclass
class SpecialistRunResult:
    """Internal record of one SpecialistRunner invocation."""

    task_id: str
    domain: str
    gap_canonical_id: str
    status: str  # "succeeded" / "partial" / "stale" / "empty_synthesised" / "tool_violation"
    specialist_done: dict[str, Any]
    turns_used: int = 0
    workspace: str = ""
    error: str = ""
    transcript_path: str = ""
    done_path: str = ""
    notes: list[str] = field(default_factory=list)


class SpecialistFailureType(str, enum.Enum):
    """Coarse failure taxonomy for a finished specialist run."""

    NONE = "none"  # succeeded
    TIMEOUT = "timeout"  # subprocess wall-clock kill
    STALE_HEARTBEAT = "stale_heartbeat"  # heartbeat went silent (hang)
    CRASH = "crash"  # nonzero exit / backend error
    NO_OUTPUT = "no_output"  # ran clean but emitted no done / empty
    TOOL_VIOLATION = "tool_violation"  # emitted a forbidden intent
    CONFIG = "config"  # unknown domain / no workspace
    UNKNOWN = "unknown"


# Transient infra failures a bounded auto-retry may re-dispatch.
RETRYABLE_SPECIALIST_FAILURES: frozenset[SpecialistFailureType] = frozenset(
    {
        SpecialistFailureType.TIMEOUT,
        SpecialistFailureType.STALE_HEARTBEAT,
        SpecialistFailureType.CRASH,
    }
)


def classify_specialist_failure(
    runner_status: str,
    error: str,
) -> tuple[SpecialistFailureType, bool]:
    """Map a :class:`SpecialistRunResult` ``(status, error)`` to a failure type + retry-eligibility flag."""
    status = (runner_status or "").strip().lower()
    err = (error or "").strip().lower()
    if status == "succeeded":
        return SpecialistFailureType.NONE, False
    if status == "tool_violation":
        return SpecialistFailureType.TOOL_VIOLATION, False
    if status == "partial":
        # Salvaged work: classify the failure but never retry over it.
        if "timeout" in err:
            return SpecialistFailureType.TIMEOUT, False
        if "stale_heartbeat" in err:
            return SpecialistFailureType.STALE_HEARTBEAT, False
        return SpecialistFailureType.CRASH, False
    if status == "stale":
        if "timeout" in err:
            ftype = SpecialistFailureType.TIMEOUT
        elif "stale_heartbeat" in err:
            ftype = SpecialistFailureType.STALE_HEARTBEAT
        else:
            ftype = SpecialistFailureType.CRASH
        return ftype, True
    if status == "empty_synthesised":
        if "unknown_domain" in err or "no_workspace" in err:
            return SpecialistFailureType.CONFIG, False
        return SpecialistFailureType.NO_OUTPUT, False
    return SpecialistFailureType.UNKNOWN, False


def build_empty_specialist_done(
    *,
    gap_canonical_id: str,
    domain: str,
    reason: str,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Return the canonical empty ``specialist_done`` payload."""
    return {
        "gap_canonical_id": gap_canonical_id,
        "domain": domain,
        "proposal_set": [],
        "empty": True,
        "summary": (reason or "specialist exited empty")[:480],
        "reason": reason or "specialist exited empty",
        "confidence": float(max(0.0, min(1.0, confidence))),
        "new_findings": [],
        "residual_questions": [],
    }


class SpecialistRunner:
    """LLM-driven sub-agent runner for the merged ``specialist`` action."""

    def __init__(
        self,
        backend_factory=None,
        *,
        subprocess_config: SpecialistSubprocessConfig | None = None,
        session_dir: Path | None = None,
        default_max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS,
    ):
        """Create a runner."""
        if backend_factory is None and subprocess_config is None:
            raise ValueError("SpecialistRunner: pass exactly one of backend_factory / subprocess_config")
        if backend_factory is not None and subprocess_config is not None:
            raise ValueError(
                "SpecialistRunner: backend_factory and subprocess_config are mutually exclusive — pick one path"
            )
        self.backend_factory = backend_factory
        self.subprocess_config = subprocess_config
        self.subprocess_dispatcher = (
            SpecialistSubprocessDispatcher(subprocess_config) if subprocess_config is not None else None
        )
        self.session_dir = Path(session_dir) if session_dir else None
        self.default_max_turns = int(default_max_turns)

    # Public entry point — dispatches to in-process or subprocess path
    async def run(
        self,
        ctx: RunnerContext,
        *,
        prompt_inputs: SpecialistPromptInputs | None = None,
    ) -> SpecialistRunResult:
        """Run a specialist task to completion (or synthesise an empty done)."""
        prep = await self._prepare(ctx, prompt_inputs=prompt_inputs)
        if prep.early_return is not None:
            return prep.early_return

        if self.subprocess_dispatcher is not None:
            return await self._run_via_subprocess(ctx, prep)
        return await self._run_via_backend(ctx, prep)

    # Setup phase (shared)
    async def _prepare(
        self,
        ctx: RunnerContext,
        *,
        prompt_inputs: SpecialistPromptInputs | None,
    ) -> "_PreparedRun":
        """Run the shared setup phase before dispatch."""
        params = ctx.task.params or {}
        domain_key = str(params.get("domain") or "").strip()
        # Back-fill domain_key from the first resolved tag for tag-only dispatch.
        if not domain_key:
            resolved_tags = normalize_dispatch_tags(params)
            if resolved_tags:
                domain_key = resolved_tags[0]
        gap = str(params.get("gap_canonical_id") or params.get("gap") or "").strip()
        max_turns = resolve_specialist_max_turns(params.get("max_turns"), default=self.default_max_turns)
        # Resolve by anchor first then key so a domain carrying the KB anchor matches its entry.
        domain = domain_for_tag(domain_key)
        profile = resolve_specialist_profile(params, domain=domain)
        task_description = str(params.get("task_description") or "").strip()

        workspace = self._resolve_workspace(ctx)

        # scope='freeform' is not bound to the domain catalogue; use the synthetic freeform domain.
        if domain is None and profile.is_freeform:
            domain = FREEFORM_DOMAIN

        if domain is None:
            done = build_empty_specialist_done(
                gap_canonical_id=gap,
                domain=domain_key,
                reason=f"unknown specialist domain={domain_key!r}",
            )
            self._write_specialist_done(workspace, done)
            return _PreparedRun(
                early_return=SpecialistRunResult(
                    task_id=ctx.task.task_id,
                    domain=domain_key,
                    gap_canonical_id=gap,
                    status="empty_synthesised",
                    specialist_done=done,
                    turns_used=0,
                    workspace=str(workspace) if workspace else "",
                    error="unknown_domain",
                    notes=[f"unknown_domain:{domain_key!r}"],
                )
            )

        # Domains outside the catalogue (today only the synthetic freeform domain) still dispatch, just without a
        # per-domain focus block.
        notes: list[str] = []
        if domain.key not in SPECIALIST_DOMAIN_KEYS:
            notes.append(f"domain={domain.key!r} is outside the domain catalogue; using generic prompt template")

        # Worktree — created only under subprocess dispatch; surfaced via ``workspace_path``.
        worktree, worktree_base, worktree_err = self._maybe_setup_worktree(
            ctx,
            workspace=workspace,
            profile=profile,
        )
        if worktree_err:
            notes.append(f"worktree_setup_failed:{worktree_err}")
        workspace_for_prompt = worktree or workspace

        allocated_gpu_ids = tuple(int(g) for g in ((ctx.extra or {}).get("gpu_ids") or []))

        if prompt_inputs is None:
            prompt_inputs = SpecialistPromptInputs(
                task_id=ctx.task.task_id,
                domain=domain,
                max_turns=max_turns,
                gap_canonical_id=gap,
                gap_symptom=str(params.get("gap_symptom") or ""),
                gap_layer=str(params.get("gap_layer") or ""),
                gap_evidence=dict(params.get("gap_evidence") or {}),
                kb_subgraph=dict(params.get("kb_subgraph") or {}),
                # Coordinator-populated roofline pre-fetch; empty when not warmed.
                roofline_evidence=dict(params.get("roofline_evidence") or {}),
                sub_kind=str(params.get("sub_kind") or ""),
                extra_focus_tags=_extra_focus_tags(params, domain),
                warm_start_recipe=dict(params.get("warm_start_recipe") or {}),
                warm_start_pitfalls=list(params.get("warm_start_pitfalls") or []),
                warm_start_lessons=list(params.get("warm_start_lessons") or []),
                pr_monitor_available=bool(params.get("pr_monitor_available", True)),
                framework=str(params.get("framework") or ""),
                framework_source_roots=tuple(params.get("framework_source_roots") or ()),
                source_hint_directories=tuple(params.get("source_hint_directories") or ()),
                model_info=dict(params.get("model_info") or {}),
                static_recon_checklist=str(params.get("static_recon_checklist") or ""),
                enablement_source_context=str(params.get("enablement_source_context") or ""),
                enablement_candidate_refs=tuple(
                    str(r).strip() for r in (params.get("enablement_candidate_refs") or ()) if str(r).strip()
                ),
                enablement_accepted_config={
                    "extra_envs": dict(params.get("base_extra_envs") or {}),
                    "extra_server_args": str(params.get("base_extra_args") or "").strip(),
                },
                gpu_type=str(params.get("gpu_type") or ""),
                allocated_gpu_ids=allocated_gpu_ids,
                tp=int(params.get("tp") or 0),
                hbm_gb=float(params.get("hbm_gb") or 0.0),
                peak_tflops=float(params.get("peak_tflops") or 0.0),
                arch_notes=str(params.get("arch_notes") or ""),
                target_gap_notes=str(params.get("target_gap_notes") or ""),
                already_proven=[p for p in (params.get("already_proven") or []) if isinstance(p, dict)],
                recipe_sites=tuple(str(s).strip() for s in (params.get("recipe_sites") or ()) if str(s).strip()),
                research_hints=str(params.get("research_hints") or ""),
                # Workload context warmed from SharedState.
                precision=str(params.get("precision") or ""),
                conc=int(params.get("conc") or 0),
                isl=int(params.get("isl") or 0),
                osl=int(params.get("osl") or 0),
                max_model_len=int(params.get("max_model_len") or 0),
                # Runtime fingerprint to flag version-mismatched lessons.
                framework_version=str(params.get("framework_version") or ""),
                workspace_path=(str(workspace_for_prompt) if workspace_for_prompt else ""),
                notes=str(params.get("notes") or ""),
                scope=profile.scope,
                mode=profile.mode,
                bench=profile.bench,
                lane=profile.lane,
                task_description=task_description,
                # Coordinator-injected note when this is a bounded auto-retry.
                auto_retry_reason=str(params.get("_auto_retry_reason") or ""),
                # WS1 wall-clock budget so the specialist can self-throttle.
                wall_budget_sec=float((ctx.extra or {}).get("wall_budget_sec") or 0.0),
                started_at_iso=datetime.now(timezone.utc).isoformat(),
                baseline_tput=float(params.get("baseline_tput") or 0.0),
                current_tput=float(params.get("current_tput") or 0.0),
                cumulative_gain_validated=float(params.get("cumulative_gain_validated") or 0.0),
                keep_threshold_pct=float(params.get("keep_threshold_pct") or 0.0),
                applied_stack=[e for e in (params.get("applied_stack") or []) if isinstance(e, dict)],
                task_kind=str(params.get("task_kind") or ""),
                prior_attempts=[e for e in (params.get("prior_attempts") or []) if isinstance(e, dict)],
                pr_lead=dict(params.get("pr_lead") or {}),
                exit_channel=("B" if self.subprocess_config is not None else "A"),
            )

        system_prompt, user_prompt = build_specialist_prompts(prompt_inputs)
        self._write_prompt(workspace, system_prompt, user_prompt)
        self._write_heartbeat(
            workspace,
            turn=0,
            max_turns=max_turns,
            status="starting",
        )

        return _PreparedRun(
            domain=domain,
            gap=gap,
            max_turns=max_turns,
            profile=profile,
            workspace=workspace,
            worktree=worktree,
            worktree_base=worktree_base,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            notes=notes,
        )

    @staticmethod
    def _ctx_tick_phase(ctx: "RunnerContext | None") -> tuple[int | None, str | None]:
        """Best-effort (tick, phase) from the live SharedState on ``ctx.extra``."""
        try:
            extra = getattr(ctx, "extra", None) or {}
            ss = extra.get("shared_state")
            if ss is None:
                return None, None
            tick = ss.tick
            phase = ss.phase
            return (
                int(tick) if tick is not None else None,
                (str(phase) or None) if phase else None,
            )
        except Exception:  # noqa: BLE001 — telemetry must never break the run
            return None, None

    def _trace_specialist_llm_call(
        self,
        *,
        task_id: str,
        turn: int,
        metadata: dict[str, Any] | None,
        latency_ms: int | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for an in-process specialist turn."""
        if self.session_dir is None:
            return
        try:
            md = metadata or {}
            has_tokens = any(
                md.get(k) is not None
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            if not has_tokens:
                return
            record = LLMCallRecord.from_metadata(
                session_id=self.session_dir.name,
                component="specialist",
                task_id=task_id,
                turn=turn,
                metadata=md,
                latency_ms=latency_ms,
                tick=tick,
                phase=phase,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist llm_call append failed for task_id=%s turn=%s",
                task_id,
                turn,
                exc_info=True,
            )

    def _trace_specialist_llm_failure(
        self,
        *,
        task_id: str,
        turn: int,
        error: BaseException,
        latency_ms: int | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Append one ``status=\"error\"`` row for a specialist turn that never returned."""
        if self.session_dir is None:
            return
        try:
            record = LLMCallRecord.for_failure(
                session_id=self.session_dir.name,
                component="specialist",
                task_id=task_id,
                turn=turn,
                error=error,
                latency_ms=latency_ms,
                tick=tick,
                phase=phase,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist llm_call failure append failed for task_id=%s turn=%s",
                task_id,
                turn,
                exc_info=True,
            )

    def _record_specialist_intel(
        self,
        *,
        task_id: str,
        turn: int,
        tool_calls: list[dict[str, Any]] | None,
    ) -> None:
        """Append the specialist's intel/tool calls to ``specialist_intel.jsonl``."""
        if self.session_dir is None or not tool_calls:
            return
        try:
            path = specialist_intel_path(self.session_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = _now_iso()
            with path.open("a", encoding="utf-8") as f:
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    row = {
                        "session_id": self.session_dir.name,
                        "component": "specialist",
                        "task_id": task_id,
                        "turn": turn,
                        "ts": ts,
                        "tool": str(call.get("tool") or "tool"),
                        "query": _redact_transcript_value(call.get("query")),
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist intel append failed for task_id=%s turn=%s",
                task_id,
                turn,
                exc_info=True,
            )

    def _record_specialist_conversation(
        self,
        *,
        task_id: str,
        turn: int,
        metadata: dict[str, Any] | None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for an in-process specialist turn."""
        if self.session_dir is None:
            return
        try:
            md = metadata or {}
            prompt = md.get("prompt")
            response = md.get("response")
            if not prompt and not response:
                return
            record = ConversationRecord(
                session_id=self.session_dir.name,
                component="specialist",
                # Same metadata dict as the token row for this turn, so both halves carry the backend's call_id when
                # it stamped one.
                call_id=md.get("call_id"),
                task_id=task_id,
                turn=turn,
                tick=tick,
                phase=phase,
                model=md.get("model"),
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist conversation append failed for task_id=%s turn=%s",
                task_id,
                turn,
                exc_info=True,
            )

    # In-process Backend path (test path)
    async def _run_via_backend(
        self,
        ctx: RunnerContext,
        prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Drive ``Backend.run`` one turn at a time until a specialist_done intent shows up."""
        assert self.backend_factory is not None  # narrowed by run()
        domain = prep.domain
        gap = prep.gap
        workspace = prep.workspace
        max_turns = prep.max_turns
        notes = list(prep.notes)

        try:
            backend = self.backend_factory(domain)
        except Exception as exc:  # noqa: BLE001 — backend init failure
            done = build_empty_specialist_done(
                gap_canonical_id=gap,
                domain=domain.key,
                reason=f"backend_init_failed: {exc!r}",
            )
            self._write_specialist_done(workspace, done)
            return SpecialistRunResult(
                task_id=ctx.task.task_id,
                domain=domain.key,
                gap_canonical_id=gap,
                status="empty_synthesised",
                specialist_done=done,
                turns_used=0,
                workspace=str(workspace) if workspace else "",
                error=f"backend_init_failed:{exc!r}",
                notes=notes + ["backend_init_failed"],
            )

        # Combined prompt so backends ignoring ``system_prompt`` still see it inline.
        combined_prompt = prep.system_prompt + "\n---\n" + prep.user_prompt

        specialist_done_intent: Intent | None = None
        tool_violations: list[str] = []
        turns_used = 0
        backend_error: str = ""

        for turn_idx in range(1, max_turns + 1):
            turns_used = turn_idx
            try:
                self._write_heartbeat(
                    workspace,
                    turn=turn_idx,
                    max_turns=max_turns,
                    status="running",
                )
                _t0 = time.perf_counter()
                turn_result = await backend.run(
                    prompt=prep.user_prompt if turn_idx == 1 else combined_prompt,
                    system_prompt=prep.system_prompt,
                    disallowed_tools=list(SPECIALIST_TOOL_DENYLIST),
                    max_turns=1,
                )
                _turn_latency_ms = int((time.perf_counter() - _t0) * 1000)
            except BackendError as exc:
                backend_error = f"backend_error:{exc!r}"
                self._append_transcript(
                    workspace,
                    turn_idx,
                    {
                        "type": "backend_error",
                        "error": str(exc),
                    },
                )
                if isinstance(exc, LLMCallFailed):
                    _tick, _phase = self._ctx_tick_phase(ctx)
                    self._trace_specialist_llm_failure(
                        task_id=ctx.task.task_id,
                        turn=turn_idx,
                        error=exc,
                        latency_ms=int((time.perf_counter() - _t0) * 1000),
                        tick=_tick,
                        phase=_phase,
                    )
                break
            except Exception as exc:  # noqa: BLE001 — defensive
                backend_error = f"backend_unexpected:{exc!r}"
                self._append_transcript(
                    workspace,
                    turn_idx,
                    {
                        "type": "backend_unexpected",
                        "error": repr(exc),
                    },
                )
                break

            self._append_transcript(
                workspace,
                turn_idx,
                {
                    "type": "turn",
                    "intents": [{"intent_type": i.type.value, "payload": i.payload} for i in turn_result.intents],
                    "raw_text_preview": _safe_redact(turn_result.raw_text[:1024]),
                    "metadata": dict(turn_result.metadata),
                },
            )
            # Mirror the turn's token spend onto the unified LLM-call ledger.
            _tick, _phase = self._ctx_tick_phase(ctx)
            self._trace_specialist_llm_call(
                task_id=ctx.task.task_id,
                turn=turn_idx,
                metadata=turn_result.metadata,
                latency_ms=_turn_latency_ms,
                tick=_tick,
                phase=_phase,
            )
            self._record_specialist_conversation(
                task_id=ctx.task.task_id,
                turn=turn_idx,
                metadata=turn_result.metadata,
                tick=_tick,
                phase=_phase,
            )

            # Tool-violation check (defense in depth).
            for intent in turn_result.intents:
                if intent.type == IntentType.SPECIALIST_DONE:
                    specialist_done_intent = intent
                elif intent.type in (
                    IntentType.SEND_MESSAGE,
                    IntentType.ALERT,
                ):
                    continue
                else:
                    tool_violations.append(intent.type.value)

            # WS1 incremental checkpoint: rewrite the partial after every turn so a budget kill leaves the best-so-far
            # result on disk.
            if specialist_done_intent is not None:
                self._write_specialist_done_partial(
                    workspace,
                    dict(specialist_done_intent.payload or {}),
                )
            else:
                self._write_specialist_done_partial(
                    workspace,
                    {
                        **build_empty_specialist_done(
                            gap_canonical_id=gap,
                            domain=domain.key,
                            reason="in_progress",
                        ),
                        "turns_used": turns_used,
                    },
                )

            if specialist_done_intent is not None:
                break

        # Final heartbeat
        self._write_heartbeat(
            workspace,
            turn=turns_used,
            max_turns=max_turns,
            status="finished",
        )

        return self._finalize(
            ctx=ctx,
            prep=prep,
            specialist_done_payload=(
                dict(specialist_done_intent.payload or {}) if specialist_done_intent is not None else None
            ),
            turns_used=turns_used,
            tool_violations=tool_violations,
            backend_error=backend_error,
            extra_notes=notes,
            patches_written=[],
        )

    # Subprocess path (production)
    async def _run_via_subprocess(
        self,
        ctx: RunnerContext,
        prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Spawn a per-task ``claude`` subprocess inside the worktree and reap its ``specialist_done.json`` / ``patches/`` output."""
        assert self.subprocess_dispatcher is not None  # narrowed by run()
        domain = prep.domain
        gap = prep.gap
        workspace = prep.workspace
        notes = list(prep.notes)

        if workspace is None:
            done = build_empty_specialist_done(
                gap_canonical_id=gap,
                domain=domain.key,
                reason="subprocess dispatch requires a workspace",
            )
            return SpecialistRunResult(
                task_id=ctx.task.task_id,
                domain=domain.key,
                gap_canonical_id=gap,
                status="empty_synthesised",
                specialist_done=done,
                turns_used=0,
                workspace="",
                error="no_workspace",
                notes=notes + ["no_workspace"],
            )

        self._write_heartbeat(
            workspace,
            turn=1,
            max_turns=prep.max_turns,
            status="subprocess_starting",
        )
        # WS1: explicit wall-clock budget injected by the Coordinator; when present it overrides the legacy
        # ``max_turns × per_turn`` ceiling.
        wall_budget_raw = (ctx.extra or {}).get("wall_budget_sec")
        wall_budget_sec = float(wall_budget_raw) if wall_budget_raw else None
        # Ray-managed GPU execution (§12 T4): when the dispatcher acquired a GpuSpecialistLease, run the whole
        # subprocess inside its num_gpus actor so any GPU command lands within Ray's assigned devices.
        sub_result: SpecialistSubprocessResult = await self.subprocess_dispatcher.run(
            task_id=ctx.task.task_id,
            workspace=workspace,
            worktree=prep.worktree,
            worktree_base=prep.worktree_base,
            system_prompt=prep.system_prompt,
            user_prompt=prep.user_prompt,
            disallowed_tools=SPECIALIST_TOOL_DENYLIST,
            max_turns=prep.max_turns,
            gpu_ids=tuple((ctx.extra or {}).get("gpu_ids") or ()),
            wall_budget_sec=wall_budget_sec,
            gpu_lease=(ctx.extra or {}).get("gpu_specialist_lease"),
            progress_cb=(ctx.extra or {}).get("specialist_progress_cb"),
        )
        self._append_transcript(
            workspace,
            1,
            {
                "type": "subprocess_result",
                "exit_code": sub_result.exit_code,
                "elapsed_seconds": sub_result.elapsed_seconds,
                "timed_out": sub_result.timed_out,
                "stale_heartbeat": sub_result.stale_heartbeat,
                "process_log_path": sub_result.process_log_path,
                "patch_count": len(sub_result.patches),
                "usage": sub_result.usage,
                "error": sub_result.error,
            },
        )
        # Fold the production specialist's token spend into the unified ledger.
        _sub_latency_ms = None
        if sub_result.elapsed_seconds is not None:
            try:
                _sub_latency_ms = int(float(sub_result.elapsed_seconds) * 1000)
            except (TypeError, ValueError):
                _sub_latency_ms = None
        _tick, _phase = self._ctx_tick_phase(ctx)
        # Prefer per-turn token rows when available; whole-session latency lands on the final turn.
        turn_usages = list(sub_result.turn_usages or [])
        if len(turn_usages) > 1:
            last_idx = len(turn_usages) - 1
            for i, tu in enumerate(turn_usages):
                md = dict(tu)
                if sub_result.usage and sub_result.usage.get("model"):
                    md.setdefault("model", sub_result.usage.get("model"))
                self._trace_specialist_llm_call(
                    task_id=ctx.task.task_id,
                    turn=i + 1,
                    metadata=md,
                    latency_ms=_sub_latency_ms if i == last_idx else None,
                    tick=_tick,
                    phase=_phase,
                )
        else:
            self._trace_specialist_llm_call(
                task_id=ctx.task.task_id,
                turn=1,
                metadata=sub_result.usage,
                latency_ms=_sub_latency_ms,
                tick=_tick,
                phase=_phase,
            )
        # Persist the tool/intel calls the specialist made, backfilled as spans.
        self._record_specialist_intel(
            task_id=ctx.task.task_id,
            turn=1,
            tool_calls=sub_result.tool_calls,
        )
        # Pair the parent-held prompt with the recovered assistant reply so the production specialist turn lands in
        # conversations.jsonl.
        if sub_result.response:
            self._record_specialist_conversation(
                task_id=ctx.task.task_id,
                turn=1,
                metadata={
                    "prompt": (prep.system_prompt + "\n---\n" + prep.user_prompt),
                    "response": sub_result.response,
                },
                tick=_tick,
                phase=_phase,
            )
        self._write_heartbeat(
            workspace,
            turn=1,
            max_turns=prep.max_turns,
            status="finished",
        )

        # Decode subprocess error: backend_error → 'stale', clean miss → empty_synthesised.
        detail = (sub_result.error or "").strip()
        backend_error = ""
        if sub_result.timed_out:
            backend_error = f"subprocess_timeout: {detail}" if detail else "subprocess_timeout"
        elif sub_result.stale_heartbeat:
            backend_error = f"subprocess_stale_heartbeat: {detail}" if detail else "subprocess_stale_heartbeat"
        elif detail:
            backend_error = f"subprocess_error:{detail}"
        elif sub_result.exit_code not in (None, 0) and sub_result.done_payload is None:
            backend_error = f"subprocess_exit_code:{sub_result.exit_code}"

        return self._finalize(
            ctx=ctx,
            prep=prep,
            specialist_done_payload=sub_result.done_payload,
            turns_used=1,
            tool_violations=[],
            backend_error=backend_error,
            extra_notes=notes,
            patches_written=list(sub_result.patches),
            patch_roots=dict(sub_result.patch_roots),
        )

    # Finalize phase (shared)
    def _finalize(
        self,
        *,
        ctx: RunnerContext,
        prep: "_PreparedRun",
        specialist_done_payload: dict[str, Any] | None,
        turns_used: int,
        tool_violations: list[str],
        backend_error: str,
        extra_notes: list[str],
        patches_written: list[str],
        patch_roots: dict[str, str] | None = None,
    ) -> SpecialistRunResult:
        """Persist the ``specialist_done`` artifact and build the result."""
        domain = prep.domain
        gap = prep.gap
        workspace = prep.workspace
        notes = list(extra_notes)
        gpu_ids = [int(g) for g in ((ctx.extra or {}).get("gpu_ids") or [])]

        if specialist_done_payload is None:
            reason = backend_error or (
                "max_turns_exhausted" if turns_used >= prep.max_turns else "no_specialist_done_emitted"
            )
            done_payload = build_empty_specialist_done(
                gap_canonical_id=gap,
                domain=domain.key,
                reason=reason,
            )
            if gpu_ids:
                done_payload["allocated_gpu_ids"] = list(gpu_ids)
            self._write_specialist_done(workspace, done_payload)
            return SpecialistRunResult(
                task_id=ctx.task.task_id,
                domain=domain.key,
                gap_canonical_id=gap,
                status="empty_synthesised" if not backend_error else "stale",
                specialist_done=done_payload,
                turns_used=turns_used,
                workspace=str(workspace) if workspace else "",
                error=backend_error or reason,
                notes=notes + ([f"tool_violations:{tool_violations}"] if tool_violations else []),
            )

        # Have a specialist_done payload — sanitise and persist.
        done_payload = dict(specialist_done_payload)
        # Re-stamp gap_canonical_id/domain so the on-disk artifact is authoritative.
        done_payload["gap_canonical_id"] = gap or done_payload.get("gap_canonical_id", "")
        done_payload["domain"] = domain.key
        if gpu_ids:
            done_payload["allocated_gpu_ids"] = list(gpu_ids)
        if "proposal_set" not in done_payload:
            done_payload["proposal_set"] = []
        if "empty" not in done_payload:
            done_payload["empty"] = not bool(done_payload["proposal_set"])
        if "summary" not in done_payload:
            done_payload["summary"] = "specialist emitted done without summary"[:480]
        # Reconcile self-reported ``patches_written`` against the filesystem: keep only claimed paths that exist on
        # disk, then union with the scan.
        claimed = done_payload.get("patches_written") or []
        if not isinstance(claimed, list):
            claimed = []
        search_bases = [b for b in (prep.worktree, workspace) if b is not None]

        def _resolve_existing_patch(p: Any) -> str | None:
            """Resolve a claimed patch path against known search bases."""
            raw = Path(str(p))
            candidates = [raw] if raw.is_absolute() else []
            for base in search_bases:
                candidates.append(base / raw)
            for c in candidates:
                try:
                    if c.is_file() and _patch_path_within_bases(c, search_bases):
                        return str(c)
                except OSError:
                    continue
            return None

        validated: list[str] = []
        missing: list[str] = []
        for p in claimed:
            resolved = _resolve_existing_patch(p)
            if resolved is not None:
                validated.append(resolved)
            else:
                missing.append(str(p))
        dropped_scanned_outside: list[str] = []
        for p in patches_written:
            if not _patch_path_within_bases(Path(str(p)), search_bases):
                dropped_scanned_outside.append(str(p))
                log.warning(
                    "specialist: scanned patch %r resolves outside the specialist worktree/workspace; dropping",
                    p,
                )
                continue
            if p not in validated:
                validated.append(p)
        _seen: set[str] = set()
        deduped: list[str] = []
        for p in validated:
            if p not in _seen:
                _seen.add(p)
                deduped.append(p)
        if missing:
            # Record dangling patch claims for the session_breakdown audit.
            notes.append("patches_claimed_but_missing:" + ",".join(missing[:8]))
        if dropped_scanned_outside:
            notes.append("patches_scanned_outside_workspace:" + ",".join(dropped_scanned_outside[:8]))

        # Stamp the dispatch scope onto every proposal for cross-domain Critic enrichment.
        for _proposal in done_payload.get("proposal_set") or []:
            if isinstance(_proposal, dict):
                _proposal.setdefault("scope", prep.profile.scope)

        # Universal patch-safety gate: drop non-diff/escaping patches, git-ground the rest, and scan for smuggled
        # claims.
        collected_roots = dict(patch_roots or {})
        base_checkout = prep.worktree_base or prep.worktree
        candidate_roots = _sibling_checkouts(
            tuple(self.subprocess_config.framework_source_roots) if self.subprocess_config else (),
            base_checkout,
        )
        explicit_root = _grounding_explicit_root(
            declared=str((ctx.task.params or {}).get("framework_source_root") or "").strip(),
            patches=deduped,
            patch_roots=collected_roots,
        )
        kept, dropped, grounding, spans_roots = _patch_safety.vet_patches(
            deduped,
            base_checkout=base_checkout,
            candidate_roots=candidate_roots,
            explicit_root=explicit_root,
        )
        # A set dropped for targets no tree holds is a distinct outcome from "the specialist wrote none", and the next
        # round has to be told which.
        all_dropped_by_grounding = bool(
            deduped
            and not kept
            and all(
                d.get("verdict") in (_patch_safety.GROUND_MISSING_TARGET, _patch_safety.GROUND_AMBIGUOUS_ROOT)
                for d in dropped
            )
        )
        numeric_warnings = _patch_safety.scan_numeric_claims(done_payload)
        # Strip, do not forward: the Critic is instructed to reject the whole proposal_set over these fields, which
        # costs the round every idea the specialist produced.
        forbidden_fields = _patch_safety.strip_forbidden_proposal_fields(done_payload)
        safety = _patch_safety.PatchSafetyReport(
            kept_patches=kept,
            dropped=dropped,
            grounding=grounding,
            numeric_warnings=numeric_warnings,
            forbidden_fields=forbidden_fields,
        )
        done_payload["patches_written"] = kept
        done_payload["patch_grounding"] = grounding
        if collected_roots:
            done_payload["patch_roots"] = {p: r for p, r in collected_roots.items() if p in kept}
        if all_dropped_by_grounding:
            done_payload["patches_dropped_by_grounding"] = [d["detail"] for d in dropped[:8]]
        if spans_roots:
            done_payload["patches_span_multiple_roots"] = True
        if not kept:
            done_payload["empty"] = not bool(done_payload.get("proposal_set"))
        notes.extend(safety.notes())
        recovered = bool(done_payload.get("_recovered_from_partial"))
        # ``partial`` keeps an infra failure visible without making the attempt retry-eligible, which would discard
        # whatever was salvaged.
        status = "succeeded"
        if tool_violations:
            status = "tool_violation"
            notes.append(f"tool_violations:{tool_violations}")
        elif backend_error or recovered:
            status = "partial"
        if recovered:
            notes.append("recovered_from_partial")
        if notes:
            done_payload["_specialist_notes"] = list(notes)

        self._write_specialist_done(workspace, done_payload)

        return SpecialistRunResult(
            task_id=ctx.task.task_id,
            domain=domain.key,
            gap_canonical_id=gap,
            status=status,
            specialist_done=done_payload,
            turns_used=turns_used,
            workspace=str(workspace) if workspace else "",
            transcript_path=str(self._transcript_path(workspace)) if workspace else "",
            done_path=str(self._done_path(workspace)) if workspace else "",
            error=backend_error,
            notes=notes,
        )

    # Worktree helpers
    def _maybe_setup_worktree(
        self,
        ctx: RunnerContext,
        *,
        workspace: Path | None,
        profile: SpecialistProfile | None = None,
    ) -> tuple[Path | None, Path | None, str]:
        """Provision a per-task git worktree when in subprocess mode."""
        if self.subprocess_config is None or workspace is None:
            return None, None, ""
        if profile is not None:
            if profile.mode != MODE_PATCH:
                return None, None, ""
        base = _pick_worktree_base(
            self.subprocess_config.framework_source_roots,
            preferred=resolve_framework_tree(str((ctx.task.params or {}).get("framework") or "")),
        )
        if base is None:
            return None, None, "no_git_framework_source_root"
        worktree_path = workspace / "worktree"
        branch = f"specialist-{ctx.task.task_id}"
        wt, err = _setup_worktree(base, worktree_path, branch)
        if wt is None:
            return None, base, err
        return wt, base, ""

    # Workspace file protocol
    def _resolve_workspace(self, ctx: RunnerContext) -> Path | None:
        """Resolve (and create) the workspace directory for a run."""
        extra = getattr(ctx, "extra", None) or {}
        ws = extra.get("workspace")
        if ws:
            p = Path(str(ws))
            p.mkdir(parents=True, exist_ok=True)
            return p
        if self.session_dir is None:
            return None
        p = runs_dir(self.session_dir, "specialist", ctx.task.task_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _prompt_path(self, workspace: Path | None) -> Path | None:
        """Return the ``prompt.md`` path within the workspace."""
        return (workspace / "prompt.md") if workspace else None

    def _transcript_path(self, workspace: Path | None) -> Path | None:
        """Return the ``transcript.jsonl`` path within the workspace."""
        return (workspace / "transcript.jsonl") if workspace else None

    def _heartbeat_path(self, workspace: Path | None) -> Path | None:
        """Return the ``heartbeat.json`` path within the workspace."""
        return (workspace / "heartbeat.json") if workspace else None

    def _done_path(self, workspace: Path | None) -> Path | None:
        """Return the ``specialist_done.json`` path within the workspace."""
        return (workspace / "specialist_done.json") if workspace else None

    def _partial_done_path(self, workspace: Path | None) -> Path | None:
        """Return the ``specialist_done.partial.json`` path in the workspace."""
        return (workspace / "specialist_done.partial.json") if workspace else None

    def _write_prompt(
        self,
        workspace: Path | None,
        system: str,
        user: str,
    ) -> None:
        """Write the combined system/user prompt to ``prompt.md``."""
        path = self._prompt_path(workspace)
        if path is None:
            return
        text = "<!-- system_prompt -->\n" + system + "\n<!-- user_prompt -->\n" + user + "\n"
        path.write_text(text, encoding="utf-8")

    def _append_transcript(
        self,
        workspace: Path | None,
        turn: int,
        entry: dict[str, Any],
    ) -> None:
        """Append one JSON line to the workspace ``transcript.jsonl``."""
        path = self._transcript_path(workspace)
        if path is None:
            return
        safe_entry = _redact_transcript_value(entry)
        line = json.dumps(
            {
                "turn": turn,
                "ts": _now_iso(),
                **safe_entry,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_heartbeat(
        self,
        workspace: Path | None,
        *,
        turn: int,
        max_turns: int,
        status: str,
    ) -> None:
        """Atomically write the workspace ``heartbeat.json``."""
        path = self._heartbeat_path(workspace)
        if path is None:
            return
        payload = {
            "ts": _now_iso(),
            "ts_unix": time.time(),
            "turn": turn,
            "max_turns": max_turns,
            "status": status,
        }
        _common_io.atomic_write_json(path, payload, indent=None, sort_keys=True, make_parents=False)

    def _write_specialist_done(
        self,
        workspace: Path | None,
        payload: dict[str, Any],
    ) -> None:
        """Write the ``specialist_done.json`` artifact with a timestamp."""
        path = self._done_path(workspace)
        if path is None:
            return
        _common_io.atomic_write_json(path, {"ts": _now_iso(), **payload}, make_parents=False)

    def _write_specialist_done_partial(
        self,
        workspace: Path | None,
        payload: dict[str, Any],
    ) -> None:
        """Atomically (re)write the incremental checkpoint partial."""
        path = self._partial_done_path(workspace)
        if path is None:
            return
        _common_io.atomic_write_json(
            path,
            {"ts": _now_iso(), "_recovered_from_partial": True, **payload},
            make_parents=False,
        )


__all__ = [
    "RETRYABLE_SPECIALIST_FAILURES",
    "SPECIALIST_TOOL_DENYLIST",
    "SpecialistFailureType",
    "SpecialistRunResult",
    "SpecialistRunner",
    "SpecialistSubprocessConfig",
    "build_empty_specialist_done",
    "classify_specialist_failure",
]
