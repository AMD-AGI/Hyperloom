# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SpecialistRunner — v0.8 M5.

LLM sub-agent runner for ``delegate{action_name='specialist', ...}``
(vs the deterministic Python executors of :class:`SubAgentRunner`).

Inv-5.3 single-exit: every exit path synthesises a ``specialist_done``
payload so the EXPLORE round never blocks; ``status`` carries the original
outcome for the audit trail.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_paths import runs_dir
from .backends.base import BackendError
from ..protocol.intent import Intent, IntentType
from .specialist_bench import BENCH_TOOL_ENABLED, TOOL_RUN_BENCH
from .trace.conversation_trace import ConversationRecord, append_conversation
from .trace.llm_trace import LLMCallRecord, append_llm_call
from .specialist_domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    FREEFORM_DOMAIN,
    SPECIALIST_DOMAINS_M5,
    SpecialistDomain,
    get_domain,
    normalize_dispatch_tags,
)
from .specialist_subprocess import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    SpecialistSubprocessResult,
    _pick_worktree_base,
    _setup_worktree,
)
from . import specialist_patch_safety as _patch_safety
from .policy import DEFAULT_SPECIALIST_MAX_PROPOSALS
from .specialist_profile import SpecialistProfile, resolve_specialist_profile
from .sub_agent_runner import RunnerContext, SubAgentResult
from .system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


log = logging.getLogger(__name__)


def _extra_focus_tags(
    params: dict[str, Any], domain: "SpecialistDomain",
) -> tuple[str, ...]:
    """Knowledge-domain tags beyond the primary domain's anchor (anchor dropped to avoid double-rendering)."""
    tags = normalize_dispatch_tags(params)
    primary_anchor = (domain.kb_anchor or "").strip()
    return tuple(t for t in tags if t and t != primary_anchor)


# R4/R5 — canonical external tool registry lives in :mod:`policy`; re-exported here for legacy importers.
from .policy import (
    CORTEX_KB_READ_TOOL_NAMES as _CORTEX_KB_READ,
    KB_WRITE_TOOL_NAMES as _KB_WRITE,
    PR_MONITOR_TOOL_NAMES as _PR_MONITOR,
    WEB_TOOL_NAMES as _WEB,
)

#: Back-compat tuple alias for the PR Monitor MCP readonly tools (tuple-typed for positional iteration).
PR_MONITOR_MCP_TOOLS: tuple[str, ...] = tuple(sorted(_PR_MONITOR))

#: Back-compat tuple alias for the Cortex KB readonly MCP tools.
CORTEX_KB_READONLY_MCP_TOOLS: tuple[str, ...] = tuple(sorted(_CORTEX_KB_READ))


# Default tool whitelist for specialists. Write tools are worktree-scoped
# via ``--add-dir <worktree>``; ``integrate_patch`` is the only path that
# applies patches to the serving workspace.
DEFAULT_SPECIALIST_TOOLS: tuple[str, ...] = (
    "emit_intent",
    "Read", "Grep", "Glob",
    # Patch authoring tools, confined to the specialist worktree.
    "Edit", "Write", "MultiEdit",
    # Restricted Bash — runners may further filter via a callback. The
    # runner's per-call hook (TODO) will block destructive invocations.
    "Bash",
) + tuple(sorted(_WEB)) + PR_MONITOR_MCP_TOOLS


# Tools explicitly denied even if the operator extends the whitelist.
SPECIALIST_TOOL_DENYLIST: frozenset[str] = frozenset(_KB_WRITE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _safe_redact(s: str) -> str:
    """Redact obvious secrets from a transcript line before writing to disk."""
    out = s
    for needle in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"):
        if needle in out:
            out = out.replace(needle, f"{needle}[REDACTED]")
    return out


@dataclass
class _PreparedRun:
    """Shared setup-phase output threaded into both execution paths."""

    domain: SpecialistDomain | None = None
    # Per-domain sub_kind selected at dispatch (empty = default prompt).
    sub_kind: str = ""
    gap: str = ""
    max_turns: int = 0
    # Resolved dispatch profile (scope / mode / bench / lane).
    profile: "SpecialistProfile" = field(default_factory=lambda: SpecialistProfile())
    workspace: Path | None = None
    worktree: Path | None = None
    worktree_base: Path | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    notes: list[str] = field(default_factory=list)
    resolved_tools: tuple[str, ...] = ()
    # When set, the caller returns this verbatim and skips execute
    # (e.g. unknown domain / missing workspace).
    early_return: "SpecialistRunResult | None" = None


@dataclass
class SpecialistRunResult:
    """Internal record of one SpecialistRunner invocation.

    Distinct from :class:`SubAgentResult` so the runner-level status is
    separable from the task state.
    """

    task_id: str
    domain: str
    gap_canonical_id: str
    status: str   # "succeeded" / "stale" / "empty_synthesised" / "tool_violation"
    specialist_done: dict[str, Any]
    turns_used: int = 0
    workspace: str = ""
    error: str = ""
    transcript_path: str = ""
    done_path: str = ""
    notes: list[str] = field(default_factory=list)


class SpecialistFailureType(str, enum.Enum):
    """Coarse failure taxonomy for a finished specialist run.

    Only the *transient infrastructure* members (``TIMEOUT`` /
    ``STALE_HEARTBEAT`` / ``CRASH``) are retry-eligible — a fresh re-dispatch
    can plausibly fix a subprocess that never produced a clean result.
    Semantic outcomes (``NO_OUTPUT`` empty findings, ``TOOL_VIOLATION``,
    ``CONFIG`` bad domain / missing workspace) are left for the orchestrator to
    act on; re-running them verbatim would just burn budget.
    """

    NONE = "none"                       # succeeded
    TIMEOUT = "timeout"                 # subprocess wall-clock kill
    STALE_HEARTBEAT = "stale_heartbeat" # heartbeat went silent (hang)
    CRASH = "crash"                     # nonzero exit / backend error
    NO_OUTPUT = "no_output"             # ran clean but emitted no done / empty
    TOOL_VIOLATION = "tool_violation"   # emitted a forbidden intent
    CONFIG = "config"                   # unknown domain / no workspace
    UNKNOWN = "unknown"


# Transient infra failures a bounded auto-retry may re-dispatch.
RETRYABLE_SPECIALIST_FAILURES: frozenset[SpecialistFailureType] = frozenset({
    SpecialistFailureType.TIMEOUT,
    SpecialistFailureType.STALE_HEARTBEAT,
    SpecialistFailureType.CRASH,
})


def classify_specialist_failure(
    runner_status: str, error: str,
) -> tuple[SpecialistFailureType, bool]:
    """Map a :class:`SpecialistRunResult` ``(status, error)`` to a failure
    type + retry-eligibility flag.

    Pure helper (no I/O) so PolicyGate, the Coordinator auto-retry hook, and
    tests share one taxonomy. ``status == 'stale'`` is the runner's marker for
    a subprocess that died with a ``backend_error`` (timeout / stale-heartbeat
    / crash); ``empty_synthesised`` means it exited cleanly without a usable
    ``specialist_done`` (genuine empty, max-turns, or a config error encoded in
    ``error``).
    """
    status = (runner_status or "").strip().lower()
    err = (error or "").strip().lower()
    if status == "succeeded":
        return SpecialistFailureType.NONE, False
    if status == "tool_violation":
        return SpecialistFailureType.TOOL_VIOLATION, False
    if status == "stale":
        if "timeout" in err:
            ftype = SpecialistFailureType.TIMEOUT
        elif "stale_heartbeat" in err:
            ftype = SpecialistFailureType.STALE_HEARTBEAT
        else:  # subprocess_error / subprocess_exit_code / backend_error
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
    """Return the canonical empty ``specialist_done`` payload.

    Satisfies PolicyGate R3 schema (``empty=true``, ``proposal_set=[]``,
    non-empty summary).
    """
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
    """LLM-driven sub-agent runner for the merged ``specialist`` action.

    Generic over the Backend protocol (MockBackend in tests, ClaudeBackend
    in production).
    """

    def __init__(
        self,
        backend_factory=None,
        *,
        subprocess_config: SpecialistSubprocessConfig | None = None,
        session_dir: Path | None = None,
        default_tools: tuple[str, ...] = DEFAULT_SPECIALIST_TOOLS,
        default_max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS,
        per_turn_max_seconds: float = 90.0,
        knowledge_plane: Any = None,
    ):
        """Create a runner.

        Exactly one of ``backend_factory`` (in-process, tests) /
        ``subprocess_config`` (PR-A2 ``claude`` subprocess, production) must
        be supplied. ``knowledge_plane`` gates ``mcp__pr_monitor__*`` tools.
        """
        if backend_factory is None and subprocess_config is None:
            raise ValueError(
                "SpecialistRunner: pass exactly one of "
                "backend_factory / subprocess_config"
            )
        if backend_factory is not None and subprocess_config is not None:
            raise ValueError(
                "SpecialistRunner: backend_factory and subprocess_config "
                "are mutually exclusive — pick one path"
            )
        self.backend_factory = backend_factory
        self.subprocess_config = subprocess_config
        self.subprocess_dispatcher = (
            SpecialistSubprocessDispatcher(subprocess_config)
            if subprocess_config is not None
            else None
        )
        self.session_dir = Path(session_dir) if session_dir else None
        self.default_tools = tuple(default_tools)
        self.default_max_turns = int(default_max_turns)
        self.per_turn_max_seconds = float(per_turn_max_seconds)
        self.knowledge_plane = knowledge_plane

    def _resolve_tools(
        self,
        task_allowed_tools: list[str] | tuple[str, ...] | None = None,
        *,
        grant_bench: bool = False,
    ) -> tuple[str, ...]:
        """Return the per-task tool whitelist.

        Strips ``mcp__pr_monitor__*`` when PR Monitor is disabled (and
        ``mcp__cortex_kb__*`` when Cortex disabled); honors a narrower
        ``Task.allowed_tools``; grants the worktree-scoped ``run_bench`` tool
        only to bench-enabled specialists (``grant_bench`` and the bench tool
        globally enabled); enforces :data:`SPECIALIST_TOOL_DENYLIST` last.
        """
        tools = (
            list(task_allowed_tools)
            if task_allowed_tools
            else list(self.default_tools)
        )
        # run_bench is granted only to bench-enabled (mode=patch & bench=true)
        # specialists, and never via the operator-narrowed allowlist.
        if grant_bench and BENCH_TOOL_ENABLED and TOOL_RUN_BENCH not in tools:
            tools.append(TOOL_RUN_BENCH)
        elif not grant_bench:
            tools = [t for t in tools if t != TOOL_RUN_BENCH]
        plane = self.knowledge_plane
        if plane is not None:
            try:
                pr_enabled = bool(plane.pr_monitor_enabled)
            except AttributeError:
                pr_enabled = True   # unknown surface; trust default
            if not pr_enabled:
                tools = [t for t in tools if not t.startswith("mcp__pr_monitor__")]
            cortex_enabled = True
            try:
                cortex_enabled = bool(plane.cortex_enabled)
            except AttributeError:
                cortex_enabled = True
            if not cortex_enabled:
                tools = [
                    t for t in tools
                    if not t.startswith("mcp__cortex_kb__")
                ]
        tools = [t for t in tools if t not in SPECIALIST_TOOL_DENYLIST]
        return tuple(tools)

    # Public entry point — dispatches to in-process or subprocess path
    async def run(
        self,
        ctx: RunnerContext,
        *,
        prompt_inputs: SpecialistPromptInputs | None = None,
    ) -> SpecialistRunResult:
        """Run a specialist task to completion (or synthesise an empty done).

        Never raises a Backend error past the boundary — every failure ends
        with a valid specialist_done payload (Inv-5.3 single exit).
        """
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
        params = ctx.task.params or {}
        domain_key = str(params.get("domain") or "").strip()
        gap = str(
            params.get("gap_canonical_id") or params.get("gap") or ""
        ).strip()
        max_turns = int(params.get("max_turns") or self.default_max_turns)
        domain = get_domain(domain_key)
        sub_kind = str(params.get("sub_kind") or "").strip()
        profile = resolve_specialist_profile(params)
        task_description = str(params.get("task_description") or "").strip()

        workspace = self._resolve_workspace(ctx)

        # scope='freeform' (absorbed dynamic_specialist) is not bound to the
        # domain catalogue: use the synthetic freeform domain so dispatch can
        # proceed on the task_description mandate alone.
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

        # Post-M5 domains still dispatch but use the generic prompt template.
        notes: list[str] = []
        if domain.key not in SPECIALIST_DOMAINS_M5:
            notes.append(
                f"domain={domain.key!r} is post-M5 (available_in="
                f"{domain.available_in!r}); using generic prompt template"
            )

        # Worktree — created only under subprocess dispatch; surfaced via
        # ``workspace_path`` so the agent knows where to write patches.
        worktree, worktree_base, worktree_err = self._maybe_setup_worktree(
            ctx, workspace=workspace,
        )
        if worktree_err:
            notes.append(f"worktree_setup_failed:{worktree_err}")
        workspace_for_prompt = worktree or workspace

        allocated_gpu_ids = tuple(
            int(g) for g in ((ctx.extra or {}).get("gpu_ids") or [])
        )

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
                warm_start_pitfalls=list(
                    params.get("warm_start_pitfalls") or []
                ),
                warm_start_lessons=list(
                    params.get("warm_start_lessons") or []
                ),
                pr_feed=list(params.get("pr_feed") or []),
                pr_monitor_available=bool(
                    params.get("pr_monitor_available", True)
                ),
                framework=str(params.get("framework") or ""),
                framework_source_roots=tuple(
                    params.get("framework_source_roots") or ()
                ),
                source_hint_directories=tuple(
                    params.get("source_hint_directories") or ()
                ),
                gpu_type=str(params.get("gpu_type") or ""),
                allocated_gpu_ids=allocated_gpu_ids,
                tp=int(params.get("tp") or 0),
                hbm_gb=float(params.get("hbm_gb") or 0.0),
                peak_tflops=float(params.get("peak_tflops") or 0.0),
                arch_notes=str(params.get("arch_notes") or ""),
                target_gap_notes=str(params.get("target_gap_notes") or ""),
                already_proven=[
                    p for p in (params.get("already_proven") or [])
                    if isinstance(p, dict)
                ],
                research_hints=str(params.get("research_hints") or ""),
                # Workload context warmed from SharedState; zero/empty
                # renders as "(none)" rather than a fabricated default.
                precision=str(params.get("precision") or ""),
                conc=int(params.get("conc") or 0),
                isl=int(params.get("isl") or 0),
                osl=int(params.get("osl") or 0),
                max_model_len=int(params.get("max_model_len") or 0),
                # runtime fingerprint for ``_format_version_note`` to flag
                # version-mismatched lessons; empty when not warmed.
                framework_version=str(params.get("framework_version") or ""),
                workspace_path=(
                    str(workspace_for_prompt) if workspace_for_prompt else ""
                ),
                notes=str(params.get("notes") or ""),
                scope=profile.scope,
                mode=profile.mode,
                bench=profile.bench,
                lane=profile.lane,
                task_description=task_description,
                # Coordinator-injected note when this is a bounded auto-retry
                # of a prior transient (timeout / crash / stale) attempt.
                auto_retry_reason=str(params.get("_auto_retry_reason") or ""),
                # proposal_set self-curation target (policy.py is the
                # source of truth); shapes the prompt, not a hard cap.
                max_proposals=max(1, int(
                    params.get("max_proposals")
                    or DEFAULT_SPECIALIST_MAX_PROPOSALS
                )),
            )

        system_prompt, user_prompt = build_specialist_prompts(prompt_inputs)
        self._write_prompt(workspace, system_prompt, user_prompt)
        self._write_heartbeat(
            workspace, turn=0, max_turns=max_turns, status="starting",
        )

        return _PreparedRun(
            domain=domain,
            sub_kind=sub_kind,
            gap=gap,
            max_turns=max_turns,
            profile=profile,
            workspace=workspace,
            worktree=worktree,
            worktree_base=worktree_base,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            notes=notes,
            resolved_tools=self._resolve_tools(
                getattr(ctx.task, "allowed_tools", None),
                grant_bench=profile.grants_bench_tool,
            ),
        )

    def _trace_specialist_llm_call(
        self,
        *,
        task_id: str,
        turn: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for an in-process specialist turn.

        No-op when ``self.session_dir`` is unset (some test harnesses run
        the runner without a session dir) or the backend reported no token
        counters. Wrapped broadly so a trace failure never aborts the run.
        """
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
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist llm_call append failed for "
                "task_id=%s turn=%s", task_id, turn, exc_info=True,
            )

    def _record_specialist_conversation(
        self,
        *,
        task_id: str,
        turn: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for an in-process specialist
        turn. Persists the full (redacted) prompt + completion the backend put
        on ``metadata``. No-op without a session dir; best-effort otherwise.
        """
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
                task_id=task_id,
                turn=turn,
                model=md.get("model"),
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the run
            log.debug(
                "full-trace: specialist conversation append failed for "
                "task_id=%s turn=%s", task_id, turn, exc_info=True,
            )

    # ------------------------------------------------------------------
    # In-process Backend path (test path)
    async def _run_via_backend(
        self, ctx: RunnerContext, prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Drive ``Backend.run`` one turn at a time until a specialist_done
        intent shows up."""
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

        # Combined prompt so backends that ignore the separate
        # ``system_prompt`` field still see the system content inline.
        combined_prompt = prep.system_prompt + "\n---\n" + prep.user_prompt

        specialist_done_intent: Intent | None = None
        tool_violations: list[str] = []
        turns_used = 0
        backend_error: str = ""

        for turn_idx in range(1, max_turns + 1):
            turns_used = turn_idx
            try:
                self._write_heartbeat(
                    workspace, turn=turn_idx, max_turns=max_turns,
                    status="running",
                )
                turn_result = await backend.run(
                    prompt=prep.user_prompt if turn_idx == 1 else combined_prompt,
                    system_prompt=prep.system_prompt,
                    tools=list(prep.resolved_tools),
                    max_turns=1,
                )
            except BackendError as exc:
                backend_error = f"backend_error:{exc!r}"
                self._append_transcript(workspace, turn_idx, {
                    "type": "backend_error",
                    "error": str(exc),
                })
                break
            except Exception as exc:  # noqa: BLE001 — defensive
                backend_error = f"backend_unexpected:{exc!r}"
                self._append_transcript(workspace, turn_idx, {
                    "type": "backend_unexpected",
                    "error": repr(exc),
                })
                break

            self._append_transcript(workspace, turn_idx, {
                "type": "turn",
                "intents": [
                    {"intent_type": i.type.value, "payload": i.payload}
                    for i in turn_result.intents
                ],
                "raw_text_preview": _safe_redact(turn_result.raw_text[:1024]),
                "metadata": dict(turn_result.metadata),
            })
            # Full-trace A3: in-process specialist fallback. The token
            # spend is already in the transcript's turn metadata; mirror it
            # onto the unified ledger keyed by task_id + turn so the
            # collector can join it to the EXPLORE decision. The production
            # default (subprocess) path is covered separately by B1.
            self._trace_specialist_llm_call(
                task_id=ctx.task.task_id,
                turn=turn_idx,
                metadata=turn_result.metadata,
            )
            self._record_specialist_conversation(
                task_id=ctx.task.task_id,
                turn=turn_idx,
                metadata=turn_result.metadata,
            )

            # Tool-violation check (defense in depth).
            for intent in turn_result.intents:
                if intent.type == IntentType.SPECIALIST_DONE:
                    specialist_done_intent = intent
                elif intent.type in (
                    IntentType.SEND_MESSAGE, IntentType.ALERT,
                ):
                    continue
                else:
                    tool_violations.append(intent.type.value)

            if specialist_done_intent is not None:
                break

        # Final heartbeat
        self._write_heartbeat(
            workspace, turn=turns_used, max_turns=max_turns, status="finished",
        )

        return self._finalize(
            ctx=ctx,
            prep=prep,
            specialist_done_payload=(
                dict(specialist_done_intent.payload or {})
                if specialist_done_intent is not None else None
            ),
            turns_used=turns_used,
            tool_violations=tool_violations,
            backend_error=backend_error,
            extra_notes=notes,
            patches_written=[],
        )

    # Subprocess path (production)
    async def _run_via_subprocess(
        self, ctx: RunnerContext, prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Spawn a per-task ``claude`` subprocess inside the worktree
        and reap its ``specialist_done.json`` / ``patches/`` output."""
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
            workspace, turn=1, max_turns=prep.max_turns, status="subprocess_starting",
        )
        sub_result: SpecialistSubprocessResult = (
            await self.subprocess_dispatcher.run(
                task_id=ctx.task.task_id,
                workspace=workspace,
                worktree=prep.worktree,
                worktree_base=prep.worktree_base,
                system_prompt=prep.system_prompt,
                user_prompt=prep.user_prompt,
                allowed_tools=prep.resolved_tools,
                max_turns=prep.max_turns,
                gpu_ids=tuple((ctx.extra or {}).get("gpu_ids") or ()),
            )
        )
        self._append_transcript(workspace, 1, {
            "type": "subprocess_result",
            "exit_code": sub_result.exit_code,
            "elapsed_seconds": sub_result.elapsed_seconds,
            "timed_out": sub_result.timed_out,
            "stale_heartbeat": sub_result.stale_heartbeat,
            "process_log_path": sub_result.process_log_path,
            "patch_count": len(sub_result.patches),
            "usage": sub_result.usage,
            "error": sub_result.error,
        })
        # Full-trace B1: fold the production specialist's token spend
        # (recovered from the Claude CLI stream-json log by the dispatcher)
        # into the unified ledger. The subprocess runs one logical agent
        # session, so we record turn=1. ``usage`` already uses the four
        # canonical counter names, so the metadata-shaped helper consumes
        # it directly.
        self._trace_specialist_llm_call(
            task_id=ctx.task.task_id,
            turn=1,
            metadata=sub_result.usage,
        )
        self._write_heartbeat(
            workspace, turn=1, max_turns=prep.max_turns, status="finished",
        )

        # Decode subprocess error: backend_error → 'stale', clean miss →
        # empty_synthesised.
        backend_error = ""
        if sub_result.timed_out:
            backend_error = "subprocess_timeout"
        elif sub_result.stale_heartbeat:
            backend_error = "subprocess_stale_heartbeat"
        elif sub_result.error:
            backend_error = f"subprocess_error:{sub_result.error}"
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
    ) -> SpecialistRunResult:
        domain = prep.domain
        gap = prep.gap
        workspace = prep.workspace
        notes = list(extra_notes)
        gpu_ids = [
            int(g) for g in ((ctx.extra or {}).get("gpu_ids") or [])
        ]

        if specialist_done_payload is None:
            reason = backend_error or (
                "max_turns_exhausted" if turns_used >= prep.max_turns
                else "no_specialist_done_emitted"
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
                notes=notes + ([
                    f"tool_violations:{tool_violations}"
                ] if tool_violations else []),
            )

        # Have a specialist_done payload — sanitise and persist.
        done_payload = dict(specialist_done_payload)
        # Re-stamp gap_canonical_id/domain so the on-disk artifact is
        # authoritative.
        done_payload["gap_canonical_id"] = gap or done_payload.get(
            "gap_canonical_id", ""
        )
        done_payload["domain"] = domain.key
        if gpu_ids:
            done_payload["allocated_gpu_ids"] = list(gpu_ids)
        if "proposal_set" not in done_payload:
            done_payload["proposal_set"] = []
        # ``max_proposals`` is a prompt-side target, not a hard cap: the
        # full proposal_set is carried back unmodified.
        if "empty" not in done_payload:
            done_payload["empty"] = not bool(done_payload["proposal_set"])
        if "summary" not in done_payload:
            done_payload["summary"] = (
                "specialist emitted done without summary"[:480]
            )
        # Reconcile self-reported ``patches_written`` against the filesystem:
        # keep only claimed paths that exist on disk (so a dangling claim
        # can't make integrate_patch a silent no-op), then union with the scan.
        claimed = done_payload.get("patches_written") or []
        if not isinstance(claimed, list):
            claimed = []
        search_bases = [b for b in (prep.worktree, workspace) if b is not None]

        def _resolve_existing_patch(p: Any) -> str | None:
            raw = Path(str(p))
            candidates = [raw] if raw.is_absolute() else []
            for base in search_bases:
                candidates.append(base / raw)
            for c in candidates:
                try:
                    if c.is_file():
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
        for p in patches_written:
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
            notes.append(
                "patches_claimed_but_missing:" + ",".join(missing[:8])
            )

        # Stamp the dispatch scope onto every proposal so the cross-domain
        # Critic enrichment fires deterministically for scope=domains (not
        # dependent on the sub-agent self-reporting it).
        for _proposal in done_payload.get("proposal_set") or []:
            if isinstance(_proposal, dict):
                _proposal.setdefault("scope", prep.profile.scope)

        # Universal patch-safety gate (applies to every scope): drop patches
        # that are not real unified diffs / escape the tree, git-ground the
        # rest against the clean base checkout, and scan for smuggled
        # quantitative claims. Stale-but-valid patches are kept (integrate_patch
        # + Critic adjudicate) with a grounding note.
        base_checkout = prep.worktree_base or prep.worktree
        kept, dropped, grounding = _patch_safety.vet_patches(
            deduped, base_checkout=base_checkout,
        )
        forbidden_fields, numeric_warnings = _patch_safety.scan_quantitative_claims(
            done_payload,
        )
        safety = _patch_safety.PatchSafetyReport(
            kept_patches=kept,
            dropped=dropped,
            grounding=grounding,
            numeric_warnings=numeric_warnings,
            forbidden_fields=forbidden_fields,
        )
        done_payload["patches_written"] = kept
        done_payload["patch_grounding"] = grounding
        if not kept:
            done_payload["empty"] = not bool(done_payload.get("proposal_set"))
        notes.extend(safety.notes())

        self._write_specialist_done(workspace, done_payload)
        status = "succeeded"
        if tool_violations:
            status = "tool_violation"
            notes.append(f"tool_violations:{tool_violations}")

        return SpecialistRunResult(
            task_id=ctx.task.task_id,
            domain=domain.key,
            gap_canonical_id=gap,
            status=status,
            specialist_done=done_payload,
            turns_used=turns_used,
            workspace=str(workspace) if workspace else "",
            transcript_path=str(self._transcript_path(workspace))
                if workspace else "",
            done_path=str(self._done_path(workspace)) if workspace else "",
            notes=notes,
        )

    # Worktree helpers
    def _maybe_setup_worktree(
        self, ctx: RunnerContext, *, workspace: Path | None,
    ) -> tuple[Path | None, Path | None, str]:
        """Provision a per-task git worktree when in subprocess mode.

        Returns ``(worktree_dir, worktree_base, error)``; empty/None in
        in-process mode or on git failure. Best-effort: the specialist still
        dispatches without isolation and the reason lands in ``notes``.
        """
        if self.subprocess_config is None or workspace is None:
            return None, None, ""
        params = ctx.task.params or {}
        readonly = bool(params.get("readonly")) or (
            str(params.get("domain") or "").strip() == "research_scout_specialist"
        )
        if readonly:
            return None, None, ""
        base = _pick_worktree_base(self.subprocess_config.framework_source_roots)
        if base is None:
            return None, None, "no_git_framework_source_root"
        worktree_path = workspace / "worktree"
        branch = f"specialist-{ctx.task.task_id}"
        wt, err = _setup_worktree(base, worktree_path, branch)
        if wt is None:
            return None, base, err
        return wt, base, ""

    # Coordinator-facing convenience: produce a SubAgentResult shape.
    @staticmethod
    def to_sub_agent_result(run_result: SpecialistRunResult) -> SubAgentResult:
        """Translate the rich runner result into the dispatcher contract."""
        state = "succeeded" if run_result.status in (
            "succeeded", "empty_synthesised", "tool_violation"
        ) else "failed"
        return SubAgentResult(
            task_id=run_result.task_id,
            state=state,
            result={
                "specialist_done": run_result.specialist_done,
                "runner_status": run_result.status,
                "turns_used": run_result.turns_used,
                "workspace": run_result.workspace,
                "transcript_path": run_result.transcript_path,
                "done_path": run_result.done_path,
                "notes": list(run_result.notes),
            },
            error=run_result.error or None,
        )

    # Workspace file protocol
    def _resolve_workspace(self, ctx: RunnerContext) -> Path | None:
        # Prefer the SubAgentRunner-premkdir'd workspace, else
        # ``runs/specialist/<task_id>/``.
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
        return (workspace / "prompt.md") if workspace else None

    def _transcript_path(self, workspace: Path | None) -> Path | None:
        return (workspace / "transcript.jsonl") if workspace else None

    def _heartbeat_path(self, workspace: Path | None) -> Path | None:
        return (workspace / "heartbeat.json") if workspace else None

    def _done_path(self, workspace: Path | None) -> Path | None:
        return (workspace / "specialist_done.json") if workspace else None

    def _write_prompt(
        self, workspace: Path | None, system: str, user: str,
    ) -> None:
        path = self._prompt_path(workspace)
        if path is None:
            return
        text = (
            "<!-- system_prompt -->\n"
            + system
            + "\n<!-- user_prompt -->\n"
            + user
            + "\n"
        )
        path.write_text(text, encoding="utf-8")

    def _append_transcript(
        self, workspace: Path | None, turn: int, entry: dict[str, Any],
    ) -> None:
        path = self._transcript_path(workspace)
        if path is None:
            return
        line = json.dumps({
            "turn": turn,
            "ts": _now_iso(),
            **entry,
        }, sort_keys=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_heartbeat(
        self, workspace: Path | None, *,
        turn: int, max_turns: int, status: str,
    ) -> None:
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
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _write_specialist_done(
        self, workspace: Path | None, payload: dict[str, Any],
    ) -> None:
        path = self._done_path(workspace)
        if path is None:
            return
        payload_with_ts = {
            "ts": _now_iso(),
            **payload,
        }
        path.write_text(
            json.dumps(payload_with_ts, sort_keys=True, indent=2),
            encoding="utf-8",
        )


__all__ = [
    "CORTEX_KB_READONLY_MCP_TOOLS",
    "DEFAULT_SPECIALIST_TOOLS",
    "PR_MONITOR_MCP_TOOLS",
    "RETRYABLE_SPECIALIST_FAILURES",
    "SPECIALIST_TOOL_DENYLIST",
    "SpecialistFailureType",
    "SpecialistRunResult",
    "SpecialistRunner",
    "SpecialistSubprocessConfig",
    "build_empty_specialist_done",
    "classify_specialist_failure",
]
