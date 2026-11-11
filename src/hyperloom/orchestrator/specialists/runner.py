# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SpecialistRunner.

LLM sub-agent runner for ``delegate{action_name='specialist', ...}``
(vs the deterministic Python executors of :class:`SubAgentRunner`).

Inv-5.3 single-exit: every exit path synthesises a ``specialist_done``
payload so the config-arm round never blocks; ``status`` carries the original
outcome for the audit trail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from hyperloom_common.coerce import first_str, first_text
from hyperloom.common.deadline import Deadline, seconds_until
from hyperloom_common.timeutil import now_iso

from hyperloom.inference_optimizer.session.session_paths import runs_dir, specialist_intel_path
from ..state.experience_citations import normalize_citations, shown_ids
from hyperloom.inference_optimizer.trace.conversation_trace import ConversationRecord, append_conversation
from hyperloom.inference_optimizer.trace.llm_trace import LLMCallRecord, LLMCallSite, append_llm_call, new_call_id
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_LLM_CALL,
    STATUS_COMPLETED,
    STATUS_FAILED,
    llm_call_summary,
    trajectory_span,
)
from .domains import (
    CUSTOM_SPECIALIST_KEY,
    DEFAULT_SPECIALIST_MAX_TURNS,
    FREEFORM_DOMAIN,
    SPECIALIST_DOMAIN_KEYS,
    SpecialistDomain,
    dispatch_domain_key,
    domain_for_tag,
    get_domain,
    normalize_dispatch_tags,
)
from .subprocess_ import SpecialistSubprocessConfig, SpecialistSubprocessDispatcher
from .subprocess_types import UNBOUNDED_REAP_CAP_SEC, SpecialistSpawn, SpecialistSubprocessResult
from .worktrees import link_worktree
from .patch_claims import grounding_explicit_root, patch_path_within_bases, resolve_claimed_patch, sibling_checkouts
from . import patch_safety as _patch_safety
from .profile import MODE_PATCH, SpecialistProfile, resolve_specialist_profile
from .prompt_inputs import SpecialistPromptInputs
from hyperloom.inference_optimizer.tree_snapshot import snapshot_repository, snapshot_tree
from .workspace_files import (
    append_transcript,
    done_path,
    redact_transcript_value,
    transcript_path,
    write_heartbeat,
    write_specialist_done,
)
from hyperloom.inference_optimizer.framework_paths import (
    FrameworkTree,
    framework_apply_tree,
    resolve_framework_tree,
)
from ..loop.sub_agent_runner import RunnerContext
from ..prompts.specialist_prompt_builder import build_specialist_prompts


log = logging.getLogger(__name__)

_T = TypeVar("_T")

NO_GIT_FRAMEWORK_SOURCE_ROOT = "no_git_framework_source_root"


def specialist_patch_preflight_error(params: dict[str, Any] | None, *, framework_repo_path: str = "") -> str:
    """Return the deterministic source-root error for a patch specialist."""
    task_params = params or {}
    domain = get_domain(str(task_params.get("domain") or ""))
    if resolve_specialist_profile(task_params, domain=domain).mode != MODE_PATCH:
        return ""
    source = _worktree_source(task_params, framework_repo_path=framework_repo_path)
    return "" if source is not None else NO_GIT_FRAMEWORK_SOURCE_ROOT


def _worktree_source(params: dict[str, Any], *, framework_repo_path: str = "") -> FrameworkTree | None:
    """The tree the session optimises, which the preflight and the worktree must both read the same way."""
    return framework_apply_tree(
        str(params.get("session_framework_tree") or "").strip()
        or resolve_framework_tree(str(params.get("framework") or ""))
        or str(framework_repo_path or "").strip()
    )


def _scope_readable() -> tuple[Path, ...]:
    """The benchmark directory, and what its prepared workload reads: the user's script and working directory.

    ``HYPERLOOM_BYPASS_SCRIPTS_DIR`` reaches the specialist's environment but no bind
    follows it -- it is not one of ``confinement.DATA_VARIABLES`` -- so in a sandbox the
    variable names a directory that is not there and reading the declaration is ``ENOENT``.
    The script is the model code the specialist rewrites around; without its directory
    bound the specialist ships a generic wrapper for a model it could not read.
    """
    from hyperloom.inference_optimizer.script_workload import benchmark_inputs

    declared = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    return tuple(dict.fromkeys((Path(declared), *benchmark_inputs(Path(declared))))) if declared else ()


def _subprocess_call_outcome(result: SpecialistSubprocessResult) -> tuple[str, dict[str, Any]]:
    """The terminal status and attributes of a specialist subprocess's ``llm.call`` span."""
    attributes: dict[str, Any] = {
        **llm_call_summary(result.trace.usage),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
    }
    try:
        attributes["latency_ms"] = int(float(result.elapsed_seconds) * 1000)
    except (TypeError, ValueError):
        pass
    if result.error or result.timed_out:
        attributes["error_message"] = (result.error or "timed out")[:500]
        return STATUS_FAILED, attributes
    return STATUS_COMPLETED, attributes


def _subprocess_backend_error(result: SpecialistSubprocessResult) -> str:
    """The backend error a reaped specialist subprocess reports; ``""`` when it ended cleanly.

    The classifier keys off the leading token; the reaper's own text is kept
    after it so the reader sees the elapsed/threshold numbers.
    """
    detail = (result.error or "").strip()
    if result.timed_out:
        return f"subprocess_timeout: {detail}" if detail else "subprocess_timeout"
    if result.stale_heartbeat:
        return f"subprocess_stale_heartbeat: {detail}" if detail else "subprocess_stale_heartbeat"
    if detail:
        return f"subprocess_error:{detail}"
    if result.exit_code not in (None, 0) and result.done_payload is None:
        return f"subprocess_exit_code:{result.exit_code}"
    return ""


def _ctx_deadline(ctx: RunnerContext) -> Deadline | None:
    """Read the dispatch deadline off a task context.

    Args:
        ctx: The specialist task context carrying dispatcher-supplied extras.

    Returns:
        Deadline | None: The dispatch deadline; ``None`` is the only spelling
        of unbounded. A duration is never coerced into one, since re-anchoring
        it here would move the bound.

    Raises:
        TypeError: When ``specialist_deadline`` is neither a Deadline nor None.
    """
    value = ctx.extra.get("specialist_deadline")
    if value is None or isinstance(value, Deadline):
        return value
    raise TypeError(f"specialist_deadline must be a Deadline or None, got {type(value).__name__}")


def resolve_specialist_max_turns(raw: Any, *, default: int) -> int:
    """Resolve the specialist turn budget from dispatch params.

    ``None`` and ``0`` both mean unbounded depth bounded by the wall-clock
    budget, implemented as the configured default cap.

    Args:
        raw: The ``max_turns`` value from dispatch params, or ``None``.
        default: The runner default when ``raw`` is omitted or zero.

    Returns:
        The resolved non-negative turn budget.
    """
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
    """Knowledge-domain tags beyond the primary domain's anchor.

    Args:
        params: The dispatch params carrying the tag list.
        domain: The primary specialist domain whose anchor is excluded.

    Returns:
        The extra focus tags, excluding the primary domain's anchor.
    """
    tags = normalize_dispatch_tags(params)
    primary_anchor = (domain.kb_anchor or "").strip()
    return tuple(t for t in tags if t and t != primary_anchor)


def _param(params: dict[str, Any], key: str, kind: type[_T]) -> _T:
    """``params[key]`` as ``kind``; a missing or falsy value is ``kind()``."""
    return kind(params.get(key) or kind())


def _param_dicts(params: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The mapping entries of the ``params[key]`` list."""
    return [entry for entry in (params.get(key) or []) if isinstance(entry, dict)]


def _param_texts(params: dict[str, Any], key: str) -> tuple[str, ...]:
    """The non-blank, stripped entries of the ``params[key]`` list."""
    return tuple(str(entry).strip() for entry in (params.get(key) or ()) if str(entry).strip())


def _workload_prompt_fields(params: dict[str, Any]) -> dict[str, Any]:
    """Prompt inputs read off the task params: model, enablement context, hardware, workload and bench state."""
    return {
        "model_name": _param(params, "model_name", str),
        "model_path": _param(params, "model_path", str),
        "model_info": _param(params, "model_info", dict),
        "static_recon_checklist": _param(params, "static_recon_checklist", str),
        "enablement_source_context": _param(params, "enablement_source_context", str),
        "enablement_candidate_refs": _param_texts(params, "enablement_candidate_refs"),
        "enablement_failure_signature": _param(params, "enablement_failure_signature", dict),
        "enablement_accepted_config": {
            "extra_envs": _param(params, "base_extra_envs", dict),
            "extra_server_args": _param(params, "base_extra_args", str).strip(),
        },
        "gpu_type": _param(params, "gpu_type", str),
        "tp": _param(params, "tp", int),
        "arch_notes": _param(params, "arch_notes", str),
        "target_gap_notes": _param(params, "target_gap_notes", str),
        "already_proven": _param_dicts(params, "already_proven"),
        "recipe_sites": _param_texts(params, "recipe_sites"),
        "research_hints": _param(params, "research_hints", str),
        # Workload context warmed from SharedState.
        "precision": _param(params, "precision", str),
        "conc": _param(params, "conc", int),
        "isl": _param(params, "isl", int),
        "osl": _param(params, "osl", int),
        "max_model_len": _param(params, "max_model_len", int),
        # benchmark_mode selects the AgentX prompt blocks; the corpus
        # shape supplies their numbers.
        "benchmark_mode": _param(params, "benchmark_mode", str),
        "agentx_corpus_shape": _param(params, "agentx_corpus_shape", dict),
        "agentx_grading": _param(params, "agentx_grading", dict),
        "agentx_backend": _param(params, "agentx_backend", str),
        # Runtime fingerprint to flag version-mismatched lessons.
        "framework_version": _param(params, "framework_version", str),
        "notes": _param(params, "notes", str),
        # Coordinator-injected note when this is a bounded auto-retry.
        "auto_retry_reason": _param(params, "_auto_retry_reason", str),
        "baseline_tput": _param(params, "baseline_tput", float),
        "current_tput": _param(params, "current_tput", float),
        "current_launch_args": _param(params, "current_launch_args", str),
        "cumulative_gain_validated": _param(params, "cumulative_gain_validated", float),
        "keep_threshold_pct": _param(params, "keep_threshold_pct", float),
        "applied_stack": _param_dicts(params, "applied_stack"),
        "task_kind": _param(params, "task_kind", str),
        "prior_attempts": _param_dicts(params, "prior_attempts"),
        "pr_lead": _param(params, "pr_lead", dict),
    }


SPECIALIST_TOOL_DENYLIST: frozenset[str] = frozenset(
    {
        "KillShell",
        "SlashCommand",
    }
)


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
    worktree_admin: Path | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    notes: list[str] = field(default_factory=list)
    # When set, the caller returns this verbatim and skips execute.
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
    status: str  # "succeeded" / "partial" / "stale" / "empty_synthesised"
    specialist_done: dict[str, Any]
    workspace: str = ""
    error: str = ""
    transcript_path: str = ""
    done_path: str = ""
    notes: list[str] = field(default_factory=list)


def build_empty_specialist_done(
    *,
    gap_canonical_id: str,
    domain: str,
    reason: str,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Return the canonical empty ``specialist_done`` payload.

    Shape: ``proposal_set=[]`` with a non-empty summary and reason.

    Args:
        gap_canonical_id: Canonical id of the gap the specialist addressed.
        domain: The specialist domain key.
        reason: Why the specialist exited empty (becomes summary/reason).
        confidence: Confidence score, clamped to ``[0.0, 1.0]``.

    Returns:
        The canonical empty ``specialist_done`` payload dict.
    """
    return {
        "gap_canonical_id": gap_canonical_id,
        "domain": domain,
        "proposal_set": [],
        "summary": (reason or "specialist exited empty")[:480],
        "reason": reason or "specialist exited empty",
        "confidence": float(max(0.0, min(1.0, confidence))),
        "new_findings": [],
        "residual_questions": [],
    }


class SpecialistRunner:
    """LLM-driven sub-agent runner for the merged ``specialist`` action.

    Each task runs the selected agent CLI as a subprocess in its own workspace.
    """

    def __init__(
        self,
        *,
        subprocess_config: SpecialistSubprocessConfig,
        session_dir: Path | None = None,
        default_max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS,
    ):
        """Create a runner.

        Args:
            subprocess_config: Subprocess spawn config.
            session_dir: Session output directory.
            default_max_turns: Default per-task max turn budget.
        """
        self.subprocess_config = subprocess_config
        self.subprocess_dispatcher = SpecialistSubprocessDispatcher(subprocess_config)
        self.session_dir = Path(session_dir) if session_dir else None
        self.default_max_turns = int(default_max_turns)

    # Public entry point
    async def run(self, ctx: RunnerContext) -> SpecialistRunResult:
        """Run a specialist task to completion (or synthesise an empty done).

        Never raises a Backend error past the boundary — every failure ends
        with a valid specialist_done payload (Inv-5.3 single exit).

        Args:
            ctx: The runner context for this specialist task.

        Returns:
            The :class:`SpecialistRunResult` for the task.
        """
        prep = await self._prepare(ctx)
        if prep.early_return is not None:
            return prep.early_return

        return await self._run_via_subprocess(ctx, prep)

    # Setup phase (shared)
    async def _prepare(self, ctx: RunnerContext) -> "_PreparedRun":
        """Run the shared setup phase before dispatch.

        Resolves the domain, gap, turn budget and workspace, optionally
        provisions a git worktree, assembles the system/user prompts and
        writes the initial prompt + heartbeat artifacts.

        Args:
            ctx (RunnerContext): Dispatch context carrying the task.

        Returns:
            _PreparedRun: The setup bundle; ``early_return`` is set when the
                execute phase should be skipped (e.g. unknown domain).
        """
        params = ctx.task.params or {}
        domain_key = dispatch_domain_key(params)
        gap = first_text(params.get("gap_canonical_id"), params.get("gap"))
        max_turns = resolve_specialist_max_turns(params.get("max_turns"), default=self.default_max_turns)
        domain = domain_for_tag(domain_key)
        profile = resolve_specialist_profile(params, domain=domain)

        workspace = self._resolve_workspace(ctx)

        # scope='freeform' is not bound to the domain catalogue; use the synthetic freeform domain.
        if domain is None and profile.is_freeform:
            domain = FREEFORM_DOMAIN

        if domain is None:
            return self._unknown_domain_run(ctx, workspace, domain_key, gap)

        # Domains outside the catalogue (today only the synthetic freeform
        # domain) still dispatch, just without a per-domain focus block.
        notes: list[str] = []
        if domain.key not in SPECIALIST_DOMAIN_KEYS:
            notes.append(f"domain={domain.key!r} is outside the domain catalogue; using generic prompt template")

        # The operator-defined specialist reads its description and focus from the session.
        shared_state = ctx.extra.get("shared_state")
        custom_specialist_prompt = first_str(getattr(shared_state, "custom_specialist_prompt", ""))
        custom_specialist_description = first_str(getattr(shared_state, "custom_specialist_description", ""))
        if domain.key == CUSTOM_SPECIALIST_KEY and custom_specialist_description:
            domain = replace(domain, description=custom_specialist_description)

        # Worktree — created only under subprocess dispatch; surfaced via ``workspace_path``.
        worktree, worktree_source, worktree_err, worktree_admin = await asyncio.to_thread(
            self._maybe_setup_worktree,
            ctx,
            workspace=workspace,
            profile=profile,
        )
        if worktree_err:
            notes.append(f"worktree_setup_failed:{worktree_err}")
        worktree_base = worktree_source.root if worktree_source is not None else None

        prep = _PreparedRun(
            domain=domain,
            gap=gap,
            max_turns=max_turns,
            profile=profile,
            workspace=workspace,
            worktree=worktree,
            worktree_base=worktree_base,
            worktree_admin=worktree_admin,
            notes=notes,
        )
        prep.system_prompt, prep.user_prompt = build_specialist_prompts(
            self._prompt_inputs(ctx, prep, worktree_source, custom_specialist_prompt)
        )
        write_heartbeat(
            workspace,
            turn=0,
            max_turns=max_turns,
            status="starting",
        )
        return prep

    def _unknown_domain_run(
        self, ctx: RunnerContext, workspace: Path | None, domain_key: str, gap: str
    ) -> "_PreparedRun":
        """Write an empty done for a task naming no known domain; the setup that skips execute."""
        done = build_empty_specialist_done(
            gap_canonical_id=gap,
            domain=domain_key,
            reason=f"unknown specialist domain={domain_key!r}",
        )
        write_specialist_done(workspace, done)
        return _PreparedRun(
            early_return=SpecialistRunResult(
                task_id=ctx.task.task_id,
                domain=domain_key,
                gap_canonical_id=gap,
                status="empty_synthesised",
                specialist_done=done,
                workspace=str(workspace) if workspace else "",
                error="unknown_domain",
                notes=[f"unknown_domain:{domain_key!r}"],
            )
        )

    @staticmethod
    def _prompt_inputs(
        ctx: RunnerContext, prep: "_PreparedRun", worktree_source: FrameworkTree | None, custom_specialist_prompt: str
    ) -> SpecialistPromptInputs:
        """Assemble the specialist prompt inputs for the setup ``prep`` resolved."""
        params = ctx.task.params or {}
        workspace_for_prompt = prep.worktree or prep.workspace
        return SpecialistPromptInputs(
            domain=prep.domain,
            max_turns=prep.max_turns,
            gap_canonical_id=prep.gap,
            gap_symptom=_param(params, "gap_symptom", str),
            gap_layer=_param(params, "gap_layer", str),
            gap_evidence=_param(params, "gap_evidence", dict),
            experience_kb_block=_param(params, "experience_kb_block", str),
            # Coordinator-populated roofline pre-fetch; empty when not warmed.
            roofline_evidence=_param(params, "roofline_evidence", dict),
            runtime_findings=_param(params, "runtime_findings", str),
            extra_focus_tags=_extra_focus_tags(params, prep.domain),
            warm_start_recipe=_param(params, "warm_start_recipe", dict),
            warm_start_pitfalls=_param(params, "warm_start_pitfalls", list),
            warm_start_lessons=_param(params, "warm_start_lessons", list),
            pr_monitor_available=bool(params.get("pr_monitor_available", True)),
            framework=_param(params, "framework", str),
            session_framework_tree=_param(params, "session_framework_tree", str),
            framework_source_roots=_param(params, "framework_source_roots", tuple),
            worktree_base=str(prep.worktree_base) if prep.worktree is not None else "",
            worktree_package_dir=(
                str(worktree_source.tree.relative_to(worktree_source.root)) if prep.worktree is not None else ""
            ),
            source_hint_directories=_param(params, "source_hint_directories", tuple),
            allocated_gpu_ids=tuple(int(g) for g in (ctx.extra.get("gpu_ids") or [])),
            workspace_path=(str(workspace_for_prompt) if workspace_for_prompt else ""),
            scope=prep.profile.scope,
            mode=prep.profile.mode,
            task_description=_param(params, "task_description", str).strip(),
            # The same instant the reaper kills at, as a duration.
            wall_budget_sec=seconds_until(
                _ctx_deadline(ctx),
                unbounded_cap=UNBOUNDED_REAP_CAP_SEC,
            ),
            started_at_iso=datetime.now(timezone.utc).isoformat(),
            custom_specialist_prompt=custom_specialist_prompt,
            **_workload_prompt_fields(params),
        )

    @staticmethod
    def _ctx_tick_phase(ctx: "RunnerContext | None") -> tuple[int | None, str | None]:
        """(tick, phase) from the live SharedState on ``ctx.extra``.

        Returns ``(None, None)`` when unavailable.
        """
        ss = (ctx.extra if ctx is not None else {}).get("shared_state")
        if ss is None:
            return None, None
        tick = ss.tick
        phase = ss.phase
        return (
            int(tick) if tick is not None else None,
            (str(phase) or None) if phase else None,
        )

    def _trace_specialist_llm_call(
        self,
        ctx: RunnerContext,
        turn: int,
        metadata: dict[str, Any] | None,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a specialist turn.

        No-op when ``self.session_dir`` is unset or the backend reported no
        token counters. ``latency_ms`` is the measured wall-clock of the turn or
        the whole subprocess session.

        Args:
            ctx: Dispatch context carrying the specialist task and the live
                state its tick and phase are read from.
            turn: The turn index being traced.
            metadata: Backend turn metadata carrying token counters.
            latency_ms: Measured wall-clock of the turn, when available.
        """
        if self.session_dir is None:
            return
        task_id = ctx.task.task_id
        tick, phase = self._ctx_tick_phase(ctx)
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
        site = LLMCallSite(
            session_id=self.session_dir.name,
            component="specialist",
            task_id=task_id,
            tick=tick,
            phase=phase,
            turn=turn,
        )
        record = LLMCallRecord.from_metadata(site, md, latency_ms=latency_ms)
        append_llm_call(session_dir=self.session_dir, record=record)

    def _record_specialist_intel(
        self,
        *,
        task_id: str,
        turn: int,
        tool_calls: list[dict[str, Any]] | None,
    ) -> None:
        """Append the specialist's intel/tool calls to ``specialist_intel.jsonl``.

        One row per recovered ``tool_use`` (``{"tool", "query"}``), stamped with
        ``task_id`` / ``turn`` / ``ts``. No-op without a session dir or when no
        tool calls were recovered.
        """
        if self.session_dir is None or not tool_calls:
            return
        try:
            path = specialist_intel_path(self.session_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = now_iso()
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
                        "query": redact_transcript_value(call.get("query")),
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
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
        """Append one ``conversations.jsonl`` row for a specialist turn.
        Persists the full (redacted) prompt + completion. No-op without a
        session dir.

        Args:
            task_id: The specialist task id.
            turn: The turn index being recorded.
            metadata: Backend turn metadata carrying the prompt + response.
        """
        if self.session_dir is None:
            return
        md = metadata or {}
        prompt = md.get("prompt")
        response = md.get("response")
        if not prompt and not response:
            return
        record = ConversationRecord(
            session_id=self.session_dir.name,
            component="specialist",
            # Same metadata dict as the token row for this turn, so both
            # halves carry the backend's call_id when it stamped one.
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

    async def _run_via_subprocess(
        self,
        ctx: RunnerContext,
        prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Spawn a per-task ``claude`` subprocess inside the worktree
        and reap its ``specialist_done.json`` / ``patches/`` output.

        Args:
            ctx (RunnerContext): Dispatch context carrying the task.
            prep (_PreparedRun): Setup bundle from :meth:`_prepare`.

        Returns:
            SpecialistRunResult: The finalized run outcome.
        """
        domain = prep.domain
        assert domain is not None  # set by a prep that did not short-circuit
        gap = prep.gap
        workspace = prep.workspace

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
                workspace="",
                error="no_workspace",
                notes=prep.notes + ["no_workspace"],
            )

        write_heartbeat(
            workspace,
            turn=1,
            max_turns=prep.max_turns,
            status="subprocess_starting",
        )
        deadline = _ctx_deadline(ctx)
        call_id = new_call_id()
        # The subprocess's tool and compaction events are parsed inside this span, so they inherit the specialist's
        # component/agent and hang off its llm.call rather than the coordinator's session scope.
        with trajectory_span(
            EVENT_LLM_CALL,
            call_id=call_id,
            component="specialist",
            agent=domain.key,
            attributes={"name": domain.key},
        ) as call_span:
            # Ray-managed GPU execution: when the dispatcher acquired a GpuSpecialistLease, run the whole subprocess
            # inside its num_gpus actor so any GPU command lands within Ray's assigned devices. ``None`` keeps the
            # local path (``gpu_ids`` pinned into *_VISIBLE_DEVICES). A sandbox shows the model when it is a local path.
            sub_result: SpecialistSubprocessResult = await self.subprocess_dispatcher.run(
                SpecialistSpawn(
                    task_id=ctx.task.task_id,
                    workspace=workspace,
                    worktree=prep.worktree,
                    worktree_base=prep.worktree_base,
                    worktree_admin=prep.worktree_admin,
                    system_prompt=prep.system_prompt,
                    user_prompt=prep.user_prompt,
                    disallowed_tools=SPECIALIST_TOOL_DENYLIST,
                    gpu_ids=tuple(ctx.extra.get("gpu_ids") or ()),
                    deadline=deadline,
                    gpu_lease=ctx.extra.get("gpu_specialist_lease"),
                    progress_cb=ctx.extra.get("specialist_progress_cb"),
                    readable=(Path(str((ctx.task.params or {}).get("model_path") or "")), *_scope_readable()),
                )
            )
            call_status, call_attributes = _subprocess_call_outcome(sub_result)
            call_span.finish(call_status, **call_attributes)
        self._record_subprocess_run(ctx, prep, sub_result, call_id, call_attributes.get("latency_ms"))
        return await asyncio.to_thread(self._finalize, ctx=ctx, prep=prep, sub_result=sub_result)

    def _record_subprocess_run(
        self,
        ctx: RunnerContext,
        prep: "_PreparedRun",
        sub_result: SpecialistSubprocessResult,
        call_id: str,
        latency_ms: int | None,
    ) -> None:
        """Record what a reaped subprocess left for the session's ledgers, then mark its heartbeat finished."""
        trace = sub_result.trace
        append_transcript(
            prep.workspace,
            1,
            {
                "type": "subprocess_result",
                "exit_code": sub_result.exit_code,
                "elapsed_seconds": sub_result.elapsed_seconds,
                "timed_out": sub_result.timed_out,
                "stale_heartbeat": sub_result.stale_heartbeat,
                "process_log_path": sub_result.process_log_path,
                "patch_count": len(sub_result.patches),
                "usage": trace.usage,
                "error": sub_result.error,
            },
        )
        tick, phase = self._ctx_tick_phase(ctx)
        # Fold the production specialist's token spend into the unified ledger: one row per turn
        # when per-turn usage is available, else one cumulative row; the session latency lands on the last.
        if len(trace.turn_usages) > 1:
            model = (trace.usage or {}).get("model")
            last_idx = len(trace.turn_usages)
            for turn, usage in enumerate(trace.turn_usages, start=1):
                metadata: dict[str, Any] = {**usage, "call_id": call_id}
                if model:
                    metadata.setdefault("model", model)
                self._trace_specialist_llm_call(
                    ctx, turn=turn, metadata=metadata, latency_ms=latency_ms if turn == last_idx else None
                )
        else:
            self._trace_specialist_llm_call(
                ctx,
                turn=1,
                metadata={**trace.usage, "call_id": call_id} if trace.usage else None,
                latency_ms=latency_ms,
            )
        self._record_specialist_intel(task_id=ctx.task.task_id, turn=1, tool_calls=trace.tool_calls)
        # Pair the parent-held prompt with the recovered assistant reply so the
        # production specialist turn lands in conversations.jsonl.
        if trace.response:
            self._record_specialist_conversation(
                task_id=ctx.task.task_id,
                turn=1,
                metadata={"prompt": prep.system_prompt + "\n---\n" + prep.user_prompt, "response": trace.response},
                tick=tick,
                phase=phase,
            )
        write_heartbeat(prep.workspace, turn=1, max_turns=prep.max_turns, status="finished")

    # Finalize phase (shared)
    def _finalize(
        self,
        *,
        ctx: RunnerContext,
        prep: "_PreparedRun",
        sub_result: SpecialistSubprocessResult,
    ) -> SpecialistRunResult:
        """Persist the ``specialist_done`` artifact and build the result.

        Blocking (``git apply --check`` per patch); call via ``asyncio.to_thread``.

        Synthesises an empty payload when none was produced, admits the
        specialist's own through :meth:`_admit_done_payload` otherwise, and
        writes the on-disk ``specialist_done.json``.

        Args:
            ctx (RunnerContext): Dispatch context carrying the task.
            prep (_PreparedRun): Setup bundle from :meth:`_prepare`.
            sub_result (SpecialistSubprocessResult): What the reaped subprocess left behind.

        Returns:
            SpecialistRunResult: The finalized run outcome record.
        """
        domain = prep.domain
        gap = prep.gap
        workspace = prep.workspace
        notes = list(prep.notes)
        backend_error = _subprocess_backend_error(sub_result)
        gpu_ids = [int(g) for g in (ctx.extra.get("gpu_ids") or [])]

        if sub_result.done_payload is None:
            reason = backend_error or "no_specialist_done_emitted"
            done_payload = build_empty_specialist_done(gap_canonical_id=gap, domain=domain.key, reason=reason)
            if gpu_ids:
                done_payload["allocated_gpu_ids"] = list(gpu_ids)
            write_specialist_done(workspace, done_payload)
            return SpecialistRunResult(
                task_id=ctx.task.task_id,
                domain=domain.key,
                gap_canonical_id=gap,
                status="empty_synthesised" if not backend_error else "stale",
                specialist_done=done_payload,
                workspace=str(workspace) if workspace else "",
                error=reason,
                notes=notes,
            )

        done_payload = dict(sub_result.done_payload)
        # Re-stamp gap_canonical_id/domain so the on-disk artifact is authoritative.
        done_payload["gap_canonical_id"] = gap or done_payload.get("gap_canonical_id", "")
        done_payload["domain"] = domain.key
        if gpu_ids:
            done_payload["allocated_gpu_ids"] = list(gpu_ids)
        done_payload.setdefault("proposal_set", [])
        done_payload.setdefault("summary", "specialist emitted done without summary")
        notes.extend(self._admit_done_payload(ctx, prep, sub_result, done_payload))
        recovered = bool(done_payload.get("_recovered_from_partial"))
        # ``partial`` keeps an infra failure visible without making the attempt
        # retry-eligible, which would discard whatever was salvaged.
        status = "partial" if backend_error or recovered else "succeeded"
        if recovered:
            notes.append("recovered_from_partial")
        if notes:
            done_payload["_specialist_notes"] = list(notes)
        write_specialist_done(workspace, done_payload)
        return SpecialistRunResult(
            task_id=ctx.task.task_id,
            domain=domain.key,
            gap_canonical_id=gap,
            status=status,
            specialist_done=done_payload,
            workspace=str(workspace) if workspace else "",
            transcript_path=str(transcript_path(workspace)) if workspace else "",
            done_path=str(done_path(workspace)) if workspace else "",
            error=backend_error,
            notes=notes,
        )

    def _admit_done_payload(
        self,
        ctx: RunnerContext,
        prep: "_PreparedRun",
        sub_result: SpecialistSubprocessResult,
        done_payload: dict[str, Any],
    ) -> list[str]:
        """Admit the specialist's ``done_payload`` in place: reconcile its patches, stamp its proposals, gate it.

        Returns the audit notes the admission leaves for the result.
        """
        notes: list[str] = []
        # Reconcile self-reported ``patches_written`` against the filesystem:
        # keep only claimed paths that exist on disk, then union with the scan.
        # Not when a harvested diff is the whole round, or the worktree was refused.
        claimed = done_payload.get("patches_written")
        if sub_result.patch_claims_ignored or not isinstance(claimed, list):
            claimed = []
        search_bases = [b for b in (prep.worktree, prep.workspace) if b is not None]
        validated: list[str] = []
        missing: list[str] = []
        for p in claimed:
            resolved = resolve_claimed_patch(p, search_bases)
            if resolved is not None:
                validated.append(resolved)
            else:
                missing.append(str(p))
        dropped_scanned_outside: list[str] = []
        for p in sub_result.patches:
            if not patch_path_within_bases(Path(p), search_bases):
                dropped_scanned_outside.append(p)
                log.warning(
                    "specialist: scanned patch %r resolves outside the specialist worktree/workspace; dropping",
                    p,
                )
                continue
            validated.append(p)
        deduped = list(dict.fromkeys(validated))
        if missing:
            # Record dangling patch claims for the session_breakdown audit.
            notes.append("patches_claimed_but_missing:" + ",".join(missing[:8]))
        if dropped_scanned_outside:
            notes.append("patches_scanned_outside_workspace:" + ",".join(dropped_scanned_outside[:8]))

        # A specialist may cite only the Experiences its dispatch read showed it.
        shown = shown_ids((ctx.task.params or {}).get("kb_rendered_refs"))
        for proposal in done_payload.get("proposal_set") or []:
            if not isinstance(proposal, dict):
                continue
            # The dispatch scope feeds the Critic's cross-domain enrichment.
            proposal.setdefault("scope", prep.profile.scope)
            if "experience_citations" in proposal:
                proposal["experience_citations"] = normalize_citations(proposal["experience_citations"], shown)
        if "experience_citations" in done_payload:
            done_payload["experience_citations"] = normalize_citations(done_payload["experience_citations"], shown)

        # Universal patch-safety gate: drop non-diff/escaping patches, git-ground
        # the rest, and scan for smuggled claims. Grounding is per set, not per
        # patch, so the root below applies to all of them or to none.
        base_checkout = prep.worktree_base or prep.worktree
        vetted = _patch_safety.vet_patches(
            deduped,
            base_checkout=base_checkout,
            candidate_roots=sibling_checkouts(tuple(self.subprocess_config.framework_source_roots), base_checkout),
            explicit_root=grounding_explicit_root(
                declared=str((ctx.task.params or {}).get("framework_source_root") or "").strip(),
                patches=deduped,
                patch_roots=sub_result.patch_roots,
            ),
        )
        return notes + _patch_safety.admit_payload(done_payload, vetted, sub_result.patch_roots).notes()

    # Worktree helpers
    def _maybe_setup_worktree(
        self,
        ctx: RunnerContext,
        *,
        workspace: Path | None,
        profile: SpecialistProfile | None = None,
    ) -> tuple[Path | None, FrameworkTree | None, str, Path | None]:
        """Provision a per-task git worktree when in subprocess mode.

        Blocking (``git worktree add``, 60s); call via ``asyncio.to_thread``.

        A tree no checkout tracks, such as a pip-installed package, is branched
        off a session-owned snapshot of it instead, so the specialist still edits
        the code the server runs.

        Best-effort: the specialist still dispatches without isolation and the
        reason lands in ``notes``.

        Args:
            ctx: The runner context for this specialist task.
            workspace: The task workspace the worktree is created under.
            profile: Resolved dispatch profile; non-patch mode skips worktree.

        Returns:
            A ``(worktree_dir, source, error, admin)`` tuple; ``worktree_dir``
            is ``None`` without a workspace or on git failure, ``source`` is
            the tree the worktree stands for, and ``admin`` its admin directory
            (:func:`~.worktrees.link_worktree`).
        """
        if workspace is None:
            return None, None, "", None
        if profile is not None:
            if profile.mode != MODE_PATCH:
                return None, None, "", None
        preflight_error = specialist_patch_preflight_error(ctx.task.params)
        if preflight_error:
            return None, None, preflight_error, None
        source = _worktree_source(ctx.task.params or {})
        if source is None:
            return None, None, "", None
        base = source.root
        if not source.checkout:
            if self.session_dir is None:
                return None, source, f"{source.tree} has no checkout and no session dir to snapshot it into", None
            repo = snapshot_repository(self.session_dir, source.tree)
            snapshot, err = snapshot_tree(source.tree, repo, refresh=not self._integrate_in_flight())
            if snapshot is None:
                return None, source, err, None
            base = snapshot
        linked = link_worktree(base, workspace / "worktree", f"specialist-{ctx.task.task_id}")
        return linked.path, source, linked.error, linked.admin

    def _integrate_in_flight(self) -> bool:
        """Whether an integrate has mutated the framework tree without deciding on it yet."""
        from ..state.shared_state import SharedState

        if self.session_dir is None:
            return False
        try:
            return bool(SharedState.load_or_init(self.session_dir).pending_integrate)
        except (OSError, ValueError):
            # An unreadable state cannot vouch for the tree.
            log.warning(
                "specialist worktree: cannot read pending_integrate; not refreshing the snapshot", exc_info=True
            )
            return True

    # Workspace file protocol
    def _resolve_workspace(self, ctx: RunnerContext) -> Path | None:
        """Resolve (and create) the workspace directory for a run.

        Prefers a pre-created workspace supplied on the context's ``extra``
        mapping, otherwise falls back to ``runs/specialist/<task_id>/``
        under the session directory.

        Args:
            ctx: Runner context for the current dispatch; an unset ``extra``
                resolves against the session directory.

        Returns:
            The workspace path, or ``None`` if no session directory is set.
        """
        ws = (ctx.extra or {}).get("workspace")
        if ws:
            p = Path(str(ws))
            p.mkdir(parents=True, exist_ok=True)
            return p
        if self.session_dir is None:
            return None
        p = runs_dir(self.session_dir, "specialist", ctx.task.task_id)
        p.mkdir(parents=True, exist_ok=True)
        return p


__all__ = [
    "NO_GIT_FRAMEWORK_SOURCE_ROOT",
    "SPECIALIST_TOOL_DENYLIST",
    "SpecialistRunResult",
    "SpecialistRunner",
    "SpecialistSubprocessConfig",
    "build_empty_specialist_done",
    "specialist_patch_preflight_error",
]
