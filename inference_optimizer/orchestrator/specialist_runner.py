"""SpecialistRunner — v0.8 M5.

The second sub-agent form factor. Whereas the deterministic
:class:`SubAgentRunner` dispatches Python executors (BaselineExecutor,
ExploreExecutor, …) for ``delegate{action_name='<action>'}``, the
SpecialistRunner drives an *LLM* sub-agent for
``delegate{action_name='specialist', params={domain, gap, ...}}``.

Both runners share the TaskRegistry state machine + lane lease
mechanism + idempotency_key contract. They differ in:

* execution: SpecialistRunner spawns an :class:`Backend.run` loop over
  up to ``max_turns`` turns instead of calling a Python executor.
* tool surface: the runner passes a tightly-scoped tool whitelist into
  the Backend.
* output: the runner harvests exactly one ``specialist_done`` intent
  from the transcript; any other intent type is logged and ignored.
* workspace: per ``runs/specialist/<task_id>/``:
  ``prompt.md`` / ``transcript.jsonl`` / ``heartbeat.json`` /
  ``tool_calls.jsonl`` / ``specialist_done.json``.

Failure modes are folded into one
recovery primitive: every exit path synthesises a ``specialist_done``
payload so the upstream EXPLORE round never blocks on a missing
result. ``status`` carries the original outcome
(``"succeeded" / "stale" / "empty_synthesised"``) for the audit trail.

Inv enforcement: PolicyGate R2/R3 sit on the Coordinator side; the
runner doesn't re-validate the dispatch payload but does refuse to
emit non-specialist intents from the transcript (defense in depth).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_paths import runs_dir
from .backends.base import Backend, BackendError
from .intent_parser import Intent, IntentType
from .specialist_domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SPECIALIST_DOMAINS_M5,
    SpecialistDomain,
    get_domain,
)
from .specialist_subprocess import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    SpecialistSubprocessResult,
    _pick_worktree_base,
    _setup_worktree,
)
from .policy import DEFAULT_SPECIALIST_MAX_PROPOSALS
from .sub_agent_runner import RunnerContext, SubAgentResult
from .system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


log = logging.getLogger(__name__)


# v0.8 §3.11 R4 / R5 — canonical external tool registry lives in
# :mod:`policy`. We re-export the tuples below so legacy importers
# (and the runner itself) still see the historical names without a
# code rewrite, but PolicyGate and SpecialistRunner now share a
# single source of truth.
from .policy import (
    CORTEX_KB_READ_TOOL_NAMES as _CORTEX_KB_READ,
    KB_WRITE_TOOL_NAMES as _KB_WRITE,
    PR_MONITOR_TOOL_NAMES as _PR_MONITOR,
    WEB_TOOL_NAMES as _WEB,
)

#: Back-compat tuple alias for the PR Monitor MCP readonly tools.
#: Kept tuple-typed because the original constant was a tuple; some
#: external scripts iterate it positionally.
PR_MONITOR_MCP_TOOLS: tuple[str, ...] = tuple(sorted(_PR_MONITOR))

#: Back-compat tuple alias for the Cortex KB readonly MCP tools.
CORTEX_KB_READONLY_MCP_TOOLS: tuple[str, ...] = tuple(sorted(_CORTEX_KB_READ))


# Default tool whitelist for specialists (KB_design §3.5 §10 / §3.11 R5;
# PR-A2 (Arbor-into-Hyperloom) added Edit / Write / MultiEdit so specialists
# can produce source patches into their per-task git worktree under
# ``runs/specialist/<task_id>/worktree/``. The subprocess dispatcher's
# ``--add-dir <worktree>`` scoping keeps them out of the main
# framework_source_roots; the orchestrator's ``integrate_patch`` action
# is the only place where worktree patches are physically applied.
#
# Note these tool names follow the Claude / Cursor convention; the actual MCP
# server names depend on operator config.
#
# Cortex KB has no MCP surface (REST only); its read context is
# pre-warmed into Section 4 of the specialist prompt by
# ``Coordinator._warm_specialist_params`` → ``select_kb_for_domain``.
# The ``mcp__cortex_kb__{traverse,find_recipe,query}`` names therefore
# stay out of the default whitelist — advertising tool names that no
# MCP server backs caused specialists to attempt orphan calls and
# silently fall back to ``WebSearch``. ``CORTEX_KB_READONLY_MCP_TOOLS``
# remains importable for PolicyGate (denial validation) and tests.
DEFAULT_SPECIALIST_TOOLS: tuple[str, ...] = (
    "emit_intent",
    "Read", "Grep", "Glob",
    # PR-A2: write tools for patch authoring. Confined to the
    # worktree via ``--add-dir`` at subprocess spawn time.
    "Edit", "Write", "MultiEdit",
    # Restricted Bash — runners may further filter via a callback. Keeping
    # ``Bash`` in the whitelist lets the LLM run rocm-smi / pgrep / cat /
    # git diff > patches/<file>.patch; the runner's per-call hook (TODO M6)
    # will block destructive invocations.
    "Bash",
) + tuple(sorted(_WEB)) + PR_MONITOR_MCP_TOOLS


# Tools explicitly denied even if the operator extends the whitelist.
# PR-A2 lifted Edit / Write / MultiEdit out of the denylist (see
# DEFAULT_SPECIALIST_TOOLS above); only the Cortex KB write surfaces
# remain blocked because the KB lifecycle is Coordinator-owned (Inv-2
# / Inv-6.1). The KB write set is sourced from
# :data:`policy.KB_WRITE_TOOL_NAMES` so we never drift between the
# policy and runner layers.
SPECIALIST_TOOL_DENYLIST: frozenset[str] = frozenset(_KB_WRITE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _safe_redact(s: str) -> str:
    """Redact obvious secrets from a transcript line before writing to disk."""
    out = s
    for needle in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"):
        if needle in out:
            # Keep the env name visible; strip the actual value (when
            # logged as KEY=...). Conservative regex-less replacement.
            out = out.replace(needle, f"{needle}[REDACTED]")
    return out


@dataclass
class _PreparedRun:
    """Shared setup-phase output threaded into both the in-process and
    subprocess execution paths. Internal to :class:`SpecialistRunner`."""

    domain: SpecialistDomain | None = None
    # F2-3: per-domain sub_kind selected at dispatch time (e.g.
    # 'framework_pr_scout'). Empty string = default per-domain prompt /
    # tool whitelist. Threaded through so :meth:`_resolve_tools` can
    # apply differential MCP-tool gating.
    sub_kind: str = ""
    gap: str = ""
    max_turns: int = 0
    workspace: Path | None = None
    worktree: Path | None = None
    worktree_base: Path | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    notes: list[str] = field(default_factory=list)
    resolved_tools: tuple[str, ...] = ()
    # When set, the caller skips the execute phase and returns this
    # result verbatim (e.g. unknown domain / missing workspace).
    early_return: "SpecialistRunResult | None" = None


@dataclass
class SpecialistRunResult:
    """Internal record of one SpecialistRunner invocation.

    Distinct from :class:`SubAgentResult` because the Coordinator may
    want to inspect the runner-level status (succeeded / stale /
    empty_synthesised) separately from the task state.
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


def build_empty_specialist_done(
    *,
    gap_canonical_id: str,
    domain: str,
    reason: str,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Return the canonical empty ``specialist_done`` payload.

    Used by every failure path in this module + by the Coordinator's
    ``kill_task`` synth path. Guarantees
    the payload satisfies PolicyGate R3 schema (``empty=true``,
    ``proposal_set=[]``, non-empty summary).
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

    Wire-up (in the dispatcher) parallels ``SubAgentRunner.run_task`` but
    branches on ``task.kind == 'specialist'`` and routes here.

    The runner is deliberately *generic over the Backend protocol* — the
    same code drives MockBackend (tests), ClaudeBackend (production), or
    any future Cursor / Codex specialist driver. Operator policy (which
    backend handles specialists) sits in the CLI wiring.
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

        Exactly one of ``backend_factory`` / ``subprocess_config`` must
        be supplied:

        * ``backend_factory`` (legacy, v0.8 M5) is a callable
          ``(domain: SpecialistDomain) -> Backend``. The factory pattern
          lets the CLI inject a per-domain Claude / Codex / Mock backend
          without baking the choice into the runner itself. The
          specialist runs **in-process** via :meth:`Backend.run`.
        * ``subprocess_config`` (PR-A2, Arbor-into-Hyperloom) configures
          a :class:`SpecialistSubprocessDispatcher` that spawns a fresh
          ``claude`` subprocess per task with ``--add-dir <worktree>``
          isolation. Production cli boot uses this path; tests keep the
          in-process backend for speed.

        ``knowledge_plane`` (optional, v0.8 M4) lets the runner consult
        the :class:`KnowledgePlane` at task dispatch to gate the
        ``mcp__pr_monitor__*`` tool block — when PR Monitor is
        disabled (``--degraded-pr``) the runner strips those tool
        names from the per-task whitelist so the LLM doesn't get
        offered an absent endpoint. ``None`` leaves the default tool
        list untouched (back-compat for callers / tests that don't
        wire a plane yet).
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

    # F2-3: framework-agent MCP tool prefix (for the optional
    # ``mcp__fa__candidates`` / ``mcp__fa__fetch_pr`` server PR #280
    # ships alongside the ``fa`` CLI). The ``fa`` CLI itself is
    # invoked via the existing ``Bash`` tool, so this prefix only
    # matters when the operator wires the fa MCP server. Centralised
    # here so the whitelist policy stays a single source of truth.
    _FA_MCP_TOOL_PREFIX: str = "mcp__fa__"

    def _resolve_tools(self, sub_kind: str = "") -> tuple[str, ...]:
        """Return the per-task tool whitelist.

        Gated on:

        * KnowledgePlane PR Monitor / Cortex KB availability — strips
          ``mcp__pr_monitor__*`` / ``mcp__cortex_kb__*`` whenever the
          corresponding surface is disabled.
        * F2-3 framework-agent sub_kind — strips ``mcp__fa__*`` when
          ``sub_kind != 'framework_pr_scout'`` so the default serving
          path can never accidentally call the fa MCP server even if
          the operator pre-loaded it into ``default_tools``. The fa CLI
          itself is invoked via the existing ``Bash`` whitelist entry
          when the sub_kind authorises it.
        * Always enforces :data:`SPECIALIST_TOOL_DENYLIST` last
          (defense in depth — caller may have extended ``default_tools``
          carelessly).
        """
        tools = list(self.default_tools)
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
        # F2-3: differential framework-agent gating. The default
        # tool set never carries ``mcp__fa__*`` today, but we strip
        # defensively so an operator-extended default_tools tuple
        # still respects the sub_kind boundary.
        if sub_kind != "framework_pr_scout":
            tools = [
                t for t in tools
                if not t.startswith(self._FA_MCP_TOOL_PREFIX)
            ]
        tools = [t for t in tools if t not in SPECIALIST_TOOL_DENYLIST]
        return tuple(tools)

    # ------------------------------------------------------------------
    # Public entry point — dispatches to in-process or subprocess path
    # ------------------------------------------------------------------
    async def run(
        self,
        ctx: RunnerContext,
        *,
        prompt_inputs: SpecialistPromptInputs | None = None,
    ) -> SpecialistRunResult:
        """Run a specialist task to completion (or synthesise an empty done).

        Returns a :class:`SpecialistRunResult`; the caller is expected to
        translate it into the standard :class:`SubAgentResult` for the
        TaskRegistry. The runner never raises a Backend error past the
        function boundary — every failure ends with a valid
        specialist_done payload (Inv-5.3 single exit protocol).

        Two execution paths share the setup + finalize halves:

        * ``subprocess_dispatcher`` is configured (PR-A2 production
          path) — spawn a ``claude`` subprocess in an isolated git
          worktree.
        * ``backend_factory`` is configured (v0.8 M5 in-process path,
          retained for tests + the MockBackend path) — drive
          ``Backend.run`` in the same process via a per-turn loop.
        """
        prep = await self._prepare(ctx, prompt_inputs=prompt_inputs)
        if prep.early_return is not None:
            return prep.early_return

        if self.subprocess_dispatcher is not None:
            return await self._run_via_subprocess(ctx, prep)
        return await self._run_via_backend(ctx, prep)

    # ------------------------------------------------------------------
    # Setup phase (shared)
    # ------------------------------------------------------------------
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
        # propagate sub_kind from the dispatch params so
        # _resolve_tools can apply differential gating. Empty = default
        # per-domain prompt + tool whitelist.
        sub_kind = str(params.get("sub_kind") or "").strip()

        workspace = self._resolve_workspace(ctx)

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

        # M5 scope guard: domains outside the M5 active set still get
        # dispatched (PolicyGate R2 already accepts them), but we log so
        # operators see we're using a generic prompt template.
        notes: list[str] = []
        if domain.key not in SPECIALIST_DOMAINS_M5:
            notes.append(
                f"domain={domain.key!r} is post-M5 (available_in="
                f"{domain.available_in!r}); using generic prompt template"
            )

        # Worktree — created only when subprocess dispatch is wired.
        # The worktree path is surfaced via ``workspace_path`` in the
        # prompt so the agent learns where to write patches.
        worktree, worktree_base, worktree_err = self._maybe_setup_worktree(
            ctx, workspace=workspace,
        )
        if worktree_err:
            notes.append(f"worktree_setup_failed:{worktree_err}")
        workspace_for_prompt = worktree or workspace

        # Assemble prompts.
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
                # Coordinator-populated roofline / fa pre-fetch payloads
                # (see Coordinator._warm_specialist_params). Both
                # default to empty so non-warmed dispatches still build
                # a valid SpecialistPromptInputs.
                roofline_evidence=dict(params.get("roofline_evidence") or {}),
                sub_kind=str(params.get("sub_kind") or ""),
                pr_candidates=list(params.get("pr_candidates") or []),
                warm_start_recipe=dict(params.get("warm_start_recipe") or {}),
                warm_start_pitfalls=list(
                    params.get("warm_start_pitfalls") or []
                ),
                warm_start_lessons=list(
                    params.get("warm_start_lessons") or []
                ),
                session_snapshot=dict(params.get("session_snapshot") or {}),
                pr_feed=list(params.get("pr_feed") or []),
                pr_monitor_available=bool(
                    params.get("pr_monitor_available", True)
                ),
                framework_source_roots=tuple(
                    params.get("framework_source_roots") or ()
                ),
                source_hint_directories=tuple(
                    params.get("source_hint_directories") or ()
                ),
                gpu_type=str(params.get("gpu_type") or ""),
                tp=int(params.get("tp") or 0),
                hbm_gb=float(params.get("hbm_gb") or 0.0),
                peak_tflops=float(params.get("peak_tflops") or 0.0),
                arch_notes=str(params.get("arch_notes") or ""),
                # Workload context — populated by
                # Coordinator._warm_specialist_params from SharedState.
                # Zero/empty means "Coordinator did not plumb this
                # field"; the prompt section 2 renderer treats those
                # as "(none)" rather than fabricating a default.
                precision=str(params.get("precision") or ""),
                conc=int(params.get("conc") or 0),
                isl=int(params.get("isl") or 0),
                osl=int(params.get("osl") or 0),
                max_model_len=int(params.get("max_model_len") or 0),
                workspace_path=(
                    str(workspace_for_prompt) if workspace_for_prompt else ""
                ),
                notes=str(params.get("notes") or ""),
                # proposal_set cap (single source of truth: policy.py).
                # Coordinator._warm_specialist_params seeds this; clamp
                # defensively in case a caller passes a larger value.
                max_proposals=max(1, min(
                    DEFAULT_SPECIALIST_MAX_PROPOSALS,
                    int(params.get("max_proposals") or
                        DEFAULT_SPECIALIST_MAX_PROPOSALS),
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
            workspace=workspace,
            worktree=worktree,
            worktree_base=worktree_base,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            notes=notes,
            resolved_tools=self._resolve_tools(sub_kind),
        )

    # ------------------------------------------------------------------
    # In-process Backend path (v0.8 M5 / test path)
    # ------------------------------------------------------------------
    async def _run_via_backend(
        self, ctx: RunnerContext, prep: "_PreparedRun",
    ) -> SpecialistRunResult:
        """Drive ``Backend.run`` in the same Python process, one turn at
        a time, until a specialist_done intent shows up. Used by tests +
        the MockBackend fast path."""
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

        # Combined prompt — backends that don't honor ``system_prompt`` as
        # a separate field still see the system content inline. The
        # Backend protocol's ``system_prompt`` parameter takes precedence
        # when present.
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

            # Tool-violation check (defense in depth — backend already
            # gates whitelist, but if a backend ignored the whitelist
            # we still scrub the result).
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

    # ------------------------------------------------------------------
    # Subprocess path (PR-A2 production)
    # ------------------------------------------------------------------
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
            "error": sub_result.error,
        })
        self._write_heartbeat(
            workspace, turn=1, max_turns=prep.max_turns, status="finished",
        )

        # Decode the subprocess error into either a backend_error
        # (sets ``status='stale'``) or a clean miss (sets
        # ``empty_synthesised``).
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

    # ------------------------------------------------------------------
    # Finalize phase (shared)
    # ------------------------------------------------------------------
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
        # Defensive: ensure gap_canonical_id/domain match dispatch
        # (PolicyGate R3 enforces this on the inbox side; we re-stamp here
        # so the on-disk artifact is authoritative).
        done_payload["gap_canonical_id"] = gap or done_payload.get(
            "gap_canonical_id", ""
        )
        done_payload["domain"] = domain.key
        if "proposal_set" not in done_payload:
            done_payload["proposal_set"] = []
        # Hard truncate proposal_set to the single-source-of-truth cap.
        # The prompt asks the specialist to self-curate, but we never
        # trust LLM output for size limits — anything beyond the cap is
        # dropped before persist so the on-disk artifact, Coordinator
        # bookkeeping, Critic review and explore-grid materialisation
        # all see the same N≤cap shape. ``proposals_truncated_from`` is
        # picked up by ``coordinator._build_specialist_round_entry`` for
        # the session_breakdown audit trail.
        _proposals = done_payload["proposal_set"]
        if (
            isinstance(_proposals, list)
            and len(_proposals) > DEFAULT_SPECIALIST_MAX_PROPOSALS
        ):
            _original_len = len(_proposals)
            done_payload["proposal_set"] = (
                _proposals[:DEFAULT_SPECIALIST_MAX_PROPOSALS]
            )
            done_payload["proposals_truncated_from"] = _original_len
            notes.append(
                f"proposal_set_truncated:{_original_len}->"
                f"{DEFAULT_SPECIALIST_MAX_PROPOSALS}"
            )
        if "empty" not in done_payload:
            done_payload["empty"] = not bool(done_payload["proposal_set"])
        if "summary" not in done_payload:
            done_payload["summary"] = (
                "specialist emitted done without summary"[:480]
            )
        # PR-A2: Merge subprocess-discovered patches into
        # ``patches_written`` so downstream Coordinator bookkeeping +
        # IntegratePatchExecutor see them. If the agent already filled
        # ``patches_written`` from inside Bash, prefer its list over the
        # filesystem scan (the agent may carry intent about which
        # patches to apply in which order via numeric prefix).
        existing_patches = done_payload.get("patches_written") or []
        if not (
            isinstance(existing_patches, list) and existing_patches
        ) and patches_written:
            done_payload["patches_written"] = patches_written

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

    # ------------------------------------------------------------------
    # Worktree helpers
    # ------------------------------------------------------------------
    def _maybe_setup_worktree(
        self, ctx: RunnerContext, *, workspace: Path | None,
    ) -> tuple[Path | None, Path | None, str]:
        """Provision a per-task git worktree when in subprocess mode.

        Returns ``(worktree_dir, worktree_base, error)``. All three are
        empty / None when:

        * the runner is in in-process Backend mode (worktree is not
          needed), OR
        * the configured ``framework_source_roots`` don't contain a
          git checkout, OR
        * ``git worktree add`` fails.

        Failures are best-effort: the runner still dispatches the
        specialist, only without worktree isolation. The reason ends
        up in the ``notes`` of the returned SpecialistRunResult so
        downstream collectors can surface it.
        """
        if self.subprocess_config is None or workspace is None:
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

    # ------------------------------------------------------------------
    # Coordinator-facing convenience: produce a SubAgentResult shape.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Workspace file protocol
    # ------------------------------------------------------------------
    def _resolve_workspace(self, ctx: RunnerContext) -> Path | None:
        # Prefer the workspace SubAgentRunner pre-mkdir'd, fall back to
        # the conventional ``runs/specialist/<task_id>/``.
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
    "SPECIALIST_TOOL_DENYLIST",
    "SpecialistRunResult",
    "SpecialistRunner",
    "SpecialistSubprocessConfig",
    "build_empty_specialist_done",
]
