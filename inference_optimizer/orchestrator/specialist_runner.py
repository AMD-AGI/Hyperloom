"""SpecialistRunner — v0.8 M5 (KB_design §3.5 + §3.13 M5).

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
  the Backend (KB_design §3.5 §10 / §3.11 R5).
* output: the runner harvests exactly one ``specialist_done`` intent
  from the transcript; any other intent type is logged and ignored.
* workspace: per ``runs/specialist/<task_id>/`` (KB_design §3.5 §8):
  ``prompt.md`` / ``transcript.jsonl`` / ``heartbeat.json`` /
  ``tool_calls.jsonl`` / ``specialist_done.json``.

Failure modes (KB_design §3.5 §9 / §3.13 M5 §6) are folded into one
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
from .sub_agent_runner import RunnerContext, SubAgentResult
from .system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
)


log = logging.getLogger(__name__)


# Default tool whitelist for specialists (KB_design §3.5 §10 / §3.11 R5).
# These tool names follow the Claude / Cursor convention; the actual MCP
# server names depend on operator config. The set is intentionally
# read-only (no Edit / Write / Bash apply commands).
DEFAULT_SPECIALIST_TOOLS: tuple[str, ...] = (
    "emit_intent",
    "Read", "Grep", "Glob",
    # Restricted Bash — runners may further filter via a callback. Keeping
    # ``Bash`` in the whitelist lets the LLM run rocm-smi / pgrep / cat;
    # the runner's per-call hook (TODO M6) will block destructive
    # invocations.
    "Bash",
    "WebSearch", "WebFetch",
    # MCP read-only surfaces.
    "mcp__pr_monitor__list_prs", "mcp__pr_monitor__get_pr",
    "mcp__cortex_kb__traverse",
    "mcp__cortex_kb__find_recipe",
    "mcp__cortex_kb__query",
)


# Tools explicitly denied even if the operator extends the whitelist —
# guard against accidental inclusion of write paths.
SPECIALIST_TOOL_DENYLIST: frozenset[str] = frozenset({
    "Edit", "Write", "MultiEdit",
    # Cortex write surfaces (KB_design §3.11 R4).
    "mcp__cortex_kb__propose_point",
    "mcp__cortex_kb__propose_edge",
    "mcp__cortex_kb__hypothesize",
    "mcp__cortex_kb__ingest_attempt",
    "mcp__cortex_kb__verify",
    "mcp__cortex_kb__commit",
})


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
    ``kill_task`` synth path (KB_design §3.5 §9 / §3.13 M5 §6). Guarantees
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
        backend_factory,
        *,
        session_dir: Path | None = None,
        default_tools: tuple[str, ...] = DEFAULT_SPECIALIST_TOOLS,
        default_max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS,
        per_turn_max_seconds: float = 90.0,
    ):
        """Create a runner.

        ``backend_factory`` is a callable ``(domain: SpecialistDomain) ->
        Backend``. The factory pattern lets the CLI inject a per-domain
        Claude / Codex / Mock backend without baking the choice into the
        runner itself.
        """
        self.backend_factory = backend_factory
        self.session_dir = Path(session_dir) if session_dir else None
        self.default_tools = tuple(default_tools)
        self.default_max_turns = int(default_max_turns)
        self.per_turn_max_seconds = float(per_turn_max_seconds)

    # ------------------------------------------------------------------
    # Public entry point
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
        """
        params = ctx.task.params or {}
        domain_key = str(params.get("domain") or "").strip()
        gap = str(
            params.get("gap_canonical_id") or params.get("gap") or ""
        ).strip()
        max_turns = int(params.get("max_turns") or self.default_max_turns)
        domain = get_domain(domain_key)

        workspace = self._resolve_workspace(ctx)

        if domain is None:
            done = build_empty_specialist_done(
                gap_canonical_id=gap,
                domain=domain_key,
                reason=f"unknown specialist domain={domain_key!r}",
            )
            self._write_specialist_done(workspace, done)
            return SpecialistRunResult(
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

        # M5 scope guard: domains outside the M5 active set still get
        # dispatched (PolicyGate R2 already accepts them), but we log so
        # operators see we're using a generic prompt template.
        notes: list[str] = []
        if domain.key not in SPECIALIST_DOMAINS_M5:
            notes.append(
                f"domain={domain.key!r} is post-M5 (available_in="
                f"{domain.available_in!r}); using generic prompt template"
            )

        # Assemble prompts (system / user). Operators may pre-build
        # ``prompt_inputs`` to inject KB / PR / source-hint context; we
        # default to a minimal version that still passes the M5
        # validators.
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
                warm_start_recipe=dict(params.get("warm_start_recipe") or {}),
                warm_start_pitfalls=list(
                    params.get("warm_start_pitfalls") or []
                ),
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
                tp=int(params.get("tp") or 1),
                hbm_gb=float(params.get("hbm_gb") or 0.0),
                peak_tflops=float(params.get("peak_tflops") or 0.0),
                arch_notes=str(params.get("arch_notes") or ""),
                workspace_path=str(workspace) if workspace else "",
                notes=str(params.get("notes") or ""),
            )

        system_prompt, user_prompt = build_specialist_prompts(prompt_inputs)
        self._write_prompt(workspace, system_prompt, user_prompt)
        self._write_heartbeat(
            workspace, turn=0, max_turns=max_turns, status="starting",
        )

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
        combined_prompt = system_prompt + "\n---\n" + user_prompt

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
                    prompt=user_prompt if turn_idx == 1 else combined_prompt,
                    system_prompt=system_prompt,
                    tools=list(self.default_tools),
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
                    # Don't break — capture the LAST done in a turn so
                    # any auxiliary intents (alerts) earlier in the same
                    # turn still log.
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

        if specialist_done_intent is None:
            reason = backend_error or (
                "max_turns_exhausted" if turns_used >= max_turns
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

        # Have a specialist_done — sanitise and persist.
        done_payload = dict(specialist_done_intent.payload or {})
        # Defensive: ensure gap_canonical_id/domain match dispatch
        # (PolicyGate R3 enforces this on the inbox side; we re-stamp here
        # so the on-disk artifact is authoritative).
        done_payload["gap_canonical_id"] = gap or done_payload.get(
            "gap_canonical_id", ""
        )
        done_payload["domain"] = domain.key
        if "proposal_set" not in done_payload:
            done_payload["proposal_set"] = []
        if "empty" not in done_payload:
            done_payload["empty"] = not bool(done_payload["proposal_set"])
        if "summary" not in done_payload:
            done_payload["summary"] = (
                "specialist emitted done without summary"[:480]
            )

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
    # Workspace file protocol (KB_design §3.5 §8)
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
    "DEFAULT_SPECIALIST_TOOLS",
    "SPECIALIST_TOOL_DENYLIST",
    "SpecialistRunResult",
    "SpecialistRunner",
    "build_empty_specialist_done",
]
