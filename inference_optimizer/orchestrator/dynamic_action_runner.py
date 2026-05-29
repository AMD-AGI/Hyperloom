"""Multi-turn ReAct sub-agent runner for ``dynamic_action`` dispatches.

Sibling to :class:`SpecialistRunner`: shared infrastructure
(TaskRegistry, worktree helpers, lane lease) is reused; prompt loop,
journal shape, tool whitelist, budget, and recovery rules are owned
here. One instance is reusable across many dispatches; each
``run`` call carries its own :class:`RunnerContext`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_proposal_set_path,
    runs_root,
)
from .backends.base import Backend, BackendError
from .dynamic_action_proposal import (
    DynamicRunnerTerminalState,
    MAX_PROPOSAL_REJECTS,
    ProposalValidationResult,
    build_proposal_set_payload,
    validate_proposal,
)
from .dynamic_action_tools import (
    BENCH_TOOL_ENABLED_V1,
    DYNAMIC_RESOURCE_TOOLS,
    TOOL_APPLY_PATCH_IN_WORKTREE,
    TOOL_EMIT_PROPOSAL,
    TOOL_READ_SESSION_ARTIFACT,
    TOOL_READ_SOURCE,
    TOOL_RUN_BENCH,
    apply_patch_in_worktree,
    capture_worktree_cumulative_diff,
    read_session_artifact,
    read_source,
    reset_worktree,
    run_bench,
)
from .specialist_subprocess import (
    _pick_worktree_base,
    _setup_worktree,
    _teardown_worktree,
)
from .sub_agent_runner import RunnerContext
from .system_prompts.dynamic_action_prompt_builder import (
    INPUT_TOKEN_CAP,
    JournalTurn,
    OUTPUT_TOKEN_CAP,
    PromptInputs,
    build_system_prompt,
    build_turn_prompt,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget defaults
# ---------------------------------------------------------------------------
DEFAULT_WALL_CLOCK_BUDGET_SEC: float = 15 * 60.0
DEFAULT_TURN_CAP: int = 12


# ---------------------------------------------------------------------------
# Action parsing — exactly one fenced JSON block per turn
# ---------------------------------------------------------------------------
_ACTION_BLOCK_RE: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.S,
)


@dataclass(frozen=True)
class ParsedAction:
    tool: str
    args: dict[str, Any]
    raw_block: str


class _UnparsableAction(RuntimeError):
    """Raised when the LLM output has no parseable action envelope."""


def parse_llm_action(text: str) -> ParsedAction:
    """Extract the single ``{"tool": ..., "args": ...}`` JSON block.

    Falls back to parsing the whole text as JSON when no fenced block
    is present.
    """
    candidates: list[str] = [m.group(1) for m in _ACTION_BLOCK_RE.finditer(text or "")]
    if not candidates:
        stripped = (text or "").strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [stripped]
    if not candidates:
        raise _UnparsableAction("no fenced JSON action block")
    if len(candidates) > 1:
        raise _UnparsableAction("multiple action blocks in single turn")
    try:
        parsed = json.loads(candidates[0])
    except json.JSONDecodeError as exc:
        raise _UnparsableAction(f"json parse failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _UnparsableAction("action envelope must be a JSON object")
    tool = str(parsed.get("tool") or "").strip()
    if not tool:
        raise _UnparsableAction("action envelope missing 'tool' field")
    args = parsed.get("args") or {}
    if not isinstance(args, dict):
        raise _UnparsableAction("action envelope 'args' must be an object")
    return ParsedAction(tool=tool, args=args, raw_block=candidates[0])


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------
@dataclass
class DynamicRunResult:
    """Outcome of one ``DynamicActionRunner.run`` invocation."""

    dyn_id: str
    terminal_state: DynamicRunnerTerminalState
    reason: str
    turns_used: int = 0
    proposal_set_payload: dict[str, Any] = field(default_factory=dict)
    journal_path: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dyn_id": self.dyn_id,
            "terminal_state": self.terminal_state.value,
            "reason": self.reason,
            "turns_used": self.turns_used,
            "proposal_set_payload": dict(self.proposal_set_payload),
            "journal_path": self.journal_path,
            "error": self.error,
            "notes": list(self.notes),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class DynamicActionRunner:
    """ReAct loop driver for one ``delegate{action='dynamic_action'}``.

    Construction is cheap; one instance per CLI session is the usual
    pattern (the runner caches no per-dispatch state). The Coordinator
    invokes :meth:`run` via the SubAgentRunner executor adapter
    registered for ``kind='dynamic_action'``.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        wall_clock_budget_sec: float = DEFAULT_WALL_CLOCK_BUDGET_SEC,
        turn_cap: int = DEFAULT_TURN_CAP,
        framework_source_roots: tuple[str, ...] = (),
    ):
        self.backend = backend
        self.wall_clock_budget_sec = float(wall_clock_budget_sec)
        self.turn_cap = int(turn_cap)
        self.framework_source_roots = tuple(framework_source_roots)

    # ------------------------------------------------------------------
    # Public entry — one dispatch
    # ------------------------------------------------------------------
    async def run(self, ctx: RunnerContext) -> DynamicRunResult:
        dyn_id = str(ctx.task.params.get("dyn_id") or ctx.task.task_id)
        session_dir = self._session_dir(ctx)
        if session_dir is None:
            return DynamicRunResult(
                dyn_id=dyn_id,
                terminal_state=DynamicRunnerTerminalState.FAILED,
                reason="runner_internal_error",
                error="session_dir missing from RunnerContext",
            )
        artefact_dir = dynamic_action_artifact_dir(session_dir, dyn_id)
        artefact_dir.mkdir(parents=True, exist_ok=True)
        journal_path = artefact_dir / "sub_agent_journal.md"
        journal_path.touch()
        spec_payload, seed_kit = self._load_dispatch_inputs(ctx, artefact_dir)
        if seed_kit is None or spec_payload is None:
            return self._finalise(
                dyn_id=dyn_id,
                state=DynamicRunnerTerminalState.FAILED,
                reason="runner_internal_error",
                journal_path=str(journal_path),
                journal=[],
                normalised_proposal=None,
                worktree=None,
                worktree_base=None,
                session_dir=session_dir,
                error="seed_kit or spec missing on disk",
            )
        worktree, worktree_base, wt_note = self._setup_worktree(
            session_dir, dyn_id,
        )
        journal: list[JournalTurn] = []
        if wt_note:
            journal.append(JournalTurn(
                turn=0,
                llm_text=f"[runner] {wt_note}",
                parsed_action={"tool": "runner_setup", "args": {}},
            ))
        deadline = time.monotonic() + self.wall_clock_budget_sec
        consecutive_rejects = 0
        terminal: tuple[DynamicRunnerTerminalState, str, dict[str, Any] | None] | None = None
        error_msg = ""

        try:
            for turn in range(1, self.turn_cap + 1):
                if time.monotonic() >= deadline:
                    terminal = (
                        DynamicRunnerTerminalState.TIMED_OUT,
                        "wall_clock_exhausted", None,
                    )
                    break
                prompt_inputs = PromptInputs(
                    dyn_id=dyn_id, seed_kit=seed_kit,
                    spec_payload=spec_payload,
                    journal=list(journal), turn_cap=self.turn_cap,
                )
                user_prompt, _ = build_turn_prompt(prompt_inputs)
                system_prompt = build_system_prompt(self.turn_cap)
                try:
                    backend_result = await self.backend.run(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        tools=None,
                        max_turns=1,
                    )
                except BackendError as exc:
                    terminal = (
                        DynamicRunnerTerminalState.FAILED,
                        "subprocess_crashed", None,
                    )
                    error_msg = repr(exc)
                    break
                raw_text = backend_result.raw_text or ""
                try:
                    action = parse_llm_action(raw_text)
                except _UnparsableAction as exc:
                    journal.append(JournalTurn(
                        turn=turn,
                        llm_text=raw_text,
                        parsed_action={"tool": "<unparsable>", "args": {}},
                        tool_result={"ok": False, "reason": str(exc)},
                    ))
                    consecutive_rejects += 1
                    if consecutive_rejects > MAX_PROPOSAL_REJECTS:
                        terminal = (
                            DynamicRunnerTerminalState.FAILED,
                            "unparsable_output", None,
                        )
                        break
                    continue

                if action.tool == TOOL_EMIT_PROPOSAL:
                    patch_text = str(action.args.get("patch_text") or "")
                    if not patch_text.strip():
                        journal.append(JournalTurn(
                            turn=turn,
                            llm_text=raw_text,
                            parsed_action={"tool": action.tool, "args": action.args},
                            proposal_validation={
                                "ok": True, "reason": "emit_empty",
                            },
                        ))
                        terminal = (
                            DynamicRunnerTerminalState.COMPLETED_EMPTY,
                            "emit_empty", None,
                        )
                        break
                    # Require the emitted ``patch_text`` to match the
                    # worktree's ``git diff HEAD`` when uncommitted
                    # changes exist; a clean worktree skips the check.
                    cumulative_diff = (
                        capture_worktree_cumulative_diff(worktree)
                        if worktree is not None else None
                    )
                    verdict: ProposalValidationResult = validate_proposal(
                        action.args,
                        spec_scope_domains=list(
                            spec_payload.get("scope_domains") or (),
                        ),
                        worktree_cumulative_diff=cumulative_diff,
                    )
                    if not verdict.ok:
                        consecutive_rejects += 1
                        journal.append(JournalTurn(
                            turn=turn,
                            llm_text=raw_text,
                            parsed_action={"tool": action.tool, "args": action.args},
                            proposal_validation=verdict.to_journal_dict(),
                        ))
                        if consecutive_rejects > MAX_PROPOSAL_REJECTS:
                            terminal = (
                                DynamicRunnerTerminalState.FAILED,
                                "proposal_validation_failed", None,
                            )
                            break
                        continue
                    journal.append(JournalTurn(
                        turn=turn,
                        llm_text=raw_text,
                        parsed_action={"tool": action.tool, "args": action.args},
                        proposal_validation={
                            "ok": True, "reason": "accepted",
                        },
                    ))
                    terminal = (
                        DynamicRunnerTerminalState.COMPLETED,
                        "emit_proposal", verdict.normalised,
                    )
                    break

                if action.tool not in DYNAMIC_RESOURCE_TOOLS:
                    journal.append(JournalTurn(
                        turn=turn,
                        llm_text=raw_text,
                        parsed_action={"tool": action.tool, "args": action.args},
                        tool_result={
                            "ok": False,
                            "reason": "unknown_tool",
                            "tool": action.tool,
                        },
                    ))
                    consecutive_rejects += 1
                    if consecutive_rejects > MAX_PROPOSAL_REJECTS:
                        terminal = (
                            DynamicRunnerTerminalState.FAILED,
                            "unparsable_output", None,
                        )
                        break
                    continue

                # Forward progress resets the consecutive-reject counter.
                consecutive_rejects = 0
                tool_result = await self._dispatch_tool(
                    action=action, session_dir=session_dir,
                    worktree=worktree, dyn_id=dyn_id, call_id=str(turn),
                )
                journal.append(JournalTurn(
                    turn=turn,
                    llm_text=raw_text,
                    parsed_action={"tool": action.tool, "args": action.args},
                    tool_result=tool_result,
                ))
            else:
                terminal = (
                    DynamicRunnerTerminalState.TIMED_OUT,
                    "turn_cap_exhausted", None,
                )
        except asyncio.CancelledError:
            terminal = (
                DynamicRunnerTerminalState.ABANDONED,
                "external_kill", None,
            )
            raise
        finally:
            if worktree is not None:
                reset_worktree(worktree)

        state, reason, normalised = (
            terminal if terminal is not None
            else (
                DynamicRunnerTerminalState.FAILED,
                "runner_internal_error", None,
            )
        )
        return self._finalise(
            dyn_id=dyn_id,
            state=state, reason=reason,
            journal_path=str(journal_path),
            journal=journal,
            normalised_proposal=normalised,
            worktree=worktree, worktree_base=worktree_base,
            session_dir=session_dir,
            error=error_msg,
            turns_used=len([j for j in journal if j.turn > 0]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _session_dir(self, ctx: RunnerContext) -> Path | None:
        sd = ctx.extra.get("session_dir")
        return Path(sd) if sd else None

    def _load_dispatch_inputs(
        self, ctx: RunnerContext, artefact_dir: Path,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Load ``spec.payload`` + ``seed_kit`` JSON; return
        ``(None, None)`` on any I/O or parse failure."""
        spec_path = ctx.task.params.get("spec_path")
        seed_kit_path = ctx.task.params.get("seed_kit_path")
        if not spec_path:
            spec_path = str(artefact_dir / "spec.json")
        if not seed_kit_path:
            seed_kit_path = str(artefact_dir / "seed_kit.json")
        try:
            spec_text = Path(spec_path).read_text(encoding="utf-8")
            seed_text = Path(seed_kit_path).read_text(encoding="utf-8")
        except OSError as exc:
            log.warning(
                "dynamic_action runner: cannot read spec/seed_kit "
                "(spec=%s seed=%s err=%r)",
                spec_path, seed_kit_path, exc,
            )
            return None, None
        try:
            spec = json.loads(spec_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "dynamic_action runner: spec.json parse failed at %s: %r",
                spec_path, exc,
            )
            return None, None
        try:
            seed = json.loads(seed_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "dynamic_action runner: seed_kit.json parse failed at "
                "%s: %r",
                seed_kit_path, exc,
            )
            return None, None
        return spec.get("payload") or {}, seed

    def _setup_worktree(
        self, session_dir: Path, dyn_id: str,
    ) -> tuple[Path | None, Path | None, str]:
        """Create ``runs/dynamic/<dyn_id>/worktree/`` over the first
        available ``framework_source_root``. Returns
        ``(worktree, base, note)``; ``note`` is non-empty when no
        isolated worktree could be set up (runner still proceeds but
        ``apply_patch_in_worktree`` will fail)."""
        base = _pick_worktree_base(self.framework_source_roots)
        if base is None:
            return None, None, (
                "no_worktree_base: framework_source_roots empty or "
                "no git checkout present; apply_patch_in_worktree "
                "will fail until configured."
            )
        worktree = (
            runs_root(session_dir) / "dynamic" / dyn_id / "worktree"
        )
        branch = f"dynamic-{dyn_id}"
        # Drop any leftover worktree + branch from a crashed / abandoned
        # prior dispatch so ``git worktree add`` does not fail with
        # "branch already exists".
        self._prune_stale_worktree(base, worktree, branch)
        wt, err = _setup_worktree(base, worktree, branch=branch)
        if wt is None:
            return None, base, f"worktree_setup_failed: {err}"
        return wt, base, ""

    @staticmethod
    def _prune_stale_worktree(base: Path, worktree: Path, branch: str) -> None:
        """Best-effort cleanup of a stale per-dyn_id worktree + branch."""
        import subprocess
        cmds = (
            ["git", "-C", str(base), "worktree", "remove", "--force",
             str(worktree)],
            ["git", "-C", str(base), "worktree", "prune"],
            ["git", "-C", str(base), "branch", "-D", branch],
        )
        for cmd in cmds:
            try:
                subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=20.0, check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    async def _dispatch_tool(
        self,
        *,
        action: ParsedAction,
        session_dir: Path,
        worktree: Path | None,
        dyn_id: str,
        call_id: str,
    ) -> dict[str, Any]:
        if action.tool == TOOL_READ_SOURCE:
            return read_source(
                str(action.args.get("path") or ""),
                action.args.get("max_bytes"),
            )
        if action.tool == TOOL_READ_SESSION_ARTIFACT:
            return read_session_artifact(
                session_dir,
                str(action.args.get("path") or ""),
                dyn_id=dyn_id,
                max_bytes=action.args.get("max_bytes"),
            )
        if action.tool == TOOL_RUN_BENCH:
            if not BENCH_TOOL_ENABLED_V1:
                return {
                    "ok": False,
                    "reason": "bench_tool_disabled_v1",
                    "tool": action.tool,
                }
            if worktree is None:
                return {
                    "ok": False,
                    "reason": "worktree_unavailable",
                    "tool": action.tool,
                }
            return await run_bench(
                str(action.args.get("bench_id") or ""),
                worktree=worktree,
                call_id=call_id,
                params=action.args.get("params") or {},
            )
        if action.tool == TOOL_APPLY_PATCH_IN_WORKTREE:
            if worktree is None:
                return {
                    "ok": False,
                    "reason": "worktree_unavailable",
                    "tool": action.tool,
                }
            return apply_patch_in_worktree(
                worktree, str(action.args.get("patch_text") or ""),
            )
        return {"ok": False, "reason": "unknown_tool", "tool": action.tool}

    def _finalise(
        self,
        *,
        dyn_id: str,
        state: DynamicRunnerTerminalState,
        reason: str,
        journal_path: str,
        journal: list[JournalTurn],
        normalised_proposal: dict[str, Any] | None,
        worktree: Path | None,
        worktree_base: Path | None = None,
        session_dir: Path,
        error: str = "",
        turns_used: int = 0,
    ) -> DynamicRunResult:
        """Write ``proposal_set.json`` + ``sub_agent_journal.md`` and
        tear down the worktree. Only these two files are recovered;
        every other in-worktree artefact is destroyed."""
        # Every terminal state gets a journal on disk for audit, even
        # FAILED with zero turns.
        try:
            Path(journal_path).write_text(
                _render_journal_markdown(dyn_id, state, reason, journal),
                encoding="utf-8",
            )
        except OSError:
            log.exception(
                "dynamic_action runner: journal write failed for "
                "dyn_id=%s",
                dyn_id,
            )
        proposal_payload = build_proposal_set_payload(
            dyn_id=dyn_id,
            normalised_proposal=(
                normalised_proposal
                if state == DynamicRunnerTerminalState.COMPLETED else None
            ),
            journal_path=journal_path,
        )
        try:
            dynamic_action_proposal_set_path(
                session_dir, dyn_id,
            ).write_text(
                json.dumps(proposal_payload, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            log.exception(
                "dynamic_action runner: proposal_set write failed for "
                "dyn_id=%s",
                dyn_id,
            )
        if worktree is not None:
            _teardown_worktree(worktree_base, worktree)
        return DynamicRunResult(
            dyn_id=dyn_id,
            terminal_state=state,
            reason=reason,
            turns_used=turns_used,
            proposal_set_payload=proposal_payload,
            journal_path=journal_path,
            error=error,
        )


def _render_journal_markdown(
    dyn_id: str,
    state: DynamicRunnerTerminalState,
    reason: str,
    journal: list[JournalTurn],
) -> str:
    lines = [
        f"# Dynamic action sub-agent journal — {dyn_id}",
        f"- terminal_state: {state.value}",
        f"- reason: {reason}",
        f"- recorded_at: {_now_iso()}",
        "",
    ]
    for entry in journal:
        lines.append(f"## turn {entry.turn}")
        if entry.llm_text:
            lines.append("### llm_text")
            lines.append("```text")
            lines.append(entry.llm_text)
            lines.append("```")
        if entry.parsed_action:
            lines.append(
                f"### parsed_action\n```json\n"
                f"{json.dumps(entry.parsed_action, sort_keys=True, indent=2)}\n```"
            )
        if entry.tool_result is not None:
            lines.append(
                f"### tool_result\n```json\n"
                f"{json.dumps(entry.tool_result, sort_keys=True, indent=2)}\n```"
            )
        if entry.proposal_validation is not None:
            lines.append(
                f"### proposal_validation\n```json\n"
                f"{json.dumps(entry.proposal_validation, sort_keys=True, indent=2)}\n```"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TURN_CAP",
    "DEFAULT_WALL_CLOCK_BUDGET_SEC",
    "DynamicActionRunner",
    "DynamicRunResult",
    "ParsedAction",
    "parse_llm_action",
]
