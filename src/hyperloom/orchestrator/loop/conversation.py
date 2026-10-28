# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator prompt composition: inbox rendering, per-tick phase/mission/advisory blocks, MCP context readers, and reactor conversation tracing."""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable
from ..bus.gpu_pool import gpus_by_task_sync
from ..mcp_tool_names import qualified_context_tool_name
from ..phases import dispatch_order
from ..phases import machine_state as _phase_state
from ..policy.projection import resource_pools_summary
from ..specialists.in_flight import in_flight_domains
from ..specialists.reap_watch import newest_activity, specialist_activity_paths
from ..roles.base import BackendTurnResult
from ..bus.message_bus import Message
from hyperloom.inference_optimizer.tracelens_md import strip_base64_data_urls
from hyperloom.inference_optimizer.trace.conversation_trace import ConversationRecord, append_conversation
from ..state.failure_evidence import UNMEASURED_OUTCOMES, render_failure_line
from ..state.shared_state import KbInjection
from hyperloom.common.coerce import to_str_list, to_unix
from hyperloom.common.prompt_safety import defang_prompt_structure as _defang_prompt_structure
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_inbox

from .verdicts import serialize_verdict_advisory
from ..state.task_registry import Task
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ..collaborator import CoordinatorCollaborator
import logging as _logging

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.experience_kb import ExperienceKBEvidence
    from .coordinator import Coordinator

log = _logging.getLogger(__name__)

# Per-variant failure lines expanded by get_recent_outcomes, and the total cap that keeps a wide top_k from flooding
# the turn.
_RECENT_OUTCOMES_VARIANT_ROWS = 12
_RECENT_OUTCOMES_LINE_CAP = 120

# Result keys surfaced in delegated_result inbox line; first match wins per group.
_OUTCOME_GAIN_KEYS: tuple[str, ...] = (
    "validated_gain_pct",
    "gain_pct",
    "predicted_gain_pct",
    "delta_pct",
)
_OUTCOME_TPUT_KEYS: tuple[str, ...] = (
    "tokens_per_s",
    "tput",
    "throughput",
    "tput_tok_s",
)
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome", "runner_status")
# Notes rendered per inbox line.
_OUTCOME_NOTES_MAX: int = 3
# Ceilings for the prompt sections that grow with the session; uncapped, a long run outgrows the context window.
_FINDINGS_PROMPT_CHARS = 24_000
_QUESTIONS_PROMPT_CHARS = 6_000
_INBOX_PROMPT_CHARS = 40_000
# Task kinds that queue for the benchmark lane, in the order the backlog line lists them.
_BENCH_LANE_KINDS = ("integrate_patch", "explore")
# Longest a KERNEL_AGENT session goes without an Orchestration turn while kernel progress is unchanged.
_KERNEL_ORCHESTRATION_HEARTBEAT_SEC = 900.0


@dataclass(frozen=True)
class ComposedPrompt:
    """One agent's prompt: ``context`` holds what changes rarely, ``turn`` what changes every tick."""

    context: str
    turn: str

    @property
    def text(self) -> str:
        """The whole prompt as one string."""
        return f"{self.context}\n{self.turn}"


@dataclass(frozen=True)
class TurnPrompt:
    """What one reactor turn hands its backend; ``context`` is the part of the prompt carried in ``system_prompt``."""

    system_prompt: str
    prompt: str
    context: str


def _fit_to_budget(lines: list[str], budget_chars: int) -> list[str]:
    """Return the leading ``lines`` that fit in ``budget_chars``.

    The first line is always kept, clipped to the budget, so a backlog led by
    one oversized entry still drains.
    """
    if not lines:
        return []
    kept = [lines[0][:budget_chars]]
    used = len(kept[0])
    for line in lines[1:]:
        used += len(line) + 1
        if used > budget_chars:
            break
        kept.append(line)
    return kept


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` present + non-None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _defang_alert_payload(value: Any) -> Any:
    """Recursively defang string leaves of an alert payload (keys untouched)."""
    if isinstance(value, str):
        return _defang_prompt_structure(value)
    if isinstance(value, dict):
        return {k: _defang_alert_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_defang_alert_payload(v) for v in value]
    return value


def _variant_failure_rows(result: dict[str, Any], max_variant_rows: int) -> list[str]:
    """Indented rows for the unmeasured per-variant outcomes of a delegated result."""
    pvos = result.get("per_variant_outcomes")
    if not isinstance(pvos, list):
        return []
    failures = [v for v in pvos if isinstance(v, dict) and str(v.get("outcome") or "").upper() in UNMEASURED_OUTCOMES]
    rows: list[str] = []
    for vo in failures[:max_variant_rows]:
        row = dict(vo)
        row["error_excerpt"] = _flatten_for_inbox(vo.get("error_excerpt") or vo.get("reason") or "")
        rows.append("  failure: " + render_failure_line(row, excerpt_chars=120))
    elided = len(failures) - max_variant_rows
    if elided > 0:
        pull = qualified_context_tool_name("get_variant_failures")
        rows.append(f"  (+{elided} more failures; pull {pull})")
    return rows


def _format_delegated_result(head: str, payload: dict[str, Any], max_variant_rows: int) -> str:
    """Render a delegated_result entry: one header line plus a row per unmeasured variant."""
    raw_result = payload.get("result")
    result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    # Only a specialist result carries this key, so its presence is the test.
    done: dict[str, Any] = result.get("specialist_done") or {}
    parts = [head, f"kind={payload.get('kind')!r}", f"state={payload.get('state')!r}"]
    for label, value, quoted in (
        ("status", _first_present(result, _OUTCOME_STATUS_KEYS), True),
        ("kept", result.get("kept"), True),
        ("gain", _first_present(result, _OUTCOME_GAIN_KEYS), False),
        ("tput", _first_present(result, _OUTCOME_TPUT_KEYS), False),
    ):
        if value is not None:
            parts.append(f"{label}={value!r}" if quoted else f"{label}={value}")
    summary = str(done.get("summary") or "").strip()
    if summary:
        parts.append(f"summary={summary[:400]!r}")
    if done.get("confidence") is not None:
        parts.append(f"confidence={done['confidence']}")
    for label, key in (("findings", "new_findings"), ("questions", "residual_questions")):
        # A model writes these, so a scalar can arrive where the schema asks for a list: it counts as one.
        items = to_str_list(done.get(key))
        if items:
            parts.append(f"{label}={len(items)}")
    # Executors that never raise report the failure inside the result envelope, leaving the top-level error None.
    error = payload.get("error") or result.get("error")
    if error:
        parts.append(f"error={str(error)[:200]!r}")
    # patch_safety_numeric is the Critic's artifact, not a lever here.
    notes = [str(n) for n in result.get("notes") or () if n and not str(n).startswith("patch_safety_numeric:")]
    if notes:
        shown = "; ".join(notes[:_OUTCOME_NOTES_MAX])
        parts.append(f"notes={shown[:300]!r}")
    header_line = " ".join(parts)
    if max_variant_rows <= 0:
        return header_line
    return "\n".join([header_line, *_variant_failure_rows(result, max_variant_rows)])


def _format_review_verdict(head: str, payload: dict[str, Any]) -> str:
    """Render a Critic verdict with the counts of the advisory it carries."""
    parts = [
        f"{head} target={payload.get('target_proposal_msg_id')!r} "
        f"verdict={payload.get('verdict')!r} "
        f"reasoning={str(payload.get('reasoning') or '')[:140]!r}"
    ]
    advisory = serialize_verdict_advisory(payload)
    required_evidence = advisory.get("required_evidence")
    if required_evidence:
        shown = "; ".join(str(item) for item in required_evidence[:3])
        parts.append(f"required_evidence[{len(required_evidence)}]={shown[:140]!r}")
    risks = advisory.get("risks")
    if risks:
        parts.append(f"risks={len(risks)}")
    advice_text = advisory.get("advice_text")
    if advice_text:
        parts.append(f"advice={advice_text[:140]!r}")
    return " ".join(parts)


def _format_inbox_event(m: "Message", *, max_variant_rows: int = 3) -> str:
    """Render one inbox ``Message`` as a compact, high-signal line."""
    topic = (m.topic or "").strip()
    payload = m.payload if isinstance(m.payload, dict) else {}
    # Canonical inbox header ordering that downstream parsers anchor on.
    msg_id = f"msg_id={m.msg_id} " if getattr(m, "msg_id", None) else ""
    head = f"seq={m.seq} {msg_id}from={m.from_agent} topic={topic}"

    if topic == "delegated_result":
        return _format_delegated_result(head, payload, max_variant_rows)
    if topic in ("policy_denial", "denial") or (topic == "observation" and payload.get("kind") == "policy_denial"):
        return (
            f"{head} action={payload.get('action_name')!r} "
            f"rule={payload.get('rule')!r} "
            f"hint={str(payload.get('hint') or '')[:140]!r}"
        )
    if topic == "review_verdict":
        return _format_review_verdict(head, payload)
    if topic == "observation" and payload.get("kind") is not None:
        return f"{head} kind={payload.get('kind')!r} payload={payload}"
    if topic == "alert":
        # Alert payloads can embed attacker-influenceable server.log excerpts; defang string leaves so a log line
        # can't inject prompt structure.
        return f"{head} payload={_defang_alert_payload(payload)}"

    return f"{head} payload={payload}"


#: A prompt section: the heading it goes under, or ``None`` for a block that carries its own, and what renders it.
_AdvisoryRow = tuple[str | None, Callable[["ConversationCollaborator"], str]]

#: The orchestration advisories that hold for the session, so they ride in the cached context. Row order is the
#: order the model reads the sections in, so it is content; the per-tick blocks follow them, in the turn.
_ADVISORY_BLOCKS: tuple[_AdvisoryRow, ...] = (
    ("=== Warm start (Recipe KB T0) ===", lambda c: c.shared_state.to_warm_start_summary()),
    ("=== Current gaps ===", lambda c: c.shared_state.to_gaps_summary()),
    (None, lambda c: c._specialist_findings_block()),
    ("=== External target gap (advisory) ===", lambda c: c.target_gap_advisory_block()),
    ("=== Specialist proposal scores (advisory) ===", lambda c: c.shared_state.to_proposal_scores_summary()),
    ("=== Priors-match (advisory ordering) ===", lambda c: c._priors_match_advisory_block()),
    ("=== Intervention mix (telemetry) ===", lambda c: c.shared_state.to_intervention_mix_summary()),
    ("=== Bottleneck redirect (advisory) ===", lambda c: c._bottleneck_redirect_advisory_block()),
    ("=== Acceptance threshold (advisory) ===", lambda c: c._acceptance_threshold_advisory_block()),
    ("=== Discarded escalation hint (advisory) ===", lambda c: c._discarded_escalate_hint_advisory_block()),
)


def _advisory_sections(owner: "ConversationCollaborator", blocks: tuple[_AdvisoryRow, ...]) -> list[str]:
    """Render the advisories that produced something, each under its heading."""
    sections: list[str] = []
    for heading, produce in blocks:
        block = produce(owner)
        if not block:
            continue
        if heading:
            sections.append(heading)
        sections.append(block)
    return sections


class ConversationCollaborator(CoordinatorCollaborator):
    """Manages conversation rounds: context tools, prompt injection, and round history."""

    def __init__(self, coordinator: "Coordinator") -> None:
        super().__init__(coordinator)
        # The Experience KB read rendered into the latest orchestration prompt; proposals emitted on that tick cite it.
        self.kb_last_read: ExperienceKBEvidence | None = None
        # Per-agent seq of the last message its prompt rendered.
        self._rendered_cursor: dict[str, int] = {}
        # The kernel progress fingerprint at Orchestration's last admitted KERNEL turn, and when it was admitted.
        self._admitted_kernel_fingerprint = ""
        self._admitted_at = 0.0

    async def _kb_prompt_block(self, untested_proposals: str) -> str:
        """Read shared Experience evidence once per FRAMEWORK_AGENT orchestration tick."""
        integration = self._coord.experience_kb
        if integration is None:
            return ""
        evidence = await asyncio.to_thread(
            integration.read_for_framework,
            self.shared_state,
            untested_proposals=untested_proposals,
        )
        self.kb_last_read = evidence
        if evidence.status != "completed" or not evidence.prompt_block:
            return ""
        block = "\n".join(
            (
                "=== Experience KB warm-start evidence ===",
                ("INVARIANT: the original Recipe benchmark measurement remains the gain-accounting baseline."),
                (
                    "Historical configurations may seed proposals/current-best "
                    "candidates only; never replace benchmark_baseline."
                ),
                evidence.prompt_block,
            )
        )
        self.shared_state.record_experience_kb_injection(
            KbInjection(
                consumer="orchestration",
                read_id=evidence.read_id,
                experience_ids=[str(ref.get("id") or "") for ref in evidence.rendered_refs],
                experiences=[dict(item) for item in evidence.experiences],
                prompt_block=block,
            )
        )
        return block

    def attach_orchestration_context_tools(self) -> None:
        """Bind a read-only ContextProvider to the orchestration backend (no-op without setter)."""
        backend = self.backends.get("orchestration")
        setter = getattr(backend, "set_context_provider", None)
        if setter is None:
            return
        from ..roles.mcp_context_tools import ContextProvider

        provider = ContextProvider(
            shared_state=self.shared_state,
            inbox_reader=self._context_inbox_reader,
            analysis_reader=self._context_analysis_reader,
            recent_outcomes_reader=self._context_recent_outcomes_reader,
            running_tasks_reader=self._context_running_tasks_reader,
            reference_reader=self._context_reference_reader,
        )
        setter(provider)

    def _context_reference_reader(self, name: str = "") -> str:
        """Resolve a reference doc by stem; reject path traversal."""
        from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir

        refs_dir = asset_prompt_references_dir()
        stem = (name or "").strip()
        if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: invalid name {name!r}; available: {available})"
        candidate = (refs_dir / stem).with_suffix(".md").resolve()
        if candidate.parent != refs_dir.resolve():
            return f"(read_reference: path traversal rejected for {name!r})"
        if not candidate.exists():
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: {name!r} not found; available: {available})"
        return candidate.read_text(encoding="utf-8")

    def _context_inbox_reader(self, since_seq: int = 0) -> str:
        """Synchronous projection of the orchestration inbox tail (sync SQLite path)."""
        msgs = self.bus.inbox_context_sync("orchestration", after_seq=int(since_seq or 0))
        if not msgs:
            return "(no inbox events)"

        lines = [_format_inbox_event(m) for m in msgs]
        return "\n".join(lines)

    def _context_recent_outcomes_reader(self, top_k: int = 8) -> str:
        """Synchronous projection of recent action outcomes."""
        k = max(1, min(top_k or 8, 50))
        newest_first = self.bus.recent_outcomes_context_sync(limit=k)
        if not newest_first:
            return "(no recent outcomes)"
        # Flip newest-first query to newest-last for chronological reading.
        msgs = newest_first[::-1]

        header = "=== Recent action outcomes (newest last) ==="
        body_lines: list[str] = []
        body_lines.extend(_format_inbox_event(m, max_variant_rows=_RECENT_OUTCOMES_VARIANT_ROWS) for m in msgs)
        rendered = "\n".join(body_lines).splitlines()
        if len(rendered) > _RECENT_OUTCOMES_LINE_CAP:
            rendered = rendered[-_RECENT_OUTCOMES_LINE_CAP:]
            return "\n".join(
                [header]
                + rendered
                + [f"(truncated at {_RECENT_OUTCOMES_LINE_CAP} lines; re-query with a smaller top_k)"]
            )
        return "\n".join([header] + rendered)

    def _context_running_tasks_reader(self) -> str:
        """Project in-flight tasks and their held resources from three reads, not one snapshot."""
        tasks = self.tasks.running_context_sync()
        backlog = "\n".join(line for line in (self._queued_backlog_line(), self._queued_specialists_line()) if line)
        if not tasks:
            return "\n".join(line for line in ("(no tasks in flight)", backlog) if line)

        lanes_by_task = self.locks.lanes_by_task_sync()
        gpus_by_task = gpus_by_task_sync(self.db)
        now_unix = time.time()
        lines = ["=== Tasks in flight ==="]
        for task in tasks:
            lanes, expires_at = lanes_by_task.get(task.task_id, ([], ""))
            gpus = gpus_by_task.get(task.task_id, [])
            params = task.params or {}
            started = to_unix(task.updated_at, 0.0)
            running_sec = max(0.0, now_unix - started) if started > 0 else 0.0
            parts = [
                f"  - task_id={task.task_id}",
                f"kind={task.kind!r}",
                f"running_sec={int(running_sec)}",
            ]
            domain = str(params.get("domain") or "")
            gap = str(params.get("gap_canonical_id") or "")
            if domain:
                parts.append(f"domain={domain!r}")
            if gap:
                parts.append(f"gap={gap!r}")
            parts.append(f"idempotency_key={task.idempotency_key!r}")
            if task.lease_ttl_sec:
                parts.append(f"lease_ttl_sec={task.lease_ttl_sec}")
            if expires_at:
                exp_unix = to_unix(expires_at, 0.0)
                if exp_unix > 0:
                    parts.append(f"lease_expires_in_sec={int(exp_unix - now_unix)}")
            if lanes:
                parts.append(f"lanes={sorted(lanes)}")
            if gpus:
                parts.append(f"gpu_ids={sorted(gpus)}")
            hb_age = self._task_heartbeat_age_sec(task, now_unix=now_unix)
            if hb_age is not None:
                parts.append(f"heartbeat_age_sec={int(hb_age)}")
            lines.append(" ".join(parts))
        if backlog:
            lines.append(backlog)
        return "\n".join(lines)

    def _queued_backlog_line(self) -> str:
        """Benchmark-lane backlog: queued integrate_patch / explore tasks in dispatch order, and untested proposals."""
        counts = self.tasks.queued_kind_counts_sync()
        counts["untested proposals"] = len(self.shared_state.untested_proposal_rows())
        parts = [f"{counts[kind]} {kind}" for kind in (*_BENCH_LANE_KINDS, "untested proposals") if counts.get(kind)]
        if not parts:
            return ""
        first = dispatch_order.PHASE_DISPATCH_FIRST.get(self.shared_state.phase, frozenset())
        queued_bench = [kind for kind in _BENCH_LANE_KINDS if counts.get(kind)]
        ahead = [kind for kind in queued_bench if kind in first]
        behind = [kind for kind in queued_bench if kind not in first]
        order = (
            f" ({', '.join(ahead)} dispatches before any queued {', '.join(behind)} in this phase,"
            " including one queued later)"
            if ahead and behind
            else ""
        )
        return "queued backlog: " + ", ".join(parts) + order

    def _queued_specialists_line(self) -> str:
        """Domains a queued specialist holds: a specialist delegate for one of them is refused until it finishes."""
        held = in_flight_domains(self.tasks.queued_context_sync())
        if not held:
            return ""
        return "queued specialists (each holds its domain): " + ", ".join(
            f"{domain} task_id={task.task_id}" for domain, task in held.items()
        )

    def _task_heartbeat_age_sec(self, task: "Task", *, now_unix: float) -> float | None:
        """Age of a specialist's freshest liveness signal, read from the same paths the reaper polls."""
        if (task.kind or "").strip() != "specialist":
            return None
        newest = newest_activity(specialist_activity_paths(runs_dir(self.session_dir, "specialist", task.task_id)))
        if newest is None:
            return None
        return max(0.0, now_unix - newest)

    def _context_analysis_reader(self) -> str:
        """Return the latest TraceLens analysis.md snapshot text."""
        lta = self.shared_state.last_trace_analyze or {}
        text = lta.get("analysis_md_text") or ""
        if text:
            return strip_base64_data_urls(text)
        path = str(lta.get("analysis_md_path") or "")
        if path:
            try:
                from pathlib import Path as _Path

                return strip_base64_data_urls(_Path(path).read_text(encoding="utf-8"))
            except OSError as exc:
                return f"(analysis.md unreadable at {path}: {exc!r})"
        return "(no analysis.md snapshot yet)"

    def record_reactor_conversation(
        self,
        agent_name: str,
        result: BackendTurnResult,
        *,
        context: str = "",
    ) -> None:
        """Append one ``conversations.jsonl`` row for a reactor turn, with ``context`` ahead of the backend's prompt."""
        metadata = result.metadata or {}
        prompt = metadata.get("prompt")
        response = metadata.get("response")
        if not prompt and not response:
            return
        record = ConversationRecord(
            session_id=self.session_dir.name,
            component=agent_name,
            # Same turn metadata the token row is built from, so both halves carry the backend's call_id when it
            # stamped one.
            call_id=metadata.get("call_id"),
            role=agent_name,
            tick=int(self.shared_state.tick or 0),
            phase=(self.shared_state.phase or "") or None,
            model=metadata.get("model"),
            prompt=f"{context}\n{prompt}" if context else prompt or "",
            response=response or "",
        )
        append_conversation(session_dir=self.session_dir, record=record)

    async def _explore_grid_fingerprints(self) -> set[str]:
        """Content fingerprints of the variants in every queued or running explore grid."""
        from ..actions.executors._proposal_identity import content_fingerprint

        return {
            content_fingerprint(variant)
            for task in (*await self.tasks.queued(), *await self.tasks.running())
            if task.kind == "explore"
            for variant in (task.params or {}).get("grid") or ()
            if isinstance(variant, dict)
        }

    def trajectory_digest(self) -> str:
        """The deterministic trajectory digest, or "" when building it fails."""
        from ..knowledge import trajectory_reviewer

        try:
            return trajectory_reviewer.build_trajectory_digest(self.session_dir, self.shared_state)
        except (OSError, ValueError, TypeError):
            log.exception("Coordinator: trajectory review digest failed")
            return ""

    def _orchestration_context_sections(self) -> list[str]:
        """The advisories only the orchestration agent reads, all of them session-stable."""
        return _advisory_sections(self, _ADVISORY_BLOCKS)

    async def _orchestration_turn_sections(self) -> list[str]:
        """The denial summary, proposal queue and plateau pair: only the orchestration agent reads them, and each
        moves with the session, so they are rendered per tick.
        """
        from ..predictor.prompt import queue_block
        from ..predictor.rows import PROVENANCE

        sections: list[str] = []
        denial_summary = self.shared_state.to_policy_denial_summary(top_k=6)
        if denial_summary:
            sections.append(denial_summary)
        if (self.shared_state.phase or "").strip().upper() == _phase_state.PHASE_FRAMEWORK_AGENT:
            # The predictor's rows lead the queue, so its block comes first, under the same section.
            blocks = (
                queue_block(self.shared_state, in_grids=await self._explore_grid_fingerprints()),
                self.shared_state.to_untested_proposals_summary(skip_provenance=PROVENANCE),
            )
            untested_block = "\n\n".join(block for block in blocks if block)
            if untested_block:
                sections.append("=== Untested proposals (current cycle) ===")
                sections.append(untested_block)
            kb_block = await self._kb_prompt_block(untested_block)
            if kb_block:
                sections.append(kb_block)
        plateau_block = self.plateau_advisory_block()
        if plateau_block:
            sections.append("=== Plateau advisory ===")
            sections.append(plateau_block)
            # Candidate directions are worth the digest's cost only once the session has stopped gaining.
            trajectory_block = self.trajectory_digest()
            if trajectory_block:
                sections.append("=== Trajectory review (advisory) ===")
                sections.append(trajectory_block)
        return sections

    async def _render_inbox_sections(self, agent_name: str) -> list[str]:
        """Render the inbox tail since this agent's cursor, oldest first, and note what it showed.

        What the character budget leaves out waits for the next turn.
        """
        cursor = await self.cursors.load(agent_name)
        unread = await self.bus.replay_for(agent_name, after_seq=cursor.last_processed_seq)
        rendered = await self._augment_critic_inbox_with_pending(unread) if agent_name == "critic" else unread
        # Only Orchestration acts on variant-level failures; the reviewers do not.
        variant_rows = 3 if agent_name == "orchestration" else 0
        inbox_lines = _fit_to_budget(
            [f"  {_format_inbox_event(m, max_variant_rows=variant_rows)}" for m in rendered],
            _INBOX_PROMPT_CHARS,
        )
        unread_ids = {m.msg_id for m in unread}
        read = [m for m in rendered[: len(inbox_lines)] if m.msg_id in unread_ids]
        last_seq = cursor.last_processed_seq
        if read:
            last_seq = int(read[-1].seq)
            self._rendered_cursor[agent_name] = last_seq
        if not inbox_lines:
            return [f"=== Inbox for {agent_name} ===", "(no new messages)"]
        sections = [f"=== Inbox for {agent_name} (newest last) ===", *inbox_lines]
        if len(read) < len(unread):
            sections.append(
                f"  (+{len(unread) - len(read)} newer messages after seq={last_seq} held for the next turn)"
            )
        return sections

    async def compose_prompt(self, agent_name: str) -> ComposedPrompt:
        """Compose the prompt for *agent_name*: the session context, and the per-tick blocks ending with the inbox tail."""
        context: list[str] = [f"SESSION_DIR={self.session_dir}"]
        turn: list[str] = []

        phase_block = _phase_state.phase_status_summary(self.shared_state)
        if phase_block:
            turn.append("=== Phase ===")
            turn.append(phase_block)

        if agent_name == "orchestration":
            turn.append("=== Mission progress ===")
            turn.append(self.shared_state.to_mission_summary())
            deadline = self._coord.run_deadline
            started = self._coord.run_started_monotonic
            if deadline is not None and started is not None:
                remaining_min = max(0.0, deadline.remaining() / 60.0)
                turn.append("=== Time budget ===")
                turn.append(
                    f"elapsed={(time.monotonic() - started) / 60.0:.1f}min  remaining={remaining_min:.1f}min  "
                    f"budget={self.shared_state.max_minutes or 0}min  "
                    f"closing_phase={self.shared_state.closing_phase}"
                )
                if remaining_min <= 5.0 and not self.shared_state.closing_phase:
                    turn.append(
                        "WARNING: < 5 min remaining. Avoid new `explore` rounds "
                        "(which bench every variant on the stack) — they will "
                        "likely be cut by the deadline. The Coordinator will "
                        "auto-enqueue `report` when the session closes."
                    )
        turn.append("=== Tick ===")
        turn.append(self.shared_state.to_tick_summary())
        # The session ledger moves with every action that lands, so it belongs to
        # the turn: carried in ``context`` it would rewrite the cached prefix the
        # stable blocks below sit in, on nearly every tick.
        turn.append("=== Shared session state ===")
        turn.append(self.shared_state.to_prompt_summary())
        turn.append(f"target_gap_pct={self._coord.target_gap_pct():.2f}")

        # Not wrapped, unlike the advisory blocks below: a latency budget changes
        # what a KEEP means, so losing it silently would have the model route as
        # if the session were unconstrained. Pure string assembly, no I/O.
        context.extend(self.shared_state.to_constraint_sections())
        context.append("=== Resource pools ===")
        context.append(resource_pools_summary(self.shared_state))
        if agent_name == "orchestration":
            context.extend(self._orchestration_context_sections())
            # A denial, a proposal leaving the queue or a plateau count lands between
            # two ticks, so these blocks follow the ledger into the turn: the cached
            # prefix ends above them and holds for the session.
            turn.extend(await self._orchestration_turn_sections())

        turn.extend(await self._render_inbox_sections(agent_name))

        return ComposedPrompt(context="\n".join(context), turn="\n".join(turn))

    def admit_turn(self, agent_name: str, now_unix: float) -> bool:
        """Whether ``agent_name`` takes a turn this tick; in KERNEL_AGENT, Orchestration waits for kernel progress or the heartbeat."""
        state = self.shared_state
        if agent_name != "orchestration" or (state.phase or "").upper() != _phase_state.PHASE_KERNEL_AGENT:
            return True
        fingerprint = state.kernel_progress_fingerprint
        last_turn = state.agent_last_active.get("orchestration", 0.0)
        handled = fingerprint == self._admitted_kernel_fingerprint and last_turn >= self._admitted_at
        if handled and now_unix - last_turn < _KERNEL_ORCHESTRATION_HEARTBEAT_SEC:
            return False
        self._admitted_kernel_fingerprint, self._admitted_at = fingerprint, now_unix
        return True

    async def compose_turn(self, agent_name: str, backend: Any) -> TurnPrompt:
        """The system prompt and prompt for one turn; a backend that caches its system prompt gets the session context in it."""
        system_prompt = await self.load_system_prompt(agent_name)
        composed = await self.compose_prompt(agent_name)
        if getattr(backend, "system_prompt_cached", False):
            return TurnPrompt(f"{system_prompt}\n\n{composed.context}", composed.turn, composed.context)
        return TurnPrompt(system_prompt, composed.text, "")

    async def advance_rendered_cursor(self, agent_name: str) -> None:
        """Advance an agent's read cursor to the last message its prompt rendered."""
        seq = self._rendered_cursor.get(agent_name)
        if seq is not None:
            await self.cursors.advance(agent_name, seq=seq)

    async def _augment_critic_inbox_with_pending(self, rendered: list["Message"]) -> list["Message"]:
        """Ensure every undecided proposal awaiting a Critic verdict is present."""
        pending = list(self.state.pending_proposals.values())
        if not pending:
            return rendered
        seen = {getattr(m, "msg_id", None) for m in rendered}
        extra: list["Message"] = []
        for p in pending:
            pid = str(getattr(p, "proposal_msg_id", "") or "")
            if not pid or pid in seen:
                continue
            pm = await self.bus.lookup_by_id(pid)
            if pm is not None:
                extra.append(pm)
                seen.add(pid)
        if not extra:
            return rendered
        merged = list(rendered) + extra
        merged.sort(key=lambda m: int(getattr(m, "seq", 0) or 0))
        return merged

    async def load_system_prompt(self, agent_name: str) -> str:
        """Load the system prompt for an agent, honoring overrides."""
        override = self.orch_prompt.get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # Advisory prompt blocks (folded in from the former AdvisoryCollaborator).
    def plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block for the current phase."""
        state = self.shared_state
        phase = (state.phase or "").strip().upper()
        lines: list[str] = []
        if phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            # The advisory reads the same predicate the exit rule does, so the
            # model is never shown a plateau the phase machine disagrees with.
            _, evidence = _phase_state.per_lever_dryness(state)
            config_dry = bool(evidence.get("config_arm_plateaued"))
            source_dry = bool(evidence.get("source_arm_plateaued"))
            config_ev = evidence
            source_ev = evidence
            if config_dry:
                lines.append("OPTIMIZE config arm plateaued: low recent KEEP gain plus specialist empty streak.")
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{config_ev.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={config_ev.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={config_ev.get('empty_streak', 0)} "
                    f"streak_threshold={config_ev.get('empty_streak_threshold', 0)}"
                )
            streak = int(source_ev.get("source_consecutive_no_keep", 0) or 0)
            if source_dry:
                lines.append("OPTIMIZE source arm plateaued: candidates exhausted or no KEEP on the trailing ones.")
            elif streak > 0:
                lines.append("OPTIMIZE source arm approaching plateau: no KEEP on the trailing candidates.")
            if source_dry or streak > 0:
                lines.append(
                    f"  consecutive_no_keep={streak} "
                    f"threshold={source_ev.get('source_threshold', 0)} "
                    f"candidates_exhausted={source_ev.get('source_candidates_exhausted', False)}"
                )
            if config_dry != source_dry:
                lines.append("  Only one arm is dry: the other lever is still live, and the phase stays open.")
        if not lines:
            return ""
        lines.append(
            "Note: OPTIMIZE advances to KERNEL_AGENT only when BOTH arms are dry "
            "(reason=optimize_no_more_leverage) -- a non-terminal lever switch, not "
            "the end of the run. Either arm going quiet also flags the next "
            "macro-cycle to steer off this bottleneck. You may request an earlier "
            "advance with an escalate_strategy_change hint, or keep working the live "
            "arm until the plateau / budget gate fires."
        )
        return "\n".join(lines)

    def dominant_roofline_direction(self) -> tuple[str, float]:
        """Return ``(direction, pct)`` for the most-saturated roofline direction in the latest snapshot; ``("", 0.0)`` when no snapshot is available."""
        from hyperloom.inference_optimizer.roofline_snapshot import dominant_direction

        snaps = self.shared_state.roofline_snapshots or []
        if not snaps or not isinstance(snaps[-1], dict):
            return "", 0.0
        return dominant_direction(snaps[-1])

    def _bottleneck_redirect_advisory_block(self) -> str:
        """Render the R3 cyclic bottleneck-redirect advisory (optimisation phase only)."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return ""
        saturated = {k: v for k, v in state.saturated_directions.items() if v.get("saturated")}
        has_switch = state.pending_bottleneck_switch
        if not has_switch and not saturated:
            return ""
        lines: list[str] = []
        if has_switch:
            lines.append(
                "The previous macro-cycle plateaued; redirect this cycle to a "
                "different bottleneck instead of re-mining the exhausted one."
            )
        if saturated:
            lines.append("Roofline ceiling signal: one or more lever families are saturated; deprioritize them.")
            lines.extend(
                f"  saturated_domain={domain} direction={row.get('direction')} "
                f"within={row.get('within_pct')}% threshold={row.get('threshold_pct')}%"
                for domain, row in sorted(saturated.items())
            )
        if state.last_cycle_bottleneck:
            lines.append(f"  plateaued_bottleneck={state.last_cycle_bottleneck} (avoid re-targeting)")
        cur_top = state.current_top_bottleneck()
        if cur_top:
            lines.append(f"  current_top_bottleneck={cur_top}")
        shift = state.bottleneck_shift
        if shift.get("from") or shift.get("to"):
            lines.append(
                f"  bottleneck_shift: {shift.get('from') or 'unknown'} → {shift.get('to') or 'unknown'} "
                f"(within_delta={shift.get('within_delta')} gap_delta={shift.get('gap_delta')})"
            )
        direction, pct = self.dominant_roofline_direction()
        if direction:
            from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

            hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
            suffix = f" → suggested specialist domain={hint[0]} tag={hint[1]}" if hint else ""
            lines.append(f"  dominant_direction={direction} ({pct:.1f}%){suffix}")
        lines.append(f"  macro_cycle={state.macro_cycle}")
        lines.append("Advisory only: pick the domain/tag yourself; this nudges focus, it does not gate dispatch.")
        return "\n".join(lines)

    def _acceptance_threshold_advisory_block(self) -> str:
        """Render the decaying acceptance bar and prior measured gains as evidence."""
        state = self.shared_state
        keep = _phase_state.resolve_keep_threshold(state)
        cycle = state.macro_cycle
        if cycle < 1:
            return ""
        stable = keep / 2.0
        search = state.explore_search
        entries = [*(search.get("tested") or {}).values(), *(search.get("rejected") or [])]
        above_bar: list[tuple[str, float]] = []
        below_bar: list[tuple[str, float]] = []
        for e in entries:
            try:
                g = float(e.get("gain_pct"))
            except (TypeError, ValueError):
                continue
            name = str(e.get("name") or e.get("fingerprint") or "")[:48]
            (above_bar if g >= keep else below_bar).append((name, g))
        above_bar.sort(key=itemgetter(1), reverse=True)
        below_bar.sort(key=itemgetter(1), reverse=True)
        lines: list[str] = [
            f"Current acceptance bar (macro_cycle={cycle}): KEEP>={keep:.2f}% stack_stable>={stable:.2f}%.",
            "Historical results are evidence only — any fingerprint may be re-proposed.",
        ]
        if above_bar:
            lines.append("Prior results above the bar (consider re-proposing if conditions changed):")
            lines.extend(f"  {name}: prior gain {g:+.2f}% >= {keep:.2f}%" for name, g in above_bar[:8])
        if below_bar:
            lines.append("Prior results below the bar (reference):")
            lines.extend(f"  {name}: prior gain {g:+.2f}% < {keep:.2f}%" for name, g in below_bar[:5])
        return "\n".join(lines)

    def _current_target_gap(self) -> dict[str, Any] | None:
        """Current-best vs the competitor target; None when the advisory is off or no target is set."""
        if not self.shared_state.target_advisory_enabled:
            return None
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        target = _research_hints.load_competitor_target(self.session_dir)
        return _research_hints.gap_for_state(target, self.shared_state) if target else None

    def target_gap_advisory_block(self) -> str:
        """Build the advisory \"External target gap\" prompt block (current-best vs competitor target; never gates)."""
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        return _research_hints.full_gap_summary(self._current_target_gap())

    def _current_primary_gap(self) -> str | None:
        """Resolve latency/throughput; None when advisory is off or no competitor target is set."""
        gap = self._current_target_gap()
        return str((gap or {}).get("primary_gap") or "").strip() or None

    def _recent_proposed_variants(
        self,
        *,
        max_rounds: int = 2,
    ) -> list[dict[str, Any]]:
        """Collect proposal_set rows from the most recent specialist rounds (deduped by name; fail-soft)."""
        rounds = [
            r
            for r in (self.shared_state.specialist_rounds or [])
            if isinstance(r, dict) and isinstance(r.get("proposal_set"), list)
        ]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rounds[-max_rounds:]:
            for variant in r.get("proposal_set") or []:
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(variant)
        return out

    def _specialist_findings_block(self) -> str:
        """Render persisted specialist findings, any domain, most recent first.

        Executable proposals are not rendered here: they go through
        ``=== Untested proposals (current cycle) ===`` alongside every other
        domain's, which also drops the ones already benched.

        Rows are ordered by recency rather than by the round's self-reported
        ``confidence``: that field is an audit record of what the specialist
        claimed, never an input to a decision here.

        ``research_hints.json`` is not read: it holds the sourced subset of
        these same findings. Each list keeps its newest entries within its
        character ceiling and counts the rest.
        """
        findings: list[str] = []
        questions: list[str] = []
        seen_questions: set[str] = set()
        for row in reversed(self.shared_state.specialist_rounds or []):
            if not isinstance(row, dict):
                continue
            domain_label = str(row.get("domain") or "").strip()
            for finding in row.get("new_findings") or []:
                text = json.dumps(finding, sort_keys=True) if isinstance(finding, dict) else str(finding)
                findings.append(f"- [{domain_label}] {text}")
            for question in row.get("residual_questions") or []:
                text = str(question).strip()
                if text and text not in seen_questions:
                    seen_questions.add(text)
                    questions.append(f"- [{domain_label}] {text}")
        if not findings and not questions:
            return ""

        lines = ["=== Specialist findings ==="]
        for title, items, budget_chars in (
            ("Findings", findings, _FINDINGS_PROMPT_CHARS),
            ("Residual questions", questions, _QUESTIONS_PROMPT_CHARS),
        ):
            if not items:
                continue
            shown = _fit_to_budget(items, budget_chars)
            lines.append(f"{title}:")
            lines.extend(shown)
            if len(shown) < len(items):
                lines.append(f"(+{len(items) - len(shown)} older omitted)")
        return "\n".join(lines)

    def _priors_match_advisory_block(self) -> str:
        """Flag recently proposed variants aligning with proven priors / dominant external gap (advisory ordering)."""
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        variants = self._recent_proposed_variants()
        if not variants:
            return ""
        hints = _research_hints.load_hints(self.session_dir)
        primary_gap = self._current_primary_gap()
        return _research_hints.priors_match_summary(
            variants,
            hints,
            primary_gap=primary_gap,
        )

    def _discarded_escalate_hint_advisory_block(self) -> str:
        """Render the advisory for an escalate_strategy_change hint that was discarded.

        A phase transition the hint did not drive drops it.

        Returns:
            The advisory string, or ``""`` when no discarded hint is recorded.
        """
        hint = str(self.shared_state.last_discarded_escalate_hint or "")
        ts = str(self.shared_state.last_discarded_escalate_hint_ts or "")
        if not hint:
            return ""
        return (
            f"ADVISORY: your escalate_strategy_change hint '{hint}' (at {ts}) was discarded "
            "because a phase transition it did not drive fired before "
            "it could be consumed. Re-emit escalate_strategy_change if still needed."
        )
