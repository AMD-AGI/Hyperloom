# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover GEAK's LLM spend from the Claude Code transcripts it leaves behind.

GEAK runs out-of-process (``interface/run_e2e.py``), drives its own
``ClaudeSDKClient`` with ``cwd`` set to its ``e2e_workflow`` directory, and fans
that client out into a Workflow whose agents are separate conversations. None of
it reaches Hyperloom's in-process ledger, yet it is the majority of a session's
model bill -- so it has to be read back off disk afterwards.

Two artifacts carry everything needed:

* ``<claude_home>/projects/<slug>/<session>.jsonl`` -- the driving conversation,
  and ``<session>/subagents/workflows/<run>/agent-<id>.jsonl`` one per Workflow
  agent. These hold the per-API-call token counts.
* ``<session>/workflows/<run>.json`` -- the Workflow's own record, whose
  ``workflowProgress`` names each agent's GEAK phase (Setup, Profile,
  HeadKernel, ...) and its label (``director:setup``,
  ``extract_op ck_..._gemm``). That is GEAK's task tree, and it becomes the
  ``task_path`` under which the report nests these calls.

Selection is deliberately narrow: a transcript counts only when its ``cwd`` is
GEAK's workflow directory *and* it mentions this session's ``exp_root``. Several
Hyperloom runs share one GEAK checkout, so the second predicate is what keeps a
neighbour's spend off this session's ledger.

Rows go to the session's ``reports/trace/ext/`` shard, never to
``llm_calls.jsonl`` directly: that append is not atomic across processes and the
session tree lives on a network filesystem. The collector already merges the
shard back in at read time.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.session_paths import (
    trace_ext_dir,
    trace_ext_shard_path,
)
from .call_detail import CallDetailRecord, append_call_detail, split_turn_timing
from .llm_trace import LLMCallRecord, append_llm_call, new_call_id
from .parse_usage import parse_claude_transcript_calls

log = logging.getLogger(__name__)

#: Trace component label for everything GEAK spends.
GEAK_COMPONENT = "geak"

#: First segment of every harvested ``task_path``, so a report can lift GEAK's
#: whole subtree out of the phase that delegated to it in one filter.
ROOT_SEGMENT = "geak"

#: Task-path segment for the driving conversation itself -- the client that
#: invokes the Workflow tool, as opposed to the agents it fans out into.
RUNNER_SEGMENT = "runner"

#: Splits a Workflow agent label into path segments. Labels are written either
#: colon-separated (``director:setup``) or as a verb and its target
#: (``extract_op ck_bpreshuffle_fp8_a8w8_blockscale_gemm``); both become the
#: role / lane / kernel levels of the tree.
_LABEL_SPLIT = re.compile(r"[:\s]+")

#: Name of the harvest bookkeeping file inside ``reports/trace/ext/``. It is
#: what makes a second harvest (one per macro cycle) additive instead of
#: duplicating every call the first one already wrote.
_STATE_FILENAME = "geak_harvest_state.json"


@dataclass(frozen=True)
class HarvestResult:
    """What one harvest pass recovered.

    Attributes:
        transcripts: Conversations that matched this session and had new calls.
        turns: Turn-level rows written to the ``ext`` shard.
        api_calls: Per-API-call rows written to the ``ext`` detail shard.
    """

    transcripts: int = 0
    turns: int = 0
    api_calls: int = 0


def claude_projects_dir(claude_home: Path | None = None) -> Path:
    """Locate the directory Claude Code keeps its session transcripts in.

    Args:
        claude_home: Explicit Claude home, for tests. Defaults to
            ``$CLAUDE_CONFIG_DIR`` and then ``~/.claude``.

    Returns:
        The ``projects`` directory, which may not exist.
    """
    if claude_home is None:
        configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
        claude_home = Path(configured) if configured else Path.home() / ".claude"
    return Path(claude_home) / "projects"


def _label_segments(label: str | None) -> tuple[str, ...]:
    """Split a Workflow agent label into ``task_path`` segments.

    An unrecognized label is kept whole rather than dropped: a call attributed
    to a strange-looking node is still attributed, whereas a dropped one
    silently shrinks the total.

    Args:
        label: The Workflow's own label for the agent.

    Returns:
        The path segments, or ``()`` when there is no label.
    """
    text = (label or "").strip()
    if not text:
        return ()
    return tuple(seg for seg in _LABEL_SPLIT.split(text) if seg)


def _parse_ts(value: Any) -> datetime | None:
    """Parse a transcript ISO-8601 timestamp.

    Args:
        value: The record's ``timestamp`` field.

    Returns:
        The parsed datetime, or ``None`` when it is absent or malformed.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _matches_session(path: Path, *, workflow_dir: str | None, marker: str) -> bool:
    """Decide whether a transcript belongs to this session's GEAK subprocess.

    Args:
        path: The transcript to test.
        workflow_dir: GEAK's ``e2e_workflow`` directory, or ``None`` to accept
            any ``cwd``.
        marker: A string only this session's run mentions -- its ``exp_root``.

    Returns:
        True when both predicates hold.
    """
    saw_cwd = workflow_dir is None
    saw_marker = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not saw_marker and marker in line:
                    saw_marker = True
                if not saw_cwd and '"cwd"' in line:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(record, dict) and str(record.get("cwd") or "") == workflow_dir:
                        saw_cwd = True
                if saw_cwd and saw_marker:
                    return True
    except OSError as exc:
        log.debug("geak_harvest: unreadable transcript %s: %r", path, exc)
    return False


def _agent_index(session_root: Path) -> dict[str, dict[str, Any]]:
    """Map every Workflow agent id to its phase, label and measured span.

    Args:
        session_root: ``<projects>/<slug>/<session_id>/``, the sidecar tree
            Claude Code writes next to a transcript.

    Returns:
        ``agentId`` -> the agent's ``workflowProgress`` entry. Empty when the
        session ran no Workflow.
    """
    index: dict[str, dict[str, Any]] = {}
    for record_path in sorted((session_root / "workflows").glob("*.json")):
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.debug("geak_harvest: unreadable workflow record %s: %r", record_path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("workflowProgress") or []:
            if not isinstance(entry, dict) or entry.get("type") != "workflow_agent":
                continue
            agent_id = str(entry.get("agentId") or "").strip()
            if agent_id:
                index[agent_id] = entry
    return index


def _load_state(path: Path) -> dict[str, Any]:
    """Read the harvest bookkeeping file.

    Args:
        path: The state file.

    Returns:
        Transcript path -> ``{"size", "calls"}``. Empty on a first harvest or
        an unreadable file, which costs at worst a repeat of work already done.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Write the harvest bookkeeping file, best-effort.

    Args:
        path: The state file.
        state: The mapping to persist.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.debug("geak_harvest: could not persist state to %s: %r", path, exc)


def _emit_conversation(
    *,
    transcript: Path,
    calls: list[dict[str, Any]],
    session_dir: Path,
    session_id: str,
    shard: Path,
    detail_shard: Path,
    task_path: tuple[str, ...],
    phase: str | None,
    task_id: str | None,
    dyn_id: str | None,
    tick: int | None,
    turn: int | None,
    started_ms: int | None,
    duration_ms: int | None,
) -> int:
    """Write one conversation's calls out as a turn row plus its detail rows.

    A GEAK agent is one conversation of many requests, so it maps onto a single
    turn row carrying the summed counters and an ``api_calls`` count, with one
    detail row per request joined on the shared ``call_id``.

    Per-request latency is the gap between message arrivals, anchored for the
    first request on the Workflow's recorded ``startedAt`` when there is one.
    That measures queueing plus generation together, which is the only span a
    transcript of whole messages actually witnesses.

    Args:
        transcript: The conversation being harvested, for logging.
        calls: Rows from :func:`parse_claude_transcript_calls`, in order.
        session_dir: Hyperloom session directory.
        session_id: Hyperloom session id, the cross-process join key.
        shard: Destination for the turn row.
        detail_shard: Destination for the per-API-call rows.
        task_path: Where this conversation sits in GEAK's task tree.
        phase: The Hyperloom phase that delegated to GEAK.
        task_id: Hyperloom task id of the delegation, when known.
        dyn_id: The Workflow agent id, which separates two agents that share a
            label.
        tick: Hyperloom timeline tick of the delegation.
        turn: Index of this conversation among the run's agents.
        started_ms: Workflow ``startedAt`` epoch-ms, when recorded.
        duration_ms: Workflow ``durationMs``, the agent's measured span.

    Returns:
        The number of detail rows written.
    """
    if not calls:
        return 0
    call_id = new_call_id()
    path_str = "/".join(task_path)
    totals: dict[str, int] = {}
    tool_total = 0
    previous_ms = started_ms
    written = 0
    for index, call in enumerate(calls):
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "reasoning_output_tokens",
        ):
            value = call.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
        tool_calls = call.get("tool_calls") or []
        tool_total += len(tool_calls)
        arrived = _parse_ts(call.get("ts"))
        arrived_ms = int(arrived.timestamp() * 1000) if arrived is not None else None
        latency_ms: int | None = None
        if arrived_ms is not None and previous_ms is not None and arrived_ms >= previous_ms:
            latency_ms = arrived_ms - previous_ms
        if arrived_ms is not None:
            previous_ms = arrived_ms
        thinking_ms, output_ms, timing_source = split_turn_timing(
            latency_ms=latency_ms,
            output_tokens=call.get("output_tokens"),
            reasoning_output_tokens=call.get("reasoning_output_tokens"),
        )
        append_call_detail(
            session_dir=session_dir,
            dest=detail_shard,
            record=CallDetailRecord(
                session_id=session_id,
                component=GEAK_COMPONENT,
                call_id=call_id,
                api_call_index=index,
                role=task_path[-1] if task_path else None,
                task_id=task_id,
                dyn_id=dyn_id,
                tick=tick,
                phase=phase,
                turn=turn,
                task_path=path_str,
                model=call.get("model"),
                input_tokens=call.get("input_tokens"),
                output_tokens=call.get("output_tokens"),
                cache_creation_input_tokens=call.get("cache_creation_input_tokens"),
                cache_read_input_tokens=call.get("cache_read_input_tokens"),
                reasoning_output_tokens=call.get("reasoning_output_tokens"),
                latency_ms=latency_ms,
                thinking_ms=thinking_ms,
                output_ms=output_ms,
                timing_source=timing_source,
                total_cost_usd=call.get("cost_usd"),
                started_ts=call.get("ts"),
                tool_calls=tool_calls,
                stop_reason=call.get("stop_reason"),
            ),
        )
        written += 1

    turn_thinking, turn_output, _ = split_turn_timing(
        latency_ms=duration_ms,
        output_tokens=totals.get("output_tokens"),
        reasoning_output_tokens=totals.get("reasoning_output_tokens"),
    )
    append_llm_call(
        session_dir=session_dir,
        dest=shard,
        record=LLMCallRecord(
            session_id=session_id,
            component=GEAK_COMPONENT,
            call_id=call_id,
            role=task_path[-1] if task_path else None,
            task_id=task_id,
            dyn_id=dyn_id,
            tick=tick,
            phase=phase,
            turn=turn,
            model=calls[-1].get("model"),
            input_tokens=totals.get("input_tokens"),
            output_tokens=totals.get("output_tokens"),
            cache_creation_input_tokens=totals.get("cache_creation_input_tokens"),
            cache_read_input_tokens=totals.get("cache_read_input_tokens"),
            reasoning_output_tokens=totals.get("reasoning_output_tokens"),
            latency_ms=duration_ms,
            thinking_ms=turn_thinking,
            output_ms=turn_output,
            task_path=path_str,
            api_calls=written,
            tool_call_count=tool_total,
            stop_reason=calls[-1].get("stop_reason"),
        ),
    )
    log.debug("geak_harvest: %s -> %d calls under %s", transcript.name, written, path_str)
    return written


def harvest_geak_calls(
    *,
    session_dir: Path,
    session_id: str,
    exp_root: str | Path,
    workflow_dir: str | Path | None = None,
    phase: str | None = None,
    task_id: str | None = None,
    tick: int | None = None,
    not_before: float | None = None,
    claude_home: Path | None = None,
    pid: int | None = None,
) -> HarvestResult:
    """Fold one GEAK subprocess's spend into this session's trace ledger.

    Safe to call more than once: what each transcript has already contributed is
    remembered in ``reports/trace/ext/geak_harvest_state.json``, so a second
    macro cycle adds only its own calls.

    Args:
        session_dir: Hyperloom session directory.
        session_id: Hyperloom session id, stamped on every row.
        exp_root: The run's GEAK experiment root. Its path string is the marker
            that distinguishes this run's transcripts from a neighbour's in a
            shared checkout.
        workflow_dir: GEAK's ``e2e_workflow`` directory, matched against each
            transcript's ``cwd``. ``None`` accepts any, which is looser than the
            caller normally wants.
        phase: Hyperloom phase that delegated to GEAK.
        task_id: Hyperloom task id of the delegation.
        tick: Hyperloom timeline tick of the delegation.
        not_before: Epoch seconds before the subprocess started. Transcripts
            untouched since then are skipped without being read, which keeps
            the scan cheap on a box with a long Claude Code history.
        claude_home: Explicit Claude home, for tests.
        pid: Owning process id for the shard filename; defaults to this one.

    Returns:
        A :class:`HarvestResult` counting what was written.
    """
    projects = claude_projects_dir(claude_home)
    if not projects.is_dir():
        log.debug("geak_harvest: no claude projects dir at %s", projects)
        return HarvestResult()

    marker = str(exp_root)
    wf_dir = str(workflow_dir) if workflow_dir is not None else None
    owner = int(pid if pid is not None else os.getpid())
    shard = trace_ext_shard_path(session_dir, GEAK_COMPONENT, owner)
    detail_shard = trace_ext_shard_path(session_dir, GEAK_COMPONENT, owner, detail=True)
    state_path = trace_ext_dir(session_dir) / _STATE_FILENAME
    state = _load_state(state_path)

    transcripts = 0
    turns = 0
    api_calls = 0
    for transcript in sorted(projects.glob("*/*.jsonl")):
        if not_before is not None:
            try:
                if transcript.stat().st_mtime < not_before:
                    continue
            except OSError:
                continue
        if not _matches_session(transcript, workflow_dir=wf_dir, marker=marker):
            continue
        session_root = transcript.with_suffix("")
        agents = _agent_index(session_root)
        conversations: list[tuple[Path, tuple[str, ...], str | None, dict[str, Any]]] = [
            (transcript, (ROOT_SEGMENT, RUNNER_SEGMENT), None, {})
        ]
        for agent_path in sorted((session_root / "subagents").rglob("agent-*.jsonl")):
            agent_id = agent_path.stem[len("agent-") :]
            entry = agents.get(agent_id, {})
            segments = (
                ROOT_SEGMENT,
                str(entry.get("phaseTitle") or "").strip() or "unlabelled",
                *_label_segments(entry.get("label")),
            )
            conversations.append((agent_path, segments, agent_id, entry))

        matched = False
        for turn, (path, segments, agent_id, entry) in enumerate(conversations):
            key = str(path)
            seen = state.get(key) if isinstance(state.get(key), dict) else {}
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if seen.get("size") == size:
                continue
            calls = parse_claude_transcript_calls(path)
            already = int(seen.get("calls") or 0)
            fresh = calls[already:]
            state[key] = {"size": size, "calls": len(calls)}
            if not fresh:
                continue
            written = _emit_conversation(
                transcript=path,
                calls=fresh,
                session_dir=session_dir,
                session_id=session_id,
                shard=shard,
                detail_shard=detail_shard,
                task_path=segments,
                phase=phase,
                task_id=task_id,
                dyn_id=agent_id,
                tick=tick,
                turn=turn,
                started_ms=entry.get("startedAt") if isinstance(entry.get("startedAt"), int) else None,
                duration_ms=entry.get("durationMs") if isinstance(entry.get("durationMs"), int) else None,
            )
            if written:
                matched = True
                turns += 1
                api_calls += written
        if matched:
            transcripts += 1

    _save_state(state_path, state)
    log.info(
        "geak_harvest: %d transcript(s), %d turn row(s), %d api-call row(s) -> %s",
        transcripts,
        turns,
        api_calls,
        shard.name,
    )
    return HarvestResult(transcripts=transcripts, turns=turns, api_calls=api_calls)


__all__ = [
    "GEAK_COMPONENT",
    "ROOT_SEGMENT",
    "RUNNER_SEGMENT",
    "HarvestResult",
    "claude_projects_dir",
    "harvest_geak_calls",
]
