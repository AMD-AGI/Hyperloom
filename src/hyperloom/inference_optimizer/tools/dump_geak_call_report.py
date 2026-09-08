#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a GEAK run as a phase -> agent -> API call -> tool call tree.

GEAK issues almost no LLM calls of its own: ``interface/run_e2e.py`` opens one
``ClaudeSDKClient`` and hands a single prompt to Claude Code, which executes
``e2e_workflow/e2e_workflow.js`` -- a Workflow script that already declares its
phases and already tags every ``agent()`` call with ``{phase, label}``. Claude
Code persists that structure, so this module reads it rather than asking GEAK
to write a ledger it has no way to fill:

* ``<claude_home>/projects/<slug>/<session>/workflows/wf_*.json`` -- the run
  record: ``phases``, ``args`` (carrying ``eval_dir``), ``result``, and a
  ``workflowProgress`` array whose ``workflow_agent`` entries name every agent
  with its label, phase, ``agentId``, model, tokens, tool calls and duration.
* ``<session>/subagents/workflows/<runId>/agent-<agentId>.jsonl`` -- one row
  per API call, with the ``usage`` counters and the ``tool_use`` blocks.

Usage::

    python3 dump_geak_call_report.py --eval-dir /path/to/geak/e2e_cycle0
    python3 dump_geak_call_report.py --run-id wf_ec8b57b0-1a7 --max-depth 3

Writes ``geak_call_report.md`` and ``geak_call_report.json`` to
``$USER_DATA_PATH/reports/<runId>/`` unless ``--output-dir`` says otherwise.

Three properties matter more than the totals:

* Assistant rows sharing a ``message.id`` are one API call split per content
  block, and their ``usage`` is cumulative -- counted once, from the last
  block, or the report overstates output by roughly the block count.
* GEAK's ``thinking_tokens`` are a **subset** of ``output_tokens``, not an
  addend beside them (the opposite of Hyperloom's own ledger), so OSL is
  ``output_tokens`` alone.
* Claude Code records no cost, so every figure here is derived from the
  shipped rate card. An unpriced model is excluded from the USD total rather
  than counted as free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "hyperloom").is_dir():
        sys.path.insert(0, str(_parent))
        break

from hyperloom.inference_optimizer.tools.dump_llm_call_report import (  # noqa: E402
    Node,
    _render_rows,
    _trim,
)
from hyperloom.orchestrator.trace.pricing import resolve_cost  # noqa: E402

WORKFLOW_GLOB = "projects/*/*/workflows/wf_*.json"
AGENT_ENTRY = "workflow_agent"
PHASE_ENTRY = "workflow_phase"
TIMING_INFERRED = "inferred"


def claude_homes(extra: Iterable[Path] = ()) -> list[Path]:
    """List the Claude Code home directories to search for run records.

    Archived GEAK runs live under whichever account drove them, so the caller
    can name additional roots; the configured home is always searched first.

    Args:
        extra: Additional roots supplied on the command line.

    Returns:
        Existing home directories, in search order, without duplicates.
    """
    seen: dict[str, Path] = {}
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    ordered = [Path(configured)] if configured else []
    ordered.append(Path.home() / ".claude")
    ordered.extend(Path(p) for p in extra)
    for home in ordered:
        try:
            resolved = home.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_dir():
            seen.setdefault(str(resolved), resolved)
    return list(seen.values())


def find_run_records(homes: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield every workflow run record found beneath *homes*.

    Args:
        homes: Claude Code home directories.

    Yields:
        ``(path, record)`` for each readable ``wf_*.json``.
    """
    for home in homes:
        for path in sorted(home.glob(WORKFLOW_GLOB)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("runId"):
                yield path, record


def _record_paths(record: dict[str, Any]) -> list[str]:
    """Return the run directories a record names, most specific first.

    A Hyperloom-driven run is identified by ``eval_dir``; a standalone one is
    usually reached for by its ``exp_root``, which is the directory the caller
    actually has to hand.

    Args:
        record: A workflow run record.

    Returns:
        Distinct directory strings, ``eval_dir`` before ``exp_root``.
    """
    found: list[str] = []
    for key in ("eval_dir", "exp_root"):
        for holder in (record.get("args"), record.get("result")):
            if not isinstance(holder, dict):
                continue
            value = holder.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip().rstrip("/")
                if cleaned not in found:
                    found.append(cleaned)
    return found


def _record_eval_dir(record: dict[str, Any]) -> str:
    """Return the run directory a record is best identified by.

    Args:
        record: A workflow run record.

    Returns:
        The most specific recorded directory, or ``""``.
    """
    paths = _record_paths(record)
    return paths[0] if paths else ""


def resolve_runs(
    *,
    homes: Iterable[Path],
    eval_dir: str | None = None,
    run_id: str | None = None,
    session_dir: Path | None = None,
    list_all: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    """Find the run records matching a selector, newest first.

    Matching is an identity check on ``args.eval_dir``, not an mtime or cwd
    heuristic: a Hyperloom session joins through its ``geak/e2e_cycle*`` dir
    and a standalone run through its experiment dir.

    Args:
        homes: Claude Code home directories to search.
        eval_dir: The GEAK eval dir to match, exactly or as a prefix.
        run_id: A ``wf_...`` run id to match instead.
        session_dir: A Hyperloom session root whose ``geak/`` subtree supplies
            the eval dir.
        list_all: Return every readable record, ignoring the selectors.

    Returns:
        Matching ``(path, record)`` pairs, newest ``timestamp`` first.
    """
    wanted = (eval_dir or "").strip().rstrip("/")
    if not wanted and session_dir is not None:
        wanted = str(session_dir.resolve() / "geak")
    hits: list[tuple[Path, dict[str, Any]]] = []
    for path, record in find_run_records(homes):
        if run_id and record.get("runId") != run_id:
            continue
        if wanted:
            if not any(
                found == wanted or found.startswith(wanted + "/") or wanted.startswith(found + "/")
                for found in _record_paths(record)
            ):
                continue
        elif not run_id and not list_all:
            continue
        hits.append((path, record))
    return sorted(hits, key=lambda pr: str(pr[1].get("timestamp") or ""), reverse=True)


def _ts_ms(value: Any) -> float | None:
    """Parse an ISO-8601 timestamp into epoch milliseconds.

    Args:
        value: A timestamp string from a transcript row.

    Returns:
        Epoch milliseconds, or ``None`` when unparseable.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError:
        return None


def _tool_uses(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the tool_use blocks of one assistant message.

    Args:
        message: The ``message`` object of a transcript row.

    Returns:
        One ``{tool, tool_use_id}`` mapping per tool_use block.
    """
    blocks = message.get("content")
    if not isinstance(blocks, list):
        return []
    return [
        {"tool": b.get("name") or "?", "tool_use_id": b.get("id")}
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def _has_thinking(message: dict[str, Any]) -> bool:
    """Report whether an assistant message carries a thinking block.

    Args:
        message: The ``message`` object of a transcript row.

    Returns:
        ``True`` when any content block is of type ``thinking``.
    """
    blocks = message.get("content")
    return isinstance(blocks, list) and any(isinstance(b, dict) and b.get("type") == "thinking" for b in blocks)


def _block_text(blocks: Any, kinds: tuple[str, ...], limit: int) -> str:
    """Join the text carried by content blocks of the given kinds.

    Args:
        blocks: A message's ``content``.
        kinds: Block types to keep, e.g. ``("text", "thinking")``.
        limit: Per-block character cap; ``0`` means uncapped.

    Returns:
        The joined text, blocks separated by blank lines.
    """
    if not isinstance(blocks, list):
        return str(blocks)[:limit] if limit and isinstance(blocks, str) else ""
    out: list[str] = []
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") not in kinds:
            continue
        text = b.get("text") or b.get("thinking") or b.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text)
        if limit and len(text) > limit:
            text = text[:limit] + f"... [+{len(text) - limit} chars]"
        out.append(text)
    return "\n\n".join(out)


def read_agent_calls(path: Path, *, capture_text: bool = False, text_chars: int = 4000) -> list[dict[str, Any]]:
    """Read one agent transcript into normalized per-API-call rows.

    Rows sharing a ``message.id`` are the same API call split across content
    blocks with cumulative usage, so they collapse to the highest
    ``apiBlockIndex`` while their tool_use blocks are unioned. Latency is the
    gap from the previous transcript row -- a tool_result timestamp is when
    the tool finished, so that gap approximates API time rather than tool time.

    Args:
        path: An ``agent-<agentId>.jsonl`` transcript.
        capture_text: Also carry the prompt and response text of each call.
        text_chars: Per-block character cap for captured text; ``0`` is uncapped.

    Returns:
        Normalized rows in the field vocabulary
        :class:`~hyperloom.inference_optimizer.tools.dump_llm_call_report.Totals`
        consumes, ordered by timestamp.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    prev_ts: float | None = None
    gap_for: dict[str, float] = {}
    pending: list[str] = []
    for line in _iter_lines(path):
        row_ts = _ts_ms(line.get("timestamp"))
        message = line.get("message")
        if capture_text and line.get("type") == "user" and isinstance(message, dict):
            # What the model was handed since it last spoke: the agent prompt on
            # the first call, tool results after that. The full context is ISL;
            # this is the increment that provoked the call.
            text = _block_text(message.get("content"), ("text", "tool_result"), text_chars)
            if text:
                pending.append(text)
        if line.get("type") == "assistant" and isinstance(message, dict) and message.get("usage"):
            key = str(message.get("id") or f"anon-{len(order)}")
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "block": -1,
                    "message": message,
                    "ts": row_ts,
                    "tools": [],
                    "thinking_block": False,
                    "prompt": "\n\n".join(pending),
                    "output": [],
                    "thinking": [],
                }
                pending = []
                grouped[key] = entry
                order.append(key)
                if prev_ts is not None and row_ts is not None:
                    gap_for[key] = max(0.0, row_ts - prev_ts)
            block = int(line.get("apiBlockIndex") or 0)
            if block >= entry["block"]:
                entry["block"] = block
                entry["message"] = message
            if row_ts is not None:
                entry["ts"] = row_ts
            entry["tools"].extend(_tool_uses(message))
            entry["thinking_block"] = entry["thinking_block"] or _has_thinking(message)
            if capture_text:
                for kind, key in (("text", "output"), ("thinking", "thinking")):
                    text = _block_text(message.get("content"), (kind,), text_chars)
                    if text:
                        entry[key].append(text)
        if row_ts is not None:
            prev_ts = row_ts

    calls: list[dict[str, Any]] = []
    for index, key in enumerate(order):
        entry = grouped[key]
        call = _normalize_call(
            message=entry["message"],
            message_id=key,
            index=index,
            ts_ms=entry["ts"],
            latency_ms=gap_for.get(key),
            tools=entry["tools"],
            thinking_block=entry["thinking_block"],
        )
        if capture_text:
            call["prompt_text"] = entry["prompt"]
            call["output_text"] = "\n\n".join(entry["output"])
            call["thinking_text"] = "\n\n".join(entry["thinking"])
        calls.append(call)
    return calls


def _iter_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the parseable JSON objects of a JSONL file.

    Args:
        path: File to read.

    Yields:
        Each decoded object; malformed lines are skipped.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def _normalize_call(
    *,
    message: dict[str, Any],
    message_id: str,
    index: int,
    ts_ms: float | None,
    latency_ms: float | None,
    tools: list[dict[str, Any]],
    thinking_block: bool,
) -> dict[str, Any]:
    """Project one API call onto the report's row vocabulary.

    Args:
        message: The assistant message carrying the final cumulative usage.
        message_id: The provider ``message.id``, the call's identity.
        index: Zero-based position within the agent.
        ts_ms: Completion time in epoch milliseconds.
        latency_ms: Inferred wall-clock for the call.
        tools: The tool_use blocks the call emitted.
        thinking_block: Whether the response contained a thinking block.

    Returns:
        A row ready for :meth:`Totals.add_call`.
    """
    usage = message.get("usage") or {}
    raw_out = int(usage.get("output_tokens") or 0)
    thinking = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
    thinking = max(0, min(thinking, raw_out))
    row: dict[str, Any] = {
        "call_id": message_id,
        "api_call_index": index,
        "model": message.get("model"),
        "stop_reason": message.get("stop_reason"),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        # thinking_tokens are part of output_tokens here, so the plain-output
        # counter is the remainder. Adding them would double-bill the split.
        "output_tokens": raw_out - thinking,
        "reasoning_output_tokens": thinking,
        "thinking_block": thinking_block,
        "tool_calls": tools,
        "ts_ms": ts_ms,
        "status": "ok",
    }
    row["isl"] = row["input_tokens"] + row["cache_read_input_tokens"] + row["cache_creation_input_tokens"]
    row["osl"] = raw_out
    if latency_ms is not None:
        row["latency_ms"] = latency_ms
        row["timing_source"] = TIMING_INFERRED
    cost = resolve_cost(model=row["model"], tokens=row)
    row.update(
        {
            "cost_usd": cost.total_usd,
            "cost_input_usd": cost.input_usd,
            "cost_output_usd": cost.output_usd,
            "cost_thinking_usd": cost.thinking_usd,
            "cost_cache_usd": cost.cache_usd,
            "cost_source": cost.source,
        }
    )
    return row


def transcript_dir(record_path: Path, run_id: str) -> Path:
    """Locate the per-agent transcripts belonging to a run record.

    Args:
        record_path: Path of the ``wf_*.json`` record.
        run_id: The record's ``runId``.

    Returns:
        The ``subagents/workflows/<runId>`` directory, which may not exist.
    """
    return record_path.parent.parent / "subagents" / "workflows" / run_id


def _phase_order(progress: Iterable[Any]) -> list[str]:
    """Return the declared phase titles in execution order.

    Args:
        progress: The ``workflowProgress`` array.

    Returns:
        Phase titles, first appearance order preserved.
    """
    titles: list[str] = []
    for entry in progress:
        if isinstance(entry, dict) and entry.get("type") == PHASE_ENTRY:
            title = str(entry.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
    return titles


def _agent_entries(progress: Iterable[Any]) -> list[dict[str, Any]]:
    """Return the latest state of every agent named in the progress array.

    An agent is reported repeatedly as it advances, so the last entry for an
    ``agentId`` is the one that carries its final tokens and duration.

    Args:
        progress: The ``workflowProgress`` array.

    Returns:
        One entry per agent, in first-appearance order.
    """
    latest: dict[str, dict[str, Any]] = {}
    for entry in progress:
        if not isinstance(entry, dict) or entry.get("type") != AGENT_ENTRY:
            continue
        key = str(entry.get("agentId") or entry.get("label") or len(latest))
        latest[key] = {**latest.get(key, {}), **entry}
    return list(latest.values())


def session_transcript(record_path: Path) -> Path:
    """Locate the orchestrator conversation that launched a run.

    ``run_e2e.py`` opens one ``ClaudeSDKClient`` and sends a single prompt; that
    conversation is a sibling ``<session-uuid>.jsonl`` of the session directory.

    Args:
        record_path: Path of the ``wf_*.json`` record.

    Returns:
        The session transcript, which may not exist.
    """
    session = record_path.parent.parent
    return session.parent / f"{session.name}.jsonl"


def nested_records(record_path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Find workflow records that ran inside this one's window.

    A nested ``workflow()`` registers its own record and Claude Code writes no
    parent link, so containment in the same session is the only join available.
    It is reported as such rather than presented as an identity match.

    Args:
        record_path: Path of the parent ``wf_*.json`` record.
        record: The parsed parent record.

    Returns:
        Parsed child records, in start order.
    """
    start = float(record.get("startTime") or 0.0)
    end = start + float(record.get("durationMs") or 0.0)
    if not start:
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(record_path.parent.glob("wf_*.json")):
        if path == record_path:
            continue
        try:
            child = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        child_start = float(child.get("startTime") or 0.0)
        if child_start and start <= child_start <= end:
            child["_path"] = str(path)
            found.append(child)
    return sorted(found, key=lambda c: float(c.get("startTime") or 0.0))


def build_tree(
    record: dict[str, Any],
    transcripts: Path,
    *,
    orchestrator: Path | None = None,
    nested: Iterable[tuple[dict[str, Any], Path]] = (),
    capture_text: bool = False,
    text_chars: int = 4000,
    sink: list[dict[str, Any]] | None = None,
) -> tuple[Node, dict[str, Any]]:
    """Place every API call under phase -> agent and roll the totals up.

    Args:
        record: A parsed ``wf_*.json`` run record.
        transcripts: Directory holding ``agent-<agentId>.jsonl``.
        orchestrator: The session transcript of the conversation that launched
            the run, folded in as its own node when given.
        nested: ``(record, transcripts)`` pairs for workflows that ran inside
            this one, grafted as sub-agents.
        capture_text: Carry each call's prompt and response text.
        text_chars: Per-block character cap for captured text.
        sink: When given, every call is appended to it as a flat row carrying
            its phase and agent, for the per-call sidecar.

    Returns:
        ``(root, coverage)``, where *coverage* records what the totals omit:
        agents whose transcript is missing, unpriced calls, and the record's
        own summary figures for reconciliation.
    """
    progress = record.get("workflowProgress")
    progress = progress if isinstance(progress, list) else []
    agents = _agent_entries(progress)
    root = Node(name=str(record.get("runId") or record.get("workflowName") or "geak"))
    for title in _phase_order(progress):
        root.child(title)

    coverage: dict[str, Any] = {
        "run_id": record.get("runId"),
        "workflow": record.get("workflowName") or record.get("name"),
        "timestamp": record.get("timestamp"),
        "default_model": record.get("defaultModel"),
        "eval_dir": _record_eval_dir(record),
        "agents_declared": len(agents),
        "agents_with_transcript": 0,
        "agents_without_transcript": 0,
        "agents_estimated_from_summary": [],
        "transcript_dir": str(transcripts),
        "record_totals": {
            "tokens": record.get("totalTokens"),
            "tool_calls": record.get("totalToolCalls"),
            "duration_ms": record.get("durationMs"),
            "agents": record.get("agentCount"),
        },
    }

    seen_labels: dict[str, int] = {}
    for entry in agents:
        phase = str(entry.get("phaseTitle") or "").strip() or "(no phase)"
        label = str(entry.get("label") or entry.get("agentId") or "(unlabelled)")
        agent_id = str(entry.get("agentId") or "")
        # One label can name several agents -- a retry, or a fan-out the script
        # did not number. Merging them would hide the retry and collide their
        # call indices, so each gets its own node.
        ordinal = seen_labels[label] = seen_labels.get(label, 0) + 1
        node = root.child(phase).child(label if ordinal == 1 else f"{label} #{ordinal}")
        path = transcripts / f"agent-{agent_id}.jsonl" if agent_id else Path()
        calls = read_agent_calls(path, capture_text=capture_text, text_chars=text_chars) if path.is_file() else []
        if calls:
            coverage["agents_with_transcript"] += 1
            for call in calls:
                _place_call(node, call)
                _record_flat(sink, call, phase=phase, agent=node.name, agent_id=agent_id)
        else:
            coverage["agents_without_transcript"] += 1
            coverage["agents_estimated_from_summary"].append(f"{phase}/{label}")
            _place_summary(node, entry)
        # durationMs is measured by the runtime; the inferred per-call gaps are
        # not, so the agent's wall-clock comes from the record either way.
        node.own.ms_total += _agent_ms(entry, node)

    for child, child_dir in nested:
        child_root, child_cov = build_tree(
            child,
            child_dir,
            capture_text=capture_text,
            text_chars=text_chars,
            sink=sink,
        )
        child_root.name = f"nested workflow {child.get('runId')}"
        root.children[child_root.name] = child_root
        coverage.setdefault("nested_runs", []).append(
            {
                "run_id": child.get("runId"),
                "agents": child_cov["agents_declared"],
                "calls": child_cov["calls_counted"],
                "join": "time containment",
            }
        )

    if orchestrator is not None and orchestrator.is_file():
        calls = read_agent_calls(orchestrator, capture_text=capture_text, text_chars=text_chars)
        node = root.child("(orchestrator)").child("SDK conversation")
        for call in calls:
            _place_call(node, call)
            _record_flat(sink, call, phase="(orchestrator)", agent="SDK conversation", agent_id="")
        coverage["orchestrator_calls"] = len(calls)
        coverage["orchestrator_transcript"] = str(orchestrator)

    total = root.roll_up()
    coverage.update(
        {
            "calls_counted": total.calls,
            "calls_unpriced": total.calls - total.calls_priced,
            "calls_untimed": total.calls - total.calls_timed,
            "tool_calls": total.tool_calls,
            "cost_sources": dict(sorted(total.cost_sources.items())),
            "models": dict(sorted(total.models.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    )
    return root, coverage


def _agent_ms(entry: dict[str, Any], node: Node) -> float:
    """Return the agent's wall-clock, preferring the runtime's own figure.

    Args:
        entry: The ``workflow_agent`` progress entry.
        node: The agent node, whose children carry inferred per-call gaps.

    Returns:
        Milliseconds to attribute to the agent, minus what its calls already
        hold, so the roll-up equals the recorded duration rather than doubling.
    """
    recorded = float(entry.get("durationMs") or 0.0)
    if recorded <= 0:
        return node.own.ms_total
    inferred = sum(kid.roll_up().ms_total for kid in node.children.values())
    return max(0.0, recorded - inferred)


def _record_flat(
    sink: list[dict[str, Any]] | None,
    call: dict[str, Any],
    *,
    phase: str,
    agent: str,
    agent_id: str,
) -> None:
    """Append one call to the flat per-call sidecar, tagged with its position.

    Args:
        sink: The accumulator, or ``None`` to skip.
        call: A normalized API-call row.
        phase: The phase the call belongs to.
        agent: The agent node's name.
        agent_id: The agent's Claude Code id, blank for the orchestrator.
    """
    if sink is None:
        return
    row = {"phase": phase, "agent": agent, "agent_id": agent_id, **call}
    row["tool_calls"] = [t.get("tool") for t in (call.get("tool_calls") or [])]
    sink.append(row)


def _place_call(node: Node, call: dict[str, Any]) -> None:
    """Attach one API call, and its tool invocations, beneath an agent node.

    Tool calls are counted on their own child nodes so the tree bottoms out at
    tool names without the API-call row counting them a second time.

    Args:
        node: The agent node.
        call: A normalized API-call row.
    """
    tools = call.get("tool_calls") or []
    leaf = node.child(f"call {int(call.get('api_call_index') or 0):04d}")
    leaf.own.add_call({**call, "tool_calls": [], "tool_call_count": 0})
    for tool in tools:
        leaf.child(f"tool:{tool.get('tool') or '?'}").own.tool_calls += 1


def _place_summary(node: Node, entry: dict[str, Any]) -> None:
    """Record an agent whose transcript is missing, from the progress entry.

    The entry reports a token total with no input/output split and no model, so
    the spend is visible in the tree but stays out of the USD total instead of
    being priced on a guess.

    Args:
        node: The agent node.
        entry: The ``workflow_agent`` progress entry.
    """
    node.own.add_call(
        {
            "model": entry.get("model"),
            "isl": 0,
            "osl": int(float(entry.get("tokens") or 0)),
            "tool_call_count": int(float(entry.get("toolCalls") or 0)),
            "cost_source": "unavailable",
            "status": str(entry.get("state") or "ok"),
        }
    )


_COLUMNS = (
    "| Node | Calls | Share | ISL | OSL | Tok in | Tok think | Tok out | "
    "s total | s think | s out | USD | USD think | USD out | Tools |"
)
_RULE = "| " + " | ".join(["---"] * 15) + " |"


def coverage_lines(coverage: dict[str, Any]) -> list[str]:
    """Render the coverage section that precedes every number.

    Args:
        coverage: The dict returned by :func:`build_tree`.

    Returns:
        Markdown lines.
    """
    rec = coverage.get("record_totals") or {}
    missing = coverage.get("agents_estimated_from_summary") or []
    lines = [
        "## Coverage — read before quoting a number",
        "",
        "| Limit | Detail |",
        "| --- | --- |",
        "| Cost | Derived from the shipped rate card. Claude Code records no "
        "provider cost, so no figure here is provider-reported. |",
        f"| Unpriced calls | {coverage.get('calls_unpriced', 0):,} of "
        f"{coverage.get('calls_counted', 0):,} excluded from the USD total, "
        "never counted as free. |",
        "| Per-call time | Inferred from consecutive message timestamps; an "
        "agent's first call absorbs its queue wait. Agent wall-clock is the "
        "runtime's own `durationMs`. |",
        "| Thinking time | Not recorded anywhere. Thinking *tokens* are exact; "
        "thinking *seconds* are 0 by construction. |",
        "| Thinking tokens | A subset of `output_tokens` in these transcripts, "
        "so OSL is output alone, not output + thinking. |",
        f"| Agents without a transcript | {len(missing)} of "
        f"{coverage.get('agents_declared', 0)} — token totals shown, no split, "
        "unpriced. |",
        "| Agents in parallel | The summed agent durations exceed the run's "
        "wall-clock wherever the script fanned out; the record's `durationMs` "
        "is the wall-clock. |",
        f"| Nested workflows | {_nested_note(coverage)} |",
        f"| Orchestrator conversation | {_orchestrator_note(coverage)} |",
        "",
        "### Reconciliation against the run record",
        "",
        "| Figure | Record | Leaf-derived |",
        "| --- | --- | --- |",
        f"| Tokens, excluding cache | {_fmt(rec.get('tokens'))} | {coverage.get('_leaf_tokens_nocache', 0):,} |",
        f"| Tokens, including cache | - | {coverage.get('_leaf_tokens', 0):,} |",
        f"| Tool calls | {_fmt(rec.get('tool_calls'))} | {coverage.get('tool_calls', 0):,} |",
        f"| Duration | {_hours(rec.get('duration_ms'))} wall-clock | "
        f"{_hours(coverage.get('_leaf_ms'))} summed over agents |",
        f"| Agents | {_fmt(rec.get('agents'))} | {coverage.get('agents_declared', 0):,} |",
        "",
        "Tool calls, agent count and per-agent duration reconcile exactly. The "
        "record's token figure does not, and is not expected to: it sums the "
        "`tokens` field of the progress entries, a streamed snapshot that lands "
        "on either side of the transcript sums agent by agent. The transcripts "
        "are authoritative for tokens.",
        "",
    ]
    if missing:
        lines += ["Agents estimated from the summary entry:", ""]
        lines += [f"* `{name}`" for name in missing]
        lines.append("")
    return lines


def _nested_note(coverage: dict[str, Any]) -> str:
    """Describe how any nested workflow records were joined.

    Args:
        coverage: The coverage dict.

    Returns:
        A table cell.
    """
    nested = coverage.get("nested_runs") or []
    if not nested:
        return (
            "None found in this session's window. Every agent here is a direct `agent()` call of the top-level script."
        )
    ids = ", ".join(f"`{n['run_id']}` ({n['calls']:,} calls)" for n in nested)
    return (
        f"{len(nested)} grafted as sub-agents: {ids}. Claude Code writes no "
        "parent link, so the join is time containment within the same session, "
        "not an identity match."
    )


def _orchestrator_note(coverage: dict[str, Any]) -> str:
    """Describe whether the launching SDK conversation is included.

    Args:
        coverage: The coverage dict.

    Returns:
        A table cell.
    """
    if "orchestrator_calls" not in coverage:
        return (
            "Excluded. `run_e2e.py`'s own `ClaudeSDKClient` turn is not counted; "
            "pass `--include-orchestrator` to fold it in."
        )
    return f"Included — {coverage['orchestrator_calls']:,} calls."


def _fmt(value: Any) -> str:
    """Format a possibly-absent integer for a table cell.

    Args:
        value: The value to render.

    Returns:
        A grouped integer, or ``"-"``.
    """
    return f"{int(value):,}" if isinstance(value, (int, float)) else "-"


def _hours(ms: Any) -> str:
    """Format milliseconds as hours.

    Args:
        ms: Milliseconds, possibly absent.

    Returns:
        A short hours string, or ``"-"``.
    """
    return f"{float(ms) / 3_600_000.0:,.2f} h" if isinstance(ms, (int, float)) else "-"


def render_markdown(root: Node, coverage: dict[str, Any]) -> str:
    """Render the whole GEAK report as markdown.

    Args:
        root: The rolled-up tree.
        coverage: The coverage dict from :func:`build_tree`.

    Returns:
        The markdown document.
    """
    t = root.total
    coverage["_leaf_tokens"] = t.isl + t.osl
    coverage["_leaf_tokens_nocache"] = t.tokens_in + t.osl
    coverage["_leaf_ms"] = t.ms_total
    lines = [
        f"# GEAK call report — `{coverage.get('run_id')}`",
        "",
        f"* workflow: `{coverage.get('workflow')}`",
        f"* started: `{coverage.get('timestamp')}`",
        f"* eval dir: `{coverage.get('eval_dir') or '(none recorded)'}`",
        f"* default model: `{coverage.get('default_model')}`",
        "",
    ]
    lines += coverage_lines(coverage)
    lines += [
        "## Run totals",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| API calls | {t.calls:,} |",
        f"| Tool calls | {t.tool_calls:,} |",
        f"| ISL | {t.isl:,} |",
        f"| OSL | {t.osl:,} |",
        f"| Tokens — input | {t.tokens_in:,} |",
        f"| Tokens — thinking | {t.tokens_thinking:,} |",
        f"| Tokens — output (visible) | {t.tokens_out:,} |",
        f"| Tokens — cache write / read | {t.tokens_cache_write:,} / {t.tokens_cache_read:,} |",
        f"| Wall-clock | {_hours(t.ms_total)} |",
        f"| USD (derived) | ${t.usd_total:,.4f} |",
        f"| Calls priced | {t.calls_priced:,} / {t.calls:,} |",
        "",
        "## Phase -> agent -> API call -> tool call",
        "",
        _COLUMNS,
        _RULE,
    ]
    _render_rows(root, 0, None, lines)
    lines += ["", "## Models", "", "| Model | Calls |", "| --- | --- |"]
    for model, count in (coverage.get("models") or {}).items():
        lines.append(f"| `{model}` | {count:,} |")
    lines.append("")
    return "\n".join(lines)


def join_hyperloom(root: Node, session_dir: Path, coverage: dict[str, Any]) -> Node:
    """Nest the GEAK tree inside the Hyperloom session that drove it.

    The session's own ledgers supply the outer phases. Rows the GEAK harvester
    already recovered into the session's ``ext`` shard are dropped first: they
    describe the same calls this tree reads from the transcripts, and keeping
    both would bill the run twice.

    Args:
        root: The rolled-up GEAK tree.
        session_dir: The Hyperloom session root.
        coverage: The GEAK coverage dict, annotated in place.

    Returns:
        The combined root.
    """
    from hyperloom.inference_optimizer.tools.dump_llm_call_report import (
        build_tree as hl_build_tree,
        load_ledgers,
    )

    turns, details = load_ledgers(session_dir)
    dropped = sum(1 for r in turns + details if str(r.get("component") or "") == "geak")
    turns = [r for r in turns if str(r.get("component") or "") != "geak"]
    details = [r for r in details if str(r.get("component") or "") != "geak"]
    outer, hl_coverage = hl_build_tree(turns, details)
    outer.name = session_dir.name
    # GEAK *is* the kernel-agent phase, so it belongs under that node whether or
    # not the session's own ledger has any rows there -- a session that never
    # harvested GEAK has the phase missing, not empty.
    host = outer.child("KERNEL_AGENT")
    root.name = f"GEAK {coverage.get('run_id')}"
    host.children[root.name] = root
    outer.roll_up()
    coverage["hyperloom_session"] = str(session_dir)
    coverage["hyperloom_geak_rows_dropped"] = dropped
    coverage["hyperloom_phases"] = sorted(outer.children)
    coverage["hyperloom_coverage"] = hl_coverage
    return outer


def _output_dir(run_id: str, explicit: Path | None) -> Path:
    """Pick where the report is written.

    Args:
        run_id: The workflow run id, used as the leaf directory.
        explicit: ``--output-dir``, when given.

    Returns:
        The directory to write into.
    """
    if explicit is not None:
        return explicit
    base = os.environ.get("USER_DATA_PATH", "").strip()
    return (Path(base) if base else Path.cwd()) / "reports" / run_id


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector; defaults to ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Render a GEAK phase/agent/call report.")
    sel = p.add_argument_group("run selection (one required)")
    sel.add_argument("--eval-dir", default=None, help="GEAK eval dir the run was pointed at")
    sel.add_argument("--run-id", default=None, help="Workflow run id, e.g. wf_ec8b57b0-1a7")
    sel.add_argument("--session-dir", type=Path, default=None, help="Hyperloom session root")
    p.add_argument(
        "--claude-home",
        type=Path,
        action="append",
        default=[],
        help="Extra Claude Code home to search (repeatable)",
    )
    p.add_argument("--join-hyperloom", type=Path, default=None, help="Nest under this session's phases")
    p.add_argument("--output-dir", "-o", type=Path, default=None, help="Where to write the report")
    p.add_argument("--max-depth", type=int, default=0, help="Trim the tree to this depth (0 = no limit)")
    p.add_argument(
        "--list",
        action="store_true",
        help="List runs and exit; with no selector, lists every record found",
    )
    p.add_argument(
        "--include-orchestrator",
        action="store_true",
        help="Also count the SDK conversation that launched the run",
    )
    p.add_argument(
        "--include-text",
        action="store_true",
        help="Write geak_calls.jsonl carrying each call's prompt and response",
    )
    p.add_argument(
        "--text-chars",
        type=int,
        default=4000,
        help="Per-block character cap for captured text (0 = uncapped)",
    )
    p.add_argument(
        "--no-nested",
        action="store_true",
        help="Do not graft workflows that ran inside this one's window",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` when no run record matches the selector.
    """
    args = _parse_args(argv)
    if not (args.eval_dir or args.run_id or args.session_dir or args.list):
        print("one of --eval-dir, --run-id or --session-dir is required", file=sys.stderr)
        return 2
    hits = resolve_runs(
        homes=claude_homes(args.claude_home),
        eval_dir=args.eval_dir,
        run_id=args.run_id,
        session_dir=args.session_dir,
        list_all=args.list and not (args.eval_dir or args.run_id or args.session_dir),
    )
    if not hits:
        print("no workflow record found for that selector", file=sys.stderr)
        return 2
    if args.list:
        for path, record in hits:
            print(f"{record.get('runId')}\t{record.get('timestamp')}\t{_record_eval_dir(record) or '-'}\t{path}")
        return 0

    path, record = hits[0]
    run_id = str(record.get("runId"))
    children = [] if args.no_nested else nested_records(path, record)
    calls: list[dict[str, Any]] = []
    root, coverage = build_tree(
        record,
        transcript_dir(path, run_id),
        orchestrator=session_transcript(path) if args.include_orchestrator else None,
        nested=[(c, transcript_dir(Path(c["_path"]), str(c.get("runId")))) for c in children],
        capture_text=args.include_text,
        text_chars=args.text_chars,
        sink=calls,
    )
    coverage["record_path"] = str(path)
    coverage["candidates"] = [
        {"run_id": r.get("runId"), "timestamp": r.get("timestamp"), "path": str(p)} for p, r in hits
    ]
    if args.max_depth:
        _trim(root, 0, args.max_depth)
        root.roll_up()
    if args.join_hyperloom is not None:
        root = join_hyperloom(root, args.join_hyperloom, coverage)

    out_dir = _output_dir(run_id, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "geak_call_report.md"
    json_path = out_dir / "geak_call_report.json"
    if args.include_text:
        calls_path = out_dir / "geak_calls.jsonl"
        with calls_path.open("w", encoding="utf-8") as fh:
            for row in calls:
                fh.write(json.dumps(row) + "\n")
        coverage["calls_sidecar"] = str(calls_path)
    md_path.write_text(render_markdown(root, coverage), encoding="utf-8")
    json_path.write_text(
        json.dumps({"coverage": coverage, "tree": root.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {md_path} and {json_path} "
        f"({root.total.calls:,} calls, {root.total.tool_calls:,} tool calls, "
        f"${root.total.usd_total:,.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
