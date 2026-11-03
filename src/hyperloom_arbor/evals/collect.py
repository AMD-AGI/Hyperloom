"""collect.py — assemble one Arbor session's trajectory from three sources.

Nothing here observes the run; it reconstructs it afterward from what the run
left behind:

  1. ``reports/trace/events.jsonl`` — the hook stream. Authoritative for
     ordering, wall-clock timing, tool exit status, and permission mode.
  2. The Claude Code transcript JSONL(s), located through ``meta.json``.
     Authoritative for prompt and response text, token usage, and — via
     ``isSidechain`` — whether a step belonged to the orchestrator or to a
     dispatched specialist.
  3. Session artifacts: ``iter_*/bench_output.json``,
     ``correctness_output.json``, ``session.json``. Authoritative for outcomes.

The three are joined on ``tool_use_id``, which the hook payload and the
transcript both carry — verified by probing the installed CLI rather than
assumed, which is what makes this an exact join instead of a timestamp
heuristic.

Outputs, all regenerated from scratch on every run so this is idempotent:

  * ``trajectory.jsonl``       — canonical step stream, reward-annotated
  * ``reports/trace/llm_calls.jsonl``      — Hyperloom closed schema
  * ``reports/trace/conversations.jsonl``  — Hyperloom closed schema
  * ``reports/trace/decision_trace.jsonl`` — one row per gate

Source 2 alone is enough to produce a usable trajectory, so a session recorded
before this tooling existed can still be collected retroactively by pointing
``--transcript`` straight at its JSONL.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbor_trace import clip, digest, now_iso, redact, redact_obj  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperloom schema parity
# ---------------------------------------------------------------------------

# Exact field sets from hyperloom/orchestrator/trace/llm_trace.py:84-106 and
# conversation_trace.py:45-60. Both are *closed* schemas on the Hyperloom side:
# an extra or missing key is a validation error there, so emitting anything
# else would silently drop Arbor rows out of a joint comparison.
LLM_CALL_FIELDS = (
    "session_id", "ts", "component", "role", "task_id", "dyn_id", "tick",
    "phase", "turn", "model", "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens", "latency_ms",
    "reviewed_msg_ids", "status", "error_type", "error_message",
)

CONVERSATION_FIELDS = (
    "session_id", "ts", "component", "role", "task_id", "dyn_id", "tick",
    "phase", "turn", "model", "prompt", "response",
)

#: Hyperloom's closed component vocabulary. Arbor's orchestrator maps to
#: ``orchestration`` and its dispatched agents to ``specialist``, which are the
#: two labels that carry the same meaning in both harnesses.
COMPONENT_ORCHESTRATOR = "orchestration"
COMPONENT_SUBAGENT = "specialist"

#: bench.py is always invoked with ``--session-dir .../iter_N``, so the
#: iteration a step belongs to is recoverable from the command line even when
#: the agent forgets its mark.
#:
#: The lookbehind is not paranoia: without it this matched ``aiter-7x-speedup``
#: in a URL and invented an iteration 7 in a run that only had one. AITER is
#: the central ROCm library in this domain, so ``aiter`` appears in nearly every
#: session — an unbounded ``iter`` match misattributes constantly.
#:
#: The leading ``/`` is the second half of the same lesson: a subagent ran
#: ``echo "=== iter_0 / baseline ==="`` and this invented an iteration 0 holding
#: one step, which dragged the agent-dispatch rate from 5/5 to 5/6 and made the
#: report show a protocol violation that never happened. Only a path segment is
#: evidence of an iteration; prose is not.
_ITER_RE = re.compile(r"/iter[_-]?(\d+)")

#: Preferred and unambiguous: the flag bench.py is actually invoked with.
_SESSION_DIR_ITER_RE = re.compile(r"--session-dir\s+\S*?/iter[_-]?(\d+)")

#: Directory names are matched whole, so they need their own pattern: the
#: path-segment rule above requires a leading slash and would never match a
#: bare ``iter_1``. Sharing one regex between "is this text a path" and "is
#: this a directory name" silently emptied the artifact cross-check.
_ITER_DIR_RE = re.compile(r"^iter[_-]?(\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed rows, skipping unparseable lines.

    A truncated final line is normal: a session killed mid-write leaves one,
    and refusing to read the other 40,000 rows because of it would be absurd.
    """
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_meta(trace_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_transcripts(meta: dict[str, Any], explicit: list[str] | None) -> list[Path]:
    """Resolve every transcript belonging to this Arbor run.

    A ``/loop``-driven session spans restarts and each restart opens a new
    file, so this is a list, not a path — collecting only the last leg would
    silently truncate a multi-day run to its final hour.
    """
    if explicit:
        return [Path(p) for p in explicit if Path(p).exists()]
    out = []
    for entry in meta.get("claude_sessions", []):
        tp = entry.get("transcript_path")
        if tp and Path(tp).exists():
            out.append(Path(tp))
    return out


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def _blocks(message: Any) -> list[dict[str, Any]]:
    """Normalize a message's content into a list of typed blocks."""
    content = (message or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def parse_transcripts(paths: list[Path]) -> dict[str, Any]:
    """Extract steps, tool results, and token usage from Claude transcripts.

    Returns raw material keyed for the join: transcript steps in order, a
    ``tool_use_id -> result`` map, and a ``tool_use_id -> is_sidechain`` map
    that is the only available signal for orchestrator-vs-subagent attribution
    (hook payloads do not carry it).
    """
    steps: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    sidechain: dict[str, bool] = {}

    for path in paths:
        for row in read_jsonl(path):
            rtype = row.get("type")
            if rtype not in ("user", "assistant"):
                continue
            msg = row.get("message") or {}
            ts = row.get("timestamp") or ""
            is_side = bool(row.get("isSidechain"))
            usage = msg.get("usage") or {}
            model = msg.get("model")
            # One assistant message becomes several steps (a thinking block, some
            # text, N tool_use blocks) and every one of them carries this same
            # usage dict. Summing usage per step therefore multiplies the true
            # cost by the number of blocks — measured at 2.6x for output tokens
            # and 3.0x for thinking on the 3-hour run. Carry the message identity
            # so aggregates can count each message exactly once.
            message_id = msg.get("id") or row.get("uuid")

            for block in _blocks(msg):
                btype = block.get("type")

                if btype == "tool_use":
                    tuid = block.get("id")
                    if tuid:
                        sidechain[tuid] = is_side
                    steps.append({
                        "ts": ts, "kind": "tool_call", "actor_sidechain": is_side,
                        "tool_use_id": tuid, "tool_name": block.get("name"),
                        "tool_input": redact_obj(block.get("input")),
                        "model": model, "usage": usage, "message_id": message_id,
                        "uuid": row.get("uuid"), "parent_uuid": row.get("parentUuid"),
                    })

                elif btype == "tool_result":
                    tuid = block.get("tool_use_id")
                    content = block.get("content")
                    text = content if isinstance(content, str) else json.dumps(content, default=str)
                    clipped, dropped = clip(redact(text or ""))
                    results[tuid] = {
                        "ts": ts, "output": clipped, "clipped_bytes": dropped,
                        "is_error": bool(block.get("is_error")),
                        "structured": redact_obj(row.get("toolUseResult")),
                    }

                elif btype == "thinking":
                    # Text is empty on this deployment — the gateway strips
                    # reasoning before it reaches the client, confirmed for both
                    # the transcript and stream-json capture routes. The block is
                    # still emitted so the gap is visible in the rendered log
                    # rather than reading as "the agent did not reason".
                    text = block.get("thinking") or ""
                    steps.append({
                        "ts": ts, "kind": "thinking", "actor_sidechain": is_side,
                        "text": redact(text), "text_available": bool(text.strip()),
                        "signature_len": len(block.get("signature") or ""),
                        "model": model, "usage": usage, "message_id": message_id,
                        "uuid": row.get("uuid"),
                    })

                elif btype == "text":
                    text = block.get("text") or ""
                    if not text.strip():
                        continue
                    clipped, _ = clip(redact(text))
                    steps.append({
                        "ts": ts,
                        "kind": "user_prompt" if rtype == "user" else "assistant_text",
                        "actor_sidechain": is_side, "text": clipped,
                        "model": model, "usage": usage if rtype == "assistant" else {},
                        "message_id": message_id,
                        "uuid": row.get("uuid"),
                    })

    steps.sort(key=lambda s: (s.get("ts") or "", s.get("uuid") or ""))
    return {"steps": steps, "results": results, "sidechain": sidechain}


# ---------------------------------------------------------------------------
# Iteration / branch attribution
# ---------------------------------------------------------------------------


def build_marks_index(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect explicit marks into a timeline the steps can be bucketed against."""
    marks = [e for e in events if e.get("event") == "mark"]
    marks.sort(key=lambda m: m.get("ts") or "")
    return {
        "all": marks,
        "branches": [m for m in marks if m.get("kind") == "branch"],
        "gates": [m for m in marks if m.get("kind") == "gate"],
        "escalations": [m for m in marks if m.get("kind") == "escalation"],
    }


def infer_iteration(step: dict[str, Any]) -> int | None:
    """Recover the iteration from a tool call's own arguments.

    The backstop for a forgotten ``mark branch``: hyperloom-optimize.md always
    routes bench and correctness through ``--session-dir .../iter_N``, and
    writes per-iteration artifacts under the same path, so the number is right
    there in the command line.
    """
    if step.get("kind") != "tool_call":
        return None
    blob = json.dumps(step.get("tool_input") or {}, default=str)

    # An explicit --session-dir is unambiguous; trust it before anything else.
    hits = _SESSION_DIR_ITER_RE.findall(blob)
    if not hits:
        # Otherwise only trust a path-like `.../iter_N/`. A bare `iter7`
        # anywhere in prose or a URL is not evidence of an iteration.
        if step.get("tool_name") not in ("Bash", "Read", "Write", "Edit"):
            return None
        hits = _ITER_RE.findall(blob)
    if hits:
        try:
            return int(hits[-1])
        except ValueError:
            return None
    return None


def attribute(steps: list[dict[str, Any]], marks: dict[str, Any]) -> None:
    """Stamp iteration / branch / tier onto every step, in place.

    Marks win where present; inference from tool arguments fills the gaps;
    otherwise a step inherits from the step before it. The last rule is what
    keeps a whole iteration's reasoning attached to that iteration even though
    only its bench call names the number.
    """
    branch_marks = sorted(marks["branches"], key=lambda m: m.get("ts") or "")
    cur = {"iteration": None, "branch_id": None, "tier": None}

    for step in steps:
        ts = step.get("ts") or ""

        for bm in branch_marks:
            if (bm.get("ts") or "") <= ts:
                cur = {
                    "iteration": bm.get("iteration"),
                    "branch_id": bm.get("branch_id"),
                    "tier": bm.get("tier"),
                }
            else:
                break

        inferred = infer_iteration(step)
        if inferred is not None:
            cur = dict(cur, iteration=inferred)

        step["iteration"] = cur["iteration"]
        step["branch_id"] = cur["branch_id"]
        step["tier"] = cur["tier"]


def attach_rewards(steps: list[dict[str, Any]], marks: dict[str, Any],
                   artifacts: dict[int, dict[str, Any]]) -> None:
    """Credit each iteration's measured outcome to its gate step.

    Episode boundary for the RL use: an iteration's steps are the episode, the
    gate is the terminal step, and ``reward_delta_pct`` is the throughput delta
    that gate actually measured. Marks are preferred; bench artifacts are the
    fallback so an unmarked iteration still carries a reward.
    """
    by_iter: dict[int, dict[str, Any]] = {}
    for gate in marks["gates"]:
        it = gate.get("iteration")
        if it is not None:
            by_iter[it] = {
                "decision": gate.get("decision"),
                "delta_pct": gate.get("delta_pct"),
                "tput": gate.get("tput"),
                "correctness": gate.get("correctness"),
                "source": "mark",
            }
    for it, art in artifacts.items():
        if it not in by_iter and art.get("output_throughput") is not None:
            by_iter[it] = {
                "decision": None,
                "delta_pct": None,
                "tput": art.get("output_throughput"),
                "correctness": "pass" if art.get("correctness_passed") else "fail",
                "source": "artifact",
            }

    for step in steps:
        it = step.get("iteration")
        outcome = by_iter.get(it) if it is not None else None
        step["reward_delta_pct"] = None
        step["decision"] = None
        if outcome and step.get("kind") in ("gate", "mark"):
            step["reward_delta_pct"] = outcome.get("delta_pct")
            step["decision"] = outcome.get("decision")
        step["iteration_outcome"] = outcome


# ---------------------------------------------------------------------------
# Session artifacts
# ---------------------------------------------------------------------------


def load_artifacts(session_dir: Path) -> dict[int, dict[str, Any]]:
    """Read per-iteration bench and correctness results off disk.

    Independent of anything the agent said it did — this is the ground truth a
    self-report can be checked against.
    """
    out: dict[int, dict[str, Any]] = {}
    if not session_dir.exists():
        return out
    for child in sorted(session_dir.glob("iter*")):
        if not child.is_dir():
            continue
        m = _ITER_DIR_RE.match(child.name)
        if not m:
            continue
        it = int(m.group(1))
        entry: dict[str, Any] = {}
        # Glob, not a fixed name. A rigorous run writes more than one bench:
        # observed in the wild were bench_output_cold1/cold2/confirm.json (two
        # cold runs plus a discarded hot one) and bench_output_normonly.json for
        # an ablation. Matching only "bench_output.json" found nothing for three
        # of five iterations and silently left the agent's self-reported number
        # unchecked — which is the one thing this cross-check exists to prevent.
        # ``bench_result*`` as well as ``bench_output*``: the filename is the
        # agent's choice, not the harness's, and a Sonnet-driven run named every
        # one of them bench_result.json. Scoring that run as "gate unverified"
        # would have been the collector's naming assumption masquerading as a
        # finding about the agent.
        measurements: list[dict[str, Any]] = []
        benches = sorted(child.glob("bench_output*.json")) + \
            sorted(child.glob("bench_result*.json"))
        for bench in benches:
            try:
                data = json.loads(bench.read_text(encoding="utf-8"))
            except Exception:
                continue
            tput = data.get("output_throughput")
            if tput is None:
                continue
            measurements.append({
                "file": bench.name,
                "output_throughput": tput,
                "tpot_ms": data.get("mean_tpot_ms") or data.get("median_tpot_ms"),
                "status": data.get("status"),
            })
        if measurements:
            entry["measurements"] = measurements
            canonical = next(
                (m for m in measurements if m["file"] == "bench_output.json"), None
            )
            values = [m["output_throughput"] for m in measurements]
            entry["output_throughput"] = (
                canonical["output_throughput"] if canonical
                else round(sum(values) / len(values), 2)
            )
            entry["bench_status"] = (canonical or measurements[0]).get("status")
            entry["bench_run_count"] = len(measurements)
            if len(values) > 1:
                # The spread across repeats of the same configuration is the
                # noise floor. A gain smaller than it is not a result.
                entry["bench_spread_pct"] = round(
                    100.0 * (max(values) - min(values)) / (sum(values) / len(values)), 2
                )
        corr = child / "correctness_output.json"
        if corr.exists():
            try:
                data = json.loads(corr.read_text(encoding="utf-8"))
                entry["correctness_passed"] = data.get("passed")
                entry["correctness_score"] = data.get("score")
            except Exception:
                pass
        if entry:
            out[it] = entry
    return out


def load_session_json(session_dir: Path) -> dict[str, Any]:
    for candidate in (session_dir / "session.json",):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _fallback_events(meta: dict[str, Any], seen_ts: set) -> list[dict[str, Any]]:
    """Recover hook rows stranded in ``~/.arbor/trajectories/<session_id>/``.

    Belt to ``_migrate_fallback_events``'s braces: that runs at session open and
    fixes new runs, this rescues runs already on disk — and any session whose
    pointer was never written at all, where the fallback holds *everything*.
    Rows are matched by the session ids meta.json already knows, so an unrelated
    session's trajectory is never pulled in.
    """
    out: list[dict[str, Any]] = []
    root = Path.home() / ".arbor" / "trajectories"
    if not root.is_dir():
        return out
    for entry in meta.get("claude_sessions", []):
        sid = entry.get("session_id")
        if not sid:
            continue
        for name in ("events.jsonl", "events.jsonl.migrated"):
            for row in read_jsonl(root / sid / "reports" / "trace" / name):
                # Migration may already have copied these in; dedupe on the
                # microsecond timestamp rather than importing them twice.
                if row.get("ts") in seen_ts:
                    continue
                out.append(row)
    return out


def _agent_windows(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """``(start_ts, end_ts, agent_tool_use_id)`` for every Agent dispatch.

    A subagent's tool calls fall between its Agent call's PreToolUse and
    PostToolUse, which is what lets a hook-only step be attributed to the
    specialist that issued it rather than to an anonymous "subagent".
    """
    starts: dict[str, str] = {}
    windows: list[tuple[str, str, str]] = []
    for ev in sorted(events, key=lambda e: e.get("ts") or ""):
        if ev.get("tool_name") != "Agent" or not ev.get("tool_use_id"):
            continue
        tuid = ev["tool_use_id"]
        if ev.get("event") == "tool_call":
            starts[tuid] = ev.get("ts") or ""
        elif ev.get("event") == "tool_result" and tuid in starts:
            windows.append((starts.pop(tuid), ev.get("ts") or "", tuid))
    # A dispatch still open when the job was killed stays open to the end.
    for tuid, start in starts.items():
        windows.append((start, "9999", tuid))
    return windows


def _enclosing_agent(ts: str | None, windows: list[tuple[str, str, str]]) -> str | None:
    if not ts:
        return None
    for start, end, tuid in windows:
        if start <= ts <= end:
            return tuid
    return None


def build_trajectory(events: list[dict[str, Any]], parsed: dict[str, Any],
                     artifacts: dict[int, dict[str, Any]],
                     meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge every source into the canonical step stream."""
    results = parsed["results"]
    sidechain = parsed["sidechain"]

    # Hook rows keyed by tool_use_id — the exact join. PostToolUse supplies
    # duration_ms and the structured response; PreToolUse supplies permission
    # mode and effort. Neither is derivable from the transcript.
    hooks_by_tuid: dict[str, dict[str, Any]] = {}
    for ev in events:
        tuid = ev.get("tool_use_id")
        if not tuid:
            continue
        slot = hooks_by_tuid.setdefault(tuid, {})
        if ev.get("event") == "tool_call":
            slot["pre"] = ev
        elif ev.get("event") == "tool_result":
            slot["post"] = ev

    steps = list(parsed["steps"])

    # Tool calls that fired hooks but have no transcript entry are the
    # subagents'. Claude Code does not write dispatched-agent turns into the
    # parent transcript — measured on a real run, 133 of 177 tool calls existed
    # only in the hook stream, among them every WebSearch and WebFetch the
    # research-protocol check is supposed to score. Reconstructing them from
    # hooks alone is the only way the specialist's work reaches the trajectory.
    transcript_tuids = {s.get("tool_use_id") for s in steps if s.get("tool_use_id")}
    agent_windows = _agent_windows(events)
    for tuid, slot in hooks_by_tuid.items():
        if tuid in transcript_tuids:
            continue
        pre = slot.get("pre") or {}
        post = slot.get("post") or {}
        src = pre or post
        if not src:
            continue
        response = post.get("tool_response")
        output = response if isinstance(response, str) else json.dumps(response, default=str)
        steps.append({
            "ts": src.get("ts"),
            "kind": "tool_call",
            # No transcript line means it was not the orchestrator's own turn.
            "actor_sidechain": True,
            "parent_agent_id": _enclosing_agent(src.get("ts"), agent_windows),
            "tool_use_id": tuid,
            "tool_name": src.get("tool_name"),
            "tool_input": src.get("tool_input"),
            "hook_only": True,
            "hook_output": output or "",
        })

    # Marks are steps too: they are the only record of intent.
    for ev in events:
        if ev.get("event") != "mark":
            continue
        steps.append({
            "ts": ev.get("ts"), "kind": "gate" if ev.get("kind") == "gate" else "mark",
            "actor_sidechain": False, "mark_kind": ev.get("kind"),
            "mark": {k: v for k, v in ev.items() if k not in ("ts", "event")},
        })

    steps.sort(key=lambda s: (s.get("ts") or "", s.get("uuid") or ""))

    marks_index = build_marks_index(events)
    attribute(steps, marks_index)
    attach_rewards(steps, marks_index, artifacts)

    session_label = Path(meta.get("arbor_session_dir", "unknown")).name
    out: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        tuid = step.get("tool_use_id")
        hook = hooks_by_tuid.get(tuid, {}) if tuid else {}
        post = hook.get("post") or {}
        pre = hook.get("pre") or {}
        result = results.get(tuid, {}) if tuid else {}

        is_side = step.get("actor_sidechain")
        if tuid and tuid in sidechain:
            is_side = sidechain[tuid]

        # Attribute transcript-derived sidechain steps too, not just hook-only
        # ones. Both kinds need to say *which* specialist they came from — a run
        # with 16 dispatches and 1,101 subagent steps is unreadable otherwise.
        parent_agent = step.get("parent_agent_id")
        if is_side and not parent_agent:
            parent_agent = _enclosing_agent(step.get("ts"), agent_windows)

        row = {
            "step": i,
            "ts": step.get("ts"),
            "arbor_session": session_label,
            "iteration": step.get("iteration"),
            "branch_id": step.get("branch_id"),
            "tier": step.get("tier"),
            "actor": "subagent" if is_side else "orchestrator",
            "kind": step.get("kind"),
            "model": step.get("model"),
            "usage": step.get("usage") or {},
            # Several steps share one message's usage; aggregates must dedupe on
            # this or they multiply the true token cost.
            "message_id": step.get("message_id"),
            "reward_delta_pct": step.get("reward_delta_pct"),
            "decision": step.get("decision"),
            "iteration_outcome": step.get("iteration_outcome"),
        }

        if step.get("kind") in ("user_prompt", "assistant_text", "thinking"):
            row["text"] = step.get("text", "")
            if step.get("kind") == "thinking":
                row["text_available"] = step.get("text_available", False)
                row["signature_len"] = step.get("signature_len", 0)
        if step.get("kind") == "tool_call":
            # Transcript-derived steps get their output from the tool_result
            # block; hook-only steps (the subagents') have no transcript entry
            # at all, so the PostToolUse payload is the only copy.
            hook_only = bool(step.get("hook_only"))
            output = step.get("hook_output", "") if hook_only else result.get("output", "")
            row.update({
                "tool_use_id": tuid,
                "tool_name": step.get("tool_name"),
                "tool_input": step.get("tool_input"),
                "tool_input_digest": digest(step.get("tool_input")),
                "tool_output": output,
                "tool_output_clipped_bytes": result.get("clipped_bytes", 0),
                "tool_error": result.get("is_error", False),
                "duration_ms": post.get("duration_ms"),
                "permission_mode": pre.get("permission_mode"),
                "effort": pre.get("effort"),
                "hook_observed": bool(hook) or hook_only,
                "source": "hook_only" if hook_only else "transcript+hook",
                "parent_agent_id": parent_agent,
            })
        if step.get("kind") in ("mark", "gate"):
            row["mark"] = step.get("mark")
            row["mark_kind"] = step.get("mark_kind")

        out.append(row)

    return out


def project_llm_calls(trajectory: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """Project assistant turns onto Hyperloom's llm_calls.jsonl closed schema."""
    rows = []
    turn = 0
    # One row per LLM call, which is one row per *message* — not per block. A
    # message emitting text plus three tool_use blocks is one call that Arbor
    # was billed for once; emitting four rows would both triple the token totals
    # and misrepresent the call count against a Hyperloom session in the same
    # table, which is the entire point of matching the schema.
    seen_messages: set[Any] = set()
    for step in trajectory:
        usage = step.get("usage") or {}
        if not usage or step.get("kind") not in ("assistant_text", "tool_call", "thinking"):
            continue
        mid = step.get("message_id")
        if mid is not None:
            if mid in seen_messages:
                continue
            seen_messages.add(mid)
        turn += 1
        row = {
            "session_id": session_id,
            "ts": step.get("ts"),
            "component": COMPONENT_SUBAGENT if step.get("actor") == "subagent" else COMPONENT_ORCHESTRATOR,
            "role": step.get("tier"),
            "task_id": step.get("branch_id"),
            "dyn_id": None,
            "tick": step.get("iteration"),
            "phase": step.get("tier"),
            "turn": turn,
            "model": step.get("model"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "latency_ms": step.get("duration_ms"),
            "reviewed_msg_ids": None,
            "status": "error" if step.get("tool_error") else "ok",
            "error_type": None,
            "error_message": None,
        }
        rows.append({k: row.get(k) for k in LLM_CALL_FIELDS})
    return rows


def project_conversations(trajectory: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """Project prompt/response text onto Hyperloom's conversations.jsonl schema."""
    rows = []
    turn = 0
    pending_prompt = ""
    for step in trajectory:
        kind = step.get("kind")
        if kind == "user_prompt":
            pending_prompt = step.get("text", "")
            continue
        if kind not in ("assistant_text", "tool_call"):
            continue
        turn += 1
        response = step.get("text") or json.dumps(
            {"tool": step.get("tool_name"), "input": step.get("tool_input")}, default=str
        )
        row = {
            "session_id": session_id,
            "ts": step.get("ts"),
            "component": COMPONENT_SUBAGENT if step.get("actor") == "subagent" else COMPONENT_ORCHESTRATOR,
            "role": step.get("tier"),
            "task_id": step.get("branch_id"),
            "dyn_id": None,
            "tick": step.get("iteration"),
            "phase": step.get("tier"),
            "turn": turn,
            "model": step.get("model"),
            "prompt": pending_prompt,
            "response": response,
        }
        pending_prompt = ""
        rows.append({k: row.get(k) for k in CONVERSATION_FIELDS})
    return rows


def project_decisions(trajectory: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """One row per gate: what was tried, how it resolved, what it moved."""
    rows = []
    for step in trajectory:
        if step.get("kind") != "gate":
            continue
        mark = step.get("mark") or {}
        rows.append({
            "session_id": session_id,
            "ts": step.get("ts"),
            "tick": step.get("iteration"),
            "phase": step.get("tier"),
            "task_id": step.get("branch_id"),
            "action": mark.get("change", ""),
            "decision": mark.get("decision"),
            "delta_pct": mark.get("delta_pct"),
            "gain_pct": mark.get("delta_pct"),
            "tput": mark.get("tput"),
            "correctness": mark.get("correctness"),
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect(session_dir: Path, transcripts: list[str] | None = None) -> dict[str, Any]:
    """Assemble and write every output for one session. Returns a summary."""
    trace_dir = session_dir / "reports" / "trace"
    meta = load_meta(trace_dir)
    if not meta.get("arbor_session_dir"):
        meta["arbor_session_dir"] = str(session_dir)

    events = list(read_jsonl(trace_dir / "events.jsonl"))
    events += _fallback_events(meta, {e.get("ts") for e in events})
    paths = discover_transcripts(meta, transcripts)
    parsed = parse_transcripts(paths)
    artifacts = load_artifacts(session_dir)

    session_id = (meta.get("claude_sessions") or [{}])[0].get("session_id") or session_dir.name

    trajectory = build_trajectory(events, parsed, artifacts, meta)

    write_jsonl(session_dir / "trajectory.jsonl", trajectory)
    write_jsonl(trace_dir / "llm_calls.jsonl", project_llm_calls(trajectory, session_id))
    write_jsonl(trace_dir / "conversations.jsonl", project_conversations(trajectory, session_id))
    write_jsonl(trace_dir / "decision_trace.jsonl", project_decisions(trajectory, session_id))

    summary = {
        "collected_at": now_iso(),
        "session_dir": str(session_dir),
        "steps": len(trajectory),
        "events": len(events),
        "transcripts": [str(p) for p in paths],
        "iterations": sorted({s["iteration"] for s in trajectory if s.get("iteration") is not None}),
        "tool_calls": sum(1 for s in trajectory if s.get("kind") == "tool_call"),
        "subagent_steps": sum(1 for s in trajectory if s.get("actor") == "subagent"),
        "gates": sum(1 for s in trajectory if s.get("kind") == "gate"),
        "artifacts": artifacts,
        "session_json": load_session_json(session_dir),
    }
    (trace_dir / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True,
                        help="Arbor session directory (the one holding reports/trace/)")
    parser.add_argument("--transcript", action="append", default=None,
                        help="Explicit transcript JSONL; repeatable. Enables retroactive "
                             "collection of a session recorded before hooks existed.")
    args = parser.parse_args(argv)

    summary = collect(Path(args.session_dir).resolve(), args.transcript)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("artifacts", "session_json")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
