#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backfill one hyperloom session's trace JSONL into Langfuse (offline).

Sibling of the *live* emitter
(:mod:`inference_optimizer.orchestrator.trace.langfuse_emitter`): the live
path mirrors calls into Langfuse while a run is in flight, this CLI replays
one finished session's ``reports/trace/`` after the fact. Both share the same
projection (:mod:`inference_optimizer.orchestrator.trace.langfuse_mapping`)
so a backfilled trace and a live-pushed trace are shaped identically.

Mapping (trace -> phase span -> agent span -> generation)::

    Trace                 = one session
      phase span          = PRELUDE / EXPLORE / KERNEL / SWEEP / ...
        agent span        = component (orchestration / kernel / specialist /
                            critic / geak / oob / proposal_scorer / ...)
          Generation      = one LLM call (llm_calls.jsonl; prompt/response
                            paired from conversations.jsonl when available)
      Score               = one decision (decision_trace.jsonl), attached to
                            the agent span that produced it (trace-level
                            fallback):
                              - gain_pct (NUMERIC)  when present
                              - outcome  (CATEGORICAL: KEEP/REVERT/no_promote)

Source files (under the session dir)
-------------------------------------
* ``reports/trace/llm_calls.jsonl``      -- every LLM call (model + token
                            usage + phase).
* ``reports/trace/conversations.jsonl``  -- prompt+response text for the
                            subset that recorded it; paired by (component,
                            tick, role, UTC-second of ts).
* ``reports/trace/decision_trace.jsonl`` -- per-action
                            KEEP/REVERT/no_promote + gain_pct.
* ``manifest.json``                      -- trace-level metadata +
                            claw_session_id.

Usage
-----
::

    # Dry run: parse + print the plan, no SDK / no network needed.
    python -m inference_optimizer.scripts.backfill_langfuse \\
        --session-dir <SD> --dry-run

    # Real backfill (needs the langfuse SDK + env keys).
    export LANGFUSE_HOST=https://langfuse.<your-domain>
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    python -m inference_optimizer.scripts.backfill_langfuse --session-dir <SD>

Notes
-----
* Correlation: ``trace_id`` and the Langfuse ``session_id`` grouping are
  derived from ``claw_session_id`` (fallback the internal session id), so a
  re-run, the live emitter, and this backfill of one PrimusClaw session all
  update the same trace rather than duplicating it.
* Historical backfill preserves original timestamps via explicit
  ``start_time`` / ``end_time`` on every observation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from inference_optimizer.orchestrator.trace import langfuse_mapping as lfmap
from inference_optimizer.orchestrator.trace.langfuse_emitter import (
    _end_obs,
    _set_trace_attrs,
    _start_obs,
)
from inference_optimizer.session_paths import recipe_snapshot_audit_jsonl

log = logging.getLogger("backfill_langfuse")

TRACE_SUBDIR = "reports/trace"
LLM_CALLS = "llm_calls.jsonl"
CONVERSATIONS = "conversations.jsonl"
DECISION_TRACE = "decision_trace.jsonl"
MANIFEST = "manifest.json"

UNPHASED = lfmap.UNPHASED


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dict records.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        The dict records; a missing file yields ``[]`` and malformed lines are
        skipped.
    """
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed object, or ``{}`` when the file is missing, unreadable, or
        not a JSON object.
    """
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Plan building (pure -- no SDK, dry-run friendly)
# ---------------------------------------------------------------------------
def build_plan(session_dir: Path) -> dict[str, Any]:
    """Parse the trace files into a Langfuse-shaped plan dict (pure)."""
    tdir = session_dir / TRACE_SUBDIR
    llm = _load_jsonl(tdir / LLM_CALLS)
    conv = _load_jsonl(tdir / CONVERSATIONS)
    decisions = _load_jsonl(tdir / DECISION_TRACE)
    recipe_audit = _load_jsonl(recipe_snapshot_audit_jsonl(session_dir))
    manifest = _load_json(session_dir / MANIFEST)

    conv_by_key: dict[tuple, dict[str, Any]] = {}
    for c in conv:
        conv_by_key[lfmap.pair_key(c)] = c

    internal_id = str(
        manifest.get("session_id")
        or (llm[0].get("session_id") if llm else "")
        or session_dir.name
    )
    # Correlate on claw_session_id (fallback internal id) so backfill and the
    # live emitter land on one Langfuse trace.
    seed = lfmap.correlation_seed(manifest, internal_id)
    session_label = lfmap.langfuse_session_id(manifest, internal_id)

    # phase -> agent -> [generation parts]; the span hierarchy mirrors the
    # live emitter (trace -> phase span -> agent span -> generation).
    phases: "OrderedDict[str, OrderedDict[str, list[dict]]]" = OrderedDict()
    paired = 0
    for row in llm:
        phase = lfmap.phase_of(row)
        agent = lfmap.agent_of(row)
        agents = phases.setdefault(phase, OrderedDict())
        gens = agents.setdefault(agent, [])
        conv_row = conv_by_key.get(lfmap.pair_key(row))
        if conv_row is not None:
            paired += 1
        gens.append({
            "ts": row.get("ts"),
            "row": row,
            "input": (conv_row or {}).get("prompt"),
            "output": (conv_row or {}).get("response"),
            "has_text": conv_row is not None,
        })

    return {
        "trace_id_seed": seed,
        "session_id": session_label,
        "internal_session_id": internal_id,
        "claw_session_id": manifest.get("claw_session_id"),
        "name": manifest.get("model_name") or session_label,
        "metadata": lfmap.trace_metadata(manifest),
        "created_at": manifest.get("created_at_utc"),
        "phases": phases,
        "decisions": decisions,
        "recipe_audit": recipe_audit,
        "stats": {
            "llm_calls": len(llm),
            "conversations": len(conv),
            "decisions": len(decisions),
            "generations_with_text": paired,
            "phase_count": len(phases),
            "recipe_audit": len(recipe_audit),
        },
    }


def _time_bounds(gens: list[dict]) -> tuple[datetime | None, datetime | None]:
    """Return the earliest and latest timestamps across generations.

    Args:
        gens: Generation rows each carrying a ``ts`` field.

    Returns:
        A ``(min, max)`` datetime tuple, or ``(None, None)`` when no row has a
        parseable timestamp.
    """
    times = [lfmap.parse_ts(g["ts"]) for g in gens]
    times = [t for t in times if t is not None]
    if not times:
        return (None, None)
    return (min(times), max(times))


def _phase_time_bounds(
    agents: "OrderedDict[str, list[dict]]",
) -> tuple[datetime | None, datetime | None]:
    """Return the time bounds across all agents in a phase.

    Args:
        agents: Mapping of agent name to its generation rows.

    Returns:
        A ``(min, max)`` datetime tuple over every generation in the phase.
    """
    flat = [g for gens in agents.values() for g in gens]
    return _time_bounds(flat)


# ---------------------------------------------------------------------------
# Dry-run printer
# ---------------------------------------------------------------------------
def print_plan(plan: dict[str, Any]) -> None:
    """Print a human-readable dry-run summary of a backfill plan.

    Args:
        plan: The plan dict produced by the plan builder.
    """
    s = plan["stats"]
    print(f"Trace: {plan['name']}  (session_id={plan['session_id']})")
    print(f"  claw_session_id = {plan.get('claw_session_id') or '(none)'}  "
          f"internal_session_id = {plan.get('internal_session_id')}")
    print(f"  trace_id = {lfmap.derive_trace_id(plan['trace_id_seed'])}")
    print(f"  llm_calls={s['llm_calls']} conversations={s['conversations']} "
          f"decisions={s['decisions']} "
          f"generations_with_text={s['generations_with_text']} "
          f"phases={s['phase_count']}")
    print("  metadata:", json.dumps(plan["metadata"], ensure_ascii=False))
    print("  Phases / agents / generations:")
    for phase, agents in plan["phases"].items():
        lo, hi = _phase_time_bounds(agents)
        total = sum(len(g) for g in agents.values())
        print(f"    [{phase}] {total} gen(s) across {len(agents)} agent(s), "
              f"{lo} .. {hi}")
        for agent, gens in agents.items():
            models = sorted({str(g["row"].get("model")) for g in gens})
            with_text = sum(1 for g in gens if g["has_text"])
            print(f"        - {agent}: {len(gens)} gen(s), {with_text} with text, "
                  f"models={models}")
    outcomes = [
        (d.get("decision") or {}).get("outcome") for d in plan["decisions"]
    ]
    keep = outcomes.count("KEEP")
    rev = outcomes.count("REVERT")
    nop = outcomes.count("no_promote")
    gainful = sum(
        1 for d in plan["decisions"]
        if (d.get("decision") or {}).get("gain_pct") is not None
    )
    print(f"  Scores: {len(plan['decisions'])} decisions "
          f"(KEEP={keep} REVERT={rev} no_promote={nop}; gain_pct set={gainful})")
    print(f"  Recipe-snapshot reads: {len(plan.get('recipe_audit') or [])} "
          f"audit row(s)")


# ---------------------------------------------------------------------------
# Real ingest (needs the langfuse SDK)
# ---------------------------------------------------------------------------
def ingest(plan: dict[str, Any]) -> int:
    """Emit a backfill plan to Langfuse as a full trace tree.

    Recreates the trace -> phase -> agent -> generation hierarchy from a
    completed session and attaches decision scores to the owning agent spans.

    Args:
        plan: The backfill plan produced by the plan builder.

    Returns:
        Process exit code: ``0`` on success, ``3`` when the Langfuse SDK is not
        importable.
    """
    try:
        from langfuse import get_client  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(
            "ERROR: langfuse SDK not importable. Install it first:\n"
            "  pip install 'hyperloom-inference_optimizer[trace]'\n"
            f"  ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 3

    client = get_client()
    trace_id = lfmap.derive_trace_id(plan["trace_id_seed"])
    trace_start = lfmap.parse_ts(plan["created_at"])

    root = _start_obs(
        client,
        name=plan["name"],
        as_type="span",
        start_time=trace_start,
        trace_context={"trace_id": trace_id},
        metadata=plan["metadata"],
    )
    # Version-tolerant (v2/v3 update_trace -> v4 OTEL attrs); never raises.
    _set_trace_attrs(
        root,
        name=plan["name"],
        session_id=plan["session_id"],
        metadata=plan["metadata"],
    )

    # trace -> phase span -> agent span -> generation. Keep the agent spans so
    # decision scores can attach to the agent that produced them.
    agent_spans: dict[tuple[str, str], Any] = {}
    last_end = trace_start
    for phase, agents in plan["phases"].items():
        p_lo, p_hi = _phase_time_bounds(agents)
        phase_span = _start_obs(
            root,
            name=f"phase:{phase}", as_type="span", start_time=p_lo or trace_start,
            metadata={"phase": phase, "agent_count": len(agents)},
        )
        for agent, gens in agents.items():
            a_lo, a_hi = _time_bounds(gens)
            agent_span = _start_obs(
                phase_span,
                name=f"agent:{agent}", as_type="span", start_time=a_lo or p_lo or trace_start,
                metadata={"phase": phase, "agent": agent, "generation_count": len(gens)},
            )
            agent_spans[(phase, agent)] = agent_span
            for g in gens:
                g_start = lfmap.parse_ts(g["ts"])
                row = g["row"]
                gen = _start_obs(
                    agent_span,
                    name=lfmap.generation_name(row),
                    as_type="generation",
                    start_time=g_start,
                    model=row.get("model"),
                    input=g.get("input"),
                    output=g.get("output"),
                    metadata=lfmap.generation_metadata(
                        row, phase=phase, has_text=g["has_text"],
                    ),
                    usage_details=lfmap.usage_details(row),
                )
                _end_obs(gen, g_start)
            _end_obs(agent_span, a_hi or a_lo or p_lo or trace_start)
        _end_obs(phase_span, p_hi or p_lo or trace_start)
        if p_hi is not None:
            last_end = p_hi

    # Recipe-snapshot / gbrain remote reads -> spans under a recipe_kb agent.
    recipe_audit = plan.get("recipe_audit") or []
    if recipe_audit:
        ra_times = [lfmap.parse_ts(r.get("ts")) for r in recipe_audit]
        ra_times = [t for t in ra_times if t is not None]
        ra_lo = min(ra_times) if ra_times else last_end
        ra_span = _start_obs(
            root,
            name=f"agent:recipe_kb", as_type="span", start_time=ra_lo or trace_start,
            metadata={"phase": UNPHASED, "agent": "recipe_kb",
                      "read_count": len(recipe_audit)},
        )
        for r in recipe_audit:
            r_start = lfmap.parse_ts(r.get("ts"))
            method = str(r.get("method") or "read")
            obs = _start_obs(
                ra_span,
                name=f"kb:recipe_snapshot:{method}", as_type="span",
                start_time=r_start, input=None, output=r,
                metadata={"kind": "recipe_snapshot", "method": method,
                          "remote": r.get("remote"),
                          "resolution": r.get("resolution"),
                          "hit": bool(r.get("hit"))},
            )
            _end_obs(obs, r_start)
        _end_obs(ra_span, (max(ra_times) if ra_times else None) or trace_start)

    # Decision scores -> the owning agent span (trace-level fallback).
    # ``component`` is the resolved proposer (specialist:<domain> / grid /
    # orchestration); normalise it back to a span-attachable agent so the score
    # still lands under a real agent span. operation_kind rides in score metadata.
    def _span_agent_for(proposer: str) -> str:
        p = (proposer or "").strip()
        if p.startswith("specialist:"):
            return "specialist"
        if p == "grid":
            return "orchestration"
        return p or lfmap.UNKNOWN_AGENT

    for i, drow in enumerate(plan["decisions"]):
        for score in lfmap.decision_to_scores(drow):
            meta = score.get("metadata") or {}
            phase = str(meta.get("phase") or lfmap.UNPHASED)
            agent = _span_agent_for(str(meta.get("component") or ""))
            span = agent_spans.get((phase, agent))
            try:
                if span is not None and hasattr(span, "score"):
                    span.score(
                        name=score["name"], value=score["value"],
                        data_type=score["data_type"],
                        comment=score.get("comment") or "",
                        metadata=meta,
                    )
                else:
                    client.create_score(
                        name=score["name"], value=score["value"],
                        trace_id=trace_id, data_type=score["data_type"],
                        comment=score.get("comment") or "", metadata=meta,
                    )
            except Exception:  # noqa: BLE001
                log.exception("create_score failed for decision %d", i)

    _end_obs(root, last_end)
    client.flush()
    print(f"Backfilled trace_id={trace_id} session_id={plan['session_id']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="backfill_langfuse",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session-dir", type=Path, required=True,
                   help="Hyperloom session directory (contains reports/trace/).")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse + print the plan; no SDK / no network.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build and (optionally) emit a backfill plan.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code from the plan/ingest path.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose >= 2
               else logging.INFO if args.verbose == 1 else logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sd = args.session_dir.resolve()
    if not (sd / TRACE_SUBDIR / LLM_CALLS).exists():
        print(f"ERROR: no {TRACE_SUBDIR}/{LLM_CALLS} under {sd}", file=sys.stderr)
        return 2

    plan = build_plan(sd)
    if args.dry_run:
        print_plan(plan)
        return 0
    return ingest(plan)


if __name__ == "__main__":
    sys.exit(main())
