#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Replay a finished session's orchestration prompts through the idle-tick gate.

Every recorded orchestration prompt stands for one tick of the old coordinator. The replay
asks the gate, in time order, whether each of those turns would have been held, and prices
the held ones from ``llm_calls.jsonl``. Only the pre-turn prompt of the last real turn is
known offline, so the count is a lower bound of what the live gate (which also matches the
post-turn state) holds.

    python scripts/meta_rsi/replay_idle_gate.py SESSION_DIR [--phase KERNEL_AGENT] [--heartbeat 300]
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401

from hyperloom.inference_optimizer.trace.meta_rsi_ledger import call_tokens, iter_jsonl, weighted_tokens
from hyperloom.orchestrator.loop.idle_gate import normalize_prompt, prompt_digest

_KEY = re.compile(r"^\s*([A-Za-z_][\w .-]{0,40}?)\s*[=:]")


def _ts(rec: dict) -> float:
    return datetime.fromisoformat(str(rec["ts"]).replace("Z", "+00:00")).timestamp()


def _changed_keys(a: str, b: str) -> list[str]:
    out = []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if line[:1] in "+-" and not line.startswith(("+++", "---")):
            m = _KEY.match(line[1:])
            out.append(m.group(1) if m else line[1:40])
    return out


def replay(session_dir: Path, *, phase: str | None, heartbeat: float) -> dict:
    trace = session_dir / "reports" / "trace"
    prompts = [
        r
        for r in iter_jsonl(trace / "conversations.jsonl")
        if r.get("component") == "orchestration" and (phase is None or r.get("phase") == phase)
    ]
    prompts.sort(key=_ts)
    cost = {}
    for call in iter_jsonl(trace / "llm_calls.jsonl"):
        cid = call.get("call_id")
        if cid:
            cost[str(cid)] = weighted_tokens(call_tokens(call))
    held = called = 0
    held_cost = called_cost = 0.0
    reopen = collections.Counter()
    last_call_ts = None
    seen: set[str] = set()
    last_norm = ""
    for rec in prompts:
        text = str(rec.get("prompt") or "")
        digest = prompt_digest(text)
        now = _ts(rec)
        c = cost.get(str(rec.get("call_id")), 0.0)
        if last_call_ts is not None and now - last_call_ts < heartbeat and digest in seen:
            held += 1
            held_cost += c
            continue
        if last_call_ts is not None:
            if now - last_call_ts >= heartbeat and digest in seen:
                reopen["heartbeat"] += 1
            else:
                for key in set(_changed_keys(last_norm, normalize_prompt(text))):
                    reopen[key] += 1
        called += 1
        called_cost += c
        last_call_ts = now
        seen = {digest}
        last_norm = normalize_prompt(text)
    total = held + called
    return {
        "session": session_dir.name,
        "phase": phase or "all",
        "turns": total,
        "held": held,
        "held_pct": round(100.0 * held / total, 1) if total else 0.0,
        "weighted_tokens_total": round(held_cost + called_cost),
        "weighted_tokens_held": round(held_cost),
        "reopened_by": reopen.most_common(15),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--heartbeat", type=float, default=300.0)
    args = ap.parse_args()
    for s in args.sessions:
        print(json.dumps(replay(s, phase=args.phase, heartbeat=args.heartbeat)))


if __name__ == "__main__":
    main()
