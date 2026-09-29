#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Replay recorded orchestration and critic calls to another model; score agreement with what was decided.

Orchestration samples come from Claude Code transcripts (the recorded ``emit_intent`` inputs are the
decision) and, for long KERNEL waits, from a session's ``conversations.jsonl`` with the decision read
off the bus. Each is replayed through the real ClaudeBackend on ``--route`` with the session's phase
system prompt; the context-pull tools are not attached, since the live state they read is gone.
Critic samples are ``conversations.jsonl`` critic rows, replayed on ``--openai-route`` and compared
verdict by verdict.

    python scripts/meta_rsi/shadow_eval.py --out DIR \
        --route '{"model": "glm-5.3-flash", "base_url": "http://127.0.0.1:30300"}' \
        --openai-route '{"model": "glm-5.3-flash", "base_url": "http://127.0.0.1:30300/v1", "protocol": "openai"}' \
        --kernel-session /wekafs/zgong/.../20260924T162544Z-d7064552
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import _paths  # noqa: F401

from hyperloom.common.llm_config import achat_completion, get_async_openai_client
from hyperloom.common.role_models import RoleModels, parse_role_models
from hyperloom.inference_optimizer.trace.meta_rsi_ledger import iter_jsonl
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.claude import ClaudeBackend

_SUFFIX = "\n\n==== OUTPUT FORMAT (REQUIRED) ===="
_PHASES = ("PRELUDE", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE")
# send_message and alerts are commentary; the decision is what the turn asks the system to do.
_COMMENTARY = {"send_message", "alert"}


def _ts(value: str) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def action_key(intent_type: str, payload: dict[str, Any]) -> tuple[str, str] | None:
    """One comparable (type, target) per intent; None for commentary."""
    t = str(intent_type)
    if t in _COMMENTARY:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {}
    payload = payload if isinstance(payload, dict) else {}
    p = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if t in ("delegate", "propose_action"):
        return t, str(payload.get("action_name") or "")
    if t == "request":
        return t, f"{payload.get('kind') or ''}:{p.get('kernel_id') or payload.get('kernel_id') or ''}"
    if t == "escalate_strategy_change":
        return t, str(payload.get("next_action_hint") or "")
    if t == "prune_branch":
        return t, str(payload.get("family") or "")
    if t == "update_state":
        return t, ",".join(sorted((payload.get("changes") or {}).keys()))
    return t, ""


def score(expected: set, got: set) -> dict[str, Any]:
    union = expected | got
    return {
        "exact": expected == got,
        "jaccard": (len(expected & got) / len(union)) if union else 1.0,
        "types_match": {k[0] for k in expected} == {k[0] for k in got},
    }


# --- orchestration samples ---------------------------------------------------------
def transcript_samples() -> list[dict[str, Any]]:
    out = []
    for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):
        first, intents = None, []
        try:
            for line in open(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                msg = row.get("message") or {}
                if row.get("type") == "user" and first is None:
                    c = msg.get("content")
                    first = (
                        c
                        if isinstance(c, str)
                        else "".join(
                            x.get("text", "") for x in c or [] if isinstance(x, dict) and x.get("type") == "text"
                        )
                    )
                if row.get("type") == "assistant" and msg.get("model") == "claude-opus-5":
                    for c in msg.get("content") or []:
                        if c.get("type") == "tool_use" and str(c.get("name", "")).endswith("emit_intent"):
                            inp = c.get("input") or {}
                            intents.append((str(inp.get("intent_type")), inp.get("payload") or {}))
        except (OSError, ValueError):
            continue
        if not first or not first.startswith("SESSION_DIR=") or not intents:
            continue
        session = first.split("\n", 1)[0].split("=", 1)[1]
        m = re.search(r"^phase\s*:\s*(\S+)", first, re.M)
        phase = m.group(1) if m else ""
        system = Path(session) / "agents" / "orchestration" / f"system_prompt.{phase}.snapshot.md"
        if phase not in _PHASES or not system.exists():
            continue
        expected = {k for k in (action_key(t, p) for t, p in intents) if k}
        out.append(
            {
                "source": "transcript",
                "session": session,
                "phase": phase,
                "system_path": str(system),
                "prompt": first.split(_SUFFIX, 1)[0],
                "expected": sorted(expected),
                "ref": f,
            }
        )
    return out


def kernel_bus_samples(session_dir: Path, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """KERNEL prompts from conversations.jsonl; the decision is what orchestration put on the bus next."""
    rows = [
        r
        for r in iter_jsonl(session_dir / "reports/trace/conversations.jsonl")
        if r.get("component") == "orchestration" and r.get("phase") == "KERNEL_AGENT"
    ]
    rows.sort(key=lambda r: _ts(r["ts"]))
    db = sqlite3.connect(f"file:{session_dir}/storage/coordinator.db?mode=ro", uri=True)
    events = [
        (_ts(ts), topic, json.loads(payload or "{}"))
        for ts, topic, payload in db.execute("select ts, topic, payload from events where from_agent='orchestration'")
    ]
    system = session_dir / "agents/orchestration/system_prompt.KERNEL_AGENT.snapshot.md"
    picks = sorted(rng.sample(range(len(rows) - 1), min(n, len(rows) - 1)))
    out = []
    for i in picks:
        lo, hi = _ts(rows[i]["ts"]) - 2.0, _ts(rows[i + 1]["ts"]) - 2.0
        expected = set()
        for ts, topic, payload in events:
            if lo <= ts < hi:
                if topic == "request":
                    expected.add(action_key("request", payload))
                elif topic == "strategy_change":
                    expected.add(action_key("escalate_strategy_change", payload))
                elif topic == "proposal":
                    expected.add(action_key("propose_action", payload))
        out.append(
            {
                "source": "bus",
                "session": str(session_dir),
                "phase": "KERNEL_AGENT",
                "system_path": str(system),
                "prompt": str(rows[i]["prompt"]).split(_SUFFIX, 1)[0],
                "expected": sorted(expected),
                "ref": rows[i].get("call_id"),
            }
        )
    return out


def stratified(samples: list[dict[str, Any]], per_phase: dict[str, int], rng: random.Random) -> list[dict[str, Any]]:
    by_phase: dict[str, list] = defaultdict(list)
    for s in samples:
        by_phase[s["phase"]].append(s)
    out = []
    for phase, k in per_phase.items():
        pool = by_phase.get(phase, [])
        rng.shuffle(pool)
        seen_sessions: set[str] = set()
        for s in pool:  # one per session first, then fill
            if len([x for x in out if x["phase"] == phase]) >= k:
                break
            if s["session"] not in seen_sessions:
                out.append(s)
                seen_sessions.add(s["session"])
        rest = [s for s in pool if s not in out]
        out += rest[: max(0, k - len([x for x in out if x["phase"] == phase]))]
    return out


async def replay_orchestration(sample: dict[str, Any], routes: RoleModels) -> dict[str, Any]:
    backend = ClaudeBackend(
        model="claude-opus-5",
        effort_role="orchestration",
        role_models=routes,
        route_role="orchestration",
        allowed_intents=default_role_registry()["orchestration"].allowed_intents,
    )
    backend.set_route_phase(sample["phase"])
    t0 = time.time()
    got: set = set()
    error = ""
    usage: dict[str, Any] = {}
    try:
        res = await backend.run(
            prompt=sample["prompt"],
            system_prompt=Path(sample["system_path"]).read_text(),
            tools=["emit_intent", "Read"],
        )
        got = {k for k in (action_key(i.type.value, i.payload) for i in res.intents) if k}
        usage = {k: res.metadata.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")}
    except Exception as exc:  # noqa: BLE001 -- a failed replay is a data point, not a crash
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    expected = {tuple(k) for k in sample["expected"]}
    return {
        **{k: sample[k] for k in ("source", "session", "phase", "ref")},
        "expected": sorted(expected),
        "got": sorted(got),
        "error": error,
        "sec": round(time.time() - t0, 1),
        "usage": usage,
        **score(expected, got),
    }


# --- critic samples ----------------------------------------------------------------------
def _json_obj(text: str) -> dict | None:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _verdicts(obj: dict | None) -> dict[str, str]:
    out = {}
    for v in (obj or {}).get("review_verdicts") or []:
        if isinstance(v, dict) and v.get("target_proposal_msg_id"):
            out[str(v["target_proposal_msg_id"])] = str(v.get("verdict") or "").lower()
    return out


def critic_samples(sessions: list[str], n: int, rng: random.Random) -> list[dict[str, Any]]:
    pool = []
    for s in sessions:
        for r in iter_jsonl(Path(s) / "reports/trace/conversations.jsonl"):
            if r.get("component") != "critic":
                continue
            verdicts = _verdicts(_json_obj(str(r.get("response") or "")))
            if verdicts:
                pool.append(
                    {
                        "session": s,
                        "prompt": str(r.get("prompt") or ""),
                        "expected": verdicts,
                        "model": r.get("model"),
                        "ref": r.get("call_id"),
                    }
                )
    rng.shuffle(pool)
    picked, per_session = [], defaultdict(int)
    for p in pool:
        if per_session[p["session"]] < 2:
            picked.append(p)
            per_session[p["session"]] += 1
        if len(picked) >= n:
            break
    return picked


async def replay_critic(sample: dict[str, Any], routes: RoleModels) -> dict[str, Any]:
    route = routes.resolve("critic")
    client = get_async_openai_client(env=route.env(os.environ))
    t0 = time.time()
    got: dict[str, str] = {}
    error = ""
    try:
        res = await achat_completion(
            client,
            component="critic",
            operation="review",
            model=route.model,
            messages=[{"role": "user", "content": sample["prompt"]}],
            max_completion_tokens=16384,
        )
        got = _verdicts(_json_obj(res.text))
        if not got:
            error = "no parseable review_verdicts"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    exp = sample["expected"]
    same = sum(1 for k, v in exp.items() if got.get(k) == v)
    return {
        "session": sample["session"],
        "ref": sample["ref"],
        "recorded_model": sample["model"],
        "expected": exp,
        "got": got,
        "error": error,
        "sec": round(time.time() - t0, 1),
        "verdicts": len(exp),
        "agree": same,
        "exact": same == len(exp) and bool(exp),
    }


# --- driver ------------------------------------------------------------------------------
async def _bounded(coros, limit: int, sink) -> list:
    sem = asyncio.Semaphore(limit)
    results = []

    async def run(c):
        async with sem:
            r = await c
            results.append(r)
            sink(r)

    await asyncio.gather(*(run(c) for c in coros))
    return results


def summarize(orch: list[dict], critic: list[dict]) -> dict[str, Any]:
    by_phase: dict[str, dict] = {}
    for phase in _PHASES:
        rows = [r for r in orch if r["phase"] == phase]
        if not rows:
            continue
        ok = [r for r in rows if not r["error"]]
        by_phase[phase] = {
            "n": len(rows),
            "failed": len(rows) - len(ok),
            "exact": round(sum(r["exact"] for r in rows) / len(rows), 3),
            "types_match": round(sum(r["types_match"] for r in rows) / len(rows), 3),
            "jaccard": round(sum(r["jaccard"] for r in rows) / len(rows), 3),
            "sec_p50": sorted(r["sec"] for r in rows)[len(rows) // 2],
        }
    n_verdicts = sum(r["verdicts"] for r in critic)
    return {
        "orchestration": by_phase,
        "orchestration_all": {
            "n": len(orch),
            "exact": round(sum(r["exact"] for r in orch) / max(1, len(orch)), 3),
            "types_match": round(sum(r["types_match"] for r in orch) / max(1, len(orch)), 3),
            "failed": sum(1 for r in orch if r["error"]),
        },
        "critic": {
            "calls": len(critic),
            "failed": sum(1 for r in critic if r["error"]),
            "verdict_agreement": round(sum(r["agree"] for r in critic) / max(1, n_verdicts), 3),
            "call_exact": round(sum(r["exact"] for r in critic) / max(1, len(critic)), 3),
            "sec_p50": sorted(r["sec"] for r in critic)[len(critic) // 2] if critic else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", required=True, help="orchestration route JSON (anthropic protocol)")
    ap.add_argument("--openai-route", required=True, help="critic route JSON (openai protocol)")
    ap.add_argument("--kernel-session", type=Path, default=None)
    ap.add_argument("--kernel-bus-samples", type=int, default=8)
    ap.add_argument("--per-phase", default="PRELUDE=8,FRAMEWORK_AGENT=12,KERNEL_AGENT=8,SWEEP=4,CLOSE=4")
    ap.add_argument("--critic", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sessions-glob", default="/wekafs/csl/Hyperloom-Sessions/meta_rsi/analysis/*.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    routes = parse_role_models(
        json.dumps({"orchestration": json.loads(args.route), "critic": json.loads(args.openai_route)})
    )
    per_phase = {k: int(v) for k, v in (x.split("=") for x in args.per_phase.split(","))}
    orch_samples = stratified(transcript_samples(), per_phase, rng)
    if args.kernel_session:
        orch_samples += kernel_bus_samples(args.kernel_session, args.kernel_bus_samples, rng)
    sessions = [json.load(open(f))["session_dir"] for f in sorted(glob.glob(args.sessions_glob))]
    crit_samples = critic_samples(sessions, args.critic, rng)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "rows.jsonl").open("a")

    def sink(row: dict) -> None:
        log.write(json.dumps(row, default=str) + "\n")
        log.flush()

    async def run_all():
        coros = [replay_orchestration(s, routes) for s in orch_samples] + [
            replay_critic(s, routes) for s in crit_samples
        ]
        return await _bounded(coros, args.concurrency, sink)

    print(f"replaying {len(orch_samples)} orchestration + {len(crit_samples)} critic calls", flush=True)
    rows = asyncio.run(run_all())
    orch = [r for r in rows if "phase" in r]
    critic = [r for r in rows if "verdicts" in r]
    summary = summarize(orch, critic)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
