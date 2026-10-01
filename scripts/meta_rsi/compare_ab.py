# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare two A/B arms and apply the round's comparison rule.

Effect kept: B's furthest phase is not earlier than A's, B's validated gain is at least A's minus
2 points or at least 90% of A's, and B has no failing stop reason A did not have. An arm that never
validated a gain fails the gain test.
Savings: B's Opus weighted tokens and its priced cost are both below A's (prices in
``metrics.py``). A model that names no priced family is an error, not a guess.

The rule is advisory: with one run per arm it cannot separate a code effect from the optimizer's
run-to-run variance, so read it together with each arm's exploration record.

    cd scripts && python -m meta_rsi.compare_ab --a <arm A user-data dir> --b <arm B user-data dir> \\
        --out <dir> [--router-log <model router requests.jsonl>]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from meta_rsi.metrics import POLL, TOKEN_FIELDS, context_tokens, cost_usd, price_family, weighted

PHASES = ["PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE"]
FAILING_STOPS = re.compile(r"fail|error|emergency|exhausted_during_prelude|escalated", re.I)
FINDINGS_HEADER = "=== Specialist findings ==="


def jsonl(path: Path):
    """Parsed rows of a JSONL file, skipping lines that do not parse; nothing when the file is absent."""
    if not path.exists():
        return
    for line in path.open(errors="ignore"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def session_dir(udp: Path) -> Path:
    hits = sorted(udp.glob("*/*/state.json"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no session under {udp}")
    return hits[-1].parent


def specialist_model_map(sess: Path) -> dict:
    """task_id -> model reported by the specialist CLI (its init event)."""
    out = {}
    for log in sess.glob("runs/specialist/*/process.log"):
        for ev in jsonl(log):
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                out[log.parent.name] = ev.get("model")
                break
    return out


def compact_ts(ts: str) -> str:
    """'2026-09-30T06:34:52Z' or '20260930T063452Z-...' -> '20260930T063452'."""
    return ts.replace("-", "").replace(":", "")[:15]


def ledger_tokens(sess: Path) -> tuple[dict, Counter, list]:
    """Tokens and priced cost per component:family, call counts, and orchestration context sizes."""
    spec_models = specialist_model_map(sess)
    tok, calls, orch_ctx = defaultdict(Counter), Counter(), []
    for r in jsonl(sess / "reports/trace/llm_calls.jsonl"):
        comp = r.get("component") or "?"
        model = (
            r.get("model")
            or (spec_models.get(r.get("task_id") or "") if comp == "specialist" else None)
            or "claude-opus-5"
        )
        f = price_family(model)
        if f is None:
            raise SystemExit(f"{sess}: model {model!r} names no single family in metrics.PRICES; add it there")
        key = f"{comp}:{f}"
        calls[comp] += 1
        for k in TOKEN_FIELDS:
            tok[key][k] += r.get(k) or 0
        tok[key]["weighted"] += weighted(r)
        tok[key]["usd"] += cost_usd(r, f)
        if comp == "orchestration":
            orch_ctx.append(context_tokens(r))
    return tok, calls, orch_ctx


def specialist_turns(sess: Path) -> tuple[list, int, int]:
    """First-turn context per specialist run, total turns, and turns that poll."""
    first_ctx, turns, polls = [], 0, 0
    for log in sess.glob("runs/specialist/*/process.log"):
        seen, first = set(), None
        for ev in jsonl(log):
            if ev.get("type") != "assistant" or ev.get("parent_tool_use_id"):
                continue
            msg = ev.get("message") or {}
            if msg.get("id") in seen:
                continue
            seen.add(msg.get("id"))
            turns += 1
            if first is None:
                first = context_tokens(msg.get("usage") or {})
            if any(
                b.get("type") == "tool_use"
                and b.get("name") == "Bash"
                and POLL.search(str((b.get("input") or {}).get("command") or ""))
                for b in msg.get("content") or []
            ):
                polls += 1
        if first:
            first_ctx.append(first)
    return first_ctx, turns, polls


def findings_block_sizes(sess: Path) -> list:
    """Characters of the specialist findings block in each orchestration prompt that has one."""
    sizes = []
    for r in jsonl(sess / "reports/trace/conversations.jsonl"):
        p = r.get("prompt") or ""
        if r.get("component") == "orchestration" and FINDINGS_HEADER in p:
            s = p.find(FINDINGS_HEADER)
            e = p.find("\n=== ", s + 10)
            sizes.append((e if e > 0 else len(p)) - s)
    return sizes


def router_counts(router_log: Path | None, window: tuple[str, str]) -> Counter:
    """Requests, request bytes and errors per route inside the arm's time window."""
    router = Counter()
    if router_log is None:
        return router
    for r in jsonl(router_log):
        ts = compact_ts(r.get("ts", ""))
        if ts < window[0] or (window[1] and ts >= window[1]):
            continue
        route = r.get("route", "?")
        router[f"{route}_requests"] += 1
        router[f"{route}_req_bytes"] += r.get("req_bytes") or 0
        router[f"{route}_errors"] += 1 if (r.get("status") or 0) >= 400 else 0
    return router


def arm_metrics(udp: Path, window: tuple[str, str], router_log: Path | None = None) -> dict:
    sess = session_dir(udp)
    state = json.loads((sess / "state.json").read_text())
    tok, calls, orch_ctx = ledger_tokens(sess)
    phases = [h.get("to_phase") for h in state.get("phase_history") or [] if h.get("to_phase")]
    furthest = max((PHASES.index(p) for p in phases if p in PHASES and p != "CLOSE"), default=0)
    decisions = Counter(
        str((row.get("decision") or {}).get("outcome")) for row in jsonl(sess / "reports/trace/decision_trace.jsonl")
    )
    first_ctx, turns, polls = specialist_turns(sess)
    findings = findings_block_sizes(sess)
    run_log = udp / "optimizer_runs/run.log"
    return {
        "session": str(sess),
        "stop_reason": state.get("stop_reason"),
        "furthest_phase": PHASES[furthest],
        "validated_gain_pct": state.get("cumulative_gain_validated"),
        "decisions": dict(decisions),
        "calls": dict(calls),
        "tokens": {k: dict(v) for k, v in sorted(tok.items())},
        "opus_weighted": sum(v["weighted"] for k, v in tok.items() if k.endswith(":opus")),
        "cost_usd": sum(v["usd"] for v in tok.values()),
        "glm_raw_tokens": sum(sum(v[f] for f in TOKEN_FIELDS) for k, v in tok.items() if k.endswith(":glm")),
        "orchestration_ctx_p50": st.median(orch_ctx) if orch_ctx else None,
        "idle_gate_skips": run_log.read_text(errors="ignore").count("idle gate: skipped") if run_log.exists() else 0,
        "specialist_runs": len(first_ctx),
        "specialist_first_ctx_p50": st.median(first_ctx) if first_ctx else None,
        "specialist_turns": turns,
        "specialist_poll_share": round(polls / turns, 3) if turns else None,
        "findings_block_chars_p50": st.median(findings) if findings else 0,
        "findings_block_chars_max": max(findings) if findings else 0,
        "router": dict(router_counts(router_log, window)),
    }


def verdict(a: dict, b: dict) -> dict:
    ga, gb = a["validated_gain_pct"], b["validated_gain_pct"]
    phase_ok = PHASES.index(b["furthest_phase"]) >= PHASES.index(a["furthest_phase"])
    gain_ok = ga is not None and gb is not None and (gb >= ga - 2.0 or (ga > 0 and gb >= 0.9 * ga))
    new_failure = bool(FAILING_STOPS.search(str(b["stop_reason"] or ""))) and not FAILING_STOPS.search(
        str(a["stop_reason"] or "")
    )
    effect_ok = phase_ok and gain_ok and not new_failure
    savings_ok = b["opus_weighted"] < a["opus_weighted"] and b["cost_usd"] < a["cost_usd"]
    return {
        "effect_kept": effect_ok,
        "phase_ok": phase_ok,
        "gain_ok": gain_ok,
        "new_failure": new_failure,
        "savings": savings_ok,
        "opus_weighted_saving_pct": round(100 * (1 - b["opus_weighted"] / a["opus_weighted"]), 1)
        if a["opus_weighted"]
        else None,
        "cost_saving_pct": round(100 * (1 - b["cost_usd"] / a["cost_usd"]), 1) if a["cost_usd"] else None,
        "pass": effect_ok and savings_ok,
    }


def last_activity(udp: Path) -> str:
    """A minute past the arm's last ledger row (compact form), or open-ended when it has none."""
    stamps = [r["ts"] for r in jsonl(session_dir(udp) / "reports/trace/llm_calls.jsonl") if r.get("ts")]
    if not stamps:
        return ""
    last = datetime.fromisoformat(max(stamps).replace("Z", "+00:00")) + timedelta(minutes=1)
    return last.strftime("%Y%m%dT%H%M%S")


def compare(arm_a: Path, arm_b: Path, router_log: Path | None = None) -> dict:
    """Both arms' metrics and the rule's verdict. An arm's router window ends where the other arm
    starts, or, for the later arm, just after its own last ledger row."""
    a_start = compact_ts(session_dir(arm_a).name)
    b_start = compact_ts(session_dir(arm_b).name)
    a_end = b_start if b_start > a_start else last_activity(arm_a)
    b_end = a_start if a_start > b_start else last_activity(arm_b)
    a = arm_metrics(arm_a, (a_start, a_end), router_log)
    b = arm_metrics(arm_b, (b_start, b_end), router_log)
    return {"A": a, "B": b, "verdict": verdict(a, b)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--router-log", default="", help="model router request log, for per-route request counts")
    args = ap.parse_args()
    result = compare(Path(args.a), Path(args.b), Path(args.router_log) if args.router_log else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["verdict"], indent=1))
    a, b = result["A"], result["B"]
    for k in (
        "stop_reason",
        "furthest_phase",
        "validated_gain_pct",
        "decisions",
        "calls",
        "opus_weighted",
        "cost_usd",
        "glm_raw_tokens",
        "orchestration_ctx_p50",
        "idle_gate_skips",
        "specialist_runs",
        "specialist_first_ctx_p50",
        "specialist_turns",
        "specialist_poll_share",
        "findings_block_chars_p50",
        "findings_block_chars_max",
        "router",
    ):
        print(f"{k:28s} A={a[k]!s:30.30s} B={b[k]!s:30.30s}")


if __name__ == "__main__":
    main()
