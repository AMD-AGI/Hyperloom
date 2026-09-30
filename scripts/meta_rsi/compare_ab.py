"""Compare two A/B arms and apply the round's verdict rule.

Effect kept: B's furthest phase is not earlier than A's, B's validated gain is at least A's minus
2 points or at least 90% of A's, and B has no failing stop reason A did not have.
Savings: B's Opus weighted tokens and its priced cost are both below A's. GLM is self-hosted and
priced at 0; its tokens are reported separately.

    python compare_ab.py --a <arm A user-data dir> --b <arm B user-data dir> --out <dir>
"""

import argparse
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

PHASES = ["PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE"]
ROUTER_LOG = Path(os.environ.get("AB_ROUTER_LOG", "/wekafs/csl/meta_rsi_ab/model_router.requests.jsonl"))
FAILING_STOPS = re.compile(r"fail|error|emergency|exhausted_during_prelude|escalated", re.I)
PRICES = {
    "opus": (5.0, 6.25, 0.50, 25.0),
    "sonnet": (3.0, 3.75, 0.30, 15.0),
    "gpt": (1.25, 1.25, 0.125, 10.0),
    "glm": (0.0, 0.0, 0.0, 0.0),
}
POLL = re.compile(
    r"\bsleep\s+\d|tail -[fF]\b|\bwatch\b|\bps\s+(-|aux)|rocm-smi|nvidia-smi|curl\s+-s[^|]*(health|v1/models)|heartbeat"
)


def fam(model: str | None) -> str:
    m = (model or "").lower()
    for f in ("glm", "sonnet", "gpt", "opus"):
        if f in m:
            return f
    return "opus"


def w(r: dict) -> float:
    return (
        (r.get("input_tokens") or 0)
        + 1.25 * (r.get("cache_creation_input_tokens") or 0)
        + 0.1 * (r.get("cache_read_input_tokens") or 0)
        + 5 * (r.get("output_tokens") or 0)
    )


def usd(r: dict, f: str) -> float:
    p = PRICES[f]
    return (
        (r.get("input_tokens") or 0) * p[0]
        + (r.get("cache_creation_input_tokens") or 0) * p[1]
        + (r.get("cache_read_input_tokens") or 0) * p[2]
        + (r.get("output_tokens") or 0) * p[3]
    ) / 1e6


def session_dir(udp: Path) -> Path:
    hits = sorted(udp.glob("*/*/state.json"), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no session under {udp}")
    return hits[-1].parent


def specialist_model_map(sess: Path) -> dict:
    """task_id -> model reported by the specialist CLI (its init event)."""
    out = {}
    for log in sess.glob("runs/specialist/*/process.log"):
        for line in log.open(errors="ignore"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                out[log.parent.name] = ev.get("model")
                break
    return out


def compact_ts(ts: str) -> str:
    """'2026-09-30T06:34:52Z' or '20260930T063452Z-...' -> '20260930T063452'."""
    return ts.replace("-", "").replace(":", "")[:15]


def arm_metrics(udp: Path, window: tuple[str, str]) -> dict:
    sess = session_dir(udp)
    state = json.loads((sess / "state.json").read_text())
    spec_models = specialist_model_map(sess)
    tok = defaultdict(Counter)
    orch_ctx, calls = [], Counter()
    for line in (sess / "reports/trace/llm_calls.jsonl").open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        comp = r.get("component") or "?"
        model = (
            r.get("model")
            or (spec_models.get(r.get("task_id") or "") if comp == "specialist" else None)
            or "claude-opus-5"
        )
        f = fam(model)
        key = f"{comp}:{f}"
        calls[comp] += 1
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            tok[key][k] += r.get(k) or 0
        tok[key]["weighted"] += w(r)
        tok[key]["usd"] += usd(r, f)
        if comp == "orchestration":
            orch_ctx.append(
                (r.get("input_tokens") or 0)
                + (r.get("cache_creation_input_tokens") or 0)
                + (r.get("cache_read_input_tokens") or 0)
            )
    opus_w = sum(v["weighted"] for k, v in tok.items() if k.endswith(":opus"))
    glm_raw = sum(
        v["input_tokens"] + v["cache_creation_input_tokens"] + v["cache_read_input_tokens"] + v["output_tokens"]
        for k, v in tok.items()
        if k.endswith(":glm")
    )
    phases = [h.get("to_phase") for h in state.get("phase_history") or [] if h.get("to_phase")]
    furthest = max((PHASES.index(p) for p in phases if p in PHASES and p != "CLOSE"), default=0)
    keeps = Counter()
    dt = sess / "reports/trace/decision_trace.jsonl"
    if dt.exists():
        for line in dt.open(errors="ignore"):
            try:
                keeps[str((json.loads(line).get("decision") or {}).get("outcome"))] += 1
            except ValueError:
                continue
    first_ctx, turns, polls = [], 0, 0
    for log in sess.glob("runs/specialist/*/process.log"):
        seen, first = set(), None
        for line in log.open(errors="ignore"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") != "assistant" or ev.get("parent_tool_use_id"):
                continue
            msg = ev.get("message") or {}
            if msg.get("id") in seen:
                continue
            seen.add(msg.get("id"))
            turns += 1
            u = msg.get("usage") or {}
            if first is None:
                first = (
                    (u.get("input_tokens") or 0)
                    + (u.get("cache_creation_input_tokens") or 0)
                    + (u.get("cache_read_input_tokens") or 0)
                )
            if any(
                b.get("type") == "tool_use"
                and b.get("name") == "Bash"
                and POLL.search(str((b.get("input") or {}).get("command") or ""))
                for b in msg.get("content") or []
            ):
                polls += 1
        if first:
            first_ctx.append(first)
    findings_chars = []
    conv = sess / "reports/trace/conversations.jsonl"
    if conv.exists():
        for line in conv.open(errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            p = r.get("prompt") or ""
            if r.get("component") == "orchestration" and "=== Specialist findings ===" in p:
                s = p.find("=== Specialist findings ===")
                e = p.find("\n=== ", s + 10)
                findings_chars.append((e if e > 0 else len(p)) - s)
    run_log = udp / "optimizer_runs/run.log"
    idle_skips = run_log.read_text(errors="ignore").count("idle gate: skipped") if run_log.exists() else 0
    # The GLM endpoint reports output tokens only; the router log gives its request count and request bytes.
    router = Counter()
    if ROUTER_LOG.exists():
        for line in ROUTER_LOG.open(errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ts = compact_ts(r.get("ts", ""))
            if ts < window[0] or (window[1] and ts >= window[1]):
                continue
            route = r.get("route", "?")
            router[f"{route}_requests"] += 1
            router[f"{route}_req_bytes"] += r.get("req_bytes") or 0
            router[f"{route}_errors"] += 1 if (r.get("status") or 0) >= 400 else 0
    return {
        "session": str(sess),
        "stop_reason": state.get("stop_reason"),
        "furthest_phase": PHASES[furthest],
        "validated_gain_pct": state.get("cumulative_gain_validated"),
        "decisions": dict(keeps),
        "calls": dict(calls),
        "tokens": {k: dict(v) for k, v in sorted(tok.items())},
        "opus_weighted": opus_w,
        "cost_usd": sum(v["usd"] for v in tok.values()),
        "glm_raw_tokens": glm_raw,
        "orchestration_ctx_p50": st.median(orch_ctx) if orch_ctx else None,
        "idle_gate_skips": idle_skips,
        "specialist_runs": len(first_ctx),
        "specialist_first_ctx_p50": st.median(first_ctx) if first_ctx else None,
        "specialist_turns": turns,
        "specialist_poll_share": round(polls / turns, 3) if turns else None,
        "findings_block_chars_p50": st.median(findings_chars) if findings_chars else 0,
        "findings_block_chars_max": max(findings_chars) if findings_chars else 0,
        "router": dict(router),
    }


def verdict(a: dict, b: dict) -> dict:
    ga, gb = a["validated_gain_pct"] or 0.0, b["validated_gain_pct"] or 0.0
    phase_ok = PHASES.index(b["furthest_phase"]) >= PHASES.index(a["furthest_phase"])
    gain_ok = gb >= ga - 2.0 or (ga > 0 and gb >= 0.9 * ga)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a_start = compact_ts(session_dir(Path(args.a)).name)
    b_start = compact_ts(session_dir(Path(args.b)).name)
    a = arm_metrics(Path(args.a), (a_start, b_start if b_start > a_start else ""))
    b = arm_metrics(Path(args.b), (b_start, a_start if a_start > b_start else ""))
    result = {"A": a, "B": b, "verdict": verdict(a, b)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["verdict"], indent=1))
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
