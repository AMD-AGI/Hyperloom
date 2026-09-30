"""Offline replay of the implemented levers over recent Pulse sessions, using the branch's own code.

Run with PYTHONPATH=<branch>/src so the idle gate is the harness implementation, not a copy.
Estimates (reported as such): a token is ~3.5 characters of prompt text; a per-tick prompt is written
to the cache once (1.25x) and read back on every further request of the turn (0.1x each); a turn's
request count is its context divided by the run's first-call prefix.
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from hyperloom.orchestrator.loop.idle_gate import IdleTickGate, prompt_digest

sys.path.insert(0, str(Path(__file__).parent))
from an_global import cost, price_family, weighted

BUNDLES = Path(os.environ.get("PULSE_BUNDLES", "/root/pulse15d/02_bundles"))
OUT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d")) / "analysis"
CHARS_PER_TOKEN = 3.5
FINDINGS_BUDGET = 12000
TOOL_SCHEMA_TOKENS = 34444 - 7103  # measured: default built-ins vs the narrowed set
ACTION = re.compile(r"\b(emitted|dispatched|queued (one|a|another)|launched|submitted|delegated)\b", re.I)


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def findings_chars(prompt: str) -> int:
    start = prompt.find("=== Specialist findings ===")
    if start < 0:
        return 0
    nxt = prompt.find("\n=== ", start + 10)
    return (nxt if nxt > 0 else len(prompt)) - start


def main() -> None:
    eras = {}
    for line in open(OUT / "runs_joined.jsonl"):
        j = json.loads(line)
        eras[j["name"]] = j["era"]
    t = Counter()
    for conv in sorted(BUNDLES.glob("*/*/**/reports/trace/conversations.jsonl")):
        archive = conv.relative_to(BUNDLES).parts[1]
        if eras.get(archive) != "recent":
            continue
        run_dir = conv.parent.parent.parent
        led = {}
        led_path = run_dir / "reports/trace/llm_calls.jsonl"
        if led_path.exists():
            for line in led_path.open(errors="ignore"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("component") == "orchestration" and r.get("call_id"):
                    led[r["call_id"]] = r
        rows = []
        for line in conv.open(errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("component") == "orchestration" and r.get("prompt") and r.get("ts"):
                rows.append(r)
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: r["ts"])
        t["runs"] += 1
        first_prefix = None
        gate = IdleTickGate(enabled=True, heartbeat_sec=900.0)
        digests = [prompt_digest(r["prompt"]) for r in rows]
        for i, r in enumerate(rows):
            lr = led.get(r.get("call_id"), {})
            c = cost(lr, price_family(lr.get("model") or r.get("model")))
            w = weighted(lr)
            ctx = (
                (lr.get("input_tokens") or 0)
                + (lr.get("cache_creation_input_tokens") or 0)
                + (lr.get("cache_read_input_tokens") or 0)
            )
            prompt_tokens = len(r["prompt"]) / CHARS_PER_TOKEN
            if first_prefix is None and ctx:
                # static prefix (system prompt + tool schemas) = first request's context minus its own prompt
                first_write = (lr.get("cache_creation_input_tokens") or 0) + (lr.get("input_tokens") or 0) or ctx
                first_prefix = max(1.0, first_write - prompt_tokens)
            requests = max(1.0, ctx / (first_prefix + prompt_tokens)) if first_prefix else 1.0
            t["calls"] += 1
            t["cost"] += c
            t["w"] += w
            now = ts(r["ts"])
            if gate.should_skip(r["prompt"], now):
                t["gate_skip"] += 1
                t["gate_cost"] += c
                t["gate_w"] += w
                changed_next = i + 1 < len(rows) and digests[i + 1] != digests[i]
                if ACTION.search(r.get("response") or "") and changed_next:
                    t["gate_miss"] += 1
                continue
            gate.record_sent(r["prompt"], now)
            # findings bound: characters above the budget leave the per-tick prompt
            excess = max(0, findings_chars(r["prompt"]) - FINDINGS_BUDGET)
            if excess:
                tokens = excess / CHARS_PER_TOKEN
                t["findings_calls"] += 1
                t["findings_w"] += tokens * (1.25 + 0.1 * (requests - 1))
            # tool narrowing: the schema tokens are cached reads on every request of the turn
            t["tools_w"] += TOOL_SCHEMA_TOKENS * 0.1 * requests
    lines = [
        f"recent runs with orchestration prompts: {t['runs']}, calls {t['calls']:,}, cost ${t['cost']:,.0f}, weighted {t['w'] / 1e6:,.1f}M",
        f"idle gate (harness code, heartbeat 900 s): skips {t['gate_skip']:,} calls ({100 * t['gate_skip'] / max(1, t['calls']):.1f}%), "
        f"${t['gate_cost']:,.0f} ({100 * t['gate_cost'] / max(1e-9, t['cost']):.1f}% of orchestration cost); "
        f"skipped calls that acted and changed the next prompt: {t['gate_miss']} ({100 * t['gate_miss'] / max(1, t['gate_skip']):.1f}%)",
        f"findings bound at {FINDINGS_BUDGET} chars (on calls still sent): {t['findings_calls']:,} calls trimmed, "
        f"~{t['findings_w'] / 1e6:,.1f}M weighted ({100 * t['findings_w'] / max(1, t['w']):.1f}% of orchestration weighted)",
        f"tool narrowing (on calls still sent): ~{t['tools_w'] / 1e6:,.1f}M weighted ({100 * t['tools_w'] / max(1, t['w']):.1f}% of orchestration weighted)",
    ]
    text = "\n".join(lines)
    (OUT / "levers_replay.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
