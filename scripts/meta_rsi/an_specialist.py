# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist turn patterns from Claude CLI stream-json process logs.

Per run: tools loaded at init, turns (unique assistant message ids), first-turn context,
tool-call mix, read-only single-call turns that follow another such turn (could have been
batched), polling turns (sleep/tail/ps/health checks), oversized tool results, files read
more than once, and the weighted tokens attributable to each pattern (the turn's own
weighted usage).

Process logs come from the fetched bundles; ``PULSE_LOCAL_SPECIALIST_GLOB`` optionally adds
local ones (an absolute glob), reported under the ``local`` label.
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from round_env import analysis_dir, bundles_dir

READ_ONLY_TOOLS = {"Read", "Grep", "Glob", "LS"}
READ_ONLY_BASH = re.compile(
    r"^\s*(cat|head|tail|grep|rg|ls|find|sed -n|wc|stat|file|tree|awk|git (log|show|diff|status|grep)|python3? -c|pip show|du|readlink|realpath|which|echo)\b"
)
POLL_BASH = re.compile(
    r"\bsleep\s+\d|tail -[fF]\b|\bwatch\b|\bps\s+(-|aux)|rocm-smi|nvidia-smi|curl\s+-s[^|]*(health|v1/models)|heartbeat"
)
BIG_RESULT = 20000


def w(u: dict) -> float:
    return (
        (u.get("input_tokens") or 0)
        + 1.25 * (u.get("cache_creation_input_tokens") or 0)
        + 0.1 * (u.get("cache_read_input_tokens") or 0)
        + 5 * (u.get("output_tokens") or 0)
    )


def parse(path: Path) -> dict | None:
    tools_loaded = []
    turns = {}  # message id -> {"usage", "tools": [(name, input)], "order"}
    order = []
    results = {}  # tool_use_id -> size
    for line in path.open(errors="ignore"):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            tools_loaded = ev.get("tools") or []
        elif t == "assistant":
            msg = ev.get("message") or {}
            if ev.get("parent_tool_use_id"):
                continue  # sub-agent traffic; counted by its own usage elsewhere
            mid = msg.get("id") or f"anon{len(order)}"
            if mid not in turns:
                turns[mid] = {"usage": msg.get("usage") or {}, "tools": []}
                order.append(mid)
            else:
                turns[mid]["usage"] = msg.get("usage") or turns[mid]["usage"]
            for block in msg.get("content") or []:
                if block.get("type") == "tool_use":
                    turns[mid]["tools"].append((block.get("name"), block.get("input") or {}, block.get("id")))
        elif t == "user":
            msg = ev.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        c = block.get("content")
                        size = len(c) if isinstance(c, str) else len(json.dumps(c)) if c is not None else 0
                        results[block.get("tool_use_id")] = size
    if not order:
        return None
    stats = Counter()
    first = turns[order[0]]["usage"]
    stats["first_ctx"] = (
        (first.get("input_tokens") or 0)
        + (first.get("cache_creation_input_tokens") or 0)
        + (first.get("cache_read_input_tokens") or 0)
    )
    stats["tools_loaded"] = len(tools_loaded)
    stats["turns"] = len(order)
    files_read = Counter()
    prev_small_ro = False
    for mid in order:
        tr = turns[mid]
        tw = w(tr["usage"])
        stats["w"] += tw
        names = [n for n, _, _ in tr["tools"]]
        for n in names:
            stats["tool_" + str(n)] += 1
        if not tr["tools"]:
            stats["no_tool_turns"] += 1
        ro = bool(tr["tools"]) and all(
            n in READ_ONLY_TOOLS or (n == "Bash" and READ_ONLY_BASH.match(str(inp.get("command") or "")))
            for n, inp, _ in tr["tools"]
        )
        small = sum(results.get(tid, 0) for _, _, tid in tr["tools"]) < 4000
        single = len(tr["tools"]) == 1
        if ro and single and small:
            if prev_small_ro:
                stats["batchable_turns"] += 1
                stats["batchable_w"] += tw
            prev_small_ro = True
        else:
            prev_small_ro = False
        if any(n == "Bash" and POLL_BASH.search(str(inp.get("command") or "")) for n, inp, _ in tr["tools"]):
            stats["poll_turns"] += 1
            stats["poll_w"] += tw
        for n, inp, tid in tr["tools"]:
            size = results.get(tid, 0)
            if size >= BIG_RESULT:
                stats["big_results"] += 1
                stats["big_result_chars"] += size
            if n == "Read" and inp.get("file_path"):
                files_read[inp["file_path"]] += 1
        if len(tr["tools"]) > 1:
            stats["parallel_turns"] += 1
    stats["reread_calls"] = sum(c - 1 for c in files_read.values() if c > 1)
    return stats


def main() -> None:
    out, bundles = analysis_dir(), bundles_dir()
    joined = {}
    for line in open(out / "runs_joined.jsonl"):
        j = json.loads(line)
        joined[j["name"]] = j["era"]
    sources = []
    for p in bundles.glob("*/*/**/runs/specialist/*/process.log"):
        sources.append((joined.get(p.relative_to(bundles).parts[1], "?"), p))
    local_glob = os.environ.get("PULSE_LOCAL_SPECIALIST_GLOB", "").strip()
    if local_glob:
        anchor = Path(local_glob.split("*", 1)[0])
        for p in anchor.glob(local_glob[len(str(anchor)) :].lstrip("/")):
            sources.append(("local", p))
    agg = defaultdict(Counter)
    dist = defaultdict(lambda: defaultdict(list))
    for era, p in sources:
        s = parse(p)
        if s is None:
            continue
        agg[era]["runs"] += 1
        for k, v in s.items():
            agg[era][k] += v
        for k in ("first_ctx", "turns", "tools_loaded"):
            dist[era][k].append(s[k])
    lines = []
    for era, a in sorted(agg.items()):
        n = a["runs"]
        tw = a["w"] or 1
        turns = a["turns"] or 1
        lines.append(f"## {era}: runs={n} turns={a['turns']:,} weighted={a['w'] / 1e6:,.1f}M")
        lines.append(
            f"  first-turn context p50={st.median(dist[era]['first_ctx']):,.0f} p90={sorted(dist[era]['first_ctx'])[int(0.9 * len(dist[era]['first_ctx']))]:,}; "
            f"turns/run p50={st.median(dist[era]['turns']):.0f} p90={sorted(dist[era]['turns'])[int(0.9 * len(dist[era]['turns']))]}; "
            f"tools loaded p50={st.median(dist[era]['tools_loaded']):.0f}"
        )
        lines.append(
            f"  batchable read-only turns: {a['batchable_turns']:,} ({100 * a['batchable_turns'] / turns:.1f}% of turns, {100 * a['batchable_w'] / tw:.1f}% of weighted)"
        )
        lines.append(
            f"  polling turns: {a['poll_turns']:,} ({100 * a['poll_turns'] / turns:.1f}% of turns, {100 * a['poll_w'] / tw:.1f}% of weighted)"
        )
        lines.append(
            f"  multi-tool turns: {a['parallel_turns']:,} ({100 * a['parallel_turns'] / turns:.1f}%); turns without tools: {a['no_tool_turns']:,}"
        )
        lines.append(
            f"  oversized tool results (>= {BIG_RESULT} chars): {a['big_results']:,}, avg {a['big_result_chars'] / max(1, a['big_results']):,.0f} chars; re-read calls: {a['reread_calls']:,}"
        )
        top_tools = sorted(((k[5:], v) for k, v in a.items() if k.startswith("tool_")), key=lambda kv: -kv[1])[:10]
        lines.append("  tool calls: " + ", ".join(f"{k}={v:,}" for k, v in top_tools))
    text = "\n".join(lines)
    (out / "specialist_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
