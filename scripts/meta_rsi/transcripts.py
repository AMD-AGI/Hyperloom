#!/usr/bin/env python3
"""Mine Claude Code transcripts (``~/.claude/projects/*/*.jsonl``) of Hyperloom agents.

    python scripts/meta_rsi/transcripts.py [--root ~/.claude/projects] [--since 20260825] --out DIR

A transcript is the full multi-turn record of one agent run: every assistant turn with
its usage, every tool call and every tool result. This is where specialist and forge
tokens go, and it is not in the session's own traces. Per transcript it reports turns,
the first-turn context (what every later turn re-reads: CLI system prompt, tool schemas,
the initial prompt), context growth per turn, tool calls and result sizes by tool, the
largest results, and repeated identical calls / re-read files.

Agent role is inferred from the transcript directory, which Claude Code names after the
run's working directory (``...-runs-specialist-<task>-worktree`` for a specialist,
``...-forge-...`` for forge, the bare session dir for orchestration).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from . import _paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _paths  # noqa: F401

from hyperloom.inference_optimizer.trace.meta_rsi_ledger import COST_WEIGHTS

_SID = re.compile(r"(\d{8}T\d{6}Z-[0-9a-f]{8})")


def classify(dirname: str, first_user: str = "") -> str:
    if "-runs-specialist-" in dirname:
        return "specialist"
    if first_user.startswith("SESSION_DIR="):
        return "orchestration"
    if first_user.startswith("You are analyzing the DECODE path"):
        return "breakdown"
    if first_user.startswith("Analyze the assigned optimization problem"):
        return "forge_analysis"
    if first_user.startswith("# Fuse "):
        return "fusion"
    if "forge" in dirname:
        return "forge"
    if "critic" in dirname:
        return "critic"
    m = _SID.search(dirname)
    if m and dirname.endswith(m.group(1)):
        return "orchestration"
    return "other"


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    total += len(c.get("text") or "")
                elif c.get("type") == "image":
                    total += 0
                else:
                    total += len(json.dumps(c.get("content") if "content" in c else c))
            else:
                total += len(str(c))
        return total
    return len(json.dumps(content)) if content is not None else 0


def call_key(name: str, inp: Any) -> str:
    """Short human-readable identity of a tool call, for grouping."""
    if not isinstance(inp, dict):
        return name
    if name == "Bash":
        cmd = re.sub(r"\s+", " ", str(inp.get("command") or "")).strip()
        return f"Bash: {cmd[:140]}"
    if name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        return f"{name}: {inp.get('file_path') or inp.get('notebook_path')}"
    if name == "Grep":
        return f"Grep: {inp.get('pattern')} @ {inp.get('path') or '.'}"
    if name == "Glob":
        return f"Glob: {inp.get('pattern')} @ {inp.get('path') or '.'}"
    args = ",".join(f"{k}={str(v)[:40]}" for k, v in sorted(inp.items())[:3])
    return f"{name}: {args}"


def command_family(key: str) -> str:
    """Coarse grouping of a call key: tool plus the first word(s) of a shell command."""
    if key.startswith("Bash: "):
        cmd = key[6:]
        cmd = re.sub(r"^(cd [^;&]+(&&|;)\s*)+", "", cmd)
        words = cmd.split()
        head = words[0] if words else ""
        if head in ("python", "python3") and len(words) > 1:
            head = f"{head} {words[1] if not words[1].startswith('-') else words[1] + ' ' + (words[2] if len(words) > 2 else '')}"
        return f"Bash: {head[:60]}"
    return key.split(":")[0]


def analyze_transcript(path: Path) -> dict[str, Any] | None:
    usage_by_msg: dict[str, dict[str, int]] = {}
    order: list[str] = []
    tool_calls: dict[str, tuple[str, str]] = {}
    exact_inputs: dict[str, str] = {}
    turn_tools: dict[str, list[str]] = defaultdict(list)
    result_chars: dict[str, int] = {}
    first_user = None
    first_text = ""
    ts_first = ts_last = None
    try:
        fh = path.open()
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            ts = r.get("timestamp")
            if ts:
                ts_first = ts_first or ts
                ts_last = ts
            m = r.get("message") or {}
            content = m.get("content")
            if r.get("type") == "user":
                if first_user is None and not (
                    isinstance(content, list)
                    and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
                ):
                    first_user = _text_len(content)
                    if isinstance(content, str):
                        first_text = content.strip()[:400]
                    elif isinstance(content, list):
                        first_text = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                        ).strip()[:400]
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            result_chars[str(c.get("tool_use_id"))] = _text_len(c.get("content"))
            elif r.get("type") == "assistant":
                mid = str(m.get("id") or r.get("uuid"))
                u = m.get("usage") or {}
                if mid not in usage_by_msg:
                    order.append(mid)
                usage_by_msg[mid] = {
                    "input": int(u.get("input_tokens") or 0),
                    "cache_read": int(u.get("cache_read_input_tokens") or 0),
                    "cache_write": int(u.get("cache_creation_input_tokens") or 0),
                    "output": int(u.get("output_tokens") or 0),
                }
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            tid = str(c.get("id"))
                            tool_calls[tid] = (str(c.get("name")), call_key(str(c.get("name")), c.get("input")))
                            exact_inputs[tid] = f"{c.get('name')}:{json.dumps(c.get('input'), sort_keys=True)}"
                            turn_tools[mid].append(tid)
    if not order:
        return None
    ctx = [usage_by_msg[i]["input"] + usage_by_msg[i]["cache_read"] + usage_by_msg[i]["cache_write"] for i in order]
    tot = Counter()
    for i in order:
        tot.update(usage_by_msg[i])
    by_tool_calls: Counter[str] = Counter()
    by_tool_chars: Counter[str] = Counter()
    by_family_chars: Counter[str] = Counter()
    keys: Counter[str] = Counter()
    read_paths: Counter[str] = Counter()
    big: list[tuple[int, str]] = []
    for tid, (name, key) in tool_calls.items():
        size = result_chars.get(tid, 0)
        by_tool_calls[name] += 1
        by_tool_chars[name] += size
        by_family_chars[command_family(key)] += size
        keys[exact_inputs.get(tid, key)] += 1
        if name == "Read":
            read_paths[key] += 1
        big.append((size, key))
    big.sort(reverse=True)
    bash_sizes = [result_chars.get(tid, 0) // 4 for tid, (name, _k) in tool_calls.items() if name == "Bash"]
    bash_hist = Counter()
    for t in bash_sizes:
        bash_hist["<1k" if t < 1000 else "1k-4k" if t < 4000 else "4k-8k" if t < 8000 else ">=8k"] += t
    repeated_calls = sum(n - 1 for n in keys.values() if n > 1)
    repeated_chars = 0
    seen: Counter[str] = Counter()
    for tid, (name, key) in tool_calls.items():
        exact = exact_inputs.get(tid, key)
        seen[exact] += 1
        if seen[exact] > 1:
            repeated_chars += result_chars.get(tid, 0)
    # Turns whose tool results were tiny are status checks: their cost is the context re-read.
    small_turn_ctx = 0
    small_turns = 0
    for mid, tids in turn_tools.items():
        if (
            tids
            and all(result_chars.get(t, 0) < 1200 for t in tids)
            and not any(tool_calls.get(t, ("", ""))[0] in ("Edit", "Write", "MultiEdit") for t in tids)
        ):
            small_turns += 1
            u = usage_by_msg.get(mid, {})
            small_turn_ctx += u.get("input", 0) + u.get("cache_read", 0) + u.get("cache_write", 0)
    return {
        "file": str(path),
        "turns": len(order),
        "ctx_first": ctx[0],
        "ctx_last": ctx[-1],
        "ctx_max": max(ctx),
        "growth_per_turn": round((ctx[-1] - ctx[0]) / max(1, len(ctx) - 1)),
        "first_user_tok": round((first_user or 0) / 4),
        "tokens": dict(tot),
        "billed": sum(tot.values()),
        "weighted": round(sum(COST_WEIGHTS[k] * tot.get(k, 0) for k in COST_WEIGHTS)),
        "tool_calls": dict(by_tool_calls),
        "tool_result_tok": {k: round(v / 4) for k, v in by_tool_chars.items()},
        "family_result_tok": {k: round(v / 4) for k, v in by_family_chars.most_common(15)},
        "repeated_calls": repeated_calls,
        "repeated_result_tok": round(repeated_chars / 4),
        "top_repeated": [[k[:200], n] for k, n in keys.most_common(5) if n > 1],
        "small_result_turns": small_turns,
        "small_result_turn_ctx": small_turn_ctx,
        "bash_result_tok_by_size": dict(bash_hist),
        "reread_files": sum(n - 1 for n in read_paths.values() if n > 1),
        "largest_results": [{"tok": round(s / 4), "call": k} for s, k in big[:6]],
        "ts_first": ts_first,
        "ts_last": ts_last,
        "first_text": first_text,
    }


def scan(root: Path, since: str) -> list[dict[str, Any]]:
    """Transcripts of runs whose session id (from the dir name, else the first message) is >= since."""
    out = []
    since_ts = f"{since[:4]}-{since[4:6]}-{since[6:8]}"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _SID.search(d.name)
        if m and m.group(1) < since:
            continue
        for f in sorted(d.glob("*.jsonl")):
            res = analyze_transcript(f)
            if not res:
                continue
            first = res.get("first_text") or ""
            sid_m = m or _SID.search(first)
            if sid_m is None and str(res.get("ts_first") or "") < since_ts:
                continue
            if sid_m is not None and sid_m.group(1) < since:
                continue
            res.update({"dir": d.name, "role": classify(d.name, first), "sid": sid_m.group(1) if sid_m else "unknown"})
            res.pop("first_text", None)
            out.append(res)
    return out


def _p50(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = []
    add = lines.append
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)
    add(f"# Claude Code transcript mining ({len(rows)} transcripts, {len({r['sid'] for r in rows})} sessions)\n")
    add(
        "| role | transcripts | turns p50 | first-turn ctx p50 | growth/turn p50 | billed | weighted | repeated calls | re-read files |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for role, rs in sorted(by_role.items(), key=lambda kv: -sum(r["weighted"] for r in kv[1])):
        add(
            f"| {role} | {len(rs)} | {_p50([r['turns'] for r in rs]):.0f} | {_p50([r['ctx_first'] for r in rs]):.0f} | "
            f"{_p50([r['growth_per_turn'] for r in rs]):.0f} | {sum(r['billed'] for r in rs) / 1e6:.1f}M | "
            f"{sum(r['weighted'] for r in rs) / 1e6:.1f}M | {sum(r['repeated_calls'] for r in rs)} | {sum(r['reread_files'] for r in rs)} |"
        )
    for role in ("specialist", "orchestration", "forge", "forge_analysis", "breakdown", "fusion"):
        rs = by_role.get(role) or []
        if not rs:
            continue
        calls: Counter[str] = Counter()
        tok: Counter[str] = Counter()
        fam: Counter[str] = Counter()
        for r in rs:
            calls.update(r["tool_calls"])
            tok.update(r["tool_result_tok"])
            fam.update(r["family_result_tok"])
        total_tool_tok = sum(tok.values()) or 1
        add(f"\n## {role}: tool results (tokens that enter the context once and are re-read every later turn)\n")
        add(
            f"First-turn context p50 {_p50([r['ctx_first'] for r in rs]):.0f} tok vs initial prompt p50 "
            f"{_p50([r['first_user_tok'] for r in rs]):.0f} tok; repeated identical calls brought "
            f"{sum(r['repeated_result_tok'] for r in rs) / 1e3:.0f}k tok of results.\n"
        )
        add("| tool | calls | result tok | share |")
        add("|---|---:|---:|---:|")
        for k, v in tok.most_common(10):
            add(f"| {k} | {calls[k]} | {v / 1e3:.0f}k | {100 * v / total_tool_tok:.0f}% |")
        add("\n| command family | result tok |")
        add("|---|---:|")
        for k, v in fam.most_common(12):
            add(f"| `{k[:80]}` | {v / 1e3:.0f}k |")
        big = sorted(((x["tok"], x["call"], r["sid"]) for r in rs for x in r["largest_results"]), reverse=True)[:10]
        add("\n| largest single results | tok | session |")
        add("|---|---:|---|")
        for t, c, sid in big:
            add(f"| `{c[:110]}` | {t / 1e3:.0f}k | {sid} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--since", default="20260825")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = scan(Path(args.root), args.since)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcripts.json").write_text(json.dumps(rows, indent=1, default=str))
    text = summarize(rows)
    (out / "transcripts_summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
