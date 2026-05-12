#!/usr/bin/env python3
"""grep_transcript.py

Stream a Cursor transcript JSONL file and count tool_use blocks matching a list
of (tool_name_pattern, arg_regex) probes. Used by inspector SKILL.md Step S4
channel A (content audit).

Each transcript line is one of:
  {"role": "user", "message": {"content": [{"type":"text","text":"..."}]}}
  {"role": "assistant", "message": {"content": [
      {"type":"text","text":"..."},
      {"type":"tool_use","name":"Shell","input":{"command":"..."}}
  ]}}

A probe matches a tool_use block iff:
  * tool_use["name"] matches probe.tool_name_pattern (re.search, case-sensitive)
  * the JSON-serialised tool_use["input"] dict matches probe.arg_regex
    (re.search, case-sensitive, run against json.dumps(input, sort_keys=True))

Probes are read from a JSON file or stdin. Schema:
  [{"id": "run_baseline_sh", "tool_name_pattern": "Shell|run_terminal_cmd",
    "arg_regex": "run_baseline\\.sh", "min_count": 1}, ...]

Output (stdout, single JSON object):
  {"transcript_path": "...", "from_line": N, "to_line": M, "lines_scanned": K,
   "results": [{"id": "...", "count": 2, "sample_lines": [432, 510],
                "min_count": 1, "passes": true}, ...]}

Streaming, line-by-line, stdlib only. ~80 LOC.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _iter_tool_uses(line: str):
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for blk in content:
        if isinstance(blk, dict) and blk.get("type") == "tool_use":
            yield blk


def _compile_probes(probes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], re.Pattern, re.Pattern]]:
    out = []
    for p in probes:
        name_re = re.compile(p.get("tool_name_pattern", ".*"))
        arg_re = re.compile(p.get("arg_regex", ".*"))
        out.append((p, name_re, arg_re))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript", required=True, help="Path to transcript JSONL.")
    ap.add_argument("--from-line", type=int, default=1,
                    help="First line (1-indexed, inclusive) to scan.")
    ap.add_argument("--to-line", type=int, default=0,
                    help="Last line (1-indexed, inclusive). 0 = end of file.")
    ap.add_argument("--probes", default="-",
                    help="Path to probes JSON file, or '-' for stdin.")
    ap.add_argument("--max-samples", type=int, default=5,
                    help="Max sample line numbers per probe to record.")
    args = ap.parse_args()

    if args.probes == "-":
        probes_data = json.load(sys.stdin)
    else:
        with open(args.probes, "r", encoding="utf-8") as f:
            probes_data = json.load(f)
    if not isinstance(probes_data, list):
        print(json.dumps({"error": "probes_must_be_list"}))
        return 2

    compiled = _compile_probes(probes_data)
    counts = [0] * len(compiled)
    samples: list[list[int]] = [[] for _ in compiled]

    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(json.dumps({"error": "transcript_not_found", "path": str(transcript_path)}))
        return 2

    lines_scanned = 0
    last_line_seen = 0
    with transcript_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno < args.from_line:
                continue
            if args.to_line and lineno > args.to_line:
                break
            lines_scanned += 1
            last_line_seen = lineno
            for blk in _iter_tool_uses(line):
                name = blk.get("name", "") or ""
                inp = blk.get("input", {})
                try:
                    inp_blob = json.dumps(inp, sort_keys=True, ensure_ascii=False)
                except (TypeError, ValueError):
                    inp_blob = str(inp)
                for idx, (_p, name_re, arg_re) in enumerate(compiled):
                    if name_re.search(name) and arg_re.search(inp_blob):
                        counts[idx] += 1
                        if len(samples[idx]) < args.max_samples:
                            samples[idx].append(lineno)

    results = []
    for (probe, _nr, _ar), c, s in zip(compiled, counts, samples):
        min_count = int(probe.get("min_count", 1))
        results.append({
            "id": probe.get("id", "<unnamed>"),
            "count": c,
            "min_count": min_count,
            "passes": c >= min_count,
            "sample_lines": s,
        })

    out = {
        "transcript_path": str(transcript_path),
        "from_line": args.from_line,
        "to_line": args.to_line if args.to_line else last_line_seen,
        "lines_scanned": lines_scanned,
        "results": results,
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
