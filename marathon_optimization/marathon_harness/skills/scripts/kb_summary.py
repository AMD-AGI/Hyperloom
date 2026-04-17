#!/usr/bin/env python3
"""Generate model-specific playbooks from KB entries for prompt injection.

Usage:
    python3 kb_summary.py --model MODEL_NAME --kb-path KB_PATH [--output OUTPUT_PATH]

Generates a concise markdown playbook that can be injected into the agent's prompt
at KB warm-up time (Step 4). The playbook summarizes:
- What worked on this model (sorted by gain)
- What failed (to avoid repeating)
- Untested strategies (to prioritize)
- Recommended launch config
- Known pitfalls
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_entries(kb_path: str) -> list[dict]:
    entries = []
    with open(kb_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def normalize_model_name(name: str) -> str:
    """Normalize model names for matching (case-insensitive, strip paths)."""
    name = name.lower().strip()
    name = name.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def match_model(entry_model: str, target: str) -> bool:
    """Check if an entry's model matches the target model."""
    if not entry_model:
        return False
    return normalize_model_name(entry_model) == normalize_model_name(target)


def generate_playbook(model: str, entries: list[dict]) -> str:
    """Generate a concise playbook for a specific model."""
    model_entries = [e for e in entries if match_model(e.get("model", ""), model)]
    cross_model = [e for e in entries if not e.get("model") and e.get("confidence", 0) >= 0.7]

    if not model_entries and not cross_model:
        return f"# Playbook: {model}\n\nNo KB entries found. This is a fresh model — explore all strategies.\n"

    sections = [f"# Playbook: {model}\n"]
    sections.append(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                    f"from {len(model_entries)} model-specific + {len(cross_model)} cross-model entries.*\n")

    wins = sorted(
        [e for e in model_entries if e.get("result", {}).get("status") in ("keep", "KEEP")],
        key=lambda e: e.get("result", {}).get("gain_pct", 0),
        reverse=True,
    )
    if wins:
        sections.append("## What Worked (by gain)")
        for e in wins[:10]:
            gain = e.get("result", {}).get("gain_pct", "?")
            sections.append(f"- **+{gain}%**: {e.get('action', '?')} — {e.get('lesson', '')[:100]}")
        sections.append("")

    fails = [e for e in model_entries
             if e.get("result", {}).get("status") in ("revert", "REVERT", "discard", "DISCARD", "crash")]
    if fails:
        sections.append("## What Failed (avoid repeating)")
        for e in fails[:10]:
            sections.append(f"- {e.get('action', '?')}: {e.get('lesson', '')[:100]}")
        sections.append("")

    all_tags = set()
    for e in model_entries:
        all_tags.update(e.get("tags", []))

    untested = []
    for strategy in ["A", "B", "C", "D", "E", "F"]:
        if f"strategy-{strategy.lower()}-untested" in all_tags:
            untested.append(strategy)
    if "call-stack-untested" in all_tags:
        untested.extend(["D", "E", "F"])
    untested = sorted(set(untested))

    if untested:
        sections.append("## Untested Strategies (PRIORITY)")
        for s in untested:
            labels = {"A": "Full torch.compile", "B": "Framework Triton",
                      "C": "Selective compile", "D": "Call stack patching",
                      "E": "Framework scheduling", "F": "Kernel fusion"}
            sections.append(f"- **Strategy {s}** ({labels.get(s, '?')}): not yet attempted")
        sections.append("")

    pitfalls = [e for e in model_entries if e.get("category") == "pitfall"]
    if pitfalls:
        sections.append("## Known Pitfalls")
        for e in pitfalls[:5]:
            sections.append(f"- {e.get('lesson', '')[:150]}")
        sections.append("")

    if cross_model:
        sections.append("## Cross-Model Insights")
        for e in sorted(cross_model, key=lambda e: e.get("confidence", 0), reverse=True)[:5]:
            sections.append(f"- [{e.get('confidence', '?')}] {e.get('lesson', '')[:150]}")
        sections.append("")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Generate model-specific playbook from KB")
    parser.add_argument("--model", required=True, help="Model name to generate playbook for")
    parser.add_argument("--kb-path", required=True, help="Path to KB entries.jsonl")
    parser.add_argument("--output", default="", help="Output path (default: stdout)")
    args = parser.parse_args()

    entries = load_entries(args.kb_path)
    playbook = generate_playbook(args.model, entries)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        dir_path = os.path.dirname(os.path.abspath(args.output))
        fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".md.tmp")
        with os.fdopen(fd, "w") as f:
            f.write(playbook)
        os.replace(tmp, args.output)
        print(f"Playbook written to {args.output}")
    else:
        print(playbook)


if __name__ == "__main__":
    main()
