# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn the census into fetch plans: token-relevant files only, never shell/env files."""

from __future__ import annotations

import gzip
import json
import re
import sys

from round_env import round_dir

SKIP = re.compile(r"\.(sh|env)$|(^|/)local\.yaml$|(^|/)\.env|credential|secret", re.I)
TIER1 = [
    re.compile(p)
    for p in (
        r"(^|/)manifest\.json$",
        r"(^|/)state\.json$",
        r"(^|/)session_breakdown\.json$",
        r"(^|/)research_hints\.md$",
        r"(^|/)lessons\.jsonl$",
        r"(^|/)llm_calls\.jsonl$",
        r"(^|/)decision_trace\.jsonl$",
        r"(^|/)specialist_intel\.jsonl$",
        r"(^|/)proposal_task_map\.jsonl$",
        r"(^|/)reports/kernel_optimization_summary\.json$",
        r"(^|/)reports/conc_sweep_summary\.json$",
        r"(^|/)agents/[^/]+/system_prompt[^/]*\.snapshot\.md$",
        r"(^|/)system_prompt[^/]*\.snapshot\.md$",
        r"(^|/)agents/orchestration/mcp_setup\.json$",
        r"(^|/)critic-session-memory/[^/]+/decisions\.jsonl$",
        r"(^|/)geak/(result|handoff)\.json$",
        r"(^|/)geak/e2e_cycle\d+/(kernel_journey|workflow_return)\.json$",
    )
]
TIER2 = [
    re.compile(p)
    for p in (
        r"(^|/)conversations\.jsonl$",
        r"(^|/)runs/specialist/[0-9a-f]{32}/process\.log$",
    )
]
TIER3 = [re.compile(p) for p in (r"(^|/)critic-workdir/\d+/(request|review|emit)\.json$",)]


def main() -> None:
    root = round_dir()
    recent = {n.strip() for n in open(root / "targets_recent.txt") if n.strip()}
    plans = {"tier1": [], "tier2": [], "tier3": []}
    with gzip.open(root / "01_census/ls.jsonl.gz", "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if row["status"] != "ready":
                continue
            for path, size, sha in row["files"]:
                if SKIP.search(path):
                    continue
                item = {"name": row["name"], "path": path, "bytes": size, "sha256": sha}
                if any(p.search(path) for p in TIER1):
                    plans["tier1"].append(item)
                elif any(p.search(path) for p in TIER2):
                    plans["tier2"].append(item)
                elif row["name"] in recent and any(p.search(path) for p in TIER3):
                    plans["tier3"].append(item)
    for tier, items in plans.items():
        with open(root / f"plan_{tier}.jsonl", "w") as out:
            for item in items:
                out.write(json.dumps(item) + "\n")
        print(
            tier,
            len(items),
            "files",
            round(sum(i["bytes"] or 0 for i in items) / 1e9, 2),
            "GB",
            len({i["name"] for i in items}),
            "sessions",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
