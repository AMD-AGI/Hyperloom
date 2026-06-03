"""Build a markdown deferred-work tracker from docstring_inventory.json."""
import json
import os

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "docstring_inventory.json"), encoding="utf-8") as fh:
    data = json.load(fh)

CORE_THIS_PASS = {
    os.path.normpath(p) for p in [
        "inference_optimizer/orchestrator/coordinator.py",
        "inference_optimizer/orchestrator/shared_state.py",
        "inference_optimizer/breakdown/collectors.py",
        "inference_optimizer/cli.py",
        "inference_optimizer/orchestrator/kernel_request_handlers.py",
        "inference_optimizer/session_paths.py",
        "inference_optimizer/orchestrator/policy.py",
        "inference_optimizer/orchestrator/phase_state.py",
        "inference_optimizer/orchestrator/objective.py",
        "inference_optimizer/orchestrator/action_executors/_grid_runner.py",
        "inference_optimizer/breakdown/schema.py",
        "inference_optimizer/paths.py",
    ]
}

reports = sorted(data["reports"], key=lambda r: len(r["missing"]), reverse=True)

lines = []
lines.append("# Docstring Deferred-Work Tracker")
lines.append("")
lines.append(f"Generated from `docstring_inventory.json`. Root: `{data['root']}`")
lines.append("")
lines.append(f"- Files with gaps: **{data['files_with_gaps']}**")
lines.append(f"- Total gaps: **{data['total_gaps']}**")
lines.append(f"- By kind: `{data['by_kind']}`")
lines.append("")
lines.append("Test files were excluded from the scan.")
lines.append("")
lines.append("## Pass 1 (this pass) - core inference_optimizer")
lines.append("")
lines.append("| Status | Gaps | File |")
lines.append("|---|---|---|")
for r in reports:
    if os.path.normpath(r["path"]) in CORE_THIS_PASS:
        lines.append(f"| [ ] | {len(r['missing'])} | `{r['path']}` |")
lines.append("")
lines.append("## Deferred to later passes")
lines.append("")
lines.append("Grouped by top-level component, sorted by gap count within each.")
lines.append("")

# group deferred by top-level dir
deferred = [r for r in reports if os.path.normpath(r["path"]) not in CORE_THIS_PASS]
groups = {}
for r in deferred:
    top = r["path"].replace("\\", "/").split("/")[0]
    groups.setdefault(top, []).append(r)

for top in sorted(groups, key=lambda g: -sum(len(r["missing"]) for r in groups[g])):
    grp = sorted(groups[top], key=lambda r: len(r["missing"]), reverse=True)
    total = sum(len(r["missing"]) for r in grp)
    lines.append(f"### `{top}/` - {total} gaps in {len(grp)} files")
    lines.append("")
    lines.append("| Gaps | File |")
    lines.append("|---|---|")
    for r in grp:
        lines.append(f"| {len(r['missing'])} | `{r['path']}` |")
    lines.append("")

with open(os.path.join(HERE, "DEFERRED_DOCSTRINGS.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("wrote DEFERRED_DOCSTRINGS.md")
print(f"deferred files: {len(deferred)}, deferred gaps: {sum(len(r['missing']) for r in deferred)}")
