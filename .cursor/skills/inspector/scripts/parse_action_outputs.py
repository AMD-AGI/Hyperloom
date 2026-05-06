#!/usr/bin/env python3
"""parse_action_outputs.py

Mechanical regex extraction from one or more action markdown files (and,
optionally, the target skill's SKILL.md for Iron-Rule intake). Implements
passes 0 (iron rules), 1 and 2 of extraction-protocol.md. The output is a JSON
proposal of candidates that the inspector's pass 3 (LLM classification)
processes one item at a time.

This script intentionally has no LLM judgment. Its purpose is to provide a
non-LLM anchor so the inspector cannot silently miss an obviously expected
artifact or tool call. The inspector's S5 step diffs the final manifest against
this proposal and reports `regex_anchors_diff_summary`.

Output (stdout, single JSON object):
  {
    "action_md_paths": ["actions/kernel-opt.md", "actions/integrate.md"],
    "action_md_sha1": "<first 12 hex chars of concatenated bytes>",
    "per_action": [{"path": "...", "sha1": "...",
                    "section_map": [...]}, ...],
    "skill_md_path": "SKILL.md" | null,
    "iron_rules": [{...}, ...]  // raw output of parse_iron_rules.py, if any
    "expected_tool_calls_candidates": [
        {"id_hint": "shell_run_baseline", "raw_text": "bash run_baseline.sh",
         "anchor": "bash_script", "line": 42, "section_tag": "PROCEDURE_SECTION",
         "source_file": "actions/baseline.md",
         "iron_rule": false},
        ...
    ],
    "expected_artifacts_candidates": [
        {"id_hint": "result_dir_baseline_json", "raw_text": "$RESULT_DIR/baseline_*.json",
         "anchor": "env_path", "line": 88, "section_tag": "OUTPUT_SECTION",
         "source_file": "actions/baseline.md", "iron_rule": false},
        ...
    ],
    "expected_state_assertions_candidates": [
        {"id_hint": "state_baseline_accuracy", "raw_text": "baseline_accuracy",
         "anchor": "set_keyword", "line": 101, "section_tag": "STATE_SECTION",
         "source_file": "actions/baseline.md", "iron_rule": false},
        ...
    ]
  }

Multiple --action paths: candidates from each are tagged with `source_file`.
Duplicates by `(bucket, anchor, raw_text)` are de-duplicated, keeping the
first occurrence and aggregating `source_files` into a list.

--skill-md: shells out to parse_iron_rules.py (same directory) and merges
its candidates as expected_tool_calls / expected_artifacts entries with
`iron_rule=true`. Iron-rule candidates are filtered by `--phase` if given.

Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# ---------- Section tagging ----------
SECTION_TAGS = [
    ("OUTPUT_SECTION", re.compile(r"\b(outputs?|produces?|writes?|emits?|artifacts?)\b", re.I)),
    ("MANDATORY_SECTION", re.compile(r"\b(iron\s*rule|mandatory|must)\b", re.I)),
    ("PROCEDURE_SECTION", re.compile(r"\b(procedure|steps?|how\s+to|execute|run)\b", re.I)),
    ("INPUT_SECTION", re.compile(r"\b(inputs?|prereq|requires?)\b", re.I)),
    ("STATE_SECTION", re.compile(r"\b(state|sets?|populates?|fields?)\b", re.I)),
]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# ---------- Tool-call anchors (pass 2a) ----------
# Backticked references (anywhere)
BASH_SCRIPT_BACKTICK_RE = re.compile(r"`([^`]*\b[\w./\$\{\}-]+\.sh\b[^`]*)`")
PY_SCRIPT_BACKTICK_RE = re.compile(r"`([^`]*python3?\s+[^`]*\.py\b[^`]*)`")
CURSOR_TOOL_RE = re.compile(r"`(Read|Write|Edit|Grep|Glob|Shell|Task|WebFetch|WebSearch)\b")
CURL_BACKTICK_RE = re.compile(r"`([^`]*\bcurl\s+[^`]*)`")
# Bare script invocations (typically inside fenced bash blocks)
BASH_SCRIPT_BARE_RE = re.compile(r"\b(?:bash|sh)\s+[\"']?([\w./\$\{\}-]+\.sh)[\"']?")
PY_SCRIPT_BARE_RE = re.compile(r"\bpython3?\s+[\"']?([\w./\$\{\}-]+\.py)[\"']?")
CURL_BARE_RE = re.compile(r"^[\s|&]*(curl\s+\S[^|&;]*)")
# MCP tool names (anywhere)
GEAK_RE = re.compile(r"\b(geak_[a-z_]+)\b")
AGENT_MCP_RE = re.compile(r"\b(agent_[a-z_]+)\b")
BROWSER_MCP_RE = re.compile(r"\b(browser_[a-z_]+)\b")
# Fence tracking
FENCE_BASH_OPEN_RE = re.compile(r"^```(?:bash|sh|shell)\s*$")
FENCE_END_RE = re.compile(r"^```\s*$")

# ---------- Artifact anchors (pass 2b) ----------
ENV_PATH_RE = re.compile(r"(\$(?:RESULT_DIR|WORK_DIR|TRACE_DIR|SESSION_DIR|SKILL_ROOT|RESULTS_DIR|BASE_DIR)/[^\s`)\"']+)")
BACKTICK_OUTPUT_RE = re.compile(r"`([^`]+\.(?:json|tsv|log|gz|bak|csv|xlsx|env|jsonl))`")
GLOB_OUTPUT_RE = re.compile(r"`([^`]*\*[^`]*\.(?:json|tsv|log|gz))`")
TOUCH_MKDIR_RE = re.compile(r"`(?:touch|mkdir\s+-p)\s+([^\s`]+)`")

# ---------- State-assertion anchors (pass 2c) ----------
SET_RE = re.compile(r"\bSet\s+`([a-z_][a-z0-9_]*)`")
POPULATE_RE = re.compile(r"\bPopulates?\s+`([a-z_][a-z0-9_]*)`")
UPDATE_RE = re.compile(r"\bUpdates?\s+`([a-z_][a-z0-9_]*)`")
STATE_DOT_RE = re.compile(r"\bstate\.([a-z_][a-z0-9_]*)\s*=")


def _tag_heading(heading_text: str) -> str:
    for tag, rx in SECTION_TAGS:
        if rx.search(heading_text):
            return tag
    return "OTHER_SECTION"


def _build_section_map(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Returns (section_map, per_line_section_tag)."""
    section_map: list[dict] = []
    per_line: list[str] = ["OTHER_SECTION"] * (len(lines) + 1)
    current_tag = "OTHER_SECTION"
    for i, line in enumerate(lines, start=1):
        m = HEADING_RE.match(line)
        if m:
            depth = len(m.group(1))
            heading = m.group(2)
            current_tag = _tag_heading(heading)
            section_map.append({
                "line": i, "depth": depth, "heading": heading, "tag": current_tag,
            })
        per_line[i] = current_tag
    return section_map, per_line


def _scan_tool_calls(lines: list[str], per_line: list[str]) -> list[dict]:
    out: list[dict] = []
    in_bash_fence = False
    for i, line in enumerate(lines, start=1):
        tag = per_line[i]
        if FENCE_BASH_OPEN_RE.match(line):
            in_bash_fence = True
            continue
        if FENCE_END_RE.match(line):
            in_bash_fence = False
            continue
        # Backticked anchors apply everywhere
        for m in BASH_SCRIPT_BACKTICK_RE.finditer(line):
            out.append({"id_hint": "bash_script", "raw_text": m.group(1),
                        "anchor": "bash_script_backtick", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in PY_SCRIPT_BACKTICK_RE.finditer(line):
            out.append({"id_hint": "python_script", "raw_text": m.group(1),
                        "anchor": "python_script_backtick", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in CURL_BACKTICK_RE.finditer(line):
            out.append({"id_hint": "curl_cmd", "raw_text": m.group(1),
                        "anchor": "curl_backtick", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        # Bare anchors (mostly trigger inside fences; harmless outside since
        # these patterns require a "bash"/"python" verb to precede the script)
        for m in BASH_SCRIPT_BARE_RE.finditer(line):
            out.append({"id_hint": "bash_script", "raw_text": m.group(1),
                        "anchor": "bash_script_bare", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in PY_SCRIPT_BARE_RE.finditer(line):
            out.append({"id_hint": "python_script", "raw_text": m.group(1),
                        "anchor": "python_script_bare", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        if in_bash_fence:
            for m in CURL_BARE_RE.finditer(line):
                out.append({"id_hint": "curl_cmd", "raw_text": m.group(1).strip(),
                            "anchor": "curl_bare", "line": i,
                            "section_tag": tag, "in_bash_fence": True})
        # MCP tool names and Cursor tool names
        for m in GEAK_RE.finditer(line):
            out.append({"id_hint": f"mcp_{m.group(1)}", "raw_text": m.group(1),
                        "anchor": "mcp_geak", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in AGENT_MCP_RE.finditer(line):
            out.append({"id_hint": f"mcp_{m.group(1)}", "raw_text": m.group(1),
                        "anchor": "mcp_agent", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in BROWSER_MCP_RE.finditer(line):
            out.append({"id_hint": f"mcp_{m.group(1)}", "raw_text": m.group(1),
                        "anchor": "mcp_browser", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
        for m in CURSOR_TOOL_RE.finditer(line):
            out.append({"id_hint": f"tool_{m.group(1)}", "raw_text": m.group(1),
                        "anchor": "cursor_tool", "line": i,
                        "section_tag": tag, "in_bash_fence": in_bash_fence})
    return out


def _scan_artifacts(lines: list[str], per_line: list[str]) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(lines, start=1):
        tag = per_line[i]
        for m in ENV_PATH_RE.finditer(line):
            out.append({"id_hint": "env_path", "raw_text": m.group(1),
                        "anchor": "env_path", "line": i, "section_tag": tag})
        for m in BACKTICK_OUTPUT_RE.finditer(line):
            out.append({"id_hint": "backtick_output", "raw_text": m.group(1),
                        "anchor": "backtick_suffix", "line": i, "section_tag": tag})
        for m in GLOB_OUTPUT_RE.finditer(line):
            out.append({"id_hint": "glob_output", "raw_text": m.group(1),
                        "anchor": "glob_suffix", "line": i, "section_tag": tag})
        for m in TOUCH_MKDIR_RE.finditer(line):
            out.append({"id_hint": "touch_mkdir", "raw_text": m.group(1),
                        "anchor": "touch_mkdir", "line": i, "section_tag": tag})
    return out


def _scan_state(lines: list[str], per_line: list[str]) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(lines, start=1):
        tag = per_line[i]
        for rx, anchor in (
            (SET_RE, "set_keyword"),
            (POPULATE_RE, "populate_keyword"),
            (UPDATE_RE, "update_keyword"),
            (STATE_DOT_RE, "state_dot_assign"),
        ):
            for m in rx.finditer(line):
                out.append({"id_hint": f"state_{m.group(1)}", "raw_text": m.group(1),
                            "anchor": anchor, "line": i, "section_tag": tag})
    return out


def _scan_one_action(path: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sha1_short = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    lines = text.splitlines()
    section_map, per_line = _build_section_map(lines)
    tcs = _scan_tool_calls(lines, per_line)
    arts = _scan_artifacts(lines, per_line)
    states = _scan_state(lines, per_line)
    src = str(path)
    for c in tcs + arts + states:
        c["source_file"] = src
        c.setdefault("iron_rule", False)
    return ({"path": src, "sha1": sha1_short, "section_map": section_map},
            tcs, arts, states)


def _phase_matches_glob(phase: str, patterns: list[str]) -> bool:
    import fnmatch
    if not patterns or "*" in patterns:
        return True
    return any(fnmatch.fnmatchcase(phase, p) for p in patterns)


def _merge_iron_rules(skill_md_path: Path, phase: str | None,
                      include_mode_files: bool) -> tuple[dict, list[dict],
                                                         list[dict]]:
    """Shell out to parse_iron_rules.py (sibling script). Returns
    (raw_iron_rules_blob, tool_call_candidates, artifact_candidates).
    State assertions are not derived from iron rules.
    """
    import subprocess
    sibling = Path(__file__).parent / "parse_iron_rules.py"
    if not sibling.is_file():
        return ({"error": "parse_iron_rules.py_not_found"}, [], [])
    cmd = [sys.executable, str(sibling), "--skill-md", str(skill_md_path)]
    if include_mode_files:
        cmd.append("--include-mode-files")
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        blob = json.loads(proc.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        return ({"error": f"iron_rules_parse_failed: {e}"}, [], [])

    tool_calls: list[dict] = []
    artifacts: list[dict] = []
    for ir in blob.get("iron_rules", []):
        applies = ir.get("applies_to_phases") or ["*"]
        if phase and not _phase_matches_glob(phase, applies):
            continue
        for cand in ir.get("candidates", []):
            entry = {
                "id_hint": cand["id_hint"],
                "raw_text": cand["raw_text"],
                "anchor": cand["anchor"],
                "line": cand["line"],
                "section_tag": cand.get("section_tag", "MANDATORY_SECTION"),
                "source_file": ir.get("source_file", str(skill_md_path)),
                "source_quote": cand.get("source_quote", ""),
                "modality": cand.get("modality", "MUST"),
                "iron_rule": True,
                "ir_id": ir["id"],
            }
            kind = cand.get("kind", "policy")
            if kind in ("tool_call", "policy"):
                tool_calls.append(entry)
            elif kind == "artifact":
                artifacts.append(entry)
            elif kind == "reference":
                # action_ref like `actions/integrate.md` — surface as a
                # tool-call expectation that the agent must "consult / execute"
                # the referenced action. The downstream LLM pass will classify.
                tool_calls.append(entry)
    return blob, tool_calls, artifacts


def _dedupe(candidates: list[dict]) -> list[dict]:
    """De-dupe by (anchor, raw_text), aggregating `source_files`."""
    seen: dict[tuple[str, str], dict] = {}
    out: list[dict] = []
    for c in candidates:
        key = (c.get("anchor", ""), c.get("raw_text", ""))
        if key in seen:
            existing = seen[key]
            srcs = existing.setdefault("source_files", [existing.get("source_file", "")])
            if c.get("source_file") and c["source_file"] not in srcs:
                srcs.append(c["source_file"])
            # Iron-rule wins over non-iron-rule
            if c.get("iron_rule") and not existing.get("iron_rule"):
                existing["iron_rule"] = True
                existing["ir_id"] = c.get("ir_id", existing.get("ir_id", ""))
                if c.get("source_quote"):
                    existing["source_quote"] = c["source_quote"]
            continue
        seen[key] = c
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", action="append", required=True,
                    help=("Path to an action .md file. May be passed multiple "
                          "times to audit a phase that mandates more than one "
                          "action (e.g. DFS_LOOP_<N> needs both kernel-opt.md "
                          "and integrate.md per IR-3)."))
    ap.add_argument("--skill-md", default=None,
                    help=("(Optional) Path to the target skill's SKILL.md. "
                          "Triggers Iron-Rule intake via parse_iron_rules.py. "
                          "Required by extraction-protocol.md Pass 0."))
    ap.add_argument("--include-mode-files", action="store_true",
                    help="Forwarded to parse_iron_rules.py")
    ap.add_argument("--phase", default=None,
                    help=("Symbolic phase name (e.g. DFS_LOOP_3, BASELINE). "
                          "Used to filter iron-rule candidates by their "
                          "applies_to_phases. If omitted, all iron rules "
                          "are merged."))
    ap.add_argument("--env-json", default=None,
                    help="(Optional) Path to JSON file with current RUN_ENV. Not used "
                         "by extraction (which is env-agnostic) but echoed back into "
                         "the output for downstream resolvers.")
    args = ap.parse_args()

    per_action_meta: list[dict] = []
    all_tcs: list[dict] = []
    all_arts: list[dict] = []
    all_states: list[dict] = []
    combined_sha = hashlib.sha1()

    for action_arg in args.action:
        action_path = Path(action_arg)
        if not action_path.is_file():
            print(json.dumps({"error": "action_md_not_found",
                              "path": str(action_path)}))
            return 2
        meta, tcs, arts, states = _scan_one_action(action_path)
        per_action_meta.append(meta)
        all_tcs.extend(tcs)
        all_arts.extend(arts)
        all_states.extend(states)
        combined_sha.update(action_path.read_bytes())

    iron_blob = None
    if args.skill_md:
        skill_path = Path(args.skill_md)
        if not skill_path.is_file():
            print(json.dumps({"error": "skill_md_not_found",
                              "path": str(skill_path)}))
            return 2
        iron_blob, ir_tcs, ir_arts = _merge_iron_rules(
            skill_path, args.phase, args.include_mode_files)
        all_tcs.extend(ir_tcs)
        all_arts.extend(ir_arts)
        combined_sha.update(skill_path.read_bytes())

    out = {
        "action_md_paths": [m["path"] for m in per_action_meta],
        "action_md_sha1": combined_sha.hexdigest()[:12],
        "per_action": per_action_meta,
        "skill_md_path": args.skill_md,
        "phase": args.phase,
        "iron_rules": iron_blob,
        "expected_tool_calls_candidates": _dedupe(all_tcs),
        "expected_artifacts_candidates": _dedupe(all_arts),
        "expected_state_assertions_candidates": _dedupe(all_states),
    }

    # Backward-compat fields for callers that only expect the single-action
    # shape (extraction-protocol.md Pass 4 readers and tests).
    if len(per_action_meta) == 1:
        out["action_md_path"] = per_action_meta[0]["path"]
        out["section_map"] = per_action_meta[0]["section_map"]

    if args.env_json:
        try:
            with open(args.env_json, "r", encoding="utf-8") as f:
                out["run_env_echo"] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            out["run_env_echo_error"] = str(e)

    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
