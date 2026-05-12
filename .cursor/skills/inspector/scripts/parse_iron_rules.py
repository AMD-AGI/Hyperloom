#!/usr/bin/env python3
"""parse_iron_rules.py

Extract `### IR-N: <title>` blocks from a target skill's SKILL.md and surface
each MUST/MUST NOT/MANDATORY/NEVER sentence as a candidate manifest entry.

Why: extraction-protocol.md previously read only phase-action `.md` files, so
SKILL.md-level Iron Rules (e.g. "Integration is MANDATORY") never made it into
the manifest. The agent was free to leave them off the audit.

Output (stdout, single JSON object):
  {
    "skill_md_path": "...",
    "skill_md_sha1": "<12 hex>",
    "iron_rules": [
       {
         "id": "IR-3",
         "title": "Integration (Phase 8) is MANDATORY",
         "section_lines": [50, 52],
         "applies_to_phases": ["DFS_LOOP_*", "SWEEP", "REPORT"],
         "candidates": [
            {
              "id_hint": "ir3_must_run_run_baseline_sh",
              "raw_text": "Re-baseline uses `run_baseline.sh`",
              "kind": "tool_call",
              "modality": "MUST",
              "anchor": "bash_script_backtick",
              "matched_token": "run_baseline.sh",
              "line": 52,
              "section_tag": "MANDATORY_SECTION",
              "iron_rule": true,
              "source_quote": "After GEAK returns ... MUST execute the integrate action ..."
            },
            ...
         ]
       },
       ...
    ]
  }

The downstream merger (parse_action_outputs.py --skill-md ...) selects only
candidates whose `applies_to_phases` matches the current phase.

Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

IR_HEADING_RE = re.compile(r"^###\s+(IR-\d+)\s*:\s*(.+?)\s*$")
NEXT_HEADING_RE = re.compile(r"^#{1,3}\s+\S")
MUST_SENTENCE_RE = re.compile(
    r"([^.\n]*\b(?:MUST(?:\s+NOT)?|MANDATORY|NEVER|REQUIRED|ALWAYS|"
    r"DO\s+NOT|DON'T)\b[^.\n]*\.?)",
    re.IGNORECASE,
)

# Modality classifier per sentence
NEGATIVE_MODALITY_RE = re.compile(
    r"\b(MUST\s+NOT|NEVER|DO\s+NOT|DON'T)\b", re.IGNORECASE)

# Reuse the same anchor patterns as parse_action_outputs.py.
BASH_SCRIPT_BACKTICK_RE = re.compile(r"`([^`]*\b[\w./\$\{\}-]+\.sh\b[^`]*)`")
PY_SCRIPT_BACKTICK_RE = re.compile(r"`([^`]*python3?\s+[^`]*\.py\b[^`]*)`")
GEAK_RE = re.compile(r"\b(geak_[a-z_]+)\b")
AGENT_MCP_RE = re.compile(r"\b(agent_[a-z_]+)\b")
BROWSER_MCP_RE = re.compile(r"\b(browser_[a-z_]+)\b")
ENV_PATH_RE = re.compile(
    r"(\$(?:RESULT_DIR|WORK_DIR|TRACE_DIR|SESSION_DIR|SKILL_ROOT|RESULTS_DIR|BASE_DIR)/[^\s`)\"']+)"
)
BACKTICK_FILE_RE = re.compile(r"`([^`]+\.(?:json|tsv|log|gz|csv|jsonl|md|py|sh))`")
ACTIONS_REF_RE = re.compile(r"`(actions/[\w./-]+\.md)`")

# Default phase mapping for IR-N when nothing more specific is detected.
# These are intentionally broad; the inspector's --phase invocation will narrow.
DEFAULT_APPLIES_TO_PHASES = ["*"]

# Specific overrides keyed by IR id (only when SKILL.md text strongly implies
# a phase). Keep this table small and audit-able.
IR_PHASE_HINTS = {
    "IR-1": ["DFS_LOOP_*", "BUILD_ACTION_STACK"],
    "IR-2": ["DFS_LOOP_*"],
    "IR-3": ["DFS_LOOP_*", "SWEEP", "REPORT"],
    "IR-4": ["BASELINE", "DFS_LOOP_*", "SWEEP"],
    "IR-5": ["BASELINE", "DFS_LOOP_*", "SWEEP"],
    "IR-6": ["DFS_LOOP_*"],
    "IR-7": ["DFS_LOOP_*"],
}


def _classify_modality(sentence: str) -> str:
    if NEGATIVE_MODALITY_RE.search(sentence):
        return "MUST"  # Prohibitions are still MUST-level commitments.
    return "MUST"  # All extracted sentences contain MUST/MANDATORY/etc.


def _find_anchors(sentence: str, line_no: int, section_tag: str) -> list[dict]:
    """Locate concrete script/MCP/path tokens inside a MUST sentence."""
    found: list[dict] = []
    for m in BASH_SCRIPT_BACKTICK_RE.finditer(sentence):
        found.append({"kind": "tool_call", "anchor": "bash_script_backtick",
                      "matched_token": m.group(1)})
    for m in PY_SCRIPT_BACKTICK_RE.finditer(sentence):
        found.append({"kind": "tool_call", "anchor": "python_script_backtick",
                      "matched_token": m.group(1)})
    for m in GEAK_RE.finditer(sentence):
        found.append({"kind": "tool_call", "anchor": "mcp_geak",
                      "matched_token": m.group(1)})
    for m in AGENT_MCP_RE.finditer(sentence):
        found.append({"kind": "tool_call", "anchor": "mcp_agent",
                      "matched_token": m.group(1)})
    for m in BROWSER_MCP_RE.finditer(sentence):
        found.append({"kind": "tool_call", "anchor": "mcp_browser",
                      "matched_token": m.group(1)})
    for m in ENV_PATH_RE.finditer(sentence):
        found.append({"kind": "artifact", "anchor": "env_path",
                      "matched_token": m.group(1)})
    for m in BACKTICK_FILE_RE.finditer(sentence):
        # Filter out re-mentions of `.sh` scripts already captured above
        tok = m.group(1)
        if tok.endswith(".sh") or tok.endswith(".py"):
            # Treat as tool_call only if not already matched as backtick script
            already = any(a["matched_token"] == tok for a in found)
            if already:
                continue
            found.append({"kind": "tool_call", "anchor": "script_filename",
                          "matched_token": tok})
        else:
            found.append({"kind": "artifact", "anchor": "backtick_suffix",
                          "matched_token": tok})
    for m in ACTIONS_REF_RE.finditer(sentence):
        found.append({"kind": "reference", "anchor": "action_ref",
                      "matched_token": m.group(1)})
    # Inject line_no / section_tag / id_hint
    out: list[dict] = []
    for f in found:
        token = f["matched_token"]
        slug = re.sub(r"[^a-z0-9]+", "_",
                      token.lower()).strip("_") or "anon"
        out.append({
            "id_hint": slug,
            "raw_text": token,
            "kind": f["kind"],
            "anchor": f["anchor"],
            "matched_token": token,
            "line": line_no,
            "section_tag": section_tag,
            "modality": "MUST",
            "iron_rule": True,
        })
    return out


def _split_ir_blocks(lines: list[str]) -> list[dict]:
    """Return a list of {id, title, start_line, end_line, body_lines}."""
    blocks: list[dict] = []
    current = None
    for i, line in enumerate(lines, start=1):
        m = IR_HEADING_RE.match(line)
        if m:
            if current is not None:
                current["end_line"] = i - 1
                blocks.append(current)
            current = {
                "id": m.group(1),
                "title": m.group(2).strip(),
                "start_line": i,
                "end_line": None,
                "body": [],
            }
            continue
        if current is not None and NEXT_HEADING_RE.match(line):
            current["end_line"] = i - 1
            blocks.append(current)
            current = None
            continue
        if current is not None:
            current["body"].append((i, line))
    if current is not None:
        current["end_line"] = len(lines)
        blocks.append(current)
    return blocks


def _extract_must_sentences(body: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Return list of (line_no, sentence_text)."""
    out: list[tuple[int, str]] = []
    for line_no, text in body:
        for m in MUST_SENTENCE_RE.finditer(text):
            out.append((line_no, m.group(1).strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skill-md", required=True,
                    help="Path to the target skill's SKILL.md")
    ap.add_argument("--include-mode-files", action="store_true",
                    help="Also scan modes/*.md siblings for IR-N blocks")
    args = ap.parse_args()

    skill_path = Path(args.skill_md)
    if not skill_path.is_file():
        print(json.dumps({"error": "skill_md_not_found",
                          "path": str(skill_path)}))
        return 2

    paths: list[Path] = [skill_path]
    if args.include_mode_files:
        modes_dir = skill_path.parent / "modes"
        if modes_dir.is_dir():
            paths.extend(sorted(modes_dir.glob("*.md")))

    all_iron_rules = []
    sha1 = hashlib.sha1()
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        sha1.update(text.encode("utf-8"))
        lines = text.splitlines()
        for block in _split_ir_blocks(lines):
            ir_id = block["id"]
            # Step 1: harvest concrete anchors from the entire IR body, since
            # the whole `### IR-N` section is mandatory by construction.
            candidates: list[dict] = []
            seen_tokens: set[tuple[str, str]] = set()
            title_quote = f"{ir_id}: {block['title']}"
            for line_no, text in block["body"]:
                # Use the line itself as source_quote context.
                anchors = _find_anchors(text, line_no, "MANDATORY_SECTION")
                for a in anchors:
                    key = (a["anchor"], a["matched_token"])
                    if key in seen_tokens:
                        continue
                    seen_tokens.add(key)
                    a["source_quote"] = (title_quote + " | " + text.strip())[:240]
                    candidates.append(a)
            # Step 2: also harvest explicit MUST/MANDATORY/NEVER sentences as
            # policy candidates so that prose-only iron rules still appear.
            sentences = _extract_must_sentences(block["body"])
            if not sentences and not candidates:
                sentences = [(block["start_line"], block["title"])]
            for line_no, sent in sentences:
                slug = re.sub(r"[^a-z0-9]+", "_",
                              sent.lower())[:60].strip("_") or "policy"
                cand = {
                    "id_hint": f"{ir_id.lower().replace('-', '_')}_{slug}",
                    "raw_text": sent,
                    "kind": "policy",
                    "anchor": "must_sentence",
                    "matched_token": "",
                    "line": line_no,
                    "section_tag": "MANDATORY_SECTION",
                    "modality": _classify_modality(sent),
                    "iron_rule": True,
                    "source_quote": (title_quote + " | " + sent)[:240],
                }
                candidates.append(cand)
            all_iron_rules.append({
                "id": ir_id,
                "title": block["title"],
                "source_file": str(p),
                "section_lines": [block["start_line"], block["end_line"]],
                "applies_to_phases": IR_PHASE_HINTS.get(
                    ir_id, DEFAULT_APPLIES_TO_PHASES),
                "candidates": candidates,
            })

    out = {
        "skill_md_path": str(skill_path),
        "skill_md_sha1": sha1.hexdigest()[:12],
        "scanned_files": [str(p) for p in paths],
        "iron_rules": all_iron_rules,
    }
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
