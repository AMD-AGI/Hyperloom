#!/usr/bin/env python3
"""compute_verdict.py

Deterministic verdict computation for inspector audits. This script is the
single point that converts (manifest + observations + semantic_rules) into the
final {PASS, WARN, BLOCK, FATAL} verdict. It exists because earlier inspector
runs let the agent write the verdict in free-form prose, which created the
"BLOCK silently downgraded to WARN with a justification sentence" failure
mode (see 2026-04-21 Qwen3-30B-A3B run).

The agent MUST NOT post-edit this script's output. If a verdict is wrong, the
remedy is to update semantic_rules.json or the extraction protocol, not the
audit reply.

Inputs (all JSON files):
  --manifest        Output of extraction-protocol.md (expected_tool_calls,
                    expected_artifacts, expected_state_assertions, each entry
                    has id, modality, source_lines, source_quote, etc.)
  --observations    Pure-fact dump produced by the inspector agent in S5a:
                    {"tool_call_observations": [{"id": ..., "count": N,
                        "sample_lines": [...]}, ...],
                     "artifact_observations":  [{"id": ..., "exists": bool,
                        "bytes": int, "json_fields": {...}, "error": str?}, ...],
                     "state_observations":     [{"id": ..., "value": any,
                        "recovered": bool}, ...],
                     "transcript":             {"path": "...", "lines": int}}
                    Forbidden top-level keys (script aborts if seen):
                    "verdict", "severity", "violations", "passes", "because",
                    "deferred", "acceptable".
  --semantic-rules  Declarative rule pack (semantic_rules.json schema below).
  --phase           Symbolic phase name. Used to evaluate rule.when.phase_in.
  --phase-action-files  Comma-separated list, used only for echoing into output.
  --target-skill-dir    Path, used only for echoing into output.

Output (stdout, single JSON object) matches audit-report-schema.md v1.1:
  {
    "schema_version": "1.1",
    "inspector_version": "1.1",
    "manifest_version": "1.1",
    "verdict_source": "compute_verdict.py@<sha1>",
    "phase": "...",
    "verdict": "PASS|WARN|BLOCK|FATAL",
    "verdict_summary": "...",
    "passes": [...],
    "violations": [...],
    "unverified": [...]
  }

Stdlib only.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.1"
INSPECTOR_VERSION = "1.1"
MANIFEST_VERSION = "1.1"

# Severity ladder — must match remediation-protocol.md §1.
SEVERITY_RANK = {"info": 0, "warn": 1, "block": 2, "fatal": 3}
RANK_TO_VERDICT = {0: "PASS", 1: "WARN", 2: "BLOCK", 3: "FATAL"}

# Modality × outcome → severity, per remediation-protocol.md §2.
# Iron-rule MUST upgrades to fatal when source_quote contains the trigger
# tokens documented in remediation-protocol.md §2 Iron-Rule trigger.
IRON_RULE_TOKENS = re.compile(
    r"(Violation\s*=\s*invalidation|Iron\s*Rule|MUST\s*NOT|"
    r"mandatory.{0,30}invalid|\bIR-\d+\b)",
    re.IGNORECASE,
)

# Tokens forbidden in the observations file. The script aborts if any of these
# appears as a top-level key (case-insensitive). This is a structural defense
# against the agent putting a verdict-y narrative into the observations file.
FORBIDDEN_OBS_KEYS = {
    "verdict", "severity", "violations", "passes",
    "because", "deferred", "acceptable", "warn", "block", "fatal",
}


def _read_json(path: str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"compute_verdict: input not found: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"compute_verdict: invalid JSON in {path}: {e}")


def _validate_observations(obs: dict) -> None:
    if not isinstance(obs, dict):
        raise SystemExit("compute_verdict: observations must be a JSON object")
    bad = sorted(k for k in obs.keys() if k.lower() in FORBIDDEN_OBS_KEYS)
    if bad:
        raise SystemExit(
            "compute_verdict: observations file contains forbidden "
            f"verdict-y keys at top level: {bad}. Observations must be pure "
            "facts; verdict computation belongs to compute_verdict.py."
        )
    for required in ("tool_call_observations", "artifact_observations",
                     "state_observations", "transcript"):
        if required not in obs:
            obs[required] = [] if required != "transcript" else {}


def _classify_modality(entry: dict) -> tuple[str, bool]:
    """Return (modality, iron_rule_flag)."""
    modality = (entry.get("modality") or "UNVERIFIED").upper()
    quote = entry.get("source_quote") or ""
    section = entry.get("section_tag") or entry.get("source_section") or ""
    iron = bool(
        IRON_RULE_TOKENS.search(quote)
        and re.search(r"iron|mandatory", str(section), re.IGNORECASE)
    ) or entry.get("iron_rule") is True
    return modality, iron


def _outcome_to_severity(modality: str, passed: bool, unresolvable: bool,
                         iron: bool) -> str | None:
    """None means 'no violation'; record in passes[] / unverified[] instead."""
    if unresolvable:
        return None
    if passed:
        return None
    if modality == "MUST":
        return "fatal" if iron else "block"
    if modality == "SHOULD":
        return "warn"
    if modality == "MAY":
        return "info"
    return None  # UNVERIFIED → record in unverified[], not violation


def _audit_tool_calls(manifest: dict, obs: dict) -> tuple[list, list, list]:
    passes, violations, unverified = [], [], []
    obs_by_id = {o["id"]: o for o in obs.get("tool_call_observations", [])
                 if isinstance(o, dict) and "id" in o}
    for entry in manifest.get("expected_tool_calls", []):
        eid = entry["id"]
        modality, iron = _classify_modality(entry)
        observation = obs_by_id.get(eid)
        if observation is None:
            unverified.append({
                "id": eid, "channel": "content",
                "expected_arg_regex": entry.get("arg_regex", ""),
                "modality_in_manifest": modality,
                "source_lines": entry.get("source_lines", []),
                "source_quote": entry.get("source_quote", ""),
                "reason": "no_observation_recorded",
            })
            continue
        count = int(observation.get("count", 0))
        min_count = int(entry.get("min_count", 1))
        passed = count >= min_count
        sev = _outcome_to_severity(modality, passed, False, iron)
        item_base = {
            "id": eid, "channel": "content",
            "expected_arg_regex": entry.get("arg_regex", ""),
            "observed_count": count,
            "modality_in_manifest": modality,
            "source_lines": entry.get("source_lines", []),
            "source_quote": entry.get("source_quote", ""),
        }
        if passed:
            passes.append(item_base)
        elif sev is None:
            unverified.append({**item_base, "reason": "modality_unverified"})
        else:
            violations.append({
                **item_base, "severity": sev,
                "observed": str(count),
                "remediation": entry.get(
                    "remediation",
                    f'Re-perform the step described at '
                    f'{entry.get("source_file","action.md")}:'
                    f'{entry.get("source_lines",[None])[0]}.'),
            })
    return passes, violations, unverified


def _audit_artifacts(manifest: dict, obs: dict) -> tuple[list, list, list]:
    passes, violations, unverified = [], [], []
    obs_by_id = {o["id"]: o for o in obs.get("artifact_observations", [])
                 if isinstance(o, dict) and "id" in o}
    for entry in manifest.get("expected_artifacts", []):
        eid = entry["id"]
        modality, iron = _classify_modality(entry)
        observation = obs_by_id.get(eid)
        if entry.get("unresolvable_env_var"):
            unverified.append({
                "id": eid, "channel": "file",
                "expected": entry.get("path_template", ""),
                "modality_in_manifest": modality,
                "source_lines": entry.get("source_lines", []),
                "source_quote": entry.get("source_quote", ""),
                "reason": "unresolvable_env_var",
            })
            continue
        if observation is None:
            unverified.append({
                "id": eid, "channel": "file",
                "expected": entry.get("path_template", ""),
                "modality_in_manifest": modality,
                "source_lines": entry.get("source_lines", []),
                "source_quote": entry.get("source_quote", ""),
                "reason": "no_observation_recorded",
            })
            continue
        exists = bool(observation.get("exists"))
        nonempty = int(observation.get("bytes", 0)) > 0
        must_be_nonempty = bool(entry.get("must_be_nonempty", False))
        passed = exists and (nonempty or not must_be_nonempty)
        sev = _outcome_to_severity(modality, passed, False, iron)
        item_base = {
            "id": eid, "channel": "file",
            "expected": entry.get("path_template", ""),
            "expected_resolved": entry.get("resolved_path", ""),
            "observed": "exists" if exists else "not_found",
            "observed_bytes": int(observation.get("bytes", 0)),
            "modality_in_manifest": modality,
            "source_lines": entry.get("source_lines", []),
            "source_quote": entry.get("source_quote", ""),
        }
        if passed:
            passes.append(item_base)
        elif sev is None:
            unverified.append({**item_base, "reason": "modality_unverified"})
        else:
            violations.append({
                **item_base, "severity": sev,
                "remediation": entry.get(
                    "remediation",
                    f'Produce {entry.get("path_template","")} per '
                    f'{entry.get("source_file","action.md")}:'
                    f'{(entry.get("source_lines",[None]) or [None])[0]}.'),
            })
    return passes, violations, unverified


def _audit_state(manifest: dict, obs: dict) -> tuple[list, list, list]:
    passes, violations, unverified = [], [], []
    obs_by_id = {o["id"]: o for o in obs.get("state_observations", [])
                 if isinstance(o, dict) and "id" in o}
    for entry in manifest.get("expected_state_assertions", []):
        eid = entry["id"]
        modality, _iron = _classify_modality(entry)
        observation = obs_by_id.get(eid)
        if observation is None or not observation.get("recovered"):
            unverified.append({
                "id": eid, "channel": "state",
                "expected": entry.get("assertion", ""),
                "field": entry.get("field", ""),
                "modality_in_manifest": modality,
                "source_lines": entry.get("source_lines", []),
                "source_quote": entry.get("source_quote", ""),
                "reason": "state_not_recoverable_from_transcript",
            })
            continue
        value = observation.get("value")
        assertion = entry.get("assertion", "is_set_and_numeric")
        passed = _eval_state_assertion(assertion, value)
        # State assertion failures never escalate to fatal — per
        # remediation-protocol.md §2 Iron-Rule trigger condition 3.
        sev = _outcome_to_severity(modality, passed, False, iron=False)
        base = {
            "id": eid, "channel": "state",
            "expected": assertion, "field": entry.get("field", ""),
            "observed": str(value),
            "modality_in_manifest": modality,
            "source_lines": entry.get("source_lines", []),
            "source_quote": entry.get("source_quote", ""),
        }
        if passed:
            passes.append(base)
        elif sev is None:
            unverified.append({**base, "reason": "modality_unverified"})
        else:
            violations.append({
                **base, "severity": sev,
                "remediation": entry.get(
                    "remediation",
                    f'Set state field `{entry.get("field","?")}` per '
                    f'{entry.get("source_file","action.md")}:'
                    f'{(entry.get("source_lines",[None]) or [None])[0]}.'),
            })
    return passes, violations, unverified


def _eval_state_assertion(assertion: str, value) -> bool:
    if assertion == "is_set_and_numeric":
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    if assertion == "is_not_none":
        return value is not None
    if assertion == "is_truthy":
        return bool(value)
    if assertion.startswith("equals:"):
        return str(value) == assertion[len("equals:"):]
    if assertion.startswith("matches:"):
        return bool(re.search(assertion[len("matches:"):], str(value or "")))
    return value is not None


# --------------------------------------------------------------------- #
#  Semantic rule evaluation
# --------------------------------------------------------------------- #

def _phase_matches(rule_when: dict, phase: str) -> bool:
    patterns = rule_when.get("phase_in") or []
    if not patterns:
        return True  # rule applies to all phases
    return any(fnmatch.fnmatchcase(phase, p) for p in patterns)


def _resolve_artifact_obs(obs: dict, artifact_id: str) -> dict | None:
    for o in obs.get("artifact_observations", []):
        if isinstance(o, dict) and o.get("id") == artifact_id:
            return o
    return None


def _json_path_get(blob, path: str):
    """Tiny dotted-path resolver. path = 'a.b.0.c' walks dict/list."""
    cur = blob
    for tok in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(tok)
        else:
            return None
    return cur


def _check_passes(check: dict, obs: dict) -> bool:
    """Return True iff the check's condition fires (i.e. rule should emit)."""
    kind = check.get("kind")

    if kind == "transcript_lines_lt":
        threshold = int(check.get("value", 5))
        lines = int((obs.get("transcript") or {}).get("lines", 0))
        return lines < threshold

    if kind == "artifact_field_equals":
        aid = check["artifact_id"]
        a = _resolve_artifact_obs(obs, aid)
        if a is None or not a.get("exists"):
            return bool(check.get("trigger_if_missing", False))
        fields = a.get("json_fields", {}) or {}
        return _json_path_get(fields, check["field_path"]) == check["value"]

    if kind == "artifact_field_lt":
        aid = check["artifact_id"]
        a = _resolve_artifact_obs(obs, aid)
        if a is None or not a.get("exists"):
            return bool(check.get("trigger_if_missing", False))
        fields = a.get("json_fields", {}) or {}
        cur = _json_path_get(fields, check["field_path"])
        try:
            return float(cur) < float(check["value"])
        except (TypeError, ValueError):
            return False

    if kind == "artifact_list_len_gt_field_len":
        # action_stack.dfs_order longer than completed_actions
        aid = check["artifact_id"]
        a = _resolve_artifact_obs(obs, aid)
        if a is None or not a.get("exists"):
            return bool(check.get("trigger_if_missing", False))
        fields = a.get("json_fields", {}) or {}
        a_path = check["a_path"]
        b_path = check["b_path"]
        a_val = _json_path_get(fields, a_path) or []
        b_val = _json_path_get(fields, b_path) or []
        try:
            return len(a_val) > len(b_val)
        except TypeError:
            return False

    if kind == "artifact_missing":
        aid = check["artifact_id"]
        a = _resolve_artifact_obs(obs, aid)
        return (a is None) or (not a.get("exists"))

    if kind == "iron_rule_must_unsatisfied":
        # A MUST tool_call from iron rules has count==0 AND transcript healthy.
        min_lines = int(check.get("min_transcript_lines", 5))
        if int((obs.get("transcript") or {}).get("lines", 0)) < min_lines:
            return False  # transcript_too_short rule will fire instead
        for tco in obs.get("tool_call_observations", []):
            if not isinstance(tco, dict):
                continue
            if tco.get("iron_rule") is True and int(tco.get("count", 0)) == 0:
                return True
        return False

    return False  # unknown kind → no fire


def _eval_semantic_rules(rules: list, obs: dict, phase: str) -> list:
    out = []
    for rule in rules:
        if not _phase_matches(rule.get("when", {}), phase):
            continue
        check = rule.get("check") or {}
        if not _check_passes(check, obs):
            continue
        out.append({
            "id": rule["id"],
            "severity": rule.get("verdict", "block"),
            "channel": rule.get("channel", "file"),
            "expected": rule.get("expected", rule["id"]),
            "observed": "semantic_rule_fired",
            "modality_in_manifest": "MUST",
            "source_lines": [],
            "source_quote": rule.get("description", ""),
            "remediation": rule.get("remediation",
                                    f'Address semantic rule {rule["id"]}.'),
            "rule_source": "semantic_rules.json",
        })
    return out


# --------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------- #

def _aggregate_verdict(violations: list) -> str:
    rank = 0
    for v in violations:
        rank = max(rank, SEVERITY_RANK.get(v.get("severity", "info"), 0))
    return RANK_TO_VERDICT[rank]


def _verdict_summary(passes, violations, unverified) -> str:
    by_sev = {"fatal": 0, "block": 0, "warn": 0, "info": 0}
    for v in violations:
        s = v.get("severity", "info")
        if s in by_sev:
            by_sev[s] += 1
    return (f"passes={len(passes)} fatal={by_sev['fatal']} "
            f"block={by_sev['block']} warn={by_sev['warn']} "
            f"info={by_sev['info']} unverified={len(unverified)}")


def _self_sha1() -> str:
    try:
        return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--observations", required=True)
    ap.add_argument("--semantic-rules", default=None,
                    help="Path to semantic_rules.json; optional but recommended.")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--phase-action-files", default="")
    ap.add_argument("--target-skill-dir", default="")
    args = ap.parse_args()

    manifest = _read_json(args.manifest)
    obs = _read_json(args.observations)
    _validate_observations(obs)
    rules = _read_json(args.semantic_rules) if args.semantic_rules else []
    if isinstance(rules, dict):
        rules = rules.get("rules", [])

    passes, violations, unverified = [], [], []

    p, v, u = _audit_tool_calls(manifest, obs)
    passes += p; violations += v; unverified += u
    p, v, u = _audit_artifacts(manifest, obs)
    passes += p; violations += v; unverified += u
    p, v, u = _audit_state(manifest, obs)
    passes += p; violations += v; unverified += u

    violations += _eval_semantic_rules(rules, obs, args.phase)

    # Synthetic: extraction_low_confidence (per audit-report-schema.md §3).
    total_expected = (
        len(manifest.get("expected_tool_calls", []))
        + len(manifest.get("expected_artifacts", []))
        + len(manifest.get("expected_state_assertions", []))
    )
    if total_expected and len(unverified) >= 0.5 * total_expected:
        violations.append({
            "id": "extraction_low_confidence",
            "severity": "warn",
            "channel": "content",
            "expected": "<half-or-more expected items verifiable>",
            "observed": f"unverified={len(unverified)}/{total_expected}",
            "modality_in_manifest": "SHOULD",
            "source_lines": [],
            "source_quote": "(synthetic)",
            "remediation": ("Inspect /tmp/inspector_obs_<phase>.json and the "
                            "manifest to identify which checks could not run; "
                            "supply missing RUN_ENV vars or extend the manifest."),
        })

    verdict = _aggregate_verdict(violations)

    out = {
        "schema_version": SCHEMA_VERSION,
        "inspector_version": INSPECTOR_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "verdict_source": f"compute_verdict.py@{_self_sha1()}",
        "phase": args.phase,
        "target_skill": args.target_skill_dir,
        "phase_action_files": [s for s in args.phase_action_files.split(",") if s],
        "verdict": verdict,
        "verdict_summary": _verdict_summary(passes, violations, unverified),
        "passes": passes,
        "violations": violations,
        "unverified": unverified,
    }
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
