# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic runtime findings read from a measured server log."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

from hyperloom.common.failure_signature import CAPABILITY_DISABLED_PATTERNS
from hyperloom.common.launch_log_evidence import engine_adjusted_settings_from_log
from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.kernel.gemm_shape_coverage import parse_aiter_shape_lookups

CORRECTNESS = "correctness"
PERF_PATH = "perf_path"

DETECTED = "detected"
NOT_DETECTED = "not_detected"
UNKNOWN = "unknown"

RUNTIME_FINDINGS_FILE = "runtime_findings.json"
#: Largest throughput drop a verified correctness fix may cost and still KEEP.
CORRECTNESS_FIX_MAX_DROP_PCT = 3.0
KEEP_REASON_CORRECTNESS_FIX = "correctness_fix"
_EVIDENCE_MAX_CHARS = 300
#: Frameworks whose launch record ``engine_adjusted_settings_from_log`` reads.
_LAUNCH_RECORD_FRAMEWORKS = frozenset({"sglang", "vllm"})
_AITER_MISS_MARKER = "not found tuned config"
_TRACEBACK_MARKER = "Traceback (most recent call last):"
#: vLLM multiprocess prefix, e.g. ``(EngineCore_DP0 pid=123) ``.
_PROCESS_PREFIX_RE = re.compile(r"^\([^)]*pid=\d+\)\s?")
_EXCEPTION_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*):")


@dataclass(frozen=True)
class _LineRule:
    rule_id: str
    category: str
    patterns: tuple[Pattern[str], ...]
    subject: Callable[[re.Match[str]], str]
    frameworks: frozenset[str] | None = None


def _first_group_or_match(match: re.Match[str]) -> str:
    return (match.group(1) if match.re.groups else match.group(0)).strip()


_LINE_RULES: tuple[_LineRule, ...] = (
    _LineRule(
        rule_id="vllm.unknown_env",
        category=CORRECTNESS,
        patterns=(re.compile(r"Unknown vLLM environment variable detected: (\S+)"),),
        subject=_first_group_or_match,
        frameworks=frozenset({"vllm"}),
    ),
    _LineRule(
        rule_id="feature_disabled",
        category=PERF_PATH,
        patterns=(re.compile(r"\bDisabling ([^.:;,(\n]{1,80})"),),
        subject=_first_group_or_match,
    ),
    _LineRule(
        rule_id="capability_disabled",
        category=PERF_PATH,
        patterns=CAPABILITY_DISABLED_PATTERNS,
        subject=_first_group_or_match,
    ),
)

_ENGINE_ADJUSTED = "engine_adjusted"
_AITER_TUNED_MISS = "aiter.tuned_miss"
_RUNTIME_TRACEBACK = "runtime.traceback"

_CATEGORIES: dict[str, str] = {
    **{rule.rule_id: rule.category for rule in _LINE_RULES},
    _ENGINE_ADJUSTED: PERF_PATH,
    _AITER_TUNED_MISS: PERF_PATH,
    _RUNTIME_TRACEBACK: CORRECTNESS,
}


def _line_rules(framework: str) -> list[_LineRule]:
    return [rule for rule in _LINE_RULES if rule.frameworks is None or framework in rule.frameworks]


def _applicable_rules(framework: str) -> list[str]:
    rules = [rule.rule_id for rule in _line_rules(framework)]
    if framework in _LAUNCH_RECORD_FRAMEWORKS:
        rules.append(_ENGINE_ADJUSTED)
    return [*rules, _AITER_TUNED_MISS, _RUNTIME_TRACEBACK]


def _evidence(line: str) -> str:
    return flatten_for_prompt(line.strip())[:_EVIDENCE_MAX_CHARS]


class _Hits:
    """Detections keyed by ``(rule_id, subject)``, first evidence kept."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], dict[str, Any]] = {}

    def add(self, rule_id: str, subject: str, line: str, count: int = 1) -> None:
        subject = flatten_for_prompt(subject.strip())
        key = (rule_id, subject)
        if key in self._hits:
            self._hits[key]["count"] += count
            return
        self._hits[key] = {
            "rule_id": rule_id,
            "category": _CATEGORIES[rule_id],
            "status": DETECTED,
            "subject": subject,
            "count": count,
            "evidence": _evidence(line),
            "reason": "",
        }

    def findings(self, rules: list[str]) -> list[dict[str, Any]]:
        detected = {rule_id for rule_id, _ in self._hits}
        out = [self._hits[key] for key in sorted(self._hits)]
        out.extend(_status_finding(rule_id, NOT_DETECTED, "") for rule_id in rules if rule_id not in detected)
        return out


def _status_finding(rule_id: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "category": _CATEGORIES[rule_id],
        "status": status,
        "subject": "",
        "count": 0,
        "evidence": "",
        "reason": reason,
    }


def _scan_lines(path: str, framework: str, hits: _Hits) -> None:
    line_rules = _line_rules(framework)
    missed_shapes: set[tuple[int, int, int]] = set()
    first_miss_line = ""
    traceback_line = ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for rule in line_rules:
                for pattern in rule.patterns:
                    match = pattern.search(line)
                    if match is not None:
                        hits.add(rule.rule_id, rule.subject(match), line)
                        break
            if _AITER_MISS_MARKER in line:
                shapes, _ = parse_aiter_shape_lookups(line)
                if shapes and not first_miss_line:
                    first_miss_line = line
                missed_shapes |= shapes
            if _TRACEBACK_MARKER in line:
                traceback_line = line
                continue
            if traceback_line:
                body = _PROCESS_PREFIX_RE.sub("", line)
                exception = _EXCEPTION_LINE_RE.match(body)
                if exception is not None:
                    hits.add(_RUNTIME_TRACEBACK, exception.group(1), body)
                    traceback_line = ""
    if traceback_line:
        hits.add(_RUNTIME_TRACEBACK, "", traceback_line)
    if missed_shapes:
        hits.add(_AITER_TUNED_MISS, "aiter_tuned_config", first_miss_line, count=len(missed_shapes))


def scan_server_log(path: str | None, framework: str) -> dict[str, Any]:
    """Scan one measured server log; every applicable rule gets one status."""
    framework = str(framework or "").strip().lower()
    rules = _applicable_rules(framework)
    report: dict[str, Any] = {
        "schema_version": 1,
        "framework": framework,
        "log_path": path or "",
        "log_mtime": 0.0,
    }
    if not path:
        report["findings"] = [_status_finding(rule_id, UNKNOWN, "no_server_log") for rule_id in rules]
        return report
    hits = _Hits()
    try:
        report["log_mtime"] = Path(path).stat().st_mtime
        _scan_lines(path, framework, hits)
    except OSError as exc:
        reason = f"unreadable: {type(exc).__name__}"
        report["findings"] = [_status_finding(rule_id, UNKNOWN, reason) for rule_id in rules]
        return report
    if framework in _LAUNCH_RECORD_FRAMEWORKS:
        for name, change in engine_adjusted_settings_from_log(path, framework).items():
            hits.add(_ENGINE_ADJUSTED, name, f"{name}: {change['requested']} -> {change['resolved']}")
    report["findings"] = hits.findings(rules)
    return report


def persist_runtime_findings(report: dict[str, Any], *, slot: Path) -> str:
    """Write ``runtime_findings.json`` into the measurement slot."""
    slot.mkdir(parents=True, exist_ok=True)
    path = slot / RUNTIME_FINDINGS_FILE
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _findings_path(measurement: Mapping[str, Any]) -> Path | None:
    evidence_path = str(measurement.get("launch_evidence_path") or "")
    return Path(evidence_path).parent / RUNTIME_FINDINGS_FILE if evidence_path else None


def load_runtime_findings(measurement: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load the findings stored beside a measurement's launch evidence; ``None`` when absent."""
    path = _findings_path(measurement)
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def correctness_fix_refusal(before: dict[str, Any] | None, after: dict[str, Any] | None, finding_id: str) -> str:
    """Why ``finding_id`` is not a correctness finding the candidate resolved; empty when it is."""
    rule_id, separator, subject = finding_id.partition(":")
    if not separator:
        return f"resolves_finding {finding_id!r} is not rule_id:subject"
    if before is None:
        return "no runtime findings for the current best"
    target = next(
        (f for f in before["findings"] if (f["rule_id"], f["subject"], f["status"]) == (rule_id, subject, DETECTED)),
        None,
    )
    if target is None:
        return f"{finding_id} is not detected on the current best"
    if target["category"] != CORRECTNESS:
        return f"{finding_id} is {target['category']}, not correctness"
    if after is None:
        return "no runtime findings for the candidate run"
    observed = [f for f in after["findings"] if f["rule_id"] == rule_id]
    if not observed or any(f["status"] == UNKNOWN for f in observed):
        return f"{rule_id} was not observable on the candidate run"
    if any(f["subject"] == subject and f["status"] == DETECTED for f in observed):
        return f"{finding_id} is still detected on the candidate run"
    return ""


def render_runtime_findings(measurement: Mapping[str, Any]) -> str:
    """Render the findings stored beside a measurement's launch evidence."""
    path = _findings_path(measurement)
    if path is None:
        return "(no measurement slot recorded for the current best)"
    report = load_runtime_findings(measurement)
    if report is None:
        return f"(no runtime findings written at {path})"
    findings = report["findings"]
    lines = [f"runtime findings for {report['log_path'] or '(no server log)'} [{report['framework']}]"]
    lines.extend(
        f"- detected [{f['category']}] {f['rule_id']} {f['subject']} x{f['count']}: {f['evidence']}"
        for f in findings
        if f["status"] == DETECTED
    )
    clear = [f["rule_id"] for f in findings if f["status"] == NOT_DETECTED]
    if clear:
        lines.append(f"- not_detected: {', '.join(clear)}")
    lines.extend(f"- unknown: {f['rule_id']} ({f['reason']})" for f in findings if f["status"] == UNKNOWN)
    return "\n".join(lines)
