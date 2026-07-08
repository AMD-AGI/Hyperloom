# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cross-framework audit helpers for framework-agent P1.

P1 does not rewrite code. It maps source-framework PR diff files to known
target-framework modules and reports whether the target side is present, so
the coordinator can route the PR to a specialist rewrite path.

Coverage note (phased rollout): the landing signal depends entirely on
``cross_framework_map.jsonl``, which currently ships only a handful of
low/medium-confidence seed mappings (some targets not yet repo-registered).
Until that map grows, most cross-framework audits fall to low-confidence
``unknown`` / ``needs_human_review`` verdicts by design rather than a concrete
``direct_apply`` / ``needs_rewrite`` — this is expected, not a defect.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._audit_common import _obtain_patch_text, _resolve_local_file, _symbols, _verdict, parse_unified_diff
from .kb import path_for_framework


log = logging.getLogger(__name__)

_MAP_FILE = "cross_framework_map.jsonl"


def cross_framework_map_path() -> Path:
    """Return the active cross-framework module-map JSONL path."""
    return path_for_framework("") / _MAP_FILE


def load_cross_framework_map(src_framework: str, dst_framework: str) -> list[dict[str, Any]]:
    """Load module mappings for one source/target framework pair.

    Args:
        src_framework: Source framework name from the candidate PR.
        dst_framework: Target framework name to port into.

    Returns:
        Matching JSONL records. Missing files and malformed rows are tolerated.
    """
    src = str(src_framework or "").strip().lower()
    dst = str(dst_framework or "").strip().lower()
    path = cross_framework_map_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning("cross_framework_map: skipping malformed line: %.80s", line)
            continue
        if not isinstance(rec, dict):
            continue
        if str(rec.get("src_framework") or "").strip().lower() != src:
            continue
        if str(rec.get("dst_framework") or "").strip().lower() != dst:
            continue
        out.append(rec)
    return out


def _paths_match(diff_path: str, mapped_path: str) -> bool:
    """Return True when two module paths match across common strip levels."""
    left = str(diff_path or "").strip("/")
    right = str(mapped_path or "").strip("/")
    if not left or not right:
        return False
    if left == right:
        return True
    left_parts = left.split("/")
    right_parts = right.split("/")
    n = min(len(left_parts), len(right_parts))
    if n <= 1:
        return left_parts[-1] == right_parts[-1]
    return left_parts[-n:] == right_parts[-n:]


def _symbol_anchors(change: Any, local: Path | None) -> list[str]:
    """Return src-diff def/class names that already anchor in the target module.

    H1 (#5-P2): upgrade the landing signal from "file exists" to "symbol
    anchor exists". Reuses ``_audit_common._symbols`` on the diff's added lines
    and checks which appear in the resolved local target file, mirroring
    ``audit._analyze_change``'s matched-symbols logic. Best-effort: unreadable
    files or no added symbols yield an empty list.

    Args:
        change: One parsed ``FileChange`` from the source diff.
        local: The resolved local target-module path, or ``None``.

    Returns:
        Ordered list of matched symbol names (empty when none anchor).
    """
    if local is None:
        return []
    src_syms = _symbols(getattr(change, "added", []) or [])
    if not src_syms:
        return []
    try:
        dst_lines = {ln.strip() for ln in local.read_text(encoding="utf-8", errors="replace").splitlines()}
    except OSError:
        return []
    return [s for s in src_syms if any(s in ln for ln in dst_lines)]


def _map_changes(changes: list[Any], mapping: list[dict[str, Any]], roots: list[Path]) -> list[dict[str, Any]]:
    """Map source diff changes to target framework modules."""
    hits: list[dict[str, Any]] = []
    for change in changes:
        for rec in mapping:
            src_module = str(rec.get("src_module") or "")
            if not src_module or not _paths_match(str(change.path), src_module):
                continue
            dst_module = str(rec.get("dst_module") or "")
            local = _resolve_local_file(dst_module, roots) if dst_module else None
            anchors = _symbol_anchors(change, local)
            hits.append(
                {
                    "src_path": str(change.path),
                    "dst_module": dst_module,
                    "dst_present": local is not None,
                    # H1: symbol-level landing anchor (empty when none matched).
                    "dst_symbol": anchors[0] if anchors else "",
                    "dst_symbol_present": bool(anchors),
                    "local_file": str(local) if local else "",
                    "feature": str(rec.get("feature") or ""),
                    "notes": str(rec.get("notes") or ""),
                    "confidence": str(rec.get("confidence") or ""),
                }
            )
            break
    return hits


def run_cross_framework_audit(request: dict[str, Any]) -> dict[str, Any]:
    """Judge whether a source-framework PR should be rewritten for a target.

    Args:
        request: Phase-audit request with ``framework`` and distinct
            ``target_framework`` plus target source roots.

    Returns:
        A semantic-audit verdict with ``layer="cross_framework"``.
    """
    candidate = request.get("candidate") or {}
    candidate_id = str(candidate.get("candidate_id") or candidate.get("pr_url") or candidate.get("ref") or "")
    src_framework = str(request.get("framework") or "").strip().lower()
    dst_framework = str(request.get("target_framework") or "").strip().lower()

    explicit_roots = request.get("target_framework_source_roots") or []
    fallback_roots = request.get("framework_source_roots") or []
    raw_roots = explicit_roots or fallback_roots
    roots = [Path(str(r)).expanduser() for r in raw_roots if str(r).strip()]
    roots_source = "explicit" if explicit_roots else ("fallback" if fallback_roots else "none")
    work_dir = Path(str(request.get("work_dir") or "/tmp/framework-agent/phase-audit")).expanduser()

    metrics: dict[str, Any] = {
        "src_framework": src_framework,
        "dst_framework": dst_framework,
        "roots_source": roots_source,
    }
    patch_text, patch_source = _obtain_patch_text(request, work_dir)
    metrics["patch_source"] = patch_source
    if not patch_text.strip():
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="unknown",
            applicability="needs_human_review",
            confidence=0.0,
            evidence=[],
            risks=["no patch material available for cross-framework audit"],
            recommended_next_step="author_via_specialist",
            layer="cross_framework",
            metrics=metrics,
        )

    mapping = load_cross_framework_map(src_framework, dst_framework)
    if not mapping:
        seed_missing = not cross_framework_map_path().is_file()
        metrics["map_source"] = "missing_seed_file" if seed_missing else "no_pair_match"
        reason = (
            f"cross_framework_map seed file missing: {cross_framework_map_path()}"
            if seed_missing
            else f"no cross_framework_map entries for {src_framework!r} -> {dst_framework!r}"
        )
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="unknown",
            applicability="needs_human_review",
            confidence=0.1,
            evidence=[],
            risks=[reason],
            recommended_next_step="author_via_specialist",
            layer="cross_framework",
            metrics=metrics,
        )

    changes = parse_unified_diff(patch_text)
    hits = _map_changes(changes, mapping, roots)
    metrics["files_total"] = len(changes)
    metrics["mapped_files"] = len(hits)
    if not hits:
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="not_present",
            applicability="needs_human_review",
            confidence=0.2,
            evidence=[],
            risks=["no mapped target module matched the PR's changed files"],
            recommended_next_step="author_via_specialist",
            layer="cross_framework",
            metrics=metrics,
        )

    present = sum(1 for hit in hits if hit.get("dst_present"))
    coverage = present / len(hits)
    # H1 (#5-P2): symbol-anchor coverage strengthens the landing signal beyond
    # mere file presence; it only ever raises confidence (capped), never lowers.
    sym_present = sum(1 for hit in hits if hit.get("dst_symbol_present"))
    sym_coverage = sym_present / len(hits)
    metrics["dst_modules_present"] = present
    metrics["dst_symbols_present"] = sym_present
    semantic_status = "partially_present" if present else "not_present"
    risks = ["cross-framework port: raw git apply impossible; specialist must rewrite"]
    if request.get("use_llm"):
        risks.append("use_llm ignored in cross-framework mode (P1); LLM validation deferred")
    if present == 0 and roots_source != "explicit":
        risks.append(f"target roots not explicit (roots_source={roots_source}); verify target_framework_source_roots")
    return _verdict(
        candidate_id=candidate_id,
        semantic_status=semantic_status,
        applicability="needs_rewrite",
        confidence=round(min(0.85, 0.4 + 0.45 * coverage + 0.1 * sym_coverage), 4),
        evidence=hits,
        risks=risks,
        recommended_next_step="author_via_specialist",
        layer="cross_framework",
        metrics=metrics,
    )


__all__ = ["cross_framework_map_path", "load_cross_framework_map", "run_cross_framework_audit"]
