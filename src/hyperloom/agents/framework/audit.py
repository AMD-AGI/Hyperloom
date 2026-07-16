# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""FRAMEWORK semantic audit — static local-source judging (Step 2 MVP).

Given a candidate PR's unified diff and the live framework source roots,
decide whether the PR's change is **already present** in the local tree
(``already_equivalent``), **absent but directly appliable**
(``direct_apply``), **partially present / drifted** (``needs_rewrite``), or
not judgeable (``unknown``). The verdict feeds the Coordinator's per-candidate
routing so it can skip already-merged PRs and seed the authoring specialist.

Two layers:

* **static** (default, hermetic): parse the diff, resolve each touched file
  under ``framework_source_roots``, and measure how much of the PR's added
  lines / symbols already exist locally + whether the diff's context anchors
  are present (raw-apply feasibility).
* **llm** (opt-in via ``use_llm``): a single chat-completion that may refine the
  static verdict. Best-effort; failure or missing creds keeps the static verdict.

An ``already_*`` verdict is downgraded to ``unknown`` when it has no concrete
static evidence backing it.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from ._audit_common import (
    FileChange,
    _obtain_patch_text,
    _resolve_local_file,
    _symbols,
    _verdict,
    parse_unified_diff,
)


log = logging.getLogger(__name__)


# Verdict thresholds (fraction of PR "signal" added-lines already found locally).
ALREADY_PRESENT_RATIO = 0.90
PARTIAL_PRESENT_RATIO = 0.20
# Context-anchor presence above which a raw ``git apply`` is judged likely.
CONTEXT_APPLY_RATIO = 0.60

_SEMANTIC_STATUSES = (
    "already_equivalent",
    "already_superset",
    "partially_present",
    "not_present",
    "unknown",
)
_APPLICABILITIES = (
    "direct_apply",
    "needs_rewrite",
    "not_applicable",
    "needs_human_review",
)

def _signal_lines(lines: list[str]) -> list[str]:
    """Keep semantically meaningful lines (drop blanks / pure punctuation).

    Args:
        lines: Raw added/removed lines.

    Returns:
        Stripped lines worth matching against local source.
    """
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if len(s) <= 3:
            continue
        if not any(ch.isalnum() for ch in s):
            continue
        out.append(s)
    return out


def _analyze_change(change: FileChange, roots: list[Path]) -> dict[str, Any]:
    """Static analysis for one file change against the local tree.

    Returns a per-file evidence dict with ``present_ratio`` (added signal lines
    already local), ``context_ratio`` (context anchors local), matched symbols,
    and ``file_present``.
    """
    local = _resolve_local_file(change.path, roots)
    result: dict[str, Any] = {
        "local_file": str(local) if local else "",
        "diff_path": change.path,
        "file_present": local is not None,
        "is_new": change.is_new,
        "is_deleted": change.is_deleted,
        "present_ratio": 0.0,
        "context_ratio": 0.0,
        "matched_symbols": [],
        "reason": "",
    }
    if change.is_new:
        # A new file existing locally => likely already merged.
        result["present_ratio"] = 1.0 if local is not None else 0.0
        result["reason"] = (
            "new-file PR target already exists locally" if local is not None else "new-file PR target absent locally"
        )
        return result
    if local is None:
        result["reason"] = "target file not found under framework_source_roots"
        return result

    try:
        text = local.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["reason"] = f"local file unreadable: {exc}"
        return result
    local_lines = {ln.strip() for ln in text.splitlines()}

    added_signal = _signal_lines(change.added)
    if added_signal:
        present = sum(1 for s in added_signal if s in local_lines)
        result["present_ratio"] = round(present / len(added_signal), 4)
    else:
        # No added signal (pure deletion / formatting): inconclusive.
        result["present_ratio"] = 0.0

    ctx_signal = _signal_lines(change.context)
    if ctx_signal:
        ctx_present = sum(1 for s in ctx_signal if s in local_lines)
        result["context_ratio"] = round(ctx_present / len(ctx_signal), 4)

    matched_syms = [s for s in _symbols(change.added) if any(s in ln for ln in local_lines)]
    result["matched_symbols"] = sorted(set(matched_syms))
    if result["present_ratio"] >= ALREADY_PRESENT_RATIO:
        result["reason"] = "added lines already present in local source"
    elif result["present_ratio"] >= PARTIAL_PRESENT_RATIO:
        result["reason"] = "added lines partially present (drift / superseded)"
    else:
        result["reason"] = "added lines absent from local source"
    return result


def _classify(
    candidate_id: str,
    per_file: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-file static signals into a single verdict.

    Args:
        candidate_id: The candidate identifier.
        per_file: Per-file analysis dicts from :func:`_analyze_change`.

    Returns:
        The semantic_audit verdict dict.
    """
    modify_files = [f for f in per_file if not f.get("is_deleted")]
    if not modify_files:
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="unknown",
            applicability="needs_human_review",
            confidence=0.1,
            evidence=[],
            risks=["diff carries no addable content (pure deletion/rename)"],
            recommended_next_step="author_via_specialist",
            metrics={"files_total": len(per_file)},
        )

    present_count = sum(1 for f in modify_files if f.get("file_present"))
    ratios = [float(f.get("present_ratio") or 0.0) for f in modify_files]
    ctx_ratios = [float(f.get("context_ratio") or 0.0) for f in modify_files if f.get("file_present")]
    mean_present = sum(ratios) / len(ratios) if ratios else 0.0
    mean_context = sum(ctx_ratios) / len(ctx_ratios) if ctx_ratios else 0.0
    all_present = present_count == len(modify_files)
    any_present = present_count > 0

    evidence: list[dict[str, Any]] = [
        {
            "local_file": f.get("local_file") or "",
            "symbol": ", ".join(f.get("matched_symbols") or []),
            "reason": f.get("reason") or "",
        }
        for f in modify_files
    ]
    metrics = {
        "files_total": len(modify_files),
        "files_present": present_count,
        "mean_present_ratio": round(mean_present, 4),
        "mean_context_ratio": round(mean_context, 4),
    }
    has_concrete_evidence = any(
        (f.get("matched_symbols") or []) or float(f.get("present_ratio") or 0.0) > 0.0 for f in modify_files
    )

    # No touched file exists in this tree => raw apply impossible here.
    if not any_present:
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="not_present",
            applicability="not_applicable",
            confidence=0.6,
            evidence=evidence,
            risks=["none of the PR's target files exist under framework_source_roots"],
            recommended_next_step="skip",
            metrics=metrics,
        )

    # Strongly present everywhere => already merged / equivalent.
    if all_present and mean_present >= ALREADY_PRESENT_RATIO:
        if not has_concrete_evidence:
            # Never claim "already" without a concrete hit.
            return _verdict(
                candidate_id=candidate_id,
                semantic_status="unknown",
                applicability="needs_human_review",
                confidence=0.2,
                evidence=evidence,
                risks=["high present-ratio but no concrete symbol/line evidence"],
                recommended_next_step="author_via_specialist",
                metrics=metrics,
            )
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="already_equivalent",
            applicability="not_applicable",
            confidence=round(min(0.99, 0.6 + 0.4 * mean_present), 4),
            evidence=evidence,
            risks=[],
            recommended_next_step="skip",
            metrics=metrics,
        )

    # Partially present => drifted / superseded; let the specialist rewrite.
    if mean_present >= PARTIAL_PRESENT_RATIO:
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="partially_present",
            applicability="needs_rewrite",
            confidence=0.5,
            evidence=evidence,
            risks=["change partially present; raw diff likely conflicts"],
            recommended_next_step="author_via_specialist",
            metrics=metrics,
        )

    # Absent locally. If targets exist and context anchors are present, a raw
    # ``git apply`` is likely to land => direct_apply; otherwise rewrite.
    if all_present and mean_context >= CONTEXT_APPLY_RATIO:
        return _verdict(
            candidate_id=candidate_id,
            semantic_status="not_present",
            applicability="direct_apply",
            confidence=round(min(0.9, 0.5 + 0.4 * mean_context), 4),
            evidence=evidence,
            risks=[],
            recommended_next_step="direct_framework",
            metrics=metrics,
        )
    return _verdict(
        candidate_id=candidate_id,
        semantic_status="not_present",
        applicability="needs_rewrite",
        confidence=0.45,
        evidence=evidence,
        risks=["target context drifted; raw diff apply uncertain"],
        recommended_next_step="author_via_specialist",
        metrics=metrics,
    )


def _persist_audit(request: dict[str, Any], result: dict[str, Any]) -> None:
    """Persist the semantic audit result next to audit material."""
    work_dir = Path(str(request.get("work_dir") or (Path(tempfile.gettempdir()) / "framework-agent" / "phase-audit"))).expanduser()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        import json

        (work_dir / "semantic_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        log.debug("phase-audit: could not persist semantic_audit.json", exc_info=True)


def run_phase_audit(request: dict[str, Any]) -> dict[str, Any]:
    """Run the FRAMEWORK semantic audit for one candidate.

    Args:
        request: ``{candidate, framework, framework_source_roots, repo_url?,
            diff_text?|patches_path?|primus_cortex_url?, work_dir?, use_llm?,
            model?, context?}``.

    Returns:
        The semantic_audit verdict dict (also written to
        ``<work_dir>/semantic_audit.json`` when ``work_dir`` is set).
    """
    candidate = request.get("candidate") or {}
    candidate_id = str(
        candidate.get("candidate_id") or candidate.get("pr_url") or candidate.get("ref") or ""
    )
    src_framework = str(request.get("framework") or "").strip().lower()
    dst_framework = str(request.get("target_framework") or "").strip().lower()
    if dst_framework and dst_framework != src_framework:
        # Lazy import to keep the audit<->cross_framework import cycle broken.
        from .cross_framework import run_cross_framework_audit

        result = run_cross_framework_audit(request)
        _persist_audit(request, result)
        return result

    roots = [Path(str(r)).expanduser() for r in (request.get("framework_source_roots") or []) if str(r).strip()]
    work_dir = Path(str(request.get("work_dir") or (Path(tempfile.gettempdir()) / "framework-agent" / "phase-audit"))).expanduser()

    patch_text, patch_source = _obtain_patch_text(request, work_dir)
    if not patch_text.strip():
        result = _verdict(
            candidate_id=candidate_id,
            semantic_status="unknown",
            applicability="needs_human_review",
            confidence=0.0,
            evidence=[],
            risks=["no patch material available (diff_text/patches_path/primus fetch all empty)"],
            recommended_next_step="author_via_specialist",
            metrics={"patch_source": patch_source},
        )
    elif not roots:
        result = _verdict(
            candidate_id=candidate_id,
            semantic_status="unknown",
            applicability="needs_human_review",
            confidence=0.0,
            evidence=[],
            risks=["no framework_source_roots provided; cannot judge locally"],
            recommended_next_step="author_via_specialist",
            metrics={"patch_source": patch_source},
        )
    else:
        changes = parse_unified_diff(patch_text)
        if not changes:
            result = _verdict(
                candidate_id=candidate_id,
                semantic_status="unknown",
                applicability="needs_human_review",
                confidence=0.0,
                evidence=[],
                risks=["diff parsed to zero file changes"],
                recommended_next_step="author_via_specialist",
                metrics={"patch_source": patch_source},
            )
        else:
            per_file = [_analyze_change(c, roots) for c in changes]
            result = _classify(candidate_id, per_file)
            result["metrics"]["patch_source"] = patch_source

    if bool(request.get("use_llm")):
        try:
            result = _maybe_llm_refine(request, result, patch_text)
        except Exception as exc:  # noqa: BLE001 — LLM refine is best-effort
            log.warning("phase-audit: LLM refine failed; keeping static verdict: %r", exc)

    _persist_audit(request, result)
    return result


def _maybe_llm_refine(
    request: dict[str, Any],
    static_result: dict[str, Any],
    patch_text: str,
) -> dict[str, Any]:
    """Optionally refine the static verdict with a single chat-completion.

    Opt-in (``use_llm=True``) and best-effort: requires ``SAFE_API_KEY`` +
    ``OPENAI_BASE_URL`` (or request ``openai_base_url``). Any failure / missing
    credential returns the static verdict unchanged. Never escalates an
    ``already_*`` claim the static layer didn't already back with evidence.

    Args:
        request: The phase-audit request (carries ``model`` / creds overrides).
        static_result: The static-layer verdict.
        patch_text: The PR's unified diff (truncated before sending).

    Returns:
        A possibly-refined verdict dict (``layer="llm"`` when refined).
    """
    import json
    import os

    api_key = str(request.get("api_key") or os.environ.get("SAFE_API_KEY") or "").strip()
    base_url = str(request.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    model = str(request.get("model") or os.environ.get("FRAMEWORK_AGENT_AUDIT_MODEL") or "gpt-5.4").strip()
    if not api_key or not base_url:
        static_result.setdefault("risks", []).append("llm refine skipped: missing SAFE_API_KEY/OPENAI_BASE_URL")
        return static_result

    try:
        from openai import OpenAI  # lazy: only when use_llm
    except Exception:  # noqa: BLE001
        static_result.setdefault("risks", []).append("llm refine skipped: openai sdk unavailable")
        return static_result

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        "You are auditing whether an upstream PR's change is already present in "
        "a local framework source tree. Given the static analysis result and the "
        "PR diff, return STRICT JSON with keys: semantic_status (one of "
        f"{list(_SEMANTIC_STATUSES)}), applicability (one of {list(_APPLICABILITIES)}), "
        "confidence (0..1), recommended_next_step (skip|direct_framework|"
        "author_via_specialist), note (short). Do not invent evidence.\n\n"
        f"STATIC_RESULT:\n{json.dumps(static_result, ensure_ascii=False)}\n\n"
        f"PR_DIFF (truncated):\n{patch_text[:6000]}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    refined = _parse_llm_json(content)
    if not refined:
        static_result.setdefault("risks", []).append("llm refine returned no parseable JSON")
        return static_result

    status = str(refined.get("semantic_status") or static_result["semantic_status"])
    appl = str(refined.get("applicability") or static_result["applicability"])
    if status not in _SEMANTIC_STATUSES or appl not in _APPLICABILITIES:
        static_result.setdefault("risks", []).append("llm refine produced invalid enum; kept static")
        return static_result
    # Don't let the LLM upgrade to already_* with no static evidence.
    if status.startswith("already_") and not static_result.get("evidence"):
        static_result.setdefault("risks", []).append("llm already_* claim rejected (no static evidence)")
        return static_result

    static_result["semantic_status"] = status
    static_result["applicability"] = appl
    if isinstance(refined.get("confidence"), (int, float)):
        static_result["confidence"] = round(float(refined["confidence"]), 4)
    nxt = str(refined.get("recommended_next_step") or "")
    if nxt in ("skip", "direct_framework", "author_via_specialist"):
        static_result["recommended_next_step"] = nxt
    note = str(refined.get("note") or "").strip()
    if note:
        static_result.setdefault("risks", []).append(f"llm: {note}")
    static_result["layer"] = "llm"
    return static_result


def _parse_llm_json(content: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM reply (tolerant of fences)."""
    import json

    if not content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


__all__ = [
    "FileChange",
    "_persist_audit",
    "parse_unified_diff",
    "run_phase_audit",
]
