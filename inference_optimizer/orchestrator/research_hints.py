"""Research-hint artifacts collected by the research scout.

The scout produces *advisory*, source-backed priors:

* ``research_hints.md`` + ``research_hints.json`` — an append-only list of
  ``{what, expected_impact, accuracy_risk, source, domain_tags[], status}``
  hints. ``source`` is mandatory; sourceless hints are dropped.
* ``competitor_target.json`` — LLM-authored target numbers where every
  per-concurrency datapoint carries its own ``source``; entries missing a
  source are discarded.

All reads/writes are fail-soft: a missing or malformed file yields an empty
result rather than raising, so the main loop never aborts on a degraded
research artifact.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .. import session_paths

log = logging.getLogger("hyperloom.research_hints")

_HINT_FIELDS = ("what", "expected_impact", "accuracy_risk", "source",
                "domain_tags", "status")


def _coerce_hint(raw: Any) -> dict[str, Any] | None:
    """Normalize one incoming hint; return ``None`` when it has no source."""
    if not isinstance(raw, dict):
        return None
    source = str(raw.get("source") or "").strip()
    if not source:
        return None
    what = str(raw.get("what") or "").strip()
    if not what:
        return None
    tags = raw.get("domain_tags") or []
    if isinstance(tags, str):
        tags = [tags]
    domain_tags = [str(t).strip() for t in tags if str(t).strip()]
    return {
        "what": what,
        "expected_impact": str(raw.get("expected_impact") or "").strip(),
        "accuracy_risk": str(raw.get("accuracy_risk") or "").strip(),
        "source": source,
        "domain_tags": domain_tags,
        "status": str(raw.get("status") or "proposed").strip() or "proposed",
    }


def _hint_key(hint: dict[str, Any]) -> str:
    """Dedup key for append-merge: a hint is the same if its claim and
    source match (case-insensitive)."""
    return f"{hint['what'].lower()}::{hint['source'].lower()}"


def load_hints(session_dir: Path) -> list[dict[str, Any]]:
    """Return the structured hints written so far (empty on miss/parse error)."""
    path = session_paths.research_hints_json(session_dir)
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("research_hints: failed to read %s", path)
        return []
    items = data.get("hints") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        coerced = _coerce_hint(item)
        if coerced is not None:
            out.append(coerced)
    return out


def _render_md(hints: list[dict[str, Any]]) -> str:
    lines = ["# Research Hints", ""]
    if not hints:
        lines += [
            "_No proven priors collected yet (scout produced an empty set "
            "or all sources are unreachable)._",
            "",
        ]
        return "\n".join(lines)
    for idx, h in enumerate(hints, start=1):
        tags = ", ".join(h["domain_tags"]) if h["domain_tags"] else "-"
        lines += [
            f"## {idx}. {h['what']}",
            f"- expected_impact: {h['expected_impact'] or '-'}",
            f"- accuracy_risk: {h['accuracy_risk'] or '-'}",
            f"- domain_tags: {tags}",
            f"- status: {h['status']}",
            f"- source: {h['source']}",
            "",
        ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_hints_skeleton(session_dir: Path) -> None:
    """Ensure both hint artifacts exist even before the scout returns.

    Guarantees the PRELUDE invariant "research_hints.md is always present"
    without clobbering hints a prior run already appended.
    """
    md_path = session_paths.research_hints_md(session_dir)
    if md_path.exists():
        return
    existing = load_hints(session_dir)
    _persist(session_dir, existing)


def _persist(session_dir: Path, hints: list[dict[str, Any]]) -> None:
    sd = Path(session_dir)
    sd.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write(
            session_paths.research_hints_json(sd),
            json.dumps({"hints": hints}, indent=2) + "\n",
        )
        _atomic_write(session_paths.research_hints_md(sd), _render_md(hints))
    except OSError as exc:
        log.warning("research_hints: persist failed (%s): %s", sd, exc)


def append_hints(
    session_dir: Path, incoming: list[Any],
) -> tuple[int, int]:
    """Append-merge ``incoming`` scout hints into the artifacts.

    Returns ``(added, dropped)`` where ``dropped`` counts entries rejected
    for a missing source. Existing hints are preserved; duplicates (same
    claim + source) are not re-added.
    """
    existing = load_hints(session_dir)
    seen = {_hint_key(h) for h in existing}
    added = 0
    dropped = 0
    for raw in incoming or []:
        coerced = _coerce_hint(raw)
        if coerced is None:
            dropped += 1
            continue
        key = _hint_key(coerced)
        if key in seen:
            continue
        seen.add(key)
        existing.append(coerced)
        added += 1
    _persist(session_dir, existing)
    return added, dropped


def _coerce_per_conc(raw: Any) -> dict[str, Any] | None:
    """Drop a per-concurrency target row that lacks a source."""
    if not isinstance(raw, dict):
        return None
    if not str(raw.get("source") or "").strip():
        return None
    row: dict[str, Any] = {"source": str(raw["source"]).strip()}
    for key in ("conc", "tput_per_gpu", "tpot_ms", "interactivity"):
        if raw.get(key) is not None:
            row[key] = raw[key]
    return row


def write_competitor_target(
    session_dir: Path, target: Any,
) -> bool:
    """Persist ``competitor_target.json`` after dropping sourceless rows.

    Returns ``True`` when a target with at least one sourced per-conc row
    was written, else ``False`` (nothing written).
    """
    if not isinstance(target, dict):
        return False
    per_conc_in = target.get("per_conc") or []
    if not isinstance(per_conc_in, list):
        per_conc_in = []
    per_conc = [r for r in (_coerce_per_conc(x) for x in per_conc_in) if r]
    if not per_conc:
        return False
    out = {
        "gpu": str(target.get("gpu") or "").strip(),
        "model": str(target.get("model") or "").strip(),
        "framework": str(target.get("framework") or "").strip(),
        "precision": str(target.get("precision") or "").strip(),
        "per_conc": per_conc,
        "notes": str(target.get("notes") or "").strip(),
    }
    try:
        _atomic_write(
            session_paths.competitor_target_json(session_dir),
            json.dumps(out, indent=2) + "\n",
        )
    except OSError as exc:
        log.warning("competitor_target: write failed: %s", exc)
        return False
    return True


def summarise_for_prompt(
    session_dir: Path, *, max_entries: int = 8,
) -> str:
    """Compact advisory block of proven priors for the orchestration prompt.

    Returns an empty string when no hints exist (section is skipped).
    Advisory only — these are priors to try earlier, not a directive.
    """
    hints = load_hints(session_dir)
    if not hints:
        return ""
    lines = [
        "Proven priors collected by the research scout. Treat as advisory "
        "hints to try earlier — each carries a source.",
    ]
    for h in hints[:max_entries]:
        impact = h["expected_impact"] or "?"
        risk = h["accuracy_risk"] or "?"
        lines.append(
            f"- {h['what']} (impact={impact}, accuracy_risk={risk}, "
            f"source={h['source']})"
        )
    extra = len(hints) - max_entries
    if extra > 0:
        lines.append(
            f"... and {extra} more in research_hints.md."
        )
    return "\n".join(lines)


__all__ = [
    "append_hints",
    "load_hints",
    "summarise_for_prompt",
    "write_competitor_target",
    "write_hints_skeleton",
]
