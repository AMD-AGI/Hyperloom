# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Research-hint artifacts collected by the research scout (advisory, source-backed priors).

Produces ``research_hints.{md,json}`` and ``competitor_target.json``
(sourceless entries dropped). All reads/writes are fail-soft.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
    """Dedup key for append-merge: claim + source (case-insensitive)."""
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
    """Render research hints as a Markdown document.

    Args:
        hints: Normalized hint records.

    Returns:
        Markdown text, with a placeholder note when there are no hints.
    """
    lines = ["# Research Hints", ""]
    if not hints:
        lines += [
            "_No proven priors collected yet (scout produced an empty set "
            + "or all sources are unreachable)._",
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
    """Write text to a file atomically via a temp file and rename.

    Args:
        path: Destination file path.
        text: Text content to write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_hints_skeleton(session_dir: Path) -> None:
    """Ensure both hint artifacts exist before the scout returns (PRELUDE invariant; preserves prior hints)."""
    md_path = session_paths.research_hints_md(session_dir)
    if md_path.exists():
        return
    existing = load_hints(session_dir)
    _persist(session_dir, existing)


def _persist(session_dir: Path, hints: list[dict[str, Any]]) -> None:
    """Persist hints to the session's JSON and Markdown artifacts.

    Args:
        session_dir: Session directory to write into.
        hints: Hint records to serialize.
    """
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
    """Append-merge ``incoming`` scout hints; returns ``(added, dropped)`` (dropped = missing-source rejects; duplicates not re-added)."""
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
    """Persist ``competitor_target.json`` after dropping sourceless rows; ``True`` when ≥1 sourced row was written."""
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


def load_competitor_target(session_dir: Path) -> dict[str, Any] | None:
    """Read ``competitor_target.json`` keeping only sourced per-conc rows; ``None`` when absent/malformed/sourceless. Fail-soft."""
    path = session_paths.competitor_target_json(session_dir)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("competitor_target: failed to read %s", path)
        return None
    if not isinstance(data, dict):
        return None
    per_conc_in = data.get("per_conc") or []
    if not isinstance(per_conc_in, list):
        return None
    per_conc = [r for r in (_coerce_per_conc(x) for x in per_conc_in) if r]
    if not per_conc:
        return None
    return {
        "gpu": str(data.get("gpu") or "").strip(),
        "model": str(data.get("model") or "").strip(),
        "framework": str(data.get("framework") or "").strip(),
        "precision": str(data.get("precision") or "").strip(),
        "per_conc": per_conc,
        "notes": str(data.get("notes") or "").strip(),
    }


def _match_target_row(
    target: dict[str, Any], conc: int | None,
) -> dict[str, Any] | None:
    """Pick the per-conc target row nearest ``conc`` (highest-throughput row when conc unknown)."""
    rows = target.get("per_conc") or []
    if not rows:
        return None
    if conc is not None:
        exact = [r for r in rows if _to_num(r.get("conc")) == conc]
        if exact:
            return max(exact, key=lambda r: _to_num(r.get("tput_per_gpu")) or 0.0)
        rows_with_conc = [r for r in rows if _to_num(r.get("conc")) is not None]
        if rows_with_conc:
            return min(
                rows_with_conc,
                key=lambda r: abs((_to_num(r.get("conc")) or 0) - conc),
            )
    return max(rows, key=lambda r: _to_num(r.get("tput_per_gpu")) or 0.0)


def gap_analysis(
    target: dict[str, Any] | None,
    *,
    our_tput_per_gpu: float | None,
    our_tpot_ms: float | None,
    conc: int | None = None,
) -> dict[str, Any] | None:
    """Compute advisory throughput/latency gaps against a competitor row; ``None`` when no comparable row. ``primary_gap`` is "latency" when TPOT ratio outweighs throughput gap."""
    if not target:
        return None
    row = _match_target_row(target, conc)
    if row is None:
        return None
    tgt_tput = _to_num(row.get("tput_per_gpu"))
    tgt_tpot = _to_num(row.get("tpot_ms"))
    tgt_inter = _to_num(row.get("interactivity"))

    throughput_gap_pct: float | None = None
    if tgt_tput and our_tput_per_gpu and tgt_tput > 0:
        throughput_gap_pct = (tgt_tput - our_tput_per_gpu) / tgt_tput * 100.0

    tpot_ratio: float | None = None
    if tgt_tpot and our_tpot_ms and tgt_tpot > 0:
        tpot_ratio = our_tpot_ms / tgt_tpot

    interactivity_gap_pct: float | None = None
    if tgt_inter and our_tpot_ms and our_tpot_ms > 0:
        our_inter = 1000.0 / our_tpot_ms
        interactivity_gap_pct = (tgt_inter - our_inter) / tgt_inter * 100.0

    primary_gap = "throughput"
    if tpot_ratio is not None and throughput_gap_pct is not None:
        if (tpot_ratio - 1.0) * 100.0 > throughput_gap_pct:
            primary_gap = "latency"
    elif tpot_ratio is not None and tpot_ratio > 1.0:
        primary_gap = "latency"

    return {
        "throughput_gap_pct": throughput_gap_pct,
        "tpot_ratio": tpot_ratio,
        "interactivity_gap_pct": interactivity_gap_pct,
        "primary_gap": primary_gap,
        "target_conc": _to_num(row.get("conc")),
        "source": row.get("source"),
    }


def full_gap_summary(
    gap: dict[str, Any] | None,
    *,
    tpot_ratio_threshold: float = 1.3,
) -> str:
    """Render an advisory "External target gap" block (empty when no gap; advisory only, never gates)."""
    if not gap:
        return ""
    lines = ["External target gap (advisory) — competitor numbers are "
             + "LLM-authored with sources; treat as direction, not a gate."]
    tg = gap.get("throughput_gap_pct")
    tr = gap.get("tpot_ratio")
    ig = gap.get("interactivity_gap_pct")
    if tg is not None:
        lines.append(f"- throughput gap vs target: {tg:+.1f}%")
    if tr is not None:
        lines.append(f"- TPOT ratio (ours/target): {tr:.2f}x")
    if ig is not None:
        lines.append(f"- interactivity gap vs target: {ig:+.1f}%")
    if gap.get("source"):
        lines.append(f"- target source: {gap['source']}")
    if tr is not None and tr > tpot_ratio_threshold:
        lines.append(
            "- Priority: TPOT is the dominant gap — favor decode-kernel, "
            "comm-overlap, MTP, and quantized-allreduce directions to cut "
            "per-output-token latency."
        )
    return "\n".join(lines)


def _to_num(value: Any) -> float | None:
    """Coerce a value to ``float``, returning ``None`` on failure.

    Args:
        value: Arbitrary value to convert.

    Returns:
        The float value, or ``None`` if it cannot be parsed.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Direction keywords for cutting per-output-token latency (flag variants aligning with a dominant TPOT gap; advisory).
_LATENCY_DIRECTION_KEYWORDS: tuple[str, ...] = (
    "mtp", "speculative", "eagle", "medusa", "decode", "comm", "overlap",
    "allreduce", "all_reduce", "quantized_allreduce", "cuda_graph",
    "cudagraph", "fused", "fuse",
)

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "use", "using", "enable", "enabled",
    "via", "this", "that", "from", "into", "per", "set", "than", "more",
    "less", "when", "then", "case", "mode", "flag", "flags", "value",
})


def _tokens(text: str) -> set[str]:
    """Tokenize text into a set of lowercase content words.

    Splits on non-alphanumeric characters and drops short tokens and
    stopwords so the remaining set is useful for overlap matching.

    Args:
        text: Free-form text to tokenize.

    Returns:
        Set of distinct content tokens (length >= 3, non-stopword).
    """
    out: set[str] = set()
    for raw in re.split(r"[^a-z0-9]+", str(text).lower()):
        tok = raw.strip()
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def match_variants_to_priors(
    variants: list[dict[str, Any]],
    hints: list[dict[str, Any]],
    *,
    primary_gap: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Annotate which variants align with proven priors (advisory; informs ordering only). Returns ``{name: {hints, latency_aligned}}`` for variants matching a hint or a dominant latency gap."""
    out: dict[str, dict[str, Any]] = {}
    latency_dominant = str(primary_gap or "").strip().lower() == "latency"
    hint_tokens: list[tuple[str, set[str]]] = []
    for h in hints or []:
        if not isinstance(h, dict):
            continue
        what = str(h.get("what") or "").strip()
        if not what:
            continue
        toks = _tokens(what)
        for tag in h.get("domain_tags") or []:
            toks |= _tokens(tag)
        if toks:
            hint_tokens.append((what, toks))
    for variant in variants or []:
        if not isinstance(variant, dict):
            continue
        name = str(variant.get("name") or "").strip()
        if not name:
            continue
        text = " ".join(
            str(variant.get(k) or "")
            for k in ("name", "extra_server_args", "extra_sglang_args",
                      "candidate_extra_server_args", "description")
        )
        text += " " + " ".join(str(t) for t in (variant.get("domain_tags") or []))
        vtoks = _tokens(text)
        matched_hints: list[str] = []
        for what, toks in hint_tokens:
            if vtoks & toks:
                matched_hints.append(what)
        latency_aligned = bool(
            latency_dominant
            and any(kw in vtoks for kw in _LATENCY_DIRECTION_KEYWORDS)
        )
        if matched_hints or latency_aligned:
            out[name] = {
                "hints": matched_hints,
                "latency_aligned": latency_aligned,
            }
    return out


def priors_match_summary(
    variants: list[dict[str, Any]],
    hints: list[dict[str, Any]],
    *,
    primary_gap: str | None = None,
    max_rows: int = 12,
) -> str:
    """Render an advisory block flagging variants that match priors (empty when none; advisory ordering only)."""
    matches = match_variants_to_priors(
        variants, hints, primary_gap=primary_gap,
    )
    if not matches:
        return ""
    lines = [
        "Recently proposed variants that align with proven priors / the "
        + "dominant external gap. Treat as a reason to TRY THESE EARLIER — "
        + "advisory ordering only, NOT a score, NOT a gate.",
    ]
    for name in sorted(matches)[:max_rows]:
        info = matches[name]
        tags: list[str] = []
        if info.get("latency_aligned"):
            tags.append("aligns-with-latency-gap")
        for what in info.get("hints") or []:
            short = what if len(what) <= 60 else what[:57] + "..."
            tags.append(f"hint:{short}")
        lines.append(f"- {name}: " + "; ".join(tags))
    return "\n".join(lines)


def summarise_for_prompt(
    session_dir: Path, *, max_entries: int = 8,
) -> str:
    """Compact advisory block of proven priors for the orchestration prompt (empty when none; advisory only)."""
    hints = load_hints(session_dir)
    if not hints:
        return ""
    lines = [
        "Proven priors collected by the research scout. Treat as advisory "
        + "hints to try earlier — each carries a source.",
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
    "full_gap_summary",
    "gap_analysis",
    "load_competitor_target",
    "load_hints",
    "match_variants_to_priors",
    "priors_match_summary",
    "summarise_for_prompt",
    "write_competitor_target",
    "write_hints_skeleton",
]
