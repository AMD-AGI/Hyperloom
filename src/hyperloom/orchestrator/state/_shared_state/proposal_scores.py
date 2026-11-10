# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The advisory multi-model proposal-score table: one row per variant, raters anonymized, no mean and no sorting."""

from __future__ import annotations

from typing import Any, NamedTuple

# A rater's reason, truncated for its table cell.
_REASON_CHARS = 80


class RaterPanel(NamedTuple):
    """One round's anonymized raters: their per-variant scores, their labels, and the order they render in."""

    scores: dict[str, Any]
    labels: dict[str, str]
    slugs: tuple[str, ...]


def rater_labels(rounds: list[dict[str, Any]]) -> dict[str, str]:
    """Map every rater slug seen across ``rounds`` — scoring or failed — to a stable ``rater_N`` label."""
    slugs: set[str] = set()
    for round_row in rounds:
        ensemble = round_row["ensemble_scores"]
        slugs.update(str(s) for s in (ensemble.get("models") or {}))
        slugs.update(str(s) for s in (ensemble.get("errors") or {}))
    return {slug: f"rater_{i}" for i, slug in enumerate(sorted(slugs), start=1)}


def _panel(ensemble: dict[str, Any], labels: dict[str, str]) -> RaterPanel:
    """The round's raters, dropping any slug that carries no label and ordering the rest by it."""
    scores = ensemble.get("models") or {}
    slugs = tuple(sorted((s for s in scores if s in labels), key=lambda s: labels[s]))
    return RaterPanel(scores=scores, labels=labels, slugs=slugs)


def _variant_names(round_row: dict[str, Any], panel: RaterPanel) -> list[str]:
    """Variant names in ``proposal_set`` order, then any a rater scored that the set omits."""
    ordered: list[str] = []
    seen: set[str] = set()
    for variant in round_row.get("proposal_set") or []:
        if isinstance(variant, dict):
            name = str(variant.get("name") or "")
            if name and name not in seen:
                ordered.append(name)
                seen.add(name)
    for per_rater in panel.scores.values():
        if isinstance(per_rater, dict):
            for name in per_rater:
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
    return ordered


def _score_cells(variant_name: str, panel: RaterPanel) -> str:
    """One variant's cell per rater, ``n/a`` where that rater did not score it."""
    parts: list[str] = []
    for slug in panel.slugs:
        per_rater = panel.scores.get(slug)
        if not isinstance(per_rater, dict):
            continue
        label = panel.labels[slug]
        cell = per_rater.get(variant_name)
        if isinstance(cell, dict) and cell.get("score") is not None:
            reason = str(cell.get("reason") or "").replace("\n", " ")
            if len(reason) > _REASON_CHARS:
                reason = reason[: _REASON_CHARS - 3] + "..."
            parts.append(f'{label}={float(cell["score"]):.1f} ("{reason}")')
        else:
            parts.append(f"{label}=n/a")
    return ", ".join(parts)


def round_rows(round_row: dict[str, Any], labels: dict[str, str]) -> list[str]:
    """One round's header line, a scored row per variant, and the raters that were unavailable for it."""
    ensemble = round_row["ensemble_scores"]
    panel = _panel(ensemble, labels)
    rows = [
        f"round={str(round_row.get('round_id') or '?')} "
        f"domain={str(round_row.get('domain') or '?')} "
        f"scale={str(ensemble.get('scale') or '0-10')}"
    ]
    rows.extend(f"  - {name}: " + _score_cells(name, panel) for name in _variant_names(round_row, panel))
    errors = ensemble.get("errors") or {}
    if errors:
        unavailable = ", ".join(sorted(labels.get(str(s), "rater_?") for s in errors))
        rows.append(f"  · raters unavailable this round: {unavailable}")
    return rows
