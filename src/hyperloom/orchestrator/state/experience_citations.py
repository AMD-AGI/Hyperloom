# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Experiences a proposal says shaped it, kept only when the agent that wrote it was shown them."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# adopt: did the cited change; adapt: did it modified; avoid: did not do it because of it;
# contrast: chose a different change designed against it.
STANCES = ("adopt", "adapt", "avoid", "contrast")
MAX_CLAIM_CHARS = 500


def shown_ids(rendered_refs: Iterable[Any] | None) -> frozenset[str]:
    """The Experience ids a read rendered into the deciding agent's prompt."""
    return frozenset(str(ref.get("id") or "").strip() for ref in (rendered_refs or ()) if isinstance(ref, Mapping)) - {
        ""
    }


def normalize_citations(raw: Any, shown: frozenset[str]) -> list[dict[str, str]]:
    """Keep each well-formed citation of an Experience in ``shown``; anything else is the agent's error and dropped."""
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, Mapping):
            continue
        experience_id = str(item.get("id") or "").strip()
        stance = str(item.get("stance") or "").strip().lower()
        if experience_id not in shown or stance not in STANCES or (experience_id, stance) in seen:
            continue
        seen.add((experience_id, stance))
        claim = " ".join(str(item.get("claim") or "").split())[:MAX_CLAIM_CHARS]
        citations.append({"id": experience_id, "stance": stance, "claim": claim})
    return citations


__all__ = ["MAX_CLAIM_CHARS", "STANCES", "normalize_citations", "shown_ids"]
