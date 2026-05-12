"""Knowledge base entry schema, validation, and conflict detection."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

VALID_CATEGORIES = {
    "backend_exploration",
    "kernel_optimization",
    "server_params",
    "pitfall",
    "benchmark_methodology",
    "architecture_constraint",
    "target_comparison",
    "framework_comparison",
    "lesson",
}

REQUIRED_FIELDS = {"category", "action", "lesson"}
OPTIONAL_FIELDS = {
    "model", "gpu", "framework", "tags", "result", "supersedes",
    "confidence", "source", "context",
}


def new_entry(
    category: str,
    action: str,
    lesson: str,
    model: str = "",
    gpu: str = "MI355X",
    framework: str = "",
    tags: Optional[list] = None,
    result: Optional[dict] = None,
    confidence: float = 0.9,
    source: str = "",
    context: str = "",
    supersedes: Optional[str] = None,
) -> dict:
    """Create a validated KB entry with auto-generated id and timestamp."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
        )
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "gpu": gpu,
        "framework": framework,
        "category": category,
        "action": action,
        "result": result or {},
        "tags": tags or [],
        "lesson": lesson,
        "supersedes": supersedes,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
        "context": context,
    }
    validate(entry)
    return entry


def validate(entry: dict) -> None:
    """Raise ValueError if the entry is malformed."""
    for field in REQUIRED_FIELDS:
        if not entry.get(field):
            raise ValueError(f"Missing required field: {field}")
    if entry.get("category") and entry["category"] not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {entry['category']}")
    if not isinstance(entry.get("tags", []), list):
        raise ValueError("'tags' must be a list")
    if not isinstance(entry.get("result", {}), dict):
        raise ValueError("'result' must be a dict")
    conf = entry.get("confidence", 0.9)
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {conf}")


def _tokenize(text: str) -> set:
    """Lowercased word tokens for similarity."""
    return set(text.lower().split())


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def detect_conflict(new_entry: dict, existing: list[dict], threshold: float = 0.55) -> list[dict]:
    """Find existing entries that potentially conflict with the new one.

    Conflict = same (model, category) AND similar action text AND divergent result.
    Returns list of conflicting entries with similarity scores.
    """
    conflicts = []
    for old in existing:
        if old.get("id") == new_entry.get("id"):
            continue
        if old.get("model") != new_entry.get("model"):
            continue
        if old.get("category") != new_entry.get("category"):
            continue

        sim = text_similarity(old.get("action", ""), new_entry.get("action", ""))
        if sim < threshold:
            continue

        old_status = old.get("result", {}).get("status", "")
        new_status = new_entry.get("result", {}).get("status", "")
        old_gain = old.get("result", {}).get("gain_pct", None)
        new_gain = new_entry.get("result", {}).get("gain_pct", None)

        result_diverges = False
        if old_status and new_status and old_status != new_status:
            result_diverges = True
        if old_gain is not None and new_gain is not None and abs(old_gain - new_gain) > 5.0:
            result_diverges = True
        if not result_diverges and sim > 0.8:
            old_has_result = bool(old.get("result"))
            new_has_result = bool(new_entry.get("result"))
            if old_has_result != new_has_result:
                result_diverges = True

        if result_diverges:
            conflicts.append({
                "existing_entry": old,
                "similarity": round(sim, 3),
                "reason": f"status: {old_status}→{new_status}, gain: {old_gain}→{new_gain}",
            })

    return conflicts


def resolve_conflict(new_entry: dict, conflict: dict) -> dict:
    """Apply automatic conflict resolution.

    Returns a resolution dict with action taken and updated entries.
    Rules:
    - Newer entry with controlled methodology supersedes
    - Both entries kept if they cover different operating points
    - Ambiguous cases flagged for review
    """
    old = conflict["existing_entry"]
    resolution = {
        "new_id": new_entry["id"],
        "old_id": old["id"],
        "similarity": conflict["similarity"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    new_is_controlled = any(
        kw in new_entry.get("context", "").lower()
        for kw in ["controlled", "fair a/b", "same session", "matched params"]
    )
    old_is_controlled = any(
        kw in old.get("context", "").lower()
        for kw in ["controlled", "fair a/b", "same session", "matched params"]
    )

    new_tags = set(new_entry.get("tags", []))
    old_tags = set(old.get("tags", []))
    different_conc = "different_conc" in new_tags or "different_conc" in old_tags

    if different_conc:
        resolution["action"] = "KEEP_BOTH"
        resolution["reason"] = "Different operating points (CONC/ISL/OSL)"
    elif new_is_controlled and not old_is_controlled:
        resolution["action"] = "SUPERSEDE"
        resolution["reason"] = "New entry has stricter methodology (controlled A/B)"
        new_entry["supersedes"] = old["id"]
        old["confidence"] = round(old.get("confidence", 0.9) * 0.5, 3)
    elif old_is_controlled and not new_is_controlled:
        resolution["action"] = "KEEP_OLD"
        resolution["reason"] = "Existing entry has stricter methodology"
        new_entry["confidence"] = round(new_entry.get("confidence", 0.9) * 0.5, 3)
    else:
        resolution["action"] = "FLAG_REVIEW"
        resolution["reason"] = "Ambiguous: same methodology quality, different results"

    resolution["old_confidence"] = old.get("confidence", 0.9)
    resolution["new_confidence"] = new_entry.get("confidence", 0.9)
    return resolution
