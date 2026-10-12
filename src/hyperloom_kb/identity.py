"""Stable identity derivation for Experience records."""

from __future__ import annotations

import uuid

EXPERIENCE_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "urn:hyperloom-kb:experience:v1",
)


def derive_experience_id(producer: str, run_id: str, seq: int) -> str:
    """Derive the only valid id for one producer/run sequence position."""

    normalized_producer = producer.strip()
    normalized_run_id = run_id.strip()
    if not normalized_producer:
        raise ValueError("producer must be a non-empty string")
    if not normalized_run_id:
        raise ValueError("run_id must be a non-empty string")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("seq must be a non-negative integer")
    name = f"{normalized_producer}\0{normalized_run_id}\0{seq}"
    return f"exp-{uuid.uuid5(EXPERIENCE_ID_NAMESPACE, name).hex}"


__all__ = ["EXPERIENCE_ID_NAMESPACE", "derive_experience_id"]
