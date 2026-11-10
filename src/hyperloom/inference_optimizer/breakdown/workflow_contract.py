# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Published workflow-evaluation contract and its canonical identity."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping

_CONTRACT_PACKAGE = "hyperloom.inference_optimizer.breakdown"
_CONTRACT_PATH = "contracts/workflow_contract.v3.json"
_SCHEMA_PATH = "contracts/session_breakdown.v6.workflow-evaluation.v3.schema.json"


@lru_cache(maxsize=1)
def _load_workflow_contract() -> dict[str, Any]:
    resource = files(_CONTRACT_PACKAGE).joinpath(_CONTRACT_PATH)
    return json.loads(resource.read_text(encoding="utf-8"))


def workflow_contract() -> dict[str, Any]:
    """Return an isolated copy of the immutable packaged workflow contract."""
    return deepcopy(_load_workflow_contract())


def workflow_contract_digest() -> str:
    """Return the SHA-256 hex identity of the packaged contract, over its documented canonicalization."""
    canonical = json.dumps(_load_workflow_contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def workflow_metadata(run_flags: Mapping[str, Any]) -> dict[str, Any]:
    """Build the author-time metadata block consumed by workflow evaluation."""
    contract = workflow_contract()
    return {
        "workflow_contract_version": contract["workflow_contract_version"],
        "contract_digest": workflow_contract_digest(),
        "run_flags": dict(run_flags),
        "phase_actions": contract["phase_actions"],
        "llm_proposable_actions": contract["llm_proposable_actions"],
        "coordinator_internal_actions": contract["coordinator_internal_actions"],
        "coordinator_reserved_actions": contract["coordinator_reserved_actions"],
        "kernel_lane_task_kinds": contract["kernel_lane_task_kinds"],
    }


def workflow_schema() -> dict[str, Any]:
    """Return the packaged workflow-evaluation JSON Schema."""
    resource = files(_CONTRACT_PACKAGE).joinpath(_SCHEMA_PATH)
    return json.loads(resource.read_text(encoding="utf-8"))


def event_semantics(event_type: str, status: str, ext: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project process and business semantics without inferring missing facts."""
    details = dict(ext or {})
    outcome: Any = None
    mapping = _load_workflow_contract()["event_business_outcomes"].get(str(event_type))
    if isinstance(mapping, Mapping):
        value: Any = details
        for part in str(mapping.get("path") or "").split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value not in (None, ""):
            # Preserve an authored but undeclared value so schema/CI rejects the
            # contract drift instead of silently converting it to missing.
            outcome = value
    failure = details.get("failure")
    blocked_by = details.get("blocked_by")
    return {
        "process_status": str(status),
        "business_outcome": outcome,
        # Preserve malformed authored values so schema validation rejects
        # contract drift instead of silently converting it to missing evidence.
        "failure": dict(failure) if isinstance(failure, Mapping) else failure,
        "blocked_by": str(blocked_by) if blocked_by else None,
    }


__all__ = [
    "event_semantics",
    "workflow_contract",
    "workflow_contract_digest",
    "workflow_metadata",
    "workflow_schema",
]
