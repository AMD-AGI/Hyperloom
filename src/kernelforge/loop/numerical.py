# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task-owned numerical contracts and repeated-output evidence for kernel acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from typing import Any

EVIDENCE_PREFIX = "__FORGE_NUMERICAL__"
REQUEST_ENV = "FORGE_NUMERICAL_REQUEST"


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_contract(value: Any) -> dict:
    """Validate tolerances at the trusted task configuration boundary."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("numerical_validation requires schema_version: 1")
    repetitions = value.get("repetitions")
    if type(repetitions) is not int or repetitions < 3:
        raise ValueError("numerical_validation requires at least three repetitions")
    cases = value.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("numerical_validation requires explicit case/output/execution-mode IDs")
    for name, limits in cases.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(limits, dict):
            raise ValueError("invalid numerical_validation case")
        for key in ("max_oracle_error", "max_error_ratio", "error_floor"):
            if not _number(limits.get(key)):
                raise ValueError(f"numerical_validation {name}: {key} must be finite and nonnegative")
        if limits["max_error_ratio"] < 1:
            raise ValueError(f"numerical_validation {name}: max_error_ratio must be >= 1")
        if limits["error_floor"] > limits["max_oracle_error"]:
            raise ValueError(f"numerical_validation {name}: error_floor exceeds max_oracle_error")
    return value


def contract_digest(contract: dict) -> str:
    """Bind acceptance evidence to the immutable task tolerance and coverage policy."""
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def judge_evidence(output: str, contract: dict, request_id: str) -> tuple[bool, str, dict]:
    """Reject absent, stale, incomplete or numerically regressed evidence."""
    records = [line[len(EVIDENCE_PREFIX) :] for line in output.splitlines() if line.startswith(EVIDENCE_PREFIX)]
    if len(records) != 1:
        raise ValueError("correctness commands must emit exactly one numerical evidence record")
    evidence = json.loads(records[0])
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise ValueError("invalid numerical evidence schema")
    if evidence.get("request_id") != request_id:
        raise ValueError("numerical evidence does not belong to this validation invocation")
    rows = evidence.get("cases")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("numerical evidence must contain cases")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(name, str) for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("numerical evidence contains invalid or duplicate case IDs")
    if set(ids) != set(contract["cases"]):
        raise ValueError("numerical evidence coverage differs from the declared cases")
    failures = []
    for row in rows:
        name = row["id"]
        limits = contract["cases"][name]
        measurements = {}
        for role in ("source_before", "candidate", "source_after"):
            item = row.get(role)
            if not isinstance(item, dict) or type(item.get("finite")) is not bool:
                raise ValueError(f"{name}/{role}: missing finite-output evidence")
            if not item["finite"]:
                failures.append(f"{name}/{role}: nonfinite output")
            for metric in ("oracle_errors", "repeat_errors"):
                errors = item.get(metric)
                count = contract["repetitions"] - (metric == "repeat_errors")
                if not isinstance(errors, list) or len(errors) != count or any(not _number(x) for x in errors):
                    raise ValueError(f"{name}/{role}: invalid {metric} or repetition count")
            measurements[role] = item
            if max(item["oracle_errors"]) > limits["max_oracle_error"]:
                failures.append(f"{name}/{role}: mathematical error exceeds task tolerance")
        for metric in ("oracle_errors", "repeat_errors"):
            source_error = max(max(measurements[role][metric]) for role in ("source_before", "source_after"))
            bound = max(source_error * limits["max_error_ratio"], limits["error_floor"])
            error = max(measurements["candidate"][metric])
            if error > bound:
                failures.append(f"{name}: candidate {metric} {error:.6g} exceeds source-relative limit {bound:.6g}")
    evidence["contract_sha256"] = contract_digest(contract)
    detail = "; ".join(failures) if failures else f"numerical stability: {len(rows)} cases passed"
    return not failures, detail, evidence


def validate_source_independence(probe: dict, baseline: dict, contract: dict) -> None:
    """Check already-validated no-op evidence against the original source envelope.

    Every candidate case must expose the no-op while both source legs remain
    healthy. Identical errors in a correct compiler roundtrip are legitimate.
    """
    if probe["contract_sha256"] != baseline["contract_sha256"] or probe["contract_sha256"] != contract_digest(contract):
        raise ValueError("numerical execution probe changed the numerical contract")
    original = {row["id"]: row for row in baseline["cases"]}
    for row in probe["cases"]:
        name = row["id"]
        limits = contract["cases"][name]
        candidate = row["candidate"]
        if candidate["finite"] and max(candidate["oracle_errors"]) <= limits["max_oracle_error"]:
            raise ValueError(f"numerical execution probe {name}: candidate did not expose the no-op")
        for role in ("source_before", "source_after"):
            source = row[role]
            if not source["finite"] or max(source["oracle_errors"]) > limits["max_oracle_error"]:
                raise ValueError(f"numerical execution probe {name}/{role}: source failed with candidate disabled")
            for metric in ("oracle_errors", "repeat_errors"):
                initial = max(max(original[name][leg][metric]) for leg in ("source_before", "source_after"))
                bound = max(initial * limits["max_error_ratio"], limits["error_floor"])
                if max(source[metric]) > bound:
                    raise ValueError(
                        f"numerical execution probe {name}/{role}: {metric} exceeded the original source envelope"
                    )


def measure_outputs(run: Callable, reference: Any, *, repetitions: int) -> dict:
    """Measure one tensor using synchronized, independently owned CPU snapshots.

    ``run`` resets ABI-owned output when needed. Errors use the fixed reference
    L2 norm, or absolute L2 for an all-zero reference. Call per output, input and mode.
    """
    import torch

    if repetitions < 3:
        raise ValueError("numerical measurements require at least three repetitions")
    device = reference.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    expected = reference.detach().cpu().to(torch.float64).clone()
    if not bool(torch.isfinite(expected).all()):
        raise ValueError("numerical reference must be finite")
    denominator = float(expected.square().sum()) or 1.0
    oracle, repeat = [], []
    previous = None
    finite = True
    for _ in range(repetitions):
        actual = run()
        synchronize()
        snapshot = actual.detach().cpu().to(torch.float64).clone()
        if snapshot.shape != expected.shape:
            raise ValueError("numerical output shape differs from reference")
        finite = finite and bool(torch.isfinite(snapshot).all())
        # Preserve a JSON-safe failure record; the finite flag remains authoritative.
        oracle.append(math.sqrt(float((snapshot - expected).square().sum()) / denominator) if finite else 0.0)
        if previous is not None:
            repeat.append(math.sqrt(float((snapshot - previous).square().sum()) / denominator) if finite else 0.0)
        previous = snapshot
    return {"finite": finite, "oracle_errors": oracle, "repeat_errors": repeat}


def emit_evidence(cases: list[dict]) -> None:
    """Emit fresh structured measurements from a protected correctness driver."""
    print(
        EVIDENCE_PREFIX
        + json.dumps(
            {"schema_version": 1, "request_id": os.environ.get(REQUEST_ENV, ""), "cases": cases}, allow_nan=False
        ),
        flush=True,
    )
