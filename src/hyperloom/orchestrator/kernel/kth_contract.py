# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's side of the Kernel Trust Harness (KTH) wire contract.

Canonical JSON, the framed subject digests and the normalized
``KernelCandidateEnvelope`` must reproduce KTH's byte-for-byte, or no
attestation could be checked against the candidate it claims to cover. KTH
owns every qualification decision; nothing here selects a check, a policy or a
verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REVIEWED_REQUEST_SCHEMA = "1.0.0"
ADAPTIVE_REQUEST_SCHEMA = "2.0.0"
ENVELOPE_SCHEMA = "1.0.0"

VERDICT_ELIGIBLE = "Eligible for performance evaluation"
VERDICT_BLOCKED = "Blocked"
VERDICT_INCONCLUSIVE = "Inconclusive"
#: ``kth-qualify`` exits with these for a completed qualification; anything else is an infrastructure failure.
VERDICT_EXIT_CODES = {VERDICT_ELIGIBLE: 0, VERDICT_BLOCKED: 2, VERDICT_INCONCLUSIVE: 3}

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
REVISION = re.compile(r"^[0-9a-f]{40,64}$")

#: Keys through which a candidate could steer its own qualification. KTH rejects them in an envelope; Hyperloom
#: rejects them anywhere in a publication so a reviewed-plan request, which carries no envelope, is held to the same
#: boundary.
CANDIDATE_CONTROL_KEYS = frozenset(
    {
        "acceptance_policy",
        "adapter",
        "argv",
        "atol",
        "attestation",
        "command",
        "commands",
        "detector_ids",
        "detectors",
        "exec",
        "executable",
        "expected_verdict",
        "import_path",
        "imports",
        "kth",
        "kth_plan_id",
        "kth_qualification",
        "max_abs",
        "max_rel",
        "max_ulp",
        "numerical_policy",
        "oracle_ids",
        "plan_id",
        "policy",
        "python_import",
        "reference",
        "reference_code",
        "reference_source",
        "rtol",
        "shell",
        "test_program",
        "thresholds",
        "tolerances",
        "verdict",
    }
)

#: ``(frame label, binding key)`` in the order KTH frames its ``kth-subject-v2`` digest.
_ADAPTIVE_SUBJECT_FIELDS = (
    ("envelope", "envelope_digest"),
    ("base_commit", "base_commit"),
    ("patch", "patch_digest"),
    ("binary", "binary_digest"),
    ("module", "module_digest"),
    ("loaded", "loaded_identity"),
    ("build", "build_digest"),
    ("compiler", "compiler"),
    ("autospec", "autospec_digest"),
    ("resolved_spec", "resolved_digest"),
    ("plan", "plan_digest"),
    ("kth_revision", "kth_revision"),
    ("environment", "environment_identity"),
)
ADAPTIVE_BINDING_KEYS = tuple(key for _, key in _ADAPTIVE_SUBJECT_FIELDS)

#: Normalized envelope sections that are objects and lists, in KTH's ``to_dict`` shape.
_ENVELOPE_OBJECTS = (
    "build",
    "artifact",
    "environment",
    "operation_schema",
    "graph",
    "implementation",
    "tracelens",
    "geak_harness",
    "hyperloom_context",
    "fallback",
    "model_config",
    "field_provenance",
)
_ENVELOPE_LISTS = ("changed_symbols", "tensors", "dispatch_paths", "call_sites", "tensor_lineage", "state_flow")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_digest(canonical_json(value).encode("utf-8"))


def framed_digest(parts: Sequence[tuple[str, bytes]], *, domain: str) -> str:
    """SHA-256 over a domain tag and length-prefixed ``(label, value)`` frames."""
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8") + b"\0")
    for name, value in parts:
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return "sha256:" + digest.hexdigest()


def reviewed_subject_digest(
    *,
    base_commit: str,
    patch: bytes,
    kernel_path: str,
    qualification_plan: Mapping[str, Any],
) -> str:
    """The ``kth-subject-v1`` digest a reviewed-plan attestation binds."""
    return framed_digest(
        [
            ("base_commit", base_commit.encode("ascii")),
            ("patch", patch),
            ("kernel_path", kernel_path.encode("utf-8")),
            ("qualification_plan", canonical_json(qualification_plan).encode("utf-8")),
        ],
        domain="kth-subject-v1",
    )


def adaptive_subject_digest(binding: Mapping[str, Any]) -> str:
    """The ``kth-subject-v2`` digest an adaptive attestation binds."""
    return framed_digest(
        [(label, str(binding.get(key) or "").encode("utf-8")) for label, key in _ADAPTIVE_SUBJECT_FIELDS],
        domain="kth-subject-v2",
    )


def environment_identity(environment: Mapping[str, Any]) -> str:
    """KTH's digest of the environment an envelope declares."""
    parts = [str(environment.get(key) or "") for key in ("hardware", "architecture", "software", "device_count")]
    return sha256_digest("|".join(parts).encode("utf-8"))


def candidate_control_fields(value: Any, where: str = "") -> Iterator[str]:
    """Yield the dotted path of every key a candidate may not use to steer qualification."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            if str(key).lower() in CANDIDATE_CONTROL_KEYS:
                yield path
            yield from candidate_control_fields(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from candidate_control_fields(item, f"{where}[{index}]")


@dataclass(frozen=True)
class CandidateArtifact:
    """The exact candidate KTH is asked to qualify: a patch applied on a known commit."""

    candidate_id: str
    attempt_id: str
    base_repository: str
    base_commit: str
    patch: bytes
    changed_paths: tuple[str, ...]

    @property
    def patch_digest(self) -> str:
        return sha256_digest(self.patch)


def build_candidate_envelope(
    artifact: CandidateArtifact,
    *,
    environment: Mapping[str, Any] | None = None,
    hyperloom_context: Mapping[str, Any] | None = None,
    geak_harness: Mapping[str, Any] | None = None,
    field_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a ``KernelCandidateEnvelope`` already in KTH's normalized form.

    Every section KTH normalizes is present, so parsing and re-serializing the
    envelope on the KTH side reproduces it and :func:`envelope_digest` equals
    the digest KTH binds. Sections Hyperloom has no trusted source for stay
    empty rather than guessed.
    """
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA,
        "candidate_id": artifact.candidate_id,
        "attempt_id": artifact.attempt_id,
        "base_repository": artifact.base_repository,
        "base_commit": artifact.base_commit,
        "patch_base64": base64.b64encode(artifact.patch).decode("ascii"),
        "patch_digest": artifact.patch_digest,
        "patch_path": "",
        "changed_paths": list(artifact.changed_paths),
    }
    envelope.update({name: [] for name in _ENVELOPE_LISTS})
    envelope.update({name: {} for name in _ENVELOPE_OBJECTS})
    envelope["environment"] = dict(environment or {})
    envelope["hyperloom_context"] = dict(hyperloom_context or {})
    envelope["geak_harness"] = dict(geak_harness or {})
    envelope["field_provenance"] = {key: dict(value) for key, value in (field_provenance or {}).items()}
    return envelope


def envelope_digest(envelope: Mapping[str, Any]) -> str:
    """The digest KTH computes over a normalized envelope; the inline patch is bound through ``patch_digest``."""
    return digest_json({key: value for key, value in envelope.items() if key != "patch_base64"})


__all__ = [
    "ADAPTIVE_BINDING_KEYS",
    "ADAPTIVE_REQUEST_SCHEMA",
    "CANDIDATE_CONTROL_KEYS",
    "ENVELOPE_SCHEMA",
    "REVIEWED_REQUEST_SCHEMA",
    "REVISION",
    "SAFE_ID",
    "VERDICT_BLOCKED",
    "VERDICT_ELIGIBLE",
    "VERDICT_EXIT_CODES",
    "VERDICT_INCONCLUSIVE",
    "CandidateArtifact",
    "adaptive_subject_digest",
    "build_candidate_envelope",
    "candidate_control_fields",
    "canonical_json",
    "digest_json",
    "envelope_digest",
    "environment_identity",
    "framed_digest",
    "reviewed_subject_digest",
    "sha256_digest",
]
