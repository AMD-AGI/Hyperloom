# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The public producer contract for framework apply-back.

The single place the rewrite protocol is defined: the version handshake a
consumer queries before committing to an integration, the rule turning a logical
operator identity into a legal Python builder symbol, the environment a
measurement driver is invoked with, and the apply-back manifest and validator.

It imports only from :mod:`kernelforge.loop.path_ownership`, which itself
imports nothing but stdlib, so the contract can be read without pulling in
agent, GPU, or git machinery.
"""

from __future__ import annotations

import hashlib
import keyword
import re
from kernelforge.loop.path_ownership import ATTEMPT_ROOT_DIR, is_producer_owned_path

REWRITE_PROTOCOL_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSIONS = (2,)
DRIVER_CONTRACT_VERSIONS = (1,)

SUPPORTED_FRAMEWORKS = ("aiter", "vllm", "sglang")

# Languages this producer can read a kernel in and port to FlyDSL. Every entry
# needs readable source, so a prebuilt binary or hand-written ASM is absent.
SUPPORTED_SOURCE_LANGUAGES = ("triton", "hip", "cuda", "cpp")

# The curated candidate kinds a consumer's profiler assigns, which routinely
# disagree with the file's language: a traced Triton kernel is reported as
# ``python`` with ``kernel_kind=triton``.
SUPPORTED_SOURCE_KINDS = ("triton", "hip_cpp")

# This producer can author or repair a non-conforming measurement driver from
# the caller's invocation evidence. A consumer that cannot synthesize a faithful
# driver for an operator reads this to decide whether handing the work over is
# an option, so it is advertised rather than assumed.
DRIVER_PREPARATION_SUPPORTED = True

# The outer rewrite exposes the same result sentinel as forge-loop so callers
# consume one backend-neutral contract.
RESULT_SENTINEL = "__FORGE_RESULT__"

ARTIFACT_KIND_FRAMEWORK_APPLYBACK = "framework_applyback"

# Correctness proven by the producer covers the standalone FlyDSL reference only.
VALIDATION_SCOPE_REFERENCE = "reference"

# Framework integration is validated by the consumer against a real serving
# workload, so the producer may only ever publish the pending status.
INTEGRATION_VALIDATION_PENDING = "pending"
PRODUCER_INTEGRATION_STATUSES = (INTEGRATION_VALIDATION_PENDING,)

# Producer-owned environment injected into every measurement driver invocation.
# A driver reads the candidate path and builder symbol from here instead of
# hardcoding producer file names or re-deriving the symbol.
ENV_SOURCE_KERNEL = "KERNELFORGE_REWRITE_SOURCE_KERNEL"
ENV_CANDIDATE_KERNEL = "KERNELFORGE_REWRITE_CANDIDATE_KERNEL"
ENV_BUILDER_SYMBOL = "KERNELFORGE_REWRITE_BUILDER_SYMBOL"
ENV_LOGICAL_OP = "KERNELFORGE_REWRITE_LOGICAL_OP"

# A readable slug stays short enough to keep the generated symbol legible; the
# digest, not the readable part, is what makes it unique.
_MAX_SLUG_CHARS = 40
_DIGEST_CHARS = 6
_PLAIN_IDENTIFIER = re.compile(r"[A-Za-z_][0-9A-Za-z_]*")


def capabilities() -> dict:
    """The machine-readable handshake a consumer uses to fail fast."""
    return {
        "rewrite_protocol_version": REWRITE_PROTOCOL_VERSION,
        "artifact_schema_versions": list(ARTIFACT_SCHEMA_VERSIONS),
        "driver_contract_versions": list(DRIVER_CONTRACT_VERSIONS),
        "frameworks": list(SUPPORTED_FRAMEWORKS),
        "source_languages": list(SUPPORTED_SOURCE_LANGUAGES),
        "source_kinds": list(SUPPORTED_SOURCE_KINDS),
        "result_sentinel": RESULT_SENTINEL,
        "driver_preparation": DRIVER_PREPARATION_SUPPORTED,
    }


def operator_slug(logical_op_name: str) -> str:
    """Derive a stable, legal identifier fragment from a logical operator name.

    A name that is already a plain ASCII identifier is used verbatim, so tasks
    and knowledge-base records keyed on simple names keep their symbol. Anything
    carrying a namespace, template, or punctuation is sanitized and suffixed
    with a digest of the original name, so distinct identities that sanitize
    alike still receive distinct symbols.
    """
    raw = str(logical_op_name or "").strip()
    if not raw:
        raise ValueError("logical operator name must not be empty")
    if len(raw) <= _MAX_SLUG_CHARS and _PLAIN_IDENTIFIER.fullmatch(raw) and not keyword.iskeyword(raw):
        return raw
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    if cleaned[:1].isdigit():
        cleaned = f"op_{cleaned}"
    cleaned = cleaned[:_MAX_SLUG_CHARS].strip("_") or "op"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"{cleaned}_{digest}"


def builder_symbol(logical_op_name: str) -> str:
    """The FlyDSL factory symbol a port must expose for ``logical_op_name``."""
    return f"build_{operator_slug(logical_op_name)}_module"


def driver_environment(
    *,
    source_kernel: str,
    candidate_kernel: str,
    logical_op_name: str,
) -> dict[str, str]:
    """Producer-owned variables every measurement driver invocation receives."""
    return {
        ENV_SOURCE_KERNEL: str(source_kernel),
        ENV_CANDIDATE_KERNEL: str(candidate_kernel),
        ENV_BUILDER_SYMBOL: builder_symbol(logical_op_name),
        ENV_LOGICAL_OP: str(logical_op_name),
    }


_REQUIRED_MANIFEST_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "artifact_kind": str,
    "validation_scope": str,
    "logical_op_name": str,
    "operator_slug": str,
    "builder_symbol": str,
    "source_entry": str,
    "reference_correctness_passed": bool,
    "reference_snr_db": (int, float, type(None)),
    "integration_validation_required": bool,
    "integration_validation_status": str,
    "base_commit": str,
    "commit_hash": str,
    "commit_ref": str,
    "flydsl_best_commit": str,
    "baseline_wall_ms": (int, float, type(None)),
    "best_wall_ms": (int, float, type(None)),
    "framework": str,
    "changed_files": list,
    "artifact_dir": str,
    "patch_path": str,
}

# ``correctness_passed`` meant "the standalone reference passed" while reading
# like "the framework patch passed". It is replaced by the explicit
# validation_scope / reference_correctness_passed / integration_validation_* set.
_FORBIDDEN_MANIFEST_FIELDS = ("correctness_passed",)

_RELATIVE_PATH_FIELDS = ("artifact_dir", "patch_path")

_REQUIRED_OUTER_RESULT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "success": bool,
    "applyback_required": bool,
    "applyback_ok": bool,
    "artifact_kind": str,
    "artifact_schema_version": int,
    "best_commit": str,
    "canonical_manifest": str,
    "canonical_patch_path": str,
    "canonical_files_root": str,
    "temporary_paths": list,
}


def _matches(value: object, expected: type | tuple[type, ...]) -> bool:
    # bool is an int subclass; a numeric field must not silently accept True.
    if isinstance(value, bool) and expected is not bool:
        return False
    return isinstance(value, expected)


def _check_relative(field: str, value: str) -> None:
    if not value:
        raise ValueError(f"apply-back manifest field is empty: {field}")
    if value.startswith("/") or ".." in value.split("/"):
        raise ValueError(f"apply-back manifest path escapes the campaign root: {field}={value}")


def validate_applyback_manifest(payload: dict) -> dict:
    """Return ``payload`` if it is a publishable apply-back manifest, else raise.

    Publication calls this before anything reaches disk, so a manifest that
    misdeclares its schema, omits a contract field, or claims an integration
    result the producer cannot prove fails the apply-back instead of shipping.
    """
    if not isinstance(payload, dict):
        raise ValueError("apply-back manifest must be a JSON object")

    version = payload.get("schema_version")
    if not _matches(version, int) or version not in ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported apply-back manifest schema version: {version!r}")

    for field in _FORBIDDEN_MANIFEST_FIELDS:
        if field in payload:
            raise ValueError(f"apply-back manifest must not carry ambiguous field: {field}")

    for field, expected in _REQUIRED_MANIFEST_FIELDS.items():
        if field not in payload:
            raise ValueError(f"apply-back manifest is missing field: {field}")
        if not _matches(payload[field], expected):
            raise ValueError(f"apply-back manifest field has the wrong type: {field}={payload[field]!r}")

    if payload["artifact_kind"] != ARTIFACT_KIND_FRAMEWORK_APPLYBACK:
        raise ValueError(f"unsupported apply-back artifact kind: {payload['artifact_kind']!r}")
    if payload["validation_scope"] != VALIDATION_SCOPE_REFERENCE:
        raise ValueError(f"unsupported apply-back validation scope: {payload['validation_scope']!r}")
    if payload["framework"] not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"unsupported apply-back framework: {payload['framework']!r}")
    status = payload["integration_validation_status"]
    if status not in PRODUCER_INTEGRATION_STATUSES:
        raise ValueError(f"the producer may not publish integration validation status: {status!r}")
    if not payload["commit_hash"]:
        raise ValueError("apply-back manifest is missing the apply-back commit")
    if not payload["base_commit"]:
        raise ValueError("apply-back manifest is missing the pristine base commit")
    if not payload["changed_files"]:
        raise ValueError("apply-back manifest declares no changed files")
    for changed in payload["changed_files"]:
        if not isinstance(changed, str):
            raise ValueError(f"apply-back manifest changed file is not a path: {changed!r}")
        _check_relative("changed_files", changed)
        if is_producer_owned_path(changed):
            raise ValueError(f"apply-back manifest publishes producer-owned state: {changed}")
    for field in _RELATIVE_PATH_FIELDS:
        _check_relative(field, payload[field])
    return payload


def validate_applyback_outer_result(payload: dict) -> dict:
    """Validate the outer result that points a consumer to one manifest bundle."""

    if not isinstance(payload, dict):
        raise ValueError("apply-back outer result must be a JSON object")
    for field, expected in _REQUIRED_OUTER_RESULT_FIELDS.items():
        if field not in payload:
            raise ValueError(f"apply-back outer result is missing field: {field}")
        if not _matches(payload[field], expected):
            raise ValueError(f"apply-back outer result field has the wrong type: {field}={payload[field]!r}")
    if payload["success"] is not True or payload["applyback_ok"] is not True:
        raise ValueError("apply-back outer result does not publish a successful patch")
    if payload["applyback_required"] is not True:
        raise ValueError("apply-back outer result does not require framework integration")
    if payload["artifact_kind"] != ARTIFACT_KIND_FRAMEWORK_APPLYBACK:
        raise ValueError(f"unsupported apply-back outer artifact: {payload['artifact_kind']!r}")
    if payload["artifact_schema_version"] not in ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported apply-back outer schema version: {payload['artifact_schema_version']!r}")
    for field in (
        "best_commit",
        "canonical_manifest",
        "canonical_patch_path",
        "canonical_files_root",
    ):
        if not payload[field]:
            raise ValueError(f"apply-back outer result field is empty: {field}")
    for temporary in payload["temporary_paths"]:
        if not isinstance(temporary, str):
            raise ValueError(f"apply-back temporary path is not a string: {temporary!r}")
        _check_relative("temporary_paths", temporary)
    return payload


def applyback_contract_example() -> dict:
    """Return one producer-authored example of both schema-2 documents."""

    base_commit = "b" * 40
    applyback_commit = "a" * 40
    manifest = validate_applyback_manifest(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND_FRAMEWORK_APPLYBACK,
            "validation_scope": VALIDATION_SCOPE_REFERENCE,
            "logical_op_name": "vllm::example",
            "operator_slug": operator_slug("vllm::example"),
            "builder_symbol": builder_symbol("vllm::example"),
            "source_entry": "example",
            "reference_correctness_passed": True,
            "reference_snr_db": 60.0,
            "integration_validation_required": True,
            "integration_validation_status": INTEGRATION_VALIDATION_PENDING,
            "base_commit": base_commit,
            "commit_hash": applyback_commit,
            "commit_ref": "refs/forge-rewrite/applyback/example-aaaaaaaaaaaa",
            "flydsl_best_commit": "c" * 40,
            "baseline_wall_ms": 2.0,
            "best_wall_ms": 1.0,
            "framework": "vllm",
            "changed_files": ["vllm/example.py"],
            "artifact_dir": "rewrite_applyback/best/iter_000",
            "patch_path": "rewrite_applyback/best/iter_000/forge.patch",
        }
    )
    outer_result = validate_applyback_outer_result(
        {
            "success": True,
            "applyback_required": True,
            "applyback_ok": True,
            "artifact_kind": ARTIFACT_KIND_FRAMEWORK_APPLYBACK,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "best_commit": applyback_commit,
            "canonical_manifest": "forge_experiments/rewrite_applyback/best/manifest.json",
            "canonical_patch_path": ("forge_experiments/rewrite_applyback/best/iter_000/forge.patch"),
            "canonical_files_root": ("forge_experiments/rewrite_applyback/best/iter_000/files"),
            "temporary_paths": [f"{ATTEMPT_ROOT_DIR}/example"],
        }
    )
    return {"manifest": manifest, "outer_result": outer_result}
