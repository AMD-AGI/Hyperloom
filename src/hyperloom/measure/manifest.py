# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The signed result manifest (``MeasurementResult`` v1) a runner writes to ``/out/manifest.json`` (3.4).

The manifest is read only after its signature has been verified over the raw bytes (see
:func:`hyperloom.measure.bundle.verify_bundle`); parsing here is strict all the same, because a manifest that is signed
but incoherent is a runner defect that must stop the line rather than reach the record book.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from hyperloom.measure.docs import (
    PREFIXED_SHA256_RE,
    SHA256_RE,
    DocumentError,
    Reader,
    canonical_json,
    parse_json_object,
    relative_path,
)

RESULT_SCHEMA = "hyperloom.measurement-result/v1"
FAILURE_CLASSES = ("infra", "unclear", "candidate")
#: Names the bundle itself owns; a manifest never lists them as result files.
RESERVED_NAMES = ("manifest.json", "manifest.sig")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16}")


@dataclass(frozen=True)
class EccDelta:
    correctable: int
    uncorrectable: int

    @classmethod
    def parse(cls, value: Any, path: str) -> "EccDelta":
        r = Reader(value, path)
        delta = cls(r.integer("correctable", minimum=0) or 0, r.integer("uncorrectable", minimum=0) or 0)
        r.done()
        return delta


@dataclass(frozen=True)
class RulePackFacts:
    """The rule pack the runner read (``hyperloom.common.rule_pack.rule_pack_identity``): its own version and the digest
    of its bytes."""

    version: str
    digest: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "RulePackFacts":
        r = Reader(value, path)
        facts = cls(r.text("version") or "", r.text("digest", pattern=PREFIXED_SHA256_RE) or "")
        r.done()
        return facts


@dataclass(frozen=True)
class RunnerFacts:
    """What the runner states about itself. ``rule_pack`` is absent from a manifest an older runner signed."""

    sandbox_id: str
    signing_key_fingerprint: str
    node_name: str | None
    image_digest: str | None
    gpu_model: str | None
    power_cap_w: float | None
    partition_mode: str | None
    hyperloom_version: str
    magpie_commit: str
    inferencex_commit: str
    gpu_ids: tuple[int, ...]
    ecc_delta: EccDelta | None
    rule_pack: RulePackFacts | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "RunnerFacts":
        r = Reader(value, path)
        ecc = r.optional("ecc_delta")
        pack = r.optional("rule_pack")
        facts = cls(
            sandbox_id=r.text("sandbox_id") or "",
            signing_key_fingerprint=r.text("signing_key_fingerprint", pattern=PREFIXED_SHA256_RE) or "",
            node_name=r.text("node_name", optional=True),
            image_digest=r.text("image_digest", optional=True),
            gpu_model=r.text("gpu_model", optional=True),
            power_cap_w=r.number("power_cap_w", minimum=0, optional=True),
            partition_mode=r.text("partition_mode", optional=True),
            hyperloom_version=r.text("hyperloom_version") or "",
            magpie_commit=r.text("magpie_commit") or "",
            inferencex_commit=r.text("inferencex_commit") or "",
            gpu_ids=r.sequence("gpu_ids", _gpu_id),
            ecc_delta=None if ecc is None else EccDelta.parse(ecc, r.at("ecc_delta")),
            rule_pack=None if pack is None else RulePackFacts.parse(pack, r.at("rule_pack")),
        )
        r.done()
        return facts


def _gpu_id(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DocumentError(f"{path}: expected a non-negative integer")
    return value


@dataclass(frozen=True)
class Outcome:
    returncode: int
    timed_out: bool
    error_class: str | None
    failure_class: str | None
    failure_reason: str | None
    stdout_tail: str
    stderr_tail: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "Outcome":
        r = Reader(value, path)
        outcome = cls(
            returncode=r.integer("returncode") or 0,
            timed_out=r.boolean("timed_out"),
            error_class=r.text("error_class", optional=True),
            failure_class=r.choice("failure_class", FAILURE_CLASSES, optional=True),
            failure_reason=r.text("failure_reason", optional=True),
            stdout_tail=r.string("stdout_tail"),
            stderr_tail=r.string("stderr_tail"),
        )
        r.done()
        failed = outcome.returncode != 0 or outcome.timed_out
        if failed and outcome.failure_class is None and outcome.error_class is None:
            raise DocumentError(f"{path}: a failed test carries a failure class")
        if not failed and (outcome.failure_class is not None or outcome.error_class is not None):
            raise DocumentError(f"{path}: a passing test carries no failure class")
        if (outcome.failure_class is None) != (outcome.failure_reason is None):
            raise DocumentError(f"{path}.failure_reason: set exactly when failure_class is set")
        return outcome


@dataclass(frozen=True)
class Timing:
    runner_started_unix: float
    spec_received_unix: float
    server_ready_unix: float | None
    finished_unix: float
    jit_compile_sec: float | None = None
    model_load_sec: float | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "Timing":
        r = Reader(value, path)
        timing = cls(
            runner_started_unix=r.number("runner_started_unix", minimum=0) or 0.0,
            spec_received_unix=r.number("spec_received_unix", minimum=0) or 0.0,
            server_ready_unix=r.number("server_ready_unix", minimum=0, optional=True),
            finished_unix=r.number("finished_unix", minimum=0) or 0.0,
            jit_compile_sec=r.number("jit_compile_sec", minimum=0, optional=True),
            model_load_sec=r.number("model_load_sec", minimum=0, optional=True),
        )
        r.done()
        return timing


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "ManifestFile":
        r = Reader(value, path)
        entry = cls(
            path=relative_path(r.raw("path"), r.at("path")),
            size=r.integer("size", minimum=0) or 0,
            sha256=r.text("sha256", pattern=SHA256_RE) or "",
        )
        r.done()
        if entry.path in RESERVED_NAMES:
            raise DocumentError(f"{path}.path: {entry.path} belongs to the bundle, not to the result files")
        return entry

    def to_document(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class ResultManifest:
    request_id: str
    spec_fingerprint: str
    spec_sha256: str
    surface_sha256: str | None
    runner: RunnerFacts
    outcome: Outcome
    platform_env_changed: tuple[str, ...]
    timing: Timing
    files: tuple[ManifestFile, ...]

    @classmethod
    def from_document(cls, document: Any) -> "ResultManifest":
        r = Reader(document, "manifest")
        if r.text("schema") != RESULT_SCHEMA:
            raise DocumentError(f"manifest.schema: expected {RESULT_SCHEMA}")
        manifest = cls(
            request_id=r.text("request_id") or "",
            spec_fingerprint=r.text("spec_fingerprint", pattern=_FINGERPRINT_RE) or "",
            spec_sha256=r.text("spec_sha256", pattern=SHA256_RE) or "",
            surface_sha256=r.text("surface_sha256", pattern=SHA256_RE, optional=True),
            runner=r.section("runner", RunnerFacts.parse),
            outcome=r.section("outcome", Outcome.parse),
            platform_env_changed=r.sequence("platform_env_changed", _name),
            timing=r.section("timing", Timing.parse),
            files=r.sequence("files", ManifestFile.parse),
        )
        r.done()
        paths = [entry.path for entry in manifest.files]
        if len(set(paths)) != len(paths):
            raise DocumentError("manifest.files: a path appears twice")
        _refuse_file_under_file(paths)
        return manifest

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ResultManifest":
        return cls.from_document(parse_json_object(raw, what="manifest"))

    def total_size(self) -> int:
        return sum(entry.size for entry in self.files)


def _name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise DocumentError(f"{path}: expected a non-empty string")
    return value


def _refuse_file_under_file(paths: list[str]) -> None:
    """``a`` and ``a/b`` cannot both be files: the second could never be written next to the first."""
    listed = set(paths)
    for path in paths:
        parts = path.split("/")
        for end in range(1, len(parts)):
            if "/".join(parts[:end]) in listed:
                raise DocumentError(f"manifest.files: {path} lies under another listed file")


def manifest_sha256(raw: bytes) -> str:
    """The identity of a result: the hash of the manifest's raw bytes, which the signature covers."""
    return hashlib.sha256(raw).hexdigest()


def encode_manifest(document: dict[str, Any]) -> bytes:
    """The bytes a runner writes for ``manifest.json``; canonical, so the same result always signs the same bytes."""
    return canonical_json(document).encode("utf-8")
