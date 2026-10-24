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

from hyperloom.contracts.measurement import (
    CARD_OCCUPANCY_STATES,
    CARD_UNKNOWN,
    MAX_PHASE_SECONDS,
    MeasurementError,
    PhaseTiming,
)
from hyperloom.measure.jit_kinds import STATES as JIT_STATES
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
    """The rule pack the runner read (``hyperloom.common.outside_facts.rule_pack.rule_pack_identity``): its own version
    and the digest of its bytes."""

    version: str
    digest: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "RulePackFacts":
        r = Reader(value, path)
        facts = cls(r.text("version") or "", r.text("digest", pattern=PREFIXED_SHA256_RE) or "")
        r.done()
        return facts


@dataclass(frozen=True)
class CardOccupancyFacts:
    """What the runner says of who else used its cards: ``clean`` or ``unknown`` with the reason it could not look, and
    how many looks read the list."""

    state: str
    reason: str | None
    checks: int

    @classmethod
    def parse(cls, value: Any, path: str) -> "CardOccupancyFacts":
        r = Reader(value, path)
        state = r.choice("state", CARD_OCCUPANCY_STATES) or ""
        reason = r.text("reason", optional=True)
        facts = cls(state, reason, r.integer("checks", minimum=0) or 0)
        r.done()
        if (state == CARD_UNKNOWN) != (reason is not None):
            raise DocumentError(f"{path}: an unknown card occupancy states why, and any other state has no reason")
        return facts


@dataclass(frozen=True)
class RunnerFacts:
    """What the runner states about itself. ``rule_pack`` and ``card_occupancy`` are absent from a manifest an older runner
    signed, and ``card_occupancy`` from one a runner without a card signed."""

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
    card_occupancy: CardOccupancyFacts | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "RunnerFacts":
        r = Reader(value, path)
        ecc = r.optional("ecc_delta")
        pack = r.optional("rule_pack")
        occupancy = r.optional("card_occupancy")
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
            card_occupancy=None if occupancy is None else CardOccupancyFacts.parse(occupancy, r.at("card_occupancy")),
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


#: How many phases a manifest may state: a test of a sweep has a few dozen.
MAX_PHASES = 4096


def _phase(value: Any, path: str) -> PhaseTiming:
    r = Reader(value, path)
    started, seconds = r.number("started_unix", minimum=0) or 0.0, r.number("seconds", minimum=0) or 0.0
    name, point = r.text("name") or "", r.integer("point", minimum=0, optional=True)
    r.done()
    try:
        return PhaseTiming(name, started, seconds, point=point)
    except MeasurementError as exc:
        raise DocumentError(f"{path}: {exc}") from None


def _phases(value: Any, path: str) -> tuple[PhaseTiming, ...]:
    """The runner's phases, lengths on its own clock; a manifest an older runner signed has none."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DocumentError(f"{path}: expected an array")
    if len(value) > MAX_PHASES:
        raise DocumentError(f"{path}: more than {MAX_PHASES} phases")
    phases = tuple(_phase(item, f"{path}[{index}]") for index, item in enumerate(value))
    if sum(phase.seconds for phase in phases) > MAX_PHASE_SECONDS:
        raise DocumentError(f"{path}: the phases add up to more than a year")
    return phases


@dataclass(frozen=True)
class Timing:
    runner_started_unix: float
    spec_received_unix: float
    server_ready_unix: float | None
    finished_unix: float
    jit_compile_sec: float | None = None
    model_load_sec: float | None = None
    phases: tuple[PhaseTiming, ...] = ()

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
            phases=_phases(r.optional("phases"), r.at("phases")),
        )
        r.done()
        return timing


@dataclass(frozen=True)
class JitContribution:
    """What a test staged for the shared cache of a kind: the hash of the snapshot list it extends and what was new."""

    sha256: str
    files: int
    bytes: int

    @classmethod
    def parse(cls, value: Any, path: str) -> "JitContribution":
        r = Reader(value, path)
        gave = cls(
            r.text("sha256", pattern=SHA256_RE) or "",
            r.integer("files", minimum=0) or 0,
            r.integer("bytes", minimum=0) or 0,
        )
        r.done()
        return gave


@dataclass(frozen=True)
class JitKindFacts:
    """How one kind of compiled-code cache went in a test: its key, the state it was found in and what it gave."""

    kind: str
    key: str | None
    state: str
    seed_files: int
    seed_bytes: int
    seed_snapshot: str | None
    contribution: JitContribution | None

    @classmethod
    def parse(cls, value: Any, path: str) -> "JitKindFacts":
        r = Reader(value, path)
        contribution = r.optional("contribution")
        facts = cls(
            kind=r.text("kind") or "",
            key=r.text("key", pattern=SHA256_RE, optional=True),
            state=r.choice("state", JIT_STATES) or "",
            seed_files=r.integer("seed_files", minimum=0) or 0,
            seed_bytes=r.integer("seed_bytes", minimum=0) or 0,
            seed_snapshot=r.text("seed_snapshot", pattern=SHA256_RE, optional=True),
            contribution=None if contribution is None else JitContribution.parse(contribution, r.at("contribution")),
        )
        r.done()
        return facts


@dataclass(frozen=True)
class JitCacheFacts:
    """What the runner signs of the shared compiled-code cache: each kind, and how much aiter built while the server ran
    (``builds_seen`` is zero when everything it needed was found built)."""

    kinds: tuple[JitKindFacts, ...]
    builds_seen: int
    jit_compile_sec: float

    @classmethod
    def parse(cls, value: Any, path: str) -> "JitCacheFacts":
        r = Reader(value, path)
        facts = cls(
            kinds=r.sequence("kinds", JitKindFacts.parse),
            builds_seen=r.integer("builds_seen", minimum=0) or 0,
            jit_compile_sec=r.number("jit_compile_sec", minimum=0) or 0.0,
        )
        r.done()
        names = [kind.kind for kind in facts.kinds]
        if len(set(names)) != len(names):
            raise DocumentError(f"{path}.kinds: a kind appears twice")
        return facts


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
    jit_cache: JitCacheFacts | None = None

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
            jit_cache=_optional(r, "jit_cache", JitCacheFacts.parse),
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


def _optional(reader: Reader, key: str, parse: Any) -> Any:
    value = reader.optional(key)
    return None if value is None else parse(value, reader.at(key))


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
