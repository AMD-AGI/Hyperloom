# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measurement instruction sheet (``MeasurementSpec`` v2): what the application asks a runner to measure.

The application writes the sheet to ``/in/spec.json`` of the authoritative sandbox. Its identity has two forms: the
sha256 of the exact bytes written (:meth:`MeasurementSpec.sha256`, recorded in the result manifest) and a short
*fingerprint* of the measurement content that is stable across sessions (:func:`fingerprint`), so the same measurement
deduplicates inside a campaign and compares across campaigns.

A v2 sheet carries the session's frozen base launch whole (``procedure.base_launch``) and the application's
inference_suite staging as its normalised manifest (``tools.inference_suite.normalised``) and that manifest's sha256
(``tools.inference_suite.staging``); a hosted runner refuses a test whose own published staging differs from it and
its result manifest names its own normalised sha256. A v1 sheet, written while tests
ran through Magpie, named a Magpie commit and a benchmark backend instead; the journal and the archive keep those, so a
v1 sheet still loads, but nothing writes one.

The result manifest lives in :mod:`hyperloom.measure.manifest`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.benchmark_config import BenchmarkPoint, BenchmarkPointError
from hyperloom.common.launch_credentials import credentials_in_the_clear
from hyperloom.contracts.measurement import Purpose
from hyperloom.measure.checkpoint import CheckpointPlan
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.surface import POINT_ENV_KEYS, REQUEST_ENV_KEYS
from inference_suite.contracts import BASE_LAUNCH_SCHEMA, BaseLaunch, NormalisedError, NormalisedManifest
from hyperloom.measure.docs import (
    PREFIXED_SHA256_RE,
    SHA256_RE,
    DocumentError,
    Reader,
    canonical_json,
    env_name,
    parse_json_object,
    relative_path,
    string_map,
)

SPEC_SCHEMA_V1 = "hyperloom.measurement-spec/v1"
SPEC_SCHEMA = "hyperloom.measurement-spec/v2"
SPEC_SCHEMAS = (SPEC_SCHEMA_V1, SPEC_SCHEMA)
#: The most bytes the carried base launch may take as canonical JSON; a resolved base is a few KiB, and the sheet is
#: written whole into a sandbox and journalled with every attempt.
MAX_BASE_LAUNCH_BYTES = 64 * 1024
_BASE_LAUNCH = "base_launch"
_BENCHMARK_BACKEND = "benchmark_backend"
_INFERENCEX = "inferencex"
_INFERENCE_SUITE = "inference_suite"
_NORMALISED = "normalised"
_KIND = "kind"
_UPSERT = "upsert"
_RUNTIME = "runtime"
_BUILDS = "builds"
_MANIFEST_SHA256 = "manifest_sha256"
_ACCURACY = "accuracy"
_SHORT_PROMPTS = "short_prompts"
_LONG_CONTEXT = "long_context"
_CHECKPOINT = "checkpoint"
_RESULT_DIR = "result_dir"
#: A key the sheet of a test that ran a greedy decode pass carried. The pass is gone and the key means nothing, but the
#: journal keeps those sheets and a result is verified again from them, so the key is read and dropped.
_REMOVED_DECODE_PASS = "decode_pass"
KINDS = ("test", "profile", _ACCURACY, "repro")
PURPOSES = tuple(purpose.value for purpose in Purpose)
DEFAULT_MAX_TOTAL_BYTES = 4 * 1024**3
#: The most levels a config YAML may nest, aliases expanded: far more than a config needs, and far fewer than the levels
#: at which what reads the loaded value next runs out of recursion.
_MAX_YAML_DEPTH = 128
#: What a loaded YAML value nests in: mappings, sequences, and the pairs of ``!!omap`` and ``!!pairs``.
_NESTED = (Mapping, list, tuple)
_TOO_DEEP = f"spec.config.yaml: nests more than {_MAX_YAML_DEPTH} levels deep"
#: The platform variables a sheet may carry reference values for: RCCL and HSA variables SaFE adds to a Pod.
#: ``NCCL_MIN_NCHANNELS`` changes how every RCCL collective is split, so a sandbox without the Pod's value measures
#: under other conditions than the local run. The interface-selection variables (``NCCL_SOCKET_IFNAME``,
#: ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``, ``GLOO_SOCKET_IFNAME``) are not listed: on one node they do not touch the
#: data path, and a sheet without them would delete them where a multi-node job needs them.
PLATFORM_ENV_NAMES = (
    "HSA_ENABLE_SDMA",
    "NCCL_CHECKS_DISABLE",
    "NCCL_CROSS_NIC",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
    "NCCL_IB_QPS_PER_CONNECTION",
    "NCCL_IB_SL",
    "NCCL_MIN_NCHANNELS",
)
#: Keys of the materialised YAML whose value is a location, replaced by a placeholder in the fingerprint.
#: ``benchmark.envs`` keys that differ from point to point of one arm: the request values and the counts that follow
#: from the concurrency level.
POINT_VALUE_KEYS = frozenset((*REQUEST_ENV_KEYS, *POINT_ENV_KEYS))
PATH_KEYS = ("pid_dir", "inferencex_path", _RESULT_DIR, "output_dir", "output_path", "log_dir")

_ATTACHMENT_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
_COMMIT = r"[0-9a-f]{40}(?:[0-9a-f]{24})?"
_FINGERPRINT_LEN = 16


_ID_RE = re.compile(_ATTACHMENT_ID)
ATTACHMENT_ID_RE = _ID_RE  # public name for callers that vet an attachment id before a sheet is built
_COMMIT_RE = re.compile(_COMMIT)
_MODE_RE = re.compile(r"0[0-7]{3}")


@dataclass(frozen=True)
class Meta:
    purpose: str
    branch_id: str
    parent_commit: str
    priority: int | None = None
    retest_of: str | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "Meta":
        r = Reader(value, path)
        meta = cls(
            purpose=r.choice("purpose", PURPOSES) or "",
            branch_id=r.text("branch_id") or "",
            parent_commit=r.text("parent_commit") or "",
            priority=r.integer("priority", minimum=0, optional=True),
            retest_of=r.text("retest_of", optional=True),
        )
        r.done()
        if meta.priority is not None and meta.priority > 4:
            raise DocumentError(f"{path}.priority: must be 0 to 4")
        if (meta.purpose == "confirm") != (meta.retest_of is not None):
            raise DocumentError(f"{path}.retest_of: given exactly for purpose 'confirm'")
        return meta

    def to_document(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "priority": self.priority,
            "branch_id": self.branch_id,
            "parent_commit": self.parent_commit,
            "retest_of": self.retest_of,
        }


@dataclass(frozen=True)
class Hardware:
    gpu_type: str
    gpu_count: int
    power_cap_w: float | None
    partition_mode: str | None

    @classmethod
    def parse(cls, value: Any, path: str) -> "Hardware":
        r = Reader(value, path)
        hardware = cls(
            gpu_type=r.text("gpu_type") or "",
            gpu_count=r.integer("gpu_count", minimum=1) or 1,
            power_cap_w=r.number("power_cap_w", minimum=0, optional=True),
            partition_mode=r.text("partition_mode", optional=True),
        )
        r.done()
        return hardware

    def to_document(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "power_cap_w": self.power_cap_w,
            "partition_mode": self.partition_mode,
        }


@dataclass(frozen=True)
class ImageRef:
    ref: str
    digest: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "ImageRef":
        r = Reader(value, path)
        image = cls(ref=r.text("ref") or "", digest=r.text("digest", pattern=PREFIXED_SHA256_RE) or "")
        r.done()
        return image

    def to_document(self) -> dict[str, Any]:
        return {"ref": self.ref, "digest": self.digest}


@dataclass(frozen=True)
class Config:
    yaml: str
    sha256: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "Config":
        r = Reader(value, path)
        config = cls(yaml=r.text("yaml") or "", sha256=r.text("sha256", pattern=SHA256_RE) or "")
        r.done()
        try:
            actual = hashlib.sha256(config.yaml.encode()).hexdigest()
        except UnicodeEncodeError:  # a lone surrogate, which a JSON escape can spell and UTF-8 cannot
            raise DocumentError(f"{path}.yaml: not Unicode text that UTF-8 can encode") from None
        if actual != config.sha256:
            raise DocumentError(f"{path}.sha256: does not match the YAML text")
        return config

    @classmethod
    def of(cls, yaml_text: str) -> "Config":
        return cls(yaml=yaml_text, sha256=hashlib.sha256(yaml_text.encode()).hexdigest())

    def to_document(self) -> dict[str, Any]:
        return {"yaml": self.yaml, "sha256": self.sha256}


@dataclass(frozen=True)
class Attachment:
    id: str
    kind: str
    blob: str
    manifest_sha256: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "Attachment":
        r = Reader(value, path)
        attachment = cls(
            id=r.text("id", pattern=_ID_RE) or "",
            kind=r.choice(_KIND, ("file", "dir")) or "",
            blob=r.text("blob", pattern=PREFIXED_SHA256_RE) or "",
            manifest_sha256=r.text(_MANIFEST_SHA256, pattern=SHA256_RE) or "",
        )
        r.done()
        return attachment

    def to_document(self) -> dict[str, Any]:
        return {"id": self.id, _KIND: self.kind, "blob": self.blob, _MANIFEST_SHA256: self.manifest_sha256}


@dataclass(frozen=True)
class ToolCommits:
    """The tools a sheet names: the InferenceX commit, and the normalised manifest of the application's inference_suite
    staging (``staging``). A v1 sheet named a Magpie commit (``magpie``) and no staging."""

    inferencex: str
    staging: NormalisedManifest | None = None
    magpie: str | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "ToolCommits":
        r = Reader(value, path)
        tools = cls(inferencex=r.section(_INFERENCEX, _commit_of), staging=r.section(_INFERENCE_SUITE, _staging_of))
        r.done()
        return tools

    @classmethod
    def parse_v1(cls, value: Any, path: str) -> "ToolCommits":
        r = Reader(value, path)
        tools = cls(inferencex=r.section(_INFERENCEX, _commit_of), magpie=r.section("magpie", _commit_of))
        r.done()
        return tools

    def to_document(self) -> dict[str, Any]:
        if self.magpie is not None:
            return {"magpie": {"commit": self.magpie}, _INFERENCEX: {"commit": self.inferencex}}
        staging = (
            None
            if self.staging is None
            else {"staging": self.staging.sha256(), _NORMALISED: self.staging.to_document()}
        )
        return {_INFERENCE_SUITE: staging, _INFERENCEX: {"commit": self.inferencex}}


def _staging_of(value: Any, path: str) -> NormalisedManifest:
    """The normalised manifest a sheet carries, refused when it is not the one its ``staging`` sha256 names."""
    r = Reader(value, path)
    sha256 = r.text("staging", pattern=SHA256_RE) or ""
    try:
        staging = NormalisedManifest.from_document(r.raw(_NORMALISED))
    except NormalisedError as exc:
        raise DocumentError(f"{r.at(_NORMALISED)}: {exc}") from None
    r.done()
    if staging.sha256() != sha256:
        raise DocumentError(f"{r.at('staging')}: {sha256} is not the sha256 of the normalised manifest it carries")
    return staging


def _commit_of(value: Any, path: str) -> str:
    r = Reader(value, path)
    commit = r.text("commit", pattern=_COMMIT_RE) or ""
    r.done()
    return commit


@dataclass(frozen=True)
class EnvEntry:
    path: str
    op: str
    blob: str | None = None
    mode: str | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "EnvEntry":
        r = Reader(value, path)
        op = r.choice("op", (_UPSERT, "delete")) or ""
        entry = cls(
            path=relative_path(r.raw("path"), r.at("path")),
            op=op,
            blob=r.text("blob", pattern=PREFIXED_SHA256_RE, optional=True),
            mode=r.text("mode", pattern=_MODE_RE, optional=True),
        )
        r.done()
        if op == _UPSERT and (entry.blob is None or entry.mode is None):
            raise DocumentError(f"{path}: an upsert names its blob and mode")
        if op == "delete" and (entry.blob is not None or entry.mode is not None):
            raise DocumentError(f"{path}: a delete carries neither blob nor mode")
        return entry

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"path": self.path, "op": self.op}
        if self.op == _UPSERT:
            document.update(blob=self.blob, mode=self.mode)
        return document


@dataclass(frozen=True)
class EnvRoot:
    id: str
    kind: str
    import_name: str
    dist_name: str
    dist_version: str
    entries: tuple[EnvEntry, ...]

    @classmethod
    def parse(cls, value: Any, path: str) -> "EnvRoot":
        r = Reader(value, path)
        dist = Reader(r.raw("dist"), r.at("dist"))
        dist_name, dist_version = dist.text("name") or "", dist.text("version") or ""
        dist.done()
        root = cls(
            id=r.text("id", pattern=_ID_RE) or "",
            kind=r.choice(_KIND, ("package", _RUNTIME)) or "",
            import_name=r.text("import_name") or "",
            dist_name=dist_name,
            dist_version=dist_version,
            entries=r.sequence("entries", EnvEntry.parse),
        )
        r.done()
        paths = [entry.path for entry in root.entries]
        if len(set(paths)) != len(paths):
            raise DocumentError(f"{path}.entries: a path appears twice")
        return root

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            _KIND: self.kind,
            "import_name": self.import_name,
            "dist": {"name": self.dist_name, "version": self.dist_version},
            "entries": [entry.to_document() for entry in self.entries],
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    snapshot: str
    manifest_sha256: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "RuntimeSnapshot":
        r = Reader(value, path)
        runtime = cls(
            snapshot=r.text("snapshot") or "", manifest_sha256=r.text(_MANIFEST_SHA256, pattern=SHA256_RE) or ""
        )
        r.done()
        return runtime

    def to_document(self) -> dict[str, Any]:
        return {"snapshot": self.snapshot, _MANIFEST_SHA256: self.manifest_sha256}


@dataclass(frozen=True)
class BuildRef:
    """The ready native build a ``runtime`` root's outputs come from: its key in the build store and the SHA-256 of the
    signed manifest it was built to."""

    root_id: str
    key: str
    manifest_sha256: str

    @classmethod
    def parse(cls, value: Any, path: str) -> "BuildRef":
        r = Reader(value, path)
        ref = cls(
            root_id=r.text("root_id", pattern=_ID_RE) or "",
            key=r.text("key", pattern=SHA256_RE) or "",
            manifest_sha256=r.text(_MANIFEST_SHA256, pattern=SHA256_RE) or "",
        )
        r.done()
        return ref

    def to_document(self) -> dict[str, Any]:
        return {"root_id": self.root_id, "key": self.key, _MANIFEST_SHA256: self.manifest_sha256}


@dataclass(frozen=True)
class Environment:
    """The overlay a runner rebuilds. ``builds`` name the ready native build of each ``runtime`` root, one each, and of
    a package root whose JIT modules a CPU build made (aiter's); a sheet with none leaves the key out, so a
    runner that does not know it reads every other sheet as before and refuses one that has builds."""

    roots: tuple[EnvRoot, ...] = ()
    runtime: RuntimeSnapshot | None = None
    blob_root: str = ""
    builds: tuple[BuildRef, ...] = ()

    @classmethod
    def parse(cls, value: Any, path: str) -> "Environment":
        r = Reader(value, path)
        runtime_raw = r.optional(_RUNTIME)
        builds_raw = r.optional(_BUILDS)
        environment = cls(
            roots=r.sequence("roots", EnvRoot.parse),
            runtime=None if runtime_raw is None else RuntimeSnapshot.parse(runtime_raw, r.at(_RUNTIME)),
            blob_root=r.text("blob_root") or "",
            builds=() if builds_raw is None else r.sequence(_BUILDS, BuildRef.parse),
        )
        r.done()
        if builds_raw is not None and not environment.builds:
            raise DocumentError(f"{path}.builds: an empty list; a sheet with no build leaves the key out")
        ids = [root.id for root in environment.roots]
        if len(set(ids)) != len(ids):
            raise DocumentError(f"{path}.roots: an id appears twice")
        kinds = {root.id: root.kind for root in environment.roots}
        runtime_ids = sorted(root_id for root_id, kind in kinds.items() if kind == _RUNTIME)
        built = [ref.root_id for ref in environment.builds]
        if len(set(built)) != len(built) or not set(runtime_ids) <= set(built) <= set(kinds):
            raise DocumentError(
                f"{path}.builds: each runtime root needs exactly one build, a package root may have one (of the "
                f"modules its JIT builds), and each build names a root (runtime roots {runtime_ids}, builds {built})"
            )
        return environment

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "roots": [root.to_document() for root in self.roots],
            _RUNTIME: None if self.runtime is None else self.runtime.to_document(),
            "blob_root": self.blob_root,
        }
        if self.builds:
            document[_BUILDS] = [ref.to_document() for ref in self.builds]
        return document


def _short_prompts(value: Any, path: str) -> ShortPromptPlan | None:
    """The sheet's short-prompt check, absent from a sheet that asks for none."""
    return None if value is None else ShortPromptPlan.parse(value, path)


def _long_context(value: Any, path: str) -> LongContextPlan | None:
    """The sheet's long-context check, absent from a sheet that asks for none."""
    return None if value is None else LongContextPlan.parse(value, path)


def _checkpoint(value: Any, path: str) -> CheckpointPlan | None:
    """The sheet's AgentX checkpoint, absent from a sheet whose measured round is not judged."""
    return None if value is None else CheckpointPlan.parse(value, path)


def _base_launch(value: Any, path: str) -> BaseLaunch:
    """The frozen base a v2 sheet carries, its digest recomputed from its content and no credential in the clear."""
    if value is None:
        raise DocumentError(
            f"{path}: a {SPEC_SCHEMA} sheet carries the session's frozen {BASE_LAUNCH_SCHEMA} base, and this one has none"
        )
    size = len(canonical_json(value).encode())
    if size > MAX_BASE_LAUNCH_BYTES:
        raise DocumentError(f"{path}: base_launch too large ({size} bytes > {MAX_BASE_LAUNCH_BYTES})")
    try:
        base = BaseLaunch.from_document(value)
    except ValueError as exc:
        raise DocumentError(f"{path}: {exc}") from None
    found = credentials_in_the_clear(base)
    if found:
        raise DocumentError(f"{path}: the base holds an unredacted credential in {', '.join(found)}")
    return base


@dataclass(frozen=True)
class Procedure:
    """How the test runs. A v2 sheet carries the session's frozen base launch (``base_launch``); a v1 sheet named the
    backend that ran it (``benchmark_backend``) instead."""

    prep: str
    warmup_rounds: int
    measure_rounds: int
    accuracy: bool
    timeout_sec: float
    silence_timeout_sec: float
    deadline_unix: float
    base_launch: BaseLaunch | None = None
    benchmark_backend: str | None = None
    short_prompts: ShortPromptPlan | None = None
    long_context: LongContextPlan | None = None
    checkpoint: CheckpointPlan | None = None

    @classmethod
    def parse(cls, value: Any, path: str) -> "Procedure":
        r = Reader(value, path)
        procedure = cls(**_procedure_fields(r), base_launch=_base_launch(r.optional(_BASE_LAUNCH), r.at(_BASE_LAUNCH)))
        r.done()
        return procedure

    @classmethod
    def parse_v1(cls, value: Any, path: str) -> "Procedure":
        r = Reader(value, path)
        procedure = cls(**_procedure_fields(r), benchmark_backend=r.choice(_BENCHMARK_BACKEND, ("magpie", "bypass")))
        r.optional(_REMOVED_DECODE_PASS)
        r.done()
        return procedure

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "prep": self.prep,
            "rounds": {"warmup": self.warmup_rounds, "measure": self.measure_rounds},
            _ACCURACY: self.accuracy,
            "timeout_sec": self.timeout_sec,
            "silence_timeout_sec": self.silence_timeout_sec,
            "deadline_unix": self.deadline_unix,
        }
        if self.base_launch is not None:
            document[_BASE_LAUNCH] = self.base_launch.to_document()
        if self.benchmark_backend is not None:
            document[_BENCHMARK_BACKEND] = self.benchmark_backend
        if self.short_prompts is not None:
            document[_SHORT_PROMPTS] = self.short_prompts.to_document()
        if self.long_context is not None:
            document[_LONG_CONTEXT] = self.long_context.to_document()
        if self.checkpoint is not None:
            document[_CHECKPOINT] = self.checkpoint.to_document()
        return document


def _procedure_fields(r: Reader) -> dict[str, Any]:
    """The fields every sheet version's procedure has."""
    rounds = Reader(r.raw("rounds"), r.at("rounds"))
    warmup, measure = rounds.integer("warmup", minimum=0) or 0, rounds.integer("measure", minimum=1) or 1
    rounds.done()
    return {
        "prep": r.choice("prep", ("grid", "conc_sweep")) or "",
        "warmup_rounds": warmup,
        "measure_rounds": measure,
        "accuracy": r.boolean(_ACCURACY),
        "timeout_sec": r.number("timeout_sec", minimum=1) or 1.0,
        "silence_timeout_sec": r.number("silence_timeout_sec", minimum=1) or 1.0,
        "deadline_unix": r.number("deadline_unix", minimum=0) or 0.0,
        "short_prompts": _short_prompts(r.optional(_SHORT_PROMPTS), r.at(_SHORT_PROMPTS)),
        "long_context": _long_context(r.optional(_LONG_CONTEXT), r.at(_LONG_CONTEXT)),
        "checkpoint": _checkpoint(r.optional(_CHECKPOINT), r.at(_CHECKPOINT)),
    }


@dataclass(frozen=True)
class _Version:
    """How one sheet version reads the sections that differ between versions."""

    tools: Callable[[Any, str], ToolCommits]
    procedure: Callable[[Any, str], Procedure]


_VERSIONS = {
    SPEC_SCHEMA_V1: _Version(ToolCommits.parse_v1, Procedure.parse_v1),
    SPEC_SCHEMA: _Version(ToolCommits.parse, Procedure.parse),
}


@dataclass(frozen=True)
class Outputs:
    result_dir: str
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    @classmethod
    def parse(cls, value: Any, path: str) -> "Outputs":
        r = Reader(value, path)
        outputs = cls(
            result_dir=relative_path(r.raw(_RESULT_DIR), r.at(_RESULT_DIR)),
            max_total_bytes=r.integer("max_total_bytes", minimum=1) or DEFAULT_MAX_TOTAL_BYTES,
        )
        r.done()
        return outputs

    def to_document(self) -> dict[str, Any]:
        return {_RESULT_DIR: self.result_dir, "max_total_bytes": self.max_total_bytes}


def _check_points(points: tuple[Config, ...], prep: str) -> None:
    """A grid test is one point; a conc_sweep arm is several that differ only in what the concurrency level decides."""
    if prep == "grid" and len(points) != 1:
        raise DocumentError("spec.config: a grid test has exactly one measurement point")
    if not points:
        raise DocumentError("spec.config: at least one measurement point is needed")
    if len({point.sha256 for point in points}) != len(points):
        raise DocumentError("spec.config: a measurement point appears twice")
    if len(points) > 1:
        rendered = [_without_point_values(_load_yaml(point.yaml)) for point in points]
        if any(other != rendered[0] for other in rendered[1:]):
            raise DocumentError("spec.config: the points of one arm differ in more than the concurrency level")


def _without_point_values(config: Any) -> dict[str, Any]:
    """The configuration with the values the concurrency level decides removed, as the points are compared."""
    try:
        envs = BenchmarkPoint.from_document(config, source="spec.config.yaml").envs
    except BenchmarkPointError as exc:
        raise DocumentError(str(exc)) from None
    benchmark = {**config["benchmark"], "envs": {k: v for k, v in envs.items() if k not in POINT_VALUE_KEYS}}
    return {**config, "benchmark": benchmark}


def _model_hashes(value: Any, path: str) -> dict[str, str]:
    hashes = string_map(value, path, key_check=relative_path)
    for key, digest in hashes.items():
        if not SHA256_RE.fullmatch(digest):
            raise DocumentError(f"{path}.{key}: not a sha256")
    return hashes


def _platform_env(value: Any, path: str) -> dict[str, str]:
    values = string_map(value, path)
    unknown = sorted(set(values) - set(PLATFORM_ENV_NAMES))
    if unknown:
        raise DocumentError(f"{path}.{unknown[0]}: not a platform variable a sheet may set")
    return values


def _unset_envs(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DocumentError(f"{path}: expected an array")
    names = tuple(env_name(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(names)) != len(names):
        raise DocumentError(f"{path}: a name appears twice")
    return names


@dataclass(frozen=True)
class MeasurementSpec:
    """One measurement to run: validated on construction through :meth:`from_document`, serialised by :meth:`to_bytes`."""

    request_id: str
    kind: str
    meta: Meta
    hardware: Hardware
    image: ImageRef
    config: tuple[Config, ...]
    attachments: tuple[Attachment, ...]
    process_env: Mapping[str, str]
    platform_env: Mapping[str, str]
    unset_envs: tuple[str, ...]
    tools: ToolCommits
    environment: Environment
    model_hashes: Mapping[str, str]
    procedure: Procedure
    outputs: Outputs
    schema: str = field(default=SPEC_SCHEMA)

    @classmethod
    def from_document(cls, document: Any) -> "MeasurementSpec":
        r = Reader(document, "spec")
        schema = r.text("schema") or ""
        version = _VERSIONS.get(schema)
        if version is None:
            raise DocumentError(
                f"spec.schema: {schema!r} is not a sheet version this reader knows ({', '.join(SPEC_SCHEMAS)})"
            )
        spec = cls(
            request_id=r.text("request_id") or "",
            kind=r.choice(_KIND, KINDS) or "",
            meta=r.section("meta", Meta.parse),
            hardware=r.section("hardware", Hardware.parse),
            image=r.section("image", ImageRef.parse),
            config=r.sequence("config", Config.parse),
            attachments=r.sequence("attachments", Attachment.parse),
            process_env=string_map(r.raw("process_env"), r.at("process_env")),
            platform_env=_platform_env(r.raw("platform_env"), r.at("platform_env")),
            unset_envs=_unset_envs(r.raw("unset_envs"), r.at("unset_envs")),
            tools=r.section("tools", version.tools),
            environment=r.section("environment", Environment.parse),
            model_hashes=_model_hashes(r.raw("model_hashes"), r.at("model_hashes")),
            procedure=r.section("procedure", version.procedure),
            outputs=r.section("outputs", Outputs.parse),
            schema=schema,
        )
        r.done()
        ids = [attachment.id for attachment in spec.attachments]
        if len(set(ids)) != len(ids):
            raise DocumentError("spec.attachments: an id appears twice")
        _check_points(spec.config, spec.procedure.prep)
        if spec.kind == _ACCURACY and not spec.procedure.accuracy:
            raise DocumentError("spec.procedure.accuracy: a kind 'accuracy' sheet asks for the accuracy evaluation")
        return spec

    @classmethod
    def from_bytes(cls, raw: bytes) -> "MeasurementSpec":
        return cls.from_document(parse_json_object(raw, what="spec"))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            _KIND: self.kind,
            "meta": self.meta.to_document(),
            "hardware": self.hardware.to_document(),
            "image": self.image.to_document(),
            "config": [point.to_document() for point in self.config],
            "attachments": [attachment.to_document() for attachment in self.attachments],
            "process_env": dict(self.process_env),
            "platform_env": dict(self.platform_env),
            "unset_envs": list(self.unset_envs),
            "tools": self.tools.to_document(),
            "environment": self.environment.to_document(),
            "model_hashes": dict(self.model_hashes),
            "procedure": self.procedure.to_document(),
            "outputs": self.outputs.to_document(),
        }

    def to_bytes(self) -> bytes:
        """The exact bytes written to ``/in/spec.json``; :meth:`sha256` is the hash of these."""
        return canonical_json(self.to_document()).encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


#: Where the caller writes every blob the sheet names, inside the sandbox: ``/in/blobs/<hex sha256>``.
BLOBS_DIR = "/in/blobs"


def blob_input_path(blob: str) -> str:
    """The sandbox path of a blob named ``sha256:<hex>`` in the sheet; the runner checks the hash when it reads it."""
    if not PREFIXED_SHA256_RE.fullmatch(blob):
        raise DocumentError(f"not a blob reference: {blob!r}")
    return f"{BLOBS_DIR}/{blob.removeprefix('sha256:')}"


def spec_sha256(raw: bytes) -> str:
    """The hash of a sheet as the runner reads it: the raw bytes, before any parsing."""
    return hashlib.sha256(raw).hexdigest()


def _replace_session_dir(value: Any, session_dir: str | None) -> Any:
    """Walk a parsed YAML value: path-valued keys become placeholders, session-directory prefixes are replaced."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, str):
                result[key] = "${PATH}"
            else:
                result[key] = _replace_session_dir(item, session_dir)
        return result
    if isinstance(value, list):
        return [_replace_session_dir(item, session_dir) for item in value]
    if isinstance(value, str) and session_dir:
        return value.replace(session_dir, "${SESSION_DIR}")
    return value


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise DocumentError(
            "the spec fingerprint needs PyYAML: install hyperloom[runtime] or hyperloom[runner]"
        ) from exc
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DocumentError(f"spec.config.yaml: not valid YAML ({type(exc).__name__})") from None
    except (ValueError, RecursionError) as exc:  # more digits than int() converts, a day no month has, deep nesting
        raise DocumentError(f"spec.config.yaml: cannot be loaded ({type(exc).__name__})") from None
    _refuse_deep(value)
    try:
        canonical_json(value).encode()
    except (TypeError, ValueError) as exc:  # a date, a set, binary, a non-finite float or a lone surrogate
        raise DocumentError(f"spec.config.yaml: holds a value UTF-8 JSON cannot hold ({exc})") from None
    return value


def _refuse_deep(value: Any) -> None:
    """Refuse a loaded YAML value that nests more than :data:`_MAX_YAML_DEPTH` levels deep.

    The loader reads an alias as the very value its anchor names, so a chain of anchors (``a1: &a1 [*a0]``) nests one
    level deeper per line of text, and an anchor used inside its own value nests without end. What reads the value next
    (its canonical JSON, the comparison of a sweep's points) recurses once per level. This walk keeps its own stack and
    measures a value an alias repeats once, so it costs no more than the value has distinct levels.
    """
    heights: dict[int, int] = {}  # id of a level already measured -> the levels it holds, itself included
    frames: list[tuple[Any, Iterator[Any]]] = []
    tallest: list[int] = []  # per frame, the most levels any of its members measured so far holds

    def enter(container: Any) -> None:
        if len(frames) == _MAX_YAML_DEPTH:
            raise DocumentError(_TOO_DEEP)
        members = container.values() if isinstance(container, Mapping) else container
        frames.append((container, iter(members)))
        tallest.append(0)

    if isinstance(value, _NESTED):
        enter(value)
    while frames:
        container, members = frames[-1]
        for member in members:
            if not isinstance(member, _NESTED):
                continue
            height = heights.get(id(member))
            if height is None:
                enter(member)
                break
            if len(frames) + height > _MAX_YAML_DEPTH:
                raise DocumentError(_TOO_DEEP)
            tallest[-1] = max(tallest[-1], height)
        else:
            frames.pop()
            heights[id(container)] = tallest.pop() + 1
            if tallest:
                tallest[-1] = max(tallest[-1], heights[id(container)])


def canonical_for_fingerprint(spec: MeasurementSpec, *, session_dir: str | None = None) -> dict[str, Any]:
    """The sheet with everything that is not measurement content removed or replaced.

    Removed: ``request_id``, ``meta``, ``procedure.deadline_unix``, ``config.sha256`` (it hashes paths), the session's
    blob root and runtime snapshot location, and the staging's normalised manifest, which its sha256 stands for. Replaced: path-valued YAML fields and session-directory prefixes become
    placeholders, and the carried base launch its digest, which leaves out where and when it was resolved. Blobs and
    attachments keep only their hashes.
    """
    document = spec.to_document()
    for key in ("request_id", "meta"):
        del document[key]
    del document["procedure"]["deadline_unix"]
    if spec.procedure.base_launch is not None:
        document["procedure"][_BASE_LAUNCH] = spec.procedure.base_launch.digest()
    if spec.tools.staging is not None:
        del document["tools"][_INFERENCE_SUITE][_NORMALISED]
    del document["environment"]["blob_root"]
    if document["environment"][_RUNTIME] is not None:
        del document["environment"][_RUNTIME]["snapshot"]
    session = session_dir.rstrip("/") if session_dir else None
    document["config"] = [{"yaml": _replace_session_dir(_load_yaml(p.yaml), session)} for p in spec.config]
    document["process_env"] = _replace_session_dir(document["process_env"], session)
    return document


def fingerprint(spec: MeasurementSpec, *, hyperloom_version: str, session_dir: str | None = None) -> str:
    """The first 16 hex digits of sha256(canonical sheet, runner Hyperloom version).

    The version is part of the identity: a measurement made by another runner version is not the same measurement.
    """
    if not hyperloom_version:
        raise DocumentError("the fingerprint needs the runner's Hyperloom version")
    canonical = canonical_json(canonical_for_fingerprint(spec, session_dir=session_dir))
    digest = hashlib.sha256(canonical.encode() + b"\n" + hyperloom_version.encode())
    return digest.hexdigest()[:_FINGERPRINT_LEN]
