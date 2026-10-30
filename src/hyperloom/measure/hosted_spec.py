# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a :class:`~hyperloom.measure.socket.TestCall` into the instruction sheet a runner executes (design 4.4.4 step 2).

The sheet carries everything the runner needs and nothing it should not have: the materialised YAML with shared-storage
paths replaced by attachment placeholders, a whitelist of process variables (never a credential), the platform variables
of this process, the environment overlay, and the facts the application knows about the session. Anything the sheet
cannot say exactly is refused with :class:`SpecRefused`, never approximated.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env import env_bool
from hyperloom.common.outside_facts import ENV_TP
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
    is_secret_shaped_env_name,
)
from hyperloom.measure.compile import BlobError
from hyperloom.measure.docs import DocumentError, canonical_json
from hyperloom.measure.spec import DEFAULT_MAX_TOTAL_BYTES, PLATFORM_ENV_NAMES, MeasurementSpec
from hyperloom.measure.socket import TestCall

__all__ = [
    "MAX_ATTACHMENT_BYTES",
    "PROCESS_ENV_WHITELIST",
    "SpecContext",
    "SpecRefused",
    "build_spec",
    "dir_entries",
    "dir_manifest_sha256",
    "gpu_count_of",
    "split_result_dir",
]

#: Process variables of the application that a sheet passes on (4.4.4); every other variable stays behind.
PROCESS_ENV_WHITELIST = (
    "HYPERLOOM_AGENTX",
    "HYPERLOOM_AGENTIC_BACKEND",
    "HYPERLOOM_PERF_METRIC",
    "HYPERLOOM_PERF_NOISE_PCT",
    "INFERENCE_OPTIMIZER_RUN_GRID_WARMUP",
    "INFERENCE_OPTIMIZER_SERVER_ARGS",
    "INFERENCE_OPTIMIZER_EXTRA_ENV",
)
#: What a request's attachments may hold in all, as blobs, and what one directory attachment may expand to.
_MIB = 1024 * 1024
MAX_ATTACHMENT_BYTES = 64 * _MIB
_HASH_CHUNK_BYTES = _MIB
_POINT_UNREADABLE = "point_unreadable"
_PLACEHOLDER = re.compile(r"\$\{HL_ATTACH:([A-Za-z0-9][A-Za-z0-9._-]{0,63})\}")
_HEX_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class SpecRefused(ValueError):
    """The call cannot be turned into a sheet that says exactly what it asks for; ``reason`` is a short stable name."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SpecContext:
    """What the application knows about the session that a single call does not carry.

    ``blobs`` is the session's content-addressed blob store (``put``/``get``); attachments are written to it.
    ``shared_roots`` are the directories on shared storage whose files a runner cannot read: a path below one of them
    (the model directory excepted) in the YAML becomes an attachment.
    """

    image_ref: str
    image_digest: str
    gpu_type: str
    magpie_commit: str
    inferencex_commit: str
    blobs: Any
    power_cap_w: float | None = None
    partition_mode: str | None = None
    model_hashes: Mapping[str, str] = field(default_factory=dict)
    benchmark_backend: str = "magpie"
    shared_roots: tuple[Path, ...] = ()
    model_dir: Path | None = None
    max_total_bytes: int | None = None
    platform_env: Mapping[str, str] | None = None
    process_environ: Mapping[str, str] | None = None
    blob_root: str = ""


# -- reading the call ------------------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> tuple[str, dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecRefused(_POINT_UNREADABLE, f"{path.name}: not valid YAML ({type(exc).__name__})") from None
    if not isinstance(document, dict):
        raise SpecRefused(_POINT_UNREADABLE, f"{path.name}: the top level is not a mapping")
    return text, document


def gpu_count_of(document: Mapping[str, Any]) -> int:
    """The tensor-parallel size of a materialised point, which is how many cards its runner needs."""
    envs = (document.get("benchmark") or {}).get("envs") or {}
    raw = envs.get(ENV_TP, 1)
    try:
        count = int(str(raw).strip() or 1)
    except ValueError:
        raise SpecRefused(_POINT_UNREADABLE, f"TP={raw!r} is not an integer") from None
    if count < 1:
        raise SpecRefused(_POINT_UNREADABLE, f"TP={raw!r} asks for no GPU")
    return count


def _process_env(call: TestCall, ctx: SpecContext) -> dict[str, str]:
    source = os.environ if ctx.process_environ is None else ctx.process_environ
    env = {name: source[name] for name in PROCESS_ENV_WHITELIST if name in source}
    allowed, dropped = filter_untrusted_env_mapping(call.env_overrides, allow_predicate=is_allowed_variant_env_key)
    secret = sorted(name for name in allowed if is_secret_shaped_env_name(name))
    if dropped or secret:
        raise SpecRefused("env_override_refused", f"env_overrides may not set {sorted(dropped) + secret}")
    env.update(allowed)
    return env


def _platform_env(ctx: SpecContext) -> dict[str, str]:
    source = os.environ if ctx.platform_env is None else ctx.platform_env
    return {name: source[name] for name in PLATFORM_ENV_NAMES if name in source}


def _unset_envs(call: TestCall) -> tuple[str, ...]:
    facts = call.session_facts
    return tuple(dict.fromkeys([*(facts.reference_unset_envs if facts else ()), *call.unset_envs]))


def _commit(value: str | None, what: str) -> str:
    if not value or not _HEX_COMMIT.fullmatch(value):
        raise SpecRefused("tools_unknown", f"{what} commit is not a full commit hash: {value!r}")
    return value


def _warmup_rounds(call: TestCall) -> int:
    configured = call.rounds.warmup_rounds
    if configured is not None:
        return int(configured)
    return 1 if env_bool("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", default=True) else 0


# -- attachments -----------------------------------------------------------------------------------------------------


def dir_manifest_sha256(entries: list[tuple[str, int, str]]) -> str:
    """The hash of a directory attachment's file list: sorted ``(relative path, size, sha256)`` triples as canonical JSON."""
    return hashlib.sha256(canonical_json([list(entry) for entry in sorted(entries)]).encode()).hexdigest()


def _pack_dir(directory: Path) -> tuple[bytes, str]:
    entries: list[tuple[str, int, str]] = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()):
            relative = path.relative_to(directory).as_posix()
            data = path.read_bytes()
            entries.append((relative, len(data), hashlib.sha256(data).hexdigest()))
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
        if any(p.is_symlink() for p in directory.rglob("*")):
            raise SpecRefused("attachment_unsafe", f"{directory} holds a symbolic link")
    return buffer.getvalue(), dir_manifest_sha256(entries)


def _attachment_from_path(path: Path, ctx: SpecContext) -> dict[str, str]:
    if path.is_symlink():
        raise SpecRefused("attachment_unsafe", f"{path} is a symbolic link")
    if path.is_dir():
        data, manifest = _pack_dir(path)
        kind = "dir"
    else:
        data = path.read_bytes()
        manifest, kind = hashlib.sha256(data).hexdigest(), "file"
    blob = ctx.blobs.put(data)
    return {"id": f"a-{blob[:16]}", "kind": kind, "blob": f"sha256:{blob}", "manifest_sha256": manifest}


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _candidate_paths(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [p for item in value.values() for p in _candidate_paths(item)]
    if isinstance(value, list):
        return [p for item in value for p in _candidate_paths(item)]
    if isinstance(value, str):
        return [part for part in value.split(":") if part.startswith("/")]
    return []


def _scan_attachments(document: Mapping[str, Any], ctx: SpecContext) -> dict[str, dict[str, str]]:
    """Shared-storage paths under ``benchmark``, keyed by the path text they appear as in the YAML."""
    found: dict[str, dict[str, str]] = {}
    model = ctx.model_dir.resolve() if ctx.model_dir else None
    roots = tuple(root.resolve() for root in ctx.shared_roots)
    for text in _candidate_paths(document.get("benchmark") or {}):
        path = Path(text)
        if text in found or not path.exists():
            continue
        real = path.resolve()
        if model is not None and _under(real, (model,)):
            continue
        if _under(real, roots):
            found[text] = _attachment_from_path(path, ctx)
    return found


def _replace_paths(yaml_text: str, found: Mapping[str, Mapping[str, str]]) -> str:
    # Longest first, so a path that is a prefix of another is not replaced inside it.
    for text in sorted(found, key=len, reverse=True):
        yaml_text = yaml_text.replace(text, "${HL_ATTACH:%s}" % found[text]["id"])
    return yaml_text


def _member_sha256(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    digest = hashlib.sha256()
    source = tar.extractfile(member)
    while source is not None and (chunk := source.read(_HASH_CHUNK_BYTES)):
        digest.update(chunk)
    return digest.hexdigest()


def _dir_entries(ref: AttachmentRef, data: bytes) -> list[tuple[str, int, str]]:
    """``(path, size, sha256)`` of each file of a directory attachment, hashed in chunks; what a request's archive
    expands to is bounded here, before the sheet is built."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if sum(m.size for m in members) > MAX_ATTACHMENT_BYTES:
                raise SpecRefused(
                    "attachment_too_large", f"attachment {ref.id} expands past {MAX_ATTACHMENT_BYTES} bytes"
                )
            return [(m.name, m.size, _member_sha256(tar, m)) for m in members]
    except tarfile.TarError as exc:
        raise SpecRefused(
            "attachment_unreadable", f"attachment {ref.id}: not a tar archive ({type(exc).__name__})"
        ) from None


dir_entries = _dir_entries  # public name for callers that vet a directory attachment before storing it


def _attachment_entries(refs: Iterable[AttachmentRef], ctx: SpecContext) -> list[dict[str, str]]:
    result = []
    for ref in refs:
        manifest = (
            dir_manifest_sha256(_dir_entries(ref, ctx.blobs.get(ref.sha256))) if ref.kind == "dir" else ref.sha256
        )
        result.append({"id": ref.id, "kind": ref.kind, "blob": f"sha256:{ref.sha256}", "manifest_sha256": manifest})
    return result


def _environment_attachments(environment: object, ctx: SpecContext) -> list[dict[str, str]]:
    return _attachment_entries(getattr(environment, "attachments", ()) or (), ctx)


def _request_attachments(call: TestCall, ctx: SpecContext) -> list[dict[str, str]]:
    """The attachments the request ships (a repro case).

    A blob the store does not hold is refused, not sent; so is one that is too large, or a request whose attachments
    together are: everything listed here is uploaded to the sandbox and unpacked there.
    """
    total = 0
    for ref in call.attachments:
        try:
            size = len(ctx.blobs.get(ref.sha256))
        except (BlobError, KeyError, OSError) as exc:
            raise SpecRefused("attachment_missing", f"attachment {ref.id}: {exc}") from None
        total += size
        if total > MAX_ATTACHMENT_BYTES:
            raise SpecRefused("attachment_too_large", f"attachment {ref.id}: the request's attachments pass the cap")
    return _attachment_entries(call.attachments, ctx)


def _environment_document(environment: object, ctx: SpecContext) -> dict[str, Any]:
    to_document = getattr(environment, "to_document", None)
    if callable(to_document):
        return to_document()  # type: ignore[no-any-return]
    return {"roots": [], "runtime": None, "blob_root": ctx.blob_root or "/"}


# -- the sheet -------------------------------------------------------------------------------------------------------


def _call_attachments(call: TestCall, ctx: SpecContext) -> dict[str, dict[str, str]]:
    """The attachments the sheet ships, by id: the environment's, then the request's; one id may not name both."""
    attachments = {a["id"]: a for a in _environment_attachments(call.environment, ctx)}
    shipped = _request_attachments(call, ctx)
    if any(a["id"] in attachments for a in shipped):
        raise SpecRefused("attachment_id_taken", "a request attachment has the id of an environment attachment")
    attachments.update({a["id"]: a for a in shipped})
    return attachments


def _point_configs(
    call: TestCall, ctx: SpecContext, attachments: dict[str, dict[str, str]]
) -> tuple[list[str], dict[str, Any] | None]:
    """Each point's YAML with its shared-storage paths replaced by placeholders, and the first point's document.

    Every attachment a point's paths name joins ``attachments``.
    """
    configs: list[str] = []
    first: dict[str, Any] | None = None
    for point in call.points:
        text, document = _load_yaml(point)
        first = first or document
        found = _scan_attachments(document, ctx)
        attachments.update({a["id"]: a for a in found.values()})
        configs.append(_replace_paths(text, found))
    return configs, first


def _refuse_unknown_placeholders(configs: list[str], attachments: Mapping[str, object]) -> None:
    unknown = sorted({p for text in configs for p in _PLACEHOLDER.findall(text)} - set(attachments))
    if unknown:
        raise SpecRefused("placeholder_unknown", f"the YAML names attachments {unknown} that nothing provides")


def _meta_section(call: TestCall) -> dict[str, Any]:
    meta = call.meta
    return {
        "purpose": meta.purpose or "explore",
        "priority": meta.priority,
        "branch_id": meta.branch_id or "none",
        "parent_commit": meta.parent_commit or "none",
        "retest_of": meta.retest_of,
    }


def _hardware_section(ctx: SpecContext, gpus: int) -> dict[str, Any]:
    return {
        "gpu_type": ctx.gpu_type,
        "gpu_count": gpus,
        "power_cap_w": ctx.power_cap_w,
        "partition_mode": ctx.partition_mode,
    }


def _tools_section(call: TestCall, ctx: SpecContext) -> dict[str, Any]:
    facts = call.session_facts
    return {
        "magpie": {"commit": _commit((facts.magpie_commit if facts else None) or ctx.magpie_commit, "Magpie")},
        "inferencex": {"commit": _commit(ctx.inferencex_commit, "InferenceX")},
    }


def _procedure_section(call: TestCall, ctx: SpecContext, deadline_unix: float) -> dict[str, Any]:
    section: dict[str, Any] = {
        "prep": call.prep,
        "benchmark_backend": ctx.benchmark_backend,
        "rounds": {"warmup": _warmup_rounds(call), "measure": int(call.rounds.measure_rounds)},
        "accuracy": bool(call.accuracy),
        "timeout_sec": float(call.timeout_sec),
        "silence_timeout_sec": float(call.silence_timeout_sec or call.timeout_sec),
        "deadline_unix": float(deadline_unix),
    }
    if call.short_prompts is not None:
        section["short_prompts"] = call.short_prompts.to_document()
    if call.long_context is not None:
        section["long_context"] = call.long_context.to_document()
    return section


def build_spec(call: TestCall, *, request_id: str, ctx: SpecContext, deadline_unix: float) -> MeasurementSpec:
    """The sheet for ``call`` (one YAML per point); any way it could differ from what was asked is a :class:`SpecRefused`.

    Raises:
        SpecRefused: No point, an unreadable point, an environment override that
            is not allowed, a missing tool commit, a placeholder with no attachment, or a document the sheet schema
            refuses.
    """
    if not call.points:
        raise SpecRefused("no_points", "a test needs at least one measurement point")
    attachments = _call_attachments(call, ctx)
    configs, first = _point_configs(call, ctx, attachments)
    _refuse_unknown_placeholders(configs, attachments)
    gpus = gpu_count_of(first or {})
    sheet = {
        "schema": "hyperloom.measurement-spec/v1",
        "request_id": request_id,
        "kind": call.kind,
        "meta": _meta_section(call),
        "hardware": _hardware_section(ctx, gpus),
        "image": {"ref": ctx.image_ref, "digest": ctx.image_digest},
        "config": [{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()} for text in configs],
        "attachments": list(attachments.values()),
        "process_env": _process_env(call, ctx),
        "platform_env": _platform_env(ctx),
        "unset_envs": list(_unset_envs(call)),
        "tools": _tools_section(call, ctx),
        "environment": _environment_document(call.environment, ctx),
        "model_hashes": dict(ctx.model_hashes),
        "procedure": _procedure_section(call, ctx, deadline_unix),
        "outputs": {"result_dir": split_result_dir(call)[0], **_max_bytes(ctx)},
    }
    try:
        return MeasurementSpec.from_document(sheet)
    except DocumentError as exc:
        raise SpecRefused("sheet_invalid", str(exc)) from None


def _max_bytes(ctx: SpecContext) -> dict[str, int]:
    return {"max_total_bytes": ctx.max_total_bytes or DEFAULT_MAX_TOTAL_BYTES}


def split_result_dir(call: TestCall) -> tuple[str, Path | None]:
    """``(result_dir as the sheet says it, the original location to copy it to afterwards)``.

    The sheet names the directory relative to the output directory. A location outside the output directory is
    renamed to a sibling below it, and the original location is returned so the backend can put the files there once
    the result has landed (4.4.4 / 3.3).
    """
    raw = call.result_dir
    if not raw:
        return "results", None
    path = Path(raw)
    if not path.is_absolute():
        if ".." in path.parts:
            raise SpecRefused("result_dir_unsafe", f"{raw!r} climbs out of the output directory")
        return path.as_posix(), None
    output = Path(call.output_dir)
    try:
        return path.relative_to(output).as_posix(), None
    except ValueError:
        return f"result_dir_{path.name or 'results'}", path
