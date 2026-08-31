# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Immutable configuration for one resumable Forge campaign."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.kernel_backends.constants import KERNEL_BACKENDS
from kernelforge.knowledge.experience_sink import (
    infer_source_owner_framework,
    resolve_operation,
)
from kernelforge.knowledge.implementation_identity import (
    hash_implementation_identity,
    implementation_signature,
)
from kernelforge.loop.new_path_allowlist import normalize_commit_new_paths
from kernelforge.mcp_server.tools.pmc import derive_kernel_names
from kernelforge.durable_io import atomic_write_text
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


SCHEMA_VERSION = 7
# Versions a campaign on disk may be written in and still be read back. Only
# ``SCHEMA_VERSION`` is ever WRITTEN; ``from_dict`` normalizes an older payload
# to it in memory, so the file itself is left untouched and the immutability
# comparison in ``CampaignConfigStore.save`` still holds.
#
# 6 differs from 7 only by the absence of ``commit_new_paths``, and a campaign
# written before the allowlist existed meant exactly what an absent allowlist
# means now: nothing may be committed. Refusing it would strand every campaign
# already on disk with no way out, since ``save`` guards on ``load``. The
# 5 -> 6 bump was a different thing -- it REMOVED a field, so an old payload
# tripped the unknown-field check and really could not be read. Precedent for
# the read-set: ``rewrite_by_flydsl.protocol.ARTIFACT_SCHEMA_VERSIONS``.
READABLE_SCHEMA_VERSIONS = (6, 7)
_GPU_TARGET_RE = re.compile(r"\bgfx[0-9a-f]+\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
log = logging.getLogger(__name__)

_FALLBACK_KERNEL_BACKEND = "flydsl"


@dataclass(frozen=True)
class CampaignConfig:
    """Inputs that must remain stable across all sessions in a campaign."""

    schema_version: int = SCHEMA_VERSION
    kernel_path: str = ""
    driver_path: str = ""
    driver_sha256: str = ""
    source_files: list[str] = field(default_factory=list)
    program_md_path: str = ""
    program_md_sha256: str = ""
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    gpu_target: str = ""
    gpu_type: str = "mi355x"
    kernel_backend: str = ""
    task_type: str = ""
    target_functions: list[str] = field(default_factory=list)
    git_branch: str = ""
    base_commit: str = ""
    framework: str = ""
    operator_name: str = ""
    # Snapshotted like every other identity dimension: a resumed campaign that
    # re-derived it would publish under an address earlier sessions never used.
    producer: str = ""
    implementation_signature: str = ""
    implementation_identity: dict = field(default_factory=dict)
    # Measurement semantics. These decide what a number MEANS, so a resumed
    # session that re-derived them from CLI defaults would compare candidates
    # against an incumbent measured under different rules -- and on a
    # collective task would also drop from nproc=4 to a single rank.
    nproc_per_node: int = 1
    bench_repeat: int = 1
    # Paths the Implementer may CREATE and still have committed with a KEEP
    # (see ``IterationConfig.commit_new_paths``). Immutable like the rest of
    # this config: what a KEEP may ship and what a REVERT deletes must not
    # change under a resumed campaign.
    commit_new_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CampaignConfig":
        if not isinstance(payload, dict):
            raise ValueError("campaign config must be a JSON object")
        version = int(payload.get("schema_version", 0) or 0)
        if version not in READABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported campaign config schema {version}; expected one "
                "of " + ", ".join(str(known) for known in READABLE_SCHEMA_VERSIONS)
            )
        unknown_fields = set(payload) - {item.name for item in fields(cls)}
        if unknown_fields:
            raise ValueError("unsupported campaign config fields: " + ", ".join(sorted(unknown_fields)))
        config = cls(
            schema_version=SCHEMA_VERSION,
            kernel_path=str(payload.get("kernel_path") or ""),
            driver_path=str(payload.get("driver_path") or ""),
            driver_sha256=str(payload.get("driver_sha256") or "").lower(),
            source_files=[str(path) for path in (payload.get("source_files") or [])],
            program_md_path=str(payload.get("program_md_path") or ""),
            program_md_sha256=str(payload.get("program_md_sha256") or ""),
            snr_threshold=float(payload.get("snr_threshold", DEFAULT_SNR_THRESHOLD_DB)),
            gpu_target=str(payload.get("gpu_target") or ""),
            gpu_type=str(payload["gpu_type"] if "gpu_type" in payload else "mi355x").strip().lower(),
            kernel_backend=str(payload.get("kernel_backend") or ""),
            task_type=str(payload.get("task_type") or ""),
            target_functions=[str(name) for name in (payload.get("target_functions") or [])],
            git_branch=str(payload.get("git_branch") or ""),
            base_commit=str(payload.get("base_commit") or ""),
            framework=str(payload.get("framework") or ""),
            operator_name=str(payload.get("operator_name") or ""),
            producer=str(payload.get("producer") or ""),
            implementation_signature=str(payload.get("implementation_signature") or "").lower(),
            implementation_identity=dict(payload.get("implementation_identity") or {}),
            # to_dict() is asdict(), so these are always written; leaving them
            # out of the reader made a resumed campaign silently fall back to
            # one rank and single-shot benching -- measuring a
            # different thing than the session it claims to continue.
            nproc_per_node=max(1, int(payload.get("nproc_per_node") or 1)),
            bench_repeat=max(1, int(payload.get("bench_repeat") or 1)),
            # Re-validated on read: this list decides which untracked files a
            # KEEP commits and a REVERT deletes, so a hand-edited pattern the
            # loop would read differently than its author meant is refused
            # here rather than acted on later.
            commit_new_paths=normalize_commit_new_paths(payload.get("commit_new_paths") or []),
        )
        if config.program_md_path and not config.program_md_sha256:
            raise ValueError("campaign program context digest is missing")
        if config.program_md_sha256 and not config.program_md_path:
            raise ValueError("campaign program context path is missing")
        if not _SHA256_RE.fullmatch(config.driver_sha256):
            raise ValueError("campaign canonical driver digest is missing or invalid")
        if not math.isfinite(config.snr_threshold) or config.snr_threshold <= 0:
            raise ValueError("campaign SNR threshold must be a positive finite float")
        if not _SHA256_RE.fullmatch(config.implementation_signature):
            raise ValueError("campaign pristine implementation signature is missing or invalid")
        if hash_implementation_identity(config.implementation_identity) != config.implementation_signature:
            raise ValueError("campaign pristine implementation identity does not match its signature")
        return config


def derive_campaign_implementation_contract(
    *,
    workspace_dir: str,
    kernel_path: str,
    source_files: list[str],
    framework: str,
    base_commit: str = "",
) -> tuple[str, dict]:
    """Derive the immutable implementation contract from pristine git sources."""
    workspace = Path(workspace_dir).resolve()
    raw_paths = _campaign_source_paths(workspace, kernel_path, source_files)
    source_contents = _read_pristine_sources(
        workspace,
        raw_paths,
        base_commit=base_commit,
    )

    kernel_absolute = str((workspace / kernel_path).resolve())
    return implementation_signature(
        workspace=str(workspace),
        kernel_path=kernel_absolute,
        source_files=raw_paths,
        framework=framework,
        source_contents=source_contents,
    )


def _campaign_source_paths(
    workspace: Path,
    kernel_path: str,
    source_files: list[str],
) -> list[str]:
    raw_paths: list[str] = []
    for relative in [kernel_path, *source_files]:
        absolute = str((workspace / relative).resolve())
        if absolute not in raw_paths:
            raw_paths.append(absolute)
    return raw_paths


def _read_pristine_sources(
    workspace: Path,
    raw_paths: list[str],
    *,
    base_commit: str,
) -> dict[str, str]:
    source_contents: dict[str, str] = {}
    for absolute in raw_paths:
        path = Path(absolute)
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError:
            continue
        source = None
        if base_commit:
            result = git("show", f"{base_commit}:{relative}", cwd=workspace, check=False)
            if result.returncode == 0:
                source = result.stdout
        if source is None:
            try:
                source = path.read_text(errors="replace")
            except OSError:
                continue
        source_contents[absolute] = source
    return source_contents


class CampaignConfigStore:
    """Atomic store for ``forge_experiments/campaign_config.json``."""

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir).resolve()
        self.root = self.workspace / "forge_experiments"
        self.path = self.root / "campaign_config.json"
        self.program_path = self.root / "program.md"

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> CampaignConfig:
        if not self.path.is_file():
            raise FileNotFoundError(f"campaign config not found: {self.path}")
        try:
            payload = json.loads(self.path.read_text())
        except Exception as error:
            raise ValueError(f"invalid campaign config: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("campaign config must be a JSON object")
        return CampaignConfig.from_dict(payload)

    def save(
        self,
        config: CampaignConfig,
        *,
        program_md: str | None = None,
    ) -> None:
        """Persist once; an existing campaign config cannot be replaced."""
        config_exists = self.path.exists()
        if config_exists:
            if self.load() != config:
                raise ValueError("campaign config is immutable")
        if config.program_md_path:
            if program_md is None:
                raise ValueError("campaign program context content is required")
            digest = hashlib.sha256(program_md.encode()).hexdigest()
            if digest != config.program_md_sha256:
                raise ValueError("campaign program context digest does not match")
            if self.program_path.exists():
                if self.program_path.read_text() != program_md:
                    raise ValueError("campaign program context is immutable")
            else:
                atomic_write_text(self.program_path, program_md)
        if not config_exists:
            payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
            atomic_write_text(self.path, payload)

    def read_program_md(self, config: CampaignConfig) -> str:
        if not config.program_md_path:
            return ""
        path = (self.workspace / config.program_md_path).resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError("campaign program path escapes workspace") from error
        if not path.is_file():
            raise ValueError(f"campaign program context is missing: {path}")
        text = path.read_text(errors="replace")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != config.program_md_sha256:
            raise ValueError("campaign program context content has changed")
        return text


def _git_value(workspace: Path, *args: str) -> str:
    return git(*args, cwd=workspace).stdout.strip()


def _relative_file(workspace: Path, raw_path: str, label: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    # Resolve BOTH sides: ``path`` is already symlink-expanded, so comparing it
    # against an unexpanded workspace makes every containment check fail when the
    # caller passes a symlinked root (e.g. USER_DATA_PATH=/primus/xiaofei/... ->
    # /primus/data/xiaofei/...), rejecting a driver that is genuinely inside it.
    workspace = workspace.resolve()
    try:
        relative = path.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{label} must be inside workspace: {path}") from error
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return relative.as_posix()


def _driver_reference(workspace: Path, raw_path: str) -> str:
    """Resolve ``--driver`` to a campaign-stable reference.

    A driver inside the workspace stays workspace-relative (the common case, and
    what keeps a campaign relocatable). A driver OUTSIDE the workspace is not an
    error: task preparation supports external drivers as a first-class mode,
    staging and publishing them transactionally (``ExternalArtifactTransaction``)
    so a failed prep cannot leak edits outside the kernel workspace. Rejecting
    the path here killed the fresh-campaign CLI before prep could ever run, which
    is what produced "Error: driver must be inside workspace: .../forge_autogen_driver.py"
    for every caller that generates the driver next to its run artifacts.

    The external form is stored absolute. ``workspace / <absolute>`` yields that
    absolute path unchanged, so both consumers of ``driver_path`` keep working.
    """
    try:
        return _relative_file(workspace, raw_path, "driver")
    except ValueError as error:
        if "must be inside workspace" not in str(error):
            raise
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"driver is not a file: {path}")
    return path.as_posix()


def detect_gpu_target() -> str:
    """Resolve the active AMD GPU architecture without a CLI option."""
    configured = os.environ.get("GPU_TARGET", "").strip().lower()
    if configured:
        if not _GPU_TARGET_RE.fullmatch(configured):
            raise ValueError(f"invalid GPU_TARGET: {configured}")
        return configured
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as error:
        raise ValueError("could not detect GPU target; ensure rocminfo is available") from error
    targets = sorted(set(_GPU_TARGET_RE.findall(result.stdout or "")))
    if result.returncode != 0 or len(targets) != 1:
        raise ValueError("could not detect exactly one GPU target; configure the runtime GPU")
    return targets[0].lower()


def normalize_kernel_backend_name(kernel_backend: str) -> str:
    """Reduce a backend label to its bare canonical name."""
    return kernel_backend.strip()


_ENV_OVERRIDE = "FORGE_KERNEL_BACKEND"


def env_backend_override() -> str:
    """The backend named by the environment, or ``""``."""
    return os.environ.get(_ENV_OVERRIDE, "").strip()


def resolve_kernel_backend_override(kernel_backend: str) -> str:
    """Resolve an override against the registered kernel-building backends."""
    name = normalize_kernel_backend_name(kernel_backend)
    if name not in KERNEL_BACKENDS:
        return _FALLBACK_KERNEL_BACKEND
    return name


def infer_kernel_backend(source_paths: list[Path]) -> str:
    """Infer the backend expertise prompt from configuration or source files."""
    override = env_backend_override()
    if override:
        return resolve_kernel_backend_override(override)

    for path in source_paths:
        try:
            text = path.read_text(errors="replace").lower()
        except Exception:
            text = ""
        path_text = str(path).lower()
        suffix = path.suffix.lower()
        if "hipblaslt" in text or "hipblaslt" in path_text:
            return "hipblaslt"
        if "/aiter/" in path_text or "import aiter" in text:
            return "aiter"
        if "flydsl" in text or "cutlass.cute" in text or "from cutlass import cute" in text:
            return "flydsl"
        # Before Triton, and matching an import or a decorator rather than the
        # word: Gluon IS Triton's low-level dialect, so a Gluon file imports
        # triton and routinely keeps a `@triton.jit` sibling kernel as its
        # fallback -- aiter's paged-MQA-logits ships exactly that shape, in a
        # file under `aiter/ops/triton/`, so neither the path nor the presence
        # of Triton markers distinguishes the two. Checked after flydsl because
        # that arm keys on its own toolchain, which this one never carries.
        if "triton.experimental.gluon" in text or "@gluon.jit" in text or "gluon.language" in text:
            return "gluon"
        if "@triton.jit" in text or "triton.language" in text:
            return "triton"
        if "composable_kernel" in text or "ck::" in text:
            return "ck"
        if suffix in {".hip", ".cu", ".cuh", ".cpp", ".cc", ".cxx"}:
            return "hip"
    raise ValueError(f"could not infer the kernel backend; set {_ENV_OVERRIDE} to a known backend")


def _derive_target_functions(
    workspace: Path,
    source_files: list[str],
    *,
    source_contents: dict[str, str] | None = None,
) -> list[str]:
    functions: list[str] = []
    for relative in source_files:
        absolute = str((workspace / relative).resolve())
        source = source_contents.get(absolute) if source_contents is not None else None
        if source is None:
            try:
                source = Path(absolute).read_text(errors="replace")
            except Exception:
                continue
        for name in derive_kernel_names(source):
            if name not in functions:
                functions.append(name)
    return functions


def validate_pending_campaign_head(workspace_dir: str, base_commit: str) -> None:
    """Require a pending fresh retry to remain at its known setup lineage."""
    workspace = Path(workspace_dir).resolve()
    base = _git_value(workspace, "rev-parse", base_commit)
    head = _git_value(workspace, "rev-parse", "HEAD")
    if head == base:
        return
    parents = _git_value(workspace, "rev-list", "--parents", "-n", "1", head).split()
    subject = _git_value(workspace, "show", "-s", "--format=%s", head)
    if len(parents) == 2 and parents[1] == base and subject.lower().startswith("kb warm-start:"):
        return
    raise ValueError(
        "pending campaign HEAD mismatch: expected the configured base commit or one direct `kb warm-start:` child"
    )


def create_campaign_config(
    *,
    workspace_dir: str,
    kernel: str,
    driver: str,
    source_files: list[str],
    program_md_file: str | None,
    base_commit: str | None = None,
    target_functions: list[str] | None = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    gpu_target: str | None = None,
    gpu_type: str | None = None,
    git_branch: str | None = None,
    kernel_backend: str | None = None,
    task_type: str | None = None,
    framework: str | None = None,
    operator_name: str | None = None,
    producer: str | None = None,
    nproc_per_node: int = 1,
    bench_repeat: int = 1,
    commit_new_paths: list[str] | None = None,
) -> CampaignConfig:
    """Resolve and normalize all immutable inputs for a fresh/legacy campaign.

    Caller-supplied ``gpu_target``/``gpu_type``/``git_branch``/``kernel_backend``/
    ``task_type`` and ``framework`` take precedence; each falls back to local inference. An
    unsupported explicit kernel backend falls back to the FlyDSL kernel_backend.
    """
    workspace = Path(workspace_dir).resolve()
    kernel_path = _relative_file(workspace, kernel, "kernel")
    driver_path = _driver_reference(workspace, driver)

    normalized_sources: list[str] = []
    for raw_path in [kernel, *source_files]:
        relative = _relative_file(workspace, raw_path, "source file")
        if relative not in normalized_sources:
            normalized_sources.append(relative)

    dirty = _git_value(
        workspace,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if dirty:
        raise ValueError("workspace has uncommitted tracked changes")

    branch = _git_value(workspace, "branch", "--show-current")
    if not branch or branch in {"main", "master"}:
        raise ValueError("forge-loop requires a non-main development branch in the workspace")
    resolved_base_commit = (base_commit or "").strip() or _git_value(workspace, "rev-parse", "HEAD")
    raw_source_paths = _campaign_source_paths(
        workspace,
        kernel_path,
        normalized_sources,
    )
    pristine_sources = _read_pristine_sources(
        workspace,
        raw_source_paths,
        base_commit=resolved_base_commit,
    )
    resolved_branch = (git_branch or "").strip() or branch
    kernel_backend_override = (kernel_backend or "").strip()
    resolved_kernel_backend = (
        resolve_kernel_backend_override(kernel_backend_override)
        if kernel_backend_override
        else infer_kernel_backend([workspace / path for path in normalized_sources])
    )
    resolved_targets = (
        list(target_functions)
        if target_functions
        else _derive_target_functions(
            workspace,
            normalized_sources,
            source_contents=pristine_sources,
        )
    )
    kernel_absolute = str((workspace / kernel_path).resolve())
    resolved_framework = infer_source_owner_framework(
        kernel_path=kernel_absolute,
        kernel_source=pristine_sources.get(kernel_absolute, ""),
        target_functions=resolved_targets,
        source_files=raw_source_paths,
        framework_override=(framework or "").strip(),
        source_contents=pristine_sources,
    )
    pristine_signature, pristine_identity = implementation_signature(
        workspace=str(workspace),
        kernel_path=kernel_absolute,
        source_files=raw_source_paths,
        framework=resolved_framework,
        source_contents=pristine_sources,
    )
    # Settled once, from the pristine sources, because it is part of the address
    # the campaign's experience is filed under. Left to be re-derived later it
    # would be read from whatever the loop has since written: a run that turns
    # eager code into its first GPU kernel would file its result under the name
    # of the kernel it just invented, at an address no read resolves to, and the
    # write would report success while the experience became unreachable.
    resolved_operator = (operator_name or "").strip() or resolve_operation(
        pristine_sources.get(kernel_absolute, ""),
        kernel_absolute,
        target_functions=resolved_targets,
    )
    program_md_path = ""
    program_md_sha256 = ""
    if program_md_file:
        source_program = Path(program_md_file).expanduser().resolve()
        if not source_program.is_file():
            raise ValueError(f"program context is not a file: {source_program}")
        program_md_path = "forge_experiments/program.md"
        program_text = source_program.read_text(errors="replace")
        program_md_sha256 = hashlib.sha256(program_text.encode()).hexdigest()

    resolved_task_type = (task_type or "").strip() or ("repository" if len(normalized_sources) > 1 else "")
    resolved_snr_threshold = float(snr_threshold)
    if not math.isfinite(resolved_snr_threshold) or resolved_snr_threshold <= 0:
        raise ValueError("SNR threshold must be a positive finite float")
    canonical_driver = workspace / driver_path
    return CampaignConfig(
        kernel_path=kernel_path,
        driver_path=driver_path,
        driver_sha256=hashlib.sha256(canonical_driver.read_bytes()).hexdigest(),
        source_files=normalized_sources,
        program_md_path=program_md_path,
        program_md_sha256=program_md_sha256,
        snr_threshold=resolved_snr_threshold,
        gpu_target=(gpu_target or "").strip() or detect_gpu_target(),
        gpu_type=str("mi355x" if gpu_type is None else gpu_type).strip().lower(),
        kernel_backend=resolved_kernel_backend,
        task_type=resolved_task_type,
        target_functions=resolved_targets,
        git_branch=resolved_branch,
        base_commit=resolved_base_commit,
        framework=resolved_framework,
        operator_name=resolved_operator,
        producer=str(producer or "").strip().lower(),
        implementation_signature=pristine_signature,
        implementation_identity=pristine_identity,
        nproc_per_node=max(1, int(nproc_per_node or 1)),
        bench_repeat=max(1, int(bench_repeat or 1)),
        commit_new_paths=normalize_commit_new_paths(commit_new_paths or []),
    )
