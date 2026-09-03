#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge submission backend running Kernel-Forge in an isolated worktree.

Emits optimized source plus an optimization_report.md artifact for integration.
"""

from __future__ import annotations

import ast
import fcntl
import json
import logging
import math
import os
import re
import signal
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent)
_TOOLS_DIR_INSERTED = _TOOLS_DIR not in sys.path
if _TOOLS_DIR_INSERTED:
    sys.path.insert(0, _TOOLS_DIR)
from _task_group_contract import (  # noqa: E402
    forge_shapes_from_candidate,
    logical_operator_name,
    task_group_shape_cases,
)
from _vendor_operator_playbooks import match_vendor_operator_playbook  # noqa: E402

if _TOOLS_DIR_INSERTED:
    sys.path.remove(_TOOLS_DIR)

_BACKENDS_DIR = str(Path(__file__).resolve().parent)
_BACKENDS_DIR_INSERTED = _BACKENDS_DIR not in sys.path
if _BACKENDS_DIR_INSERTED:
    sys.path.insert(0, _BACKENDS_DIR)
import _flydsl_rewrite  # noqa: E402

if _BACKENDS_DIR_INSERTED:
    sys.path.remove(_BACKENDS_DIR)

log = logging.getLogger(__name__)

_KNOWLEDGE_CONFIG_CACHE = None
_KNOWLEDGE_CONFIG_RESOLVED = False
_KNOWLEDGE_CONFIG_LOCK = threading.Lock()


def _reset_knowledge_config_cache() -> None:
    """Clear Forge's prevalidated knowledge configuration (tests only)."""

    global _KNOWLEDGE_CONFIG_CACHE, _KNOWLEDGE_CONFIG_RESOLVED
    _KNOWLEDGE_CONFIG_CACHE = None
    _KNOWLEDGE_CONFIG_RESOLVED = False


def _knowledge_config_for_forge():
    """Resolve process-level knowledge configuration once."""

    global _KNOWLEDGE_CONFIG_CACHE, _KNOWLEDGE_CONFIG_RESOLVED
    if _KNOWLEDGE_CONFIG_RESOLVED:
        return _KNOWLEDGE_CONFIG_CACHE

    with _KNOWLEDGE_CONFIG_LOCK:
        if _KNOWLEDGE_CONFIG_RESOLVED:
            return _KNOWLEDGE_CONFIG_CACHE

        from hyperloom.orchestrator.knowledge.config import KnowledgeConfig

        source = dict(os.environ)
        try:
            config = KnowledgeConfig.from_env(source)
        except Exception as exc:  # noqa: BLE001 - submit hot paths must remain available
            log.warning(
                "Forge knowledge configuration is invalid (%s); disabling remote "
                "knowledge for this process. Validate KNOWLEDGE_STORE_MODE and "
                "KB Store credentials during startup.",
                exc,
            )
            fallback = dict(source)
            fallback["KNOWLEDGE_STORE_MODE"] = "local"
            fallback.pop("KB_STORE_URL", None)
            fallback.pop("KB_STORE_TOKEN", None)
            config = KnowledgeConfig.from_env(fallback)
        _KNOWLEDGE_CONFIG_CACHE = config
        _KNOWLEDGE_CONFIG_RESOLVED = True
        return config


_FORGE_EXPERIMENT_ID = "hyperloom"
# Mirrors kernelforge.cli.MIN_MAX_HOURS (1.0h): forge-loop refuses a shorter
# runtime budget rather than running a non-productive campaign.
_FORGE_MIN_BUDGET_SEC = 3600
_FORGE_SHUTDOWN_GRACE_SEC = 30


def _forge_failure_tail(output: str, *, max_chars: int = 500) -> str:
    """Summarize why the forge child failed, for the error the caller reads.

    The whole transcript already goes to the forge log, which nobody opens while
    the only thing reaching the orchestrator is a return code -- so a producer
    that rejected its own argv looked identical to one that crashed measuring.

    A usage error outranks the tail: the CLI names it on one line and exits
    before emitting any of the progress output the tail would otherwise capture.
    Result sentinels are skipped because one such line is a whole JSON document
    and would crowd out everything else.
    """
    lines = [line.strip() for line in (output or "").splitlines() if line.strip() and "__FORGE_RESULT__" not in line]
    if not lines:
        return "no output"
    flagged = [line for line in lines if line.startswith(("Error:", "Usage:"))]
    text = " | ".join(flagged or lines[-3:])
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


class ForgeLoopOutcome(NamedTuple):
    """Result and recovery evidence from one forge-loop subprocess."""

    baseline_ms: float | None
    best_ms: float | None
    improved: bool
    output: str
    error: Exception | None
    timed_out: bool
    checkpoint: dict | None
    pristine_baseline_ms: float | None = None
    search_start_ms: float | None = None
    improved_during_search: bool = False
    structured_result: dict | None = None
    mean_case_speedup: float | None = None
    search_start_mean_case_speedup: float | None = None
    total_improved: bool = False
    incremental_improved: bool = False


class _WorktreePreparationError(RuntimeError):
    """A new isolated workspace could not be prepared safely."""


class _RetainedWorkspaceCollision(FileExistsError):
    """The requested workspace path already contains a retained attempt."""


# Platform -> gfx target.
_PLATFORM_TO_GFX = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}

# Triton/python source maps to the triton kernel_backend.
_SOURCE_TYPE_TO_KERNEL_BACKEND = {
    "triton": "triton",
    "python": "triton",
}

# Compiled-kernel kernel_backends. Opt out with FORGE_DISABLE_COMPILED_KERNEL_BACKENDS=1.
_COMPILED_SOURCE_TYPE_TO_KERNEL_BACKEND = {
    "hip_cpp": "hip",
    "hip": "hip",
    "cuda_cpp": "hip",
    "ck": "ck",
    "aiter": "aiter",
    "hipblaslt": "hipblaslt",
    "flydsl": "flydsl",
}


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing text output (never raises on non-zero)."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _git_argv(args: list[str], cwd: str | None = None) -> list[str]:
    """Build a ``git`` argv carrying a ``safe.directory`` exception for the target repo.

    ``args`` excludes the executable. The kernel repo is routinely bind-mounted
    and owned by another uid, which git refuses to read or write without this.
    """
    try:
        from hyperloom.common.git_safety import safe_directory_args  # noqa: PLC0415 - standalone import-light
    except ImportError:
        # tools/ scripts also run on remote nodes with no hyperloom installed.
        return ["git", *args]
    return ["git", *safe_directory_args(args, cwd=cwd)]


def _run_git(args: list[str], cwd: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run ``git <args>`` (``args`` excludes the executable) on a possibly foreign-owned repo."""
    return _run(_git_argv(args, cwd=cwd), cwd=cwd, timeout=timeout)


def _resolve_gpu_target(candidate: dict) -> str:
    """Resolve the gfx target: env GPU_TARGET -> candidate platform -> probe.

    Never hard-codes; falls back to rocminfo when nothing else is available.
    """
    env_target = (os.environ.get("GPU_TARGET") or os.environ.get("GPU_TYPE") or "").strip()
    if env_target:
        normalized = _normalize_gpu_target(env_target)
        if normalized:
            return normalized
    platform = str(candidate.get("platform") or candidate.get("arch") or "").strip().lower()
    normalized = _normalize_gpu_target(platform)
    if normalized:
        return normalized
    # Probe via rocminfo as a last resort.
    try:
        proc = _run(["rocminfo"], timeout=30)
        m = re.search(r"\bgfx\d+[a-z]*\b", proc.stdout or "")
        if m:
            return m.group(0)
    except Exception:
        pass
    # Honor the "never hard-codes" contract: a wrong default (e.g. gfx942 on a
    # gfx950 host) silently mis-targets kernel compilation. Fail loudly instead.
    raise RuntimeError(
        "Cannot resolve gfx target: set GPU_TARGET/GPU_TYPE or a candidate "
        "'platform', and ensure rocminfo is available."
    )


def _known_gpu_model(value: str) -> str:
    """Return the canonical card name, or "" when this is not one.

    The command line and the environment must agree on the model, so both
    render it through here rather than each trusting what it was handed.
    """
    model = str(value or "").strip().lower()
    return model if model in _PLATFORM_TO_GFX else ""


def _resolve_gpu_type(candidate: dict) -> str:
    """Resolve the hardware model: env GPU_TYPE -> candidate platform.

    KernelForge files a kernel's experience under the card it was measured on,
    not under the architecture it was compiled for. The two are not
    interchangeable: mi300x, mi308x and mi325x all build for gfx942 while
    differing in bandwidth and cache, so a recipe tuned on one is not a
    recommendation for the others, and the target cannot be reversed into a
    model. Returns "" when the model is unknown; KernelForge then declines to
    read or write rather than filing under an address nothing resolves to.
    """
    offered = (
        os.environ.get("GPU_TYPE"),
        candidate.get("platform"),
        candidate.get("arch"),
    )
    for raw in offered:
        model = _known_gpu_model(raw)
        if model:
            return model
    # Nothing downstream fails on this: the loop optimizes, the result looks
    # ordinary, and only the experience is missing. So it is said here.
    rejected = ", ".join(repr(str(v)) for v in offered if str(v or "").strip())
    log.warning(
        "forge: no known hardware model for this run%s; kernel experience is "
        "addressed by model, so this run has no address to read or record one. "
        "Set GPU_TYPE to a card such as %s.",
        f" (offered {rejected})" if rejected else "",
        ", ".join(sorted(_PLATFORM_TO_GFX)),
    )
    return ""


def _apply_gpu_type_env(env: dict, gpu_type: str) -> None:
    """Hand the child a hardware model, or none at all.

    The child inherits this process's environment, where ``GPU_TYPE`` is also
    accepted as a way to name a gfx target. Passing that through would file the
    run's experience under ``gfx950`` as though it were a card, so an
    unresolved model is removed rather than forwarded: KernelForge then declines
    to read or write instead of addressing a record by a value that means
    something else. The reason it could not be resolved is reported by
    :func:`_resolve_gpu_type`, which is where it is known.
    """
    model = _known_gpu_model(gpu_type)
    if model:
        env["GPU_TYPE"] = model
    else:
        env.pop("GPU_TYPE", None)


def _normalize_gpu_target(value: str) -> str:
    """Return a canonical lowercase gfx architecture or an empty string."""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    mapped = _PLATFORM_TO_GFX.get(normalized)
    if mapped:
        return mapped
    match = re.search(r"\bgfx\d+[a-z]*\b", normalized)
    return match.group(0) if match else ""


# Framework aliases -> canonical KB framework identity. MUST stay in sync with
# the arena launcher's _FRAMEWORK_ALIASES so producer/consumer agree. aiter_meta
# is aiter's C++/CK companion package and shares aiter's identity.
_FRAMEWORK_ALIASES = {
    "vllm": "vllm",
    "sglang": "sglang",
    "aiter": "aiter",
    "aiter_meta": "aiter",
}


def _framework_from_path(path: str) -> str:
    """First (shallowest == owning package) known framework component in a path."""
    for comp in Path(path).parts:
        canon = _FRAMEWORK_ALIASES.get(comp.lower())
        if canon:
            return canon
    return ""


def _resolve_framework(candidate: dict, kernel_path: str = "") -> str:
    """Best-effort framework identity for the KB slug. Empty == let forge-loop infer.

    framework is a SOFT slug component, so this never raises and never guesses a
    wrong value: it returns a framework only when confident, else "" so the
    caller omits ``--framework`` and forge-loop falls back to its own path scan
    (then ``unknown``). Passing it explicitly matters because a producer (arena)
    and consumer (hyperloom) can have different workspace layouts — pinning the
    framework keeps both on the SAME kernel page. Resolution order:

      1. an explicit, recognized ``source_framework`` on the candidate;
      2. the owning package of a KERNEL SOURCE definition file
         (``kernel_sources``) — this is where the real compute kernel lives,
         which can be aiter even when the traced entry/anchor is a vLLM/SGLang
         dispatch that merely CALLS it; matching the definition keeps the slug
         aligned with the arena producer;
      3. the owning framework package in the kernel path — scanned shallowest
         first so a kernel that lives DIRECTLY in vllm/sglang (e.g.
         ``.../vllm/model_executor/layers/fused_moe/...``) resolves to that
         package, not a deep subdir name;
      4. "" (defer to forge-loop).
    """
    raw = str((candidate or {}).get("source_framework") or "").strip().lower()
    canon = _FRAMEWORK_ALIASES.get(raw)
    if canon:
        return canon
    kernel_sources = (candidate or {}).get("kernel_sources") or []
    if isinstance(kernel_sources, str):
        kernel_sources = [kernel_sources]
    for src in kernel_sources:
        framework = _framework_from_path(str(src))
        if framework:
            return framework
    candidate_source = str((candidate or {}).get("source_file") or "").strip()
    if candidate_source:
        framework = _framework_from_path(candidate_source)
        if framework:
            return framework
    return _framework_from_path(kernel_path)


def _logical_operator(candidate: dict | None) -> str:
    """Derive the stable logical operator without conflating implementation symbols."""
    return logical_operator_name(candidate)


def _stable_implementation_symbols(
    candidate: dict | None,
    invocation_spec_file: str = "",
    source_files: list[str] | None = None,
) -> list[str]:
    """Collect curated and source-level symbols for ``--target-functions``."""
    candidate = candidate or {}
    values: list[object] = []
    for key in ("source_symbol", "target_functions"):
        value = candidate.get(key)
        if value:
            values.extend(value if isinstance(value, (list, tuple)) else [value])
    if invocation_spec_file:
        try:
            spec = json.loads(Path(invocation_spec_file).read_text(encoding="utf-8"))
            implementation = spec.get("implementation") if isinstance(spec, dict) else {}
            if isinstance(implementation, dict):
                values.extend(implementation.get("symbols") or [])
        except (OSError, json.JSONDecodeError, TypeError):
            # The optional spec only augments symbols; source inspection remains available.
            pass
    for source_file in source_files or []:
        try:
            source_text = Path(source_file).read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue
        if str(source_file).endswith((".py", ".pyi")):
            try:
                tree = ast.parse(source_text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = []
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if isinstance(target, ast.Name):
                        decorators.append(target.id)
                    elif isinstance(target, ast.Attribute):
                        decorators.append(target.attr)
                if "jit" in decorators:
                    values.append(node.name)
            continue
        values.extend(
            match.group(1)
            for match in re.finditer(
                r"\b(?:__global__|__device__)\b[^;{}]*?\b"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                source_text,
            )
        )
    symbols: list[str] = []
    for value in values:
        symbol = str(value or "").strip()
        if not symbol or symbol.endswith("...") or any(character.isspace() for character in symbol):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _resolve_kernel_kind(source_type: str, kernel_kind: str) -> str:
    """Use explicit kind first, then proven source classifications."""
    explicit = str(kernel_kind or "").strip().lower().replace("-", "_")
    if explicit:
        return explicit
    proven = str(source_type or "").strip().lower()
    if proven in {"triton", "flydsl", "ck"}:
        return proven
    return ""


# ``FORGE_DISABLE_COMPILED_FELLOWS`` was this knob's name before the
# fellow -> kernel_backend rename. It cannot simply be dropped: ``FORGE_`` is on
# env_safety's dotenv prefix allowlist, so an operator's old value is still
# forwarded into the run and then ignored, which silently re-enables the
# compiled kernel backends they had switched off. Honouring the old spelling
# would keep the retired vocabulary alive, so it is refused instead -- once, and
# loudly enough to be actionable.
_RETIRED_COMPILED_OPT_OUT = "FORGE_DISABLE_COMPILED_FELLOWS"
_retired_opt_out_warned = False


def _warn_on_retired_compiled_opt_out() -> None:
    """Warn once if the pre-rename opt-out variable is still set."""
    global _retired_opt_out_warned
    if _retired_opt_out_warned or not os.environ.get(_RETIRED_COMPILED_OPT_OUT, "").strip():
        return
    _retired_opt_out_warned = True
    log.warning(
        "%s is set but no longer read; it was renamed to "
        "FORGE_DISABLE_COMPILED_KERNEL_BACKENDS. Compiled kernel backends are "
        "ENABLED for this run -- set the new name to keep them off.",
        _RETIRED_COMPILED_OPT_OUT,
    )


def _kernel_backend_for_source_type(source_type: str) -> str | None:
    """Map source_type to a Forge kernel_backend. None if unsupported.

    Triton/python map to triton. Compiled source types
    (hip_cpp/ck/aiter/hipblaslt/flydsl) map to their native kernel backend by default;
    opt out with FORGE_DISABLE_COMPILED_KERNEL_BACKENDS=1 for triton-only.
    """
    st = (source_type or "").strip().lower()
    kernel_backend = _SOURCE_TYPE_TO_KERNEL_BACKEND.get(st)
    if kernel_backend is not None:
        return kernel_backend
    _warn_on_retired_compiled_opt_out()
    if os.environ.get("FORGE_DISABLE_COMPILED_KERNEL_BACKENDS", "").strip().lower() in ("1", "true", "yes"):
        return None
    return _COMPILED_SOURCE_TYPE_TO_KERNEL_BACKEND.get(st)


def _resolve_kernel_backend(source_type: str, kernel_kind: str) -> str | None:
    """Resolve the kernel backend deterministically from language and curated kernel kind."""
    kind = str(kernel_kind or "").strip().lower().replace("-", "_")
    if "flydsl" in kind:
        return _kernel_backend_for_source_type("flydsl")
    if kind == "ck" or kind.endswith("_ck") or kind.startswith("ck_"):
        return _kernel_backend_for_source_type("ck")
    if "triton" in kind:
        return _kernel_backend_for_source_type("triton")
    return _kernel_backend_for_source_type(source_type)


def _git_toplevel(path: str) -> str:
    """Return the git repo root containing `path`, or '' if not a git repo."""
    try:
        proc = _run_git(["-C", str(Path(path).parent), "rev-parse", "--show-toplevel"], timeout=30)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _default_branch(repo: str) -> str:
    """Best-effort default branch name for `repo` (e.g. 'main'/'master').

    Prefers the remote's advertised default, then falls back to common local
    branch names.
    """
    p = _run_git(["-C", repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], timeout=30)
    ref = (p.stdout or "").strip()
    if ref.startswith("origin/"):
        return ref[len("origin/") :]
    for name in ("main", "master"):
        if _run_git(["-C", repo, "rev-parse", "--verify", name], timeout=30).returncode == 0:
            return name
    return ""


def _new_forge_branch(output_dir: Path, source_file: str) -> str:
    """Return a valid, unique retained branch name for one Forge attempt."""

    def _component(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
        return cleaned or fallback

    session_id = _component(output_dir.parent.name, "session")
    kernel_id = _component(Path(source_file).stem, "kernel")
    return f"forge/{session_id}/{kernel_id}-{uuid.uuid4().hex[:12]}"


def _prepare_worktree(source_file: str, kernel_repo: str, output_dir: Path, branch: str) -> tuple[str, str, str] | None:
    """Create a git worktree of kernel_repo at output_dir/worktree (R1/W1).

    Returns (worktree_dir, worktree_kernel_file, base_commit) or None when the
    repo is not a clean git checkout / source_file is not tracked (forge then
    skips, never mutating the live repo). base_commit is the commit the worktree
    was created at (HEAD); export diffs the best state against it.
    """
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo or not (Path(repo) / ".git").exists():
        return None
    src_abs = Path(source_file).resolve()
    try:
        rel = src_abs.relative_to(Path(repo).resolve())
    except ValueError:
        return None  # source_file not inside the repo

    # Being inside the repo is not the same as being tracked by it. A framework
    # tree can host a git repo that indexes only part of itself (a scratch repo
    # over site-packages that only added ``vllm/``, say). ``git worktree add``
    # then succeeds and produces a worktree WITHOUT the kernel, and the failure
    # surfaces far downstream as "prepared kernel does not exist". Fall back to
    # the no-git scratch path instead, which copies the file in.
    tracked = _run_git(["-C", repo, "ls-files", "--error-unmatch", "--", rel.as_posix()], timeout=30)
    if tracked.returncode != 0:
        return None

    wt = output_dir / "worktree"
    # A prior attempt at this path is retained for inspection. Never remove or
    # reuse it, and never let the caller reinterpret it as a no-git scratch.
    if wt.exists() or wt.is_symlink():
        raise _RetainedWorkspaceCollision(f"retained Forge workspace already exists: {wt}")
    _run_git(["-C", repo, "worktree", "prune"], timeout=60)

    base = _run_git(["-C", repo, "rev-parse", "--verify", "HEAD"], timeout=30)
    if base.returncode != 0 or not base.stdout.strip():
        raise _WorktreePreparationError("could not resolve the source repository HEAD")
    base_commit = base.stdout.strip()
    add = _run_git(["-C", repo, "worktree", "add", "-b", branch, str(wt), "HEAD"], timeout=120)
    if add.returncode != 0:
        raise _WorktreePreparationError("git worktree creation failed: " + (add.stderr.strip() or add.stdout.strip()))

    return str(wt), str(wt / rel), base_commit


def _remap_implementation_sources(
    *,
    candidate: dict,
    source_file: str,
    workspace: str,
    worktree_kernel: str,
    kernel_repo: str,
) -> list[str]:
    """Map all editable sources into the prepared workspace and include the anchor."""
    workspace_path = Path(workspace).resolve()
    original_anchor = Path(source_file).expanduser().resolve()
    mapped_anchor = Path(worktree_kernel).expanduser().resolve()
    try:
        anchor_relative = mapped_anchor.relative_to(workspace_path)
    except ValueError as error:
        raise _WorktreePreparationError(f"prepared kernel escapes Forge workspace: {mapped_anchor}") from error
    if not mapped_anchor.is_file():
        raise _WorktreePreparationError(f"prepared kernel does not exist: {mapped_anchor}")

    original_roots: list[Path] = []
    if kernel_repo:
        original_roots.append(Path(kernel_repo).expanduser().resolve())
    if len(anchor_relative.parts) <= len(original_anchor.parts):
        root = original_anchor
        for _ in anchor_relative.parts:
            root = root.parent
        original_roots.append(root)

    raw_kernel_sources = candidate.get("kernel_sources") or []
    raw_sources = [raw_kernel_sources] if isinstance(raw_kernel_sources, str) else list(raw_kernel_sources)
    raw_sources.append(source_file)
    remapped: list[str] = []
    for raw_source in raw_sources:
        raw = str(raw_source or "").strip()
        if not raw:
            continue
        original = Path(raw).expanduser()
        if not original.is_absolute():
            base = Path(kernel_repo).expanduser() if kernel_repo else original_anchor.parent
            original = base / original
        original = original.resolve()

        candidates: list[Path] = []
        try:
            original.relative_to(workspace_path)
            candidates.append(original)
        except ValueError:
            # Sources outside the workspace may still map through a known root below.
            pass
        for root in original_roots:
            try:
                candidates.append(workspace_path / original.relative_to(root))
            except ValueError:
                continue
        if original == original_anchor:
            candidates.insert(0, mapped_anchor)

        mapped = next(
            (
                path.resolve()
                for path in candidates
                if path.is_file() and _path_is_within(path.resolve(), workspace_path)
            ),
            None,
        )
        if mapped is None:
            raise _WorktreePreparationError(
                "declared implementation source could not be mapped into the "
                f"prepared workspace: source={original} workspace={workspace_path}"
            )
        mapped_value = str(mapped)
        if mapped_value not in remapped:
            remapped.append(mapped_value)

    anchor_value = str(mapped_anchor)
    if anchor_value in remapped:
        remapped.remove(anchor_value)
    remapped.insert(0, anchor_value)
    return remapped


def _pkg_toplevel(source_file: str) -> str:
    """Return the topmost importable package directory containing ``source_file``.

    Ascends while an ``__init__.py`` is present and returns the *last* directory
    that still has one — i.e. the root package directory itself (e.g. ``vllm/``
    for ``.../dist-packages/vllm/model_executor/models/deepseek_v2.py``), NOT its
    parent. Its parent is the directory you would add to ``sys.path``; use
    :func:`_pkg_sys_path_root` for that.

    Falls back to the parent directory of ``source_file`` when the file is not
    part of a package (no ``__init__.py`` beside it).
    """
    parent = Path(source_file).resolve().parent
    if not (parent / "__init__.py").exists():
        # Not inside a package — the file's own directory is the top level.
        return str(parent)
    top = parent
    while (top.parent / "__init__.py").exists():
        top = top.parent
    return str(top)


def _pkg_sys_path_root(source_file: str) -> str:
    """Return the directory to place on ``sys.path`` / ``PYTHONPATH``.

    This is the parent of the topmost importable package (so ``import <pkg>``
    resolves), or ``source_file``'s own directory when it is not part of a
    package.
    """
    top = Path(_pkg_toplevel(source_file))
    parent = Path(source_file).resolve().parent
    if str(top) == str(parent) and not (parent / "__init__.py").exists():
        # Non-package file: its own directory is already the import root.
        return str(parent)
    return str(top.parent)


def _is_on_network_fs(path: Path) -> bool:
    """True when ``path`` or its nearest existing ancestor is on a network FS.

    Imported inside the call: this file also runs as a standalone script on
    nodes with no ``hyperloom`` on the path, the invariant
    ``test_forge_submit_stays_import_light`` pins.
    """
    from hyperloom.common.fs_utils import is_network_fs  # noqa: PLC0415

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return is_network_fs(str(probe))


def _local_scratch_dir(output_dir: Path) -> Path:
    """Return where the scratch worktree belongs for ``output_dir``.

    A worktree is throwaway but gets copied and git-hashed in full, so on a
    network mount it is placed on local disk instead. Only the worktree moves;
    the durable archive under ``output_dir`` stays where the operator put it.
    Override the local root with ``$FORGE_LOCAL_SCRATCH_ROOT``.

    The local tree mirrors ``<session_id>/<attempt>`` from the durable side.
    The attempt name alone is the kernel's, so two sessions optimizing one
    kernel would claim one path and the second would be refused for a collision
    with the first. Carrying the session id also scopes the owner marker that
    lets the sweep identify live trees regardless of durable-root differences.
    """
    if not _is_on_network_fs(output_dir):
        return output_dir / "worktree"
    root_env = os.environ.get("FORGE_LOCAL_SCRATCH_ROOT", "").strip()
    local_root = Path(root_env).expanduser() if root_env else Path.home() / ".cache" / "hyperloom" / "forge_scratch"
    session_dir = local_root / output_dir.parent.name
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_scratch_owner(session_dir)
    _sweep_orphaned_scratch(local_root, output_dir.parent.name)
    return session_dir / output_dir.name / "worktree"


def _scratch_owner_file(session_dir: Path) -> Path:
    return session_dir / ".owner"


def _write_scratch_owner(session_dir: Path) -> None:
    """Record pid and start-time so the sweep can tell live from dead sessions."""
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            starttime = fh.read().split()[21]
    except OSError:
        starttime = "0"
    _scratch_owner_file(session_dir).write_text(f"{pid}:{starttime}", encoding="ascii")


def _scratch_owner_alive(session_dir: Path) -> bool:
    """True when the process that created this scratch session is still running."""
    try:
        text = _scratch_owner_file(session_dir).read_text(encoding="ascii").strip()
        pid_str, starttime = text.split(":", 1)
        pid = int(pid_str)
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # EPERM proves the pid exists, owned by another uid
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            current_starttime = fh.read().split()[21]
        return current_starttime == starttime
    except OSError:
        return True  # unreadable stat: assume live to avoid deleting a running tree


def _sweep_orphaned_scratch(local_root: Path, current_session: str) -> None:
    """Delete local scratch whose owning process is no longer running."""
    for child in local_root.iterdir():
        if not child.is_dir() or child.name == current_session:
            continue
        if not any(child.glob("*/worktree")):
            continue
        if not _scratch_owner_alive(child):
            shutil.rmtree(child, ignore_errors=True)
            log.info("forge: swept orphaned scratch session %s", child)


def _prepare_worktree_nogit(
    source_file: str,
    kernel_repo: str,
    output_dir: Path,
    branch: str,
) -> tuple[str, str, str] | None:
    """Ephemeral git-scaffold scratch worktree for non-git source trees (scheme A).

    When ``source_file`` lives outside any git repository (e.g. a pip-installed
    package under ``/usr/local/lib/python3.12/dist-packages/``), this function:

    1. Determines the scratch layout root (== the PYTHONPATH root): the explicit
       ``kernel_repo`` when provided, otherwise the *parent* of the single
       top-level package containing ``source_file`` (so ``import <pkg>`` still
       resolves from the scratch copy).
    2. Copies only what is needed to ``output_dir/worktree`` — the whole tree
       for an explicit ``kernel_repo``, but for a pip-installed package only that
       one top-level package subtree (e.g. ``vllm/``), NEVER the entire
       ``dist-packages``/``site-packages`` directory (which would copy every
       installed package — torch, vllm, ... — 5-15 GB per submit, risking
       ENOSPC). Ignores ``.git`` and the producer's runtime-artefact names.
    3. ``git init`` + sets ``user.name``/``user.email`` + excludes regenerated
       bytecode caches + ``git add -A`` + initial commit so Forge's
       ``IterationLoop`` (which uses ``git commit``/``reset --hard``) can manage
       its iterative keep/revert loop.
    4. Returns ``(scratch_dir, scratch_kernel_file, base_commit)`` with the same
       signature as :func:`_prepare_worktree`.

    The measurement driver is staged inside this root and executed from it, so
    the scratch copy shadows the dist-packages install at import time
    (pure-Python only; editable-finder installs are excluded — those are handled
    by :func:`_prepare_inplace`).

    Returns ``None`` on any error (e.g. ``shutil.copytree`` failure).

    .. note::
        This path is intentionally **not** used for editable-finder packages.
        Those are detected by :func:`_needs_inplace` before this function is
        ever called.
    """
    src_abs = Path(source_file).resolve()

    # Scratch layout root == the directory placed on PYTHONPATH. Honour an
    # explicit kernel_repo; otherwise derive the single top-level package's
    # parent (not the whole dist-packages dir — ENOSPC risk).
    if kernel_repo:
        layout_root = Path(kernel_repo).resolve()
        copy_subtrees: list[Path] | None = None  # copy the whole repo
    else:
        layout_root = Path(_pkg_sys_path_root(source_file))
        pkg_top = Path(_pkg_toplevel(source_file))
        # Copy only the top-level package subtree, unless the file is not part
        # of a package.
        copy_subtrees = None if str(pkg_top) == str(layout_root) else [pkg_top]

    try:
        rel = src_abs.relative_to(layout_root)
    except ValueError:
        # source_file not inside layout_root — fall back to a flat copy of just
        # its parent dir. This DROPS the framework directory structure from the
        # kernel path, which impairs cross-repo KB reuse: the slug's framework
        # component now relies entirely on the explicit --framework we forward
        # (see _resolve_framework), and a KB diff produced with the full repo
        # path applies here only via forge-loop's strip-depth normalization.
        # Surface it rather than degrade silently.
        log.warning(
            "forge: kernel %s is outside its package root %s; using a FLAT "
            "scratch layout. KB framework detection falls back to the explicit "
            "--framework, and cross-workspace diff apply relies on strip-depth "
            "normalization. Pass an explicit kernel_repo to preserve structure.",
            src_abs,
            layout_root,
        )
        layout_root = src_abs.parent
        rel = Path(src_abs.name)
        copy_subtrees = None

    scratch_dir = _local_scratch_dir(output_dir)
    if scratch_dir.exists() or scratch_dir.is_symlink():
        raise _RetainedWorkspaceCollision(f"retained Forge workspace already exists: {scratch_dir}")
    if not branch or branch in {"main", "master"}:
        raise _WorktreePreparationError("no-git scratch requires a supplied non-main Forge branch")

    # In-call for the reason ``_is_on_network_fs`` states. The copy takes the
    # narrow set; only the index may drop what a package is imported through.
    from kernelforge.loop.path_ownership import (  # noqa: PLC0415
        COPY_FILTER_DIRECTORY_NAMES,
        RUNTIME_DIRECTORY_GLOBS,
        RUNTIME_DIRECTORY_NAMES,
        RUNTIME_FILE_SUFFIXES,
    )
    import fnmatch as _fnmatch  # noqa: PLC0415

    skipped_dirs = RUNTIME_DIRECTORY_NAMES | COPY_FILTER_DIRECTORY_NAMES | {".git"}
    runtime_suffixes = tuple(RUNTIME_FILE_SUFFIXES)
    dir_globs = RUNTIME_DIRECTORY_GLOBS

    def _ignore(directory: str, names: list[str]) -> list[str]:
        return [
            n
            for n in names
            if n in skipped_dirs or n.endswith(runtime_suffixes) or any(_fnmatch.fnmatchcase(n, g) for g in dir_globs)
        ]

    try:
        if copy_subtrees is None:
            # Whole layout_root.
            shutil.copytree(str(layout_root), str(scratch_dir), ignore=_ignore)
        else:
            # Only the named top-level package(s), preserving their path relative
            # to layout_root so ``import <pkg>`` still resolves.
            scratch_dir.mkdir(parents=True, exist_ok=True)
            for sub in copy_subtrees:
                dest = scratch_dir / sub.relative_to(layout_root)
                shutil.copytree(str(sub), str(dest), ignore=_ignore)
    except OSError as exc:
        log.warning("forge: non-git scratch copy failed (root=%s): %s", layout_root, exc)
        shutil.rmtree(scratch_dir, ignore_errors=True)
        return None

    def _scaffold(cmds: list[list[str]]) -> bool:
        for cmd in cmds:
            # 120s, not less: the baseline ``git add -A`` hashes every copied
            # extension module, and a framework package carries GBs of them.
            try:
                proc = _run_git(cmd, timeout=120)
            except subprocess.TimeoutExpired:
                log.warning("forge: non-git scaffold git step timed out: %s", cmd)
                shutil.rmtree(scratch_dir, ignore_errors=True)
                return False
            if proc.returncode != 0:
                log.warning(
                    "forge: non-git scaffold git init step failed: %s -> %s",
                    cmd,
                    proc.stderr.strip() or proc.stdout.strip(),
                )
                shutil.rmtree(scratch_dir, ignore_errors=True)
                return False
        return True

    # Bootstrap a real git repo so IterationLoop's commit/revert works.
    if not _scaffold(
        [
            ["-C", str(scratch_dir), "init", "-b", branch],
            ["-C", str(scratch_dir), "config", "user.name", "forge-bot"],
            ["-C", str(scratch_dir), "config", "user.email", "forge-bot@local"],
        ]
    ):
        return None

    # Must precede the baseline `git add -A`, so the pattern is in force for
    # every commit the loop later makes against this repository.
    _exclude_runtime_artifacts(scratch_dir)

    if not _scaffold(
        [
            ["-C", str(scratch_dir), "add", "-A"],
            ["-C", str(scratch_dir), "commit", "-q", "-m", "forge: scratch baseline"],
        ]
    ):
        return None

    base_commit_proc = _run_git(["-C", str(scratch_dir), "rev-parse", "HEAD"], timeout=30)
    if base_commit_proc.returncode != 0:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        return None
    base_commit = base_commit_proc.stdout.strip()
    scratch_kernel = str(scratch_dir / rel)
    log.info("forge: non-git scratch worktree ready at %s (kernel=%s)", scratch_dir, scratch_kernel)
    return str(scratch_dir), scratch_kernel, base_commit


def _editable_roots() -> list[str]:
    """Collect filesystem roots of PEP 660 editable-finder installs.

    Scans site-packages for ``__editable__*.pth`` and ``__editable___*_finder.py``
    and extracts the absolute paths they map into. Such packages are imported via
    a sys.meta_path finder that points at the *live* repo and CANNOT be overridden
    by PYTHONPATH, so a git worktree copy is never imported.

    Handles two finder layouts:
      1. Path-string .pth files that contain absolute paths in quotes.
      2. Setuptools-style .pth files that ``import __editable___<pkg>_finder``;
         the finder .py has a ``MAPPING`` dict mapping package names to paths.
    """
    roots: set[str] = set()
    seen_dirs: set[str] = set()
    scan_dirs = list(sys.path)
    try:
        scan_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            scan_dirs.append(site.getusersitepackages())
        except Exception:
            pass
    # Venv / conda site-packages may not appear in sys.path; probe conventional
    # locations for sys.prefix, VIRTUAL_ENV, CONDA_PREFIX, and the interpreter.
    _pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    _prefixes = {sys.prefix, sys.exec_prefix, sys.base_prefix}
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        v = os.environ.get(var)
        if v:
            _prefixes.add(v)
    # Derive the venv from the interpreter path.
    _interp = os.path.realpath(sys.executable)
    if os.sep + "bin" + os.sep in _interp:
        _prefixes.add(_interp.rsplit(os.sep + "bin" + os.sep, 1)[0])
    for prefix in _prefixes:
        for sub in (f"lib/{_pyver}/site-packages", f"lib/{_pyver}/dist-packages"):
            cand = os.path.join(prefix, sub)
            if os.path.isdir(cand):
                scan_dirs.append(cand)
    for d in scan_dirs:
        if not d or d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if not n.startswith("__editable__"):
                continue
            if not (n.endswith(".pth") or n.endswith("_finder.py")):
                continue
            fpath = os.path.join(d, n)
            try:
                with open(fpath, errors="replace") as _fh:
                    txt = _fh.read()
            except OSError:
                continue
            # Layout 0: bare absolute path on a line (no quotes, no import).
            for line in txt.splitlines():
                line = line.strip()
                if line.startswith("/") and not line.startswith("#") and "import" not in line and os.path.isdir(line):
                    roots.add(os.path.realpath(line))
            # Layout 1: quoted absolute paths directly in the file.
            for m in re.findall(r"['\"](/[^'\"]+)['\"]", txt):
                if os.path.isdir(m):
                    roots.add(os.path.realpath(m))
            # Layout 2: .pth imports a _finder.py; read its MAPPING dict for
            # paths. The finder file lives next to the .pth in site-packages.
            if n.endswith(".pth"):
                fm = re.search(r"import\s+(__editable___\w+_finder)", txt)
                if fm:
                    finder_file = os.path.join(d, fm.group(1) + ".py")
                    try:
                        with open(finder_file, errors="replace") as _fh2:
                            ftxt = _fh2.read()
                    except OSError:
                        continue
                    for m in re.findall(r"['\"](/[^'\"]+)['\"]", ftxt):
                        if os.path.isdir(m):
                            roots.add(os.path.realpath(m))
    return sorted(roots)


def _needs_inplace(kernel_repo: str) -> bool:
    """True when kernel_repo is (or contains/sits under) an editable-finder root.

    In that case forge must edit the live repo in place (the finder imports the
    live path; a worktree copy would be invisible -> the loop would no-op).
    """
    if not kernel_repo:
        return False
    repo = os.path.realpath(kernel_repo)
    for r in _editable_roots():
        if r == repo or r.startswith(repo + os.sep) or repo.startswith(r + os.sep):
            return True
    return False


class _RepoLock:
    """Owned in-place repo lock; released explicitly after restore."""

    def __init__(self, fh) -> None:
        self._fh = fh

    @property
    def fd(self) -> int:
        return self._fh.fileno()

    def close(self) -> None:
        self._fh.close()


def _acquire_repo_lock(repo: str) -> _RepoLock | None:
    """Take a non-blocking exclusive lock on the live repo for in-place editing.

    In-place mode mutates the shared live repo, so two concurrent forge sessions
    on the same repo would race. The lock serializes them; a caller that cannot
    get it must skip in-place. Returns the held lock (release with
    _release_repo_lock) or None when already held.
    """
    lock_path = os.path.join(repo, ".git", "forge_inplace.lock")
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        os.chmod(lock_path, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return _RepoLock(fh)


def _release_repo_lock(lock: _RepoLock | None) -> None:
    """Release + close the in-place repo lock (best-effort)."""
    if lock is None:
        return
    try:
        fcntl.flock(lock.fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        lock.close()
    except OSError:
        pass


def _prepare_inplace(
    source_file: str,
    kernel_repo: str,
    branch: str,
    *,
    lock_fd: _RepoLock | None = None,
) -> tuple[str, str, dict] | None:
    """In-place mode (Option 1): edit the LIVE repo so an editable-finder import
    sees the changes. Snapshots the original branch/HEAD + source bytes for a
    per-file restore in finally. Returns (workspace=repo, kernel_file=source_file,
    restore_info) or None when the repo is not a usable git checkout.

    Safety:
      - if HEAD is already on a forge/ temp branch (a prior crashed/SIGKILL'd
        run that never restored), AUTO-RECOVER: force-checkout the repo's
        default branch and delete the stale temp branch, then proceed from a
        pristine baseline (falls back to skip only if the default branch can't
        be resolved),
      - hold a per-repo lock so concurrent forge runs never interleave,
      - dirty working trees are allowed and preserved: the caller may record a
        tracked-baseline patch and the untracked inventory, which
        ``_restore_inplace`` replays so uncommitted work survives the campaign.
        Files the campaign itself created are removed on restore; there is
        still no ``reset --hard``.
    """
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo or not (Path(repo) / ".git").exists():
        _release_repo_lock(lock_fd)
        return None
    if not Path(source_file).is_file():
        _release_repo_lock(lock_fd)
        return None
    try:
        relpath = str(Path(source_file).resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        _release_repo_lock(lock_fd)
        return None  # source not inside repo

    # Serialize in-place runs on this repo before touching any git state.
    lock_fd = lock_fd or _acquire_repo_lock(repo)
    if lock_fd is None:
        return None  # another forge in-place run holds this repo; skip cleanly

    def _skip() -> None:
        """Release the lock and report the repo as unusable."""
        _release_repo_lock(lock_fd)
        return None

    try:
        orig_branch = _run_git(["-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], timeout=30).stdout.strip()
        orig_head = _run_git(["-C", repo, "rev-parse", "HEAD"], timeout=30).stdout.strip()
        if not orig_head:
            return _skip()
        # Auto-recover from a leftover forge temp branch: force the repo back
        # onto its default branch and delete the stale temp branch.
        if orig_branch.startswith("forge/"):
            default_branch = _default_branch(repo)
            if not default_branch:
                return _skip()
            stale = orig_branch
            co = _run_git(["-C", repo, "checkout", "-f", default_branch], timeout=120)
            if co.returncode != 0:
                return _skip()
            _run_git(["-C", repo, "branch", "-D", stale], timeout=30)
            orig_branch = default_branch
            orig_head = _run_git(["-C", repo, "rev-parse", "HEAD"], timeout=30).stdout.strip()
            if not orig_head:
                return _skip()
        # Drop any stale temp branch from a prior crashed run.
        _run_git(["-C", repo, "branch", "-D", branch], timeout=30)
        # Snapshot the source_file bytes on disk (restored exactly on exit).
        try:
            backup = Path(source_file).read_bytes()
        except OSError:
            return _skip()
        # Create a temp branch for the forge loop to commit/revert on (deleted
        # in _restore_inplace).
        cb = _run_git(["-C", repo, "checkout", "-b", branch], timeout=60)
        if cb.returncode != 0:
            return _skip()
        # Snapshot any pre-existing dirty tracked files as a baseline commit so
        # a later revert can't destroy them. base_commit is the pre-forge tree
        # that agent edits stack on top of; when the tree is clean it equals
        # orig_head.
        _run_git(["-C", repo, "add", "-u"], timeout=60)
        dirty = _run_git(["-C", repo, "diff", "--cached", "--quiet"], timeout=30)
        if dirty.returncode != 0:
            _run_git(
                [
                    "-C",
                    repo,
                    "-c",
                    "user.name=forge-bot",
                    "-c",
                    "user.email=forge-bot@local",
                    "commit",
                    "-m",
                    "forge: pre-existing dirty baseline",
                ],
                timeout=60,
            )
            base_commit = _run_git(["-C", repo, "rev-parse", "HEAD"], timeout=30).stdout.strip() or orig_head
        else:
            base_commit = orig_head
    except Exception:
        _release_repo_lock(lock_fd)
        raise

    restore = {
        "repo": repo,
        "orig_branch": orig_branch,
        "orig_head": orig_head,
        "branch": branch,
        "source_file": source_file,
        "backup": backup,
        "relpath": relpath,
        "lock_fd": lock_fd,
        "base_commit": base_commit,
    }
    return repo, source_file, restore


def _untracked_paths(repo: str) -> set[str]:
    """Return untracked repository paths without shell quoting."""
    proc = _run_git(["-C", repo, "ls-files", "--others", "--exclude-standard", "-z"], timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"could not inspect untracked files in {repo}")
    return {path for path in (proc.stdout or "").split("\0") if path}


def _remove_new_untracked(repo: str, baseline: set[str]) -> None:
    """Remove only untracked paths created after the baseline."""
    root = Path(repo)
    created = _untracked_paths(repo) - baseline
    for relpath in sorted(
        created,
        key=lambda value: len(Path(value).parts),
        reverse=True,
    ):
        relative = Path(relpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe untracked path: {relpath}")
        target = root / relative
        try:
            if target.is_symlink() or target.is_file():
                target.unlink()
        except OSError as exc:
            raise RuntimeError(f"could not remove campaign file: {target}") from exc
        parent = target.parent
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _apply_tracked_baseline(repo: str, patch: bytes) -> None:
    """Restore a journaled tracked baseline patch to the working tree."""
    if not patch:
        return
    proc = subprocess.run(
        _git_argv(["-C", repo, "apply", "--binary", "--whitespace=nowarn", "-"]),
        input=patch,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        detail = (
            (proc.stderr or proc.stdout or b"")
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
        raise RuntimeError(f"could not restore tracked repository baseline: {detail}")


def _restore_inplace(restore: dict) -> None:
    """Restore the live repo after in-place editing: revert EVERY file the agent
    changed back to its pre-forge content, return to the original branch/HEAD,
    and drop the temp branch.

    Restores the full changed-file set (not just ``source_file``): the agent may
    have edited a sibling tracked file (e.g. a config defaults module), and the
    loop's ``git add -u`` commits mean those edits live on the temp branch.
    ``base_commit`` holds the exact pre-forge tree (including any pre-existing
    dirty content snapshotted at prepare time), so checking files out of it
    restores precisely what was there before forge ran.

    Untracked files are handled by inventory, not by ``reset --hard``: when the
    caller recorded ``baseline_untracked`` at prepare time, untracked paths that
    did NOT exist then are deleted, because a campaign's leftover artifacts
    (notably ``forge_experiments/``) otherwise make the next run refuse to
    start. Untracked files present in the baseline are preserved.
    """
    if not restore:
        return
    repo = restore["repo"]
    # Abort any in-progress revert the loop may have left.
    _run_git(["-C", repo, "revert", "--abort"], timeout=30)
    orig_branch = restore.get("orig_branch") or ""
    orig_head = restore.get("orig_head") or ""
    base_commit = restore.get("base_commit") or orig_head
    # Restore every file that differs from the pre-forge baseline back to its
    # base_commit content (working tree + index), undoing all tracked edits.
    # Done while still on the temp branch so base_commit is reachable.
    if base_commit:
        diff = _run_git(["-C", repo, "diff", "--name-only", base_commit], timeout=60)
        for rel in (diff.stdout or "").splitlines():
            rel = rel.strip()
            if rel:
                _run_git(["-C", repo, "checkout", base_commit, "--", rel], timeout=30)
    # Move HEAD back to the original ref WITHOUT touching the working tree.
    if orig_branch and orig_branch != "HEAD":
        # Was on a named branch: point HEAD back at it via symbolic-ref.
        _run_git(["-C", repo, "symbolic-ref", "HEAD", f"refs/heads/{orig_branch}"], timeout=30)
    elif orig_head:
        # Was on detached HEAD: re-detach via update-ref --no-deref so the
        # working tree is not touched.
        _run_git(["-C", repo, "update-ref", "--no-deref", "HEAD", orig_head], timeout=30)
    # Reset the index to match orig_head (without touching working tree).
    if orig_head:
        _run_git(["-C", repo, "reset", orig_head, "--", "."], timeout=30)
    # Any baseline failure below must still drop the temp branch and release the
    # per-repo lock, otherwise the next in-place session cannot run.
    try:
        baseline_patch = restore.get("baseline_tracked_patch")
        if baseline_patch is not None:
            if not isinstance(baseline_patch, bytes):
                raise RuntimeError("invalid tracked repository baseline")
            baseline_in_base_commit = restore.get(
                "baseline_in_base_commit",
                False,
            )
            if not isinstance(baseline_in_base_commit, bool):
                raise RuntimeError("invalid tracked baseline commit marker")
            if baseline_patch and not baseline_in_base_commit:
                _apply_tracked_baseline(repo, baseline_patch)
        # Ensure the primary source_file is exactly the pre-forge bytes even if
        # the git restore above raced or partially applied.
        try:
            Path(restore["source_file"]).write_bytes(restore["backup"])
        except OSError as exc:
            # Best-effort rewrite; the git restore above already reverted it.
            # Surfaced rather than swallowed: if it fires alongside a failed
            # git restore, the file is the one the caller must inspect.
            log.warning(
                "in-place restore could not rewrite %s: %s",
                restore.get("source_file"),
                exc,
            )
        baseline_untracked = restore.get("baseline_untracked")
        if baseline_untracked is not None:
            if not isinstance(baseline_untracked, list) or any(
                not isinstance(path, str) or not path for path in baseline_untracked
            ):
                raise RuntimeError("invalid in-place untracked baseline")
            _remove_new_untracked(repo, set(baseline_untracked))
    finally:
        # Delete the temp branch (safe now that HEAD points elsewhere).
        if restore.get("branch"):
            _run_git(["-C", repo, "branch", "-D", restore["branch"]], timeout=30)
        # Release the per-repo in-place lock last, after full restore.
        _release_repo_lock(restore.get("lock_fd"))
        restore["lock_fd"] = None


def _remove_worktree(kernel_repo: str, source_file: str, wt: str, branch: str) -> None:
    """Tear down the worktree + temp branch; live repo untouched (W3)."""
    repo = kernel_repo or _git_toplevel(source_file)
    if not repo:
        return
    _run_git(["-C", repo, "worktree", "remove", "--force", wt], timeout=60)
    shutil.rmtree(wt, ignore_errors=True)
    _run_git(["-C", repo, "branch", "-D", branch], timeout=30)
    _run_git(["-C", repo, "worktree", "prune"], timeout=60)


# forge-loop requires --driver to exist before preflight_task repairs it in
# place, so the delegated driver is a file that fails loudly rather than one
# that could be mistaken for a conforming measurement driver.
_TASK_PREPARER_PLACEHOLDER = '''#!/usr/bin/env python3
"""Placeholder driver — forge-loop's task preparer authors the real one."""
import sys

sys.exit("forge task-preparer placeholder: no measurement driver authored yet")
'''


_GENERATED_DRIVER_GLOB = ".forge_driver_*.py"


def _exclude_generated_drivers(workspace: Path) -> None:
    """Keep generated drivers out of whatever the producer stages.

    Drivers are Hyperloom scratch files living inside the producer's workspace,
    so a broad ``git add`` would otherwise sweep one into the framework patch.
    Registering the pattern in the repository's own exclude file is idempotent
    and leaves the working tree untouched.

    ``--git-common-dir`` lands this in the live repository even from a linked
    worktree, so :func:`_restore_generated_driver_exclude` takes it back out when
    the run ends.
    """
    exclude = _git_exclude_file(workspace)
    if exclude is None:
        return
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        if _GENERATED_DRIVER_GLOB in existing.split():
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with open(exclude, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(_GENERATED_DRIVER_GLOB + "\n")
    except OSError as error:
        log.warning("forge: could not exclude generated drivers from git: %s", error)


def _git_exclude_file(workspace: Path) -> Path | None:
    """Resolve the exclude file git actually reads for ``workspace``."""
    probe = _run_git(
        ["-C", str(workspace), "rev-parse", "--git-common-dir"],
        timeout=30,
    )
    if probe.returncode != 0:
        return None
    common = Path(probe.stdout.strip())
    if not common.is_absolute():
        common = (Path(workspace) / common).resolve()
    return common / "info" / "exclude"


def _exclude_runtime_artifacts(workspace: Path) -> None:
    """Keep machine-generated artefacts out of the scratch repository's index.

    Only the throwaway scratch repository needs this. Real repositories carry
    their own ignore rules, and entries written there would outlive the run.
    """
    from kernelforge.loop.path_ownership import runtime_gitignore_globs  # noqa: PLC0415

    exclude = _git_exclude_file(workspace)
    if exclude is None:
        return
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        present = set(existing.split())
        missing = [g for g in runtime_gitignore_globs() if g not in present]
        if not missing:
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with open(exclude, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            for glob in missing:
                handle.write(glob + "\n")
    except OSError as error:
        log.warning("forge: could not write scratch exclude patterns: %s", error)


def _restore_generated_driver_exclude(workspace: Path) -> None:
    """Drop the driver pattern again so the live repository is left as found.

    Run beside the deletion of the drivers themselves, so the entry never outlives
    the files it hid. Only the exact pattern line is removed.
    """
    exclude = _git_exclude_file(workspace)
    if exclude is None or not exclude.is_file():
        return
    try:
        lines = exclude.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if line.strip() != _GENERATED_DRIVER_GLOB]
        if len(kept) == len(lines):
            return
        exclude.write_text("".join(kept), encoding="utf-8")
    except OSError as error:
        log.warning("forge: could not restore the git exclude file: %s", error)


def _write_generated_driver(workspace: str | Path, content: str) -> str:
    """Atomically allocate a unique hidden driver inside ``workspace``.

    The long-horizon forge-loop CLI resolves ``--driver`` relative to
    ``--workspace`` and rejects anything outside it, so generated drivers must
    live in the workspace rather than in the attempt output dir. The
    ``.forge_driver_`` prefix is the contract ``_finalize_forge_workspace``
    uses to clean these up after an in-place run.
    """
    workspace_path = Path(workspace)
    _exclude_generated_drivers(workspace_path)
    fd, raw_path = tempfile.mkstemp(
        prefix=".forge_driver_",
        suffix=".py",
        dir=str(workspace_path),
        text=True,
    )
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "w") as file:
            file.write(content)
        path.chmod(0o755)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return str(path)


def _shapes_from_candidate(candidate: dict) -> dict:
    """Build primary-first Forge selectors for every distinct workload case."""
    return forge_shapes_from_candidate(candidate)


def _invocation_spec_covers_cases(
    invocation_spec_file: str,
    grouped_cases: list[dict],
) -> bool:
    """Validate that the persisted task-preparer contract contains every case."""
    if not invocation_spec_file:
        return False
    try:
        payload = json.loads(Path(invocation_spec_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        schema_version = int(payload.get("schema_version") or 0)
    except (TypeError, ValueError):
        return False
    if schema_version < 2:
        return False

    expected_selectors = [
        dict(case.get("selector") or {}) for case in grouped_cases if isinstance(case.get("selector"), dict)
    ]
    task_group = (payload.get("workload") or {}).get("task_group") or {}
    spec_cases = task_group.get("cases") if isinstance(task_group, dict) else None
    actual_selectors = [
        dict(case.get("selector") or {})
        for case in (spec_cases or [])
        if isinstance(case, dict) and isinstance(case.get("selector"), dict)
    ]
    driver_contract = (payload.get("tests") or {}).get("driver_contract") or {}
    contract_selectors = driver_contract.get("case_selectors") if isinstance(driver_contract, dict) else None
    return (
        len(expected_selectors) == len(grouped_cases)
        and len(expected_selectors) > 1
        and actual_selectors == expected_selectors
        and contract_selectors == expected_selectors
        and driver_contract.get("requires_all_cases") is True
    )


def _write_report(
    output_dir: Path,
    baseline_ms: float | None,
    best_ms: float | None,
    improved: bool,
    *,
    mean_case_speedup: float | None = None,
    search_start_ms: float | None = None,
    improved_during_search: bool = False,
    integration_validation: str = "",
) -> Path:
    """Write optimization_report.md with the locked anchors (doc Section 6.4).

    Only claims a KEEP-worthy result when Forge reports a validated
    ``mean_case_speedup > 1``. Raw aggregate timings are diagnostic and may
    regress because they are not the optimization objective.

    ``integration_validation`` adds a second marker, so ``[correctness]`` keeps
    meaning the micro gate while the report still states integration is unproven.
    """
    lines = ["# Forge optimization report", ""]
    if improved and mean_case_speedup and mean_case_speedup > 1.0:
        lines.append(f"[micro_speedup] {mean_case_speedup:.4f}x")
        lines.append(f"mean_case_speedup={mean_case_speedup:.6f}")
        if search_start_ms:
            lines.append(
                f"search_start_ms={search_start_ms:.4f} improved_during_search={str(improved_during_search).lower()}"
            )
        if baseline_ms and best_ms and best_ms > 0:
            lines.append(
                f"# diagnostic raw means (not monotonic): baseline_ms={baseline_ms:.4f} selected_ms={best_ms:.4f}"
            )
        lines.append("[correctness] pass")
    else:
        lines.append("micro_speedup: N/A (no validated improvement kept)")
        lines.append("[correctness] fail")
        # When both baseline and best were measured but not kept, record the
        # observed timing informationally. Deliberately avoids the word
        # "speedup" and the "Nx" form so the report scanners never treat it as a
        # KEEP-worthy figure.
        if baseline_ms and best_ms and best_ms > 0:
            lines.append(f"# observed timing (not kept): baseline_ms={baseline_ms:.4f} selected_ms={best_ms:.4f}")
    if integration_validation:
        lines.append(f"[integration_validation] {integration_validation}")
    report = output_dir / "optimization_report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def _export_best_artifacts(
    workspace: str,
    base_commit: str,
    worktree_kernel_file: str,
    source_file: str,
    output_dir: Path,
    best_commit: str = "",
) -> tuple[str, list[str]]:
    """Export the best-kept state — ALL files the agent changed, not just the kernel.

    The loop now commits every tracked edit (``runner._git_commit`` uses
    ``git add -u``), so the agent's winning change may live in a sibling tracked
    file (e.g. a ``*_config.py`` defaults module) rather than ``source_file``.
    Exporting only ``source_file`` would yield a byte-identical artifact that
    carries none of the optimization (the in-place bench measured it, but it
    would not transfer on integration), and the sibling file would be left dirty.

    This:
      - copies the primary kernel to ``optimized_versions/v1_forge.<ext>`` (the
        Hyperloom report scan's drop-in-replacement contract), and
      - copies EVERY file changed since ``base_commit`` under
        ``optimized_versions/files/<repo-relative-path>``, and
      - writes a single ``optimized_versions/forge.patch`` (``git diff
        base_commit``) so a multi-file change can be applied at integration time.

    Returns (primary_artifact_path, changed_relpaths).
    """
    dst_dir = output_dir / "optimized_versions"
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _blob_at_commit(commit: str, relative_path: str) -> bytes | None:
        proc = subprocess.run(
            _git_argv(["-C", workspace, "show", f"{commit}:{relative_path}"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout if proc.returncode == 0 else None

    # Primary kernel artifact (drop-in replacement contract).
    ext = Path(source_file).suffix or ".py"
    primary = dst_dir / f"v1_forge{ext}"
    if best_commit:
        try:
            primary_rel = str(Path(worktree_kernel_file).resolve().relative_to(Path(workspace).resolve()))
        except ValueError:
            primary_rel = ""
        primary_bytes = _blob_at_commit(best_commit, primary_rel) if primary_rel else None
        if primary_bytes is None:
            raise RuntimeError(
                f"validated best commit does not contain primary source: {primary_rel or worktree_kernel_file}"
            )
        primary.write_bytes(primary_bytes)
    else:
        try:
            shutil.copy2(worktree_kernel_file, primary)
        except OSError as exc:
            log.warning(
                "forge export: could not copy primary artifact %s to %s: %s",
                worktree_kernel_file,
                primary,
                exc,
            )

    # A recovered run exports only the validated commit. A normally completed
    # run without checkpoint evidence retains the legacy working-tree export.
    changed: list[str] = []
    diff_cmd = ["-C", workspace, "diff", "--name-only", base_commit]
    if best_commit:
        diff_cmd.append(best_commit)
    diff = _run_git(diff_cmd, timeout=60)
    if best_commit and diff.returncode != 0:
        raise RuntimeError(f"could not list files changed by validated best {best_commit}")
    for rel in (diff.stdout or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        changed.append(rel)
        dstp = dst_dir / "files" / rel
        dstp.parent.mkdir(parents=True, exist_ok=True)
        if best_commit:
            blob = _blob_at_commit(best_commit, rel)
            if blob is not None:
                dstp.write_bytes(blob)
        else:
            srcp = Path(workspace) / rel
            if not srcp.is_file():
                continue
            try:
                shutil.copy2(srcp, dstp)
            except OSError as exc:
                log.warning(
                    "forge export: could not copy changed artifact %s to %s: %s",
                    srcp,
                    dstp,
                    exc,
                )

    # Full multi-file patch (excludes pre-existing dirty). --binary keeps the
    # patch appliable when a change touches a non-text artifact.
    patch_cmd = ["-C", workspace, "diff", "--binary", base_commit]
    if best_commit:
        patch_cmd.append(best_commit)
    patch = _run_git(patch_cmd, timeout=60)
    if best_commit and patch.returncode != 0:
        raise RuntimeError(f"could not export validated best patch {best_commit}")
    patch_text = patch.stdout or ""
    if best_commit and (not changed or not patch_text.strip()):
        raise RuntimeError(f"validated best commit {best_commit} has no exportable source diff")
    (dst_dir / "forge.patch").write_text(patch_text)

    if best_commit and not primary.is_file():
        raise RuntimeError(f"validated best primary artifact was not written: {primary}")

    return str(primary), changed


def _normalized(returncode: int, stdout: str, stderr: str, elapsed_s: float, skipped: bool = False) -> dict:
    """Shape the kernel-backend result dict (``returncode`` / ``skipped`` /
    ``stdout_tail`` / ``stderr_tail`` / ``stdout`` / ``gpu_ids`` / ``elapsed_s``
    / ``cmd``).

    ``skipped=True`` marks a forge self-skip: forge bailed before any real
    optimization attempt (unsupported source type, repo not a clean git
    checkout, etc.). It is the structured signal downstream uses to classify the
    kernel outcome as ``skip`` rather than a kernel failure; forge returns
    ``returncode=2`` for every such path, but consumers should read this flag
    rather than the return code.
    """
    return {
        "returncode": returncode,
        "skipped": bool(skipped),
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "stdout": stdout or "",
        "gpu_ids": os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "",
        "elapsed_s": round(elapsed_s, 2),
        "cmd": ["forge_submit.submit"],
    }


def _ensure_flydsl_aiter_compat(protocol_path: str = "") -> bool:
    """Self-heal aiter's flydsl dependency so HIP/CK ops aren't disabled.

    flydsl >=0.2 renamed ``fly_values`` to ``extract_to_ir_values``, but aiter's
    flydsl kernels still ``from flydsl.compiler.protocol import fly_values``. The
    failed import makes aiter disable ALL CK/HIP ops -> any aiter forge loop is
    dead on arrival. The sglang sandbox image ships the incompatible flydsl, and
    the container FS is ephemeral, so idempotently append a back-compat alias
    before running an aiter loop. Returns True when the alias is present.

    Args:
        protocol_path: Override for flydsl.compiler.protocol's file (tests);
            resolved via importlib when empty.
    """
    try:
        path = protocol_path
        if not path:
            import importlib.util

            spec = importlib.util.find_spec("flydsl.compiler.protocol")
            path = spec.origin if (spec and spec.origin) else ""
        if not path or not os.path.isfile(path):
            return False
        text = ""
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            return False
        if "fly_values" in text:
            return True  # original export or our shim already present
        if "def extract_to_ir_values" not in text:
            return False  # unexpected flydsl layout
        with open(path, "a") as f:
            f.write(
                "\n\n# Forge compat shim: aiter imports fly_values, renamed to\n"
                "# extract_to_ir_values in flydsl>=0.2 (same List[ir.Value] result).\n"
                "fly_values = extract_to_ir_values\n"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def _openai_only_provider() -> bool:
    """Return true when the OpenAI side is the only configured provider.

    The forge kernel backend reaches an OpenAI-protocol gateway only through
    KernelForge's codex provider, so this predicate is what selects it over the
    claude provider that ``Config.agent_backend='auto'`` would otherwise resolve
    to. The shape test lives in :mod:`hyperloom.common.llm_config` so that the
    kernel backend, backend selection and the TraceLens runner cannot disagree.
    """
    from hyperloom.common import llm_config  # local import: keep module import-light

    return llm_config.is_openai_only()


def _apply_kernel_backend_env(env: dict) -> None:
    """Apply kernel backend (claude CLI / codex SDK) stability defaults to ``env``.

    Mutates the given child-process env dict ONLY -- never the parent
    ``os.environ`` -- so the rewrite (notably the ANTHROPIC_BASE_URL streaming
    proxy) cannot leak outside this forge attempt. The forge-loop subprocess
    inherits this env; inside it the kernel backend drives either the claude CLI
    streaming transport or the codex SDK, per the configured provider side.
    ``setdefault`` keeps operator overrides authoritative.
    """
    claude_kernel_backend = not _openai_only_provider()
    # bypassPermissions refuses to start under root unless IS_SANDBOX=1.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("IS_SANDBOX", "1")
    if claude_kernel_backend:
        # claude CLI discovery: the child may inherit a stripped PATH, so resolve
        # claude's absolute path here, export FORGE_CLAUDE_BIN, and prepend its dir
        # to the child PATH.
        claude_bin = env.get("FORGE_CLAUDE_BIN", "").strip() or shutil.which("claude")
        if not claude_bin:
            for cand in ("/usr/local/bin/claude", "/usr/bin/claude", str(Path.home() / ".local/bin/claude")):
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    claude_bin = cand
                    break
        if claude_bin and os.path.isfile(claude_bin):
            env.setdefault("FORGE_CLAUDE_BIN", claude_bin)
            bindir = os.path.dirname(claude_bin)
            cur_path = env.get("PATH", "")
            if bindir and bindir not in cur_path.split(os.pathsep):
                env["PATH"] = bindir + os.pathsep + cur_path if cur_path else bindir
        # Public defaults keep TLS verification enabled. Internal deployments with
        # self-signed proxies can opt out by exporting their own TLS override envs.
        base_url = str(env.get("ANTHROPIC_BASE_URL") or "").strip()
        if base_url.endswith("/llm-gateway"):
            env["ANTHROPIC_BASE_URL"] = base_url[: -len("/llm-gateway")] + "/api/v1/llm-proxy"
    # KernelBackend-hung mitigation: bound the claude CLI's own request timeout and cut
    # non-essential traffic / autoupdate that can block in headless containers.
    from _llm_stability_env import apply_llm_stability_env

    apply_llm_stability_env(env)
    # The forge loop spends against the gateway for the whole of its run, and
    # every agent it drives inherits this env, so without a tag here that spend
    # arrives naming no component at all. Injecting from this side is what makes
    # the phase and the action travel: both live in this process only, and the
    # loop can refine the component it is given without having to restate them.
    from hyperloom.common.llm_attribution import inject_env

    inject_env(env, component="forge", operation="forge_loop")
    # Shared KnowledgePlane contract. KernelForge remains responsible for its
    # own local knowledge implementation and remote kernel-experience behavior.
    from hyperloom.orchestrator.knowledge.kernel_experience_bridge import (
        KernelExperienceBridge,
    )

    # The process-level configuration was validated at startup/first use and is
    # cached. A malformed child mapping therefore cannot fail every submission.
    KernelExperienceBridge(_knowledge_config_for_forge()).configure_child_env(env)

    # Auth fallback: seed ANTHROPIC_API_KEY from the claude CLI's config.json
    # primaryApiKey when it is not already exported. Skipped on the OpenAI-only
    # side, where the codex provider authenticates from OPENAI_API_KEY, and under
    # a subscription token, which any API key would silently override.
    if (
        claude_kernel_backend
        and not env.get("ANTHROPIC_API_KEY", "").strip()
        and not env.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    ):
        try:
            import json as _json

            _cfg = _json.loads((Path.home() / ".claude" / "config.json").read_text())
            _key = str(_cfg.get("primaryApiKey") or "").strip()
            if _key:
                env["ANTHROPIC_API_KEY"] = _key
        except Exception:  # noqa: S110
            pass


def _read_forge_checkpoint(experiments_dir: Path) -> dict | None:
    """Read the caller-owned experiment checkpoint written after each KEEP."""
    path = experiments_dir / f"{_FORGE_EXPERIMENT_ID}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoint") if isinstance(payload, dict) else None
    return checkpoint if isinstance(checkpoint, dict) else None


def _proc_identity(pid: int) -> tuple[int, int] | None:
    """Return ``(parent_pid, start_time_ticks)`` from Linux procfs."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
        closing_paren = stat_text.rfind(")")
        fields_after_name = stat_text[closing_paren + 2 :].split()
        return int(fields_after_name[1]), int(fields_after_name[19])
    except (OSError, ValueError, IndexError):
        return None


def _descendant_processes(root_pid: int) -> list[tuple[int, int]]:
    """Return ``(pid, start_time)`` descendants, deepest first."""
    children: dict[int, list[tuple[int, int]]] = {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        identity = _proc_identity(pid)
        if identity is None:
            continue
        parent_pid, start_time = identity
        children.setdefault(parent_pid, []).append((pid, start_time))

    descendants: list[tuple[int, int]] = []

    def _walk(parent_pid: int) -> None:
        for child_pid, start_time in children.get(parent_pid, []):
            _walk(child_pid)
            descendants.append((child_pid, start_time))

    _walk(root_pid)
    return descendants


def _signal_processes(processes: list[tuple[int, int]], sig: int) -> None:
    """Signal captured processes only while their procfs identity still matches."""
    for pid, expected_start_time in processes:
        identity = _proc_identity(pid)
        if identity is None or identity[1] != expected_start_time:
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            continue


def _process_group_members(pgid: int) -> list[tuple[int, int, str]]:
    """Return live ``(pid, start_time, state)`` members of one process group."""
    members: list[tuple[int, int, str]] = []
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return members
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text()
            closing_paren = stat_text.rfind(")")
            fields = stat_text[closing_paren + 2 :].split()
            if int(fields[2]) != pgid:
                continue
            members.append((int(entry.name), int(fields[19]), fields[0]))
        except (OSError, ValueError, IndexError):
            continue
    return members


def _signal_process_group(
    pgid: int,
    sig: int,
    *,
    phase: str,
) -> bool | None:
    """Signal a Forge-owned process group and warn on non-race failures."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return None
    except (PermissionError, OSError) as exc:
        log.warning(
            "forge process-group %s failed: pgid=%d signal=%d error=%s",
            phase,
            pgid,
            sig,
            exc,
        )
        return False


def _terminate_forge_process(
    proc: subprocess.Popen,
    *,
    grace_sec: int = _FORGE_SHUTDOWN_GRACE_SEC,
) -> tuple[str, str]:
    """Terminate the forge-loop process group, escalating after a grace period."""
    pgid = proc.pid
    descendants = _descendant_processes(proc.pid)
    if (
        _signal_process_group(
            pgid,
            signal.SIGTERM,
            phase="SIGTERM",
        )
        is False
    ):
        _signal_processes(descendants, signal.SIGTERM)
        try:
            proc.terminate()
        except OSError as exc:
            log.warning(
                "forge direct-process terminate fallback failed: pid=%d error=%s",
                proc.pid,
                exc,
            )
    try:
        stdout, stderr = proc.communicate(timeout=grace_sec)
        _signal_processes(descendants, signal.SIGKILL)
        _signal_process_group(
            pgid,
            signal.SIGKILL,
            phase="post-reap SIGKILL",
        )
        return stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        descendants = list(
            dict.fromkeys(
                [
                    *descendants,
                    *_descendant_processes(proc.pid),
                ]
            )
        )
        _signal_processes(descendants, signal.SIGKILL)
        if (
            _signal_process_group(
                pgid,
                signal.SIGKILL,
                phase="timeout SIGKILL",
            )
            is False
        ):
            try:
                proc.kill()
            except OSError as exc:
                log.warning(
                    "forge direct-process kill fallback failed: pid=%d error=%s",
                    proc.pid,
                    exc,
                )
        try:
            stdout, stderr = proc.communicate(timeout=5)
            _signal_process_group(
                pgid,
                signal.SIGKILL,
                phase="final SIGKILL",
            )
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            _signal_process_group(
                pgid,
                signal.SIGKILL,
                phase="reap-timeout SIGKILL",
            )
            residual = _process_group_members(pgid)
            _signal_processes(
                [(pid, start_time) for pid, start_time, _state in residual],
                signal.SIGKILL,
            )
            log.warning(
                "forge process group was not reaped after SIGKILL: pgid=%d residual=%s",
                pgid,
                [{"pid": pid, "state": state} for pid, _start_time, state in residual],
            )
            return "", ""


def _read_forge_best_result(workspace: str) -> dict | None:
    """Read the published best manifest forge atomically rewrites on every KEEP.

    Anchored to the campaign root under the workspace (not --experiments-dir):
    resume artifacts always live there, so this file is present and current after
    a clean finish, a soft budget exhaustion, or a hard kill mid-run.
    """
    path = Path(workspace) / "forge_experiments" / "best_result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _path_within(root: Path, relative: str) -> Path | None:
    """Resolve a manifest-relative path without allowing root escape."""
    if not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _canonical_forge_artifacts(
    workspace: str,
    published: dict | None,
) -> dict[str, object]:
    """Normalize KernelForge's immutable best bundle for downstream deploy.

    KernelForge schema-v1 paths are relative to the campaign root
    (``<workspace>/forge_experiments``), not to ``best/manifest.json``.
    Preserve changed paths as repo-relative POSIX paths and expose absolute
    artifact locations only after containment validation.
    """
    if not isinstance(published, dict):
        log.warning(
            "canonical Forge bundle unavailable: published manifest payload "
            "is missing or invalid; compatibility artifact fallback may be used"
        )
        return {}
    campaign_root = (Path(workspace) / "forge_experiments").resolve()
    manifest_path = campaign_root / "best" / "manifest.json"
    artifact_relative = str(published.get("artifact_dir") or "")
    artifact_dir = _path_within(
        campaign_root,
        artifact_relative,
    )
    patch_relative = str(published.get("patch_path") or "")
    patch_path = _path_within(
        campaign_root,
        patch_relative,
    )
    changed_files: list[str] = []
    for raw in published.get("changed_files") or []:
        rel = Path(str(raw))
        if rel.is_absolute() or ".." in rel.parts:
            log.warning(
                "canonical Forge bundle unavailable: changed file path escapes "
                "the campaign root (%s); compatibility artifact fallback may "
                "be used",
                raw,
            )
            return {}
        changed_files.append(rel.as_posix())
    if not manifest_path.is_file():
        log.warning(
            "canonical Forge bundle unavailable: best manifest does not exist "
            "at %s; compatibility artifact fallback may be used",
            manifest_path,
        )
        return {}
    if artifact_dir is None:
        log.warning(
            "canonical Forge bundle unavailable: artifact_dir %r is not a "
            "safe campaign-relative path under %s; compatibility artifact "
            "fallback may be used",
            artifact_relative,
            campaign_root,
        )
        return {}
    if patch_path is None or not patch_path.is_file():
        log.warning(
            "canonical Forge bundle unavailable: patch_path %r did not resolve "
            "to a file under %s; compatibility artifact fallback may be used",
            patch_relative,
            campaign_root,
        )
        return {}
    files_root = artifact_dir / "files"
    if not files_root.is_dir():
        log.warning(
            "canonical Forge bundle unavailable: expected files directory does "
            "not exist at %s (artifact_dir=%s); compatibility artifact fallback "
            "may be used",
            files_root,
            artifact_dir,
        )
        return {}
    try:
        resolved_files_root = files_root.resolve(strict=True)
    except OSError as error:
        log.warning(
            "canonical Forge bundle unavailable: files directory could not be "
            "resolved at %s (%s); compatibility artifact fallback may be used",
            files_root,
            error,
        )
        return {}
    if not _path_is_within(resolved_files_root, campaign_root):
        log.warning(
            "canonical Forge bundle unavailable: files directory resolves "
            "outside the campaign root (%s -> %s, root=%s); compatibility "
            "artifact fallback may be used",
            files_root,
            resolved_files_root,
            campaign_root,
        )
        return {}
    if not changed_files:
        log.warning(
            "canonical Forge bundle unavailable: manifest changed_files is "
            "empty; compatibility artifact fallback may be used"
        )
        return {}
    return {
        "best_manifest": str(manifest_path),
        "canonical_patch_path": str(patch_path),
        "canonical_files_root": str(resolved_files_root),
        "changed_files": changed_files,
        "forge_workspace": str(Path(workspace).resolve()),
    }


def _observed_mean_case_result_fields(
    payload: dict,
) -> tuple[float | None, float | None, bool, bool]:
    """Parse measured Forge scores without conflating regressions with missing data."""
    try:
        mean_case_speedup = float(payload.get("mean_case_speedup"))
    except (TypeError, ValueError):
        mean_case_speedup = None
    if mean_case_speedup is not None and (not math.isfinite(mean_case_speedup) or mean_case_speedup <= 0.0):
        mean_case_speedup = None
    try:
        search_start = float(payload.get("search_start_mean_case_speedup"))
    except (TypeError, ValueError):
        search_start = None
    if search_start is not None and (not math.isfinite(search_start) or search_start <= 0.0):
        search_start = None
    total_improved = bool(mean_case_speedup is not None and mean_case_speedup > 1.0)
    incremental = bool(
        payload.get(
            "incremental_improved",
            payload.get(
                "improved_during_search",
                mean_case_speedup is not None and search_start is not None and mean_case_speedup > search_start,
            ),
        )
    )
    return mean_case_speedup, search_start, total_improved, incremental


def _mean_case_result_fields(
    payload: dict,
) -> dict | None:
    """Normalize an improving, fully anchored Forge recovery result."""
    (
        mean_case_speedup,
        search_start,
        total_improved,
        incremental,
    ) = _observed_mean_case_result_fields(payload)
    if mean_case_speedup is None or search_start is None or not total_improved:
        return None
    return {
        "mean_case_speedup": mean_case_speedup,
        "search_start_mean_case_speedup": search_start,
        "total_improved": total_improved,
        "incremental_improved": incremental,
        "improved": True,
        "improved_during_search": incremental,
    }


def _validated_commit_lineage_and_timing(
    payload: dict,
    *,
    workspace: str,
    base_commit: str,
) -> tuple[str, float, float] | None:
    """Confirm a manifest names a real descendant of this run's base.

    A manifest is written by another process and may be stale from an earlier
    run against a different base, so the commit it names is re-checked against
    the workspace history and its wall timings must be usable numbers.

    Args:
        payload: A manifest carrying ``commit_hash`` and both wall timings.
        workspace: The git workspace the commit must live in.
        base_commit: The commit this attempt started from.

    Returns:
        tuple[str, float, float] | None: ``(commit, baseline_ms, best_ms)``, or
            ``None`` when the lineage or the timings do not hold up.
    """
    best_commit = str(payload.get("commit_hash") or "").strip()
    if not best_commit or best_commit == base_commit:
        return None
    exists = _run_git(
        ["-C", workspace, "cat-file", "-e", f"{best_commit}^{{commit}}"],
        timeout=30,
    )
    if exists.returncode != 0:
        return None
    ancestor = _run_git(
        [
            "-C",
            workspace,
            "merge-base",
            "--is-ancestor",
            base_commit,
            best_commit,
        ],
        timeout=30,
    )
    if ancestor.returncode != 0:
        return None
    try:
        baseline_ms = float(payload.get("baseline_wall_ms"))
        best_ms = float(payload.get("best_wall_ms"))
    except (TypeError, ValueError):
        return None
    if baseline_ms <= 0 or best_ms <= 0:
        return None
    return best_commit, baseline_ms, best_ms


def _validated_forge_best_result(
    payload: dict | None,
    *,
    workspace: str,
    base_commit: str,
) -> dict | None:
    """Return normalized evidence only for a published, correctness-passed best.

    Forge publishes this file only after a KEEP whose validation passed and whose
    commit is already in the workspace history, so it is the authoritative record
    of what to keep. Re-verify the commit lineage and the speedup here anyway --
    the file is written by another process and may be stale from an earlier run
    against a different base.

    ``schema_version`` is deliberately not gated on. Every field read below is
    checked on its own -- the commit against the workspace history, the timings
    for being positive numbers, the score for actually improving -- so pinning a
    version bought nothing those checks do not, while a producer bump that
    changed none of them silently rejected every published best for six days.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("correctness_passed") is not True:
        return None
    lineage = _validated_commit_lineage_and_timing(
        payload,
        workspace=workspace,
        base_commit=base_commit,
    )
    if lineage is None:
        return None
    best_commit, baseline_ms, best_ms = lineage
    score_fields = _mean_case_result_fields(payload)
    if score_fields is None:
        return None
    return {
        "best_commit": best_commit,
        "baseline_ms": baseline_ms,
        "best_ms": best_ms,
        **score_fields,
        "iteration": payload.get("iteration"),
        "snr_db": payload.get("snr_db"),
        "source": "best_result.json",
    }


_REWRITE_ARTIFACT_KIND = "framework_applyback"
_REWRITE_ARTIFACT_SCHEMA_VERSION = 2
_REWRITE_VALIDATION_SCOPE = "reference"
_REWRITE_INTEGRATION_PENDING = "pending"
_REWRITE_SUPPORTED_FRAMEWORKS = frozenset({"aiter", "vllm", "sglang"})
_REWRITE_MANIFEST_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
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
_PATCH_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _rewrite_manifest_has_producer_shape(manifest: dict) -> bool:
    """Require every field the installed producer's schema-2 manifest owns."""
    for field, expected in _REWRITE_MANIFEST_REQUIRED_FIELDS.items():
        if field not in manifest:
            return False
        value = manifest[field]
        if isinstance(value, bool) and expected is not bool:
            return False
        if not isinstance(value, expected):
            return False
    return True


def _rewrite_contained_path(
    workspace_root: Path,
    value: object,
    *,
    allow_absolute: bool,
) -> Path | None:
    """Resolve a producer-reported path, rejecting anything outside the workspace."""
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        return _path_within(workspace_root, text)
    if not allow_absolute:
        return None
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    return resolved if _path_is_within(resolved, workspace_root) else None


def _patch_touched_paths(patch_path: Path) -> set[str] | None:
    """Return the repo-relative paths a git patch claims to touch.

    Only the post-image side of each header, matching the producer's
    ``git diff --name-only`` declaration. The two differ only on a rename or copy,
    where the header names both ends but the declaration names the destination.
    """
    try:
        text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    touched: set[str] = set()
    for _old, new in _PATCH_HEADER_RE.findall(text):
        path = new.strip().strip('"')
        if path and path != "/dev/null":
            touched.add(path)
    return touched


def _validated_rewrite_applyback_result(
    payload: dict | None,
    *,
    workspace: str,
    base_commit: str,
    problems: list[str] | None = None,
) -> dict | None:
    """Return normalized evidence only for a published framework apply-back.

    The producer reports two documents with disjoint key sets: an outer result
    naming the canonical artifacts, and the schema-2 manifest those artifacts
    are described by. Both are checked with their own keys, and the manifest is
    opened only through the path the outer result declares -- never by scanning
    the workspace for whichever manifest looks newest.

    Args:
        payload: The outer result read from the caller-chosen result file.
        workspace: The git workspace every reported path must stay inside.
        base_commit: The commit this attempt started from.
        problems: Collector for the clause that refused the artifact. Without it
            every refusal here reaches an operator as the same sentence.

    Returns:
        dict | None: Normalized apply-back evidence, or ``None`` when any part
            of the two-document contract does not hold.
    """

    def _reject(reason: str) -> dict | None:
        if problems is not None:
            problems.append(reason)
        log.warning("forge rewrite: apply-back refused: %s", reason)
        return None

    if not isinstance(payload, dict):
        return _reject("the producer wrote no result object")
    if payload.get("success") is not True or payload.get("applyback_ok") is not True:
        return _reject(
            f"the producer did not report success (success={payload.get('success')!r} "
            f"applyback_ok={payload.get('applyback_ok')!r})"
        )
    if payload.get("artifact_kind") != _REWRITE_ARTIFACT_KIND:
        return _reject(f"result artifact_kind={payload.get('artifact_kind')!r} is not {_REWRITE_ARTIFACT_KIND!r}")
    try:
        artifact_schema_version = int(payload.get("artifact_schema_version"))
    except (TypeError, ValueError):
        return _reject(f"result artifact_schema_version is not an integer: {payload.get('artifact_schema_version')!r}")
    if artifact_schema_version != _REWRITE_ARTIFACT_SCHEMA_VERSION:
        return _reject(
            f"result artifact_schema_version={artifact_schema_version} is not the "
            f"supported {_REWRITE_ARTIFACT_SCHEMA_VERSION}"
        )
    outer_commit = str(payload.get("best_commit") or "").strip()
    if not outer_commit:
        return _reject("the result names no best_commit")

    workspace_root = Path(workspace).resolve()
    manifest_path = _rewrite_contained_path(workspace_root, payload.get("canonical_manifest"), allow_absolute=True)
    patch_path = _rewrite_contained_path(workspace_root, payload.get("canonical_patch_path"), allow_absolute=True)
    files_root = _rewrite_contained_path(workspace_root, payload.get("canonical_files_root"), allow_absolute=True)
    if manifest_path is None or not manifest_path.is_file():
        return _reject(
            f"canonical_manifest is not a readable file inside the workspace: {payload.get('canonical_manifest')!r}"
        )
    if patch_path is None or not patch_path.is_file():
        return _reject(
            f"canonical_patch_path is not a readable file inside the workspace: {payload.get('canonical_patch_path')!r}"
        )
    if files_root is None or not files_root.is_dir():
        return _reject(
            f"canonical_files_root is not a directory inside the workspace: {payload.get('canonical_files_root')!r}"
        )

    # Reclaiming these paths is destructive and keys off this declaration alone,
    # so an absent or non-relative one fails the result rather than defaulting.
    declared_temporary = payload.get("temporary_paths")
    if not isinstance(declared_temporary, list):
        return _reject(f"temporary_paths is not a list: {declared_temporary!r}")
    temporary_paths: list[str] = []
    for raw in declared_temporary:
        resolved = _rewrite_contained_path(workspace_root, raw, allow_absolute=False)
        if resolved is None:
            return _reject(f"declared temporary path escapes the workspace or is absolute: {raw!r}")
        temporary_paths.append(str(resolved))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _reject(f"the manifest at {manifest_path} is unreadable: {error}")
    if not isinstance(manifest, dict):
        return _reject(f"the manifest at {manifest_path} is not a JSON object")
    if not _rewrite_manifest_has_producer_shape(manifest):
        return _reject("the manifest is missing producer-owned fields, so it was not written by the rewrite producer")
    if manifest.get("schema_version") != _REWRITE_ARTIFACT_SCHEMA_VERSION:
        return _reject(
            f"manifest schema_version={manifest.get('schema_version')!r} is not the "
            f"supported {_REWRITE_ARTIFACT_SCHEMA_VERSION}"
        )
    if manifest.get("artifact_kind") != _REWRITE_ARTIFACT_KIND:
        return _reject(f"manifest artifact_kind={manifest.get('artifact_kind')!r} is not {_REWRITE_ARTIFACT_KIND!r}")
    if manifest.get("validation_scope") != _REWRITE_VALIDATION_SCOPE:
        return _reject(
            f"manifest validation_scope={manifest.get('validation_scope')!r} is not {_REWRITE_VALIDATION_SCOPE!r}"
        )
    if str(manifest.get("framework") or "") not in _REWRITE_SUPPORTED_FRAMEWORKS:
        return _reject(
            f"manifest framework={manifest.get('framework')!r} is not one this "
            f"consumer can apply back {sorted(_REWRITE_SUPPORTED_FRAMEWORKS)}"
        )
    if manifest.get("reference_correctness_passed") is not True:
        return _reject("the manifest does not claim reference correctness passed")
    if manifest.get("integration_validation_required") is not True:
        return _reject(
            "the manifest does not require integration validation, which only a "
            "producer claiming to have proven the integration itself would do"
        )
    if str(manifest.get("integration_validation_status") or "") != _REWRITE_INTEGRATION_PENDING:
        return _reject(
            "manifest integration_validation_status="
            f"{manifest.get('integration_validation_status')!r} is not "
            f"{_REWRITE_INTEGRATION_PENDING!r}; only the consumer may record a verdict"
        )
    if str(manifest.get("base_commit") or "").strip() != base_commit:
        return _reject(
            f"manifest base_commit={manifest.get('base_commit')!r} is not the commit "
            f"this attempt started from ({base_commit})"
        )
    manifest_artifact_dir = _rewrite_contained_path(
        workspace_root / "forge_experiments",
        manifest.get("artifact_dir"),
        allow_absolute=False,
    )
    manifest_patch_path = _rewrite_contained_path(
        workspace_root / "forge_experiments",
        manifest.get("patch_path"),
        allow_absolute=False,
    )
    if manifest_artifact_dir is None or manifest_artifact_dir != files_root.parent:
        return _reject(
            f"manifest artifact_dir={manifest.get('artifact_dir')!r} does not resolve "
            f"to the parent of the declared canonical_files_root ({files_root.parent})"
        )
    if manifest_patch_path is None or manifest_patch_path != patch_path:
        return _reject(
            f"manifest patch_path={manifest.get('patch_path')!r} does not resolve to "
            f"the declared canonical_patch_path ({patch_path})"
        )

    lineage = _validated_commit_lineage_and_timing(
        manifest,
        workspace=workspace,
        base_commit=base_commit,
    )
    if lineage is None:
        return _reject("the manifest's commit lineage or its reference timings did not validate against the workspace")
    best_commit, baseline_ms, best_ms = lineage
    if best_commit != outer_commit:
        return _reject(
            f"the manifest's best commit ({best_commit[:12]}) is not the one the result names ({outer_commit[:12]})"
        )
    # Whether the rewrite is *faster* is not part of the producer contract: it
    # may publish a correct-but-not-faster port. That is a consumer policy call,
    # graded by the caller so a rejected win is not reported as a bad artifact.

    changed_files: list[str] = []
    for raw in manifest.get("changed_files") or []:
        relative = str(raw or "").strip()
        if not relative:
            return _reject("the manifest declares an empty changed_files entry")
        parts = Path(relative)
        if parts.is_absolute() or ".." in parts.parts:
            return _reject(f"the manifest declares a changed file outside the framework: {relative!r}")
        changed_files.append(parts.as_posix())
    if not changed_files:
        return _reject("the manifest declares no changed files")
    touched = _patch_touched_paths(patch_path)
    if touched != set(changed_files):
        return _reject(
            "the patch and the manifest disagree on which files change (patch: "
            f"{sorted(touched or [])}, manifest: {sorted(changed_files)})"
        )

    commit_ref = str(manifest.get("commit_ref") or "").strip()
    if not commit_ref:
        return _reject("the manifest names no commit_ref to pin the artifact to")
    pinned = _run_git(
        ["-C", workspace, "rev-parse", "--verify", f"{commit_ref}^{{commit}}"],
        timeout=30,
    )
    if pinned.returncode != 0 or pinned.stdout.strip() != best_commit:
        return _reject(
            f"commit_ref {commit_ref!r} does not resolve to the best commit ({best_commit[:12]}) in the workspace"
        )

    return {
        "best_commit": best_commit,
        "baseline_ms": baseline_ms,
        "best_ms": best_ms,
        "artifact_kind": _REWRITE_ARTIFACT_KIND,
        "artifact_schema_version": artifact_schema_version,
        "validation_scope": _REWRITE_VALIDATION_SCOPE,
        "reference_correctness_passed": True,
        "reference_snr_db": manifest.get("reference_snr_db"),
        "integration_validation_required": True,
        "integration_validation_status": _REWRITE_INTEGRATION_PENDING,
        "base_commit": base_commit,
        "commit_ref": commit_ref,
        "builder_symbol": str(manifest.get("builder_symbol") or ""),
        "framework": str(manifest.get("framework") or ""),
        "logical_op_name": str(manifest.get("logical_op_name") or ""),
        "source_entry": str(manifest.get("source_entry") or ""),
        "canonical_manifest": str(manifest_path),
        "canonical_patch_path": str(patch_path),
        "canonical_files_root": str(files_root),
        "changed_files": changed_files,
        "temporary_paths": temporary_paths,
        "applyback_required": payload.get("applyback_required"),
        "source": "framework_applyback",
    }


def _validated_forge_checkpoint(
    checkpoint: dict | None,
    *,
    workspace: str,
    base_commit: str,
    shapes: dict,
) -> dict | None:
    """Return normalized recovery evidence only for a validated improved commit."""
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get("schema_version") != 1:
        return None
    if checkpoint.get("experiment_id") != _FORGE_EXPERIMENT_ID:
        return None
    if checkpoint.get("state") != "best_committed":
        return None
    if checkpoint.get("validation_passed") is not True:
        return None
    best_commit = str(checkpoint.get("best_commit") or "").strip()
    checkpoint_base = str(checkpoint.get("base_commit") or "").strip()
    if checkpoint_base != base_commit:
        return None
    if not best_commit or best_commit == base_commit:
        return None
    exists = _run_git(
        ["-C", workspace, "cat-file", "-e", f"{best_commit}^{{commit}}"],
        timeout=30,
    )
    if exists.returncode != 0:
        return None
    ancestor = _run_git(
        [
            "-C",
            workspace,
            "merge-base",
            "--is-ancestor",
            base_commit,
            best_commit,
        ],
        timeout=30,
    )
    if ancestor.returncode != 0:
        return None
    score_fields = _mean_case_result_fields(checkpoint)
    if score_fields is None:
        return None
    try:
        baseline_ms = float(checkpoint.get("baseline_ms"))
        best_ms = float(checkpoint.get("best_ms"))
    except (TypeError, ValueError):
        return None
    if baseline_ms <= 0 or best_ms <= 0:
        return None
    expected_coverage = list(shapes.get("validation") or [])
    if not expected_coverage:
        for shape in (shapes.get("minimal"), shapes.get("primary")):
            if isinstance(shape, dict) and shape and shape not in expected_coverage:
                expected_coverage.append(shape)
    # forge-loop stopped reporting case coverage once drivers took over suite
    # evaluation, so silence here says nothing about what was measured -- an
    # older loop reports an empty list for the same reason. Only a coverage that
    # is reported and disagrees is evidence the checkpoint measured something
    # else; vetoing on absence discards every salvageable best from a timeout.
    actual_coverage = checkpoint.get("case_coverage")
    if actual_coverage and expected_coverage and actual_coverage != expected_coverage:
        # Discarding a best the producer already validated and committed is too
        # expensive an outcome to leave to a return value nobody can attribute.
        log.warning(
            "forge recovery: dropping checkpoint for %s -- case coverage mismatch: expected %r, checkpoint reported %r",
            best_commit[:12],
            expected_coverage,
            actual_coverage,
        )
        return None
    normalized = dict(checkpoint)
    normalized["best_commit"] = best_commit
    normalized["baseline_ms"] = baseline_ms
    normalized["best_ms"] = best_ms
    normalized.update(score_fields)
    return normalized


def _validated_warm_start_result(
    result: dict | None,
    *,
    workspace: str,
    base_commit: str,
) -> dict | None:
    """Validate a KB-applied best when the search produced no later KEEP."""
    if not isinstance(result, dict):
        return None
    kb_experience = result.get("kb_experience")
    read = kb_experience.get("read") if isinstance(kb_experience, dict) else None
    if not isinstance(read, dict) or read.get("applied") is not True:
        return None
    warm_best = result.get("warm_start")
    if not isinstance(warm_best, dict):
        warm_best = result.get("warm_start_best")
    if not isinstance(warm_best, dict):
        warm_best = {}
    if read.get("validated") is False or read.get("validation_passed") is False:
        return None
    if warm_best.get("validated") is False or warm_best.get("validation_passed") is False:
        return None

    best_commit = str(
        result.get("warm_start_best_commit")
        or warm_best.get("best_commit")
        or warm_best.get("commit_hash")
        or read.get("best_commit")
        or read.get("applied_commit")
        or read.get("commit_hash")
        or (result.get("best_commit") if result.get("best_iteration") in (None, 0, "0") else "")
        or ""
    ).strip()
    if not best_commit or best_commit == base_commit:
        return None
    exists = _run_git(
        ["-C", workspace, "cat-file", "-e", f"{best_commit}^{{commit}}"],
        timeout=30,
    )
    if exists.returncode != 0:
        return None
    ancestor = _run_git(
        [
            "-C",
            workspace,
            "merge-base",
            "--is-ancestor",
            base_commit,
            best_commit,
        ],
        timeout=30,
    )
    if ancestor.returncode != 0:
        return None
    score_fields = _mean_case_result_fields(result)
    if score_fields is None:
        return None
    try:
        pristine_ms = float(
            result.get("pristine_baseline_ms")
            or warm_best.get("pristine_baseline_ms")
            or warm_best.get("pristine_ms")
            or read.get("pristine_ms")
        )
        best_ms = float(
            result.get("best_ms")
            or warm_best.get("best_ms")
            or warm_best.get("validated_best_ms")
            or read.get("keep_baseline_ms")
        )
    except (TypeError, ValueError):
        return None
    if pristine_ms <= 0 or best_ms <= 0:
        return None
    return {
        "best_commit": best_commit,
        "baseline_ms": pristine_ms,
        "best_ms": best_ms,
        **score_fields,
        "source": "kb_warm_start",
        "kb_experience": kb_experience,
    }


def _run_loop_via_cli(
    *,
    worktree_kernel: str,
    driver: str,
    workspace: str,
    snr_threshold: float,
    max_hours: float,
    branch: str,
    gpu_target: str,
    gpu_type: str,
    kernel_backend: str,
    program_md_file: str,
    invocation_spec_file: str,
    experiments_dir: Path,
    forge_log: Path,
    timeout_s: int,
    deadline_unix: float = 0.0,
    operator_name: str = "",
    experience_id: str = "",
    framework: str = "",
    target_functions: list[str] | None = None,
    source_files: list[str] | None = None,
) -> ForgeLoopOutcome:
    """Run the Forge IterationLoop as an isolated subprocess (CLI mode).

    Shells out to ``kernelforge forge-loop`` (like the GEAK backend shells
    out to its CLI) so the LLM-driven loop runs in a hard-killable child
    process. A hung kernel backend can no longer freeze the orchestrator: the timeout
    terminates the whole process group, then returns any persisted best
    checkpoint for recovery.

    The child runs ``python -m kernelforge.cli forge-loop`` against the
    installed package, which ships inside this distribution.
    """
    import json as _json

    if deadline_unix <= 0:
        deadline_unix = time.time() + timeout_s
    result_json = experiments_dir.parent / "forge_cli_result.json"
    checkpoint_json = experiments_dir / f"{_FORGE_EXPERIMENT_ID}.json"
    for stale_path in (result_json, checkpoint_json):
        try:
            stale_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"could not clear stale Forge recovery artifact {stale_path}: {exc}") from exc
        if stale_path.exists():
            raise RuntimeError(f"stale Forge recovery artifact still exists: {stale_path}")
    env = dict(os.environ)
    env["GPU_TARGET"] = gpu_target
    _apply_gpu_type_env(env, gpu_type)
    # KernelBackend stability defaults scoped to this child env only.
    _apply_kernel_backend_env(env)
    # Identity for the commits the loop makes, so no repo .git/config is touched.
    env.setdefault("GIT_AUTHOR_NAME", "forge-bot")
    env.setdefault("GIT_AUTHOR_EMAIL", "forge-bot@local")
    env.setdefault("GIT_COMMITTER_NAME", "forge-bot")
    env.setdefault("GIT_COMMITTER_EMAIL", "forge-bot@local")
    # KernelForge owns content-addressed AITER cache invalidation. Do not set
    # AITER_REBUILD globally: cpp_itfs interprets it by deleting the whole build
    # tree on every driver-process import, causing repeated attention rebuilds.
    if "/aiter/" in (worktree_kernel or ""):
        env.pop("AITER_REBUILD", None)
        # Self-heal aiter's flydsl dep (fly_values rename) so HIP/CK ops aren't
        # disabled before the loop imports aiter.
        _ensure_flydsl_aiter_compat()
    cmd = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--kernel",
        worktree_kernel,
        "--driver",
        driver,
        "--workspace",
        workspace,
        "--snr-threshold",
        str(snr_threshold),
        "--max-hours",
        str(max_hours),
        "--git-branch",
        branch,
        "--gpu-target",
        gpu_target,
        "--kernel-backend",
        kernel_backend,
        "--experiments-dir",
        str(experiments_dir),
        "--experiment-id",
        _FORGE_EXPERIMENT_ID,
        "--experience-id",
        experience_id or experiments_dir.parent.name,
        "--deadline-unix",
        str(deadline_unix),
        "--result-json",
        str(result_json),
    ]
    # Named on the command line as well as in the environment: KernelForge
    # skips its KB and reports ``missing_gpu_type`` rather than stopping, so an
    # identity that arrived only by inheritance could be lost without the run
    # ever failing. Passed even when it resolves to nothing, because an omitted
    # option is how KernelForge is told to use its own default -- saying nothing
    # would file the run under a card it may never have run on.
    cmd += ["--gpu-type", _known_gpu_model(gpu_type)]
    # Provider selection. KernelForge defaults agent_backend to "auto", which
    # resolves to its claude provider; an OpenAI-only deployment has no Anthropic
    # credential and no Claude CLI login, so every attempt would REVERT on "Not
    # logged in". Pin codex instead, and disable the provider fallback (it
    # defaults to claude) so a missing Codex SDK fails loudly here rather than
    # degrading into an unauthenticated claude run.
    #
    # Model id uses the shared Forge ladder (FORGE_* → CLAUDE/CODEX_MODEL) so
    # rewrite honors the same overrides as fusion and collective. Omit --model
    # when unset so KernelForge keeps its own provider default.
    from hyperloom.common.llm_config import resolve_forge_llm_model

    if _openai_only_provider():
        cmd += ["--agent-backend", "codex", "--agent-fallback-provider", "none"]
        forge_model = resolve_forge_llm_model("codex")
    else:
        forge_model = resolve_forge_llm_model("claude")
    if forge_model:
        cmd += ["--model", forge_model]
    if program_md_file and Path(program_md_file).exists():
        cmd += ["--program-md-file", str(program_md_file)]
    if invocation_spec_file and Path(invocation_spec_file).is_file():
        cmd += ["--invocation-spec-file", str(Path(invocation_spec_file).resolve())]
    if operator_name:
        cmd += ["--operator-name", operator_name]
    if target_functions:
        cmd += ["--target-functions", ",".join(target_functions)]
    if source_files:
        cmd += ["--source-files", ",".join(source_files)]
    # Pin the KB framework identity so producer/consumer resolve the same kernel
    # page across differing workspace layouts. Omitted when unknown, in which
    # case forge-loop infers it from the kernel path (soft, never fatal).
    if framework:
        cmd += ["--framework", framework]

    loop_exc = None
    out = ""
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=workspace,
            start_new_session=True,
        )
        try:
            remaining = max(1.0, deadline_unix - time.time())
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_forge_process(proc)
        out = (stdout or "") + "\n" + (stderr or "")
        if timed_out:
            loop_exc = RuntimeError(f"forge-loop exceeded absolute deadline after {timeout_s}s")
        if proc.returncode != 0:
            if loop_exc is None:
                loop_exc = RuntimeError(f"forge-loop exited rc={proc.returncode}: {_forge_failure_tail(out)}")
    except Exception as exc:  # noqa: BLE001
        loop_exc = exc

    try:
        with open(forge_log, "a") as f:
            f.write("\n=== forge-loop (cli) stdout ===\n")
            f.write(out)
            if loop_exc:
                f.write(f"\n=== forge-loop exception ===\n{loop_exc}\n")
    except OSError:  # noqa: S110
        pass

    # Parse the result: prefer the JSON sidecar, else the sentinel line.
    baseline_ms = best_ms = None
    pristine_baseline_ms = search_start_ms = None
    mean_case_speedup = search_start_mean_case_speedup = None
    improved = False
    improved_during_search = False
    total_improved = False
    incremental_improved = False
    parsed = None
    try:
        if result_json.exists():
            parsed = _json.loads(result_json.read_text())
    except Exception:
        parsed = None
    if parsed is None and "__FORGE_RESULT__" in out:
        try:
            seg = out.split("__FORGE_RESULT__")[1]
            parsed = _json.loads(seg)
        except Exception:
            parsed = None
    if parsed:
        baseline_ms = parsed.get("baseline_ms")
        best_ms = parsed.get("best_ms")
        pristine_baseline_ms = parsed.get("pristine_baseline_ms", baseline_ms)
        search_start_ms = parsed.get("search_start_ms", baseline_ms)
        (
            mean_case_speedup,
            search_start_mean_case_speedup,
            total_improved,
            incremental_improved,
        ) = _observed_mean_case_result_fields(parsed)
        improved = total_improved
        improved_during_search = incremental_improved
        if parsed.get("deadline_expired"):
            timed_out = True
            if loop_exc is None:
                loop_exc = RuntimeError("forge-loop reached its graceful absolute deadline")
    checkpoint = _read_forge_checkpoint(experiments_dir)
    return ForgeLoopOutcome(
        baseline_ms=baseline_ms,
        best_ms=best_ms,
        improved=improved,
        output=out,
        error=loop_exc,
        timed_out=timed_out,
        checkpoint=checkpoint,
        pristine_baseline_ms=pristine_baseline_ms,
        search_start_ms=search_start_ms,
        improved_during_search=improved_during_search,
        structured_result=parsed if isinstance(parsed, dict) else None,
        mean_case_speedup=mean_case_speedup,
        search_start_mean_case_speedup=search_start_mean_case_speedup,
        total_improved=total_improved,
        incremental_improved=incremental_improved,
    )


def _write_changed_files_index(output_dir: Path, changed_files: list[str]) -> None:
    """Record the exported bundle's file list beside the artifacts."""
    if not changed_files:
        return
    try:
        (output_dir / "optimized_versions" / "changed_files.txt").write_text("\n".join(changed_files) + "\n")
    except OSError:
        log.warning("forge export: could not write the changed-files index")


class RewriteRunOutcome(NamedTuple):
    """Result and failure evidence from one forge-rewrite-by-flydsl subprocess."""

    result: dict | None
    output: str
    error: Exception | None
    timed_out: bool


def _run_rewrite_via_cli(
    *,
    source_kernel: str,
    driver: str,
    logical_op_name: str,
    source_entry: str,
    source_language: str,
    workspace: str,
    experiments_dir: Path,
    result_json: Path,
    target_functions: list[str] | None,
    shapes: list[dict],
    invocation_spec_file: str,
    driver_preparation: bool,
    snr_threshold: float,
    gpu_target: str,
    gpu_type: str,
    max_hours: float,
    branch: str,
    framework: str,
    forge_log: Path,
    timeout_s: int,
    deadline_unix: float = 0.0,
) -> RewriteRunOutcome:
    """Run the source-to-FlyDSL rewrite as an isolated subprocess (CLI mode).

    Shares the forge-loop launcher's containment guarantees -- child env,
    isolated process group, absolute deadline and escalating termination -- but
    builds the producer's own argv rather than stripping options off the
    generic one, and reads only the caller-chosen result file.

    ``shapes`` is a list of per-case dimension mappings, not the selector dict
    Hyperloom carries internally: the rewrite producer coerces this argument
    with ``list()``, so a mapping would degrade into a list of its keys.

    ``invocation_spec_file`` is the evidence the producer's driver-preparation
    stage reads when the handed-over driver does not conform. A synthesized
    driver can also be found non-conforming and repaired from the same evidence,
    so it is offered on both routes -- but only to a producer that advertised
    ``driver_preparation``, since an older one rejects the options outright.

    ``source_language`` is stated rather than left for the producer to infer: this
    consumer resolved it from a trace, and a traced Triton kernel lives in a ``.py``
    that names no language.
    """
    if deadline_unix <= 0:
        deadline_unix = time.time() + timeout_s
    # Aim the producer one reserve short of the hard kill so it publishes the
    # apply-back inside its own budget instead of racing the kill. The floor keeps
    # a rounding error from passing a ``--max-hours`` the producer would reject.
    producer_deadline_unix = max(
        time.time() + 1.0,
        deadline_unix - _flydsl_rewrite.APPLYBACK_RESERVE_SEC,
    )
    max_hours = max(
        _flydsl_rewrite.PRODUCER_MIN_BUDGET_SEC / 3600.0,
        min(max_hours, (producer_deadline_unix - time.time()) / 3600.0),
    )
    try:
        result_json.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"could not clear stale rewrite result {result_json}: {exc}") from exc
    if result_json.exists():
        raise RuntimeError(f"stale rewrite result still exists: {result_json}")

    env = dict(os.environ)
    env["GPU_TARGET"] = gpu_target
    _apply_gpu_type_env(env, gpu_type)
    _apply_kernel_backend_env(env)
    # Same provider pin the generic loop applies through argv, which this command
    # has no options for: it takes no --agent-backend, so its Config reads these.
    # Without them an OpenAI-only deployment resolves "auto" to the claude
    # provider and every session fails "Not logged in", after the whole budget.
    #
    # forge-rewrite-by-flydsl accepts --model (overrides KERNEL_AGENTS_MODEL /
    # FORGE_AGENT_MODEL). KernelForge Config does not read FORGE_CLAUDE_MODEL /
    # FORGE_CODEX_MODEL, so Hyperloom must resolve and pass the id explicitly —
    # the same ladder forge-loop / fusion / collective already use.
    from hyperloom.common.llm_config import resolve_forge_llm_model

    if _openai_only_provider():
        env["FORGE_AGENT_BACKEND"] = "codex"
        env["FORGE_AGENT_FALLBACK_PROVIDER"] = "none"
        forge_model = resolve_forge_llm_model("codex")
    else:
        forge_model = resolve_forge_llm_model("claude")
    if "/aiter/" in (source_kernel or ""):
        env.pop("AITER_REBUILD", None)
        _ensure_flydsl_aiter_compat()

    cmd = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        _flydsl_rewrite.REWRITE_COMMAND,
        "--source-kernel",
        source_kernel,
        "--driver",
        driver,
        "--logical-op-name",
        logical_op_name,
        "--workspace",
        workspace,
        "--experiments-dir",
        str(experiments_dir),
        "--shapes-json",
        json.dumps(shapes),
        "--snr-threshold",
        str(snr_threshold),
        "--gpu-target",
        gpu_target,
        "--max-hours",
        str(max_hours),
        "--deadline-unix",
        str(producer_deadline_unix),
        "--git-branch",
        branch,
        "--result-json",
        str(result_json),
    ]
    # Named on the command line for the same reason the loop names it: the
    # rewrite producer files its port under an identity the model is part of,
    # and an unresolved model has to be said rather than left out.
    cmd += ["--gpu-type", _known_gpu_model(gpu_type)]
    if forge_model:
        cmd += ["--model", forge_model]
    if source_entry:
        cmd += ["--source-entry", source_entry]
    if source_language:
        cmd += ["--source-language", source_language]
    if target_functions:
        cmd += ["--target-functions", ",".join(target_functions)]
    if framework:
        cmd += ["--framework", framework]
    if driver_preparation:
        cmd += ["--prepare-driver"]
        if invocation_spec_file and Path(invocation_spec_file).is_file():
            cmd += ["--invocation-spec-file", str(Path(invocation_spec_file).resolve())]

    run_exc: Exception | None = None
    out = ""
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=workspace,
            start_new_session=True,
        )
        try:
            remaining = max(1.0, deadline_unix - time.time())
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_forge_process(proc)
        out = (stdout or "") + "\n" + (stderr or "")
        if timed_out:
            run_exc = RuntimeError(f"forge rewrite exceeded absolute deadline after {timeout_s}s")
        if proc.returncode != 0 and run_exc is None:
            run_exc = RuntimeError(f"forge rewrite exited rc={proc.returncode}: {_forge_failure_tail(out)}")
    except Exception as exc:  # noqa: BLE001
        run_exc = exc

    try:
        with open(forge_log, "a") as handle:
            handle.write("\n=== forge-rewrite-by-flydsl (cli) stdout ===\n")
            handle.write(out)
            if run_exc:
                handle.write(f"\n=== forge rewrite exception ===\n{run_exc}\n")
    except OSError:  # noqa: S110
        pass

    parsed = None
    try:
        if result_json.exists():
            parsed = json.loads(result_json.read_text())
    except (OSError, json.JSONDecodeError):
        parsed = None
    sentinel = _flydsl_rewrite.RESULT_SENTINEL
    if parsed is None and sentinel in out:
        try:
            parsed = json.loads(out.split(sentinel)[1])
        except (IndexError, json.JSONDecodeError):
            parsed = None
    return RewriteRunOutcome(
        result=parsed if isinstance(parsed, dict) else None,
        output=out,
        error=run_exc,
        timed_out=timed_out,
    )


# Canonical claude/usage token counters (mirrors parse_usage.normalize_usage).
_FORGE_USAGE_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _usage_has_token_counter(usage: object) -> bool:
    """True when ``usage`` carries at least one int-coercible canonical counter.

    Mirrors the FORGE_LLM_USAGE consumer's contract
    (``parse_usage.normalize_usage``): a usage block is meaningful as soon as
    any of the four canonical token counters is present and int-coercible. The
    per-iteration ``calls`` field is optional metadata, not a precondition.
    """
    if not isinstance(usage, dict):
        return False
    for key in _FORGE_USAGE_TOKEN_KEYS:
        value = usage.get(key)
        if value is None:
            continue
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            continue
    return False


def _forge_trace_from_sidecar(output_dir: Path) -> tuple[dict | None, dict | None]:
    """Recover the forge run's LLM usage + key-step timeline from the CLI sidecar.

    The forge loop runs in an isolated subprocess, so its in-process usage /
    IterationResults are not reachable here. When the forge-loop CLI serializes
    them into ``forge_cli_result.json`` (keys ``llm_usage`` / ``steps``),
    surface them so ``submit`` can re-emit the canonical FORGE_LLM_USAGE /
    FORGE_STEPS markers.

    Returns ``(llm_usage, steps)``; either is ``None`` when the sidecar is
    missing or lacks that field, leaving the markers a no-op.
    """
    sidecar = Path(output_dir) / "forge_cli_result.json"
    try:
        if not sidecar.exists():
            return None, None
        import json as _json

        parsed = _json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001 — best-effort: a bad sidecar is not fatal
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    usage = parsed.get("llm_usage")
    usage = usage if _usage_has_token_counter(usage) else None
    steps = parsed.get("steps")
    steps = steps if isinstance(steps, dict) and steps.get("steps") else None
    return usage, steps


def _run_rewrite_attempt(
    *,
    route: "_flydsl_rewrite.RewriteDecision",
    workspace: str,
    base_commit: str,
    source_file: str,
    implementation_sources: list[str],
    kernel_kind: str,
    output_dir: Path,
    experiments_dir: Path,
    forge_log: Path,
    invocation_spec_file: str,
    snr_threshold: float,
    gpu_type: str,
    max_hours: float,
    deadline_unix: float,
    timeout_s: int,
    started: float,
) -> tuple[dict, list[str]]:
    """Run one FlyDSL rewrite attempt and accept only a canonical apply-back.

    Returns:
        tuple[dict, list[str]]: The backend result dict, and the producer's
            declared temporary paths -- empty unless an apply-back validated,
            because reclaiming them is destructive and needs a trusted source.
    """
    spec = route.spec
    outcome = _run_rewrite_via_cli(
        source_kernel=spec.source_kernel,
        driver=spec.driver,
        logical_op_name=spec.logical_operator,
        source_entry=spec.source_entry,
        source_language=spec.source_language,
        workspace=workspace,
        experiments_dir=experiments_dir,
        result_json=output_dir / "forge_rewrite_result.json",
        target_functions=list(spec.implementation_symbols),
        shapes=[dict(case) for case in spec.shape_cases],
        invocation_spec_file=invocation_spec_file,
        driver_preparation=bool(route.capabilities and route.capabilities.driver_preparation),
        snr_threshold=snr_threshold,
        gpu_target=spec.gpu_target,
        gpu_type=gpu_type,
        max_hours=max_hours,
        branch=spec.branch,
        framework=spec.framework,
        forge_log=forge_log,
        timeout_s=timeout_s,
        deadline_unix=deadline_unix,
    )
    # The producer is never given an experiment id, so it writes no forge-loop
    # checkpoint: a published apply-back is the only evidence this route takes.
    applyback_problems: list[str] = []
    applyback = _validated_rewrite_applyback_result(
        outcome.result,
        workspace=workspace,
        base_commit=base_commit,
        problems=applyback_problems,
    )

    def _rejected(detail: str) -> tuple[dict, list[str]]:
        """Report a rewrite attempt that produced nothing this route can keep."""
        failed = _normalized(1, "", detail, time.time() - started)
        failed["timed_out"] = outcome.timed_out
        failed["salvaged"] = False
        failed["output_dir"] = str(output_dir)
        failed["cli_workspace"] = str(output_dir)
        failed["flydsl_rewrite"] = route.as_dict()
        return failed, []

    if applyback is None:
        detail = (
            "forge rewrite timed out before publishing an apply-back patch"
            if outcome.timed_out
            else "forge rewrite returned without a validated apply-back patch"
        )
        if applyback_problems:
            detail = f"{detail}: {applyback_problems[-1]}"
        if outcome.error is not None:
            detail = f"{detail}: {outcome.error}"
        return _rejected(detail)

    # A contract-valid apply-back that is not faster is a policy rejection, not
    # a malformed artifact. Naming it separately keeps the two apart in the log,
    # and keeps the producer's scratch unreclaimed on any rejection.
    if applyback["best_ms"] >= applyback["baseline_ms"]:
        log.info(
            "forge rewrite: rejecting a valid apply-back that is not faster (best=%sms baseline=%sms commit=%s)",
            applyback["best_ms"],
            applyback["baseline_ms"],
            applyback["best_commit"][:12],
        )
        return _rejected(
            "forge rewrite published a reference-verified apply-back that is not "
            f"faster than the source: best={applyback['best_ms']}ms vs "
            f"baseline={applyback['baseline_ms']}ms"
        )

    salvaged = bool(outcome.error)
    _, changed_files = _export_best_artifacts(
        workspace,
        base_commit,
        spec.source_kernel,
        source_file,
        output_dir,
        best_commit=applyback["best_commit"],
    )
    _write_changed_files_index(output_dir, changed_files)
    baseline_ms = applyback["baseline_ms"]
    best_ms = applyback["best_ms"]
    # The rewrite oracle reports one aggregate timing per implementation, so
    # their ratio is this route's validated micro gain.
    micro_speedup = baseline_ms / best_ms
    _write_report(
        output_dir,
        baseline_ms,
        best_ms,
        True,
        mean_case_speedup=micro_speedup,
        search_start_ms=baseline_ms,
        improved_during_search=True,
        integration_validation=applyback["integration_validation_status"],
    )
    msg = (
        f"forge rewrite done (cli): baseline={baseline_ms} best={best_ms} "
        f"micro_speedup={micro_speedup:.4f} "
        f"commit={applyback['best_commit'][:12]} "
        f"changed_files={len(applyback['changed_files'])} "
        f"integration={applyback['integration_validation_status']} "
        f"salvaged={'yes' if salvaged else 'no'}"
    )
    res = _normalized(
        0,
        msg + "\n" + (outcome.output or "")[-3000:],
        "",
        time.time() - started,
    )
    res["cli_workspace"] = str(output_dir)
    res["output_dir"] = str(output_dir)
    res["timed_out"] = outcome.timed_out
    res["salvaged"] = salvaged
    res["pristine_baseline_ms"] = baseline_ms
    res["search_start_ms"] = baseline_ms
    res["best_ms"] = best_ms
    res["mean_case_speedup"] = micro_speedup
    res["search_start_mean_case_speedup"] = 1.0
    res["improved"] = True
    res["total_improved"] = True
    res["incremental_improved"] = True
    res["improved_during_search"] = True
    res["best_commit"] = applyback["best_commit"]
    res["canonical_patch_path"] = applyback["canonical_patch_path"]
    res["canonical_files_root"] = applyback["canonical_files_root"]
    res["best_manifest"] = applyback["canonical_manifest"]
    res["changed_files"] = applyback["changed_files"]
    res["forge_workspace"] = str(Path(workspace).resolve())
    res["artifacts"] = [applyback["canonical_patch_path"]]
    res["flydsl_rewrite"] = route.as_dict()
    res["flydsl_applyback"] = applyback
    res["logical_operator"] = spec.logical_operator
    res["source_framework"] = spec.framework
    res["implementation_sources"] = implementation_sources
    res["kernel_kind"] = kernel_kind
    res["target_functions"] = list(spec.implementation_symbols)
    return res, applyback["temporary_paths"]


_RELOCATABLE_ARTIFACT_KEYS = (
    "canonical_patch_path",
    "canonical_files_root",
    "canonical_manifest",
    "best_manifest",
    "checkpoint_path",
)


def _repoint_relocated_artifacts(
    result: dict[str, Any] | None,
    *,
    moved_from: Path,
    moved_to: Path,
) -> None:
    """Rewrite artifact paths in ``result`` that a directory move just invalidated."""
    if not result:
        return

    def relocated(raw: Any) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            relative = Path(text).relative_to(moved_from)
        except ValueError:
            return None
        return str(moved_to / relative)

    for key in _RELOCATABLE_ARTIFACT_KEYS:
        moved = relocated(result.get(key))
        if moved:
            result[key] = moved
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        result["artifacts"] = [relocated(entry) or str(entry) for entry in artifacts]
    applyback = result.get("flydsl_applyback")
    if isinstance(applyback, dict):
        for key in _RELOCATABLE_ARTIFACT_KEYS:
            moved = relocated(applyback.get(key))
            if moved:
                applyback[key] = moved


def _finalize_forge_workspace(
    *,
    inplace: bool,
    restore_info: dict | None,
    driver: str,
    workspace: str,
    output_dir: Path,
    branch: str,
    nogit_scratch: bool,
    temporary_paths: list[str] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Restore live repos, but retain isolated Forge workspaces for inspection.

    ``temporary_paths`` are scratch files a producer declared in a validated
    result. They are reclaimed only in place, and only after re-confirming
    containment, so an unvalidated run never deletes anything it merely guessed.

    ``result`` is the dict about to be returned. Relocating an in-place campaign
    directory moves the producer's published bundle with it, so its artifact paths
    are repointed rather than left naming a directory this just emptied.
    """
    # --git-common-dir resolves to the live repo even from a linked worktree, so
    # the exclude entry outlives a worktree run unless it is removed here too.
    if not nogit_scratch:
        _restore_generated_driver_exclude(Path(workspace))

    if inplace:
        cleanup_errors: list[str] = []
        campaign_root = Path(workspace) / "forge_experiments"
        if campaign_root.is_dir():
            destination = Path(output_dir) / "forge_experiments"
            try:
                # ``--experiments-dir`` already points at (and mkdir's) this
                # path, so an empty destination is the normal case and must not
                # abort cleanup. Only a destination holding real artifacts is
                # preserved, by moving the workspace campaign beside it.
                if destination.is_dir() and not any(destination.iterdir()):
                    destination.rmdir()
                elif destination.exists():
                    preserved = destination
                    suffix = 1
                    while preserved.exists():
                        preserved = destination.with_name(f"{destination.name}_workspace_{suffix}")
                        suffix += 1
                    log.warning(
                        "forge: %s already holds campaign artifacts; preserving the in-place campaign at %s instead",
                        destination,
                        preserved,
                    )
                    destination = preserved
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(campaign_root), str(destination))
                _repoint_relocated_artifacts(
                    result,
                    moved_from=campaign_root,
                    moved_to=destination,
                )
            except OSError as error:
                cleanup_errors.append(f"failed to preserve in-place campaign artifacts: {error}")
        driver_paths: set[Path] = set()
        try:
            driver_paths.update(Path(workspace).glob(".forge_driver_*.py"))
        except OSError as error:
            cleanup_errors.append(f"failed to enumerate generated in-place drivers: {error}")
        if driver:
            driver_paths.add(Path(driver))
        for driver_path in driver_paths:
            if not driver_path.name.startswith(".forge_driver_"):
                continue
            try:
                driver_path.unlink()
            except FileNotFoundError:
                pass  # already gone -- nothing to clean up
            except OSError as error:
                cleanup_errors.append(f"failed to remove generated in-place driver: {error}")
        workspace_root = Path(workspace).resolve()
        for raw in temporary_paths or []:
            declared = Path(str(raw))
            if not declared.is_absolute() or not _path_is_within(declared, workspace_root):
                cleanup_errors.append(f"declared temporary path escapes the workspace: {raw}")
                continue
            if declared.resolve() == workspace_root:
                cleanup_errors.append("declared temporary path is the workspace root itself")
                continue
            try:
                if declared.is_dir() and not declared.is_symlink():
                    shutil.rmtree(declared)
                else:
                    declared.unlink(missing_ok=True)
            except OSError as error:
                cleanup_errors.append(f"failed to remove producer temporary path {declared}: {error}")
        try:
            _restore_inplace(restore_info)
        except Exception as error:  # noqa: BLE001 - combine cleanup/restore failures
            cleanup_errors.append(f"failed to restore in-place repository: {error}")
        if cleanup_errors:
            raise RuntimeError("in-place workspace cleanup failed: " + "; ".join(cleanup_errors))
        return
    log.info(
        "forge: retaining workspace for inspection: %s (branch=%s, nogit=%s)",
        workspace,
        branch,
        nogit_scratch,
    )


# --- Vendor-operator-playbook route -----------------------------------------
#
# A vendor-playbook candidate (mori's EP dispatch/combine is the first case,
# see _vendor_operator_playbooks.py and KernelForge PR #88) has no rewritable
# device source: it's a pip-installed compiled library. Instead of the
# git-worktree / source-rewrite pipeline `submit()` otherwise runs, this copies
# a validated KernelForge `examples/<task>/` bundle into a scratch workspace
# and runs forge-loop against that bundle's own driver/config/program.md.
#
# mori's dispatch and combine are two separate hot-kernel candidates that
# share one playbook id and are deliberately invoked as **one** Forge
# task/session (not two) -- the lock/result files below de-duplicate so a
# session that dispatches both candidates only launches forge-loop once.

_VENDOR_PLAYBOOK_CLAIM_POLL_S = 5.0

# A cached FAILURE only de-dupes submissions within this window -- long
# enough to catch a genuinely concurrent dispatch+combine pair, short enough
# that Hyperloom's normal "fail -> add budget -> retry" model gets a fresh
# attempt instead of the whole group being permanently retired for the rest
# of the session by one transient failure (see PR #1191 review finding #2).
# A cached SUCCESS has no such expiry: sharing one session's result for the
# rest of the session is the intended dedup behavior this module implements.
_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S = 600.0

# Extra time past an attempt's own timeout_s before its claim is presumed
# abandoned (SIGKILL budget enforcement, OOM, node restart -- anything that
# kills the holder without a chance to write result.json) rather than merely
# still finishing up (writing the report, staging the artifact copy, etc).
_VENDOR_PLAYBOOK_CLAIM_STALE_GRACE_S = 120.0


def _vendor_playbook_lock_dir(output_dir: Path, group_id: str) -> Path:
    """Return the session-scoped directory used to de-duplicate a playbook group.

    ``output_dir`` is per-attempt (``.../forge/<session_id>/<prompt_stem>``);
    the lock lives one level up so every kernel_id in the same analysis
    session and playbook group shares it.
    """
    safe_group = re.sub(r"[^A-Za-z0-9_-]+", "-", group_id).strip("-") or "vendor-playbook"
    return output_dir.parent / "vendor_playbook_locks" / safe_group


def _read_vendor_playbook_cached_result(lock_dir: Path, *, max_failure_age_s: float | None = None) -> dict | None:
    """Read a previously-cached vendor-playbook result, if any.

    A cached SUCCESS (``returncode == 0``) is returned unconditionally --
    sharing one session's validated result for the rest of the session is
    the intended dedup behavior. A cached FAILURE is only returned while it
    is younger than ``max_failure_age_s``; once it ages out it is treated as
    absent so a fresh submission actually retries instead of one transient
    failure (a flaky bundle copy, a transient git failure, etc.)
    permanently wedging the whole playbook group for the rest of the session
    (PR #1191 review finding #2).
    """
    result_path = lock_dir / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if max_failure_age_s is not None and result.get("returncode") != 0:
        try:
            age_s = time.time() - result_path.stat().st_mtime
        except OSError:
            age_s = None
        if age_s is not None and age_s > max_failure_age_s:
            return None
    return result


def _write_vendor_playbook_result(lock_dir: Path, result: dict) -> None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = lock_dir / f".result.json.{uuid.uuid4().hex[:8]}.tmp"
    tmp_path.write_text(json.dumps(result, sort_keys=True, default=str), encoding="utf-8")
    tmp_path.replace(lock_dir / "result.json")


def _write_claim_marker(claim_path: Path, *, nonce: str | None = None) -> str:
    """Write ``{pid, claimed_at, nonce}`` into an already-created claim file.

    The timestamp lets any waiter compute how long the claim has been held
    without a result appearing; the nonce lets ``_steal_stale_claim`` verify
    which of several racing stealers actually won the replace.
    """
    nonce = nonce or uuid.uuid4().hex
    payload = {"pid": os.getpid(), "claimed_at": time.time(), "nonce": nonce}
    claim_path.write_text(json.dumps(payload), encoding="utf-8")
    return nonce


def _claim_marker_age_s(claim_path: Path) -> float | None:
    """Return how long ago ``claim_path`` was claimed, or ``None`` if unknown."""
    try:
        raw = claim_path.read_text(encoding="utf-8")
    except OSError:
        return None
    claimed_at: float | None = None
    if raw.strip():
        try:
            claimed_at = float(json.loads(raw).get("claimed_at"))
        except (ValueError, TypeError):
            claimed_at = None
    if claimed_at is None:
        # Marker predates this format (or failed to write its JSON body) --
        # fall back to the file's own mtime rather than treating age as
        # unknown, since os.O_CREAT|O_EXCL always sets one.
        try:
            claimed_at = claim_path.stat().st_mtime
        except OSError:
            return None
    return max(0.0, time.time() - claimed_at)


def _claim_is_stale(claim_path: Path, timeout_s: int) -> bool:
    """A claim is stale (its holder is presumed done or dead) when either:

    1. A result already exists but has aged out of the failure-cache TTL
       (``_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S``) -- the completed attempt's
       ``claimed.lock`` is never deleted, so without this check a lingering
       claim from a long-finished, now-expired failure would block every
       later retry from ever running (PR #1191 review finding #2 combined
       with #3: the claim and the result cache must age out together).
    2. The claim is older than its own attempt budget plus grace, with no
       result at all -- covers SIGKILL budget enforcement, OOM, and node
       restarts, none of which give the holder a chance to write
       ``result.json``. Without this check every subsequent submission for
       the group would poll ``_wait_for_vendor_playbook_result`` all the way
       to its deadline and still find nothing -- for a 60-minute-budget
       attempt, that is an hour burned per submission (PR #1191 review
       finding #3).
    """
    lock_dir = claim_path.parent
    cached = _read_vendor_playbook_cached_result(lock_dir, max_failure_age_s=_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S)
    if cached is None and (lock_dir / "result.json").is_file():
        return True
    age_s = _claim_marker_age_s(claim_path)
    if age_s is None:
        return False
    return age_s > (float(timeout_s) + _VENDOR_PLAYBOOK_CLAIM_STALE_GRACE_S)


def _steal_stale_claim(claim_path: Path) -> bool:
    """Atomically replace a stale claim, verifying this caller actually won it.

    ``os.replace`` never raises when the target already exists, so two
    waiters racing to steal the same stale claim could both believe they
    succeeded. Tag the write with a nonce and read it back afterwards: only
    the caller whose nonce is what's on disk is the new owner.
    """
    nonce = uuid.uuid4().hex
    tmp_path = claim_path.with_name(f".{claim_path.name}.{nonce}.tmp")
    try:
        _write_claim_marker(tmp_path, nonce=nonce)
        os.replace(str(tmp_path), str(claim_path))
        current = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            # Best-effort scratch-file cleanup only: by now tmp_path has
            # already been atomically replaced onto claim_path (success) or
            # never fully written (failure), so nothing downstream depends
            # on this unlink -- it must never raise out of a claim-stealing
            # attempt over something as inconsequential as a leftover temp
            # file (missing_ok=True already covers the common "already
            # gone" case; this only guards rarer failures like EPERM).
            pass
    return current.get("nonce") == nonce


def _claim_vendor_playbook_run(lock_dir: Path, timeout_s: int) -> bool:
    """Atomically claim the right to run this group's one forge-loop session.

    Returns ``True`` for whichever caller wins the race (dispatch or
    combine, whichever the orchestrator happened to submit first); the loser
    waits for the winner's result instead of launching a second session.
    Also returns ``True`` when the existing claim is stale (its holder is
    presumed dead) and this caller wins the steal.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    claim_path = lock_dir / "claimed.lock"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if not _claim_is_stale(claim_path, timeout_s):
            return False
        return _steal_stale_claim(claim_path)
    try:
        _write_claim_marker(claim_path)
    except OSError:
        log.warning(
            "forge: claimed %s but could not write its pid/timestamp marker; "
            "staleness detection will fall back to file mtime",
            claim_path,
        )
    return True


def _wait_for_vendor_playbook_result(lock_dir: Path, deadline_unix: float, timeout_s: int) -> dict | None:
    """Poll for the winner's result until it appears, the claim looks
    abandoned, or ``deadline_unix`` passes.

    Returns early (well before ``deadline_unix``) the moment the claim looks
    stale, so a waiter never burns its entire poll window on a holder that
    was SIGKILLed/OOM-killed and will never write a result (PR #1191 review
    finding #3). The caller is responsible for then trying to claim (steal)
    the group itself rather than treating an early ``None`` as a hard
    failure.
    """
    claim_path = lock_dir / "claimed.lock"
    while True:
        cached = _read_vendor_playbook_cached_result(lock_dir, max_failure_age_s=_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S)
        if cached is not None:
            return cached
        if _claim_is_stale(claim_path, timeout_s):
            return None
        if time.time() >= deadline_unix:
            return None
        time.sleep(min(_VENDOR_PLAYBOOK_CLAIM_POLL_S, max(0.0, deadline_unix - time.time())))


def _stage_vendor_playbook_artifact_for_reuse(cached: dict, output_dir: Path) -> None:
    """Duplicate a reused vendor-playbook result's artifact under ``output_dir``.

    ``kernel_optimization.py``'s ``invoke_backend()`` unconditionally resets
    ``result["output_dir"]`` to *this* attempt's own directory right after
    ``submit()`` returns (``result["output_dir"] = str(out_dir)``), and
    ``_candidate_artifact_paths()`` looks under both ``cli_workspace`` and
    ``output_dir`` for an ``optimized_versions/`` directory. A cache-hit
    result's ``cli_workspace``/``output_dir`` fields describe the *winner's*
    directory (correct at the time they were written to ``result.json``,
    before that later overwrite mutates the in-memory dict this call
    returns), so the winner's directory alone would silently stop being
    reachable for a reused sibling once the caller clobbers ``output_dir``.
    Physically copying the file(s) here makes the reused result
    self-contained regardless of that overwrite.
    """
    src_opt = None
    for key in ("cli_workspace", "output_dir"):
        candidate_dir = cached.get(key)
        if not candidate_dir:
            continue
        probe = Path(candidate_dir) / "optimized_versions"
        if probe.is_dir():
            src_opt = probe
            break
    if src_opt is None:
        return
    dest_opt = output_dir / "optimized_versions"
    try:
        dest_opt.mkdir(parents=True, exist_ok=True)
        for item in src_opt.iterdir():
            if not item.is_file():
                continue
            dest = dest_opt / item.name
            if not dest.exists():
                shutil.copy2(item, dest)
    except OSError as exc:
        log.warning("forge: could not stage reused vendor playbook artifact copy: %s", exc)


def _copy_vendor_task_bundle(task_bundle_root: Path, workspace: Path) -> None:
    """Copy a KernelForge ``examples/<task>/`` bundle into ``workspace`` and
    git-init it there.

    forge-loop's IterationLoop runs ``git status``/``git checkout`` against
    the workspace to snapshot and restore each attempt (see the bundle's own
    ``run_example.sh``, which does the identical ``git init`` + commit before
    invoking forge-loop directly); a bare directory of copied files with no
    ``.git`` fails the very first git call with "not a git repository".
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for item in sorted(task_bundle_root.iterdir()):
        if item.name in (".git", "__pycache__"):
            continue
        dest = workspace / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    gitignore = workspace / ".gitignore"
    if not gitignore.is_file():
        gitignore.write_text("__pycache__/\n*.pyc\n*.log\nbuild/\nforge_experiments/\n", encoding="utf-8")
    _git = shutil.which("git") or "git"
    subprocess.run([_git, "init", "-q"], cwd=str(workspace), check=True)
    subprocess.run(
        [_git, "config", "user.email", "forge-vendor-playbook@local"],
        cwd=str(workspace),
        check=True,
    )
    subprocess.run(
        [_git, "config", "user.name", "forge-vendor-playbook"],
        cwd=str(workspace),
        check=True,
    )
    subprocess.run([_git, "add", "-A"], cwd=str(workspace), check=True)
    subprocess.run(
        [_git, "commit", "-q", "-m", "vendor playbook: initial task bundle", "--allow-empty"],
        cwd=str(workspace),
        check=True,
    )


def _run_vendor_playbook_loop_via_cli(
    *,
    kernel_anchor: str,
    driver: str,
    workspace: str,
    snr_threshold: float,
    max_hours: float,
    branch: str,
    gpu_target: str,
    gpu_type: str,
    kernel_backend: str,
    program_md_file: str,
    target_functions: list[str],
    experiments_dir: Path,
    forge_log: Path,
    timeout_s: int,
    deadline_unix: float,
    experience_id: str,
    extra_env: dict[str, str] | None = None,
) -> ForgeLoopOutcome:
    """Run forge-loop against a copied vendor-playbook task bundle.

    Mirrors ``_run_loop_via_cli``'s subprocess/result-parsing conventions, but
    always passes ``--no-profiling --no-prepare-task`` (the bundle already
    ships a hand-written, validated ``driver.py`` -- forge-loop's own
    task-preparer/profiler must not try to author or reprofile it) and forwards
    the playbook's own env requirements (e.g. ``KERNELFORGE_INCLUDE_MORI_KB``).
    """
    result_json = experiments_dir.parent / "forge_cli_result.json"
    checkpoint_json = experiments_dir / f"{_FORGE_EXPERIMENT_ID}.json"
    for stale_path in (result_json, checkpoint_json):
        try:
            stale_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"could not clear stale Forge recovery artifact {stale_path}: {exc}") from exc

    env = dict(os.environ)
    env["GPU_TARGET"] = gpu_target
    _apply_gpu_type_env(env, gpu_type)
    _apply_kernel_backend_env(env)
    for key, value in (extra_env or {}).items():
        env[str(key)] = str(value)

    cmd = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--kernel",
        kernel_anchor,
        "--driver",
        driver,
        "--workspace",
        workspace,
        "--snr-threshold",
        str(snr_threshold),
        "--max-hours",
        str(max_hours),
        "--git-branch",
        branch,
        "--gpu-target",
        gpu_target,
        "--kernel-backend",
        kernel_backend,
        "--experiments-dir",
        str(experiments_dir),
        "--experiment-id",
        _FORGE_EXPERIMENT_ID,
        "--experience-id",
        experience_id or experiments_dir.parent.name,
        "--deadline-unix",
        str(deadline_unix),
        "--result-json",
        str(result_json),
        "--no-profiling",
        "--no-prepare-task",
    ]
    cmd += ["--gpu-type", _known_gpu_model(gpu_type)]
    if _openai_only_provider():
        cmd += ["--agent-backend", "codex", "--agent-fallback-provider", "none"]
        codex_model = (os.environ.get("CODEX_MODEL") or "").strip()
        if codex_model:
            cmd += ["--model", codex_model]
    if program_md_file and Path(program_md_file).exists():
        cmd += ["--program-md-file", str(program_md_file)]
    if target_functions:
        cmd += ["--target-functions", ",".join(target_functions)]

    loop_exc = None
    out = ""
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=workspace,
            start_new_session=True,
        )
        try:
            remaining = max(1.0, deadline_unix - time.time())
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_forge_process(proc)
        out = (stdout or "") + "\n" + (stderr or "")
        if timed_out:
            loop_exc = RuntimeError(f"forge-loop exceeded absolute deadline after {timeout_s}s")
        if proc.returncode != 0:
            if loop_exc is None:
                loop_exc = RuntimeError(f"forge-loop exited rc={proc.returncode}: {_forge_failure_tail(out)}")
    except Exception as exc:  # noqa: BLE001
        loop_exc = exc

    try:
        with open(forge_log, "a") as f:
            f.write("\n=== forge-loop vendor-playbook (cli) stdout ===\n")
            f.write(out)
            if loop_exc:
                f.write(f"\n=== forge-loop exception ===\n{loop_exc}\n")
    except OSError:  # noqa: S110
        pass

    baseline_ms = best_ms = None
    pristine_baseline_ms = search_start_ms = None
    mean_case_speedup = search_start_mean_case_speedup = None
    improved = improved_during_search = total_improved = incremental_improved = False
    parsed = None
    try:
        if result_json.exists():
            parsed = json.loads(result_json.read_text())
    except Exception:
        parsed = None
    if parsed is None and "__FORGE_RESULT__" in out:
        try:
            parsed = json.loads(out.split("__FORGE_RESULT__")[1])
        except Exception:
            parsed = None
    if parsed:
        baseline_ms = parsed.get("baseline_ms")
        best_ms = parsed.get("best_ms")
        pristine_baseline_ms = parsed.get("pristine_baseline_ms", baseline_ms)
        search_start_ms = parsed.get("search_start_ms", baseline_ms)
        (
            mean_case_speedup,
            search_start_mean_case_speedup,
            total_improved,
            incremental_improved,
        ) = _observed_mean_case_result_fields(parsed)
        improved = total_improved
        improved_during_search = incremental_improved
        if parsed.get("deadline_expired"):
            timed_out = True
            if loop_exc is None:
                loop_exc = RuntimeError("forge-loop reached its graceful absolute deadline")
    checkpoint = _read_forge_checkpoint(experiments_dir)
    return ForgeLoopOutcome(
        baseline_ms=baseline_ms,
        best_ms=best_ms,
        improved=improved,
        output=out,
        error=loop_exc,
        timed_out=timed_out,
        checkpoint=checkpoint,
        pristine_baseline_ms=pristine_baseline_ms,
        search_start_ms=search_start_ms,
        improved_during_search=improved_during_search,
        structured_result=parsed if isinstance(parsed, dict) else None,
        mean_case_speedup=mean_case_speedup,
        search_start_mean_case_speedup=search_start_mean_case_speedup,
        total_improved=total_improved,
        incremental_improved=incremental_improved,
    )


def _resolve_vendor_task_bundle(relative: str) -> Path | None:
    """Locate a vendor playbook's task bundle under KernelForge's ``examples/``.

    The bundle ships inside the installed ``kernelforge`` package, so this needs
    no environment at all -- it used to hard-fail with "FORGE_PATH is not set",
    which is no longer a precondition. An operator who must substitute a bundle
    without reinstalling points ``$KERNELFORGE_PROJECT_ROOT`` at a tree holding
    it, which :func:`resource_path` honours ahead of the packaged copy.

    Returns ``None`` for an empty ``relative``. ``missing_ok`` keeps a bundle
    the package does not carry reportable as a concrete path, which the caller
    turns into ``skipped`` rather than a failure.
    """
    if not relative:
        return None

    from kernelforge.resources import default_project_root, resource_path

    return resource_path(relative, default_project_root(), missing_ok=True)


def _run_claimed_vendor_playbook(
    *,
    candidate: dict[str, Any],
    prompt_file: Path,
    output_dir: Path,
    timeout_s: int,
    playbook: dict[str, Any],
    group_id: str,
    role: str,
    lock_dir: Path,
    started: float,
) -> dict:
    """Copy the task bundle and run forge-loop, having already won the claim.

    May raise (e.g. ``subprocess.CalledProcessError`` from the git-init
    calls in ``_copy_vendor_task_bundle``, or anything else unexpected from
    forge-loop setup) -- the caller (``_submit_vendor_playbook``) must catch
    broadly and always write a result to ``lock_dir``, or a raised exception
    here leaves ``claimed.lock`` in place forever with no result for any
    waiting sibling or later retry to find.
    """
    task_bundle_root = _resolve_vendor_task_bundle(str(playbook.get("task_bundle") or ""))
    if task_bundle_root is None or not task_bundle_root.is_dir():
        result = _normalized(
            2,
            "",
            f"forge: vendor playbook task bundle not found: {task_bundle_root} (playbook {group_id!r})",
            time.time() - started,
            skipped=True,
        )
        _write_vendor_playbook_result(lock_dir, result)
        return result

    workspace = output_dir / "worktree"
    if workspace.exists() or workspace.is_symlink():
        result = _normalized(
            2,
            "",
            f"forge: retained vendor playbook workspace already exists: {workspace}",
            time.time() - started,
            skipped=True,
        )
        _write_vendor_playbook_result(lock_dir, result)
        return result

    try:
        _copy_vendor_task_bundle(task_bundle_root, workspace)
    except (OSError, subprocess.CalledProcessError) as exc:
        # _copy_vendor_task_bundle's git init/config/add/commit calls run with
        # check=True and raise CalledProcessError, not OSError, on failure.
        result = _normalized(
            2,
            "",
            f"forge: failed to copy vendor playbook task bundle {task_bundle_root} -> {workspace}: {exc}",
            time.time() - started,
            skipped=True,
        )
        _write_vendor_playbook_result(lock_dir, result)
        return result

    kernel_anchor = workspace / str(playbook.get("kernel_anchor") or "")
    driver = workspace / str(playbook.get("driver") or "driver.py")
    program_md = workspace / str(playbook.get("program_md") or "program.md")
    if not program_md.is_file():
        program_md = Path(prompt_file)

    experiments_dir = output_dir / "forge_experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    forge_log = output_dir / "forge_loop.log"
    branch = _new_forge_branch(output_dir, str(kernel_anchor))
    gpu_target = _resolve_gpu_target(candidate)
    gpu_type = _resolve_gpu_type(candidate)
    snr_threshold = float(playbook.get("snr_threshold", 30.0))
    if timeout_s < _FORGE_MIN_BUDGET_SEC:
        log.warning(
            "forge budget %.0f min is below the %d-min minimum forge-loop "
            "accepts for vendor playbook %r; running with the floored "
            "--max-hours and hard-killing at the requested budget",
            timeout_s / 60.0,
            _FORGE_MIN_BUDGET_SEC // 60,
            group_id,
        )
    deadline_unix = max(time.time() + 1.0, started + timeout_s)

    loop_outcome = _run_vendor_playbook_loop_via_cli(
        kernel_anchor=str(kernel_anchor),
        driver=str(driver),
        workspace=str(workspace),
        snr_threshold=snr_threshold,
        max_hours=max(_FORGE_MIN_BUDGET_SEC / 3600.0, timeout_s / 3600.0),
        branch=branch,
        gpu_target=gpu_target,
        gpu_type=gpu_type,
        kernel_backend=str(playbook.get("kernel_backend") or "aiter"),
        program_md_file=str(program_md),
        target_functions=[str(f) for f in (playbook.get("target_functions") or [])],
        experiments_dir=experiments_dir,
        forge_log=forge_log,
        timeout_s=timeout_s,
        deadline_unix=deadline_unix,
        experience_id=output_dir.name,
        extra_env={str(k): str(v) for k, v in (playbook.get("env") or {}).items()},
    )
    result = _normalized(
        0 if loop_outcome.error is None else 1,
        loop_outcome.output or "",
        "" if loop_outcome.error is None else str(loop_outcome.error),
        time.time() - started,
    )
    # kernel_optimization.py's build_verification() only recognizes a forge
    # attempt's measured speedup when total_improved/mean_case_speedup are
    # BOTH present on the result dict it reads (see run_attempt's field
    # copy); leaving any of these out silently downgrades a real KEEP-worthy
    # improvement to PARTIAL ("no measurable speedup found"), even though
    # forge-loop itself committed and validated a faster config.
    result.update(
        {
            # NOTE: cli_workspace intentionally equals output_dir here (the
            # convention the ordinary per-file forge path uses, see
            # `res["cli_workspace"] = str(output_dir)` elsewhere in this
            # module), NOT the git worktree -- optimized_versions/ below is
            # written directly under output_dir, and _candidate_artifact_paths()
            # checks cli_workspace/optimized_versions first. forge_workspace
            # separately carries the real git worktree for anything that needs
            # the live tree (e.g. a future patch-based snapshot).
            "cli_workspace": str(output_dir),
            "forge_workspace": str(workspace),
            "output_dir": str(output_dir),
            "improved": bool(loop_outcome.total_improved),
            "total_improved": bool(loop_outcome.total_improved),
            "incremental_improved": bool(loop_outcome.incremental_improved),
            "improved_during_search": bool(loop_outcome.improved_during_search),
            "mean_case_speedup": loop_outcome.mean_case_speedup,
            "search_start_mean_case_speedup": loop_outcome.search_start_mean_case_speedup,
            "best_ms": loop_outcome.best_ms,
            "baseline_ms": loop_outcome.baseline_ms,
            "pristine_baseline_ms": loop_outcome.pristine_baseline_ms,
            "search_start_ms": loop_outcome.search_start_ms,
            "vendor_playbook_id": group_id,
            "vendor_playbook_role": role,
            "vendor_playbook_task_bundle": str(task_bundle_root),
            "vendor_playbook_reused": False,
            # This role is the one that actually ran forge-loop and produced
            # the measurement; a sibling role that reuses this same result
            # (see _submit_vendor_playbook's ``_reuse``) marks itself False
            # so downstream benefit accounting sums this speedup once, not
            # once per role sharing it (PR #1191 review finding #4).
            "vendor_playbook_independently_counted": True,
        }
    )
    # The ordinary per-file forge path's correctness signal comes from
    # optimization_report.md's "[correctness] pass" marker (kernel_optimization
    # .py's _extract_correctness_from_report scans cli_workspace for it). The
    # vendor-playbook path reused this same forge-loop run but never wrote
    # that file, so correctness_passed stayed False and make_proposal()
    # could never return KEEP even when SNR validation had already passed
    # inside forge-loop (PR #1191 review finding #5). cli_workspace ==
    # output_dir here (see the NOTE above), so writing it here is exactly
    # where the correctness scan will look.
    _write_report(
        output_dir,
        loop_outcome.baseline_ms,
        loop_outcome.best_ms,
        loop_outcome.total_improved,
        mean_case_speedup=loop_outcome.mean_case_speedup,
        search_start_ms=loop_outcome.search_start_ms,
        improved_during_search=loop_outcome.improved_during_search,
    )
    if loop_outcome.total_improved and kernel_anchor.is_file():
        # There is no separate "deploy" artifact for a vendor launch-config:
        # the tuned values live in the anchor file forge-loop already
        # committed in-place in workspace. Materialize a copy under the
        # attempt's own optimized_versions/ so _select_source_artifact()
        # (which only looks in that conventional directory) can find it,
        # exactly like the ordinary per-file-rewrite forge path does.
        try:
            opt_dir = output_dir / "optimized_versions"
            opt_dir.mkdir(parents=True, exist_ok=True)
            dest = opt_dir / f"{group_id}_{role or 'optimized'}{kernel_anchor.suffix}"
            shutil.copy2(kernel_anchor, dest)
        except OSError as exc:
            log.warning("forge: could not stage vendor playbook artifact copy: %s", exc)
    _write_vendor_playbook_result(lock_dir, result)
    return result


def _submit_vendor_playbook(
    *,
    candidate: dict[str, Any],
    prompt_file: Path,
    output_dir: Path,
    timeout_s: int,
) -> dict:
    """Run (or reuse) the one forge-loop session for a vendor-playbook group."""
    started = time.time()
    playbook = candidate.get("vendor_operator_playbook")
    if not isinstance(playbook, dict) or not playbook.get("id"):
        # Defensive re-resolve: a candidate dict round-tripped through JSON by
        # a caller that dropped nested fields still carries enough identity
        # (name/library/source_file) to re-match the registry.
        playbook = match_vendor_operator_playbook(candidate)
    if not isinstance(playbook, dict) or not playbook.get("id"):
        return _normalized(
            2,
            "",
            "forge: patch_strategy=vendor_playbook but candidate carries no "
            "vendor_operator_playbook entry (re-run classify_patchability?)",
            time.time() - started,
            skipped=True,
        )

    group_id = str(playbook.get("id"))
    role = str(candidate.get("vendor_playbook_role") or playbook.get("role") or "")
    lock_dir = _vendor_playbook_lock_dir(output_dir, group_id)

    def _reuse(cached_result: dict) -> dict:
        result = dict(cached_result)
        result["vendor_playbook_reused"] = True
        result["vendor_playbook_role"] = role
        # This measurement was already counted once, on the role that
        # actually ran forge-loop; a reused sibling must not add its
        # identical mean_case_speedup/best_ms again into downstream benefit
        # totals (PR #1191 review finding #4).
        result["vendor_playbook_independently_counted"] = False
        _stage_vendor_playbook_artifact_for_reuse(cached_result, output_dir)
        return result

    def _run_and_guard() -> dict:
        # From here on we (believe we) hold the claim: any unhandled
        # exception MUST still produce a result.json, or claimed.lock is
        # orphaned forever and no sibling/retry for this group can ever run
        # again (see _run_claimed_vendor_playbook's docstring).
        # _copy_vendor_task_bundle's git subprocess calls and the forge-loop
        # launch are the known risks, but this is a deliberate catch-all,
        # not just those two.
        try:
            return _run_claimed_vendor_playbook(
                candidate=candidate,
                prompt_file=prompt_file,
                output_dir=output_dir,
                timeout_s=timeout_s,
                playbook=playbook,
                group_id=group_id,
                role=role,
                lock_dir=lock_dir,
                started=started,
            )
        except Exception as exc:  # noqa: BLE001
            result = _normalized(
                2,
                "",
                f"forge: vendor playbook group {group_id!r} raised after claiming "
                f"the shared session, before producing a result "
                f"({type(exc).__name__}: {exc})",
                time.time() - started,
                skipped=True,
            )
            try:
                _write_vendor_playbook_result(lock_dir, result)
            except OSError:
                log.exception(
                    "forge: could not write a failure result for vendor playbook "
                    "group %r after claiming it; claimed.lock will remain until "
                    "manually cleared",
                    group_id,
                )
            return result

    cached = _read_vendor_playbook_cached_result(lock_dir, max_failure_age_s=_VENDOR_PLAYBOOK_FAILURE_CACHE_TTL_S)
    if cached is not None:
        return _reuse(cached)

    if _claim_vendor_playbook_run(lock_dir, timeout_s):
        return _run_and_guard()

    # A sibling role (e.g. this is "combine" and "dispatch" already claimed
    # the group) is running the one shared session; wait for it rather than
    # launching a second forge-loop for the same task.
    deadline = time.time() + max(60.0, float(timeout_s) + 300.0)
    cached = _wait_for_vendor_playbook_result(lock_dir, deadline, timeout_s)
    if cached is not None:
        return _reuse(cached)

    # The wait ended without a result either because the deadline passed or
    # because the holder's claim looked abandoned (SIGKILL/OOM/node restart
    # -- PR #1191 review finding #3). Try once more to claim the group: if
    # the claim really is stale this steals it and we run for real instead
    # of failing outright; if the original holder is alive and simply still
    # running, this correctly fails again.
    if _claim_vendor_playbook_run(lock_dir, timeout_s):
        return _run_and_guard()

    return _normalized(
        2,
        "",
        f"forge: vendor playbook group {group_id!r} was claimed by a "
        "concurrent submission but never produced a result before the wait "
        "deadline",
        time.time() - started,
        skipped=True,
    )


def submit(
    source_file: str,
    prompt_file: Path,
    output_dir: Path,
    source_type: str = "unknown",
    candidate: dict | None = None,
    timeout_s: int = 1800,
    kernel_repo: str = "",
    invocation_spec_file: str = "",
) -> dict:
    """Run Forge's autonomous loop on one kernel; emit Hyperloom-contract artifacts.

    Hyperloom prepares an isolated git worktree / in-place edit, then runs the
    Forge IterationLoop in a hard-killable CLI subprocess (`kernelforge
    forge-loop`) so a hung kernel backend can never freeze the orchestrator. Returns a
    normalized result dict and writes optimized_versions/ +
    optimization_report.md under output_dir.
    """
    started = time.time()
    candidate = candidate or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vendor-operator-playbook route: a closed-source vendor op (e.g. mori's EP
    # dispatch/combine) has no rewritable device source to worktree/rewrite --
    # skip the entire git-worktree / kernel_backend-resolution / rewrite-route pipeline
    # below and copy the validated KernelForge task bundle instead. See
    # _vendor_operator_playbooks.py and KernelForge PR #88.
    if candidate.get("patch_strategy") == "vendor_playbook":
        return _submit_vendor_playbook(
            candidate=candidate,
            prompt_file=Path(prompt_file),
            output_dir=output_dir,
            timeout_s=timeout_s,
        )

    from hyperloom.orchestrator.knowledge.kernel_experience_bridge import (
        KernelExperienceBridge,
    )

    knowledge_bridge = KernelExperienceBridge(_knowledge_config_for_forge())

    # Re-derive source_type from the file extension when it's unknown: an aiter
    # .cu/.cuh kernel can arrive as "unknown" and be wrongly skipped. A real
    # device-source extension means hip_cpp.
    if (source_type or "").strip().lower() in ("", "unknown") and str(source_file).lower().endswith(
        (".cu", ".cuh", ".hip")
    ):
        source_type = "hip_cpp"
    # Curated kernel_kind refines the kernel backend choice: an aiter CK .cu is best
    # tuned by the ck, not generic HIP; aiter_asm is a prebuilt assembly
    # core the agent cannot rewrite -> skip cleanly.
    kernel_kind = _resolve_kernel_kind(
        source_type,
        str((candidate or {}).get("kernel_kind") or ""),
    )
    if kernel_kind == "aiter_asm":
        return _normalized(
            2,
            "",
            "forge: aiter_asm prebuilt assembly compute-core (.co) is not "
            "editable from source; skipping (no rewritable kernel, no tuner)",
            time.time() - started,
            skipped=True,
        )
    kernel_backend = _resolve_kernel_backend(source_type, kernel_kind)
    log.info(
        "forge dispatch: source_file=%s source_type=%s kernel_kind=%s kernel_backend=%s op=%s",
        source_file,
        source_type,
        kernel_kind or "-",
        kernel_backend,
        (candidate or {}).get("operation", ""),
    )
    if kernel_backend is None:
        return _normalized(
            2,
            "",
            f"forge stage-1 supports triton only; got source_type={source_type}",
            time.time() - started,
            skipped=True,
        )

    branch = _new_forge_branch(output_dir, source_file)

    repo = kernel_repo or _git_toplevel(source_file)
    # Editable-finder packages import the live path via a meta_path finder that
    # PYTHONPATH can't override, so a worktree copy is invisible; edit in place
    # on a temp branch and hard-restore afterward.
    inplace = _needs_inplace(repo)
    restore_info: dict | None = None
    nogit_scratch = False
    try:
        if inplace:
            prep = _prepare_inplace(source_file, repo, branch)
            if prep is None:
                return _normalized(
                    2,
                    "",
                    "forge: editable-finder package but repo is not a usable git checkout; skipping",
                    time.time() - started,
                    skipped=True,
                )
            workspace, worktree_kernel, restore_info = prep
            base_commit = restore_info.get("base_commit") or ""
        else:
            wt_info = _prepare_worktree(source_file, kernel_repo, output_dir, branch)
            if wt_info is None:
                # Non-git source (e.g. pip-installed dist-packages): scaffold an
                # isolated scratch worktree with git init. Disable with
                # FORGE_DISABLE_NOGIT=1.
                if os.environ.get("FORGE_DISABLE_NOGIT", "").strip().lower() in ("1", "true", "yes"):
                    return _normalized(
                        2,
                        "",
                        "forge: kernel_repo is not a clean git checkout or source_file "
                        "not tracked; skipping (live repo untouched; FORGE_DISABLE_NOGIT set)",
                        time.time() - started,
                        skipped=True,
                    )
                wt_info = _prepare_worktree_nogit(source_file, kernel_repo, output_dir, branch)
                if wt_info is None:
                    return _normalized(
                        2,
                        "",
                        "forge: kernel_repo is not a clean git checkout or source_file "
                        "not tracked; skipping (live repo untouched)",
                        time.time() - started,
                        skipped=True,
                    )
                nogit_scratch = True
            workspace, worktree_kernel, base_commit = wt_info
    except (_RetainedWorkspaceCollision, _WorktreePreparationError) as error:
        result = _normalized(
            2,
            "",
            f"forge: workspace preparation skipped safely: {error}",
            time.time() - started,
            skipped=True,
        )
        result["cli_workspace"] = str(output_dir / "worktree")
        result["output_dir"] = str(output_dir)
        return result

    driver = ""
    producer_temporary_paths: list[str] = []
    # Repointed by finalization, which runs in this function's ``finally`` -- before
    # the value reaches the caller, so mutating it there is visible to them.
    finalized_result: dict[str, Any] = {}
    try:
        shapes = _shapes_from_candidate(candidate)
        grouped_cases = task_group_shape_cases(candidate)
        requires_multi_case_driver = len(grouped_cases) > 1
        implementation_sources = _remap_implementation_sources(
            candidate=candidate,
            source_file=source_file,
            workspace=workspace,
            worktree_kernel=worktree_kernel,
            kernel_repo=repo,
        )
        logical_operator = _logical_operator(candidate)
        implementation_symbols = _stable_implementation_symbols(
            candidate,
            invocation_spec_file,
            implementation_sources,
        )
        source_framework = _resolve_framework(candidate, source_file)
        gpu_target = _resolve_gpu_target(candidate)
        gpu_type = _resolve_gpu_type(candidate)
        # Decided before any driver exists: the rewrite route seeds its own.
        rewrite_route = _flydsl_rewrite.evaluate_rewrite_route(
            candidate=candidate,
            source_type=source_type,
            kernel_kind=kernel_kind,
            logical_operator=logical_operator,
            source_kernel=worktree_kernel,
            workspace=workspace,
            implementation_sources=implementation_sources,
            implementation_symbols=implementation_symbols,
            framework=source_framework,
            gpu_target=gpu_target,
            shape_cases=grouped_cases,
            shapes=shapes,
            branch=branch,
            attempt_id=output_dir.name,
            timeout_s=timeout_s,
            invocation_spec_file=invocation_spec_file,
        )
        if not rewrite_route.eligible and rewrite_route.reason != "route_disabled":
            log.info(
                "forge route: FlyDSL rewrite declined for op=%s (%s: %s); using forge-loop",
                logical_operator or worktree_kernel,
                rewrite_route.reason,
                rewrite_route.detail,
            )

        if rewrite_route.eligible:
            driver = _flydsl_rewrite.build_rewrite_driver_seed(
                workspace=workspace,
                writer=_write_generated_driver,
            )
            rewrite_route = rewrite_route.with_driver(driver)
            log.info(
                "forge driver: FlyDSL rewrite seed for op=%s -> %s "
                "(the producer authors the real one from the invocation spec)",
                logical_operator or worktree_kernel,
                driver,
            )
        else:
            # A grouped task must carry every shape before the preparer sees it;
            # a single-shape task has nothing to check.
            if requires_multi_case_driver and not _invocation_spec_covers_cases(
                invocation_spec_file,
                grouped_cases,
            ):
                return _normalized(
                    1,
                    "",
                    "forge: grouped multi-shape invocation spec is missing or incomplete",
                    time.time() - started,
                )
            driver = _write_generated_driver(workspace, _TASK_PREPARER_PLACEHOLDER)
            log.info(
                "forge driver: delegating %d-shape task to forge-loop task-preparer -> %s",
                len(grouped_cases),
                driver,
            )
        # GPU_TARGET is passed via the forge-loop child env (not the parent
        # os.environ, which would leak to sibling ladder backends).
        forge_log = output_dir / "forge_loop.log"
        experiments_dir = output_dir / "forge_experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        snr_threshold = float((candidate.get("targets") or {}).get("snr_db", 30.0))

        # Run the loop in an isolated, hard-killable subprocess so a hung kernel backend
        # can never freeze the orchestrator. KernelBackend stability env defaults are
        # applied inside _run_loop_via_cli, scoped to the child env only.
        # forge-loop rejects --max-hours below its own MIN_MAX_HOURS (1.0) with a
        # click BadParameter (exit 2) that reads like a forge crash and leaves no
        # checkpoint to salvage. Floor the soft budget at that minimum so the
        # campaign always starts; timeout_s still bounds the hard kill, and any
        # KEEP committed before it is recoverable from the checkpoint.
        if timeout_s < _FORGE_MIN_BUDGET_SEC:
            log.warning(
                "forge budget %.0f min is below the %d-min minimum forge-loop "
                "accepts; running with --max-hours %.1f and hard-killing at "
                "%.0f min (raise --budget-minutes to avoid a truncated run)",
                timeout_s / 60.0,
                _FORGE_MIN_BUDGET_SEC // 60,
                _FORGE_MIN_BUDGET_SEC / 3600.0,
                timeout_s / 60.0,
            )
        # Returns before the generic recovery channels below, which are
        # schema-1 forge-loop semantics an apply-back must never take.
        if rewrite_route.eligible:
            rewrite_result, producer_temporary_paths = _run_rewrite_attempt(
                route=rewrite_route,
                workspace=workspace,
                base_commit=base_commit,
                source_file=source_file,
                implementation_sources=implementation_sources,
                kernel_kind=kernel_kind,
                output_dir=output_dir,
                experiments_dir=experiments_dir,
                forge_log=forge_log,
                invocation_spec_file=invocation_spec_file,
                snr_threshold=snr_threshold,
                gpu_type=gpu_type,
                max_hours=max(_FORGE_MIN_BUDGET_SEC / 3600.0, timeout_s / 3600.0),
                deadline_unix=max(time.time() + 1.0, started + timeout_s),
                timeout_s=timeout_s,
                started=started,
            )
            finalized_result = rewrite_result
            return rewrite_result

        loop_outcome = _run_loop_via_cli(
            worktree_kernel=worktree_kernel,
            driver=driver,
            workspace=workspace,
            snr_threshold=snr_threshold,
            max_hours=max(_FORGE_MIN_BUDGET_SEC / 3600.0, timeout_s / 3600.0),
            branch=branch,
            gpu_target=gpu_target,
            gpu_type=gpu_type,
            kernel_backend=kernel_backend,
            program_md_file=str(prompt_file),
            invocation_spec_file=invocation_spec_file,
            experiments_dir=experiments_dir,
            forge_log=forge_log,
            timeout_s=timeout_s,
            deadline_unix=max(
                time.time() + 1.0,
                started + timeout_s,
            ),
            operator_name=logical_operator,
            experience_id=output_dir.name,
            framework=source_framework,
            target_functions=implementation_symbols,
            source_files=implementation_sources,
        )
        # keep/revert is decided from forge's own published best, in descending
        # order of trust:
        #   1. best_result.json -- rewritten atomically on every KEEP, gated on
        #      correctness, and pointing at a commit already in the history. It
        #      is current whether the loop finished, exhausted its soft budget,
        #      or was hard-killed, so it is the authoritative record.
        #   2. the caller-owned checkpoint -- same guarantees, but routed through
        #      --experiments-dir and only as fresh as the last KEEP callback.
        #   3. the final-result sidecar / stdout sentinel -- only produced on a
        #      graceful return, and never sufficient on its own after a kill.
        raw_published = _read_forge_best_result(workspace)
        published = _validated_forge_best_result(
            raw_published,
            workspace=workspace,
            base_commit=base_commit,
        )
        checkpoint_recovery = _validated_forge_checkpoint(
            loop_outcome.checkpoint,
            workspace=workspace,
            base_commit=base_commit,
            shapes=shapes,
        )
        warm_start_recovery = _validated_warm_start_result(
            loop_outcome.structured_result,
            workspace=workspace,
            base_commit=base_commit,
        )
        recovery = published or checkpoint_recovery or warm_start_recovery
        if loop_outcome.error is not None and recovery is None:
            failure_detail = (
                "forge cli loop timed out without recoverable checkpoint"
                if loop_outcome.timed_out
                else "forge cli loop failed without validated recovery"
            )
            failed = _normalized(
                1,
                "",
                f"{failure_detail}: {loop_outcome.error}",
                time.time() - started,
            )
            failed["timed_out"] = loop_outcome.timed_out
            failed["salvaged"] = False
            failed["output_dir"] = str(output_dir)
            if loop_outcome.structured_result is not None:
                failed["forge_result"] = loop_outcome.structured_result
                kb_experience = loop_outcome.structured_result.get("kb_experience")
                if isinstance(kb_experience, dict):
                    failed["kb_experience"] = kb_experience
            return failed
        baseline_ms = (
            loop_outcome.pristine_baseline_ms
            if loop_outcome.pristine_baseline_ms is not None
            else loop_outcome.baseline_ms
        )
        search_start_ms = (
            loop_outcome.search_start_ms if loop_outcome.search_start_ms is not None else loop_outcome.baseline_ms
        )
        best_ms = loop_outcome.best_ms
        mean_case_speedup = loop_outcome.mean_case_speedup
        search_start_mean_case_speedup = loop_outcome.search_start_mean_case_speedup
        improved = loop_outcome.total_improved
        improved_during_search = loop_outcome.incremental_improved
        best_commit = ""
        if recovery is not None:
            if loop_outcome.pristine_baseline_ms is None:
                baseline_ms = recovery["baseline_ms"]
            best_ms = recovery["best_ms"]
            mean_case_speedup = recovery["mean_case_speedup"]
            search_start_mean_case_speedup = recovery["search_start_mean_case_speedup"]
            improved = recovery["total_improved"]
            improved_during_search = recovery["incremental_improved"]
            best_commit = recovery["best_commit"]
        salvaged = bool(loop_outcome.error and recovery is not None)
        if published is not None and checkpoint_recovery is not None:
            # Both channels are validated; disagreement means one is stale. The
            # published manifest wins (it is rewritten per KEEP), but surface it
            # -- a persistent mismatch is a forge-side bug, not noise.
            if published["best_commit"] != checkpoint_recovery["best_commit"]:
                log.warning(
                    "forge best_result.json (%s) and checkpoint (%s) disagree; keeping the published manifest",
                    published["best_commit"][:12],
                    checkpoint_recovery["best_commit"][:12],
                )
        changed_files: list[str] = []
        _, changed_files = _export_best_artifacts(
            workspace,
            base_commit,
            worktree_kernel,
            source_file,
            output_dir,
            best_commit=best_commit,
        )
        _write_changed_files_index(output_dir, changed_files)
        _write_report(
            output_dir,
            baseline_ms,
            best_ms,
            improved,
            mean_case_speedup=mean_case_speedup,
            search_start_ms=search_start_ms,
            improved_during_search=improved_during_search,
        )
        knowledge_status = knowledge_bridge.status
        msg = (
            f"forge done (cli): pristine_baseline={baseline_ms} "
            f"search_start={search_start_ms} best={best_ms} "
            f"mean_case_speedup={mean_case_speedup} improved={improved} "
            f"improved_during_search={improved_during_search} "
            f"kernel_backend={kernel_backend} gpu={gpu_target} "
            f"knowledge={knowledge_status.mode}/{knowledge_status.backend} "
            f"salvaged={'yes' if salvaged else 'no'}"
        )
        # Surface the run's LLM token spend + key-step timeline from the CLI
        # sidecar as the canonical markers (FORGE_LLM_USAGE / FORGE_STEPS) so
        # the tracer can attribute forge's cost + decision process.
        forge_usage, forge_steps = _forge_trace_from_sidecar(output_dir)
        if forge_usage:
            import json as _json_usage

            msg += "\nFORGE_LLM_USAGE " + _json_usage.dumps(forge_usage, sort_keys=True)
        if forge_steps:
            import json as _json_steps

            msg += "\nFORGE_STEPS " + _json_steps.dumps(forge_steps, sort_keys=True)
        res = _normalized(
            0,
            msg + "\n" + (loop_outcome.output or "")[-3000:],
            "",
            time.time() - started,
        )
        if forge_usage:
            res["llm_usage"] = forge_usage
        if forge_steps:
            res["steps"] = forge_steps
        res["cli_workspace"] = str(output_dir)
        res["output_dir"] = str(output_dir)
        res["timed_out"] = loop_outcome.timed_out
        res["salvaged"] = salvaged
        res["pristine_baseline_ms"] = baseline_ms
        res["search_start_ms"] = search_start_ms
        res["best_ms"] = best_ms
        res["mean_case_speedup"] = mean_case_speedup
        res["search_start_mean_case_speedup"] = search_start_mean_case_speedup
        res["improved"] = improved
        res["total_improved"] = improved
        res["incremental_improved"] = improved_during_search
        res["improved_during_search"] = improved_during_search
        canonical_artifacts = _canonical_forge_artifacts(
            workspace,
            raw_published if published is not None else None,
        )
        if canonical_artifacts:
            res.update(canonical_artifacts)
            res["artifacts"] = [str(canonical_artifacts["canonical_patch_path"])]
        if loop_outcome.structured_result is not None:
            res["forge_result"] = loop_outcome.structured_result
            kb_experience = loop_outcome.structured_result.get("kb_experience")
            if isinstance(kb_experience, dict):
                res["kb_experience"] = kb_experience
        kernel_experience = knowledge_bridge.collect_result(loop_outcome.structured_result)
        res["kernel_experience"] = kernel_experience
        res["knowledge_audit"] = [
            {
                "op": "kernel_experience_passthrough",
                "mode": knowledge_status.mode,
                "backend": knowledge_status.backend,
                "resolution": "collected" if kernel_experience["result_present"] else "not_reported",
                "success": True,
                "provenance": kernel_experience["provenance"],
            }
        ]
        res["logical_operator"] = logical_operator
        res["source_framework"] = source_framework
        if rewrite_route.reason != "route_disabled":
            res["flydsl_rewrite"] = rewrite_route.as_dict()
        res["implementation_sources"] = implementation_sources
        res["kernel_kind"] = kernel_kind
        res["target_functions"] = implementation_symbols
        if recovery is not None:
            res["best_commit"] = recovery["best_commit"]
            res["checkpoint_path"] = str(experiments_dir / f"{_FORGE_EXPERIMENT_ID}.json")
        finalized_result = res
        return res
    except Exception as exc:  # noqa: BLE001
        return _normalized(1, "", f"forge submit failed: {type(exc).__name__}: {exc}", time.time() - started)
    finally:
        # Never let workspace cleanup failure swallow the forge result dict.
        try:
            _finalize_forge_workspace(
                inplace=inplace,
                restore_info=restore_info,
                driver=driver,
                workspace=workspace,
                output_dir=output_dir,
                branch=branch,
                nogit_scratch=nogit_scratch,
                temporary_paths=producer_temporary_paths,
                result=finalized_result,
            )
        except Exception:
            log.exception("forge workspace finalization failed")


class _NominationStagingError(RuntimeError):
    """The brief names no candidate that a workspace could be staged from."""


@dataclass(frozen=True)
class _StagedBrief:
    """Where a nomination will run, and what has to be undone afterwards.

    ``workspace`` equals ``live_repo`` in place, so the path rewriting and the
    inverse mapping both collapse to identity and only the restore differs.
    """

    workspace: str
    live_repo: str
    staged_request: str
    base_commit: str
    offtree_rows: int
    inplace: bool = False
    restore_info: dict | None = None
    nogit_scratch: bool = False


def _nominatable_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows the nominator could actually pick, hottest first.

    Mirrors the stub's own eligibility -- a row with no source file, or one this
    session already rejected, can never be selected -- so a row that could never
    be picked cannot decide which tree gets staged.
    """
    rows = manifest.get("hot_kernels")
    rows = rows if isinstance(rows, list) else []
    eligible = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("source_file") or "").strip() and not row.get("rejected")
    ]
    eligible.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
    return eligible


def _stage_nomination_brief(
    *,
    nomination_input: str,
    output_dir: Path,
    branch: str,
) -> _StagedBrief:
    """Stage a workspace the nominator can pick inside, and re-point the brief.

    forge resolves the nominated kernel against ``--workspace`` and refuses
    anything outside it, so a brief listing paths in the live install tree can
    never yield a runnable target. A worktree of that tree is staged instead --
    the live tree is never edited -- and every candidate path is rewritten into
    it, so whichever row the nominator picks is already inside the workspace.

    The tree is the hottest eligible row's, which is the row the nominator ranks
    first. Rows from any other tree are dropped from the staged brief and
    counted: one worktree cannot hold two trees, and leaving them in would offer
    the nominator targets that must fail the containment check.

    Args:
        nomination_input: Path to the request JSON written by Hyperloom.
        output_dir: Per-attempt directory the worktree is created under.
        branch: Branch name for the staged worktree.

    An editable install is the exception: its finder imports the live tree, so a
    copy is invisible to the campaign's own measurement and the loop would no-op.
    That tree is edited in place instead, under the same lock and restore contract
    the named-kernel path uses, and the caller must replay the restore.

    Returns:
        The staged brief, whose ``live_repo`` is the tree the re-baselined server
        imports and the one patch targets must name.

    Raises:
        _NominationStagingError: When no eligible row resolves into a stageable
            tree, so no workspace could make any nomination runnable, or when an
            editable tree could not be prepared in place.
    """
    request_path = Path(nomination_input).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(request.get("candidates_path") or "")).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    eligible = _nominatable_rows(manifest)
    if not eligible:
        raise _NominationStagingError("the nomination brief lists no candidate with a resolved source file")

    hottest = eligible[0]
    source_file = str(hottest.get("source_file") or "").strip()
    repo = str(hottest.get("kernel_repo") or "").strip() or _git_toplevel(source_file)
    if not repo:
        raise _NominationStagingError(f"could not resolve a source tree to stage from {source_file}")
    # An editable install imports the live tree, so a copy is invisible to the
    # campaign's own measurement: edit in place and restore on the way out.
    inplace = _needs_inplace(repo)
    restore_info: dict | None = None
    nogit_scratch = False
    if inplace:
        prepared = _prepare_inplace(source_file, repo, branch)
        if prepared is None:
            raise _NominationStagingError(
                f"{repo} is an editable install that could not be prepared in place; another forge "
                "run may hold it, or it is not a usable git checkout"
            )
        workspace, _staged_kernel, restore_info = prepared
        # The dirty-baseline commit when there was one, else the original HEAD.
        base_commit = str(restore_info.get("base_commit") or "")
    else:
        staged = _prepare_worktree(source_file, repo, output_dir, branch)
        if staged is None:
            staged = _prepare_worktree_nogit(source_file, repo, output_dir, branch)
            nogit_scratch = staged is not None
        if staged is None:
            raise _NominationStagingError(f"{repo} is neither a usable git checkout nor a copyable source tree")
        workspace, _staged_kernel, base_commit = staged

    repo_root = Path(repo).resolve()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in manifest.get("hot_kernels") or []:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("source_file") or "").strip()
        if not raw:
            # Unresolved rows carry no path to rewrite and are the whole point of
            # handing the full list over, so they travel untouched.
            kept.append(row)
            continue
        try:
            relative = Path(raw).resolve().relative_to(repo_root)
        except ValueError:
            dropped += 1
            continue
        kept.append({**row, "source_file": str(Path(workspace) / relative), "kernel_repo": workspace})

    staged_manifest = output_dir / "forge_candidate_manifest.staged.json"
    staged_manifest.write_text(json.dumps({**manifest, "hot_kernels": kept}), encoding="utf-8")
    staged_request = output_dir / "forge_nomination_input.staged.json"
    staged_request.write_text(json.dumps({**request, "candidates_path": str(staged_manifest)}), encoding="utf-8")
    return _StagedBrief(
        workspace=workspace,
        live_repo=str(repo_root),
        staged_request=str(staged_request),
        base_commit=base_commit,
        offtree_rows=dropped,
        inplace=inplace,
        restore_info=restore_info,
        nogit_scratch=nogit_scratch,
    )


def _reanchor_patch_targets(envelope: Any, *, workspace: str, live_repo: str) -> Any:
    """Map a patch's targets from the staged workspace onto the live tree.

    Staging rewrites candidate paths into a worktree, so the producer answers
    with worktree paths. Applied as given they land in the copy, which nothing
    imports: the re-baselined server measures unmodified code and a real gain
    reads as none. Paths outside the workspace travel untouched.

    Only the fields naming a file the patch is applied to move. The diff and the
    snapshot are producer artifacts and must stay where they were written.

    Args:
        envelope: The raw ``--auto`` result envelope.
        workspace: The staged workspace the producer worked in.
        live_repo: The tree the workspace was copied from.

    Returns:
        The envelope with every patch target and repo root named in the live tree.
    """
    if not isinstance(envelope, dict):
        return envelope
    rows = envelope.get("patches")
    if not isinstance(rows, list):
        return envelope
    staged = Path(workspace).resolve()
    live = Path(live_repo).resolve()

    def _mapped(raw: str) -> str:
        try:
            relative = Path(raw).resolve().relative_to(staged)
        except (ValueError, OSError):
            return raw
        return str(live / relative)

    remapped: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            remapped.append(row)
            continue
        updated = dict(row)
        target = str(updated.get("target_file") or "").strip()
        if target:
            updated["target_file"] = _mapped(target)
        paths = updated.get("write_paths")
        if isinstance(paths, list):
            updated["write_paths"] = [_mapped(str(p)) if str(p).strip() else p for p in paths]
        if str(updated.get("kernel_repo") or "").strip():
            updated["kernel_repo"] = str(live)
        remapped.append(updated)
    return {**envelope, "patches": remapped}


def submit_auto(
    *,
    nomination_input: str,
    output_dir: Path,
    timeout_s: int = 1800,
    gpu_target: str = "",
    gpu_type: str = "",
    kernel_backend: str = "",
) -> dict:
    """Run forge's ``--auto`` self-nomination loop and return its raw envelope.

    Sibling of :func:`submit`, not a variant of it. The named-kernel path is
    predicated from its first lines on a ``source_file`` + ``candidate`` and a
    prepared worktree, so ``--auto`` -- which hands forge no kernel and lets it
    pick from the nomination -- cannot flow through it. This entry instead builds
    the ``forge-loop`` argv directly, OMITTING ``--kernel``/``--driver``/
    ``--source-files`` (forge derives the target from the nomination), and adding
    ``--auto --nomination-input <path>``. The two ends agree only by the static
    ``PROTOCOL_VERSION`` the request carries; there is no capability probe.

    Containment mirrors :func:`_run_loop_via_cli` -- child env, isolated process
    group, absolute deadline, escalating termination, ``__FORGE_RESULT__`` /
    ``--result-json`` parsing. Unlike ``submit``/``_run_loop_via_cli`` it returns
    the parsed forge envelope UNCHANGED (``{"patches": [...], "nomination": {...},
    "improved": bool}``) rather than collapsing it to a scalar
    ``ForgeLoopOutcome``: the handler needs every sibling patch, so nothing may be
    lost in a single-best normalization.

    The workspace is deliberately not an argument. A campaign edits the tree it
    is handed, so this entry stages its own worktree of whatever tree the
    candidates live in rather than letting a caller aim it at the live install.

    Args:
        nomination_input: Path to the ``forge_nomination_input.json`` request.
        output_dir: Where diagnostics/checkpoints and the result sidecar live.
        timeout_s: Wall-clock budget; the child is hard-killed at the deadline.
        gpu_target: gfx arch override; resolved from env/probe when empty.
        gpu_type: Hardware model override; resolved from env when empty.
        kernel_backend: Kernel-backend override; forge picks its own when empty.

    Returns:
        The forge result envelope on success, or a minimal
        ``{"status": "failed"|"timeout", "patches": [], "error": ...}`` when the
        child never produced one.
    """
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = output_dir / "forge_experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    forge_log = output_dir / "forge_loop.log"
    result_json = experiments_dir.parent / "forge_cli_result.json"
    try:
        result_json.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"could not clear stale forge result artifact {result_json}: {exc}") from exc

    resolved_target = (gpu_target or "").strip() or _resolve_gpu_target({})
    resolved_type = (gpu_type or "").strip() or _resolve_gpu_type({})
    branch = _new_forge_branch(output_dir, "auto-nomination")
    deadline_unix = max(time.time() + 1.0, started + timeout_s)

    # A fresh campaign resolves both a kernel and a driver against the workspace;
    # --auto names neither, so both are arranged here before forge launches.
    try:
        # forge reports the commit each patch is diffed against on the envelope,
        # so the staged base is not needed again here.
        brief = _stage_nomination_brief(
            nomination_input=nomination_input,
            output_dir=output_dir,
            branch=branch,
        )
    except (
        _NominationStagingError,
        _RetainedWorkspaceCollision,
        _WorktreePreparationError,
        OSError,
        ValueError,
    ) as exc:
        return {
            "status": "failed",
            "patches": [],
            "error": f"forge --auto workspace staging failed: {exc}",
        }
    workspace = brief.workspace
    if brief.offtree_rows:
        log.warning(
            "forge --auto: %d candidate row(s) live outside the staged tree %s and were not offered",
            brief.offtree_rows,
            workspace,
        )
    # forge-loop requires --driver to exist before its task preparer authors the
    # real one; the placeholder fails loudly rather than passing for a driver.
    driver = _write_generated_driver(workspace, _TASK_PREPARER_PLACEHOLDER)
    try:
        return _run_auto_campaign(
            brief=brief,
            driver=driver,
            output_dir=output_dir,
            experiments_dir=experiments_dir,
            result_json=result_json,
            forge_log=forge_log,
            branch=branch,
            deadline_unix=deadline_unix,
            timeout_s=timeout_s,
            resolved_target=resolved_target,
            resolved_type=resolved_type,
            kernel_backend=kernel_backend,
        )
    finally:
        # In place this restores the live tree; a worktree is retained because
        # the patches it produced still reference paths inside it.
        try:
            _finalize_forge_workspace(
                inplace=brief.inplace,
                restore_info=brief.restore_info,
                driver=driver,
                workspace=workspace,
                output_dir=output_dir,
                branch=branch,
                nogit_scratch=brief.nogit_scratch,
            )
        except Exception:
            log.exception("forge --auto workspace finalization failed")


def _run_auto_campaign(
    *,
    brief: _StagedBrief,
    driver: str,
    output_dir: Path,
    experiments_dir: Path,
    result_json: Path,
    forge_log: Path,
    branch: str,
    deadline_unix: float,
    timeout_s: int,
    resolved_target: str,
    resolved_type: str,
    kernel_backend: str,
) -> dict:
    """Launch ``forge-loop --auto`` in a staged workspace and read its envelope.

    Split from :func:`submit_auto` so every exit runs the staging teardown: in
    place the live tree stays edited until the restore replays.

    Args:
        brief: The staged workspace and the live tree it stands for.
        driver: The placeholder driver already allocated inside the workspace.
        output_dir: Per-attempt directory holding the campaign and its logs.
        experiments_dir: Campaign directory handed to ``--experiments-dir``.
        result_json: Sidecar the child writes its envelope to.
        forge_log: Log file the child's output is appended to.
        branch: Branch name the campaign commits on.
        deadline_unix: Absolute instant the child is killed at.
        timeout_s: Budget reported in the timeout error.
        resolved_target: GPU target passed through to forge.
        resolved_type: GPU model passed through to forge.
        kernel_backend: Optional kernel-backend override.

    Returns:
        The envelope, with patch targets mapped onto the live tree.
    """
    workspace = brief.workspace

    env = dict(os.environ)
    env["GPU_TARGET"] = resolved_target
    _apply_gpu_type_env(env, resolved_type)
    _apply_kernel_backend_env(env)
    env.setdefault("GIT_AUTHOR_NAME", "forge-bot")
    env.setdefault("GIT_AUTHOR_EMAIL", "forge-bot@local")
    env.setdefault("GIT_COMMITTER_NAME", "forge-bot")
    env.setdefault("GIT_COMMITTER_EMAIL", "forge-bot@local")

    cmd = [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "forge-loop",
        "--auto",
        "--nomination-input",
        str(Path(brief.staged_request).resolve()),
        "--workspace",
        workspace,
        "--driver",
        driver,
        "--max-hours",
        str(max(_FORGE_MIN_BUDGET_SEC / 3600.0, timeout_s / 3600.0)),
        "--git-branch",
        branch,
        "--gpu-target",
        resolved_target,
        "--gpu-type",
        _known_gpu_model(resolved_type),
        "--experiments-dir",
        str(experiments_dir),
        "--experiment-id",
        _FORGE_EXPERIMENT_ID,
        "--experience-id",
        output_dir.name,
        "--deadline-unix",
        str(deadline_unix),
        "--result-json",
        str(result_json),
    ]
    if kernel_backend.strip():
        cmd += ["--kernel-backend", kernel_backend.strip()]

    from hyperloom.common.llm_config import resolve_forge_llm_model

    if _openai_only_provider():
        cmd += ["--agent-backend", "codex", "--agent-fallback-provider", "none"]
        forge_model = resolve_forge_llm_model("codex")
    else:
        forge_model = resolve_forge_llm_model("claude")
    if forge_model:
        cmd += ["--model", forge_model]

    loop_exc: Exception | None = None
    out = ""
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=workspace,
            start_new_session=True,
        )
        try:
            remaining = max(1.0, deadline_unix - time.time())
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = _terminate_forge_process(proc)
        out = (stdout or "") + "\n" + (stderr or "")
        if timed_out:
            loop_exc = RuntimeError(f"forge-loop --auto exceeded absolute deadline after {timeout_s}s")
        elif proc.returncode != 0:
            loop_exc = RuntimeError(f"forge-loop --auto exited rc={proc.returncode}: {_forge_failure_tail(out)}")
    except Exception as exc:  # noqa: BLE001
        loop_exc = exc

    try:
        with open(forge_log, "a") as f:
            f.write("\n=== forge-loop --auto (cli) stdout ===\n")
            f.write(out)
            if loop_exc:
                f.write(f"\n=== forge-loop --auto exception ===\n{loop_exc}\n")
    except OSError:  # noqa: S110
        pass

    parsed = None
    try:
        if result_json.exists():
            parsed = json.loads(result_json.read_text())
    except Exception:
        parsed = None
    if parsed is None and "__FORGE_RESULT__" in out:
        try:
            parsed = json.loads(out.split("__FORGE_RESULT__")[1])
        except Exception:
            parsed = None
    if loop_exc is not None:
        # Only exit code 0 makes an empty ``patches`` a valid answer, so the
        # process outcome decides and the sidecar may only enrich the error.
        envelope: dict[str, Any] = {
            "status": "timeout" if timed_out else "failed",
            "patches": [],
            "error": str(loop_exc),
        }
        if isinstance(parsed, dict):
            detail = str(parsed.get("error") or "").strip()
            if detail:
                envelope["error"] = f"{envelope['error']}: {detail}"
            summary = parsed.get("nomination")
            if isinstance(summary, dict):
                envelope["nomination"] = summary
        return envelope
    if isinstance(parsed, dict):
        # forge owns the envelope's shape; only the staged paths are rewritten,
        # since the tree it worked in is not the tree the server imports.
        return _reanchor_patch_targets(parsed, workspace=workspace, live_repo=brief.live_repo)
    return {
        "status": "failed",
        "patches": [],
        "error": "forge-loop --auto produced no result",
    }
