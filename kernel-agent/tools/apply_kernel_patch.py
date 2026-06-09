#!/usr/bin/env python3
"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
PYTHON_SOURCE_SUFFIXES = {".py"}
COMPILED_ARTIFACT_SUFFIXES = {".so", ".co", ".hsaco"}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}
# Fallback when ``inference_optimizer`` is not on ``sys.path`` (standalone CLI).
_FALLBACK_KNOWN_TARGET_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
)

_CACHED_KNOWN_TARGET_ROOTS: tuple[str, ...] | None = None


def known_target_roots() -> tuple[str, ...]:
    """Resolved framework roots (importlib/glob when orchestrator is importable)."""
    global _CACHED_KNOWN_TARGET_ROOTS
    if _CACHED_KNOWN_TARGET_ROOTS is not None:
        return _CACHED_KNOWN_TARGET_ROOTS
    try:
        from inference_optimizer.orchestrator.framework_paths import (
            resolve_patch_target_roots,
        )

        _CACHED_KNOWN_TARGET_ROOTS = resolve_patch_target_roots()
    except ImportError:
        _CACHED_KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS
    return _CACHED_KNOWN_TARGET_ROOTS


# Backward-compat alias for tests / external imports.
KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS


# Default location of the multi-node patch backup directory on the
# RayJob pod. Mirrors the apply_kernel_patch.backup_root convention
# (per-target subdir) but lives on the pod's local fs, not the
# sandbox's, so it survives sglang restarts without depending on
# wekafs. Overridable via $HYPERLOOM_MN_KERNEL_BACKUP_DIR for tests.
_MN_POD_BACKUP_DIR_DEFAULT = "/var/kernel_patch_backups"

# State file written by ``inference_optimizer.multi_node create-rayjob``.
# Presence of ``nodes >= 2`` is the multi-node signal used to decide
# whether apply_kernel_patch.py should fan-out the patch to RayJob pods
# via the multi_node CLI in addition to writing the sandbox-local copy.
#
# Resolution mirrors ``inference_optimizer.orchestrator.action_executors
# ._multi_node_env._state_path``: ``$MULTI_NODE_STATE_FILE`` wins,
# default ``/tmp/multi_node_state.json``. Honouring the env var keeps
# test runs isolated — pytest can point this at a non-existent path
# so the fan-out branch is never taken, even on a sandbox whose
# hardcoded ``/tmp`` file is left over from a prior real multi-node
# session (without this override, an active inference_optimizer's
# ``/tmp/multi_node_state.json`` would silently turn ``test_p2_4``
# integrate fixtures into multi-node fan-out attempts that
# mock-mismatch ``subprocess.run``).
_MN_STATE_FILE_DEFAULT = "/tmp/multi_node_state.json"


def _mn_state_path() -> Path:
    """Resolve where ``inference_optimizer.multi_node`` dropped its state."""
    return Path(os.environ.get("MULTI_NODE_STATE_FILE", _MN_STATE_FILE_DEFAULT))


# Legacy module attribute. Kept for any caller / test that imports
# ``_MN_STATE_FILE`` directly; runtime checks go through
# :func:`_mn_state_path` so each call re-resolves the env override.
_MN_STATE_FILE = Path(_MN_STATE_FILE_DEFAULT)


def _is_multi_node() -> bool:
    """True iff a multi-node RayJob is active (nodes >= 2).

    Reads ``$MULTI_NODE_STATE_FILE`` (default
    ``/tmp/multi_node_state.json``) — the same checkpoint
    ``inference_optimizer.multi_node.cli`` writes after
    ``create-rayjob``. Missing file / unreadable / ``nodes < 2`` →
    ``False``, so single-node and standalone CLI use of this tool
    keep their pre-multinode behaviour bit-for-bit.
    """
    state_path = _mn_state_path()
    try:
        if not state_path.is_file():
            return False
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return int(data.get("nodes") or 0) >= 2
    except (OSError, ValueError):
        return False


def _dispatch_multinode_apply(
    *,
    target_file: Path,
    patch_path: Path,
    kernel_id: str,
    backup_dir_on_pod: str,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Run ``python3 -m inference_optimizer.multi_node apply-patch`` to
    fan the same patch out to every pod (head + workers).

    Returns the parsed JSON document produced by
    kernel_patch_multinode.py — caller checks ``status == "ok"`` and
    persists ``per_node`` (host → backup_path map) into the
    apply-kernel-patch manifest so revert can reach the same pods.

    Raises RuntimeError on subprocess failure / non-JSON output / pod
    status != ok so the apply_kernel_patch caller can roll back the
    sandbox-local copy.
    """
    cmd = [
        sys.executable, "-m", "inference_optimizer.multi_node",
        "apply-patch",
        "--patch-file", str(patch_path),
        "--target-path", str(target_file),
        "--backup-dir", backup_dir_on_pod,
        "--kernel-id", kernel_id or "",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"multi-node apply-patch returned rc={proc.returncode}: "
            f"stderr={(proc.stderr or '')[-2000:]!r}"
        )
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1]) \
            if proc.stdout.strip().startswith("{") is False \
            else json.loads(proc.stdout)
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"multi-node apply-patch stdout not JSON: {exc!r}; "
            f"stdout_tail={(proc.stdout or '')[-2000:]!r}"
        ) from exc
    if str(parsed.get("status", "")).lower() != "ok":
        raise RuntimeError(
            f"multi-node apply-patch reported status={parsed.get('status')!r}: "
            f"failures={parsed.get('failures')!r}"
        )
    return parsed


def _dispatch_multinode_revert(
    *,
    target_path: str,
    backup_map: dict[str, str],
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Run ``python3 -m inference_optimizer.multi_node revert-patch`` to
    restore the original file on every pod that received the apply.

    Best-effort: a partial revert is logged but not raised — the
    caller (revert_kernel_patch) has already restored the sandbox
    copy, and re-running revert is idempotent on noop_missing_backup.
    """
    cmd = [
        sys.executable, "-m", "inference_optimizer.multi_node",
        "revert-patch",
        "--target-path", str(target_path),
        "--backup-map-json", json.dumps(backup_map, sort_keys=True),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec,
    )
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out.startswith("{") else {}
    except json.JSONDecodeError:
        parsed = {}
    if proc.returncode != 0 or str(parsed.get("status", "")).lower() != "ok":
        # Don't raise — sandbox revert already won; warn-only so caller
        # can mark the manifest reverted regardless.
        sys.stderr.write(
            f"WARN multi-node revert-patch rc={proc.returncode} "
            f"status={parsed.get('status')!r} "
            f"stderr_tail={(proc.stderr or '')[-1000:]!r}\n"
        )
    return parsed or {
        "status": "partial",
        "returncode": proc.returncode,
        "stdout_tail": out[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _now() -> str:
    """Return the current UTC timestamp as an ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Sanitize a filename component to a safe, short identifier."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel"


def _path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    """Copy a file into the backup tree and return the manifest entry."""
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


def _source_text_looks_complete(text: str, suffix: str) -> bool:
    """Heuristically check that the patch text is a full source file."""
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(
            marker in stripped
            for marker in ("def ", "class ", "import ", "@triton.jit", "torch.")
        )
    if suffix in COMPILED_SOURCE_SUFFIXES:
        return any(
            marker in stripped
            for marker in (
                "#include", "__global__", "__device__", "extern ", "namespace ",
                "template", "void ", "int ", "float ", "half", "torch::",
            )
        )
    return False


def _validate_patch_source(patch: Path, target: Path) -> None:
    patch_suffix = patch.suffix.lower()
    target_suffix = target.suffix.lower()
    if patch_suffix in TEXT_ARTIFACT_SUFFIXES:
        raise ValueError(f"patch_path is not a complete source file: {patch}")
    if patch_suffix != target_suffix:
        raise ValueError(
            f"patch suffix {patch_suffix or '<none>'} does not match "
            f"target suffix {target_suffix or '<none>'}"
        )
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise ValueError(f"patch_path is not text source: {patch}") from exc
    if not _source_text_looks_complete(text, target_suffix):
        raise ValueError(f"patch_path does not look like a complete {target_suffix} source file: {patch}")
    target_text = target.read_text(encoding="utf-8", errors="replace")
    _validate_replacement_compatibility(text, target_text, target)
    if target_suffix == ".py":
        py_compile.compile(str(patch), doraise=True)


def _host_entry_functions(source_text: str) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(
        r"(?m)^\s*(?!__global__)(?:template\s*<[^>]+>\s*)?"
        r"(?:void|int|bool|float|double|auto|at::Tensor|torch::Tensor)\s+"
        r"([A-Za-z_]\w*)\s*\("
    )
    for match in pattern.finditer(source_text):
        name = match.group(1)
        if name.startswith("_") or name in {"main"}:
            continue
        names.add(name)
    return names


def _validate_replacement_compatibility(patch_text: str, target_text: str, target: Path) -> None:
    if "PYBIND11_MODULE" in patch_text and "PYBIND11_MODULE" not in target_text:
        raise ValueError(
            "patch creates a standalone PYBIND11 module but target is a framework source file"
        )
    if "TORCH_LIBRARY" in patch_text and "TORCH_LIBRARY" not in target_text:
        raise ValueError(
            "patch creates standalone TORCH_LIBRARY registration absent from target"
        )
    if "namespace aiter" in target_text and "namespace aiter" not in patch_text:
        raise ValueError("patch does not preserve namespace aiter")

    required = _host_entry_functions(target_text)
    if required:
        present = _host_entry_functions(patch_text)
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"patch does not preserve target host entry function(s) for {target}: "
                + ", ".join(missing[:12])
            )


def _clear_python_kernel_caches(target: Path) -> dict[str, Any]:
    removed: list[str] = []

    def remove_path(path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed.append(str(path))
        except OSError:
            return

    for cache_dir in target.parent.rglob("__pycache__"):
        remove_path(cache_dir)

    home = Path(os.environ.get("HOME", "~")).expanduser()
    for path in (
        home / ".triton" / "cache",
        home / ".cache" / "triton",
        home / ".cache" / "torch" / "inductor",
    ):
        if path.exists():
            remove_path(path)
    for pattern in ("/tmp/torchinductor_*", "/tmp/triton_*"):
        for path in Path("/tmp").glob(Path(pattern).name):
            if path.exists():
                remove_path(path)
    return {"status": "ok", "removed": removed}


# ---------------------------------------------------------------------------
# PR-K: aiter JIT cache invalidation around compiled-source rebuilds.
#
# aiter ships ``@compile_ops("module_<name>", gen_func=...)`` decorators that
# JIT-codegen + hipcc-compile per-instance ``.so`` files into
# ``<aiter>/jit/build/module_<name>_<sig>/``. ``setup.py develop`` rebuilds the
# python package + statically-compiled ``.so`` but does NOT invalidate the
# jit/build entries: a patch under ``aiter/csrc/ck_gemm_moe_2stages_codegen/``
# would rebuild the wheel yet the next ``import aiter.ops.moe_op`` would still
# pick up the pre-patch ``module_moe_ck2stages_*.so`` from jit/build/, leaving
# the integrate benchmark to measure unchanged performance and emit REVERT.
#
# We move (NOT copy) the entire jit/build/ directory aside before the rebuild
# step so the post-rebuild first-import re-codegens + re-compiles every module
# from clean state. ``shutil.move`` is atomic on the same filesystem and zero-
# copy. Revert moves the backup back, removing any regenerated jit/build/ dir
# first so the pre-patch state is restored bit-for-bit.
#
# Scope: ONLY aiter is affected. sglang's sgl-kernel and vllm have no JIT
# codegen layer — their ``.so`` are produced by setup.py at install time, so
# the standard ``setup.py develop`` rebuild + cache_clear is sufficient and
# this invalidation step is a no-op for those targets.
# ---------------------------------------------------------------------------
_AITER_CSRC_MARKER = "/aiter/csrc/"


def _target_is_in_aiter_csrc(target_file: Path) -> bool:
    """Return True iff ``target_file`` resides under any ``aiter/csrc/`` tree.

    Matches both the editable checkout (``/sgl-workspace/aiter/csrc/...``) and
    the dist-packages layout (``/usr/local/lib/python3.10/dist-packages/aiter/
    csrc/...``) — in both cases the relative segment ``aiter/csrc/`` appears
    verbatim in the absolute path.
    """
    return _AITER_CSRC_MARKER in str(target_file).replace(os.sep, "/")


def _aiter_jit_build_dir() -> Path | None:
    """Return ``<aiter>/jit/build`` for the importable aiter, or ``None``.

    Resolved via ``importlib.util.find_spec("aiter")`` so editable installs
    (``/sgl-workspace/aiter/aiter/__init__.py``), wheel installs
    (``/usr/local/lib/python3.12/dist-packages/aiter/__init__.py``) and any
    other layout on ``sys.path`` resolve correctly without hardcoding.

    Returns ``None`` when aiter is not importable in the current interpreter
    (the kernel-agent sandbox container always has aiter available; this
    fallback exists so unit tests on a host without aiter still pass).
    """
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    aiter_pkg = Path(list(spec.submodule_search_locations)[0])
    return aiter_pkg / "jit" / "build"


def _invalidate_aiter_jit_build(
    target_file: Path,
    backup_dir: Path,
    *,
    jit_build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter ``jit/build/`` aside so a post-rebuild first import re-JITs.

    No-op for targets outside ``aiter/csrc/`` and for sandboxes where the
    aiter package isn't importable / hasn't populated its jit/build/ yet.
    Returns one of:

      * ``{"status": "ok", "src": ..., "backup_path": ..., "moved_at": ...}``
        — backup written; caller must persist this in the manifest before
        rebuild so revert can find it.
      * ``{"status": "skipped", "reason": ...}`` — non-aiter target, aiter
        not importable, or jit/build already absent/empty.
      * ``{"status": "failed", "error": ...}`` — backup path collision; the
        caller is expected to abort apply rather than rebuild against an
        inconsistent jit cache.

    ``jit_build_dir_override`` is a test-only escape hatch so unit tests can
    point at a synthetic jit/build/ tree without an importable aiter package.
    """
    if not _target_is_in_aiter_csrc(target_file):
        return {"status": "skipped", "reason": "target not under aiter/csrc/"}
    jit_build = jit_build_dir_override or _aiter_jit_build_dir()
    if jit_build is None:
        return {"status": "skipped", "reason": "aiter package not importable"}
    if not jit_build.exists():
        return {"status": "skipped", "reason": "aiter jit/build/ does not exist"}
    try:
        is_empty = not any(jit_build.iterdir())
    except OSError as exc:
        return {
            "status": "failed",
            "error": f"failed to scan {jit_build}: {exc}",
            "src": str(jit_build),
        }
    if is_empty:
        return {"status": "skipped", "reason": "aiter jit/build/ is empty"}
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "jit_build"
    if backup_path.exists():
        return {
            "status": "failed",
            "error": f"jit/build backup path already exists: {backup_path}",
            "src": str(jit_build),
        }
    try:
        shutil.move(str(jit_build), str(backup_path))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed: {exc}",
            "src": str(jit_build),
            "backup_path": str(backup_path),
        }
    return {
        "status": "ok",
        "src": str(jit_build),
        "backup_path": str(backup_path),
        "moved_at": _now(),
    }


def _restore_aiter_jit_build(jit_build_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse of :func:`_invalidate_aiter_jit_build`.

    Moves the backup back to its original location. If the post-rebuild first
    import already regenerated a fresh jit/build/, that fresh dir is removed
    first so the pre-patch state is restored bit-for-bit (revert semantics).
    """
    if not isinstance(jit_build_backup, dict) or jit_build_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no backup recorded"}
    src = Path(jit_build_backup.get("src", ""))
    backup_path = Path(jit_build_backup.get("backup_path", ""))
    if not src or not backup_path:
        return {"status": "skipped", "reason": "incomplete backup record"}
    if not backup_path.exists():
        return {
            "status": "skipped",
            "reason": f"backup path missing: {backup_path}",
        }
    if src.exists():
        try:
            shutil.rmtree(src)
        except OSError as exc:
            return {
                "status": "failed",
                "error": f"failed to clear regenerated jit/build/: {exc}",
                "src": str(src),
            }
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup_path), str(src))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed during restore: {exc}",
            "src": str(src),
            "backup_path": str(backup_path),
        }
    return {"status": "ok", "restored_to": str(src)}


# ---------------------------------------------------------------------------
# PR-K2 (#459): aiter cpp_itfs RUNTIME-compiled cache invalidation.
#
# Distinct from the ``@compile_ops`` jit/build cache handled above. aiter's
# ``csrc/cpp_itfs`` kernels (e.g. paged_attention -> ``pa_ragged``) are NOT
# produced by ``setup.py develop``; they are runtime-compiled on first call
# by ``compile_template_op`` (aiter ``csrc/cpp_itfs/utils.py``) into
# ``$AITER_ROOT_DIR/build/<md_name>_<md5(params)>/lib.so`` (default
# ``$HOME/.aiter/build``). The cache folder name hashes kernel *parameters*,
# NOT source content, so the pristine and the patched build of the same
# kernel collide on the SAME directory. ``compile_template_op`` rebuilds only
# when ``lib.so`` is missing (``not_built``), so after we patch the ``.cuh``
# and run the (no-op for this class) ``setup.py develop``, the next server
# reuses the STALE pristine ``lib.so``; the integrate re-baseline then
# measures ~0% and a genuinely-good kernel is flagged NEEDS_REVIEW / REVERT
# (observed -0.17% on a +2.5% paged_attention kernel; see GH #458).
#
# ``setup.py develop`` cannot refresh this kernel class, and the jit/build
# move above never touches ``$HOME/.aiter/build``. So for cpp_itfs targets we
# ALSO move the affected runtime-cache dirs aside before the rebuild step --
# scoped to the patched module's ``MD_NAME`` prefix(es) when determinable,
# else the whole cpp_itfs build root the scheduler uses -- so the re-baseline
# server runtime-recompiles the patched kernel from clean state.
# ``shutil.move`` keeps it reversible; :func:`_restore_aiter_cpp_itfs_cache`
# moves the backup back on revert. ``integrate_handler`` additionally sets
# ``AITER_REBUILD=1`` on the re-baseline server and gates KEEP on a verified
# fresh rebuild via :func:`verify_cpp_itfs_rebuilt`.
#
# Scope: ONLY aiter cpp_itfs targets. Non-cpp_itfs aiter targets, sglang and
# vllm keep their current behaviour bit-for-bit (this is a no-op for them).
# ---------------------------------------------------------------------------
_AITER_CPP_ITFS_MARKER = "/aiter/csrc/cpp_itfs/"
_MD_NAME_RE = re.compile(r"""(?m)^\s*MD_NAME\s*=\s*["']([^"']+)["']""")


def _target_is_in_aiter_cpp_itfs(target_file: Path) -> bool:
    """True iff ``target_file`` lives under any ``aiter/csrc/cpp_itfs/`` tree.

    Strict subset of :func:`_target_is_in_aiter_csrc`: these are the
    runtime-compiled kernels whose served ``.so`` lives in
    ``$HOME/.aiter/build`` rather than in ``<aiter>/jit/build`` or the
    statically-linked wheel. Matches both the editable checkout
    (``/sgl-workspace/aiter/csrc/cpp_itfs/...``) and the dist-packages
    layout (``.../aiter/csrc/cpp_itfs/...``).
    """
    return _AITER_CPP_ITFS_MARKER in str(target_file).replace(os.sep, "/")


def _aiter_cpp_itfs_build_dir() -> Path:
    """Resolve aiter's cpp_itfs runtime ``BUILD_DIR``.

    Mirrors aiter ``csrc/cpp_itfs/utils.py``: ``$AITER_ROOT_DIR/build`` with
    ``$AITER_ROOT_DIR`` defaulting to ``$HOME/.aiter``. Honouring both env
    vars keeps non-default deployments + unit tests correct without importing
    aiter into this standalone tool.
    """
    root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if not root:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        root = str(home / ".aiter")
    return Path(root) / "build"


def _cpp_itfs_module_names(target_file: Path) -> list[str]:
    """Best-effort ``MD_NAME`` prefix(es) for the cpp_itfs module(s) the
    patched source feeds.

    The cpp_itfs ``.py`` driver next to the patched source declares
    ``MD_NAME = "pa_ragged"`` (etc.), which becomes the
    ``<md_name>_<hash>`` runtime-cache folder prefix. A single shared
    ``.cuh`` (e.g. ``pa_kernels.cuh``) is pulled into several drivers in the
    same directory, so we collect EVERY ``MD_NAME`` declared in the target's
    directory. An empty result tells the caller to fall back to clearing the
    whole cpp_itfs build root.
    """
    names: set[str] = set()
    search_dir = target_file.parent
    try:
        py_files = sorted(search_dir.glob("*.py"))
    except OSError:
        py_files = []
    if target_file.suffix.lower() == ".py":
        py_files = list(dict.fromkeys([target_file, *py_files]))
    for py in py_files:
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _MD_NAME_RE.finditer(text):
            names.add(match.group(1))
    return sorted(names)


def _invalidate_aiter_cpp_itfs_cache(
    target_file: Path,
    backup_dir: Path,
    *,
    build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter cpp_itfs runtime-cache dirs aside so the re-baseline server
    runtime-recompiles the patched kernel from clean state.

    No-op (``skipped``, ``is_cpp_itfs=False``) for non-cpp_itfs targets. For
    cpp_itfs targets the affected ``<build_dir>/<md_name>_*`` dirs (scoped by
    ``MD_NAME`` when determinable, else every child of the build root) are
    MOVED into ``backup_dir/cpp_itfs_cache/`` so the operation is reversible.
    The record always carries ``is_cpp_itfs`` + ``build_dir`` +
    ``module_names`` + ``invalidated_unix`` (even when nothing was cached
    yet) so integrate can later verify a fresh rebuild actually landed.

    Returns one of ``ok`` / ``skipped`` / ``failed`` mirroring
    :func:`_invalidate_aiter_jit_build`. ``build_dir_override`` is a
    test-only hook.
    """
    if not _target_is_in_aiter_cpp_itfs(target_file):
        return {
            "status": "skipped",
            "is_cpp_itfs": False,
            "reason": "target not under aiter/csrc/cpp_itfs/",
        }
    build_dir = build_dir_override or _aiter_cpp_itfs_build_dir()
    module_names = _cpp_itfs_module_names(target_file)
    scope = "module" if module_names else "build_root"
    record: dict[str, Any] = {
        "is_cpp_itfs": True,
        "build_dir": str(build_dir),
        "module_names": module_names,
        "scope": scope,
        "invalidated_at": _now(),
        "invalidated_unix": time.time(),
    }
    if not build_dir.exists():
        # Nothing cached yet -> the re-baseline server will build fresh into
        # this dir on first kernel call. No stale binary to mask.
        record.update({
            "status": "skipped",
            "reason": "cpp_itfs build dir does not exist",
            "moved": [],
        })
        return record
    try:
        if module_names:
            to_move: list[Path] = []
            for md in module_names:
                to_move.extend(p for p in build_dir.glob(f"{md}_*") if p.is_dir())
        else:
            to_move = [p for p in build_dir.iterdir() if p.is_dir()]
    except OSError as exc:
        record.update({"status": "failed", "error": f"failed to scan {build_dir}: {exc}"})
        return record
    # De-dup: a shared .cuh can match overlapping MD_NAME globs.
    to_move = sorted({p.resolve() for p in to_move})
    if not to_move:
        record.update({
            "status": "skipped",
            "reason": "no matching cpp_itfs cache entries",
            "moved": [],
        })
        return record
    cache_backup_root = backup_dir / "cpp_itfs_cache"
    moved: list[dict[str, str]] = []
    try:
        cache_backup_root.mkdir(parents=True, exist_ok=True)
        for src in to_move:
            dst = cache_backup_root / src.name
            if dst.exists():
                record.update({
                    "status": "failed",
                    "error": f"cpp_itfs cache backup path already exists: {dst}",
                    "moved": moved,
                })
                return record
            shutil.move(str(src), str(dst))
            moved.append({"src": str(src), "backup_path": str(dst)})
    except (OSError, shutil.Error) as exc:
        record.update({
            "status": "failed",
            "error": f"shutil.move failed: {exc}",
            "moved": moved,
        })
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def _restore_aiter_cpp_itfs_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_invalidate_aiter_cpp_itfs_cache` (revert path).

    Moves each backed-up cache dir back to its original location, removing
    any dir the re-baseline server regenerated there first so the pre-patch
    runtime cache is restored bit-for-bit.
    """
    if not isinstance(cache_backup, dict) or cache_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no cpp_itfs cache backup recorded"}
    moved = cache_backup.get("moved") or []
    if not moved:
        return {"status": "skipped", "reason": "nothing was moved"}
    restored: list[str] = []
    for entry in moved:
        src = Path(entry.get("src", ""))  # original cache location
        backup_path = Path(entry.get("backup_path", ""))
        if not str(src) or not str(backup_path) or not backup_path.exists():
            continue
        if src.exists():
            try:
                shutil.rmtree(src)
            except OSError as exc:
                return {
                    "status": "failed",
                    "error": f"failed to clear regenerated cache dir {src}: {exc}",
                }
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(src))
        except (OSError, shutil.Error) as exc:
            return {
                "status": "failed",
                "error": f"shutil.move failed during restore: {exc}",
            }
        restored.append(str(src))
    return {"status": "ok", "restored": restored}


def verify_cpp_itfs_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline actually runtime-recompiled the patched kernel.

    After :func:`_invalidate_aiter_cpp_itfs_cache` moved the stale cache
    aside, a faithful re-baseline must repopulate
    ``<build_dir>/<md_name>_*/lib.so`` (the served binary) with an mtime at
    or after the invalidation. If no such fresh ``lib.so`` exists, the server
    dlopened a stale binary (or the kernel was never exercised) and the
    measured gain is meaningless -- the integrate KEEP/REVERT gate uses this
    to flag/abort instead of scoring a stale binary (GH #458 point 2).

    Returns ``{"verified": bool, ...}``. ``verified`` is True for
    non-cpp_itfs targets so the caller's gate is a strict no-op off the
    cpp_itfs path.
    """
    if not isinstance(cache_backup, dict) or not cache_backup.get("is_cpp_itfs"):
        return {"verified": True, "status": "skipped", "reason": "non-cpp_itfs target"}
    build_dir = Path(cache_backup.get("build_dir", ""))
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    module_names = list(cache_backup.get("module_names") or [])
    if not str(build_dir) or not build_dir.exists():
        return {
            "verified": False,
            "status": "stale",
            "reason": f"cpp_itfs build dir absent after re-baseline: {build_dir}",
        }
    globs = [f"{md}_*/lib.so" for md in module_names] or ["*/lib.so"]
    fresh: list[str] = []
    for pattern in globs:
        for so in build_dir.glob(pattern):
            try:
                mtime = so.stat().st_mtime
            except OSError:
                continue
            # 1s slack absorbs build-dir-create vs lib.so-flush ordering.
            if mtime + 1.0 >= since:
                fresh.append(str(so))
    if fresh:
        return {"verified": True, "status": "ok", "fresh_lib_so": sorted(set(fresh))[:8]}
    return {
        "verified": False,
        "status": "stale",
        "reason": (
            "no freshly-built cpp_itfs lib.so found after re-baseline; "
            "served binary is stale"
        ),
        "build_dir": str(build_dir),
        "module_names": module_names,
    }


# ---------------------------------------------------------------------------
# PR-K3 (#485): Triton (@triton.jit) kernel cache invalidation.
#
# Triton kernels -- e.g. sglang's editable fused_moe at
# ``python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`` or aiter's
# triton ops -- are NOT built by ``setup.py``; the Triton runtime
# AOT-compiles each kernel on first launch into ``$TRITON_CACHE_DIR``
# (default ``~/.triton/cache``) keyed by a hash of the kernel IR + signature,
# emitting ``*.hsaco`` / ``*.json`` / IR artifacts. A whole-file source patch
# usually changes that hash so the patched kernel recompiles -- but a
# config-only retune (e.g. the fused_moe ``64x64x32 -> 128x128x64`` tile
# change) or an autotune-key collision can re-serve a STALE compiled artifact,
# so the integrate re-baseline would measure the PRE-patch kernel. We move the
# Triton cache dir aside before the re-baseline so the server recompiles from
# clean state; revert moves it back. No-op for non-Triton targets. THIS makes
# the integrate re-baseline robust for the proven editable Triton fused_moe.
# ---------------------------------------------------------------------------
_TRITON_PATH_MARKERS = ("/triton/", "fused_moe_triton", "triton_kernels", "_triton_kernels")
# chaojhou review (point a): detect a real Triton kernel by the ``@triton.jit``
# decorator (optionally under ``@triton.autotune`` / ``@triton.heuristics``,
# which only ever sit ON TOP of a @triton.jit kernel) or by a triton source
# *path* marker -- DELIBERATELY NOT by a bare ``import triton`` / ``from
# triton`` / ``tl.load`` substring. Many non-kernel files import triton or
# reference ``tl.*`` helpers without defining a compiled kernel, and matching
# those would move the WHOLE Triton cache aside for an unrelated edit (and
# then make integrate hard-gate KEEP on a "stale" verify that never had a
# patched kernel to recompile). The decorator is the unambiguous marker of an
# actual JIT kernel definition that owns a cache entry.
_TRITON_DECORATOR_MARKERS = ("@triton.jit", "@triton.autotune", "@triton.heuristics")


def _target_is_triton(target_file: Path) -> bool:
    """Best-effort: is ``target_file`` an editable Triton (@triton.jit) kernel?

    Conservative (chaojhou-tightened): a ``.py`` file under a well-known
    Triton path segment, or whose text declares a ``@triton.jit`` (/
    ``@triton.autotune`` / ``@triton.heuristics``) kernel. Non-``.py`` targets
    and plain-python files that merely ``import triton`` / use ``tl.*`` without
    a JIT-kernel decorator are rejected so the whole-cache move-aside only
    fires for files that actually own a Triton compile-cache entry.
    """
    if target_file.suffix.lower() != ".py":
        return False
    s = str(target_file).replace(os.sep, "/")
    if any(m in s for m in _TRITON_PATH_MARKERS):
        return True
    try:
        text = target_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(m in text for m in _TRITON_DECORATOR_MARKERS)


def _triton_cache_dir() -> Path:
    """Resolve the Triton compile cache dir: ``$TRITON_CACHE_DIR`` else
    ``~/.triton/cache`` (the Triton runtime default)."""
    cache = os.environ.get("TRITON_CACHE_DIR", "").strip()
    if cache:
        return Path(cache)
    home = Path(os.environ.get("HOME", "~")).expanduser()
    return home / ".triton" / "cache"


# ---------------------------------------------------------------------------
# PR-K4 (#485): torch inductor cache invalidation.
#
# ``torch.compile`` / TorchInductor AOT-compiles fused kernels into an on-disk
# cache (``$TORCHINDUCTOR_CACHE_DIR`` else ``~/.cache/torch/inductor`` and/or
# ``/tmp/torchinductor_<user>``). Like Triton, a stale entry can be re-served
# so the integrate re-baseline measures the pre-patch kernel. We move the
# inductor cache dir(s) aside before the re-baseline so the server recompiles
# from clean state; revert moves them back. No-op for non-inductor targets.
# ---------------------------------------------------------------------------
_INDUCTOR_PATH_MARKERS = ("torchinductor", "/torch/_inductor/", "_inductor_cache")
# chaojhou review (point a, "same spirit"): require an actual ``torch.compile``
# invocation / decorator or an inductor import -- NOT a bare ``torch.compile``
# / ``torch._inductor`` substring (which shows up in comments, docstrings and
# unrelated helper code). This keeps the inductor move-aside scoped to files
# that actually drive an inductor compile.
_INDUCTOR_USAGE_MARKERS = (
    "@torch.compile",
    "torch.compile(",
    "import torch._inductor",
    "from torch._inductor",
)


def _target_is_inductor(target_file: Path) -> bool:
    """Best-effort: is ``target_file`` a torch inductor target?

    Path under a torchinductor cache / ``torch/_inductor`` tree, or a ``.py``
    file that actually invokes ``torch.compile(`` / ``@torch.compile`` or
    imports ``torch._inductor``. Conservative (chaojhou-tightened) so the
    cache move-aside only fires for real inductor targets, not files that
    merely mention ``torch.compile`` in a comment.
    """
    s = str(target_file).replace(os.sep, "/")
    if any(m in s for m in _INDUCTOR_PATH_MARKERS):
        return True
    if target_file.suffix.lower() != ".py":
        return False
    try:
        text = target_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(m in text for m in _INDUCTOR_USAGE_MARKERS)


def _inductor_cache_dirs() -> list[Path]:
    """Resolve torch inductor cache dir(s): ``$TORCHINDUCTOR_CACHE_DIR`` if set,
    else ``~/.cache/torch/inductor`` plus ``/tmp/torchinductor_<user>``.

    chaojhou review (point b): on a SHARED pod, inductor cache isolation
    relies on ``$TORCHINDUCTOR_CACHE_DIR`` pointing at a PER-RUN directory --
    when it is set we move ONLY that dir aside, so concurrent runs with their
    own per-run dirs are unaffected. The fallback ``/tmp/torchinductor_<user>``
    (and ``~/.cache/torch/inductor``) is keyed by OS user, NOT by run, so on a
    shared pod where several runs share a user it is NOT isolated; treat the
    fallback as best-effort and prefer exporting a per-run
    ``$TORCHINDUCTOR_CACHE_DIR`` (e.g. ``/tmp/torchinductor_<run_id>``) so this
    move-aside cannot disturb a co-tenant's compile cache.
    """
    dirs: list[Path] = []
    env = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "").strip()
    if env:
        # Per-run isolated dir: move ONLY this one aside.
        dirs.append(Path(env))
    else:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        dirs.append(home / ".cache" / "torch" / "inductor")
        user = (os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()
        if user:
            # NOTE: shared-pod hazard -- this dir is per-user, not per-run.
            dirs.append(Path("/tmp") / f"torchinductor_{user}")
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def _move_cache_dirs_aside(
    dirs: Iterable[Path], backup_dir: Path, *, label: str,
) -> tuple[list[dict[str, str]], str | None]:
    """Move each existing dir in ``dirs`` into ``backup_dir/<label>/<name>``.

    Returns ``(moved, error)``. ``moved`` is a reversible list of
    ``{"src", "backup_path"}`` records (empty when nothing existed). ``error``
    is non-None on the first failure (backup collision / move error) so the
    caller can bail without scoring against an inconsistent cache.
    """
    moved: list[dict[str, str]] = []
    existing = [d for d in dirs if d.exists()]
    if not existing:
        return moved, None
    dest_root = backup_dir / label
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        for src in existing:
            dst = dest_root / src.name
            if dst.exists():
                return moved, f"cache backup path already exists: {dst}"
            shutil.move(str(src), str(dst))
            moved.append({"src": str(src), "backup_path": str(dst)})
    except (OSError, shutil.Error) as exc:
        return moved, f"shutil.move failed: {exc}"
    return moved, None


def _restore_moved_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_move_cache_dirs_aside` (shared Triton/inductor restore).

    Moves each backed-up dir back to its original location, removing any dir
    the re-baseline regenerated there first so the pre-patch state is restored
    bit-for-bit.
    """
    if not isinstance(cache_backup, dict) or cache_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no cache backup recorded"}
    moved = cache_backup.get("moved") or []
    if not moved:
        return {"status": "skipped", "reason": "nothing was moved"}
    restored: list[str] = []
    for entry in moved:
        src = Path(entry.get("src", ""))
        backup_path = Path(entry.get("backup_path", ""))
        if not str(src) or not str(backup_path) or not backup_path.exists():
            continue
        if src.exists():
            try:
                shutil.rmtree(src)
            except OSError as exc:
                return {
                    "status": "failed",
                    "error": f"failed to clear regenerated cache dir {src}: {exc}",
                }
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(src))
        except (OSError, shutil.Error) as exc:
            return {
                "status": "failed",
                "error": f"shutil.move failed during restore: {exc}",
            }
        restored.append(str(src))
    return {"status": "ok", "restored": restored}


def _verify_moved_cache_rebuilt(
    cache_backup: dict[str, Any], *, kind_key: str, label: str,
) -> dict[str, Any]:
    """Shared fresh-build check for the move-aside toolchains (Triton/inductor).

    ``verified`` is True for non-matching targets and when nothing was moved
    aside (cache absent/empty at invalidation -> can't conclude staleness, so
    never false-abort). When a non-empty cache WAS moved aside, requires a
    file in the recreated cache dir with mtime at/after the invalidation
    (proof the patched kernel recompiled); otherwise reports ``stale``.
    """
    if not isinstance(cache_backup, dict) or not cache_backup.get(kind_key):
        return {"verified": True, "status": "skipped", "reason": f"non-{label} target"}
    moved = cache_backup.get("moved") or []
    if cache_backup.get("status") != "ok" or not moved:
        return {
            "verified": True,
            "status": "skipped",
            "reason": f"{label} cache was empty/absent at invalidation; nothing to verify",
        }
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    fresh: list[str] = []
    for m in moved:
        cache_dir = Path(m.get("src", ""))  # original (now-recreated) location
        if not cache_dir.exists():
            continue
        try:
            for p in cache_dir.rglob("*"):
                try:
                    if p.is_file() and p.stat().st_mtime + 1.0 >= since:
                        fresh.append(str(p))
                except OSError:
                    continue
                if len(fresh) >= 8:
                    break
        except OSError:
            continue
        if len(fresh) >= 8:
            break
    if fresh:
        return {"verified": True, "status": "ok", "fresh_artifacts": sorted(set(fresh))[:8]}
    return {
        "verified": False,
        "status": "stale",
        "reason": (
            f"no freshly-compiled {label} artifact found after re-baseline; "
            "served binary may be stale"
        ),
    }


def _invalidate_triton_cache(
    target_file: Path,
    backup_dir: Path,
    *,
    cache_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move the Triton compile cache aside so the re-baseline recompiles.

    No-op (``skipped``, ``is_triton=False``) for non-Triton targets.
    ``cache_dir_override`` is a test-only hook.
    """
    if not _target_is_triton(target_file):
        return {
            "status": "skipped",
            "is_triton": False,
            "reason": "target is not a Triton kernel",
        }
    cache_dir = cache_dir_override or _triton_cache_dir()
    record: dict[str, Any] = {
        "is_triton": True,
        "toolchain": "triton",
        "cache_dirs": [str(cache_dir)],
        "scope": "triton_cache_dir",
        "invalidated_at": _now(),
        "invalidated_unix": time.time(),
    }
    moved, error = _move_cache_dirs_aside([cache_dir], backup_dir, label="triton_cache")
    if error is not None:
        record.update({"status": "failed", "error": error, "moved": moved})
        return record
    if not moved:
        record.update({
            "status": "skipped",
            "reason": "triton cache dir does not exist",
            "moved": [],
        })
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def _restore_triton_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_invalidate_triton_cache` (revert path)."""
    return _restore_moved_cache(cache_backup)


def verify_triton_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline freshly recompiled the patched Triton kernel."""
    return _verify_moved_cache_rebuilt(cache_backup, kind_key="is_triton", label="triton")


def _invalidate_torch_inductor_cache(
    target_file: Path,
    backup_dir: Path,
    *,
    cache_dirs_override: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Move the torch inductor cache dir(s) aside so the re-baseline recompiles.

    No-op (``skipped``, ``is_inductor=False``) for non-inductor targets.
    ``cache_dirs_override`` is a test-only hook.
    """
    if not _target_is_inductor(target_file):
        return {
            "status": "skipped",
            "is_inductor": False,
            "reason": "target is not a torch inductor kernel",
        }
    cache_dirs = list(cache_dirs_override) if cache_dirs_override is not None else _inductor_cache_dirs()
    record: dict[str, Any] = {
        "is_inductor": True,
        "toolchain": "torch_inductor",
        "cache_dirs": [str(d) for d in cache_dirs],
        "scope": "inductor_cache_dirs",
        "invalidated_at": _now(),
        "invalidated_unix": time.time(),
    }
    moved, error = _move_cache_dirs_aside(cache_dirs, backup_dir, label="inductor_cache")
    if error is not None:
        record.update({"status": "failed", "error": error, "moved": moved})
        return record
    if not moved:
        record.update({
            "status": "skipped",
            "reason": "torch inductor cache dir(s) do not exist",
            "moved": [],
        })
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def _restore_torch_inductor_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_invalidate_torch_inductor_cache` (revert path)."""
    return _restore_moved_cache(cache_backup)


def verify_inductor_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline freshly recompiled the patched inductor kernel."""
    return _verify_moved_cache_rebuilt(cache_backup, kind_key="is_inductor", label="inductor")


# ---------------------------------------------------------------------------
# Cache-invalidation REGISTRY (#485): ONE dispatch keyed by toolchain.
#
# Each :class:`ToolchainCacheEntry` declares, for one toolchain, the cache it
# owns and how to {invalidate (move-aside), restore, verify a fresh build} +
# the rebuild env the integrate re-baseline must set. ``apply_kernel_patch``
# drives invalidation through this table, ``revert_kernel_patch`` restores
# through it, and ``integrate_handler`` reads ``rebuild_env_for_apply_result``
# / ``verify_rebuilt_for_apply_result`` to set the right env + run the right
# verification gate. The aiter entries delegate to the pre-existing
# ``_invalidate_aiter_jit_build`` / ``_invalidate_aiter_cpp_itfs_cache`` (and
# their restore/verify) so behaviour is byte-for-byte preserved and the
# GH #458 (#459) public names stay importable + identically-shaped. Every
# entry self-gates (``invalidate`` returns ``skipped`` off its toolchain) so a
# target the registry doesn't handle is bit-for-bit unaffected.
# ---------------------------------------------------------------------------
class ToolchainCacheEntry:
    """One toolchain's cache-invalidation policy.

    A plain class (not a ``@dataclass``) on purpose: this module is loaded via
    ``importlib.util.spec_from_file_location`` (kernel_request_handlers'
    ``_load_apply_tool`` + the unit tests) WITHOUT being registered in
    ``sys.modules``, and ``@dataclass`` + ``from __future__ import annotations``
    introspects ``sys.modules[cls.__module__]`` at class-creation time and
    crashes when the module isn't registered. The fields below are stored
    verbatim; callables are kept as plain attributes (never bound as methods,
    so ``entry.invalidate(target, backup_dir)`` calls the bare function).

    Fields:
      name                    stable toolchain id
      manifest_key            where the invalidation record is stored
      restore_key             where the restore record is stored
      matches(path)->bool     path-based detection (for classification)
      invalidate(target, backup_dir)->record
      restore(record)->result
      verify(record)->{"verified": bool, ...}
      engaged(record)->bool   did this toolchain apply to the target?
      requires_compiled       only run inside the compiled-rebuild path
      gates_keep              integrate hard-gates KEEP on this entry's verify
      on_failure_error_class  error_class surfaced when invalidate fails
      failure_error_prefix    human prefix for the failure error message
      rebuild_env             env the re-baseline server must set
      skipped_default         record shape when the entry does not run
    """

    def __init__(
        self,
        *,
        name: str,
        manifest_key: str,
        restore_key: str,
        matches: Callable[[Path], bool],
        invalidate: Callable[..., dict[str, Any]],
        restore: Callable[[dict[str, Any]], dict[str, Any]],
        verify: Callable[[dict[str, Any]], dict[str, Any]],
        engaged: Callable[[dict[str, Any]], bool],
        requires_compiled: bool,
        gates_keep: bool,
        on_failure_error_class: str,
        failure_error_prefix: str,
        rebuild_env: dict[str, str],
        skipped_default: dict[str, Any],
    ) -> None:
        self.name = name
        self.manifest_key = manifest_key
        self.restore_key = restore_key
        self.matches = matches
        self.invalidate = invalidate
        self.restore = restore
        self.verify = verify
        self.engaged = engaged
        self.requires_compiled = requires_compiled
        self.gates_keep = gates_keep
        self.on_failure_error_class = on_failure_error_class
        self.failure_error_prefix = failure_error_prefix
        self.rebuild_env = rebuild_env
        self.skipped_default = skipped_default


CACHE_INVALIDATION_REGISTRY: list[ToolchainCacheEntry] = [
    ToolchainCacheEntry(
        name="aiter_compile_ops",
        manifest_key="jit_build_backup",
        restore_key="jit_build_restore",
        matches=_target_is_in_aiter_csrc,
        invalidate=_invalidate_aiter_jit_build,
        restore=_restore_aiter_jit_build,
        verify=lambda rec: {"verified": True, "status": "skipped", "reason": "jit/build not gated"},
        engaged=lambda rec: rec.get("status") == "ok",
        requires_compiled=True,
        gates_keep=False,
        on_failure_error_class="aiter_jit_invalidation_failed",
        failure_error_prefix="aiter jit/build/ invalidation failed",
        rebuild_env={},
        skipped_default={"status": "skipped", "reason": "rebuild not run"},
    ),
    ToolchainCacheEntry(
        name="aiter_cpp_itfs",
        manifest_key="cpp_itfs_cache_backup",
        restore_key="cpp_itfs_cache_restore",
        matches=_target_is_in_aiter_cpp_itfs,
        invalidate=_invalidate_aiter_cpp_itfs_cache,
        restore=_restore_aiter_cpp_itfs_cache,
        verify=verify_cpp_itfs_rebuilt,
        engaged=lambda rec: bool(rec.get("is_cpp_itfs")),
        requires_compiled=True,
        gates_keep=True,
        on_failure_error_class="aiter_cpp_itfs_invalidation_failed",
        failure_error_prefix="aiter cpp_itfs runtime cache invalidation failed",
        rebuild_env={"AITER_REBUILD": "1"},
        skipped_default={"status": "skipped", "reason": "rebuild not run", "is_cpp_itfs": False},
    ),
    ToolchainCacheEntry(
        name="triton",
        manifest_key="triton_cache_backup",
        restore_key="triton_cache_restore",
        matches=_target_is_triton,
        invalidate=_invalidate_triton_cache,
        restore=_restore_triton_cache,
        verify=verify_triton_rebuilt,
        engaged=lambda rec: bool(rec.get("is_triton")),
        requires_compiled=False,
        gates_keep=True,
        on_failure_error_class="triton_cache_invalidation_failed",
        failure_error_prefix="triton cache invalidation failed",
        rebuild_env={},
        skipped_default={"status": "skipped", "reason": "not run", "is_triton": False},
    ),
    ToolchainCacheEntry(
        name="torch_inductor",
        manifest_key="inductor_cache_backup",
        restore_key="inductor_cache_restore",
        matches=_target_is_inductor,
        invalidate=_invalidate_torch_inductor_cache,
        restore=_restore_torch_inductor_cache,
        verify=verify_inductor_rebuilt,
        engaged=lambda rec: bool(rec.get("is_inductor")),
        requires_compiled=False,
        gates_keep=True,
        on_failure_error_class="torch_inductor_cache_invalidation_failed",
        failure_error_prefix="torch inductor cache invalidation failed",
        rebuild_env={},
        skipped_default={"status": "skipped", "reason": "not run", "is_inductor": False},
    ),
]


def toolchain_for(target_file: str | Path, strategy: dict[str, Any] | None = None) -> str | None:
    """Classify ``target_file`` to a single registry toolchain (most-specific
    first), or ``None`` when no toolchain owns its cache.

    Note this is for observability/classification only; the apply-time
    invalidation runs EVERY matching entry (a cpp_itfs target is also under
    aiter/csrc, so both the @compile_ops jit/build and cpp_itfs caches are
    invalidated, preserving the pre-registry behaviour).
    """
    p = Path(target_file)
    if _target_is_in_aiter_cpp_itfs(p):
        return "aiter_cpp_itfs"
    if _target_is_inductor(p):
        return "torch_inductor"
    if _target_is_triton(p):
        return "triton"
    if _target_is_in_aiter_csrc(p):
        return "aiter_compile_ops"
    return None


def _invalidate_caches_for_apply(
    target: Path,
    backup_dir: Path,
    *,
    strategy: dict[str, Any],
    skip_rebuild: bool,
    manifest: dict[str, Any],
    manifest_path: Path,
    source_backup: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Run the registry's invalidations for ``target`` before re-baseline.

    Returns ``(records, failure)``:
      * ``records`` maps every entry's ``manifest_key`` -> its record
        (skipped-default for entries that did not run), preserving the apply
        return/manifest shape for all callers + the GH #458 tests.
      * ``failure`` is ``None`` on success, else a ready-to-return failed dict
        (source + any already-moved caches have already been restored).

    Each successful (``ok``) record is persisted to the manifest immediately so
    a later rebuild failure can restore the moved-aside cache via
    :func:`revert_kernel_patch`. Compiled-only entries (aiter jit/build,
    cpp_itfs) run only inside the compiled-rebuild path -- identical to the
    pre-registry gating; the source-cache entries (Triton, inductor) run for
    ``.py`` targets too since their cache is invalidated by relocation, not by
    ``setup.py``.
    """
    records: dict[str, dict[str, Any]] = {
        e.manifest_key: dict(e.skipped_default) for e in CACHE_INVALIDATION_REGISTRY
    }
    if skip_rebuild:
        return records, None
    succeeded: list[tuple[ToolchainCacheEntry, dict[str, Any]]] = []
    for entry in CACHE_INVALIDATION_REGISTRY:
        if entry.requires_compiled and not strategy.get("compiled"):
            continue
        rec = entry.invalidate(target, backup_dir)
        records[entry.manifest_key] = rec
        if rec.get("status") == "failed":
            # Refuse to re-baseline against an inconsistent cache: restore the
            # source + every already-moved cache (reverse order), then bail.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            for done_entry, done_rec in reversed(succeeded):
                done_entry.restore(done_rec)
            return records, {
                "status": "failed",
                "error_class": entry.on_failure_error_class,
                "error": f"{entry.failure_error_prefix}: {rec.get('error')}",
                "manifest_path": str(manifest_path),
                entry.manifest_key: rec,
            }
        if rec.get("status") == "ok":
            succeeded.append((entry, rec))
            # Persist BEFORE rebuild so a rebuild failure can restore it.
            manifest[entry.manifest_key] = rec
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return records, None


def restore_caches_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Restore every toolchain cache the manifest records as moved-aside.

    Iterates the registry (aiter jit/build, cpp_itfs, Triton, inductor),
    restoring each ``ok`` record and writing the restore result under the
    entry's ``restore_key``. Returns the list of restored paths. No-op for
    caches that were not moved. Used by :func:`revert_kernel_patch`.
    """
    restored: list[str] = []
    for entry in CACHE_INVALIDATION_REGISTRY:
        rec = manifest.get(entry.manifest_key) or {}
        if rec.get("status") != "ok":
            continue
        res = entry.restore(rec)
        manifest[entry.restore_key] = res
        if res.get("status") == "ok":
            if res.get("restored_to"):
                restored.append(str(res["restored_to"]))
            restored.extend(str(p) for p in (res.get("restored") or []))
    return restored


def rebuild_env_for_apply_result(apply_result: dict[str, Any]) -> dict[str, str]:
    """Env vars the integrate re-baseline server must set for whatever
    toolchain caches the apply invalidated.

    Preserves GH #458: a cpp_itfs apply -> ``{"AITER_REBUILD": "1"}``. Empty
    for toolchains whose move-aside alone forces recompilation (Triton,
    inductor) and for targets the registry doesn't touch.
    """
    env: dict[str, str] = {}
    if not isinstance(apply_result, dict):
        return env
    for entry in CACHE_INVALIDATION_REGISTRY:
        rec = apply_result.get(entry.manifest_key) or {}
        if entry.engaged(rec):
            env.update(entry.rebuild_env)
    return env


def verify_rebuilt_for_apply_result(apply_result: dict[str, Any]) -> dict[str, Any]:
    """Run the fresh-build verification for every gating toolchain whose cache
    the apply invalidated; aggregate into one ``{"verified": bool, ...}``.

    A strict no-op (``verified=True``) when no gating cache was invalidated, so
    integrate's KEEP/REVERT gate is unaffected off the cache-invalidation
    paths. Generalizes the GH #458 cpp_itfs gate to Triton + inductor.
    """
    if not isinstance(apply_result, dict):
        return {"verified": True, "status": "skipped", "reason": "no apply result"}
    per: dict[str, Any] = {}
    verified = True
    for entry in CACHE_INVALIDATION_REGISTRY:
        if not entry.gates_keep:
            continue
        rec = apply_result.get(entry.manifest_key) or {}
        if not entry.engaged(rec):
            continue
        v = entry.verify(rec)
        per[entry.name] = v
        if not v.get("verified", True):
            verified = False
    if not per:
        return {"verified": True, "status": "skipped", "reason": "no gating cache invalidated"}
    return {
        "verified": verified,
        "status": "ok" if verified else "stale",
        "per_toolchain": per,
    }


def _detect_strategy(target_file: Path, *, allow_unknown_target: bool) -> dict[str, Any]:
    target = str(target_file)
    lower = target.lower()
    roots = known_target_roots()
    if not allow_unknown_target and not any(root in lower for root in roots):
        raise ValueError(f"target_file is outside known reusable source roots: {target_file}")

    suffix = target_file.suffix.lower()
    compiled = suffix in COMPILED_SOURCE_SUFFIXES
    root = None
    rebuild_command: list[str] = []
    artifact_roots: list[Path] = []

    if "/sgl-workspace/aiter/" in lower:
        root = Path("/sgl-workspace/aiter")
        rebuild_command = ["/opt/venv/bin/python", "setup.py", "develop"]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/sgl-kernel/" in lower:
        root = Path("/sgl-workspace/sglang/sgl-kernel")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/" in lower:
        root = Path("/sgl-workspace/sglang")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "python"]
        artifact_roots = [root]
    elif "/sgl-workspace/vllm/" in lower:
        root = Path("/sgl-workspace/vllm")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif allow_unknown_target:
        root = target_file.parent

    if suffix in PYTHON_SOURCE_SUFFIXES:
        compiled = False
        rebuild_command = []
        artifact_roots = []

    return {
        "compiled": compiled,
        "root": str(root) if root else "",
        "rebuild_command": rebuild_command,
        "artifact_roots": artifact_roots,
    }


def _discover_artifacts(
    roots: Iterable[Path],
    explicit_artifact_paths: Iterable[str] | None,
    *,
    max_artifacts: int = 400,
) -> list[Path]:
    seen: set[Path] = set()
    artifacts: list[Path] = []
    for raw in explicit_artifact_paths or []:
        p = Path(raw)
        if p.is_file() and p not in seen:
            artifacts.append(p)
            seen.add(p)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if len(artifacts) >= max_artifacts:
                break
            if p.is_file() and p.suffix.lower() in COMPILED_ARTIFACT_SUFFIXES and p not in seen:
                artifacts.append(p)
                seen.add(p)
    return artifacts


def _run_rebuild(command: list[str], cwd: Path, timeout_sec: int) -> dict[str, Any]:
    if not command:
        return {"status": "skipped", "reason": "no rebuild command"}
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env={**os.environ, "PATH": f"/opt/venv/bin:{os.environ.get('PATH', '')}"},
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "command": command,
        "cwd": str(cwd),
    }


def revert_kernel_patch(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    restored: list[str] = []
    for item in manifest.get("artifacts", []):
        src = Path(item["backup_path"])
        dst = Path(item["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
    source_backup = manifest.get("source_backup") or {}
    if source_backup:
        src = Path(source_backup["backup_path"])
        dst = Path(source_backup["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
            if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES:
                manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)

    # Registry-driven cache restore: put every toolchain cache the apply
    # moved aside back (aiter jit/build, aiter cpp_itfs runtime cache, Triton,
    # torch inductor), removing any dir the re-baseline regenerated first so
    # the pre-patch state is restored bit-for-bit. Done after source/artifact
    # restore but before multi-node fan-out so the host-local caches are back
    # in place even if pod-side revert subsequently fails. No-op for caches
    # that weren't moved (status != "ok"). Preserves the PR-K jit/build +
    # #459 cpp_itfs restore order + restored-path collection (the
    # ``aiter_compile_ops`` registry entry delegates to
    # :func:`_restore_aiter_jit_build`, writing the same ``jit_build_restore``
    # manifest key).
    restored.extend(restore_caches_from_manifest(manifest))

    # Multi-node: also fan-out a revert to every pod that received the
    # corresponding apply. Best-effort; sandbox revert above already
    # restored the LLM-visible source, but we still want pod-side
    # sglang to load v0 on the next restart so the integrate
    # baseline rerun measures the right thing.
    multinode_info = manifest.get("multinode") or {}
    mn_revert: dict[str, Any] = {}
    if multinode_info and multinode_info.get("host_backup_map"):
        target_path = multinode_info.get("target_path") or (
            source_backup.get("path") if source_backup else ""
        )
        backup_map = multinode_info.get("host_backup_map") or {}
        if target_path and backup_map:
            try:
                mn_revert = _dispatch_multinode_revert(
                    target_path=target_path,
                    backup_map=backup_map,
                )
            except Exception as exc:  # noqa: BLE001
                mn_revert = {"status": "failed", "error": str(exc)}

    reverted_at = _now()
    manifest["status"] = "reverted"
    manifest["reverted_at"] = reverted_at
    manifest["restored_paths"] = restored
    if mn_revert:
        manifest["multinode_revert"] = mn_revert
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_file),
        "restored_paths": restored,
        "reverted_at": reverted_at,
    }
    # Only attach multinode_revert when fan-out actually ran. Preserves
    # single-node revert return shape bit-for-bit.
    if mn_revert:
        result["multinode_revert"] = mn_revert
    return result


def apply_kernel_patch(
    *,
    patch_path: str | Path,
    target_file: str | Path,
    backup_root: str | Path,
    kernel_id: str = "",
    artifact_paths: Iterable[str] | None = None,
    rebuild_command: list[str] | str | None = None,
    rebuild_timeout_sec: int = 1800,
    skip_rebuild: bool = False,
    allow_unknown_target: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    patch = Path(patch_path).resolve()
    target = Path(target_file).resolve()
    if not patch.is_file():
        return {"status": "failed", "error": f"patch_path does not exist: {patch}"}
    if not target.is_file():
        return {"status": "failed", "error": f"target_file does not exist: {target}"}

    try:
        strategy = _detect_strategy(target, allow_unknown_target=allow_unknown_target)
        _validate_patch_source(patch, target)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}

    backup_dir = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    manifest_path = backup_dir / "manifest.json"
    source_backup = _copy_to_backup(target, backup_dir, "source")
    artifacts: list[dict[str, str]] = []
    if strategy["compiled"]:
        found = _discover_artifacts(strategy["artifact_roots"], artifact_paths)
        artifacts = [_copy_to_backup(path, backup_dir, "artifacts") for path in found]

    manifest = {
        "status": "prepared",
        "kernel_id": kernel_id,
        "patch_path": str(patch),
        "target_file": str(target),
        "source_backup": source_backup,
        "artifacts": artifacts,
        "strategy": {
            "compiled": strategy["compiled"],
            "root": strategy["root"],
        },
        "created_at": _now(),
    }
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(manifest_path)}

    shutil.copy2(patch, target)
    cache_clear = (
        _clear_python_kernel_caches(target)
        if target.suffix.lower() in PYTHON_SOURCE_SUFFIXES else
        {"status": "skipped", "reason": "not a python source target"}
    )

    # Multi-node: fan-out the SAME patch to every RayJob pod so head +
    # workers see identical source. Sandbox-local write above already
    # succeeded, so the apply is in a "sandbox-applied, pod-pending"
    # state; the dispatch either promotes it to a fully-applied state
    # across all three (sandbox + head + workers) or we hard-revert the
    # sandbox copy to avoid a partial multinode state where LLM would
    # next round see v1 source locally but pod-side sglang still runs
    # v0.
    multinode_info: dict[str, Any] = {}
    if _is_multi_node():
        pod_backup_dir = os.environ.get(
            "HYPERLOOM_MN_KERNEL_BACKUP_DIR", _MN_POD_BACKUP_DIR_DEFAULT,
        )
        try:
            mn_apply = _dispatch_multinode_apply(
                target_file=target,
                patch_path=patch,
                kernel_id=kernel_id,
                backup_dir_on_pod=pod_backup_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Pod fan-out failed: revert sandbox copy from the source
            # backup so the three sides agree on v0 again.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error": (
                    "multi-node apply fan-out failed; sandbox copy "
                    f"reverted to {source_backup['backup_path']}: {exc}"
                ),
                "manifest_path": str(manifest_path),
            }
        # Persist per-host backups into the manifest so revert can find
        # them. Map shape: {hostname: backup_path}.
        backup_map: dict[str, str] = {}
        for entry in mn_apply.get("per_node", []) or []:
            host = (entry.get("host") or "").strip()
            bp = (entry.get("backup_path") or "").strip()
            if host and bp:
                backup_map[host] = bp
        multinode_info = {
            "status": "ok",
            "target_path": str(target),
            "backup_dir_on_pod": pod_backup_dir,
            "host_backup_map": backup_map,
            "per_node": mn_apply.get("per_node", []),
        }
        # CRITICAL: persist multinode info to the manifest NOW (not at
        # the end with status=applied) so a later rebuild failure
        # triggers revert_kernel_patch with the multinode block still
        # visible — without this, rebuild failure would revert only
        # the sandbox copy and leave pod-side patches stranded.
        manifest["multinode"] = multinode_info
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    command: list[str] = []
    if isinstance(rebuild_command, str):
        command = ["/bin/bash", "-lc", rebuild_command]
    elif rebuild_command:
        command = list(rebuild_command)
    elif not skip_rebuild:
        command = list(strategy["rebuild_command"])

    rebuild = {"status": "skipped", "reason": "source-only patch or skip_rebuild=true"}
    # Registry-driven cache invalidation (#485): ONE dispatch keyed by
    # toolchain (aiter @compile_ops jit/build, aiter cpp_itfs runtime cache,
    # Triton cache, torch inductor cache). Each entry self-gates so a target
    # that doesn't match a toolchain is bit-for-bit unaffected. Moves the right
    # cache(s) aside BEFORE rebuild/re-baseline so the patched kernel is
    # (re)compiled from clean state; on invalidation failure the source +
    # any already-moved caches are restored and we bail rather than score
    # against an inconsistent cache. The aiter @compile_ops entry delegates to
    # the PR-K :func:`_invalidate_aiter_jit_build` so jit-build behaviour
    # (incl. the ``aiter_jit_invalidation_failed`` error_class +
    # ``jit_build_backup`` manifest/result key) is preserved bit-for-bit; the
    # cpp_itfs entry delegates to the #459 functions.
    cache_records, invalidation_failure = _invalidate_caches_for_apply(
        target, backup_dir,
        strategy=strategy, skip_rebuild=skip_rebuild,
        manifest=manifest, manifest_path=manifest_path, source_backup=source_backup,
    )
    if invalidation_failure is not None:
        return invalidation_failure

    if strategy["compiled"] and not skip_rebuild:
        cwd = Path(strategy["root"] or target.parent)
        rebuild = _run_rebuild(command, cwd, rebuild_timeout_sec)
        if rebuild["status"] != "ok":
            revert = revert_kernel_patch(manifest_path)
            return {
                "status": "failed",
                "error": "rebuild failed; original source/artifacts restored",
                "manifest_path": str(manifest_path),
                "rebuild": rebuild,
                "revert": revert,
            }

    manifest["status"] = "applied"
    manifest["applied_at"] = _now()
    manifest["rebuild"] = rebuild
    manifest["cache_clear"] = cache_clear
    # Surface every toolchain cache record (incl. skipped reasons) so manifest
    # readers can audit which caches were invalidated, integrate can verify a
    # fresh rebuild landed, and revert can restore the moved-aside caches.
    for _ck, _crec in cache_records.items():
        if _crec.get("status") in {"ok", "skipped"}:
            manifest[_ck] = _crec
    # multinode block (when present) is already persisted at fan-out
    # time above; we don't rewrite it here to avoid clobbering the
    # per-host backup map.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "target_file": str(target),
        "backup_dir": str(backup_dir),
        "compiled": bool(strategy["compiled"]),
        "artifact_count": len(artifacts),
        "cache_clear": cache_clear,
        "rebuild": rebuild,
        "toolchain": toolchain_for(target, strategy),
        **cache_records,
    }
    # Only attach the multinode key when fan-out actually ran. Keeps
    # single-node callers' return shape bit-for-bit identical to
    # pre-multinode behaviour.
    if multinode_info:
        result["multinode"] = multinode_info
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply or revert a kernel patch")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--patch-path", required=True)
    apply_p.add_argument("--target-file", required=True)
    apply_p.add_argument("--backup-root", required=True)
    apply_p.add_argument("--kernel-id", default="")
    apply_p.add_argument("--artifact-path", action="append", default=[])
    apply_p.add_argument("--rebuild-command", default="")
    apply_p.add_argument("--rebuild-timeout-sec", type=int, default=1800)
    apply_p.add_argument("--skip-rebuild", action="store_true")
    apply_p.add_argument("--allow-unknown-target", action="store_true")
    apply_p.add_argument("--dry-run", action="store_true")

    revert_p = sub.add_parser("revert")
    revert_p.add_argument("--manifest-path", required=True)

    args = parser.parse_args()
    if args.command == "revert":
        result = revert_kernel_patch(args.manifest_path)
    else:
        result = apply_kernel_patch(
            patch_path=args.patch_path,
            target_file=args.target_file,
            backup_root=args.backup_root,
            kernel_id=args.kernel_id,
            artifact_paths=args.artifact_path,
            rebuild_command=args.rebuild_command or None,
            rebuild_timeout_sec=args.rebuild_timeout_sec,
            skip_rebuild=args.skip_rebuild,
            allow_unknown_target=args.allow_unknown_target,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
