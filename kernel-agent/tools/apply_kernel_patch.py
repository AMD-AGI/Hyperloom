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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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

    # PR-K: restore aiter jit/build/ if it was moved aside during apply.
    # Done after source/artifact restore but before multi-node fan-out so
    # the host-local jit cache is back in place even if pod-side revert
    # subsequently fails. Manifest's ``jit_build_backup`` carries ``src``
    # and ``backup_path`` set by :func:`_invalidate_aiter_jit_build` and
    # is only present (with status=ok) when the apply actually moved the
    # dir aside.
    jit_build_backup = manifest.get("jit_build_backup") or {}
    if jit_build_backup.get("status") == "ok":
        jit_build_restore = _restore_aiter_jit_build(jit_build_backup)
        manifest["jit_build_restore"] = jit_build_restore
        if jit_build_restore.get("status") == "ok" and jit_build_restore.get("restored_to"):
            restored.append(str(jit_build_restore["restored_to"]))

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
    jit_build_backup: dict[str, Any] = {
        "status": "skipped", "reason": "rebuild not run",
    }
    if strategy["compiled"] and not skip_rebuild:
        # PR-K: aiter @compile_ops modules cache JIT-built .so under
        # <aiter>/jit/build/module_*/. setup.py develop rebuilds the
        # python package + statically-linked .so but does NOT touch
        # jit/build/, so a patched .cu under aiter/csrc/ would rebuild
        # yet the next import would still load the pre-patch .so. Move
        # jit/build/ aside before rebuild so the post-rebuild first
        # import re-codegens + re-compiles cleanly. No-op for sglang/
        # vllm targets (they have no JIT codegen layer).
        jit_build_backup = _invalidate_aiter_jit_build(target, backup_dir)
        if jit_build_backup.get("status") == "failed":
            # Refuse to rebuild against an inconsistent jit cache state:
            # restore source from backup so the on-disk file matches v0
            # again, then bail out.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error_class": "aiter_jit_invalidation_failed",
                "error": (
                    "aiter jit/build/ invalidation failed: "
                    f"{jit_build_backup.get('error')}"
                ),
                "manifest_path": str(manifest_path),
                "jit_build_backup": jit_build_backup,
            }
        if jit_build_backup.get("status") == "ok":
            # Persist the backup record into the manifest BEFORE rebuild
            # so a rebuild failure can still trigger restore via
            # revert_kernel_patch (which reads the manifest).
            manifest["jit_build_backup"] = jit_build_backup
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

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
    if jit_build_backup.get("status") in {"ok", "skipped"}:
        # Surface skipped reason too so manifest readers can audit why
        # invalidation didn't run on a particular apply (e.g. non-aiter
        # target, aiter not importable in this sandbox).
        manifest["jit_build_backup"] = jit_build_backup
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
        "jit_build_backup": jit_build_backup,
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
