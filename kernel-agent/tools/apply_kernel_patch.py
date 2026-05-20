#!/usr/bin/env python3
"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import argparse
import hashlib
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
KNOWN_TARGET_ROOTS = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
)


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
_MN_STATE_FILE = Path("/tmp/multi_node_state.json")


def _is_multi_node() -> bool:
    """True iff a multi-node RayJob is active (nodes >= 2).

    Reads ``/tmp/multi_node_state.json`` (the same checkpoint
    inference_optimizer.multi_node.cli writes after create-rayjob).
    Missing file / unreadable / nodes < 2 → False, so single-node and
    standalone CLI use of this tool keep their pre-multinode behaviour
    bit-for-bit.
    """
    try:
        if not _MN_STATE_FILE.is_file():
            return False
        data = json.loads(_MN_STATE_FILE.read_text(encoding="utf-8"))
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
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel"


def _path_hash(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


def _source_text_looks_complete(text: str, suffix: str) -> bool:
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


def _detect_strategy(target_file: Path, *, allow_unknown_target: bool) -> dict[str, Any]:
    target = str(target_file)
    lower = target.lower()
    if not allow_unknown_target and not any(root in lower for root in KNOWN_TARGET_ROOTS):
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
