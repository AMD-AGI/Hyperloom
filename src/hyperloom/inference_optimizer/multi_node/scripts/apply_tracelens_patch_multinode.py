#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node TraceLens SGLang patch fan-out.

Fans out one NodeAffinity-pinned actor per alive pod to apply the
TraceLens patches where SGLang lives: the roofline set below 0.5.18,
``sglang_gc_patch`` from 0.5.18 on. Each actor resolves the
sglang version + apply root, skips if the sentinel markers are already
present (idempotent), ``git apply --check``s then applies every
``$TRACELENS_ROOT/.../sglang_<X_Y_Z>/*.patch`` (rolling back on mid-set
failure). Emits one JSON summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# NOTE: ``ray`` is imported lazily inside ``_fanout_to_all_nodes`` only.


# Pod-side subset check: scheduler_profiler_mixin.py + io_struct.py only, and a pod counts as patched iff ALL markers
# are present.
_SENTINEL_RELPATH = "python/sglang/srt/managers/scheduler_profiler_mixin.py"
_SENTINEL_MARKERS: tuple[str, ...] = (
    "shape_discovery",
    "detailed_annotations",
)
# io_struct must also be patched, else the request body fails to deserialise.
_EXTRA_SENTINEL_RELPATH = "python/sglang/srt/managers/io_struct.py"
_EXTRA_SENTINEL_MARKERS: tuple[str, ...] = (
    "shape_discovery",
    "detailed_annotations",
)

# Path within the TraceLens checkout that hosts the patch sets.
_PATCH_TREE_REL = (
    "examples",
    "custom_workflows",
    "inference_analysis",
    "sglang_roofline_patches",
)
_SGLANG_GC_MIN_VERSION: tuple[int, ...] = (0, 5, 18)
_GC_PATCH_TREE_REL = (
    "examples",
    "custom_workflows",
    "inference_analysis",
    "sglang_gc_patch",
)
_GC_SENTINEL_RELPATH = "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
_GC_SENTINEL_MARKERS: tuple[str, ...] = (
    "_set_profile_trace_tag",
    "_profile_runner_name",
)

# Per ``git apply`` timeout.
_GIT_TIMEOUT_SEC = 30


def _log(msg: str) -> None:
    """Stderr-only timestamped log line (stdout is reserved for the final JSON)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[tracelens_patch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _versioned_patches_subdir_name(version: str) -> str | None:
    """``0.5.11`` -> ``sglang_0_5_11`` (tolerates ``-rc1`` / ``+local`` suffixes)."""
    text = (version or "").strip()
    if not text:
        return None
    head = text.split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".") if head else []
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return "sglang_" + "_".join(parts)


def _numeric_version_prefix(version: str) -> tuple[int, ...] | None:
    """Leading numeric components of a version (``0.5.21.dev1`` -> ``(0, 5, 21)``)."""
    text = (version or "").strip()
    if not text:
        return None
    head = text.split("-", 1)[0].split("+", 1)[0]
    numeric: list[int] = []
    for part in head.split("."):
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    return tuple(numeric) if len(numeric) >= 2 else None


def _uses_gc_patch(version: str) -> bool:
    """Pod-side mirror of the SGLang shape gate: sitecustomize mode gets ``sglang_gc_patch``."""
    override = os.environ.get("HYPERLOOM_SGLANG_SHAPE_MODE", "auto").strip().lower()
    if override in {"patch", "patched"}:
        return False
    if override == "sitecustomize":
        return True
    running = _numeric_version_prefix(version)
    return running is not None and running >= _SGLANG_GC_MIN_VERSION


def _subdir_version_tuple(name: str) -> tuple[int, ...] | None:
    """``sglang_0_5_21`` -> ``(0, 5, 21)``. A ``_sgldev`` suffix stops the run."""
    head = "sglang_"
    if not name.startswith(head):
        return None
    numeric: list[int] = []
    for part in name[len(head) :].split("_"):
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    return tuple(numeric) if len(numeric) >= 2 else None


def _resolve_gc_patches_dir(patches_root: Path, version: str) -> Path | None:
    """Exact ``sglang_<X_Y_Z>`` dir, else the nearest not-newer one."""
    running = _numeric_version_prefix(version)
    if running is None or not patches_root.is_dir():
        return None
    base = "sglang_" + "_".join(str(part) for part in running)
    names = [base]
    text = version or ""
    if ".dev" in text or "+g" in text:
        names = [f"{base}_sgldev", base]
    for name in names:
        candidate = patches_root / name
        if candidate.is_dir() and any(candidate.glob("*.patch")):
            return candidate
    available: dict[tuple[int, ...], Path] = {}
    for entry in sorted(patches_root.iterdir()):
        if not entry.is_dir() or entry.name.endswith("_sgldev"):
            continue
        if not any(entry.glob("*.patch")):
            continue
        vt = _subdir_version_tuple(entry.name)
        if vt:
            available[vt] = entry
    if not available:
        return None
    same_minor = [vt for vt in available if vt[:2] == running[:2] and vt <= running]
    if same_minor:
        return available[max(same_minor)]
    older = [vt for vt in available if vt <= running]
    if older:
        return available[max(older)]
    return None


def _resolve_sglang_install(sglang_module_path: Path) -> tuple[Path, int] | None:
    """Decide ``(apply_root, -p<N> strip)`` from any sglang anchor (wheel/editable/namespace-dir layouts); ``None`` if unrecognised."""
    resolved = sglang_module_path.resolve()
    # Pass 1: walk up to the ``sglang/`` package dir (the one with a ``srt/`` subdir).
    pkg_dir: Path | None = None
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name == "sglang" and (ancestor / "srt").is_dir():
            pkg_dir = ancestor
            break

    # Pass 2: anchor is a namespace dir; probe well-known child paths.
    if pkg_dir is None and resolved.is_dir():
        editable_inside = resolved / "python" / "sglang"
        wheel_inside = resolved / "sglang"  # rare: anchor is site-packages parent
        for cand in (editable_inside, wheel_inside, resolved):
            if (cand / "srt").is_dir():
                pkg_dir = cand
                break

    if pkg_dir is None:
        return None
    # editable: .../<repo_root>/python/sglang/...
    if pkg_dir.parent.name == "python":
        repo_root = pkg_dir.parent.parent
        if (repo_root / "python" / "sglang").is_dir():
            return repo_root, 1
    # wheel: .../site-packages/sglang/...
    return pkg_dir, 3


def _all_markers_present(path: Path, markers: tuple[str, ...]) -> bool:
    """Check whether a file contains every marker substring."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all(m in text for m in markers)


def _safe_directory_args(args: tuple[str, ...], cwd: Path) -> list[str]:
    """Prepend a ``safe.directory`` exception for the checkout at ``cwd``."""
    try:
        from hyperloom.common.git_safety import safe_directory_args
    except ImportError:
        return list(args)
    return safe_directory_args(list(args), cwd=cwd)


def _run_git(args: tuple[str, ...], cwd: Path) -> tuple[int, str, str]:
    """Run ``git <args>``; return ``(rc, stdout, stderr)`` (never raises on non-zero exit)."""
    proc = subprocess.run(
        ["git", *_safe_directory_args(args, cwd)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _apply_on_pod(
    *,
    tracelens_root: str,
    tracelens_internal_root: str,
    sglang_version_pin: str | None,
) -> dict[str, Any]:
    """Apply (or verify) the TraceLens SGLang patch set on this pod; never raises (failures become ``status=failed``)."""
    host = socket.gethostname()
    started = time.time()
    result: dict[str, Any] = {
        "host": host,
        "status": "unknown",
        "sglang_version": None,
        "patches_applied": [],
        "patches_skipped_already_present": False,
        "error": None,
        "elapsed_sec": 0.0,
    }
    try:
        try:
            import sglang  # type: ignore
        except Exception as e:  # noqa: BLE001
            result["status"] = "failed"
            result["error"] = f"sglang not importable: {e}"
            return result

        # Version resolution: sglang.version, then pip metadata, then attr.
        version = ""
        try:
            from sglang.version import __version__ as _sv  # type: ignore[import-not-found]

            version = (_sv or "").strip()
        except Exception:  # noqa: BLE001
            pass
        if not version:
            try:
                import importlib.metadata as _md

                version = (_md.version("sglang") or "").strip()
            except Exception:  # noqa: BLE001
                pass
        if not version:
            version = (getattr(sglang, "__version__", "") or "").strip()
        result["sglang_version"] = version or None
        if sglang_version_pin and version and version != sglang_version_pin:
            _log(f"version pin {sglang_version_pin!r} != installed {version!r} — proceeding (pin is advisory)")

        # Install-root anchor: sglang.__file__, else the scheduler_profiler_mixin file, else sglang.__path__[0].
        anchor_path: Path | None = None
        if sglang.__file__:  # editable layout
            anchor_path = Path(sglang.__file__)
        else:
            try:
                import sglang.srt.managers.scheduler_profiler_mixin as _spm  # type: ignore[import-not-found]

                if _spm.__file__:
                    anchor_path = Path(_spm.__file__)
            except Exception:  # noqa: BLE001
                pass
        if anchor_path is None:
            sp = list(getattr(sglang, "__path__", []) or [])
            if sp:
                anchor_path = Path(sp[0])
        if anchor_path is None:
            result["status"] = "failed"
            result["error"] = (
                "cannot locate sglang install root (sglang.__file__ is None, "
                "scheduler_profiler_mixin not importable, __path__ empty)"
            )
            return result
        layout = _resolve_sglang_install(anchor_path)
        if layout is None:
            result["status"] = "failed"
            result["error"] = (
                f"unrecognised sglang layout at {anchor_path} "
                "(expected editable .../python/sglang/... or wheel "
                ".../site-packages/sglang/...)"
            )
            return result
        apply_root, strip = layout
        graph_capture = _uses_gc_patch(version)
        result["patch_set"] = "graph-capture" if graph_capture else "roofline"
        sentinel_rel = _GC_SENTINEL_RELPATH if graph_capture else _SENTINEL_RELPATH
        sentinel_markers = _GC_SENTINEL_MARKERS if graph_capture else _SENTINEL_MARKERS
        # strip=1: apply_root is the repo root; strip=3: the wheel sglang/ dir.
        if strip == 1:
            sentinel_path = apply_root / sentinel_rel
            extra_sentinel = apply_root / _EXTRA_SENTINEL_RELPATH
        else:
            sentinel_path = apply_root / Path(*Path(sentinel_rel).parts[2:])
            extra_sentinel = apply_root / Path(*Path(_EXTRA_SENTINEL_RELPATH).parts[2:])

        extra_ok = graph_capture or _all_markers_present(extra_sentinel, _EXTRA_SENTINEL_MARKERS)
        if _all_markers_present(sentinel_path, sentinel_markers) and extra_ok:
            result["status"] = "skipped"
            result["patches_skipped_already_present"] = True
            return result

        if graph_capture:
            patches_dir = _resolve_gc_patches_dir(Path(tracelens_root, *_GC_PATCH_TREE_REL), version)
            if patches_dir is None:
                result["status"] = "failed"
                result["error"] = (
                    f"no sglang_gc_patch set for SGLang {version!r} under {Path(tracelens_root, *_GC_PATCH_TREE_REL)}"
                )
                return result
        else:
            subdir = _versioned_patches_subdir_name(version)
            if subdir is None:
                result["status"] = "failed"
                result["error"] = f"cannot derive per-version patches subdir from version {version!r}"
                return result
            patches_dir = Path(tracelens_root, *_PATCH_TREE_REL, subdir)
            if not patches_dir.is_dir():
                result["status"] = "failed"
                result["error"] = (
                    f"TraceLens patches dir missing: {patches_dir} (upgrade TraceLens to Hyperloom_integration_v0.3.1+)"
                )
                return result
        patches = tuple(sorted(patches_dir.glob("*.patch")))
        if not patches:
            result["status"] = "failed"
            result["error"] = f"no *.patch files in {patches_dir}"
            return result

        # Pre-check every patch (atomic-transaction policy: all or none).
        strip_arg = f"-p{strip}"
        for p in patches:
            rc, _, stderr = _run_git(
                ("apply", "--check", strip_arg, str(p)),
                apply_root,
            )
            if rc != 0:
                result["status"] = "failed"
                result["error"] = f"git apply --check {strip_arg} {p.name} failed (rc={rc}): {stderr.strip()[:240]}"
                return result

        # Apply for real; track applied for rollback on mid-set failure.
        applied: list[Path] = []
        for p in patches:
            rc, _, stderr = _run_git(
                ("apply", strip_arg, str(p)),
                apply_root,
            )
            if rc != 0:
                _log(f"apply failed at {p.name} (rc={rc}); rolling back {len(applied)} patches")
                for prev in reversed(applied):
                    _run_git(
                        ("apply", "-R", strip_arg, str(prev)),
                        apply_root,
                    )
                result["status"] = "failed"
                result["error"] = f"git apply {strip_arg} {p.name} failed (rc={rc}): {stderr.strip()[:240]}"
                return result
            applied.append(p)
            result["patches_applied"].append(p.name)

        result["status"] = "applied"
        return result
    except Exception as e:  # noqa: BLE001 - actor must never raise
        result["status"] = "failed"
        result["error"] = f"unexpected: {type(e).__name__}: {e}"
        return result
    finally:
        result["elapsed_sec"] = round(time.time() - started, 3)


# Ray actor scaffolding (per-node fan-out).
def _fanout_to_all_nodes(
    *,
    tracelens_root: str,
    tracelens_internal_root: str,
    sglang_version_pin: str | None,
) -> list[dict[str, Any]]:
    """Spawn one actor per alive node; collect all summaries."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    actor_apply = ray.remote(_apply_on_pod)
    ray.init(address="auto", ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("ray.nodes() returned no alive nodes")
    _log(f"discovered {len(nodes)} alive node(s); fanning out")

    actors = []
    for n in nodes:
        node_id = n["NodeID"]
        opts = actor_apply.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        )
        actors.append(
            opts.remote(
                tracelens_root=tracelens_root,
                tracelens_internal_root=tracelens_internal_root,
                sglang_version_pin=sglang_version_pin,
            )
        )
    results = ray.get(actors)
    return list(results)


def main() -> int:
    """Parse CLI arguments, fan out the patch to all pods, and aggregate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracelens-root",
        default=os.environ.get("TRACELENS_ROOT", ""),
        help="path to public TraceLens checkout (default: $TRACELENS_ROOT)",
    )
    parser.add_argument(
        "--tracelens-internal-root",
        default=os.environ.get("TRACELENS_INTERNAL_ROOT", ""),
        help="path to TraceLens-internal checkout (default: $TRACELENS_INTERNAL_ROOT). "
        "Reserved for future use; not consumed by current patch logic.",
    )
    parser.add_argument(
        "--sglang-version-pin",
        default=os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "") or None,
        help="optional advisory pin (e.g. '0.5.11'); logged on mismatch",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="patch THIS pod only (no ray fan-out). Used by the infera SSH "
        "backend, which ships+runs this script on each GPU pod directly. "
        "ray is never imported in this mode.",
    )
    args = parser.parse_args()

    if not args.tracelens_root or not Path(args.tracelens_root).is_dir():
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": (f"--tracelens-root invalid or missing: {args.tracelens_root!r}"),
                    "per_pod": [],
                },
                indent=2,
            )
        )
        return 2

    if shutil.which("git") is None:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "git not on PATH in pod image",
                    "per_pod": [],
                },
                indent=2,
            )
        )
        return 2

    # --local: infera SSH backend.
    if args.local:
        r = _apply_on_pod(
            tracelens_root=args.tracelens_root,
            tracelens_internal_root=args.tracelens_internal_root,
            sglang_version_pin=args.sglang_version_pin or None,
        )
        overall = r.get("status") if r.get("status") in ("applied", "skipped") else "failed"
        print(json.dumps({"status": overall, "per_pod": [r]}, indent=2, sort_keys=True))
        return 0 if overall in ("applied", "skipped") else 1

    try:
        per_pod = _fanout_to_all_nodes(
            tracelens_root=args.tracelens_root,
            tracelens_internal_root=args.tracelens_internal_root,
            sglang_version_pin=args.sglang_version_pin or None,
        )
    except Exception as e:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"ray fan-out aborted: {type(e).__name__}: {e}",
                    "per_pod": [],
                },
                indent=2,
            )
        )
        return 3

    # Aggregate: ``applied`` only if every pod applied or skipped.
    overall = "applied"
    any_fresh = False
    for r in per_pod:
        if r.get("status") == "applied":
            any_fresh = True
        elif r.get("status") != "skipped":
            overall = "failed"
            break
    if overall == "applied" and not any_fresh:
        overall = "skipped"  # every pod already patched

    print(
        json.dumps(
            {
                "status": overall,
                "per_pod": per_pod,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if overall in ("applied", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
