#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node TraceLens SGLang patch fan-out.

Fans out one NodeAffinity-pinned actor per alive pod to apply the
TraceLens patches where SGLang lives: the roofline set or
``sglang_gc_patch``, as chosen by the controller's ``--patch-set`` (when
omitted, roofline below 0.5.18 and ``sglang_gc_patch`` from 0.5.18 on,
gated on each pod's own SGLang). Each actor resolves the
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
import traceback
from pathlib import Path
from typing import Any

from sglang_shape_gate import SGLANG_SITECUSTOMIZE_MIN_VERSION

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
# ``--patch-set`` values; the controller resolves its shape mode once and passes one of these.
_PATCH_SET_ROOFLINE = "roofline"
_PATCH_SET_GRAPH_CAPTURE = "graph-capture"
_PATCH_SETS: tuple[str, ...] = (_PATCH_SET_ROOFLINE, _PATCH_SET_GRAPH_CAPTURE)

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
    """Fallback when no ``--patch-set`` is given: mirror the SGLang shape gate from the pod's own version."""
    running = _numeric_version_prefix(version)
    return running is not None and running >= SGLANG_SITECUSTOMIZE_MIN_VERSION


def _resolve_gc_patches_dir(patches_root: Path, version: str) -> Path | None:
    """Exact ``sglang_<X_Y_Z>`` dir (``_sgldev`` first for dev builds), else ``None``.

    No nearest-version fallback: the patch adds an unconditional call to the
    capture loop, and a newer release may already carry the upstream fix.
    """
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
    # Wheel install, e.g. .../site-packages/sglang/...
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


def _installed_sglang_version(sglang_module: Any) -> str:
    """The installed SGLang version: ``sglang.version``, then pip metadata, then the module attribute."""
    try:
        from sglang.version import __version__ as module_version  # type: ignore[import-not-found]

        version = (module_version or "").strip()
    except ImportError:
        version = ""
    if not version:
        try:
            import importlib.metadata as _md

            version = (_md.version("sglang") or "").strip()
        except _md.PackageNotFoundError:
            version = ""
    if not version:
        version = (getattr(sglang_module, "__version__", "") or "").strip()
    return version


def _sglang_anchor_path(sglang_module: Any) -> Path | None:
    """A file inside the sglang install, which its layout is resolved from.

    ``sglang.__file__`` in an editable layout, else the scheduler profiler
    mixin, else the first entry of ``__path__``.
    """
    if sglang_module.__file__:  # editable layout
        return Path(sglang_module.__file__)
    try:
        import sglang.srt.managers.scheduler_profiler_mixin as _spm  # type: ignore[import-not-found]

        if _spm.__file__:
            return Path(_spm.__file__)
    except ImportError:
        pass
    search_path = list(getattr(sglang_module, "__path__", []) or [])
    return Path(search_path[0]) if search_path else None


def _resolve_patches_dir(tracelens_root: str, version: str, *, graph_capture: bool) -> tuple[Path | None, str, str]:
    """The patch directory for this SGLang version, as ``(dir, skip_reason, error)``.

    A skip reason is not a failure: a release with no TraceLens graph-capture
    set runs unpatched.
    """
    if graph_capture:
        tree = Path(tracelens_root, *_GC_PATCH_TREE_REL)
        patches_dir = _resolve_gc_patches_dir(tree, version)
        if patches_dir is None:
            return None, f"no sglang_gc_patch set for SGLang {version!r} under {tree}", ""
        return patches_dir, "", ""
    subdir = _versioned_patches_subdir_name(version)
    if subdir is None:
        return None, "", f"cannot derive per-version patches subdir from version {version!r}"
    patches_dir = Path(tracelens_root, *_PATCH_TREE_REL, subdir)
    if not patches_dir.is_dir():
        return (
            None,
            "",
            f"TraceLens patches dir missing: {patches_dir} (upgrade TraceLens to Hyperloom_integration_v0.3.1+)",
        )
    return patches_dir, "", ""


def _apply_patch_set(patches: tuple[Path, ...], apply_root: Path, strip: int) -> tuple[list[str], str]:
    """Apply every patch or none, and say which landed.

    The whole set is pre-checked first, then applied; a failure mid-set rolls
    back what landed, because a half-patched install runs a mix of two SGLang
    versions.
    """
    strip_arg = f"-p{strip}"
    for patch in patches:
        rc, _, stderr = _run_git(("apply", "--check", strip_arg, str(patch)), apply_root)
        if rc != 0:
            return [], f"git apply --check {strip_arg} {patch.name} failed (rc={rc}): {stderr.strip()[:240]}"
    applied: list[Path] = []
    for patch in patches:
        rc, _, stderr = _run_git(("apply", strip_arg, str(patch)), apply_root)
        if rc != 0:
            _log(f"apply failed at {patch.name} (rc={rc}); rolling back {len(applied)} patches")
            for previous in reversed(applied):
                _run_git(("apply", "-R", strip_arg, str(previous)), apply_root)
            error = f"git apply {strip_arg} {patch.name} failed (rc={rc}): {stderr.strip()[:240]}"
            return [path.name for path in applied], error
        applied.append(patch)
    return [path.name for path in applied], ""


def _sentinel_path(apply_root: Path, strip: int, relpath: str) -> Path:
    """``relpath`` (repo-relative) under ``apply_root``; strip=1: apply_root is the repo root; strip=3: the wheel sglang/ dir."""
    return apply_root / Path(*Path(relpath).parts[strip - 1 :])


def _markers_already_applied(apply_root: Path, strip: int, *, graph_capture: bool) -> bool:
    """True when the patch-set sentinel markers are already present under ``apply_root``."""
    rel = _GC_SENTINEL_RELPATH if graph_capture else _SENTINEL_RELPATH
    markers = _GC_SENTINEL_MARKERS if graph_capture else _SENTINEL_MARKERS
    return _all_markers_present(_sentinel_path(apply_root, strip, rel), markers) and (
        graph_capture
        or _all_markers_present(_sentinel_path(apply_root, strip, _EXTRA_SENTINEL_RELPATH), _EXTRA_SENTINEL_MARKERS)
    )


def _sglang_apply_layout(sglang: Any) -> tuple[tuple[Path, int] | None, str]:
    """``(apply_root, strip)`` for imported sglang, or ``(None, error)``."""
    anchor_path = _sglang_anchor_path(sglang)
    if anchor_path is None:
        return None, (
            "cannot locate sglang install root (sglang.__file__ is None, "
            "scheduler_profiler_mixin not importable, __path__ empty)"
        )
    layout = _resolve_sglang_install(anchor_path)
    if layout is None:
        return None, (
            f"unrecognised sglang layout at {anchor_path} "
            "(expected editable .../python/sglang/... or wheel .../site-packages/sglang/...)"
        )
    return layout, ""


def _apply_on_pod(
    *,
    tracelens_root: str,
    sglang_version_pin: str | None,
    patch_set: str | None = None,
) -> dict[str, Any]:
    """Apply (or verify) the TraceLens SGLang patch set on this pod; never raises (failures become ``status=failed``).

    ``patch_set`` is the controller's choice and wins over the pod-side gate.
    """
    started = time.time()
    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "status": "unknown",
        "sglang_version": None,
        "patches_applied": [],
        "patches_skipped_already_present": False,
        "error": None,
        "elapsed_sec": 0.0,
    }
    try:
        try:
            import sglang  # type: ignore[import-not-found]
        except (ImportError, OSError) as e:
            result["status"] = "failed"
            result["error"] = f"sglang not importable: {e}"
            return result
        version = _installed_sglang_version(sglang)
        result["sglang_version"] = version or None
        if sglang_version_pin and version and version != sglang_version_pin:
            _log(f"version pin {sglang_version_pin!r} != installed {version!r} — proceeding (pin is advisory)")
        layout, layout_error = _sglang_apply_layout(sglang)
        if layout is None:
            result["status"] = "failed"
            result["error"] = layout_error
            return result
        apply_root, strip = layout
        graph_capture = (patch_set == _PATCH_SET_GRAPH_CAPTURE) if patch_set else _uses_gc_patch(version)
        result["patch_set"] = _PATCH_SET_GRAPH_CAPTURE if graph_capture else _PATCH_SET_ROOFLINE
        if _markers_already_applied(apply_root, strip, graph_capture=graph_capture):
            result["status"] = "skipped"
            result["patches_skipped_already_present"] = True
            return result
        patches_dir, skip_reason, error = _resolve_patches_dir(tracelens_root, version, graph_capture=graph_capture)
        if skip_reason:
            result["status"] = "skipped"
            result["skip_reason"] = skip_reason
            _log(f"{skip_reason}; continuing unpatched")
            return result
        if patches_dir is None:
            result["status"] = "failed"
            result["error"] = error
            return result
        patches = tuple(sorted(patches_dir.glob("*.patch")))
        if not patches:
            result["status"] = "failed"
            result["error"] = f"no *.patch files in {patches_dir}"
            return result
        applied, apply_error = _apply_patch_set(patches, apply_root, strip)
        result["patches_applied"] = applied
        if apply_error:
            result["status"] = "failed"
            result["error"] = apply_error
            return result
        result["status"] = "applied"
        return result
    except Exception as e:  # noqa: BLE001 - actor must never raise
        result["status"] = "failed"
        result["error"] = f"unexpected: {type(e).__name__}: {e}"
        _log(f"{result['error']}\n{traceback.format_exc().rstrip()}")
        return result
    finally:
        result["elapsed_sec"] = round(time.time() - started, 3)


# Ray actor scaffolding (per-node fan-out).
def _fanout_to_all_nodes(
    *,
    tracelens_root: str,
    sglang_version_pin: str | None,
    patch_set: str | None = None,
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
                sglang_version_pin=sglang_version_pin,
                patch_set=patch_set,
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
        "--sglang-version-pin",
        default=os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "") or None,
        help="optional advisory pin (e.g. '0.5.11'); logged on mismatch",
    )
    parser.add_argument(
        "--patch-set",
        choices=_PATCH_SETS,
        default=None,
        help="patch set resolved by the controller; when omitted each pod gates on its own SGLang version",
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
            sglang_version_pin=args.sglang_version_pin or None,
            patch_set=args.patch_set,
        )
        overall = r.get("status") if r.get("status") in ("applied", "skipped") else "failed"
        print(json.dumps({"status": overall, "per_pod": [r]}, indent=2, sort_keys=True))
        return 0 if overall in ("applied", "skipped") else 1

    try:
        per_pod = _fanout_to_all_nodes(
            tracelens_root=args.tracelens_root,
            sglang_version_pin=args.sglang_version_pin or None,
            patch_set=args.patch_set,
        )
    except Exception as e:  # noqa: BLE001 - outermost CLI main: the Ray fan-out, reported as failed JSON
        _log(f"ray fan-out aborted:\n{traceback.format_exc().rstrip()}")
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
