#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node TraceLens SGLang patch fan-out.

The single-node patcher can't reach SGLang across pods, so this fans out
one NodeAffinity-pinned actor per alive pod to apply the TraceLens
roofline patches where SGLang lives (else the trace splitter sees no
steady-state and the agent hangs at proposals=0). Each actor resolves the
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

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# Sentinel markers (keep in sync with _server_patcher._discover_sglang_plan).
# A pod counts as patched iff ALL markers are present (PR-D §4: avoids false
# positives when upstream merges one identifier but not the whole patch).
_SENTINEL_RELPATH = "python/sglang/srt/managers/scheduler_profiler_mixin.py"
_SENTINEL_MARKERS: tuple[str, ...] = (
    "shape_discovery",
    "roofline_annotations",
)
# Extra check: io_struct must also be patched, else the request body fails to deserialise.
_EXTRA_SENTINEL_RELPATH = "python/sglang/srt/managers/io_struct.py"
_EXTRA_SENTINEL_MARKERS: tuple[str, ...] = (
    "shape_discovery",
    "roofline_annotations",
)

# Path within the TraceLens checkout that hosts the patch sets.
_PATCH_TREE_REL = (
    "examples", "custom_workflows", "inference_analysis",
    "sglang_roofline_patches",
)

# Per ``git apply`` timeout (a single patch is <1s; headroom for I/O hiccups).
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
    """Check whether a file contains every marker substring.

    Args:
        path (Path): File to inspect.
        markers (tuple[str, ...]): Substrings that must all be present.

    Returns:
        bool: ``True`` iff ``path`` is readable and contains every marker;
        ``False`` otherwise (including when the file cannot be read).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all(m in text for m in markers)


def _run_git(args: tuple[str, ...], cwd: Path) -> tuple[int, str, str]:
    """Run ``git <args>``; return ``(rc, stdout, stderr)`` (never raises on non-zero exit)."""
    proc = subprocess.run(  # noqa: S603
        ["git", *args],
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
            import sglang  # type: ignore  # noqa: I001 - runtime probe
        except Exception as e:  # noqa: BLE001
            result["status"] = "failed"
            result["error"] = f"sglang not importable: {e}"
            return result

        # Version resolution chain (namespace-packaged sglang has __version__
        # = None): sglang.version, then pip metadata, then top-level attr.
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
            _log(
                f"version pin {sglang_version_pin!r} != installed {version!r} "
                "— proceeding (pin is advisory)"
            )

        # Install-root anchor: sglang.__file__ (editable), else the
        # scheduler_profiler_mixin submodule file, else sglang.__path__[0].
        anchor_path: Path | None = None
        if sglang.__file__:  # legacy editable layout
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
        # strip=1: apply_root is the repo root; strip=3: the wheel sglang/ dir
        # (drop the leading "python/sglang/" segments for the sentinel path).
        if strip == 1:
            sentinel_path = apply_root / _SENTINEL_RELPATH
            extra_sentinel = apply_root / _EXTRA_SENTINEL_RELPATH
        else:
            sentinel_path = apply_root / Path(*Path(_SENTINEL_RELPATH).parts[2:])
            extra_sentinel = apply_root / Path(*Path(_EXTRA_SENTINEL_RELPATH).parts[2:])

        if (
            _all_markers_present(sentinel_path, _SENTINEL_MARKERS)
            and _all_markers_present(extra_sentinel, _EXTRA_SENTINEL_MARKERS)
        ):
            result["status"] = "skipped"
            result["patches_skipped_already_present"] = True
            return result

        subdir = _versioned_patches_subdir_name(version)
        if subdir is None:
            result["status"] = "failed"
            result["error"] = (
                f"cannot derive per-version patches subdir from version "
                f"{version!r}"
            )
            return result
        patches_dir = Path(tracelens_root, *_PATCH_TREE_REL, subdir)
        if not patches_dir.is_dir():
            result["status"] = "failed"
            result["error"] = (
                f"TraceLens patches dir missing: {patches_dir} "
                "(upgrade TraceLens to Hyperloom_integration_v0.3.1+)"
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
                ("apply", "--check", strip_arg, str(p)), apply_root,
            )
            if rc != 0:
                result["status"] = "failed"
                result["error"] = (
                    f"git apply --check {strip_arg} {p.name} failed "
                    f"(rc={rc}): {stderr.strip()[:240]}"
                )
                return result

        # Apply for real; track applied for rollback on mid-set failure.
        applied: list[Path] = []
        for p in patches:
            rc, _, stderr = _run_git(
                ("apply", strip_arg, str(p)), apply_root,
            )
            if rc != 0:
                _log(f"apply failed at {p.name} (rc={rc}); rolling back {len(applied)} patches")
                for prev in reversed(applied):
                    _run_git(
                        ("apply", "-R", strip_arg, str(prev)), apply_root,
                    )
                result["status"] = "failed"
                result["error"] = (
                    f"git apply {strip_arg} {p.name} failed (rc={rc}): "
                    f"{stderr.strip()[:240]}"
                )
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


# ---------------------------------------------------------------------------
# Ray actor scaffolding (mirrors kernel_patch_multinode.py's per-node fan-out)
# ---------------------------------------------------------------------------
@ray.remote
def _actor_apply(
    *,
    tracelens_root: str,
    tracelens_internal_root: str,
    sglang_version_pin: str | None,
) -> dict[str, Any]:
    """Ray actor entrypoint that patches one pod.

    Pinned to one node via NodeAffinityScheduling so every pod (head +
    workers) runs the patcher exactly once.

    Args:
        tracelens_root (str): Path to the TraceLens checkout on the pod.
        sglang_version_pin (str | None): Optional advisory version pin.

    Returns:
        dict[str, Any]: The per-pod summary from :func:`_apply_on_pod`.
    """
    return _apply_on_pod(
        tracelens_root=tracelens_root,
        tracelens_internal_root=tracelens_internal_root,
        sglang_version_pin=sglang_version_pin,
    )


def _fanout_to_all_nodes(
    *,
    tracelens_root: str,
    tracelens_internal_root: str,
    sglang_version_pin: str | None,
) -> list[dict[str, Any]]:
    """Spawn one actor per alive node and collect all summaries.

    Args:
        tracelens_root (str): Path to the TraceLens checkout on the pods.
        sglang_version_pin (str | None): Optional advisory version pin.

    Returns:
        list[dict[str, Any]]: One per-pod summary dict per alive node.

    Raises:
        RuntimeError: If ``ray.nodes()`` reports no alive nodes.
    """
    ray.init(address="auto", ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("ray.nodes() returned no alive nodes")
    _log(f"discovered {len(nodes)} alive node(s); fanning out")

    actors = []
    for n in nodes:
        node_id = n["NodeID"]
        opts = _actor_apply.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        )
        actors.append(opts.remote(
            tracelens_root=tracelens_root,
            tracelens_internal_root=tracelens_internal_root,
            sglang_version_pin=sglang_version_pin,
        ))
    results = ray.get(actors)
    return list(results)


def main() -> int:
    """Parse CLI arguments, fan out the patch to all pods, and aggregate.

    Validates ``--tracelens-root`` and the presence of ``git``, fans the
    patcher out to every alive node, then prints an aggregate JSON document
    (overall status plus per-pod summaries) to stdout.

    Returns:
        int: ``0`` if every pod was applied or skipped; ``1`` on a per-pod
        failure; ``2`` for invalid inputs / missing git; ``3`` if the Ray
        fan-out itself aborted.
    """
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
    args = parser.parse_args()

    if not args.tracelens_root or not Path(args.tracelens_root).is_dir():
        print(json.dumps({
            "status": "failed",
            "error": (
                f"--tracelens-root invalid or missing: {args.tracelens_root!r}"
            ),
            "per_pod": [],
        }, indent=2))
        return 2

    if shutil.which("git") is None:
        print(json.dumps({
            "status": "failed",
            "error": "git not on PATH in pod image",
            "per_pod": [],
        }, indent=2))
        return 2

    try:
        per_pod = _fanout_to_all_nodes(
            tracelens_root=args.tracelens_root,
            tracelens_internal_root=args.tracelens_internal_root,
            sglang_version_pin=args.sglang_version_pin or None,
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({
            "status": "failed",
            "error": f"ray fan-out aborted: {type(e).__name__}: {e}",
            "per_pod": [],
        }, indent=2))
        return 3

    # Aggregate: overall is ``applied`` only if every pod succeeded
    # (``applied`` or ``skipped``); otherwise ``failed``.
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

    print(json.dumps({
        "status": overall,
        "per_pod": per_pod,
    }, indent=2, sort_keys=True))
    return 0 if overall in ("applied", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
