#!/usr/bin/env python3
"""Multi-node TraceLens SGLang patch fan-out.

Counterpart to ``kernel_patch_multinode.py``. Submitted via Ray Dashboard
REST by ``inference_optimizer.multi_node apply-tracelens-patch`` when the
workload has ``nodes >= 2``.

Why this script exists
----------------------

The single-node ``_server_patcher.ensure_sglang_patched_for_tracelens``
runs inside the same Python process that imports SGLang (Hyperloom + SGLang
share the host). On multi-node, Hyperloom runs in the sandbox controller
and SGLang runs in head/worker pods; the controller cannot ``import
sglang`` so the local patcher silently skips ('sglang not importable').
Without the TraceLens patches SGLang's torch.profiler emits step
boundaries named ``step[DECODE bs=N]`` / ``step[EXTEND bs=N toks=M]``
which the TraceLens splitter does not recognise — splitter returns 0
steady-state chunks and ``tracelens_analysis.py`` raises
``trace_split_no_steady_state``, hanging the orchestration agent in a
``proposals=0`` loop.

This entrypoint mirrors ``kernel_patch_multinode.py``'s actor fan-out
pattern so the patches apply inside every pod (head + workers) where
SGLang actually lives. ``apply-tracelens-patch`` is idempotent: if the
sentinel marker is already present the actor returns ``status=skipped``
without re-applying, so calling it on every ``restart_server_for_round``
costs only the network round-trip + sentinel grep on already-patched
pods.

Algorithm
---------

  1. ``ray.init()`` (no address; in-pod).
  2. Enumerate all alive nodes (one actor per pod).
  3. For each pod, in a NodeAffinity-pinned actor:
     a. ``import sglang`` → resolve installed version and apply root.
     b. Resolve ``$TRACELENS_ROOT/<patch_tree>/sglang_<X_Y_Z>/*.patch``.
        Per-version subdirs are the v0.3.1+ layout; flat fallback is NOT
        supported here (matches Hyperloom main's
        ``_resolve_sglang_patches_dir`` post-c839a20 simplification).
     c. Sentinel check: if every required marker substring is already
        present in ``scheduler_profiler_mixin.py``, ``return
        status=skipped`` (already patched).
     d. ``git apply --check`` every patch; if any pre-check fails fall
        back to ``patch -p<N> --fuzz=2 --dry-run`` (point-release drift
        tolerance, same policy as ``_server_patcher._apply_atomic``).
     e. Apply every patch in order. On mid-set failure, reverse-apply
        the ones we already committed so the install stays consistent.
  4. Collect per-node results; emit a single JSON document on stdout.
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


# ---------------------------------------------------------------------------
# Sentinel markers — keep in sync with _server_patcher._discover_sglang_plan.
# A pod is treated as "already patched" iff every marker below is present in
# the sentinel file. PR-D §4: requiring ALL N markers raises the bar for
# false positives if upstream merges one identifier but not the whole patch.
# ---------------------------------------------------------------------------
_SENTINEL_RELPATH = "python/sglang/srt/managers/scheduler_profiler_mixin.py"
_SENTINEL_MARKERS: tuple[str, ...] = (
    "shape_discovery",
    "roofline_annotations",
)
# Extra check: the io_struct patch adds the field on the request schema. If
# only scheduler_profiler_mixin is patched but io_struct is missing, the
# request body fails to deserialise — guard against that partial state.
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

# Per ``git apply`` invocation timeout. ``git apply`` on a single SGLang
# patch is <1s in practice; allow generous headroom for I/O hiccups.
_GIT_TIMEOUT_SEC = 30


def _log(msg: str) -> None:
    """Stderr-only timestamped log line.

    stdout is reserved for the final JSON document the dashboard caller
    parses, so all progress chatter goes to stderr.

    Args:
        msg (str): The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[tracelens_patch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _versioned_patches_subdir_name(version: str) -> str | None:
    """Derive the per-version patches subdir name from a version string.

    ``0.5.11`` -> ``sglang_0_5_11``. Tolerates ``-rc1`` / ``+local``
    suffixes (same logic as _server_patcher._versioned_patches_subdir_name).

    Args:
        version (str): The installed sglang version string.

    Returns:
        str | None: The subdir name (e.g. ``sglang_0_5_11``), or ``None`` if
        the version cannot be parsed into dotted numeric components.
    """
    text = (version or "").strip()
    if not text:
        return None
    head = text.split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".") if head else []
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return "sglang_" + "_".join(parts)


def _resolve_sglang_install(sglang_module_path: Path) -> tuple[Path, int] | None:
    """Decide ``(apply_root, -p<N> strip)`` from any sglang anchor path.

    Accepts three input shapes:

    1. Top-level ``sglang/__init__.py`` (legacy wheel install). Walks up
       to land on ``site-packages/sglang/``, returns ``(<pkg_dir>, 3)``.
    2. A submodule ``__file__`` like
       ``.../python/sglang/srt/managers/scheduler_profiler_mixin.py``.
       Walks up to find the ``sglang/`` package dir, returns
       ``(<repo_root>, 1)`` for editable / ``(<pkg_dir>, 3)`` for wheel.
    3. A bare namespace-package directory like ``/sgl-workspace/sglang``
       (where ``sglang.__path__[0]`` points when the docker image ships
       sglang via ``sys.path`` insertion without ``__init__.py``).
       Probes ``<dir>/python/sglang/srt`` for editable layout and
       ``<dir>/srt`` for wheel-shape inside that dir.

    Returns ``None`` only if none of these layouts match — caller fail-softs.
    Mirrors the intent of ``_server_patcher._resolve_sglang_apply_root``
    but adds namespace-dir handling.

    Args:
        sglang_module_path (Path): An anchor path into the sglang install
            (a module ``__file__`` or a namespace-package directory).

    Returns:
        tuple[Path, int] | None: ``(apply_root, strip)`` where ``strip`` is
        the ``-p<N>`` level for ``git apply``, or ``None`` if no known
        layout matched.
    """
    resolved = sglang_module_path.resolve()
    # Pass 1: walk up the ancestor chain looking for the ``sglang/``
    # package dir (the one that contains a ``srt/`` subdir, which is the
    # canonical "this is really sglang" marker).
    pkg_dir: Path | None = None
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name == "sglang" and (ancestor / "srt").is_dir():
            pkg_dir = ancestor
            break

    # Pass 2: anchor is a namespace dir (no ``__init__.py``); search a few
    # well-known child paths for the real package root. ``__path__[0]`` of
    # a namespace-packaged sglang typically points HERE.
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
    """Run ``git <args>`` and capture its result.

    Never raises for a non-zero exit — caller inspects ``rc`` to branch.

    Args:
        args (tuple[str, ...]): Arguments appended after ``git``.
        cwd (Path): Working directory for the git invocation.

    Returns:
        tuple[int, str, str]: ``(returncode, stdout, stderr)``.
    """
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
    sglang_version_pin: str | None,
) -> dict[str, Any]:
    """Apply (or verify) the TraceLens SGLang patch set on THIS pod.

    Returns a dict summary (host, status, version, patches, skipped reason
    if any). Never raises — wraps failures into ``status=failed`` so the
    caller sees them via ``ray.get`` without exception propagation.

    Args:
        tracelens_root (str): Path to the TraceLens checkout that hosts the
            sglang roofline patch sets.
        sglang_version_pin (str | None): Optional advisory version pin;
            logged on mismatch but never enforced.

    Returns:
        dict[str, Any]: Per-pod summary including ``host``, ``status``
        (``applied`` / ``skipped`` / ``failed``), ``sglang_version``,
        ``patches_applied``, and ``error`` / ``elapsed_sec`` fields.
    """
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

        # Version resolution: SGLang ships as a namespace package in the
        # standard MI300X/MI355X docker images, so ``sglang.__version__``
        # and ``sglang.__file__`` are both ``None``. Fall through a chain:
        #   1. ``sglang.version.__version__`` (the real version module the
        #      package itself imports lazily)
        #   2. ``importlib.metadata.version("sglang")`` (PEP 566 / pip
        #      metadata; works for wheel installs)
        #   3. ``getattr(sglang, "__version__", "")`` (legacy editable
        #      installs that DO set it on the top-level module)
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

        # Install-root resolution: pick the most reliable ``__file__`` we
        # can grab. Namespace-packaged sglang has ``sglang.__file__ ==
        # None``; submodule ``__file__`` is always populated, so we pull
        # the location from ``sglang.srt.managers.scheduler_profiler_mixin``
        # which is the file we'll be patching anyway. Fall through to
        # ``sglang.__path__[0]`` if scheduler_profiler_mixin is not
        # importable (e.g. older sglang point release that ships the file
        # under a different path).
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
        # The strip count tells us how deep the patch ``a/`` prefix is
        # relative to ``apply_root``. Use the same path math the patches
        # themselves use to locate the sentinel:
        #   strip=1: apply_root is the repo root; sentinel under
        #       apply_root/python/sglang/srt/managers/...
        #   strip=3: apply_root is the wheel sglang/ dir; sentinel under
        #       apply_root/srt/managers/...
        if strip == 1:
            sentinel_path = apply_root / _SENTINEL_RELPATH
            extra_sentinel = apply_root / _EXTRA_SENTINEL_RELPATH
        else:
            # _SENTINEL_RELPATH starts with "python/sglang/"; strip two
            # leading segments to land under the wheel sglang/ dir.
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
        sglang_version_pin=sglang_version_pin,
    )


def _fanout_to_all_nodes(
    *,
    tracelens_root: str,
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
        help="path to TraceLens-internal checkout (default: $TRACELENS_ROOT)",
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
