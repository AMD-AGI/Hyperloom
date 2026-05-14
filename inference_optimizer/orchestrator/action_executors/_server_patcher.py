"""Idempotent run-time patcher for vLLM and SGLang server installs
(Hyperloom issue #194 §4 / §5).

Background
----------

The TraceLens magpie-benchmark-profiling skill requires three flags
that are *not* in upstream vLLM / SGLang:

* vLLM:   ``--profiler-config.capture_torch_profiler_dir``
* vLLM:   ``--profiler-config.detailed_trace_annotation``
* SGLang: ``--enable-shape-discovery-for-cuda-graph-profile``

These only exist in builds that have the TraceLens patch set applied
(``TraceLens-internal/examples/custom_workflows/inference_analysis/``).
Without the patch, vLLM rejects ``capture_torch_profiler_dir`` as an
"unknown JSON key" and SGLang ``argparse`` errors on the unknown flag —
the server fails to start and the entire profile run is wasted.

TraceLens's official path is "rebuild a docker image with the patch
baked in" via ``build_docker_vllm.sh`` / ``build_docker_sglang_*.sh``.
This module gives Hyperloom users an automatic fallback: at the start
of every profile run we runtime-patch the in-container Python install
so the flags become available. Users who already run a TraceLens-
patched image short-circuit on the idempotent check after the first
session.

Design contract
---------------

* **Per-framework, independent**: vLLM and SGLang have separate
  patchers. The caller invokes only the one matching the YAML's
  framework (no point patching SGLang if we're running vLLM).
* **Default on, kill-switchable**: caller is responsible for honouring
  ``HYPERLOOM_ENABLE_PATCH != "0"`` — this module just answers
  "did patching succeed?" so the caller knows whether to inject the
  TraceLens-only flags.
* **Fail-soft**: any failure (TraceLens repo missing, version
  unsupported, install layout unexpected, patches don't apply
  cleanly, file not writable, ``git`` unavailable) returns ``False``.
  Callers MUST treat ``False`` as "skip TraceLens flags" so the
  benchmark continues with today's safe behaviour.
* **Idempotent**: a sentinel-substring check on a known patched file
  short-circuits subsequent calls in O(1). The Coordinator may invoke
  this many times in one session (profile, sweep, params, …) — only
  the first call does real work.
* **Concurrency-safe**: ``fcntl.flock`` serializes the read-then-write
  window across processes. Subprocess-level ``git apply`` writes are
  themselves atomic on a per-file basis (``git`` uses tmp+rename).
* **All-or-nothing for multi-patch sets**: SGLang requires 10 patches
  applied together. We run ``git apply --check`` on every patch
  first; only if every check passes do we apply for real. A mid-flight
  failure rolls back via ``git apply -R`` on the patches we already
  applied.
* **Patches are TraceLens's responsibility**: this module never reads
  patch contents or maps versions in code. The vLLM patch filename is
  derived from ``vllm.__version__``; the SGLang patch set is the
  whole directory. When TraceLens ships a new vLLM version's patch,
  Hyperloom picks it up transparently.

The patches are backward-compatible by design: vLLM ones add new
``ProfilerConfig`` dataclass fields with default ``""`` / ``False``;
SGLang ones add a new ``store_true`` server argument. Behaviour is
unchanged unless callers explicitly opt in via the new flags. Hence
the patches are safe to leave applied permanently — no revert path.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

log = logging.getLogger(__name__)


# Tunables -------------------------------------------------------------

# System-wide lock file. ``/tmp`` is writable inside containers and on
# the validation Slurm nodes; cross-reboot persistence isn't needed (the
# patch itself is persistent on disk inside the container).
_LOCK_PATH = "/tmp/hyperloom_server_patcher.lock"

# Per-``git`` invocation timeout. ``git apply`` on a single patch is
# millisecond-fast in practice; this cap is purely defensive against
# hung NFS / weird filesystems.
_GIT_TIMEOUT_SEC = 30

# TraceLens currently ships SGLang patches only for v0.5.9. Hyperloom's
# default deployment uses v0.5.10 — version mismatch is the *common* case
# and must fail-soft cleanly. Bump this string when TraceLens adds new
# versions (or refactor to discover via a manifest file inside the
# patches directory).
_SGLANG_SUPPORTED_VERSIONS: frozenset[str] = frozenset({"0.5.9"})

# Path within the TraceLens checkout that hosts the patch sets.
_PATCH_TREE_REL = ("examples", "custom_workflows", "inference_analysis")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_vllm_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """Apply the TraceLens config patch matching the installed vLLM
    version. Returns ``True`` if the install is in patched state at
    exit (already-patched or freshly-patched both count); ``False`` on
    any fail-soft outcome. Callers MUST treat ``False`` as "do not
    inject TraceLens-only profiler flags".
    """
    plan = _discover_vllm_plan(tracelens_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


def ensure_sglang_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """SGLang counterpart of :func:`ensure_vllm_patched_for_tracelens`.
    Applies the 10-patch roofline / shape-discovery set as a single
    atomic transaction (``--check`` all first, rollback on mid-apply
    failure)."""
    plan = _discover_sglang_plan(tracelens_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


# ---------------------------------------------------------------------
# Plan discovery
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _PatchPlan:
    """All information needed to apply (or verify) a patch set."""
    framework: str
    version: str
    apply_root: Path             # cwd for ``git apply``
    patches: tuple[Path, ...]    # in apply order
    sentinel_file: Path          # file we grep to detect "already patched"
    sentinel_text: str           # substring expected in sentinel_file


def _resolve_tracelens_root(arg: Path | str | None) -> Path | None:
    """Resolve TRACELENS_ROOT from arg → env → None. Fail-soft when
    the path is unset or missing on disk so tests / dry-runs without
    a real TraceLens checkout don't crash."""
    if arg:
        root = Path(arg)
    else:
        env = os.environ.get("TRACELENS_ROOT", "").strip()
        if not env:
            return None
        root = Path(env)
    return root if root.is_dir() else None


def _patch_tree(tracelens_root: Path, leaf: str) -> Path:
    """``<tracelens>/examples/custom_workflows/inference_analysis/<leaf>``."""
    return tracelens_root.joinpath(*_PATCH_TREE_REL, leaf)


def _discover_vllm_plan(arg: Path | str | None) -> _PatchPlan | None:
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.info(
            "_server_patcher: TRACELENS_ROOT unset/missing — skip vLLM patch"
        )
        return None

    try:
        import vllm  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001 - any import failure → fail-soft
        log.info("_server_patcher: vllm not importable (%s); skip patch", e)
        return None

    version = (getattr(vllm, "__version__", "") or "").strip()
    if not version:
        log.info("_server_patcher: vllm has no __version__; skip patch")
        return None

    patch_file = _patch_tree(tracelens_root, "vllm_patches") / (
        f"config_vllm_v{version}.patch"
    )
    if not patch_file.is_file():
        log.info(
            "_server_patcher: no TraceLens patch for vLLM %s "
            "(looked for %s); skip", version, patch_file,
        )
        return None

    # ``Path(vllm.__file__)`` is ``.../site-packages/vllm/__init__.py``;
    # the apply root for the patch (which uses ``a/vllm/...`` prefix) is
    # the parent of the ``vllm/`` package directory, i.e. site-packages.
    install_root = Path(vllm.__file__).resolve().parent.parent
    sentinel = install_root / "vllm" / "config" / "profiler.py"
    if not sentinel.is_file():
        log.info(
            "_server_patcher: vLLM install layout unexpected "
            "(no %s); skip patch", sentinel,
        )
        return None

    return _PatchPlan(
        framework="vllm",
        version=version,
        apply_root=install_root,
        patches=(patch_file,),
        sentinel_file=sentinel,
        sentinel_text="capture_torch_profiler_dir",
    )


def _discover_sglang_plan(arg: Path | str | None) -> _PatchPlan | None:
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.info(
            "_server_patcher: TRACELENS_ROOT unset/missing — skip SGLang patch"
        )
        return None

    try:
        import sglang  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001
        log.info("_server_patcher: sglang not importable (%s); skip patch", e)
        return None

    version = (getattr(sglang, "__version__", "") or "").strip()
    if version not in _SGLANG_SUPPORTED_VERSIONS:
        log.info(
            "_server_patcher: SGLang %s not in supported set %s; skip "
            "(TraceLens ships patches for these versions only — bump "
            "_SGLANG_SUPPORTED_VERSIONS when TraceLens adds support)",
            version, sorted(_SGLANG_SUPPORTED_VERSIONS),
        )
        return None

    patches_dir = _patch_tree(tracelens_root, "sglang_roofline_patches")
    if not patches_dir.is_dir():
        log.info(
            "_server_patcher: SGLang patches directory missing (%s); skip",
            patches_dir,
        )
        return None
    patches = tuple(sorted(patches_dir.glob("*.patch")))
    if not patches:
        log.info("_server_patcher: SGLang patches directory empty; skip")
        return None

    # The SGLang patches use ``a/python/sglang/...`` prefix — they
    # assume an editable install where the repo's ``python/sglang/...``
    # layout is preserved. For pip-wheel installs ``sglang/`` sits
    # directly under site-packages with no ``python/`` parent, so the
    # patches' ``-p1`` strip won't match.
    sglang_module = Path(sglang.__file__).resolve()
    if sglang_module.parent.parent.name != "python":
        log.info(
            "_server_patcher: SGLang install at %s is not an editable "
            "`python/sglang/` layout — patches assume that layout, so "
            "skip patching", sglang_module,
        )
        return None
    apply_root = sglang_module.parent.parent.parent

    # Sentinel: the kernel_shape_profiler patch creates an entirely new
    # file. Its presence + a unique content marker is a robust "already
    # patched" signal.
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    return _PatchPlan(
        framework="sglang",
        version=version,
        apply_root=apply_root,
        patches=patches,
        sentinel_file=sentinel,
        sentinel_text="kernel_shape_profiler",
    )


# ---------------------------------------------------------------------
# Application core
# ---------------------------------------------------------------------


def _ensure_patched(plan: _PatchPlan) -> bool:
    """Fast no-lock path → lock → re-check → atomic apply."""
    if _is_patched(plan):
        return True
    with _file_lock(_LOCK_PATH):
        if _is_patched(plan):
            return True
        return _apply_atomic(plan)


def _is_patched(plan: _PatchPlan) -> bool:
    try:
        if not plan.sentinel_file.exists():
            return False
        return plan.sentinel_text in plan.sentinel_file.read_text(
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return False


@contextmanager
def _file_lock(path: str) -> Iterator[None]:
    """Best-effort cross-process exclusion. If ``/tmp`` is read-only we
    proceed unsynchronized — the worst case is two patchers each
    deciding "I'll apply", at which point only the first wins (``git
    apply`` of an already-applied patch errors out, and the second
    patcher then re-checks the sentinel and short-circuits)."""
    try:
        fp = open(path, "w")
    except OSError as e:
        log.warning(
            "_server_patcher: cannot open lock %s (%s); proceeding without "
            "exclusion", path, e,
        )
        yield
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()


def _apply_atomic(plan: _PatchPlan) -> bool:
    """Apply every patch in ``plan.patches`` as a transaction.

    Strategy:

    1. ``git apply --check`` every patch first. If any pre-check
       fails, none get applied → caller stays in fail-soft mode.
    2. Apply for real one at a time. If patch *k* fails after we have
       already applied 0..k-1, reverse-apply 0..k-1 to leave the
       install in its original state, then return ``False``.
    """
    git = shutil.which("git")
    if git is None:
        log.warning(
            "_server_patcher: `git` not on PATH — cannot apply %s patches "
            "(install git in the runtime image to enable TraceLens flags)",
            plan.framework,
        )
        return False

    for p in plan.patches:
        if not _git(git, ("apply", "--check", str(p)), plan.apply_root):
            log.warning(
                "_server_patcher: `git apply --check` failed for %s "
                "(version %s, patch %s); fail-soft skip",
                plan.framework, plan.version, p.name,
            )
            return False

    applied: list[Path] = []
    for p in plan.patches:
        if _git(git, ("apply", str(p)), plan.apply_root):
            applied.append(p)
            continue
        log.error(
            "_server_patcher: %s patch %s failed during apply after "
            "passing --check; rolling back %d previously-applied patches",
            plan.framework, p.name, len(applied),
        )
        for prev in reversed(applied):
            if not _git(git, ("apply", "-R", str(prev)), plan.apply_root):
                log.error(
                    "_server_patcher: rollback of %s also failed — "
                    "install may be in inconsistent state; manual review "
                    "required", prev.name,
                )
        return False

    log.info(
        "_server_patcher: applied %d TraceLens patch(es) for %s %s "
        "(issue #194 §4/§5)",
        len(plan.patches), plan.framework, plan.version,
    )
    return True


def _git(git: str, args: Sequence[str], cwd: Path) -> bool:
    """Run ``git <args>`` in ``cwd``. Returns True iff rc == 0."""
    try:
        result = subprocess.run(
            (git, *args),
            cwd=str(cwd),
            capture_output=True,
            timeout=_GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(
            "_server_patcher: git %s in %s failed to spawn (%s)",
            list(args), cwd, e,
        )
        return False
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:500] \
            if result.stderr else ""
        log.debug(
            "_server_patcher: git %s rc=%d stderr=%r",
            list(args), result.returncode, err,
        )
        return False
    return True


__all__ = [
    "ensure_vllm_patched_for_tracelens",
    "ensure_sglang_patched_for_tracelens",
]
