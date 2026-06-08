# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, backward-compatible patcher for InferenceX
``benchmarks/benchmark_lib.sh`` (Hyperloom issue #194 §2).

Background
----------

When ``PROFILE=1``, upstream ``run_benchmark_serving`` in
``benchmark_lib.sh`` unconditionally resets ``num_prompts`` after CLI
parsing:

.. code-block:: bash

   if [[ "${PROFILE:-}" == "1" ]]; then
       ...
       num_prompts="$max_concurrency"
   fi

This stomps anything the caller passes via ``--num-prompts`` *and*
ignores any environment variable.  With the TraceLens-aligned steady-
state window (``delay_iters`` reaches into the thousands for large
``OSL``), the benchmark engine finishes long before the profiling
window opens — yielding empty or warmup-only traces (see issue #194 §1
follow-up).

Approach
--------

Apply the smallest possible **backward-compatible** patch:

.. code-block:: diff

   -        num_prompts="$max_concurrency"
   +        num_prompts="${NUM_PROMPTS:-$max_concurrency}"

When ``NUM_PROMPTS`` env is *unset* the line behaves bit-for-bit
identically to upstream — so every existing InferenceX consumer
(``single_node/*.sh``, ``multi_node/*.sh``, etc.) is unaffected.  When
Hyperloom's profile path exports ``NUM_PROMPTS`` (sized to cover
``delay_iters + max_iters``), the engine has enough prompts to reach
the profiling window.

Lifecycle
---------

The patch is applied **in place, once, and never reverted**.  Repeated
calls are O(1) no-ops (we grep for a sentinel substring).  Concurrent
attempts from multiple processes are serialized via ``fcntl.flock`` on
a system-wide lock file so two writers never race on the same byte
range.  The write itself is atomic (temp file + ``os.replace``) so a
crash mid-write can never corrupt ``benchmark_lib.sh``.

If the expected legacy line is missing (e.g. someone has hand-patched
the file to a different shape, or the upstream layout changed), the
patcher logs a warning and returns ``False`` so callers can decide
whether to fail-loud.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# Exact upstream line. Leading-whitespace-anchored so we cannot
# accidentally match an unrelated reference to ``num_prompts`` elsewhere
# in the file (and there are several — e.g. in helper-function arg
# lists).
_LEGACY_LINE = '        num_prompts="$max_concurrency"'
_PATCHED_LINE = '        num_prompts="${NUM_PROMPTS:-$max_concurrency}"'
# Substring uniquely present after patching; used as the "already
# patched?" sentinel so we don't have to re-derive the patched line.
_PATCH_SENTINEL = '${NUM_PROMPTS:-$max_concurrency}'

# System-wide lock. ``/tmp`` is writable inside containers and on the
# validation Slurm nodes; persistence across reboots isn't needed (the
# patch itself is persistent on disk).
_LOCK_PATH = "/tmp/hyperloom_benchmark_lib_patcher.lock"


# ``benchmark_serving.py`` hardcodes
# ``extra_body={"num_steps": 1, "merge_profiles": True, "profile_by_stage":
# True}`` on the ``/start_profile`` request to SGLang. Hyperloom sets
# ``PROFILE_EXTRA_BODY`` env var (with shape_discovery / roofline_annotations
# + steady-state start_step/num_steps computed by ``_workload_envs``), but
# upstream InferenceX never reads it. Without this patch the carefully-tuned
# steady-state window collapses to the upstream default and TraceLens
# loses the shape/roofline annotations the analysis depends on.
#
# Patch is a single-line replacement gated on the exact legacy text so we
# never double-patch; sentinel is the ``PROFILE_EXTRA_BODY`` substring.
_BENCH_SERVING_LEGACY = (
    '                                         extra_body={"num_steps": 1, '
    '"merge_profiles": True, "profile_by_stage": True},'
)
# JSON fallback uses lowercase ``true`` (the Python literal ``True``
# would be a JSONDecodeError on the unset-env path, defeating the
# backward-compat invariant). ``json.loads`` correctly maps ``true``
# back to Python ``True`` so the resulting dict matches the upstream
# literal byte-for-byte.
_BENCH_SERVING_PATCHED = (
    "                                         extra_body=__import__('json')."
    "loads(__import__('os').environ.get('PROFILE_EXTRA_BODY') or "
    '\'{"num_steps": 1, "merge_profiles": true, "profile_by_stage": true}\'),'
)
_BENCH_SERVING_SENTINEL = "PROFILE_EXTRA_BODY"
_BENCH_SERVING_LOCK_PATH = "/tmp/hyperloom_benchmark_serving_patcher.lock"


def _discover_inferencex_roots(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every InferenceX checkout root Hyperloom should patch.

    #210 root cause (Deval, 2026-05-15 — see issue #210 comments 4 + 6):
    Magpie loads its own bundled InferenceX from
    ``$MAGPIE_DIR/InferenceX`` at runtime, NOT from
    ``$INFERENCEX_PATH``. If those two paths differ, patching only
    ``$INFERENCEX_PATH`` leaves Magpie's actual runtime copy untouched
    — ``profile_by_stage=True`` leaks through, ``PROFILE_EXTRA_BODY``
    is ignored, and the trace folder ends up with separate
    ``_extend_*`` / ``_decode_*`` files instead of the expected
    ``_steady_state_*`` (the smoking-gun symptom mohbasit reported on
    the same issue).

    Patches ALL discovered roots (deduplicated by resolved absolute
    path) so the patch reaches whichever InferenceX Magpie actually
    imports at runtime:

    1. ``inferencex_path`` arg (caller-provided override)
    2. ``$INFERENCEX_PATH`` env (existing behaviour, PR #207's path)
    3. ``$MAGPIE_DIR/InferenceX`` (the new path — the #210 fix)

    Returns ``[]`` when no roots resolve to existing directories;
    callers fail-soft as before. Same paths from multiple sources
    collapse to one entry (the standard install layout where
    ``$INFERENCEX_PATH = $MAGPIE_DIR/InferenceX`` produces one root,
    not two).

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root, considered first.

    Returns:
        list[Path]: Resolved, de-duplicated InferenceX checkout roots
        that exist on disk.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path | str | None) -> None:
        """Resolve and append a candidate root if it is a new directory.

        Args:
            candidate (Path | str | None): A candidate InferenceX root.

        Returns:
            None: Mutates the enclosing ``roots``/``seen`` collections.
        """
        if not candidate:
            return
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    _add(inferencex_path)
    _add(os.environ.get("INFERENCEX_PATH", "").strip() or None)
    magpie_dir = os.environ.get("MAGPIE_DIR", "").strip()
    if magpie_dir:
        _add(Path(magpie_dir) / "InferenceX")
    return roots


def _resolve_benchmark_lib_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing ``<root>/benchmarks/benchmark_lib.sh`` to
    patch (one per InferenceX root from :func:`_discover_inferencex_roots`).

    Returns ``[]`` when no root has the expected file so callers can
    treat "no InferenceX checkout" as "skip patching".

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root forwarded to :func:`_discover_inferencex_roots`.

    Returns:
        list[Path]: Existing ``benchmark_lib.sh`` paths, one per root.
    """
    out: list[Path] = []
    for root in _discover_inferencex_roots(inferencex_path):
        candidate = root / "benchmarks" / "benchmark_lib.sh"
        if candidate.is_file():
            out.append(candidate)
    return out


# Back-compat: existing callers / tests that import this single-path
# helper get the first discovered candidate (or None). New code must
# use :func:`_resolve_benchmark_lib_paths` to patch every root.
def _resolve_benchmark_lib_path(
    inferencex_path: Path | str | None,
) -> Path | None:
    """Single-path back-compat shim for :func:`_resolve_benchmark_lib_paths`.

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root.

    Returns:
        Path | None: The first resolved ``benchmark_lib.sh`` path, or
        ``None`` when none exist.
    """
    paths = _resolve_benchmark_lib_paths(inferencex_path)
    return paths[0] if paths else None


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    If the lock file can't be opened (read-only ``/tmp``, exotic
    sandbox), we fall through *without* exclusion rather than crash:
    the atomic-replace path below still guarantees no torn writes, and
    in the worst case two concurrent patchers each produce the same
    patched bytes (idempotent).

    Args:
        lock_path (str): Filesystem path used as the lock file.

    Yields:
        None: Control is yielded while the exclusive lock is held.
    """
    try:
        fp = open(lock_path, "w")
    except OSError as e:
        log.warning(
            "_inferencex_patcher: cannot open lock file %s (%s); "
            "proceeding without exclusion",
            lock_path, e,
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


def _is_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already carries the patch.

    Args:
        src (Path): The ``benchmark_lib.sh`` file to inspect.

    Returns:
        bool: ``True`` if the patch sentinel is present; ``False`` on a
        miss or read error.
    """
    try:
        return _PATCH_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False


def _apply_patch_atomic(src: Path) -> bool:
    """Rewrite ``src`` via temp-file + atomic rename so a crash
    mid-write cannot leave a corrupt ``benchmark_lib.sh``.

    Args:
        src (Path): The ``benchmark_lib.sh`` file to patch in place.

    Returns:
        bool: ``True`` when the patched bytes were written; ``False``
        when the legacy line is missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False

    if _LEGACY_LINE not in original:
        log.warning(
            "_inferencex_patcher: expected legacy line not found in %s; "
            "the file may already have been hand-patched to a "
            "different shape, or the upstream layout has changed. "
            "Manual review needed.",
            src,
        )
        return False

    patched = original.replace(_LEGACY_LINE, _PATCHED_LINE, 1)
    if patched == original:
        # Defence in depth — should never trip given the membership
        # check above, but cheap to verify.
        return False

    tmp_dir = src.parent
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".benchmark_lib.sh.hyperloom_",
            dir=str(tmp_dir),
        )
    except OSError as e:
        log.warning(
            "_inferencex_patcher: cannot create temp file in %s: %s",
            tmp_dir, e,
        )
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patched)
        # Preserve the executable bit (and any other perms) from the
        # original so the patched file is still runnable as a sourced
        # library.
        os.chmod(tmp_name, src.stat().st_mode)
        os.replace(tmp_name, src)
    except OSError as e:
        log.warning("_inferencex_patcher: cannot write %s: %s", src, e)
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_err:
            # Best-effort temp cleanup; the main write already failed so
            # we propagate that failure regardless. Log at debug so the
            # ignored exception is grep-discoverable without spamming
            # warning logs on the unhappy path.
            log.debug(
                "_inferencex_patcher: best-effort cleanup failed for temp "
                "file %s: %s", tmp_name, cleanup_err,
            )
        return False

    log.info(
        "_inferencex_patcher: applied NUM_PROMPTS-respecting patch to "
        "%s (Hyperloom issue #194 §2)",
        src,
    )
    return True


def ensure_benchmark_lib_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_lib.sh`` honours ``$NUM_PROMPTS``.

    Returns ``True`` when the file is in patched state at exit
    (already-patched or freshly-patched both count); ``False`` when
    the file could not be located or the expected legacy line is
    missing.  The latter is non-fatal — callers are expected to log
    and continue so that smoke / dry-run paths without a real
    InferenceX checkout still work.

    Safe to call from any number of processes concurrently — the
    flock serializes the read-then-write window; the temp-file +
    rename guarantees no torn writes; and the fast-path bypasses the
    lock entirely when the file is already patched (which is the case
    for every call after the first on a given checkout).

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root; defaults to env-discovered roots.

    Returns:
        bool: ``True`` if at least one discovered ``benchmark_lib.sh``
        is in patched state at exit; ``False`` otherwise.
    """
    sources = _resolve_benchmark_lib_paths(inferencex_path)
    if not sources:
        log.info(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_DIR/InferenceX) or "
            "benchmark_lib.sh missing — skipping patch (this is fine "
            "for tests and dry-runs without a real InferenceX tree)",
        )
        return False

    # #210 fix: patch every discovered InferenceX root, not just the
    # first. Magpie loads its bundled InferenceX from
    # $MAGPIE_DIR/InferenceX at runtime regardless of $INFERENCEX_PATH;
    # patching only one of the two leaves the actual runtime copy
    # untouched. Single-lock-for-all-paths is fine — each individual
    # patch is microsecond-fast, and the loop preserves atomic-replace
    # per-file.
    any_patched = False
    pre_patched = [src for src in sources if _is_patched(src)]
    if len(pre_patched) == len(sources):
        return True  # all paths already patched, fast-path no lock

    with _file_lock(_LOCK_PATH):
        for src in sources:
            # Re-check under the lock: another process may have patched
            # while we were blocked on flock.
            if _is_patched(src):
                any_patched = True
                continue
            if _apply_patch_atomic(src):
                any_patched = True
            else:
                log.warning(
                    "_inferencex_patcher: failed to patch %s; other "
                    "discovered roots will still be attempted", src,
                )
    return any_patched


# =====================================================================
# PROFILE_EXTRA_BODY consumer patch for benchmark_serving.py
# =====================================================================
def _resolve_benchmark_serving_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing
    ``<root>/utils/bench_serving/benchmark_serving.py`` to patch
    (one per InferenceX root from :func:`_discover_inferencex_roots`).

    Independent of the benchmark_lib.sh resolver because the two
    patches are independently useful: shape-aware steady-state windows
    (this one — PROFILE_EXTRA_BODY) and NUM_PROMPTS honouring (the
    other) sit on different files.

    #210 fix: includes Magpie's bundled InferenceX so Hyperloom
    patches the file Magpie actually loads at runtime, not just
    whatever ``$INFERENCEX_PATH`` resolves to.

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root forwarded to :func:`_discover_inferencex_roots`.

    Returns:
        list[Path]: Existing ``benchmark_serving.py`` paths, one per
        root.
    """
    out: list[Path] = []
    for root in _discover_inferencex_roots(inferencex_path):
        candidate = root / "utils" / "bench_serving" / "benchmark_serving.py"
        if candidate.is_file():
            out.append(candidate)
    return out


def _resolve_benchmark_serving_path(
    inferencex_path: Path | str | None,
) -> Path | None:
    """Single-path back-compat shim for :func:`_resolve_benchmark_serving_paths`.

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root.

    Returns:
        Path | None: The first resolved ``benchmark_serving.py`` path,
        or ``None`` when none exist.
    """
    paths = _resolve_benchmark_serving_paths(inferencex_path)
    return paths[0] if paths else None


def _is_benchmark_serving_patched(src: Path) -> bool:
    """Return whether ``benchmark_serving.py`` already carries the patch.

    Args:
        src (Path): The ``benchmark_serving.py`` file to inspect.

    Returns:
        bool: ``True`` if the ``PROFILE_EXTRA_BODY`` sentinel is present;
        ``False`` on a miss or read error.
    """
    try:
        return _BENCH_SERVING_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False


def _apply_benchmark_serving_patch_atomic(src: Path) -> bool:
    """Rewrite the single hardcoded ``extra_body=`` line in
    ``benchmark_serving.py`` to consult ``PROFILE_EXTRA_BODY`` first,
    via temp-file + atomic rename so a crash mid-write cannot leave a
    corrupt ``benchmark_serving.py``.

    Args:
        src (Path): The ``benchmark_serving.py`` file to patch in place.

    Returns:
        bool: ``True`` when the patched bytes were written; ``False``
        when the legacy line is missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False

    if _BENCH_SERVING_LEGACY not in original:
        log.warning(
            "_inferencex_patcher: expected legacy `extra_body=` line not "
            "found in %s; InferenceX layout may have changed and Hyperloom "
            "needs an updated patch. PROFILE_EXTRA_BODY env var will be "
            "ignored — TraceLens shape_discovery / roofline_annotations / "
            "steady-state start_step won't reach the server. Manual review "
            "needed.", src,
        )
        return False

    patched = original.replace(
        _BENCH_SERVING_LEGACY, _BENCH_SERVING_PATCHED, 1,
    )
    if patched == original:
        return False

    tmp_dir = src.parent
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".benchmark_serving.py.hyperloom_",
            dir=str(tmp_dir),
        )
    except OSError as e:
        log.warning(
            "_inferencex_patcher: cannot create temp file in %s: %s",
            tmp_dir, e,
        )
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patched)
        os.chmod(tmp_name, src.stat().st_mode)
        os.replace(tmp_name, src)
    except OSError as e:
        log.warning("_inferencex_patcher: cannot write %s: %s", src, e)
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_err:
            log.debug(
                "_inferencex_patcher: best-effort cleanup failed for temp "
                "file %s: %s", tmp_name, cleanup_err,
            )
        return False

    log.info(
        "_inferencex_patcher: patched %s to consume PROFILE_EXTRA_BODY env "
        "var (PR-D §2: fixes silently-ignored shape_discovery / "
        "roofline_annotations / steady-state start_step from "
        "_workload_envs.py)", src,
    )
    return True


def ensure_benchmark_serving_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_serving.py`` reads
    ``PROFILE_EXTRA_BODY`` env var on ``/start_profile``.

    Returns ``True`` when the file is in patched state at exit
    (already-patched or freshly-patched both count); ``False`` when
    the file could not be located or the expected legacy line is
    missing. Non-fatal — callers are expected to log and continue
    so that smoke / dry-run paths without a real InferenceX checkout
    still work.

    Safe to call concurrently — same flock + atomic-replace shape as
    :func:`ensure_benchmark_lib_patched`. Independent lock file so
    the two patches don't serialize on each other (typical workflow
    calls both once per profile run).

    Args:
        inferencex_path (Path | str | None): Caller-provided override
            root; defaults to env-discovered roots.

    Returns:
        bool: ``True`` if at least one discovered ``benchmark_serving.py``
        is in patched state at exit; ``False`` otherwise.
    """
    sources = _resolve_benchmark_serving_paths(inferencex_path)
    if not sources:
        log.info(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_DIR/InferenceX) or "
            "benchmark_serving.py missing — skipping PROFILE_EXTRA_BODY "
            "patch (this is fine for tests and dry-runs without a real "
            "InferenceX tree)",
        )
        return False

    # #210 fix: patch every discovered InferenceX root, not just the
    # first. Magpie loads its bundled InferenceX at runtime; patching
    # only $INFERENCEX_PATH leaves Magpie's $MAGPIE_DIR/InferenceX
    # copy untouched, which is exactly the silent failure mode Deval
    # diagnosed in #210 comments 4 + 6.
    if all(_is_benchmark_serving_patched(src) for src in sources):
        return True  # all paths already patched, fast-path no lock

    any_patched = False
    with _file_lock(_BENCH_SERVING_LOCK_PATH):
        for src in sources:
            if _is_benchmark_serving_patched(src):
                any_patched = True
                continue
            if _apply_benchmark_serving_patch_atomic(src):
                any_patched = True
            else:
                log.warning(
                    "_inferencex_patcher: failed to PROFILE_EXTRA_BODY-"
                    "patch %s; other discovered roots will still be "
                    "attempted", src,
                )
    return any_patched


__all__ = ["ensure_benchmark_lib_patched", "ensure_benchmark_serving_patched"]
