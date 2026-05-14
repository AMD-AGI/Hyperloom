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


def _resolve_benchmark_lib_path(
    inferencex_path: Path | str | None,
) -> Path | None:
    """Resolve ``<inferencex_root>/benchmarks/benchmark_lib.sh``.

    Returns ``None`` when the path is unconfigured or missing on disk
    so callers can treat "no InferenceX checkout" as "skip patching"
    (this is what unit tests need — they exercise profile YAML
    rendering without provisioning a real InferenceX tree).
    """
    root: Path | None = None
    if inferencex_path:
        root = Path(inferencex_path)
    else:
        env = os.environ.get("INFERENCEX_PATH", "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "benchmarks" / "benchmark_lib.sh"
    return candidate if candidate.is_file() else None


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    If the lock file can't be opened (read-only ``/tmp``, exotic
    sandbox), we fall through *without* exclusion rather than crash:
    the atomic-replace path below still guarantees no torn writes, and
    in the worst case two concurrent patchers each produce the same
    patched bytes (idempotent).
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
    try:
        return _PATCH_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False


def _apply_patch_atomic(src: Path) -> bool:
    """Rewrite ``src`` via temp-file + atomic rename so a crash
    mid-write cannot leave a corrupt ``benchmark_lib.sh``.
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
        except OSError:
            pass
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
    """
    src = _resolve_benchmark_lib_path(inferencex_path)
    if src is None:
        log.info(
            "_inferencex_patcher: INFERENCEX_PATH unset or "
            "benchmark_lib.sh missing — skipping patch (this is fine "
            "for tests and dry-runs without a real InferenceX tree)",
        )
        return False

    if _is_patched(src):
        return True

    with _file_lock(_LOCK_PATH):
        # Re-check under the lock: another process may have patched
        # while we were blocked on flock.
        if _is_patched(src):
            return True
        return _apply_patch_atomic(src)


__all__ = ["ensure_benchmark_lib_patched"]
