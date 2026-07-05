# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, backward-compatible patcher for InferenceX
``benchmarks/benchmark_lib.sh`` (Hyperloom issue #194 §2).

Upstream resets ``num_prompts="$max_concurrency"`` under ``PROFILE=1``,
stomping ``--num-prompts`` so the engine finishes before the steady-state
profiling window opens (empty traces). The patch makes that line honour
``${NUM_PROMPTS:-$max_concurrency}`` — bit-for-bit identical when the env is
unset, so existing consumers are unaffected.

Applied in place, once, never reverted: idempotent via a sentinel substring,
serialized across processes via ``fcntl.flock``, written atomically (temp file
+ ``os.replace``). Returns ``False`` (non-fatal) when the legacy line is
missing so callers can decide whether to fail-loud.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Callable

from ._file_lock import best_effort_file_lock
from ._magpie_patcher import atomic_write_text
from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)


# Exact upstream line, whitespace-anchored so we don't match an unrelated
# ``num_prompts`` reference elsewhere in the file.
_LEGACY_LINE = '        num_prompts="$max_concurrency"'
_PATCHED_LINE = '        num_prompts="${NUM_PROMPTS:-$max_concurrency}"'
# "Already patched?" sentinel.
_PATCH_SENTINEL = "${NUM_PROMPTS:-$max_concurrency}"

# System-wide lock (``/tmp`` is writable; cross-reboot persistence not needed).
_LOCK_PATH = "/tmp/hyperloom_benchmark_lib_patcher.lock"


# ``benchmark_serving.py`` hardcodes the ``/start_profile`` ``extra_body`` and
# never reads Hyperloom's ``PROFILE_EXTRA_BODY`` env, collapsing the tuned
# steady-state window. Single-line replacement gated on the exact legacy text;
# sentinel is the ``PROFILE_EXTRA_BODY`` substring.
_BENCH_SERVING_LEGACY = (
    '                                         extra_body={"num_steps": 1, '
    '"merge_profiles": True, "profile_by_stage": True},'
)
# JSON fallback uses lowercase ``true`` (Python ``True`` would be a
# JSONDecodeError on the unset-env path); ``json.loads`` maps it back so the
# dict matches the upstream literal byte-for-byte.
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

    #210: Magpie loads its bundled ``$MAGPIE_PATH/InferenceX`` at runtime, not
    ``$INFERENCEX_PATH``; patching only the latter leaves Magpie's actual copy
    unpatched. Patches ALL discovered roots (deduped by resolved path):
    ``inferencex_path`` arg, ``$INFERENCEX_PATH``, ``$MAGPIE_PATH/InferenceX``.
    Returns ``[]`` when none resolve (callers fail-soft).

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A deduped list of resolved InferenceX checkout directories, or ``[]``
        when none resolve.
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
    magpie_dir = (os.environ.get("MAGPIE_PATH") or "").strip()
    if magpie_dir:
        _add(Path(magpie_dir) / "InferenceX")
    return roots


def _resolve_inferencex_files(
    inferencex_path: Path | str | None,
    *rel_parts: str,
) -> list[Path]:
    """Return every existing ``<root>/<*rel_parts>`` across discovered roots.

    One entry per :func:`_discover_inferencex_roots` root whose joined relative
    path is an existing file. ``[]`` = skip patching.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.
        *rel_parts: Relative path components joined onto each discovered root.

    Returns:
        A list of existing files, or ``[]`` when none exist.
    """
    out: list[Path] = []
    for root in _discover_inferencex_roots(inferencex_path):
        candidate = root.joinpath(*rel_parts)
        if candidate.is_file():
            out.append(candidate)
    return out


def _resolve_benchmark_lib_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing ``<root>/benchmarks/benchmark_lib.sh`` to patch
    (one per :func:`_discover_inferencex_roots` root). ``[]`` = skip patching.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A list of existing ``benchmark_lib.sh`` paths, or ``[]`` when none
        exist.
    """
    return _resolve_inferencex_files(inferencex_path, "benchmarks", "benchmark_lib.sh")


def _is_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already carries the patch.

    Args:
        src (Path): The ``benchmark_lib.sh`` file to inspect.

    Returns:
        bool: ``True`` if the patch sentinel is present; ``False`` on a
        miss or read error.
    """
    return file_contains_sentinel(src, _PATCH_SENTINEL, log, "_inferencex_patcher")


def _apply_line_replacement_atomic(
    src: Path,
    legacy: str,
    patched_line: str,
    *,
    tmp_prefix: str,
    missing_msg: str,
    success_msg: str,
) -> bool:
    """Replace a single exact ``legacy`` line with ``patched_line`` in ``src``
    via temp-file + atomic rename so a crash mid-write cannot leave a corrupt
    file.

    Shared by both InferenceX patches (``benchmark_lib.sh`` and
    ``benchmark_serving.py``); they differ only in the legacy/patched text,
    temp-file prefix, and log messages.

    Args:
        src: The file to patch in place.
        legacy: Exact legacy line that must be present to patch.
        patched_line: Replacement text for ``legacy`` (first occurrence).
        tmp_prefix: Temp-file prefix for the atomic write.
        missing_msg: Warning (one ``%s`` for ``src``) when ``legacy`` is absent.
        success_msg: Info (one ``%s`` for ``src``) logged on a successful write.

    Returns:
        bool: ``True`` when the patched bytes were written; ``False`` when the
        legacy line is missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False

    if legacy not in original:
        log.warning(missing_msg, src)
        return False

    patched = original.replace(legacy, patched_line, 1)
    if patched == original:
        return False

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=tmp_prefix,
        log_prefix="_inferencex_patcher",
    ):
        return False

    log.info(success_msg, src)
    return True


def _ensure_patched(
    sources: list[Path],
    is_patched: Callable[[Path], bool],
    apply_patch: Callable[[Path], bool],
    lock_path: str,
    *,
    empty_msg: str,
    failure_msg: str,
) -> bool:
    """Drive a set of discovered files to patched state (#210 multi-root).

    Empty fast-path: ``log.info(empty_msg)`` + ``False``. All-already-patched
    fast-path skips the lock. Otherwise, under the lock, each source is
    re-checked and patched; a failed apply emits ``log.warning(failure_msg,
    src)`` and the remaining roots are still attempted.

    Args:
        sources: Discovered files to patch.
        is_patched: "Already patched?" predicate for one file.
        apply_patch: In-place atomic patcher for one file (True on success).
        lock_path: Cross-process lock file path.
        empty_msg: Info message logged when ``sources`` is empty.
        failure_msg: Warning message (one ``%s`` for ``src``) on apply failure.

    Returns:
        True when at least one source is patched (or already patched), False
        when none could be patched.
    """
    if not sources:
        log.info(empty_msg)
        return False

    # #210 fix: patch every discovered InferenceX root, not just the first.
    if all(is_patched(s) for s in sources):
        return True  # all paths already patched, fast-path no lock

    any_patched = False
    with best_effort_file_lock(lock_path, label="_inferencex_patcher"):
        for src in sources:
            # Re-check under the lock (another process may have patched).
            if is_patched(src):
                any_patched = True
                continue
            if apply_patch(src):
                any_patched = True
            else:
                log.warning(failure_msg, src)
    return any_patched


def ensure_benchmark_lib_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_lib.sh`` honours ``$NUM_PROMPTS``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when the file
    is missing or the legacy line is absent. Concurrency-safe (flock +
    atomic rename; already-patched fast-path skips the lock).

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_lib.sh`` is patched (or
        already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_lib_paths(inferencex_path),
        _is_patched,
        # Preserve perms so the patched file stays runnable as a sourced lib.
        partial(
            _apply_line_replacement_atomic,
            legacy=_LEGACY_LINE,
            patched_line=_PATCHED_LINE,
            tmp_prefix=".benchmark_lib.sh.hyperloom_",
            missing_msg=(
                "_inferencex_patcher: expected legacy line not found in %s; "
                "the file may already have been hand-patched to a "
                "different shape, or the upstream layout has changed. "
                "Manual review needed."
            ),
            success_msg=(
                "_inferencex_patcher: applied NUM_PROMPTS-respecting patch to %s (Hyperloom issue #194 §2)"
            ),
        ),
        _LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_lib.sh missing — skipping patch (this is fine "
            "for tests and dry-runs without a real InferenceX tree)"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to patch %s; other discovered roots will still be attempted"
        ),
    )


# =====================================================================
# PROFILE_EXTRA_BODY consumer patch for benchmark_serving.py
# =====================================================================
def _resolve_benchmark_serving_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing
    ``<root>/utils/bench_serving/benchmark_serving.py`` to patch (one per
    :func:`_discover_inferencex_roots` root, including Magpie's bundled copy
    per the #210 fix). Independent of the benchmark_lib.sh resolver.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A list of existing ``benchmark_serving.py`` paths, or ``[]`` when none
        exist.
    """
    return _resolve_inferencex_files(
        inferencex_path, "utils", "bench_serving", "benchmark_serving.py"
    )


def _is_benchmark_serving_patched(src: Path) -> bool:
    """Return whether ``benchmark_serving.py`` already carries the patch.

    Args:
        src (Path): The ``benchmark_serving.py`` file to inspect.

    Returns:
        bool: ``True`` if the ``PROFILE_EXTRA_BODY`` sentinel is present;
        ``False`` on a miss or read error.
    """
    return file_contains_sentinel(src, _BENCH_SERVING_SENTINEL, log, "_inferencex_patcher")


def ensure_benchmark_serving_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_serving.py`` reads ``PROFILE_EXTRA_BODY``
    on ``/start_profile``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when missing.
    Concurrency-safe; independent lock file from
    :func:`ensure_benchmark_lib_patched` so the two patches don't serialize.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_serving.py`` is patched
        (or already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_serving_paths(inferencex_path),
        _is_benchmark_serving_patched,
        partial(
            _apply_line_replacement_atomic,
            legacy=_BENCH_SERVING_LEGACY,
            patched_line=_BENCH_SERVING_PATCHED,
            tmp_prefix=".benchmark_serving.py.hyperloom_",
            missing_msg=(
                "_inferencex_patcher: expected legacy `extra_body=` line not "
                "found in %s; InferenceX layout may have changed and Hyperloom "
                "needs an updated patch. PROFILE_EXTRA_BODY env var will be "
                "ignored — TraceLens shape_discovery / roofline_annotations / "
                "steady-state start_step won't reach the server. Manual review "
                "needed."
            ),
            success_msg=(
                "_inferencex_patcher: patched %s to consume PROFILE_EXTRA_BODY env "
                "var (PR-D §2: fixes silently-ignored shape_discovery / "
                "roofline_annotations / steady-state start_step from "
                "_workload_envs.py)"
            ),
        ),
        _BENCH_SERVING_LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_serving.py missing — skipping PROFILE_EXTRA_BODY "
            "patch (this is fine for tests and dry-runs without a real "
            "InferenceX tree)"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to PROFILE_EXTRA_BODY-"
            "patch %s; other discovered roots will still be "
            "attempted"
        ),
    )


__all__ = ["ensure_benchmark_lib_patched", "ensure_benchmark_serving_patched"]
