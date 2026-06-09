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

import fcntl
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# Exact upstream line, whitespace-anchored so we don't match an unrelated
# ``num_prompts`` reference elsewhere in the file.
_LEGACY_LINE = '        num_prompts="$max_concurrency"'
_PATCHED_LINE = '        num_prompts="${NUM_PROMPTS:-$max_concurrency}"'
# "Already patched?" sentinel.
_PATCH_SENTINEL = '${NUM_PROMPTS:-$max_concurrency}'

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

    #210: Magpie loads its bundled ``$MAGPIE_DIR/InferenceX`` at runtime, not
    ``$INFERENCEX_PATH``; patching only the latter leaves Magpie's actual copy
    unpatched. Patches ALL discovered roots (deduped by resolved path):
    ``inferencex_path`` arg, ``$INFERENCEX_PATH``, ``$MAGPIE_DIR/InferenceX``.
    Returns ``[]`` when none resolve (callers fail-soft).
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path | str | None) -> None:
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
    """Return every existing ``<root>/benchmarks/benchmark_lib.sh`` to patch
    (one per :func:`_discover_inferencex_roots` root). ``[]`` = skip patching.
    """
    out: list[Path] = []
    for root in _discover_inferencex_roots(inferencex_path):
        candidate = root / "benchmarks" / "benchmark_lib.sh"
        if candidate.is_file():
            out.append(candidate)
    return out


# Back-compat single-path helper; new code uses
# :func:`_resolve_benchmark_lib_paths` to patch every root.
def _resolve_benchmark_lib_path(
    inferencex_path: Path | str | None,
) -> Path | None:
    """Single-path back-compat shim — returns first
    :func:`_resolve_benchmark_lib_paths` result."""
    paths = _resolve_benchmark_lib_paths(inferencex_path)
    return paths[0] if paths else None


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    Falls through without exclusion if the lock file can't be opened; the
    atomic-replace path still guarantees no torn writes (idempotent).
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
        # Preserve perms so the patched file stays runnable as a sourced lib.
        os.chmod(tmp_name, src.stat().st_mode)
        os.replace(tmp_name, src)
    except OSError as e:
        log.warning("_inferencex_patcher: cannot write %s: %s", src, e)
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_err:
            # Best-effort temp cleanup; main write already failed.
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

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when the file
    is missing or the legacy line is absent. Concurrency-safe (flock +
    atomic rename; already-patched fast-path skips the lock).
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

    # #210 fix: patch every discovered InferenceX root, not just the first.
    any_patched = False
    pre_patched = [src for src in sources if _is_patched(src)]
    if len(pre_patched) == len(sources):
        return True  # all paths already patched, fast-path no lock

    with _file_lock(_LOCK_PATH):
        for src in sources:
            # Re-check under the lock (another process may have patched).
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
    ``<root>/utils/bench_serving/benchmark_serving.py`` to patch (one per
    :func:`_discover_inferencex_roots` root, including Magpie's bundled copy
    per the #210 fix). Independent of the benchmark_lib.sh resolver.
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
    """Single-path back-compat shim — returns first
    :func:`_resolve_benchmark_serving_paths` result."""
    paths = _resolve_benchmark_serving_paths(inferencex_path)
    return paths[0] if paths else None


def _is_benchmark_serving_patched(src: Path) -> bool:
    try:
        return _BENCH_SERVING_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False


def _apply_benchmark_serving_patch_atomic(src: Path) -> bool:
    """Rewrite the hardcoded ``extra_body=`` line to consult
    ``PROFILE_EXTRA_BODY`` first, via temp-file + atomic rename.
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
    """Ensure InferenceX ``benchmark_serving.py`` reads ``PROFILE_EXTRA_BODY``
    on ``/start_profile``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when missing.
    Concurrency-safe; independent lock file from
    :func:`ensure_benchmark_lib_patched` so the two patches don't serialize.
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

    # #210 fix: patch every discovered InferenceX root, not just the first.
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
