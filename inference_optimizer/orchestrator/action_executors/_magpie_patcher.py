# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, atomic-write patcher for Magpie ``_prepare_benchmark_scripts``
(Hyperloom ``bugs.md`` §C #1 root-cause fix).

Background
----------

``Magpie/modes/benchmark/benchmarker.py:_prepare_benchmark_scripts`` copies
Magpie's generic ``scripts/benchmark/*.sh`` into
``<InferenceX>/benchmarks/`` at the start of every benchmark invocation.
The upstream loop is:

.. code-block:: python

    for script in magpie_scripts.glob("*.sh"):
        target_file = target_dir / script.name
        shutil.copy2(script, target_file)
        target_file.chmod(0o755)

``shutil.copy2`` opens ``target_file`` with ``O_WRONLY|O_CREAT|O_TRUNC`` and
writes in chunks. The chunked write is NOT atomic from a concurrent
reader's point of view:

* ``O_TRUNC`` truncates ``target_file`` to length 0 immediately.
* Each ``write(2)`` system call is atomic, but the full copy spans many.
* Any process that ``open()`` s the path during the write window sees the
  current partial length, not the eventual final bytes.

In Hyperloom this manifests as ``bash vllm_mi300x.sh: line 125: syntax
error near unexpected token 'fi'`` (``bugs.md`` §C #1) whenever a leaked
``bash`` interpreter from a prior task (kept alive by ``bugs.md`` §B —
``profile_executor`` failing to kill its spawned vLLM subtree) re-sources
the script while a new Magpie subprocess is mid-copy.

Why we patch in place at install time
-------------------------------------

Hyperloom invokes Magpie as a **subprocess** (``python -m Magpie -v
benchmark …`` — see ``baseline.py:383-388`` and ``_grid_runner.py:531``).
Monkey-patching ``Magpie.modes.benchmark.benchmarker._prepare_benchmark_scripts``
inside the Coordinator's Python process has no effect — the subprocess
imports a fresh, unmodified Magpie.

The only options that actually fix the race in the cloned Magpie's own
code path are:

(a) Upstream PR to Magpie (best long-term, but blocks on review).
(b) In-place patch of the cloned ``benchmarker.py`` at Hyperloom install
    time, mirroring the existing ``_inferencex_patcher.py`` pattern.

This module is (b). When upstream eventually adopts atomic writes, the
sentinel substring below will already be present and ``ensure_*`` becomes
a no-op (fast path, no flock).

Patch shape
-----------

We replace:

.. code-block:: python

            shutil.copy2(script, target_file)
            target_file.chmod(0o755)

with an atomic temp-file + ``os.replace`` form that preserves the
executable bit and the source mtime/perms (``shutil.copy2`` does both):

.. code-block:: python

            # Hyperloom #C1 patch: atomic write so a concurrent bash
            # `source` cannot see a half-truncated file. Skip the write
            # entirely when the target is already byte-identical.
            import os as _hyperloom_os
            import shutil as _hyperloom_shutil
            import tempfile as _hyperloom_tempfile
            import filecmp as _hyperloom_filecmp
            if target_file.exists() and _hyperloom_filecmp.cmp(
                str(script), str(target_file), shallow=False
            ):
                pass
            else:
                try:
                    _tmp_fd, _tmp_name = _hyperloom_tempfile.mkstemp(
                        prefix=f".{script.name}.hyperloom_", dir=str(target_dir),
                    )
                except OSError as _hyperloom_err:
                    raise OSError(  # names script + read-only dir + the fix
                        ...
                    ) from _hyperloom_err
                _hyperloom_os.close(_tmp_fd)
                _hyperloom_shutil.copy2(script, _tmp_name)
                _hyperloom_os.chmod(_tmp_name, 0o755)
                _hyperloom_os.replace(_tmp_name, target_file)

``os.replace`` is a single ``rename(2)`` syscall. POSIX guarantees that
any reader that already has a file descriptor on the old inode keeps
seeing the old (consistent) bytes; any reader that ``open()`` s the path
after the rename sees the new (consistent) bytes. There is no observable
intermediate state.

Two refinements over a naive atomic copy (``bugs.md`` §C #1 follow-up —
the read-only-mount regression):

* **Idempotent skip.** When ``target_file`` already byte-matches ``script``
  we do nothing — a shared, read-only ``InferenceX/benchmarks`` deployment
  with scripts pre-staged is a no-op instead of failing in
  ``mkstemp(dir=target_dir)`` with ``[Errno 30] Read-only file system``
  (the failure that killed every model's first benchmark session).
* **Actionable read-only error.** If the dir really is read-only AND the
  script is missing/stale, we raise a clear error naming the script and
  directory (and the per-install-clone fix), never a bare ``[Errno 30]``.

The replacement uses fully-qualified ``_hyperloom_*`` aliases so we don't
collide with any names already in upstream scope (``os`` / ``shutil`` /
``tempfile`` / ``filecmp`` happen to already be importable, but we do not
rely on that — defence in depth keeps the patch self-contained and immune
to upstream import-reordering).

Lifecycle
---------

The patch is applied **in place, once, and never reverted**. Repeated
calls are O(1) no-ops via a sentinel-substring check. Concurrent install
attempts are serialised via ``fcntl.flock`` on a dedicated lock file so
two writers never race on the same byte range. The patch write itself is
atomic (temp file + ``os.replace``) so a crash mid-write cannot corrupt
``benchmarker.py``.

If the expected legacy two-line block is missing the patcher is
upstream-aware: it first checks whether the cloned Magpie already copies
scripts atomically (newer upstream extracted a ``_copy_benchmark_script_atomic``
helper, or otherwise does ``tempfile.mkstemp`` + ``os.replace`` inside
``_prepare_benchmark_scripts``). If so the race is already closed, the
Hyperloom #C1 patch is redundant, and ``ensure_*`` logs an ``info`` line
and returns ``True`` without touching the file. Only when *neither* the
legacy block *nor* any atomic implementation is found does the patcher log
a warning and return ``False`` (a genuinely-unexpected shape) so the
install script can decide whether to fail-loud; that case means
``bugs.md`` §C #1 cannot be confirmed mitigated.
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


# Exact upstream two-line block we replace. Whitespace-anchored so we
# can't match an unrelated reference to ``shutil.copy2`` elsewhere in
# ``benchmarker.py`` (there are no others today, but defence in depth).
_LEGACY_BLOCK = (
    "            shutil.copy2(script, target_file)\n"
    "            target_file.chmod(0o755)\n"
)

# Replacement block. Uses ``_hyperloom_*`` aliases so the injected
# ``import os`` / ``import tempfile`` lines cannot shadow anything
# upstream code may rely on at that scope (those modules happen to be
# imported at module top today, but the aliases keep the patch immune
# to upstream import-reorderings).
_PATCHED_BLOCK = (
    "            # Hyperloom #C1 patch: atomic write so a concurrent bash\n"
    "            # `source` cannot see a half-truncated file. Skip the write\n"
    "            # entirely when the target is already byte-identical, so a\n"
    "            # read-only / shared InferenceX/benchmarks deployment (scripts\n"
    "            # pre-staged, dir not writable) is a no-op instead of\n"
    "            # OSError: [Errno 30] Read-only file system.\n"
    "            import os as _hyperloom_os\n"
    "            import shutil as _hyperloom_shutil\n"
    "            import tempfile as _hyperloom_tempfile\n"
    "            import filecmp as _hyperloom_filecmp\n"
    "            if target_file.exists() and _hyperloom_filecmp.cmp(\n"
    "                str(script), str(target_file), shallow=False\n"
    "            ):\n"
    "                pass\n"
    "            else:\n"
    "                try:\n"
    "                    _tmp_fd, _tmp_name = _hyperloom_tempfile.mkstemp(\n"
    "                        prefix=f\".{script.name}.hyperloom_\", dir=str(target_dir),\n"
    "                    )\n"
    "                except OSError as _hyperloom_err:\n"
    "                    raise OSError(\n"
    "                        f\"Hyperloom #C1: cannot stage benchmark script \"\n"
    "                        f\"{script.name} into read-only {target_dir}: \"\n"
    "                        f\"{_hyperloom_err}. Use a writable per-install \"\n"
    "                        f\"InferenceX clone (unset INFERENCEX_PATH so \"\n"
    "                        f\"install.sh clones a per-session copy).\"\n"
    "                    ) from _hyperloom_err\n"
    "                _hyperloom_os.close(_tmp_fd)\n"
    "                _hyperloom_shutil.copy2(script, _tmp_name)\n"
    "                _hyperloom_os.chmod(_tmp_name, 0o755)\n"
    "                _hyperloom_os.replace(_tmp_name, target_file)\n"
)

# Substring uniquely present after patching; used as the "already
# patched?" sentinel so we don't have to re-derive the patched block.
_PATCH_SENTINEL = "Hyperloom #C1 patch"

# Name of the static helper upstream Magpie introduced when it refactored
# the copy loop to be race-safe on its own (``tempfile.mkstemp`` +
# ``shutil.copy2`` + ``os.chmod`` + ``os.replace``). Its presence means the
# legacy two-line block is *gone because upstream already fixed it*, not
# because the checkout drifted into an unrecognised shape.
_UPSTREAM_ATOMIC_HELPER = "_copy_benchmark_script_atomic"

# Atomic-write primitives we look for *within* the
# ``_prepare_benchmark_scripts`` body when upstream inlined the temp-file +
# rename dance instead of extracting the named helper above.
_ATOMIC_MKSTEMP = "tempfile.mkstemp("
_ATOMIC_REPLACE = "os.replace("

# Header used to locate the method body we scope inline-atomic detection to,
# so an unrelated ``os.replace`` elsewhere in ``benchmarker.py`` cannot be
# mistaken for an already-fixed copy loop.
_PREPARE_METHOD_MARKER = "def _prepare_benchmark_scripts"

# System-wide lock. ``/tmp`` is writable inside containers and on the
# validation Slurm nodes; persistence across reboots isn't needed (the
# patch itself is persistent on disk).
_LOCK_PATH = "/tmp/hyperloom_magpie_benchmarker_patcher.lock"


def _resolve_benchmarker_path(magpie_dir: Path | str | None) -> Path | None:
    """Resolve ``<magpie_dir>/Magpie/modes/benchmark/benchmarker.py``.

    Returns ``None`` when the path is unconfigured or missing on disk
    so callers can treat "no Magpie checkout" as "skip patching" — this
    is what unit tests need (they may pass ``tmp_path`` fixtures without
    populating a full Magpie tree).
    """
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = (
            os.environ.get("MAGPIE_PATH") or os.environ.get("MAGPIE_DIR") or ""
        ).strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    return candidate if candidate.is_file() else None


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    Falls through without exclusion if the lock file can't be opened
    (read-only ``/tmp``, exotic sandbox): the atomic-replace below
    still guarantees no torn writes, and concurrent patchers each
    produce identical bytes (idempotent).
    """
    try:
        fp = open(lock_path, "w")  # noqa: SIM115 — kept open across yield
    except OSError as e:
        log.warning(
            "_magpie_patcher: cannot open lock file %s (%s); "
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
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False


def _extract_prepare_region(text: str) -> str:
    """Return the source slice covering the ``_prepare_benchmark_scripts``
    method body, or ``""`` when the method header is absent.

    Inline-atomic detection (``mkstemp`` + ``os.replace``) is scoped to this
    slice so an unrelated ``os.replace`` elsewhere in ``benchmarker.py``
    cannot masquerade as an already-fixed copy loop. The slice spans from
    the ``def`` header to the first subsequent non-blank line indented at or
    below the header column (the next sibling ``def``/``class`` or dedented
    code), which bounds exactly one method body.
    """
    start = text.find(_PREPARE_METHOD_MARKER)
    if start == -1:
        return ""
    line_start = text.rfind("\n", 0, start) + 1
    def_indent = start - line_start
    lines = text[line_start:].splitlines(keepends=True)
    region = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            region.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= def_indent:
            break
        region.append(line)
    return "".join(region)


def _upstream_is_already_atomic(text: str) -> bool:
    """True when the cloned Magpie already copies scripts atomically, so the
    Hyperloom #C1 patch is redundant (a no-op) rather than missing.

    Either signal is sufficient:

    * the file defines/uses ``_copy_benchmark_script_atomic`` — the helper
      upstream Magpie introduced when it made the copy loop race-safe; or
    * the ``_prepare_benchmark_scripts`` body itself performs an inline
      temp-file + atomic rename (``tempfile.mkstemp(`` and ``os.replace(``).
    """
    if _UPSTREAM_ATOMIC_HELPER in text:
        return True
    region = _extract_prepare_region(text)
    return _ATOMIC_MKSTEMP in region and _ATOMIC_REPLACE in region


def _apply_patch_atomic(src: Path) -> bool:
    """Rewrite ``src`` via temp-file + atomic rename so a crash
    mid-write cannot leave a corrupt ``benchmarker.py``.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False

    if _LEGACY_BLOCK not in original:
        if _upstream_is_already_atomic(original):
            log.info(
                "_magpie_patcher: Magpie upstream already performs atomic "
                "script copy (found _copy_benchmark_script_atomic / "
                "mkstemp+os.replace); Hyperloom #C1 patch is a no-op for %s",
                src,
            )
            return True
        log.warning(
            "_magpie_patcher: neither the legacy shutil.copy2/chmod block nor "
            "an atomic copy implementation found in %s; Magpie may have been "
            "refactored into an unrecognised shape, or this checkout was "
            "hand-patched. Hyperloom bugs.md §C #1 (script-tearing race) "
            "cannot be confirmed mitigated — `profile`/`baseline` may hit "
            "`syntax error near unexpected token 'fi'` again. Manual review "
            "needed.",
            src,
        )
        return False

    patched = original.replace(_LEGACY_BLOCK, _PATCHED_BLOCK, 1)
    if patched == original:
        return False

    tmp_dir = src.parent
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".benchmarker.py.hyperloom_",
            dir=str(tmp_dir),
        )
    except OSError as e:
        log.warning(
            "_magpie_patcher: cannot create temp file in %s: %s",
            tmp_dir, e,
        )
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patched)
        os.chmod(tmp_name, src.stat().st_mode)
        os.replace(tmp_name, src)
    except OSError as e:
        log.warning("_magpie_patcher: cannot write %s: %s", src, e)
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_err:
            log.debug(
                "_magpie_patcher: best-effort cleanup failed for temp "
                "file %s: %s", tmp_name, cleanup_err,
            )
        return False

    log.info(
        "_magpie_patcher: applied Hyperloom #C1 atomic-write patch to %s",
        src,
    )
    return True


def ensure_magpie_atomic_scripts_patch(
    magpie_dir: Path | str | None = None,
) -> bool:
    """Ensure cloned Magpie's ``_prepare_benchmark_scripts`` copies each
    script atomically (via ``os.replace``).

    Returns ``True`` when the race is closed at exit, which covers three
    cases: freshly-patched, already-patched (sentinel present), or an
    upstream that is *already* atomic (``_copy_benchmark_script_atomic`` /
    inline ``mkstemp`` + ``os.replace``) — in the last case the Hyperloom
    #C1 patch is a redundant no-op and the file is left untouched.

    Returns ``False`` only when the file could not be located, or when
    *neither* the legacy block *nor* any atomic implementation is found
    (a genuinely-unexpected shape). The install script is expected to
    ``fail-loud`` on ``False`` (unlike the InferenceX patcher which is
    best-effort) — this is a known root-cause fix, and that case means
    ``bugs.md`` §C #1 cannot be confirmed mitigated.

    Safe to call from any number of processes concurrently — flock
    serialises the read-then-write window; temp-file + rename
    guarantees no torn writes; and the fast path bypasses the lock
    entirely once the file is patched (which is the case for every
    call after the first on a given Magpie checkout).
    """
    src = _resolve_benchmarker_path(magpie_dir)
    if src is None:
        log.info(
            "_magpie_patcher: MAGPIE_DIR unset or benchmarker.py missing — "
            "skipping patch (fine for tests / dry-runs without a real "
            "Magpie tree)",
        )
        return False

    if _is_patched(src):
        return True

    with _file_lock(_LOCK_PATH):
        if _is_patched(src):
            return True
        return _apply_patch_atomic(src)


__all__ = [
    "ensure_magpie_atomic_scripts_patch",
]
