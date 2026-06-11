# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, atomic-write patcher for Magpie ``_prepare_benchmark_scripts``
(Hyperloom ``bugs.md`` §C #1 root-cause fix).

Magpie copies its generic ``scripts/benchmark/*.sh`` into
``<InferenceX>/benchmarks/`` via ``shutil.copy2`` (``O_TRUNC`` + chunked,
non-atomic to a concurrent reader). A leaked ``bash`` from a prior task
(bugs.md §B) that re-sources a script mid-copy then hits ``syntax error near
unexpected token 'fi'``. We can't monkey-patch the subprocess Magpie, so we
patch the cloned ``benchmarker.py`` in place to use a temp-file + ``os.replace``
copy (no observable intermediate state), with an idempotent byte-identical skip
so a read-only pre-staged ``InferenceX/benchmarks`` deployment no-ops instead of
hitting ``[Errno 30]``. The replacement uses ``_hyperloom_*`` aliases to avoid
shadowing upstream names.

Applied in place, once, never reverted: idempotent via a sentinel substring,
serialized via ``fcntl.flock``, written atomically. When the legacy block is
absent the patcher is upstream-aware: if Magpie already copies atomically it
returns ``True`` (redundant no-op); only a genuinely-unexpected shape returns
``False`` so the install script can fail-loud.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# Exact upstream two-line block we replace, whitespace-anchored so we don't
# match an unrelated ``shutil.copy2`` elsewhere.
_LEGACY_BLOCK = (
    "            shutil.copy2(script, target_file)\n"
    "            target_file.chmod(0o755)\n"
)

# Replacement block; ``_hyperloom_*`` aliases keep the injected imports from
# shadowing upstream names.
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

# "Already patched?" sentinel.
_PATCH_SENTINEL = "Hyperloom #C1 patch"
_REMOTE_TRUST_SENTINEL = "MAGPIE_TRUST_REMOTE_CODE"

# Magpie's remote-server SGLang client path bypasses the local run_benchmark
# helper, so it used to miss --trust-remote-code for custom tokenizer models.
_REMOTE_DIRECT_LEGACY_BLOCK = (
    "    SERVER_MONITOR_ARGS=()\n"
    "    magpie_run_benchmark_serving_remote_direct || exit $?\n"
)
_REMOTE_DIRECT_PATCHED_BLOCK = (
    "    SERVER_MONITOR_ARGS=()\n"
    "    if [[ \"${MAGPIE_TRUST_REMOTE_CODE:-0}\" == \"1\" ]]; then\n"
    "      magpie_run_benchmark_serving_remote_direct trust || exit $?\n"
    "    else\n"
    "      magpie_run_benchmark_serving_remote_direct || exit $?\n"
    "    fi\n"
)

# Helper upstream Magpie introduced when it made the copy loop race-safe; its
# presence means the legacy block is gone because upstream already fixed it.
_UPSTREAM_ATOMIC_HELPER = "_copy_benchmark_script_atomic"

# Atomic-write primitives we look for when upstream inlined the temp-file +
# rename dance instead of extracting the named helper.
_ATOMIC_MKSTEMP = "tempfile.mkstemp("
_ATOMIC_REPLACE = "os.replace("

# Method header used to scope inline-atomic detection (so an unrelated
# ``os.replace`` elsewhere isn't mistaken for a fixed copy loop).
_PREPARE_METHOD_MARKER = "def _prepare_benchmark_scripts"

# System-wide lock (``/tmp`` is writable; cross-reboot persistence not needed).
_LOCK_PATH = "/tmp/hyperloom_magpie_benchmarker_patcher.lock"


def _resolve_benchmarker_path(magpie_dir: Path | str | None) -> Path | None:
    """Resolve ``<magpie_dir>/Magpie/modes/benchmark/benchmarker.py``.

    Returns ``None`` when unconfigured or missing on disk (callers skip
    patching).
    """
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = os.environ.get("MAGPIE_DIR", "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    return candidate if candidate.is_file() else None


def _resolve_sglang_mi300x_script_path(
    magpie_dir: Path | str | None,
) -> Path | None:
    """Resolve Magpie's generic SGLang MI300X benchmark script when present."""
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = os.environ.get("MAGPIE_DIR", "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh"
    return candidate if candidate.is_file() else None


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    Falls through without exclusion if the lock file can't be opened; the
    atomic-replace still guarantees no torn writes (idempotent).
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
    """Return whether ``src`` already contains the patch sentinel.

    Args:
        src (Path): The ``benchmarker.py`` file to inspect.

    Returns:
        bool: True iff the Hyperloom patch sentinel is present (and False on
            any read error).
    """
    try:
        return _PATCH_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False


def _extract_prepare_region(text: str) -> str:
    """Return the source slice covering the ``_prepare_benchmark_scripts``
    method body, or ``""`` when the header is absent.

    Scopes inline-atomic detection to one method body (header down to the next
    line indented at/below the header column) so an unrelated ``os.replace``
    can't masquerade as a fixed copy loop.
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
    """True when the cloned Magpie already copies scripts atomically (#C1 patch
    redundant). Either signal suffices: ``_copy_benchmark_script_atomic``
    present, or an inline ``tempfile.mkstemp(`` + ``os.replace(`` in the
    ``_prepare_benchmark_scripts`` body.
    """
    if _UPSTREAM_ATOMIC_HELPER in text:
        return True
    region = _extract_prepare_region(text)
    return _ATOMIC_MKSTEMP in region and _ATOMIC_REPLACE in region


def _apply_patch_atomic(src: Path) -> bool:
    """Rewrite ``src`` via temp-file + atomic rename so a crash
    mid-write cannot leave a corrupt ``benchmarker.py``.

    Args:
        src (Path): The ``benchmarker.py`` file to patch in place.

    Returns:
        bool: True when the patch was written; False when the legacy block was
            missing or any read/write step failed.
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


def _is_remote_trust_patched(src: Path) -> bool:
    """Return whether the SGLang remote-client trust gate is already present."""
    try:
        return _REMOTE_TRUST_SENTINEL in src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False


def _apply_remote_trust_patch_atomic(src: Path) -> bool:
    """Patch ``sglang_mi300x.sh`` so remote clients can pass trust mode."""
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False

    if _REMOTE_TRUST_SENTINEL in original:
        return True
    if _REMOTE_DIRECT_LEGACY_BLOCK not in original:
        log.warning(
            "_magpie_patcher: remote benchmark direct-call block not found in "
            "%s; Magpie custom-tokenizer trust patch could not be applied",
            src,
        )
        return False

    patched = original.replace(
        _REMOTE_DIRECT_LEGACY_BLOCK,
        _REMOTE_DIRECT_PATCHED_BLOCK,
        1,
    )
    tmp_dir = src.parent
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".sglang_mi300x.sh.hyperloom_",
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
        "_magpie_patcher: applied SGLang remote trust patch to %s",
        src,
    )
    return True


@dataclass(frozen=True)
class MagpiePatchStatus:
    atomic_ok: bool
    remote_trust_ok: bool

    @property
    def ok(self) -> bool:
        return self.atomic_ok and self.remote_trust_ok


def magpie_scripts_patch_status(
    magpie_dir: Path | str | None = None,
) -> MagpiePatchStatus:
    """Return independent status for atomic-copy and remote-trust patches.

    This keeps a drift in the optional SGLang remote-client trust patch from
    being reported as a generic atomic-copy failure. The bool-valued
    ``ensure_magpie_atomic_scripts_patch`` wrapper remains for compatibility.
    """
    src = _resolve_benchmarker_path(magpie_dir)
    if src is None:
        log.info(
            "_magpie_patcher: MAGPIE_DIR unset or benchmarker.py missing — "
            "skipping patch (fine for tests / dry-runs)",
        )
        return MagpiePatchStatus(atomic_ok=False, remote_trust_ok=True)

    with _file_lock(_LOCK_PATH):
        atomic_ok = _is_patched(src) or _apply_patch_atomic(src)
        sglang_script = _resolve_sglang_mi300x_script_path(magpie_dir)
        if sglang_script is None:
            log.info(
                "_magpie_patcher: sglang_mi300x.sh missing — skipping "
                "remote trust patch (fine for reduced tests / non-SGLang "
                "Magpie layouts)",
            )
            remote_trust_ok = True
        else:
            remote_trust_ok = (
                _is_remote_trust_patched(sglang_script)
                or _apply_remote_trust_patch_atomic(sglang_script)
            )
        if not remote_trust_ok:
            log.warning(
                "_magpie_patcher: SGLang remote trust patch did not apply; "
                "MAGPIE_TRUST_REMOTE_CODE=1 will not reach remote "
                "benchmark_serving.py for custom-code models",
            )
        return MagpiePatchStatus(
            atomic_ok=atomic_ok,
            remote_trust_ok=remote_trust_ok,
        )


def ensure_magpie_atomic_scripts_patch(
    magpie_dir: Path | str | None = None,
) -> bool:
    """Ensure cloned Magpie's ``_prepare_benchmark_scripts`` copies each
    script atomically (via ``os.replace``).

    Returns ``True`` when the race is closed (freshly-patched, already-patched,
    or upstream already atomic). Returns ``False`` only when the file is missing
    or neither the legacy block nor an atomic impl is found — the install script
    should fail-loud on ``False`` (this is a known root-cause fix).
    Concurrency-safe (flock + atomic rename; patched fast-path skips the lock).

    Reflects the atomic-copy patch only (matching this function's name). The
    optional SGLang remote-client trust patch is independent and can drift
    without the atomic race being open, so it is intentionally NOT folded in
    here; callers that need both must use :func:`magpie_scripts_patch_status`
    and check ``remote_trust_ok`` / ``ok`` (install.sh does this).
    """
    return magpie_scripts_patch_status(magpie_dir).atomic_ok


__all__ = [
    "MagpiePatchStatus",
    "ensure_magpie_atomic_scripts_patch",
    "magpie_scripts_patch_status",
]
