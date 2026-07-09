# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, atomic-write patcher for Magpie ``_prepare_benchmark_scripts``.

Magpie copies its generic ``scripts/benchmark/*.sh`` into
``<InferenceX>/benchmarks/`` via ``shutil.copy2`` (``O_TRUNC`` + chunked,
non-atomic to a concurrent reader). A concurrent ``bash`` that re-sources a
script mid-copy then hits ``syntax error near unexpected token 'fi'``. We
can't monkey-patch the subprocess Magpie, so we
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

:class:`MagpiePatchStatus` carries a classified ``atomic_reason`` so a caller
can tell an EXPECTED no-op (``upstream_atomic`` / ``already_patched`` /
``missing``) apart from a GENUINE failure (``unrecognized_shape`` / ``io_error``)
where the script-tearing race is actually unmitigated. ``install.sh`` reads
``atomic_genuine_failure`` to fail-loud by default (``MAGPIE_PATCH_STRICT=1``)
on a real failure while still warning-and-continuing on a benign no-op.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._file_lock import best_effort_file_lock
from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)


# Atomic-patch outcome reasons. These let a caller (install.sh) tell an
# EXPECTED no-op apart from a GENUINE failure instead of collapsing both into a
# single ``False``. Only ``UNRECOGNIZED_SHAPE`` and ``IO_ERROR`` mean the
# script-tearing race may be unmitigated; the rest are benign.
_ATOMIC_REASON_APPLIED = "applied"  # legacy block rewritten this run
_ATOMIC_REASON_ALREADY_PATCHED = "already_patched"  # sentinel already present
_ATOMIC_REASON_UPSTREAM_ATOMIC = "upstream_atomic"  # Magpie already atomic
_ATOMIC_REASON_MISSING = "missing"  # MAGPIE_PATH unset / file absent
_ATOMIC_REASON_UNRECOGNIZED_SHAPE = "unrecognized_shape"  # genuine: unpatched
_ATOMIC_REASON_IO_ERROR = "io_error"  # read/write failed mid-patch

# Reasons that mean the atomic-copy race is genuinely NOT mitigated — a strict
# caller should fail-loud on these, a lenient one warns conspicuously.
_ATOMIC_REASONS_GENUINE_FAILURE = frozenset(
    {
        _ATOMIC_REASON_UNRECOGNIZED_SHAPE,
        _ATOMIC_REASON_IO_ERROR,
    }
)


# Exact upstream two-line block we replace, whitespace-anchored so we don't
# match an unrelated ``shutil.copy2`` elsewhere.
_LEGACY_BLOCK = "            shutil.copy2(script, target_file)\n            target_file.chmod(0o755)\n"

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
    '                        prefix=f".{script.name}.hyperloom_", dir=str(target_dir),\n'
    "                    )\n"
    "                except OSError as _hyperloom_err:\n"
    "                    raise OSError(\n"
    '                        f"Hyperloom #C1: cannot stage benchmark script "\n'
    '                        f"{script.name} into read-only {target_dir}: "\n'
    '                        f"{_hyperloom_err}. Use a writable per-install "\n'
    '                        f"InferenceX clone (unset INFERENCEX_PATH so "\n'
    '                        f"install.sh clones a per-session copy)."\n'
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
# helper, so the trust gate for --trust-remote-code (custom tokenizer models)
# must be injected into the remote-direct path here.
_REMOTE_DIRECT_LEGACY_BLOCK = "    SERVER_MONITOR_ARGS=()\n    magpie_run_benchmark_serving_remote_direct || exit $?\n"
_REMOTE_DIRECT_PATCHED_BLOCK = (
    "    SERVER_MONITOR_ARGS=()\n"
    '    if [[ "${MAGPIE_TRUST_REMOTE_CODE:-0}" == "1" ]]; then\n'
    "      magpie_run_benchmark_serving_remote_direct trust || exit $?\n"
    "    else\n"
    "      magpie_run_benchmark_serving_remote_direct || exit $?\n"
    "    fi\n"
)

# --- Redundant eval-concurrency flag strip -------------------------------
# Magpie's generic ``{framework}_{gpu}.sh`` scripts call
# ``run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC``.
# But InferenceX's ``run_lm_eval`` (benchmark_lib.sh) does NOT accept
# ``--concurrent-requests`` — its arg parser rejects any unknown flag with
# ``Unknown parameter`` and ``return 1``. It already derives concurrency from
# ``EVAL_CONCURRENT_REQUESTS``/``CONC`` env. So the flag is redundant AND fatal:
# ``run_eval`` exits non-zero, the ``|| exit $?`` aborts the whole benchmark
# script, and an otherwise-healthy throughput baseline is thrown away. Strip the
# flag at install time (concurrency still flows via the ``CONC`` env). Matches
# ``$CONC`` / ``"$CONC"`` / ``${CONC}`` and the surrounding space. Idempotent:
# the absence of ``--concurrent-requests`` IS the patched state, so a re-run /
# already-fixed upstream script is a no-op.
_EVAL_CONCURRENCY_FLAG_MARKER = "--concurrent-requests"
_EVAL_CONCURRENCY_FLAG_RE = re.compile(r"\s*--concurrent-requests\s+(?:\"\$CONC\"|\$\{CONC\}|\$CONC)")

# Name of the upstream atomic-copy helper; its presence signals Magpie already
# copies benchmark scripts atomically.
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

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        The resolved ``benchmarker.py`` path, or ``None`` when unconfigured or
        absent on disk.
    """
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = (os.environ.get("MAGPIE_PATH") or "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    return candidate if candidate.is_file() else None


def _resolve_sglang_mi300x_script_path(
    magpie_dir: Path | str | None,
) -> Path | None:
    """Resolve Magpie's generic SGLang MI300X benchmark script when present.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        The resolved ``sglang_mi300x.sh`` path, or ``None`` when unconfigured
        or absent on disk.
    """
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = os.environ.get("MAGPIE_PATH", "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh"
    return candidate if candidate.is_file() else None


def _resolve_benchmark_scripts_dir(
    magpie_dir: Path | str | None,
) -> Path | None:
    """Resolve Magpie's ``scripts/benchmark`` directory when present.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        The resolved ``scripts/benchmark`` directory, or ``None`` when
        unconfigured or absent on disk.
    """
    root: Path | None = None
    if magpie_dir:
        root = Path(magpie_dir)
    else:
        env = os.environ.get("MAGPIE_PATH", "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "Magpie" / "scripts" / "benchmark"
    return candidate if candidate.is_dir() else None


def _strip_eval_concurrency_flag(text: str) -> str | None:
    """Return ``text`` with the redundant ``--concurrent-requests <CONC>`` flag
    removed, or ``None`` when nothing needed changing.

    Args:
        text: The benchmark-script source text to scrub.

    Returns:
        The scrubbed text when at least one occurrence was removed, else
        ``None`` (no marker present / already patched).
    """
    if _EVAL_CONCURRENCY_FLAG_MARKER not in text:
        return None
    patched = _EVAL_CONCURRENCY_FLAG_RE.sub("", text)
    if patched == text:
        # Marker present but in an unexpected shape (e.g. a literal value we
        # don't recognise). Leave it untouched and let the caller report a
        # genuine miss rather than silently no-op.
        return None
    return patched


def _apply_eval_flag_patch_atomic(scripts_dir: Path) -> bool:
    """Strip the redundant ``--concurrent-requests`` eval flag from every
    generic Magpie benchmark script under ``scripts_dir``.

    Idempotent: scripts without the marker (already patched / upstream fixed)
    are skipped. A script whose marker is present but in an unrecognised shape
    (regex miss) makes this return ``False`` so the caller can warn that the
    fatal-eval flag may still be live.

    Args:
        scripts_dir: Magpie ``scripts/benchmark`` directory to scan.

    Returns:
        ``True`` when every script is either clean or successfully scrubbed,
        ``False`` when a marker survived (unrecognised shape) or an IO step
        failed.
    """
    ok = True
    for script in sorted(scripts_dir.glob("*.sh")):
        try:
            original = script.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("_magpie_patcher: cannot read %s: %s", script, e)
            ok = False
            continue
        if _EVAL_CONCURRENCY_FLAG_MARKER not in original:
            continue
        patched = _strip_eval_concurrency_flag(original)
        if patched is None:
            log.warning(
                "_magpie_patcher: %s still contains '%s' in an unrecognised "
                "shape; the redundant eval flag could not be stripped and "
                "RUN_EVAL=true baselines may abort on 'Unknown parameter'. "
                "Review the script's run_eval line.",
                script,
                _EVAL_CONCURRENCY_FLAG_MARKER,
            )
            ok = False
            continue
        if not atomic_write_text(
            script,
            patched,
            tmp_prefix=f".{script.name}.hyperloom_",
            log_prefix="_magpie_patcher",
        ):
            ok = False
            continue
        log.info(
            "_magpie_patcher: stripped redundant '%s' eval flag from %s "
            "(concurrency still flows via the CONC env)",
            _EVAL_CONCURRENCY_FLAG_MARKER,
            script,
        )
    return ok


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    Thin delegator to :func:`best_effort_file_lock` preserved for tests / call
    sites that import ``_file_lock`` from this module.

    Args:
        lock_path: Filesystem path of the lock file to acquire exclusively.

    Yields:
        Control while the exclusive lock is held; the lock is released on exit.
    """
    with best_effort_file_lock(lock_path, label="_magpie_patcher"):
        yield


def _is_patched(src: Path) -> bool:
    """Return whether ``src`` already contains the patch sentinel.

    Args:
        src (Path): The ``benchmarker.py`` file to inspect.

    Returns:
        bool: True iff the Hyperloom patch sentinel is present (and False on
            any read error).
    """
    return file_contains_sentinel(src, _PATCH_SENTINEL, log, "_magpie_patcher")


def _extract_prepare_region(text: str) -> str:
    """Return the source slice covering the ``_prepare_benchmark_scripts``
    method body, or ``""`` when the header is absent.

    Scopes inline-atomic detection to one method body (header down to the next
    line indented at/below the header column) so an unrelated ``os.replace``
    can't masquerade as a fixed copy loop.

    Args:
        text: The full ``benchmarker.py`` source text to slice.

    Returns:
        The source slice covering the ``_prepare_benchmark_scripts`` method
        body, or ``""`` when the header is absent.
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

    Args:
        text: The full ``benchmarker.py`` source text to inspect.

    Returns:
        True when the cloned Magpie already copies scripts atomically (making
        the #C1 patch redundant), False otherwise.
    """
    if _UPSTREAM_ATOMIC_HELPER in text:
        return True
    region = _extract_prepare_region(text)
    return _ATOMIC_MKSTEMP in region and _ATOMIC_REPLACE in region


def atomic_write_text(
    src: Path,
    content: str,
    *,
    tmp_prefix: str,
    log_prefix: str,
) -> bool:
    """Write ``content`` to ``src`` via temp-file + atomic rename.

    ``tempfile.mkstemp`` into ``src.parent`` -> ``os.fdopen`` write ->
    ``os.chmod`` to ``src``'s mode -> ``os.replace`` so a crash mid-write
    cannot leave a corrupt file. On any ``OSError`` the temp file is unlinked
    best-effort and ``False`` is returned. Lives in this module so the
    module-global ``tempfile`` / ``os`` names (which tests monkeypatch) still
    intercept; ``_inferencex_patcher`` imports it.

    Args:
        src: The file to (atomically) overwrite.
        content: The new file contents.
        tmp_prefix: Prefix for the staged temp file.
        log_prefix: Caller-identifying prefix used in the warning/debug logs.

    Returns:
        ``True`` when the bytes were written and renamed into place, ``False``
        when any IO step failed.
    """
    tmp_dir = src.parent
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=tmp_prefix,
            dir=str(tmp_dir),
        )
    except OSError as e:
        log.warning(
            "%s: cannot create temp file in %s: %s",
            log_prefix,
            tmp_dir,
            e,
        )
        return False

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp_name, src.stat().st_mode)
        os.replace(tmp_name, src)
    except OSError as e:
        log.warning("%s: cannot write %s: %s", log_prefix, src, e)
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_err:
            log.debug(
                "%s: best-effort cleanup failed for temp file %s: %s",
                log_prefix,
                tmp_name,
                cleanup_err,
            )
        return False
    return True


def _apply_patch_atomic_reason(src: Path) -> str:
    """Rewrite ``src`` via temp-file + atomic rename so a crash mid-write
    cannot leave a corrupt ``benchmarker.py``, returning a classified reason.

    Unlike a bare bool, the reason lets a caller distinguish an EXPECTED no-op
    (already-patched / upstream-atomic) from a GENUINE failure (unrecognized
    shape / I/O error) where the script-tearing race is actually unmitigated.

    Args:
        src (Path): The ``benchmarker.py`` file to patch in place.

    Returns:
        str: One of the ``_ATOMIC_REASON_*`` constants.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return _ATOMIC_REASON_IO_ERROR

    if _PATCH_SENTINEL in original:
        return _ATOMIC_REASON_ALREADY_PATCHED

    if _LEGACY_BLOCK not in original:
        if _upstream_is_already_atomic(original):
            log.info(
                "_magpie_patcher: Magpie upstream already performs atomic "
                "script copy (found _copy_benchmark_script_atomic / "
                "mkstemp+os.replace); Hyperloom #C1 patch is a no-op for %s",
                src,
            )
            return _ATOMIC_REASON_UPSTREAM_ATOMIC
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
        return _ATOMIC_REASON_UNRECOGNIZED_SHAPE

    patched = original.replace(_LEGACY_BLOCK, _PATCHED_BLOCK, 1)
    if patched == original:
        return _ATOMIC_REASON_UNRECOGNIZED_SHAPE

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=".benchmarker.py.hyperloom_",
        log_prefix="_magpie_patcher",
    ):
        return _ATOMIC_REASON_IO_ERROR

    log.info(
        "_magpie_patcher: applied Hyperloom #C1 atomic-write patch to %s",
        src,
    )
    return _ATOMIC_REASON_APPLIED


def _apply_patch_atomic(src: Path) -> bool:
    """Bool wrapper over :func:`_apply_patch_atomic_reason`: True when the
    atomic-copy race is closed (applied / already-patched / upstream-atomic).

    Args:
        src: The ``benchmarker.py`` file to patch in place.

    Returns:
        True when the atomic-copy race is closed, False on a genuine failure.
    """
    return _apply_patch_atomic_reason(src) not in _ATOMIC_REASONS_GENUINE_FAILURE


def _is_remote_trust_patched(src: Path) -> bool:
    """Return whether the SGLang remote-client trust gate is already present.

    Args:
        src: The ``sglang_mi300x.sh`` file to inspect.

    Returns:
        True iff the remote-trust sentinel is present, False on a miss or read
        error.
    """
    return file_contains_sentinel(src, _REMOTE_TRUST_SENTINEL, log, "_magpie_patcher")


def _apply_remote_trust_patch_atomic(src: Path) -> bool:
    """Patch ``sglang_mi300x.sh`` so remote clients can pass trust mode.

    Args:
        src: The ``sglang_mi300x.sh`` file to patch in place.

    Returns:
        True when the trust gate is present after the call (already patched or
        freshly written), False when the legacy block is missing or any IO
        step fails.
    """
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
    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=".sglang_mi300x.sh.hyperloom_",
        log_prefix="_magpie_patcher",
    ):
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
    # Classified atomic-patch outcome (``_ATOMIC_REASON_*``). Lets callers tell
    # an EXPECTED no-op (upstream-atomic / already-patched / missing tree) apart
    # from a GENUINE failure where the atomic-write safeguard is absent.
    atomic_reason: str = _ATOMIC_REASON_MISSING
    # Whether the redundant ``--concurrent-requests`` eval flag was stripped
    # from every generic benchmark script (or none needed it). ``False`` means
    # a script kept the flag in an unrecognised shape, so RUN_EVAL=true
    # baselines may still abort on InferenceX's ``Unknown parameter``. Defaults
    # True (not-applicable / no scripts dir) so it never falsely fails install.
    eval_flag_ok: bool = True

    @property
    def ok(self) -> bool:
        """Whether the patch result is fully successful.

        Returns:
            ``True`` only when the atomic write, remote-trust, and
            eval-flag-strip checks all succeeded.
        """
        return self.atomic_ok and self.remote_trust_ok and self.eval_flag_ok

    @property
    def atomic_genuine_failure(self) -> bool:
        """True when ``atomic_ok`` is False for a real reason (unrecognized
        shape / I/O error) — i.e. the script-tearing race is NOT mitigated, as
        opposed to a benign no-op. A strict install should fail-loud on this.

        Returns:
            True when the atomic-copy patch failed for a genuine reason
            (unrecognized shape / I/O error), False for a benign no-op.
        """
        return self.atomic_reason in _ATOMIC_REASONS_GENUINE_FAILURE


def magpie_scripts_patch_status(
    magpie_dir: Path | str | None = None,
) -> MagpiePatchStatus:
    """Return independent status for atomic-copy and remote-trust patches.

    This keeps a drift in the optional SGLang remote-client trust patch from
    being reported as a generic atomic-copy failure. The bool-valued
    ``ensure_magpie_atomic_scripts_patch`` wrapper remains for compatibility.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        A ``MagpiePatchStatus`` carrying the atomic-copy and remote-trust
        outcomes plus the classified ``atomic_reason``.
    """
    src = _resolve_benchmarker_path(magpie_dir)
    if src is None:
        log.info(
            "_magpie_patcher: MAGPIE_PATH unset or benchmarker.py missing — skipping patch (fine for tests / dry-runs)",
        )
        # remote_trust_ok=True here means "not applicable / not checked"
        # (no Magpie tree to inspect), NOT "trust patch verified". It is set
        # True only so this no-op path does not emit a spurious remote-trust
        # warning. atomic_ok=False + reason=missing keeps the legacy fail-soft
        # (install.sh warns, does not abort) but is NOT a genuine failure.
        # eval_flag_ok=True is likewise "not applicable" (no scripts to scrub).
        return MagpiePatchStatus(
            atomic_ok=False,
            remote_trust_ok=True,
            atomic_reason=_ATOMIC_REASON_MISSING,
            eval_flag_ok=True,
        )

    with _file_lock(_LOCK_PATH):
        atomic_reason = _apply_patch_atomic_reason(src)
        atomic_ok = atomic_reason not in _ATOMIC_REASONS_GENUINE_FAILURE
        sglang_script = _resolve_sglang_mi300x_script_path(magpie_dir)
        if sglang_script is None:
            log.info(
                "_magpie_patcher: sglang_mi300x.sh missing — skipping "
                "remote trust patch (fine for reduced tests / non-SGLang "
                "Magpie layouts)",
            )
            remote_trust_ok = True
        else:
            remote_trust_ok = _is_remote_trust_patched(sglang_script) or _apply_remote_trust_patch_atomic(sglang_script)
        if not remote_trust_ok:
            log.warning(
                "_magpie_patcher: SGLang remote trust patch did not apply; "
                "MAGPIE_TRUST_REMOTE_CODE=1 will not reach remote "
                "benchmark_serving.py for custom-code models",
            )
        scripts_dir = _resolve_benchmark_scripts_dir(magpie_dir)
        if scripts_dir is None:
            # No scripts/benchmark dir (reduced test layout / dry run): nothing
            # to scrub, so the redundant-flag patch is not-applicable.
            eval_flag_ok = True
        else:
            eval_flag_ok = _apply_eval_flag_patch_atomic(scripts_dir)
        return MagpiePatchStatus(
            atomic_ok=atomic_ok,
            remote_trust_ok=remote_trust_ok,
            atomic_reason=atomic_reason,
            eval_flag_ok=eval_flag_ok,
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

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        True when the atomic-copy race is closed, False when the file is
        missing or neither the legacy block nor an atomic impl is found.
    """
    return magpie_scripts_patch_status(magpie_dir).atomic_ok


__all__ = [
    "MagpiePatchStatus",
    "ensure_magpie_atomic_scripts_patch",
    "magpie_scripts_patch_status",
    "_ATOMIC_REASON_APPLIED",
    "_ATOMIC_REASON_ALREADY_PATCHED",
    "_ATOMIC_REASON_UPSTREAM_ATOMIC",
    "_ATOMIC_REASON_MISSING",
    "_ATOMIC_REASON_UNRECOGNIZED_SHAPE",
    "_ATOMIC_REASON_IO_ERROR",
]
