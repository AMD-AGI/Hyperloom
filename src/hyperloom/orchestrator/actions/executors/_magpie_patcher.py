# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Idempotent, atomic-write patcher for Magpie ``_prepare_benchmark_scripts``.

Patches the cloned ``benchmarker.py`` in place so benchmark scripts are copied
via a temp-file + ``os.replace`` (no observable intermediate state a concurrent
``bash source`` could tear), with a byte-identical skip so a read-only
pre-staged deployment no-ops. Applied once, idempotent via a sentinel substring,
serialized via ``fcntl.flock``, written atomically. When the legacy block is
absent the patcher is upstream-aware: an already-atomic Magpie returns ``True``;
only a genuinely-unexpected shape returns ``False`` so install.sh can fail-loud.

:class:`MagpiePatchStatus` carries a classified ``atomic_reason`` so a caller
can tell an EXPECTED no-op (``upstream_atomic`` / ``already_patched`` /
``missing``) apart from a GENUINE failure (``unrecognized_shape`` / ``io_error``)
where the script-tearing race is actually unmitigated. ``install.sh`` reads
``atomic_genuine_failure`` to fail-loud by default on a real failure.
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


# Atomic-patch outcome reasons: distinguish an EXPECTED no-op from a GENUINE
# failure. Only ``UNRECOGNIZED_SHAPE`` and ``IO_ERROR`` mean the script-tearing
# race may be unmitigated; the rest are benign.
_ATOMIC_REASON_APPLIED = "applied"
_ATOMIC_REASON_ALREADY_PATCHED = "already_patched"
_ATOMIC_REASON_UPSTREAM_ATOMIC = "upstream_atomic"
_ATOMIC_REASON_MISSING = "missing"
_ATOMIC_REASON_UNRECOGNIZED_SHAPE = "unrecognized_shape"
_ATOMIC_REASON_IO_ERROR = "io_error"

# Reasons that mean the atomic-copy race is genuinely NOT mitigated.
_ATOMIC_REASONS_GENUINE_FAILURE = frozenset(
    {
        _ATOMIC_REASON_UNRECOGNIZED_SHAPE,
        _ATOMIC_REASON_IO_ERROR,
    }
)


# Exact upstream two-line block we replace, whitespace-anchored so we don't
# match an unrelated ``shutil.copy2`` elsewhere.
_LEGACY_BLOCK = "            shutil.copy2(script, target_file)\n            target_file.chmod(0o755)\n"

# Replacement block; ``_hyperloom_*`` aliases avoid shadowing upstream names.
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

# "Already patched?" sentinels.
_PATCH_SENTINEL = "Hyperloom #C1 patch"
_REMOTE_TRUST_SENTINEL = "MAGPIE_TRUST_REMOTE_CODE"
_EVAL_CONC_SENTINEL = "HYPERLOOM_EVAL_CONCURRENCY_FIX"

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

# Magpie's SGLang local-client path (``BENCHMARK_BASE_URL`` unset — e.g. the
# baseline ``server_lifecycle`` reuse path) calls ``run_benchmark_serving``
# directly. Unlike the vLLM scripts it does not pass ``--trust-remote-code``, so
# custom tokenizer models (for example Kimi) fail before issuing a request.
# Build an optional argv array from the same env gate used by the remote-direct
# path, then splice it into the local client command. Both the ``sglang_mi300x``
# and ``sglang_mi355x`` scripts share these byte-identical client blocks.
_LOCAL_TRUST_SENTINEL = "HYPERLOOM_SGLANG_LOCAL_TRUST"
# Marks that a script actually has a local-server client path to patch. When
# absent (reduced test layouts / scripts without the local branch) the local
# trust splice is skipped rather than treated as drift.
_LOCAL_PATH_MARKER = "--result-dir ${RESULT_DIR"
_LOCAL_TRUST_ARGS_LEGACY_BLOCK = 'SERVER_MONITOR_ARGS=()\nif [[ -n "${SERVER_PID:-}" ]]; then\n'
_LOCAL_TRUST_ARGS_PATCHED_BLOCK = (
    "SERVER_MONITOR_ARGS=()\n"
    f"# {_LOCAL_TRUST_SENTINEL}: custom-tokenizer client argv\n"
    "CLIENT_TRUST_ARGS=()\n"
    'if [[ "${MAGPIE_TRUST_REMOTE_CODE:-0}" == "1" ]]; then\n'
    "  CLIENT_TRUST_ARGS+=(--trust-remote-code)\n"
    "fi\n"
    'if [[ -n "${SERVER_PID:-}" ]]; then\n'
)
_LOCAL_CLIENT_LEGACY_BLOCK = (
    '        "${SERVER_MONITOR_ARGS[@]}" \\\n'
    "        --result-dir ${RESULT_DIR:-/workspace/} || exit $?\n"
)
_LOCAL_CLIENT_PATCHED_BLOCK = (
    '        "${SERVER_MONITOR_ARGS[@]}" \\\n'
    '        "${CLIENT_TRUST_ARGS[@]}" \\\n'
    "        --result-dir ${RESULT_DIR:-/workspace/} || exit $?\n"
)

# Strip the redundant, fatal ``--concurrent-requests <CONC>`` flag from Magpie's
# generic benchmark scripts: InferenceX's ``run_lm_eval`` rejects it as an
# unknown flag, aborting the whole script; concurrency still flows via the
# ``CONC`` env. Idempotent (absence of the flag IS the patched state).
_EVAL_CONCURRENCY_FLAG_MARKER = "--concurrent-requests"
_EVAL_CONCURRENCY_FLAG_RE = re.compile(r"\s*--concurrent-requests\s+(?:\"\$CONC\"|\$\{CONC\}|\$CONC)")

# A ``run_eval`` invocation that STILL passes the rejected flag — the one shape
# that actually aborts a benchmark. Anchored on ``run_eval`` (and on a line with
# no preceding ``#``) so ``benchmark_lib.sh``'s own arg-parser case, which
# legitimately names the flag, and our own explanatory comments never match.
_LIVE_RUN_EVAL_FLAG_RE = re.compile(
    r"^[^\n#]*\brun_eval\b[^\n]*--concurrent[-_]requests",
    re.MULTILINE,
)

# Belt-and-suspenders for the run-time re-copy: Magpie's
# ``_prepare_benchmark_scripts`` re-copies its (possibly still-flagged) generic
# scripts into ``$INFERENCEX_PATH/benchmarks`` on every run, so a stray
# ``--concurrent-requests`` can survive the strip. Teach InferenceX's
# ``benchmark_lib.sh::run_lm_eval`` arg-parser to accept the flag (it already
# owns a ``concurrent_requests`` local, wired into ``num_concurrent=``); this
# turns a fatal ``Unknown parameter`` abort into a no-op. Idempotent via the
# sentinel comment.
_RUN_LM_EVAL_PARSER_SENTINEL = "HYPERLOOM_EVAL_CONCURRENCY_ARG"
_RUN_LM_EVAL_PARSER_LEGACY_BLOCK = (
    '            --top-p)          top_p="$2"; shift 2 ;;\n'
    '            *)                echo "Unknown parameter: $1"; return 1 ;;\n'
)
_RUN_LM_EVAL_PARSER_PATCHED_BLOCK = (
    '            --top-p)          top_p="$2"; shift 2 ;;\n'
    "            # HYPERLOOM_EVAL_CONCURRENCY_ARG: accept the redundant flag\n"
    "            # (concurrency also flows via EVAL_CONCURRENT_REQUESTS/CONC).\n"
    '            --concurrent-requests|--concurrent_requests) concurrent_requests="$2"; shift 2 ;;\n'
    '            *)                echo "Unknown parameter: $1"; return 1 ;;\n'
)

# InferenceX a4bb43af+ refactored the parser into a single merged case
# (``--port|--task|...|--top-p)`` with an inner dispatch and a ``>&2`` /
# ``return 2`` catch-all, so the per-flag legacy block above no longer matches.
# Match that catch-all (any ``return N``) and splice a dedicated
# ``--concurrent-requests`` case in front of it, preserving the leading indent.
# The scan is scoped to the ``run_lm_eval`` body (see
# :func:`_extract_run_lm_eval_region`) so it never lands in an earlier
# function's identical ``*)`` catch-all (benchmark_lib.sh has several).
_RUN_LM_EVAL_MERGED_CATCHALL_RE = re.compile(
    r"^(?P<indent>[ \t]*)\*\)\s*\n"
    r"[ \t]*echo\s+\"Unknown parameter: \$1\"(?:\s+>&2)?\s*\n"
    r"[ \t]*return\s+\d+\s*\n"
    r"[ \t]*;;\s*\n",
    re.MULTILINE,
)

# Header of the InferenceX ``run_lm_eval`` shell function; used to scope the
# merged-case parser patch and the tolerance check to that function's body.
_RUN_LM_EVAL_FN_MARKER = "run_lm_eval()"

# InferenceX benchmark_lib.sh::run_lm_eval reads concurrency from env
# (EVAL_CONCURRENT_REQUESTS, fallback CONC). Passing --concurrent-requests to
# run_eval is rejected as an unknown argument.
_RUN_EVAL_LEGACY_BLOCK = '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n'
_RUN_EVAL_PATCHED_BLOCK = (
    "        # HYPERLOOM_EVAL_CONCURRENCY_FIX: benchmark_lib.sh resolves eval\n"
    "        # concurrency from EVAL_CONCURRENT_REQUESTS (fallback CONC).\n"
    '        EVAL_CONCURRENT_REQUESTS="${EVAL_CONCURRENT_REQUESTS:-$CONC}" '
    'run_eval --framework lm-eval --port "$PORT" || exit $?\n'
)

# Upstream atomic-copy helper; its presence signals Magpie already copies
# benchmark scripts atomically.
_UPSTREAM_ATOMIC_HELPER = "_copy_benchmark_script_atomic"

# Atomic-write primitives looked for when upstream inlined the temp-file + rename
# dance instead of extracting the named helper.
_ATOMIC_MKSTEMP = "tempfile.mkstemp("
_ATOMIC_REPLACE = "os.replace("

# Method header used to scope inline-atomic detection.
_PREPARE_METHOD_MARKER = "def _prepare_benchmark_scripts"

# System-wide lock.
_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_magpie_benchmarker_patcher.lock")


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


def _resolve_sglang_mi355x_script_path(
    magpie_dir: Path | str | None,
) -> Path | None:
    """Resolve Magpie's SGLang MI355X benchmark script when present.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        The resolved ``sglang_mi355x.sh`` path, or ``None`` when unconfigured
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
    candidate = root / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh"
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


def _resolve_inferencex_benchmarks_dir(
    inferencex_dir: Path | str | None,
) -> Path | None:
    """Resolve InferenceX's ``benchmarks`` directory when present.

    Magpie's ``_prepare_benchmark_scripts`` copies its generic ``*.sh`` scripts
    INTO ``$INFERENCEX_PATH/benchmarks`` at run time, and those copies are what
    actually execute (they source InferenceX's ``benchmark_lib.sh``). The
    redundant ``--concurrent-requests`` eval flag therefore also has to be
    scrubbed here, not just in the Magpie source dir.

    Args:
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH`` when falsy.

    Returns:
        The resolved ``benchmarks`` directory, or ``None`` when unconfigured or
        absent on disk.
    """
    root: Path | None = None
    if inferencex_dir:
        root = Path(inferencex_dir)
    else:
        env = (os.environ.get("INFERENCEX_PATH") or "").strip()
        if env:
            root = Path(env)
    if root is None:
        return None
    candidate = root / "benchmarks"
    return candidate if candidate.is_dir() else None


def _resolve_inferencex_benchmark_lib(
    inferencex_dir: Path | str | None,
) -> Path | None:
    """Resolve InferenceX's ``benchmarks/benchmark_lib.sh`` when present.

    Args:
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH`` when falsy.

    Returns:
        The resolved ``benchmark_lib.sh`` path, or ``None`` when unconfigured
        or absent on disk.
    """
    scripts_dir = _resolve_inferencex_benchmarks_dir(inferencex_dir)
    if scripts_dir is None:
        return None
    candidate = scripts_dir / "benchmark_lib.sh"
    return candidate if candidate.is_file() else None


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
        # Marker present but in an unrecognised shape; report a genuine miss.
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
        # ``benchmark_lib.sh`` is the shared library, not a caller script: it
        # legitimately references ``--concurrent-requests`` in run_lm_eval's arg
        # parser (patched separately by _apply_run_lm_eval_arg_patch_atomic).
        # Stripping there would corrupt the parser and the regex miss would be
        # mis-reported as an "unrecognised shape" failure — skip it.
        if script.name == "benchmark_lib.sh":
            continue
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
            # ERROR, not WARNING: a surviving flag makes every RUN_EVAL=true
            # baseline abort in InferenceX's run_lm_eval arg parser, which the
            # accuracy gate turns into a whole-run stop. Callers escalate.
            log.error(
                "_magpie_patcher: %s still contains '%s' in an unrecognised "
                "shape; the redundant eval flag could not be stripped and "
                "RUN_EVAL=true baselines WILL abort on 'Unknown parameter'. "
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
            "_magpie_patcher: stripped redundant '%s' eval flag from %s (concurrency still flows via the CONC env)",
            _EVAL_CONCURRENCY_FLAG_MARKER,
            script,
        )
    return ok


def _extract_run_lm_eval_region(text: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` char offsets of the ``run_lm_eval`` function body.

    Scopes any parser-shape scan to that one function so an identical ``*)``
    catch-all in an earlier ``benchmark_lib.sh`` function (there are several)
    can never be mistaken for ``run_lm_eval``'s. The body runs from the
    ``run_lm_eval()`` header to the next line-start ``}`` (the function's
    closing brace at column 0), or end-of-text when that brace is absent.

    Args:
        text: The full ``benchmark_lib.sh`` source text.

    Returns:
        The ``(start, end)`` offsets of the function body, or ``None`` when the
        ``run_lm_eval()`` header is absent.
    """
    start = text.find(_RUN_LM_EVAL_FN_MARKER)
    if start == -1:
        return None
    close = re.search(r"^\}", text[start:], re.MULTILINE)
    end = start + close.end() if close else len(text)
    return start, end


def _patch_merged_case_parser(text: str) -> str | None:
    """Splice a ``--concurrent-requests`` case before the merged-case parser's
    ``*)`` catch-all, or return ``None`` when that catch-all is not found inside
    the ``run_lm_eval`` body.

    Handles the InferenceX a4bb43af+ shape where every flag shares one
    ``case`` arm; the new arm sets the ``concurrent_requests`` local the parser
    already owns. Indentation is inherited from the matched catch-all. Matching
    is scoped to the ``run_lm_eval`` body so the patch never lands in an
    earlier function's identical catch-all.
    """
    region = _extract_run_lm_eval_region(text)
    if region is None:
        return None
    start, end = region
    m = _RUN_LM_EVAL_MERGED_CATCHALL_RE.search(text, start, end)
    if m is None:
        return None
    indent = m.group("indent")
    new_case = (
        f"{indent}# {_RUN_LM_EVAL_PARSER_SENTINEL}: accept the redundant flag\n"
        f"{indent}# (concurrency also flows via EVAL_CONCURRENT_REQUESTS/CONC).\n"
        f'{indent}--concurrent-requests|--concurrent_requests) concurrent_requests="$2"; shift 2 ;;\n'
    )
    return text[: m.start()] + new_case + text[m.start() :]


def _apply_run_lm_eval_arg_patch_atomic(benchmark_lib: Path) -> bool:
    """Teach InferenceX's ``benchmark_lib.sh::run_lm_eval`` to accept the
    ``--concurrent-requests`` flag instead of aborting on ``Unknown parameter``.

    Defence-in-depth for the run-time re-copy: even when the strip above misses
    a script (wrong ``MAGPIE_PATH`` at strip time, or Magpie re-copies an
    unstripped source over a stripped ``benchmarks`` copy), the parser now
    tolerates the flag. Idempotent via the sentinel; a genuinely-unrecognised
    parser shape returns ``False`` so the caller can warn.

    Args:
        benchmark_lib: The InferenceX ``benchmark_lib.sh`` file to patch.

    Returns:
        ``True`` when the parser already tolerates the flag or was patched to,
        ``False`` when the expected parser block was not found or an IO step
        failed.
    """
    try:
        original = benchmark_lib.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", benchmark_lib, e)
        return False

    # Already tolerant (our sentinel, or an upstream that added the flag) --
    # scoped to the run_lm_eval body so a sentinel/flag elsewhere in the file
    # does not short-circuit the patch of run_lm_eval itself.
    _region = _extract_run_lm_eval_region(original)
    _body = original[_region[0] : _region[1]] if _region is not None else ""
    if _RUN_LM_EVAL_PARSER_SENTINEL in _body or _EVAL_CONCURRENCY_FLAG_MARKER in _body:
        return True

    if _RUN_LM_EVAL_PARSER_LEGACY_BLOCK in original:
        patched = original.replace(
            _RUN_LM_EVAL_PARSER_LEGACY_BLOCK,
            _RUN_LM_EVAL_PARSER_PATCHED_BLOCK,
            1,
        )
    else:
        # InferenceX a4bb43af+ merged-case parser: splice a dedicated
        # --concurrent-requests case in front of the ``*)`` catch-all,
        # preserving its indentation.
        patched = _patch_merged_case_parser(original)
        if patched is None:
            log.warning(
                "_magpie_patcher: run_lm_eval arg-parser block not found in %s; "
                "cannot make it tolerate '--concurrent-requests'. RUN_EVAL=true "
                "baselines may still abort if a stray flag survives the strip.",
                benchmark_lib,
            )
            return False
    if patched == original:
        return False

    if not atomic_write_text(
        benchmark_lib,
        patched,
        tmp_prefix=".benchmark_lib.sh.hyperloom_",
        log_prefix="_magpie_patcher",
    ):
        return False

    log.info(
        "_magpie_patcher: patched %s run_lm_eval to accept '--concurrent-requests'",
        benchmark_lib,
    )
    return True


def _apply_eval_concurrency_fixes(
    magpie_dir: Path | str | None,
    inferencex_dir: Path | str | None,
) -> bool:
    """Apply every eval-concurrency compatibility fix, independent of the
    ``benchmarker.py`` atomic-copy patch.

    Scrubs the redundant ``--concurrent-requests`` flag from the Magpie source
    scripts dir AND the InferenceX ``benchmarks`` dir (where Magpie copies them
    to execute), then makes InferenceX's ``run_lm_eval`` tolerant of the flag as
    a belt for the run-time re-copy. Missing dirs / files are treated as
    not-applicable (no failure).

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH``.
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH``.

    Returns:
        ``True`` when every resolved target is clean / successfully patched,
        ``False`` when a target was found in an unrecognised shape or an IO step
        failed.
    """
    ok = True
    scanned: set[Path] = set()
    for scripts_dir in (
        _resolve_benchmark_scripts_dir(magpie_dir),
        _resolve_inferencex_benchmarks_dir(inferencex_dir),
    ):
        if scripts_dir is None or scripts_dir in scanned:
            continue
        scanned.add(scripts_dir)
        if not _apply_eval_flag_patch_atomic(scripts_dir):
            ok = False
    benchmark_lib = _resolve_inferencex_benchmark_lib(inferencex_dir)
    if benchmark_lib is not None and not _apply_run_lm_eval_arg_patch_atomic(benchmark_lib):
        ok = False
    return ok


def _inferencex_tolerates_eval_flag(inferencex_dir: Path | str | None) -> bool:
    """Whether InferenceX's ``run_lm_eval`` accepts ``--concurrent-requests``.

    When it does, a stray flag in a caller script is harmless (it is parsed into
    the existing ``concurrent_requests`` local) instead of fatal.

    Args:
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH``.

    Returns:
        ``True`` when the resolved ``benchmark_lib.sh`` parses the flag,
        ``False`` when it does not or could not be resolved/read (conservative:
        an unknown parser is assumed intolerant).
    """
    lib = _resolve_inferencex_benchmark_lib(inferencex_dir)
    if lib is None:
        return False
    try:
        text = lib.read_text(encoding="utf-8")
    except OSError:
        return False
    # Scope the check to the run_lm_eval body: a sentinel / flag anywhere else
    # in the file (e.g. a mis-placed patch in another function's catch-all, or
    # an unrelated comment) must NOT be read as run_lm_eval tolerating the flag.
    region = _extract_run_lm_eval_region(text)
    if region is None:
        return False
    body = text[region[0] : region[1]]
    return _RUN_LM_EVAL_PARSER_SENTINEL in body or _EVAL_CONCURRENCY_FLAG_MARKER in body


def live_eval_concurrency_flag_scripts(
    magpie_dir: Path | str | None = None,
    inferencex_dir: Path | str | None = None,
) -> list[Path]:
    """Benchmark scripts that still invoke ``run_eval`` with the rejected flag.

    This is the *fatal* condition, as opposed to "a defence-in-depth patch did
    not apply": each returned script aborts its benchmark with
    ``Unknown parameter: --concurrent-requests`` before any ``results*.json``
    is written. ``benchmark_lib.sh`` is skipped — it is the library whose arg
    parser legitimately names the flag, not a caller.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH``.
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH``.

    Returns:
        The offending script paths (empty when nothing is blocked).
    """
    hits: list[Path] = []
    scanned: set[Path] = set()
    for scripts_dir in (
        _resolve_benchmark_scripts_dir(magpie_dir),
        _resolve_inferencex_benchmarks_dir(inferencex_dir),
    ):
        if scripts_dir is None or scripts_dir in scanned:
            continue
        scanned.add(scripts_dir)
        for script in sorted(scripts_dir.glob("*.sh")):
            if script.name == "benchmark_lib.sh":
                continue
            try:
                text = script.read_text(encoding="utf-8")
            except OSError:
                continue
            if _LIVE_RUN_EVAL_FLAG_RE.search(text):
                hits.append(script)
    return hits


def ensure_eval_concurrency_compat(
    magpie_dir: Path | str | None = None,
    inferencex_dir: Path | str | None = None,
) -> bool:
    """Public, run-time-safe entry point for the eval-concurrency fixes.

    ``install.sh`` is not the only way Magpie and InferenceX land on a box:
    :mod:`hyperloom.inference_optimizer.cli.preflight` pip-installs Magpie and
    clones InferenceX on its own, and :func:`baseline._ensure_local_inferencex`
    re-mirrors the InferenceX checkout from scratch on **every** run. Magpie's
    ``_prepare_benchmark_scripts`` then re-copies its generic ``*.sh`` scripts
    into that mirror's ``benchmarks/`` dir at run time. So a Magpie tree that
    was never patched at install time silently re-introduces the fatal
    ``run_eval ... --concurrent-requests $CONC`` line into the copy that
    actually executes, and ``run_lm_eval`` aborts the whole benchmark with
    ``Unknown parameter: --concurrent-requests`` (no ``results*.json`` ->
    ``baseline_accuracy_failed``).

    Callers invoke this immediately before launching a benchmark so the fixes
    are (re)asserted against the trees that will really run. Cheap and
    idempotent: scripts already clean are skipped without a write.

    Returns ``False`` **only** for the genuinely fatal state: a caller script
    still invokes ``run_eval`` with the flag AND InferenceX's ``run_lm_eval``
    would reject it. A defence-in-depth patch that merely could not be applied
    (e.g. an unrecognised ``benchmark_lib.sh`` parser shape in a reduced or
    already-fixed tree) is logged, not escalated — nothing is actually blocked.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH``.
        inferencex_dir: InferenceX root override (pass the *effective* /
            mirrored checkout, not the pristine source); falls back to
            ``$INFERENCEX_PATH``.

    Returns:
        ``True`` when accuracy eval is unblocked, ``False`` when a live
        ``run_eval --concurrent-requests`` invocation survives and cannot be
        absorbed. ``False`` means accuracy eval is *certain* to abort — callers
        should fail loudly rather than continue.
    """
    with _file_lock(_LOCK_PATH):
        applied_ok = _apply_eval_concurrency_fixes(magpie_dir, inferencex_dir)
        return _eval_concurrency_unblocked(applied_ok, magpie_dir, inferencex_dir)


def _eval_concurrency_unblocked(
    applied_ok: bool,
    magpie_dir: Path | str | None,
    inferencex_dir: Path | str | None,
) -> bool:
    """Whether accuracy eval is unblocked given the eval-fix apply result.

    The single source of truth shared by the run-time entry point
    (:func:`ensure_eval_concurrency_compat`) and the install-time status
    (:func:`magpie_scripts_patch_status`): a defence-in-depth patch that merely
    could not be applied is NOT fatal on its own — only a *live*
    ``run_eval --concurrent-requests`` that InferenceX would reject blocks eval.
    Callers must already hold ``_LOCK_PATH``.

    Args:
        applied_ok: Result of :func:`_apply_eval_concurrency_fixes`.
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH``.
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH``.

    Returns:
        ``False`` only when a live flag survives and cannot be absorbed, else
        ``True``.
    """
    blockers = live_eval_concurrency_flag_scripts(magpie_dir, inferencex_dir)
    if blockers and not _inferencex_tolerates_eval_flag(inferencex_dir):
        log.error(
            "_magpie_patcher: %d benchmark script(s) still call run_eval "
            "with '%s' and InferenceX's run_lm_eval will reject it: %s. "
            "Accuracy eval WILL abort ('Unknown parameter'); concurrency "
            "must flow via EVAL_CONCURRENT_REQUESTS (fallback CONC).",
            len(blockers),
            _EVAL_CONCURRENCY_FLAG_MARKER,
            ", ".join(str(p) for p in blockers),
        )
        return False
    if not applied_ok:
        log.warning(
            "_magpie_patcher: an eval-concurrency defence-in-depth patch "
            "could not be applied (magpie=%s inferencex=%s), but no live "
            "'run_eval ... %s' invocation was found, so accuracy eval is "
            "not blocked.",
            magpie_dir or os.environ.get("MAGPIE_PATH", "") or "<unset>",
            inferencex_dir or os.environ.get("INFERENCEX_PATH", "") or "<unset>",
            _EVAL_CONCURRENCY_FLAG_MARKER,
        )
    return True


@contextmanager
def _file_lock(lock_path: str) -> Iterator[None]:
    """Best-effort cross-process mutex via ``fcntl.flock``.

    Thin delegator to :func:`best_effort_file_lock`.

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
    """True when installed Magpie already copies scripts atomically (#C1 patch
    redundant). Either signal suffices: ``_copy_benchmark_script_atomic``
    present, or an inline ``tempfile.mkstemp(`` + ``os.replace(`` in the
    ``_prepare_benchmark_scripts`` body.

    Args:
        text: The full ``benchmarker.py`` source text to inspect.

    Returns:
        True when installed Magpie already copies scripts atomically (making
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
    best-effort and ``False`` is returned.

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


def _is_remote_trust_patched(src: Path) -> bool:
    """Return whether SGLang compatibility sentinels are already present.

    Checks BOTH ``MAGPIE_TRUST_REMOTE_CODE`` (remote trust) and
    ``HYPERLOOM_EVAL_CONCURRENCY_FIX`` (eval concurrency) so a pre-existing
    remote-trust patch does not short-circuit the eval-concurrency fix.

    Args:
        src: The ``sglang_mi300x.sh`` file to inspect.

    Returns:
        True iff both compatibility sentinels are present, False on a miss or
        read error.
    """
    return file_contains_sentinel(src, _REMOTE_TRUST_SENTINEL, log, "_magpie_patcher") and file_contains_sentinel(
        src,
        _EVAL_CONC_SENTINEL,
        log,
        "_magpie_patcher",
    )


def _apply_remote_trust_patch_atomic(src: Path) -> bool:
    """Patch ``sglang_mi300x.sh`` for Hyperloom compatibility.

    Applies two independent compatibility fixes:
    - remote client trust gating via ``MAGPIE_TRUST_REMOTE_CODE``
    - eval concurrency wiring via ``EVAL_CONCURRENT_REQUESTS`` env (no
      unsupported ``--concurrent-requests`` arg)

    Args:
        src: The ``sglang_mi300x.sh`` file to patch in place.

    Returns:
        True when both compatibility fixes are present after the call (already
        patched or freshly written), False when a required legacy block is
        missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False

    patched = original

    if _REMOTE_TRUST_SENTINEL not in patched:
        if _REMOTE_DIRECT_LEGACY_BLOCK not in patched:
            log.warning(
                "_magpie_patcher: remote benchmark direct-call block not found in "
                "%s; Magpie custom-tokenizer trust patch could not be applied",
                src,
            )
            return False
        patched = patched.replace(
            _REMOTE_DIRECT_LEGACY_BLOCK,
            _REMOTE_DIRECT_PATCHED_BLOCK,
            1,
        )

    if _EVAL_CONC_SENTINEL not in patched:
        if _RUN_EVAL_LEGACY_BLOCK not in patched:
            log.warning(
                "_magpie_patcher: run_eval concurrency block not found in %s; "
                "eval concurrency compatibility patch could not be applied",
                src,
            )
            return False
        patched = patched.replace(
            _RUN_EVAL_LEGACY_BLOCK,
            _RUN_EVAL_PATCHED_BLOCK,
            1,
        )

    if patched == original:
        return True

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=".sglang_mi300x.sh.hyperloom_",
        log_prefix="_magpie_patcher",
    ):
        return False

    log.info(
        "_magpie_patcher: applied SGLang script compatibility patches to %s",
        src,
    )
    return True


def _is_sglang_client_trust_patched(src: Path) -> bool:
    """Return whether an SGLang script's client paths already carry trust gating.

    Covers both the remote-direct path and, when the script has a local-server
    client path (:data:`_LOCAL_PATH_MARKER`), the local splice. Scripts without
    a local path are considered patched once the remote-direct gate is present.

    Args:
        src: The ``sglang_mi300x.sh`` / ``sglang_mi355x.sh`` file to inspect.

    Returns:
        True iff the applicable client trust patches are already present, False
        on a miss or read error.
    """
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return False
    remote_ok = "magpie_run_benchmark_serving_remote_direct trust" in text
    local_ok = _LOCAL_TRUST_SENTINEL in text or _LOCAL_PATH_MARKER not in text
    return remote_ok and local_ok


def _apply_sglang_client_trust_patch_atomic(src: Path) -> bool:
    """Patch an SGLang script's remote and local benchmark clients for custom code.

    Applies to both ``sglang_mi300x.sh`` and ``sglang_mi355x.sh`` — their client
    blocks are byte-identical. The remote-direct gate is required (a missing
    block signals layout drift). The local splice is applied only when the
    script actually has a local-server client path (:data:`_LOCAL_PATH_MARKER`),
    and a drifted local block there is reported as a failure.

    Args:
        src: The SGLang benchmark script to patch in place.

    Returns:
        True when the applicable trust patches are present after the call
        (already patched or freshly written), False when a required legacy block
        is missing / drifted or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_magpie_patcher: cannot read %s: %s", src, e)
        return False

    patched = original
    if "magpie_run_benchmark_serving_remote_direct trust" not in patched:
        if _REMOTE_DIRECT_LEGACY_BLOCK not in patched:
            log.warning(
                "_magpie_patcher: remote benchmark direct-call block "
                "not found in %s; custom-tokenizer trust patch could not be applied",
                src,
            )
            return False
        patched = patched.replace(
            _REMOTE_DIRECT_LEGACY_BLOCK,
            _REMOTE_DIRECT_PATCHED_BLOCK,
            1,
        )

    if _LOCAL_TRUST_SENTINEL not in patched and _LOCAL_PATH_MARKER in patched:
        if _LOCAL_TRUST_ARGS_LEGACY_BLOCK not in patched or _LOCAL_CLIENT_LEGACY_BLOCK not in patched:
            log.warning(
                "_magpie_patcher: local benchmark client block not found "
                "in %s; custom-tokenizer trust patch could not be applied",
                src,
            )
            return False
        patched = patched.replace(
            _LOCAL_TRUST_ARGS_LEGACY_BLOCK,
            _LOCAL_TRUST_ARGS_PATCHED_BLOCK,
            1,
        )
        patched = patched.replace(
            _LOCAL_CLIENT_LEGACY_BLOCK,
            _LOCAL_CLIENT_PATCHED_BLOCK,
            1,
        )

    if patched == original:
        return True

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=f".{src.name}.hyperloom_",
        log_prefix="_magpie_patcher",
    ):
        return False

    log.info(
        "_magpie_patcher: applied SGLang client trust patches to %s",
        src,
    )
    return True


def ensure_client_trust_compat(magpie_dir: Path | str | None = None) -> bool:
    """Public, run-time-safe entry point for the SGLang client trust patches.

    ``install.sh`` is the only caller that applies the custom-tokenizer trust
    patches, but it is not the only way Magpie lands on a box:
    :mod:`hyperloom.inference_optimizer.cli.preflight` pip-installs Magpie on
    its own. An unpatched tree ignores ``MAGPIE_TRUST_REMOTE_CODE`` entirely —
    upstream's client call sites never pass the ``trust`` argument — so
    ``benchmark_serving.py`` keeps its ``--trust-remote-code`` default of False
    and loads the tokenizer with ``trust_remote_code=False``. A model shipping
    custom tokenizer code then raises before a single request is issued.
    transformers offers no environment-variable fallback for this, so the CLI
    flag is the only opt-in.

    Only :func:`_apply_sglang_client_trust_patch_atomic` is used here. It
    covers the remote-direct and local client paths of both SGLang scripts and,
    unlike the legacy MI300X patcher, does not require the still-flagged
    ``run_eval`` block — so it remains applicable on a tree whose
    eval-concurrency strip has already run.

    Cheap and idempotent: an already-patched script is skipped without a write.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.

    Returns:
        ``True`` when every resolved SGLang script carries the trust gating, or
        when no SGLang script exists (not applicable). ``False`` when a script
        drifted from the expected shape and could not be patched.
    """
    scripts = [
        s
        for s in (
            _resolve_sglang_mi300x_script_path(magpie_dir),
            _resolve_sglang_mi355x_script_path(magpie_dir),
        )
        if s is not None
    ]
    if not scripts:
        log.info(
            "_magpie_patcher: no SGLang MI300X/MI355X script resolved — "
            "skipping client trust patches (not applicable)",
        )
        return True
    with _file_lock(_LOCK_PATH):
        # Materialized, not short-circuited: every script must be attempted so
        # one drifted file cannot leave a healthy sibling unpatched.
        results = [
            _is_sglang_client_trust_patched(s) or _apply_sglang_client_trust_patch_atomic(s)
            for s in scripts
        ]
    return all(results)


@dataclass(frozen=True)
class MagpiePatchStatus:
    atomic_ok: bool
    remote_trust_ok: bool
    # Classified atomic-patch outcome (``_ATOMIC_REASON_*``): tells an EXPECTED
    # no-op apart from a GENUINE failure where the atomic-write safeguard is absent.
    atomic_reason: str = _ATOMIC_REASON_MISSING
    # Whether the redundant ``--concurrent-requests`` eval flag was stripped from
    # every generic benchmark script (or none needed it). Defaults True
    # (not-applicable) so it never falsely fails install.
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
    inferencex_dir: Path | str | None = None,
) -> MagpiePatchStatus:
    """Return independent status for atomic-copy and remote-trust patches.

    Keeps a drift in the optional SGLang remote-client trust patch from being
    reported as a generic atomic-copy failure.

    The eval-concurrency fixes (strip the redundant ``--concurrent-requests``
    flag from the Magpie + InferenceX benchmark scripts, and make InferenceX's
    ``run_lm_eval`` tolerant of it) are applied **independently** of the
    ``benchmarker.py`` atomic-copy patch: a missing / stale ``benchmarker.py``
    used to early-return and silently skip them, so an unresolved ``MAGPIE_PATH``
    left the fatal flag live in the InferenceX copies that actually execute.

    Args:
        magpie_dir: Magpie root override; falls back to ``$MAGPIE_PATH`` when
            falsy.
        inferencex_dir: InferenceX root override; falls back to
            ``$INFERENCEX_PATH`` when falsy.

    Returns:
        A ``MagpiePatchStatus`` carrying the atomic-copy and remote-trust
        outcomes plus the classified ``atomic_reason``.
    """
    src = _resolve_benchmarker_path(magpie_dir)

    with _file_lock(_LOCK_PATH):
        if src is None:
            # Eval-concurrency fixes run REGARDLESS of benchmarker.py resolution:
            # the fatal --concurrent-requests flag lives in the generic *.sh
            # scripts (Magpie source + the InferenceX/benchmarks copies that
            # actually execute), not in benchmarker.py. A missing / stale
            # benchmarker.py must NOT silently skip them.
            applied_ok = _apply_eval_concurrency_fixes(magpie_dir, inferencex_dir)
            # Align install-time with run-time: a defence-in-depth patch that
            # could not be applied is NOT fatal unless a live flag survives.
            eval_flag_ok = _eval_concurrency_unblocked(applied_ok, magpie_dir, inferencex_dir)
            log.info(
                "_magpie_patcher: MAGPIE_PATH unset or benchmarker.py missing — "
                "skipping atomic-copy patch (fine for tests / dry-runs); "
                "eval-concurrency fixes still applied where scripts were found "
                "(eval_flag_ok=%s)",
                eval_flag_ok,
            )
            # remote_trust_ok True here means "not applicable" (no Magpie tree to
            # inspect); atomic_ok=False + reason=missing is fail-soft (install.sh
            # warns, does not abort) and is NOT a genuine failure.
            return MagpiePatchStatus(
                atomic_ok=False,
                remote_trust_ok=True,
                atomic_reason=_ATOMIC_REASON_MISSING,
                eval_flag_ok=eval_flag_ok,
            )

        atomic_reason = _apply_patch_atomic_reason(src)
        atomic_ok = atomic_reason not in _ATOMIC_REASONS_GENUINE_FAILURE
        sglang_mi300x_script = _resolve_sglang_mi300x_script_path(magpie_dir)
        sglang_mi355x_script = _resolve_sglang_mi355x_script_path(magpie_dir)
        sglang_scripts = [s for s in (sglang_mi300x_script, sglang_mi355x_script) if s is not None]
        trust_results: list[bool] = []
        # MI300X additionally carries the eval-concurrency inline rewrite; keep
        # that patcher so the script's existing behaviour is unchanged.
        if sglang_mi300x_script is not None:
            trust_results.append(
                _is_remote_trust_patched(sglang_mi300x_script)
                or _apply_remote_trust_patch_atomic(sglang_mi300x_script)
            )
        # Both MI300X and MI355X get the full remote + local client trust patch;
        # the client blocks are byte-identical across the two scripts, so one
        # patcher covers each remote-direct and local-server client path.
        for script in sglang_scripts:
            trust_results.append(
                _is_sglang_client_trust_patched(script)
                or _apply_sglang_client_trust_patch_atomic(script)
            )
        if not trust_results:
            log.info(
                "_magpie_patcher: SGLang MI300X/MI355X scripts missing — "
                "skipping client trust patches (fine for reduced tests / "
                "non-SGLang Magpie layouts)",
            )
            remote_trust_ok = True
        else:
            remote_trust_ok = all(trust_results)
        if not remote_trust_ok:
            log.warning(
                "_magpie_patcher: SGLang remote trust patch did not apply "
                "for one or more remote/local client paths; "
                "MAGPIE_TRUST_REMOTE_CODE=1 will not reach one or more "
                "benchmark_serving.py paths for custom-code models",
            )
        # Eval-concurrency fixes run LAST so the remote-trust patch on
        # sglang_mi300x.sh still finds its (flagged) legacy run_eval block
        # before the generic strip removes the flag from it.
        applied_ok = _apply_eval_concurrency_fixes(magpie_dir, inferencex_dir)
        # Align install-time with run-time: a defence-in-depth patch that could
        # not be applied is NOT fatal unless a live flag actually survives.
        eval_flag_ok = _eval_concurrency_unblocked(applied_ok, magpie_dir, inferencex_dir)
        return MagpiePatchStatus(
            atomic_ok=atomic_ok,
            remote_trust_ok=remote_trust_ok,
            atomic_reason=atomic_reason,
            eval_flag_ok=eval_flag_ok,
        )


def ensure_magpie_atomic_scripts_patch(
    magpie_dir: Path | str | None = None,
) -> bool:
    """Ensure installed Magpie's ``_prepare_benchmark_scripts`` copies each
    script atomically (via ``os.replace``).

    Returns ``True`` when the race is closed (freshly-patched, already-patched,
    or upstream already atomic). Returns ``False`` only when the file is missing
    or neither the legacy block nor an atomic impl is found — the install script
    should fail-loud on ``False``. Concurrency-safe (flock + atomic rename).

    Reflects the atomic-copy patch only. The optional SGLang remote-client trust
    patch is independent; callers that need both must use
    :func:`magpie_scripts_patch_status` and check ``remote_trust_ok`` / ``ok``.

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
    "ensure_eval_concurrency_compat",
    "live_eval_concurrency_flag_scripts",
    "ensure_magpie_atomic_scripts_patch",
    "magpie_scripts_patch_status",
    "_ATOMIC_REASON_APPLIED",
    "_ATOMIC_REASON_ALREADY_PATCHED",
    "_ATOMIC_REASON_UPSTREAM_ATOMIC",
    "_ATOMIC_REASON_MISSING",
    "_ATOMIC_REASON_UNRECOGNIZED_SHAPE",
    "_ATOMIC_REASON_IO_ERROR",
]
