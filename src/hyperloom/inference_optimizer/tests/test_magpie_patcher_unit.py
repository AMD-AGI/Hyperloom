# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the idempotent, atomic Magpie ``benchmarker.py`` patcher
(path resolution, sentinel/legacy detection, upstream-atomic awareness, and
the classified atomic-reason outcomes)."""

from __future__ import annotations

from pathlib import Path


from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp


_LEGACY_SRC = (
    "class Benchmarker:\n"
    "    def _prepare_benchmark_scripts(self, target_dir):\n"
    "        for script in scripts:\n"
    "            shutil.copy2(script, target_file)\n"
    "            target_file.chmod(0o755)\n"
    "        return\n"
)

_SGLANG_LEGACY = (
    "#!/bin/bash\n"
    "    SERVER_MONITOR_ARGS=()\n"
    "    magpie_run_benchmark_serving_remote_direct || exit $?\n"
    '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n'
)

_SGLANG_MI355X_LEGACY = (
    "#!/bin/bash\n"
    "SERVER_MONITOR_ARGS=()\n"
    'if [[ -n "${SERVER_PID:-}" ]]; then\n'
    '  SERVER_MONITOR_ARGS+=(--server-pid "$SERVER_PID")\n'
    "fi\n"
    "    SERVER_MONITOR_ARGS=()\n"
    "    magpie_run_benchmark_serving_remote_direct || exit $?\n"
    '        "${SERVER_MONITOR_ARGS[@]}" \\\n'
    "        --result-dir ${RESULT_DIR:-/workspace/} || exit $?\n"
)


def _make_magpie(root: Path, *, benchmarker: str | None = _LEGACY_SRC, sglang: str | None = _SGLANG_LEGACY) -> Path:
    if benchmarker is not None:
        bp = root / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(benchmarker, encoding="utf-8")
    if sglang is not None:
        sp = root / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(sglang, encoding="utf-8")
    return root


# ---- path resolution ------------------------------------------------------
def test_resolve_benchmarker_none(monkeypatch):
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert mp._resolve_benchmarker_path(None) is None


def test_resolve_benchmarker_env(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path))
    p = mp._resolve_benchmarker_path(None)
    assert p is not None and p.name == "benchmarker.py"


def test_resolve_benchmarker_missing_file(tmp_path):
    assert mp._resolve_benchmarker_path(tmp_path) is None


def test_resolve_sglang(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    assert mp._resolve_sglang_mi300x_script_path(tmp_path) is not None
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert mp._resolve_sglang_mi300x_script_path(None) is None


def test_resolve_sglang_env(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path))
    assert mp._resolve_sglang_mi300x_script_path(None) is not None


def test_resolve_sglang_mi355x(monkeypatch, tmp_path):
    script = tmp_path / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh"
    script.parent.mkdir(parents=True)
    script.write_text(_SGLANG_MI355X_LEGACY, encoding="utf-8")
    assert mp._resolve_sglang_mi355x_script_path(tmp_path) == script
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert mp._resolve_sglang_mi355x_script_path(None) is None


# ---- file lock ------------------------------------------------------------
def test_file_lock_normal(tmp_path):
    lock = str(tmp_path / "x.lock")
    with mp._file_lock(lock):
        pass
    assert Path(lock).exists()


def test_file_lock_unopenable():
    # directory path can't be opened "w" -> yield without exclusion
    with mp._file_lock("/nonexistent_dir_zzz/sub/lock"):
        pass


# ---- _is_patched ----------------------------------------------------------
def test_is_patched(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("nothing here", encoding="utf-8")
    assert mp._is_patched(f) is False
    f.write_text("... Hyperloom #C1 patch ...", encoding="utf-8")
    assert mp._is_patched(f) is True
    assert mp._is_patched(tmp_path / "missing.py") is False


# ---- prepare region + upstream atomic -------------------------------------
def test_extract_prepare_region():
    region = mp._extract_prepare_region(_LEGACY_SRC)
    assert "shutil.copy2" in region
    assert mp._extract_prepare_region("no method here") == ""


def test_extract_prepare_region_blank_and_dedent():
    src = (
        "class C:\n"
        "    def _prepare_benchmark_scripts(self):\n"
        "        a = 1\n"
        "\n"
        "        b = 2\n"
        "    def other(self):\n"
        "        c = 3\n"
    )
    region = mp._extract_prepare_region(src)
    assert "a = 1" in region and "b = 2" in region
    assert "c = 3" not in region


def test_upstream_already_atomic_helper():
    txt = "def x():\n    _copy_benchmark_script_atomic()\n"
    assert mp._upstream_is_already_atomic(txt) is True


def test_upstream_already_atomic_inline():
    txt = (
        "    def _prepare_benchmark_scripts(self):\n"
        "        fd = tempfile.mkstemp(dir=d)\n"
        "        os.replace(tmp, target)\n"
    )
    assert mp._upstream_is_already_atomic(txt) is True


def test_upstream_not_atomic():
    assert mp._upstream_is_already_atomic(_LEGACY_SRC) is False


# ---- _apply_patch_atomic_reason -------------------------------------------
def test_apply_reason_io_error_read(tmp_path):
    assert mp._apply_patch_atomic_reason(tmp_path) == mp._ATOMIC_REASON_IO_ERROR


def test_apply_reason_already_patched(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("Hyperloom #C1 patch present", encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_ALREADY_PATCHED


def test_apply_reason_upstream_atomic(tmp_path):
    f = tmp_path / "b.py"
    f.write_text(
        "def _prepare_benchmark_scripts(self):\n    tempfile.mkstemp(dir=d)\n    os.replace(a, b)\n",
        encoding="utf-8",
    )
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_UPSTREAM_ATOMIC


def test_apply_reason_unrecognized(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("totally different code\n", encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_UNRECOGNIZED_SHAPE


def test_apply_reason_applied(tmp_path):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_APPLIED
    assert "Hyperloom #C1 patch" in f.read_text(encoding="utf-8")


def test_apply_reason_write_error(tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("no space")

    monkeypatch.setattr(mp.tempfile, "mkstemp", _boom)
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_IO_ERROR


def test_apply_reason_fdopen_write_error(tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    # mkstemp succeeds but os.replace fails -> fdopen-path OSError + cleanup.
    monkeypatch.setattr(mp.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_IO_ERROR


# ---- remote trust patch ---------------------------------------------------
def test_is_remote_trust_patched(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text("no sentinel", encoding="utf-8")
    assert mp._is_remote_trust_patched(f) is False
    f.write_text("MAGPIE_TRUST_REMOTE_CODE here", encoding="utf-8")
    assert mp._is_remote_trust_patched(f) is False
    f.write_text(
        "MAGPIE_TRUST_REMOTE_CODE here\nHYPERLOOM_EVAL_CONCURRENCY_FIX\n",
        encoding="utf-8",
    )
    assert mp._is_remote_trust_patched(f) is True
    assert mp._is_remote_trust_patched(tmp_path / "missing") is False


def test_apply_remote_trust_already(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text(
        "MAGPIE_TRUST_REMOTE_CODE\nHYPERLOOM_EVAL_CONCURRENCY_FIX\n",
        encoding="utf-8",
    )
    assert mp._apply_remote_trust_patch_atomic(f) is True


def test_apply_remote_trust_legacy_missing(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text("unrelated", encoding="utf-8")
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_remote_trust_applied(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    assert mp._apply_remote_trust_patch_atomic(f) is True
    assert "MAGPIE_TRUST_REMOTE_CODE" in f.read_text(encoding="utf-8")


def test_apply_remote_trust_read_error(tmp_path):
    assert mp._apply_remote_trust_patch_atomic(tmp_path) is False


def test_apply_remote_trust_write_error(tmp_path, monkeypatch):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    monkeypatch.setattr(mp.tempfile, "mkstemp", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_remote_trust_fdopen_write_error(tmp_path, monkeypatch):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    monkeypatch.setattr(mp.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_sglang_client_trust_applied_and_idempotent(tmp_path):
    # A script carrying the local-server client path (either mi300x or mi355x —
    # the client blocks are byte-identical) gets both client paths gated.
    f = tmp_path / "sglang_mi355x.sh"
    f.write_text(_SGLANG_MI355X_LEGACY, encoding="utf-8")

    assert mp._apply_sglang_client_trust_patch_atomic(f) is True
    first = f.read_text(encoding="utf-8")
    assert "HYPERLOOM_SGLANG_LOCAL_TRUST" in first
    assert "magpie_run_benchmark_serving_remote_direct trust" in first
    assert "CLIENT_TRUST_ARGS+=(--trust-remote-code)" in first
    assert '"${CLIENT_TRUST_ARGS[@]}"' in first
    assert mp._is_sglang_client_trust_patched(f) is True

    assert mp._apply_sglang_client_trust_patch_atomic(f) is True
    assert f.read_text(encoding="utf-8") == first


def test_apply_sglang_client_trust_rejects_drifted_local_shape(tmp_path):
    # Local path present (marker) but the splice block drifted -> fail loud.
    f = tmp_path / "sglang_mi355x.sh"
    f.write_text(
        _SGLANG_MI355X_LEGACY.replace(
            '        "${SERVER_MONITOR_ARGS[@]}" \\\n',
            '        "${SERVER_MONITOR_ARGS[@]}" --changed \\\n',
        ),
        encoding="utf-8",
    )
    assert mp._apply_sglang_client_trust_patch_atomic(f) is False


def test_apply_sglang_client_trust_remote_only_skips_local(tmp_path):
    # Reduced script with only the remote-direct path (no local marker): remote
    # gets gated, the local splice is skipped rather than reported as drift.
    f = tmp_path / "sglang_mi300x.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")

    assert mp._apply_sglang_client_trust_patch_atomic(f) is True
    text = f.read_text(encoding="utf-8")
    assert "magpie_run_benchmark_serving_remote_direct trust" in text
    assert "HYPERLOOM_SGLANG_LOCAL_TRUST" not in text
    assert mp._is_sglang_client_trust_patched(f) is True
    # Idempotent.
    assert mp._apply_sglang_client_trust_patch_atomic(f) is True


def test_apply_sglang_client_trust_full_mi300x_gets_local(tmp_path):
    # A realistic mi300x script (with the local-server client path) gets BOTH
    # the remote-direct and local-server trust gates — closing the gap where
    # the earlier patch only ever reached mi355x.
    f = tmp_path / "sglang_mi300x.sh"
    f.write_text(_SGLANG_MI355X_LEGACY, encoding="utf-8")

    assert mp._apply_sglang_client_trust_patch_atomic(f) is True
    text = f.read_text(encoding="utf-8")
    assert "magpie_run_benchmark_serving_remote_direct trust" in text
    assert "HYPERLOOM_SGLANG_LOCAL_TRUST" in text
    assert '"${CLIENT_TRUST_ARGS[@]}"' in text


# ---- MagpiePatchStatus ----------------------------------------------------
def test_status_properties():
    s = mp.MagpiePatchStatus(atomic_ok=True, remote_trust_ok=True, atomic_reason=mp._ATOMIC_REASON_APPLIED)
    assert s.ok is True
    assert s.atomic_genuine_failure is False
    s2 = mp.MagpiePatchStatus(atomic_ok=False, remote_trust_ok=True, atomic_reason=mp._ATOMIC_REASON_IO_ERROR)
    assert s2.ok is False
    assert s2.atomic_genuine_failure is True


# ---- top-level orchestration ----------------------------------------------
def test_patch_status_missing(monkeypatch):
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    s = mp.magpie_scripts_patch_status(None)
    assert s.atomic_ok is False
    assert s.atomic_reason == mp._ATOMIC_REASON_MISSING
    assert s.remote_trust_ok is True


def test_patch_status_full_flow(tmp_path):
    _make_magpie(tmp_path)
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.atomic_ok is True
    assert s.atomic_reason == mp._ATOMIC_REASON_APPLIED
    assert s.remote_trust_ok is True
    assert s.ok is True


def test_patch_status_no_sglang(tmp_path):
    _make_magpie(tmp_path, sglang=None)
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.remote_trust_ok is True  # no script -> not applicable


def test_patch_status_remote_trust_fails(tmp_path):
    _make_magpie(tmp_path, sglang="#!/bin/bash\nunrelated content\n")
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.remote_trust_ok is False
    assert s.ok is False


def test_ensure_wrapper(tmp_path):
    _make_magpie(tmp_path)
    assert mp.ensure_magpie_atomic_scripts_patch(tmp_path) is True


# ---- eval-concurrency fixes (--concurrent-requests) -----------------------
_VLLM_LEGACY = (
    "#!/bin/bash\n"
    'if [[ "$RUN_EVAL" = "true" ]]; then\n'
    '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n'
    "fi\n"
)

_BENCHMARK_LIB_LEGACY = (
    "#!/bin/bash\n"
    "run_lm_eval() {\n"
    '    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"\n'
    "    while [[ $# -gt 0 ]]; do\n"
    "        case $1 in\n"
    '            --port)           port="$2"; shift 2 ;;\n'
    '            --top-p)          top_p="$2"; shift 2 ;;\n'
    '            *)                echo "Unknown parameter: $1"; return 1 ;;\n'
    "        esac\n"
    "    done\n"
    "}\n"
)


def _make_inferencex(
    root: Path,
    *,
    vllm: str | None = _VLLM_LEGACY,
    benchmark_lib: str | None = _BENCHMARK_LIB_LEGACY,
) -> Path:
    bench = root / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    if vllm is not None:
        (bench / "vllm_mi355x.sh").write_text(vllm, encoding="utf-8")
    if benchmark_lib is not None:
        (bench / "benchmark_lib.sh").write_text(benchmark_lib, encoding="utf-8")
    return root


def test_resolve_inferencex_benchmarks_dir(monkeypatch, tmp_path):
    _make_inferencex(tmp_path)
    assert mp._resolve_inferencex_benchmarks_dir(tmp_path) == tmp_path / "benchmarks"
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    assert mp._resolve_inferencex_benchmarks_dir(None) == tmp_path / "benchmarks"
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert mp._resolve_inferencex_benchmarks_dir(None) is None
    assert mp._resolve_inferencex_benchmarks_dir(tmp_path / "nope") is None


def test_resolve_inferencex_benchmark_lib(tmp_path):
    _make_inferencex(tmp_path)
    lib = mp._resolve_inferencex_benchmark_lib(tmp_path)
    assert lib is not None and lib.name == "benchmark_lib.sh"
    assert mp._resolve_inferencex_benchmark_lib(tmp_path / "nope") is None


def test_run_lm_eval_arg_patch_applied(tmp_path):
    _make_inferencex(tmp_path)
    lib = tmp_path / "benchmarks" / "benchmark_lib.sh"
    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    text = lib.read_text(encoding="utf-8")
    assert "--concurrent-requests|--concurrent_requests" in text
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in text
    # Idempotent second call.
    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True


def test_run_lm_eval_arg_patch_unrecognized(tmp_path):
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is False


def test_eval_flag_stripped_from_inferencex_dir(tmp_path):
    _make_inferencex(tmp_path)
    assert mp._apply_eval_concurrency_fixes(None, tmp_path) is True
    vllm = (tmp_path / "benchmarks" / "vllm_mi355x.sh").read_text(encoding="utf-8")
    assert "--concurrent-requests" not in vllm
    lib = (tmp_path / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in lib


def test_eval_concurrency_fixes_idempotent(tmp_path):
    """Regression: a 2nd pass must stay ok. The parser patch leaves a legit
    ``--concurrent-requests`` case in benchmark_lib.sh; the flag-strip scan must
    skip the library rather than mis-report it as an unrecognised shape."""
    _make_inferencex(tmp_path)
    assert mp._apply_eval_concurrency_fixes(None, tmp_path) is True
    # Second pass: benchmark_lib.sh now carries the parser sentinel + flag.
    assert mp._apply_eval_concurrency_fixes(None, tmp_path) is True
    lib = (tmp_path / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    # The parser case survived (not stripped) and stayed idempotent.
    assert lib.count("--concurrent-requests|--concurrent_requests") == 1
    assert "--concurrent-requests" not in (
        tmp_path / "benchmarks" / "vllm_mi355x.sh"
    ).read_text(encoding="utf-8")


def test_eval_fixes_run_when_benchmarker_missing(monkeypatch, tmp_path):
    """Regression: a missing/stale benchmarker.py must NOT skip the eval fixes.

    Previously ``magpie_scripts_patch_status`` early-returned when
    ``benchmarker.py`` was unresolved, leaving the fatal ``--concurrent-requests``
    flag live in the InferenceX copies that actually execute.
    """
    ix = _make_inferencex(tmp_path / "ix")
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.setenv("INFERENCEX_PATH", str(ix))
    status = mp.magpie_scripts_patch_status(None, str(ix))
    # Atomic patch is a no-op (no benchmarker.py) but the eval fixes ran.
    assert status.atomic_reason == mp._ATOMIC_REASON_MISSING
    assert status.eval_flag_ok is True
    vllm = (ix / "benchmarks" / "vllm_mi355x.sh").read_text(encoding="utf-8")
    assert "--concurrent-requests" not in vllm
    lib = (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in lib


def test_full_flow_covers_inferencex_and_ordering(tmp_path):
    """Full status flow: atomic + remote-trust + eval strip across both dirs,
    with the remote-trust patch on sglang running BEFORE the generic strip."""
    magpie = _make_magpie(tmp_path / "magpie")
    # Add a flagged generic vllm script to the Magpie scripts dir too.
    (magpie / "Magpie" / "scripts" / "benchmark" / "vllm_mi355x.sh").write_text(
        _VLLM_LEGACY, encoding="utf-8"
    )
    ix = _make_inferencex(tmp_path / "ix")
    status = mp.magpie_scripts_patch_status(str(magpie), str(ix))
    assert status.atomic_ok is True
    assert status.remote_trust_ok is True  # sglang patched before strip removed its flag
    assert status.eval_flag_ok is True
    assert status.ok is True
    # sglang got the remote-trust rewrite (no bare flag left).
    sglang = (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh").read_text(encoding="utf-8")
    assert "MAGPIE_TRUST_REMOTE_CODE" in sglang
    assert "--concurrent-requests" not in sglang
    # Both Magpie's and InferenceX's generic vllm scripts were stripped.
    assert "--concurrent-requests" not in (
        magpie / "Magpie" / "scripts" / "benchmark" / "vllm_mi355x.sh"
    ).read_text(encoding="utf-8")
    assert "--concurrent-requests" not in (
        ix / "benchmarks" / "vllm_mi355x.sh"
    ).read_text(encoding="utf-8")


# ---- regression: run-time eval-concurrency compat (2026-07-27 outage) ------
# Reproduces the exact failure that killed a Qwen3-8B optimization run:
# preflight pip-installed Magpie into site-packages and cloned InferenceX
# WITHOUT ever running the patcher (only install.sh did), so
# ``Magpie/scripts/benchmark/sglang_mi355x.sh`` kept upstream's
#     run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
# Magpie's ``_prepare_benchmark_scripts`` then re-copied that script into
# ``<inferencex>/benchmarks/`` at run time, InferenceX's ``run_lm_eval``
# rejected the flag ("Unknown parameter: --concurrent-requests"), the benchmark
# aborted with no ``results*.json``, and the run stopped with
# ``baseline_accuracy_failed``.
_SGLANG_MI355X_FLAGGED = (
    "#!/bin/bash\n"
    'if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then\n'
    '    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then\n'
    "        magpie_run_eval_remote_direct || exit $?\n"
    "    else\n"
    '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n'
    "        append_lm_eval_summary\n"
    "    fi\n"
    "fi\n"
)


def _make_sitepackages_magpie(root: Path) -> Path:
    """Magpie as pip installs it: package root with scripts/benchmark/*.sh."""
    bench = root / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "sglang_mi355x.sh").write_text(_SGLANG_MI355X_FLAGGED, encoding="utf-8")
    return root


def test_ensure_eval_concurrency_compat_strips_sglang_mi355x(tmp_path):
    """The public run-time entry point removes the flag from the Magpie tree
    Magpie re-copies from, so the executed copy is clean."""
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    ix = _make_inferencex(tmp_path / "ix", vllm=None)

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is True

    script = (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert "--concurrent-requests" not in script
    # Concurrency still reaches lm-eval: run_lm_eval resolves it from
    # EVAL_CONCURRENT_REQUESTS (fallback CONC), which the untouched call keeps.
    assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in script
    # The remote-direct shim (which never took the flag) is untouched.
    assert "magpie_run_eval_remote_direct || exit $?" in script


def test_ensure_eval_concurrency_compat_makes_run_lm_eval_tolerant(tmp_path):
    """Belt for Magpie's run-time re-copy: even if a flagged script slips into
    ``<inferencex>/benchmarks/``, ``run_lm_eval`` must not abort on it."""
    ix = _make_inferencex(tmp_path / "ix", vllm=None)

    assert mp.ensure_eval_concurrency_compat(None, str(ix)) is True

    lib = (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in lib
    assert '--concurrent-requests|--concurrent_requests) concurrent_requests="$2"' in lib
    # The catch-all that produced "Unknown parameter: --concurrent-requests" is
    # now reached only by genuinely unknown flags.
    assert lib.index("--concurrent-requests|--concurrent_requests") < lib.index('echo "Unknown parameter: $1"')


def test_ensure_eval_concurrency_compat_falls_back_to_env(monkeypatch, tmp_path):
    """With no explicit args the entry point resolves $MAGPIE_PATH / $INFERENCEX_PATH."""
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    ix = _make_inferencex(tmp_path / "ix", vllm=None)
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))
    monkeypatch.setenv("INFERENCEX_PATH", str(ix))

    assert mp.ensure_eval_concurrency_compat() is True

    assert (
        "--concurrent-requests"
        not in (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    )
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")


def test_ensure_eval_concurrency_compat_idempotent(tmp_path):
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    ix = _make_inferencex(tmp_path / "ix", vllm=None)
    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is True
    first_script = (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    first_lib = (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is True

    assert (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(
        encoding="utf-8"
    ) == first_script
    assert (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8") == first_lib


def test_ensure_eval_concurrency_compat_reports_unstrippable(tmp_path):
    """An unrecognised flag shape must report False (callers fail loudly), not
    silently leave a fatal flag live."""
    magpie = tmp_path / "site-packages"
    bench = magpie / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n',
        encoding="utf-8",
    )
    assert mp.ensure_eval_concurrency_compat(str(magpie), None) is False


def test_ensure_eval_concurrency_compat_noop_without_trees(monkeypatch, tmp_path):
    """No Magpie / InferenceX on disk is 'not applicable', not a failure."""
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert mp.ensure_eval_concurrency_compat(str(tmp_path / "nope"), str(tmp_path / "nope2")) is True


def test_ensure_eval_concurrency_compat_exported():
    assert "ensure_eval_concurrency_compat" in mp.__all__


# ---- live-flag detection: the precise "eval will abort" condition ----------
def test_live_flag_scan_finds_flagged_caller(tmp_path):
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    hits = mp.live_eval_concurrency_flag_scripts(str(magpie), None)
    assert [p.name for p in hits] == ["sglang_mi355x.sh"]


def test_live_flag_scan_ignores_benchmark_lib_parser_case(tmp_path):
    """benchmark_lib.sh's own arg parser names the flag legitimately."""
    ix = _make_inferencex(tmp_path / "ix", vllm=None)
    lib = ix / "benchmarks" / "benchmark_lib.sh"
    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    assert "--concurrent-requests" in lib.read_text(encoding="utf-8")
    assert mp.live_eval_concurrency_flag_scripts(None, str(ix)) == []


def test_live_flag_scan_ignores_env_prefixed_patched_form(tmp_path):
    """The supported rewrite (EVAL_CONCURRENT_REQUESTS=... run_eval) is clean."""
    bench = tmp_path / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "sglang_mi300x.sh").write_text(
        mp._RUN_EVAL_PATCHED_BLOCK,
        encoding="utf-8",
    )
    assert mp.live_eval_concurrency_flag_scripts(str(tmp_path), None) == []


def test_compat_true_when_only_the_belt_fails(tmp_path):
    """Regression: a reduced / already-fixed benchmark_lib.sh whose parser block
    is unrecognised must NOT be reported as blocking. Nothing is actually
    passing the flag, so accuracy eval runs fine."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(
        "run_lm_eval() { : ; }\n", encoding="utf-8"
    )
    assert mp._apply_eval_concurrency_fixes(None, str(ix)) is False
    assert mp.ensure_eval_concurrency_compat(None, str(ix)) is True


def test_compat_true_when_parser_absorbs_an_unstrippable_flag(tmp_path):
    """A flag shape the strip cannot rewrite is harmless once run_lm_eval parses
    it — the belt is doing its job, so do not block the run."""
    magpie = tmp_path / "site-packages"
    bench = magpie / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n',
        encoding="utf-8",
    )
    ix = _make_inferencex(tmp_path / "ix", vllm=None)

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is True

    # Flag survived (unrecognised shape) but the parser now accepts it.
    assert "--concurrent-requests 64" in (bench / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
