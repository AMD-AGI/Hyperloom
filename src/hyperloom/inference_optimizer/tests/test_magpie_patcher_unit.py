# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the idempotent, atomic Magpie ``benchmarker.py`` patcher (path resolution, sentinel/legacy detection, upstream-atomic awareness, and the classified atomic-reason outcomes)."""

from __future__ import annotations

from pathlib import Path

import pytest

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
_BENCHMARKER_REL = ("Magpie", "modes", "benchmark", "benchmarker.py")


def test_resolve_component_path_none(monkeypatch):
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert mp._resolve_component_path(None, "MAGPIE_PATH", *_BENCHMARKER_REL) is None


def test_resolve_component_path_explicit_dir_and_env(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    assert mp._resolve_component_path(tmp_path, "MAGPIE_PATH", *_BENCHMARKER_REL) == tmp_path.joinpath(
        *_BENCHMARKER_REL
    )
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path))
    p = mp._resolve_component_path(None, "MAGPIE_PATH", *_BENCHMARKER_REL)
    assert p is not None and p.name == "benchmarker.py"


def test_resolve_component_path_missing_file(tmp_path):
    assert mp._resolve_component_path(tmp_path, "MAGPIE_PATH", *_BENCHMARKER_REL) is None


def test_resolve_component_path_dir_check(monkeypatch, tmp_path):
    _make_inferencex(tmp_path)
    assert mp._resolve_component_path(tmp_path, "INFERENCEX_PATH", "benchmarks", check="dir") == tmp_path / "benchmarks"
    assert mp._resolve_component_path(tmp_path, "INFERENCEX_PATH", "benchmarks") is None
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    assert mp._resolve_component_path(None, "INFERENCEX_PATH", "benchmarks", check="dir") == tmp_path / "benchmarks"
    assert mp._resolve_component_path(tmp_path / "nope", "INFERENCEX_PATH", "benchmarks", check="dir") is None


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


def _fail_replace(monkeypatch):
    monkeypatch.setattr(mp._common_io.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))


def test_apply_reason_write_error(tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    _fail_replace(monkeypatch)
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_IO_ERROR
    assert f.read_text(encoding="utf-8") == _LEGACY_SRC
    assert [p.name for p in tmp_path.iterdir()] == ["b.py"]


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
    _fail_replace(monkeypatch)
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_sglang_client_trust_applied_and_idempotent(tmp_path):
    # A script carrying the local-server client path (either mi300x or mi355x — the client blocks are byte-identical)
    # gets both client paths gated.
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
    # Reduced script with only the remote-direct path (no local marker): remote gets gated, the local splice is
    # skipped rather than reported as drift.
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
    # A realistic mi300x script (with the local-server client path) gets BOTH the remote-direct and local-server trust
    # gates — closing the gap where the earlier patch only ever reached mi355x.
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
    """Regression: a 2nd pass must stay ok."""
    _make_inferencex(tmp_path)
    assert mp._apply_eval_concurrency_fixes(None, tmp_path) is True
    # Second pass: benchmark_lib.sh now carries the parser sentinel + flag.
    assert mp._apply_eval_concurrency_fixes(None, tmp_path) is True
    lib = (tmp_path / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    # The parser case survived (not stripped) and stayed idempotent.
    assert lib.count("--concurrent-requests|--concurrent_requests") == 1
    assert "--concurrent-requests" not in (tmp_path / "benchmarks" / "vllm_mi355x.sh").read_text(encoding="utf-8")


def test_eval_fixes_run_when_benchmarker_missing(monkeypatch, tmp_path):
    """Regression: a missing/stale benchmarker.py must NOT skip the eval fixes."""
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
    """Full status flow: atomic + remote-trust + eval strip across both dirs, with the remote-trust patch on sglang
    running BEFORE the generic strip.
    """
    magpie = _make_magpie(tmp_path / "magpie")
    # Add a flagged generic vllm script to the Magpie scripts dir too.
    (magpie / "Magpie" / "scripts" / "benchmark" / "vllm_mi355x.sh").write_text(_VLLM_LEGACY, encoding="utf-8")
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
    assert "--concurrent-requests" not in (magpie / "Magpie" / "scripts" / "benchmark" / "vllm_mi355x.sh").read_text(
        encoding="utf-8"
    )
    assert "--concurrent-requests" not in (ix / "benchmarks" / "vllm_mi355x.sh").read_text(encoding="utf-8")


# ---- regression: run-time eval-concurrency compat (2026-07-27 outage) ------ Reproduces the exact failure that
# killed a Qwen3-8B optimization run: preflight pip-installed Magpie into site-packages and cloned InferenceX WITHOUT
# ever running the patcher (only install.sh did), so ``Magpie/scripts/benchmark/sglang_mi355x.sh`` kept upstream's
# run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC Magpie's ``_prepare_benchmark_scripts`` then
# re-copied that script into ``<inferencex>/benchmarks/`` at run time, InferenceX's ``run_lm_eval`` rejected the flag
# ("Unknown parameter: --concurrent-requests"), the benchmark aborted with no ``results*.json``, and the run stopped
# with ``baseline_accuracy_failed``.
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
    """The public run-time entry point removes the flag from the Magpie tree Magpie re-copies from, so the executed copy is clean."""
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    ix = _make_inferencex(tmp_path / "ix", vllm=None)

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is True

    script = (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert "--concurrent-requests" not in script
    # Concurrency still reaches lm-eval: run_lm_eval resolves it from EVAL_CONCURRENT_REQUESTS (fallback CONC), which
    # the untouched call keeps.
    assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in script
    # The remote-direct shim (which never took the flag) is untouched.
    assert "magpie_run_eval_remote_direct || exit $?" in script


def test_ensure_eval_concurrency_compat_makes_run_lm_eval_tolerant(tmp_path):
    """Belt for Magpie's run-time re-copy: even if a flagged script slips into ``<inferencex>/benchmarks/``, ``run_lm_eval`` must not abort on it."""
    ix = _make_inferencex(tmp_path / "ix", vllm=None)

    assert mp.ensure_eval_concurrency_compat(None, str(ix)) is True

    lib = (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in lib
    assert '--concurrent-requests|--concurrent_requests) concurrent_requests="$2"' in lib
    # The catch-all that produced "Unknown parameter: --concurrent-requests" is now reached only by genuinely unknown
    # flags.
    assert lib.index("--concurrent-requests|--concurrent_requests") < lib.index('echo "Unknown parameter: $1"')


def test_ensure_eval_concurrency_compat_falls_back_to_env(monkeypatch, tmp_path):
    """With no explicit args the entry point resolves $MAGPIE_PATH / $INFERENCEX_PATH."""
    magpie = _make_sitepackages_magpie(tmp_path / "site-packages")
    ix = _make_inferencex(tmp_path / "ix", vllm=None)
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))
    monkeypatch.setenv("INFERENCEX_PATH", str(ix))

    assert mp.ensure_eval_concurrency_compat() is True

    assert "--concurrent-requests" not in (magpie / "Magpie" / "scripts" / "benchmark" / "sglang_mi355x.sh").read_text(
        encoding="utf-8"
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
    """An unrecognised flag shape must report False (callers fail loudly), not silently leave a fatal flag live."""
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
    """Regression: a reduced / already-fixed benchmark_lib.sh whose parser block is unrecognised must NOT be reported as blocking."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    assert mp._apply_eval_concurrency_fixes(None, str(ix)) is False
    assert mp.ensure_eval_concurrency_compat(None, str(ix)) is True


def test_compat_true_when_parser_absorbs_an_unstrippable_flag(tmp_path):
    """A flag shape the strip cannot rewrite is harmless once run_lm_eval parses it — the belt is doing its job, so do not block the run."""
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


def test_compat_false_when_an_unstrippable_flag_meets_a_strict_parser(tmp_path):
    """The one genuinely fatal state: a caller still passes the flag AND run_lm_eval still rejects it."""
    magpie = tmp_path / "site-packages"
    bench = magpie / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    # A shape the strip cannot rewrite, so the flag survives the patch.
    (bench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n',
        encoding="utf-8",
    )
    # A benchmark_lib.sh whose parser cannot be taught the flag either: no run_lm_eval definition at all, so the belt
    # has nothing to patch.
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text("# no run_lm_eval here\n", encoding="utf-8")

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is False
    # The blocker is still reported by the scanner, so callers can name the file.
    assert [p.name for p in mp.live_eval_concurrency_flag_scripts(str(magpie), None)] == ["sglang_mi355x.sh"]


# ---- unreadable files: the patcher must degrade, never crash a run ---------
def _unreadable(path):
    """A path that exists but raises OSError on read."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_unreadable_caller_script_is_reported_not_raised(tmp_path):
    """A script the patcher cannot read must fail the pass, not kill preflight."""
    bench = tmp_path / "site-packages" / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    _unreadable(bench / "sglang_mi355x.sh")

    assert mp._apply_eval_concurrency_fixes(str(tmp_path / "site-packages"), None) is False


def test_unreadable_benchmark_lib_reads_as_intolerant(tmp_path):
    """Cannot prove the parser accepts the flag -> must assume it does not."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    _unreadable(ix / "benchmarks" / "benchmark_lib.sh")

    assert mp._inferencex_tolerates_eval_flag(str(ix)) is False


def test_flag_scan_skips_unreadable_scripts_without_failing(tmp_path):
    """The scanner reports what it can read; an unreadable entry is not a hit."""
    bench = tmp_path / "site-packages" / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    _unreadable(bench / "vllm_mi355x.sh")
    (bench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n',
        encoding="utf-8",
    )

    hits = mp.live_eval_concurrency_flag_scripts(str(tmp_path / "site-packages"), None)
    assert [p.name for p in hits] == ["sglang_mi355x.sh"]


def test_parser_patch_reports_failure_when_the_lib_cannot_be_read(tmp_path):
    """An unreadable benchmark_lib.sh cannot be taught the flag -> False."""
    lib = tmp_path / "benchmark_lib.sh"
    lib.mkdir()  # a directory: read_text raises OSError even as root

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is False


def test_parser_patch_reports_failure_on_an_unrecognised_parser_block(tmp_path):
    """No legacy parser block to rewrite -> nothing patched, report False."""
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text("run_lm_eval() { : ; }\n", encoding="utf-8")

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is False


# ---- merged-case parser (InferenceX a4bb43af+) ---------------------------- The pinned InferenceX (a4bb43af)
# refactored run_lm_eval's arg parser into a single merged ``--port|--task|...|--top-p)`` case with an inner dispatch
# and a ``>&2`` / ``return 2`` catch-all.
_BENCHMARK_LIB_MERGED_CASE = (
    "#!/bin/bash\n"
    "run_lm_eval() {\n"
    '    local port="${PORT:-8888}"\n'
    "    local top_p=1\n"
    '    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"\n'
    "    while [[ $# -gt 0 ]]; do\n"
    '        case "$1" in\n'
    "            --port|--task|--results-dir|--gen-max-tokens|--temperature|--top-p)\n"
    '                case "$1" in\n'
    '                    --port)           port="$2" ;;\n'
    '                    --top-p)          top_p="$2" ;;\n'
    "                esac\n"
    "                shift 2\n"
    "                ;;\n"
    "            *)\n"
    '                echo "Unknown parameter: $1" >&2\n'
    "                return 2\n"
    "                ;;\n"
    "        esac\n"
    "    done\n"
    "}\n"
)


def test_merged_case_parser_is_taught_the_flag(tmp_path):
    """The a4bb43af merged-case parser must be patched to accept the flag."""
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text(_BENCHMARK_LIB_MERGED_CASE, encoding="utf-8")

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    text = lib.read_text(encoding="utf-8")
    assert "--concurrent-requests|--concurrent_requests" in text
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in text
    # Idempotent second pass.
    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    assert lib.read_text(encoding="utf-8").count("--concurrent-requests|--concurrent_requests") == 1


def test_merged_case_env_only_ix_is_not_a_false_positive(tmp_path):
    """Full status: merged-case parser + env concurrency + no live flag => ok."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "benchmark_lib.sh").write_text(_BENCHMARK_LIB_MERGED_CASE, encoding="utf-8")
    # A caller script that takes concurrency via env, not the flag (no live flag).
    (bench / "vllm_mi355x.sh").write_text(
        "#!/bin/bash\n"
        'if [[ "$RUN_EVAL" = "true" ]]; then\n'
        '        run_eval --framework lm-eval --port "$PORT" || exit $?\n'
        "fi\n",
        encoding="utf-8",
    )

    status = mp.magpie_scripts_patch_status(None, str(ix))
    # The merged-case parser was taught the flag, so the eval fix succeeded.
    assert status.eval_flag_ok is True
    # atomic is a benign no-op here (no MAGPIE_PATH / benchmarker.py), not a genuine failure; install.sh treats
    # reason=missing as fail-soft.
    assert status.atomic_reason == mp._ATOMIC_REASON_MISSING
    assert status.atomic_genuine_failure is False
    assert mp.live_eval_concurrency_flag_scripts(None, str(ix)) == []


def test_unpatchable_parser_without_live_flag_is_not_fatal(tmp_path):
    """Narrowed judgement: even a parser we cannot teach must not fail install when no caller passes the flag (aligns install-time with run-time)."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    # A run_lm_eval whose parser shape we cannot recognise at all.
    (bench / "benchmark_lib.sh").write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    # No live --concurrent-requests anywhere.
    (bench / "vllm_mi355x.sh").write_text(
        "#!/bin/bash\n"
        'if [[ "$RUN_EVAL" = "true" ]]; then\n'
        '        run_eval --framework lm-eval --port "$PORT" || exit $?\n'
        "fi\n",
        encoding="utf-8",
    )

    status = mp.magpie_scripts_patch_status(None, str(ix))
    assert mp.live_eval_concurrency_flag_scripts(None, str(ix)) == []
    # The belt patch could not apply, but nothing is blocked -> not fatal.
    assert status.eval_flag_ok is True


# A benchmark_lib.sh with EARLIER functions that carry an identical ``*)`` catch-all (real a4bb43af has several before
# run_lm_eval, e.g. at lines 285 & 451).
_BENCHMARK_LIB_MULTI_CATCHALL = (
    "#!/bin/bash\n"
    "wait_for_server_ready() {\n"
    "    while [[ $# -gt 0 ]]; do\n"
    '        case "$1" in\n'
    '            --port) port="$2"; shift 2 ;;\n'
    "            *)\n"
    '                echo "Unknown parameter: $1" >&2\n'
    "                return 2\n"
    "                ;;\n"
    "        esac\n"
    "    done\n"
    "}\n"
    "\n"
    "parse_other() {\n"
    '    case "$1" in\n'
    "        *)\n"
    '            echo "Unknown parameter: $1" >&2\n'
    "            return 1\n"
    "            ;;\n"
    "    esac\n"
    "}\n"
    "\n" + _BENCHMARK_LIB_MERGED_CASE
)


def test_merged_case_patch_lands_inside_run_lm_eval_only(tmp_path):
    """Regression for the mis-patch bug: with earlier functions sharing the same ``*)`` catch-all, the flag case must be spliced into run_lm_eval, not the first matching catch-all in the file."""
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text(_BENCHMARK_LIB_MULTI_CATCHALL, encoding="utf-8")

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    text = lib.read_text(encoding="utf-8")

    # Exactly one flag case was added, and it sits inside run_lm_eval's body.
    assert text.count("--concurrent-requests|--concurrent_requests") == 1
    region = mp._extract_run_lm_eval_region(text)
    assert region is not None
    body = text[region[0] : region[1]]
    assert "--concurrent-requests|--concurrent_requests" in body
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in body
    # The earlier functions' catch-alls were left untouched.
    before = text[: region[0]]
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL not in before
    assert "--concurrent-requests" not in before
    # Tolerance check (scoped to run_lm_eval) now reports True for this tree.
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(text, encoding="utf-8")
    assert mp._inferencex_tolerates_eval_flag(str(ix)) is True


def test_tolerance_not_fooled_by_outer_catchall_sentinel(tmp_path):
    """A sentinel/flag that lives OUTSIDE run_lm_eval must not be read as run_lm_eval tolerating the flag (guards the fatal path)."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    # run_lm_eval itself is an unteachable stub (no flag inside), but an earlier function carries the sentinel + a
    # --concurrent-requests case.
    poisoned = (
        "#!/bin/bash\n"
        "other_fn() {\n"
        f"    # {mp._RUN_LM_EVAL_PARSER_SENTINEL}: not the real parser\n"
        '    --concurrent-requests|--concurrent_requests) x="$2" ;;\n'
        "}\n"
        "run_lm_eval() { : ; }\n"
    )
    (bench / "benchmark_lib.sh").write_text(poisoned, encoding="utf-8")

    assert mp._inferencex_tolerates_eval_flag(str(ix)) is False


def test_real_pinned_benchmark_lib_patches_run_lm_eval(tmp_path):
    """Integration against a real benchmark_lib.sh, when one is checked in."""
    fixture = Path(__file__).parent / "fixtures" / "benchmark_lib_a4bb43af.sh"
    if not fixture.is_file():
        pytest.skip(
            "no local benchmark_lib.sh fixture; the pinned file is verified by "
            "test_inferencex_anchor_contract (magpie_patch)"
        )
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is True
    text = lib.read_text(encoding="utf-8")
    region = mp._extract_run_lm_eval_region(text)
    assert region is not None
    body = text[region[0] : region[1]]
    assert "--concurrent-requests|--concurrent_requests" in body
    assert mp._RUN_LM_EVAL_PARSER_SENTINEL in body
    assert text.count("--concurrent-requests|--concurrent_requests") == 1


def test_unpatchable_parser_with_live_flag_stays_fatal(tmp_path):
    """The narrowed judgement must still fail when a live flag really survives an unteachable parser (no false negative)."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "benchmark_lib.sh").write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    # A caller that STILL passes the rejected flag in a shape the strip regex (which expects the $CONC variable)
    # cannot remove: a literal value.
    (bench / "vllm_mi355x.sh").write_text(
        "#!/bin/bash\n"
        'if [[ "$RUN_EVAL" = "true" ]]; then\n'
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n'
        "fi\n",
        encoding="utf-8",
    )

    status = mp.magpie_scripts_patch_status(None, str(ix))
    assert [p.name for p in mp.live_eval_concurrency_flag_scripts(None, str(ix))] == ["vllm_mi355x.sh"]
    assert status.eval_flag_ok is False


# ---- the generic client must be able to name its tokenizer -------------------

_GENERIC_CLIENT = """#!/usr/bin/env bash
if true; then
    run_benchmark_serving \\
        --model "$MODEL" \\
        --result-dir "$WORKSPACE_DIR/" \\
        "${SERVER_MONITOR_ARGS[@]}" \\
        --trust-remote-code || exit $?
fi
"""


def _patch_client(tmp_path, text=_GENERIC_CLIENT):
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    script = tmp_path / "vllm_mi300x.sh"
    script.write_text(text, encoding="utf-8")
    applied = mp._apply_client_tokenizer_mode_patch_atomic(script)
    return applied, script.read_text(encoding="utf-8")


def test_the_generic_client_gains_a_tokenizer_hook(tmp_path):
    """Without it the client dies in HF AutoConfig before its first request."""
    applied, text = _patch_client(tmp_path)
    assert applied
    assert "HYPERLOOM_CLIENT_TOKENIZER_MODE:+--tokenizer-mode" in text
    assert "--trust-remote-code || exit $?" in text


def test_the_hook_is_not_a_comment_inside_the_continuation(tmp_path):
    """After a trailing backslash a ``#`` is an argument, not a comment.

    A comment line spliced into the continuation would be handed to the client
    as argv and break the call.
    """
    _applied, text = _patch_client(tmp_path)
    body = text[text.index("run_benchmark_serving") : text.index("|| exit $?")]
    assert "#" not in body, body


def test_patching_the_client_is_idempotent(tmp_path):
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    _applied, text = _patch_client(tmp_path)
    script = tmp_path / "vllm_mi300x.sh"
    assert mp._apply_client_tokenizer_mode_patch_atomic(script)
    assert script.read_text(encoding="utf-8").count("HYPERLOOM_CLIENT_TOKENIZER_MODE:+--tokenizer-mode") == 1


def test_a_script_with_no_client_shape_is_left_alone(tmp_path):
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    script = tmp_path / "unrelated.sh"
    script.write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    assert mp._apply_client_tokenizer_mode_patch_atomic(script)
    assert script.read_text(encoding="utf-8") == "#!/usr/bin/env bash\necho hi\n"


def test_an_unpatchable_client_is_reported_not_merely_logged(tmp_path):
    """A missing hook must not hide behind the fail-soft eval-concurrency status.

    That status deliberately returns True whenever no live --concurrent-requests
    flag survives. Folding the tokenizer hook into it would let a client that
    still dies in HF AutoConfig report success.
    """
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    # Carries the client shape this patch targets, but not the exact legacy block.
    (scripts / "vllm_mi300x.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --drifted\n',
        encoding="utf-8",
    )
    assert not mp._client_tokenizer_hook_installed(None, scripts.parent)
    assert not mp.ensure_client_tokenizer_hook(None, scripts.parent)


def test_a_patched_tree_reports_installed(tmp_path):
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    (scripts / "vllm_mi300x.sh").write_text(_GENERIC_CLIENT, encoding="utf-8")
    assert mp.ensure_client_tokenizer_hook(None, scripts.parent)
    assert mp._client_tokenizer_hook_installed(None, scripts.parent)


def test_the_production_status_path_installs_the_hook(tmp_path):
    """The install entry point must TRANSFORM an unpatched tree, not just grade it.

    Verification alone would leave a fresh checkout unpatched forever while
    faithfully reporting that it is unpatched.
    """
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    client = scripts / "vllm_mi300x.sh"
    client.write_text(_GENERIC_CLIENT, encoding="utf-8")
    assert "HYPERLOOM_CLIENT_TOKENIZER_MODE" not in client.read_text(encoding="utf-8")

    status = mp.magpie_scripts_patch_status(None, scripts.parent)

    assert status.client_tokenizer_ok
    assert "HYPERLOOM_CLIENT_TOKENIZER_MODE:+--tokenizer-mode" in client.read_text(encoding="utf-8")


def test_the_runtime_entry_point_installs_the_hook(tmp_path):
    """``ensure_eval_concurrency_compat`` is what a run actually calls.

    baseline.py and preflight.py call it; nothing in a run calls
    magpie_scripts_patch_status. A hook installed only on the status path would
    never reach a launch.
    """
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    client = scripts / "vllm_mi300x.sh"
    client.write_text(_GENERIC_CLIENT, encoding="utf-8")

    mp.ensure_eval_concurrency_compat(None, scripts.parent)

    assert "HYPERLOOM_CLIENT_TOKENIZER_MODE:+--tokenizer-mode" in client.read_text(encoding="utf-8")


def test_a_failed_hook_does_not_fail_the_eval_concurrency_result(tmp_path):
    """The two are reported separately, and the runtime checks the hook on its own.

    ``ensure_eval_concurrency_compat`` stays fail-soft about its own patch; it is
    NOT the thing that decides whether a launch may proceed without the hook.
    ``baseline._after_materialize_config`` calls ``ensure_client_tokenizer_hook``
    separately and refuses the round with ``client_tokenizer_unpatchable`` when
    the model is one whose tokenizer has to be named.
    """
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    # Client shape present, legacy block drifted: the hook cannot be installed.
    (scripts / "vllm_mi300x.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --drifted\n',
        encoding="utf-8",
    )
    assert mp.ensure_eval_concurrency_compat(None, scripts.parent)
    assert not mp._client_tokenizer_hook_installed(None, scripts.parent)


def test_a_model_needing_a_named_tokenizer_refuses_an_unpatchable_checkout(tmp_path, monkeypatch):
    """The runtime consequence: no hook, no round -- but only for such a model."""
    import yaml
    from hyperloom.orchestrator.actions.executors import baseline as bl

    scripts = tmp_path / "ix" / "benchmarks"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --drifted\n',
        encoding="utf-8",
    )
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "deepseek_v4"}', encoding="utf-8")
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(
        yaml.safe_dump({"benchmark": {"model": str(model), "inferencex_path": str(tmp_path / "ix")}}),
        encoding="utf-8",
    )

    ex = bl.BaselineExecutor(session_dir=tmp_path) if hasattr(bl, "BaselineExecutor") else None
    if ex is None:
        import pytest as _pytest

        _pytest.skip("BaselineExecutor not exposed under this name")
    monkeypatch.setattr(bl, "materialized_run_eval_disabled", lambda _p: False)
    # The subject is the refusal, not the mode detection that precedes it.
    # ``_client_tokenizer_mode`` answers "" when ``transformers`` cannot be
    # imported -- true of the lint/test images this suite runs on -- and also
    # when the installed transformers happens to know ``deepseek_v4``. Left to
    # the environment, this test asserted a refusal on the machines that had
    # transformers and silently asserted nothing on the ones that did not.
    monkeypatch.setattr(bl, "_client_tokenizer_mode", lambda _model: "deepseek_v4")
    res = ex._after_materialize_config(cfg, tmp_path / "out")
    assert res is not None and res.get("error_class") == "client_tokenizer_unpatchable", res


def test_the_hook_is_required_even_with_evaluation_disabled(tmp_path, monkeypatch):
    """The hook fixes the THROUGHPUT client, which runs whether or not lm-eval does.

    Gating it on eval would leave an eval-disabled DeepSeek-V4 run dying exactly
    as before, with correctness resting on a preflight side effect.
    """
    import yaml
    from hyperloom.orchestrator.actions.executors import baseline as bl

    scripts = tmp_path / "ix" / "benchmarks"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --drifted\n',
        encoding="utf-8",
    )
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "deepseek_v4"}', encoding="utf-8")
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(
        yaml.safe_dump({"benchmark": {"model": str(model), "inferencex_path": str(tmp_path / "ix")}}),
        encoding="utf-8",
    )

    # Evaluation OFF: the eval probe and eval-concurrency checks must not run,
    # but the tokenizer hook still must.
    monkeypatch.setattr(bl, "materialized_run_eval_disabled", lambda _p: True)
    ex = bl.BaselineExecutor(session_dir=tmp_path)
    # Same environment dependence as the refusal test above: "" when
    # ``transformers`` is missing, which is the state of the test images.
    monkeypatch.setattr(bl, "_client_tokenizer_mode", lambda _model: "deepseek_v4")
    res = ex._after_materialize_config(cfg, tmp_path / "out")
    assert res is not None and res.get("error_class") == "client_tokenizer_unpatchable", res


def test_an_unrelated_client_shape_does_not_veto_the_round(tmp_path):
    """The multimodal variants carry the same marker with a different call.

    Judging every sibling would refuse a workload whose own script is patched
    and correct - which is what happened live: vllm_mi300x.sh was patched, and
    vllm_mi300x_mm.sh / vllm_mi355x_mm.sh failed the check and stopped the round.
    """
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    (scripts / "vllm_mi300x.sh").write_text(_GENERIC_CLIENT, encoding="utf-8")
    # Same marker, different client call: unpatchable by design.
    (scripts / "vllm_mi300x_mm.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --multimodal\n',
        encoding="utf-8",
    )

    assert mp.ensure_client_tokenizer_hook(None, scripts.parent, script_name="vllm_mi300x.sh")
    # Unscoped, the sibling still vetoes - that is the behaviour being narrowed.
    assert not mp.ensure_client_tokenizer_hook(None, scripts.parent)


def test_naming_an_unpatchable_script_still_refuses(tmp_path):
    """Narrowing must not become permissive: the named script still has to pass."""
    from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp

    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    (scripts / "vllm_mi300x.sh").write_text(
        '#!/usr/bin/env bash\nrun_benchmark_serving --result-dir "$WORKSPACE_DIR/" --drifted\n',
        encoding="utf-8",
    )
    assert not mp.ensure_client_tokenizer_hook(None, scripts.parent, script_name="vllm_mi300x.sh")


def test_baseline_names_the_script_from_the_config(tmp_path):
    """framework + runner_type is how Magpie picks it; an override wins."""
    import yaml
    from hyperloom.orchestrator.actions.executors import baseline as bl

    cfg = tmp_path / "b.yaml"
    cfg.write_text(yaml.safe_dump({"benchmark": {"framework": "vllm", "runner_type": "mi300x"}}), encoding="utf-8")
    assert bl.BaselineExecutor._client_script_from_config(cfg) == "vllm_mi300x.sh"

    cfg.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "runner_type": "mi300x", "benchmark_script": "custom.sh"}}),
        encoding="utf-8",
    )
    assert bl.BaselineExecutor._client_script_from_config(cfg) == "custom.sh"

    cfg.write_text(yaml.safe_dump({"benchmark": {}}), encoding="utf-8")
    assert bl.BaselineExecutor._client_script_from_config(cfg) is None


def test_an_unfittable_sibling_script_does_not_fail_the_install():
    """The hook is workload-specific; the install contract is not.

    A Magpie layout carries sibling scripts -- the multimodal ``*_mm.sh`` among
    them -- whose client shape this hook does not fit and was never meant to.
    Folding ``client_tokenizer_ok`` into ``ok`` failed installation over a
    script the run would never execute, on a layout the run would never touch.
    The status still reports it; the hard failure lives where the config names
    both the model that needs the hook and the one script that will run it.
    """
    from hyperloom.orchestrator.actions.executors._magpie_patcher import MagpiePatchStatus

    status = MagpiePatchStatus(
        atomic_ok=True,
        remote_trust_ok=True,
        eval_flag_ok=True,
        client_tokenizer_ok=False,
    )

    assert status.ok is True, "an unfittable sibling must not fail the install"
    assert status.client_tokenizer_ok is False, "it must still be reported"
