# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the idempotent, atomic Magpie ``benchmarker.py`` patcher
(path resolution, sentinel/legacy detection, upstream-atomic awareness, and
the classified atomic-reason outcomes)."""

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


_REMOTE_COMPAT_MODEL_ARGS = (
    '  local model_args="model=${MODEL},base_url=${base_url},'
    'num_concurrent=${conc},tokenizer_backend=huggingface,trust_remote_code=true"\n'
)


def test_lm_eval_tokenized_patch_adds_env_hook(tmp_path):
    # The remote-compat model_args line gains the $MAGPIE_EVAL_TOKENIZED_REQUESTS hook.
    d = tmp_path / "benchmark"
    d.mkdir()
    script = d / "magpie_bench_remote_compat.sh"
    script.write_text("#!/bin/bash\n" + _REMOTE_COMPAT_MODEL_ARGS, encoding="utf-8")

    assert mp._apply_lm_eval_tokenized_requests_patch_atomic(d) is True
    patched = script.read_text(encoding="utf-8")
    assert "${MAGPIE_EVAL_TOKENIZED_REQUESTS:+,tokenized_requests=${MAGPIE_EVAL_TOKENIZED_REQUESTS}}" in patched
    # tokenizer_backend line stays otherwise intact.
    assert "tokenizer_backend=huggingface,trust_remote_code=true" in patched

    # Idempotent: a second pass is a no-op (marker already present).
    before = patched
    assert mp._apply_lm_eval_tokenized_requests_patch_atomic(d) is True
    assert script.read_text(encoding="utf-8") == before


def test_lm_eval_tokenized_patch_noop_for_unrelated_script(tmp_path):
    # A script without the exact model_args line is left byte-for-byte unchanged.
    d = tmp_path / "benchmark"
    d.mkdir()
    other = d / "sglang_mi300x.sh"
    body = "#!/bin/bash\necho hello\nrun_eval --framework lm-eval\n"
    other.write_text(body, encoding="utf-8")

    assert mp._apply_lm_eval_tokenized_requests_patch_atomic(d) is True
    assert other.read_text(encoding="utf-8") == body


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


def test_compat_false_when_an_unstrippable_flag_meets_a_strict_parser(tmp_path):
    """The one genuinely fatal state: a caller still passes the flag AND
    run_lm_eval still rejects it.

    This is exactly the shape that killed a run at baseline_accuracy_failed, so
    it must report False and let the caller escalate rather than proceed into a
    doomed eval.
    """
    magpie = tmp_path / "site-packages"
    bench = magpie / "Magpie" / "scripts" / "benchmark"
    bench.mkdir(parents=True)
    # A shape the strip cannot rewrite, so the flag survives the patch.
    (bench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n',
        encoding="utf-8",
    )
    # A benchmark_lib.sh whose parser cannot be taught the flag either: no
    # run_lm_eval definition at all, so the belt has nothing to patch.
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(
        "# no run_lm_eval here\n", encoding="utf-8"
    )

    assert mp.ensure_eval_concurrency_compat(str(magpie), str(ix)) is False
    # The blocker is still reported by the scanner, so callers can name the file.
    assert [p.name for p in mp.live_eval_concurrency_flag_scripts(str(magpie), None)] == [
        "sglang_mi355x.sh"
    ]


# ---- unreadable files: the patcher must degrade, never crash a run ---------
def _unreadable(path):
    """A path that exists but raises OSError on read.

    Uses a directory rather than chmod: these suites run as root, where mode
    bits do not deny access, so a permission-based fixture would silently not
    exercise the error branch at all.
    """
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
    """No legacy parser block to rewrite -> nothing patched, report False.

    This is the shape that must NOT be mistaken for success: silently returning
    True here would let a run proceed into an eval the parser still rejects.
    """
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text("run_lm_eval() { : ; }\n", encoding="utf-8")

    assert mp._apply_run_lm_eval_arg_patch_atomic(lib) is False


# ---- merged-case parser (InferenceX a4bb43af+) ----------------------------
# The pinned InferenceX (a4bb43af) refactored run_lm_eval's arg parser into a
# single merged ``--port|--task|...|--top-p)`` case with an inner dispatch and a
# ``>&2`` / ``return 2`` catch-all. It already reads concurrency from
# EVAL_CONCURRENT_REQUESTS/CONC and no caller passes --concurrent-requests, so
# accuracy eval is NOT blocked. The old per-flag legacy block no longer matches,
# which used to make eval_flag_ok=False and (post 3166da7f) fail install with a
# false positive.
_BENCHMARK_LIB_MERGED_CASE = (
    "#!/bin/bash\n"
    "run_lm_eval() {\n"
    '    local port="${PORT:-8888}"\n'
    '    local top_p=1\n'
    '    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"\n'
    "    while [[ $# -gt 0 ]]; do\n"
    "        case \"$1\" in\n"
    "            --port|--task|--results-dir|--gen-max-tokens|--temperature|--top-p)\n"
    "                case \"$1\" in\n"
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
    """Full status: merged-case parser + env concurrency + no live flag => ok.

    Reproduces the shuoshuo-dev install failure: the defence-in-depth parser
    patch could not match the refactored parser, but nothing passes the flag, so
    the install must NOT be failed (status.ok stays True).
    """
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
    # atomic is a benign no-op here (no MAGPIE_PATH / benchmarker.py), not a
    # genuine failure; install.sh treats reason=missing as fail-soft.
    assert status.atomic_reason == mp._ATOMIC_REASON_MISSING
    assert status.atomic_genuine_failure is False
    assert mp.live_eval_concurrency_flag_scripts(None, str(ix)) == []


def test_unpatchable_parser_without_live_flag_is_not_fatal(tmp_path):
    """Narrowed judgement: even a parser we cannot teach must not fail install
    when no caller passes the flag (aligns install-time with run-time)."""
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


# A benchmark_lib.sh with EARLIER functions that carry an identical ``*)``
# catch-all (real a4bb43af has several before run_lm_eval, e.g. at lines 285 &
# 451). The merged-case patch must skip these and only touch run_lm_eval's.
_BENCHMARK_LIB_MULTI_CATCHALL = (
    "#!/bin/bash\n"
    "wait_for_server_ready() {\n"
    "    while [[ $# -gt 0 ]]; do\n"
    "        case \"$1\" in\n"
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
    "    case \"$1\" in\n"
    "        *)\n"
    '            echo "Unknown parameter: $1" >&2\n'
    "            return 1\n"
    "            ;;\n"
    "    esac\n"
    "}\n"
    "\n"
    + _BENCHMARK_LIB_MERGED_CASE
)


def test_merged_case_patch_lands_inside_run_lm_eval_only(tmp_path):
    """Regression for the mis-patch bug: with earlier functions sharing the same
    ``*)`` catch-all, the flag case must be spliced into run_lm_eval, not the
    first matching catch-all in the file."""
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
    """A sentinel/flag that lives OUTSIDE run_lm_eval must not be read as
    run_lm_eval tolerating the flag (guards the fatal path)."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    # run_lm_eval itself is an unteachable stub (no flag inside), but an earlier
    # function carries the sentinel + a --concurrent-requests case.
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
    """Integration: the real a4bb43af benchmark_lib.sh (if present) must get its
    run_lm_eval taught the flag, with the sentinel landing inside that function.

    Skips silently when the fixture is not checked in, so the suite stays
    hermetic;     the logic is already covered by the multi-catch-all stub above."""
    fixture = Path(__file__).parent / "fixtures" / "benchmark_lib_a4bb43af.sh"
    if not fixture.is_file():
        pytest.skip("real pinned benchmark_lib.sh fixture not present")
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
    """The narrowed judgement must still fail when a live flag really survives
    an unteachable parser (no false negative)."""
    ix = tmp_path / "ix"
    bench = ix / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "benchmark_lib.sh").write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    # A caller that STILL passes the rejected flag in a shape the strip regex
    # (which expects the $CONC variable) cannot remove: a literal value. The
    # live-flag scan still recognises it, so it is a genuine, unstrippable blocker.
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
