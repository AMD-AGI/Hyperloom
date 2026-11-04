# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX's lm-eval install, patched by ``_magpie_patcher``, runs once per Python environment, under a lock; the
eval command's own output is kept in the round's directory as it is written."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _inferencex_patcher as inferencex_patcher
from hyperloom.orchestrator.actions.executors import _magpie_patcher as patcher
from hyperloom.orchestrator.actions.executors.benchmark_config import prepare_benchmark_runtime

REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
LIB = f"""#!/usr/bin/env bash
before() {{ :; }}

_install_lm_eval_deps() {{
    local lm_eval_ref="{REF}"
    echo install >> "$INSTALLS"
}}

after() {{ :; }}
"""


RUN_LM_EVAL = """
run_lm_eval() {
    _install_lm_eval_deps
    echo "Eval budget: 4096"
    set -x
    python3 -m lm_eval --tasks gsm8k \\
      --output_path "$RESULT_DIR/eval"
    local eval_exit=$?
    set +x
    return $eval_exit
}
"""
#: The install-once wrapper as an earlier revision laid it: from the install on, the whole shell's output was teed.
SHELL_WIDE_CAPTURE = """_hyperloom_keep_eval_output() {
    [[ -n "${RESULT_DIR:-}" && -d "${RESULT_DIR:-}" ]] || return 0
    exec > >(tee -a "$RESULT_DIR/hyperloom_eval.log") 2> >(tee -a "$RESULT_DIR/hyperloom_eval.log" >&2)
}

"""


def _lib(tmp_path: Path, text: str = LIB) -> Path:
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text(text)
    return lib


def test_the_install_is_wrapped_once_and_keeps_the_rest_of_the_library(tmp_path: Path) -> None:
    lib = _lib(tmp_path)

    assert patcher._apply_lm_eval_install_once_patch_atomic(lib)
    patched = lib.read_text()
    assert patcher._apply_lm_eval_install_once_patch_atomic(lib)

    assert lib.read_text() == patched
    assert patched.count("_install_lm_eval_deps() {") == 1
    assert f"{patcher._LM_EVAL_UNLOCKED_FN}() {{" in patched
    assert f".hyperloom-lm-eval-{REF}" in patched
    assert "before() {" in patched and "after() {" in patched


def test_the_benchmark_preparation_patches_the_runner_s_own_inferencex_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "InferenceX"
    (checkout / "benchmarks").mkdir(parents=True)
    _lib(checkout / "benchmarks")
    config = tmp_path / "point.yaml"
    config.write_text("benchmark:\n  framework: sglang\n  envs:\n    RUN_EVAL: 'false'\n")
    monkeypatch.setenv("INFERENCEX_PATH", str(checkout))
    monkeypatch.setattr(inferencex_patcher, "ensure_eval_probe_patched", lambda root: True)

    assert prepare_benchmark_runtime(config, {}) is None

    assert patcher._LM_EVAL_ONCE_SENTINEL in (checkout / "benchmarks" / "benchmark_lib.sh").read_text()


def test_the_eval_concurrency_sweep_lays_the_patch_whatever_it_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight and baseline reach the patch through this sweep; a library it cannot fully fix still gets it."""
    checkout = tmp_path / "InferenceX"
    (checkout / "benchmarks").mkdir(parents=True)
    lib = _lib(checkout / "benchmarks")
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    patcher.ensure_eval_concurrency_compat(inferencex_dir=checkout)

    assert patcher._LM_EVAL_ONCE_SENTINEL in lib.read_text()


def test_a_library_without_the_install_is_left_alone_and_said_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text("before() { :; }\n")

    assert not patcher._apply_lm_eval_install_once_patch_atomic(lib)

    assert lib.read_text() == "before() { :; }\n"
    assert "_install_lm_eval_deps not found" in caplog.text


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_two_tests_on_one_environment_install_lm_eval_once(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python3 = bin_dir / "python3"
    python3.write_text(f'#!/bin/sh\nif [ "$2" = "import sys; print(sys.prefix)" ]; then echo {prefix}; fi\nexit 0\n')
    python3.chmod(0o755)
    installs = tmp_path / "installs"
    script = f'source "{lib}"; _install_lm_eval_deps; _install_lm_eval_deps'

    subprocess.run(
        ["bash", "-c", script],
        check=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "INSTALLS": str(installs)},
    )

    assert installs.read_text() == "install\n"
    assert (prefix / f".hyperloom-lm-eval-{REF}").is_file()


def _python3(tmp_path: Path, prefix: Path) -> Path:
    """A ``python3`` with ``prefix`` for ``sys.prefix`` that imports lm_eval once ``$INSTALLS`` says it was installed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python3 = bin_dir / "python3"
    python3.write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = "import sys; print(sys.prefix)" ]; then echo {prefix}; exit 0; fi\n'
        'if [ "$2" = "import lm_eval" ]; then [ -s "$INSTALLS" ]; exit $?; fi\n'
        'if [ "$2" = "lm_eval" ]; then\n'
        '    echo "lm_eval: 42 requests done"; echo "loading gsm8k from the Hub" >&2\n'
        '    if [ -n "${EVAL_HANG:-}" ]; then sleep 60; fi\n'
        "    exit 3\n"
        "fi\n"
        "exit 0\n"
    )
    python3.chmod(0o755)
    return bin_dir


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_an_environment_whose_prefix_cannot_be_written_installs_once_too(tmp_path: Path) -> None:
    """The image's interpreter may sit where the sandbox's account cannot write, while pip still installs (into a
    writable site-packages or the user site). The marker then goes below /tmp, and the round does not install again."""
    lib = _lib(tmp_path)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    (tmp_path / "file").write_text("")
    prefix = tmp_path / "file" / "prefix"
    bin_dir = _python3(tmp_path, prefix)
    installs = tmp_path / "installs"
    environ = {"PATH": f"{bin_dir}:/usr/bin:/bin", "INSTALLS": str(installs)}
    spare = subprocess.run(
        ["bash", "-c", f'source "{lib}" && {patcher._LM_EVAL_SPARE_MARKER_FN} "$1"', "bash", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
    ).stdout.strip()
    try:
        script = f'source "{lib}"; _install_lm_eval_deps; _install_lm_eval_deps; {patcher.LM_EVAL_READY_FN}'
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=environ)

        assert done.returncode == 0, done.stderr
        assert installs.read_text() == "install\n"
        assert Path(spare).is_file() and spare.startswith(f"/tmp/hyperloom-lm-eval-{REF}-")
        assert "is installed for" in done.stderr, "the skip says so"
    finally:
        Path(spare).unlink(missing_ok=True)


def _eval_round(tmp_path: Path, text: str = LIB + RUN_LM_EVAL) -> tuple[Path, dict[str, str]]:
    lib = _lib(tmp_path, text)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    environ = {
        "PATH": f"{_python3(tmp_path, prefix)}:/usr/bin:/bin",
        "INSTALLS": str(tmp_path / "installs"),
        "RESULT_DIR": str(tmp_path),
    }
    return lib, environ


@pytest.mark.parametrize(
    "lay",
    [
        lambda checkout: patcher.ensure_lm_eval_install_once(checkout),
        lambda checkout: patcher.ensure_eval_concurrency_compat(inferencex_dir=checkout),
    ],
    ids=["install_once", "eval_concurrency_sweep"],
)
def test_each_entry_point_lays_the_eval_output_patch_with_the_install_once_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lay
) -> None:
    checkout = tmp_path / "InferenceX"
    (checkout / "benchmarks").mkdir(parents=True)
    lib = _lib(checkout / "benchmarks", LIB + RUN_LM_EVAL)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    lay(checkout)

    text = lib.read_text()
    assert patcher._LM_EVAL_ONCE_SENTINEL in text and patcher._EVAL_OUTPUT_SENTINEL in text


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_only_what_the_eval_command_writes_is_kept_in_the_round_s_directory(tmp_path: Path) -> None:
    """Magpie holds the round's output until the round ends; the eval's own output is also kept as it is written, and
    what the shell writes before or after it is not."""
    lib, environ = _eval_round(tmp_path)
    assert patcher._apply_eval_output_patch_atomic(lib)
    patched = lib.read_text()
    assert patcher._apply_eval_output_patch_atomic(lib)
    assert lib.read_text() == patched
    assert patched.count(f"{patcher._EVAL_OUTPUT_FN} python3 -m lm_eval --tasks gsm8k") == 1
    assert patcher._apply_lm_eval_install_once_patch_atomic(lib)
    script = f'source "{lib}"; run_lm_eval; echo "eval exit $?"; echo "benchmarking again"; echo "benchmarking" >&2'

    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=environ, timeout=60)

    kept = (tmp_path / patcher.EVAL_OUTPUT_LOG).read_text()
    assert "42 requests done" in kept and "loading gsm8k from the Hub" in kept
    assert "benchmarking" not in kept and "Eval budget" not in kept and "installing lm-eval" not in kept
    assert "eval exit 3" in done.stdout, "the eval's status reaches the caller of run_lm_eval"
    assert "42 requests done" in done.stdout and "loading gsm8k from the Hub" in done.stderr, "the round still sees it"


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_what_the_eval_printed_is_kept_when_the_round_is_killed_during_it(tmp_path: Path) -> None:
    lib, environ = _eval_round(tmp_path)
    patcher._apply_eval_output_patch_atomic(lib)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    log = tmp_path / patcher.EVAL_OUTPUT_LOG
    process = subprocess.Popen(
        ["bash", "-c", f'source "{lib}"; run_lm_eval'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**environ, "EVAL_HANG": "1"},
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (log.is_file() and "loading gsm8k" in log.read_text()):
            time.sleep(0.05)
    finally:
        os.killpg(process.pid, signal.SIGKILL)  # as the watchdog ends a round: the whole group, mid-eval
        process.wait()

    assert "loading gsm8k from the Hub" in log.read_text()


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_a_library_patched_with_the_shell_wide_capture_keeps_nothing_after_the_install(tmp_path: Path) -> None:
    """A checkout patched by the earlier revision calls the capture bare, first thing in the install; that call now does
    nothing, so the benchmark shell's later output stays out of the eval's log."""
    lib, environ = _eval_round(tmp_path)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    text = lib.read_text().replace(
        "    local prefix marker lock\n", "    _hyperloom_keep_eval_output\n    local prefix marker lock\n"
    )
    lib.write_text(text.replace("_install_lm_eval_deps() {", SHELL_WIDE_CAPTURE + "_install_lm_eval_deps() {", 1))
    log = tmp_path / patcher.EVAL_OUTPUT_LOG
    script = f'source "{lib}"; _install_lm_eval_deps; echo "benchmarking again"'
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, env=environ, timeout=60)
    assert "benchmarking again" in log.read_text(), "the fixture is the shell-wide capture"
    log.unlink()

    assert patcher._apply_eval_output_patch_atomic(lib)
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, env=environ, timeout=60)

    assert not log.exists(), "the bare call keeps nothing, not even an empty log"


def test_a_library_without_the_eval_command_is_left_alone_and_said_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lib = _lib(tmp_path)

    assert not patcher._apply_eval_output_patch_atomic(lib)

    assert lib.read_text() == LIB
    assert "no lm_eval command in run_lm_eval" in caplog.text
