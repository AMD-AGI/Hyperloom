# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX's lm-eval install, patched by ``_magpie_patcher``, runs once per Python environment, under a lock."""

from __future__ import annotations

import shutil
import subprocess
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


def _lib(tmp_path: Path) -> Path:
    lib = tmp_path / "benchmark_lib.sh"
    lib.write_text(LIB)
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


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_what_the_eval_prints_is_kept_in_the_round_s_directory_as_it_is_written(tmp_path: Path) -> None:
    """Magpie holds the round's output until the round ends; a round the watchdog ends kept nothing of its eval."""
    lib = _lib(tmp_path)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    environ = {
        "PATH": f"{_python3(tmp_path, prefix)}:/usr/bin:/bin",
        "INSTALLS": str(tmp_path / "installs"),
        "RESULT_DIR": str(tmp_path),
    }
    log = tmp_path / patcher.EVAL_OUTPUT_LOG
    script = (
        f'source "{lib}"; _install_lm_eval_deps; echo "Eval budget: 4096"; echo "loading gsm8k from the Hub" >&2; '
        f'while ! grep -q "loading gsm8k" "{log}" 2>/dev/null; do sleep 0.1; done; kill -9 $$'
    )

    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=environ, timeout=60)

    assert done.returncode == -9, "the shell was killed, as the watchdog ends a round"
    kept = log.read_text()
    assert "installing lm-eval" in kept and "Eval budget: 4096" in kept and "loading gsm8k from the Hub" in kept
    assert "loading gsm8k from the Hub" in done.stderr and "Eval budget: 4096" in done.stdout, "the round still sees it"


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")
def test_an_install_whose_output_is_kept_elsewhere_keeps_none_in_the_round_s_directory(tmp_path: Path) -> None:
    lib = _lib(tmp_path)
    patcher._apply_lm_eval_install_once_patch_atomic(lib)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    environ = {
        "PATH": f"{_python3(tmp_path, prefix)}:/usr/bin:/bin",
        "INSTALLS": str(tmp_path / "installs"),
        "RESULT_DIR": str(tmp_path),
        patcher.EVAL_OUTPUT_ENV: str(tmp_path / "eval_deps.log"),
    }

    subprocess.run(["bash", "-c", f'source "{lib}"; _install_lm_eval_deps'], check=True, env=environ)

    assert not (tmp_path / patcher.EVAL_OUTPUT_LOG).exists()
