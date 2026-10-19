# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX's lm-eval install, patched by ``_magpie_patcher``, runs once per Python environment, under a lock."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _magpie_patcher as patcher

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
