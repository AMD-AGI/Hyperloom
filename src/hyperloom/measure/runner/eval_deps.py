# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy evaluator, installed before the test's server boots.

InferenceX's ``run_lm_eval`` installs lm-eval (``pip`` from the image's indexes, then the pinned harness from GitHub)
as the first thing an accuracy round does once its benchmark is over, with the server up and idle. Magpie keeps the
round's output until the round ends, so nothing of that install reaches the logs the silence watchdog reads: in a fresh
sandbox of the vLLM 0.16 image it ran past the 600 s window, and the watchdog reaped the idle server as a hung engine
before the evaluation sent a request (hyperloom-multibranch #156 c4287). Here it runs once, before the server boots,
with a limit and a log of its own, and the round's own call finds it done through the install-once marker
(``_magpie_patcher.ensure_lm_eval_install_once``).

It runs in the environment the round starts its benchmark script with, not the runner's own: the round's ``python3``
is the one ``PATH`` names there, and the marker is kept by that interpreter's ``sys.prefix``, so an install for another
interpreter would leave the round to install again in its silent window. What decides that it worked is the round's own
check, run in that environment: ``python3`` imports lm-eval and the round's ``_install_lm_eval_deps`` would install
nothing (hyperloom-multibranch #156 c4469).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from hyperloom.orchestrator.actions.executors import EVAL_OUTPUT_ENV, LM_EVAL_READY_FN, ensure_lm_eval_install_once

__all__ = ["EVAL_DEPS_LOG", "EVAL_DEPS_TIMEOUT_SEC", "EvalDepsError", "install_eval_deps"]

#: Where the install's own output goes, below the test's output directory, so the bundle carries it.
EVAL_DEPS_LOG = "eval_deps.log"
#: How long the install may take; the tool checkouts are given as long (``tools._FETCH_TIMEOUT_SEC``).
EVAL_DEPS_TIMEOUT_SEC = 900.0
_LIB = Path("benchmarks") / "benchmark_lib.sh"
#: The round sources the same library and calls the same function, so this is the install the round would run.
_INSTALL = 'source "$1" && _install_lm_eval_deps'
_READY = f'source "$1" && {LM_EVAL_READY_FN}'
_IMPORT = ("python3", "-c", "import lm_eval")
_TAIL_CHARS = 1500


class EvalDepsError(Exception):
    """lm-eval could not be installed for the accuracy evaluation; the message ends with what the install printed."""


def install_eval_deps(
    inferencex: Path, *, environ: Mapping[str, str], log_path: Path, timeout_sec: float = EVAL_DEPS_TIMEOUT_SEC
) -> str | None:
    """Install lm-eval the way the accuracy round would, before it, and make the round's own install a no-op.

    ``environ`` is the environment the round runs its benchmark script with (its ``python3`` is the one installed for).
    Returns a note when it did not run (the checkout cannot be made to install once, so the round installs anyway).

    Raises:
        EvalDepsError: the install outlived ``timeout_sec``, lm-eval does not import after it, or the round's install
            would not find it installed.
    """
    if not ensure_lm_eval_install_once(inferencex):
        return (
            f"lm-eval was not installed before the server booted: {inferencex / _LIB} cannot be made to install it "
            "once, so the accuracy round installs it itself"
        )
    # The install's output is kept in its own log already; the round's eval keeps its output where the round says.
    env = {**environ, EVAL_OUTPUT_ENV: str(log_path)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            ["bash", "-c", _INSTALL, "bash", str(inferencex / _LIB)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=inferencex,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _end(process)
            raise EvalDepsError(
                f"installing lm-eval for the accuracy evaluation did not end within {timeout_sec:.0f} s; "
                f"its output: {_tail(log_path)}"
            ) from None
    lib = str(inferencex / _LIB)
    ready = subprocess.run(
        ["bash", "-c", _READY, "bash", lib], env=env, cwd=inferencex, capture_output=True, timeout=300
    )
    if ready.returncode == 0:
        return None
    probe = subprocess.run(_IMPORT, env=env, cwd=inferencex, capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise EvalDepsError(
            f"lm-eval does not import after its install (install exit {returncode}): "
            f"{probe.stderr.strip()[-_TAIL_CHARS:]}; the install's output: {_tail(log_path)}"
        )
    python3 = shutil.which("python3", path=env.get("PATH")) or "python3"
    raise EvalDepsError(
        f"lm-eval imports in {python3}, but the install left no marker the round's install finds, so the round would "
        f"install it again on an idle server; the install's output: {_tail(log_path)}"
    )


def _end(process: subprocess.Popen) -> None:
    """Kill the install's whole process group: pip and git run below the shell that started them."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait()


def _tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(unreadable)"
    return text.strip()[-_TAIL_CHARS:] or "(nothing)"
