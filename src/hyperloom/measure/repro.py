# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the case of a ``repro`` test against a live server and keep its verdict where the signed bundle carries it.

A repro case is an attachment directory with a ``repro.sh`` at its top. The runner unpacks every attachment under
``attach/<id>/``; the case is run with that directory as its working directory and two variables, ``HL_BASE_URL`` (the
server the test started) and ``HL_MODEL`` (the model it serves). Its exit code is the whole contract:

* ``0``: the failure does not occur (``not_reproduced``);
* ``1``: the failure occurs (``reproduced``);
* anything else, a timeout or a missing ``repro.sh``: the case did not run to a verdict. That is a
  :class:`ReproError`, never a verdict, so a case that could not run is not read as a fix.

The verdict and the tail of the case's output are written below ``<output_dir>/repro``, so the result manifest lists
them and the signature covers them; :func:`read_verdict` reads the verdict back from a landed bundle.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from hyperloom.contracts.measurement import NOT_REPRODUCED, REPRO_VERDICTS, REPRODUCED

__all__ = [
    "CASE_SCRIPT",
    "LOG_FILE",
    "REPRO_DIR",
    "VERDICT_FILE",
    "ReproError",
    "find_case",
    "read_verdict",
    "run_case",
]

#: The directory, below a test's output directory, that holds the verdict and the case's log tail.
REPRO_DIR = "repro"
VERDICT_FILE = "verdict.json"
LOG_FILE = "case.log"
#: The entry point of a case: a script at the top of one attachment directory.
CASE_SCRIPT = "repro.sh"
_EXIT_VERDICTS = {0: NOT_REPRODUCED, 1: REPRODUCED}
_KIB = 1024
_MIB = _KIB * _KIB
_LOG_TAIL_BYTES = 64 * _KIB
#: The largest file the case may write (its output sink included): a case that floods the disk is killed, not a verdict.
_MAX_FILE_BYTES = 256 * _MIB
#: What the case sees of this process: enough to find tools, nothing that could carry a credential.
_INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
_LOCAL_HOSTS = "127.0.0.1,localhost"


class ReproError(Exception):
    """The case did not run to a verdict; the message is the reason the result carries."""


def find_case(attach_dir: Path, attachment_ids: Sequence[str]) -> Path:
    """The ``repro.sh`` of the one attachment directory that holds it."""
    found = [
        script
        for attachment_id in attachment_ids
        if (script := Path(attach_dir) / attachment_id / CASE_SCRIPT).is_file()
    ]
    if not found:
        raise ReproError(f"no attachment holds {CASE_SCRIPT} at its top")
    if len(found) > 1:
        raise ReproError(f"more than one attachment holds {CASE_SCRIPT}")
    return found[0]


def _limit_file_size() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))


def _case_env(base_url: str, model: str) -> dict[str, str]:
    env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
    env.update({"HL_BASE_URL": base_url, "HL_MODEL": model, "no_proxy": _LOCAL_HOSTS, "NO_PROXY": _LOCAL_HOSTS})
    return env


def _tail(handle: IO[bytes]) -> bytes:
    handle.seek(0, os.SEEK_END)
    handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
    return handle.read()


def _wait(process: subprocess.Popen[bytes], timeout_sec: float) -> int | None:
    """The exit code, or ``None`` when the case outlived ``timeout_sec`` and its process group was killed."""
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)  # the case's own group: the child was started as its leader
        process.wait()
        return None


def _write_log(output_dir: Path, log: bytes) -> None:
    directory = Path(output_dir) / REPRO_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / LOG_FILE).write_bytes(log)


def run_case(script: Path, *, base_url: str, model: str, timeout_sec: float, output_dir: Path) -> str:
    """Run ``script`` against ``base_url``; return the verdict, which is also written below ``output_dir``.

    The log tail is written whatever the end, so a case that errors out can be diagnosed from the bundle.
    ``repro.sh`` is run with ``bash``: an attachment does not carry file modes.

    Raises:
        ReproError: The script could not be started, timed out or exited with a code that is not a verdict.
    """
    with tempfile.TemporaryFile() as sink:
        try:
            process = subprocess.Popen(
                ["bash", script.name],
                cwd=script.parent,
                env=_case_env(base_url, model),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_limit_file_size,
            )
        except OSError as exc:
            raise ReproError(f"{CASE_SCRIPT} could not be started: {type(exc).__name__}") from exc
        code = _wait(process, timeout_sec)
        _write_log(output_dir, _tail(sink))
    if code is None:
        raise ReproError(f"{CASE_SCRIPT} did not finish within {timeout_sec:.0f} s")
    verdict = _EXIT_VERDICTS.get(code)
    if verdict is None:
        raise ReproError(f"{CASE_SCRIPT} exited with {code}: neither 0 (not reproduced) nor 1 (reproduced)")
    (Path(output_dir) / REPRO_DIR / VERDICT_FILE).write_text(
        json.dumps({"verdict": verdict, "exit_code": code}), encoding="utf-8"
    )
    return verdict


def read_verdict(directory: Path) -> str:
    """The verdict a landed bundle holds below ``directory`` (the test's output directory).

    Raises:
        ReproError: The bundle holds no verdict, an unreadable one, or one that is not a verdict.
    """
    path = Path(directory) / REPRO_DIR / VERDICT_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReproError(f"the bundle holds no readable {REPRO_DIR}/{VERDICT_FILE}") from exc
    verdict = document.get("verdict") if isinstance(document, dict) else None
    if verdict not in REPRO_VERDICTS:
        raise ReproError(f"{REPRO_DIR}/{VERDICT_FILE} holds {verdict!r}, not one of {REPRO_VERDICTS}")
    return str(verdict)
