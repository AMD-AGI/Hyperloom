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

A probe is a case with another entry script: ``served-probe.sh`` runs against the server the test started,
``probe.sh`` with no server, on the test's cards alone. It is run with the environment its caller gives it and has no
verdict: its exit code is data, the last line of the log tail, which is its whole result (:func:`run_probe`).
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from hyperloom.contracts.measurement import NOT_REPRODUCED, REPRO_VERDICTS, REPRODUCED

__all__ = [
    "CASE_SCRIPT",
    "LOG_FILE",
    "PROBE_SCRIPT",
    "REPRO_DIR",
    "SERVED_PROBE_SCRIPT",
    "VERDICT_FILE",
    "Case",
    "ReproError",
    "find_case",
    "read_verdict",
    "run_case",
    "run_probe",
]

#: The directory, below a test's output directory, that holds the verdict and the case's log tail.
REPRO_DIR = "repro"
VERDICT_FILE = "verdict.json"
LOG_FILE = "case.log"
#: The entry point of a case: a script at the top of one attachment directory.
CASE_SCRIPT = "repro.sh"
PROBE_SCRIPT = "probe.sh"
SERVED_PROBE_SCRIPT = "served-probe.sh"
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


@dataclass(frozen=True)
class Case:
    script: Path
    is_probe: bool
    needs_server: bool


_ENTRY_SCRIPTS = {CASE_SCRIPT: (False, True), SERVED_PROBE_SCRIPT: (True, True), PROBE_SCRIPT: (True, False)}


def find_case(attach_dir: Path, attachment_ids: Sequence[str]) -> Case:
    """The one entry script the attachment directories hold at their top, and what it says of the case."""
    found = [
        Case(script, *traits)
        for attachment_id in attachment_ids
        for name, traits in _ENTRY_SCRIPTS.items()
        if (script := Path(attach_dir) / attachment_id / name).is_file()
    ]
    names = ", ".join(_ENTRY_SCRIPTS)
    if not found:
        raise ReproError(f"no attachment holds an entry script ({names}) at its top")
    if len(found) > 1:
        raise ReproError(f"the attachments hold more than one entry script ({names})")
    return found[0]


def _limit_file_size() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))


def _server_env(base_url: str) -> dict[str, str]:
    return {"HL_BASE_URL": base_url, "no_proxy": _LOCAL_HOSTS, "NO_PROXY": _LOCAL_HOSTS}


def _case_env(base_url: str, model: str) -> dict[str, str]:
    env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
    return {**env, "HL_MODEL": model, **_server_env(base_url)}


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


def _run(script: Path, env: Mapping[str, str], timeout_sec: float) -> tuple[int | None, bytes]:
    """Run ``script`` with ``bash`` (an attachment does not carry file modes); its exit code, ``None`` for one killed
    at ``timeout_sec``, and the tail of its output."""
    with tempfile.TemporaryFile() as sink:
        try:
            process = subprocess.Popen(
                ["bash", script.name],
                cwd=script.parent,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_limit_file_size,
            )
        except OSError as exc:
            raise ReproError(f"{script.name} could not be started: {type(exc).__name__}") from exc
        return _wait(process, timeout_sec), _tail(sink)


def run_probe(
    script: Path, *, env: Mapping[str, str], base_url: str | None, timeout_sec: float, output_dir: Path
) -> None:
    """Run the probe ``script`` with ``env``, against ``base_url`` when it has a server; write its log tail below
    ``output_dir``, ending with the line that says how it ended.

    Raises:
        ReproError: The script could not be started.
    """
    code, log = _run(script, {**env, **(_server_env(base_url) if base_url else {})}, timeout_sec)
    ending = f"killed: not finished within {timeout_sec:.0f} s" if code is None else f"exit code {code}"
    _write_log(output_dir, log + f"\n[{ending}]\n".encode())


def run_case(script: Path, *, base_url: str, model: str, timeout_sec: float, output_dir: Path) -> str:
    """Run ``script`` against ``base_url``; return the verdict, which is also written below ``output_dir``.

    The log tail is written whatever the end, so a case that errors out can be diagnosed from the bundle.

    Raises:
        ReproError: The script could not be started, timed out or exited with a code that is not a verdict.
    """
    code, log = _run(script, _case_env(base_url, model), timeout_sec)
    _write_log(output_dir, log)
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
