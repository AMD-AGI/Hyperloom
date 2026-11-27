# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom hosted run``: the hosted entry, which hands a campaign to the ``hyperloom-multibranch`` command.

The host (the application launcher) injects a campaign spec file and a set of environment variables; this entry checks
both, runs the packaged installer and then replaces itself with ``hyperloom-multibranch hosted run``, which reads the host
inputs itself and reports how the run ended in ``$HYPERLOOM_STATE_DIR/exit.json``.  The host decides whether to restart
the run from that file's ``final`` field and nothing else:

* ``final: true`` -- the run is over, or the request was refused here (return code 2, with the reason in ``error``).
  Never restarted.
* no ``exit.json``, or ``final: false`` -- the process died.  Restarted.

A spec may name its ``flow``: ``multibranch``, the default, or ``conc_sweep``, which the multi-branch command runs too.
The retired ``single`` flow is refused, as is a spec with neither ``flow`` nor ``multibranch`` (written for the retired default) and any other spec the shared
validation refuses (:func:`hyperloom.hosted.campaign.parse_campaign`).

Before the hand-over, on every leg, the entry runs the packaged ``install.sh`` -- the installer the legacy launchers run
at every pod start -- with its output in ``$HYPERLOOM_STATE_DIR/install.log``; an installer that fails ends the run
finally (return code 2).  The installer never sees a host-only input other than ``HYPERLOOM_STATE_DIR``, which it needs
to install the hosted extra.  The command's environment is the entry's own, with the running interpreter's directory
first on ``PATH`` so the command resolves to, and runs with, the interpreter the installer installed into.  The
isolation of the run's specialists is checked by that command, not here.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.shared_storage import (
    make_shared_dirs,
    open_shared_replacement,
    setgid_warnings,
    writable_problems,
)
from hyperloom.hosted.campaign import CampaignError, parse_campaign
from hyperloom.hosted.host_info import HOST_ONLY_VARS, HostInfo, HostInfoError, Var, read_host_info
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH

EXIT_FILE = "exit.json"
MULTIBRANCH_COMMAND = "hyperloom-multibranch"
INSTALL_SCRIPT = Path(__file__).resolve().parent.parent / "inference_optimizer" / "assets" / "install.sh"
INSTALL_LOG = "install.log"
RETURNCODE_REFUSED = 2
ENCODING = "utf-8"
#: Where install.sh and the run keep their runtime state; a hosted leg defaults it to the container's own disk.
ENV_RUNTIME_DIR = "HYPERLOOM_RUNTIME_DIR"
#: ``exit.json``: replaced by rename, readable by the group (the next leg's account).
_DURABLE = WriteOptions(fsync=True, shared=True)
_USAGE = "usage: hyperloom hosted run"


class Refused(Exception):
    """The request cannot be served as given; the run ends, finally, with return code 2."""


class Unrecordable(Refused):
    """A refusal ``exit.json`` cannot record (this account cannot write the state directory): said on stderr only."""


def main(argv: list[str] | None = None) -> int:
    """Run the hosted campaign; the process return code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["run"]:
        print(_USAGE, file=sys.stderr)
        return RETURNCODE_REFUSED
    state_dir = _state_dir_from_environment()
    try:
        return _run()
    except Refused as exc:
        print(f"[hosted] refused: {exc}", file=sys.stderr)
        if state_dir is not None and not isinstance(exc, Unrecordable):
            write_exit(state_dir, final=True, returncode=RETURNCODE_REFUSED, error=str(exc))
        return RETURNCODE_REFUSED


def _run() -> int:
    info = _read_inputs()
    _check_spec(info)
    user_data = Path(os.environ.get(ENV_USER_DATA_PATH, ""))
    if not user_data.is_absolute():
        raise Refused(f"{ENV_USER_DATA_PATH} must be set to an absolute path")
    _check_shared_storage(info.state_dir, user_data)
    _default_runtime_dir()
    # install.sh's ``ensure_hosted_extra`` step installs the wheel's ``hosted`` extra into this interpreter before the command starts.
    _put_interpreter_first_on_path()
    _run_installer(info.state_dir)
    _exec_multibranch()
    return 0


def _check_shared_storage(state_dir: Path, user_data: Path) -> None:
    """Refuse, naming each, the key directories this account cannot create and remove a file in.

    A leg may be another account than the one before it; a directory that account left without group write fails here,
    before anything is written, instead of at the first write of a resume. A key directory that is not setgid only
    warns: what is made there takes this account's group.
    """
    with contextlib.suppress(OSError):  # one that cannot be made is named by the probe below
        make_shared_dirs(user_data)
    for warning in setgid_warnings({Var.STATE_DIR: state_dir, ENV_USER_DATA_PATH: user_data}):
        print(f"[hosted] WARNING: {warning}", file=sys.stderr, flush=True)
    unrecordable = writable_problems({Var.STATE_DIR: state_dir})
    problems = unrecordable + writable_problems({ENV_USER_DATA_PATH: user_data})
    if problems:
        message = f"this account (uid {os.geteuid()}) cannot write the shared storage: {'; '.join(problems)}"
        raise (Unrecordable if unrecordable else Refused)(message)


def _default_runtime_dir() -> None:
    """Give the leg a runtime directory of its own on the container's disk unless the host named one.

    The default under ``USER_DATA_PATH`` is shared with the other accounts of the run, and the installer writes files
    there that later steps source and run.
    """
    if not os.environ.get(ENV_RUNTIME_DIR):
        os.environ[ENV_RUNTIME_DIR] = tempfile.mkdtemp(prefix="hyperloom-runtime-")


def _put_interpreter_first_on_path() -> None:
    """Make the directory of the running interpreter the first entry of ``PATH`` for the installer and the command.

    The console scripts the installer puts into the run's environment (the TraceLens CLIs) live there.
    """
    bin_dir = str(Path(sys.executable).parent)
    entries = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item and item != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *entries])


def _installer_environment(state_dir: Path) -> dict[str, str]:
    """The environment of ``install.sh``: the running interpreter and its environment are the ones it installs into.

    Without ``INFERENCE_OPTIMIZER_FORCE_PYTHON`` the script prefers the image's ``/opt/venv`` over ``PYTHON``, where this
    wheel is not installed.  ``HYPERLOOM_STATE_DIR`` is the one host input it gets back: it is how the script knows the
    install is hosted and installs the wheel's ``hosted`` extra.
    """
    environment = {name: value for name, value in os.environ.items() if name not in HOST_ONLY_VARS}
    environment[Var.STATE_DIR] = str(state_dir)
    environment["PYTHON"] = sys.executable
    environment["INFERENCE_OPTIMIZER_FORCE_PYTHON"] = "1"
    if sys.prefix != sys.base_prefix:
        environment["VIRTUAL_ENV"] = sys.prefix
    return environment


def _run_installer(state_dir: Path) -> None:
    """Run the packaged ``install.sh`` (no flags, like the legacy launchers) and log its output; a failure ends the run.

    The script comes from the installed wheel, so it is of the commit that runs.  It is idempotent and runs on every leg.
    Its output goes to a file.  The log is only the script's own output; the entry adds nothing from the environment.
    Each leg's log is a new file of its own, renamed over ``install.log`` before the script starts
    (:func:`~hyperloom_common.shared_storage.open_shared_replacement`): the previous leg's, which may be another
    account's, and whatever a member of the group put there are replaced, never opened.
    """
    log_path = state_dir / INSTALL_LOG
    print(f"[hosted] running {INSTALL_SCRIPT.name}; log {log_path}", flush=True)
    try:
        with os.fdopen(open_shared_replacement(log_path), "wb") as log:
            returncode = subprocess.run(
                ["bash", str(INSTALL_SCRIPT)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                env=_installer_environment(state_dir),
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
    except OSError as exc:
        raise Refused(f"install.sh could not run: {exc.strerror or type(exc).__name__}; log {log_path}") from None
    if returncode != 0:
        raise Refused(f"install.sh failed with exit code {returncode}; see {log_path}")


def _state_dir_from_environment() -> Path | None:
    """The state directory, read early so that even a refusal of the other inputs can be reported there."""
    raw = os.environ.get(Var.STATE_DIR, "")
    path = Path(raw)
    return path if raw and path.is_absolute() and path.is_dir() else None


def _read_inputs() -> HostInfo:
    try:
        return read_host_info(os.environ, require=())
    except HostInfoError as exc:
        raise Refused(str(exc)) from None


def _check_spec(info: HostInfo) -> None:
    """Refuse here, finally, a spec the command would refuse: its shared part is validated before the installer runs."""
    try:
        parse_campaign(json.loads(info.input_file.read_text(encoding=ENCODING)))
    except (ValueError, UnicodeDecodeError) as exc:  # json.JSONDecodeError and CampaignError are ValueErrors
        raise Refused(str(exc) if isinstance(exc, CampaignError) else f"{Var.INPUT_FILE} is not valid JSON") from None


def _exec_multibranch() -> None:
    if shutil.which(MULTIBRANCH_COMMAND) is None:
        raise Refused(f"the hosted run needs the {MULTIBRANCH_COMMAND} command, which is not installed")
    try:
        os.execvp(MULTIBRANCH_COMMAND, [MULTIBRANCH_COMMAND, "hosted", "run"])
    except OSError as exc:
        raise Refused(f"could not start {MULTIBRANCH_COMMAND}: {exc.strerror or type(exc).__name__}") from None


def write_exit(state_dir: Path, *, final: bool, returncode: int, **details: object) -> None:
    """Durably write ``exit.json`` in *state_dir*: ``final`` tells the host whether to start the run again."""
    atomic_write_json(state_dir / EXIT_FILE, {"final": final, "returncode": returncode, **details}, options=_DURABLE)
