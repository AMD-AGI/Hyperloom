# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom hosted run``: the hosted entry for the legacy ``optimize`` flow.

The host (the application launcher) injects a campaign spec file and a set of environment variables; this entry
checks both, picks the session directory, runs ``optimize`` in this process and reports how it ended in
``$HYPERLOOM_STATE_DIR/exit.json``.  The host decides whether to restart the run from that file's ``final`` field and
nothing else:

* ``final: true`` -- the run is over, whatever the return code (budget used, target reached, stop requested --
  by signal or by a ``hosted/control/stop.json`` in the session directory, which a restart finds before it starts ``optimize`` --
  ``optimize`` failed), or the request was refused (return code 2, with the reason in ``error``).  Never restarted.
* no ``exit.json``, or ``final: false`` -- the process died, or a watchdog ended a leg.  Restarted, and the entry
  resumes the same session.

Before ``optimize`` starts or resumes, on every leg, the entry runs the packaged ``install.sh`` -- the installer the
legacy launchers run at every pod start -- with its output in ``$HYPERLOOM_STATE_DIR/install.log``; an installer that
fails ends the run finally (return code 2).  The session directory is a function of the run (see :func:`hyperloom.inference_optimizer.session.paths.hosted_session_dir`),
so a restart finds it again; a directory that holds every file of :data:`SESSION_FILES` is resumed, any other is
started fresh, after the stray files of a session that never finished its first write are removed.  A spec whose ``flow`` is ``multibranch`` is validated here and then handed, environment unchanged, to
the ``hyperloom-multibranch`` command, which reads the host inputs itself.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hyperloom.common.io import WriteOptions, atomic_write_json
from hyperloom.common.owner_scope import local_owner_scope
from hyperloom.hosted.campaign import FLOW_MULTIBRANCH, CampaignError, CampaignSpec, optimize_args, parse_campaign
from hyperloom.hosted.host_info import HostInfo, HostInfoError, Var, read_host_info, take_host_info
from hyperloom.hosted.mcp_host_tools import register_host_tools
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.inference_optimizer.breakdown.session_package import ENV_PACKAGE_DEST_ROOT
from hyperloom.inference_optimizer.breakdown.stop_reasons import SUPERVISOR_RESTART_REASON
from hyperloom.inference_optimizer.session.paths import (
    ENV_USER_DATA_PATH,
    SESSION_FILES,
    ensure_archive_root,
    hosted_session_dir,
    stop_request_path,
)
from hyperloom.inference_optimizer.session.resume_guard import (
    CleanupConfirmationError,
    confirm_scope_gone,
    foreign_owner_scopes,
)
from hyperloom.plugs.errors import ManifestError
from hyperloom.plugs.manifest import LOCKABLE_SOCKETS, load_builtin_profile

EXIT_FILE = "exit.json"
SESSION_FILE = "session.json"
MULTIBRANCH_COMMAND = "hyperloom-multibranch"
ENV_IMAGE = "HYPERLOOM_IMAGE"
INSTALL_SCRIPT = Path(__file__).resolve().parent.parent / "inference_optimizer" / "assets" / "install.sh"
INSTALL_LOG = "install.log"
#: ``stop_reason`` of a run ended by a stop request before it started; the coordinator's name for an operator stop.
STOP_REQUEST_REASON = "signal"
DEFAULT_MCP_HOST = "127.0.0.1"
RETURNCODE_REFUSED = 2
ENCODING = "utf-8"
_DURABLE = WriteOptions(fsync=True)
_USAGE = "usage: hyperloom hosted run"


class Refused(Exception):
    """The request cannot be served as given; the run ends, finally, with return code 2."""


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
        if state_dir is not None:
            write_exit(state_dir, final=True, returncode=RETURNCODE_REFUSED, error=str(exc))
        return RETURNCODE_REFUSED


def _drop_cwd_from_sys_path() -> None:
    """Keep the working directory off ``sys.path``: the host chooses it, and a ``python -m`` start puts it first.

    A directory there named like an installed package (``sglang``) but without ``__init__.py`` is found as a namespace
    package, which wins over an editable install's finder, so the package has no ``__file__`` for ``optimize``.
    """
    cwd = os.path.realpath(os.getcwd())
    sys.path[:] = [item for item in sys.path if item and os.path.realpath(item) != cwd]


def _run() -> int:
    _drop_cwd_from_sys_path()
    info = _read_inputs()
    spec = _read_spec(info)
    if not Path(os.environ.get(ENV_USER_DATA_PATH, "")).is_absolute():
        raise Refused(f"{ENV_USER_DATA_PATH} must be set to an absolute path")
    if spec.flow == FLOW_MULTIBRANCH:
        _exec_multibranch()
        return 0
    lock = _plug_lock(spec)
    # From here on the host-only inputs live in ``info`` only: no child process may inherit them.
    info = take_host_info(os.environ, require=())
    session_dir = _prepare_session_dir(spec, info)
    os.environ[ENV_PACKAGE_DEST_ROOT] = str(ensure_archive_root(session_dir))  # the package outlives the container
    os.environ[ENV_IMAGE] = spec.framework_image
    _announce_grant(info)
    if stop_request_path(session_dir).is_file():
        print("[hosted] a stop was requested before this leg started; ending the run", flush=True)
        write_exit(info.state_dir, final=True, returncode=0, stop_reason=STOP_REQUEST_REASON)
        return 0
    _put_interpreter_first_on_path()
    _run_installer(info.state_dir)
    _confirm_previous_containers_gone(session_dir)
    with _plug_inputs(lock, info), _mcp_service(info, session_dir):
        return _optimize(optimize_command(spec, session_dir), info.state_dir, session_dir)


SCOPE_GONE_REASON = "hosted resume: the host starts a leg only once the previous application container no longer exists"


def _confirm_previous_containers_gone(session_dir: Path) -> None:
    """On a resume in a new container, mark the work a previous container left running as interrupted.

    The host runs one application container per run at a time and starts a new leg only after the previous container
    is gone, so an owner scope in the session that is not this container's belongs to a container that no longer
    exists.  This is ``hyperloom recover --confirm-scope-gone`` for each such scope, from the same code; a leg in the
    container that wrote the session finds no other scope and goes through the preflight's own checks unchanged.
    """
    scope = local_owner_scope()
    if not scope or not is_resumable(session_dir):
        return
    for gone in foreign_owner_scopes(session_dir, scope):
        try:
            confirm_scope_gone(session_dir, owner_scope=gone, reason=SCOPE_GONE_REASON, current_scope=scope)
        except CleanupConfirmationError as exc:
            raise Refused(f"the previous container's work cannot be confirmed gone: {exc}") from None
        print(
            "[hosted] the previous application container is gone; its unfinished work is marked interrupted", flush=True
        )


def _put_interpreter_first_on_path() -> None:
    """Make the directory of the running interpreter the first entry of ``PATH`` for the installer and ``optimize``.

    The console scripts the installer puts into the run's environment (the TraceLens CLIs) live there.
    """
    bin_dir = str(Path(sys.executable).parent)
    entries = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item and item != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *entries])


def _installer_environment() -> dict[str, str]:
    """The environment of ``install.sh``: the running interpreter and its environment are the ones it installs into.

    Without ``INFERENCE_OPTIMIZER_FORCE_PYTHON`` the script prefers the image's ``/opt/venv`` over ``PYTHON``, where this
    wheel is not installed.
    """
    environment = dict(os.environ)
    environment["PYTHON"] = sys.executable
    environment["INFERENCE_OPTIMIZER_FORCE_PYTHON"] = "1"
    if sys.prefix != sys.base_prefix:
        environment["VIRTUAL_ENV"] = sys.prefix
    return environment


def _run_installer(state_dir: Path) -> None:
    """Run the packaged ``install.sh`` (no flags, like the legacy launchers) and log its output; a failure ends the run.

    The script comes from the installed wheel, so it is of the commit that runs.  It is idempotent and runs on every leg.
    Its output goes to a file, never a pipe: the Ray daemon it starts would hold a pipe open.  The log is only the
    script's own output; the entry adds nothing from the environment.
    """
    log_path = state_dir / INSTALL_LOG
    print(f"[hosted] running {INSTALL_SCRIPT.name}; log {log_path}", flush=True)
    try:
        with log_path.open("wb") as log:
            returncode = subprocess.run(
                ["bash", str(INSTALL_SCRIPT)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                env=_installer_environment(),
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
    except OSError as exc:
        raise Refused(f"install.sh could not run: {exc.strerror or type(exc).__name__}; log {log_path}") from None
    if returncode != 0:
        raise Refused(f"install.sh failed with exit code {returncode}; see {log_path}")


@contextmanager
def _mcp_service(info: HostInfo, session_dir: Path) -> Iterator[None]:
    """Serve the host tools for the length of ``optimize`` when the host gave a port and a token; otherwise serve nothing."""
    if info.mcp_port is None or info.mcp_token is None:
        yield
        return
    server = McpServer(
        McpServerConfig(host=info.mcp_host or DEFAULT_MCP_HOST, port=info.mcp_port, host_token=info.mcp_token.reveal()),
        campaign=info.run_id,
    )
    register_host_tools(server, session_dir, info.events_dir)
    try:
        server.start()
    except OSError as exc:
        raise Refused(f"the MCP server cannot start: {type(exc).__name__}") from None
    try:
        with _stop_request_guard(session_dir):
            yield
    finally:
        server.stop()


@contextmanager
def _stop_request_guard(session_dir: Path) -> Iterator[None]:
    """Keep a SIGTERM that ``request_stop`` sends from killing the run while no ``SignalDrain`` is armed.

    The drain exists only inside the coordinator's tick loop; before it (preflight) and after it (teardown) SIGTERM
    would end the process with no ``exit.json``, and the host would restart the run it was asked to stop.  While
    ``stop.json`` exists the signal is only a nudge: the coordinator reads the file every tick and the entry reads it
    before every leg.  Any other SIGTERM keeps its default meaning.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_term(_signum: int, _frame: object) -> None:
        if stop_request_path(session_dir).is_file():
            return
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


@contextmanager
def _plug_inputs(lock: Mapping[str, Mapping[str, str]], info: HostInfo) -> Iterator[None]:
    """Give plug resolution the lock and the host inputs for the length of ``optimize``, however it ends.

    ``optimize`` cannot import this package (it would be a cycle), so the entry hands the values over.
    """
    from hyperloom.inference_optimizer.cli.bootstrap import HostedPlugInputs, set_hosted_plug_inputs

    set_hosted_plug_inputs(HostedPlugInputs(lock=lock, host_info=info))
    try:
        yield
    finally:
        set_hosted_plug_inputs(None)


def _state_dir_from_environment() -> Path | None:
    """The state directory, read early so that even a refusal of the other inputs can be reported there."""
    raw = os.environ.get(Var.STATE_DIR, "")
    path = Path(raw)
    return path if raw and path.is_absolute() and path.is_dir() else None


def _read_inputs() -> HostInfo:
    try:
        return read_host_info(
            os.environ, require=()
        )  # the MCP group is optional: the server runs only when the host gives a port and a token
    except HostInfoError as exc:
        raise Refused(str(exc)) from None


def _read_spec(info: HostInfo) -> CampaignSpec:
    try:
        return parse_campaign(json.loads(info.input_file.read_text(encoding=ENCODING)))
    except (ValueError, UnicodeDecodeError) as exc:  # json.JSONDecodeError and CampaignError are ValueErrors
        raise Refused(str(exc) if isinstance(exc, CampaignError) else f"{Var.INPUT_FILE} is not valid JSON") from None


def _plug_lock(spec: CampaignSpec) -> dict[str, dict[str, str]]:
    """The hosted lock layer: every lockable socket the spec's profile declares, with the implementation it names.

    A declared ``measurement`` must agree with ``placement.measurement``.
    """
    try:
        profile = load_builtin_profile(spec.profile)
    except ManifestError as exc:
        raise Refused(str(exc)) from None
    lock = {name: {"impl": setting["impl"]} for name, setting in profile["sockets"].items() if name in LOCKABLE_SOCKETS}
    measured = lock.get("measurement", {}).get("impl", spec.measurement)
    if measured != spec.measurement:
        raise Refused(
            f"placement.measurement {spec.measurement!r} does not match profile {spec.profile!r}, "
            f"which measures with {measured!r}; the legacy flow measures locally"
        )
    return lock


def _exec_multibranch() -> None:
    if shutil.which(MULTIBRANCH_COMMAND) is None:
        raise Refused(f"flow 'multibranch' needs the {MULTIBRANCH_COMMAND} command, which is not installed")
    try:
        os.execvp(MULTIBRANCH_COMMAND, [MULTIBRANCH_COMMAND, "hosted", "run"])
    except OSError as exc:
        raise Refused(f"could not start {MULTIBRANCH_COMMAND}: {exc.strerror or type(exc).__name__}") from None


def _prepare_session_dir(spec: CampaignSpec, info: HostInfo) -> Path:
    """Create the run's session directory and say where it is, before anything else can fail."""
    session_dir = hosted_session_dir(spec.model_path, info.run_created_at, info.run_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(info.state_dir / SESSION_FILE, {"session_dir": str(session_dir)}, options=_DURABLE)
    return session_dir


def is_resumable(session_dir: Path) -> bool:
    """Whether the directory holds a session that ``optimize --resume-from`` can pick up."""
    return all((session_dir / name).is_file() for name in SESSION_FILES)


def _clear_partial_session(session_dir: Path) -> None:
    """Remove the session files of a first write that was cut short: ``--session-dir`` refuses a directory holding any."""
    for name in SESSION_FILES:
        (session_dir / name).unlink(missing_ok=True)


def optimize_command(spec: CampaignSpec, session_dir: Path) -> list[str]:
    """The ``optimize`` command line: a resume of the directory when it holds a session, else a fresh run of the spec."""
    if is_resumable(session_dir):
        return ["optimize", f"--resume-from={session_dir}", f"--plug-profile={spec.profile}"]
    _clear_partial_session(session_dir)
    return optimize_args(spec, plug_profile=spec.profile, session_dir=str(session_dir))


def _announce_grant(info: HostInfo) -> None:
    """The legacy flow only reports the granted totals; it plans nothing from them."""
    if info.grant_limits is not None:
        print(f"[hosted] grant limits: {info.grant_limits}", flush=True)


def _announce(event_type: str, data: Mapping[str, Any]) -> None:
    """Seam for the progress events (``hyperloom.campaign_failed`` / ``campaign_finished``): the events socket emits here."""


def _optimize_main(argv: list[str]) -> int:
    from hyperloom.inference_optimizer.cli import main as optimize_main

    return optimize_main(argv)


def _optimize(argv: list[str], state_dir: Path, session_dir: Path) -> int:
    """Run ``optimize`` here and record how it ended.  An exception leaves no ``final``: the host restarts the run."""
    print(
        f"[hosted] {'resuming' if argv[1].startswith('--resume-from') else 'starting'} session {session_dir}",
        flush=True,
    )
    try:
        returncode = _optimize_main(argv)
    except SystemExit as exc:
        returncode = _returncode_of(exc)
    reason = _stop_reason(session_dir)
    final = reason != SUPERVISOR_RESTART_REASON  # a watchdog restart ends a leg, not the run
    write_exit(state_dir, final=final, returncode=returncode, stop_reason=reason)
    _announce("hyperloom.campaign_finished" if final else "hyperloom.campaign_leg_ended", {"returncode": returncode})
    return returncode


def _returncode_of(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1


def _stop_reason(session_dir: Path) -> str | None:
    """The ``stop_reason`` the session recorded, or ``None`` when it recorded none or the state cannot be read."""
    try:
        reason = json.loads((session_dir / "state.json").read_text(encoding=ENCODING)).get("stop_reason")
    except (OSError, ValueError, AttributeError):
        return None
    return reason if isinstance(reason, str) and reason else None


def write_exit(state_dir: Path, *, final: bool, returncode: int, **details: object) -> None:
    """Durably write ``exit.json`` in *state_dir*: ``final`` tells the host whether to start the run again."""
    atomic_write_json(state_dir / EXIT_FILE, {"final": final, "returncode": returncode, **details}, options=_DURABLE)
