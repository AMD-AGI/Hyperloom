# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom hosted run``: the hosted entry for the legacy ``optimize`` flow.

The host (the application launcher) injects a campaign spec file and a set of environment variables; this entry
checks both, picks the session directory, runs ``optimize`` in this process and reports how it ended in
``$HYPERLOOM_STATE_DIR/exit.json``.  The host decides whether to restart the run from that file's ``final`` field and
nothing else:

* ``final: true`` -- the run is over, whatever the return code (budget used, target reached, stop requested,
  ``optimize`` failed), or the request was refused (return code 2, with the reason in ``error``).  Never restarted.
* no ``exit.json``, or ``final: false`` -- the process died, or a watchdog ended a leg.  Restarted, and the entry
  resumes the same session.

The session directory is a function of the run (see :func:`hyperloom.inference_optimizer.session.paths.hosted_session_dir`),
so a restart finds it again; a directory that holds every file of :data:`SESSION_FILES` is resumed, any other is
started fresh, after the stray files of a session that never finished its first write are removed.  A spec whose ``flow`` is ``multibranch`` is validated here and then handed, environment unchanged, to
the ``hyperloom-multibranch`` command, which reads the host inputs itself.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.hosted.context import HostedRun, install
from hyperloom.hosted.campaign import FLOW_MULTIBRANCH, CampaignError, CampaignSpec, optimize_args, parse_campaign
from hyperloom.hosted.host_info import HostInfo, HostInfoError, Var, read_host_info, take_host_info
from hyperloom.inference_optimizer.breakdown.stop_reasons import SUPERVISOR_RESTART_REASON
from hyperloom.inference_optimizer.session.paths import (
    ENV_USER_DATA_PATH,
    SESSION_FILES,
    ensure_archive_root,
    hosted_session_dir,
)
from hyperloom.plugs.errors import ManifestError
from hyperloom.plugs.manifest import LOCKABLE_SOCKETS, load_builtin_profile

EXIT_FILE = "exit.json"
SESSION_FILE = "session.json"
MULTIBRANCH_COMMAND = "hyperloom-multibranch"
ENV_IMAGE = "HYPERLOOM_IMAGE"
RETURNCODE_REFUSED = 2
ENCODING = "utf-8"
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
            _write_exit(state_dir, final=True, returncode=RETURNCODE_REFUSED, error=str(exc))
        return RETURNCODE_REFUSED


def _run() -> int:
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
    run = HostedRun(
        spec=spec,
        host_info=info,
        plug_lock=lock,
        session_dir=session_dir,
        archive_root=ensure_archive_root(session_dir),
    )
    install(run)
    _hand_to_plug_host(run)
    os.environ[ENV_IMAGE] = spec.framework_image
    _announce_grant(info)
    return _optimize(optimize_command(spec, session_dir), info.state_dir, session_dir)


def _hand_to_plug_host(run: HostedRun) -> None:
    """Give plug resolution the lock and the host inputs; ``optimize`` cannot import this package (it would be a cycle)."""
    from hyperloom.inference_optimizer.cli.bootstrap import HostedPlugInputs, set_hosted_plug_inputs

    set_hosted_plug_inputs(HostedPlugInputs(lock=run.plug_lock, host_info=run.host_info))


def _state_dir_from_environment() -> Path | None:
    """The state directory, read early so that even a refusal of the other inputs can be reported there."""
    raw = os.environ.get(Var.STATE_DIR, "")
    path = Path(raw)
    return path if raw and path.is_absolute() and path.is_dir() else None


def _read_inputs() -> HostInfo:
    try:
        return read_host_info(os.environ, require=())  # the MCP server is not hosted by this entry yet
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
    _write_json(info.state_dir / SESSION_FILE, {"session_dir": str(session_dir)})
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
    _write_exit(state_dir, final=final, returncode=returncode, stop_reason=reason)
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


def _write_exit(state_dir: Path, *, final: bool, returncode: int, **details: object) -> None:
    _write_json(state_dir / EXIT_FILE, {"final": final, "returncode": returncode, **details})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write *payload* so that a reader sees the old file or the whole new one, never half."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding=ENCODING) as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
