# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the account a session's specialists run as can reach the session directory, asked as the session starts.

Where specialists run as another account (:class:`~hyperloom.common.agent_isolation.AccountIsolation`), each is started
in its workspace below the session's branch directory, which the launch shares with the account's group
(:meth:`~hyperloom.common.env_safety.RunAs.share_tree`). What the launch cannot share is the way there: a directory
above the session the account cannot search fails every specialist at its start, the agent CLI unable to read its
system prompt, after the session's PRELUDE has run for hours. So the session asks first, the way a launch would: it
shares a directory below the branch directory as a launch shares a workspace and runs, as the account, a shell that
searches each directory down to it and writes and reads a file in it, and refuses the session when that fails, naming
the first directory the account cannot search, its owner and mode, and how to open it. Specialists not started as an
account (under bubblewrap or unconfined, as this process's own account) are not asked about.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.agent_isolation import AccountIsolation, AgentIsolation, IsolationRefused
from hyperloom.common.env_safety import RunAs, RunAsError
from hyperloom.orchestrator.specialists.launch_env import resolve_launch_isolation
from hyperloom_common.shared_storage import ownership
from hyperloom_multibranch.scheduling.session_environment import EnvironmentIncomplete

log = logging.getLogger(__name__)

#: The name a probe directory starts with below the branch directory; it is removed once the probe ends.
PROBE_PREFIX = ".MB-reach-"
PROBE_TIMEOUT_SEC = 60.0
#: Searches each directory after the first argument, then writes and reads a file in the first; says which failed.
PROBE_SCRIPT = """\
probe=$1
shift
for dir in "$@"; do
  test -x "$dir" || { printf 'search\\t%s\\n' "$dir"; exit 3; }
done
printf ok > "$probe/probe" 2>/dev/null && test "$(cat "$probe/probe" 2>/dev/null)" = ok || {
  printf 'write\\t%s\\n' "$probe"; exit 4; }
"""
_SEARCH = "search"
_WRITE = "write"


@dataclass(frozen=True)
class ProbeResult:
    """How the probe run as the account ended: its exit code, and what it printed."""

    returncode: int
    output: str


#: Runs an argv as an account, as a specialist is started, and says how it ended.
Prober = Callable[[RunAs, Sequence[str]], ProbeResult]


def run_as_account(run_as: RunAs, argv: Sequence[str]) -> ProbeResult:
    """``argv`` run as ``run_as`` the way a specialist is started: demoted to its uid, gid and groups, with no
    capability (:meth:`~hyperloom.common.env_safety.RunAs.command`). A probe that cannot be started is an exit of
    ``-1`` saying why."""

    try:
        done = subprocess.run(
            run_as.command(argv),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SEC,
            check=False,
            env={"PATH": os.defpath},
            cwd="/",
            **run_as.popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(-1, str(exc))
    return ProbeResult(done.returncode, done.stdout + done.stderr)


def require_specialist_reach(
    branches: Path,
    *,
    isolation: Callable[[], AgentIsolation] | None = None,
    prober: Prober | None = None,
) -> None:
    """Refuse a session whose specialists run as an account (``isolation``, as the agent runner chooses it) that
    cannot search the way to ``branches``, the session's branch directory, or write and read a file in a directory
    below it shared with its group as a launch shares a workspace, asked by running a probe as the account through
    ``prober``. Nothing is asked when specialists do not run as an account, or when no tier works: the start of the
    session reports that (:mod:`~hyperloom_multibranch.scheduling.launch_isolation`).

    Raises:
        EnvironmentIncomplete: The account cannot reach ``branches``, naming the first directory it cannot search with
            its owner and mode and how to open it, or cannot write below it, or the probe could not run.
    """

    try:
        found = (isolation or resolve_launch_isolation)()
    except (IsolationRefused, RunAsError):
        return
    if not isinstance(found, AccountIsolation):
        return
    run_as = found.run_as
    probe = Path(tempfile.mkdtemp(prefix=PROBE_PREFIX, dir=branches))
    try:
        try:
            run_as.share_tree(probe)
        except OSError as exc:
            raise EnvironmentIncomplete(
                f"cannot share a directory below {branches} with the specialists' account {_account(run_as)}, as each"
                f" launch shares its workspace: {exc}"
            ) from exc
        result = (prober or run_as_account)(
            run_as, ["sh", "-c", PROBE_SCRIPT, "sh", str(probe), *map(str, _way_to(branches))]
        )
    finally:
        _remove(probe)
    if result.returncode != 0:
        raise EnvironmentIncomplete(_refusal(run_as, branches, result))
    log.info("specialist reach: the account %s can reach %s", run_as.user, branches)


def _way_to(directory: Path) -> list[Path]:
    """Every directory from ``/`` down to ``directory``, which the account must search to reach what is below it."""

    return [*reversed(directory.parents), directory]


def _account(run_as: RunAs) -> str:
    groups = ",".join(str(gid) for gid in dict.fromkeys((run_as.gid, *run_as.groups)))
    return f"{run_as.user} (uid {run_as.uid}, groups {groups})"


def _refusal(run_as: RunAs, branches: Path, result: ProbeResult) -> str:
    """Why the account cannot reach ``branches``, from what the probe printed, and how to let it."""

    said = result.output.strip()
    what, _, path = said.splitlines()[0].partition("\t") if said else ("", "", "")
    account = _account(run_as)
    lead = "every specialist of this session would fail at its start: the specialists' account"
    if what == _SEARCH and path:
        return (
            f"{lead} {account} cannot search {path}{ownership(Path(path))}, on the way to {branches};"
            f" give {path} search permission to others (chmod o+x {path}) or give the account that directory's group"
            f"{_group_of(Path(path))}"
        )
    if what == _WRITE:
        return (
            f"{lead} {account} cannot write and read a file in a directory below {branches} shared with its group"
            f" {run_as.gid}, as each launch shares its workspace"
        )
    return (
        f"{lead} {account} could not be asked whether it reaches {branches}: the probe exited {result.returncode}"
        + (f": {said}" if said else "")
    )


def _group_of(path: Path) -> str:
    try:
        return f" ({path.stat().st_gid})"
    except OSError:
        return ""


def _remove(probe: Path) -> None:
    try:
        shutil.rmtree(probe)
    except OSError as exc:
        log.warning("specialist reach: cannot remove the probe directory %s: %s", probe, exc)


__all__ = [
    "PROBE_PREFIX",
    "ProbeResult",
    "Prober",
    "require_specialist_reach",
    "run_as_account",
]
