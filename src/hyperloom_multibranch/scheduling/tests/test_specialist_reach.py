# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session whose specialists run as an account that cannot reach its directory is refused as it starts, naming the
directory in the way and how to open it; nothing is asked where specialists do not run as an account.

The probe is a stand-in that runs the real probe script as this account, or answers as the account would; one test
runs it as a real other account where the test may switch to one (root)."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from hyperloom.common.agent_isolation import (
    AccountIsolation,
    AgentIsolation,
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
)
from hyperloom.common.env_safety import RunAs, RunAsError
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_multibranch import cli
from hyperloom_multibranch.scheduling import specialist_reach
from hyperloom_multibranch.scheduling.session_environment import EnvironmentIncomplete
from hyperloom_multibranch.scheduling.specialist_reach import (
    PROBE_PREFIX,
    ProbeResult,
    require_specialist_reach,
    run_as_account,
)

#: The account specialists run as, here the test's own ids: sharing a tree with its group needs no privilege.
ACCOUNT = RunAs("hl-specialist", os.getuid(), os.getgid(), (), "/")


class Prober:
    """A stand-in for the probe run as the account: the real script run as this process, unless ``answer`` says how
    the account would answer; records each argv."""

    def __init__(self, answer: ProbeResult | None = None) -> None:
        self.answer = answer
        self.asked: list[list[str]] = []
        self.probe_dirs: list[Path] = []
        self.modes: list[int] = []

    def __call__(self, run_as: RunAs, argv: Sequence[str]) -> ProbeResult:
        self.asked.append(list(argv))
        self.probe_dirs.append(Path(argv[4]))
        self.modes.append(Path(argv[4]).stat().st_mode)
        if self.answer is not None:
            return self.answer
        done = subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=60)
        return ProbeResult(done.returncode, done.stdout + done.stderr)


def _as(found: AgentIsolation | Exception):
    def isolation() -> AgentIsolation:
        if isinstance(found, Exception):
            raise found
        return found

    return isolation


def _specialists(monkeypatch: pytest.MonkeyPatch, found: AgentIsolation | Exception, prober: object) -> None:
    """Specialists launched as ``found`` says, the probe run as the account through ``prober``."""

    monkeypatch.setattr(specialist_reach, "resolve_launch_isolation", _as(found))
    monkeypatch.setattr(specialist_reach, "run_as_account", prober)


def _branches(tmp_path: Path) -> Path:
    branches = tmp_path / "users" / "a1b2" / "model" / "MB-1" / "MB-branches"
    branches.mkdir(parents=True)
    return branches


@pytest.mark.parametrize(
    "found",
    [
        NoIsolation(("no bubblewrap here",)),
        BwrapIsolation(BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())),
        IsolationRefused(("must be isolated",)),
        RunAsError("no such user"),
    ],
    ids=["unconfined", "bwrap", "refused", "no-account"],
)
def test_nothing_is_asked_where_specialists_do_not_run_as_an_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: AgentIsolation | Exception
) -> None:
    branches = _branches(tmp_path)
    prober = Prober(ProbeResult(3, f"search\t{tmp_path}\n"))
    _specialists(monkeypatch, found, prober)

    require_specialist_reach(branches)

    assert prober.asked == []
    assert list(branches.iterdir()) == [], "no probe directory is made"


def test_an_account_that_reaches_the_session_starts_it_and_the_probe_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branches = _branches(tmp_path)
    prober = Prober()
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), prober)

    require_specialist_reach(branches)

    (argv,) = prober.asked
    (probe,) = prober.probe_dirs
    assert argv[:2] == ["sh", "-c"]
    assert argv[5:] == [str(path) for path in (*reversed(branches.parents), branches)], "every directory on the way"
    assert probe.parent == branches and probe.name.startswith(PROBE_PREFIX)
    (mode,) = prober.modes
    assert mode & 0o2070 == 0o2070, "shared with the account's group as a launch shares a workspace"
    assert list(branches.iterdir()) == [], "the probe directory and the file the account wrote are removed"


def test_an_account_that_cannot_search_a_directory_above_the_session_is_refused_naming_it_and_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branches = _branches(tmp_path)
    user_dir = tmp_path / "users" / "a1b2"
    user_dir.chmod(0o770)
    info = user_dir.stat()
    prober = Prober(ProbeResult(3, f"search\t{user_dir}\n"))
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), prober)

    with pytest.raises(EnvironmentIncomplete) as refused:
        require_specialist_reach(branches)

    message = str(refused.value)
    assert f"hl-specialist (uid {ACCOUNT.uid}, groups {ACCOUNT.gid})" in message
    assert f"cannot search {user_dir} (owner {info.st_uid}:{info.st_gid} 0770)" in message
    assert f"chmod o+x {user_dir}" in message
    assert f"that directory's group ({info.st_gid})" in message
    assert list(branches.iterdir()) == []


def test_the_probe_script_names_the_first_directory_it_cannot_search(tmp_path: Path) -> None:
    """The real script, run as this process, over a way whose middle directory is not searchable by anyone here: a
    stand-in for the account that cannot search it."""

    branches = _branches(tmp_path)
    blocked = tmp_path / "nowhere"  # not a directory: ``test -x`` fails for root too
    prober = Prober()

    result = prober(ACCOUNT, ["sh", "-c", specialist_reach.PROBE_SCRIPT, "sh", str(branches), str(blocked), "/"])

    assert (result.returncode, result.output) == (3, f"search\t{blocked}\n")


def test_an_account_that_cannot_write_below_the_session_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branches = _branches(tmp_path)
    prober = Prober(ProbeResult(4, f"write\t{branches}/{PROBE_PREFIX}x\n"))
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), prober)

    with pytest.raises(EnvironmentIncomplete, match=r"cannot write and read a file in a directory below .*MB-branches"):
        require_specialist_reach(branches)


def test_a_probe_that_could_not_run_refuses_the_session_saying_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branches = _branches(tmp_path)
    prober = Prober(ProbeResult(-1, "[Errno 1] Operation not permitted"))
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), prober)

    with pytest.raises(EnvironmentIncomplete, match=r"the probe exited -1: \[Errno 1\] Operation not permitted"):
        require_specialist_reach(branches)


@pytest.mark.skipif(os.geteuid() != 0, reason="running the probe as another account needs root")
def test_a_real_account_that_cannot_search_the_users_directory_is_refused_by_the_real_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident's shape, with a real other account: the users directory is ``0770`` and the account is neither its
    owner nor in its group."""

    nobody = RunAs("nobody", 65534, 65534, (), "/")
    branches = _branches(tmp_path)
    for directory in branches.parents:  # pytest's own directories are 0700
        directory.chmod(directory.stat().st_mode | 0o005)
    user_dir = tmp_path / "users" / "a1b2"
    os.chown(user_dir, 0, 0)
    user_dir.chmod(0o770)
    _specialists(monkeypatch, AccountIsolation(nobody), run_as_account)

    with pytest.raises(EnvironmentIncomplete, match=rf"cannot search {user_dir} \(owner 0:0 0770\)"):
        require_specialist_reach(branches)

    user_dir.chmod(0o771)
    require_specialist_reach(branches)
    assert list(branches.iterdir()) == []


def _settings(*, stop_after_map: bool) -> object:
    workload = type("Workload", (), {"model_basename": "model"})()
    return type("Settings", (), {"workload": workload, "stop_after_map": stop_after_map, "conc_sweep": False})()


def _run_session(settings: object) -> list[Path]:
    """:func:`cli.run_session` over ``settings``; the session roots it reported. The session's own log handler, which
    it adds as the session is created, is removed again."""

    roots: list[Path] = []
    handlers = list(logging.getLogger().handlers)
    try:
        cli.run_session(settings, None, None, roots.append)  # type: ignore[arg-type]
    finally:
        for handler in set(logging.getLogger().handlers) - set(handlers):
            logging.getLogger().removeHandler(handler)
            handler.close()
    return roots


def test_a_new_session_whose_account_cannot_reach_it_is_refused_before_it_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), Prober(ProbeResult(3, f"search\t{tmp_path}\n")))

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the session was built")

    monkeypatch.setattr(cli, "Session", never)

    with pytest.raises(EnvironmentIncomplete, match=f"cannot search {tmp_path}"):
        _run_session(_settings(stop_after_map=False))


def test_a_map_only_session_which_starts_no_specialist_is_not_asked_about_the_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that stops once its map is made returns before EXPLORE, the only stage that starts a specialist, so an
    account that cannot reach it does not refuse it."""

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    prober = Prober(ProbeResult(3, f"search\t{tmp_path}\n"))
    _specialists(monkeypatch, AccountIsolation(ACCOUNT), prober)
    built: list[object] = []

    class MapOnly:
        def __init__(self, settings: object, *_: object) -> None:
            built.append(settings)

        async def run(self, cap: object) -> object:
            return type("Outcome", (), {"exit_code": 0})()

    monkeypatch.setattr(cli, "Session", MapOnly)
    settings = _settings(stop_after_map=True)

    (root,) = _run_session(settings)

    assert built == [settings], "the session was built and ran"
    assert prober.asked == []
    assert not any(path.name.startswith(PROBE_PREFIX) for path in root.rglob("*"))
