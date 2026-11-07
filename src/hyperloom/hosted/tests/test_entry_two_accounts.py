# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two legs of one hosted run as two accounts of one group: the second starts on what the first left in the state
directory and the session's archive, and gets past the installer and every file the entry and its run write again."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hyperloom.hosted import entry
from hyperloom.hosted.events import segment_name
from hyperloom.hosted.host_info import Var
from hyperloom.hosted.tests.test_campaign import MINIMAL
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts

_SRC = str(Path(entry.__file__).parents[2])

#: One leg: the entry, with the packaged installer and ``optimize`` stood in for, run as this account. The stand-in
#: writes what a run writes outside the session's own files: an event, and the session package into the archive.
_LEG = """
import json, os, sys
sys.dont_write_bytecode = True
leg = json.loads(sys.argv[1])
for name in [name for name in os.environ if name.startswith(("HYPERLOOM_", "E2B_"))]:
    del os.environ[name]
os.environ.update(leg["env"])
sys.path.insert(0, leg["src"])
from pathlib import Path
from hyperloom.hosted import entry
from hyperloom.hosted.events import HostedProgressEvents
from hyperloom.inference_optimizer.breakdown.session_package import package_session_artifacts
from hyperloom_common.shared_storage import make_shared_dirs

entry.INSTALL_SCRIPT = Path(leg["installer"])
entry.require_agent_isolation = lambda *, bwrap: None  # how agents are isolated is not this test's subject

def optimize(argv):
    session = Path(next(a for a in argv if a.startswith("--session-dir=")).partition("=")[2])
    events = HostedProgressEvents(Path(leg["env"]["HYPERLOOM_STATE_DIR"]) / "events", "run-1")
    events.emit("hyperloom.warning", {"code": "leg", "message": leg["name"]})
    events.close()
    summary = session / "runs" / leg["name"] / "summary.txt"
    make_shared_dirs(summary.parent)  # as the session skeleton is made
    summary.write_text(leg["name"] + "\\n")
    summary.chmod(0o664)  # the session's own files are not this test's subject
    package_session_artifacts(session, session_id="s")
    return 0

entry._optimize_main = optimize
sys.exit(entry.main(["run"]))
"""

#: What a member of the group can do to the state directory between two legs: put a link where the log goes.
_PLANT = """
import os, sys
os.unlink(sys.argv[1])
os.symlink(sys.argv[2], sys.argv[1])
"""

#: And to the session: a file of its own choosing that the next leg packages, and a link where the archive's copy of it
#: goes, to a directory of the next leg's account, for that account to write the file into.
_PLANT_ON_THE_WAY = """
import os, sys
session, archive, victim = sys.argv[1:]
os.makedirs(os.path.join(session, "runs", "planted"))
with open(os.path.join(session, "runs", "planted", "summary.txt"), "w") as chosen:
    chosen.write("chosen by another account\\n")
os.symlink(victim, os.path.join(archive, "runs", "planted"))
"""


def _private_dir(accounts: Accounts, name: str, uid: int) -> Path:
    path = accounts.root / name
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    return path


def _owner_and_mode(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    return info.st_uid, stat.S_IMODE(info.st_mode)


def _leg(accounts: Accounts, uid: int, name: str, env: dict[str, str], installer: Path) -> str:
    leg = {"name": name, "src": _SRC, "installer": str(installer), "env": {**env, "TMPDIR": env[f"TMPDIR_{uid}"]}}
    return accounts.run(uid, f"import sys\nsys.argv[1:] = [{json.dumps(leg)!r}]\n{_LEG}", umask=0o077)


def _reaches_the_checkout(accounts: Accounts) -> bool:
    probe = f"import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, {_SRC!r})\nimport hyperloom.hosted.entry\n"
    try:
        accounts.run(accounts.second, probe)
    except AssertionError:
        return False
    return True


def test_a_leg_of_another_account_of_the_group_takes_over_everything_the_previous_leg_left():
    with other_accounts() as accounts:
        if not _reaches_the_checkout(accounts):
            pytest.skip("the other accounts cannot read this checkout or its interpreter's packages")
        first, second = accounts.first, accounts.second
        state, user_data = accounts.shared_dir("state"), accounts.shared_dir("user-data")
        spec = accounts.root / "campaign.json"
        spec.write_text(json.dumps(MINIMAL))
        spec.chmod(0o644)
        installer = accounts.root / "install.sh"
        installer.write_text('#!/bin/bash\necho "installer ran as $(id -u)"\n')
        installer.chmod(0o755)
        env = {
            Var.INPUT_FILE: str(spec),
            Var.RUN_ID: "run-1",
            Var.RUN_CREATED_AT: "2026-10-23T08:30:05Z",
            Var.STATE_DIR: str(state),
            "USER_DATA_PATH": str(user_data),
            **{f"TMPDIR_{uid}": str(_private_dir(accounts, f"tmp-{uid}", uid)) for uid in (first, second)},
        }

        _leg(accounts, first, "leg-1", env, installer)

        session = Path(json.loads((state / "session.json").read_text())["session_dir"])
        archive = session / "archive"
        assert (state / "install.log").read_text() == f"installer ran as {first}\n"
        for left in (state / "install.log", state / "exit.json", state / "session.json"):
            assert _owner_and_mode(left) == (first, 0o664), left
        assert _owner_and_mode(state / "events" / segment_name(1)) == (first, 0o664)
        assert _owner_and_mode(archive / "runs") == (first, 0o2770)
        assert _owner_and_mode(archive / "hyperloom-session-packages" / "s.zip") == (first, 0o660)
        # Between the legs a member of the group puts a link where the log goes, to a file of the next leg's account.
        victim = _private_dir(accounts, "second-home", second) / "kept"
        victim.write_text("kept\n")
        os.chown(victim, second, second)
        accounts.run(first, f"import sys\nsys.argv[1:] = [{str(state / 'install.log')!r}, {str(victim)!r}]\n{_PLANT}")
        # And a link at a directory of the archive, to a directory of the next leg's account holding a file the
        # package's copy would replace.
        kept = victim.parent / "summary.txt"
        kept.write_text("the next account's own\n")
        os.chown(kept, second, second)
        kept.chmod(0o600)
        planted = [str(session), str(archive), str(victim.parent)]
        accounts.run(first, f"import sys\nsys.argv[1:] = {planted!r}\n{_PLANT_ON_THE_WAY}", umask=0o002)

        out = _leg(accounts, second, "leg-2", env, installer)

        assert "[hosted] starting session" in out
        assert victim.read_text() == "kept\n", "the planted link is replaced, never written through"
        assert kept.read_text() == "the next account's own\n", "a link on the way is never followed"
        assert sorted(path.name for path in victim.parent.iterdir()) == ["kept", "summary.txt"]
        assert (state / "install.log").read_text() == f"installer ran as {second}\n"
        assert json.loads((state / "exit.json").read_text()) == {"final": True, "returncode": 0, "stop_reason": None}
        for written in (state / "install.log", state / "exit.json", state / "session.json"):
            assert _owner_and_mode(written) == (second, 0o664), written
        events = state / "events"
        assert [_owner_and_mode(events / segment_name(n)) for n in (1, 2)] == [(first, 0o664), (second, 0o664)]
        lines = [json.loads(line) for n in (1, 2) for line in (events / segment_name(n)).read_bytes().splitlines()]
        assert [(event["id"], event["data"]["message"]) for event in lines] == [("1", "leg-1"), ("2", "leg-2")]
        # The second leg's package replaces the first's, and its loose copy of the first leg's file replaces the one
        # the first leg left, in the directories the first leg made.
        assert _owner_and_mode(archive / "hyperloom-session-packages" / "s.zip") == (second, 0o660)
        assert _owner_and_mode(archive / "runs" / "leg-1" / "summary.txt") == (second, 0o660)
        assert _owner_and_mode(archive / "runs" / "leg-2") == (second, 0o2770)
        assert (archive / "runs" / "leg-1" / "summary.txt").read_text() == "leg-1\n"
