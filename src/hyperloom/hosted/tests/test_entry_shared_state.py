# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two legs of one hosted run as two accounts of one group: the second starts on what the first left in the state
directory, gets past the installer, reads the session the first named, and replaces every file the entry and the
multibranch command write there, readable by the group."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hyperloom.hosted import entry
from hyperloom.hosted.host_info import Var
from hyperloom.hosted.tests.test_campaign import MINIMAL
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts

_SRC = str(Path(entry.__file__).parents[2])

#: One leg: the entry, with the packaged installer stood in for, run as this account. The hand-over is stood in for by
#: what ``hyperloom-multibranch hosted run`` writes in the state directory, with its own writers: it takes over the
#: session an earlier leg named in ``session.json``, else starts one, and ends the run with ``exit.json``.
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
from hyperloom_common.shared_storage import make_shared_dirs
from hyperloom_multibranch.scheduling.hosted_run import HostedExit, record_session, recorded_session, write_final_exit

entry.INSTALL_SCRIPT = Path(leg["installer"])

def multibranch():
    state = Path(os.environ["HYPERLOOM_STATE_DIR"])
    session = recorded_session(state) or Path(os.environ["USER_DATA_PATH"]) / "m" / "session"
    make_shared_dirs(session)
    print(f"[leg] session {session}", flush=True)
    record_session(state, session)
    write_final_exit(state, HostedExit(returncode=0, stop_reason=leg["name"]))

entry._exec_multibranch = multibranch
sys.exit(entry.main(["run"]))
"""

#: What a member of the group can do to the state directory between two legs: put a link where the log goes.
_PLANT = """
import os, sys
os.unlink(sys.argv[1])
os.symlink(sys.argv[2], sys.argv[1])
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
    probe = (
        f"import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, {_SRC!r})\n"
        "import hyperloom.hosted.entry, hyperloom_multibranch.scheduling.hosted_run\n"
    )
    try:
        accounts.run(accounts.second, probe)
    except AssertionError:
        return False
    return True


def test_a_leg_of_another_account_of_the_group_takes_over_everything_the_previous_leg_left_in_the_state_dir():
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
        assert (state / "install.log").read_text() == f"installer ran as {first}\n"
        for left in (state / "install.log", state / "exit.json", state / "session.json"):
            assert _owner_and_mode(left) == (first, 0o664), left
        # Between the legs a member of the group puts a link where the log goes, to a file of the next leg's account.
        victim = _private_dir(accounts, "second-home", second) / "kept"
        victim.write_text("kept\n")
        os.chown(victim, second, second)
        accounts.run(first, f"import sys\nsys.argv[1:] = [{str(state / 'install.log')!r}, {str(victim)!r}]\n{_PLANT}")

        out = _leg(accounts, second, "leg-2", env, installer)

        assert f"[leg] session {session}" in out, "the second account reads the session the first named"
        assert victim.read_text() == "kept\n", "the planted link is replaced, never written through"
        assert (state / "install.log").read_text() == f"installer ran as {second}\n"
        assert json.loads((state / "exit.json").read_text()) == {"final": True, "returncode": 0, "stop_reason": "leg-2"}
        assert json.loads((state / "session.json").read_text()) == {"session_dir": str(session)}
        for written in (state / "install.log", state / "exit.json", state / "session.json"):
            assert _owner_and_mode(written) == (second, 0o664), written
