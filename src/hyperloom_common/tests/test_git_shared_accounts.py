# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two real accounts of one group take turns writing a repository the first made as the app makes it below shared
storage (:func:`hyperloom_common.git_safety.share_objects_and_refs`): the second commits, and cannot touch the config,
``info`` or hooks git reads. Root only, through :mod:`hyperloom_common.tests.shared_accounts`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import hyperloom_common
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts

_PRELUDE = """
import json, os, subprocess
from pathlib import Path
from hyperloom_common import git_safety
ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
       "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.local", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.local"}
EXCLUDES = ("-c", "core.excludesFile=/dev/null")
def git(*args, umask=-1, **env):
    done = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], env={**ENV, **env}, umask=umask,
                          capture_output=True, text=True, check=False)
    return done.returncode, done.stdout.strip() + done.stderr.strip()
def ok(*args, **kwargs):
    code, out = git(*args, **kwargs)
    assert code == 0, (args, out)
    return out
def errno_of(action):
    try:
        action()
    except OSError as error:
        return error.errno
    return 0
"""

_FIRST = """
session = Path({session!r})
os.mkdir(session); os.chmod(session, 0o2775)
base = session / "source"
os.mkdir(base, 0o755); os.chmod(base, 0o2755)
(base / "mod.py").write_text("old = 1\\n")
ok("init", "-q", "--template=", "--shared=group", str(base), umask=git_safety.git_dir_umask())
common = base / ".git"
git_safety.share_objects_and_refs(common)
(common / "info" / "exclude").write_text("")
ok("-C", str(base), "add", "-A"); ok("-C", str(base), "commit", "-qm", "base")
ok("-C", str(base), "worktree", "add", "-q", "-b", "first", str(session / "first"), umask=0o002, GIT_COMMON_DIR=str(common))
repo = session / "repo"
os.mkdir(repo, 0o755)
ok("init", "-q", "--bare", "--template=", "--shared=group", "--initial-branch=main", str(repo), umask=git_safety.git_dir_umask())
git_safety.share_objects_and_refs(repo)
blob = ok("hash-object", "-w", "--stdin", GIT_DIR=str(repo))
tree = subprocess.run(["git", "mktree"], input="", env={{**ENV, "GIT_DIR": str(repo)}}, capture_output=True, text=True).stdout.strip()
root = ok("commit-tree", "-m", "root", tree, GIT_DIR=str(repo))
ok("update-ref", "refs/heads/main", root, GIT_DIR=str(repo))
print(json.dumps({{"group": git_safety.check_shared_repository(common), "root": root}}))
"""

_SECOND = """
session = Path({session!r})
base, repo = session / "source", session / "repo"
common = base / ".git"
found = {{"group": git_safety.check_shared_repository(common), "bare_group": git_safety.check_shared_repository(repo)}}
ok("-c", f"safe.directory={{base}}", "-C", str(base), "worktree", "add", "-q", "-b", "second", str(session / "second"), umask=0o002, GIT_COMMON_DIR=str(common))
for name in ("second", "first"):
    worktree = session / name
    (worktree / f"{{name}}.py").write_text("by the second account\\n")
    admin = git_safety.trusted_admin_dir(worktree, common)
    added = git_safety.app_git(worktree, *EXCLUDES, "add", "-A", admin=admin, env={{"HOME": str(session)}})
    made = git_safety.app_git(worktree, *EXCLUDES, "commit", "-qm", "second", admin=admin, env={{**ENV, "HOME": str(session)}})
    found[name] = [added.returncode, made.returncode, made.stderr]
bare = {{"GIT_DIR": str(repo), "GIT_COMMON_DIR": str(repo)}}
blob = ok("hash-object", "-w", "--stdin", **bare)
head = ok("rev-parse", "refs/heads/main", **bare)
tree = subprocess.run(["git", "mktree"], input="", env={{**ENV, **bare}}, capture_output=True, text=True).stdout.strip()
moved = ok("commit-tree", "-p", head, "-m", "second", tree, **bare)
found["move_head_s_branch"] = git("update-ref", "refs/heads/main", moved, head, **bare)
found["create"] = git("update-ref", "refs/heads/made", moved, **bare)
found["delete"] = git("update-ref", "-d", "refs/heads/made", **bare)
mine = common / "mine"
mine.write_text("[filter \\"own\\"]\\n")
found["append_config"] = errno_of(lambda: open(common / "config", "a").close())
found["replace_config"] = errno_of(lambda: os.replace(mine, common / "config"))
found["replace_bare_config"] = errno_of(lambda: os.replace(mine, repo / "config"))
found["plant_attributes"] = errno_of(lambda: (common / "info" / "attributes").write_text("* filter=own\\n"))
found["rename_git_dir"] = errno_of(lambda: os.rename(common, base / "moved"))
os.unlink(mine)
found["make_hooks"] = errno_of(lambda: os.mkdir(common / "hooks"))
found["plant_commondir"] = errno_of(lambda: (common / "commondir").write_text(str(session / "elsewhere")))
print(json.dumps(found))
"""

_THIRD = """
common = Path({session!r}) / "source" / ".git"
print(json.dumps({{"refused": errno_of(lambda: git_safety.check_shared_repository(common))}}))
"""


def _lib(accounts: Accounts) -> None:
    here = Path(hyperloom_common.__file__).parent
    for name in ("fs_utils.py", "git_safety.py"):
        shutil.copyfile(here / name, accounts.lib / "hyperloom_common" / name)


def test_the_next_account_of_the_group_commits_but_writes_none_of_git_s_own_state():
    with other_accounts() as accounts:
        _lib(accounts)
        session = str(accounts.shared_dir("work") / "session")

        first = json.loads(accounts.run(accounts.first, _PRELUDE + _FIRST.format(session=session)))
        second = json.loads(accounts.run(accounts.second, _PRELUDE + _SECOND.format(session=session)))
        third = json.loads(accounts.run(accounts.first, _PRELUDE + _THIRD.format(session=session)))

    assert first["group"] == accounts.group
    assert second["group"] == second["bare_group"] == accounts.group
    assert second["second"] == [0, 0, ""], "the second account commits in a worktree of its own"
    assert second["first"] == [0, 0, ""], "and in the first account's, whose admin directory is the group's"
    assert second["move_head_s_branch"][0] == 0, "HEAD.lock is made in the sticky git directory"
    assert second["create"][0] == 0 and second["delete"][0] == 0, "packed-refs.lock too"
    denied = {13, 1}  # EACCES, EPERM
    assert {second[key] for key in ("append_config", "replace_config", "replace_bare_config")} <= denied
    assert second["plant_attributes"] in denied and second["rename_git_dir"] in denied
    assert second["make_hooks"] == 17, "the first account made an empty hooks directory (EEXIST)"
    assert second["plant_commondir"] == 0, "the sticky git directory still lets the group make a name of its own"
    assert third["refused"] == 1, "which the first account's next git refuses (EPERM)"
