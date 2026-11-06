# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A confinement derived from configuration, launched for real through :meth:`BwrapIsolation.launch`: the program
runs, the home's logins do not exist inside, a credential inside a bound tree is masked at every path that shows it,
and the mark is set inside only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common.tests._bwrap_host import usable_bwrap

from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.common.confinement import Confinement, credential_paths, toolchain_paths

pytestmark = pytest.mark.agent_isolation

MARK = "HYPERLOOM_REAP_MARK"
#: Run inside: each named path's read (``ok:<text>`` or the errno name), the tool's output, a write, the mark.
CHECK = r"""
import errno, json, os, shutil, subprocess, sys
def outcome(action):
    try:
        return "ok:" + action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
def write(path):
    with open(path, "w") as handle:
        handle.write("from inside")
    return "written"
paths, work = json.loads(sys.argv[1]), sys.argv[2]
seen = {name: outcome(lambda path=path: open(path).read()) for name, path in paths.items()}
seen["tool"] = shutil.which("tool") and subprocess.run(["tool"], capture_output=True, text=True).stdout.strip()
seen["write"] = outcome(lambda: write(os.path.join(work, "out")))
seen["mark"] = os.environ.get(sys.argv[3], "")
print(json.dumps(seen))
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return path


def _run(isolation: BwrapIsolation, confinement: Confinement, env: dict[str, str], paths: dict[str, Path], work: Path):
    argv, child_env = isolation.launch(
        [sys.executable, "-I", "-S", "-c", CHECK, json.dumps({k: str(v) for k, v in paths.items()}), str(work), MARK],
        confinement.policy(chdir=work),
        {**env, MARK: "mark-1"},
        marks=(MARK,),
    )
    assert MARK not in child_env
    done = subprocess.run(argv, env=child_env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "SECRET" not in done.stdout + done.stderr
    return json.loads(done.stdout)


def test_a_derived_confinement_runs_the_program_and_shows_no_login_of_the_home(tmp_path):
    isolation = BwrapIsolation(usable_bwrap())
    home, hf, work = tmp_path / "home", tmp_path / "hf", tmp_path / "work"
    work.mkdir()
    tool = _write(home / ".local/share/tool/versions/1", "#!/bin/sh\necho tool ran\n")
    tool.chmod(0o755)
    (home / ".local/bin").mkdir(parents=True)
    (home / ".local/bin/tool").symlink_to(tool)
    paths = {
        "auth": _write(home / ".local/share/opencode/auth.json", "SECRET-AUTH"),
        "aws": _write(home / ".aws/credentials", "SECRET-AWS"),
        "hf_token": _write(hf / "token", "SECRET-HF"),
        "hf_model": _write(hf / "hub/model.txt", "weights"),
    }
    env = {"HOME": str(home), "PATH": f"{home / '.local/bin'}:/usr/bin:/bin", "HF_HOME": str(hf)}
    readable = toolchain_paths(env, "tool", writable=(work,))
    masked = credential_paths(env, app_home=home, allowed=(*readable, work))

    seen = _run(isolation, Confinement(writable=(work,), readable=readable, masked=masked), env, paths, work)

    assert seen == {
        "auth": "ENOENT",
        "aws": "ENOENT",
        "hf_token": "EACCES",
        "hf_model": "ok:weights",
        "tool": "tool ran",
        "write": "ok:written",
        "mark": "mark-1",
    }
    assert (work / "out").read_text(encoding="ascii") == "from inside"


def test_a_credential_reached_through_a_linked_directory_is_masked_at_both_paths(tmp_path):
    isolation = BwrapIsolation(usable_bwrap())
    work = tmp_path / "work"
    work.mkdir()
    real, link = tmp_path / "real", tmp_path / "link"
    paths = {
        "real_secret": _write(real / "data/secret", "SECRET-DATA"),
        "link_secret": link / "data/secret",
        "link_visible": _write(real / "data/visible", "shown") and link / "data/visible",
    }
    link.symlink_to(real)
    env = {"PATH": "/usr/bin:/bin"}
    readable = (*toolchain_paths(env, sys.executable, writable=(work,)), link / "data")

    seen = _run(
        isolation, Confinement(writable=(work,), readable=readable, masked=(real / "data/secret",)), env, paths, work
    )

    assert {name: seen[name] for name in paths} == {
        "real_secret": "EACCES",
        "link_secret": "EACCES",
        "link_visible": "ok:shown",
    }
