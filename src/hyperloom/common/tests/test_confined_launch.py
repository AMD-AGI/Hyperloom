# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A confinement derived from configuration, launched for real through :meth:`BwrapIsolation.launch`: the program
runs, the home's logins do not exist inside, a credential inside a bound tree is absent (a file) or masked (a
directory) at every path that shows it, and stays so while the app replaces it or logs in during the launch, the mark
is set inside only, and links an agent leaves in its tree reach nothing at the next launch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common.tests._bwrap_host import usable_bwrap

from hyperloom.common import confinement as confinement_module
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.common.confinement import Confinement, PrivatePaths, credential_paths, toolchain_paths

pytestmark = pytest.mark.agent_isolation

MARK = "HYPERLOOM_REAP_MARK"
#: Run inside: each named path's read (``ok:<text>`` or the errno name), the tool's output, each named write, the mark.
CHECK = r"""
import errno, json, os, shutil, subprocess, sys
def outcome(action):
    try:
        return "ok:" + action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
def write(path):
    with open(path, "a") as handle:
        handle.write("from inside")
    return "written"
paths, writes = json.loads(sys.argv[1]), json.loads(sys.argv[2])
seen = {name: outcome(lambda path=path: open(path).read()) for name, path in paths.items()}
seen["tool"] = shutil.which("tool") and subprocess.run(["tool"], capture_output=True, text=True).stdout.strip()
seen.update({name: outcome(lambda path=path: write(path)) for name, path in writes.items()})
seen["mark"] = os.environ.get(sys.argv[3], "")
print(json.dumps(seen))
"""
#: Run inside: each named path's read (``ok:<text>`` or the errno name), once at start and once more after a line on
#: stdin, which the test sends when the app has replaced and created its logins.
WATCH = r"""
import errno, json, sys
def outcome(path):
    try:
        return "ok:" + open(path).read()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
paths = json.loads(sys.argv[1])
print(json.dumps({name: outcome(path) for name, path in paths.items()}), flush=True)
sys.stdin.readline()
print(json.dumps({name: outcome(path) for name, path in paths.items()}), flush=True)
"""
#: Run inside by an agent: turn each named directory or file of its tree into a link to the path given with it.
PLANT = r"""
import json, os, shutil, sys
for name, target in json.loads(sys.argv[1]).items():
    shutil.rmtree(name) if os.path.isdir(name) and not os.path.islink(name) else os.remove(name)
    os.symlink(target, name)
print("planted")
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return path


def _named(paths: dict[str, Path]) -> str:
    return json.dumps({name: str(path) for name, path in paths.items()})


def _launch(isolation: BwrapIsolation, confinement: Confinement, env: dict[str, str], argv: list[str], work: Path):
    command, child_env = isolation.launch(
        [sys.executable, "-I", "-S", "-c", *argv],
        confinement.policy(chdir=work),
        {**env, MARK: "mark-1"},
        marks=(MARK,),
    )
    assert MARK not in child_env
    done = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "SECRET" not in done.stdout + done.stderr
    return done.stdout


def _run(
    isolation: BwrapIsolation,
    confinement: Confinement,
    env: dict[str, str],
    paths: dict[str, Path],
    work: Path,
    writes: dict[str, Path] | None = None,
):
    writes = {"write": work / "out"} if writes is None else writes
    return json.loads(_launch(isolation, confinement, env, [CHECK, _named(paths), _named(writes), MARK], work))


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
    masked = credential_paths(env, allowed=(*readable, work))

    seen = _run(isolation, Confinement(writable=(work,), readable=readable, masked=masked), env, paths, work)

    assert seen == {
        "auth": "ENOENT",
        "aws": "ENOENT",
        "hf_token": "ENOENT",
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
        "real_secret": "ENOENT",
        "link_secret": "ENOENT",
        "link_visible": "ok:shown",
    }


def test_links_an_agent_leaves_in_its_tree_reach_nothing_at_the_next_launch(tmp_path):
    """The first launch's agent makes its ``PATH`` entry and its re-pointed config directory links to the app's
    ``~/.ssh``; the next launch's confinement, derived the same way, neither reads the key nor writes beside it."""
    isolation = BwrapIsolation(usable_bwrap())
    home, work = tmp_path / "home", tmp_path / "work"
    key = _write(home / ".ssh/id_rsa", "SECRET-SSH-KEY")
    keys = _write(home / ".ssh/authorized_keys", "app-key\n")
    for directory in (work / "bin", work / ".claude"):
        directory.mkdir(parents=True)
    env = {"HOME": str(home), "PATH": f"{work / 'bin'}:/usr/bin:/bin"}
    writable = (work, work / ".claude")

    def confinement() -> Confinement:
        readable = toolchain_paths(env, sys.executable, writable=writable)
        allowed = (*readable, *writable)
        masked = credential_paths(env, allowed=allowed, writable=writable)
        return Confinement(writable=writable, readable=readable, masked=masked)

    planted = {str(work / "bin"): str(home / ".ssh"), str(work / ".claude"): str(home / ".ssh")}
    assert _launch(isolation, confinement(), env, [PLANT, json.dumps(planted)], work).strip() == "planted"
    assert (work / "bin").is_symlink() and (work / ".claude").is_symlink()

    seen = _run(
        isolation,
        confinement(),
        env,
        {"key": key, "via_path": work / "bin/id_rsa", "via_config": work / ".claude/id_rsa"},
        work,
        {"keys": keys, "keys_via_config": work / ".claude/authorized_keys", "write": work / "out"},
    )

    assert seen == {
        "key": "ENOENT",
        "via_path": "ENOENT",
        "via_config": "ENOENT",
        "tool": None,
        "keys": "ENOENT",
        "keys_via_config": "ENOENT",
        "write": "ok:written",
        "mark": "mark-1",
    }
    assert keys.read_text(encoding="ascii") == "app-key\n"


@pytest.mark.parametrize("through", ["writable", "given"])
def test_a_credential_an_agent_links_to_from_its_tree_stays_masked_at_the_next_launch(tmp_path, through):
    """The first launch's agent turns its re-pointed config directory (a writable path below its tree) or the prompt
    it was handed (a given path) into a link to Hugging Face's token, in the model cache it reads; the next launch's
    confinement, derived the same way, still masks the token, at its own path and through the link."""
    isolation = BwrapIsolation(usable_bwrap())
    hf, work = tmp_path / "hf", tmp_path / "work"
    token = _write(hf / "token", "SECRET-HF")
    model = _write(hf / "hub/model.txt", "weights")
    (work / ".claude").mkdir(parents=True)
    prompt = _write(work / "prompt", "the task")
    env = {"PATH": "/usr/bin:/bin", "HF_HOME": str(hf)}
    writable = (work, work / ".claude")

    def confinement() -> Confinement:
        readable = toolchain_paths(env, sys.executable, writable=writable)
        allowed = (*readable, *writable)
        masked = credential_paths(env, given=(prompt,), allowed=allowed, writable=writable)
        return Confinement(writable=writable, readable=readable, masked=masked)

    planted = work / ".claude" if through == "writable" else prompt
    paths = {"token": token, "via_link": planted, "model": model}
    assert {name: _run(isolation, confinement(), env, paths, work)[name] for name in paths} == {
        "token": "ENOENT",
        "via_link": "EISDIR" if through == "writable" else "ok:the task",
        "model": "ok:weights",
    }
    assert _launch(isolation, confinement(), env, [PLANT, _named({str(planted): token})], work).strip() == "planted"
    assert planted.is_symlink()

    seen = _run(isolation, confinement(), env, paths, work)

    assert {name: seen[name] for name in paths} == {"token": "ENOENT", "via_link": "ENOENT", "model": "ok:weights"}


@pytest.mark.parametrize("target", ["root", "parent", "link_holder"])
def test_a_writable_path_an_agent_re_points_through_another_never_takes_its_tree_away(tmp_path, target):
    """``into-work/sub`` is not below the worktree as given (``via/work``), but both app links lead to ``work``, so the
    agent decides what ``sub`` is. The first launch's agent points it at ``/``, at the directory holding ``work``, or
    at ``via``, the directory holding the app's link to ``work``: each has ``work`` reached through it. It also turns
    the prompt it was handed into a link to Hugging Face's token, beside the ``python3`` it left on its ``PATH``. The
    next derivation still takes ``work`` for a tree the agent wrote: it runs no program of it, binds nothing the agent
    chose, and the token stays masked. Pointed at ``via``, the two paths are reached through each other's trees, and
    neither is bound: the agent can withhold its own tree, never get another."""
    isolation = BwrapIsolation(usable_bwrap())
    hf, work, ran = tmp_path / "hf", tmp_path / "work", tmp_path / "ran"
    token = _write(hf / "token", "SECRET-HF")
    model = _write(hf / "hub/model.txt", "weights")
    python = _write(work / "bin/python3", f"#!/bin/sh\nid > {ran}\n")
    python.chmod(0o755)
    (work / "sub").mkdir()
    prompt = _write(work / "prompt", "the task")
    (tmp_path / "via").mkdir()
    (tmp_path / "via/work").symlink_to(work)
    (tmp_path / "into-work").symlink_to(work)
    env = {"PATH": f"{work / 'bin'}:/usr/bin:/bin", "HF_HOME": str(hf)}
    writable = (tmp_path / "via/work", tmp_path / "into-work/sub")

    def confinement() -> Confinement:
        readable = toolchain_paths(env, sys.executable, writable=writable)
        allowed = (*readable, *writable)
        masked = credential_paths(env, given=(prompt,), allowed=allowed, writable=writable)
        return Confinement(writable=writable, readable=readable, masked=masked)

    chosen = {"root": Path("/"), "parent": tmp_path, "link_holder": tmp_path / "via"}[target]
    planted = _named({str(work / "sub"): chosen, str(prompt): token})
    assert _launch(isolation, confinement(), env, [PLANT, planted], work).strip() == "planted"
    assert (work / "sub").is_symlink() and prompt.is_symlink()

    after = confinement()
    assert not ran.exists(), "the agent's python3 ran as the app"
    inside = Path("/") if target == "link_holder" else work
    assert after.policy(chdir=inside).writable == (() if target == "link_holder" else (str(work),))

    seen = _run(isolation, after, env, {"token": token, "via_prompt": prompt, "model": model}, inside, {})

    assert {name: seen[name] for name in ("token", "via_prompt", "model")} == {
        "token": "ENOENT",
        "via_prompt": "ENOENT",
        "model": "ok:weights",
    }


def _replace(path: Path, text: str) -> None:
    """Write ``path`` as git and the login tools do: a new file beside it, renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(path.name + ".lock")
    fresh.write_text(text, encoding="ascii")
    os.replace(fresh, path)


def test_logins_the_app_replaces_or_makes_during_a_launch_never_show_in_it(tmp_path, monkeypatch):
    """The app's home lies in a tree the child reads (a workspace), opened up around the program it runs; Hugging
    Face's home is read whole for its hub cache; a registered state file lies in a data tree it reads. While the
    sandbox runs, the app rewrites its git credentials, gh login, Claude login, Hugging Face token and state file by
    rename, and logs in anew (Hugging Face's ``stored_tokens``, a ``~/.netrc``, a second gh host file): none of them
    shows inside, before or after, while the cache, the data and the program stay readable."""
    isolation = BwrapIsolation(usable_bwrap())
    workspace, hf, data, work = (tmp_path / name for name in ("workspace", "hf", "data", "work"))
    home = workspace / "home"
    work.mkdir()
    tool = _write(home / ".local/share/tool/versions/1", "#!/bin/sh\necho tool ran\n")
    tool.chmod(0o755)
    (home / ".local/bin").mkdir(parents=True)
    (home / ".local/bin/tool").symlink_to(tool)
    logins = {
        "git": _write(home / ".git-credentials", "SECRET-GIT"),
        "gh": _write(home / ".config/gh/hosts.yml", "SECRET-GH"),
        "claude": _write(home / ".claude/.credentials.json", "SECRET-CLAUDE"),
        "hf_token": _write(hf / "token", "SECRET-HF"),
        "state": _write(data / "run/exit.json", "SECRET-STATE"),
    }
    made = {
        "hf_stored": hf / "stored_tokens",
        "netrc": home / ".netrc",
        "gh_other": home / ".config/gh/other.yml",
    }
    shown = {
        "model": _write(hf / "hub/model.txt", "weights"),
        "data": _write(data / "visible.txt", "data"),
        "workspace": _write(workspace / "readme", "readme"),
    }
    registry = PrivatePaths()
    registry.add(data / "run/exit.json", "the run's state file")
    monkeypatch.setattr(confinement_module, "PRIVATE_PATHS", registry)
    env = {"HOME": str(home), "PATH": f"{home / '.local/bin'}:/usr/bin:/bin", "HF_HOME": str(hf)}
    readable = (*toolchain_paths(env, "tool", writable=(work,)), workspace, data)
    masked = credential_paths(env, app_environ={}, allowed=(*readable, work))
    policy = Confinement(writable=(work,), readable=readable, masked=masked).policy(chdir=work)
    assert policy.problems(isolation.probe.version or (0, 0, 0)) == []
    paths = {**logins, **made, **shown}
    command, child_env = isolation.launch([sys.executable, "-I", "-S", "-c", WATCH, _named(paths)], policy, env)

    with subprocess.Popen(
        command, env=child_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as child:
        assert child.stdout is not None and child.stdin is not None
        before = json.loads(child.stdout.readline() or "{}")
        for name, path in logins.items():
            _replace(path, f"NEW-SECRET-{name}")
        for name, path in made.items():
            _replace(path, f"NEW-SECRET-{name}")
        child.stdin.write("go\n")
        child.stdin.flush()
        after = json.loads(child.stdout.readline() or "{}")
        _out, err = child.communicate(timeout=60)
    assert child.returncode == 0, err

    hidden = dict.fromkeys((*logins, *made), "ENOENT")
    expected = {**hidden, "model": "ok:weights", "data": "ok:data", "workspace": "ok:readme"}
    assert before == expected
    assert after == expected
    assert all(
        path.read_text(encoding="ascii").startswith("NEW-SECRET-") for path in (*logins.values(), *made.values())
    )
