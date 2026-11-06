# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/confinement``: the masks derived from configuration, the toolchain a child reads, and the bubblewrap
policy a confinement becomes (no sandbox is made here: ``test_confined_launch.py`` launches one)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom_common.bwrap import SYSTEM_ROOTS, BwrapPolicyError, BwrapProbe, Scratch, interpreter_roots

from hyperloom.common import confinement
from hyperloom.common.confinement import (
    PRIVATE_PATHS,
    Confinement,
    PrivatePaths,
    credential_paths,
    interpreter_search_path,
    toolchain_paths,
)

PROBE = BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """An app home with tool logins, a venv and a scratch tree holding a worktree four levels down."""
    root = tmp_path / "home"
    for path in (".aws/credentials", ".claude/.credentials.json", ".ssh/id_ed25519", ".venv/bin/python"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("x", encoding="ascii")
    (root / "scratch" / "a" / "b" / "wt").mkdir(parents=True)
    (root / "scratch" / "a" / "notes.txt").write_text("x", encoding="ascii")
    (root / "scratch" / "other").mkdir()
    return root


@pytest.fixture(autouse=True)
def no_passwd_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """This process has no passwd entry, as the module under test sees it (pytest's own lookups are left alone), so
    the masks below come only from what each test configures."""

    def unknown(uid: int):
        raise KeyError(uid)

    monkeypatch.setattr(confinement, "pwd", SimpleNamespace(getpwuid=unknown))


@pytest.fixture
def no_child_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child's interpreter adds nothing to the toolchain (it is asked for real in the tests below that need it)."""
    monkeypatch.setattr(confinement, "interpreter_search_path", lambda _environ, **_kwargs: ())


def _masks(environ: dict[str, str], **kwargs) -> set[Path]:
    """The masks from ``environ`` alone: this process's own environment is ``app_environ`` only where a test says."""
    return set(credential_paths(environ, **{"app_environ": {}, **kwargs}))


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return path


def _absent_inside(path: Path, policy) -> bool:
    """Whether ``path`` does not exist inside ``policy``'s sandbox, or is masked there."""
    real = os.path.realpath(path)
    binds = (*policy.read_only, *policy.writable)
    shown = [bind for bind in binds if real == bind or real.startswith(bind.rstrip("/") + "/")]
    masked = [mask for mask in policy.masked if real == mask or real.startswith(mask + "/")]
    return not shown or bool(masked)


# ---- credential locations ----


def test_the_apps_home_is_masked_whole_when_nothing_allowed_is_in_it(home):
    assert _masks({"HOME": str(home)}) == {home}


@pytest.mark.parametrize("written", [False, True], ids=["read", "written"])
def test_an_allowed_path_below_the_home_opens_up_only_its_ancestors(home, written):
    """Also when the worktree is the tree the agent writes: a tree is not reached through itself, so it still counts."""
    worktree = home / "scratch" / "a" / "b" / "wt"
    writable = [worktree] if written else []
    found = _masks({"HOME": str(home)}, allowed=[worktree, home / ".venv"], writable=writable)
    assert found == {
        home / ".aws",
        home / ".claude",
        home / ".ssh",
        home / "scratch" / "other",
        home / "scratch/a/notes.txt",
    }
    assert not any(worktree == mask or mask in worktree.parents for mask in found)


def test_past_the_depth_limit_a_directory_that_leads_to_an_allowed_path_is_left_unmasked(home):
    deep = home / "scratch" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (home / "scratch" / "a" / "b" / "c" / "d" / "sibling").mkdir()
    found = _masks({"HOME": str(home)}, allowed=[deep])
    assert home / "scratch" / "a" / "b" / "wt" in found, "depth 4 is still opened up"
    assert home / "scratch" / "a" / "b" / "c" / "d" / "sibling" not in found, "depth 5 is not"
    assert not any(mask == deep or mask in deep.parents for mask in found)


def test_the_passwd_home_is_used_without_HOME_and_masked_beside_another(monkeypatch, home, tmp_path):
    monkeypatch.setattr(confinement.pwd, "getpwuid", lambda _uid: type("E", (), {"pw_dir": str(home)})())
    assert _masks({}) == {home}
    other, child = tmp_path / "other-home", tmp_path / "child-home"
    other.mkdir()
    child.mkdir()
    found = _masks({"HOME": str(child)}, app_environ={"HOME": str(other)})
    assert found == {child, other, home}, "logins stay in the passwd home, whatever either HOME says"


def test_every_variable_that_names_a_credential_location_is_masked(tmp_path):
    key, kube, token, secret_dir = (tmp_path / name for name in ("signing.pem", "kubeconfig", "tok", "secrets"))
    for path in (key, kube, token):
        path.write_text("x", encoding="ascii")
    secret_dir.mkdir()
    environ = {
        "HOME": "/nonexistent-home",
        "HYPERLOOM_BUILD_SIGNING_KEY_FILE": str(key),
        "KUBECONFIG": f"{kube}:{tmp_path / 'missing'}:relative/path",
        "MY_SERVICE_TOKEN_PATH": str(token),
        "SOME_SECRET_DIR": str(secret_dir),
        "ANTHROPIC_API_KEY": "sk-not-a-path",
        "TOKENIZERS_PARALLELISM": "false",
        "UNRELATED_DIR": str(tmp_path),
    }
    assert _masks(environ) == {key, kube, token, secret_dir}


def test_hugging_faces_login_files_are_masked_and_its_cache_is_not(tmp_path):
    hf = tmp_path / "hf"
    (hf / "hub").mkdir(parents=True)
    (hf / "token").write_text("hf_x", encoding="ascii")
    (hf / "stored_tokens").write_text("hf_y", encoding="ascii")
    found = _masks({"HOME": "/nonexistent-home", "HF_HOME": str(hf)}, allowed=[hf])
    assert found == {hf / "token", hf / "stored_tokens"}


def test_xdg_directories_outside_the_home_and_registered_files_are_masked(tmp_path, monkeypatch):
    config, state, private = tmp_path / "config", tmp_path / "state", tmp_path / "state-dir"
    for path in (config, state, private):
        path.mkdir()
    registry = PrivatePaths()
    registry.add(private, "hosted state dir")
    registry.add(private, "registered twice")
    environ = {"HOME": "/nonexistent-home", "XDG_CONFIG_HOME": str(config), "XDG_STATE_HOME": str(state)}
    assert registry.paths() == (private,)
    assert _masks(environ) == {config, state}, "the process registry is empty"
    monkeypatch.setattr(confinement, "PRIVATE_PATHS", registry)
    assert _masks(environ) == {config, state, private}


def test_a_given_path_stays_readable_even_inside_the_masked_home(home):
    given = home / ".aws" / "credentials"
    found = _masks({"HOME": str(home), "AWS_SHARED_CREDENTIALS_FILE": str(given)}, given=[given])
    assert given not in found and home not in found
    assert {home / ".claude", home / ".ssh", home / ".venv", home / "scratch"} <= found


def test_the_root_directory_is_never_a_mask_nor_opened_up(home):
    """Opened up around ``/usr``, a home of ``/`` would mask ``/etc``, ``/opt``, ``/root`` and the rest."""
    assert _masks({"HOME": "/", "SOME_TOKEN_DIR": "/"}, allowed=[Path("/usr")]) == set()
    assert credential_paths({}, app_environ={"HOME": "/"}, allowed=[Path("/usr")]) == ()


def test_this_processs_own_credential_locations_are_masked_whatever_the_child_is_given(tmp_path, monkeypatch):
    """The child's home is re-pointed into its work tree and its environment carries no credential variable; the
    app's home and ``KUBECONFIG`` inside a readable tree are still masked, and the child's own home is not."""
    home, tools, work = tmp_path / "home", tmp_path / "tools", tmp_path / "work"
    kube = _write(tools / "kube" / "config")
    _write(home / ".ssh" / "id_ed25519")
    (work / ".home").mkdir(parents=True)
    _write(tools / "bin" / "tool")
    child = {"HOME": str(work / ".home"), "PATH": str(tools / "bin")}
    app = {"HOME": str(home), "KUBECONFIG": str(kube)}

    found = credential_paths(child, app_environ=app, allowed=[tools, work], writable=[work])

    assert {kube, home} <= set(found)
    assert work / ".home" not in found, "the child's own home, in the tree it writes"
    policy = Confinement(writable=(work,), readable=(tools,), masked=found).policy()
    assert policy.masked == (str(kube),)
    monkeypatch.setenv("KUBECONFIG", str(kube))
    assert kube in credential_paths(child, allowed=[tools]), "this process's environment by default"


def test_the_process_registry_starts_empty():
    assert isinstance(PRIVATE_PATHS, PrivatePaths)


# ---- the toolchain ----


def test_the_toolchain_is_the_system_roots_this_interpreter_the_search_paths_and_the_program(tmp_path, monkeypatch):
    bin_dir = tmp_path / "layer" / "bin"
    claude = _write(bin_dir / "claude", "#!/bin/sh\n")
    claude.chmod(0o755)
    ca = _write(tmp_path / "ca.pem")
    model = tmp_path / "models" / "m"
    model.mkdir(parents=True)
    monkeypatch.setattr(confinement, "interpreter_search_path", lambda _environ, **_kwargs: (Path("/from/sys/path"),))
    environ = {
        "PATH": f"{bin_dir}:/usr/bin:relative",
        "LD_LIBRARY_PATH": "/opt/accel/lib",
        "SSL_CERT_FILE": str(ca),
        "HF_HUB_CACHE": str(tmp_path),
        "MODEL_PATH": str(model),
    }

    found = toolchain_paths(environ, "claude", writable=())

    assert {Path(root) for root in SYSTEM_ROOTS} <= set(found)
    assert {Path(root) for root in interpreter_roots()} <= set(found)
    assert {Path("/from/sys/path"), bin_dir, bin_dir.parent, Path("/usr/bin"), ca, tmp_path, model} <= set(found)
    assert Path("relative") not in found and Path("/") not in found
    assert Path("/proc") not in found, "the sandbox mounts its own /proc; a bind of it is refused"
    assert Path("/sys") not in found, "/sys is the policy's sys_paths, which a compile narrows"


def test_a_native_installer_binds_only_the_programs_directories_never_all_of_local(tmp_path, no_child_python):
    """``~/.local/bin/claude -> ~/.local/share/claude/versions/V``: the directory and parent of both would make all of
    ``~/.local`` readable, where other tools keep their logins, and mask nothing under it."""
    home = tmp_path / "home"
    version = _write(home / ".local/share/claude/versions/2.1.319", "#!/bin/sh\n")
    version.chmod(0o755)
    (home / ".local/bin").mkdir(parents=True)
    (home / ".local/bin/claude").symlink_to(version)
    logins = [
        _write(home / ".local/share/opencode/auth.json"),
        _write(home / ".local/share/uv/credentials"),
        _write(home / ".local/state/gh/hosts.yml"),
        _write(home / ".aws/credentials"),
    ]
    environ = {"HOME": str(home), "PATH": f"{home / '.local/bin'}:/usr/bin"}

    found = toolchain_paths(environ, "claude", writable=())

    assert {home / ".local/bin", home / ".local/share/claude/versions"} <= set(found)
    assert not {home, home / ".local", home / ".local/share", home / ".local/share/claude"} & set(found)
    policy = Confinement(readable=found, masked=credential_paths(environ, app_environ={}, allowed=found)).policy()
    assert [login for login in logins if not _absent_inside(login, policy)] == []


def test_nothing_in_a_writable_tree_is_made_read_only(tmp_path, monkeypatch):
    """The writable trees already show what lies in them; a read-only bind there would take the child's own files
    from it, and on bubblewrap before 0.12 refuse the launch."""
    work = tmp_path / "work"
    for path in (work / "bin", work / "src", work / "py"):
        path.mkdir(parents=True)
    monkeypatch.setattr(
        confinement, "interpreter_search_path", lambda _environ, **_kwargs: (work / "src", Path("/from/sys/path"))
    )
    environ = {"PATH": f"{work / 'bin'}:/usr/bin", "PYTHONPATH": str(work / "py"), "HOME": str(tmp_path / "home")}

    found = toolchain_paths(environ, "sh", writable=(work,))

    assert {Path("/usr/bin"), Path("/from/sys/path")} <= set(found)
    assert [path for path in found if path == work or work in path.parents] == []
    assert Confinement(writable=(work,), readable=found).policy().problems((0, 9, 0)) == []


def test_a_program_directly_in_a_credential_location_binds_only_itself(tmp_path, no_child_python):
    home = tmp_path / "home"
    tool = _write(home / "tool", "#!/bin/sh\n")
    tool.chmod(0o755)
    found = toolchain_paths({"HOME": str(home), "PATH": "/usr/bin"}, str(tool), writable=())
    assert tool in found and home not in found and home.parent not in found


def test_the_resolvers_target_is_readable_when_it_lies_outside_etc(tmp_path, monkeypatch, no_child_python):
    stub = _write(tmp_path / "run" / "stub-resolv.conf")
    link = tmp_path / "etc" / "resolv.conf"
    link.parent.mkdir()
    link.symlink_to(stub)
    monkeypatch.setattr(confinement, "RESOLV_CONF", link)
    assert stub in toolchain_paths({"PATH": "/usr/bin"}, "sh", writable=())
    monkeypatch.setattr(confinement, "RESOLV_CONF", stub)
    assert stub not in toolchain_paths({"PATH": "/usr/bin"}, "sh", writable=()), (
        "a plain file in /etc is under the /etc bind"
    )


def _venv(root: Path) -> Path:
    """A virtual environment at ``root`` on this interpreter's base installation; returns its site directory."""
    base = Path(os.path.realpath(sys.executable))
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text(f"home = {base.parent}\ninclude-system-site-packages = false\n", encoding="ascii")
    (root / "bin" / "python3").symlink_to(base)
    site = root / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    site.mkdir(parents=True)
    return site


def _fake_python(directory: Path, answer: str, seen: Path) -> Path:
    """A ``python3`` in ``directory`` that records its arguments and environment in ``seen`` and prints ``answer``."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "python3"
    record = f"\"import json, os, sys; json.dump([sys.argv[1:], dict(os.environ)], open({str(seen)!r}, 'w'))\""
    script.write_text(f"#!/bin/sh\n{sys.executable} -I -S -c {record} \"$@\"\necho '{answer}'\n", encoding="ascii")
    script.chmod(0o755)
    return script


def test_the_childs_interpreter_reports_its_venvs_site_the_pth_directories_and_editable_trees_running_none(tmp_path):
    """A setuptools editable install names its source tree only in its finder's MAPPING, which no import path entry
    has; its .pth file imports that finder. Both are read as data: nothing in them, nor a sitecustomize, runs."""
    ran = tmp_path / "ran"
    site = _venv(tmp_path / "venv")
    extra, other = tmp_path / "extra", tmp_path / "other"
    extra.mkdir()
    other.mkdir()
    (site / "__editable__.demo-1.0.pth").write_text("import __editable___demo_1_0_finder\n", encoding="ascii")
    (site / "__editable___demo_1_0_finder.py").write_text(
        f"open({str(ran)!r}, 'a').write('finder ')\nMAPPING: dict[str, str] = {{'demo': '/src/demo/demo'}}\n",
        encoding="ascii",
    )
    (site / "paths.pth").write_text(
        f"# a comment\n{extra}\n../../../../other\n{tmp_path / 'missing'}\n", encoding="ascii"
    )
    pythonpath = tmp_path / "py"
    pythonpath.mkdir()
    (pythonpath / "sitecustomize.py").write_text(f"open({str(ran)!r}, 'a').write('site ')\n", encoding="ascii")
    environ = {"PATH": str(tmp_path / "venv" / "bin"), "PYTHONPATH": str(pythonpath)}

    found = interpreter_search_path(environ, writable=())

    assert {site, extra, other, Path("/src/demo/demo")} <= set(found)
    assert tmp_path / "missing" not in found and pythonpath not in found
    assert not ran.exists(), ran.read_text(encoding="ascii")
    assert interpreter_search_path({"PATH": str(tmp_path / "nowhere")}, writable=()) == ()


def test_an_interpreter_in_a_writable_tree_is_never_run_and_its_answer_never_bound(tmp_path):
    """A ``python3`` the agent could have written runs as the app and names what the agent may read."""
    work, secret = tmp_path / "work", tmp_path / "home" / ".ssh"
    secret.mkdir(parents=True)
    planted = _fake_python(work / "bin", json.dumps({"path": [str(secret)], "site": []}), tmp_path / "planted-ran")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "python3").symlink_to(planted)
    honest = _fake_python(tmp_path / "usr" / "bin", json.dumps({"path": ["/honest"], "site": []}), tmp_path / "seen")

    found = interpreter_search_path({"PATH": f"{work / 'bin'}:{linked}:{honest.parent}"}, writable=(work,))

    assert found == (Path("/honest"),)
    assert not (tmp_path / "planted-ran").exists()
    assert interpreter_search_path({"PATH": f"{work / 'bin'}:{linked}"}, writable=(work,)) == ()
    assert not (tmp_path / "planted-ran").exists()


def test_an_interpreter_reached_through_a_writable_tree_is_never_run_even_where_it_points_outside(tmp_path):
    """``work/bin/python3 -> /usr/bin/python3`` beside a ``work/pyvenv.cfg`` whose ``home`` holds a planted standard
    library: run by that path, a distribution's interpreter imports the agent's ``encodings`` as the app."""
    work, seen = tmp_path / "work", tmp_path / "seen"
    honest = _fake_python(tmp_path / "usr" / "bin", json.dumps({"path": ["/honest"], "site": []}), seen)
    (work / "bin").mkdir(parents=True)
    (work / "bin" / "python3").symlink_to(honest)
    (work / "linked-bin").symlink_to(honest.parent)

    assert interpreter_search_path({"PATH": f"{work / 'bin'}:{work / 'linked-bin'}"}, writable=(work,)) == ()
    assert not seen.exists()
    found = interpreter_search_path({"PATH": f"{work / 'bin'}:{honest.parent}"}, writable=(work,))
    assert found == (Path("/honest"),)


def test_the_interpreter_runs_isolated_with_only_the_library_path_outside_the_writable_trees(tmp_path):
    work, lib, elsewhere = tmp_path / "work", tmp_path / "lib", tmp_path / "elsewhere"
    (work / "lib").mkdir(parents=True)
    lib.mkdir()
    elsewhere.mkdir()
    (work / "linked-lib").symlink_to(elsewhere)
    seen = tmp_path / "seen"
    python = _fake_python(tmp_path / "bin", json.dumps({"path": [], "site": []}), seen)
    environ = {
        "PATH": str(python.parent),
        "LD_LIBRARY_PATH": f"{work / 'lib'}:{work / 'linked-lib'}:{lib}",
        "LD_PRELOAD": str(work / "lib" / "x.so"),
        "PYTHONPATH": str(work),
        "HOME": str(work),
        "ANTHROPIC_API_KEY": "sk-not-for-the-probe",
    }

    interpreter_search_path(environ, writable=(work,))

    argv, env = json.loads(seen.read_text(encoding="ascii"))
    assert argv[:3] == ["-I", "-S", "-c"]
    assert {name: value for name, value in env.items() if name not in ("PWD", "SHLVL", "_", "LC_CTYPE")} == {
        "LD_LIBRARY_PATH": str(lib)
    }


def test_a_site_directory_in_a_writable_tree_is_not_read(tmp_path):
    work, secret, shown = tmp_path / "work", tmp_path / "secret", tmp_path / "shown"
    for path in (work / "site", tmp_path / "site", tmp_path / "planted-site", secret, shown):
        path.mkdir(parents=True)
    (work / "site" / "planted.pth").write_text(f"{secret}\n", encoding="ascii")
    (tmp_path / "planted-site" / "planted.pth").write_text(f"{secret}\n", encoding="ascii")
    (work / "linked-site").symlink_to(tmp_path / "planted-site")
    (tmp_path / "site" / "image.pth").write_text(f"{shown}\n", encoding="ascii")
    sites = [str(work / "site"), str(work / "linked-site"), str(tmp_path / "site")]
    answer = json.dumps({"path": [], "site": sites})
    python = _fake_python(tmp_path / "bin", answer, tmp_path / "seen")

    found = interpreter_search_path({"PATH": str(python.parent)}, writable=(work,))

    assert shown in found and secret not in found


@pytest.mark.parametrize("answer", ["not json", "[]", '{"path": "/x", "site": []}', '{"path": []}'])
def test_an_interpreter_that_does_not_answer_gives_no_path(tmp_path, answer):
    python = _fake_python(tmp_path / f"bin-{len(answer)}", answer, tmp_path / "seen")
    assert interpreter_search_path({"PATH": str(python.parent)}, writable=()) == ()


# ---- the policy ----


def test_a_policy_drops_the_missing_binds_once_and_passes_the_rest_through(tmp_path):
    work, ca, kfd = tmp_path / "wt", _write(tmp_path / "ca.pem"), Path("/dev/kfd")
    work.mkdir()
    scratch = Scratch(str(tmp_path / "tmp"), str(tmp_path / "shm"))
    policy = Confinement(
        writable=(work, tmp_path / "missing"),
        readable=(work, ca, tmp_path / "gone"),
        masked=(tmp_path / "gone",),
        devices=(kfd,),
        app_only=(tmp_path / "state",),
        sys_paths=(Path("/sys/devices/system/cpu"),),
        own_network=True,
    ).policy(scratch=scratch, chdir=work)
    assert (policy.writable, policy.read_only, policy.masked) == ((str(work),), (str(ca),), ())
    assert (policy.devices, policy.app_only, policy.sys_paths) == (
        ("/dev/kfd",),
        (str(tmp_path / "state"),),
        ("/sys/devices/system/cpu",),
    )
    assert (policy.tmp, policy.shm, policy.chdir, policy.own_network) == (scratch.tmp, scratch.shm, str(work), True)


def test_a_linked_path_is_bound_at_its_target_and_is_the_same_link_inside(tmp_path):
    real = tmp_path / "usr" / "bin"
    real.mkdir(parents=True)
    link = tmp_path / "bin"
    link.symlink_to(real)
    policy = Confinement(readable=(tmp_path / "usr", link)).policy()
    assert policy.read_only == (str(tmp_path / "usr"), str(link)), "usr/bin is under usr, already shown"
    argv = policy.argv(PROBE, ["true"])
    assert argv[argv.index(str(link)) - 2 : argv.index(str(link)) + 1] == ["--symlink", str(real), str(link)]


def test_a_read_only_path_below_a_read_only_bind_is_left_out_and_a_writable_one_also_readable_kept_once(tmp_path):
    (tmp_path / "ro" / "deeper").mkdir(parents=True)
    (tmp_path / "rw").mkdir()
    policy = Confinement(
        writable=(tmp_path / "rw",),
        readable=(tmp_path / "ro", tmp_path / "ro" / "deeper", tmp_path / "rw"),
    ).policy()
    assert policy.read_only == (str(tmp_path / "ro"),)
    assert policy.writable == (str(tmp_path / "rw"),)


# ---- nothing is resolved through what the agent writes ----


def _planted(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An app home with an SSH key, and the agent's work tree: what it leaves for the next launch is up to each test."""
    home, work = tmp_path / "home", tmp_path / "work"
    key = _write(home / ".ssh" / "id_rsa", "SECRET-SSH-KEY")
    _write(home / ".ssh" / "authorized_keys")
    work.mkdir()
    return home, work, key


def _shows(policy, path: Path) -> list[str]:
    """The binds of ``policy`` that show ``path`` (resolved) at its own place."""
    real = os.path.realpath(path)
    return [bind for bind in (*policy.read_only, *policy.writable) if real == bind or real.startswith(bind + "/")]


def test_a_path_entry_the_agent_turned_into_a_link_is_never_bound_at_its_target(tmp_path, no_child_python):
    home, work, key = _planted(tmp_path)
    (work / "bin").symlink_to(home / ".ssh")
    environ = {"HOME": str(home), "PATH": f"{work / 'bin'}:/usr/bin"}

    found = toolchain_paths(environ, "sh", writable=(work,))

    assert [path for path in found if os.path.realpath(path).startswith(str(home))] == []
    masked = credential_paths(environ, app_environ={}, allowed=(*found, work), writable=(work,))
    policy = Confinement(writable=(work,), readable=found, masked=masked).policy()
    assert _shows(policy, key) == [] and _absent_inside(key, policy)


def test_a_writable_path_below_another_is_never_bound_at_a_target_the_agent_chose(tmp_path):
    """The re-pointed config directory, home and caches lie in the work tree; the agent can make one a link."""
    home, work, key = _planted(tmp_path)
    (work / ".claude").symlink_to(home / ".ssh")
    (work / ".cache").mkdir()

    policy = Confinement(writable=(work, work / ".claude", work / ".cache")).policy()

    assert (policy.writable, policy.read_only) == ((str(work),), ())
    assert _shows(policy, key) == [] and _shows(policy, home / ".ssh" / "authorized_keys") == []


def test_a_nested_writable_path_the_agent_re_points_never_becomes_a_tree_the_agent_writes(tmp_path, no_child_python):
    """Else the agent could point its config directory at the toolchain and have its next launch refused."""
    _home, work, _key = _planted(tmp_path)
    tools = tmp_path / "tools"
    (tools / "bin").mkdir(parents=True)
    (work / ".claude").symlink_to(tools)

    policy = Confinement(writable=(work, work / ".claude"), readable=(tools / "bin",)).policy()

    assert (policy.writable, policy.read_only) == ((str(work),), (str(tools / "bin"),))
    environ = {"PATH": f"{tools / 'bin'}:/usr/bin"}
    assert tools / "bin" in toolchain_paths(environ, "sh", writable=(work, work / ".claude"))


def test_a_writable_path_reached_through_another_never_adds_where_it_points_to_the_trees(tmp_path, no_child_python):
    """``into-work/sub`` is not below ``work`` as given, but ``into-work`` links to ``work``, so the agent decides what
    ``sub`` is. Pointed at the toolchain, it is left out, and the toolchain is not taken for a tree the agent writes:
    else its next launch would be refused, and a credential location there left unmasked as the agent's own."""
    _home, work, _key = _planted(tmp_path)
    tools = tmp_path / "tools"
    (tools / "bin").mkdir(parents=True)
    kube = _write(tools / "kube" / "config")
    (tmp_path / "into-work").symlink_to(work)
    (work / "sub").symlink_to(tools)
    writable = (work, tmp_path / "into-work" / "sub")

    policy = Confinement(writable=writable, readable=(tools / "bin",)).policy()

    assert (policy.writable, policy.read_only) == ((str(work),), (str(tools / "bin"),))
    environ = {"PATH": f"{tools / 'bin'}:/usr/bin", "KUBECONFIG": str(kube)}
    assert tools / "bin" in toolchain_paths(environ, "sh", writable=writable)
    assert kube in credential_paths(environ, app_environ={}, allowed=(tools, work), writable=writable)


@pytest.mark.parametrize("target", ["root", "parent", "link_holder"])
def test_a_writable_path_re_pointed_through_another_never_takes_a_tree_from_the_agent(
    tmp_path, no_child_python, target
):
    """``into-work/sub`` leads into ``work`` (as ``via/work``, the worktree as given, does), so the agent decides what
    it is. Pointed at ``/``, at ``work``'s parent, or at ``via`` (which holds the app's link), it has ``work`` reached
    through it; yet ``work`` stays a tree: nothing reached through it is bound or read. Pointed at ``via``, the two
    are reached through each other's trees: neither is bound, both stay trees, and a credential location of the
    child's in ``via`` is masked, not taken for one the agent writes."""
    _home, work, _key = _planted(tmp_path)
    (work / "bin").mkdir()
    (tmp_path / "via").mkdir()
    (tmp_path / "via" / "work").symlink_to(work)
    (tmp_path / "into-work").symlink_to(work)
    (work / "sub").symlink_to({"root": Path("/"), "parent": tmp_path, "link_holder": tmp_path / "via"}[target])
    kube = _write(tmp_path / "via" / "kubeconfig")
    writable = (tmp_path / "via" / "work", tmp_path / "into-work" / "sub")

    assert Confinement(writable=writable).policy().writable == (() if target == "link_holder" else (str(work),))
    with pytest.raises(BwrapPolicyError):
        Confinement(writable=writable, readable=(work / "bin",)).policy()
    environ = {"PATH": f"{work / 'bin'}:/usr/bin", "KUBECONFIG": str(kube)}
    assert work / "bin" not in toolchain_paths(environ, "sh", writable=writable)
    assert kube in credential_paths(environ, app_environ={}, writable=writable)


@pytest.mark.parametrize("planted", [False, True], ids=["directory", "link"])
def test_a_readable_path_reached_through_a_tree_the_agent_writes_is_refused(tmp_path, planted):
    """Its target would be the agent's choice, and a read-only bind over a tree the agent writes cannot hold."""
    home, work, _key = _planted(tmp_path)
    if planted:
        (work / "ro").symlink_to(home / ".ssh")
    else:
        (work / "ro").mkdir()
    with pytest.raises(BwrapPolicyError) as refused:
        Confinement(writable=(work,), readable=(Path("/usr"), work / "ro")).policy()
    assert refused.value.problems == (
        f"{work / 'ro'} is reached through a tree the agent writes, which decides what it is",
    )


def test_a_path_reached_through_an_app_link_into_the_agents_tree_or_its_scratch_is_not_bound(tmp_path):
    """Also as a bind names it: ``link/../tmp/left`` resolves beside ``link``'s target, and bubblewrap is handed it
    normalized, ``tmp/left``, inside the scratch the agent writes."""
    home, work, _key = _planted(tmp_path)
    (work / "sub").symlink_to(home / ".ssh")
    (tmp_path / "into-work").symlink_to(work)
    scratch = Scratch(str(tmp_path / "tmp"), str(tmp_path / "shm"))
    (tmp_path / "tmp" / "left").mkdir(parents=True)
    (tmp_path / "shm").mkdir()
    (tmp_path / "image" / "tmp" / "left").mkdir(parents=True)
    (tmp_path / "image" / "inner").mkdir()
    (tmp_path / "to-image").symlink_to(tmp_path / "image" / "inner")
    beside = Path(f"{tmp_path / 'to-image'}/../tmp/left")

    policy = Confinement(writable=(work, tmp_path / "into-work" / "sub")).policy(scratch=scratch)

    assert (policy.writable, policy.read_only) == ((str(work),), ())
    for readable in (tmp_path / "into-work" / "sub", tmp_path / "tmp" / "left", beside):
        with pytest.raises(BwrapPolicyError):
            Confinement(writable=(work,), readable=(readable,)).policy(scratch=scratch)
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    assert Confinement(writable=(work,), readable=(tmp_path / "loop" / "x",)).policy().read_only == (), (
        "a link loop ends the walk where the kernel's ends (ELOOP), and nothing is bound"
    )
    through_work = Path(f"{tmp_path / 'into-work'}/../work")
    assert Confinement(writable=(work,), readable=(through_work,)).policy().writable == (str(work),), (
        "a path that resolves to the writable tree itself is that tree"
    )


def test_a_program_the_agent_linked_elsewhere_binds_nothing_of_where_it_points(tmp_path, monkeypatch, no_child_python):
    """By name, by absolute path, and by a relative path, which names a program relative to the working directory
    inside (this process's own one is the agent's tree here)."""
    home, work, key = _planted(tmp_path)
    (work / "bin").mkdir()
    (work / "bin" / "claude").symlink_to(key)
    environ = {"HOME": str(home), "PATH": f"{work / 'bin'}:/usr/bin"}
    monkeypatch.chdir(work)

    for argv0 in ("claude", str(work / "bin" / "claude"), "bin/claude", "./bin/claude"):
        found = toolchain_paths(environ, argv0, writable=(work,))
        assert [path for path in found if os.path.realpath(path).startswith(str(home))] == [], argv0
        assert [path for path in found if not path.is_absolute()] == [], argv0


def test_a_relative_program_path_binds_nothing_wherever_it_lies(tmp_path, monkeypatch, no_child_python):
    """Resolved against this process's working directory, a relative path names a program the child, started in
    another one, may not run; nothing is bound for it, outside the agent's tree too."""
    tools = tmp_path / "tools"
    program = _write(tools / "bin" / "tool", "#!/bin/sh\n")
    program.chmod(0o755)
    monkeypatch.chdir(tools)

    assert tools / "bin" in toolchain_paths({"PATH": "/usr/bin"}, str(program), writable=())
    found = toolchain_paths({"PATH": "/usr/bin"}, "bin/tool", writable=())
    assert [path for path in found if os.path.realpath(path).startswith(str(tools))] == []


def test_masks_are_only_paths_a_bind_shows(tmp_path):
    hf = tmp_path / "hf"
    token, elsewhere = _write(hf / "token"), _write(tmp_path / "home" / ".aws" / "credentials")
    policy = Confinement(readable=(hf,), masked=(token, elsewhere, tmp_path / "missing")).policy()
    assert policy.masked == (str(token),), "a credential no bind shows does not exist inside: nothing to mask"


def test_a_mask_reached_through_a_linked_directory_is_masked_at_both_paths(tmp_path):
    """``link/data`` with ``link -> real`` shows ``real/data`` at both paths, so its credential is masked at both."""
    secret = _write(tmp_path / "real" / "data" / "secret")
    _write(tmp_path / "real" / "data" / "visible")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    given = tmp_path / "link" / "data"
    policy = Confinement(readable=(given,), masked=(secret,)).policy()
    assert policy.read_only == (str(tmp_path / "real" / "data"), str(given))
    assert policy.masked == (str(secret), str(given / "secret"))


def test_a_mask_below_another_mask_is_left_out(tmp_path):
    inner = _write(tmp_path / "tree" / "a" / "b")
    policy = Confinement(readable=(tmp_path / "tree",), masked=(inner, tmp_path / "tree" / "a")).policy()
    assert policy.masked == (str(tmp_path / "tree" / "a"),), "a mask below a read-only mask could not be mounted"


def test_a_toolchain_policy_never_binds_proc_or_sys():
    environ = dict(os.environ)
    policy = Confinement(readable=toolchain_paths(environ, sys.executable, writable=())).policy()
    argv = policy.argv(PROBE, ["true"])
    binds = [argv[at + 2] for at, arg in enumerate(argv) if arg in ("--ro-bind-try", "--bind")]
    assert [path for path in binds if path in ("/proc", "/sys") or path.startswith("/proc/")] == ["/sys"]
    assert argv[argv.index("--proc") + 1] == "/proc"
    assert "/sys" in policy.sys_paths and "/sys" not in policy.read_only
