# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/confinement``: the masks derived from configuration, the toolchain a child reads, and the bubblewrap
policy a confinement becomes (no sandbox is made here: ``test_confined_launch.py`` launches one)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hyperloom_common.bwrap import SYSTEM_ROOTS, BwrapProbe, Scratch, interpreter_roots

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


@pytest.fixture
def no_child_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child's interpreter adds nothing to the toolchain (it is asked for real in the tests below that need it)."""
    monkeypatch.setattr(confinement, "interpreter_search_path", lambda _environ: ())


def _masks(environ: dict[str, str], **kwargs) -> set[Path]:
    return set(credential_paths(environ, **kwargs))


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


def test_an_allowed_path_below_the_home_opens_up_only_its_ancestors(home):
    worktree = home / "scratch" / "a" / "b" / "wt"
    found = _masks({"HOME": str(home)}, allowed=[worktree, home / ".venv"])
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


def test_the_passwd_home_is_used_without_HOME(monkeypatch, home):
    monkeypatch.setattr(confinement.pwd, "getpwuid", lambda _uid: type("E", (), {"pw_dir": str(home)})())
    assert _masks({}) == {home}


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


def test_xdg_directories_outside_the_home_and_registered_files_are_masked(tmp_path):
    config, state, private = tmp_path / "config", tmp_path / "state", tmp_path / "state-dir"
    for path in (config, state, private):
        path.mkdir()
    registry = PrivatePaths()
    registry.add(private, "hosted state dir")
    registry.add(private, "registered twice")
    environ = {"HOME": "/nonexistent-home", "XDG_CONFIG_HOME": str(config), "XDG_STATE_HOME": str(state)}
    assert registry.paths() == (private,)
    assert _masks(environ, registered=registry.paths()) == {config, state, private}


def test_a_given_path_stays_readable_even_inside_the_masked_home(home):
    given = home / ".aws" / "credentials"
    found = _masks({"HOME": str(home), "AWS_SHARED_CREDENTIALS_FILE": str(given)}, given=[given])
    assert given not in found and home not in found
    assert {home / ".claude", home / ".ssh", home / ".venv", home / "scratch"} <= found


def test_the_root_directory_is_never_a_mask(home):
    assert Path("/") not in _masks({"HOME": "/", "SOME_TOKEN_DIR": "/"}, allowed=[Path("/usr")])


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
    monkeypatch.setattr(confinement, "interpreter_search_path", lambda _environ: (Path("/from/sys/path"),))
    environ = {
        "PATH": f"{bin_dir}:/usr/bin:relative",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "SSL_CERT_FILE": str(ca),
        "HF_HUB_CACHE": str(tmp_path),
        "MODEL_PATH": str(model),
    }

    found = toolchain_paths(environ, "claude")

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

    found = toolchain_paths(environ, "claude")

    assert {home / ".local/bin", home / ".local/share/claude/versions"} <= set(found)
    assert not {home, home / ".local", home / ".local/share", home / ".local/share/claude"} & set(found)
    policy = Confinement(readable=found, masked=credential_paths(environ, app_home=home, allowed=found)).policy()
    assert [login for login in logins if not _absent_inside(login, policy)] == []


def test_a_program_directly_in_a_credential_location_binds_only_itself(tmp_path, no_child_python):
    home = tmp_path / "home"
    tool = _write(home / "tool", "#!/bin/sh\n")
    tool.chmod(0o755)
    found = toolchain_paths({"HOME": str(home), "PATH": "/usr/bin"}, str(tool))
    assert tool in found and home not in found and home.parent not in found


def test_the_resolvers_target_is_readable_when_it_lies_outside_etc(tmp_path, monkeypatch, no_child_python):
    stub = _write(tmp_path / "run" / "stub-resolv.conf")
    link = tmp_path / "etc" / "resolv.conf"
    link.parent.mkdir()
    link.symlink_to(stub)
    monkeypatch.setattr(confinement, "RESOLV_CONF", link)
    assert stub in toolchain_paths({"PATH": "/usr/bin"}, "sh")
    monkeypatch.setattr(confinement, "RESOLV_CONF", stub)
    assert stub not in toolchain_paths({"PATH": "/usr/bin"}, "sh"), "a plain file in /etc is under the /etc bind"


def test_the_childs_interpreter_reports_its_import_path_and_editable_source_trees(tmp_path):
    """A setuptools editable install names its source tree only in its finder's MAPPING, which sys.path lacks."""
    bin_dir, site = tmp_path / "bin", tmp_path / "site"
    bin_dir.mkdir()
    site.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    (site / "__editable___demo_finder.py").write_text("MAPPING = {'demo': '/src/demo/demo'}\n", encoding="ascii")
    (site / "sitecustomize.py").write_text("import __editable___demo_finder\n", encoding="ascii")
    environ = {"PATH": str(bin_dir), "PYTHONPATH": str(site)}

    found = interpreter_search_path(environ)

    assert site in found and Path("/src/demo/demo") in found
    assert interpreter_search_path({"PATH": str(tmp_path / "nowhere")}) == ()


def test_an_interpreter_that_does_not_answer_gives_no_path(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text("#!/bin/sh\necho not json\n", encoding="ascii")
    (bin_dir / "python3").chmod(0o755)
    assert interpreter_search_path({"PATH": str(bin_dir)}) == ()


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


def test_a_read_only_path_below_a_read_only_bind_is_left_out_and_below_a_writable_one_kept(tmp_path):
    (tmp_path / "ro" / "deeper").mkdir(parents=True)
    (tmp_path / "rw" / "ro").mkdir(parents=True)
    policy = Confinement(
        writable=(tmp_path / "rw",),
        readable=(tmp_path / "ro", tmp_path / "ro" / "deeper", tmp_path / "rw" / "ro", tmp_path / "rw"),
    ).policy()
    assert policy.read_only == (str(tmp_path / "ro"), str(tmp_path / "rw" / "ro"))
    assert policy.writable == (str(tmp_path / "rw"),)


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
    policy = Confinement(readable=toolchain_paths(environ, sys.executable)).policy()
    argv = policy.argv(PROBE, ["true"])
    binds = [argv[at + 2] for at, arg in enumerate(argv) if arg in ("--ro-bind-try", "--bind")]
    assert [path for path in binds if path in ("/proc", "/sys") or path.startswith("/proc/")] == ["/sys"]
    assert argv[argv.index("--proc") + 1] == "/proc"
    assert "/sys" in policy.sys_paths and "/sys" not in policy.read_only
