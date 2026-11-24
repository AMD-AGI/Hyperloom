# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bootstrap's refusals: which inputs end a run for good (a final exit.json) and which are left to a restart; and a
source on this machine that another account owns, which the fetch alone trusts, by its exact path."""

from __future__ import annotations

import ensurepip
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

BASH = shutil.which("bash") or "bash"
SCRIPT = Path(__file__).resolve().parent.parent / "app_bootstrap.sh"
_SHA = "0123456789abcdef0123456789abcdef01234567"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return done.stdout.strip()


@pytest.fixture
def source(tmp_path) -> Path:
    """A git repository with one commit and no lock file."""
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README").write_text("x")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "c")
    return repo


def _bootstrap(tmp_path, *, umask=0o022, **env):
    state = Path(env.get("HYPERLOOM_STATE_DIR", tmp_path / "state"))
    if not state.exists():
        state.mkdir()
    (tmp_path / "container-tmp").mkdir(exist_ok=True)
    full = {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HYPERLOOM_STATE_DIR": str(state),
        "USER_DATA_PATH": str(tmp_path / "data"),
        "TMPDIR": str(tmp_path / "container-tmp"),
    }
    full.update(env)
    done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True, env=full, timeout=120, umask=umask)
    written = state / "exit.json"
    return done.returncode, (json.loads(written.read_text()) if written.is_file() else None), done.stderr


def _expect_final(code, exit_file, stderr, message):
    assert code == 2
    assert exit_file["final"] is True and exit_file["returncode"] == 2 and message in exit_file["error"], stderr


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": "main"},
            "HYPERLOOM_SOURCE_SHA must be a full 40-character",
        ),
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": _SHA.upper()},
            "HYPERLOOM_SOURCE_SHA must be a full 40-character",
        ),
        ({"HYPERLOOM_SOURCE_SHA": _SHA}, "HYPERLOOM_SOURCE_REPO is not set"),
        ({"HYPERLOOM_SOURCE_REPO": "https://", "HYPERLOOM_SOURCE_SHA": _SHA}, "neither OWNER/NAME nor a git URL"),
        (
            {"HYPERLOOM_SOURCE_REPO": "https://host-without-a-path", "HYPERLOOM_SOURCE_SHA": _SHA},
            "neither OWNER/NAME nor a git URL",
        ),
        ({"HYPERLOOM_SOURCE_REPO": "not a repo", "HYPERLOOM_SOURCE_SHA": _SHA}, "neither OWNER/NAME nor a git URL"),
        (
            {"HYPERLOOM_SOURCE_REPO": "o/r", "HYPERLOOM_SOURCE_SHA": _SHA, "USER_DATA_PATH": "relative"},
            "USER_DATA_PATH is not an absolute path",
        ),
    ],
)
def test_an_input_the_script_cannot_work_with_ends_the_run_for_good(tmp_path, env, message):
    _expect_final(*_bootstrap(tmp_path, **env), message)


def test_a_commit_the_repository_does_not_have_ends_the_run_for_good(tmp_path, source):
    _expect_final(
        *_bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=_SHA), "has no such commit"
    )


def test_a_repository_that_cannot_be_reached_is_left_to_a_restart(tmp_path):
    code, exit_file, stderr = _bootstrap(
        tmp_path, HYPERLOOM_SOURCE_REPO=str(tmp_path / "missing"), HYPERLOOM_SOURCE_SHA=_SHA
    )
    assert code == 1 and exit_file is None and "cannot fetch the source commit" in stderr


def test_the_script_leaves_no_build_directory_behind_when_it_fails(tmp_path, source):
    _bootstrap(tmp_path, HYPERLOOM_SOURCE_REPO=str(source), HYPERLOOM_SOURCE_SHA=_SHA)
    assert list((tmp_path / "container-tmp").iterdir()) == []
    assert list((tmp_path / "data").iterdir()) == []


# -- the shared storage, which each leg may write as another account ---------------------------------------------
def test_a_user_data_path_this_account_cannot_write_ends_the_run_for_good_naming_it(tmp_path):
    data = tmp_path / 'we"ird\\data'
    data.write_text("")  # no account can create a file in it
    code, exit_file, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO="o/r",
        HYPERLOOM_SOURCE_SHA=_SHA,
        USER_DATA_PATH=str(data),
        PYTHON=sys.executable,
    )
    _expect_final(
        code, exit_file, stderr, f"USER_DATA_PATH {data} cannot be written by this account (uid {os.geteuid()})"
    )


def test_a_state_dir_this_account_cannot_write_is_said_on_stderr_only(tmp_path):
    state = tmp_path / "state-file"
    state.write_text("")
    code, exit_file, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO="o/r",
        HYPERLOOM_SOURCE_SHA=_SHA,
        HYPERLOOM_STATE_DIR=str(state),
        PYTHON=sys.executable,
    )
    assert (code, exit_file) == (2, None)
    assert f"HYPERLOOM_STATE_DIR {state} cannot be written by this account" in stderr
    assert state.read_text() == ""


def test_an_exit_json_an_earlier_leg_left_is_replaced_never_rewritten(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "exit.json").write_text("{}")
    (tmp_path / "the-old-inode").hardlink_to(state / "exit.json")
    _expect_final(*_bootstrap(tmp_path, HYPERLOOM_SOURCE_SHA=_SHA), "HYPERLOOM_SOURCE_REPO is not set")
    assert (tmp_path / "the-old-inode").read_text() == "{}"
    assert sorted(path.name for path in state.iterdir()) == ["exit.json"]


def test_the_exit_json_the_bootstrap_writes_is_the_groups_to_read_and_replace_under_an_owner_only_umask(tmp_path):
    _expect_final(*_bootstrap(tmp_path, umask=0o077, HYPERLOOM_SOURCE_SHA=_SHA), "HYPERLOOM_SOURCE_REPO is not set")
    assert stat.S_IMODE((tmp_path / "state" / "exit.json").stat().st_mode) == 0o664


def test_a_directory_put_where_exit_json_goes_is_never_written_into(tmp_path):
    state = tmp_path / "state"
    (state / "exit.json").mkdir(parents=True)
    code, exit_file, stderr = _bootstrap(tmp_path, HYPERLOOM_SOURCE_SHA=_SHA)
    assert (code, exit_file) == (2, None)
    assert f"cannot write {state}/exit.json" in stderr
    assert list((state / "exit.json").iterdir()) == []
    assert sorted(path.name for path in state.iterdir()) == ["exit.json"], "nothing of the attempt stays behind"


_STUB = """#!/bin/bash
# A stand-in for the framework Python: answers the probes and records every other call; FAKE_PROBE_FAILS makes the
# writability probe fail as a Python that does not run would, FAKE_VENV_FAILS makes `-m venv` fail after its record.
if [[ "$1" == "-c" ]]; then
  [[ "$2" != *W_OK* || -z "${FAKE_PROBE_FAILS:-}" ]] || exit 1
  case "$2" in *W_OK*) echo "${FAKE_WRITABLE:-1}" ;; *base_prefix*) echo 1 ;; *) echo "$FAKE_PREFIX" ;; esac
  exit 0
fi
echo "$*" >>"$FAKE_LOG"
[[ "$1 $2" != "-m venv" || -z "${FAKE_VENV_FAILS:-}" ]] || exit 1
if [[ " $* " == *" wheel "* ]]; then
  while [[ "$1" != "-w" ]]; do shift; done
  : >"$2/hyperloom-0-py3-none-any.whl"
fi
"""


def _framework_run(tmp_path, source, repo=None, **env):
    stub = tmp_path / "framework" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text(_STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls"
    code, exit_file, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO=repo or str(source),
        HYPERLOOM_SOURCE_SHA=_git(source, "rev-parse", "HEAD"),
        PYTHON=str(stub),
        FAKE_PREFIX=str(tmp_path / "framework"),
        FAKE_LOG=str(log),
        **env,
    )
    return code, stderr, stub, (log.read_text().splitlines() if log.exists() else [])


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_only_the_wheel_goes_into_the_framework_python_and_it_runs_the_entry(tmp_path, source):
    code, stderr, stub, calls = _framework_run(tmp_path, source)
    assert code == 0, stderr
    installs = [c for c in calls if " install " in c]
    assert len(installs) == 1, calls  # the wheel itself: no lock, no dependency
    assert "install -q --force-reinstall --no-deps --no-index" in installs[0] and installs[0].endswith(".whl")
    assert not [c for c in calls if "--require-hashes" in c or " -r " in c]
    assert [c for c in calls if " wheel " in c and "--no-deps" in c]
    assert calls[-1] == "-m hyperloom hosted run"  # exec'd with the framework Python
    assert not (tmp_path / "state" / "venv").exists()  # no environment of its own
    assert (tmp_path / "framework" / ".hyperloom-install").read_text().strip() == _git(source, "rev-parse", "HEAD")


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_the_wheel_is_built_and_installed_off_the_shared_storage_and_leaves_nothing_there(tmp_path, source):
    planted = tmp_path / "data" / ".hyperloom" / "wheels" / _git(source, "rev-parse", "HEAD")
    planted.mkdir(parents=True)
    (planted / ".complete").write_text("")
    (planted / "hyperloom-0-py3-none-any.whl").write_text("another account's")
    code, stderr, _, calls = _framework_run(tmp_path, source)
    assert code == 0, stderr
    (install,) = [c for c in calls if " install " in c]
    wheel = Path(install.split()[-1])
    assert wheel.parent.parent.parent == tmp_path / "container-tmp" and wheel.parent.parent.name.startswith(
        "hyperloom-wheel."
    )
    assert list((tmp_path / "container-tmp").iterdir()) == [], "the build directory is removed"
    assert sorted(path.name for path in (tmp_path / "data").rglob("*")) == sorted(
        [".hyperloom", "wheels", planted.name, ".complete", "hyperloom-0-py3-none-any.whl"]
    ), "nothing written to the shared storage, nothing read from it"


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_new_container_builds_the_wheel_again(tmp_path, source):
    _framework_run(tmp_path, source)
    (tmp_path / "calls").unlink()
    (tmp_path / "framework" / ".hyperloom-install").unlink()  # a new container's environment holds no install
    code, _, _, calls = _framework_run_again(tmp_path, source)
    assert code == 0 and [c for c in calls if " wheel " in c] and [c for c in calls if " install " in c]


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_restart_of_the_same_commit_installs_nothing_again(tmp_path, source):
    _framework_run(tmp_path, source)
    (tmp_path / "calls").unlink()
    code, _, _, calls = _framework_run_again(tmp_path, source)
    assert code == 0 and calls == ["-m hyperloom hosted run"]


def _framework_run_again(tmp_path, source):
    stub = tmp_path / "framework" / "bin" / "python"
    log = tmp_path / "calls"
    code, _, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO=str(source),
        HYPERLOOM_SOURCE_SHA=_git(source, "rev-parse", "HEAD"),
        PYTHON=str(stub),
        FAKE_PREFIX=str(tmp_path / "framework"),
        FAKE_LOG=str(log),
    )
    return code, stderr, stub, log.read_text().splitlines()


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_without_any_python_the_run_ends_for_good(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for tool in ("mktemp", "mv"):  # the only tools the refusal itself needs
        (empty / tool).symlink_to(shutil.which(tool))
    _expect_final(
        *_bootstrap(tmp_path, PATH=str(empty), HYPERLOOM_SOURCE_REPO="o/r", HYPERLOOM_SOURCE_SHA=_SHA),
        "no framework Python was found",
    )


# A git that records, for every fetch, the global config it runs with (a copy) and its URL, then runs the real git.
_RECORDING_GIT = """#!/bin/bash
if [[ " $* " == *" fetch "* ]]; then
  n=$(ls "$RECORD" | wc -l)
  printf '%s\\n' "${@: -2}" >"$RECORD/$n.args"
  if [[ -n "${GIT_CONFIG_GLOBAL+set}" ]]; then cp "$GIT_CONFIG_GLOBAL" "$RECORD/$n.config"; fi
elif [[ -n "${GIT_CONFIG_GLOBAL+set}" ]]; then
  echo "$*" >>"$RECORD/outside-a-fetch"
fi
exec "$REAL_GIT" "$@"
"""


def _recording_path(tmp_path: Path) -> tuple[str, Path]:
    """A PATH whose git records each fetch into the returned directory and is otherwise the git on this PATH."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "git").write_text(_RECORDING_GIT)
    (shim / "git").chmod(0o755)
    record = tmp_path / "record"
    record.mkdir()
    return f"{shim}:{_git_path()}", record


def _git_path() -> str:
    """The PATH the bootstrap runs with: the git this test finds first, then the defaults."""
    return f"{Path(shutil.which('git')).parent}:{Path(sys.executable).parent}:/usr/bin:/bin"


def _safe_directories(config: Path) -> list[str]:
    done = subprocess.run(
        ["git", "config", "--file", str(config), "--get-all", "safe.directory"], capture_output=True, text=True
    )
    return done.stdout.splitlines()


@pytest.fixture
def mirror(tmp_path, source) -> Path:
    """A bare mirror of ``source`` reached through a symlink, as a host names its mirror."""
    real = tmp_path / "mirrors" / "real" / "hyperloom.git"
    real.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", "--mirror", str(source), str(real)], check=True)
    (tmp_path / "mirrors" / "link").symlink_to(real.parent)
    return real


def _named_through_a_link_then_up(tmp_path: Path, source: Path, mirror: Path) -> tuple[Path, Path]:
    """``a/link/../hyperloom.git`` with ``a/link`` -> ``mirror``'s directory's ``inner``: physically ``mirror``, while
    dropping ``link/..`` as text gives ``a/hyperloom.git``, which is made a repository too (the decoy, returned)."""
    inner = mirror.parent / "inner"
    inner.mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "link").symlink_to(inner)
    decoy = tmp_path / "a" / "hyperloom.git"
    subprocess.run(["git", "clone", "-q", "--mirror", str(source), str(decoy)], check=True)
    return tmp_path / "a" / "link" / ".." / "hyperloom.git", decoy


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
@pytest.mark.parametrize("scheme", ["", "file://"], ids=["path", "file-url"])
@pytest.mark.parametrize("up", [False, True], ids=["through-a-link", "up-after-a-link"])
def test_a_local_source_is_trusted_for_its_fetch_alone_by_its_exact_resolved_path(tmp_path, source, mirror, scheme, up):
    path, record = _recording_path(tmp_path)
    local_tmp = tmp_path / "local-tmp"
    local_tmp.mkdir()
    named, decoy = tmp_path / "mirrors" / "link" / "hyperloom.git", None
    if up:
        named, decoy = _named_through_a_link_then_up(tmp_path, source, mirror)
    code, stderr, _, calls = _framework_run(
        tmp_path,
        source,
        repo=f"{scheme}{named}",
        PATH=path,
        REAL_GIT=shutil.which("git"),
        RECORD=str(record),
        TMPDIR=str(local_tmp),
    )
    assert code == 0, stderr
    assert calls[-1] == "-m hyperloom hosted run"
    resolved = os.path.realpath(mirror)
    fetch = sorted(record.glob("*.args"))[0]
    assert fetch.read_text().splitlines()[0] == f"{scheme}{resolved}", "the fetch names the path it trusts"
    assert _safe_directories(fetch.with_suffix(".config")) == [resolved, f"{resolved}/.git"]
    assert decoy is None or str(decoy) not in fetch.read_text(), "the directory a textual '..' names is not fetched"
    assert not (record / "outside-a-fetch").exists(), "no other git command runs with the trust"
    assert not list(local_tmp.iterdir()), "the trust is removed once the wheel is built"


def test_a_remote_source_is_fetched_with_no_config_of_ours(tmp_path):
    path, record = _recording_path(tmp_path)
    code, _, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO="ssh://example.invalid/o/r.git",
        HYPERLOOM_SOURCE_SHA=_SHA,
        PATH=path,
        REAL_GIT=shutil.which("git"),
        RECORD=str(record),
        GIT_SSH_COMMAND="false",
    )
    assert code == 1 and "cannot fetch the source commit" in stderr
    assert sorted(p.name for p in record.iterdir()) == ["0.args", "1.args"], "no trust for either fetch"
    assert (record / "0.args").read_text().splitlines()[0] == "ssh://example.invalid/o/r.git"


@pytest.mark.skipif(os.geteuid() != 0, reason="a repository of another owner needs root to chown")
@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
@pytest.mark.parametrize("bare", [True, False], ids=["bare-mirror", "checkout"])
def test_a_source_another_account_owns_is_fetched_and_no_other_repository_is_trusted(tmp_path, source, mirror, bare):
    """git builds that check the owner in upload-pack (Ubuntu's 2.34.1 and 2.43.0, 2.45.1 to 2.47) refused this fetch
    with "detected dubious ownership"; on every build the trust the fetch ran with names no other repository."""
    other, checkout = tmp_path / "other", tmp_path / "checkout"
    for clone in (other, checkout):
        subprocess.run(["git", "clone", "-q", str(source), str(clone)], check=True)
    served = mirror if bare else checkout
    for repo in (served, other):
        for root, dirs, files in os.walk(repo):
            for name in (root, *(os.path.join(root, entry) for entry in (*dirs, *files))):
                os.lchown(name, 65534, 65534)
    path, record = _recording_path(tmp_path)
    code, stderr, _, calls = _framework_run(
        tmp_path, source, repo=str(served), PATH=path, REAL_GIT=shutil.which("git"), RECORD=str(record)
    )
    assert code == 0, stderr
    assert calls[-1] == "-m hyperloom hosted run"
    config = sorted(record.glob("*.config"))[0]
    trusted = {**os.environ, "GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"}

    def rev_parse(repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(repo), "rev-parse"], capture_output=True, text=True, env=trusted)

    assert rev_parse(served).returncode == 0
    refused = rev_parse(other)
    assert refused.returncode != 0 and "dubious ownership" in refused.stderr


# -- a framework environment this account cannot write (a uid other than root, a read-only image) ------------------
# A backend in the source tree itself, so pip builds the wheel with no index: the package is `hyperloom`, whose
# `__main__` reports what the entry would see instead of running it.
_BACKEND = r"""
import base64, hashlib, os, zipfile

_FILES = {
    "hyperloom/__init__.py": "",
    "hyperloom/__main__.py": (
        "import importlib.util, json, os, sys\n"
        "spec = importlib.util.find_spec('framework_pkg')\n"
        "report = {'argv': sys.argv[1:], 'executable': sys.executable, 'prefix': sys.prefix,\n"
        "          'base_prefix': sys.base_prefix, 'framework_pkg': spec.origin if spec else None,\n"
        "          'path': os.environ['PATH'].split(os.pathsep), 'virtual_env': os.environ.get('VIRTUAL_ENV'),\n"
        "          'revision': os.environ.get('HYPERLOOM_CODE_REVISION'), 'hyperloom': __file__}\n"
        "open(os.environ['REPORT'], 'w').write(json.dumps(report))\n"
    ),
}
_DIST = "hyperloom-0.dist-info"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = "hyperloom-0-py3-none-any.whl"
    files = {
        **_FILES,
        f"{_DIST}/METADATA": "Metadata-Version: 2.1\nName: hyperloom\nVersion: 0\n",
        f"{_DIST}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    with zipfile.ZipFile(os.path.join(wheel_directory, name), "w") as wheel:
        for path, text in files.items():
            data = text.encode()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            record.append(f"{path},sha256={digest},{len(data)}")
            wheel.writestr(path, data)
        record.append(f"{_DIST}/RECORD,,")
        wheel.writestr(f"{_DIST}/RECORD", "\n".join(record) + "\n")
    return name
"""
_PYPROJECT = """[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
"""
# The framework Python as an account that cannot write it sees it: the writability probe says no, the rest is the real
# interpreter.
_READ_ONLY_FRAMEWORK = """#!/bin/bash
if [[ "$1" == "-c" && "$2" == *W_OK* ]]; then echo 0; exit 0; fi
exec "$REAL_FRAMEWORK_PY" "$@"
"""


@pytest.fixture
def package_source(tmp_path) -> Path:
    """A git repository whose one commit builds a ``hyperloom`` wheel offline."""
    repo = tmp_path / "package-source"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "backend.py").write_text(_BACKEND)
    (repo / "pyproject.toml").write_text(_PYPROJECT)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    return repo


@pytest.fixture(scope="module")
def framework_venv(tmp_path_factory) -> Path:
    """A framework environment that is itself a virtual environment (the /opt/venv layout), with pip and a package of
    its own, ``framework_pkg``, that a plain ``--system-site-packages`` venv made from it would not see."""
    root = tmp_path_factory.mktemp("framework") / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(root)], check=True, capture_output=True, timeout=300)
    python = root / "bin" / "python"
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (Path(purelib) / "framework_pkg").mkdir()
    (Path(purelib) / "framework_pkg" / "__init__.py").write_text("")
    return root


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.lstat().st_mtime_ns, path.lstat().st_size) for path in root.rglob("*")}


def _read_only_run(tmp_path, framework_venv, package_source, **env):
    wrapper = tmp_path / "read-only-framework" / "bin" / "python"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(_READ_ONLY_FRAMEWORK)
    wrapper.chmod(0o755)
    report = tmp_path / "report.json"
    report.unlink(missing_ok=True)
    full = {
        "HYPERLOOM_SOURCE_REPO": str(package_source),
        "HYPERLOOM_SOURCE_SHA": _git(package_source, "rev-parse", "HEAD"),
        "PYTHON": str(wrapper),
        "REAL_FRAMEWORK_PY": str(framework_venv / "bin" / "python"),
        "REPORT": str(report),
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    full.update(env)
    code, exit_file, stderr = _bootstrap(tmp_path, **full)
    return code, exit_file, stderr, (json.loads(report.read_text()) if report.is_file() else None)


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_framework_this_account_cannot_write_gets_a_venv_beside_it_that_sees_its_packages(
    tmp_path, framework_venv, package_source
):
    before = _tree(framework_venv)
    code, exit_file, stderr, report = _read_only_run(tmp_path, framework_venv, package_source)
    assert code == 0 and exit_file is None, stderr
    venv = tmp_path / "container-tmp" / "hyperloom-venv"
    assert report["argv"] == ["hosted", "run"]
    assert report["executable"] == str(venv / "bin" / "python") and report["prefix"] == str(venv)
    assert report["path"][:2] == [str(venv / "bin"), str(tmp_path / "read-only-framework" / "bin")], (
        "the venv's bin leads PATH and the framework's bin stays on it, behind"
    )
    assert report["virtual_env"] == str(venv)
    assert Path(report["hyperloom"]).is_relative_to(venv), "Hyperloom is installed into the new environment"
    assert report["framework_pkg"] and Path(report["framework_pkg"]).is_relative_to(framework_venv), (
        "the framework's own package is found where the image has it"
    )
    sha = _git(package_source, "rev-parse", "HEAD")
    assert report["revision"] == sha and (venv / ".hyperloom-install").read_text().strip() == sha
    assert not (venv / "bin" / "pip").exists(), "the environment has no pip of its own"
    assert _tree(framework_venv) == before, "the framework's environment is not written"
    assert sorted(path.name for path in (tmp_path / "container-tmp").iterdir()) == ["hyperloom-venv"]


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_restart_reuses_the_venv_and_a_new_commit_makes_it_anew(tmp_path, framework_venv, package_source):
    assert _read_only_run(tmp_path, framework_venv, package_source)[0] == 0
    venv = tmp_path / "container-tmp" / "hyperloom-venv"
    (venv / "left-by-the-first-start").write_text("")
    code, _, stderr, report = _read_only_run(tmp_path, framework_venv, package_source)
    assert code == 0 and report["prefix"] == str(venv), stderr
    assert (venv / "left-by-the-first-start").exists(), "a restart of the same commit installs nothing"
    assert "installing into" not in stderr, "and does not say it does"
    (package_source / "another").write_text("")
    _git(package_source, "add", "another")
    _git(package_source, "commit", "-q", "-m", "next")
    code, _, stderr, report = _read_only_run(tmp_path, framework_venv, package_source)
    assert code == 0 and report["revision"] == _git(package_source, "rev-parse", "HEAD"), stderr
    assert not (venv / "left-by-the-first-start").exists(), "another commit gets a new environment"
    assert report["framework_pkg"], "and it still sees the framework's packages"


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_with_no_writable_place_for_the_venv_the_run_ends_for_good(tmp_path, source):
    not_a_dir = tmp_path / "tmp-file"
    not_a_dir.write_text("")
    code, stderr, _, calls = _framework_run(tmp_path, source, FAKE_WRITABLE="0", TMPDIR=str(not_a_dir))
    _expect_final(code, _exit_file(tmp_path), stderr, "the place for a virtual environment beside the framework's")
    assert not [c for c in calls if " install " in c], "nothing is installed into the framework's environment"
    assert not (tmp_path / "framework" / ".hyperloom-install").exists()


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_venv_path_that_is_not_this_accounts_directory_ends_the_run_for_good(tmp_path, source):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "container-tmp").mkdir()
    (tmp_path / "container-tmp" / "hyperloom-venv").symlink_to(elsewhere)
    code, stderr, _, calls = _framework_run(tmp_path, source, FAKE_WRITABLE="0")
    _expect_final(code, _exit_file(tmp_path), stderr, "hyperloom-venv is not a directory of this account")
    assert calls == [] and list(elsewhere.iterdir()) == []


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_framework_python_that_cannot_make_a_venv_ends_the_run_for_good(tmp_path, source):
    code, stderr, _, calls = _framework_run(tmp_path, source, FAKE_WRITABLE="0", FAKE_VENV_FAILS="1")
    _expect_final(code, _exit_file(tmp_path), stderr, "cannot make a virtual environment in")
    assert [c for c in calls if c.startswith("-m venv --clear --without-pip ")]
    assert not [c for c in calls if " install " in c]


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_venv_the_pth_step_cannot_complete_in_ends_the_run_for_good(tmp_path, source):
    code, stderr, _, calls = _framework_run(tmp_path, source, FAKE_WRITABLE="0")
    # The stub's `-m venv` succeeds and makes nothing, so the .pth step finds no interpreter in the venv.
    _expect_final(code, _exit_file(tmp_path), stderr, "hyperloom-venv see the framework's packages")
    assert [c for c in calls if c.startswith("-m venv --clear --without-pip ")]
    assert not [c for c in calls if " install " in c]


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_a_framework_python_whose_probe_does_not_run_is_left_to_a_restart(tmp_path, source):
    code, stderr, _, calls = _framework_run(tmp_path, source, FAKE_PROBE_FAILS="1")
    assert (code, _exit_file(tmp_path)) == (1, None), stderr
    assert "does not run" in stderr and calls == []


def _exit_file(tmp_path):
    written = tmp_path / "state" / "exit.json"
    return json.loads(written.read_text()) if written.is_file() else None


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a directory without write permission")
@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
def test_the_bootstrap_itself_finds_that_this_account_cannot_write_the_framework(
    tmp_path, framework_venv, package_source
):
    """The real probe, with no stand-in: a framework environment without write permission for this account."""
    directories = [
        framework_venv,
        *(path for path in framework_venv.rglob("*") if path.is_dir() and not path.is_symlink()),
    ]
    for directory in directories:
        directory.chmod(directory.stat().st_mode & ~0o222)
    try:
        code, exit_file, stderr, report = _read_only_run(
            tmp_path, framework_venv, package_source, PYTHON=str(framework_venv / "bin" / "python")
        )
    finally:
        for directory in directories:
            directory.chmod(directory.stat().st_mode | 0o200)
    assert code == 0 and exit_file is None, stderr
    assert report["prefix"] == str(tmp_path / "container-tmp" / "hyperloom-venv") and report["framework_pkg"]


def _plain_framework(root: Path, *venv_args: str) -> Path:
    """A framework environment of this account's own, made by ``python -m venv`` with ``venv_args``."""
    subprocess.run([sys.executable, "-m", "venv", *venv_args, str(root)], check=True, capture_output=True, timeout=300)
    return root


def _framework_path(framework: Path, name: str) -> Path:
    python = framework / "bin" / "python"
    done = subprocess.run(
        [str(python), "-c", f"import sysconfig; print(sysconfig.get_path({name!r}))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(done.stdout.strip())


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
@pytest.mark.parametrize("layout", ["complete", "site-packages-not-made-yet"])
def test_the_real_probe_keeps_a_framework_this_account_can_write_on_the_old_path(tmp_path, package_source, layout):
    """The real probe, with no stand-in, on a framework this account can write: the wheel and the stamp go into it and
    its Python runs the entry. A framework whose site-packages does not exist yet (pip makes it) is writable through
    the nearest directory above it."""
    framework = tmp_path / "writable-framework"
    env = {}
    if layout == "complete":
        _plain_framework(framework)
    else:
        _plain_framework(framework, "--without-pip")
        _framework_path(framework, "purelib").rmdir()
        bundled = sorted((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
        if not bundled:
            pytest.skip("this Python carries no bundled pip wheel")
        env["PYTHONPATH"] = str(bundled[-1])
    code, exit_file, stderr, report = _read_only_run(
        tmp_path, framework, package_source, PYTHON=str(framework / "bin" / "python"), **env
    )
    assert code == 0 and exit_file is None, stderr
    assert report["executable"] == str(framework / "bin" / "python") and report["prefix"] == str(framework)
    assert report["path"][0] == str(framework / "bin") and report["virtual_env"] == str(framework)
    assert Path(report["hyperloom"]).is_relative_to(framework), "Hyperloom is installed into the framework"
    sha = _git(package_source, "rev-parse", "HEAD")
    assert (framework / ".hyperloom-install").read_text().strip() == sha
    assert not (tmp_path / "container-tmp" / "hyperloom-venv").exists() and "installing into" not in stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a directory without write permission")
@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
@pytest.mark.parametrize("read_only", ["purelib", "scripts"])
def test_the_real_probe_asks_about_each_directory_pip_installs_into(tmp_path, package_source, read_only):
    """A framework whose prefix this account can write but one of the directories pip installs into it cannot."""
    framework = _plain_framework(tmp_path / "framework-with-one-read-only-directory")
    directory = _framework_path(framework, read_only)
    directory.chmod(directory.stat().st_mode & ~0o222)
    try:
        code, exit_file, stderr, report = _read_only_run(
            tmp_path, framework, package_source, PYTHON=str(framework / "bin" / "python")
        )
    finally:
        directory.chmod(directory.stat().st_mode | 0o200)
    venv = tmp_path / "container-tmp" / "hyperloom-venv"
    assert code == 0 and exit_file is None, stderr
    assert report["prefix"] == str(venv) and (venv / ".hyperloom-install").is_file()
    assert not (framework / ".hyperloom-install").exists()


# A stand-in for a system Python (no virtual environment) and its pip, in two shapes (FAKE_PIP):
#   old    a pip before 23.0.1: it has no --break-system-packages, so any command given it, `install --help` included,
#          fails with "no such option";
#   23.0.1 pip 23.0.1 and later: `pip install` takes the option, `pip wheel` rejects it.
# Like real pip, every shape rejects the option before the subcommand and on `pip wheel`. A rejected option exits 2.
_SYSTEM_STUB = """#!/bin/bash
if [[ "$1" == "-c" ]]; then
  case "$2" in *W_OK*) echo 1 ;; *base_prefix*) echo 0 ;; *) echo "$FAKE_PREFIX" ;; esac
  exit 0
fi
subcommand=""
for arg in "$@"; do
  case "$arg" in
    install | wheel) [[ -n "$subcommand" ]] || subcommand="$arg" ;;
    --break-system-packages)
      if [[ "$subcommand" != install || "$FAKE_PIP" == old ]]; then
        echo "no such option: --break-system-packages" >&2
        exit 2
      fi
      ;;
  esac
done
if [[ " $* " == *" --help "* ]]; then
  exit 0
fi
echo "$*" >>"$FAKE_LOG"
if [[ "$subcommand" == wheel ]]; then
  while [[ "$1" != "-w" ]]; do shift; done
  : >"$2/hyperloom-0-py3-none-any.whl"
fi
"""


@pytest.mark.skipif(Path("/opt/venv/bin/python").exists(), reason="the image's framework Python takes precedence")
@pytest.mark.parametrize(
    ("fake_pip", "install_gets_it"),
    [("old", False), ("23.0.1", True)],
    ids=["pip-before-23.0.1", "pip-23.0.1-or-later"],
)
def test_a_system_python_gets_break_system_packages_on_its_install_only(tmp_path, source, fake_pip, install_gets_it):
    stub = tmp_path / "framework" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text(_SYSTEM_STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls"
    code, exit_file, stderr = _bootstrap(
        tmp_path,
        HYPERLOOM_SOURCE_REPO=str(source),
        HYPERLOOM_SOURCE_SHA=_git(source, "rev-parse", "HEAD"),
        PYTHON=str(stub),
        FAKE_PREFIX=str(tmp_path / "framework"),
        FAKE_LOG=str(log),
        FAKE_PIP=fake_pip,
    )
    assert (code, exit_file) == (0, None), stderr
    calls = log.read_text().splitlines()
    (wheel,) = [c for c in calls if " wheel " in c]
    (install,) = [c for c in calls if " install " in c]
    assert "--break-system-packages" not in wheel.split()
    assert ("--break-system-packages" in install.split()) is install_gets_it, install
    assert calls[-1] == "-m hyperloom hosted run"
