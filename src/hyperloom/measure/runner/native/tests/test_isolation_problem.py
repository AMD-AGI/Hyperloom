# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether this host can compile as user 65534: root, ``setpriv``, and 65534 mapped in this process's user namespace;
where it cannot, no isolation is offered and the build refuses naming why."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.common.tests._id_map_helpers import HOST_MAP, ONE_ID_MAP, use_id_maps
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_UID,
    SOURCE_ROOT,
    IsolatedBuilds,
    isolation_problem,
    local_isolation,
)


@pytest.fixture
def root_with_setpriv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "setpriv" else None)


@pytest.fixture
def id_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Have the check read stand-in uid and gid maps (the gid map is the uid map unless given); returns their paths."""

    def use(uid_map: str | None, gid_map: str | None = None) -> tuple[Path, Path]:
        return use_id_maps(monkeypatch, tmp_path, uid_map, gid_map)

    return use


def test_on_a_host_namespace_the_compile_runs_as_65534(root_with_setpriv: None, id_maps) -> None:
    id_maps(HOST_MAP)

    found = local_isolation(lambda unshare: pytest.fail("there is no unshare to probe"))

    assert isolation_problem() is None
    assert found is not None and (found.uid, found.level) == (65534, ISOLATION_UID)
    assert found.prefix[:4] == ("/usr/bin/setpriv", "--reuid=65534", "--regid=65534", "--clear-groups")


def test_a_one_id_namespace_gives_no_isolation_and_says_65534_is_not_mapped(root_with_setpriv: None, id_maps) -> None:
    uid_map, gid_map = id_maps(ONE_ID_MAP)

    assert local_isolation(lambda unshare: True) is None
    problem = isolation_problem()
    assert problem is not None and problem.startswith(
        f"uid 65534, gid 65534 are not mapped in this process's user namespace ({uid_map}: '0 1234 1'; {gid_map}: "
        "'0 1234 1'): setpriv cannot become user 65534 there"
    ), problem
    assert "fakeroot" in problem


@pytest.mark.parametrize(
    ("uid_map", "gid_map", "says"),
    [
        (HOST_MAP, ONE_ID_MAP, "gid 65534 is not mapped"),
        (ONE_ID_MAP, HOST_MAP, "uid 65534 is not mapped"),
        ("0 0 65534\n", HOST_MAP, "uid 65534 is not mapped"),
        (None, HOST_MAP, "cannot read {uid_map}: FileNotFoundError"),
        ("0 0 x\n", HOST_MAP, "cannot parse {uid_map}: '0 0 x' is not 'inside outside count'"),
    ],
    ids=["gid-only", "uid-only", "range-ending-below-65534", "missing", "garbled"],
)
def test_either_id_unmapped_or_a_map_unread_gives_no_isolation_naming_it(
    root_with_setpriv: None, id_maps, uid_map: str | None, gid_map: str, says: str
) -> None:
    paths = id_maps(uid_map, gid_map)

    assert local_isolation(lambda unshare: True) is None
    problem = isolation_problem()
    assert problem is not None and problem.startswith(says.format(uid_map=paths[0])), problem


def test_a_host_that_is_not_root_or_has_no_setpriv_is_named(monkeypatch: pytest.MonkeyPatch, id_maps) -> None:
    id_maps(HOST_MAP)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 1000)
    assert isolation_problem() == "it runs as uid 1000, not root, and only root can compile as user 65534"
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: None)
    assert isolation_problem() == "setpriv (util-linux) is not on PATH, and the compile runs as user 65534 through it"


class _Planner:
    def planned(self, inputs: object) -> object:
        return inputs


def test_the_build_on_a_one_id_namespace_refuses_naming_the_cause(
    root_with_setpriv: None, id_maps, tmp_path: Path
) -> None:
    id_maps(ONE_ID_MAP)
    runner = IsolatedBuilds(_Planner(), None, local_isolation(lambda unshare: True), {})
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(BuildInfraError) as refused:
        runner(object())(scratch)

    message = str(refused.value)
    assert message.startswith(
        "this host cannot isolate a native build: uid 65534, gid 65534 are not mapped in this process's user namespace"
    ), message
    assert message.endswith("; nothing was compiled") and "setpriv cannot become user 65534" in message
    assert not list(scratch.iterdir()), "nothing was laid"


def test_a_build_given_no_isolation_on_a_host_that_has_one_says_so(
    root_with_setpriv: None, id_maps, tmp_path: Path
) -> None:
    id_maps(HOST_MAP)

    with pytest.raises(BuildInfraError) as refused:
        IsolatedBuilds(_Planner(), None, None, {})(object())(tmp_path)

    assert str(refused.value) == (
        "this host cannot isolate a native build: the build runner was given no isolation; nothing was compiled"
    )


_ISOLATION_IN_NAMESPACE = """
from hyperloom.measure.runner.native.isolated import isolation_problem, local_isolation
print(local_isolation(lambda unshare: False))
print(isolation_problem())
"""


def _in_one_id_namespace(*command: str) -> subprocess.CompletedProcess[str] | None:
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    namespace = [unshare, "--user", "--map-root-user", "--"]
    if subprocess.run([*namespace, "true"], capture_output=True, check=False).returncode != 0:
        return None
    return subprocess.run([*namespace, *command], capture_output=True, text=True, check=False)


def test_in_a_real_one_id_user_namespace_setpriv_fails_and_no_isolation_is_offered() -> None:
    if shutil.which("setpriv") is None:
        pytest.skip("no setpriv")
    premise = _in_one_id_namespace("setpriv", "--reuid=65534", "--regid=65534", "--clear-groups", "true")
    if premise is None:
        pytest.skip("this host cannot make a user namespace (unshare --user)")
    assert premise.returncode != 0 and "Invalid argument" in premise.stderr, premise.stderr

    done = _in_one_id_namespace(
        sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + _ISOLATION_IN_NAMESPACE
    )

    assert done is not None and done.returncode == 0, done.stderr if done else "no namespace"
    found, problem = done.stdout.splitlines()
    assert found == "None"
    assert problem.startswith(
        "uid 65534, gid 65534 are not mapped in this process's user namespace (/proc/self/uid_map: '0 "
    )
