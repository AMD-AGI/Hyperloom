# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether this host can compile as user 65534: root, ``setpriv``, and 65534 mapped in this process's user namespace;
where it cannot, bubblewrap is the compile's isolation when its probe passes, and where neither works no isolation is
offered and the build refuses naming both reasons."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common.bwrap import BwrapProbe

from hyperloom.common.agent_isolation import BwrapIsolation, NoIsolation
from hyperloom.common.tests._id_map_helpers import HOST_MAP, ONE_ID_MAP, use_id_maps
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_BWRAP_NETNS,
    ISOLATION_UID,
    SOURCE_ROOT,
    IsolatedBuilds,
    Isolation,
    account_problem,
    isolation_problem,
    local_isolation,
)

NO_BWRAP = NoIsolation(("bubblewrap: not found on PATH",))
WORKING_BWRAP = BwrapIsolation(BwrapProbe("/usr/bin/bwrap", (0, 9, 0), ()))


class Bwrap:
    """Stands in for :func:`~hyperloom.common.agent_isolation.bwrap_isolation`: gives ``found``, counting the asks."""

    def __init__(self, found: BwrapIsolation | NoIsolation) -> None:
        self.found, self.asked = found, 0

    def __call__(self) -> BwrapIsolation | NoIsolation:
        self.asked += 1
        return self.found


@pytest.fixture(autouse=True)
def bwrap(monkeypatch: pytest.MonkeyPatch) -> Bwrap:
    """No bubblewrap, whatever this host has: a test that wants one sets ``found``."""
    stand_in = Bwrap(NO_BWRAP)
    monkeypatch.setattr(isolated, "bwrap_isolation", stand_in)
    return stand_in


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


def test_on_a_host_namespace_the_compile_runs_as_65534_and_bubblewrap_is_not_asked(
    root_with_setpriv: None, id_maps, bwrap: Bwrap
) -> None:
    id_maps(HOST_MAP)
    bwrap.found = WORKING_BWRAP

    found = local_isolation(lambda unshare: pytest.fail("there is no unshare to probe"))

    assert account_problem() is None and isolation_problem() is None
    assert found is not None and (found.uid, found.level, found.bwrap) == (65534, ISOLATION_UID, None)
    assert found.prefix[:4] == ("/usr/bin/setpriv", "--reuid=65534", "--regid=65534", "--clear-groups")
    assert bwrap.asked == 0, "the account tier is taken first"


def test_a_one_id_namespace_without_bubblewrap_gives_no_isolation_and_names_both(
    root_with_setpriv: None, id_maps
) -> None:
    uid_map, gid_map = id_maps(ONE_ID_MAP)

    assert local_isolation(lambda unshare: True) is None
    unmapped = (
        f"uid 65534, gid 65534 are not mapped in this process's user namespace ({uid_map}: '0 1234 1'; {gid_map}: "
        "'0 1234 1'): setpriv cannot become user 65534 there"
    )
    problem = account_problem()
    assert problem is not None and problem.startswith(unmapped), problem
    assert "fakeroot" in problem
    assert isolation_problem() == (
        f"it can compile neither as user 65534 ({problem}) nor under bubblewrap (bubblewrap: not found on PATH)"
    )


def test_a_one_id_namespace_with_bubblewrap_compiles_under_it_saying_why_not_as_65534(
    root_with_setpriv: None, id_maps, bwrap: Bwrap
) -> None:
    id_maps(ONE_ID_MAP)
    bwrap.found = WORKING_BWRAP

    found = local_isolation(lambda unshare: pytest.fail("a sandbox has a network of its own; nothing to probe"))

    assert found is not None and (found.prefix, found.uid, found.level) == ((), None, ISOLATION_BWRAP_NETNS)
    assert found.bwrap is not None and found.bwrap.probe == WORKING_BWRAP.probe
    (note,) = found.bwrap.notes
    assert note.startswith("compiling as user 65534: uid 65534, gid 65534 are not mapped"), note
    assert isolation_problem() is None


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
    problem = account_problem()
    assert problem is not None and problem.startswith(says.format(uid_map=paths[0])), problem


def test_a_host_that_is_not_root_or_has_no_setpriv_is_named(monkeypatch: pytest.MonkeyPatch, id_maps) -> None:
    id_maps(HOST_MAP)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 1000)
    assert account_problem() == "it runs as uid 1000, not root, and only root can compile as user 65534"
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: None)
    assert account_problem() == "setpriv (util-linux) is not on PATH, and the compile runs as user 65534 through it"
    assert local_isolation(lambda unshare: True) is None


def test_a_host_without_setpriv_compiles_under_bubblewrap_where_it_works(
    monkeypatch: pytest.MonkeyPatch, id_maps, bwrap: Bwrap
) -> None:
    id_maps(HOST_MAP)
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    bwrap.found = WORKING_BWRAP

    found = local_isolation(lambda unshare: pytest.fail("nothing to probe"))

    assert found == Isolation(
        (),
        None,
        ISOLATION_BWRAP_NETNS,
        BwrapIsolation(
            WORKING_BWRAP.probe,
            ("compiling as user 65534: it runs as uid 1000, not root, and only root can compile as user 65534",),
        ),
    )


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
        "this host cannot isolate a native build: it can compile neither as user 65534 (uid 65534, gid 65534 are not "
        "mapped in this process's user namespace"
    ), message
    assert "setpriv cannot become user 65534" in message
    assert message.endswith(" nor under bubblewrap (bubblewrap: not found on PATH); nothing was compiled"), message
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
import json
from hyperloom.common.agent_isolation import BwrapIsolation, bwrap_isolation
from hyperloom.measure.runner.native.isolated import account_problem, local_isolation
found, bwrap = local_isolation(lambda unshare: False), bwrap_isolation()
level = None if found is None else found.level
print(json.dumps([level, account_problem(), isinstance(bwrap, BwrapIsolation), str(bwrap)]))
"""


def _in_one_id_namespace(*command: str) -> subprocess.CompletedProcess[str] | None:
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    namespace = [unshare, "--user", "--map-root-user", "--"]
    if subprocess.run([*namespace, "true"], capture_output=True, check=False).returncode != 0:
        return None
    return subprocess.run([*namespace, *command], capture_output=True, text=True, check=False)


def test_in_a_real_one_id_user_namespace_setpriv_fails_and_only_bubblewrap_is_offered() -> None:
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
    level, problem, bwrap_works, bwrap = json.loads(done.stdout.splitlines()[-1])
    assert problem.startswith(
        "uid 65534, gid 65534 are not mapped in this process's user namespace (/proc/self/uid_map: '0 "
    )
    assert level == (ISOLATION_BWRAP_NETNS if bwrap_works else None), bwrap
