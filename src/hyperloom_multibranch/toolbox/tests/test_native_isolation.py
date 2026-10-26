# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a native build's compile is isolated: always another user, a network namespace of its own only where the host
lets this process make one, the level written in the build's record and logged once; and the child compiles with the
builder of the tree this process runs from, never with an installed copy."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import hyperloom
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_KEY,
    ISOLATION_UID,
    ISOLATION_UID_NETNS,
    SOURCE_ROOT,
    ChildInterpreter,
    IsolatedBuilds,
    Isolation,
    local_isolation,
    private_network,
)
from hyperloom.measure.tests._recipes import IMAGE
from hyperloom_multibranch.toolbox import native_builds as native_builds_module
from hyperloom_multibranch.toolbox.native_builds import native_builds
from hyperloom_multibranch.toolbox.tests import native_support as fake
from hyperloom_multibranch.toolbox.tests.test_native_builds import EDITED_CU, Hooks, Rig, _configured

SETPRIV = "/usr/bin/setpriv"
UNSHARE = "/usr/bin/unshare"
#: A child, in a tree of its own, that stages a record saying which tree it came from.
_MARKED_CHILD = """\
import json
from pathlib import Path


def main(argv):
    record = {"status": "failed", "outputs": [], "toolchain": {"tree": "marked"}, "log": "", "failure": "marked"}
    (Path(argv[0]) / "staged.json").write_text(json.dumps(record))
    return 0
"""
_PACKAGES = ("hyperloom", "hyperloom/measure", "hyperloom/measure/runner", "hyperloom/measure/runner/native")


@pytest.fixture
def root_with_tools(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A root process on a host with ``setpriv`` and ``unshare``; the list is every probe of a network namespace."""
    probed: list[str] = []
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    return probed


def test_where_the_host_makes_a_network_namespace_the_compile_gets_one_and_another_user(
    root_with_tools: list[str],
) -> None:
    found = local_isolation(lambda unshare: root_with_tools.append(unshare) or True)

    assert found is not None and (found.uid, found.level) == (65534, ISOLATION_UID_NETNS)
    assert found.prefix[:4] == (UNSHARE, "--net", "--", SETPRIV) and "--reuid=65534" in found.prefix
    assert root_with_tools == [UNSHARE], "the host is probed once"


def test_where_it_cannot_the_compile_still_runs_as_another_user_on_the_hosts_network(
    root_with_tools: list[str],
) -> None:
    found = local_isolation(lambda unshare: root_with_tools.append(unshare) and False)

    assert found is not None and (found.uid, found.level) == (65534, ISOLATION_UID)
    assert found.prefix[0] == SETPRIV and UNSHARE not in found.prefix and "--net" not in found.prefix
    assert {"--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs"} <= set(found.prefix)
    assert root_with_tools == [UNSHARE]


def test_without_unshare_the_compile_runs_as_another_user_and_nothing_is_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(isolated.shutil, "which", lambda name: SETPRIV if name == "setpriv" else None)
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)

    found = local_isolation(lambda unshare: pytest.fail("there is nothing to probe"))

    assert found is not None and (found.prefix[0], found.level) == (SETPRIV, ISOLATION_UID)


def test_a_host_that_cannot_change_the_user_gives_no_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 1000)
    assert local_isolation(lambda unshare: True) is None, "only root can give the compile another user"
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: UNSHARE if name == "unshare" else None)
    assert local_isolation(lambda unshare: True) is None, "a network namespace alone does not keep the key from it"


@pytest.mark.parametrize(("exit_code", "made"), [(0, True), (1, False)])
def test_the_probe_runs_unshare_net_once_and_reads_its_exit(tmp_path: Path, exit_code: int, made: bool) -> None:
    calls = tmp_path / "calls"
    tool = tmp_path / "unshare"
    tool.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nexit {exit_code}\n')
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)

    assert private_network(str(tool)) is made
    assert calls.read_text() == "--net -- true\n"


def test_a_probe_that_cannot_start_finds_no_network_namespace(tmp_path: Path) -> None:
    assert private_network(str(tmp_path / "missing-unshare")) is False


@pytest.mark.parametrize("level", [ISOLATION_UID, ISOLATION_UID_NETNS])
async def test_the_build_record_names_the_isolation_its_compile_had(tmp_path: Path, level: str) -> None:
    rig = Rig(tmp_path, isolation=Isolation((), None, level))
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())

    await rig.wait(AcceptedMeasurement(compiled, ()), Hooks())

    assert rig.store.read(compiled.builds[0].inputs).toolchain[ISOLATION_KEY] == level


@pytest.mark.parametrize(
    ("isolation", "level", "says"),
    [
        (Isolation((UNSHARE, "--net", "--", SETPRIV), 65534, ISOLATION_UID_NETNS), logging.INFO, "of their own"),
        (Isolation((SETPRIV,), 65534, ISOLATION_UID), logging.WARNING, "with this host's network"),
        (None, logging.WARNING, "every build will fail"),
    ],
)
def test_the_session_logs_its_isolation_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, isolation: Isolation | None, level: int, says: str
) -> None:
    probes: list[int] = []

    def probe() -> Isolation | None:
        probes.append(1)
        return isolation

    with caplog.at_level(logging.INFO, logger=native_builds_module.logger.name):
        native_builds(_configured(tmp_path), LocalBlobStore(tmp_path / "blobs"), IMAGE, isolation=probe)

    (record,) = [r for r in caplog.records if r.name == native_builds_module.logger.name]
    assert (record.levelno, says in record.getMessage()) == (level, True)
    assert isolation is None or isolation.level in record.getMessage()
    assert probes == [1], "the host is probed once, at the start"


def _built_with_source(
    tmp_path: Path, source: Path, isolation: Isolation = Isolation((), None), python: str = sys.executable
):
    rig = Rig(tmp_path / "rig")
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())
    child = ChildInterpreter(python, source)
    runner = IsolatedBuilds(rig.planner, rig.blobs, isolation, {fake.COMPONENT: fake.FACTORY}, child)
    scratch = Path(tempfile.mkdtemp(prefix="hl-native-", dir=tmp_path))  # private to this user, as the service makes it
    return runner(compiled.builds[0].inputs)(scratch)


def _tree(root: Path, child: str) -> Path:
    for package in _PACKAGES:
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("")
    (root / _PACKAGES[-1] / "child.py").write_text(child)
    return root


def test_the_child_compiles_with_the_builder_of_the_tree_it_is_given(tmp_path: Path) -> None:
    staged = _built_with_source(tmp_path, _tree(tmp_path / "tree", _MARKED_CHILD))

    assert staged.toolchain == {"tree": "marked", ISOLATION_KEY: "none"}, "the installed builder ran instead"


def test_a_child_whose_builder_resolves_outside_the_tree_refuses_to_build(tmp_path: Path) -> None:
    elsewhere = tmp_path / "tree-without-hyperloom"
    elsewhere.mkdir()

    with pytest.raises(BuildInfraError, match="not to the tree under test"):
        _built_with_source(tmp_path, elsewhere)


def test_by_default_the_child_uses_the_tree_this_process_runs_from() -> None:
    assert Path(hyperloom.__file__).resolve().parent.parent == SOURCE_ROOT == ChildInterpreter().source


def test_the_compile_user_can_reach_its_work_below_the_services_private_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch.chmod(0o700)  # what tempfile.TemporaryDirectory makes
    (scratch / "work").mkdir()
    (scratch / "work" / "job.json").write_text("{}")

    isolated._give(scratch / "work", os.getuid())

    assert stat.S_IMODE(scratch.stat().st_mode) == 0o701, "others may pass through it, never list or read it"


#: A child that stages what it could do: its user, whether it could read the key, and its network namespace.
_PROBING_CHILD = """\
import json
import os
from pathlib import Path


def main(argv):
    try:
        Path({key!r}).read_bytes()
        key = "read"
    except OSError as exc:
        key = type(exc).__name__
    seen = {{"uid": str(os.getuid()), "key": key, "net": os.readlink("/proc/self/ns/net")}}
    record = {{"status": "failed", "outputs": [], "toolchain": seen, "log": "", "failure": "probe"}}
    (Path(argv[0]) / "staged.json").write_text(json.dumps(record))
    return 0
"""


def _nobody_python() -> str | None:
    """An interpreter user 65534 can run on this host (this process's may live where only its own user reads)."""
    found = local_isolation(lambda unshare: False)
    for python in (sys.executable, shutil.which("python3")):
        if found is not None and python is not None:
            ran = subprocess.run([*found.prefix, python, "-I", "-c", "pass"], capture_output=True, check=False)
            if ran.returncode == 0:
                return python
    return None


@pytest.mark.skipif(_nobody_python() is None, reason="this host cannot run Python as user 65534 (needs root, setpriv)")
@pytest.mark.parametrize("network", [True, False], ids=["uid+netns", "uid"])
def test_on_a_real_host_the_compile_runs_as_nobody_cannot_read_the_key_and_has_a_network_only_where_made(
    network: bool,
) -> None:
    isolation = local_isolation(lambda unshare: network and private_network(unshare))
    if isolation is None or (network and isolation.level != ISOLATION_UID_NETNS):
        pytest.skip("this host cannot make a network namespace")
    base = Path(tempfile.mkdtemp(prefix="w201-isolation-"))
    try:
        base.chmod(0o755)  # the compile user reads the tree from here, as it reads the toolbox's own
        key = base / "service.key"
        key.write_text("not a key")
        key.chmod(0o600)

        tree = _tree(base / "tree", _PROBING_CHILD.format(key=str(key)))
        staged = _built_with_source(base, tree, isolation, str(_nobody_python()))

        seen = dict(staged.toolchain)
        own_network = os.readlink("/proc/self/ns/net")
        assert (seen["uid"], seen["key"], seen[ISOLATION_KEY]) == ("65534", "PermissionError", isolation.level)
        assert (seen["net"] != own_network) is network
    finally:
        subprocess.run(["rm", "-rf", str(base)], check=False)
