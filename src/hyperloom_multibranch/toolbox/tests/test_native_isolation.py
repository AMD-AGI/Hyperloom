# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a native build's compile is isolated: another user, with a network namespace of its own only where the host
lets this process make one, else a bubblewrap sandbox with a network of its own; the level written in the build's
record and logged once; and the child compiles with the builder of the tree this process runs from, never with an
installed copy."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom_common.bwrap import BwrapProbe

import hyperloom
from hyperloom.common import confinement
from hyperloom.common.agent_isolation import BwrapIsolation, NoIsolation
from hyperloom.common.tests._id_map_helpers import HOST_MAP, use_id_maps
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_BWRAP_NETNS,
    ISOLATION_KEY,
    ISOLATION_UID,
    ISOLATION_UID_NETNS,
    SOURCE_ROOT,
    ChildInterpreter,
    IsolatedBuilds,
    Isolation,
    Unisolated,
    local_isolation,
    private_network,
)
from hyperloom.measure.runner.native.tests._bwrap_compile import bwrap_compile_isolation
from hyperloom.measure.tests._recipes import IMAGE
from hyperloom_multibranch.toolbox import native_builds as native_builds_module
from hyperloom_multibranch.toolbox.native_builds import native_builds
from hyperloom_multibranch.toolbox.tests import native_support as fake
from hyperloom_multibranch.toolbox.tests.test_native_builds import EDITED_CU, Hooks, Rig, _configured, farm_of


@pytest.fixture(autouse=True)
def ready_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host checks pass here (test_build_preflight has them)."""
    monkeypatch.setattr(native_builds_module, "build_preflight", lambda site: None)


SETPRIV = "/usr/bin/setpriv"
UNSHARE = "/usr/bin/unshare"
_BWRAP = BwrapProbe("/usr/bin/bwrap", (0, 9, 0), ())
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
def root_with_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """A root process on a host with ``setpriv`` and ``unshare`` whose user namespace maps every id; the list is every
    probe of a network namespace."""
    use_id_maps(monkeypatch, tmp_path, HOST_MAP)
    probed: list[str] = []
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    return probed


def test_where_the_host_makes_a_network_namespace_the_compile_gets_one_and_another_user(
    root_with_tools: list[str],
) -> None:
    found = local_isolation(lambda unshare: root_with_tools.append(unshare) or True)

    assert isinstance(found, Isolation) and (found.uid, found.level) == (65534, ISOLATION_UID_NETNS)
    assert found.prefix[:4] == (UNSHARE, "--net", "--", SETPRIV) and "--reuid=65534" in found.prefix
    assert root_with_tools == [UNSHARE], "the host is probed once"


def test_where_it_cannot_the_compile_still_runs_as_another_user_on_the_hosts_network(
    root_with_tools: list[str],
) -> None:
    found = local_isolation(lambda unshare: root_with_tools.append(unshare) and False)

    assert isinstance(found, Isolation) and (found.uid, found.level) == (65534, ISOLATION_UID)
    assert found.prefix[0] == SETPRIV and UNSHARE not in found.prefix and "--net" not in found.prefix
    assert {"--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs"} <= set(found.prefix)
    assert root_with_tools == [UNSHARE]


def test_without_unshare_the_compile_runs_as_another_user_and_nothing_is_probed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_id_maps(monkeypatch, tmp_path, HOST_MAP)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: SETPRIV if name == "setpriv" else None)
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)

    found = local_isolation(lambda unshare: pytest.fail("there is nothing to probe"))

    assert isinstance(found, Isolation) and (found.prefix[0], found.level) == (SETPRIV, ISOLATION_UID)


def test_a_host_that_cannot_change_the_user_or_run_bubblewrap_gives_no_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(isolated, "bwrap_isolation", lambda: NoIsolation(("bubblewrap: not found on PATH",)))
    monkeypatch.setattr(isolated.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 1000)
    assert isinstance(local_isolation(lambda unshare: True), Unisolated), "only root can give the compile another user"
    monkeypatch.setattr(isolated.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolated.shutil, "which", lambda name: UNSHARE if name == "unshare" else None)
    assert isinstance(local_isolation(lambda unshare: True), Unisolated), (
        "a network namespace alone does not keep the key from it"
    )


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


@pytest.mark.parametrize("level", [ISOLATION_UID, ISOLATION_UID_NETNS, ISOLATION_BWRAP_NETNS])
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
        (
            Isolation((), None, ISOLATION_BWRAP_NETNS, BwrapIsolation(_BWRAP, ("compiling as user 65534: not root",))),
            logging.INFO,
            "under bubblewrap 0.9.0 /usr/bin/bwrap as this process's user, in a network namespace of their own "
            f"({ISOLATION_BWRAP_NETNS}): compiling as user 65534: not root",
        ),
        (Unisolated("no tier (a stand-in)"), logging.WARNING, "cannot isolate a compile now: each build asks again"),
    ],
)
def test_the_session_logs_its_isolation_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, isolation: Isolation | Unisolated, level: int, says: str
) -> None:
    probes: list[int] = []

    def probe() -> Isolation | Unisolated:
        probes.append(1)
        return isolation

    with caplog.at_level(logging.INFO, logger=native_builds_module.logger.name):
        native_builds(_configured(tmp_path), LocalBlobStore(tmp_path / "blobs"), IMAGE, isolation=probe)

    (record,) = [r for r in caplog.records if r.name == native_builds_module.logger.name]
    assert (record.levelno, says in record.getMessage()) == (level, True)
    if isinstance(isolation, Unisolated):
        assert record.getMessage().endswith(": no tier (a stand-in)"), "the reasons the choice found, not asked again"
    else:
        assert isolation.level in record.getMessage()
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


#: A child that stages what it could do: its user, whether it could read the key and its job, and its network namespace.
_PROBING_CHILD = """\
import json
import os
from pathlib import Path


def readable(path):
    try:
        path.read_bytes()
    except OSError as exc:
        return type(exc).__name__
    return "read"


def main(argv):
    key, job = readable(Path({key!r})), readable(Path(argv[0]) / "job.json")
    seen = {{"uid": str(os.getuid()), "key": key, "job": job, "net": os.readlink("/proc/self/ns/net")}}
    record = {{"status": "failed", "outputs": [], "toolchain": seen, "log": "", "failure": "probe"}}
    (Path(argv[0]) / "staged.json").write_text(json.dumps(record))
    return 0
"""


def _nobody_python() -> str | None:
    """An interpreter user 65534 can run on this host (this process's may live where only its own user reads)."""
    found = local_isolation(lambda unshare: False)
    for python in (sys.executable, shutil.which("python3")):
        if isinstance(found, Isolation) and found.uid is not None and python is not None:
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
    assert isinstance(isolation, Isolation) and isolation.uid is not None, "the account tier, as for _nobody_python"
    if network and isolation.level != ISOLATION_UID_NETNS:
        pytest.skip("this host cannot make a network namespace")
    base = Path(tempfile.mkdtemp(prefix="native-isolation-"))
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
        assert seen["job"] == "read", "the job is the compile user's own once its work directory is given to it"
        assert (seen["net"] != own_network) is network
    finally:
        subprocess.run(["rm", "-rf", str(base)], check=False)


# --- under bubblewrap ----------------------------------------------------------------------------------------------

#: A stand-in ``bwrap``: it writes its arguments to the file ``BWRAP_LOG`` names and runs the command after ``--``.
_STAND_IN_BWRAP = """\
#!/bin/sh
printf '%s\\n' "$@" > "{log}"
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""


def test_a_build_under_bubblewrap_runs_its_child_in_the_compile_sandbox_and_gives_nothing_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, stand_in = tmp_path / "bwrap.log", tmp_path / "bwrap"
    stand_in.write_text(_STAND_IN_BWRAP.format(log=log))
    stand_in.chmod(0o755)
    monkeypatch.setattr(isolated, "_give", lambda work, uid: pytest.fail("the work directory stays this user's"))
    sandbox = Isolation((), None, ISOLATION_BWRAP_NETNS, BwrapIsolation(BwrapProbe(str(stand_in), (0, 13, 1), ())))

    staged = _built_with_source(tmp_path, _tree(tmp_path / "tree", _MARKED_CHILD), sandbox)

    assert staged.toolchain == {"tree": "marked", ISOLATION_KEY: ISOLATION_BWRAP_NETNS}
    argv = log.read_text().splitlines()
    work = str(staged.staging)
    flags = argv[: argv.index("--")]
    assert {"--unshare-net", "--die-with-parent", "--unshare-pid", "--new-session"} <= set(flags)
    assert ["--bind", work, work] == flags[flags.index(work) - 1 : flags.index(work) + 2]
    assert ["--chdir", work] == flags[flags.index("--chdir") : flags.index("--chdir") + 2]
    assert argv[argv.index("--") + 1 :][1:3] == ["-I", "-c"], "the child's own command line follows"
    assert stat.S_IMODE(staged.staging.parent.stat().st_mode) == 0o700, "nobody else is let through the scratch"


#: A child that stages what its sandbox lets it do (``outside`` an address beyond loopback, ``key`` and ``bound_key``
#: the signing key outside every bind and inside a read-only one, ``read_only`` a path in the tree it imports from).
_SANDBOX_CHILD = """\
import errno
import json
import os
import socket
import subprocess
from pathlib import Path


def outcome(action):
    try:
        action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
    return "ok"


def write(path):
    with open(path, "w") as out:
        out.write("x")


def main(argv):
    work = Path(argv[0])
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    quiet = subprocess.run(["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    seen = {{
        "egress": outcome(lambda: socket.create_connection(({outside!r}, 443), timeout=5).close()),
        "key": outcome(lambda: Path({key!r}).read_bytes()),
        "bound_key": outcome(lambda: Path({bound_key!r}).read_bytes()),
        "dev_null": outcome(lambda: write("/dev/null")),
        "dev_null_child": str(quiet.returncode),
        "job": outcome(lambda: (work / "job.json").read_bytes()),
        "work": outcome(lambda: write(work / "written")),
        "read_only": outcome(lambda: write({read_only!r})),
        "cpu": outcome(lambda: os.listdir("/sys/devices/system/cpu")),
        "sys_class": outcome(lambda: os.listdir("/sys/class")),
        "uid": str(os.getuid()),
        "caps": status["CapEff"].strip(),
        "no_new_privs": status["NoNewPrivs"].strip(),
        "net": os.readlink("/proc/self/ns/net"),
        "pid": os.readlink("/proc/self/ns/pid"),
        "environ": ",".join(sorted(os.environ)),
    }}
    record = {{"status": "failed", "outputs": [], "toolchain": seen, "log": "", "failure": "probe"}}
    (work / "staged.json").write_text(json.dumps(record))
    return 0
"""
#: An address no sandbox with a network of its own can route to (TEST-NET-1, never a real host).
_OUTSIDE = "192.0.2.1"


@pytest.fixture
def bwrap_tier(monkeypatch: pytest.MonkeyPatch) -> Isolation:
    return bwrap_compile_isolation(monkeypatch)


@pytest.fixture
def outside_tmp() -> Iterator[Path]:
    """A directory below no ``/tmp`` of the host's: there a tree bound entry by entry is in the sandbox's root, not in
    its own ``/tmp``, which it writes."""
    with tempfile.TemporaryDirectory(dir="/var/tmp") as base:
        yield Path(base)


@pytest.mark.agent_isolation
def test_under_bubblewrap_the_compile_has_no_egress_no_key_a_dev_null_and_writes_only_its_work(
    tmp_path: Path, outside_tmp: Path, bwrap_tier: Isolation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_TEST_SERVICE_TOKEN", "SECRET token of the build service")
    registered = confinement.PRIVATE_PATHS  # this test's own (conftest)
    key = tmp_path / "service.key"
    tree = outside_tmp / "tree"  # it holds the key, so it is bound entry by entry
    bound_key = tree / "service.key"
    child = _SANDBOX_CHILD.format(
        outside=_OUTSIDE, key=str(key), bound_key=str(bound_key), read_only=str(tree / "written")
    )
    _tree(tree, child)
    for path in (key, bound_key):
        path.write_text("SECRET signing key")
        path.chmod(0o600)
        registered.add(path, "the build service's signing key")

    staged = _built_with_source(tmp_path, tree, bwrap_tier)

    seen = dict(staged.toolchain)
    namespaces = {name: seen.pop(name) for name in ("net", "pid")}
    assert namespaces != {
        "net": os.readlink("/proc/self/ns/net"),
        "pid": os.readlink("/proc/self/ns/pid"),
    }
    assert seen == {
        "egress": "ENETUNREACH",
        "key": "ENOENT",
        "bound_key": "ENOENT",
        "dev_null": "ok",
        "dev_null_child": "0",
        "job": "ok",
        "work": "ok",
        "read_only": "EROFS",
        "cpu": "ok",
        "sys_class": "ENOENT",
        "uid": str(os.getuid()),
        "caps": "0000000000000000",
        "no_new_privs": "1",
        "environ": "HOME,LC_ALL,PATH,PWD,TMPDIR",
        ISOLATION_KEY: ISOLATION_BWRAP_NETNS,
    }
    assert (staged.staging / "written").read_text() == "x" and not (tree / "written").exists()
    assert "SECRET" not in staged.log


@pytest.mark.agent_isolation
async def test_under_bubblewrap_a_cu_edit_is_built_stored_and_laid_from_a_worker_thread(
    tmp_path: Path, bwrap_tier: Isolation
) -> None:
    rig = Rig(tmp_path, isolation=bwrap_tier)
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())

    hooks = Hooks()
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), hooks)

    assert (hooks.ready, hooks.ended) == (["r1"], [])
    assert rig.store.read(compiled.builds[0].inputs).toolchain[ISOLATION_KEY] == ISOLATION_BWRAP_NETNS
    _, farm = farm_of(rig, builds.resolved(AcceptedMeasurement(compiled, ())))
    assert (farm.site / fake.COMPONENT / fake.extension().name).read_bytes() == fake.compiled(EDITED_CU)


def test_a_runner_given_what_chooses_its_isolation_asks_it_at_each_build(tmp_path: Path) -> None:
    """A bubblewrap whose probe gave no answer in time at the start of the session may answer at a later build."""
    answers: list[Isolation | Unisolated] = [Unisolated("no answer in time"), Isolation((), None)]
    tree = _tree(tmp_path / "tree", _MARKED_CHILD)

    with pytest.raises(BuildInfraError, match="cannot isolate a native build"):
        _built_with_source(tmp_path / "first", tree, lambda: answers.pop(0))
    staged = _built_with_source(tmp_path / "second", tree, lambda: answers.pop(0))

    assert staged.toolchain == {"tree": "marked", ISOLATION_KEY: "none"} and answers == []


@pytest.mark.parametrize(
    "found", [Unisolated("no tier (a stand-in)"), Isolation((), None, ISOLATION_BWRAP_NETNS)], ids=["none", "found"]
)
def test_a_session_that_finds_no_isolation_at_its_start_asks_again_at_each_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: Isolation | Unisolated
) -> None:
    given: list[object] = []
    monkeypatch.setattr(
        native_builds_module, "IsolatedBuilds", lambda planner, blobs, isolation, factories: given.append(isolation)
    )

    def choose() -> Isolation | Unisolated:
        return found

    native_builds(_configured(tmp_path), LocalBlobStore(tmp_path / "blobs"), IMAGE, isolation=choose)

    assert given == [found if isinstance(found, Isolation) else choose], (
        "asked again where the start found none, else kept"
    )


def test_the_build_service_registers_its_signing_key_and_trust_file_as_the_apps_private_paths(tmp_path: Path) -> None:
    environ = _configured(tmp_path)

    native_builds(environ, LocalBlobStore(tmp_path / "blobs"), IMAGE, isolation=lambda: Unisolated("a stand-in"))

    registered = set(confinement.PRIVATE_PATHS.paths())
    assert {Path(environ["HYPERLOOM_BUILD_SIGNING_KEY"]), Path(environ["HYPERLOOM_BUILD_TRUST"])} <= registered
