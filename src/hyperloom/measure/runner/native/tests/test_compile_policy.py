# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bubblewrap sandbox of a compile, as a policy and the command line it becomes: only the work directory writable,
the toolchain, the child's source tree, the image's packages and the upstream build directory (where they are not laid
in the work directory) and the image's directories a laid farm links into read-only, the CPU tree of ``/sys``, the base
``/dev`` nodes and no GPU, a network of its own, and the signing key and every credential location of the app absent or
masked; and a compile bubblewrap cannot sandbox fails as the machine's, running nothing."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_common.bwrap import BwrapPolicyError, BwrapProbe, interpreter_roots

from hyperloom.common import confinement
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.common.confinement import PrivatePaths
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.builder import BuildMachine, LaidTree
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_BWRAP_NETNS,
    SOURCE_ROOT,
    ChildInterpreter,
    IsolatedBuilds,
    Isolation,
    compile_policy,
)
from hyperloom.measure.runner.native.tests._native import BUILDER_IMAGE, CountingPreflight, fake_recipe
from hyperloom.measure.tests._recipes import IMAGE

PROBE = BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> PrivatePaths:
    """The app's private paths, empty for this test."""
    fresh = PrivatePaths()
    monkeypatch.setattr(confinement, "PRIVATE_PATHS", fresh)
    return fresh


def _laid(root: Path) -> LaidTree:
    """A laid tree whose build runs an image interpreter of its own (a link to this one in a directory nothing else
    binds)."""
    for name in ("work", "image/site-packages", "sources/fakepkg/native", "image-python/bin"):
        (root / name).mkdir(parents=True)
    (root / "image-python/bin/python3").symlink_to(sys.executable)
    machine = BuildMachine(IMAGE, BUILDER_IMAGE, "gfx942", str(root / "image-python/bin/python3"))
    recipe = fake_recipe(CountingPreflight())
    return LaidTree(recipe, machine, root / "sources/fakepkg/native", root / "image/site-packages", frozenset())


def _environment(work: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(work), "TMPDIR": str(work), "LC_ALL": "C.UTF-8"}


def _pairs(argv: list[str]) -> set[tuple[str, ...]]:
    return {tuple(argv[index : index + 3]) for index in range(len(argv))}


def _at(argv: list[str], flag: str, value: str) -> int:
    return next(index for index in range(len(argv) - 1) if argv[index : index + 2] == [flag, value])


def test_the_compile_writes_only_its_work_and_reads_its_toolchain_source_and_image(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert policy.writable == (str(work),)
    read_only = set(policy.read_only)
    assert {str(SOURCE_ROOT), str(laid.image_root), str(laid.tracked)} <= read_only
    for root in ("/usr", "/etc"):  # bound entry by entry where the environment names a credential file in it
        assert root in read_only or any(path.startswith(f"{root}/") for path in read_only), root
    assert set(interpreter_roots()) <= read_only, "the child's interpreter, wherever it is installed"
    image_bin = tmp_path / "image-python/bin"
    assert any(image_bin == Path(path) or Path(path) in image_bin.parents for path in read_only), "the image's python"
    assert not any(path == "/proc" or path.startswith(("/proc/", "/sys", "/dev")) for path in read_only)
    assert (policy.devices, policy.sys_paths, policy.tmp, policy.shm) == ((), ("/sys/devices/system/cpu",), None, None)
    assert (policy.own_network, policy.die_with_parent, policy.chdir) == (True, True, str(work))


def test_its_command_line_has_its_own_network_dies_with_its_parent_and_binds_no_gpu(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"
    child = ChildInterpreter()

    argv = compile_policy(work, child, laid, _environment(work)).argv(PROBE, child.command(work))

    flags = argv[: argv.index("--")]
    assert argv[0] == "/usr/bin/bwrap" and argv[argv.index("--") + 1 :] == child.command(work)
    assert {"--unshare-net", "--die-with-parent", "--unshare-pid", "--unshare-user", "--new-session"} <= set(flags)
    assert _at(flags, "--cap-drop", "ALL") >= 0
    pairs = _pairs(flags)
    assert ("--bind", str(work), str(work)) in pairs
    assert ("--ro-bind-try", "/sys/devices/system/cpu", "/sys/devices/system/cpu") in pairs
    assert ("--dev-bind-try", "/dev/null", "/dev/null") in pairs
    dev, null = _at(flags, "--tmpfs", "/dev"), _at(flags, "--dev-bind-try", "/dev/null")
    assert dev < null, "the base nodes are bound into a fresh /dev"
    assert "--dev" not in flags and not any("kfd" in arg or "/dev/dri" in arg for arg in flags)
    assert ("--ro-bind-try", "/sys", "/sys") not in pairs


def test_the_signing_key_is_absent_where_a_bound_tree_holds_it_too(tmp_path: Path, registered: PrivatePaths) -> None:
    laid, work = _laid(tmp_path), tmp_path / "work"
    outside, inside = tmp_path / "service.key", laid.image_root / "service.key"
    for key in (outside, inside):
        key.write_text("not a key")
        registered.add(key, "the build service's signing key")
    (laid.image_root / "package").mkdir()

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert not _shown(inside, policy) and _shown(laid.image_root / "package", policy), "the image is bound beside it"
    bound = (*policy.read_only, *policy.writable)
    assert not any(str(outside) == path or str(outside).startswith(path.rstrip("/") + "/") for path in bound)
    assert str(outside) not in policy.masked and str(inside) not in policy.masked, "a file is left out, not masked"


def test_a_path_the_compile_reads_through_its_own_work_is_refused(tmp_path: Path, registered: PrivatePaths) -> None:
    """Named below ``work`` but a link there to elsewhere: not part of the laid tree, so never left to ``work``'s bind
    (where it would dangle, and the compile fail as the candidate's) and never bound at its target."""
    laid, work = _laid(tmp_path), tmp_path / "work"
    (work / "image").symlink_to(laid.image_root)
    reached = LaidTree(laid.recipe, laid.machine, laid.tracked, work / "image", frozenset())

    with pytest.raises(BwrapPolicyError, match="reached through a tree the agent writes"):
        compile_policy(work, ChildInterpreter(), reached, _environment(work))


def _shown(path: Path, policy) -> bool:
    """Whether ``path`` is at or below a bind of ``policy`` and not at or below one of its masks."""
    text = str(path)

    def under(roots) -> bool:
        return any(text == root or text.startswith(root.rstrip("/") + "/") for root in roots)

    return under((*policy.read_only, *policy.writable)) and not under(policy.masked)


def _farm(root: Path) -> tuple[LaidTree, Path]:
    """A tree laid as aiter's module build lays it: the farm, its JIT directory and its upstream tree in ``work``, its
    untouched files links into the image's aiter (a directory nothing else binds)."""
    laid, work = _laid(root), root / "work"
    image = root / "image-aiter"
    (image / "aiter/jit").mkdir(parents=True)
    (image / "aiter/jit/core.py").write_text("# the image's\n")
    site = work / "tree/site"
    (site / "aiter/jit").mkdir(parents=True)
    (site / "aiter/jit/core.py").symlink_to(image / "aiter/jit/core.py")
    farm = replace(laid, tracked=work / "tree", image_root=site / "aiter/jit", linked=(image / "aiter", image))
    return farm, image


def test_a_farm_laid_in_the_work_directory_is_its_own_and_the_image_it_links_to_is_read_only(
    tmp_path: Path, registered: PrivatePaths
) -> None:
    laid, image = _farm(tmp_path)
    work = tmp_path / "work"

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert policy.writable == (str(work),)
    assert not [path for path in policy.read_only if path == str(work) or path.startswith(f"{work}/")]
    assert str(image) in policy.read_only, "the farm's links resolve inside"
    assert _shown(image / "aiter/jit/core.py", policy)


def test_a_credential_location_the_apps_own_environment_names_is_hidden_where_a_bound_tree_holds_it(
    tmp_path: Path, registered: PrivatePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compile's environment names no credential: the app's does, and its locations are masked all the same."""
    laid, work = _laid(tmp_path), tmp_path / "work"
    aws, hf = laid.image_root / "aws-credentials", laid.image_root / "hf"
    aws.write_text("SECRET aws")
    (hf / "hub").mkdir(parents=True)
    (hf / "token").write_text("SECRET hf")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(aws))
    monkeypatch.setenv("HF_HOME", str(hf))

    policy = compile_policy(work, ChildInterpreter(), laid, _environment(work))

    assert not {str(aws), str(hf / "token")} & set(policy.masked), "files are left out of the tree, not masked"
    assert not _shown(aws, policy) and not _shown(hf / "token", policy) and _shown(hf / "hub", policy)


def test_a_path_entry_that_is_the_apps_home_does_not_show_the_home(
    tmp_path: Path, registered: PrivatePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compile runs with the app's ``PATH``; an entry that is the home itself would show all of it, logins
    included, unmasked; a directory of programs below the home is bound as before."""
    laid, work = _laid(tmp_path), tmp_path / "work"
    home = tmp_path / "home"
    for login in (".ssh/id_ed25519", ".config/gh/hosts.yml"):
        (home / login).parent.mkdir(parents=True)
        (home / login).write_text("SECRET")
    (home / "bin").mkdir()
    monkeypatch.setenv("HOME", str(home))
    environ = {**_environment(work), "PATH": f"{home}:{home / 'bin'}:{os.environ.get('PATH', os.defpath)}"}

    policy = compile_policy(work, ChildInterpreter(), laid, environ)

    assert str(home) not in policy.read_only and str(home / "bin") in policy.read_only
    assert not _shown(home / ".ssh/id_ed25519", policy) and not _shown(home / ".config/gh/hosts.yml", policy)


class _Planner:
    """Plans one build whose tree ``lay`` lays."""

    def __init__(self, lay) -> None:
        self.lay = lay

    def planned(self, inputs: object) -> _Planner:
        return self


def test_a_compile_bubblewrap_cannot_sandbox_fails_as_the_machines_and_nothing_runs(
    tmp_path: Path, registered: PrivatePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy's refusal is the build's: never a compile started some other way."""
    laid = _laid(tmp_path)

    def lay(blobs: object, work: Path) -> LaidTree:
        (work / "image").symlink_to(laid.image_root)
        return replace(laid, image_root=work / "image")

    monkeypatch.setattr(isolated, "run_in_group", lambda *args, **kwargs: pytest.fail(f"a compile ran: {args}"))
    sandbox = Isolation((), None, ISOLATION_BWRAP_NETNS, BwrapIsolation(PROBE))
    builds = IsolatedBuilds(_Planner(lay), None, sandbox, {laid.recipe.component: "unused:factory"})
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(BuildInfraError) as refused:
        builds(object())(scratch)

    message = str(refused.value)
    assert message.startswith("bubblewrap cannot sandbox this compile: ") and message.endswith("; nothing was compiled")
    assert f"{scratch}/work/image is reached through a tree the agent writes" in message


def test_under_bubblewrap_the_compile_is_given_only_its_own_environment(
    tmp_path: Path, registered: PrivatePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under bubblewrap the compile is this process's own user: none of this process's variables may reach it."""
    laid = _laid(tmp_path)
    monkeypatch.setenv("HYPERLOOM_TEST_SERVICE_TOKEN", "SECRET")
    seen: list[dict[str, str]] = []

    def run(command, **kwargs):
        seen.append(dict(kwargs["env"]))
        raise isolated.subprocess.TimeoutExpired(command, 0)

    monkeypatch.setattr(isolated, "run_in_group", run)
    sandbox = Isolation((), None, ISOLATION_BWRAP_NETNS, BwrapIsolation(PROBE))
    builds = IsolatedBuilds(
        _Planner(lambda blobs, work: laid), None, sandbox, {laid.recipe.component: "unused:factory"}
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(BuildInfraError, match="did not end in time"):
        builds(object())(scratch)

    (given,) = seen
    assert sorted(given) == ["HOME", "LC_ALL", "PATH", "TMPDIR"], "names only: a failure must not print the values"
    work = scratch / "work"
    assert given == {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(work),
        "TMPDIR": str(work),
        "LC_ALL": "C.UTF-8",
    }
