# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one-call factory: the shipped runner in a local sandbox, only the GPU measurement stubbed."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common import confinement
from hyperloom.common.outside_facts.rule_pack import PACK_ENV, load_pack
from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.archive import DirResultArchive
from hyperloom.measure.framework.base import FRAMEWORK_BASE_FILE, SNAPSHOT_ENV, load_snapshot, take_snapshot
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT, MeasurementJournal, Record
from hyperloom.measure.live_log import last_output
from hyperloom.measure.runner.tools import ToolError, ensure_inferencex
from hyperloom.measure.tests.framework_fixtures import make_framework
from hyperloom.measure.tests.gpu_free_host import idle_cards_python
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.rule_pack_fixtures import identity_of, other_pack, reading_pack  # noqa: F401
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo, _point
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    _runner_template,
    build_local_toolbox,
    host_tool_repos,
    hub_cache,
    max_parallel_tests,
    runner_plan,
)
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req, lineage_blackboard
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner


@pytest.mark.parametrize(("cards", "tp", "expected"), [(8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 2, 4), (2, 1, 2)])
def test_at_most_eight_over_tp_tests_run_at_once(cards: int, tp: int, expected: int) -> None:
    assert max_parallel_tests(cards, tp) == expected


def test_a_test_that_needs_more_cards_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot run a test of TP=4"):
        max_parallel_tests(2, 4)


def test_runner_envs_add_to_the_image_variables_rather_than_replace_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the image's ``ROCM_PATH`` and ``LD_LIBRARY_PATH`` the runner dies importing torch on an image whose ROCm
    libraries are not in the loader's path; a runner given ``runner_envs`` keeps them, and the process sees them."""
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/rocm/lib")
    options = LocalOptions(runner_envs={"INFERENCEX_REPO": "/repo", "ROCM_PATH": "/opt/rocm-own"})
    template = _runner_template(LocalSession(tmp_path, "run-1", [0]), options)
    assert template.envs["LD_LIBRARY_PATH"] == "/opt/rocm/lib"
    assert template.envs["INFERENCEX_REPO"] == "/repo"
    assert template.envs["ROCM_PATH"] == "/opt/rocm-own", "runner_envs win over the image's value"
    probe = ("/bin/sh", "-c", 'printf "%s|%s" "$ROCM_PATH" "$LD_LIBRARY_PATH" > "$HYPERLOOM_SANDBOX_ROOT/out/env"')
    provider = LocalSandboxProvider([replace(template, command=probe)], gpus=(0,), base_dir=tmp_path / "sandboxes")
    sandbox = provider.create(template.name, role="measurement", timeout_sec=60, metadata={})
    try:
        seen = _wait_for_file(sandbox, "/out/env")
    finally:
        sandbox.kill()
    assert seen == b"/opt/rocm-own|/opt/rocm/lib"


def test_the_host_tool_repositories_reach_the_runner_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local runner starts from an emptied environment, so the repositories it clones InferenceX and Magpie from are
    carried from the host's: both when the host sets both."""
    monkeypatch.setenv("INFERENCEX_REPO", "/mirrors/inferencex.git")
    monkeypatch.setenv("MAGPIE_REPO", "/mirrors/magpie.git")
    template = _runner_template(LocalSession(tmp_path, "run-1", [0]), LocalOptions())
    assert template.envs["INFERENCEX_REPO"] == "/mirrors/inferencex.git"
    assert template.envs["MAGPIE_REPO"] == "/mirrors/magpie.git"


def test_runner_envs_win_over_the_host_tool_repositories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCEX_REPO", "/mirrors/inferencex.git")
    monkeypatch.setenv("MAGPIE_REPO", "/mirrors/magpie.git")
    options = LocalOptions(runner_envs={"INFERENCEX_REPO": "/explicit/inferencex"})
    template = _runner_template(LocalSession(tmp_path, "run-1", [0]), options)
    assert template.envs["INFERENCEX_REPO"] == "/explicit/inferencex"
    assert template.envs["MAGPIE_REPO"] == "/mirrors/magpie.git", "the name runner_envs does not give is the host's"


def test_a_host_tool_repository_that_is_not_set_is_not_in_the_runner_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INFERENCEX_REPO", raising=False)
    monkeypatch.setenv("MAGPIE_REPO", "  ")
    template = _runner_template(LocalSession(tmp_path, "run-1", [0]), LocalOptions())
    assert "INFERENCEX_REPO" not in template.envs, "unset is absent, not empty"
    assert "MAGPIE_REPO" not in template.envs, "a blank value is absent too"


def test_only_the_named_tool_repositories_are_taken_from_the_host() -> None:
    environ = {"INFERENCEX_REPO": "/a", "MAGPIE_REPO": "/b", "GITHUB_TOKEN": "t", "INFERENCEX_REF": "main"}
    assert host_tool_repos(environ) == {"INFERENCEX_REPO": "/a", "MAGPIE_REPO": "/b"}
    assert host_tool_repos({}) == {}


def _wait_for_file(sandbox: Any, path: str, limit_sec: float = 30) -> bytes:
    end = time.monotonic() + limit_sec
    while True:
        try:
            return sandbox.read_file(path)
        except FileNotFoundError:
            if time.monotonic() > end:
                raise
            time.sleep(0.05)


def _app_environ(monkeypatch: pytest.MonkeyPatch, environ: dict[str, str]) -> None:
    """This process's environment is ``environ`` and ``PATH`` alone: the plan reads the app's homes, credential
    locations and hub cache from the process, the one place the masks read them from."""
    for name in list(os.environ):
        if name != "PATH":
            monkeypatch.delenv(name)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)


def test_the_runner_plan_names_what_a_runner_reads_and_never_the_session_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confined runner reads its interpreter, the trees its template's variables name, the model with what its links
    point into, and the hub cache; a repository named by URL and the image's own settings name nothing it binds."""
    session, blobs, hf = tmp_path / "session", tmp_path / "hf" / "hub" / "models--m" / "blobs", tmp_path / "hf"
    model = hf / "hub" / "models--m" / "snapshots" / "abc"
    model.mkdir(parents=True)
    blobs.mkdir(parents=True)
    (blobs / "weights-blob").write_text("weights")
    (model / "weights").symlink_to("../../blobs/weights-blob")
    envs = {
        "HYPERLOOM_FRAMEWORK_SNAPSHOT": str(session / "framework"),
        PACK_ENV: str(session / "rule-pack" / "pack.yaml"),
        "HYPERLOOM_JIT_CACHE": "/tmp/hyperloom-jit-store-0",
        "HYPERLOOM_RUNNER_CACHE": "/runner-cache",
        "HYPERLOOM_BUILD_STORE": "/stores/build",
        "HYPERLOOM_BUILD_TRUST": "/stores/trust.json",
        "INFERENCEX_REPO": "file:///mirrors/inferencex.git",
        "MAGPIE_REPO": "https://github.com/AMD-AGI/Magpie.git",
        "ROCM_PATH": "/opt/rocm",
    }
    _app_environ(monkeypatch, {"HF_HOME": str(hf)})
    plan = runner_plan(session, model, envs, "/usr/bin/python3")

    assert plan.readable == (
        Path("/usr/bin/python3"),
        session / "framework",
        session / "rule-pack" / "pack.yaml",
        Path("/tmp/hyperloom-jit-store-0"),
        Path("/runner-cache"),
        Path("/stores/build"),
        Path("/stores/trust.json"),
        Path("/mirrors/inferencex.git"),
        model,
        blobs,
        hf / "hub",
    )
    assert plan.app_only == (session,)


def test_a_model_entry_linking_out_of_the_hub_binds_what_it_links_to_and_never_the_directory_it_is_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link to a file or directory in the home binds that file or directory, never the home with every login in it,
    and a link to a directory elsewhere never binds that directory's parent."""
    home, data, model = tmp_path / "home", tmp_path / "data", tmp_path / "model"
    (home / "tokdir").mkdir(parents=True)
    (home / "config.json").write_text("{}")
    (data / "weights").mkdir(parents=True)
    model.mkdir()
    (model / "config.json").symlink_to(home / "config.json")
    (model / "tokenizer").symlink_to(home / "tokdir")
    (model / "weights").symlink_to(data / "weights")
    _app_environ(monkeypatch, {"HOME": str(home)})
    plan = runner_plan(tmp_path / "session", model, {}, "/usr/bin/python3")

    linked = set(plan.readable) - {Path("/usr/bin/python3"), model}
    assert linked == {home / "config.json", home / "tokdir", data / "weights"}, plan.readable
    assert plan.refused == ()


_HIDDEN = "a hidden entry of a home where tools keep their logins"


@pytest.mark.parametrize(
    ("name", "target", "environ", "why"),
    [
        ("keys", "home/.ssh", {}, f"in {{root}}/home/.ssh, {_HIDDEN}"),
        ("c", "home/.claude", {}, f"in {{root}}/home/.claude, {_HIDDEN}"),
        ("conf", "home/.config", {}, f"in {{root}}/home/.config, {_HIDDEN}"),
        ("a", "home/.aws", {}, f"in {{root}}/home/.aws, {_HIDDEN}"),
        ("n", "home/.netrc", {}, f"in {{root}}/home/.netrc, {_HIDDEN}"),
        ("id", "home/.ssh/id_rsa", {}, f"in {{root}}/home/.ssh, {_HIDDEN}"),
        ("h", "home", {}, "a credential location"),
        ("n", "logins/netrc", {"NETRC": "{root}/logins/netrc"}, "a credential location"),
        ("gh", "xdg/gh", {"XDG_CONFIG_HOME": "{root}/xdg"}, "below the credential location {root}/xdg"),
    ],
)
def test_a_model_entry_linking_into_a_credential_location_binds_nothing_and_the_plan_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, target: str, environ: dict[str, str], why: str
) -> None:
    """A mask hides a credential location below a bound path, never one the bound path is in: a link to the home's
    ``.ssh``, ``.claude``, ``.config``, ``.aws`` or ``.netrc``, to the home, to a login file a variable names, or below
    a credential directory a variable names, binds nothing. The plan says why, so its tests never run confined with it;
    the model's other links bind as before."""
    root = tmp_path.resolve()
    home, model = root / "home", root / "model"
    for path in ("home/.ssh/id_rsa", "home/.claude/.credentials.json", "home/.config/gh/hosts.yml", "home/.aws/x"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("SECRET")
    for path in ("home/.netrc", "logins/netrc", "xdg/gh/hosts.yml", "home/tokdir/tok.json"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("SECRET" if "tokdir" not in path else "tokenizer")
    model.mkdir()
    (model / name).symlink_to(root / target)
    (model / "tokenizer").symlink_to(home / "tokdir")
    named = {"HOME": str(home), **{key: value.format(root=root) for key, value in environ.items()}}
    _app_environ(monkeypatch, named)
    plan = runner_plan(root / "session", model, {}, "/usr/bin/python3")

    assert plan.readable == (Path("/usr/bin/python3"), model, home / "tokdir")
    assert plan.refused == (
        f"{model / name} links to {root / target}, {why.format(root=root)}, which its bind would show unmasked",
    )


def test_a_model_in_the_hub_cache_below_the_home_binds_its_blobs_and_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default hub cache is in the home's ``.cache``, which the plan binds anyway: a snapshot's links to their
    blobs, and a model elsewhere linking into the hub, bind as they do anywhere else."""
    home = tmp_path.resolve() / "home"
    hub = home / ".cache" / "huggingface" / "hub"
    blobs, snapshot = hub / "models--m" / "blobs", hub / "models--m" / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "w").write_text("weights")
    (snapshot / "weights").symlink_to("../../blobs/w")
    elsewhere = tmp_path.resolve() / "model"
    elsewhere.mkdir()
    (elsewhere / "weights").symlink_to(blobs / "w")
    _app_environ(monkeypatch, {"HOME": str(home)})
    for model in (snapshot, elsewhere):
        plan = runner_plan(tmp_path / "session", model, {}, "/usr/bin/python3")

        assert plan.refused == ()
        assert plan.readable == (Path("/usr/bin/python3"), model, blobs, hub)


#: A hub cache that is the home's parent, which the plan reads whole, masking the home in it.
_HUB_ABOVE = {"HOME": "{root}/hp/home", "HF_HUB_CACHE": "{root}/hp"}
#: A hub cache in a hidden entry of the home, its Hugging Face login beside its blobs.
_HUB_HIDDEN = {"HOME": "{root}/home2", "HF_HOME": "{root}/home2/.hf", "HF_HUB_CACHE": "{root}/home2/.hf"}


def _link_area(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Homes with logins and a tokenizer beside them: the app's (the default hub cache in its ``.cache``), the
    runner's, the passwd entry's (``pw``, made this process's), one below a hub cache and one with a hub cache in its
    ``.hf``; a runner ``CLAUDE_CONFIG_DIR``; links to the app home's ``.ssh`` and tokenizer through a link
    (``hop/a``, ``hop/tok``) and through a directory link (``hop/d``). Returns the model directory, empty."""
    for path in (
        "home/.ssh/id_rsa",
        "home/tokdir/tok.json",
        "home/.cache/pip/creds",
        "home/.cache/huggingface/hub/models--m/blobs/w",
        "rhome/.ssh/id_rsa",
        "rhome/tokdir/tok.json",
        "rcc/sub/creds",
        "pw/.ssh/id_rsa",
        "pw/tokdir/tok.json",
        "hp/home/.ssh/id_rsa",
        "hp/home/tokdir/tok.json",
        "home2/.hf/token",
        "home2/.hf/models--m/blobs/w",
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(path)
    hop = root / "hop"
    hop.mkdir()
    (hop / "b").symlink_to(root / "home" / ".ssh")
    (hop / "a").symlink_to(hop / "b")
    (hop / "d").symlink_to(root / "home" / ".ssh")
    (hop / "tok").symlink_to(root / "home" / "tokdir")
    monkeypatch.setattr(local_hosted, "_passwd_home", lambda: root / "pw")
    monkeypatch.setattr(confinement, "_passwd_home", lambda: root / "pw")
    model = root / "model"
    model.mkdir()
    return model


def _named(values: dict[str, str], root: Path) -> dict[str, str]:
    return {name: value.format(root=root) for name, value in values.items()}


@pytest.mark.parametrize(
    ("link", "named", "envs", "tree", "why"),
    [
        ("hop/a", {}, {}, "home/.ssh", f"in {{root}}/home/.ssh, {_HIDDEN}"),
        ("hop/d/id_rsa", {}, {}, "home/.ssh/id_rsa", f"in {{root}}/home/.ssh, {_HIDDEN}"),
        ("rcc/sub", {}, {"CLAUDE_CONFIG_DIR": "{root}/rcc"}, "rcc/sub", "below the credential location {root}/rcc"),
        ("rhome/.ssh", {}, {"HOME": "{root}/rhome"}, "rhome/.ssh", f"in {{root}}/rhome/.ssh, {_HIDDEN}"),
        ("pw/.ssh", {}, {}, "pw/.ssh", f"in {{root}}/pw/.ssh, {_HIDDEN}"),
        ("home/.cache/pip", {}, {}, "home/.cache/pip", f"in {{root}}/home/.cache, {_HIDDEN}"),
        ("hp/home/.ssh", _HUB_ABOVE, {}, "hp/home/.ssh", f"in {{root}}/hp/home/.ssh, {_HIDDEN}"),
        ("home2/.hf/token", _HUB_HIDDEN, {}, "home2/.hf/token", "a credential location"),
    ],
    ids=[
        "link-to-link",
        "directory-link",
        "runner-claude",
        "runner-home",
        "passwd-home",
        "beside-hub",
        "hub-above",
        "login-in-hub",
    ],
)
def test_a_model_link_into_a_credential_location_binds_nothing_however_it_gets_there(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link: str,
    named: dict[str, str],
    envs: dict[str, str],
    tree: str,
    why: str,
) -> None:
    """A link is judged by where it ends, through links to links and directory links; the runner's credential
    locations and home and the passwd entry's home count as the app's do; and a tree the plan reads whole excuses a
    link only from a credential location it is in: one above it (a hub cache that is the home's parent) masks the
    location, which the link's bind would open up, and a login in a hub cache stays a login."""
    root = tmp_path.resolve()
    model = _link_area(root, monkeypatch)
    (model / "x").symlink_to(root / link)
    (model / "tokenizer").symlink_to(root / "hop" / "tok")
    _app_environ(monkeypatch, {"HOME": str(root / "home"), **_named(named, root)})
    plan = runner_plan(root / "session", model, _named(envs, root), "/usr/bin/python3")

    reason = f"{model / 'x'} links to {root / tree}, {why.format(root=root)}, which its bind would show unmasked"
    assert plan.refused == (reason,)
    assert root / tree not in plan.readable and root / "home" / "tokdir" in plan.readable


@pytest.mark.parametrize(
    ("link", "named", "envs", "tree"),
    [
        ("hop/tok", {}, {}, "home/tokdir"),
        ("rhome/tokdir", {}, {"HOME": "{root}/rhome"}, "rhome/tokdir"),
        ("pw/tokdir", {}, {}, "pw/tokdir"),
        ("home/.cache/huggingface/hub/models--m/blobs/w", {}, {}, "home/.cache/huggingface/hub/models--m/blobs"),
        ("hp/home/tokdir", _HUB_ABOVE, {}, "hp/home/tokdir"),
        ("home2/.hf/models--m/blobs/w", _HUB_HIDDEN, {}, "home2/.hf/models--m/blobs"),
    ],
    ids=["link-to-link", "runner-home", "passwd-home", "default-hub", "hub-above", "hub-in-hidden-entry"],
)
def test_a_model_link_to_a_home_s_other_entries_or_into_a_hub_cache_in_a_home_binds_what_it_links_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link: str, named: dict[str, str], envs: dict[str, str], tree: str
) -> None:
    """Every home's entries but its hidden ones (the app's, the runner's, the passwd entry's) may be linked to, and a
    hub cache's blobs in a home's hidden entry, whose masks the plan's read of the hub cache opens up anyway."""
    root = tmp_path.resolve()
    model = _link_area(root, monkeypatch)
    (model / "x").symlink_to(root / link)
    _app_environ(monkeypatch, {"HOME": str(root / "home"), **_named(named, root)})
    plan = runner_plan(root / "session", model, _named(envs, root), "/usr/bin/python3")

    assert plan.refused == ()
    assert root / tree in plan.readable, plan.readable


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        ({"HF_HUB_CACHE": "{root}/own-hub", "HF_HOME": "{root}/hf"}, "{root}/own-hub"),
        ({"HF_HOME": "{root}/hf", "XDG_CACHE_HOME": "{root}/xdg"}, "{root}/hf/hub"),
        ({"XDG_CACHE_HOME": "{root}/xdg", "HOME": "{root}/home"}, "{root}/xdg/huggingface/hub"),
        ({"HOME": "{root}/home"}, "{root}/home/.cache/huggingface/hub"),
        ({"HOME": "{root}/nowhere"}, None),
    ],
)
def test_the_hub_cache_is_the_one_the_environment_names_and_only_where_it_exists(
    tmp_path: Path, named: dict[str, str], expected: str | None
) -> None:
    for hub in ("own-hub", "hf/hub", "xdg/huggingface/hub", "home/.cache/huggingface/hub"):
        (tmp_path / hub).mkdir(parents=True)
    environ = {name: value.format(root=tmp_path) for name, value in named.items()}
    assert hub_cache(environ) == (None if expected is None else Path(expected.format(root=tmp_path)))


def _write(path: Path, text: str, written: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (written, written))


def test_the_live_output_is_the_last_line_of_the_live_log_written_last(tmp_path: Path) -> None:
    variant = tmp_path / "variant_00_test"
    _write(variant / "warmup_round" / "benchmark_sglang_1" / "server.log", "boot\nready\n", 1000.0)
    _write(variant / "benchmark_sglang_1" / "benchmark_stderr.log", "\x1b[32mINFO\x1b[0m\t42/100\r\x07\n\n", 1100.0)
    _write(variant / "notes.log", "written last, but no live log\n", 1200.0)
    assert last_output(tmp_path, 1130.0) == (30.0, "INFO 42/100")
    _write(variant / "benchmark_sglang_1" / "benchmark_stdout.log", "x" * 1000 + "END\n", 1300.0)
    seconds, line = last_output(tmp_path, 1300.0) or (None, "")
    assert seconds == 0.0 and len(line) == 300 and line.endswith("END")
    assert last_output(tmp_path / "missing", 1300.0) is None


def test_the_live_output_is_read_from_the_sandbox_of_the_unfinished_attempt(tmp_path: Path) -> None:
    toolbox = LocalHostedToolbox.__new__(LocalHostedToolbox)
    journal = MeasurementJournal(tmp_path / "hosted")
    toolbox.backend, toolbox.sandboxes = SimpleNamespace(journal=journal), tmp_path / "sandboxes"  # type: ignore[assignment]
    _write(
        tmp_path / "sandboxes" / "lsb-1" / "fs" / "out" / "files" / "v" / "server.log", "decode batch\n", time.time()
    )
    assert toolbox._live_log("r1") is None
    journal.append(Record("r1", 1, STATE_SENT, "lsb-1", "tests/r1/attempt-1", "r1", {}))
    output = toolbox._live_log("r1")
    assert output is not None and output[1] == "decode batch" and 0.0 <= output[0] < 60.0
    journal.append(Record("r1", 1, STATE_FINISHED, "lsb-1", "tests/r1/attempt-1", "r1", {}))
    assert toolbox._live_log("r1") is None


def test_a_runner_reads_the_shared_tables_its_framework_snapshot_was_taken_from_and_none_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each test hashes the shared tables against the snapshot, to show it changed none: a confined runner reads them,
    read-only as every tree of its plan, and the session's directory, which holds the snapshot, never whole."""
    configs = make_framework(tmp_path)
    snapshot = tmp_path / "session" / "framework"
    take_snapshot(configs, snapshot)

    _app_environ(monkeypatch, {})
    taken = runner_plan(tmp_path / "session", None, {SNAPSHOT_ENV: str(snapshot)}, "/usr/bin/python3")
    none = runner_plan(tmp_path / "session", None, {SNAPSHOT_ENV: str(tmp_path / "absent")}, "/usr/bin/python3")

    assert configs in taken.readable and snapshot in taken.readable
    assert configs not in none.readable


def _require_tool_checkout(repo: Path, commit: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The precondition of the end-to-end test: the runner can check the stand-in repository out as it runs it.

    The runner fetches the tools by hash under an empty home, like a sandbox. Where that fails (no usable ``git``, a
    refused local fetch) the test would otherwise show only ``measurement_failed``, so it fails here and says why.
    """
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    monkeypatch.setenv("HOME", str(home))
    try:
        ensure_inferencex(commit, home / "cache")
    except ToolError as error:
        pytest.fail(
            f"this machine cannot check the stand-in InferenceX repository out, so no runner can start: {error}"
        )


def _toolbox_of(
    blackboard: LocalBlackboard, tmp_path: Path, session: Path, commit: str, **extra: Any
) -> LocalHostedToolbox:
    return build_local_toolbox(
        LocalSession(session, "run-1", range(8)),
        blackboard,
        AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        ),
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
            python=idle_cards_python(tmp_path),
            poll_interval_sec=0.1,
            timeout_sec=120,
            **extra,
        ),
    )


async def test_a_runner_that_cannot_start_says_why_and_leaves_no_log_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The runner fails before it writes ``out/files``: the result carries its message and saving logs stays quiet."""
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    session = tmp_path / "session"
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    envs = {"INFERENCEX_REPO": str(repo) + "-missing"}
    toolbox = _toolbox_of(blackboard, tmp_path, session, commit, runner_envs=envs, max_infra_retries=0)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        with caplog.at_level(logging.DEBUG):
            await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
        assert result.failure_reason and "runner_start_failed" in result.failure_reason
        assert "git fetch failed" in result.failure_reason, result.failure_reason
        assert result.sandbox and f"(sandbox {result.sandbox.sandbox_id} on node " in result.failure_reason
        journal = [json.loads(line) for line in (session / "hosted" / "measurements.jsonl").read_text().splitlines()]
        status = [line["runner_status"] for line in journal if line["state"] == "finished"]
        assert [s["error_class"] for s in status] == ["runner_start_failed"], journal
        assert "git fetch failed" in status[0]["message"], "the runner's own message is in the journal line"
        assert "could not be saved" not in caplog.text, "a runner with no out/files has no logs: that is not a warning"
    finally:
        await toolbox.close()


@pytest.mark.agent_isolation
async def test_the_toolbox_s_test_sandboxes_run_in_bubblewrap_where_it_can_sandbox_them(tmp_path: Path) -> None:
    usable_bwrap()
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(blackboard, tmp_path, tmp_path / "session", "0" * 40)
    try:
        sandbox = toolbox.provider.create("runner-1", role="measurement", timeout_sec=60, metadata={})
        confined = sandbox._confined is not None
        sandbox.kill()
    finally:
        await toolbox.close()

    assert confined, "the runner and the server it starts run in bubblewrap"


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is not None,
    reason="proves the wiring through the runner's import failure; with sglang installed the server fails otherwise",
)
async def test_the_factory_builds_a_toolbox_that_runs_a_test_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    session = tmp_path / "session"
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(blackboard, tmp_path, session, commit, runner_envs={"INFERENCEX_REPO": str(repo)})
    try:
        assert toolbox._queue.policy.global_concurrency == 8, "the policy is held to 8 / TP"
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        # No GPU here, so the shipped runner cannot serve a model: what this proves is the wiring. The real runner
        # started from the local template, signed a bundle, the backend verified it, and the toolbox took the cause
        # the server log names, quoted (no sglang module) as the candidate's failure, did not rerun it and wrote it to the
        # blackboard.
        assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
        assert result.failure_reason is not None and result.failure_reason.startswith("import_error: ")
        assert "No module named 'sglang'" in result.failure_reason, (
            "the reason quotes the line the server log ends with"
        )
        assert len(result.attempts) == 1 and result.manifest_sha256, result
        sandboxes = {rec.sandbox_id for rec in toolbox.backend.journal.records() if rec.request_id == request_id}
        assert len(sandboxes) == 1
        record = await blackboard.record(request_id)
        assert record.result is not None and record.result.manifest_sha256 == result.manifest_sha256
        assert toolbox.backend.journal.latest(request_id).extra["verified"] is True  # type: ignore[union-attr]
    finally:
        await toolbox.close()


async def test_every_test_of_the_session_reads_private_copies_of_the_tables_the_toolbox_snapshotted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    configs = make_framework(tmp_path)
    session = tmp_path / "session"
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(
        blackboard,
        tmp_path,
        session,
        commit,
        runner_envs={"INFERENCEX_REPO": str(repo)},
        framework_configs=str(configs),
        max_infra_retries=0,
    )
    try:
        snapshot = load_snapshot(session / "framework")
        assert snapshot.source == configs and "AITER_CONFIG_GEMM" in snapshot.variables
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        landed = json.loads((session / result.output_dir / FRAMEWORK_BASE_FILE).read_text())
        assert landed["isolation"] == "private_copies" and landed["matches_session"] is True
        assert landed["session_sha256"] == snapshot.sha256 and landed["framework_base_changed"] is False
        assert "AITER_CONFIG_GEMM" in landed["redirected"]
    finally:
        await toolbox.close()


async def test_a_session_that_names_a_rule_pack_gives_its_sandboxes_a_staged_copy_and_one_that_does_not_gives_none(
    tmp_path: Path, reading_pack
) -> None:
    blackboard = lineage_blackboard(tmp_path / "blackboard", ("branch-named", "branch-reference"))
    named = other_pack(tmp_path)
    for pack, session in ((named, tmp_path / "named"), (None, tmp_path / "reference")):
        reading_pack(pack)
        toolbox = _toolbox_of(blackboard, tmp_path, session, "a" * 40)
        try:
            env = (await toolbox.lease_branch_sandbox(f"branch-{session.name}", 1)).env
        finally:
            await toolbox.close()
        if pack is None:
            assert PACK_ENV not in env and not (session / "rule-pack").exists()
        else:
            staged = Path(env[PACK_ENV])
            assert staged.parent == (session / "rule-pack").resolve() and staged != named
            assert load_pack(staged).digest == identity_of(named)["digest"]


async def test_every_test_of_a_session_runs_with_the_rule_pack_the_session_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reading_pack
) -> None:
    """The runner starts from an emptied environment: the pack reaches it through the template, its signed manifest
    states the digest it read, and the backend accepted the result only because that is the session's."""
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    named = other_pack(tmp_path)
    reading_pack(named)
    session = tmp_path / "session"
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(
        blackboard, tmp_path, session, commit, runner_envs={"INFERENCEX_REPO": str(repo)}, max_infra_retries=0
    )
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        assert result.manifest_sha256
        assert toolbox.backend.journal.latest(request_id).extra["verified"] is True  # type: ignore[union-attr]
        archived = DirResultArchive(session / "archive").bundle_path(request_id, result.manifest_sha256)
        manifest = json.loads((archived / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["runner"]["rule_pack"] == identity_of(named)
    finally:
        await toolbox.close()


async def test_a_result_of_a_runner_that_read_another_pack_than_the_session_is_not_a_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reading_pack
) -> None:
    """The template is told another pack than the one the session read (here through ``runner_envs``): the runner
    states what it read, and the toolbox refuses to take a result of other facts as the session's."""
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    other = other_pack(tmp_path)
    reading_pack(None)
    session = tmp_path / "session"
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    envs = {"INFERENCEX_REPO": str(repo), PACK_ENV: str(other)}
    toolbox = _toolbox_of(blackboard, tmp_path, session, commit, runner_envs=envs, max_infra_retries=0)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
        assert result.failure_reason and "(rule_pack_mismatch)" in result.failure_reason, result.failure_reason
        assert identity_of(other)["digest"] in result.failure_reason
        assert result.manifest_sha256 is None, "nothing of the refused bundle is recorded as the test's result"
        extra = toolbox.backend.journal.latest(request_id).extra  # type: ignore[union-attr]
        assert extra["reject"] == "rule_pack_mismatch" and "verified" not in extra
    finally:
        await toolbox.close()
