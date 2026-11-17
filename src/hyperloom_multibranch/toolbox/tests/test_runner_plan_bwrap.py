# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's runner plan (:func:`~hyperloom_multibranch.toolbox.local_hosted.runner_plan`) under real bubblewrap:
a test whose model links into the app's home reads what the links name and none of the home's logins, and one whose
model links to a login there never runs confined with it: as before with a WARNING, or refused in a hosted run."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.common import agent_isolation
from hyperloom.sandbox import confined
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers, RunnerPlan
from hyperloom.sandbox.socket import SandboxDenied
from hyperloom_common.tests._bwrap_host import unavailable, usable_bwrap
from hyperloom_multibranch.toolbox.local_hosted import runner_plan

pytestmark = pytest.mark.agent_isolation

#: Reads each path its argument names and writes what it got to ``/out/seen.json``.
READER = r"""
import errno, json, os, sys
def outcome(path):
    try:
        with open(path) as handle:
            return "ok:" + handle.read()[:16]
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
out = os.path.join(os.environ["HYPERLOOM_SANDBOX_ROOT"], "out")
with open(os.path.join(out, "seen.tmp"), "w") as handle:
    json.dump({name: outcome(path) for name, path in json.loads(sys.argv[1]).items()}, handle)
os.rename(os.path.join(out, "seen.tmp"), os.path.join(out, "seen.json"))
"""


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(confined, "_said", set())
    monkeypatch.delenv("HYPERLOOM_STATE_DIR", raising=False)


@pytest.fixture
def area() -> Iterator[Path]:
    """Outside ``/tmp``, below which a bubblewrap before 0.12 binds nothing."""
    if not os.access("/var/tmp", os.W_OK):
        unavailable("no writable /var/tmp to hold the session outside /tmp")
    made = Path(tempfile.mkdtemp(prefix="hl-test-plan-", dir="/var/tmp")).resolve()
    yield made
    shutil.rmtree(made, ignore_errors=True)


def _hub_from_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan's hub cache is the default one below ``HOME``: no variable of this process's names another."""
    for name in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)


def _home_and_model(area: Path) -> tuple[Path, Path]:
    """A home with logins, a file and a directory a model may link to, and the default hub cache (in the home's
    ``.cache``, beside the hub's login); a model directory whose links lead to those, to data beside the home, and to
    a blob of the hub."""
    home, data, model = area / "home", area / "data", area / "model"
    hub = home / ".cache" / "huggingface" / "hub"
    for path, text in (
        (home / ".claude" / ".credentials.json", "SECRET-LOGIN"),
        (home / ".ssh" / "id_rsa", "SECRET-SSH"),
        (home / ".netrc", "SECRET-NETRC"),
        (home / ".config" / "gh" / "hosts.yml", "SECRET-GH"),
        (home / ".aws" / "credentials", "SECRET-AWS"),
        (home / ".cache" / "huggingface" / "token", "SECRET-HF"),
        (hub / "models--m" / "blobs" / "w", "hub-weights"),
        (home / "config.json", "model-config"),
        (home / "tokdir" / "tok.json", "tokenizer"),
        (data / "weights" / "w.bin", "weights"),
        (data / "other.txt", "beside-weights"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    model.mkdir()
    (model / "config.json").symlink_to(home / "config.json")
    (model / "tokenizer").symlink_to(home / "tokdir")
    (model / "weights").symlink_to(data / "weights")
    (model / "shard").symlink_to(hub / "models--m" / "blobs" / "w")
    return home, model


def _read_in_sandbox(plan: RunnerPlan, session: Path, reads: dict[str, str]) -> tuple[bool, dict[str, str]]:
    """Whether a sealed test sandbox of ``plan`` ran confined, and what its process got reading each of ``reads``."""
    template = LocalTemplate(
        "runner",
        "measurement",
        command=(sys.executable, "-I", "-S", "-c", READER, json.dumps(reads)),
        sealed=True,
        signing=True,
    )
    with LocalSandboxProvider([template], base_dir=session / "sandboxes", users=ProcessUsers(plan=plan)) as provider:
        sandbox = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        until = time.monotonic() + 60
        while True:
            try:
                return sandbox._confined is not None, json.loads(sandbox.read_file("/out/seen.json"))
            except FileNotFoundError:
                assert time.monotonic() < until, "the test never wrote what it read"
                time.sleep(0.05)


def _refused_twice(
    plan: RunnerPlan, session: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bwrap: str
) -> tuple[bool, str]:
    """Whether a test sandbox of ``plan`` ran confined, said at WARNING, and why a hosted run refused one."""
    template = LocalTemplate(
        "runner", "measurement", command=(sys.executable, "-c", "import time; time.sleep(600)"), sealed=True
    )
    with LocalSandboxProvider([template], base_dir=session / "sandboxes", users=ProcessUsers(plan=plan)) as provider:
        with caplog.at_level(logging.WARNING):
            sandbox = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        sandbox.kill()
        monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=bwrap))
        with pytest.raises(SandboxDenied, match="must run confined") as refused:
            provider.create("runner", role="measurement", timeout_sec=120, metadata={})
    return sandbox._confined is not None, str(refused.value)


def test_a_model_linking_into_the_home_shows_its_test_what_the_links_name_and_none_of_the_home_s_logins(
    area: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usable_bwrap()
    home, model = _home_and_model(area)
    data = area / "data"
    monkeypatch.setenv("HOME", str(home))
    session = area / "session"
    _hub_from_home(monkeypatch)
    plan = runner_plan(session, model, {}, sys.executable)
    reads = {
        "config": str(model / "config.json"),
        "tokenizer": str(model / "tokenizer" / "tok.json"),
        "weights": str(model / "weights" / "w.bin"),
        "shard": str(model / "shard"),
        "login": str(home / ".claude" / ".credentials.json"),
        "ssh": str(home / ".ssh" / "id_rsa"),
        "netrc": str(home / ".netrc"),
        "gh": str(home / ".config" / "gh" / "hosts.yml"),
        "aws": str(home / ".aws" / "credentials"),
        "hf_login": str(home / ".cache" / "huggingface" / "token"),
        "through_link": str(model / "tokenizer" / ".." / ".netrc"),
        "beside_weights": str(data / "other.txt"),
    }
    confined_, seen = _read_in_sandbox(plan, session, reads)

    assert plan.refused == ()
    assert confined_
    assert seen == {
        "config": "ok:model-config",
        "tokenizer": "ok:tokenizer",
        "weights": "ok:weights",
        "shard": "ok:hub-weights",
        "login": "ENOENT",
        "ssh": "ENOENT",
        "netrc": "ENOENT",
        "gh": "ENOENT",
        "aws": "ENOENT",
        "hf_login": "ENOENT",
        "through_link": "ENOENT",
        "beside_weights": "ENOENT",
    }


@pytest.mark.parametrize(
    ("name", "login"), [("keys", ".ssh"), ("c", ".claude"), ("conf", ".config"), ("a", ".aws"), ("n", ".netrc")]
)
def test_a_model_linking_to_a_login_in_the_home_never_runs_its_test_confined_with_it(
    area: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: str, login: str
) -> None:
    """A mask hides a login below a bound path, never one the bound path is in: a model entry linking to the home's
    ``.ssh``, ``.claude``, ``.config``, ``.aws`` or ``.netrc`` would show it whole. The plan refuses the link, so its
    test runs as before, said at WARNING, and a hosted run refuses it, though bubblewrap could sandbox it."""
    probe = usable_bwrap()
    home, model = _home_and_model(area)
    (model / name).symlink_to(home / login)
    monkeypatch.setenv("HOME", str(home))
    session = area / "session"
    _hub_from_home(monkeypatch)
    plan = runner_plan(session, model, {}, sys.executable)
    confined_, refused = _refused_twice(plan, session, monkeypatch, caplog, probe.path)

    why = f"{model / name} links to {home / login}, in {home / login}, a hidden entry of a home"
    assert not confined_
    assert "runs unconfined, as before" in caplog.text and why in caplog.text
    assert why in refused
    assert home / login not in plan.readable and home / "tokdir" in plan.readable


def test_a_hub_cache_above_the_home_masks_the_home_s_logins_and_a_model_link_to_one_never_runs_confined(
    area: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The plan reads the hub cache whole, here the home's parent, and masks the home's logins in it; a model entry
    linking to the home's ``.ssh`` would open that mask up around it. Without that link the test runs confined and
    reads the tokenizer the model links to, and no login; with it the plan refuses the link, so its test runs as
    before, said at WARNING, and a hosted run refuses it."""
    probe = usable_bwrap()
    hub = area / "hp"
    home, model, session = hub / "home", area / "model", area / "session"
    for path, text in ((home / ".ssh" / "id_rsa", "SECRET-SSH"), (home / "tokdir" / "tok.json", "tokenizer")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    model.mkdir()
    (model / "tokenizer").symlink_to(home / "tokdir")
    monkeypatch.setenv("HOME", str(home))
    _hub_from_home(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    plan = runner_plan(session, model, {}, sys.executable)
    reads = {"tokenizer": str(model / "tokenizer" / "tok.json"), "ssh": str(home / ".ssh" / "id_rsa")}
    confined_, seen = _read_in_sandbox(plan, session, reads)

    assert plan.refused == () and hub in plan.readable
    assert confined_
    # Hugging Face's login not made yet below the home hides the first missing path on its way, so the home in the hub
    # is bound entry by entry, and its .ssh is left out of it rather than masked.
    assert seen == {"tokenizer": "ok:tokenizer", "ssh": "ENOENT"}

    (model / "keys").symlink_to(home / ".ssh")
    plan = runner_plan(session, model, {}, sys.executable)
    confined_, refused = _refused_twice(plan, area / "session-2", monkeypatch, caplog, probe.path)

    why = f"{model / 'keys'} links to {home / '.ssh'}, in {home / '.ssh'}, a hidden entry of a home"
    assert plan.refused == (f"{why} where tools keep their logins, which its bind would show unmasked",)
    assert not confined_
    assert "runs unconfined, as before" in caplog.text and why in caplog.text
    assert why in refused
    assert home / ".ssh" not in plan.readable
