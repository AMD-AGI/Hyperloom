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
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers
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


def test_a_model_linking_into_the_home_shows_its_test_what_the_links_name_and_none_of_the_home_s_logins(
    area: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usable_bwrap()
    home, model = _home_and_model(area)
    data = area / "data"
    monkeypatch.setenv("HOME", str(home))
    session = area / "session"
    plan = runner_plan(session, model, {}, sys.executable, {"HOME": str(home)})
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
    template = LocalTemplate(
        "runner",
        "measurement",
        command=(sys.executable, "-I", "-S", "-c", READER, json.dumps(reads)),
        sealed=True,
        signing=True,
    )
    provider = LocalSandboxProvider([template], base_dir=session / "sandboxes", users=ProcessUsers(plan=plan))
    try:
        sandbox = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        until = time.monotonic() + 60
        while True:
            try:
                seen = json.loads(sandbox.read_file("/out/seen.json"))
                break
            except FileNotFoundError:
                assert time.monotonic() < until, "the test never wrote what it read"
                time.sleep(0.05)
    finally:
        provider.close()

    assert plan.refused == ()
    assert sandbox.confined
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
    plan = runner_plan(session, model, {}, sys.executable, {"HOME": str(home)})
    template = LocalTemplate(
        "runner", "measurement", command=(sys.executable, "-c", "import time; time.sleep(600)"), sealed=True
    )
    provider = LocalSandboxProvider([template], base_dir=session / "sandboxes", users=ProcessUsers(plan=plan))
    try:
        with caplog.at_level(logging.WARNING):
            sandbox = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        sandbox.kill()
        monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process(hosted=True, layer_bwrap=probe.path))
        with pytest.raises(SandboxDenied, match="must run confined") as refused:
            provider.create("runner", role="measurement", timeout_sec=120, metadata={})
    finally:
        provider.close()

    why = f"{model / name} links to {home / login}, in {home / login}, a hidden entry of a home"
    assert not sandbox.confined
    assert "runs unconfined, as before" in caplog.text and why in caplog.text
    assert why in str(refused.value)
    assert home / login not in plan.readable and home / "tokdir" in plan.readable
