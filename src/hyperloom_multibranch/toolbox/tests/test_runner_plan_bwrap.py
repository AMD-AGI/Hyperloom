# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's runner plan (:func:`~hyperloom_multibranch.toolbox.local_hosted.runner_plan`) under real bubblewrap:
a test whose model links into the app's home reads what the links name and none of the home's logins."""

from __future__ import annotations

import json
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


def test_a_model_linking_into_the_home_shows_its_test_what_the_links_name_and_none_of_the_home_s_logins(
    area: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usable_bwrap()
    home, data, model = area / "home", area / "data", area / "model"
    for path, text in (
        (home / ".claude" / ".credentials.json", "SECRET-LOGIN"),
        (home / ".ssh" / "id_rsa", "SECRET-SSH"),
        (home / ".netrc", "SECRET-NETRC"),
        (home / ".config" / "gh" / "hosts.yml", "SECRET-GH"),
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
    monkeypatch.setenv("HOME", str(home))
    session = area / "session"
    plan = runner_plan(session, model, {}, sys.executable, {"HOME": str(home)})
    reads = {
        "config": str(model / "config.json"),
        "tokenizer": str(model / "tokenizer" / "tok.json"),
        "weights": str(model / "weights" / "w.bin"),
        "login": str(home / ".claude" / ".credentials.json"),
        "ssh": str(home / ".ssh" / "id_rsa"),
        "netrc": str(home / ".netrc"),
        "gh": str(home / ".config" / "gh" / "hosts.yml"),
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

    assert sandbox.confined
    assert seen == {
        "config": "ok:model-config",
        "tokenizer": "ok:tokenizer",
        "weights": "ok:weights",
        "login": "ENOENT",
        "ssh": "ENOENT",
        "netrc": "ENOENT",
        "gh": "ENOENT",
        "through_link": "ENOENT",
        "beside_weights": "ENOENT",
    }
