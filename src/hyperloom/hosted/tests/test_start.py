# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""start.sh of the hosting layer: what it refuses on an image, and what it hands app_bootstrap.sh."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SH = shutil.which("sh") or "sh"
START = Path(__file__).resolve().parent.parent / "start.sh"
_CERT = "-----BEGIN CERTIFICATE-----\nQUJD\n-----END CERTIFICATE-----\n"


def _tool(path: Path, name: str, body: str = "exit 0") -> None:
    path.mkdir(exist_ok=True)
    (path / name).write_text(f"#!/bin/sh\n{body}\n")
    (path / name).chmod(0o755)


@pytest.fixture
def layer(tmp_path) -> Path:
    """A layer directory whose bootstrap records what it was started with."""
    root = tmp_path / "layer"
    root.mkdir()
    shutil.copy(START, root / "start.sh")
    _tool(
        root,
        "app_bootstrap.sh",
        'echo "PATH=$PATH" >"$CALLS.boot"; env | grep -E "^(SSL_CERT_FILE|NODE_EXTRA_CA_CERTS)=" >>"$CALLS.boot" || true',
    )
    _tool(root / "bin", "claude", "echo 1.0.0")
    return root


@pytest.fixture
def image(tmp_path) -> Path:
    """The tools of an image as stubs: root, git, setpriv, bash and the account tools, each recording its call."""
    tools = tmp_path / "tools"
    for name in ("git", "setpriv"):
        _tool(tools, name)
    _tool(tools, "bash", 'exec /bin/bash "$@"')
    _tool(tools, "id", "echo 0")
    _tool(tools, "getent", 'echo "getent $*" >>"$CALLS"; exit 2')
    for name in ("groupadd", "useradd"):
        _tool(tools, name, f'echo "{name} $*" >>"$CALLS"')
    return tools


def _run(tmp_path, layer, image, **env):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    full = {
        "PATH": f"{image}:/usr/bin:/bin",
        "CALLS": str(tmp_path / "calls"),
        "HYPERLOOM_STATE_DIR": str(state),
        **env,
    }
    done = subprocess.run([SH, str(layer / "start.sh")], capture_output=True, text=True, env=full, timeout=60)
    calls = tmp_path / "calls"
    return done, (calls.read_text().splitlines() if calls.exists() else [])


@pytest.mark.parametrize("missing", ["git", "setpriv"])
def test_an_image_without_git_or_setpriv_is_refused_by_name(tmp_path, layer, image, missing):
    (image / missing).unlink()
    # Keep the real tool out of reach: only the stub dir and a dir with the bare essentials are searched.
    essentials = tmp_path / "essentials"
    essentials.mkdir()
    for name in ("cat", "cut", "dirname", "mv", "chmod", "ls"):
        found = shutil.which(name)
        if found:
            (essentials / name).symlink_to(found)
    done, _ = _run(tmp_path, layer, image, PATH=f"{image}:{essentials}")
    assert done.returncode == 1 and f"has no {missing}" in done.stderr, done.stderr


def test_not_root_is_refused(tmp_path, layer, image):
    _tool(image, "id", "echo 1000")
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "must run as root" in done.stderr and not calls


def test_a_uid_taken_by_another_name_is_refused(tmp_path, layer, image):
    _tool(
        image,
        "getent",
        'case "$1" in passwd) [ "$2" = 2000 ] && echo "someone:x:2000:2000::/h:/bin/sh" && exit 0;; esac; exit 2',
    )
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "uid 2000 is taken by the account someone" in done.stderr


def test_a_name_with_another_uid_is_refused(tmp_path, layer, image):
    _tool(
        image,
        "getent",
        'case "$1 $2" in "passwd hl-specialist") echo "hl-specialist:x:999:999::/h:/bin/sh"; exit 0;; esac; exit 2',
    )
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "exists with a uid other than 2000" in done.stderr


def test_a_layer_without_the_cli_is_refused(tmp_path, layer, image):
    (layer / "bin" / "claude").unlink()
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "no claude CLI" in done.stderr


def test_the_account_is_created_and_the_bootstrap_gets_the_cli_and_the_ca(tmp_path, layer, image):
    (layer / "ca.pem").write_text(_CERT)
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert "groupadd --gid 2000 hl-specialist" in calls
    assert any(
        c.startswith("useradd ") and "--uid 2000" in c and "--gid 2000" in c and c.endswith("hl-specialist")
        for c in calls
    )
    boot = (tmp_path / "calls.boot").read_text().splitlines()
    assert boot[0].startswith(f"PATH={layer}/bin:")
    bundle = tmp_path / "state" / "ca-bundle.pem"
    assert f"SSL_CERT_FILE={bundle}" in boot and f"NODE_EXTRA_CA_CERTS={bundle}" in boot
    assert bundle.read_text().endswith(_CERT)


def test_an_existing_account_is_left_alone_and_no_ca_means_no_bundle(tmp_path, layer, image):
    _tool(image, "getent", 'echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert not [c for c in calls if c.startswith(("useradd", "groupadd"))]
    assert not (tmp_path / "state" / "ca-bundle.pem").exists()
