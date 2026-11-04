# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""start.sh of the hosting layer: what it refuses on an image, and what it hands app_bootstrap.sh."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
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
    state.chmod(0o700)  # private to root, as shared storage may be
    full = {
        "PATH": f"{image}:/usr/bin:/bin",
        "CALLS": str(tmp_path / "calls"),
        "HYPERLOOM_STATE_DIR": str(state),
        "TMPDIR": str(tmp_path / "local-tmp"),
        **env,
    }
    Path(full["TMPDIR"]).mkdir(exist_ok=True)
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
    bundle = tmp_path / "local-tmp" / "hyperloom-host" / "ca-bundle.pem"
    assert f"SSL_CERT_FILE={bundle}" in boot and f"NODE_EXTRA_CA_CERTS={bundle}" in boot
    assert bundle.read_text().endswith(_CERT)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o644 and stat.S_IMODE(bundle.parent.stat().st_mode) == 0o755
    assert not list((tmp_path / "state").iterdir())  # nothing written to the (shared) state dir


def test_the_specialist_can_read_the_bundle_and_the_state_dir_stays_empty(tmp_path, layer, image):
    if os.geteuid() != 0 or shutil.which("setpriv") is None:
        pytest.skip("needs root and setpriv to act as uid 2000")
    (layer / "ca.pem").write_text(_CERT)
    local = Path(tempfile.mkdtemp())  # a path uid 2000 can traverse, unlike pytest's private tmp dirs
    try:
        local.chmod(0o755)
        done, _ = _run(tmp_path, layer, image, TMPDIR=str(local))
        assert done.returncode == 0, done.stderr
        bundle = local / "hyperloom-host" / "ca-bundle.pem"
        read = subprocess.run(
            ["setpriv", "--reuid=2000", "--regid=2000", "--clear-groups", "cat", str(bundle)],
            capture_output=True,
            text=True,
        )
        assert read.returncode == 0 and read.stdout.endswith(_CERT), read.stderr
        assert not list((tmp_path / "state").iterdir())
    finally:
        shutil.rmtree(local, ignore_errors=True)


def test_an_existing_account_is_left_alone_and_no_ca_means_no_bundle(tmp_path, layer, image):
    _tool(image, "getent", 'echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert not [c for c in calls if c.startswith(("useradd", "groupadd"))]
    assert not (tmp_path / "local-tmp" / "hyperloom-host").exists()


def test_a_layer_built_under_a_private_umask_is_readable_and_runnable_by_every_account(tmp_path):
    """build_layer.sh in a copy of the repo layout, with the installer stubbed, under umask 027."""
    root = tmp_path / "repo"
    for rel in (
        "scripts/hosted_app_image/build_layer.sh",
        "src/hyperloom/hosted/app_bootstrap.sh",
        "src/hyperloom/hosted/start.sh",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(START.parent.parent.parent.parent / rel, root / rel)
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c"],
    ):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    stub = tmp_path / "stub"
    _tool(
        stub,
        "curl",
        'echo \'mkdir -p "$HOME/.local/bin" && printf "#!/bin/sh\\necho 2.0.0 stub\\n" >"$HOME/.local/bin/claude" && chmod 700 "$HOME/.local/bin/claude"\'',
    )
    ca = tmp_path / "ca.crt"
    ca.write_text(_CERT)
    out = tmp_path / "layers"
    done = subprocess.run(
        [
            "sh",
            "-c",
            f'umask 027; exec bash "{root}/scripts/hosted_app_image/build_layer.sh" --out "{out}" --ca-file "{ca}"',
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{stub}:/usr/bin:/bin"},
    )
    assert done.returncode == 0, done.stderr
    layer = next(out.iterdir())
    assert layer.name.endswith("-claude2.0.0")
    for path in [layer, *layer.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            assert mode == 0o755, path
        else:
            assert mode & 0o444 == 0o444, path  # a+r
            if path.name != "ca.pem":
                assert mode & 0o111 == 0o111, path  # the scripts and the CLI: a+x
