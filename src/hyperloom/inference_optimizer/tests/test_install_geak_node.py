# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the Node and Codex binaries the installer pins for GEAK."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common.llm.llm_config import ANTHROPIC_CREDENTIAL_ENV_ORDER

from ._kernel_tools_harness import sourceable_kernel_tools

_SRC = Path(__file__).resolve().parents[3]
_ARCH = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine()]
_NODE_DIR = f"node-v22.23.3-linux-{_ARCH}"
_NO_PROVIDER_KEYS = dict.fromkeys(
    ("ANTHROPIC_BASE_URL", *ANTHROPIC_CREDENTIAL_ENV_ORDER, "OPENAI_BASE_URL", "OPENAI_API_KEY"), ""
)


def _fake_node(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho v{version}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _install(
    tmp_path: Path, body: str, *, node_on_path: str, ok: bool = True, **env: str
) -> subprocess.CompletedProcess[str]:
    """Source the kernel-agent steps with a fake ``node`` of ``node_on_path`` first on PATH, then run ``body``."""
    fake_bin = tmp_path / "bin"
    _fake_node(fake_bin / "node", node_on_path)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    # Sourcing the steps puts the system directories first on PATH, so the fake node goes in front afterwards.
    script = (
        f'set -euo pipefail\nsource {sourceable_kernel_tools(tmp_path)!s}\nexport PATH="{fake_bin}:$PATH"\n{body}\n'
    )
    controlled = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_", "HYPERLOOM_GEAK_"))}
    controlled.update(
        REPO_ROOT=str(repo),
        HYPERLOOM_CACHE_DIR=str(tmp_path / "cache"),
        GEAK_ROOT=str(tmp_path / "geak"),
        PYTHON=sys.executable,
        PYTHONPATH=str(_SRC),
        **env,
    )
    proc = subprocess.run(["bash", "-c", script], env=controlled, text=True, capture_output=True, timeout=120)
    assert (proc.returncode == 0) == ok, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _node_bin(proc: subprocess.CompletedProcess[str]) -> str:
    return next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("NODE="))


def test_a_new_enough_node_on_path_is_reused(tmp_path):
    proc = _install(tmp_path, 'ensure_geak_node\necho "NODE=${_geak_node_bin}"', node_on_path="22.4.0")

    assert _node_bin(proc) == str(tmp_path / "bin" / "node")


def test_an_old_node_on_path_gives_way_to_the_cached_build(tmp_path):
    cached = _fake_node(tmp_path / "cache" / _NODE_DIR / "bin" / "node", "22.23.3")

    proc = _install(tmp_path, 'ensure_geak_node\necho "NODE=${_geak_node_bin}"', node_on_path="18.19.0")

    assert _node_bin(proc) == str(cached)


def test_an_operator_pinned_node_is_kept(tmp_path):
    proc = _install(
        tmp_path,
        'ensure_geak_node\necho "NODE=${_geak_node_bin}"',
        node_on_path="22.4.0",
        GEAK_NODE_BIN="/opt/node22/bin/node",
    )

    assert _node_bin(proc) == "/opt/node22/bin/node"


def test_the_env_file_and_dotenv_pin_node_and_codex_for_geak(tmp_path):
    env_file = tmp_path / "runtime" / "kernel-agent.env.sh"

    _install(
        tmp_path,
        'ensure_geak_node\n_geak_codex_bin="$(_resolve_geak_codex_bin)"\nwrite_env_file',
        node_on_path="22.4.0",
        KERNEL_AGENT_ENV=str(env_file),
        GEAK_CODEX_BIN="/opt/codex/bin/codex",
    )

    node = tmp_path / "bin" / "node"
    exported = env_file.read_text(encoding="utf-8")
    assert f"export GEAK_NODE_BIN='{node}'" in exported
    assert "export GEAK_CODEX_BIN='/opt/codex/bin/codex'" in exported
    dotenv = (tmp_path / "repo" / ".env").read_text(encoding="utf-8").splitlines()
    assert f"GEAK_NODE_BIN={node}" in dotenv
    assert "GEAK_CODEX_BIN=/opt/codex/bin/codex" in dotenv


@pytest.mark.parametrize(
    ("env", "refused"),
    [
        ({}, False),
        ({"GEAK_AGENT_PROFILE": "codex-gpt56"}, True),
        ({"GEAK_AGENT_BACKEND": "claude"}, True),
        ({"OPENAI_API_KEY": "sk-test"}, True),
        ({"OPENAI_API_KEY": "sk-test", "GEAK_AGENT_AUTO": "0"}, False),
    ],
)
def test_an_unfetchable_node_stops_the_install_only_when_geak_agent_is_selected(tmp_path, env, refused):
    curl = tmp_path / "bin" / "curl"
    curl.parent.mkdir(parents=True)
    curl.write_text("#!/bin/sh\nexit 22\n", encoding="utf-8")
    curl.chmod(0o755)

    proc = _install(
        tmp_path,
        'ensure_geak_node\necho "NODE=${_geak_node_bin}"',
        node_on_path="18.19.0",
        ok=not refused,
        **{**_NO_PROVIDER_KEYS, **env},
    )

    assert "downloading" in proc.stderr
    assert refused or _node_bin(proc) == ""


def _served_node_archive(tmp_path: Path) -> Path:
    """A fake curl serving a Node archive whose members belong to uid/gid 1000, as nodejs.org's do."""
    import hashlib
    import io
    import tarfile

    served = tmp_path / "served"
    served.mkdir()
    archive = served / f"{_NODE_DIR}.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        for name, data, mode in (
            (_NODE_DIR, None, 0o755),
            (f"{_NODE_DIR}/bin", None, 0o755),
            (f"{_NODE_DIR}/bin/node", b"#!/bin/sh\necho v22.23.3\n", 0o755),
        ):
            info = tarfile.TarInfo(name)
            info.uid = info.gid = 1000
            info.uname = info.gname = "builder"
            info.mode = mode
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (served / "SHASUMS256.txt").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    curl = tmp_path / "bin" / "curl"
    curl.parent.mkdir(parents=True, exist_ok=True)
    curl.write_text(
        '#!/bin/sh\nout=""\nwhile [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; *) url="$1"; shift;; esac; done\n'
        f'cp "{served}/$(basename "$url")" "$out"\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    return archive


@pytest.mark.skipif(shutil.which("xz") is None, reason="the installer unpacks Node with xz")
def test_the_fetched_node_belongs_to_the_installing_account(tmp_path):
    _served_node_archive(tmp_path)

    proc = _install(tmp_path, 'ensure_geak_node\necho "NODE=${_geak_node_bin}"', node_on_path="18.19.0")

    node = Path(_node_bin(proc))
    assert node == tmp_path / "cache" / _NODE_DIR / "bin" / "node"
    for path in (node, node.parent, node.parent.parent):
        assert path.stat().st_uid == os.geteuid(), f"{path} kept the archive's owner"


def _one_id_user_namespace() -> bool:
    try:
        probe = subprocess.run(["unshare", "--user", "--map-root-user", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(shutil.which("xz") is None, reason="the installer unpacks Node with xz")
@pytest.mark.skipif(not _one_id_user_namespace(), reason="this host cannot make a user namespace")
def test_node_is_fetched_inside_a_one_id_user_namespace(tmp_path):
    """Root of a namespace that maps one id cannot chown to the archive's uid 1000; the unpack must not try."""
    _served_node_archive(tmp_path)
    body = 'ensure_geak_node\necho "NODE=${_geak_node_bin}"'
    script = (
        f"set -euo pipefail\nsource {sourceable_kernel_tools(tmp_path)!s}\n"
        f'export PATH="{tmp_path / "bin"}:$PATH"\n{body}\n'
    )
    _fake_node(tmp_path / "bin" / "node", "18.19.0")
    (tmp_path / "repo").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_", "HYPERLOOM_GEAK_"))}
    env.update(
        REPO_ROOT=str(tmp_path / "repo"),
        HYPERLOOM_CACHE_DIR=str(tmp_path / "cache"),
        GEAK_ROOT=str(tmp_path / "geak"),
        PYTHON=sys.executable,
        PYTHONPATH=str(_SRC),
    )

    proc = subprocess.run(
        ["unshare", "--user", "--map-root-user", "bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _node_bin(proc) == str(tmp_path / "cache" / _NODE_DIR / "bin" / "node")
