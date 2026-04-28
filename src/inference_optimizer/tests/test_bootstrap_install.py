"""Tests for ``bootstrap.install`` — subprocess + urllib are mocked."""
from __future__ import annotations

import platform
from pathlib import Path

import pytest

from inference_optimizer.bootstrap import install as install_mod
from inference_optimizer.bootstrap.errors import InstallFailed, UnsupportedPlatform
from inference_optimizer.bootstrap.install import (
    DEFAULT_NODE_VERSION,
    _resolve_archive,
    install_claude_global,
    install_node_portable,
)


# ---------------------------------------------------------------------------
def test_resolve_archive_linux_x64(monkeypatch):
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "x86_64")
    fname, url, ext = _resolve_archive("20.18.0")
    assert fname == "node-v20.18.0-linux-x64.tar.xz"
    assert url.endswith("/v20.18.0/" + fname)
    assert ext == "tar.xz"


def test_resolve_archive_windows(monkeypatch):
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "AMD64")
    fname, _url, ext = _resolve_archive("20.18.0")
    assert fname.endswith(".zip")
    assert ext == "zip"


def test_resolve_archive_unsupported_raises(monkeypatch):
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "RISCV")
    with pytest.raises(UnsupportedPlatform):
        _resolve_archive("20.18.0")


# ---------------------------------------------------------------------------
def _mk_fake_extract(monkeypatch, fake_node_layout):
    """Replace ``_download`` and ``_extract`` so we can test ``install_node_portable``
    without touching the network or filesystem at the OS level.
    """
    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"FAKE-NODE-ARCHIVE")

    def fake_extract(archive, target_dir):
        # Lay out a fake node-vX.Y.Z directory.
        extracted = target_dir / fake_node_layout.name
        extracted.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            (extracted / "node.exe").write_text("noop")
            (extracted / "npm.cmd").write_text("noop")
        else:
            bin_dir = extracted / "bin"
            bin_dir.mkdir()
            (bin_dir / "node").write_text("noop")
            (bin_dir / "npm").write_text("noop")
        return extracted

    monkeypatch.setattr(install_mod, "_download", fake_download)
    monkeypatch.setattr(install_mod, "_extract", fake_extract)


def test_install_node_portable_creates_layout(monkeypatch, tmp_path):
    layout = Path(f"node-v{DEFAULT_NODE_VERSION}-linux-x64")
    _mk_fake_extract(monkeypatch, layout)
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "x86_64")

    out = install_node_portable(cache_dir=tmp_path)
    assert out.install_dir.exists()
    assert out.node_bin.exists()
    assert out.npm_bin.exists()
    assert out.bin_dir == out.node_bin.parent


def test_install_node_portable_is_idempotent(monkeypatch, tmp_path):
    layout = Path(f"node-v{DEFAULT_NODE_VERSION}-linux-x64")
    _mk_fake_extract(monkeypatch, layout)
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_mod.platform, "machine", lambda: "x86_64")

    a = install_node_portable(cache_dir=tmp_path)
    # Second call: download must NOT be called again.
    calls = {"download": 0}
    real_download = install_mod._download

    def counting_download(*args, **kwargs):
        calls["download"] += 1
        return real_download(*args, **kwargs)

    monkeypatch.setattr(install_mod, "_download", counting_download)
    b = install_node_portable(cache_dir=tmp_path)
    assert b.node_bin == a.node_bin
    assert calls["download"] == 0


def test_install_claude_global_runs_npm(monkeypatch, tmp_path):
    """``install_claude_global`` should shell out to ``npm install -g``."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["env"] = kwargs.get("env")
        # Pretend npm placed the binary in the prefix.
        prefix = Path(args[args.index("--prefix") + 1])
        if platform.system() == "Windows":
            (prefix).mkdir(parents=True, exist_ok=True)
            (prefix / "claude.cmd").write_text("noop")
        else:
            bin_dir = prefix / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "claude").write_text("noop")
        return ""

    monkeypatch.setattr(install_mod, "_run_blocking", fake_run)

    out = install_claude_global(
        npm_bin=tmp_path / "npm",
        node_bin=tmp_path / "node",
        prefix_dir=tmp_path / "prefix",
    )
    assert out.claude_bin.exists()
    assert "@anthropic-ai/claude-code" in captured["args"]
    assert "--prefix" in captured["args"]
    assert "install" in captured["args"]


def test_install_claude_global_raises_when_binary_missing(monkeypatch, tmp_path):
    """If npm reports success but no claude binary appears, raise."""
    monkeypatch.setattr(install_mod, "_run_blocking", lambda *a, **kw: "")
    with pytest.raises(InstallFailed):
        install_claude_global(
            npm_bin=tmp_path / "npm",
            node_bin=tmp_path / "node",
            prefix_dir=tmp_path / "prefix",
        )
