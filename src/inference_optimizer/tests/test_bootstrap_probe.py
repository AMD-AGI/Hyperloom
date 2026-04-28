"""Tests for ``bootstrap.probe`` — pure stdlib, no real subprocess calls."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.bootstrap import probe as probe_mod
from inference_optimizer.bootstrap.probe import (
    NODE_MIN_VERSION,
    ProbeResult,
    parse_version,
    probe_environment,
)


# ---------------------------------------------------------------------------
def test_parse_version_basic():
    assert parse_version("v20.18.0\n") == (20, 18, 0)
    assert parse_version("Node.js v18.0.1") == (18, 0, 1)
    assert parse_version("16.20.5  ") == (16, 20, 5)


def test_parse_version_returns_none_on_garbage():
    assert parse_version("") is None
    assert parse_version("not a version") is None


def test_node_min_version_is_18():
    assert NODE_MIN_VERSION == (18, 0, 0)


# ---------------------------------------------------------------------------
def test_probe_no_binaries(monkeypatch):
    monkeypatch.setattr(probe_mod, "_which", lambda *_a, **_k: None)
    monkeypatch.setattr(probe_mod, "_run", lambda *_a, **_k: "")
    r = probe_environment()
    assert r.has_node is False
    assert r.has_claude is False
    assert r.node_is_recent_enough is False


def test_probe_node_only_no_claude(monkeypatch):
    fake_node = Path("/usr/bin/node")
    fake_npm = Path("/usr/bin/npm")
    which_table = {
        "node": fake_node,
        "npm": fake_npm,
        "claude": None,
    }
    monkeypatch.setattr(
        probe_mod, "_which",
        lambda binary, **_k: which_table.get(binary),
    )
    monkeypatch.setattr(
        probe_mod, "_run",
        lambda args, **_k: "v20.18.0\n" if "node" in args[0] else "",
    )

    r = probe_environment()
    assert r.has_node is True
    assert r.node_is_recent_enough is True
    assert r.has_claude is False
    assert r.npm_path == fake_npm


def test_probe_node_too_old(monkeypatch):
    fake_node = Path("/usr/bin/node")
    fake_npm = Path("/usr/bin/npm")
    which_table = {"node": fake_node, "npm": fake_npm, "claude": None}
    monkeypatch.setattr(
        probe_mod, "_which", lambda binary, **_k: which_table.get(binary),
    )
    monkeypatch.setattr(
        probe_mod, "_run",
        lambda args, **_k: "v16.13.0\n" if "node" in args[0] else "",
    )

    r = probe_environment()
    assert r.has_node is True
    assert r.node_version == (16, 13, 0)
    assert r.node_is_recent_enough is False


def test_probe_full_stack(monkeypatch):
    fake_node = Path("/opt/node/bin/node")
    fake_npm = Path("/opt/node/bin/npm")
    fake_claude = Path("/opt/npm-prefix/bin/claude")

    which_table = {"node": fake_node, "npm": fake_npm, "claude": fake_claude}
    monkeypatch.setattr(
        probe_mod, "_which", lambda binary, **_k: which_table.get(binary),
    )

    def fake_run(args, **_k):
        if "node" in args[0]:
            return "v22.5.1\n"
        if "claude" in args[0]:
            return "1.0.42\n"
        return ""

    monkeypatch.setattr(probe_mod, "_run", fake_run)
    r = probe_environment()
    assert isinstance(r, ProbeResult)
    assert r.has_node and r.has_claude
    assert r.node_version == (22, 5, 1)
    assert r.node_is_recent_enough is True
    assert r.claude_version == "1.0.42"


def test_probe_with_extra_dirs(monkeypatch, tmp_path):
    """Extra dirs should be forwarded into shutil.which's path argument."""
    captured = {}

    def fake_which(binary, *, path=None):
        captured.setdefault("paths", []).append((binary, path))
        return None

    monkeypatch.setattr(probe_mod.shutil, "which", fake_which)
    overlay = (tmp_path / "node-bin",)
    probe_environment(extra_dirs=overlay)
    paths = [p for _b, p in captured["paths"] if p]
    assert any(str(overlay[0]) in p for p in paths)
