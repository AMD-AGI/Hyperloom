"""Tests for ``ensure_claude_cli`` end-to-end orchestration (mocked)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_optimizer.bootstrap import (
    InstallReport,
    MissingDependency,
    ensure_claude_cli,
)
from inference_optimizer.bootstrap import orchestrator as orch_mod
from inference_optimizer.bootstrap import probe as probe_mod
from inference_optimizer.bootstrap.install import ClaudeInstall, NodeInstall
from inference_optimizer.bootstrap.probe import ProbeResult


# ---------------------------------------------------------------------------
def _stub_probe(monkeypatch, *, satisfied: bool, claude: bool = False, node: bool = False):
    """Return ``probe_environment`` results that drive the path under test."""
    states = []

    def make_probe(*, has_node: bool, has_claude: bool) -> ProbeResult:
        return ProbeResult(
            node_path=Path("/usr/bin/node") if has_node else None,
            node_version=(20, 18, 0) if has_node else None,
            npm_path=Path("/usr/bin/npm") if has_node else None,
            claude_path=Path("/usr/bin/claude") if has_claude else None,
            claude_version="1.0.0" if has_claude else None,
            extra_path_dirs=(),
        )

    if satisfied:
        states.extend([make_probe(has_node=True, has_claude=True)] * 4)
    else:
        # Pre-install state — both missing
        states.append(make_probe(has_node=node, has_claude=claude))
        # Post-install state — satisfied
        states.append(make_probe(has_node=True, has_claude=True))

    counter = {"i": 0}

    def fake_probe(extra_dirs=()):
        i = counter["i"]
        counter["i"] += 1
        return states[min(i, len(states) - 1)]

    monkeypatch.setattr(orch_mod, "probe_environment", fake_probe)
    monkeypatch.setattr(probe_mod, "probe_environment", fake_probe)


# ---------------------------------------------------------------------------
def test_already_satisfied_returns_clean_report(monkeypatch, tmp_path):
    _stub_probe(monkeypatch, satisfied=True)

    report = ensure_claude_cli(auto_install=False, cache_dir=tmp_path)

    assert isinstance(report, InstallReport)
    assert report.installed_node is False
    assert report.installed_claude is False
    assert "already satisfied" in " ".join(report.notes)


def test_missing_without_auto_install_raises(monkeypatch, tmp_path):
    _stub_probe(monkeypatch, satisfied=False, claude=False, node=False)

    with pytest.raises(MissingDependency) as exc:
        ensure_claude_cli(auto_install=False, cache_dir=tmp_path)
    msg = str(exc.value)
    assert "claude CLI" in msg
    assert "node" in msg
    assert exc.value.missing == ("node", "claude")


def test_auto_install_full_path(monkeypatch, tmp_path):
    """node missing + claude missing + auto_install=True --> both installed."""
    _stub_probe(monkeypatch, satisfied=False, node=False, claude=False)

    fake_node = NodeInstall(
        install_dir=tmp_path / "node-v20",
        node_bin=tmp_path / "node-v20" / "bin" / "node",
        npm_bin=tmp_path / "node-v20" / "bin" / "npm",
        bin_dir=tmp_path / "node-v20" / "bin",
    )
    fake_claude = ClaudeInstall(
        prefix_dir=tmp_path / "npm-prefix",
        claude_bin=tmp_path / "npm-prefix" / "bin" / "claude",
        bin_dir=tmp_path / "npm-prefix" / "bin",
    )
    install_calls = {"node": 0, "claude": 0}

    def fake_install_node(**kwargs):
        install_calls["node"] += 1
        return fake_node

    def fake_install_claude(**kwargs):
        install_calls["claude"] += 1
        return fake_claude

    monkeypatch.setattr(orch_mod, "install_node_portable", fake_install_node)
    monkeypatch.setattr(orch_mod, "install_claude_global", fake_install_claude)

    saved_path = os.environ.get("PATH", "")
    try:
        report = ensure_claude_cli(auto_install=True, cache_dir=tmp_path)
    finally:
        os.environ["PATH"] = saved_path

    assert install_calls == {"node": 1, "claude": 1}
    assert report.installed_node and report.installed_claude
    assert fake_node.bin_dir in report.extra_path_dirs
    assert fake_claude.bin_dir in report.extra_path_dirs


def test_auto_install_only_claude_when_node_present(monkeypatch, tmp_path):
    _stub_probe(monkeypatch, satisfied=False, node=True, claude=False)

    fake_claude = ClaudeInstall(
        prefix_dir=tmp_path / "npm-prefix",
        claude_bin=tmp_path / "npm-prefix" / "bin" / "claude",
        bin_dir=tmp_path / "npm-prefix" / "bin",
    )
    install_calls = {"node": 0, "claude": 0}

    def fake_install_node(**kwargs):
        install_calls["node"] += 1
        raise AssertionError("install_node_portable should not be called")

    def fake_install_claude(**kwargs):
        install_calls["claude"] += 1
        return fake_claude

    monkeypatch.setattr(orch_mod, "install_node_portable", fake_install_node)
    monkeypatch.setattr(orch_mod, "install_claude_global", fake_install_claude)

    report = ensure_claude_cli(auto_install=True, cache_dir=tmp_path)

    assert install_calls == {"node": 0, "claude": 1}
    assert report.installed_node is False
    assert report.installed_claude is True


def test_summary_renders(monkeypatch, tmp_path):
    _stub_probe(monkeypatch, satisfied=True)
    report = ensure_claude_cli(auto_install=False, cache_dir=tmp_path)
    s = report.summary()
    assert "node:" in s
    assert "claude:" in s
