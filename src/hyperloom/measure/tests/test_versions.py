"""Tests for socket_versions: stability, sensitivity and the tests/ exclusion."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hyperloom.measure import versions


def _layout(root: Path) -> None:
    for rel, text in {
        "measure/a.py": "a = 1\n",
        "measure/runner/r.py": "r = 1\n",
        "sandbox/s.py": "s = 1\n",
        "orchestrator/specialists/agent_runner.py": "x = 1\n",
        "orchestrator/specialists/resources.py": "y = 1\n",
        "measure/tests/test_a.py": "t = 1\n",
        "sandbox/tests/test_s.py": "t = 1\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    _layout(tmp_path)
    monkeypatch.setattr(versions, "_package_root", lambda: tmp_path)
    return tmp_path


def test_real_package_shape_and_stability():
    first = versions.socket_versions()
    assert first == versions.socket_versions()
    assert {"measurement", "ai_work"} <= set(first)
    for value in first.values():
        assert re.fullmatch(r"[0-9a-f]{12}", value)


def test_edit_changes_only_the_owning_socket(pkg):
    before = versions.socket_versions()
    (pkg / "measure/runner/r.py").write_text("r = 2\n")
    after = versions.socket_versions()
    assert after["measurement"] != before["measurement"]
    assert after["ai_work"] == before["ai_work"]
    (pkg / "orchestrator/specialists/resources.py").write_text("y = 2\n")
    assert versions.socket_versions()["ai_work"] != before["ai_work"]


def test_tests_directories_are_excluded(pkg):
    before = versions.socket_versions()
    (pkg / "measure/tests/test_a.py").write_text("t = 2\n")
    (pkg / "sandbox/tests/new_test.py").write_text("n = 1\n")
    (pkg / "measure/runner/tests").mkdir()
    (pkg / "measure/runner/tests/test_r.py").write_text("n = 1\n")
    assert versions.socket_versions() == before


def test_rename_changes_the_version(pkg):
    before = versions.socket_versions()["measurement"]
    (pkg / "sandbox/s.py").rename(pkg / "sandbox/s2.py")
    assert versions.socket_versions()["measurement"] != before


def test_equal_content_in_two_roots_gives_equal_versions(tmp_path, monkeypatch):
    results = []
    for name in ("one", "two"):
        root = tmp_path / name
        _layout(root)
        monkeypatch.setattr(versions, "_package_root", lambda root=root: root)
        results.append(versions.socket_versions())
    assert results[0] == results[1]
