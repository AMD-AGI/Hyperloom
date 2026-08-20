"""Tests for the InferenceX preflight clone + read-only guard.

Covers detection (no hard-coded read-only host candidates), cloning a fresh
writable checkout when none is found, and hard-failing when the clone fails or
the resolved path is not writable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch


from hyperloom.inference_optimizer.cli import preflight as cli_preflight


def test_clone_inferencex_sha_ref_uses_shallow_fetch(tmp_path, monkeypatch):
    """A 40-hex INFERENCEX_REF triggers init + shallow fetch + checkout."""
    monkeypatch.setenv("INFERENCEX_REF", "a" * 40)
    monkeypatch.setenv("INFERENCEX_REPO", "https://example.invalid/X.git")
    dest = tmp_path / "InferenceX"
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        # checkout materializes the validity marker (benchmarks/benchmark_lib.sh).
        if "checkout" in cmd:
            (dest / "benchmarks").mkdir(parents=True, exist_ok=True)
            (dest / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hyperloom.inference_optimizer.cli.preflight.subprocess.run", side_effect=fake_run):
        out = cli_preflight._clone_inferencex(dest)

    assert out == str(dest)
    # init + fetch + checkout (not a --branch clone)
    assert any(c[:2] == ["git", "init"] for c in calls)
    assert any("fetch" in c for c in calls)
    assert not any("clone" in c for c in calls)


def test_clone_inferencex_branch_ref_uses_clone(tmp_path, monkeypatch):
    """A non-hex ref (branch/tag) uses ``git clone --branch``."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        if "clone" in cmd:
            (dest / "benchmarks").mkdir(parents=True, exist_ok=True)
            (dest / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hyperloom.inference_optimizer.cli.preflight.subprocess.run", side_effect=fake_run):
        out = cli_preflight._clone_inferencex(dest)

    assert out == str(dest)
    assert any("clone" in c and "--branch" in c for c in calls)


def test_clone_inferencex_failure_returns_none(tmp_path, monkeypatch):
    """A git failure soft-degrades to None (caller decides to hard-fail)."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def boom(cmd, *a, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    with patch("hyperloom.inference_optimizer.cli.preflight.subprocess.run", side_effect=boom):
        out = cli_preflight._clone_inferencex(dest)

    assert out is None


def test_clone_inferencex_removes_partial_dir_on_failure(tmp_path, monkeypatch):
    """A bare ``git init`` that then fails to fetch leaves a stub dir; the
    clone must delete it so a later preflight does not mistake the stub for
    a valid checkout and skip re-cloning."""
    monkeypatch.setenv("INFERENCEX_REF", "a" * 40)
    dest = tmp_path / "InferenceX"

    def init_then_fail(cmd, *a, **kw):
        if "init" in cmd:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".git").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(128, cmd)  # fetch fails

    with patch("hyperloom.inference_optimizer.cli.preflight.subprocess.run", side_effect=init_then_fail):
        out = cli_preflight._clone_inferencex(dest)

    assert out is None
    assert not dest.exists()


def test_clone_inferencex_rejects_checkout_without_marker(tmp_path, monkeypatch):
    """git reports success but the tree lacks benchmarks/benchmark_lib.sh →
    treat as failure and clean up, never return a half-checkout."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def fake_run(cmd, *a, **kw):
        dest.mkdir(parents=True, exist_ok=True)  # empty, no marker
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hyperloom.inference_optimizer.cli.preflight.subprocess.run", side_effect=fake_run):
        out = cli_preflight._clone_inferencex(dest)

    assert out is None
    assert not dest.exists()


def test_inferencex_checkout_ok_requires_benchmark_lib(tmp_path):
    """The validity check rejects a bare dir and accepts a real checkout."""
    stub = tmp_path / "stub"
    stub.mkdir()
    assert cli_preflight._inferencex_checkout_ok(stub) is False

    good = tmp_path / "good"
    (good / "benchmarks").mkdir(parents=True)
    (good / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    assert cli_preflight._inferencex_checkout_ok(good) is True


def test_preflight_detects_checkout_via_validity_not_isdir():
    """Detection and post-clone guards must use the validity helper, not a
    bare ``is_dir()`` that would accept a half-cloned stub."""
    # _preflight and its InferenceX detection loop live in cli/preflight.py,
    # not cli/__init__.py.
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    assert "_inferencex_checkout_ok(candidate)" in src
    assert "_inferencex_checkout_ok(inferencex_path)" in src


def test_detection_candidates_exclude_wekafs_host_mounts():
    """The removed read-only fallbacks must not reappear in the source."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    assert 'Path("/path/hyperloom/InferenceX")' not in src
    assert 'Path("/opt/hyperloom/InferenceX")' not in src
    assert 'Path("/path/fully-local/inference_optimization/InferenceX")' not in src


def test_validated_inferencex_path_overwrites_env_not_setdefault():
    """A stale/broken INFERENCEX_PATH that triggers the clone must be
    overwritten with the validated path. ``setdefault`` would leave the bad
    value in place, so Magpie would still read the broken mount."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    # The final export must be an unconditional assignment, never setdefault.
    assert 'os.environ["INFERENCEX_PATH"] = inferencex_path' in src
    assert 'os.environ.setdefault("INFERENCEX_PATH"' not in src


def test_auto_detected_inferencex_candidates_must_be_writable():
    """Auto-detected read-only checkouts are skipped so preflight can clone."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    assert "if os.access(candidate, os.W_OK):" in src
    assert "skipping non-writable auto-detected" in src


# --- revision validation -------------------------------------------------------
#
# Completeness alone let a checkout cloned before a pin bump stay "usable"
# forever, so the bump never reached the box -- and since the synthetic path
# sources benchmark_lib.sh from whichever checkout wins, that drift was not
# AgentX-scoped. Two processes on the same Hyperloom commit could measure
# against different InferenceX revisions with nothing in the logs naming either.

_PIN = "3d5581562f643f9bdeb8410cd924e2c70906c966"
_OTHER = "a4bb43afa7fd74c1356583ed29e51421be010f0f"


def _checkout(tmp_path, name="co"):
    d = tmp_path / name
    (d / "benchmarks").mkdir(parents=True)
    (d / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    return d


def _at(monkeypatch, sha):
    monkeypatch.setattr(cli_preflight, "_inferencex_head_sha", lambda _p: sha)


def test_checkout_at_the_pin_is_accepted(tmp_path, monkeypatch):
    _at(monkeypatch, _PIN)
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref=_PIN) is True


def test_checkout_at_another_revision_is_rejected(tmp_path, monkeypatch):
    """The case that used to pass: complete tree, wrong code."""
    _at(monkeypatch, _OTHER)
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref=_PIN) is False


def test_short_pin_matches_a_full_head(tmp_path, monkeypatch):
    _at(monkeypatch, _PIN)
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref=_PIN[:12]) is True


def test_unreadable_head_is_tolerated(tmp_path, monkeypatch):
    """A tarball drop with no .git works today; do not reject it over metadata
    we only just started asking for."""
    _at(monkeypatch, "")
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref=_PIN) is True


def test_branch_ref_skips_the_comparison(tmp_path, monkeypatch):
    """A branch name cannot be compared to a HEAD without a network round trip."""
    _at(monkeypatch, _OTHER)
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref="main") is True


def test_explicit_empty_ref_skips_the_comparison(tmp_path, monkeypatch):
    """Used right after a clone, which is at the ref by construction."""
    _at(monkeypatch, _OTHER)
    assert cli_preflight._inferencex_checkout_ok(_checkout(tmp_path), ref="") is True


def test_incomplete_checkout_still_rejected_before_any_ref_work(tmp_path, monkeypatch):
    def _boom(_p):
        raise AssertionError("ref check must not run on an incomplete checkout")

    monkeypatch.setattr(cli_preflight, "_inferencex_head_sha", _boom)
    stub = tmp_path / "stub"
    stub.mkdir()
    assert cli_preflight._inferencex_checkout_ok(stub, ref=_PIN) is False


def test_clone_destination_is_per_revision():
    """A shared directory name is what let one revision win by existing."""
    assert cli_preflight._inferencex_dest_name(_PIN) == f"InferenceX@{_PIN}"
    assert cli_preflight._inferencex_dest_name("main") == "InferenceX@main"
    assert "/" not in cli_preflight._inferencex_dest_name("feature/x y")
