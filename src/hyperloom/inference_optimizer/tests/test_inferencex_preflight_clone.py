"""Tests for the InferenceX preflight clone + read-only guard."""

from __future__ import annotations

import subprocess
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.inference_optimizer.cli import preflight as cli_preflight
from hyperloom.inference_optimizer import inferencex_checkout


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

    with patch("hyperloom.inference_optimizer.inferencex_checkout.subprocess.run", side_effect=fake_run):
        out = inferencex_checkout.clone(dest)

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

    with patch("hyperloom.inference_optimizer.inferencex_checkout.subprocess.run", side_effect=fake_run):
        out = inferencex_checkout.clone(dest)

    assert out == str(dest)
    assert any("clone" in c and "--branch" in c for c in calls)


def test_clone_inferencex_failure_returns_none(tmp_path, monkeypatch):
    """A git failure soft-degrades to None (caller decides to hard-fail)."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def boom(cmd, *a, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    with patch("hyperloom.inference_optimizer.inferencex_checkout.subprocess.run", side_effect=boom):
        out = inferencex_checkout.clone(dest)

    assert out is None


def test_clone_inferencex_removes_partial_dir_on_failure(tmp_path, monkeypatch):
    """A bare ``git init`` that then fails to fetch leaves a stub dir; the clone must delete it so a later preflight does not mistake the stub for a valid checkout and skip re-cloning."""
    monkeypatch.setenv("INFERENCEX_REF", "a" * 40)
    dest = tmp_path / "InferenceX"

    def init_then_fail(cmd, *a, **kw):
        if "init" in cmd:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".git").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(128, cmd)  # fetch fails

    with patch("hyperloom.inference_optimizer.inferencex_checkout.subprocess.run", side_effect=init_then_fail):
        out = inferencex_checkout.clone(dest)

    assert out is None
    assert not dest.exists()


def test_clone_inferencex_rejects_checkout_without_marker(tmp_path, monkeypatch):
    """git reports success but the tree lacks benchmarks/benchmark_lib.sh → treat as failure and clean up, never return a half-checkout."""
    monkeypatch.setenv("INFERENCEX_REF", "main")
    dest = tmp_path / "InferenceX"

    def fake_run(cmd, *a, **kw):
        dest.mkdir(parents=True, exist_ok=True)  # empty, no marker
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hyperloom.inference_optimizer.inferencex_checkout.subprocess.run", side_effect=fake_run):
        out = inferencex_checkout.clone(dest)

    assert out is None
    assert not dest.exists()


def test_inferencex_checkout_ok_requires_benchmark_lib(tmp_path):
    """The validity check rejects a bare dir and accepts a real checkout."""
    stub = tmp_path / "stub"
    stub.mkdir()
    assert inferencex_checkout.checkout_ok(stub) is False

    good = tmp_path / "good"
    (good / "benchmarks").mkdir(parents=True)
    (good / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    assert inferencex_checkout.checkout_ok(good) is True


def test_preflight_detects_checkout_via_validity_not_isdir():
    """Detection and post-clone guards must use the validity helper, not a bare ``is_dir()`` that would accept a half-cloned stub."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    assert "checkout_ok(candidate)" in Path(inferencex_checkout.__file__).read_text()
    assert "inferencex_checkout.checkout_ok(inferencex_path)" in src


def test_detection_candidates_exclude_wekafs_host_mounts():
    """The removed read-only fallbacks must not reappear in the source."""
    src = Path(inferencex_checkout.__file__).read_text(encoding="utf-8")
    assert 'Path("/path/hyperloom/InferenceX")' not in src
    assert 'Path("/opt/hyperloom/InferenceX")' not in src
    assert 'Path("/path/fully-local/inference_optimization/InferenceX")' not in src


def test_validated_inferencex_path_overwrites_env_not_setdefault():
    """A stale/broken INFERENCEX_PATH that triggers the clone must be overwritten with the validated path."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    # The final export must be an unconditional assignment, never setdefault.
    assert 'os.environ["INFERENCEX_PATH"] = inferencex_path' in src
    assert 'os.environ.setdefault("INFERENCEX_PATH"' not in src


def test_a_network_mount_checkout_is_recorded_rather_than_copied():
    """Nothing relocates the checkout, so the ledger is what tells a flapped round from one that cannot serve."""
    src = Path(cli_preflight.__file__).read_text(encoding="utf-8")
    assert "inferencex_network_fs = is_network_fs(inferencex_path)" in src
    assert '"network_fs": inferencex_network_fs,' in src


def test_auto_detected_inferencex_candidates_must_be_writable():
    """Auto-detected read-only checkouts are skipped so preflight can clone."""
    src = Path(inferencex_checkout.__file__).read_text(encoding="utf-8")
    assert "if os.access(candidate, os.W_OK):" in src
    assert "skipping non-writable auto-detected" in src


# --- revision validation -------------------------------------------------------

_PIN = "3d5581562f643f9bdeb8410cd924e2c70906c966"
_OTHER = "a4bb43afa7fd74c1356583ed29e51421be010f0f"


def _checkout(tmp_path, name="co"):
    d = tmp_path / name
    (d / "benchmarks").mkdir(parents=True)
    (d / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    return d


def _at(monkeypatch, sha):
    monkeypatch.setattr(inferencex_checkout, "head_sha", lambda _p: sha)


def test_checkout_at_the_pin_is_accepted(tmp_path, monkeypatch):
    _at(monkeypatch, _PIN)
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref=_PIN) is True


def test_checkout_at_another_revision_is_rejected(tmp_path, monkeypatch):
    """The case that used to pass: complete tree, wrong code."""
    _at(monkeypatch, _OTHER)
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref=_PIN) is False


def test_short_pin_matches_a_full_head(tmp_path, monkeypatch):
    _at(monkeypatch, _PIN)
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref=_PIN[:12]) is True


def test_unreadable_head_is_tolerated(tmp_path, monkeypatch):
    """A tarball drop with no .git works today; do not reject it over metadata we only just started asking for."""
    _at(monkeypatch, "")
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref=_PIN) is True


def test_branch_ref_skips_the_comparison(tmp_path, monkeypatch):
    """A branch name cannot be compared to a HEAD without a network round trip."""
    _at(monkeypatch, _OTHER)
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref="main") is True


def test_explicit_empty_ref_skips_the_comparison(tmp_path, monkeypatch):
    """Used right after a clone, which is at the ref by construction."""
    _at(monkeypatch, _OTHER)
    assert inferencex_checkout.checkout_ok(_checkout(tmp_path), ref="") is True


def test_incomplete_checkout_still_rejected_before_any_ref_work(tmp_path, monkeypatch):
    def _boom(_p):
        raise AssertionError("ref check must not run on an incomplete checkout")

    monkeypatch.setattr(inferencex_checkout, "head_sha", _boom)
    stub = tmp_path / "stub"
    stub.mkdir()
    assert inferencex_checkout.checkout_ok(stub, ref=_PIN) is False


def test_clone_destination_is_per_revision():
    """A shared directory name is what let one revision win by existing."""
    assert inferencex_checkout.destination_name(_PIN) == f"InferenceX@{_PIN}"
    assert inferencex_checkout.destination_name("main") == "InferenceX@main"
    assert "/" not in inferencex_checkout.destination_name("feature/x y")


def _installer_pin() -> str:
    installer = Path(cli_preflight.__file__).resolve().parents[1] / "assets" / "install.sh"
    match = re.search(r'^INFERENCEX_REF="\$\{INFERENCEX_REF:-([0-9a-f]{40})\}"$', installer.read_text(), re.M)
    assert match, "installer must declare its reproducible default revision"
    return match[1]


def test_default_inferencex_pin_is_consistent():
    assert _installer_pin() == inferencex_checkout.DEFAULT_REF


@pytest.mark.parametrize("explicit_ref", [None, "a" * 40])
def test_ensure_inferencex_reuses_installed_pin_without_network(tmp_path, monkeypatch, explicit_ref):
    """An installer checkout survives CLI preflight, including an operator's matching pin override."""
    installed_ref = explicit_ref or _installer_pin()
    checkout = _checkout(tmp_path, f"InferenceX@{installed_ref}")
    monkeypatch.setenv("INFERENCEX_PATH", str(checkout))
    if explicit_ref:
        monkeypatch.setenv("INFERENCEX_REF", explicit_ref)
    else:
        monkeypatch.delenv("INFERENCEX_REF", raising=False)
    monkeypatch.setattr(cli_preflight, "is_network_fs", lambda _path: False)

    def local_git_only(command, **_kwargs):
        assert command == ["git", "-C", str(checkout), "rev-parse", "HEAD"], "checkout reuse must not fetch or clone"
        return subprocess.CompletedProcess(command, 0, stdout=installed_ref + "\n")

    monkeypatch.setattr(cli_preflight.subprocess, "run", local_git_only)
    event = {"ext": {"steps": []}}

    assert cli_preflight._ensure_inferencex(event) == str(checkout)
    assert os.environ["INFERENCEX_PATH"] == str(checkout)
    step = event["ext"]["steps"][-1]
    assert step["status"] == "already_present"
    assert step["version_after"] == installed_ref
    assert step["detail"]["ref"] == installed_ref


@pytest.mark.parametrize("selection", ["environment", "yaml"])
def test_native_preflight_leaves_checkout_and_patches_to_magpie(tmp_path, monkeypatch, selection):
    """Neither an inherited synthetic checkout nor the new project layout is cloned/patched by Hyperloom."""
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    if selection == "environment":
        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    else:
        config = tmp_path / "benchmark.yaml"
        config.write_text("benchmark:\n  agentx:\n    enabled: true\n")
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(config))
    inherited_checkout = str(tmp_path / "InferenceX@synthetic")
    native_checkout = str(tmp_path / "InferenceX@native" / "inferencex-e2e")
    monkeypatch.setenv("INFERENCEX_PATH", inherited_checkout)
    monkeypatch.setenv("MAGPIE_INFERENCEX_PATH", native_checkout)

    def unexpected(*_args, **_kwargs):
        pytest.fail("native AgentX must not clone or patch the synthetic InferenceX runtime")

    monkeypatch.setattr(cli_preflight, "_ensure_inferencex", unexpected)
    monkeypatch.setattr(cli_preflight, "_patch_magpie_compat", unexpected)
    event = {"ext": {"steps": []}}

    assert cli_preflight._prepare_inferencex(event, magpie_active=True) == ""
    assert os.environ["INFERENCEX_PATH"] == inherited_checkout
    assert os.environ["MAGPIE_INFERENCEX_PATH"] == native_checkout
    assert event["ext"]["steps"][-1]["skip_reason"] == "magpie_manages_native_agentx"


@pytest.mark.parametrize("backend", ["magpie", "bypass", "mlperf"])
def test_synthetic_and_mlperf_preflight_keep_checkout_preparation(tmp_path, monkeypatch, backend):
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1" if backend == "mlperf" else "0")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf" if backend == "mlperf" else "magpie")
    checkout = str(tmp_path / "InferenceX")
    monkeypatch.setattr(cli_preflight, "_ensure_inferencex", lambda _event: checkout)
    patches = []
    monkeypatch.setattr(
        cli_preflight,
        "_patch_magpie_compat",
        lambda _event, path, *, magpie_active: patches.append((path, magpie_active)),
    )

    assert cli_preflight._prepare_inferencex({"ext": {"steps": []}}, magpie_active=backend != "bypass") == checkout
    assert patches == [(checkout, backend != "bypass")]
