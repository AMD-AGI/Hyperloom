# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""TraceLens self-heal at the trace_analyze use-site.

The optimizer's ``trace_analyze`` subprocess reads ``TRACELENS_ROOT`` long
after install time. When the pod-local checkout vanishes mid-run (a
concurrent install rm+re-clones it, or /tmp is reaped), the use-site must
idempotently re-clone it under a shared flock instead of dying with
``FileNotFoundError``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
TL_PATH = TOOLS_DIR / "tracelens_analysis.py"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@pytest.fixture(scope="module")
def tl_module():
    mod_name = "tracelens_analysis_selfheal_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, TL_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the
    # module in sys.modules (cls.__module__ lookup) during import.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_source_repo(path: Path) -> str:
    """Create a tiny git repo that stands in for AMD-AGI/TraceLens."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "marker.txt").write_text("tracelens-source\n", encoding="utf-8")
    env_args = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", *env_args, "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", *env_args, "-C", str(path), "commit", "-q", "-m", "init"],
        check=True,
    )
    return str(path)


def test_selfheal_reclones_when_root_missing(tl_module, tmp_path, monkeypatch):
    """RED: a missing TRACELENS_ROOT must be rebuilt, not raise."""
    source = _make_source_repo(tmp_path / "src" / "TraceLens")
    tl_root = tmp_path / "open-source-repos" / "TraceLens"
    log_path = tmp_path / "run.log"

    monkeypatch.setenv("TRACELENS_REPO", source)
    monkeypatch.setenv("TRACELENS_REF", "HEAD")

    assert not tl_root.exists()
    tl_module._ensure_tracelens_checkout(tl_root, log_path=log_path)
    assert tl_root.exists()
    assert (tl_root / "marker.txt").exists()


def test_selfheal_is_idempotent_when_present(tl_module, tmp_path, monkeypatch):
    """An existing checkout is left untouched (no re-clone)."""
    source = _make_source_repo(tmp_path / "src" / "TraceLens")
    tl_root = tmp_path / "open-source-repos" / "TraceLens"
    log_path = tmp_path / "run.log"
    monkeypatch.setenv("TRACELENS_REPO", source)
    monkeypatch.setenv("TRACELENS_REF", "HEAD")

    tl_module._ensure_tracelens_checkout(tl_root, log_path=log_path)
    (tl_root / "sentinel").write_text("keep", encoding="utf-8")
    tl_module._ensure_tracelens_checkout(tl_root, log_path=log_path)
    # Idempotent: existing tree (and our sentinel) preserved, not wiped.
    assert (tl_root / "sentinel").exists()


def test_selfheal_rebuilds_half_cloned_tree_without_git(tl_module, tmp_path, monkeypatch):
    """A dir that exists but lacks .git (installer's in-progress clone) is
    treated as incomplete and rebuilt."""
    source = _make_source_repo(tmp_path / "src" / "TraceLens")
    tl_root = tmp_path / "open-source-repos" / "TraceLens"
    tl_root.mkdir(parents=True)
    (tl_root / "partial").write_text("half", encoding="utf-8")  # no .git yet
    log_path = tmp_path / "run.log"
    monkeypatch.setenv("TRACELENS_REPO", source)
    monkeypatch.setenv("TRACELENS_REF", "HEAD")

    tl_module._ensure_tracelens_checkout(tl_root, log_path=log_path)
    assert (tl_root / ".git").exists()
    assert (tl_root / "marker.txt").exists()


def test_is_default_tracelens_root_distinguishes_override(tl_module, tmp_path, monkeypatch):
    """Only the installer-managed default path is self-healed; an operator
    override path is not (mirrors handler / install.sh semantics)."""
    monkeypatch.setenv("HYPERLOOM_OPEN_SOURCE_ROOT", str(tmp_path / "podlocal"))
    default_root = tmp_path / "podlocal" / "TraceLens"
    override_root = tmp_path / "operator" / "TraceLens"
    assert tl_module._is_default_tracelens_root(default_root) is True
    assert tl_module._is_default_tracelens_root(override_root) is False


def test_incomplete_non_default_override_is_unusable_and_not_default(tl_module, tmp_path, monkeypatch):
    """A non-default override dir that exists but lacks .git is both
    'not default' (so main won't self-heal it) and 'not complete' (so main's
    post-check fails fast)."""
    monkeypatch.setenv("HYPERLOOM_OPEN_SOURCE_ROOT", str(tmp_path / "podlocal"))
    override = tmp_path / "operator" / "TraceLens"
    override.mkdir(parents=True)
    (override / "partial").write_text("half", encoding="utf-8")  # no .git
    assert tl_module._is_default_tracelens_root(override) is False
    assert tl_module._tracelens_checkout_complete(override) is False


def test_selfheal_raises_and_cleans_up_when_ref_unpinnable(tl_module, tmp_path, monkeypatch):
    """A non-HEAD ref that cannot be fetched must raise (never ship an
    unpinned default HEAD) and leave no target or temp dir behind."""
    source = _make_source_repo(tmp_path / "src" / "TraceLens")
    tl_root = tmp_path / "open-source-repos" / "TraceLens"
    log_path = tmp_path / "run.log"
    monkeypatch.setenv("TRACELENS_REPO", source)
    monkeypatch.setenv("TRACELENS_REF", "0" * 40)  # nonexistent sha

    with pytest.raises(FileNotFoundError):
        tl_module._ensure_tracelens_checkout(tl_root, log_path=log_path)
    assert not tl_root.exists()
    # No leftover temp/heal dirs in the parent.
    leftovers = [p.name for p in (tl_root.parent).glob(".TraceLens.*")]
    assert leftovers == [], leftovers
