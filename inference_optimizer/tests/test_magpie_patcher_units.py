"""Targeted unit tests for ``action_executors._magpie_patcher``.

Covers the resolution helpers (``_resolve_benchmarker_path``,
``_is_patched``) and the apply/skip branches of
``ensure_magpie_atomic_scripts_patch``. The full apply path is also
exercised here via a temporary benchmarker.py fixture so we lock in
the sentinel-check + atomic-replace semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _magpie_patcher as mp


# Minimal stand-in for the upstream legacy block, byte-for-byte equal to
# the production ``_LEGACY_BLOCK``. Wrapped in a tiny function body so the
# patched file still tokenises as valid Python and we can re-import to
# verify roundtrip.
_LEGACY_FILE = (
    "def stub():\n"
    "    pass\n"
    "    # block\n"
    + mp._LEGACY_BLOCK
)


@pytest.fixture
def magpie_dir(tmp_path: Path) -> Path:
    target = tmp_path / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    target.parent.mkdir(parents=True)
    target.write_text(_LEGACY_FILE)
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_benchmarker_path
# ---------------------------------------------------------------------------

class TestResolveBenchmarkerPath:
    def test_explicit_dir_returns_existing_file(self, magpie_dir):
        out = mp._resolve_benchmarker_path(magpie_dir)
        assert out is not None
        assert out.name == "benchmarker.py"

    def test_returns_none_when_dir_missing(self, tmp_path):
        assert mp._resolve_benchmarker_path(tmp_path) is None

    def test_returns_none_when_no_input_or_env(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mp._resolve_benchmarker_path(None) is None

    def test_env_fallback(self, monkeypatch, magpie_dir):
        monkeypatch.setenv("MAGPIE_DIR", str(magpie_dir))
        out = mp._resolve_benchmarker_path(None)
        assert out is not None


# ---------------------------------------------------------------------------
# _is_patched
# ---------------------------------------------------------------------------

class TestIsPatched:
    def test_false_when_legacy(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._is_patched(target) is False

    def test_true_after_apply(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._apply_patch_atomic(target) is True
        assert mp._is_patched(target) is True

    def test_returns_false_when_unreadable(self, tmp_path, monkeypatch):
        ghost = tmp_path / "no.py"

        def boom(self, **kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(Path, "read_text", boom)
        assert mp._is_patched(ghost) is False


# ---------------------------------------------------------------------------
# _apply_patch_atomic
# ---------------------------------------------------------------------------

class TestApplyPatchAtomic:
    def test_returns_false_when_legacy_block_missing(self, tmp_path):
        target = tmp_path / "benchmarker.py"
        target.write_text("def foo():\n    pass\n")
        assert mp._apply_patch_atomic(target) is False
        # Content unchanged.
        assert "Hyperloom #C1 patch" not in target.read_text()

    def test_returns_false_when_read_fails(self, tmp_path, monkeypatch):
        target = tmp_path / "benchmarker.py"

        def boom(self, **kwargs):
            raise OSError("io")

        monkeypatch.setattr(Path, "read_text", boom)
        assert mp._apply_patch_atomic(target) is False

    def test_applies_patch_and_returns_true(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._apply_patch_atomic(target) is True
        assert mp._PATCH_SENTINEL in target.read_text()
        # File still has the original ``def stub`` body.
        assert "def stub" in target.read_text()


# ---------------------------------------------------------------------------
# ensure_magpie_atomic_scripts_patch
# ---------------------------------------------------------------------------

class TestEnsurePatch:
    def test_returns_false_without_magpie_tree(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mp.ensure_magpie_atomic_scripts_patch(None) is False

    def test_full_roundtrip_idempotent(self, magpie_dir):
        # First call: applies the patch.
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        # Second call: hits the fast path (no flock).
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        # Only one copy of the sentinel — the patch ran exactly once.
        assert target.read_text().count(mp._PATCH_SENTINEL) == 1
