# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_xdit_patcher.ensure_xdit_profiler_patched``.

Pins the patcher contract: a backward-compatible, idempotent, concurrency-safe,
fail-soft patch that inserts ``repeat=1`` into xDiT's ``torch.profiler.schedule``
so the diffusion profiler retains its active window (upstream default
``repeat=0`` discards it -> empty trace, Op count == 0). Fixtures synthesize a
fake xfuser tree in ``tmp_path`` discovered via ``$XDIT_PATH``.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _xdit_patcher
from inference_optimizer.orchestrator.action_executors._xdit_patcher import (
    _apply_patch_atomic,
    _discover_xfuser_base_models,
    ensure_xdit_profiler_patched,
)

# Verbatim upstream shape (incl. the exact indentation the patcher matches on).
_UPSTREAM_FIXTURE = """\
class xFuserModel:
    def profile(self, input_args):
        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
        )
        num_repetitions = self.config.profile_wait + self.config.profile_warmup + self.config.profile_active
        with profile(schedule=schedule) as profile_object:
            for iteration in range(num_repetitions):
                self._run_timed_pipe(input_args)
                profile_object.step()
"""

_REL = ("xfuser", "model_executor", "models", "runner_models", "base_model.py")
_SENTINEL = "# hyperloom: retain active window"


@pytest.fixture(autouse=True)
def _isolate_xdit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``$XDIT_PATH`` so a synthetic ``tmp_path`` test never discovers a
    real on-pod xDiT checkout (tests that exercise discovery re-set it)."""
    monkeypatch.delenv("XDIT_PATH", raising=False)


@pytest.fixture
def fake_xdit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal ``$XDIT_PATH/xfuser/.../base_model.py`` tree."""
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(_UPSTREAM_FIXTURE, encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    return target


def test_first_call_inserts_repeat_one(fake_xdit: Path) -> None:
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    text = fake_xdit.read_text()
    assert "repeat=1," in text
    assert _SENTINEL in text


def test_second_call_is_a_noop(fake_xdit: Path) -> None:
    """Idempotency: re-applying must not double-patch or change bytes."""
    ensure_xdit_profiler_patched()
    after_first = fake_xdit.read_text()
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    assert fake_xdit.read_text() == after_first


def test_repeat_appears_exactly_once(fake_xdit: Path) -> None:
    for _ in range(5):
        ensure_xdit_profiler_patched()
    assert fake_xdit.read_text().count("repeat=1") == 1


def test_patched_file_is_valid_python(fake_xdit: Path) -> None:
    import ast

    ensure_xdit_profiler_patched()
    ast.parse(fake_xdit.read_text())


def test_missing_legacy_block_is_fail_soft(tmp_path: Path, monkeypatch) -> None:
    """A file without the schedule block leaves it untouched, returns False."""
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    # Restrict discovery to the fake file so a real installed xfuser in the
    # environment cannot make the patcher report success for another root.
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: [target]
    )
    assert ensure_xdit_profiler_patched() is False
    assert target.read_text() == "x = 1\n"


def test_no_xdit_tree_is_fail_soft(monkeypatch) -> None:
    """No discoverable base_model.py (empty $XDIT_PATH, no xfuser) -> False."""
    monkeypatch.delenv("XDIT_PATH", raising=False)
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: []
    )
    assert ensure_xdit_profiler_patched() is False


# ---------------------------------------------------------------------------
# _discover_xfuser_base_models: discovery-path edge cases (fail-soft)
# ---------------------------------------------------------------------------
def test_discovery_skips_when_base_model_missing(tmp_path: Path, monkeypatch) -> None:
    """``$XDIT_PATH`` set but the ``base_model.py`` file is absent → skipped."""
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _discover_xfuser_base_models() == []


def test_discovery_survives_find_spec_error(monkeypatch) -> None:
    """A raising ``importlib.util.find_spec`` must not propagate → returns []."""
    monkeypatch.delenv("XDIT_PATH", raising=False)

    def _boom(name: str):
        raise ValueError("namespace package without a real spec")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert _discover_xfuser_base_models() == []


def test_discovery_uses_importable_xfuser_spec(tmp_path: Path, monkeypatch) -> None:
    """When ``$XDIT_PATH`` is unset, discovery falls back to the importable
    ``xfuser`` package location (``find_spec.submodule_search_locations``)."""
    pkg_root = tmp_path / "site-packages"
    target = pkg_root.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(_UPSTREAM_FIXTURE, encoding="utf-8")
    monkeypatch.delenv("XDIT_PATH", raising=False)

    class _Spec:
        submodule_search_locations = [str(pkg_root / "xfuser")]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: _Spec())
    found = _discover_xfuser_base_models()
    assert target.resolve() in found


# ---------------------------------------------------------------------------
# _apply_patch_atomic: IO / no-op edge cases (fail-soft, never corrupts)
# ---------------------------------------------------------------------------
def test_apply_returns_false_on_read_error(tmp_path: Path) -> None:
    """An unreadable path (here a directory) fails soft, returns False."""
    unreadable = tmp_path / "a_directory_not_a_file"
    unreadable.mkdir()
    assert _apply_patch_atomic(unreadable) is False


def test_apply_returns_false_when_patch_is_a_noop(fake_xdit: Path, monkeypatch) -> None:
    """If replacing the legacy block yields identical bytes, return False."""
    monkeypatch.setattr(
        _xdit_patcher, "_PATCHED_BLOCK", _xdit_patcher._LEGACY_BLOCK
    )
    assert _apply_patch_atomic(fake_xdit) is False


def test_apply_returns_false_when_write_fails(fake_xdit: Path, monkeypatch) -> None:
    """A failed atomic write is reported (False) without touching the file."""
    original = fake_xdit.read_text()
    monkeypatch.setattr(
        _xdit_patcher, "atomic_write_text", lambda *a, **k: False
    )
    assert _apply_patch_atomic(fake_xdit) is False
    assert fake_xdit.read_text() == original


def test_concurrent_calls_are_safe(fake_xdit: Path) -> None:
    """flock + atomic rename: concurrent patchers must not corrupt the file."""
    results: list[bool] = []

    def _worker() -> None:
        results.append(ensure_xdit_profiler_patched())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results)
    assert fake_xdit.read_text().count("repeat=1") == 1
