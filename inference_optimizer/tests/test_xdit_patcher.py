# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_xdit_patcher.ensure_xdit_profiler_patched``.

Pins the patcher contract: a backward-compatible, idempotent, concurrency-safe,
fail-soft patch that inserts ``repeat=1`` into xDiT's ``torch.profiler.schedule``
so the diffusion profiler retains its active window (upstream default
``repeat=0`` discards it -> empty trace, Op count == 0). Fixtures synthesize a
fake xfuser tree in ``tmp_path`` discovered via ``$XDIT_PATH``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _xdit_patcher
from inference_optimizer.orchestrator.action_executors._xdit_patcher import (
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
