# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_xdit_patcher`` — a VERIFIER, not a mutator.

The two diffusion-profiling adaptations (``repeat=1`` + per-denoise-step
markers) are baked into the sandbox image. This module verifies the baked
sentinels are present in the running xfuser and fails soft (returns ``False``,
logs remediation) when they are missing.

Fixtures synthesize a fake xfuser tree in ``tmp_path`` discovered via
``$XDIT_PATH`` (``base_model.py`` with / without the baked sentinels).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _xdit_patcher
from hyperloom.orchestrator.actions.executors._xdit_patcher import (
    _discover_xfuser_base_models,
    _is_baked,
    ensure_xdit_profiler_patched,
    verify_xdit_profiler_baked,
)

_REL = ("xfuser", "model_executor", "models", "runner_models", "base_model.py")
_PROFILER_SENTINEL = "# hyperloom: retain active window"
_ANNOT_SENTINEL = "# hyperloom: per-denoise-step annotation"

# A base_model.py carrying BOTH baked adaptations.
_BAKED_FIXTURE = """\
class xFuserModel:
    def profile(self, input_args):
        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
            repeat=1,  # hyperloom: retain active window (upstream repeat=0 -> empty trace)
        )
        with profile(schedule=schedule) as profile_object:
            # hyperloom: per-denoise-step annotation -- wrap scheduler.step
            _hl_denoise_step = {"i": 0}
            for iteration in range(1):
                _hl_reset_denoise_counter()  # hyperloom: per-denoise-step annotation
                with record_function("model_inference"):
                    self._run_timed_pipe(input_args)
                profile_object.step()
"""

# The clean upstream shape (neither adaptation baked) -> verify must be False.
_PRISTINE_FIXTURE = """\
class xFuserModel:
    def profile(self, input_args):
        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
        )
        with profile(schedule=schedule) as profile_object:
            for iteration in range(1):
                with record_function("model_inference"):
                    self._run_timed_pipe(input_args)
                profile_object.step()
"""


@pytest.fixture(autouse=True)
def _isolate_xdit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``$XDIT_PATH`` so a synthetic ``tmp_path`` test never discovers a
    real on-pod xDiT checkout (tests that exercise discovery re-set it)."""
    monkeypatch.delenv("XDIT_PATH", raising=False)


def _write_fake_xdit(tmp_path: Path, contents: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    return target


# ---------------------------------------------------------------------------
# verify_xdit_profiler_baked
# ---------------------------------------------------------------------------
def test_verify_true_when_both_sentinels_present(tmp_path: Path, monkeypatch) -> None:
    target = _write_fake_xdit(tmp_path, _BAKED_FIXTURE, monkeypatch)
    assert verify_xdit_profiler_baked() is True
    assert _is_baked(target) is True


def test_verify_false_when_pristine(tmp_path: Path, monkeypatch) -> None:
    """A clean upstream base_model.py (no bake) fails soft: False, file untouched."""
    target = _write_fake_xdit(tmp_path, _PRISTINE_FIXTURE, monkeypatch)
    before = target.read_text()
    assert verify_xdit_profiler_baked() is False
    assert _is_baked(target) is False
    assert target.read_text() == before  # verifier never mutates


def test_verify_false_when_only_profiler_sentinel(tmp_path: Path, monkeypatch) -> None:
    """Partial bake (repeat=1 only, no per-step annotation) is not accepted."""
    partial = _PRISTINE_FIXTURE.replace(
        "            active=self.config.profile_active,\n",
        "            active=self.config.profile_active,\n"
        "            repeat=1,  # hyperloom: retain active window\n",
    )
    target = _write_fake_xdit(tmp_path, partial, monkeypatch)
    assert _PROFILER_SENTINEL in target.read_text()
    assert _ANNOT_SENTINEL not in target.read_text()
    assert verify_xdit_profiler_baked() is False


def test_verify_warns_with_remediation_when_missing(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    _write_fake_xdit(tmp_path, _PRISTINE_FIXTURE, monkeypatch)
    with caplog.at_level("WARNING"):
        assert verify_xdit_profiler_baked() is False
    assert "MISSING the baked" in caplog.text
    assert _xdit_patcher._REQUIRED_IMAGE in caplog.text


def test_verify_false_no_xdit_tree(monkeypatch) -> None:
    """No discoverable base_model.py (empty $XDIT_PATH, no xfuser) -> False."""
    monkeypatch.setattr(_xdit_patcher, "_discover_xfuser_base_models", lambda: [])
    assert verify_xdit_profiler_baked() is False


def test_verify_scans_all_discovered_and_accepts_any_baked(
    tmp_path: Path, monkeypatch
) -> None:
    """With multiple discovered files, one fully-baked copy is enough."""
    pristine = tmp_path / "a" / "base_model.py"
    pristine.parent.mkdir(parents=True)
    pristine.write_text(_PRISTINE_FIXTURE, encoding="utf-8")
    baked = tmp_path / "b" / "base_model.py"
    baked.parent.mkdir(parents=True)
    baked.write_text(_BAKED_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: [pristine, baked]
    )
    assert verify_xdit_profiler_baked() is True


def test_ensure_alias_is_verify() -> None:
    """The backward-compatible name maps to the verifier (callers unchanged)."""
    assert ensure_xdit_profiler_patched is verify_xdit_profiler_baked


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
    target.write_text(_BAKED_FIXTURE, encoding="utf-8")
    monkeypatch.delenv("XDIT_PATH", raising=False)

    class _Spec:
        submodule_search_locations = [str(pkg_root / "xfuser")]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: _Spec())
    found = _discover_xfuser_base_models()
    assert target.resolve() in found


def test_discovery_via_xdit_path(tmp_path: Path, monkeypatch) -> None:
    """``$XDIT_PATH`` resolves ``$XDIT_PATH/xfuser/.../base_model.py``."""
    target = _write_fake_xdit(tmp_path, _BAKED_FIXTURE, monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert target.resolve() in _discover_xfuser_base_models()
