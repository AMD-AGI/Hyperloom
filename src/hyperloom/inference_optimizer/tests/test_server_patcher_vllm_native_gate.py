"""vLLM >= 0.26 ships the profiler fields upstream, so the patch path is skipped.

The patch picker falls back to the nearest patch not newer than the running
vLLM, so without this gate a 0.26 install would be handed the 0.25 patch and
fail to apply against already-upstreamed code.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import _server_patcher as sp


def _fake_install(monkeypatch, tmp_path, version, *, markers=sp._VLLM_PROFILER_SENTINELS):
    """Stand up an install root whose profiler.py carries ``markers``."""
    sentinel = tmp_path / "vllm" / "config" / "profiler.py"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("\n".join(markers), encoding="utf-8")
    monkeypatch.setattr(sp, "_discover_vllm_install", lambda: (version, tmp_path))
    return sentinel


def _forbid_patching(monkeypatch):
    """Record any attempt to take the patch path."""
    calls: list[str] = []

    def _plan(arg, *, install=None):
        calls.append("plan")
        return None

    def _apply(plan):
        calls.append("apply")
        return True

    monkeypatch.setattr(sp, "_discover_vllm_plan", _plan)
    monkeypatch.setattr(sp, "_ensure_patched", _apply)
    return calls


def test_native_version_is_accepted_without_patching(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    calls = _forbid_patching(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    assert calls == []


def test_native_local_build_is_accepted_without_patching(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.27.1rc1.dev12+g23d65ff98.rocm722")
    calls = _forbid_patching(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    assert calls == []


def test_native_version_missing_the_fields_fails_soft(tmp_path, monkeypatch):
    # A 0.26 build without the upstream fields would reject the flags, and the
    # patch set does not cover it, so the caller has to drop them.
    _fake_install(monkeypatch, tmp_path, "0.26.0", markers=("something_else",))
    _forbid_patching(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is False


def test_pre_native_version_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.25.0")
    seen: dict[str, object] = {}

    def _plan(arg, *, install=None):
        seen["install"] = install
        return "plan"

    monkeypatch.setattr(sp, "_discover_vllm_plan", _plan)
    monkeypatch.setattr(sp, "_ensure_patched", lambda plan: plan == "plan")
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    # The probe is reused rather than repeated.
    assert seen["install"] == ("0.25.0", tmp_path)


def test_operator_pin_forces_the_patch_path_on_a_native_version(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    monkeypatch.setenv(sp._VLLM_EXACT_VERSIONS_ENV, "0.25.0")
    monkeypatch.setattr(sp, "_discover_vllm_plan", lambda arg, *, install=None: "plan")
    monkeypatch.setattr(sp, "_ensure_patched", lambda plan: True)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True


def test_undiscoverable_vllm_fails_soft(monkeypatch):
    monkeypatch.setattr(sp, "_discover_vllm_install", lambda: None)
    calls = _forbid_patching(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(None) is False
    assert calls == []


def test_native_boundary(monkeypatch):
    monkeypatch.delenv(sp._VLLM_EXACT_VERSIONS_ENV, raising=False)
    assert sp._vllm_profiler_is_native("0.26.0") is True
    assert sp._vllm_profiler_is_native("0.26") is True
    assert sp._vllm_profiler_is_native("0.25.9") is False
    assert sp._vllm_profiler_is_native("") is False


def test_markers_present_requires_every_marker(tmp_path):
    path = Path(tmp_path) / "f.py"
    path.write_text("alpha\n", encoding="utf-8")
    assert sp._markers_present(path, ("alpha",)) is True
    assert sp._markers_present(path, ("alpha", "beta")) is False
    assert sp._markers_present(tmp_path / "missing.py", ("alpha",)) is False
