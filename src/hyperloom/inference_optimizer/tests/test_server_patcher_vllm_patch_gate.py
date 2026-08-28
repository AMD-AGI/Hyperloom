"""vLLM takes the TraceLens patch at every supported version, 0.26+ included.

Upstream took the profiler *config fields* from 0.26 but not the graph-capture
implementation, which moved into the v2 runner and stayed patch-only. So on
0.26+ the config fields no longer prove the patch landed -- only the module the
patch creates at ``vllm/profiler/graph_capture.py`` does. Trusting the fields
there reports success while ``capture_traces/`` is never written, which is a
silent loss of the graph-capture sidecars for a whole profile run.
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


def _write_graph_capture(tmp_path, *, markers=sp._VLLM_GRAPH_CAPTURE_MARKERS):
    """Create the module the 0.26+ patch adds, standing in for a patched tree."""
    path = tmp_path.joinpath(*sp._VLLM_GRAPH_CAPTURE_SENTINEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(markers), encoding="utf-8")
    return path


def _fake_tracelens(tmp_path, versions):
    """A TraceLens root carrying patch files for ``versions``."""
    root = tmp_path / "tracelens"
    patches = sp._patch_tree(root, "vllm_patches")
    patches.mkdir(parents=True)
    for v in versions:
        (patches / f"config_vllm_v{v}.patch").write_text("dummy\n", encoding="utf-8")
    return root


def _stub_patch_path(monkeypatch, *, applied=True):
    """Route the patch path through a stub, recording the reused probe."""
    seen: dict[str, object] = {}

    def _plan(arg, *, install=None):
        seen["install"] = install
        return "plan"

    monkeypatch.setattr(sp, "_discover_vllm_plan", _plan)
    monkeypatch.setattr(sp, "_ensure_patched", lambda plan: applied and plan == "plan")
    return seen


def test_native_version_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    # The probe is reused rather than repeated.
    assert seen["install"] == ("0.26.0", tmp_path)


def test_native_local_build_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.27.1rc1.dev12+g23d65ff98.rocm722")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    assert seen["install"][0] == "0.27.1rc1.dev12+g23d65ff98.rocm722"


def test_pre_native_version_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.25.0")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is True
    assert seen["install"] == ("0.25.0", tmp_path)


def test_failed_apply_fails_soft(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    _stub_patch_path(monkeypatch, applied=False)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is False


def test_absent_plan_fails_soft(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    monkeypatch.setattr(sp, "_discover_vllm_plan", lambda arg, *, install=None: None)
    assert sp.ensure_vllm_patched_for_tracelens(tmp_path / "tracelens") is False


def test_undiscoverable_vllm_fails_soft(monkeypatch):
    monkeypatch.setattr(sp, "_discover_vllm_install", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        sp,
        "_discover_vllm_plan",
        lambda arg, *, install=None: calls.append("plan"),
    )
    assert sp.ensure_vllm_patched_for_tracelens(None) is False
    assert calls == []


def test_stock_native_install_is_not_mistaken_for_patched(tmp_path, monkeypatch):
    # The regression: a stock 0.28 carries both config markers natively, so the
    # base sentinel alone says "already patched" and the patch gets skipped.
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    root = _fake_tracelens(tmp_path, ["0.25.0", "0.28.0"])
    plan = sp._discover_vllm_plan(root, install=("0.28.1", tmp_path))
    assert plan is not None
    assert sp._markers_present(plan.sentinel_file, plan.sentinel_text) is True
    assert plan.extra_sentinels != ()
    assert sp._is_patched(plan) is False


def test_patched_native_install_is_detected(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    _write_graph_capture(tmp_path)
    root = _fake_tracelens(tmp_path, ["0.28.0"])
    plan = sp._discover_vllm_plan(root, install=("0.28.1", tmp_path))
    assert plan is not None
    assert sp._is_patched(plan) is True


def test_partial_graph_capture_module_is_not_patched(tmp_path, monkeypatch):
    # A module present but missing a marker is an incomplete apply, not a patch.
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    _write_graph_capture(tmp_path, markers=("graph_capture_profiler",))
    root = _fake_tracelens(tmp_path, ["0.28.0"])
    plan = sp._discover_vllm_plan(root, install=("0.28.1", tmp_path))
    assert plan is not None
    assert sp._is_patched(plan) is False


def test_pre_native_plan_keeps_the_config_fields_as_proof(tmp_path, monkeypatch):
    # Below 0.26 the patch writes the config fields, so they do prove it landed
    # and no graph-capture module is expected.
    _fake_install(monkeypatch, tmp_path, "0.25.0")
    root = _fake_tracelens(tmp_path, ["0.25.0"])
    plan = sp._discover_vllm_plan(root, install=("0.25.0", tmp_path))
    assert plan is not None
    assert plan.extra_sentinels == ()
    assert sp._is_patched(plan) is True


def test_graph_capture_sentinel_boundary(tmp_path):
    assert sp._vllm_graph_capture_sentinel("0.25.9", tmp_path) == ()
    assert sp._vllm_graph_capture_sentinel("", tmp_path) == ()
    for version in ("0.26", "0.26.0", "0.28.1rc0.dev8+g75dea9b4ae"):
        got = sp._vllm_graph_capture_sentinel(version, tmp_path)
        assert len(got) == 1
        path, markers = got[0]
        assert path == tmp_path / "vllm" / "profiler" / "graph_capture.py"
        assert markers == sp._VLLM_GRAPH_CAPTURE_MARKERS


def test_markers_present_requires_every_marker(tmp_path):
    path = Path(tmp_path) / "f.py"
    path.write_text("alpha\n", encoding="utf-8")
    assert sp._markers_present(path, ("alpha",)) is True
    assert sp._markers_present(path, ("alpha", "beta")) is False
    assert sp._markers_present(tmp_path / "missing.py", ("alpha",)) is False
