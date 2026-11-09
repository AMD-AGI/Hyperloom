"""vLLM takes the TraceLens patch at every supported version, 0.26+ included."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import PatchedServer
from hyperloom.orchestrator.actions.executors import _patch_laying as laying
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


def _write_relocated_markers(tmp_path, *, capture_traces=True):
    """Create a patched 0.32+ tree: ``capture_traces`` lives in ``wrapper.py``, not ``graph_capture.py``."""
    graph_capture = tmp_path / "vllm" / "profiler" / "graph_capture.py"
    graph_capture.parent.mkdir(parents=True, exist_ok=True)
    graph_capture.write_text("graph_capture_profiler\n", encoding="utf-8")
    wrapper = tmp_path / "vllm" / "profiler" / "wrapper.py"
    wrapper.write_text("capture_traces\n" if capture_traces else "unrelated\n", encoding="utf-8")


def _fake_tracelens(monkeypatch, tmp_path, versions):
    """Point ``$TRACELENS_ROOT`` at a root carrying patch files for ``versions``."""
    root = tmp_path / "tracelens"
    patches = sp._patch_tree(root, "vllm_patches")
    patches.mkdir(parents=True)
    for v in versions:
        (patches / f"config_vllm_v{v}.patch").write_text("dummy\n", encoding="utf-8")
    monkeypatch.setenv("TRACELENS_ROOT", str(root))


def _stub_patch_path(monkeypatch, *, applied=True):
    """Route the patch path through a stub, recording the reused probe."""
    seen: dict[str, object] = {}

    def _plan(*, install):
        seen["install"] = install
        return "plan"

    monkeypatch.setattr(sp, "_discover_vllm_plan", _plan)
    monkeypatch.setattr(sp, "_ensure_patched", lambda plan: PatchedServer(None) if applied and plan == "plan" else None)
    return seen


def test_native_version_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens() is not None
    # The probe is reused rather than repeated.
    assert seen["install"] == ("0.26.0", tmp_path)


def test_native_local_build_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.27.1rc1.dev12+g23d65ff98.rocm722")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens() is not None
    assert seen["install"][0] == "0.27.1rc1.dev12+g23d65ff98.rocm722"


def test_pre_native_version_still_takes_the_patch_path(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.25.0")
    seen = _stub_patch_path(monkeypatch)
    assert sp.ensure_vllm_patched_for_tracelens() is not None
    assert seen["install"] == ("0.25.0", tmp_path)


def test_failed_apply_fails_soft(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    _stub_patch_path(monkeypatch, applied=False)
    assert sp.ensure_vllm_patched_for_tracelens() is None


def test_absent_plan_fails_soft(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.26.0")
    monkeypatch.setattr(sp, "_discover_vllm_plan", lambda *, install: None)
    assert sp.ensure_vllm_patched_for_tracelens() is None


def test_undiscoverable_vllm_fails_soft(monkeypatch):
    monkeypatch.setattr(sp, "_discover_vllm_install", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(
        sp,
        "_discover_vllm_plan",
        lambda *, install: calls.append("plan"),
    )
    assert sp.ensure_vllm_patched_for_tracelens() is None
    assert calls == []


def test_stock_native_install_is_not_mistaken_for_patched(tmp_path, monkeypatch):
    # The regression: a stock 0.28 carries both config markers natively, so the base sentinel alone says "already
    # patched" and the patch gets skipped.
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    _fake_tracelens(monkeypatch, tmp_path, ["0.25.0", "0.28.0"])
    plan = sp._discover_vllm_plan(install=("0.28.1", tmp_path))
    assert plan is not None
    assert laying._markers_present(plan.sentinel_file, plan.sentinel_text) is True
    assert plan.extra_sentinels != ()
    assert laying._is_patched(plan) is False


def test_patched_native_install_is_detected(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    _write_graph_capture(tmp_path)
    _fake_tracelens(monkeypatch, tmp_path, ["0.28.0"])
    plan = sp._discover_vllm_plan(install=("0.28.1", tmp_path))
    assert plan is not None
    assert laying._is_patched(plan) is True


def test_partial_graph_capture_module_is_not_patched(tmp_path, monkeypatch):
    # A module present but missing a marker is an incomplete apply, not a patch.
    _fake_install(monkeypatch, tmp_path, "0.28.1")
    _write_graph_capture(tmp_path, markers=("graph_capture_profiler",))
    _fake_tracelens(monkeypatch, tmp_path, ["0.28.0"])
    plan = sp._discover_vllm_plan(install=("0.28.1", tmp_path))
    assert plan is not None
    assert laying._is_patched(plan) is False


def test_pre_native_plan_keeps_the_config_fields_as_proof(tmp_path, monkeypatch):
    # Below 0.26 the patch writes the config fields, so they do prove it landed and no graph-capture module is
    # expected.
    _fake_install(monkeypatch, tmp_path, "0.25.0")
    _fake_tracelens(monkeypatch, tmp_path, ["0.25.0"])
    plan = sp._discover_vllm_plan(install=("0.25.0", tmp_path))
    assert plan is not None
    assert plan.extra_sentinels == ()
    assert laying._is_patched(plan) is True


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
    assert laying._markers_present(path, ("alpha",)) is True
    assert laying._markers_present(path, ("alpha", "beta")) is False
    assert laying._markers_present(tmp_path / "missing.py", ("alpha",)) is False


def test_patched_relocated_marker_install_is_detected(tmp_path, monkeypatch):
    # From 0.32 the patch writes ``capture_traces`` into ``wrapper.py`` and only the context manager stays in
    # ``graph_capture.py``; a correctly patched tree must read as patched.
    _fake_install(monkeypatch, tmp_path, "0.33.0+rocm100")
    _write_relocated_markers(tmp_path)
    _fake_tracelens(monkeypatch, tmp_path, ["0.33.0"])
    plan = sp._discover_vllm_plan(install=("0.33.0+rocm100", tmp_path))
    assert plan is not None
    assert laying._is_patched(plan) is True


def test_relocated_marker_missing_from_wrapper_is_not_patched(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.33.0+rocm100")
    _write_relocated_markers(tmp_path, capture_traces=False)
    _fake_tracelens(monkeypatch, tmp_path, ["0.33.0"])
    plan = sp._discover_vllm_plan(install=("0.33.0+rocm100", tmp_path))
    assert plan is not None
    assert laying._is_patched(plan) is False


def test_stock_relocated_marker_install_is_not_mistaken_for_patched(tmp_path, monkeypatch):
    _fake_install(monkeypatch, tmp_path, "0.33.0+rocm100")
    wrapper = tmp_path / "vllm" / "profiler" / "wrapper.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("unrelated\n", encoding="utf-8")
    _fake_tracelens(monkeypatch, tmp_path, ["0.33.0"])
    plan = sp._discover_vllm_plan(install=("0.33.0+rocm100", tmp_path))
    assert plan is not None
    assert laying._is_patched(plan) is False


def test_graph_capture_sentinel_relocates_at_0_32(tmp_path):
    legacy = sp._vllm_graph_capture_sentinel("0.31.0", tmp_path)
    assert legacy == ((tmp_path / "vllm" / "profiler" / "graph_capture.py", sp._VLLM_GRAPH_CAPTURE_MARKERS),)
    for version in ("0.32.0", "0.33.0", "0.33.0+rocm100"):
        got = sp._vllm_graph_capture_sentinel(version, tmp_path)
        assert got == (
            (tmp_path / "vllm" / "profiler" / "graph_capture.py", ("graph_capture_profiler",)),
            (tmp_path / "vllm" / "profiler" / "wrapper.py", ("capture_traces",)),
        )
