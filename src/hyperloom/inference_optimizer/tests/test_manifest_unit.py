# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the session manifest writer (provenance helpers, image
detection, objective derivation, and the atomic write/load round-trip)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.timeutil import utc_now_compact
from hyperloom.inference_optimizer.session import manifest as mf


class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


# ---- small helpers --------------------------------------------------------
def test_utc_now_compact_and_session_id():
    assert utc_now_compact().endswith("Z")
    sid = mf.build_session_id("meta/llama")
    assert sid.startswith("meta_llama_")
    assert mf.build_session_id("").startswith("session_")


# ---- git helpers ----------------------------------------------------------
def test_git_revision_at_success(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(0, "abc1234\n"))
    assert mf._git_revision_at(Path("/repo")) == "abc1234"
    assert mf._git_revision() == "abc1234"


def test_git_revision_at_nonzero_and_error(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(1, ""))
    assert mf._git_revision_at(Path("/repo")) == ""

    def _raise(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(mf.subprocess, "run", _raise)
    assert mf._git_revision_at(Path("/repo")) == ""


def test_git_remote_at(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(0, "git@github.com:x/y.git\n"))
    assert mf._git_remote_at(Path("/repo")) == "git@github.com:x/y.git"
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(2, ""))
    assert mf._git_remote_at(Path("/repo")) == ""

    def _raise(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(mf.subprocess, "run", _raise)
    assert mf._git_remote_at(Path("/repo")) == ""


# ---- path containment + dependency escape warning -------------------------
def test_is_path_within(tmp_path):
    assert mf._paths.is_path_within(tmp_path / "a", tmp_path) is True
    assert mf._paths.is_path_within(Path("/etc"), tmp_path) is False
    assert mf._paths.is_path_within(tmp_path / ".." / "elsewhere", tmp_path) is False


def test_warn_if_dependency_escapes_no_user_data(monkeypatch):
    monkeypatch.delenv(mf._paths.ENV_USER_DATA_PATH, raising=False)
    mf._warn_if_dependency_escapes_user_data("MAGPIE_PATH", "/workspace/x")


def test_warn_if_dependency_inside_user_data(monkeypatch, tmp_path):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, str(tmp_path))
    mf._warn_if_dependency_escapes_user_data("MAGPIE_PATH", str(tmp_path / "dep"))


def test_warn_if_dependency_pod_local(monkeypatch):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, "/data/persist")
    # /workspace is pod-local and outside user_data -> warns.
    mf._warn_if_dependency_escapes_user_data("MAGPIE_PATH", "/workspace/dep")


def test_warn_if_dependency_shared_no_warn(monkeypatch):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, "/data/persist")
    mf._warn_if_dependency_escapes_user_data("MAGPIE_PATH", "/shared/mirror/dep")


# ---- _describe_dep / _build_dependencies ----------------------------------
def test_describe_dep_unset(monkeypatch):
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert mf._describe_dep("MAGPIE_PATH") == {"path": "", "commit": "", "remote": ""}


def test_describe_dep_not_a_dir(monkeypatch):
    monkeypatch.setenv("MAGPIE_PATH", "/nonexistent/path/xyz")
    out = mf._describe_dep("MAGPIE_PATH")
    assert out["path"] == "/nonexistent/path/xyz"
    assert out["commit"] == ""


def test_describe_dep_real_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path))
    monkeypatch.setattr(mf, "_git_revision_at", lambda p: "deadbee")
    monkeypatch.setattr(mf, "_git_remote_at", lambda p: "http://r")
    out = mf._describe_dep("MAGPIE_PATH")
    assert out == {"path": str(tmp_path), "commit": "deadbee", "remote": "http://r"}


def test_build_dependencies(monkeypatch):
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    deps = mf._build_dependencies()
    assert set(deps) == {"magpie", "inferencex"}


# ---- _detect_image --------------------------------------------------------
def test_detect_image_env(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_IMAGE", "myimage:tag")
    assert mf._detect_image() == "myimage:tag"


def test_detect_image_marker(monkeypatch, tmp_path):
    for v in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(v, raising=False)
    marker = tmp_path / "image"
    marker.write_text("frommarker:1\n", encoding="utf-8")
    monkeypatch.setattr(mf, "Path", Path)  # keep real Path
    import hyperloom.inference_optimizer.session.manifest as mod

    real_path = mod.Path

    def _fake_path(arg):
        if arg == "/etc/podinfo/image":
            return marker
        return real_path(arg)

    monkeypatch.setattr(mod, "Path", _fake_path)
    assert mf._detect_image() == "frommarker:1"


def test_detect_image_cgroup(monkeypatch):
    for v in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(v, raising=False)
    import hyperloom.inference_optimizer.session.manifest as mod

    cgroup_text = "12:devices:/docker/0123456789abcdef0123\n7:cpu:/other\n"

    class _Cgroup:
        def exists(self):
            return True

        def read_text(self, **k):
            return cgroup_text

    class _NoMarker:
        def exists(self):
            return False

    def _fake_path(arg):
        if str(arg) == "/proc/1/cgroup":
            return _Cgroup()
        return _NoMarker()

    monkeypatch.setattr(mod, "Path", _fake_path)
    assert mf._detect_image() == "unknown@0123456789ab"


def test_detect_image_marker_oserror(monkeypatch):
    for v in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(v, raising=False)
    import hyperloom.inference_optimizer.session.manifest as mod

    class _BadMarker:
        def exists(self):
            return True

        def read_text(self, **k):
            raise OSError("denied")

    class _NoCgroup:
        def exists(self):
            return False

    def _fake_path(arg):
        if str(arg).startswith("/etc/"):
            return _BadMarker()
        return _NoCgroup()

    monkeypatch.setattr(mod, "Path", _fake_path)
    assert mf._detect_image() is None


def test_detect_image_none(monkeypatch):
    for v in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(v, raising=False)
    import hyperloom.inference_optimizer.session.manifest as mod

    real_path = mod.Path

    class _Missing:
        def exists(self):
            return False

    def _fake_path(arg):
        if str(arg).startswith("/etc/") or str(arg) == "/proc/1/cgroup":
            return _Missing()
        return real_path(arg)

    monkeypatch.setattr(mod, "Path", _fake_path)
    assert mf._detect_image() is None


# ---- _objective_summary ---------------------------------------------------
def test_objective_summary_variants():
    assert mf._objective_summary(SimpleNamespace(target_gain=10))["kind"] == "gain_pct"
    assert mf._objective_summary(SimpleNamespace(target_gain=None, target_tput=900))["kind"] == "tput"
    assert (
        mf._objective_summary(SimpleNamespace(target_gain=None, target_tput=None, target_baseline_dir="/b"))["kind"]
        == "baseline"
    )
    assert (
        mf._objective_summary(SimpleNamespace(target_gain=None, target_tput=None, target_baseline_dir=None))["kind"]
        == "time_only"
    )


# ---- build / write / load -------------------------------------------------
def test_build_manifest_without_args(monkeypatch):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["schema_version"] == mf.SCHEMA_VERSION
    assert m["framework"] == "sglang"
    assert m["objective"]["kind"] == "time_only"


def test_build_manifest_with_args(monkeypatch):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    for v in ("ISL", "OSL", "CONC", "TP", "MAX_MODEL_LEN"):
        monkeypatch.delenv(v, raising=False)
    args = argparse.Namespace(
        model="/models/llama",
        framework="vllm",
        gpu_type="mi300x",
        isl=128,
        osl=256,
        precision="fp8",
        target_gain=5.0,
        target_tput=None,
        target_baseline_dir=None,
        max_hours=2,
        research_lane_capacity=3,
        gpu_specialist_capacity=2,
        kb_degraded_reason=None,
        pr_degraded_reason=None,
    )
    m = mf.build_manifest(Path("/tmp/sd"), args=args, session_id="sid-1")
    assert m["session_id"] == "sid-1"
    assert m["model_name"] == "llama"
    assert m["framework"] == "vllm"
    assert m["workload"]["isl"] == 128
    assert m["objective"]["kind"] == "gain_pct"
    assert m["max_minutes"] == 120
    assert m["research_lane_capacity"] == 3


def test_build_manifest_shared_provenance_fields(monkeypatch):
    """Schema v4 carries gfx/EP/graph-mode/server-args from the shared WP-0
    provenance builder (kept in lockstep with the TraceShapeManifest)."""
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.setattr(
        mf,
        "build_provenance",
        lambda *a, **k: {
            "gfx_arch": "gfx950",
            "ep": 8,
            "graph_mode": "graph_capture",
            "server_args": ["--tp", "1"],
            "server_args_hash": "abc123",
        },
    )
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["schema_version"] == 4
    assert m["gfx_arch"] == "gfx950"
    assert m["ep"] == 8
    assert m["graph_mode"] == "graph_capture"
    assert m["server_args"] == ["--tp", "1"]
    assert m["server_args_hash"] == "abc123"


def test_manifest_versions_a_framework_installed_in_its_own_venv(monkeypatch, tmp_path):
    """``--framework-env isolated`` is the default for vLLM, so the framework is
    installed where the orchestrator's interpreter cannot see it. The manifest is
    the copy the KB row, the specialist prompt and resume all read, and each of
    them drops ``unknown`` -- so a fingerprint that degrades here is absent from
    all three, not just from the run report.
    """
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    venv_root = tmp_path / "vllm-venv"
    info = venv_root / "lib" / "python3.12" / "site-packages" / "vllm-0.27.1+rocm723.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.4\nName: vllm\nVersion: 0.27.1+rocm723\n", encoding="utf-8")
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setenv("HYPERLOOM_RESOLVED_FRAMEWORK", "vllm")
    monkeypatch.setenv("HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON", str(venv_root / "bin" / "python"))
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["stack_fingerprint"]["vllm"] == "0.27.1+rocm723"


def test_build_manifest_snapshots_user_data_path_from_env(monkeypatch, tmp_path):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, str(tmp_path / "ud"))
    m = mf.build_manifest(tmp_path / "ud" / "sess")
    assert m["user_data_path"] == str(tmp_path / "ud")


def test_build_manifest_user_data_path_falls_back_to_workspace_root(monkeypatch):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.delenv(mf._paths.ENV_USER_DATA_PATH, raising=False)
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["user_data_path"] == str(mf._paths.workspace_root())
    assert m["user_data_path"]


def test_write_and_load_manifest_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    written = mf.write_manifest(tmp_path, session_id="sid-x")
    loaded = mf.load_manifest(tmp_path)
    assert loaded["session_id"] == "sid-x"
    assert loaded == written


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.load_manifest(tmp_path / "no-session")
