# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the session manifest writer (provenance helpers, image
detection, objective derivation, and the atomic write/load round-trip)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer import manifest as mf


class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


# ---- small helpers --------------------------------------------------------
def test_utc_now_compact_and_session_id():
    assert mf._utc_now_compact().endswith("Z")
    sid = mf.build_session_id("meta/llama")
    assert sid.startswith("meta_llama_")
    assert mf.build_session_id("").startswith("session_")


def test_read_first_line(tmp_path):
    assert mf._read_first_line(tmp_path / "missing.txt") == ""
    f = tmp_path / "v.txt"
    f.write_text("\n\n  hello \nworld\n", encoding="utf-8")
    assert mf._read_first_line(f) == "hello"


def test_read_first_line_blank_only(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("\n\n   \n", encoding="utf-8")
    assert mf._read_first_line(f) == ""


def test_read_first_line_oserror(tmp_path):
    # A directory exists() True but read_text raises IsADirectoryError (OSError).
    assert mf._read_first_line(tmp_path) == ""


def test_detect_stack_fingerprint_package_imports(monkeypatch):
    import sys
    for var in ("SGLANG_VERSION", "SGL_VERSION", "VLLM_VERSION",
                "AITER_COMMIT", "AITER_VERSION", "ROCM_VERSION", "HIP_VERSION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mf, "_read_first_line", lambda p: "")
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(__version__="0.5"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.7"))
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace(__commit__="cafe"))
    out = mf._detect_stack_fingerprint()
    assert out["sglang"] == "0.5"
    assert out["vllm"] == "0.7"
    assert out["aiter"] == "cafe"


# ---- stack fingerprint ----------------------------------------------------
def test_detect_stack_fingerprint_env_and_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_VERSION", "0.4.1")
    monkeypatch.setenv("VLLM_VERSION", "0.6.0")
    monkeypatch.delenv("ROCM_VERSION", raising=False)
    monkeypatch.delenv("HIP_VERSION", raising=False)
    monkeypatch.delenv("AITER_COMMIT", raising=False)
    monkeypatch.delenv("AITER_VERSION", raising=False)
    # rocm marker file read
    marker = tmp_path / "version"
    marker.write_text("6.2.0\n", encoding="utf-8")
    monkeypatch.setattr(mf, "_read_first_line",
                        lambda p: "6.2.0" if "version" in str(p) else "")
    out = mf._detect_stack_fingerprint()
    assert out["sglang"] == "0.4.1"
    assert out["vllm"] == "0.6.0"
    assert out["rocm"] == "6.2.0"
    assert out["aiter"] == "unknown"  # no env, no pkg


# ---- git helpers ----------------------------------------------------------
def test_git_revision_at_success(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run",
                        lambda *a, **k: _Proc(0, "abc1234\n"))
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
    monkeypatch.setattr(mf.subprocess, "run",
                        lambda *a, **k: _Proc(0, "git@github.com:x/y.git\n"))
    assert mf._git_remote_at(Path("/repo")) == "git@github.com:x/y.git"
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(2, ""))
    assert mf._git_remote_at(Path("/repo")) == ""

    def _raise(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(mf.subprocess, "run", _raise)
    assert mf._git_remote_at(Path("/repo")) == ""


# ---- path containment + dependency escape warning -------------------------
def test_path_is_relative_to(tmp_path):
    assert mf._path_is_relative_to(tmp_path / "a", tmp_path) is True
    assert mf._path_is_relative_to(Path("/etc"), tmp_path) is False


def test_warn_if_dependency_escapes_no_user_data(monkeypatch):
    monkeypatch.delenv(mf._paths.ENV_USER_DATA_PATH, raising=False)
    # no USER_DATA_PATH -> early return, no raise
    mf._warn_if_dependency_escapes_user_data("MAGPIE_DIR", "/workspace/x")


def test_warn_if_dependency_inside_user_data(monkeypatch, tmp_path):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, str(tmp_path))
    mf._warn_if_dependency_escapes_user_data("MAGPIE_DIR", str(tmp_path / "dep"))


def test_warn_if_dependency_pod_local(monkeypatch):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, "/data/persist")
    # /workspace is pod-local and outside user_data -> warns (no raise)
    mf._warn_if_dependency_escapes_user_data("MAGPIE_DIR", "/workspace/dep")


def test_warn_if_dependency_shared_no_warn(monkeypatch):
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, "/data/persist")
    mf._warn_if_dependency_escapes_user_data("MAGPIE_DIR", "/shared/mirror/dep")


# ---- _describe_dep / _build_dependencies ----------------------------------
def test_describe_dep_unset(monkeypatch):
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert mf._describe_dep("MAGPIE_DIR") == {"path": "", "commit": "", "remote": ""}


def test_describe_dep_not_a_dir(monkeypatch):
    monkeypatch.setenv("MAGPIE_DIR", "/nonexistent/path/xyz")
    out = mf._describe_dep("MAGPIE_DIR")
    assert out["path"] == "/nonexistent/path/xyz"
    assert out["commit"] == ""


def test_describe_dep_real_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGPIE_DIR", str(tmp_path))
    monkeypatch.setattr(mf, "_git_revision_at", lambda p: "deadbee")
    monkeypatch.setattr(mf, "_git_remote_at", lambda p: "http://r")
    out = mf._describe_dep("MAGPIE_DIR")
    assert out == {"path": str(tmp_path), "commit": "deadbee", "remote": "http://r"}


def test_build_dependencies(monkeypatch):
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
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
    # Point the marker scan at our temp file via monkeypatching the tuple
    import inference_optimizer.manifest as mod

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
    import inference_optimizer.manifest as mod
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
    import inference_optimizer.manifest as mod

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
    import inference_optimizer.manifest as mod
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
    assert mf._objective_summary(
        SimpleNamespace(target_gain=None, target_tput=900))["kind"] == "tput"
    assert mf._objective_summary(SimpleNamespace(
        target_gain=None, target_tput=None,
        target_baseline_dir="/b"))["kind"] == "baseline"
    assert mf._objective_summary(SimpleNamespace(
        target_gain=None, target_tput=None,
        target_baseline_dir=None))["kind"] == "time_only"


# ---- build / write / load -------------------------------------------------
def test_build_manifest_without_args(monkeypatch):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.setattr(mf, "_detect_stack_fingerprint", lambda: {})
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["schema_version"] == mf.SCHEMA_VERSION
    assert m["framework"] == "sglang"
    assert m["objective"]["kind"] == "time_only"
    assert m["warm_replay_enabled"] is True


def test_build_manifest_with_args(monkeypatch):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.setattr(mf, "_detect_stack_fingerprint", lambda: {})
    for v in ("ISL", "OSL", "CONC", "TP", "MAX_MODEL_LEN"):
        monkeypatch.delenv(v, raising=False)
    args = argparse.Namespace(
        model="/models/llama", framework="vllm", gpu_type="mi300x",
        isl=128, osl=256, precision="fp8", target_gain=5.0,
        target_tput=None, target_baseline_dir=None, max_hours=2,
        research_lane_capacity=3, gpu_specialist_capacity=2,
        kb_degraded_reason=None, pr_degraded_reason=None,
        no_warm_replay=True, warm_replay_min_confidence=0.6,
        warm_replay_min_reproduce_pct=0.9,
    )
    m = mf.build_manifest(Path("/tmp/sd"), args=args, session_id="sid-1")
    assert m["session_id"] == "sid-1"
    assert m["model_name"] == "llama"
    assert m["framework"] == "vllm"
    assert m["workload"]["isl"] == 128
    assert m["objective"]["kind"] == "gain_pct"
    assert m["max_minutes"] == 120
    assert m["research_lane_capacity"] == 3
    assert m["warm_replay_enabled"] is False


def test_write_and_load_manifest_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(mf, "_git_revision", lambda: "rev1")
    monkeypatch.setattr(mf, "_build_dependencies", lambda: {})
    monkeypatch.setattr(mf, "_detect_image", lambda: None)
    monkeypatch.setattr(mf, "_detect_stack_fingerprint", lambda: {})
    written = mf.write_manifest(tmp_path, session_id="sid-x")
    loaded = mf.load_manifest(tmp_path)
    assert loaded["session_id"] == "sid-x"
    assert loaded == written


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.load_manifest(tmp_path / "no-session")
