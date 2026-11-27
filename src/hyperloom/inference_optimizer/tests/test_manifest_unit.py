# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the session manifest writer (provenance helpers, objective derivation, and the atomic write/load round-trip)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom_common.timeutil import utc_now_compact
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
def test_git_revision_at_nonzero_and_error(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", lambda *a, **k: _Proc(1, ""))
    assert mf._git_revision_at(Path("/repo")) == ""

    def _raise(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(mf.subprocess, "run", _raise)
    assert mf._git_revision_at(Path("/repo")) == ""


# ---- path containment -----------------------------------------------------
def test_is_path_within(tmp_path):
    assert mf._paths.is_path_within(tmp_path / "a", tmp_path) is True
    assert mf._paths.is_path_within(Path("/etc"), tmp_path) is False
    assert mf._paths.is_path_within(tmp_path / ".." / "elsewhere", tmp_path) is False


# ---- _build_dependencies --------------------------------------------------
def test_build_dependencies(monkeypatch):
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    deps = mf._build_dependencies()
    assert set(deps) == {"inferencex"}


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
    monkeypatch.setattr(mf, "_build_dependencies", dict)
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["schema_version"] == mf.SCHEMA_VERSION
    assert m["framework"] == "sglang"
    assert m["objective"]["kind"] == "time_only"


def test_build_manifest_with_args(monkeypatch):
    monkeypatch.setattr(mf, "_build_dependencies", dict)
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


def test_build_manifest_shared_provenance_fields(monkeypatch):
    """Schema v4 carries gfx/EP/graph-mode/server-args from the shared WP-0 provenance builder."""
    monkeypatch.setattr(mf, "_build_dependencies", dict)
    monkeypatch.setattr(
        mf,
        "build_provenance",
        lambda *a, **k: {
            "gfx_arch": "gfx950",
            "ep": 8,
            "graph_mode": "graph_capture",
            "server_args": ["--tp", "1"],
            "server_args_hash": "abc123",
            "code_revision": "rev1",
            "image": "img:tag",
        },
    )
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["schema_version"] == 4
    assert m["gfx_arch"] == "gfx950"
    assert m["ep"] == 8
    assert m["graph_mode"] == "graph_capture"
    assert m["server_args"] == ["--tp", "1"]
    assert m["server_args_hash"] == "abc123"
    assert (m["code_revision"], m["image"]) == ("rev1", "img:tag")


def test_the_manifest_records_the_gpu_identity_the_launch_published(monkeypatch):
    from hyperloom.common.provenance import GPU_IDENTITY_ENV

    monkeypatch.setattr(mf, "_build_dependencies", dict)
    identity = {
        "product": "w7900",
        "marketing_name": "AMD Radeon Pro W7900 Dual Slot",
        "pci_device_id": "1002:744a",
        "gfx_arch": "gfx1100",
        "visible_gpus": 1,
    }
    monkeypatch.setenv(GPU_IDENTITY_ENV, json.dumps(identity))
    assert mf.build_manifest(Path("/tmp/sd"))["gpu_identity"] == identity


@pytest.mark.parametrize("published", [None, "", "{}", "not json", "[1]"])
def test_a_launch_that_published_no_identity_records_none(monkeypatch, published):
    from hyperloom.common.provenance import GPU_IDENTITY_ENV

    monkeypatch.setattr(mf, "_build_dependencies", dict)
    if published is None:
        monkeypatch.delenv(GPU_IDENTITY_ENV, raising=False)
    else:
        monkeypatch.setenv(GPU_IDENTITY_ENV, published)
    assert mf.build_manifest(Path("/tmp/sd"))["gpu_identity"] is None


def test_manifest_versions_a_framework_installed_in_its_own_venv(monkeypatch, tmp_path):
    """``--framework-env isolated`` is the default for vLLM, so the framework is installed where the orchestrator's interpreter cannot see it."""
    monkeypatch.setattr(mf, "_build_dependencies", dict)
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
    monkeypatch.setattr(mf, "_build_dependencies", dict)
    monkeypatch.setenv(mf._paths.ENV_USER_DATA_PATH, str(tmp_path / "ud"))
    m = mf.build_manifest(tmp_path / "ud" / "sess")
    assert m["user_data_path"] == str(tmp_path / "ud")


def test_build_manifest_user_data_path_falls_back_to_workspace_root(monkeypatch):
    monkeypatch.setattr(mf, "_build_dependencies", dict)
    monkeypatch.delenv(mf._paths.ENV_USER_DATA_PATH, raising=False)
    m = mf.build_manifest(Path("/tmp/sd"))
    assert m["user_data_path"] == str(mf._paths.workspace_root())
    assert m["user_data_path"]


def test_write_and_load_manifest_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(mf, "_build_dependencies", dict)
    written = mf.write_manifest(tmp_path, session_id="sid-x")
    loaded = mf.load_manifest(tmp_path)
    assert loaded["session_id"] == "sid-x"
    assert loaded == written


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.load_manifest(tmp_path / "no-session")


def test_git_remote_userinfo_is_stripped(monkeypatch):
    monkeypatch.setattr(mf, "_git_capture", lambda path, args: "https://user:tok3n@github.com/x/y.git")
    remote = mf._git_remote_at(Path("/repo"))
    assert "tok3n" not in remote and "user" not in remote
    assert remote == "https://github.com/x/y.git"
