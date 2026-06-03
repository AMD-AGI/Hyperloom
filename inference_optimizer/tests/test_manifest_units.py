"""Unit tests for ``inference_optimizer.manifest`` helpers.

Existing tests round-trip the manifest writer at a high level; we
target the auxiliary helpers (objective summary, dependency provenance,
image detection fallbacks) so each branch has explicit coverage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer import manifest as mf


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

class TestObjectiveSummary:
    def test_gain_pct(self):
        ns = argparse.Namespace(target_gain=10.0)
        assert mf._objective_summary(ns) == {"kind": "gain_pct", "value": 10.0}

    def test_tput(self):
        ns = argparse.Namespace(target_gain=None, target_tput=200.0)
        assert mf._objective_summary(ns) == {"kind": "tput", "value": 200.0}

    def test_baseline_dir(self, tmp_path):
        ns = argparse.Namespace(
            target_gain=None,
            target_tput=None,
            target_baseline_dir=tmp_path / "x",
        )
        out = mf._objective_summary(ns)
        assert out["kind"] == "baseline"
        assert str(tmp_path / "x") in out["value"]

    def test_time_only_default(self):
        ns = argparse.Namespace()
        assert mf._objective_summary(ns) == {"kind": "time_only", "value": None}


# ---------------------------------------------------------------------------
# build_session_id
# ---------------------------------------------------------------------------

class TestBuildSessionId:
    def test_uses_model_name_when_provided(self):
        sid = mf.build_session_id("meta-llama/Llama-3.1-8B")
        assert "meta-llama_Llama-3.1-8B_" in sid
        # uuid suffix length is 8 hex chars.
        assert len(sid.rsplit("_", 1)[-1]) == 8

    def test_defaults_to_session_when_blank(self):
        sid = mf.build_session_id("")
        assert sid.startswith("session_")


# ---------------------------------------------------------------------------
# _describe_dep + _build_dependencies
# ---------------------------------------------------------------------------

class TestDescribeDep:
    def test_unset_env(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mf._describe_dep("MAGPIE_DIR") == {
            "path": "", "commit": "", "remote": "",
        }

    def test_missing_dir_yields_path_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAGPIE_DIR", str(tmp_path / "ghost"))
        out = mf._describe_dep("MAGPIE_DIR")
        assert out["path"] == str(tmp_path / "ghost")
        assert out["commit"] == "" and out["remote"] == ""

    def test_directory_present_calls_git_helpers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAGPIE_DIR", str(tmp_path))
        monkeypatch.setattr(mf, "_git_revision_at", lambda p: "abc1234")
        monkeypatch.setattr(mf, "_git_remote_at", lambda p: "https://x/y.git")
        out = mf._describe_dep("MAGPIE_DIR")
        assert out["commit"] == "abc1234"
        assert out["remote"] == "https://x/y.git"

    def test_first_env_var_wins(self, tmp_path, monkeypatch):
        """MAGPIE_PATH (preferred) wins over the legacy MAGPIE_DIR fallback."""
        monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "preferred"))
        monkeypatch.setenv("MAGPIE_DIR", str(tmp_path / "legacy"))
        out = mf._describe_dep("MAGPIE_PATH", "MAGPIE_DIR")
        assert out["path"] == str(tmp_path / "preferred")

    def test_falls_back_to_legacy_env_var(self, tmp_path, monkeypatch):
        """Only the legacy MAGPIE_DIR is set → fallback is honoured
        (backward compatibility for the MAGPIE_DIR → MAGPIE_PATH rename)."""
        monkeypatch.delenv("MAGPIE_PATH", raising=False)
        monkeypatch.setenv("MAGPIE_DIR", str(tmp_path / "legacy"))
        out = mf._describe_dep("MAGPIE_PATH", "MAGPIE_DIR")
        assert out["path"] == str(tmp_path / "legacy")


# ---------------------------------------------------------------------------
# _detect_image
# ---------------------------------------------------------------------------

class TestDetectImage:
    def test_returns_env_when_set(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_IMAGE", "registry/x:tag")
        assert mf._detect_image() == "registry/x:tag"

    def test_falls_back_to_marker_file(self, tmp_path, monkeypatch):
        for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
            monkeypatch.delenv(var, raising=False)

        marker = tmp_path / "marker_image"
        marker.write_text("custom/image:1\n")

        original_init = Path.__init__

        # Substitute the well-known marker paths used by _detect_image.
        original_exists = Path.exists

        def fake_exists(self):
            if str(self) in ("/etc/podinfo/image", "/etc/hyperloom-image"):
                return True
            return original_exists(self)

        def fake_read_text(self, *args, **kwargs):
            if str(self) == "/etc/podinfo/image":
                return "custom/image:1"
            return Path.read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "read_text", fake_read_text)
        # Image detection should pick up the patched marker.
        assert mf._detect_image() == "custom/image:1"

    def test_returns_none_when_no_signal(self, monkeypatch):
        for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert mf._detect_image() is None


# ---------------------------------------------------------------------------
# build_manifest end-to-end
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_default_no_args(self, tmp_path, monkeypatch):
        for var in ("FRAMEWORK", "GPU_TYPE", "ISL", "OSL", "MAX_MODEL_LEN",
                    "PRECISION", "CONC", "TP", "CLAW_SESSION_ID",
                    "SANDBOX_USER_ID"):
            monkeypatch.delenv(var, raising=False)
        out = mf.build_manifest(tmp_path)
        assert out["schema_version"] == mf.SCHEMA_VERSION
        assert out["framework"] == "sglang"
        assert out["max_minutes"] == 0
        assert out["session_dir"] == str(tmp_path)
        assert out["pid"] == os.getpid()

    def test_args_override_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "vllm")
        ns = SimpleNamespace(
            model="/weights/m",
            framework="sglang",
            gpu_type="MI300X",
            isl=128,
            osl=64,
            precision="fp16",
            target_gain=5.0,
            target_tput=None,
            target_baseline_dir=None,
            max_hours=2,
        )
        out = mf.build_manifest(tmp_path, args=ns)
        assert out["framework"] == "sglang"  # arg wins over env
        assert out["model_path"] == "/weights/m"
        assert out["model_name"] == "m"
        assert out["workload"]["isl"] == 128
        assert out["workload"]["osl"] == 64
        assert out["workload"]["precision"] == "fp16"
        assert out["objective"] == {"kind": "gain_pct", "value": 5.0}
        assert out["max_minutes"] == 120


class TestWriteAndLoad:
    def test_round_trip(self, tmp_path, monkeypatch):
        for var in ("FRAMEWORK", "GPU_TYPE", "ISL", "OSL", "MAX_MODEL_LEN",
                    "PRECISION", "CONC", "TP"):
            monkeypatch.delenv(var, raising=False)
        manifest = mf.write_manifest(tmp_path)
        assert (tmp_path / "manifest.json").is_file()
        loaded = mf.load_manifest(tmp_path)
        assert loaded["session_id"] == manifest["session_id"]

    def test_load_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mf.load_manifest(tmp_path)
