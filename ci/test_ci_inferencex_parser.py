# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/inferenceX_parser.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import inferenceX_parser as ix  # noqa: E402


# ── _unmangle_msys_path / get_nfs_root / resolve_var ──


def test_unmangle_msys_path():
    mangled = r"C:/Program Files/Git/wekafs/models/x"
    assert ix._unmangle_msys_path(mangled) == "/wekafs/models/x"


def test_unmangle_msys_path_noop():
    assert ix._unmangle_msys_path("/wekafs/models/x") == "/wekafs/models/x"


def test_get_nfs_root_default(monkeypatch):
    monkeypatch.delenv("NFS_ROOT", raising=False)
    assert ix.get_nfs_root() == "/wekafs"


def test_get_nfs_root_env(monkeypatch):
    monkeypatch.setenv("NFS_ROOT", "/custom")
    assert ix.get_nfs_root() == "/custom"


def test_resolve_var_basic():
    assert ix.resolve_var("a/${X}/b", {"X": "mid"}) == "a/mid/b"


def test_resolve_var_unset_keeps_placeholder():
    assert ix.resolve_var("${MISSING}", {}) == "${MISSING}"


def test_resolve_var_non_string():
    assert ix.resolve_var(123) == 123


# ── synthesize_entry_from_ci_config / parse_model_entry ──


def test_synthesize_and_parse_roundtrip():
    cfg = {
        "model_hf": "org/m",
        "image": "img:1",
        "framework": "vllm",
        "precision": "fp8",
        "conc": 128,
        "tp": 4,
        "ep": 2,
        "isl_osl_configs": [[1024, 2048]],
        "key": "abc-def",
    }
    entry = ix.synthesize_entry_from_ci_config(cfg)
    assert entry["model"] == "org/m"
    assert entry["model-prefix"] == "abc"
    parsed = ix.parse_model_entry(entry)
    assert parsed["model_hf"] == "org/m"
    assert parsed["tp"] == 4
    assert parsed["conc_end"] == 128
    assert parsed["isl_osl_configs"] == [(1024, 2048)]


def test_synthesize_defaults():
    entry = ix.synthesize_entry_from_ci_config({})
    assert entry["runner"] == "mi300x"
    assert entry["scenarios"]["fixed-seq-len"][0]["isl"] == 1024


def test_parse_model_entry_legacy_seq_len_configs():
    entry = {"model": "m", "seq-len-configs": [{"isl": 512, "osl": 512, "search-space": [{"tp": 2, "conc-end": 32}]}]}
    parsed = ix.parse_model_entry(entry)
    assert parsed["tp"] == 2
    assert parsed["isl_osl_configs"] == [(512, 512)]


def test_parse_model_entry_empty():
    parsed = ix.parse_model_entry({})
    assert parsed["tp"] == 8
    assert parsed["isl_osl_configs"] == []


# ── find_benchmark ──


def _bench(hw="b200", isl=1024, osl=1024, prec="fp8", out_tput=100, **kw):
    b = {"hardware": hw, "isl": isl, "osl": osl, "precision": prec, "metrics": {"output_tput_per_gpu": out_tput}}
    b.update(kw)
    return b


def test_find_benchmark_basic_best_tput():
    benches = [_bench(out_tput=100), _bench(out_tput=200)]
    assert ix.find_benchmark(benches, "b200", 1024, 1024)["metrics"]["output_tput_per_gpu"] == 200


def test_find_benchmark_no_match():
    assert ix.find_benchmark([_bench(hw="h100")], "b200", 1024, 1024) is None


def test_find_benchmark_precision_filter():
    benches = [_bench(prec="fp8"), _bench(prec="bf16", out_tput=999)]
    got = ix.find_benchmark(benches, "b200", 1024, 1024, precision="fp8")
    assert got["precision"] == "fp8"


def test_find_benchmark_image_tp_conc_preference():
    benches = [
        _bench(out_tput=300, image="other", decode_tp=2, conc=64),
        _bench(out_tput=10, image="want-img", decode_tp=8, conc=32),
    ]
    got = ix.find_benchmark(benches, "b200", 1024, 1024, image="want-img", tp=8, conc=32)
    assert got["image"] == "want-img"


def test_find_benchmark_tput_per_gpu_fallback():
    b = {"hardware": "b200", "isl": 1024, "osl": 1024, "precision": "fp8", "metrics": {"tput_per_gpu": 55}}
    assert ix.find_benchmark([b], "b200", 1024, 1024)["metrics"]["tput_per_gpu"] == 55


# ── format_benchmark_for_prompt ──


def test_format_benchmark_for_prompt_match():
    text = ix.format_benchmark_for_prompt([_bench()], "b200", 1024, 1024, "fp8")
    assert "Hardware: b200" in text
    assert "Output Throughput/GPU" in text


def test_format_benchmark_for_prompt_no_data():
    text = ix.format_benchmark_for_prompt([], "b200", 1024, 1024, "fp8")
    assert "No InferenceX data" in text


# ── find_benchmark_script ──


def test_find_benchmark_script_exact(tmp_path: Path):
    sdir = tmp_path / "benchmarks" / "single_node"
    sdir.mkdir(parents=True)
    (sdir / "minimaxm25_fp8_mi355x.sh").write_text("#!/bin/sh", encoding="utf-8")
    got = ix.find_benchmark_script(tmp_path, "minimaxm2.5-fp8-mi355x")
    assert got == "benchmarks/single_node/minimaxm25_fp8_mi355x.sh"


def test_find_benchmark_script_prefix(tmp_path: Path):
    sdir = tmp_path / "benchmarks" / "single_node"
    sdir.mkdir(parents=True)
    (sdir / "qwen3_fp8_b200.sh").write_text("x", encoding="utf-8")
    got = ix.find_benchmark_script(tmp_path, "qwen3-fp8-h100")
    assert got == "benchmarks/single_node/qwen3_fp8_b200.sh"


def test_find_benchmark_script_no_dir(tmp_path: Path):
    assert ix.find_benchmark_script(tmp_path, "x") is None


def test_find_benchmark_script_no_match(tmp_path: Path):
    sdir = tmp_path / "benchmarks" / "single_node"
    sdir.mkdir(parents=True)
    (sdir / "zzz.sh").write_text("x", encoding="utf-8")
    assert ix.find_benchmark_script(tmp_path, "qwen3-fp8-h100") is None


# ── merge_model_config ──


def test_merge_model_config(monkeypatch):
    monkeypatch.setenv("NFS_ROOT", "/nfs")
    ifx_entry = ix.synthesize_entry_from_ci_config(
        {
            "model_hf": "org/M",
            "image": "img:1",
            "framework": "sglang",
            "precision": "fp8",
            "conc": 64,
            "isl_osl_configs": [[1024, 1024]],
        }
    )
    cfg = {"inferenceX_key": "k", "tp": 4}
    merged = ix.merge_model_config(cfg, ifx_entry, {}, "harbor", [])
    assert merged["model_hf"] == "org/M"
    assert merged["model_path"] == "/nfs/models/org-M"
    assert merged["sandbox_image"] == "harbor/img:1"
    assert merged["tp"] == 4
    assert merged["claw_plugin_id"] == 4
    assert merged["inferencex_path"] == "/nfs/InferenceX"


def test_merge_model_config_no_harbor_and_plugin_override(monkeypatch):
    monkeypatch.setenv("NFS_ROOT", "/nfs")
    ifx_entry = ix.synthesize_entry_from_ci_config({"model_hf": "m", "image": "img"})
    merged = ix.merge_model_config({"claw_plugin_id": None}, ifx_entry, {}, "", [])
    assert merged["sandbox_image"] == "img"
    assert merged["claw_plugin_id"] is None


# ── network functions (mocked) ──


def test_get_latest_commit(monkeypatch):
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(stdout="abc123\trefs/heads/main\n")

    monkeypatch.setattr(ix.subprocess, "run", fake_run)
    assert ix.get_latest_commit("url") == "abc123"


def test_get_latest_commit_empty(monkeypatch):
    monkeypatch.setattr(ix.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="  \n"))
    assert ix.get_latest_commit("url") == ""


def test_fetch_amd_master_yaml(monkeypatch, tmp_path: Path):
    def fake_run(cmd, **kw):
        # cmd: ["git","clone","--depth=1","--branch=main", url, tmpdir]
        tmpdir = Path(cmd[-1])
        cfgp = tmpdir / ".github" / "configs" / "amd-master.yaml"
        cfgp.parent.mkdir(parents=True, exist_ok=True)
        cfgp.write_text("models:\n  - model: x\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ix.subprocess, "run", fake_run)
    cfg = ix.fetch_amd_master_yaml("url")
    assert cfg == {"models": [{"model": "x"}]}


def test_find_benchmark_script_from_clone(monkeypatch):
    def fake_run(cmd, **kw):
        tmpdir = Path(cmd[-1])
        sdir = tmpdir / "benchmarks" / "single_node"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "m_fp8.sh").write_text("x", encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ix.subprocess, "run", fake_run)
    got = ix.find_benchmark_script_from_clone("url", "m-fp8")
    assert got == "benchmarks/single_node/m_fp8.sh"


def test_fetch_benchmarks_ok(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"hardware": "b200"}]

    monkeypatch.setattr(ix.requests, "get", lambda *a, **k: FakeResp())
    assert ix.fetch_benchmarks("model") == [{"hardware": "b200"}]


def test_fetch_benchmarks_api_error(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "nope"}

    monkeypatch.setattr(ix.requests, "get", lambda *a, **k: FakeResp())
    assert ix.fetch_benchmarks("model") == []
