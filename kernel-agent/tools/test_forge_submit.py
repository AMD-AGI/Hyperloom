# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the forge backend (parse_backends gate, gpu_target, fellow
map, report anchors roundtrip, and skip paths). No GPU / gateway required.

Design ref: claw-dev/docs-zh/forge-as-hyperloom-backend-integration.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import forge_submit  # noqa: E402
import kernel_optimization as ko  # noqa: E402


def test_parse_backends_accepts_forge():
    assert ko.parse_backends("forge") == ["forge"]
    assert ko.parse_backends("geak,forge") == ["geak", "forge"]


def test_parse_backends_still_rejects_unknown():
    with pytest.raises(ValueError):
        ko.parse_backends("forge,bogus")


def test_resolve_gpu_target_env_wins(monkeypatch):
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    assert forge_submit._resolve_gpu_target({"platform": "mi300x"}) == "gfx950"


def test_resolve_gpu_target_platform_map(monkeypatch):
    monkeypatch.delenv("GPU_TARGET", raising=False)
    assert forge_submit._resolve_gpu_target({"platform": "MI300X"}) == "gfx942"
    assert forge_submit._resolve_gpu_target({"platform": "mi355x"}) == "gfx950"


def test_fellow_for_source_type():
    assert forge_submit._fellow_for_source_type("triton") == "triton-fellow"
    assert forge_submit._fellow_for_source_type("python") == "triton-fellow"
    assert forge_submit._fellow_for_source_type("hip_cpp") is None
    assert forge_submit._fellow_for_source_type("unknown") is None


def test_report_anchors_roundtrip(tmp_path):
    """_write_report output must be parseable by kernel_optimization's extractors."""
    report = forge_submit._write_report(tmp_path, 0.183, 0.106, improved=True)
    speedup = ko._extract_speedup_from_report(report)
    assert speedup is not None and abs(speedup - (0.183 / 0.106)) < 1e-3
    assert ko._extract_correctness_from_report(report) is True


def test_report_never_fabricates_when_no_best(tmp_path):
    report = forge_submit._write_report(tmp_path, None, None, improved=False)
    assert ko._extract_speedup_from_report(report) is None
    assert ko._extract_correctness_from_report(report) is False


def test_shapes_from_candidate_named_passthrough():
    """Back-compat: an explicit pre-named dim dict is honored."""
    shapes = forge_submit._shapes_from_candidate(
        {"operation": "gemm", "input_shapes": [{"M": 2048, "N": 2048, "K": 2048}]})
    assert shapes["primary"] == {"M": 2048, "N": 2048, "K": 2048}
    assert shapes["minimal"] == shapes["primary"]
    assert shapes["validation"] == [shapes["primary"]]


def test_shapes_from_candidate_gemm_real_format():
    """Real TraceLens format: per-tensor {call_num, shape}; derive M/N/K from A@B."""
    cand = {"operation": "gemm", "input_shapes": [
        {"call_num": 1, "shape": [4096, 8192]},   # A: [M, K]
        {"call_num": 2, "shape": [8192, 1024]},   # B: [K, N]
    ]}
    assert forge_submit._shapes_from_candidate(cand)["primary"] == {"M": 4096, "K": 8192, "N": 1024}


def test_shapes_from_candidate_moe_real_format():
    """fused_moe: derive M/N/K/E/TOPK from hidden/w1/w2/topk tensor shapes."""
    cand = {"operation": "fused_moe", "input_shapes": [
        {"call_num": 1, "shape": [16384, 2048]},      # hidden_states [M, K]
        {"call_num": 2, "shape": [128, 3072, 2048]},  # w1 [E, 2N, K]
        {"call_num": 3, "shape": [128, 2048, 1536]},  # w2 [E, K, N]
        {"call_num": 4, "shape": [16384, 8]},         # topk ids [M, TOPK]
    ]}
    p = forge_submit._shapes_from_candidate(cand)["primary"]
    assert p == {"M": 16384, "K": 2048, "TOPK": 8, "E": 128, "N": 1536}


def test_shapes_from_candidate_unparseable_falls_back():
    """Non-derivable shapes -> empty primary so the driver keeps its defaults."""
    cand = {"operation": "fused_moe", "input_shapes": [{"call_num": 1, "shape": []}]}
    assert forge_submit._shapes_from_candidate(cand)["primary"] == {}


def test_submit_skips_non_triton(tmp_path):
    """Stage 1 supports triton only; other source_types return a clean skip, no GPU work."""
    res = forge_submit.submit(source_file=str(tmp_path / "k.cpp"), prompt_file=tmp_path / "p.txt",
                              output_dir=tmp_path / "out", test_command="echo hi",
                              source_type="hip_cpp", candidate={})
    assert res["returncode"] == 2
    assert "triton only" in res["stderr_tail"]


def test_autogen_driver_selection():
    """No harness: gemm/matmul/moe ops get an auto-generated driver; others skip cleanly."""
    import tempfile
    o = Path(tempfile.mkdtemp())
    assert forge_submit._autogen_forge_driver(
        {"operation": "gemm", "input_shapes": [{"M": 8, "N": 8, "K": 8}]},
        "/tmp/gemm.py", o) is not None
    assert (o / "forge_autogen_driver.py").exists()
    # fused_moe imports sglang by package, so it requires in-place mode.
    assert forge_submit._autogen_forge_driver(
        {"operation": "fused_moe", "name": "fused_moe_triton"}, "/x/y.py", o, inplace=True) is not None
    # Without in-place mode the moe template would no-op -> skip cleanly.
    assert forge_submit._autogen_forge_driver(
        {"operation": "fused_moe", "name": "fused_moe_triton"}, "/x/y.py", o, inplace=False) is None
    # Ops without a template still skip cleanly.
    assert forge_submit._autogen_forge_driver(
        {"operation": "softmax", "name": "softmax_kernel"}, "/x/y.py", o) is None


def test_autogen_templates_compile():
    """Generated driver templates must be valid Python (CI can't import sglang/torch,
    but a syntax-level compile catches template breakage before a live run)."""
    import tempfile
    moe = forge_submit._autogen_forge_driver(
        {"operation": "fused_moe"}, "/x/y.py", Path(tempfile.mkdtemp()), inplace=True)
    gemm = forge_submit._autogen_forge_driver(
        {"operation": "gemm", "input_shapes": [{"M": 8}]}, "/tmp/k.py", Path(tempfile.mkdtemp()))
    for p in (moe, gemm):
        src = Path(p).read_text()
        compile(src, p, "exec")


def test_submit_skips_without_harness_or_template(tmp_path):
    """Empty test_command + non-git/unknown-op skips cleanly (rc=2), never crashes."""
    res = forge_submit.submit(source_file=str(tmp_path / "k.py"), prompt_file=tmp_path / "p.txt",
                              output_dir=tmp_path / "out", test_command="",
                              source_type="triton", candidate={})
    assert res["returncode"] == 2
