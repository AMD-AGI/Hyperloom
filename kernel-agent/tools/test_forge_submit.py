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
        {"operation": "gemm", "input_shapes": [{"M": 2048, "N": 2048, "K": 2048}]}
    )
    assert shapes["primary"] == {"M": 2048, "N": 2048, "K": 2048}
    assert shapes["minimal"] == shapes["primary"]
    assert shapes["validation"] == [shapes["primary"]]


def test_shapes_from_candidate_gemm_real_format():
    """Real TraceLens format: per-tensor {call_num, shape}; derive M/N/K from A@B."""
    cand = {
        "operation": "gemm",
        "input_shapes": [
            {"call_num": 1, "shape": [4096, 8192]},  # A: [M, K]
            {"call_num": 2, "shape": [8192, 1024]},  # B: [K, N]
        ],
    }
    assert forge_submit._shapes_from_candidate(cand)["primary"] == {"M": 4096, "K": 8192, "N": 1024}


def test_shapes_from_candidate_moe_real_format():
    """fused_moe: derive M/N/K/E/TOPK from hidden/w1/w2/topk tensor shapes."""
    cand = {
        "operation": "fused_moe",
        "input_shapes": [
            {"call_num": 1, "shape": [16384, 2048]},  # hidden_states [M, K]
            {"call_num": 2, "shape": [128, 3072, 2048]},  # w1 [E, 2N, K]
            {"call_num": 3, "shape": [128, 2048, 1536]},  # w2 [E, K, N]
            {"call_num": 4, "shape": [16384, 8]},  # topk ids [M, TOPK]
        ],
    }
    p = forge_submit._shapes_from_candidate(cand)["primary"]
    assert p == {"M": 16384, "K": 2048, "TOPK": 8, "E": 128, "N": 1536}


def test_shapes_from_candidate_unparseable_falls_back():
    """Non-derivable shapes -> empty primary so the driver keeps its defaults."""
    cand = {"operation": "fused_moe", "input_shapes": [{"call_num": 1, "shape": []}]}
    assert forge_submit._shapes_from_candidate(cand)["primary"] == {}


def test_submit_skips_non_triton(tmp_path):
    """Stage 1 supports triton only; other source_types return a clean skip, no GPU work."""
    res = forge_submit.submit(
        source_file=str(tmp_path / "k.cpp"),
        prompt_file=tmp_path / "p.txt",
        output_dir=tmp_path / "out",
        test_command="echo hi",
        source_type="hip_cpp",
        candidate={},
    )
    assert res["returncode"] == 2
    assert "triton only" in res["stderr_tail"]


def test_autogen_driver_selection():
    """No harness: gemm/matmul/moe ops get an auto-generated driver; others skip cleanly."""
    import tempfile

    o = Path(tempfile.mkdtemp())
    assert (
        forge_submit._autogen_forge_driver(
            {"operation": "gemm", "input_shapes": [{"M": 8, "N": 8, "K": 8}]}, "/tmp/gemm.py", o
        )
        is not None
    )
    assert (o / "forge_autogen_driver.py").exists()
    # fused_moe imports sglang by package, so it requires in-place mode.
    assert (
        forge_submit._autogen_forge_driver(
            {"operation": "fused_moe", "name": "fused_moe_triton"}, "/x/y.py", o, inplace=True
        )
        is not None
    )
    # Without in-place mode the moe template would no-op -> skip cleanly.
    assert (
        forge_submit._autogen_forge_driver(
            {"operation": "fused_moe", "name": "fused_moe_triton"}, "/x/y.py", o, inplace=False
        )
        is None
    )
    # Ops without a template still skip cleanly.
    assert forge_submit._autogen_forge_driver({"operation": "softmax", "name": "softmax_kernel"}, "/x/y.py", o) is None


def test_autogen_templates_compile():
    """Generated driver templates must be valid Python (CI can't import sglang/torch,
    but a syntax-level compile catches template breakage before a live run)."""
    import tempfile

    moe = forge_submit._autogen_forge_driver(
        {"operation": "fused_moe"}, "/x/y.py", Path(tempfile.mkdtemp()), inplace=True
    )
    gemm = forge_submit._autogen_forge_driver(
        {"operation": "gemm", "input_shapes": [{"M": 8}]}, "/tmp/k.py", Path(tempfile.mkdtemp())
    )
    for p in (moe, gemm):
        src = Path(p).read_text()
        compile(src, p, "exec")


def test_submit_skips_without_harness_or_template(tmp_path):
    """Empty test_command + non-git/unknown-op skips cleanly (rc=2), never crashes."""
    res = forge_submit.submit(
        source_file=str(tmp_path / "k.py"),
        prompt_file=tmp_path / "p.txt",
        output_dir=tmp_path / "out",
        test_command="",
        source_type="triton",
        candidate={},
    )
    assert res["returncode"] == 2


def test_run_loop_via_cli_parses_result(tmp_path, monkeypatch):
    """CLI mode parses the subprocess JSON result (sidecar + sentinel) and never
    runs the loop in-process."""
    import subprocess as _sp

    exp_dir = tmp_path / "forge_experiments"
    exp_dir.mkdir(parents=True)
    sidecar = tmp_path / "forge_cli_result.json"

    def fake_run(cmd, **kwargs):
        # Emulate the forge-loop CLI: write the result sidecar + sentinel stdout.
        sidecar.write_text('{"baseline_ms": 0.20, "best_ms": 0.18, '
                           '"improved": true, "experiment_id": "abc123"}')

        class P:
            returncode = 0
            stdout = "loop log...\n__FORGE_RESULT__{\"baseline_ms\":0.20}__FORGE_RESULT__\n"
            stderr = ""
        return P()

    monkeypatch.setattr(_sp, "run", fake_run)
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")

    baseline, best, improved, out, exc = forge_submit._run_loop_via_cli(
        worktree_kernel=str(tmp_path / "k.py"), driver=str(tmp_path / "d.py"),
        workspace=str(tmp_path), shapes={"primary": {"M": 8}}, snr_threshold=30.0,
        max_iters=2, max_hours=0.1, branch="forge/t/k", gpu_target="gfx942",
        fellow="triton-fellow", program_md_file=str(tmp_path / "nope.md"),
        experiments_dir=exp_dir, forge_log=tmp_path / "forge_loop.log", timeout_s=60)

    assert exc is None
    assert baseline == 0.20 and best == 0.18 and improved is True


def test_run_loop_via_cli_timeout_returns_exc(tmp_path, monkeypatch):
    """A subprocess timeout surfaces as loop_exc with no measurement (the caller
    then reports a forge failure) — proving the hard-kill path works."""
    import subprocess as _sp

    def fake_run(cmd, **kwargs):
        raise _sp.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(_sp, "run", fake_run)
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    exp_dir = tmp_path / "forge_experiments"
    exp_dir.mkdir(parents=True)

    baseline, best, improved, out, exc = forge_submit._run_loop_via_cli(
        worktree_kernel=str(tmp_path / "k.py"), driver=str(tmp_path / "d.py"),
        workspace=str(tmp_path), shapes={}, snr_threshold=30.0, max_iters=2,
        max_hours=0.1, branch="forge/t/k", gpu_target="gfx942",
        fellow="triton-fellow", program_md_file="", experiments_dir=exp_dir,
        forge_log=tmp_path / "forge_loop.log", timeout_s=1)

    assert exc is not None and baseline is None and improved is False


def test_apply_fellow_env_rewrites_gateway_to_proxy():
    """/llm-gateway is rewritten to the streaming /api/v1/llm-proxy endpoint."""
    env = {"ANTHROPIC_BASE_URL": "https://host/llm-gateway"}
    forge_submit._apply_fellow_env(env)
    assert env["ANTHROPIC_BASE_URL"] == "https://host/api/v1/llm-proxy"
    assert env["ANTHROPIC_SKIP_TLS_VERIFY"] == "true"
    assert env["NODE_TLS_REJECT_UNAUTHORIZED"] == "0"


def test_apply_fellow_env_keeps_existing_proxy_and_operator_overrides():
    """An already-proxy URL is left as-is; operator-set TLS values are kept."""
    env = {
        "ANTHROPIC_BASE_URL": "https://host/api/v1/llm-proxy",
        "ANTHROPIC_SKIP_TLS_VERIFY": "false",
    }
    forge_submit._apply_fellow_env(env)
    assert env["ANTHROPIC_BASE_URL"] == "https://host/api/v1/llm-proxy"
    # setdefault must not clobber an explicit operator value.
    assert env["ANTHROPIC_SKIP_TLS_VERIFY"] == "false"


def test_apply_fellow_env_does_not_mutate_os_environ(monkeypatch):
    """Finding-1 regression guard: the rewrite is scoped to the passed child env
    dict and never leaks into the parent os.environ (which sibling ladder
    backends claude/codex read)."""
    import os as _os
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://host/llm-gateway")
    env = dict(_os.environ)
    forge_submit._apply_fellow_env(env)
    # child env rewritten...
    assert env["ANTHROPIC_BASE_URL"] == "https://host/api/v1/llm-proxy"
    # ...but the process-global env is untouched (no leak to claude/codex).
    assert _os.environ["ANTHROPIC_BASE_URL"] == "https://host/llm-gateway"
