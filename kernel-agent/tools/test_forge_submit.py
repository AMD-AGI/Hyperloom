# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the forge backend (parse_backends gate, gpu_target, fellow
map, report anchors roundtrip, and skip paths). No GPU / gateway required.

Design ref: claw-dev/docs-zh/forge-as-hyperloom-backend-integration.md
"""

from __future__ import annotations

import os
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


def test_fellow_compiled_gated_by_env(monkeypatch):
    # Compiled fellows stay off by default (clean skip -> geak fallback).
    monkeypatch.delenv("FORGE_ENABLE_COMPILED_FELLOWS", raising=False)
    assert forge_submit._fellow_for_source_type("hip_cpp") is None
    assert forge_submit._fellow_for_source_type("ck") is None
    # Opt-in enables Kernel-Forge's native compiled fellows.
    monkeypatch.setenv("FORGE_ENABLE_COMPILED_FELLOWS", "1")
    assert forge_submit._fellow_for_source_type("hip_cpp") == "hip-fellow"
    assert forge_submit._fellow_for_source_type("ck") == "ck-fellow"
    assert forge_submit._fellow_for_source_type("aiter") == "aiter-fellow"
    assert forge_submit._fellow_for_source_type("hipblaslt") == "hipblaslt-fellow"
    # Still None for genuinely unsupported types.
    assert forge_submit._fellow_for_source_type("vendor_binary") is None


def _backends_args(backends=""):
    import argparse
    return argparse.Namespace(backends=backends, benchmark_file="", test_harness_path="")


def test_choose_backends_appends_geak_fallback_for_forge_only(monkeypatch):
    # RCA root cause A: forge-only must not run without a geak safety net.
    monkeypatch.delenv("FORGE_DISABLE_GEAK_FALLBACK", raising=False)
    selected, notes = ko.choose_backends(_backends_args("forge"), {})
    assert selected == ["forge", "geak"]
    assert notes.get("geak_fallback_appended") is True


def test_choose_backends_geak_fallback_opt_out(monkeypatch):
    monkeypatch.setenv("FORGE_DISABLE_GEAK_FALLBACK", "1")
    selected, _ = ko.choose_backends(_backends_args("forge"), {})
    assert selected == ["forge"]


def test_choose_backends_no_double_geak(monkeypatch):
    monkeypatch.delenv("FORGE_DISABLE_GEAK_FALLBACK", raising=False)
    selected, _ = ko.choose_backends(_backends_args("forge,geak"), {})
    assert selected == ["forge", "geak"]


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


def test_adapter_bench_mode_rewrites_correctness_to_benchmark(tmp_path):
    """Bench mode must run the harness's --benchmark path (which emits timing),
    not reuse --correctness which prints no latency (RCA root cause 3)."""
    # Fake harness: prints latency only under --benchmark, nothing under
    # --correctness, so the adapter must rewrite the flag to measure anything.
    harness = tmp_path / "fake_harness.py"
    harness.write_text(
        "import sys\n"
        "if '--benchmark' in sys.argv:\n"
        "    print('GEAK_RESULT_LATENCY_MS=4.2000')\n"
        "elif '--correctness' in sys.argv:\n"
        "    print('correctness passed')\n"
    )
    test_command = f"{sys.executable} {harness} --correctness"
    driver = forge_submit._build_driver_adapter(test_command, str(tmp_path), tmp_path)
    import subprocess as _sp
    out = _sp.run([sys.executable, driver, "--bench-mode"],
                  capture_output=True, text=True)
    assert "wall_ms: 4.2000" in out.stdout, out.stdout + out.stderr


def test_adapter_bench_parses_aiter_us_per_iter(tmp_path):
    """B: aiter op_tests have no --benchmark flag (benchmark by default) and log
    'avg: N us/iter'. The adapter must run them verbatim and convert us->ms."""
    harness = tmp_path / "aiter" / "op_tests" / "test_activation.py"
    harness.parent.mkdir(parents=True)
    # Emulate aiter perftest output; error out if a --benchmark flag is appended
    # (aiter argparse would reject it).
    harness.write_text(
        "import sys\n"
        "if '--benchmark' in sys.argv:\n"
        "    sys.stderr.write('error: unrecognized arguments: --benchmark\\n'); sys.exit(2)\n"
        "print('avg: 2500.0 us/iter from cuda.Event')\n"
        "print('avg: 1800.0 us/iter from cuda.Event')\n"
    )
    test_command = f"{sys.executable} {harness}"
    driver = forge_submit._build_driver_adapter(test_command, str(tmp_path), tmp_path)
    import subprocess as _sp
    out = _sp.run([sys.executable, driver, "--bench-mode"], capture_output=True, text=True)
    # min(2500,1800)=1800 us -> 1.8 ms; and no --benchmark was appended.
    assert "wall_ms: 1.8" in out.stdout, out.stdout + out.stderr


def test_report_informational_timing_not_kept_does_not_trigger_keep(tmp_path):
    """When not kept, the report records observed timing for post-mortem but must
    NOT be parsed as a KEEP-worthy speedup (no false KEEP)."""
    report = forge_submit._write_report(tmp_path, 7.5778, 4.0310, improved=False)
    text = report.read_text()
    assert "ratio=" in text and "best_ms=4.0310" in text
    # Critical: extractors must still see no speedup + failed correctness.
    assert ko._extract_speedup_from_report(report) is None
    assert ko._extract_correctness_from_report(report) is False


def test_apply_fellow_env_claude_path_and_timeout(tmp_path, monkeypatch):
    """G3+G4: child env gets FORGE_CLAUDE_BIN + claude dir on PATH, plus the
    fellow-hung mitigations (API_TIMEOUT_MS / disable flags)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    claude = bindir / "claude"
    claude.write_text("#!/bin/sh\n")
    claude.chmod(0o755)
    # FORGE_CLAUDE_BIN is read from the child env dict the function mutates.
    env = {"PATH": "/usr/bin", "FORGE_CLAUDE_BIN": str(claude)}
    forge_submit._apply_fellow_env(env)
    assert env["FORGE_CLAUDE_BIN"] == str(claude)
    assert str(bindir) in env["PATH"].split(os.pathsep)
    assert env["API_TIMEOUT_MS"] == "300000"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_apply_fellow_env_timeout_respects_operator_override(monkeypatch):
    """setdefault must not clobber an operator-set API_TIMEOUT_MS."""
    env = {"API_TIMEOUT_MS": "60000"}
    forge_submit._apply_fellow_env(env)
    assert env["API_TIMEOUT_MS"] == "60000"
