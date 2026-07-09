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

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import forge_submit  # noqa: E402
import kernel_optimization as ko  # noqa: E402


def test_parse_backends_accepts_forge():
    assert ko.parse_backends("forge") == ["forge"]
    assert ko.parse_backends("geak_v3,forge") == ["geak_v3", "forge"]


def test_parse_backends_still_rejects_unknown():
    with pytest.raises(ValueError):
        ko.parse_backends("forge,bogus")


def test_parse_backends_tolerates_stringified_list():
    # An upstream dispatch slip can hand the repr() of a Python
    # list to --backends ("['geak_v3']") instead of a bare name. parse_backends
    # must recover the inner token rather than rejecting a valid backend.
    assert ko.parse_backends("['geak_v3']") == ["geak_v3"]
    assert ko.parse_backends('["geak_v3"]') == ["geak_v3"]
    assert ko.parse_backends("['geak_v3', 'claude']") == ["geak_v3", "claude"]


def test_parse_backends_stringified_list_still_rejects_unknown():
    # The recovery must not weaken validation: a genuinely-unknown name inside
    # a stringified list is still rejected.
    with pytest.raises(ValueError):
        ko.parse_backends("['bogus']")


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
    assert forge_submit._fellow_for_source_type("unknown") is None


def test_fellow_compiled_enabled_by_default(monkeypatch):
    # Compiled fellows are ON by default.
    monkeypatch.delenv("FORGE_DISABLE_COMPILED_FELLOWS", raising=False)
    assert forge_submit._fellow_for_source_type("hip_cpp") == "hip-fellow"
    assert forge_submit._fellow_for_source_type("ck") == "ck-fellow"
    assert forge_submit._fellow_for_source_type("aiter") == "aiter-fellow"
    assert forge_submit._fellow_for_source_type("hipblaslt") == "hipblaslt-fellow"
    # Still None for genuinely unsupported types.
    assert forge_submit._fellow_for_source_type("vendor_binary") is None
    # Opt-out disables compiled fellows (revert to triton-only -> geak_v3 fallback).
    monkeypatch.setenv("FORGE_DISABLE_COMPILED_FELLOWS", "1")
    assert forge_submit._fellow_for_source_type("hip_cpp") is None
    assert forge_submit._fellow_for_source_type("ck") is None


def _backends_args(backends=""):
    import argparse
    return argparse.Namespace(backends=backends, benchmark_file="", test_harness_path="")


def test_choose_backends_respects_forge_only_order(monkeypatch):
    # KERNEL_OPT_BACKEND_ORDER / --backends is authoritative: forge means
    # strict forge-only, no hidden GEAK fallback.
    selected, notes = ko.choose_backends(_backends_args("forge"), {})
    assert selected == ["forge"]
    assert "geak_fallback_appended" not in notes


def test_choose_backends_no_double_geak(monkeypatch):
    selected, _ = ko.choose_backends(_backends_args("forge,geak_v3"), {})
    assert selected == ["forge", "geak_v3"]


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


def test_submit_rederives_aiter_cu_source_type(tmp_path, monkeypatch):
    """An aiter .cu kernel arriving with source_type='unknown' is re-derived to
    hip_cpp so forge maps it to hip-fellow (compiled fellows enabled by default)."""
    monkeypatch.delenv("FORGE_DISABLE_COMPILED_FELLOWS", raising=False)
    # unknown + .cu -> hip_cpp -> hip-fellow (not the triton-only skip).
    res = forge_submit.submit(
        source_file="/sgl-workspace/aiter/csrc/py_itfs_ck/mha_batch_prefill_kernels.cu",
        prompt_file=tmp_path / "p.txt", output_dir=tmp_path / "out",
        test_command="", source_type="unknown", candidate={"operation": "attention"},
        kernel_repo="")
    # It must NOT be the "supports triton only" skip (rc=2 with that message).
    assert "supports triton only" not in (res.get("stderr_tail") or "")


def test_submit_skips_untracked_source(tmp_path):
    """Untracked source files return a clean skip before any live-tree work."""
    res = forge_submit.submit(
        source_file=str(tmp_path / "k.cpp"),
        prompt_file=tmp_path / "p.txt",
        output_dir=tmp_path / "out",
        test_command="echo hi",
        source_type="hip_cpp",
        candidate={},
    )
    assert res["returncode"] == 2
    assert res["skipped"] is True
    assert "kernel_repo is not a clean git checkout" in res["stderr_tail"]


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
    assert res["skipped"] is True


def test_normalized_skipped_flag():
    """``_normalized`` carries the structured self-skip marker; default is False."""
    skip = forge_submit._normalized(2, "", "forge skipped: ...", 0.1, skipped=True)
    assert skip["skipped"] is True and skip["returncode"] == 2
    ran = forge_submit._normalized(0, "forge done", "", 0.1)
    assert ran["skipped"] is False


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


def test_apply_fellow_env_claude_path_and_stability_flags(tmp_path, monkeypatch):
    """G3+G4: child env gets FORGE_CLAUDE_BIN + claude dir on PATH, plus the
    low-risk fellow stability flags. API_TIMEOUT_MS is opt-in because external
    clients may treat it as a total request timeout."""
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
    assert "API_TIMEOUT_MS" not in env
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_apply_fellow_env_timeout_respects_operator_override(monkeypatch):
    """Existing operator-set API_TIMEOUT_MS must pass through untouched."""
    env = {"API_TIMEOUT_MS": "60000"}
    forge_submit._apply_fellow_env(env)
    assert env["API_TIMEOUT_MS"] == "60000"


def test_llm_stability_env_helper_sets_defaults():
    """The shared kernel-side helper sets only low-risk defaults."""
    import _llm_stability_env

    env: dict[str, str] = {}
    _llm_stability_env.apply_llm_stability_env(env)
    assert "API_TIMEOUT_MS" not in env
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_llm_stability_env_helper_respects_override():
    import _llm_stability_env

    env = {"API_TIMEOUT_MS": "1000"}
    _llm_stability_env.apply_llm_stability_env(env)
    assert env["API_TIMEOUT_MS"] == "1000"


def test_llm_stability_env_helper_can_opt_in_to_api_timeout():
    import _llm_stability_env

    env: dict[str, str] = {}
    _llm_stability_env.apply_llm_stability_env(env, api_timeout_ms="120000")
    assert env["API_TIMEOUT_MS"] == "120000"


def _capture_cli_env(tmp_path, monkeypatch, worktree_kernel):
    """Run _run_loop_via_cli with subprocess mocked, return the child env."""
    import subprocess as _sp
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})

        class P:
            returncode = 0
            stdout = '__FORGE_RESULT__{"baseline_ms":1.0,"best_ms":1.0,"improved":false}__FORGE_RESULT__'
            stderr = ""
        return P()

    monkeypatch.setattr(_sp, "run", fake_run)
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    exp = tmp_path / "forge_experiments"
    exp.mkdir(parents=True)
    forge_submit._run_loop_via_cli(
        worktree_kernel=worktree_kernel, driver=str(tmp_path / "d.py"),
        workspace=str(tmp_path), shapes={}, snr_threshold=30.0, max_iters=1,
        max_hours=0.1, branch="forge/t/k", gpu_target="gfx942",
        fellow="hip-fellow", program_md_file="", experiments_dir=exp,
        forge_log=tmp_path / "forge_loop.log", timeout_s=60)
    return captured["env"]


def test_cli_sets_aiter_rebuild_for_aiter_kernel(tmp_path, monkeypatch):
    """C: an aiter kernel forces AITER_REBUILD=1 so edits recompile."""
    env = _capture_cli_env(tmp_path, monkeypatch, "/sgl-workspace/aiter/csrc/x.cuh")
    assert env.get("AITER_REBUILD") == "1"


def test_cli_no_aiter_rebuild_for_non_aiter_kernel(tmp_path, monkeypatch):
    """A non-aiter (e.g. triton) kernel must not set AITER_REBUILD."""
    env = _capture_cli_env(tmp_path, monkeypatch, "/sgl-workspace/sglang/python/x.py")
    assert "AITER_REBUILD" not in env


def test_flydsl_compat_adds_alias_when_missing(tmp_path):
    """flydsl shim appends fly_values alias when only extract_to_ir_values exists."""
    proto = tmp_path / "protocol.py"
    proto.write_text("def extract_to_ir_values(obj):\n    return []\n")
    assert forge_submit._ensure_flydsl_aiter_compat(str(proto)) is True
    text = proto.read_text()
    assert "fly_values = extract_to_ir_values" in text
    # Importable and the alias actually binds.
    ns: dict = {}
    exec(compile(text, str(proto), "exec"), ns)
    assert ns["fly_values"] is ns["extract_to_ir_values"]


def test_flydsl_compat_idempotent(tmp_path):
    """Second call must not append a duplicate alias."""
    proto = tmp_path / "protocol.py"
    proto.write_text("def extract_to_ir_values(obj):\n    return []\n")
    forge_submit._ensure_flydsl_aiter_compat(str(proto))
    once = proto.read_text()
    forge_submit._ensure_flydsl_aiter_compat(str(proto))
    assert proto.read_text() == once


def test_flydsl_compat_untouched_when_no_extract(tmp_path):
    """Unexpected flydsl layout (no extract_to_ir_values) is left untouched."""
    proto = tmp_path / "protocol.py"
    proto.write_text("def something_else():\n    return 1\n")
    assert forge_submit._ensure_flydsl_aiter_compat(str(proto)) is False
    assert "fly_values" not in proto.read_text()


# ---- forge_cli_result.json sidecar -> FORGE_LLM_USAGE / FORGE_STEPS ----


def _write_forge_sidecar(tmp_path, payload):
    import json
    (tmp_path / "forge_cli_result.json").write_text(json.dumps(payload))
    return tmp_path


def test_sidecar_usage_surfaced_without_calls(tmp_path):
    """The parser only needs token counters; ``calls`` is optional metadata.

    A sidecar that reports aggregate token counters with NO ``calls`` field
    must still surface so ``submit`` emits FORGE_LLM_USAGE — the previous gate
    (``usage.get("calls")``) silently dropped it, losing the forge token row.
    """
    out_dir = _write_forge_sidecar(tmp_path, {
        "baseline_ms": 10, "best_ms": 8, "improved": True,
        "llm_usage": {"input_tokens": 120, "output_tokens": 30},
    })
    usage, steps = forge_submit._forge_trace_from_sidecar(out_dir)
    assert usage == {"input_tokens": 120, "output_tokens": 30}
    assert steps is None


def test_sidecar_usage_surfaced_with_zero_calls(tmp_path):
    """``calls == 0`` but real token counts present -> still surfaced."""
    out_dir = _write_forge_sidecar(tmp_path, {
        "llm_usage": {"input_tokens": 5, "calls": 0},
    })
    usage, _ = forge_submit._forge_trace_from_sidecar(out_dir)
    assert usage["input_tokens"] == 5


def test_sidecar_usage_dropped_without_token_counters(tmp_path):
    """Only metadata, no int-coercible counter -> nothing meaningful to emit."""
    out_dir = _write_forge_sidecar(tmp_path, {"llm_usage": {"calls": 3}})
    usage, _ = forge_submit._forge_trace_from_sidecar(out_dir)
    assert usage is None


def test_sidecar_missing_or_non_dict(tmp_path):
    """No sidecar -> (None, None); a non-dict sidecar -> (None, None)."""
    assert forge_submit._forge_trace_from_sidecar(tmp_path) == (None, None)
    (tmp_path / "forge_cli_result.json").write_text("[1, 2, 3]")
    assert forge_submit._forge_trace_from_sidecar(tmp_path) == (None, None)


def test_sidecar_steps_surfaced(tmp_path):
    out_dir = _write_forge_sidecar(tmp_path, {
        "steps": {"steps": [{"iteration": 1, "decision": "KEEP"}],
                  "summary": {"iterations": 1}},
    })
    _, steps = forge_submit._forge_trace_from_sidecar(out_dir)
    assert steps["summary"]["iterations"] == 1


def test_sidecar_empty_steps_list_is_none(tmp_path):
    """An empty ``steps`` list carries no timeline -> not surfaced."""
    out_dir = _write_forge_sidecar(tmp_path, {"steps": {"steps": []}})
    _, steps = forge_submit._forge_trace_from_sidecar(out_dir)
    assert steps is None


def test_sidecar_usage_roundtrips_through_parser(tmp_path):
    """Full contract: sidecar -> _forge_trace_from_sidecar -> FORGE_LLM_USAGE
    marker (as ``submit`` emits it) -> ``parse_forge_usage`` recovers counters."""
    import json

    from hyperloom.orchestrator.trace.parse_usage import parse_forge_usage

    out_dir = _write_forge_sidecar(tmp_path, {
        "llm_usage": {"input_tokens": 7, "output_tokens": 11},
    })
    usage, _ = forge_submit._forge_trace_from_sidecar(out_dir)
    marker = "FORGE_LLM_USAGE " + json.dumps(usage, sort_keys=True)
    recovered = parse_forge_usage(marker)
    assert recovered["input_tokens"] == 7
    assert recovered["output_tokens"] == 11
