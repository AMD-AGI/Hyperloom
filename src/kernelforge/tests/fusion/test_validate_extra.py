# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover serving_smoke, runtime helpers, and harness error paths."""

from __future__ import annotations

import subprocess

from kernelforge.fusion.models import Recipe
from kernelforge.fusion.validate import (
    HarnessKernelRunner,
    _parse_harness_json,
    _runtime_dir,
    _serving_crash_reason,
    _serving_smoke_launch_cmd,
    _tail_text,
    _vllm_decode_probe,
    classify_serving_smoke_failure,
    serving_smoke,
    snr_db,
    validate_recipe,
)
from kernelforge.fusion import validate

import urllib.request as _urllib_rq


class _FakeResp:
    def __init__(self, payload):
        import json as _j

        self._b = _j.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _make_urlopen(models_payload=None, models_exc=None, completion_payload=None, completion_exc=None):
    def fake(arg, timeout=None):
        url = arg.full_url if hasattr(arg, "full_url") else str(arg)
        if url.endswith("/v1/models"):
            if models_exc:
                raise models_exc
            return _FakeResp(models_payload)
        if completion_exc:
            raise completion_exc
        return _FakeResp(completion_payload)

    return fake


def test_vllm_probe_models_http_error(monkeypatch):
    monkeypatch.setattr(_urllib_rq, "urlopen", _make_urlopen(models_exc=OSError("boom")))
    ok, detail = _vllm_decode_probe(8899, isl=64, osl=8, num_prompts=16, conc=4, timeout_s=5)
    assert ok is False and "/v1/models probe error" in detail


def test_vllm_probe_empty_model_id(monkeypatch):
    monkeypatch.setattr(_urllib_rq, "urlopen", _make_urlopen(models_payload={"data": [{}]}))
    ok, detail = _vllm_decode_probe(8899, isl=64, osl=8, num_prompts=16, conc=4, timeout_s=5)
    assert ok is False and "no served model id" in detail


def test_vllm_probe_empty_completion_text(monkeypatch):
    monkeypatch.setattr(
        _urllib_rq,
        "urlopen",
        _make_urlopen(models_payload={"data": [{"id": "m"}]}, completion_payload={"choices": [{"text": ""}]}),
    )
    ok, detail = _vllm_decode_probe(8899, isl=64, osl=8, num_prompts=16, conc=4, timeout_s=5)
    assert ok is False and "no output tokens" in detail


def test_vllm_probe_completion_http_error(monkeypatch):
    monkeypatch.setattr(
        _urllib_rq, "urlopen", _make_urlopen(models_payload={"data": [{"id": "m"}]}, completion_exc=OSError("net"))
    )
    ok, detail = _vllm_decode_probe(8899, isl=64, osl=8, num_prompts=16, conc=4, timeout_s=5)
    assert ok is False and "error" in detail


def test_vllm_probe_all_ok(monkeypatch):
    monkeypatch.setattr(
        _urllib_rq,
        "urlopen",
        _make_urlopen(models_payload={"data": [{"id": "m"}]}, completion_payload={"choices": [{"text": "hi"}]}),
    )
    ok, detail = _vllm_decode_probe(8899, isl=64, osl=8, num_prompts=16, conc=4, timeout_s=5)
    assert ok is True and "decode completions ok" in detail


def test_launch_cmd_matches_framework():
    """Repro: serving smoke must launch the framework's own server, not always sglang.

    A vLLM run previously always shelled out to ``sglang.launch_server`` and died
    with ``ModuleNotFoundError: sglang`` before the fusion was ever validated.
    """
    for fw in ("vllm", "vllm-aiter"):
        cmd = _serving_smoke_launch_cmd(fw, "/m", 8977, "")
        assert cmd[:2] == ["vllm", "serve"], f"{fw} must launch vllm serve, got {cmd[:2]}"
        assert not any("sglang" in str(c) for c in cmd), f"{fw} must not launch sglang: {cmd}"
    scmd = _serving_smoke_launch_cmd("sglang", "/m", 8977, "")
    assert any("sglang.launch_server" in str(c) for c in scmd)


def test_launch_cmd_matches_session_tp_block_size_and_max_model_len():
    """Serving smoke must boot with the session's TP / KV block size / max len.

    MiniMax MSA died on TP=1 + default block-size 16 ("No common block size for 16")
    while the real session served TP=8 and --block-size 128.
    """
    cmd = _serving_smoke_launch_cmd(
        "vllm",
        "/m",
        8977,
        "",
        tp=8,
        block_size=128,
        max_model_len=13312,
    )
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "8"
    assert cmd[cmd.index("--block-size") + 1] == "128"
    assert cmd[cmd.index("--max-model-len") + 1] == "13312"
    scmd = _serving_smoke_launch_cmd(
        "sglang",
        "/m",
        8977,
        "",
        tp=8,
        block_size=128,
        max_model_len=13312,
    )
    assert scmd[scmd.index("--tp") + 1] == "8"
    assert scmd[scmd.index("--context-length") + 1] == "13312"
    assert "--block-size" not in scmd


def test_classify_serving_smoke_failure_only_blames_explicit_gpu_faults():
    """The reason-only fallback needs fault EVIDENCE, not a keyword that resembles it."""
    assert (
        classify_serving_smoke_failure("server exited rc=1 before ready: ValueError: No common block size for 16")
        == "env_or_boot"
    )
    assert classify_serving_smoke_failure("server not ready within 1200s") == "env_or_boot"
    assert classify_serving_smoke_failure("serving smoke harness error: RuntimeError: popen exploded") == "env_or_boot"
    # Memory exhaustion reaches us through the same "HIP error:" channel as a fault.
    assert (
        classify_serving_smoke_failure("server exited rc=1 before ready: RuntimeError: HIP error: out of memory")
        == "env_or_boot"
    )
    # A live server that refused a request is not the kernel faulting.
    assert classify_serving_smoke_failure("decode probe failed: /v1/models probe error: OSError: boom") == "env_or_boot"
    assert (
        classify_serving_smoke_failure("decode bench failed rc=1: ModuleNotFoundError: sglang.bench_serving")
        == "env_or_boot"
    )
    assert (
        classify_serving_smoke_failure("scheduler crashed during CUDA-graph decode: HSA_STATUS_ERROR_EXCEPTION")
        == "kernel_fault"
    )
    assert classify_serving_smoke_failure("server crashed at startup: hardware exception") == "kernel_fault"
    assert classify_serving_smoke_failure("decode bench timed out (possible hang in fused kernel)") == "kernel_fault"


def test_serving_smoke_tp_exposes_enough_gpus(monkeypatch, tmp_path):
    _patch_smoke_common(monkeypatch, tmp_path)
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["env"] = k.get("env") or {}
        captured["cmd"] = cmd
        return _FakeServer([0])

    monkeypatch.setattr(validate.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(validate, "_tail_text", lambda *a, **k: "")
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    serving_smoke("/m", {"F": "1"}, framework="vllm", gpu="0", tp=8, timeout_s=1, log_path=str(tmp_path / "tp.log"))
    assert captured["env"].get("HIP_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7"
    assert captured["cmd"][captured["cmd"].index("--tensor-parallel-size") + 1] == "8"


def test_serving_smoke_vllm_uses_vllm_launcher_and_probe(monkeypatch, tmp_path):
    _patch_smoke_common(monkeypatch, tmp_path)
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakeServer([None, None])

    monkeypatch.setattr(validate.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(validate, "_tail_text", lambda *a, **k: "Application startup complete.\n")
    # vLLM path must NOT use sglang.bench_serving; it uses the HTTP decode probe.
    monkeypatch.setattr(validate, "_vllm_decode_probe", lambda *a, **k: (True, "3 decode completions ok"))
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    ok, reason = serving_smoke("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "v.log"))
    assert ok is True and "survives" in reason
    assert captured["cmd"][:2] == ["vllm", "serve"]
    assert not any("sglang" in str(c) for c in captured["cmd"])


def _recipe(**over) -> Recipe:
    base = dict(
        pattern_id="p",
        description="d",
        env_flag="F",
        source_file="/m.py",
        source_hints=["a"],
        fusion_math="y=x",
        eager_reference_hint="h",
        shapes={"T": 8},
        matched_categories=["c"],
        trigger_share=0.3,
    )
    base.update(over)
    return Recipe(**base)


class _Proc:
    def __init__(self, stdout="", stderr="", rc=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = rc


def test_snr_zero_signal_returns_zero():
    # noise>0 but signal==0 -> 0.0 branch (line 332)
    assert snr_db([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_runtime_dir_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    d = _runtime_dir("mykind")
    assert d.exists() and d.name == "mykind"


def test_tail_text_missing_and_present(tmp_path):
    assert _tail_text(str(tmp_path / "nope.log")) == ""
    p = tmp_path / "l.log"
    p.write_text("abcdefgh")
    assert _tail_text(str(p), n=3) == "fgh"


def test_serving_crash_reason_finds_marker():
    tail = "boot ok\nMemory access fault by GPU node-1\nother"
    assert "Memory access fault" in _serving_crash_reason(tail)


def test_serving_crash_reason_default_when_no_marker():
    assert "no explicit GPU-fault" in _serving_crash_reason("all fine\nstill fine")


# ── serving_smoke ────────────────────────────────────────────────────────────
class _FakeServer:
    def __init__(self, poll_seq):
        self._poll = iter(poll_seq)
        self.returncode = 1
        self.pid = 4242

    def poll(self):
        try:
            return next(self._poll)
        except StopIteration:
            return None


def _patch_smoke_common(monkeypatch, tmp_path):
    monkeypatch.setattr(validate, "_runtime_dir", lambda kind: tmp_path)
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(validate.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(validate.os, "getpgid", lambda pid: pid)


def test_serving_smoke_server_exits_before_ready(monkeypatch, tmp_path):
    # _tail_text returns a static string: serving_smoke truncates the log file.
    _patch_smoke_common(monkeypatch, tmp_path)
    slog = tmp_path / "s.log"
    monkeypatch.setattr(validate, "_tail_text", lambda *a, **k: "boot...\nCUDA error: illegal\n")
    monkeypatch.setattr(validate.subprocess, "Popen", lambda *a, **k: _FakeServer([0]))
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    ok, reason = serving_smoke("/m", {"F": "1"}, timeout_s=5, log_path=str(slog))
    assert ok is False and "before ready" in reason


def test_serving_smoke_ready_then_bench_ok(monkeypatch, tmp_path):
    _patch_smoke_common(monkeypatch, tmp_path)
    slog = tmp_path / "s2.log"

    def fake_run(cmd, *a, **k):
        if any("bench_serving" in str(c) for c in cmd):
            return _Proc(stdout="Output token throughput: 999\n", rc=0)
        return _Proc()

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    monkeypatch.setattr(validate, "_tail_text", lambda *a, **k: "The server is fired up and ready to roll!\n")
    monkeypatch.setattr(validate.subprocess, "Popen", lambda *a, **k: _FakeServer([None, None]))
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    ok, reason = serving_smoke("/m", {"F": "1"}, timeout_s=5, log_path=str(slog))
    assert ok is True and "survives" in reason


def test_serving_smoke_crash_during_decode(monkeypatch, tmp_path):
    _patch_smoke_common(monkeypatch, tmp_path)
    slog = tmp_path / "s3.log"
    ready = "Application startup complete.\n"
    crash = ready + "HSA_STATUS_ERROR_EXCEPTION hardware exception\n"
    state = {"tail": ready}

    def fake_run(cmd, *a, **k):
        if any("bench_serving" in str(c) for c in cmd):
            state["tail"] = crash
            return _Proc(stdout="", rc=1)
        return _Proc()

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    monkeypatch.setattr(validate, "_tail_text", lambda *a, **k: state["tail"])
    monkeypatch.setattr(validate.subprocess, "Popen", lambda *a, **k: _FakeServer([None, None, None]))
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    ok, reason = serving_smoke("/m", {"F": "1"}, timeout_s=5, log_path=str(slog))
    assert ok is False and "scheduler crashed" in reason


def test_serving_smoke_harness_error_is_soft_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(validate, "_runtime_dir", lambda kind: tmp_path)
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: _Proc())

    def boom(*a, **k):
        raise RuntimeError("popen exploded")

    monkeypatch.setattr(validate.subprocess, "Popen", boom)
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    ok, reason = serving_smoke("/m", {"F": "1"}, timeout_s=1, log_path=str(tmp_path / "x.log"))
    assert ok is False and "harness error" in reason


# ── HarnessKernelRunner error paths ──────────────────────────────────────────
def test_harness_timeout(tmp_path, monkeypatch):
    h = tmp_path / "h.py"
    h.write_text("print('{}')\n")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(validate.subprocess, "run", boom)
    runner = HarnessKernelRunner(str(h), workdir=str(tmp_path), timeout_s=1)
    comp = runner.compile_check(_recipe())
    assert comp.ok is False and "timed out" in comp.error


def test_harness_oserror(tmp_path, monkeypatch):
    h = tmp_path / "h.py"
    h.write_text("print('{}')\n")

    def boom(*a, **k):
        raise OSError("no python")

    monkeypatch.setattr(validate.subprocess, "run", boom)
    runner = HarnessKernelRunner(str(h), workdir=str(tmp_path))
    comp = runner.compile_check(_recipe())
    assert comp.ok is False and "could not run" in comp.error


def test_harness_cache_reused(tmp_path, monkeypatch):
    h = tmp_path / "h.py"
    h.write_text("x")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _Proc(stdout='{"compiled": true, "is_triton": false}')

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    runner = HarnessKernelRunner(str(h), workdir=str(tmp_path))
    runner.compile_check(_recipe())
    runner.parity_samples(_recipe())
    runner.microbench(_recipe())
    assert calls["n"] == 1  # cached across the three gate calls


def test_parse_harness_json_no_json_fallback():
    d = _parse_harness_json("no json here", "stderr blob", 3)
    assert d["compiled"] is False and "no JSON" in d["error"]


def test_parse_harness_json_skips_bad_line_uses_last_valid():
    stdout = '{"bad": }\n{"compiled": true, "is_triton": true}\n'
    d = _parse_harness_json(stdout, "", 0)
    assert d["compiled"] is True and d["is_triton"] is True


def test_validate_parity_all_none_fails():
    from kernelforge.fusion.validate import BenchOutcome, CompileOutcome, ParitySample

    class R:
        def compile_check(self, r):
            return CompileOutcome(ok=True)

        def parity_samples(self, r):
            return [ParitySample(snr_db=None, max_abs_err=None)]

        def microbench(self, r):
            return BenchOutcome()

    vr = validate_recipe(_recipe(), R())
    assert vr.correctness_passed is False and "PARITY FAILED" in vr.note


def test_validate_bench_no_timing_not_kept():
    from kernelforge.fusion.validate import BenchOutcome, CompileOutcome, ParitySample

    class R:
        def compile_check(self, r):
            return CompileOutcome(ok=True)

        def parity_samples(self, r):
            return [ParitySample(snr_db=50.0)]

        def microbench(self, r):
            return BenchOutcome(eager_us=None, fused_us=None)

    vr = validate_recipe(_recipe(), R())
    assert vr.correctness_passed is True and vr.kept is False
    assert "no timing" in vr.note


def test_apply_serving_gate_env_boot_failure_keeps_micro_keep(tmp_path, monkeypatch):
    """A MiniMax-style KV boot failure is not a fused-kernel loss."""
    from types import SimpleNamespace

    from kernelforge.fusion import command as cli_module
    from kernelforge.fusion.models import ValidationResult

    from kernelforge.fusion.validate import SMOKE_STAGE_STARTUP_CRASH, SmokeVerdict

    monkeypatch.setattr(cli_module, "_serving_check_enabled", lambda: True)
    captured = {}

    def fake_smoke(*a, **k):
        captured.update(k)
        return SmokeVerdict(
            ok=False,
            reason="server exited rc=1 before ready: ValueError: No common block size for 16",
            stage=SMOKE_STAGE_STARTUP_CRASH,
            blames_kernel=False,
        )

    monkeypatch.setattr(cli_module, "serving_smoke_verdict", fake_smoke)
    vr = ValidationResult(
        correctness_passed=True,
        max_abs_err=0.0,
        rtol=0.02,
        kernel_speedup=2.5,
        eager_us=100.0,
        fused_us=40.0,
        kept=True,
        note="KERNEL OK",
    )
    result = SimpleNamespace(
        kept=True,
        best=vr,
        best_recipe=SimpleNamespace(
            env_flag="X_FUSED",
            pattern_id="llm:x",
            source_file="/s.py",
        ),
        termination_reason="",
    )
    cli_module.apply_serving_gate(
        result,
        framework="vllm",
        out=tmp_path,
        gpu="0",
        model_path="/m",
        isl=8,
        osl=8,
        tp=8,
        block_size=128,
        max_model_len=13312,
    )
    assert result.kept is True
    assert vr.kernel_speedup == 2.5
    assert "defer" in vr.note.lower() or "e2e" in vr.note.lower()
    assert captured.get("tp") == 8
    assert captured.get("block_size") == 128
    assert captured.get("max_model_len") == 13312


def test_apply_serving_gate_cuda_graph_crash_clears_keep(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kernelforge.fusion import command as cli_module
    from kernelforge.fusion.models import ValidationResult

    from kernelforge.fusion.validate import SMOKE_STAGE_DECODE_CRASH, SmokeVerdict

    monkeypatch.setattr(cli_module, "_serving_check_enabled", lambda: True)
    monkeypatch.setattr(
        cli_module,
        "serving_smoke_verdict",
        lambda *a, **k: SmokeVerdict(
            ok=False,
            reason="scheduler crashed during CUDA-graph decode: HSA_STATUS_ERROR_EXCEPTION",
            stage=SMOKE_STAGE_DECODE_CRASH,
            blames_kernel=True,
        ),
    )
    vr = ValidationResult(
        correctness_passed=True,
        max_abs_err=0.0,
        rtol=0.02,
        kernel_speedup=2.5,
        eager_us=100.0,
        fused_us=40.0,
        kept=True,
        note="KERNEL OK",
    )
    result = SimpleNamespace(
        kept=True,
        best=vr,
        best_recipe=SimpleNamespace(
            env_flag="X_FUSED",
            pattern_id="llm:x",
            source_file="/s.py",
        ),
        termination_reason="",
    )
    cli_module.apply_serving_gate(
        result,
        framework="sglang",
        out=tmp_path,
        gpu="0",
        model_path="/m",
        isl=8,
        osl=8,
    )
    assert result.kept is False
    assert vr.kept is False
    assert vr.kernel_speedup is None
    assert "SERVING CRASHED" in vr.note
    assert not (tmp_path / "kernel_keep_checkpoint.json").exists()
