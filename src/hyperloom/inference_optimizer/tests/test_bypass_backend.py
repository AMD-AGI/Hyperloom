# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stage 2a/3 tests for the bypass benchmark backend + Python engine.

No real GPU/server: server launch, client subprocess, and HTTP readiness are
all injected/monkeypatched. Verifies backend selection, argv construction, the
Magpie-compatible report contract, and end-to-end orchestration.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.orchestrator.actions.executors import bypass_engine
from hyperloom.orchestrator.actions.executors import bypass_report
from hyperloom.orchestrator.actions.executors import bypass_runner
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
)


def test_bypass_backend_selected(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    backend = bb.resolve_backend()
    assert backend.name == "bypass"
    cmd = backend.build_command(
        python_exe="PY",
        config_path=Path("/cfg.yaml"),
        output_dir=Path("/out"),
    )
    assert cmd == [
        "PY",
        "-m",
        "hyperloom.orchestrator.actions.executors.bypass_runner",
        "benchmark",
        "--benchmark-config",
        "/cfg.yaml",
        "--output-dir",
        "/out",
        "--run-mode",
        "local",
    ]


def test_server_command_sglang():
    cmd = bypass_engine.build_server_command(
        framework="sglang", model="/m", tp=2, port=8888,
        max_model_len=None, extra_args=["--foo", "1"], profile_dir=None,
    )
    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--tensor-parallel-size" in cmd and "2" in cmd
    assert cmd[-2:] == ["--foo", "1"]


def test_server_command_vllm_max_len():
    cmd = bypass_engine.build_server_command(
        framework="vllm", model="/m", tp=1, port=9000,
        max_model_len=4096, extra_args=[], profile_dir=None,
    )
    assert cmd[:2] == ["vllm", "serve"]
    assert "--max-model-len" in cmd and "4096" in cmd


def test_server_command_atom_profile():
    cmd = bypass_engine.build_server_command(
        framework="atom", model="/m", tp=8, port=8888,
        max_model_len=4090, extra_args=[], profile_dir="/ws/torch_trace",
    )
    assert "atom.entrypoints.openai_server" in cmd
    assert "--torch-profiler-dir" in cmd and "/ws/torch_trace" in cmd


def test_client_command_shape():
    cmd = bypass_engine.build_client_command(
        inferencex_root="/ix", python_exe="PY", model="/m",
        base_url="http://127.0.0.1:8888", isl=128, osl=64, conc=4,
        random_range_ratio=0.5, result_dir="/ws", result_filename="inferencex_result",
    )
    assert cmd[0] == "PY"
    assert cmd[1] == "/ix/utils/bench_serving/benchmark_serving.py"
    assert "--base-url" in cmd and "http://127.0.0.1:8888" in cmd
    assert "--num-prompts" in cmd and "40" in cmd  # conc*10
    assert "--result-filename" in cmd and "inferencex_result.json" in cmd


def test_eval_command_shape():
    cmd = bypass_engine.build_eval_command(
        python_exe="PY", model="/m", base_url="http://127.0.0.1:8888",
        conc=8, out_dir="/ws/lm_eval",
    )
    assert cmd[:5] == ["PY", "-m", "lm_eval", "--model", "local-completions"]
    assert "--tasks" in cmd and "gsm8k" in cmd
    joined = " ".join(cmd)
    assert "base_url=http://127.0.0.1:8888/v1/completions" in joined


def test_wait_for_server_ready_polls_until_200():
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        return 200 if calls["n"] >= 3 else 503

    ok = bypass_engine.wait_for_server_ready(
        "http://127.0.0.1:8888",
        timeout_s=100.0,
        poll_s=0.0,
        probe=probe,
        sleep=lambda _s: None,
    )
    assert ok is True
    assert calls["n"] == 3


def test_wait_for_server_ready_times_out():
    ticks = iter([0.0, 1.0, 2.0, 3.0, 100.0])

    ok = bypass_engine.wait_for_server_ready(
        "http://127.0.0.1:8888",
        timeout_s=5.0,
        poll_s=0.0,
        probe=lambda _u: 503,
        sleep=lambda _s: None,
        now=lambda: next(ticks),
    )
    assert ok is False


def test_bypass_report_is_measurement_compatible():
    raw = {
        "request_throughput": 2.0,
        "output_throughput": 1234.5,
        "total_token_throughput": 2000.0,
        "completed": 64,
        "duration": 60.0,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 200.0,
        "mean_tpot_ms": 5.0,
        "mean_e2el_ms": 2000.0,
        "p99_e2el_ms": 3000.0,
    }
    report = bypass_report.build_report(
        raw, framework="sglang", model="/models/x", success=True,
        workspace_dir="/ws/benchmark_sglang_x", execution_time=61.0,
    )
    m = extract_benchmark_measurement(report)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 1234.5
    assert m["completed_requests"] == 64


def _write_cfg(tmp_path, inferencex, run_eval="false"):
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/x",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "envs": {"TP": 1, "CONC": 4, "ISL": 128, "OSL": 64, "RUN_EVAL": run_eval},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_bypass_run_end_to_end(tmp_path, monkeypatch):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)

    class _FakeServer:
        pid = 4242

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(
        bypass_engine, "wait_for_server_ready", lambda *a, **k: True
    )

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        # The client writes inferencex_result.json into --result-dir.
        result_dir = Path(cmd[cmd.index("--result-dir") + 1])
        raw = {
            "output_throughput": 999.0,
            "request_throughput": 1.0,
            "total_token_throughput": 1500.0,
            "completed": 40,
            "duration": 30.0,
            "mean_ttft_ms": 50.0,
            "mean_e2el_ms": 900.0,
        }
        (result_dir / "inferencex_result.json").write_text(json.dumps(raw), encoding="utf-8")

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0

    workspaces = list((tmp_path / "out").glob("benchmark_sglang_*"))
    assert len(workspaces) == 1
    ws = workspaces[0]
    report = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is True
    m = extract_benchmark_measurement(report, workspace=ws)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 999.0


def test_bypass_run_missing_inferencex_fails(tmp_path):
    cfg_path = _write_cfg(tmp_path, tmp_path / "does-not-exist")
    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 2
    workspaces = list((tmp_path / "out").glob("benchmark_sglang_*"))
    assert len(workspaces) == 1
    report = json.loads((workspaces[0] / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is False
    assert report["errors"]


def test_bypass_run_server_not_ready_fails(tmp_path, monkeypatch):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)

    class _FakeServer:
        pid = 1

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: False)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 1


def test_bypass_cli_rejects_non_local(tmp_path):
    rc = bypass_runner.main(
        [
            "benchmark",
            "--benchmark-config",
            str(tmp_path / "c.yaml"),
            "--output-dir",
            str(tmp_path / "o"),
            "--run-mode",
            "docker",
        ]
    )
    assert rc == 2

def test_bypass_eval_env_passthrough(tmp_path, monkeypatch):
    """RUN_EVAL uses MAGPIE_EVAL_TASKS/MAGPIE_EVAL_LIMIT env in the eval command."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex, run_eval="true")

    monkeypatch.setenv("MAGPIE_EVAL_TASKS", "gsm8k")
    monkeypatch.setenv("MAGPIE_EVAL_LIMIT", "8")

    class _FakeServer:
        pid = 7

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)

    captured = {}

    def fake_eval_cmd(**kwargs):
        captured.update(kwargs)
        return ["true"]  # harmless no-op command

    monkeypatch.setattr(bypass_engine, "build_eval_command", fake_eval_cmd)

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--result-dir" in cmd:
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": 100.0, "completed": 40, "duration": 30.0}),
                encoding="utf-8",
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert captured.get("tasks") == "gsm8k"
    assert captured.get("limit") == "8"


def test_bypass_eval_limit_absent_is_none(tmp_path, monkeypatch):
    """Without MAGPIE_EVAL_LIMIT, the eval command limit is None (full run)."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex, run_eval="true")

    monkeypatch.delenv("MAGPIE_EVAL_LIMIT", raising=False)
    monkeypatch.delenv("MAGPIE_EVAL_TASKS", raising=False)

    class _FakeServer:
        pid = 7

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)

    captured = {}

    def fake_eval_cmd(**kwargs):
        captured.update(kwargs)
        return ["true"]

    monkeypatch.setattr(bypass_engine, "build_eval_command", fake_eval_cmd)

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--result-dir" in cmd:
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": 100.0, "completed": 40, "duration": 30.0}),
                encoding="utf-8",
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert captured.get("tasks") == "gsm8k"
    assert captured.get("limit") is None


def test_vllm_server_command_enables_torch_profiler():
    """vllm 0.24 needs --profiler-config flags to enable the torch profiler.

    Regression: setting only VLLM_TORCH_PROFILER_DIR env is ignored by vllm
    0.24 (Unknown env var), so /start_profile returns 404 and no trace lands.
    """
    cmd = bypass_engine.build_server_command(
        framework="vllm", model="/m", tp=1, port=8888,
        max_model_len=2048, extra_args=[], profile_dir="/ws/torch_trace",
    )
    joined = " ".join(cmd)
    assert "--profiler-config.profiler" in joined
    assert "torch" in cmd
    assert "--profiler-config.torch_profiler_dir" in joined
    assert "/ws/torch_trace" in cmd


def test_vllm_server_command_no_profiler_when_dir_none():
    """No profiler flags when profiling is off (profile_dir=None)."""
    cmd = bypass_engine.build_server_command(
        framework="vllm", model="/m", tp=1, port=8888,
        max_model_len=2048, extra_args=[], profile_dir=None,
    )
    assert not any("profiler-config" in c for c in cmd)


def test_bypass_pid_meta_helpers(tmp_path):
    pid_dir = str(tmp_path)
    bypass_engine.write_lifecycle_files(
        pid_dir=pid_dir, framework="vllm", port=8888, pid=123, pgid=123, model="/m",
    )
    pidf = bypass_engine.lifecycle_pid_file(pid_dir, "vllm", 8888)
    metaf = bypass_engine.lifecycle_meta_file(pid_dir, "vllm", 8888)
    assert pidf.read_text(encoding="utf-8").split() == ["123", "123"]
    meta = json.loads(metaf.read_text(encoding="utf-8"))
    assert meta["pid"] == 123 and meta["port"] == 8888
    assert meta["base_url"] == "http://127.0.0.1:8888"


def test_server_health_ok_probe():
    assert bypass_engine.server_health_ok("http://127.0.0.1:8888", probe=lambda u: 200) is True
    assert bypass_engine.server_health_ok("http://127.0.0.1:8888", probe=lambda u: 503) is False

    def boom(u):
        raise OSError("refused")

    assert bypass_engine.server_health_ok("http://127.0.0.1:8888", probe=boom) is False


def test_server_phase_writes_pid_meta_and_persists(tmp_path, monkeypatch):
    """phase=server starts a server, writes pid/meta, and does NOT terminate it."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()

    class _FakeServer:
        pid = 4321

        def wait(self, timeout=None):
            return 0

    terminated = {"called": False}
    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: terminated.__setitem__("called", True))
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)
    monkeypatch.setattr(bypass_runner.os, "getpgid", lambda pid: pid)

    rc = bypass_runner.run_benchmark(
        cfg_path, tmp_path / "out", phase="server", pid_dir=str(pid_dir),
    )
    assert rc == 0
    assert terminated["called"] is False  # server must persist
    pidf = bypass_engine.lifecycle_pid_file(str(pid_dir), "sglang", 8888)
    assert pidf.exists()
    assert pidf.read_text(encoding="utf-8").split()[0] == "4321"


def test_server_phase_requires_pid_dir(tmp_path, monkeypatch):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)
    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out", phase="server", pid_dir=None)
    assert rc == 2


def test_client_phase_reuses_healthy_server(tmp_path, monkeypatch):
    """phase=client reuses a running server, runs client, writes report."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    # phase=client reuses a server a prior server phase persisted: pid/meta exist.
    bypass_engine.write_lifecycle_files(
        pid_dir=str(pid_dir), framework="sglang", port=8888, pid=4321, pgid=4321, model="/models/x",
    )

    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: True)
    teardown = {"called": False}
    import hyperloom.orchestrator.actions.executors._server_lifecycle as sl
    monkeypatch.setattr(sl, "teardown_lifecycle_server", lambda **k: teardown.__setitem__("called", True))

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--result-dir" in cmd:
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": 500.0, "completed": 40, "duration": 30.0}),
                encoding="utf-8",
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    # cleanup=False -> server must NOT be torn down.
    rc = bypass_runner.run_benchmark(
        cfg_path, tmp_path / "out", phase="client", pid_dir=str(pid_dir), cleanup=False,
    )
    assert rc == 0
    assert teardown["called"] is False
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is True

    # cleanup=True -> server torn down.
    rc = bypass_runner.run_benchmark(
        cfg_path, tmp_path / "out2", phase="client", pid_dir=str(pid_dir), cleanup=True,
    )
    assert rc == 0
    assert teardown["called"] is True


def test_client_phase_no_server_fails(tmp_path, monkeypatch):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: False)
    rc = bypass_runner.run_benchmark(
        cfg_path, tmp_path / "out", phase="client", pid_dir=str(tmp_path), cleanup=True,
    )
    assert rc == 1


def _write_cfg_lifecycle(tmp_path, inferencex, cleanup, pid_dir):
    import yaml
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/x",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "envs": {"TP": 1, "CONC": 4, "ISL": 128, "OSL": 64, "RUN_EVAL": "false", "PORT": 8888},
            "server_lifecycle": {"enabled": True, "cleanup": cleanup, "pid_dir": str(pid_dir)},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _fake_client_run(monkeypatch, tput=700.0):
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--result-dir" in cmd:
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": tput, "completed": 40, "duration": 30.0}),
                encoding="utf-8",
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_yaml_lifecycle_first_round_persists(tmp_path, monkeypatch):
    """server_lifecycle warmup round (cleanup=false, no server yet): start + persist, no teardown."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    cfg_path = _write_cfg_lifecycle(tmp_path, inferencex, cleanup=False, pid_dir=pid_dir)

    class _FakeServer:
        pid = 5555

        def wait(self, timeout=None):
            return 0

    terminated = {"n": 0}
    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: terminated.__setitem__("n", terminated["n"] + 1))
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: False)  # no server yet
    monkeypatch.setattr(bypass_runner.os, "getpgid", lambda pid: pid)
    _fake_client_run(monkeypatch)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")  # phase defaults to all
    assert rc == 0
    assert terminated["n"] == 0  # cleanup=false -> persist
    assert bypass_engine.lifecycle_pid_file(str(pid_dir), "sglang", 8888).exists()


def test_yaml_lifecycle_reuse_round_teardown(tmp_path, monkeypatch):
    """server_lifecycle measure round (cleanup=true, healthy server present): reuse + teardown."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    cfg_path = _write_cfg_lifecycle(tmp_path, inferencex, cleanup=True, pid_dir=pid_dir)

    # A prior round persisted the server: pid/meta exist alongside a healthy port.
    bypass_engine.write_lifecycle_files(
        pid_dir=str(pid_dir), framework="sglang", port=8888, pid=4321, pgid=4321, model="/models/x",
    )
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: True)  # server already up
    launched = {"n": 0}
    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: launched.__setitem__("n", launched["n"] + 1))
    teardown = {"called": False}
    import hyperloom.orchestrator.actions.executors._server_lifecycle as sl
    monkeypatch.setattr(sl, "teardown_lifecycle_server", lambda **k: teardown.__setitem__("called", True))
    _fake_client_run(monkeypatch)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert launched["n"] == 0  # reused existing server, did not launch a new one
    assert teardown["called"] is True  # cleanup=true -> torn down


def test_lifecycle_server_ready_timeout_honored(tmp_path, monkeypatch):
    """server_lifecycle.server_ready_timeout_s bounds server-boot, not the client.

    wait_for_server_ready must receive the lifecycle server_ready_timeout_s
    (not benchmark.timeout_seconds), while the client benchmark keeps using
    timeout_seconds.
    """
    import yaml

    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/x",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "envs": {"TP": 1, "CONC": 4, "ISL": 128, "OSL": 64, "RUN_EVAL": "false", "PORT": 8888},
            "server_lifecycle": {
                "enabled": True,
                "cleanup": False,
                "pid_dir": str(pid_dir),
                "server_ready_timeout_s": 1234,
            },
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class _FakeServer:
        pid = 5555

        def wait(self, timeout=None):
            return 0

    seen = {}

    def fake_wait(base_url, *, timeout_s, **kwargs):
        seen["timeout_s"] = timeout_s
        return True

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", fake_wait)
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: False)  # no server yet
    monkeypatch.setattr(bypass_runner.os, "getpgid", lambda pid: pid)
    _fake_client_run(monkeypatch)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    # server-boot budget comes from server_ready_timeout_s, not timeout_seconds.
    assert seen["timeout_s"] == 1234


def test_remote_multinode_client_no_server(tmp_path, monkeypatch):
    """BENCHMARK_BASE_URL set: bypass runs client against remote, no local server."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)

    monkeypatch.setenv("BENCHMARK_BASE_URL", "http://head-pod:8888")
    launched = {"n": 0}
    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: launched.__setitem__("n", launched["n"] + 1))

    captured = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--result-dir" in cmd:
            captured["base_url"] = cmd[cmd.index("--base-url") + 1] if "--base-url" in cmd else None
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": 600.0, "completed": 40, "duration": 30.0}),
                encoding="utf-8",
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert launched["n"] == 0  # no local server launched
    assert captured.get("base_url") == "http://head-pod:8888"  # client hit remote
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is True


def test_scriptable_script_resolution(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import bypass_scriptable as bs

    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    ix_script = inferencex / "benchmarks" / "xdit_mi300x.sh"
    ix_script.write_text("#!/bin/bash\n", encoding="utf-8")

    monkeypatch.delenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    got = bs.resolve_scriptable_script("xdit", "mi300x", str(inferencex))
    assert got == ix_script

    override = tmp_path / "scripts"
    override.mkdir()
    ov_script = override / "xdit_mi300x.sh"
    ov_script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(override))
    got = bs.resolve_scriptable_script("xdit", "mi300x", str(inferencex))
    assert got == ov_script  # override wins


def test_scriptable_script_missing_returns_none(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import bypass_scriptable as bs

    monkeypatch.delenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert bs.resolve_scriptable_script("xdit", "mi300x", str(tmp_path)) is None


def test_scriptable_run_end_to_end(tmp_path, monkeypatch):
    """xdit scriptable run produces a report carrying quality_gate; eval maps it."""
    import yaml
    from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results

    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    # Fake xdit script: write an InferenceX-shaped result with a passing gate.
    (scripts_dir / "xdit_mi300x.sh").write_text(
        "#!/bin/bash\n"
        "cat > \"$RESULT_DIR/$RESULT_FILENAME.json\" <<JSON\n"
        "{\"framework\": \"xdit\", \"workload_kind\": \"scriptable\", "
        "\"throughput_unit\": \"img/s\", \"output_throughput\": 1.5, "
        "\"latency_s\": 0.66, "
        "\"quality_gate\": {\"passed\": true, \"lpips\": 0.01, \"ssim\": 0.99}}\n"
        "JSON\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts_dir))

    cfg = {
        "benchmark": {
            "framework": "xdit",
            "model": "/primus/models/FLUX",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "workload_kind": "scriptable",
            "envs": {"TP": 1},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    ws = sorted((tmp_path / "out").glob("benchmark_xdit_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is True
    assert rep["framework"] == "xdit"
    assert rep["throughput_unit"] == "img/s"
    assert rep["quality_gate"]["passed"] is True

    # parse_eval_results maps a passing quality_gate onto accuracy=1.0 for xdit.
    out = parse_eval_results(ws, framework="xdit")
    assert out["accuracy"] == 1.0


def test_scriptable_profile_passthrough(tmp_path, monkeypatch):
    """torch_profiler.enabled=true must reach the scriptable script as PROFILE=1.

    The serving path injects PROFILE/profiler-dir env, but the scriptable
    (xDiT) path previously dropped it, so profiler never engaged. The fake
    script records the PROFILE env + profiler dir it received.
    """
    import yaml

    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    # Fake xdit script: echo the PROFILE env + profiler dir into the result JSON.
    (scripts_dir / "xdit_mi300x.sh").write_text(
        "#!/bin/bash\n"
        "PROF_DIR=\"${VLLM_TORCH_PROFILER_DIR:-${SGLANG_TORCH_PROFILER_DIR:-}}\"\n"
        "cat > \"$RESULT_DIR/$RESULT_FILENAME.json\" <<JSON\n"
        "{\"framework\": \"xdit\", \"workload_kind\": \"scriptable\", "
        "\"throughput_unit\": \"img/s\", \"output_throughput\": 1.5, "
        "\"latency_s\": 0.66, \"seen_profile\": \"${PROFILE:-unset}\", "
        "\"seen_profile_dir\": \"${PROF_DIR}\", "
        "\"quality_gate\": {\"passed\": true}}\n"
        "JSON\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.delenv("PROFILE", raising=False)

    cfg = {
        "benchmark": {
            "framework": "xdit",
            "model": "/primus/models/FLUX",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "workload_kind": "scriptable",
            "envs": {"TP": 1},
            "profiler": {"torch_profiler": {"enabled": True}},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    ws = sorted((tmp_path / "out").glob("benchmark_xdit_*"))[-1]
    raw = json.loads((ws / "inferencex_result.json").read_text(encoding="utf-8"))
    # The script must have observed PROFILE=1 and a profiler dir.
    assert raw["seen_profile"] == "1"
    assert raw["seen_profile_dir"]
    assert "torch_trace" in raw["seen_profile_dir"]
    # The profiler dir must have been created under the workspace.
    assert (ws / "torch_trace").is_dir()


def test_scriptable_profile_disabled_by_default(tmp_path, monkeypatch):
    """Without torch_profiler, the scriptable script sees no PROFILE=1."""
    import yaml

    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "xdit_mi300x.sh").write_text(
        "#!/bin/bash\n"
        "cat > \"$RESULT_DIR/$RESULT_FILENAME.json\" <<JSON\n"
        "{\"framework\": \"xdit\", \"workload_kind\": \"scriptable\", "
        "\"throughput_unit\": \"img/s\", \"output_throughput\": 1.5, "
        "\"latency_s\": 0.66, \"seen_profile\": \"${PROFILE:-unset}\", "
        "\"quality_gate\": {\"passed\": true}}\n"
        "JSON\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.delenv("PROFILE", raising=False)

    cfg = {
        "benchmark": {
            "framework": "xdit",
            "model": "/primus/models/FLUX",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "workload_kind": "scriptable",
            "envs": {"TP": 1},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    ws = sorted((tmp_path / "out").glob("benchmark_xdit_*"))[-1]
    raw = json.loads((ws / "inferencex_result.json").read_text(encoding="utf-8"))
    assert raw["seen_profile"] in ("unset", "0")


def test_num_prompts_warmups_passthrough(tmp_path, monkeypatch):
    """NUM_PROMPTS/NUM_WARMUPS from YAML envs reach the client command."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    import yaml
    cfg = {
        "benchmark": {
            "framework": "sglang", "model": "/m", "precision": "bf16",
            "runner_type": "mi300x", "run_mode": "local",
            "inferencex_path": str(inferencex), "timeout_seconds": 60,
            "envs": {"TP": 1, "CONC": 4, "ISL": 128, "OSL": 64, "RUN_EVAL": "false",
                     "NUM_PROMPTS": 37, "NUM_WARMUPS": 3},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class _FakeServer:
        pid = 1
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)

    captured = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        if "--num-prompts" in cmd:
            captured["num_prompts"] = cmd[cmd.index("--num-prompts") + 1]
            captured["num_warmups"] = cmd[cmd.index("--num-warmups") + 1]
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "inferencex_result.json").write_text(
                json.dumps({"output_throughput": 1.0, "completed": 1, "duration": 1.0}), encoding="utf-8"
            )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert captured.get("num_prompts") == "37"
    assert captured.get("num_warmups") == "3"


def test_server_env_injects_rocr_visible_devices():
    env = bypass_runner._server_env(False, None, {"ROCR_VISIBLE_DEVICES": "0,1,2,3"})
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
    env2 = bypass_runner._server_env(False, None, {})
    # No pin in bench_envs: whatever the parent env had (may be unset).
    assert env2.get("ROCR_VISIBLE_DEVICES") == os.environ.get("ROCR_VISIBLE_DEVICES")


def _eval_client_run(monkeypatch, *, client_rc=0, eval_rc=1):
    """Fake subprocess.run: client writes result (client_rc), eval returns eval_rc.

    The client is identified by --result-dir (writes inferencex_result.json);
    anything else is treated as the eval subprocess.
    """
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        is_client = "--result-dir" in cmd and "benchmark_serving.py" in " ".join(cmd)
        if is_client:
            rd = Path(cmd[cmd.index("--result-dir") + 1])
            rd.mkdir(parents=True, exist_ok=True)
            if client_rc == 0:
                (rd / "inferencex_result.json").write_text(
                    json.dumps({"output_throughput": 700.0, "completed": 40, "duration": 30.0}),
                    encoding="utf-8",
                )

            class _C:
                returncode = client_rc
                stdout = "client ok"
                stderr = ""

            return _C()

        class _E:
            returncode = eval_rc
            stdout = ""
            stderr = "run_eval failed with exit code 1" if eval_rc else ""

        return _E()

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_eval_failure_propagates_as_run_failure(tmp_path, monkeypatch):
    """client succeeds but lm-eval fails: the whole run must fail (no silent 0).

    Magpie's ``run_eval ... || exit $?`` aborts the benchmark on eval failure;
    bypass must mirror that so baseline's eval-rooted RUN_EVAL=false fallback can
    detect it. The report must be success=false and carry the eval marker.
    """
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex, run_eval="true")

    class _FakeServer:
        pid = 7

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)
    _eval_client_run(monkeypatch, client_rc=0, eval_rc=1)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc != 0  # eval failure must not be swallowed
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is False
    assert any("run_eval failed with exit code" in e for e in rep["errors"])


def test_eval_success_keeps_run_success(tmp_path, monkeypatch):
    """client + eval both succeed: run succeeds (regression guard for the fix)."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex, run_eval="true")

    class _FakeServer:
        pid = 7

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bypass_runner, "_launch_server", lambda cmd, env, log: _FakeServer())
    monkeypatch.setattr(bypass_runner, "_terminate_server", lambda proc: None)
    monkeypatch.setattr(bypass_engine, "wait_for_server_ready", lambda *a, **k: True)
    _eval_client_run(monkeypatch, client_rc=0, eval_rc=0)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is True


def test_lifecycle_reuse_without_metadata_fails(tmp_path, monkeypatch):
    """YAML-lifecycle: /health=200 but no pid/meta files means a foreign/zombie
    server occupies the port. bypass must NOT silently reuse or re-boot over it;
    it fails explicitly so the reuse-key mismatch surfaces instead of being
    papered over.
    """
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    cfg_path = _write_cfg_lifecycle(tmp_path, inferencex, cleanup=True, pid_dir=pid_dir)

    # Healthy port, but pid/meta are absent (no prior bypass round wrote them).
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: True)
    launched = {"n": 0}
    monkeypatch.setattr(
        bypass_runner, "_launch_server",
        lambda cmd, env, log: launched.__setitem__("n", launched["n"] + 1),
    )
    _fake_client_run(monkeypatch)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc != 0  # explicit failure, not a silent reuse/boot
    assert launched["n"] == 0  # must not boot over a foreign server
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is False


def test_lifecycle_reuse_with_metadata_reuses(tmp_path, monkeypatch):
    """YAML-lifecycle: /health=200 AND pid/meta present -> reuse (no new boot)."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    cfg_path = _write_cfg_lifecycle(tmp_path, inferencex, cleanup=True, pid_dir=pid_dir)

    # A prior round persisted the server: pid/meta exist and port is healthy.
    bypass_engine.write_lifecycle_files(
        pid_dir=str(pid_dir), framework="sglang", port=8888, pid=4321, pgid=4321, model="/models/x",
    )
    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: True)
    launched = {"n": 0}
    monkeypatch.setattr(
        bypass_runner, "_launch_server",
        lambda cmd, env, log: launched.__setitem__("n", launched["n"] + 1),
    )
    teardown = {"called": False}
    import hyperloom.orchestrator.actions.executors._server_lifecycle as sl
    monkeypatch.setattr(sl, "teardown_lifecycle_server", lambda **k: teardown.__setitem__("called", True))
    _fake_client_run(monkeypatch)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0
    assert launched["n"] == 0  # reused existing server
    assert teardown["called"] is True  # cleanup=true -> torn down


def test_client_phase_requires_pid_dir(tmp_path, monkeypatch):
    """phase=client without pid_dir must fail with a configuration error."""
    inferencex = tmp_path / "InferenceX"
    (inferencex / "utils" / "bench_serving").mkdir(parents=True)
    (inferencex / "utils" / "bench_serving" / "benchmark_serving.py").write_text("", encoding="utf-8")
    cfg_path = _write_cfg(tmp_path, inferencex)

    monkeypatch.setattr(bypass_engine, "server_health_ok", lambda *a, **k: True)

    rc = bypass_runner.run_benchmark(
        cfg_path, tmp_path / "out", phase="client", pid_dir=None, cleanup=True,
    )
    assert rc == 1
    ws = sorted((tmp_path / "out").glob("benchmark_sglang_*"))[-1]
    rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["success"] is False
    assert any("phase=client requires pid_dir" in e for e in rep["errors"])


def test_write_report_emits_magpie_compat_artifacts(tmp_path):
    """summary.txt, log aliases, and profiling_enabled mirror Magpie outputs."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "client_stdout.log").write_text("client-out\n", encoding="utf-8")
    (workspace / "client_stderr.log").write_text("client-err\n", encoding="utf-8")

    report = bypass_report.build_report(
        {
            "output_throughput": 123.4,
            "request_throughput": 12.3,
            "total_token_throughput": 200.0,
            "completed": 10,
            "duration": 5.0,
            "mean_ttft_ms": 11.0,
            "mean_tpot_ms": 2.0,
        },
        framework="sglang",
        model="/models/x",
        success=True,
        workspace_dir=str(workspace),
        execution_time=42.0,
        profiling_enabled=True,
    )
    bypass_report.write_report(workspace, report)

    rep = json.loads((workspace / "benchmark_report.json").read_text(encoding="utf-8"))
    assert rep["profiling_enabled"] is True
    summary = (workspace / "summary.txt").read_text(encoding="utf-8")
    assert "success: True" in summary
    assert "output_throughput: 123.4" in summary
    assert "profiling_enabled: True" in summary
    assert (workspace / "benchmark_stdout.log").read_text(encoding="utf-8") == "client-out\n"
    assert (workspace / "benchmark_stderr.log").read_text(encoding="utf-8") == "client-err\n"
