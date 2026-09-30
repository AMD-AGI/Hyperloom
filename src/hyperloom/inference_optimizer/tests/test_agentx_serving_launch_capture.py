# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU proof of capture from the real AgentX recipe invocation boundary."""

from __future__ import annotations

import json
import ctypes
import os
from pathlib import Path
import shlex
import socket
import signal
import subprocess
import sys
import time

import pytest

from hyperloom.common.serving_launch import digest, load_capture, snapshot
from hyperloom.common.proctree import descendants, proc_identity, running
from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

ENVIRONMENTS = [
    {
        "PYTHONNOUSERSITE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "0",
        "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": "256",
    },
    {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "AITER_TRITON_LOG_LEVEL": "ERROR",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "OMP_NUM_THREADS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_USE_RUST_FRONTEND": "1",
        "PYTHONUNBUFFERED": "1",
    },
]


@pytest.mark.parametrize("serving", ENVIRONMENTS, ids=["minimax_exports", "deepseek_exports"])
def test_recipe_process_capture_preserves_exports_and_binds_measurement(tmp_path, serving):
    bench = tmp_path / "benchmarks"
    deploy_agentx_assets(bench)
    result = tmp_path / "result"
    result.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    server = binary / "vllm"
    server.write_text(
        f"#!{sys.executable}\n"
        + """import socket, sys, time
port = int(sys.argv[sys.argv.index('--port') + 1])
s = socket.socket(); s.bind(('127.0.0.1', port)); s.listen()
while True: time.sleep(0.1)
"""
    )
    server.chmod(0o700)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    config = tmp_path / "config.yaml"
    config.write_text("benchmark:\n  framework: vllm\n  envs:\n    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: '1200'\n")
    script = bench / "single_node/agentic/fixture.sh"
    script.parent.mkdir(parents=True)
    flags = [
        "--tokenizer",
        "/models/tokenizer",
        "--tensor-parallel-size",
        "4",
        "--seed",
        "7",
        "--attention-backend",
        "aiter",
        "--speculative-config",
        '{"nested":{"label":"a b"}}',
    ]
    script.write_text(
        "#!/bin/bash\nset -eu\n"
        + "\n".join(f"export {k}={shlex.quote(v)}" for k, v in serving.items())
        + "\n"
        + "export VLLM_API_KEY=credential-sentinel UNRELATED_VALUE=unrelated-sentinel\n"
        + 'vllm serve /models/accepted --port "$PORT" '
        + shlex.join(flags)
        + ' > "$RESULT_DIR/server.log" 2>&1 &\n'
        + 'pid=$!\ntrap \'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true\' EXIT\n'
        + shlex.quote(sys.executable)
        + " - <<'CHILD'\nimport os, time, json\nfrom pathlib import Path\ntime.sleep(0.8)\n"
        + "Path(os.environ['RESULT_DIR'], 'inferencex_result.json').write_text(json.dumps({'throughput': 1}))\nCHILD\n"
    )
    env = {
        "PATH": str(binary) + ":" + str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "MODEL": "/models/accepted",
        "CONC": "1",
        "FRAMEWORK": "vllm",
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/fixture.sh",
        "AIPERF_BIN": "/bin/true",
        "PORT": str(port),
        "RESULT_DIR": str(result),
        "AGENTX_KEEP_SERVER": "1",
        "HYPERLOOM_LAUNCH_CONFIG_PATH": str(config),
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1200",
    }
    run = subprocess.run(["bash", str(bench / "aiperf_client.sh")], env=env, text=True, capture_output=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
    evidence = build_launch_evidence(
        config_path=config, actual_server_log=str(result / "server.log"), framework="vllm", slot=result
    )
    assert evidence["server_launch_argv_complete"] is True
    observed = evidence["observed_server_env"]
    assert all(observed.get(k) == v for k, v in serving.items())
    serialized = json.dumps(evidence["server_launch_capture"])
    assert "credential-sentinel" not in serialized and "unrelated-sentinel" not in serialized
    assert evidence["server_launch_capture"]["server"]["semantic_binding"] == {
        "model": "/models/accepted",
        "tokenizer": "/models/tokenizer",
        "served_model_name": "/models/accepted",
        "tp": "4",
        "dp": "1",
        "pp": "1",
        "seed": "7",
        "nnodes": "1",
        "node_rank": "0",
    }
    assert shlex.split(evidence["observed_server_launch_flags"]) == flags
    assert load_capture(result / "server.log", "sha256:wrong-config") == {}
    receipt_path = result / "server_launch_capture.json"
    original_receipt = receipt_path.read_text()
    changed = json.loads(original_receipt)
    changed["server"]["launch_nonce"] = "f" * 32
    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
    receipt_path.write_text(json.dumps(changed))
    assert load_capture(result / "server.log", evidence["recipe_digest"]) == {}
    receipt_path.write_text(original_receipt)
    measurement = result / "inferencex_result.json"
    measurement.write_text('{"throughput": 2}')
    assert load_capture(result / "server.log", evidence["recipe_digest"]) == {}


def test_recycled_pid_cannot_supply_capture(tmp_path):
    with pytest.raises(ValueError, match="identity changed"):
        snapshot(os.getpid(), 1, tmp_path / "server.log", 8000)


def test_recipe_cancellation_preserves_group_cleanup_without_orphan_listener(tmp_path):
    bench, result, binary = tmp_path / "benchmarks", tmp_path / "result", tmp_path / "bin"
    deploy_agentx_assets(bench)
    result.mkdir()
    binary.mkdir()
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    server = binary / "vllm"
    server.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, socket, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
        "s = socket.socket(); s.bind(('127.0.0.1', int(os.environ['PORT']))); s.listen()\n"
        "Path(os.environ['RESULT_DIR'], 'listener.ready').write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.1)\n"
    )
    server.chmod(0o700)
    recipe = bench / "single_node/agentic/cancel.sh"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        "#!/bin/bash\nset -eu\n"
        'vllm serve /models/fixture --port "$PORT" > "$RESULT_DIR/server.log" 2>&1 &\n'
        'pid=$!\ncleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }\n'
        "trap cleanup EXIT\ntrap 'exit 143' TERM INT\n"
        'wait "$pid"\n'
    )
    config = tmp_path / "config.yaml"
    config.write_text("benchmark:\n  framework: vllm\n")
    env = {
        "PATH": f"{binary}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "MODEL": "/models/fixture",
        "CONC": "1",
        "FRAMEWORK": "vllm",
        "PORT": str(port),
        "RESULT_DIR": str(result),
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/cancel.sh",
        "AIPERF_BIN": "/bin/true",
        "HYPERLOOM_LAUNCH_CONFIG_PATH": str(config),
    }
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    owned = []
    process = None

    def reset_signals():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        with (result / "client.log").open("w") as output:
            process = subprocess.Popen(
                ["bash", str(bench / "aiperf_client.sh")],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=reset_signals,
            )
            assert os.getpgid(process.pid) == process.pid != os.getpgrp()
            deadline = time.monotonic() + 5
            while not (result / "listener.ready").exists():
                assert process.poll() is None, (result / "client.log").read_text()
                assert time.monotonic() < deadline, "CPU listener did not become ready"
                time.sleep(0.02)
            owned = descendants(process.pid)
            assert len(owned) >= 3, "expected collector, recipe shell and vLLM stub"
            assert all(os.getpgid(pid) == process.pid for pid, _ in owned)
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            os.killpg(process.pid, signal.SIGTERM)
            assert process.wait(timeout=5) != 0
            deadline = time.monotonic() + 3
            while any(
                (identity := proc_identity(pid)) is not None and identity[1] == ticks and running(pid)
                for pid, ticks in owned
            ):
                assert time.monotonic() < deadline, "an owned process survived cancellation"
                time.sleep(0.02)
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0, "orphan listener survived cancellation"
        assert not (result / "server_launch_capture.json").exists()
        assert not (result / "inferencex_result.json").exists()
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        for pid, _ in owned:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0
