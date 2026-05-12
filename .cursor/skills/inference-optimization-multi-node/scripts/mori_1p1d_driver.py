#!/usr/bin/env python3
"""MoRI 1P1D baseline/profile driver for remote RayJob runs.

This is the official script path for the multi-node skill when the serving
topology is one prefill worker and one decode worker, each using a full MI300X
node. It runs inside the RayJob via Ray Dashboard REST, uses in-cluster Ray
node affinity to place prefill/decode on different nodes, sends benchmark
traffic through the router, and sends profile control to the prefill backend.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} must be set")
    return val


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


@ray.remote(num_cpus=1)
def cleanup_node() -> dict[str, Any]:
    cmd = r"""
set +e
ps -eo pid,comm,args | awk '$2 ~ /sglang::router|sglang::scheduler|sglang::detoken/ {print $1}' | xargs -r kill -TERM 2>/dev/null
ps aux | grep '[p]ython3 -m sglang.launch_server' | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null
ps aux | grep '[p]ython -m sglang_router.launch_router' | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null
sleep 5
ps -eo pid,comm,args | awk '$2 ~ /sglang::router|sglang::scheduler|sglang::detoken/ {print $1}' | xargs -r kill -9 2>/dev/null
ps aux | grep '[p]ython3 -m sglang.launch_server' | awk '{print $2}' | xargs -r kill -9 2>/dev/null
ps aux | grep '[p]ython -m sglang_router.launch_router' | awk '{print $2}' | xargs -r kill -9 2>/dev/null
true
"""
    proc = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=45
    )
    return {
        "host": socket.gethostname(),
        "rc": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


@ray.remote(num_gpus=8, num_cpus=16)
class ProcessNode:
    def __init__(self) -> None:
        self.processes: dict[str, tuple[subprocess.Popen[str], Any, str]] = {}

    def start(self, name: str, cmd: str, env: dict[str, str], log_path: str) -> dict:
        run_env = os.environ.copy()
        run_env.update(env)
        visible = run_env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        run_env["HIP_VISIBLE_DEVICES"] = visible
        run_env["ROCR_VISIBLE_DEVICES"] = visible
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            env=run_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self.processes[name] = (proc, log_fh, log_path)
        return {
            "host": socket.gethostname(),
            "name": name,
            "pid": proc.pid,
            "log": log_path,
        }

    def run(self, cmd: str, env: dict[str, str] | None = None, timeout: int = 600) -> dict:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "host": socket.gethostname(),
            "rc": proc.returncode,
            "stdout": proc.stdout[-5000:],
            "stderr": proc.stderr[-5000:],
        }

    def stop(self) -> bool:
        for proc, _fh, _log in list(self.processes.values()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(5)
        for proc, fh, _log in list(self.processes.values()):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            try:
                fh.close()
            except Exception:
                pass
        self.processes.clear()
        return True


def wait_port(ip: str, port: int, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((ip, port), timeout=3):
                return
        except OSError:
            time.sleep(5)
    raise RuntimeError(f"timeout waiting for {ip}:{port}")


def wait_http(url: str, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            urllib.request.urlopen(url, timeout=5).read()
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"timeout waiting for {url}")


def post_text(url: str, payload: dict[str, Any] | None, timeout_s: int) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data or b"", headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout_s).read().decode("utf-8", "replace")


def wait_for_trace_stability(trace_dir: Path, expected_count: int, timeout_s: int) -> list[Path]:
    start = time.time()
    previous_size = -1
    while time.time() - start < timeout_s:
        traces = sorted([*trace_dir.rglob("*.json.gz"), *trace_dir.rglob("*.json")])
        total_size = sum(p.stat().st_size for p in traces if p.exists())
        if traces and len(traces) >= expected_count and total_size == previous_size:
            return traces
        previous_size = total_size
        time.sleep(5)
    return sorted([*trace_dir.rglob("*.json.gz"), *trace_dir.rglob("*.json")])


def validate_traces(traces: list[Path], limit: int = 2) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trace in traces[:limit]:
        try:
            opener = gzip.open if trace.suffix == ".gz" else open
            with opener(trace, "rt", errors="replace") as fh:
                data = json.load(fh)
            events = data.get("traceEvents", [])
            output.append(
                {
                    "path": str(trace),
                    "events": len(events),
                    "has_gpu": any(event.get("cat") == "kernel" for event in events),
                }
            )
        except Exception as exc:
            output.append({"path": str(trace), "error": repr(exc)})
    return output


def write_run_context(
    path: Path,
    *,
    model: str,
    prefill_tp: int,
    decode_tp: int,
    conc: int,
    isl: int,
    osl: int,
    inferencex_path: str,
    result_dir: Path,
    trace_dir: Path,
    router_port: int,
    prefill_port: int,
    decode_port: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"MODEL={sh_quote(model)}",
                f"TP={prefill_tp}",
                f"PREFILL_TP={prefill_tp}",
                f"DECODE_TP={decode_tp}",
                f"CONC={conc}",
                f"ISL={isl}",
                f"OSL={osl}",
                f"INFERENCEX_PATH={sh_quote(inferencex_path)}",
                f"PORT={router_port}",
                f"ROUTER_PORT={router_port}",
                f"PROFILE_PORT={prefill_port}",
                f"PREFILL_PORT={prefill_port}",
                f"DECODE_PORT={decode_port}",
                "FRAMEWORK=sglang",
                f"RESULT_DIR={sh_quote(str(result_dir))}",
                f"TRACE_DIR={sh_quote(str(trace_dir))}",
                "RESULT_FILENAME=profile_run",
                "TRACE_FOR_ANALYSIS=",
                "",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "profile"), required=True)
    args = parser.parse_args()

    model = required_env("MODEL")
    inferencex_path = required_env("INFERENCEX_PATH")
    prefill_tp = int(os.environ.get("PREFILL_TP", "8"))
    decode_tp = int(os.environ.get("DECODE_TP", "8"))
    prefill_ep = int(os.environ.get("PREFILL_EP", "1"))
    decode_ep = int(os.environ.get("DECODE_EP", "1"))
    prefill_dp_attn = env_bool("PREFILL_DP_ATTN")
    decode_dp_attn = env_bool("DECODE_DP_ATTN")
    conc = int(required_env("CONC"))
    isl = int(required_env("ISL"))
    osl = int(required_env("OSL"))
    router_port = int(os.environ.get("ROUTER_PORT", "30000"))
    prefill_port = int(os.environ.get("PREFILL_PORT", "8000"))
    decode_port = int(os.environ.get("DECODE_PORT", "8000"))
    run_id = os.environ.get("RUN_ID") or time.strftime("mori_1p1d_%Y%m%d_%H%M%S")
    result_dir = Path(os.environ.get("RESULT_DIR", f"/workspace/hyperloom/results/{run_id}"))
    trace_dir = Path(os.environ.get("TRACE_DIR", f"/workspace/hyperloom/{run_id}/traces"))
    profile_prompts = int(os.environ.get("PROFILE_PROMPTS", str(min(conc, 16))))
    baseline_prompts = int(os.environ.get("NUM_PROMPTS", str(conc * int(os.environ.get("NUM_PROMPTS_MULTIPLIER", "3")))))
    keep_server = env_bool("KEEP_SERVER")

    if prefill_tp != 8 or decode_tp != 8 or prefill_ep != 1 or decode_ep != 1:
        raise RuntimeError("This multi-node skill supports only MoRI 1P1D TP=8 EP=1 per role")
    if prefill_dp_attn or decode_dp_attn:
        raise RuntimeError("This multi-node skill supports only DP-attn=false for 1P1D")

    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    context_file = result_dir / "run_context.env"
    write_run_context(
        context_file,
        model=model,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        conc=conc,
        isl=isl,
        osl=osl,
        inferencex_path=inferencex_path,
        result_dir=result_dir,
        trace_dir=trace_dir,
        router_port=router_port,
        prefill_port=prefill_port,
        decode_port=decode_port,
    )

    ray.init()
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    head = next(
        node
        for node in nodes
        if node.get("Resources", {}).get("node:__internal_head__", 0) > 0
    )
    workers = [node for node in nodes if node["NodeID"] != head["NodeID"]]
    if not workers:
        raise RuntimeError("MoRI 1P1D requires a Ray worker node")
    worker = sorted(workers, key=lambda node: node["NodeManagerAddress"])[0]
    head_ip = head["NodeManagerAddress"]
    worker_ip = worker["NodeManagerAddress"]

    ibdevices = os.environ.get(
        "IBDEVICES",
        "rocep28s0,rocep29s0,rocep62s0,rocep79s0,rocep96s0,rocep158s0,rocep159s0,rocep190s0",
    )
    base_flags = (
        "--decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 "
        "--ep-dispatch-algorithm fake --load-balance-method round_robin "
        "--kv-cache-dtype fp8_e4m3 --attention-backend aiter "
        "--disaggregation-transfer-backend mori"
    )
    cuda_graph_bs = os.environ.get("MORI_CUDA_GRAPH_BS", "1 2 4 8 16")
    prefill_flags = (
        f"--tp-size {prefill_tp} {base_flags} --mem-fraction-static "
        f"{os.environ.get('PREFILL_MEM_FRACTION_STATIC', '0.8')} "
        f"--max-running-requests {os.environ.get('PREFILL_MAX_RUNNING_REQUESTS', str(max(conc, 16)))} "
        f"--chunked-prefill-size {os.environ.get('PREFILL_CHUNKED_PREFILL_SIZE', '262144')} "
        f"--cuda-graph-bs {cuda_graph_bs} --disable-radix-cache"
    )
    decode_flags = (
        f"--tp-size {decode_tp} {base_flags} --mem-fraction-static "
        f"{os.environ.get('DECODE_MEM_FRACTION_STATIC', '0.85')} "
        f"--max-running-requests {os.environ.get('DECODE_MAX_RUNNING_REQUESTS', str(max(conc, 16)))} "
        f"--cuda-graph-bs {cuda_graph_bs} --prefill-round-robin-balance"
    )
    common_env = {
        "PATH": "/opt/venv/bin:" + os.environ.get("PATH", ""),
        "MODEL_NAME": Path(model).name,
        "IBDEVICES": ibdevices,
        "SGLANG_USE_AITER": "1",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
        "MORI_SHMEM_MODE": "ISOLATION",
        "SGLANG_MORI_FP8_DISP": os.environ.get("SGLANG_MORI_FP8_DISP", "True"),
        "SGLANG_MORI_FP4_DISP": os.environ.get("SGLANG_MORI_FP4_DISP", "False"),
        "SGLANG_MORI_FP8_COMB": os.environ.get("SGLANG_MORI_FP8_COMB", "False"),
        "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD": os.environ.get(
            "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", "320"
        ),
        "MORI_EP_LAUNCH_CONFIG_MODE": os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "AUTO"),
        "MORI_IO_QP_MAX_SEND_WR": os.environ.get("MORI_IO_QP_MAX_SEND_WR", "16384"),
        "MORI_IO_QP_MAX_CQE": os.environ.get("MORI_IO_QP_MAX_CQE", "32768"),
        "MORI_IO_QP_MAX_SGE": os.environ.get("MORI_IO_QP_MAX_SGE", "4"),
        "MORI_APP_LOG_LEVEL": os.environ.get("MORI_APP_LOG_LEVEL", "INFO"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "enp159s0np0"),
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "enp159s0np0"),
        "NCCL_IB_HCA": ibdevices,
        "SGLANG_TORCH_PROFILER_DIR": str(trace_dir),
        "SGLANG_PROFILE_WITH_STACK": "1",
        "SGLANG_PROFILE_RECORD_SHAPES": "1",
    }

    print(json.dumps({"event": "cleanup", "results": ray.get([
        cleanup_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False
            )
        ).remote()
        for node in nodes
    ])}, indent=2))

    head_actor = ProcessNode.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head["NodeID"], soft=False)
    ).remote()
    worker_actor = ProcessNode.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=worker["NodeID"], soft=False)
    ).remote()

    try:
        ray.get(
            [
                head_actor.start.remote(
                    "prefill",
                    (
                        "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384 "
                        f"python3 -m sglang.launch_server --model-path {sh_quote(model)} "
                        f"--disaggregation-mode prefill --disaggregation-ib-device {sh_quote(ibdevices)} "
                        f"--host 0.0.0.0 --port {prefill_port} --trust-remote-code "
                        f"{prefill_flags} --log-level-http warning"
                    ),
                    common_env,
                    str(result_dir / "prefill.log"),
                ),
                worker_actor.start.remote(
                    "decode",
                    (
                        "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160 "
                        f"python3 -m sglang.launch_server --model-path {sh_quote(model)} "
                        f"--disaggregation-mode decode --disaggregation-ib-device {sh_quote(ibdevices)} "
                        f"--host 0.0.0.0 --port {decode_port} --trust-remote-code "
                        f"{decode_flags} --log-level-http warning"
                    ),
                    common_env,
                    str(result_dir / "decode.log"),
                ),
            ]
        )
        (result_dir / "state.json").write_text(
            json.dumps({"phase": "loading", "head_ip": head_ip, "worker_ip": worker_ip}, indent=2)
        )
        wait_port(head_ip, prefill_port, int(os.environ.get("SERVER_READY_TIMEOUT", "1800")))
        wait_port(worker_ip, decode_port, int(os.environ.get("SERVER_READY_TIMEOUT", "1800")))
        ray.get(
            head_actor.start.remote(
                "router",
                (
                    "python -m sglang_router.launch_router "
                    f"--pd-disaggregation --port {router_port} --policy random "
                    f"--prefill-policy random --decode-policy random "
                    f"--prefill http://{head_ip}:{prefill_port} "
                    f"--decode http://{worker_ip}:{decode_port}"
                ),
                common_env,
                str(result_dir / "router.log"),
            )
        )
        wait_http(
            f"http://{head_ip}:{router_port}/readiness",
            int(os.environ.get("ROUTER_READY_TIMEOUT", "600")),
        )
        (result_dir / "state.json").write_text(
            json.dumps({"phase": "ready", "head_ip": head_ip, "worker_ip": worker_ip}, indent=2)
        )

        if args.mode == "baseline":
            baseline_cmd = (
                f"cd {sh_quote(inferencex_path)} && "
                "python3 utils/bench_serving/benchmark_serving.py "
                f"--model {sh_quote(model)} --backend openai "
                f"--base-url http://0.0.0.0:{router_port} "
                f"--dataset-name random --random-input-len {isl} --random-output-len {osl} "
                f"--random-range-ratio 1 --num-prompts {baseline_prompts} "
                f"--max-concurrency {conc} --request-rate inf --ignore-eos --num-warmups 0 "
                f"--save-result --result-dir {sh_quote(str(result_dir))} "
                "--result-filename baseline_mori_1p1d --trust-remote-code"
            )
            baseline = ray.get(
                head_actor.run.remote(
                    baseline_cmd,
                    common_env,
                    int(os.environ.get("BENCH_TIMEOUT", "1800")),
                )
            )
            (result_dir / "baseline_result.json").write_text(json.dumps(baseline, indent=2))

        payload = {
            "output_dir": str(trace_dir),
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": True,
            "profile_prefix": "prefill",
        }
        start_profile = post_text(
            f"http://{head_ip}:{prefill_port}/start_profile",
            payload,
            int(os.environ.get("START_PROFILE_TIMEOUT", "30")),
        )
        (result_dir / "start_profile.txt").write_text(start_profile)

        profile_cmd = (
            f"cd {sh_quote(inferencex_path)} && "
            "python3 utils/bench_serving/benchmark_serving.py "
            f"--model {sh_quote(model)} --backend openai "
            f"--base-url http://0.0.0.0:{router_port} "
            f"--dataset-name random --random-input-len {isl} --random-output-len {osl} "
            f"--random-range-ratio 1 --num-prompts {profile_prompts} "
            f"--max-concurrency {min(profile_prompts, conc)} --request-rate inf "
            "--ignore-eos --num-warmups 0 --save-result "
            f"--result-dir {sh_quote(str(result_dir))} --result-filename profile_run "
            "--trust-remote-code"
        )
        profile_result = ray.get(
            head_actor.run.remote(
                profile_cmd,
                common_env,
                int(os.environ.get("PROFILE_BENCH_TIMEOUT", "1800")),
            )
        )
        (result_dir / "profile_bench_result.json").write_text(
            json.dumps(profile_result, indent=2)
        )

        try:
            stop_profile = post_text(
                f"http://{head_ip}:{prefill_port}/stop_profile",
                None,
                int(os.environ.get("STOP_PROFILE_TIMEOUT", "900")),
            )
        except Exception as exc:
            stop_profile = "ERR " + repr(exc)
        (result_dir / "stop_profile.txt").write_text(stop_profile)

        traces = wait_for_trace_stability(trace_dir, prefill_tp, 600)
        (result_dir / "trace_files.json").write_text(
            json.dumps([str(path) for path in traces], indent=2)
        )
        validation = validate_traces(traces)
        (result_dir / "trace_validation.json").write_text(json.dumps(validation, indent=2))
        final = {
            "phase": "done",
            "mode": args.mode,
            "result_dir": str(result_dir),
            "trace_dir": str(trace_dir),
            "start_profile": start_profile,
            "stop_profile": stop_profile,
            "trace_count": len(traces),
            "trace_validation": validation,
            "trace_for_analysis": str(next((p for p in traces if "TP-0" in p.name), traces[0])) if traces else None,
        }
        (result_dir / "mori_1p1d_summary.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final, indent=2))
    finally:
        if not keep_server:
            try:
                ray.get([head_actor.stop.remote(), worker_actor.stop.remote()], timeout=120)
            except Exception as exc:
                print(f"cleanup_error: {exc!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
