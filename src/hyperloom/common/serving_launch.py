# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Capture a local recipe's owned vLLM launch and bind it to its result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import cast
import uuid

from .env_safety import is_allowed_external_env_key
from .proctree import descendants, proc_identity, running

SCHEMA = "hyperloom.serving_launch.v1"
ENV_SCOPE = "serving-knobs-v1"
ENV_PREFIXES = (
    "VLLM_",
    "SGLANG_",
    "AITER_",
    "TORCH_",
    "TORCHINDUCTOR_",
    "TORCHDYNAMO_",
    "PYTORCH_",
    "TRITON_",
    "NCCL_",
    "OMP_",
    "KMP_",
    "MIOPEN_",
    "ROCM_",
    "MORI_",
    "RCCL_",
    "HSA_",
    "HIP_",
)
ENV_EXACT = {
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "TOKENIZERS_PARALLELISM",
}


def serving_env(env: dict[str, str]) -> dict[str, str]:
    """Project stable serving knobs; paths, profiling controls and secrets do not travel."""
    return {
        k: v
        for k, v in sorted(env.items())
        if is_allowed_external_env_key(k)
        and (k.startswith(ENV_PREFIXES) or k in ENV_EXACT)
        and not re.search(r"AUTH|HEADERS", k)
        and not k.endswith(("PATH", "_DIR", "_FILE", "_PORT", "_URL", "_ENDPOINT", "_ROOT", "_HOME"))
        and not any(word in k for word in ("PROFILER", "PROFILE", "TRACE", "DUMP"))
    }


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_binding(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def vllm_tail(argv: list[str]) -> list[str]:
    """Require an exact supported entrypoint, retaining the model operand."""
    index = 0
    if argv and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
        index = 1
        while index < len(argv) and argv[index] in ("-u", "-B", "-s", "-E", "-I", "-O", "-OO"):
            index += 1
        if argv[index : index + 2] == ["-m", "vllm.entrypoints.openai.api_server"]:
            return argv[index + 2 :]
    if index < len(argv) and Path(argv[index]).name == "vllm" and argv[index + 1 : index + 2] == ["serve"]:
        tail = argv[index + 2 :]
        if tail and not tail[0].startswith("-"):
            return ["--model", tail[0], *tail[1:]]
    raise ValueError("unsupported vLLM entrypoint")


def semantics(tokens: list[str]) -> dict[str, str | None]:
    aliases = {
        "--model": "model",
        "--model-path": "model",
        "--tokenizer": "tokenizer",
        "--tokenizer-path": "tokenizer",
        "--served-model-name": "served_model_name",
        "--tensor-parallel-size": "tp",
        "--tp-size": "tp",
        "--tp": "tp",
        "--data-parallel-size": "dp",
        "--dp-size": "dp",
        "--pipeline-parallel-size": "pp",
        "--pp-size": "pp",
        "--random-seed": "seed",
        "--seed": "seed",
        "--nnodes": "nnodes",
        "--node-rank": "node_rank",
    }
    out = {
        "model": None,
        "tokenizer": None,
        "served_model_name": None,
        "tp": "1",
        "dp": "1",
        "pp": "1",
        "seed": None,
        "nnodes": "1",
        "node_rank": "0",
    }
    for i, token in enumerate(tokens):
        name, equal, value = token.partition("=")
        if name in aliases:
            value = value if equal else tokens[i + 1] if i + 1 < len(tokens) else ""
            if not value or value.startswith("--"):
                raise ValueError("missing semantic launch operand")
            out[aliases[name]] = value
    if not out["model"]:
        raise ValueError("server launch has no model")
    out["tokenizer"] = out["tokenizer"] or out["model"]
    out["served_model_name"] = out["served_model_name"] or out["model"]
    return out


def snapshot(pid: int, ticks: int, log: Path, port: int, nonce: str = "") -> dict:
    process = Path("/proc") / str(pid)
    before = proc_identity(pid)
    if before is None or before[1] != ticks or not running(pid) or process.stat().st_uid != os.geteuid():
        raise ValueError("server identity changed")
    raw = (process / "cmdline").read_bytes()
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("incomplete argv")
    argv = [part.decode("utf-8") for part in raw[:-1].split(b"\0")]
    tokens = vllm_tail(argv)
    if any(
        t.split("=", 1)[0] in ("--api-key", "--api_key", "--auth-token", "--password", "--hf-token") for t in tokens
    ):
        raise ValueError("credential-bearing argv cannot be captured")
    options = {}
    for i, token in enumerate(tokens):
        name, equal, value = token.partition("=")
        if name == "--port":
            options[name] = value if equal else tokens[i + 1]
    if options.get("--port") != str(port):
        raise ValueError("server port does not match the recipe")
    output = (process / "fd/1").stat()
    target = log.stat()
    if (output.st_dev, output.st_ino) != (target.st_dev, target.st_ino):
        raise ValueError("server output does not belong to this measurement")
    selected = {}
    observed_nonce = ""
    for entry in (process / "environ").read_bytes().split(b"\0"):
        key, sep, raw_value = entry.partition(b"=")
        if sep:
            name = key.decode("utf-8")
            if name == "HYPERLOOM_SERVER_LAUNCH_ID":
                observed_nonce = raw_value.decode("utf-8")
            if serving_env({name: ""}):
                selected[name] = raw_value.decode("utf-8")
    if proc_identity(pid) != before or not running(pid) or not nonce or observed_nonce != nonce:
        raise ValueError("server identity changed during capture")
    return {
        "pid": pid,
        "start_ticks": ticks,
        "process_uid": os.geteuid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "argv": argv,
        "semantic_binding": semantics(tokens),
        "serving_env": selected,
        "launch_nonce": observed_nonce,
        "serving_env_scope": ENV_SCOPE,
        "log_device": target.st_dev,
        "log_inode": target.st_ino,
    }


def listening_process(pid: int, ticks: int, port: int) -> tuple[int, int] | None:
    identity = proc_identity(pid)
    if identity is None or identity[1] != ticks:
        return None
    sockets = set()
    for table in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / table).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                sockets.add("socket:[" + fields[9] + "]")
    for candidate, start in [(pid, ticks), *descendants(pid)]:
        try:
            if any(os.readlink(fd) in sockets for fd in (Path("/proc") / str(candidate) / "fd").iterdir()):
                identity = proc_identity(candidate)
                if identity is not None and identity[1] == start:
                    return candidate, start
        except OSError:
            continue
    return None


def run_recipe(command: list[str], result: Path, port: int, config: Path) -> int:
    pending = result.parent / "server_launch_capture.pending.json"
    receipt = result.parent / "server_launch_capture.json"
    pending.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)
    started = time.time_ns()
    owner_pid = os.getppid()
    owner = proc_identity(owner_pid)
    try:
        config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    except OSError:
        config_digest = ""
    if not config_digest or not Path("/proc/self/stat").is_file() or os.environ.get("FRAMEWORK") != "vllm":
        return subprocess.call(command)
    candidates: dict[tuple[int, int], dict] = {}
    nonce = uuid.uuid4().hex
    with subprocess.Popen(command, env={**os.environ, "HYPERLOOM_SERVER_LAUNCH_ID": nonce}) as child:
        child_identity = proc_identity(child.pid)
        while child.poll() is None:
            if any(
                captured.get("endpoint_identity")
                and (identity := proc_identity(pid)) is not None
                and identity[1] == ticks
                for (pid, ticks), captured in candidates.items()
            ):
                time.sleep(0.1)
                continue
            for pid, ticks in descendants(child.pid):
                if (pid, ticks) not in candidates:
                    try:
                        candidates[(pid, ticks)] = snapshot(pid, ticks, result.parent / "server.log", port, nonce)
                    except (OSError, ValueError, IndexError, UnicodeError):
                        continue
                captured = candidates[(pid, ticks)]
                try:
                    endpoint = listening_process(pid, ticks, port)
                except OSError:
                    continue
                if endpoint:
                    captured["endpoint_identity"] = list(endpoint)
                    captured["ready_observed_ns"] = time.time_ns()
            time.sleep(0.1)
        rc = child.returncode
    verified = [value for value in candidates.values() if value.get("endpoint_identity")]
    if rc == 0 and len(verified) == 1 and config_digest and owner and child_identity:
        capture = {
            "schema": SCHEMA,
            "capture_id": nonce,
            "started_ns": started,
            "owner_pid": owner_pid,
            "owner_start_ticks": owner[1],
            "recipe_pid": child.pid,
            "recipe_start_ticks": child_identity[1],
            "recipe_digest": "sha256:" + config_digest,
            "workspace": str(result.parent.resolve()),
            "server": verified[0],
        }
        pending.write_text(json.dumps(capture))
    return rc


def seal_result(result: Path) -> bool:
    pending = result.parent / "server_launch_capture.pending.json"
    try:
        capture = json.loads(pending.read_text())
        owner = proc_identity(os.getppid())
        if not owner or capture["owner_pid"] != os.getppid() or capture["owner_start_ticks"] != owner[1]:
            return False
        binding = file_binding(result)
        if binding["mtime_ns"] < capture["started_ns"]:
            return False
        capture["measurement"] = binding
        capture["sha256"] = digest(capture)
        (result.parent / "server_launch_capture.json").write_text(json.dumps(capture))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
    finally:
        pending.unlink(missing_ok=True)


def load_capture(log: Path, recipe_digest: str) -> dict:
    try:
        capture = cast(dict, json.loads(log.with_name("server_launch_capture.json").read_text()))
        unsigned = {k: v for k, v in capture.items() if k != "sha256"}
        measured = capture["measurement"]
        result = Path(measured["path"])
        server = capture["server"]
        stat = log.stat()
        if (
            capture["schema"] != SCHEMA
            or capture["sha256"] != digest(unsigned)
            or capture["recipe_digest"] != recipe_digest
            or capture["workspace"] != str(log.parent.resolve())
            or result.parent.resolve() != log.parent.resolve()
            or file_binding(result) != measured
            or measured["mtime_ns"] < capture["started_ns"]
            or (stat.st_dev, stat.st_ino) != (server["log_device"], server["log_inode"])
            or server["serving_env_scope"] != ENV_SCOPE
            or serving_env(server["serving_env"]) != server["serving_env"]
        ):
            return {}
        if (
            not re.fullmatch(r"[0-9a-f]{32}", capture["capture_id"])
            or server["launch_nonce"] != capture["capture_id"]
            or server["semantic_binding"] != semantics(vllm_tail(server["argv"]))
            or not all(
                server.get(k) for k in ("pid", "start_ticks", "boot_id", "endpoint_identity", "ready_observed_ns")
            )
        ):
            return {}
        return capture
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "seal"))
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, default=Path("/nonexistent"))
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:boundary])
    if args.operation == "seal":
        seal_result(args.result)
        return 0
    command = argv[boundary + 1 :]
    return run_recipe(command, args.result, args.port, args.config)
