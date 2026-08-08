#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run forge-loop against a multi-GPU collective kernel, as a kernel-agent tool.

The orchestrator writes an input JSON and calls this script; the optimisation
loop itself lives in the standalone KernelForge ``kernel-agents`` CLI.

This wrapper owns three things the loop does not:

* **Rig generation.** A collective task needs a torchrun driver, which
  ``collective_driver_generator`` builds from the candidate's kernel contract
  and traced shapes.
* **Rank-aware invocation.** ``--nproc-per-node`` makes the loop profile every
  rank; wrapping the launcher process alone would profile a process that runs no
  kernel.
* **Contract translation.** forge-loop's result dict is normalised into the
  Hyperloom kernel-result contract (a ``FORGE_COLLECTIVE_RESULT_BEGIN/END``
  stdout sentinel plus an on-disk ``result.json``).

A KEPT collective carries kernel-level parity only, so ``requires_e2e_validation``
is always set: the orchestrator's integrate gate confirms the end-to-end gain.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collective_driver_generator import generate_collective_driver  # noqa: E402

sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
try:
    from _llm_stability_env import apply_llm_stability_env  # noqa: E402
except ImportError:  # pragma: no cover - backends dir is optional in unit tests

    def apply_llm_stability_env(_env) -> None:
        """No-op when the stability-env helper is unavailable."""


sys.path.pop(0)

RESULT_BEGIN = "FORGE_COLLECTIVE_RESULT_BEGIN"
RESULT_END = "FORGE_COLLECTIVE_RESULT_END"
DEFAULT_TIMEOUT_SEC = 14400  # 4h: a collective iterates over N ranks per bench.
DEFAULT_SNR_THRESHOLD = 30.0
# forge-loop refuses to run on main/master so an optimisation campaign can never
# rewrite the pristine baseline it measures against. It checks out this branch
# in the workspace before snapshotting the campaign config.
DEFAULT_GIT_BRANCH = "forge-collective-opt"


def _inject_author_gateway_env() -> None:
    """Seed the author subprocess's gateway auth from the OpenAI-proxy env.

    forge-loop drives an agent CLI that authenticates via ``ANTHROPIC_*``, while
    Hyperloom's session env only carries the OpenAI-compatible proxy variables.
    Only fills what is absent; explicit operator values always win.
    """
    openai_base = str(os.environ.get("OPENAI_BASE_URL") or "").strip()
    if openai_base and not os.environ.get("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = openai_base[:-3] if openai_base.endswith("/v1") else openai_base
    token = str(os.environ.get("SAFE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if token:
        os.environ.setdefault("ANTHROPIC_API_KEY", token)
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", token)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    apply_llm_stability_env(os.environ)


def _load_input_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {path}")
    return data


def _add_opt(cmd: list[str], value: Any, flag: str) -> None:
    if value not in (None, "", []):
        cmd.extend([flag, str(value)])


def _build_cmd(args: dict[str, Any], rig: dict[str, str], output_dir: Path) -> list[str]:
    """Assemble the ``forge-loop`` invocation.

    Invoked as a module rather than through the ``kernel-agents`` console
    script: kernel_agents is resolved from ``$FORGE_PATH`` on PYTHONPATH, so the
    entry point is frequently not installed and the bare name raises
    FileNotFoundError. This mirrors ``forge_submit._run_forge_loop``.
    """
    candidate = args.get("candidate") or {}
    source_file = str(args.get("source_file") or candidate.get("source_file") or "")
    workspace = str(args.get("kernel_repo") or candidate.get("kernel_repo") or "")
    if not source_file:
        raise ValueError("source_file is required")
    if not workspace:
        raise ValueError("kernel_repo is required")

    cli = args.get("cli")
    cmd = [str(cli), "forge-loop"] if cli else [sys.executable, "-m", "kernel_agents.cli", "forge-loop"]
    _add_opt(cmd, workspace, "--workspace")
    _add_opt(cmd, source_file, "--kernel")
    _add_opt(cmd, rig["driver"], "--driver")
    _add_opt(cmd, rig["program"], "--program-md-file")
    _add_opt(cmd, "repository", "--task-type")
    _add_opt(cmd, args.get("git_branch") or DEFAULT_GIT_BRANCH, "--git-branch")
    _add_opt(cmd, rig["world_size"], "--nproc-per-node")
    _add_opt(cmd, args.get("snr_threshold") or DEFAULT_SNR_THRESHOLD, "--snr-threshold")
    _add_opt(cmd, args.get("gpu_target"), "--gpu-target")
    _add_opt(cmd, args.get("max_iters"), "--max-iters")
    _add_opt(cmd, args.get("max_hours"), "--max-hours")
    _add_opt(cmd, args.get("llm_model"), "--model")
    _add_opt(cmd, str(output_dir / "forge_result.json"), "--result-json")
    _add_opt(cmd, str(output_dir / "experiments"), "--experiments-dir")
    # A collective's true speedup often sits near the noise floor, so repeat the
    # per-case measurement and calibrate the floor before judging.
    _add_opt(cmd, args.get("bench_repeat") or 3, "--bench-repeat")
    _add_opt(cmd, args.get("calibrate_noise_floor") or 5, "--calibrate-noise-floor")
    _add_opt(cmd, args.get("structural_edits"), "--structural-edits")
    target_functions = args.get("target_functions")
    if isinstance(target_functions, (list, tuple)):
        target_functions = ",".join(str(t) for t in target_functions if t)
    _add_opt(cmd, target_functions, "--target-functions")
    _add_opt(cmd, args.get("workload_key"), "--workload-key")
    return cmd


def _timeout_sec(args: dict[str, Any]) -> int:
    raw = args.get("timeout") or args.get("timeout_sec") or os.environ.get("FORGE_COLLECTIVE_TIMEOUT")
    try:
        return max(1, int(float(raw or DEFAULT_TIMEOUT_SEC)))
    except (OverflowError, TypeError, ValueError):
        return DEFAULT_TIMEOUT_SEC


def _terminate_process_tree(proc: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    """Best-effort teardown of the loop and every rank it spawned."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
            own_pgid = os.getpgid(0)
        except OSError:
            pgid = own_pgid = None
        if pgid is not None and pgid != own_pgid:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=grace_seconds)
                return
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
                return
    try:
        proc.terminate()
        proc.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def _run_with_tree_timeout(cmd: list[str], timeout_sec: int) -> subprocess.CompletedProcess:
    """Run forge-loop in its own process group and reap the group on timeout."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **({"start_new_session": True} if os.name == "posix" else {}),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            stdout = stderr = ""
        raise subprocess.TimeoutExpired(cmd, timeout_sec, output=stdout, stderr=stderr)


def _persist_logs(output_dir: str, stdout: str | None, stderr: str | None) -> None:
    """Persist forge-loop output next to the result so failures stay diagnosable.

    The orchestrator keeps only the sentinel-wrapped JSON it scrapes from stdout,
    so without this a non-zero exit leaves no record of why forge-loop refused to
    run -- the wrapper reports ``no forge_result.json`` and the actual message is
    gone.
    """
    base = Path(output_dir or "")
    if not base.is_dir():
        return
    for name, text in (
        ("forge_loop_stdout.log", stdout),
        ("forge_loop_stderr.log", stderr),
    ):
        if not text:
            continue
        try:
            (base / name).write_text(text)
        except OSError:
            pass


def _base_result(output_dir: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "engine": "forge_collective",
        "micro_decision": "failed",
        "decision": "REVERT",
        "kept": False,
        "kernel_speedup": None,
        "env_flags": {},
        "artifact_files": [],
        "patch": None,
        "requires_e2e_validation": False,
        "workspace": str(output_dir or ""),
    }


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    """First present, non-null value among ``keys`` (schema-tolerant)."""
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _normalize_result(output_dir: str, rc: int, rig: dict[str, str]) -> dict[str, Any]:
    """Map forge-loop's result dict onto the Hyperloom kernel-result contract."""
    result = _base_result(output_dir)
    result["collective_op"] = rig.get("collective_op", "")
    result["world_size"] = rig.get("world_size", "")
    result["driver"] = rig.get("driver", "")

    path = Path(output_dir or ".") / "forge_result.json"
    if not path.is_file():
        result["error"] = f"no forge_result.json at {path} (forge-loop rc={rc})"
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"forge_result.json parse error: {exc!r}"
        return result
    if not isinstance(payload, dict):
        result["error"] = "forge_result.json is not a JSON object"
        return result

    kept = bool(_first(payload, "kept", "improved", "success") or False)
    speedup = _first(payload, "speedup", "best_speedup", "kernel_speedup")
    changed = _first(payload, "changed_files", "artifact_files") or []
    if isinstance(changed, str):
        changed = [changed]

    result.update(
        {
            "status": "ok" if kept else "complete",
            "micro_decision": "candidate" if kept else "no_improvement",
            "decision": "KEEP" if kept else "REVERT",
            "kept": kept,
            "kernel_speedup": speedup,
            "artifact_files": list(changed),
            "patch": _first(payload, "patch", "patch_path"),
            "source_file": str(_first(payload, "kernel", "source_file") or ""),
            "kernel_repo": str(_first(payload, "workspace", "workspace_dir") or ""),
            "iterations": _first(payload, "iterations", "iters"),
            "experiment_id": _first(payload, "experiment_id", "run_id"),
            # Kernel parity only; integrate confirms the real end-to-end gain.
            "requires_e2e_validation": kept,
        }
    )
    return result


def _timeout_result(output_dir: str, timeout_sec: int, exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    result = _base_result(output_dir)
    cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or []))
    result["error_class"] = "subprocess_timeout"
    result["error"] = f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}"
    return result


def _relay(stdout: Any, stderr: Any) -> None:
    if stdout:
        sys.stdout.write(stdout if isinstance(stdout, str) else stdout.decode("utf-8", "replace"))
    if stderr:
        sys.stderr.write(stderr if isinstance(stderr, str) else stderr.decode("utf-8", "replace"))


def _emit(result: dict[str, Any], output_dir: str) -> None:
    """Write result.json (disk fallback) and print the stdout sentinel."""
    if output_dir:
        try:
            (Path(output_dir) / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except OSError:
            pass
    print(f"\n{RESULT_BEGIN}\n{json.dumps(result, sort_keys=True)}\n{RESULT_END}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hyperloom wrapper for forge-loop (collective)")
    parser.add_argument("--input-json", required=True)
    try:
        args = parser.parse_args(list(argv or sys.argv[1:]))
        payload = _load_input_json(args.input_json)
        output_dir = str(payload.get("output_dir") or "")
        if not output_dir:
            raise ValueError("output_dir is required")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rig = generate_collective_driver(
            payload.get("candidate") or {},
            output_dir,
            tp=int(payload.get("tp") or 8),
        )
        cmd = _build_cmd(payload, rig, Path(output_dir))
    except Exception as exc:  # noqa: BLE001 - structured wrapper failure
        print(
            json.dumps(
                {
                    "status": "failed",
                    "engine": "forge_collective",
                    "micro_decision": "failed",
                    "decision": "REVERT",
                    "kept": False,
                    "error_class": exc.__class__.__name__,
                    "error": repr(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    _inject_author_gateway_env()
    timeout_sec = _timeout_sec(payload)
    try:
        proc = _run_with_tree_timeout(cmd, timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _relay(getattr(exc, "stdout", None), getattr(exc, "stderr", None))
        _persist_logs(output_dir, getattr(exc, "stdout", None), getattr(exc, "stderr", None))
        result = _timeout_result(output_dir, timeout_sec, exc)
        _emit(result, output_dir)
        return 124

    _relay(proc.stdout, proc.stderr)
    _persist_logs(output_dir, proc.stdout, proc.stderr)
    result = _normalize_result(output_dir, proc.returncode, rig)
    _emit(result, output_dir)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
