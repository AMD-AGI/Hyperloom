#!/usr/bin/env python3
"""GEAK submission via Ray (preferred) or direct CLI fallback.

This is a self-contained alternative to inference-optimization's
`geak_ray_submit.py`. It does not import or depend on that script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ray_runtime import (
    quiet_ray_init,
)


def _find_geak_bin() -> str:
    for name in ("geak", "mini", "geak-gaagent"):
        path = shutil.which(name)
        if path:
            return path
    return "geak"


def _resolve_geak_config() -> Path:
    geak_config = os.environ.get("GEAK_CONFIG", "").strip()
    if not geak_config:
        raise ValueError(
            "GEAK_CONFIG is required; run inference_optimizer/scripts/install.sh "
            "and source $KERNEL_AGENT_ENV "
            "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
        )
    path = Path(geak_config)
    if not path.is_file():
        raise ValueError(f"GEAK_CONFIG does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"(?m)^\s*model_class\s*:\s*litellm\s*$", text):
        raise ValueError(f"GEAK_CONFIG must set model.model_class: litellm: {path}")
    return path


def _build_cmd(prompt_file: Path, output_dir: Path, kernel_path: str, gpu_ids: str,
               cost_limit: float | None, kernel_repo: str = "",
               test_command: str = "") -> list[str]:
    cmd = [_find_geak_bin(), "-t", str(prompt_file), "--yolo",
           "--output", str(output_dir), "--gpu-ids", gpu_ids]
    cmd.extend(["--config", str(_resolve_geak_config())])
    if kernel_path:
        cmd.extend(["--kernel-path", kernel_path])
    if kernel_repo:
        cmd.extend(["--repo", kernel_repo])
    if test_command:
        cmd.extend(["--test-command", test_command])
    # cost_limit semantics (matches GEAK's ``-l/--cost-limit`` option):
    #   * ``None``  — caller did not pass a value; do NOT add the flag, so
    #                 GEAK falls back to its config-file value. For Hyperloom
    #                 callers this branch is unreachable today because
    #                 ``kernel_optimization.py`` defaults to ``0.0`` (see the
    #                 long comment there). Kept for direct CLI users.
    #   * ``0.0``   — explicitly disable the cap. GEAK's ``mini.py:194-195``
    #                 writes ``config["agent"]["cost_limit"] = 0`` which is
    #                 honoured by every child agent spawned from that config;
    #                 this is the only way to defeat the sub-agent path that
    #                 silently falls back to ``AgentConfig.cost_limit = 3.0``.
    #   * ``> 0.0`` — finite per-attempt budget in USD (CI guardrail).
    if cost_limit is not None:
        cmd.extend(["--cost-limit", str(cost_limit)])
    return cmd


def run_via_ray(prompt_file: Path, output_dir: Path, kernel_path: str,
                cost_limit: float | None, num_gpus: int, timeout_s: int,
                kernel_repo: str = "", test_command: str = "") -> dict:
    import ray
    runtime_env = quiet_ray_init()

    @ray.remote(num_gpus=num_gpus)
    def _task(prompt_file_str: str, output_dir_str: str, kernel_path: str,
              cost_limit, timeout_s: int, kernel_repo: str, test_command: str) -> dict:
        # Self-contained: do NOT import kernel-agent modules here, Ray workers
        # don't share the driver's sys.path patches.
        import os as _os, shutil as _shutil, subprocess as _sp, time as _t
        import re as _re
        from pathlib import Path as _Path
        # GPU visibility on AMD/ROCm + Ray:
        #   * Ray sets ROCR_VISIBLE_DEVICES (NOT CUDA_VISIBLE_DEVICES) to a
        #     comma-list of *physical* GPU ids it allocated to this worker.
        #   * ROCR pre-filters at the lower layer, so HIP/CUDA APIs see those
        #     N physical GPUs as logical device 0..N-1.
        # To make GEAK's --gpu-ids and any nested torchrun rank that calls
        # `torch.cuda.set_device(local_rank)` work for BOTH single-GPU and
        # multi-GPU (e.g. set_device(1) when num_gpus=2), we must pass the
        # ROCR-filtered logical ids 0..N-1 to HIP/CUDA, NOT the raw physical
        # ids. Symptoms this fixes:
        #   * r17 GEAK single-GPU: "No HIP GPUs available" (we previously
        #     overwrote ROCR with the wrong value, double-filtering).
        #   * r20 GEAK multi-GPU: "invalid device ordinal" on rank 1 because
        #     HIP only saw device 0 when --gpu-ids was "0" (or unset).
        rocr_raw = _os.environ.get("ROCR_VISIBLE_DEVICES", "")
        if rocr_raw:
            n_visible = len([x for x in rocr_raw.split(",") if x.strip()])
            logical_ids = ",".join(str(i) for i in range(n_visible))
            _os.environ["HIP_VISIBLE_DEVICES"] = logical_ids
            _os.environ["CUDA_VISIBLE_DEVICES"] = logical_ids
            gpu_ids = logical_ids
        else:
            cuda_vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_vis:
                _os.environ["HIP_VISIBLE_DEVICES"] = cuda_vis
            gpu_ids = cuda_vis or "0"
        geak_bin = _shutil.which("geak") or _shutil.which("mini") or "geak"
        cmd = [geak_bin, "-t", prompt_file_str, "--yolo",
               "--output", output_dir_str, "--gpu-ids", gpu_ids]
        geak_config = _os.environ.get("GEAK_CONFIG", "").strip()
        if not geak_config:
            return {
                "returncode": 2,
                "stdout_tail": "",
                "stderr_tail": (
                    "GEAK_CONFIG is required; run inference_optimizer/scripts/install.sh "
                    "and source $KERNEL_AGENT_ENV "
                    "(default: $USER_DATA_PATH/runtime/kernel-agent.env.sh)"
                ),
                "stdout": "",
                "gpu_ids": gpu_ids,
                "elapsed_s": 0.0,
                "cmd": cmd,
            }
        geak_config_path = _Path(geak_config)
        if not geak_config_path.is_file():
            return {
                "returncode": 2,
                "stdout_tail": "",
                "stderr_tail": f"GEAK_CONFIG does not exist: {geak_config_path}",
                "stdout": "",
                "gpu_ids": gpu_ids,
                "elapsed_s": 0.0,
                "cmd": cmd,
            }
        geak_config_text = geak_config_path.read_text(encoding="utf-8", errors="replace")
        if not _re.search(r"(?m)^\s*model_class\s*:\s*litellm\s*$", geak_config_text):
            return {
                "returncode": 2,
                "stdout_tail": "",
                "stderr_tail": f"GEAK_CONFIG must set model.model_class: litellm: {geak_config_path}",
                "stdout": "",
                "gpu_ids": gpu_ids,
                "elapsed_s": 0.0,
                "cmd": cmd,
            }
        cmd.extend(["--config", str(geak_config_path)])
        if kernel_path:
            cmd.extend(["--kernel-path", kernel_path])
        if kernel_repo:
            cmd.extend(["--repo", kernel_repo])
        if test_command:
            cmd.extend(["--test-command", test_command])
        # Mirrors ``_build_cmd``: only emit ``--cost-limit`` when the
        # caller specified one. Hyperloom's default (0.0) means we
        # always pass the flag and disable GEAK's $3 sub-agent
        # fallback; see the cost_limit semantics comment in
        # ``_build_cmd`` above for the full rationale.
        if cost_limit is not None:
            cmd.extend(["--cost-limit", str(cost_limit)])
        started = _t.time()
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            return {
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-4000:],
                "stderr_tail": (proc.stderr or "")[-4000:],
                "stdout": proc.stdout or "",
                "gpu_ids": gpu_ids,
                "elapsed_s": round(_t.time() - started, 2),
                "cmd": cmd,
            }
        except _sp.TimeoutExpired as exc:
            return {
                "returncode": 124,
                "stdout_tail": "",
                "stderr_tail": f"TimeoutExpired after {timeout_s}s",
                "stdout": "",
                "gpu_ids": gpu_ids,
                "elapsed_s": round(_t.time() - started, 2),
                "cmd": cmd,
            }

    ref = _task.options(num_gpus=num_gpus, runtime_env=runtime_env).remote(
        str(prompt_file), str(output_dir), kernel_path, cost_limit, timeout_s,
        kernel_repo, test_command,
    )
    result = ray.get(ref)
    return result


def run_via_cli(prompt_file: Path, output_dir: Path, kernel_path: str,
                cost_limit: float | None, timeout_s: int,
                kernel_repo: str = "", test_command: str = "") -> dict:
    # Build a child env with ROCR→logical GPU mapping instead of
    # mutating os.environ (avoids leaking GPU vars to later steps).
    child_env = os.environ.copy()
    rocr_raw = child_env.get("ROCR_VISIBLE_DEVICES", "")
    if rocr_raw:
        n_visible = len([x for x in rocr_raw.split(",") if x.strip()])
        logical_ids = ",".join(str(i) for i in range(n_visible))
        child_env["HIP_VISIBLE_DEVICES"] = logical_ids
        child_env["CUDA_VISIBLE_DEVICES"] = logical_ids
        gpu_ids = logical_ids
    else:
        cuda_vis = child_env.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_vis and not child_env.get("HIP_VISIBLE_DEVICES"):
            child_env["HIP_VISIBLE_DEVICES"] = cuda_vis
        gpu_ids = cuda_vis or "0"
    started = time.time()
    try:
        cmd = _build_cmd(prompt_file, output_dir, kernel_path, gpu_ids, cost_limit,
                         kernel_repo=kernel_repo, test_command=test_command)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=child_env)
        return {
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "gpu_ids": gpu_ids,
            "elapsed_s": round(time.time() - started, 2),
            "cmd": cmd,
        }
    except ValueError as exc:
        return {
            "returncode": 2,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "gpu_ids": gpu_ids,
            "elapsed_s": round(time.time() - started, 2),
            "cmd": [],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, (str, bytes)) else "",
            "stderr_tail": f"TimeoutExpired after {timeout_s}s",
            "gpu_ids": gpu_ids,
            "elapsed_s": round(time.time() - started, 2),
            "cmd": cmd,
        }


def submit(prompt_file: Path, output_dir: Path, kernel_path: str = "",
           cost_limit: float | None = None, timeout_s: int = 1800,
           num_gpus: int = 1, prefer_ray: bool = True,
           kernel_repo: str = "", test_command: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefer_ray:
        try:
            import ray  # noqa: F401
            # Don't burn 30 s of ray.init retries on a wedged cluster. If
            # `ray status` fails, ``ensure_ray_cluster`` will start a fresh
            # head node here (safe no-op when the cluster is already healthy).
            ensure_ray_cluster(num_gpus=num_gpus,
                               log_path=output_dir / "ray_lifecycle.log")
            return run_via_ray(prompt_file, output_dir, kernel_path, cost_limit,
                               num_gpus, timeout_s, kernel_repo=kernel_repo,
                               test_command=test_command)
        except Exception as exc:
            return {
                "returncode": 1,
                "stdout_tail": "",
                "stderr_tail": (
                    f"ray submission failed: {type(exc).__name__}: {exc}\n"
                    f"hint: check `ray status` in container; raylet zombie symptom is "
                    f"`global_state_accessor.cc:500 ... retrying ... 'ray start' on this node'`."
                ),
                "gpu_ids": "",
                "elapsed_s": 0.0,
                "cmd": [],
            }
    return run_via_cli(prompt_file, output_dir, kernel_path, cost_limit, timeout_s,
                       kernel_repo=kernel_repo, test_command=test_command)


def main() -> int:
    parser = argparse.ArgumentParser(description="kernel-agent self-contained GEAK submitter")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kernel-path", default="")
    parser.add_argument("--kernel-repo", default="")
    parser.add_argument("--test-command", default="")
    parser.add_argument("--cost-limit", type=float, default=None)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--prefer-cli", action="store_true")
    args = parser.parse_args()
    result = submit(
        prompt_file=Path(args.prompt_file),
        output_dir=Path(args.output_dir),
        kernel_path=args.kernel_path,
        cost_limit=args.cost_limit,
        timeout_s=args.timeout_s,
        num_gpus=args.num_gpus,
        prefer_ray=not args.prefer_cli,
        kernel_repo=args.kernel_repo,
        test_command=args.test_command,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
