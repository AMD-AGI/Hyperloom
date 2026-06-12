#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    ensure_ray_cluster,
    isolated_compile_cache_env,
    quiet_ray_init,
)


def _find_geak_bin() -> str:
    """Locate the GEAK CLI executable on ``PATH``.

    Returns:
        str: The absolute path to the first of ``geak`` / ``mini`` /
            ``geak-gaagent`` found on ``PATH``, falling back to the bare
            name ``"geak"`` when none resolve.
    """
    for name in ("geak", "mini", "geak-gaagent"):
        path = shutil.which(name)
        if path:
            return path
    return "geak"


def _resolve_geak_config() -> Path:
    """Resolve and validate the GEAK config path from ``$GEAK_CONFIG``.

    Returns:
        Path: The validated config file path.

    Raises:
        ValueError: If ``$GEAK_CONFIG`` is unset, points at a missing
            file, or does not set ``model.model_class: litellm``.
    """
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
    """Assemble the GEAK CLI argument vector.

    Args:
        prompt_file (Path): Path to the task prompt file (``-t``).
        output_dir (Path): Directory for GEAK output (``--output``).
        kernel_path (str): Optional kernel file path (``--kernel-path``);
            omitted when empty.
        gpu_ids (str): Comma-separated logical GPU ids (``--gpu-ids``).
        cost_limit (float | None): Per-attempt USD budget. ``None`` omits
            the flag (GEAK uses its config value); ``0.0`` disables the
            cap; ``> 0.0`` sets a finite cap.
        kernel_repo (str): Optional repo path (``--repo``); omitted when
            empty.
        test_command (str): Optional test command (``--test-command``);
            omitted when empty.

    Returns:
        list[str]: The full command vector ready for ``subprocess.run``.
    """
    cmd = [_find_geak_bin(), "-t", str(prompt_file), "--yolo",
           "--output", str(output_dir), "--gpu-ids", gpu_ids]
    cmd.extend(["--config", str(_resolve_geak_config())])
    if kernel_path:
        cmd.extend(["--kernel-path", kernel_path])
    if kernel_repo:
        cmd.extend(["--repo", kernel_repo])
    if test_command:
        cmd.extend(["--test-command", test_command])
    # cost_limit: None omits the flag; 0.0 disables GEAK's sub-agent fallback; >0.0 is a USD budget.
    if cost_limit is not None:
        cmd.extend(["--cost-limit", str(cost_limit)])
    return cmd


def run_via_ray(prompt_file: Path, output_dir: Path, kernel_path: str,
                cost_limit: float | None, num_gpus: int, timeout_s: int,
                kernel_repo: str = "", test_command: str = "") -> dict:
    """Run a GEAK submission inside a Ray GPU task.

    Initializes Ray (quietly), then dispatches a ``num_gpus``-pinned
    remote task that maps ROCR-visible physical GPUs to logical ids and
    invokes the GEAK CLI.

    Args:
        prompt_file (Path): Task prompt file passed to GEAK.
        output_dir (Path): Directory for GEAK output.
        kernel_path (str): Optional kernel file path.
        cost_limit (float | None): Per-attempt USD budget; see
            :func:`_build_cmd` for the ``None`` / ``0.0`` / ``> 0.0``
            semantics.
        num_gpus (int): GPUs to reserve for the remote task.
        timeout_s (int): Subprocess timeout in seconds.
        kernel_repo (str): Optional repo path.
        test_command (str): Optional test command.

    Returns:
        dict: The result mapping from the remote task with keys such as
            ``returncode``, ``stdout_tail``, ``stderr_tail``,
            ``gpu_ids``, ``elapsed_s``, and ``cmd``.
    """
    import ray
    runtime_env = quiet_ray_init(
        num_gpus=num_gpus, log_path=output_dir / "ray_lifecycle.log")

    @ray.remote(num_gpus=num_gpus)
    def _task(prompt_file_str: str, output_dir_str: str, kernel_path: str,
              cost_limit, timeout_s: int, kernel_repo: str, test_command: str) -> dict:
        """Run one GEAK CLI attempt inside a Ray worker.

        Self-contained because Ray workers do not inherit the driver's
        ``sys.path``; all imports and GPU-visibility remapping happen here.

        Args:
            prompt_file_str: Path to the prompt file.
            output_dir_str: Output directory for this attempt.
            kernel_path: Path to the kernel under optimization.
            cost_limit: Optional cost ceiling for the GEAK run.
            timeout_s: Per-attempt timeout in seconds.
            kernel_repo: Kernel repository identifier.
            test_command: Command used to validate the kernel.

        Returns:
            A result dict with ``returncode``, ``stdout_tail``,
            ``stderr_tail``, ``gpu_ids``, ``elapsed_s``, and ``cmd``.
        """
        # Self-contained: Ray workers lack the driver's sys.path.
        import os as _os, shutil as _shutil, subprocess as _sp, time as _t
        import re as _re
        from pathlib import Path as _Path
        # r17/r20: map ROCR physical ids to logical 0..N-1 for HIP/CUDA so set_device works.
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
        # Per-attempt compile caches so a co-running OOB ladder can't clobber
        # this run's aiter/triton/inductor artifacts (see isolated_compile_cache_env).
        for _var, _sub in (("TRITON_CACHE_DIR", "triton"),
                           ("AITER_ROOT_DIR", "aiter"),
                           ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
            _cdir = _os.path.join(output_dir_str, ".cache", _sub)
            _os.makedirs(_cdir, exist_ok=True)
            _os.environ[_var] = _cdir
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
        # Mirrors ``_build_cmd``: only emit ``--cost-limit`` when specified.
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
        except _sp.TimeoutExpired:
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
    """Run one GEAK CLI attempt in a subprocess (non-Ray path).

    Builds a child environment with ROCR→logical GPU remapping and per-attempt
    compile caches, then runs the GEAK CLI under a timeout.

    Args:
        prompt_file: Path to the prompt file.
        output_dir: Output directory for this attempt.
        kernel_path: Path to the kernel under optimization.
        cost_limit: Optional cost ceiling for the GEAK run.
        timeout_s: Per-attempt timeout in seconds.
        kernel_repo: Kernel repository identifier.
        test_command: Command used to validate the kernel.

    Returns:
        A result dict with ``returncode``, ``stdout_tail``, ``stderr_tail``,
        ``gpu_ids``, ``elapsed_s``, and ``cmd``. Timeouts return code ``124``
        and input errors return code ``2``.
    """
    # Child env with ROCR→logical GPU mapping; avoids leaking GPU vars to later steps.
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
    # Per-attempt compile caches (see isolated_compile_cache_env).
    child_env = isolated_compile_cache_env(output_dir, base_env=child_env)
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
    """Submit a GEAK run, preferring Ray with a CLI fallback.

    Ensures the output directory exists, then (when ``prefer_ray``)
    starts/attaches to a Ray cluster and runs via :func:`run_via_ray`,
    falling back to a structured error dict on any Ray failure. When Ray
    is not preferred, runs via :func:`run_via_cli`.

    Args:
        prompt_file (Path): Task prompt file passed to GEAK.
        output_dir (Path): Directory for GEAK output; created if absent.
        kernel_path (str): Optional kernel file path.
        cost_limit (float | None): Per-attempt USD budget; see
            :func:`_build_cmd`.
        timeout_s (int): Subprocess timeout in seconds.
        num_gpus (int): GPUs to reserve for the Ray task.
        prefer_ray (bool): When True, try Ray first; otherwise use the
            CLI path.
        kernel_repo (str): Optional repo path.
        test_command (str): Optional test command.

    Returns:
        dict: The result mapping from the chosen submission path (see
            :func:`run_via_ray` / :func:`run_via_cli`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefer_ray:
        try:
            import ray  # noqa: F401
            # ensure_ray_cluster starts a fresh head if `ray status` fails (no-op when healthy).
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
    """CLI entry point: parse args, submit, and print the JSON result.

    Returns:
        int: 0 when the submission's ``returncode`` is 0, otherwise 1.
    """
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
