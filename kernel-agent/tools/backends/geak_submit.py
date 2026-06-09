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
import signal
import subprocess
import time
from pathlib import Path

from ray_runtime import (
    ensure_ray_cluster,
    bootstrap_ray_cluster,
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


def _apply_geak_child_env(env: dict[str, str]) -> None:
    """Ensure GEAK subprocesses resolve subagents/ and repo metadata."""
    hyperloom_root = env.get("HYPERLOOM_ROOT", "").strip()
    if hyperloom_root and not env.get("GEAK_ROOT", "").strip():
        geak_root = Path(hyperloom_root) / "geak"
        if geak_root.is_dir():
            env["GEAK_ROOT"] = str(geak_root)


def _num_parallel_from_env(env: dict[str, str] | None = None) -> int | None:
    raw = (env or os.environ).get("MAX_PARALLEL_WORKERS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _append_num_parallel(cmd: list[str], env: dict[str, str] | None = None) -> None:
    n = _num_parallel_from_env(env)
    if n is not None:
        cmd.extend(["--num-parallel", str(n)])


def _write_geak_subprocess_log(
    output_dir: Path | str,
    *,
    stdout: str,
    stderr: str,
    returncode: int,
    cmd: list[str],
) -> Path:
    log_path = Path(output_dir) / "geak_subprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"returncode={returncode}",
                f"cmd={' '.join(str(c) for c in cmd)}",
                "",
                "=== stdout ===",
                stdout or "",
                "",
                "=== stderr ===",
                stderr or "",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def _geak_cmd_result(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    elapsed_s: float,
    cmd: list[str],
    output_dir: Path | str = "",
    gpu_ids: str = "",
) -> dict:
    stderr_text = stderr or ""
    result = {
        "returncode": returncode,
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": stderr_text[-16000:],
        "stderr": stderr_text,
        "stdout": stdout or "",
        "elapsed_s": elapsed_s,
        "cmd": cmd,
    }
    if gpu_ids:
        result["gpu_ids"] = gpu_ids
    if output_dir:
        log_path = _write_geak_subprocess_log(
            output_dir,
            stdout=stdout or "",
            stderr=stderr_text,
            returncode=returncode,
            cmd=cmd,
        )
        result["geak_subprocess_log"] = str(log_path)
    return result


def _reap_zombie_geak(output_dir: Path | str = "") -> None:
    """Reap idle geak/mini parents left after mini-swe completion."""
    needle = str(output_dir).strip() if output_dir else ""
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    for line in (proc.stdout or "").splitlines():
        if not any(tok in line for tok in (" geak ", " mini ", "/geak ", "/mini ")):
            continue
        if needle and needle not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid <= 1 or pid == os.getpid():
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except OSError:
                continue
            time.sleep(0.2)


def _kill_orphan_kernel_profile(output_dir: Path | str = "") -> None:
    """Reap orphaned ``kernel-profile`` children after GEAK timeout/cleanup."""
    needle = str(output_dir).strip() if output_dir else ""
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    for line in (proc.stdout or "").splitlines():
        if "kernel-profile" not in line:
            continue
        if needle and needle not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid <= 1:
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except OSError:
                continue
            time.sleep(0.2)


def _run_geak_cmd(
    cmd: list[str],
    *,
    timeout_s: int,
    env: dict[str, str],
    output_dir: Path | str = "",
) -> dict:
    """Run GEAK in its own process group so timeouts can kill the whole tree."""
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        _reap_zombie_geak(next((str(a) for a in cmd if str(a).startswith("/")), ""))
        return _geak_cmd_result(
            returncode=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            elapsed_s=round(time.time() - started, 2),
            cmd=cmd,
            output_dir=output_dir,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        time.sleep(1.0)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        _kill_orphan_kernel_profile(
            next((str(a) for a in cmd if str(a).startswith("/")), ""),
        )
        _reap_zombie_geak(next((str(a) for a in cmd if str(a).startswith("/")), ""))
        return _geak_cmd_result(
            returncode=124,
            stdout="",
            stderr=f"TimeoutExpired after {timeout_s}s",
            elapsed_s=round(time.time() - started, 2),
            cmd=cmd,
            output_dir=output_dir,
        )


def _build_cmd(prompt_file: Path, output_dir: Path, kernel_path: str, gpu_ids: str,
               cost_limit: float | None, kernel_repo: str = "",
               test_command: str = "", total_budget_s: int | None = None) -> list[str]:
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
    if total_budget_s is not None and total_budget_s > 0:
        cmd.extend(["--total-budget-s", str(int(total_budget_s))])
    _append_num_parallel(cmd)
    return cmd


def run_via_ray(prompt_file: Path, output_dir: Path, kernel_path: str,
                cost_limit: float | None, num_gpus: int, timeout_s: int,
                kernel_repo: str = "", test_command: str = "",
                total_budget_s: int | None = None) -> dict:
    import ray
    runtime_env = quiet_ray_init()

    @ray.remote(num_gpus=num_gpus)
    def _task(prompt_file_str: str, output_dir_str: str, kernel_path: str,
              cost_limit, timeout_s: int, kernel_repo: str, test_command: str,
              total_budget_s: int | None) -> dict:
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
        _apply_geak_child_env(_os.environ)
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
        if total_budget_s is not None and total_budget_s > 0:
            cmd.extend(["--total-budget-s", str(int(total_budget_s))])
        _append_num_parallel(cmd, _os.environ)
        started = _t.time()
        import signal as _signal

        def _kill_orphans() -> None:
            try:
                ps = _sp.run(
                    ["ps", "-eo", "pid,args"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, _sp.TimeoutExpired):
                return
            for line in (ps.stdout or "").splitlines():
                if "kernel-profile" not in line or output_dir_str not in line:
                    continue
                parts = line.strip().split(None, 1)
                if not parts:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                for sig in (_signal.SIGTERM, _signal.SIGKILL):
                    try:
                        _os.kill(pid, sig)
                    except ProcessLookupError:
                        break
                    except OSError:
                        continue
                    _t.sleep(0.2)

        proc = _sp.Popen(
            cmd,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            text=True,
            env=_os.environ,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            _kill_orphans()
            return _geak_cmd_result(
                returncode=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                elapsed_s=round(_t.time() - started, 2),
                cmd=cmd,
                output_dir=output_dir_str,
                gpu_ids=gpu_ids,
            )
        except _sp.TimeoutExpired:
            try:
                _os.killpg(proc.pid, _signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
            _t.sleep(1.0)
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            _kill_orphans()
            return _geak_cmd_result(
                returncode=124,
                stdout="",
                stderr=f"TimeoutExpired after {timeout_s}s",
                elapsed_s=round(_t.time() - started, 2),
                cmd=cmd,
                output_dir=output_dir_str,
                gpu_ids=gpu_ids,
            )

    _total_budget = total_budget_s if total_budget_s is not None else timeout_s
    ref = _task.options(num_gpus=num_gpus, runtime_env=runtime_env).remote(
        str(prompt_file), str(output_dir), kernel_path, cost_limit, timeout_s,
        kernel_repo, test_command, _total_budget,
    )
    result = ray.get(ref)
    return result


def run_via_cli(prompt_file: Path, output_dir: Path, kernel_path: str,
                cost_limit: float | None, timeout_s: int,
                kernel_repo: str = "", test_command: str = "",
                total_budget_s: int | None = None) -> dict:
    # Build a child env with ROCR→logical GPU mapping instead of
    # mutating os.environ (avoids leaking GPU vars to later steps).
    child_env = os.environ.copy()
    _apply_geak_child_env(child_env)
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
        _total_budget = total_budget_s if total_budget_s is not None else timeout_s
        cmd = _build_cmd(prompt_file, output_dir, kernel_path, gpu_ids, cost_limit,
                         kernel_repo=kernel_repo, test_command=test_command,
                         total_budget_s=_total_budget)
        run_out = _run_geak_cmd(
            cmd, timeout_s=timeout_s, env=child_env, output_dir=output_dir,
        )
        if run_out["returncode"] == 124:
            _kill_orphan_kernel_profile(output_dir)
        return {
            "returncode": run_out["returncode"],
            "stdout_tail": run_out["stdout_tail"],
            "stderr_tail": run_out["stderr_tail"],
            "stderr": run_out.get("stderr", ""),
            "geak_subprocess_log": run_out.get("geak_subprocess_log", ""),
            "gpu_ids": gpu_ids,
            "elapsed_s": run_out["elapsed_s"],
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


def submit(prompt_file: Path, output_dir: Path, kernel_path: str = "",
           cost_limit: float | None = None, timeout_s: int = 1800,
           num_gpus: int = 1, prefer_ray: bool = True,
           kernel_repo: str = "", test_command: str = "",
           total_budget_s: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefer_ray:
        try:
            import ray  # noqa: F401
            # Bootstrap Ray before GEAK dispatch: stale raylets / low FD limits
            # surface as LocalRayletDiedError and must not permanently retire
            # hot kernels (zero-touch run9).
            bootstrap_ray_cluster(
                num_gpus=num_gpus,
                log_path=output_dir / "ray_lifecycle.log",
                force_restart=False,
            )
            return run_via_ray(prompt_file, output_dir, kernel_path, cost_limit,
                               num_gpus, timeout_s, kernel_repo=kernel_repo,
                               test_command=test_command,
                               total_budget_s=total_budget_s)
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
                       kernel_repo=kernel_repo, test_command=test_command,
                       total_budget_s=total_budget_s)


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
