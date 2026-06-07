#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""OOB (claude/codex/cursor) submission via Ray (preferred) or direct CLI fallback.

Self-contained: does not depend on inference-optimization scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ray_runtime import ensure_ray_cluster, quiet_ray_init, safe_runtime_env


def _safety_system_prompt(kernel_repo: str, budget_minutes: float = 30.0,
                          num_gpus: int = 1) -> str:
    forbidden = (kernel_repo
                 or "/sgl-workspace, /opt, /usr, /etc, "
                    "/sgl-workspace/aiter, /sgl-workspace/sglang")
    soft_deadline = max(1, int(budget_minutes * 0.85))
    gpu_hint = (
        f"You have {num_gpus} GPU(s) available. "
        + ("For multi-GPU benchmarks use `torchrun --nproc_per_node="
           f"{num_gpus} ...` or set HIP_VISIBLE_DEVICES to the assigned ids "
           "(see the `gpu_ids` line in the prompt body).\n"
           if num_gpus > 1 else "Single-GPU sandbox; no torchrun needed.\n")
    )
    base = (
        "You are a kernel optimization agent running inside a sandboxed workspace. "
        "Hard rules you MUST follow:\n"
        "1. ALWAYS use ABSOLUTE PATHS for any file you write. Your first Bash "
        "step MUST be `pwd` to capture your absolute working directory; then "
        "save outputs ONLY under `<pwd>/optimized_versions/v<N>_<desc>.<ext>` "
        "and `<pwd>/optimization_report.md`. NEVER use `./`, `~/`, `/home/...` "
        "or any other path — agents historically default to `/home/user/` and "
        "the driver loses your output.\n"
        "   The optimized file MUST be a COMPLETE source file with the SAME "
        "extension as the input kernel. Do not write markdown, a patch/diff, "
        "or a code excerpt as the final optimized artifact.\n"
        "   It must be an in-place replacement for the input file: preserve "
        "the original namespace, exported host entry functions, registration "
        "macros, includes, and public signatures. Do NOT create a standalone "
        "`torch.utils.cpp_extension`/`PYBIND11_MODULE` file unless the original "
        "file already used that pattern.\n"
        f"2. NEVER modify files outside your working directory. Specifically, "
        f"do NOT Edit/Write/Bash-write files under: {forbidden}. You MAY "
        "freely READ any file there for reference.\n"
        f"3. TIME BUDGET: hard wall-clock is {budget_minutes:.0f} minutes. "
        f"At minute {soft_deadline} you MUST stop iterating and produce a "
        "final `optimization_report.md` summarizing your best version + "
        "measured numbers (or N/A). Don't keep optimizing past that point.\n"
        f"4. {gpu_hint}"
        "5. Do not invent benchmark numbers. Either compile+run a real "
        "benchmark, or explicitly mark latency/speedup as N/A.\n"
    )
    return base


def _build_cmd(agent: str, prompt_file: Path, output_dir: Path,
               source_file: str, max_turns: int, timeout_s: int,
               extra_files: list[str] | None = None,
               kernel_repo: str = "", num_gpus: int = 1) -> list[str]:
    if not shutil.which("oob"):
        raise FileNotFoundError("oob CLI not in PATH; run install.sh")
    cmd = [
        "oob", "run", "-a", agent,
        "--prompt-file", str(prompt_file),
        "--max-turns", str(max_turns),
        "--timeout", str(timeout_s),
        "--system-prompt", _safety_system_prompt(
            kernel_repo, budget_minutes=timeout_s / 60.0, num_gpus=num_gpus),
        "--json", "--no-live",
        "-o", str(output_dir),
    ]
    if source_file:
        cmd.extend(["-f", source_file])
    for ef in extra_files or []:
        if ef:
            cmd.extend(["-f", ef])
    return cmd


def _parse_oob_init(stdout: str) -> dict[str, str]:
    """Extract cwd / session_id / thread_id from oob run --json output.

    `oob run --json` prints a final summary block of the form
        {"task_id": "...", "status": "completed",
         "workspace": "...tasks/cli/<uuid>/workspace",
         "log_file": "...", "usage": ...}
    Parse the trailing JSON object first; fall back to scanning ndjson
    `system/init` events if the trailing block is missing (e.g. when the
    run was killed mid-stream).
    """
    info = {"cli_workspace": "", "session_id": "", "thread_id": ""}
    if not stdout:
        return info

    # 1) Try the trailing oob-run JSON summary.
    end = stdout.rfind("}")
    if end != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(end, -1, -1):
            ch = stdout[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        evt = json.loads(stdout[i:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(evt, dict) and evt.get("workspace"):
                        info["cli_workspace"] = str(evt.get("workspace") or "")
                        info["session_id"] = str(evt.get("task_id") or "")
                        return info
                    break

    # 2) Fall back to ndjson init line (raw stream from claude-code-sdk).
    for line in stdout.splitlines()[:200]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "system" and evt.get("subtype") == "init":
            info["cli_workspace"] = str(evt.get("cwd") or "")
            info["session_id"] = str(evt.get("session_id") or "")
            return info
        if evt.get("type") == "thread.started":
            info["thread_id"] = str(evt.get("thread_id") or "")
    return info


def run_via_ray(agent: str, prompt_file: Path, output_dir: Path, source_file: str,
                max_turns: int, num_gpus: int, timeout_s: int,
                extra_files: list[str] | None = None,
                kernel_repo: str = "") -> dict:
    import ray
    runtime_env = quiet_ray_init()
    system_prompt_text = _safety_system_prompt(
        kernel_repo, budget_minutes=timeout_s / 60.0, num_gpus=num_gpus)

    @ray.remote(num_gpus=num_gpus)
    def _task(agent: str, prompt_file_str: str, output_dir_str: str,
              source_file: str, max_turns: int, timeout_s: int,
              extra_files: list[str], system_prompt: str) -> dict:
        # Self-contained: workers don't share driver sys.path.
        import os as _os, shutil as _shutil, subprocess as _sp, time as _t
        if not _shutil.which("oob"):
            return {
                "returncode": 127, "stdout_tail": "", "stdout": "",
                "stderr_tail": "oob CLI not in PATH on Ray worker",
                "gpu_ids": "", "elapsed_s": 0.0, "cmd": [],
            }
        gpu_ids = (_os.environ.get("ROCR_VISIBLE_DEVICES")
                   or _os.environ.get("HIP_VISIBLE_DEVICES")
                   or _os.environ.get("CUDA_VISIBLE_DEVICES")
                   or "0")
        cmd = ["oob", "run", "-a", agent,
               "--prompt-file", prompt_file_str,
               "--max-turns", str(max_turns),
               "--timeout", str(timeout_s),
               "--system-prompt", system_prompt,
               "--json", "--no-live", "-o", output_dir_str]
        if source_file:
            cmd.extend(["-f", source_file])
        for ef in extra_files or []:
            if ef:
                cmd.extend(["-f", ef])
        started = _t.time()
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=timeout_s + 60)
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
            # Capture whatever oob already wrote so the driver can scan the
            # workspace for partial outputs (optimized_versions/ etc.).
            partial_out = ""
            try:
                partial_out = (exc.stdout or "").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            except Exception:
                partial_out = ""
            return {
                "returncode": 124,
                "stdout_tail": partial_out[-4000:],
                "stderr_tail": f"TimeoutExpired after {timeout_s}s",
                "stdout": partial_out,
                "gpu_ids": gpu_ids,
                "elapsed_s": round(_t.time() - started, 2),
                "cmd": cmd,
            }

    ref = _task.options(num_gpus=num_gpus, runtime_env=runtime_env).remote(
        agent, str(prompt_file), str(output_dir), source_file, max_turns, timeout_s,
        list(extra_files or []), system_prompt_text,
    )
    result = ray.get(ref)
    # Attribution: parse the oob ndjson init line so the driver knows exactly
    # which tasks/cli/<uuid>/workspace this attempt produced (no mtime races).
    result.update(_parse_oob_init(result.get("stdout", "")))
    return result


def run_via_cli(agent: str, prompt_file: Path, output_dir: Path, source_file: str,
                max_turns: int, timeout_s: int,
                extra_files: list[str] | None = None,
                kernel_repo: str = "", num_gpus: int = 1) -> dict:
    cmd = _build_cmd(agent, prompt_file, output_dir, source_file, max_turns, timeout_s,
                     extra_files=extra_files, kernel_repo=kernel_repo, num_gpus=num_gpus)
    gpu_ids = os.environ.get("ROCR_VISIBLE_DEVICES") \
        or os.environ.get("HIP_VISIBLE_DEVICES") \
        or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 60)
        result = {
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "stdout": proc.stdout or "",
            "gpu_ids": gpu_ids,
            "elapsed_s": round(time.time() - started, 2),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        partial_out = ""
        try:
            partial_out = (exc.stdout or "").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        except Exception:
            partial_out = ""
        result = {
            "returncode": 124,
            "stdout_tail": partial_out[-4000:],
            "stderr_tail": f"TimeoutExpired after {timeout_s}s",
            "stdout": partial_out,
            "gpu_ids": gpu_ids,
            "elapsed_s": round(time.time() - started, 2),
            "cmd": cmd,
        }
    result.update(_parse_oob_init(result.get("stdout", "")))
    return result


def submit(agent: str, prompt_file: Path, output_dir: Path, source_file: str = "",
           max_turns: int = 100, timeout_s: int = 1800, num_gpus: int = 1,
           prefer_ray: bool = True, extra_files: list[str] | None = None,
           kernel_repo: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Placement precedence: ssh (Dynamo multi-node, env-gated) > ray > cli.
    # ssh_placement_active() is True ONLY when the orchestrator set
    # KERNEL_AGENT_GPU_PLACEMENT=ssh (Dynamo backend); ray/cli paths below are
    # byte-for-byte unchanged otherwise.
    try:
        from ssh_runtime import ssh_placement_active, run_oob_over_ssh
    except Exception:  # noqa: BLE001 — ssh_runtime optional; never block ray/cli
        ssh_placement_active = lambda: False  # noqa: E731
    if ssh_placement_active():
        return run_oob_over_ssh(
            agent, prompt_file, output_dir, source_file, max_turns, num_gpus,
            timeout_s, extra_files=extra_files, kernel_repo=kernel_repo,
            system_prompt_text=_safety_system_prompt(
                kernel_repo, budget_minutes=timeout_s / 60.0, num_gpus=num_gpus),
        )
    if prefer_ray:
        try:
            import ray  # noqa: F401
            # Don't burn 30 s of ray.init retries on a wedged cluster. If
            # `ray status` fails, ``ensure_ray_cluster`` will start a fresh
            # head node here (safe no-op when the cluster is already healthy).
            ensure_ray_cluster(num_gpus=num_gpus,
                               log_path=output_dir / "ray_lifecycle.log")
            return run_via_ray(agent, prompt_file, output_dir, source_file,
                               max_turns, num_gpus, timeout_s,
                               extra_files=extra_files, kernel_repo=kernel_repo)
        except Exception as exc:
            return {
                "returncode": 1,
                "stdout_tail": "",
                "stderr_tail": (
                    f"ray submission failed: {type(exc).__name__}: {exc}\n"
                    f"hint: check `ray status` in container; raylet zombie symptom is "
                    f"`global_state_accessor.cc:500 ... retrying ... 'ray start' on this node'`."
                ),
                "stdout": "",
                "gpu_ids": "",
                "elapsed_s": 0.0,
                "cmd": [],
            }
    return run_via_cli(agent, prompt_file, output_dir, source_file, max_turns, timeout_s,
                       extra_files=extra_files, kernel_repo=kernel_repo, num_gpus=num_gpus)


def main() -> int:
    parser = argparse.ArgumentParser(description="kernel-agent self-contained OOB submitter")
    parser.add_argument("--agent", required=True, choices=["claude", "codex", "cursor"])
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-file", default="")
    parser.add_argument("--extra-file", action="append", default=[],
                        help="Additional file copied into the OOB workspace (repeatable)")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--prefer-cli", action="store_true")
    args = parser.parse_args()
    result = submit(
        agent=args.agent,
        prompt_file=Path(args.prompt_file),
        output_dir=Path(args.output_dir),
        source_file=args.source_file,
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
        num_gpus=args.num_gpus,
        prefer_ray=not args.prefer_cli,
        extra_files=args.extra_file,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
