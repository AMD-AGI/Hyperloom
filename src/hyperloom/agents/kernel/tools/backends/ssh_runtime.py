"""SSH placement for GEAK on the Dynamo multi-node backend.

Counterpart to ``ray_runtime.py``. When the optimizer runs multi-node on the
**Dynamo** backend there is no Ray cluster to submit a GPU task to, so GEAK is
executed on a GPU-bearing Dynamo pod over SSH instead — the SSH analogue of
``geak_submit.run_via_ray``.

ISOLATION CONTRACT: this path is *only* taken when the orchestrator explicitly
sets ``KERNEL_AGENT_GPU_PLACEMENT=ssh`` (which it does solely for the
``--mn-backend dynamo`` multi-node case). Single-node and RayJob runs never set
it, so ``run_via_ray`` / ``run_via_cli`` behave exactly as before — see
``ssh_placement_active``.

Design:
* **No file transfer**: ``prompt_file`` / ``output_dir`` / ``GEAK_CONFIG`` all
  live under ``$USER_DATA_PATH`` (a cluster-shared NFS/wekafs mount the pod
  also mounts), so the pod reads/writes the SAME absolute paths — nothing is
  copied. (rsync is intentionally avoided: the sandbox has none.)
* **Creds never hit argv or disk**: the GEAK env (``GEAK_*`` / ``*_API_KEY`` /
  ``*_BASE_URL``) is piped to the pod over the SSH **stdin** as a bash
  prologue, not passed on the command line and not written to a shared file.
* **Blocking SSH**: one long-lived SSH call per GEAK attempt (timeout =
  GEAK budget + buffer); GEAK writes its artifacts to the shared ``output_dir``
  which the sandbox reads back locally afterwards.
* **Pod GEAK precondition**: the GPU pod must have the ``geak`` CLI on PATH
  (image-baked or installed once over SSH). The pod-side runner reports a
  clear ``returncode=2`` error if it is missing.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

# Reuse the exact env allowlist + credential-alias logic the Ray path uses, so
# GEAK sees the same GEAK_* / creds / base-url config regardless of placement.
from ray_runtime import SAFE_ENV_KEYS, safe_runtime_env

DEFAULT_SSH_PORT = 2222

_SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "LogLevel=ERROR",
]


def ssh_placement_active() -> bool:
    """True only when the orchestrator opted this run into SSH GPU placement.

    The single switch that isolates the Dynamo SSH path from every other run.

    Returns:
        ``True`` when ``KERNEL_AGENT_GPU_PLACEMENT`` is set to ``ssh``.
    """
    return os.environ.get("KERNEL_AGENT_GPU_PLACEMENT", "").strip().lower() == "ssh"


def ssh_target() -> tuple[str, int, str]:
    """Resolve (host, port, key_path) for the GPU pod from MN_SSH_* env.

    Returns:
        A ``(host, port, key_path)`` tuple read from ``MN_SSH_HOST`` /
        ``MN_SSH_PORT`` / ``MN_SSH_KEY``.
    """
    host = os.environ.get("MN_SSH_HOST", "").strip()
    port = int(os.environ.get("MN_SSH_PORT", str(DEFAULT_SSH_PORT)) or DEFAULT_SSH_PORT)
    key = os.environ.get("MN_SSH_KEY", "").strip()
    return host, port, key


def _env_prologue() -> str:
    """Bash ``export K=V`` lines for the GEAK env, sent over SSH stdin.

    Values are shell-quoted; nothing reaches argv. Mirrors
    ``safe_runtime_env``'s allowlist + credential aliasing.

    Returns:
        Newline-joined ``export K=V`` lines for the allowlisted GEAK env,
        including a ``PATH`` export leading with ``/opt/venv/bin``.
    """
    env = safe_runtime_env().get("env_vars", {})
    lines = [
        f"export {k}={shlex.quote(str(v))}"
        for k, v in env.items()
        if k in SAFE_ENV_KEYS
        or k
        in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OOB_API_KEY",
            "GEAK_API_KEY",
            "LLM_API_KEY",
            "AMD_LLM_API_KEY",
            "LLM_GATEWAY_KEY",
            "ANTHROPIC_BASE_URL",
            "OOB_BASE_URL",
            "GEAK_BASE_URL",
            "LLM_API_BASE",
        )
    ]
    # Ensure /opt/venv/bin (framework venv with geak/torch) leads PATH.
    lines.append('export PATH="/opt/venv/bin:${PATH:-}"')
    return "\n".join(lines)


# Self-contained pod-side runner. Mirrors geak_submit.run_via_ray._task: maps
# GPUs to logical 0..N-1, builds the geak CLI, runs it, prints a JSON result.
# Stdlib only — the Dynamo pod has no kernel-agent checkout.
_POD_RUNNER = r"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prompt", required=True)
p.add_argument("--output", required=True)
p.add_argument("--kernel-path", default="")
p.add_argument("--kernel-repo", default="")
p.add_argument("--test-command", default="")
p.add_argument("--cost-limit", default="")
p.add_argument("--num-gpus", type=int, default=1)
p.add_argument("--timeout-s", type=int, default=1800)
a = p.parse_args()

ng = max(1, a.num_gpus)
ids = ",".join(str(i) for i in range(ng))
os.environ["HIP_VISIBLE_DEVICES"] = ids
os.environ["CUDA_VISIBLE_DEVICES"] = ids
os.environ["ROCR_VISIBLE_DEVICES"] = ids
# Mirror the Ray/CLI GEAK defaults on the Dynamo pod (both keys are in
# SAFE_ENV_KEYS, so an explicit orchestrator-side value is forwarded over SSH and
# wins via setdefault): the advisory profiler-mcp pass is hang-prone -> default off.
os.environ.setdefault("GEAK_SKIP_PROFILE", "1")

def fail(msg):
    print(json.dumps({"returncode": 2, "stdout_tail": "", "stderr_tail": msg,
                      "gpu_ids": ids, "elapsed_s": 0.0, "cmd": []}))
    raise SystemExit(0)

geak_bin = shutil.which("geak") or shutil.which("mini")
if not geak_bin:
    fail("geak CLI not found on the Dynamo GPU pod PATH; bake geak into the "
         "image or install it once over SSH before kernel optimization")
cfg = os.environ.get("GEAK_CONFIG", "").strip()
if not cfg or not Path(cfg).is_file():
    fail("GEAK_CONFIG missing or not a file on the pod: %r" % cfg)
if not re.search(r"(?m)^\s*model_class\s*:\s*litellm\s*$",
                 Path(cfg).read_text(encoding="utf-8", errors="replace")):
    fail("GEAK_CONFIG must set model.model_class: litellm: %s" % cfg)

cmd = [geak_bin, "-t", a.prompt, "--yolo", "--output", a.output,
       "--gpu-ids", ids, "--config", cfg]
# Forward kernel-time scoring (the E2E-faithful default) as an explicit --target,
# mirroring the Ray/CLI paths, so pod-side GEAK does not fall back to its --target
# default. An explicit GEAK_SCORE_TARGET=wall (forwarded over SSH) is honored.
_score_target = os.environ.get("GEAK_SCORE_TARGET", "kernel").strip().lower()
if _score_target in ("wall", "kernel"):
    cmd += ["--target", _score_target]
if a.kernel_path:
    cmd += ["--kernel-path", a.kernel_path]
if a.kernel_repo:
    cmd += ["--repo", a.kernel_repo]
if a.test_command:
    cmd += ["--test-command", a.test_command]
if a.cost_limit != "":
    cmd += ["--cost-limit", a.cost_limit]

started = time.time()
try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout_s)
    print(json.dumps({
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "gpu_ids": ids,
        "elapsed_s": round(time.time() - started, 2),
        "cmd": cmd,
    }))
except subprocess.TimeoutExpired:
    print(json.dumps({"returncode": 124, "stdout_tail": "",
                      "stderr_tail": "TimeoutExpired after %ds" % a.timeout_s,
                      "gpu_ids": ids, "elapsed_s": round(time.time()-started, 2),
                      "cmd": cmd}))
"""


def _extract_last_json(text: str) -> dict | None:
    """Parse the last top-level JSON object from the pod's stdout.

    Args:
        text: The pod's stdout to scan.

    Returns:
        The last top-level JSON object as a dict, or ``None`` when none is
        found or it fails to parse.
    """
    if not text:
        return None
    s = text.rstrip()
    end = s.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if s[i] == "}":
            depth += 1
        elif s[i] == "{":
            depth -= 1
            if depth == 0:
                import json

                try:
                    return json.loads(s[i : end + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _ssh_unconfigured() -> dict | None:
    """Return an error dict if MN_SSH_HOST/KEY are missing, else None.

    Returns:
        A result-shaped error dict when SSH is unconfigured, else ``None``.
    """
    host, _port, key = ssh_target()
    if not host or not key:
        return {
            "returncode": 1,
            "stdout_tail": "",
            "stderr_tail": "KERNEL_AGENT_GPU_PLACEMENT=ssh but MN_SSH_HOST / "
            "MN_SSH_KEY are not set; orchestrator must supply them",
            "gpu_ids": "",
            "elapsed_s": 0.0,
            "cmd": [],
        }
    return None


def _run_pod_runner_over_ssh(
    runner_name: str,
    runner_py: str,
    runner_args: list[str],
    timeout_s: int,
) -> tuple[dict | None, subprocess.CompletedProcess | None, str]:
    """Ship a self-contained pod runner to the GPU pod over SSH and run it.

    Env (creds) is piped via SSH stdin (never argv/disk). Returns
    ``(parsed_json_or_None, completed_proc_or_None, host)``.

    Args:
        runner_name: Short name for the runner (used in temp paths / heredoc).
        runner_py: The self-contained pod runner Python source.
        runner_args: CLI args passed to the pod runner.
        timeout_s: Runner budget in seconds (SSH timeout adds a buffer).

    Returns:
        A ``(parsed_json_or_None, completed_proc_or_None, host)`` tuple; the
        proc is ``None`` when the SSH call itself timed out.
    """
    host, port, key = ssh_target()
    quoted_args = " ".join(shlex.quote(x) for x in runner_args)
    eof = f"__MN_{runner_name.upper()}_PY__"
    script = (
        "set -euo pipefail\n"
        f"{_env_prologue()}\n"
        f"cat > /tmp/mn_{runner_name}_runner.py <<'{eof}'\n"
        f"{runner_py}\n"
        f"{eof}\n"
        f"exec python3 /tmp/mn_{runner_name}_runner.py {quoted_args}\n"
    )
    argv = ["ssh", *_SSH_OPTS, "-i", key, "-p", str(port), f"root@{host}", "bash", "-s"]
    try:
        proc = subprocess.run(
            argv,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_s + 300,
        )
    except subprocess.TimeoutExpired:
        return None, None, host
    return _extract_last_json(proc.stdout or ""), proc, host


def run_geak_over_ssh(
    prompt_file: Path,
    output_dir: Path,
    kernel_path: str,
    cost_limit: float | None,
    num_gpus: int,
    timeout_s: int,
    kernel_repo: str = "",
    test_command: str = "",
) -> dict:
    """Run GEAK on a Dynamo GPU pod over SSH (no Ray). Returns the same dict
    shape as ``run_via_cli`` / ``run_via_ray`` so callers are placement-blind.

    Args:
        prompt_file: Path to the GEAK prompt (on the shared mount).
        output_dir: Output directory for GEAK artifacts (shared mount).
        kernel_path: Optional kernel source path passed to GEAK.
        cost_limit: Optional cost cap forwarded to GEAK.
        num_gpus: Number of GPUs to expose to the pod runner.
        timeout_s: GEAK budget in seconds.
        kernel_repo: Optional kernel repo path passed to GEAK.
        test_command: Optional test command passed to GEAK.

    Returns:
        The GEAK result dict (``returncode`` / ``stdout_tail`` /
        ``stderr_tail`` / ``gpu_ids`` / ``elapsed_s`` / ``cmd``).
    """
    unconfigured = _ssh_unconfigured()
    if unconfigured is not None:
        return unconfigured
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_args = [
        "--prompt",
        str(prompt_file),
        "--output",
        str(output_dir),
        "--num-gpus",
        str(max(1, num_gpus)),
        "--timeout-s",
        str(timeout_s),
    ]
    if kernel_path:
        runner_args += ["--kernel-path", kernel_path]
    if kernel_repo:
        runner_args += ["--kernel-repo", kernel_repo]
    if test_command:
        runner_args += ["--test-command", test_command]
    if cost_limit is not None:
        runner_args += ["--cost-limit", str(cost_limit)]

    parsed, proc, host = _run_pod_runner_over_ssh(
        "geak",
        _POD_RUNNER,
        runner_args,
        timeout_s,
    )
    if proc is None:
        return {
            "returncode": 124,
            "stdout_tail": "",
            "stderr_tail": f"ssh GEAK run timed out (host={host})",
            "gpu_ids": "",
            "elapsed_s": 0.0,
            "cmd": [],
        }
    if parsed is not None:
        return parsed
    return {
        "returncode": proc.returncode or 1,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": f"ssh transport/runner error (host={host}): {(proc.stderr or '')[-2000:]}",
        "gpu_ids": "",
        "elapsed_s": 0.0,
        "cmd": [],
    }
