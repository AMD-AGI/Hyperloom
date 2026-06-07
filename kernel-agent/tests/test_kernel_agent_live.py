#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in live tests for Kernel Agent tools.

Run with:
  KERNEL_AGENT_RUN_LIVE=1 \
  KERNEL_AGENT_ENV_FILE=/wekafs/xiaofei/AgentKernelArena/.env \
  KERNEL_AGENT_LIVE_BACKENDS=geak,claude,codex,llm \
  python3 -m unittest kernel-agent/tests/test_kernel_agent_live.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_TOOL = ROOT / "tools" / "tracelens_analysis.py"
OPT_TOOL = ROOT / "tools" / "kernel_optimization.py"


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    if "SAFE_API_KEY" in env:
        env.setdefault("OOB_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("ANTHROPIC_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("OPENAI_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("ANTHROPIC_AUTH_TOKEN", env["SAFE_API_KEY"])
    if "ANTHROPIC_AUTH_TOKEN" in env:
        env.setdefault("ANTHROPIC_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
        env.setdefault("OPENAI_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
        env.setdefault("OOB_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
    if "AMD_API_KEY" in env:
        env.setdefault("AMD_LLM_API_KEY", env["AMD_API_KEY"])
        env.setdefault("LLM_API_KEY", env["AMD_API_KEY"])
        env.setdefault("GEAK_API_KEY", env["AMD_API_KEY"])
    if "OPENAI_BASE_URL" in env:
        env.setdefault("ANTHROPIC_BASE_URL", env["OPENAI_BASE_URL"])
        env.setdefault("OOB_BASE_URL", env["OPENAI_BASE_URL"])
        env.setdefault("LLM_API_BASE", env["OPENAI_BASE_URL"])
    elif "ANTHROPIC_BASE_URL" in env:
        env.setdefault("OPENAI_BASE_URL", env["ANTHROPIC_BASE_URL"])
        env.setdefault("OOB_BASE_URL", env["ANTHROPIC_BASE_URL"])
        env.setdefault("LLM_API_BASE", env["ANTHROPIC_BASE_URL"])
    return env


def gpu_available() -> bool:
    if shutil.which("rocm-smi") and subprocess.run(
        ["rocm-smi", "--showid"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        return True
    if shutil.which("amd-smi") and subprocess.run(
        ["amd-smi", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0:
        return True
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_json(cmd: list[str], *, workspace: Path, env: dict[str, str], timeout: int = 180) -> dict:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env={**os.environ, **env, "USER_DATA_PATH": str(workspace)},
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "command failed\n"
            f"cmd={cmd}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


def write_live_files(workspace: Path) -> tuple[Path, Path, Path]:
    source = workspace / "rmsnorm_kernel.py"
    source.write_text(
        "def triton_rmsnorm_kernel(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    harness = workspace / "bench_rmsnorm.py"
    harness.write_text(
        "import torch\n"
        "x = torch.ones((16, 16), device='cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print(float(x.sum().item()))\n",
        encoding="utf-8",
    )
    trace = workspace / "live_trace.json"
    trace.write_text(json.dumps({
        "traceEvents": [
            {
                "name": "triton_rmsnorm_kernel",
                "cat": "kernel",
                "dur": 1000,
                "args": {
                    "source_file": str(source),
                    "shape": {"M": 16, "N": 16},
                },
            }
        ]
    }), encoding="utf-8")
    return trace, source, harness


@unittest.skipUnless(os.environ.get("KERNEL_AGENT_RUN_LIVE") == "1", "live test disabled")
class KernelAgentLiveTests(unittest.TestCase):
    def test_live_gpu_and_backend_attempts_are_recorded(self) -> None:
        self.assertTrue(gpu_available(), "GPU is not available to the live test")

        env_file = Path(os.environ.get(
            "KERNEL_AGENT_ENV_FILE",
            "/wekafs/xiaofei/AgentKernelArena/.env",
        ))
        env = load_dotenv(env_file)
        env.setdefault("KERNEL_AGENT_LLM_MODEL", "claude4.7")
        backends = os.environ.get("KERNEL_AGENT_LIVE_BACKENDS", "geak,claude,codex,llm")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace, source, harness = write_live_files(workspace)
            analysis = run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "live",
                "--model-name", "live-test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace, env=env)
            self.assertEqual(analysis["hot_kernels"][0]["kernel_id"], "k001")

            result = run_json([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "live",
                "--backends", backends,
                "--source-file", str(source),
                "--test-harness-path", str(harness),
                "--budget-minutes", os.environ.get("KERNEL_AGENT_LIVE_BUDGET_MIN", "0.25"),
            ], workspace=workspace, env=env, timeout=240)

            self.assertEqual(result["selected_backends"], [b.strip() for b in backends.split(",")])
            self.assertEqual(len(result["attempts"]), len(result["selected_backends"]))
            self.assertTrue(Path(result["cli_log_path"]).exists())
            self.assertTrue(Path(result["status_path"]).exists())
            for attempt in result["attempts"]:
                self.assertIn(attempt["status"], {"completed", "failed", "timeout", "partial"})
                self.assertIn("attempt_id", attempt)
            self.assertIn(result["proposal"]["decision"], {"KEEP", "REVERT", "NEEDS_REVIEW"})


if __name__ == "__main__":
    unittest.main()
