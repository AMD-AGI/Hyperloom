"""Profiling — torch profiler for kernel discovery + TraceLens integration.

Primary profiling is via torch profiler (no rocprof — reliability issues).
TraceLens is used when available for deeper trace analysis via Claude SDK.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class KernelInfo:
    """Information about a GPU kernel from profiling."""

    name: str
    duration_us: float = 0.0
    percentage: float = 0.0  # fraction of total GPU time
    count: int = 0
    source_function: str = ""
    shape: str = ""  # e.g., "M=128,N=7168,K=2048"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result of a profiling run."""

    kernels: list[KernelInfo] = field(default_factory=list)
    total_gpu_time_us: float = 0.0
    idle_pct: float = 0.0
    trace_path: str = ""
    raw_output: str = ""

    @property
    def hot_kernels(self) -> list[KernelInfo]:
        """Top kernels by GPU time (>5% of total)."""
        return [k for k in self.kernels if k.percentage > 0.05]


def run_torch_profiler(
    benchmark_cmd: list[str],
    output_dir: str,
    env: dict[str, str] | None = None,
    warmup_steps: int = 3,
    active_steps: int = 5,
) -> ProfileResult:
    """Run a benchmark under torch profiler and parse the trace.

    Injects HYPERLOOM_PROFILE=1 into the environment so the benchmark
    script can enable profiling if it supports it. Otherwise uses a
    wrapper that activates torch.profiler.profile() around the workload.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trace_path = out_path / "trace.json"

    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["HYPERLOOM_PROFILE"] = "1"
    child_env["HYPERLOOM_PROFILE_OUTPUT"] = str(trace_path)
    child_env["HYPERLOOM_PROFILE_WARMUP"] = str(warmup_steps)
    child_env["HYPERLOOM_PROFILE_ACTIVE"] = str(active_steps)

    log.info("Running profiling pass: %s", " ".join(benchmark_cmd[:3]))
    try:
        result = subprocess.run(
            benchmark_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            env=child_env,
            cwd=output_dir,
        )
    except subprocess.TimeoutExpired:
        log.error("Profiling run timed out")
        return ProfileResult(raw_output="timeout")

    if trace_path.exists():
        return parse_torch_trace(str(trace_path))

    return ProfileResult(
        raw_output=result.stdout + "\n" + result.stderr,
    )


def parse_torch_trace(trace_path: str) -> ProfileResult:
    """Parse a Chrome trace JSON from torch.profiler into KernelInfo list."""
    path = Path(trace_path)
    if not path.exists():
        return ProfileResult()

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.error("Failed to parse trace JSON: %s", trace_path)
        return ProfileResult(trace_path=trace_path)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    kernel_times: dict[str, float] = {}
    kernel_counts: dict[str, int] = {}

    for event in events:
        if event.get("cat") == "kernel" or event.get("cat") == "gpu_memcpy":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)  # microseconds
            kernel_times[name] = kernel_times.get(name, 0) + dur
            kernel_counts[name] = kernel_counts.get(name, 0) + 1

    total_time = sum(kernel_times.values()) or 1.0

    kernels = []
    for name, dur in sorted(kernel_times.items(), key=lambda x: -x[1]):
        kernels.append(KernelInfo(
            name=name,
            duration_us=dur,
            percentage=dur / total_time,
            count=kernel_counts.get(name, 0),
        ))

    return ProfileResult(
        kernels=kernels,
        total_gpu_time_us=total_time,
        trace_path=trace_path,
    )


def strip_tracelens_charts(analysis_text: str) -> str:
    """Remove base64-encoded PNG/SVG chart blobs from TraceLens analysis output.

    TraceLens analysis.md often contains embedded base64 images that bloat
    LLM context windows. This strips them while preserving all textual analysis.
    """
    import re

    stripped = re.sub(
        r"!\[([^\]]*)\]\(data:image/[^)]+\)",
        r"[Chart: \1 — removed for context efficiency]",
        analysis_text,
    )
    stripped = re.sub(
        r"<img[^>]*src=[\"']data:image/[^\"']+[\"'][^>]*>",
        "[Embedded chart removed for context efficiency]",
        stripped,
    )
    stripped = re.sub(
        r"```[^\n]*\n(?:[A-Za-z0-9+/=\n]{200,})\n```",
        "[Base64 data block removed]",
        stripped,
    )
    return stripped


def prepare_server_for_profiling(
    server_cmd: list[str],
    env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Prepare a serving framework launch command for profiling.

    Adds --enforce-eager flag (disables CUDA graphs for accurate profiling)
    and sets torch profiler environment variables.

    Returns (modified_cmd, modified_env).
    """
    modified_cmd = list(server_cmd)

    if "--enforce-eager" not in modified_cmd:
        modified_cmd.append("--enforce-eager")

    modified_env = dict(env or os.environ.copy())

    modified_env["TORCH_PROFILER_ENABLED"] = "1"
    modified_env["KINETO_LOG_LEVEL"] = "0"
    modified_env.setdefault("NCCL_DEBUG", "WARN")
    modified_env["PYTORCH_CUDA_ALLOC_CONF"] = modified_env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    return modified_cmd, modified_env


async def run_tracelens(
    trace_path: str,
    output_dir: str,
    skill_path: str | None = None,
    model: str = "claude-opus-4-7",
    platform: str = "mi300x",
    framework: str = "",
) -> ProfileResult:
    """Run TraceLens analysis via Claude SDK agent.

    Requires claude_agent_sdk to be installed. Falls back gracefully
    if not available.
    """
    try:
        import claude_agent_sdk as sdk
    except ImportError:
        log.warning("claude_agent_sdk not available, skipping TraceLens")
        return ProfileResult()

    if not skill_path:
        skill_path = os.environ.get("TRACELENS_SKILL_PATH", "")
    if not skill_path or not Path(skill_path).exists():
        log.warning("TraceLens skill not found at %s", skill_path)
        return ProfileResult()

    tracelens_root = str(Path(skill_path).parent)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Analyze the trace at {trace_path}. "
        f"Platform: {platform}. Framework: {framework}. "
        f"Write the analysis report to {output_dir}/analysis.md. "
        f"Use `cd {tracelens_root} && <CMD>` for all shell commands. "
        "Execute the analysis-orchestrator workflow through Step 11."
    )

    options = sdk.ClaudeAgentOptions(
        max_turns=300,
        system_prompt=f"You are a TraceLens analysis runner. Skill file: {skill_path}",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Task"],
        model=model,
        cwd=tracelens_root,
    )

    chunks: list[str] = []
    async for message in sdk.query(prompt=prompt, options=options):
        if hasattr(message, "content"):
            chunks.append(str(message.content))

    analysis_path = out_path / "analysis.md"
    if analysis_path.exists():
        log.info("TraceLens analysis complete: %s", analysis_path)

    return ProfileResult(
        trace_path=trace_path,
        raw_output="\n".join(chunks),
    )
