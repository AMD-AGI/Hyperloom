# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Select an Ave kernel from supplied inference evidence and prepare its driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

from ave.task import OptimizationRequest, scope_prompts
from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL, claude_sdk_env_options
from ave.claude import query_result


SYSTEM_PROMPT = """You prepare one AveLang kernel optimization for Hyperloom.
Use the supplied inference workload, serving configuration, trace evidence and
source repositories to choose ONE existing single-GPU AveLang kernel that the
serving workload actually calls and to which ALL supplied optimization prompts apply.
Read available trace analysis and source/call sites before selecting it. TraceLens
conclusions and candidate lists are hints: missing analysis or source resolution
does not rule out an Ave kernel. Cross-check raw trace events, runtime dispatch
and logs against the current serving configuration and editable AveLang source.
Establish the active call path from execution evidence or deterministic dispatch
conditions, and distinguish measured facts from source-based inference in reason.
The prompts describe transformations; they are not evidence that a kernel is used
by the model.
Read/Glob/Grep are your only tools. Never modify sources or run commands.
If no suitable kernel can be established, or its workload cases/reference cannot
be established, return {"kernel": null, "reason": "explain the missing evidence"}.
Do not invent a kernel, shapes, dtype, tolerances, or a serving call path.

Otherwise return only JSON:
{"kernel": {"repo_root": "/absolute/source/repository", "kernel_path": "relative/kernel.py",
"source_files": ["relative/kernel.py"], "operator_name": "operator",
"driver_source": "complete Python driver source"}, "reason": "serving call path, trace evidence and workload cases"}

The repository must be one of the supplied source repositories. Include the kernel
and necessary editable dependencies in source_files; paths are repository-relative,
tracked files. Tests and references are not editable sources.
Generate a driver importing the selected kernel from the CURRENT WORKING DIRECTORY,
whose root, python/ and src/ lead PYTHONPATH; never hard-code the serving checkout.
The driver must cover the current workload's shapes/dtypes with an independent
reference and explicit numerical tolerances. With no arguments it must assert
correctness and print 'allclose: True' only after all cases pass.
With --bench-mode --warmup 3 --iters 20 it must warm up compilation, then time the
actual kernel using synchronized GPU graph replay, excluding compilation and host
overhead. Print 'case_ms: <stable_case_id> <positive_milliseconds>' for every case.
Keep cases identical between validation and timing. Use runtime APIs and test
conventions verified from source. Do not run or benchmark the driver yourself;
the host validates it against the unchanged kernel before allowing optimization.
"""


class NoAveKernel(ValueError):
    """The workload does not provide enough evidence for a supported Ave target."""


@dataclass(frozen=True)
class PreparedKernel:
    """A Hyperloom-selected target and a fixed correctness/timing program."""

    repo_root: Path
    kernel_path: str
    source_files: tuple[str, ...]
    operator_name: str
    driver_source: str
    reason: str


def read_selection(payload: dict, repositories: tuple[Path, ...]) -> PreparedKernel:
    """Validate model output at the preparation boundary."""
    if not isinstance(payload, dict) or not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        raise ValueError("Ave preparation requires a reason with workload evidence")
    kernel = payload.get("kernel")
    if kernel is None:
        raise NoAveKernel(payload["reason"])
    required = ("repo_root", "kernel_path", "operator_name", "driver_source")
    if not isinstance(kernel, dict) or any(
        not isinstance(kernel.get(key), str) or not kernel[key].strip() for key in required
    ):
        raise ValueError("Ave preparation requires repository, kernel, operator and driver source")
    repo = Path(kernel["repo_root"]).resolve()
    if repo not in repositories:
        raise ValueError("Ave selected a repository outside the serving source repositories")
    sources = kernel.get("source_files")
    if not isinstance(sources, list) or not sources or any(not isinstance(source, str) for source in sources):
        raise ValueError("Ave preparation requires a source_files list")
    sources = tuple(dict.fromkeys((kernel["kernel_path"], *sources)))
    for source in sources:
        path = Path(source)
        resolved = (repo / path).resolve()
        if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(repo):
            raise ValueError(f"Ave source must be repository-relative: {source}")
        if not resolved.is_file() or (repo / path).is_symlink():
            raise ValueError(f"Ave source must be a regular file: {source}")
    compile(kernel["driver_source"], "ave-driver.py", "exec")
    return PreparedKernel(
        repo, kernel["kernel_path"], sources, kernel["operator_name"], kernel["driver_source"], payload["reason"]
    )


async def prepare_kernel(request: OptimizationRequest, output: Path) -> PreparedKernel:
    """Use a read-only Claude session to turn inference evidence into an Ave target."""
    from claude_agent_sdk import ClaudeAgentOptions

    repositories = request.repositories
    if not repositories:
        raise NoAveKernel("No serving source repositories are available")
    context = {
        "workload": request.workload,
        "serving": request.serving,
        "trace": request.trace,
        "source_repositories": [str(repo) for repo in repositories],
        "steps": [asdict(step) for step in request.steps],
    }
    atomic_write_json(output / "preparation-context.json", context)
    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
    settings = {"setting_sources": [], **claude_sdk_env_options(model=model, component="ave_preparation")}
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        cwd=str(output),
        add_dirs=[str(output.parent.parent), *(str(repo) for repo in repositories)],
        tools=["Read", "Glob", "Grep"],
        allowed_tools=["Read", "Glob", "Grep"],
        mcp_servers={},
        strict_mcp_config=True,
        plugins=[],
        skills=[],
        max_turns=24,
        **settings,
    )
    payload = extract_last_json_with_key(await query_result(json.dumps(context), options), "kernel")
    atomic_write_json(output / "selection.json", payload)
    kernel = read_selection(payload, repositories)
    scope_prompts(request.steps, kernel.source_files)
    return kernel
