# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Select an Ave kernel from supplied inference evidence and prepare its driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from ave.task import NonemptyText, OptimizationRequest, RelativeSource, scope_prompts
from hyperloom_common.io import atomic_write_json
from hyperloom_common.jsonio import extract_last_json_with_key
from ave.claude import query_result
from ave.workspace import git


class NoAveKernel(ValueError):
    """The workload does not provide enough evidence for a supported Ave target."""


@dataclass(frozen=True)
class PreparedKernel:
    """A Hyperloom-selected target and a fixed correctness/timing program."""

    repo_root: Path
    kernel_path: RelativeSource
    source_files: Annotated[tuple[RelativeSource, ...], Field(min_length=1)]
    operator_name: NonemptyText
    driver_source: NonemptyText
    reason: NonemptyText


def read_selection(payload: dict, repositories: tuple[Path, ...]) -> PreparedKernel:
    """Validate model output at the preparation boundary."""
    reason = TypeAdapter(NonemptyText).validate_python(payload.get("reason"))
    if payload.get("kernel") is None:
        raise NoAveKernel(reason)
    if not isinstance(payload["kernel"], dict):
        raise ValueError("Ave selection must be a kernel object")
    kernel = TypeAdapter(PreparedKernel).validate_python({**payload["kernel"], "reason": reason})
    kernel = replace(kernel, repo_root=kernel.repo_root.resolve())
    if kernel.repo_root not in repositories:
        raise ValueError("Ave selected a repository outside the serving source repositories")
    sources = tuple(dict.fromkeys((kernel.kernel_path, *kernel.source_files)))
    for source in sources:
        path = kernel.repo_root / source
        if not path.resolve().is_relative_to(kernel.repo_root) or not path.is_file() or path.is_symlink():
            raise ValueError(f"Ave source must be a regular file inside the repository: {source}")
        git("ls-files", "--error-unmatch", "--", source, cwd=kernel.repo_root)
    compile(kernel.driver_source, "ave-driver.py", "exec")
    return replace(kernel, source_files=sources)


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
    options = ClaudeAgentOptions(
        cwd=str(output),
        add_dirs=[str(output.parent.parent), *(str(repo) for repo in repositories)],
        tools=["Read", "Glob", "Grep"],
        max_turns=24,
    )
    result = await query_result(context, options, operation="prepare_kernel", transcript=output / "preparation.jsonl")
    payload = extract_last_json_with_key(result, "kernel")
    atomic_write_json(output / "selection.json", payload)
    kernel = read_selection(payload, repositories)
    scope_prompts(request.steps, kernel.source_files)
    return kernel
