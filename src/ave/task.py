# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ave optimization instructions, requests and prepared worker context."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from ave.workspace import git

DEFAULT_PROMPT = (
    "Optimize the selected AveLang kernel for the supplied inference workload. "
    "Use the source and baseline measurements to choose a concrete optimization, "
    "preserving correctness for every workload case."
)


@dataclass(frozen=True)
class PromptStep:
    """One required transformation and the files it may edit."""

    id: str
    files: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class OptimizationRequest:
    """Inference evidence and ordered instructions supplied by the caller."""

    workload: str
    serving: str
    trace: str
    repositories: tuple[Path, ...]
    steps: tuple[PromptStep, ...]


@dataclass(frozen=True)
class OptimizationRun:
    """A selected kernel, scoped instructions, isolated workspace and worker deadline."""

    repo_root: Path
    base_commit: str
    kernel_path: str
    source_files: tuple[str, ...]
    operator_name: str
    steps: tuple[PromptStep, ...]
    workspace: Path
    driver: Path
    result_json: Path
    deadline_unix: float
    reason: str = "Optimize the supplied workload cases without changing semantics."

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.kernel_path, *self.source_files)))

    @property
    def logs(self) -> Path:
        return self.result_json.parent / "ave-iterations"

    def validate(self) -> None:
        """Refuse a workspace that cannot measure this prepared task."""
        head = git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()
        if head != self.base_commit:
            raise ValueError("Ave workspace HEAD must match base_commit")
        if git("status", "--porcelain", "--untracked-files=no", cwd=self.workspace).stdout.strip():
            raise ValueError("Ave requires a clean tracked workspace")
        for source in self.sources:
            path = Path(source)
            resolved = (self.workspace / path).resolve()
            if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(self.workspace):
                raise ValueError(f"source must be relative to the workspace: {source}")
            if not resolved.is_file() or (self.workspace / path).is_symlink():
                raise ValueError(f"source must be a regular file: {source}")
            git("cat-file", "-e", f"{head}:{source}", cwd=self.workspace)
        if any(not step.files or not set(step.files) <= set(self.sources) for step in self.steps):
            raise ValueError("step files must belong to the selected kernel's source_files")
        if not self.driver.is_file() or self.driver in {(self.workspace / source).resolve() for source in self.sources}:
            raise ValueError("the prepared driver must exist and cannot be an editable source")
        if self.result_json.is_relative_to(self.workspace) or self.result_json.exists():
            raise ValueError("Ave requires a fresh result path outside the editable workspace")


def read_run(payload: dict) -> OptimizationRun:
    """Decode the run context supplied by Hyperloom."""
    return OptimizationRun(
        repo_root=Path(payload["repo_root"]).resolve(),
        base_commit=payload["base_commit"],
        kernel_path=payload["kernel_path"],
        source_files=tuple(payload["source_files"]),
        operator_name=payload["operator_name"],
        steps=parse_prompts({"steps": payload["steps"]}),
        reason=payload["reason"],
        workspace=Path(payload["workspace"]).resolve(),
        driver=Path(payload["driver"]).resolve(),
        result_json=Path(payload["result_json"]).resolve(),
        deadline_unix=float(payload["deadline_unix"]),
    )


def read_prompts(path: Path) -> tuple[PromptStep, ...]:
    """Read ordered instructions independently of kernel selection."""
    return parse_prompts(json.loads(path.read_text(encoding="utf-8")))


def read_env_prompts() -> tuple[PromptStep, ...]:
    """Use the built-in prompt unless a file supplies replacement steps."""
    path = os.environ.get("AVE_PROMPTS_FILE", "").strip()
    if not path:
        return (PromptStep("optimize", (), DEFAULT_PROMPT),)
    try:
        return read_prompts(Path(path).expanduser())
    except (OSError, ValueError) as error:
        raise ValueError(f"AVE_PROMPTS_FILE: {error}") from error


def parse_prompts(payload: dict) -> tuple[PromptStep, ...]:
    """Validate the prompt file or an equivalent subprocess payload."""
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError("prompts file requires a nonempty steps list")
    result = []
    seen = set()
    for step in steps:
        if not isinstance(step, dict) or any(
            not isinstance(step.get(key), str) or not step[key].strip() for key in ("id", "prompt")
        ):
            raise ValueError("each step requires nonempty id and prompt strings")
        files = step.get("files", [])
        if not isinstance(files, list) or any(not isinstance(path, str) or not path for path in files):
            raise ValueError("step files must be a list of relative paths")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in files):
            raise ValueError("step files must be relative paths without '..'")
        if step["id"] in seen:
            raise ValueError(f"duplicate step id: {step['id']}")
        seen.add(step["id"])
        result.append(PromptStep(step["id"], tuple(dict.fromkeys(files)), step["prompt"]))
    return tuple(result)


def scope_prompts(steps: tuple[PromptStep, ...], sources: tuple[str, ...]) -> tuple[PromptStep, ...]:
    """Bind instructions to the selected kernel, respecting narrower per-step scopes."""
    if any(not set(step.files) <= set(sources) for step in steps):
        raise ValueError("step files must belong to the selected kernel's source_files")
    return tuple(PromptStep(step.id, step.files or sources, step.prompt) for step in steps)
