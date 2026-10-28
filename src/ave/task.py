# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ave optimization instructions, requests and prepared worker context."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, FiniteFloat, StringConstraints, TypeAdapter


DEFAULT_PROMPT = (
    "Optimize the selected AveLang kernel for the supplied inference workload. "
    "Use the source and baseline measurements to choose a concrete optimization, "
    "preserving correctness for every workload case."
)


class DeadlineExceeded(RuntimeError):
    """The Ave campaign has no time budget left to continue."""


def relative_source(value: str) -> str:
    """Reject paths that can escape the selected source repository."""
    if not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("source files must be relative paths without '..'")
    return value


NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RelativeSource = Annotated[str, AfterValidator(relative_source)]


@dataclass(frozen=True)
class PromptStep:
    """One required transformation and the files it may edit."""

    id: NonemptyText
    prompt: NonemptyText
    files: tuple[RelativeSource, ...] = ()


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
    deadline_unix: FiniteFloat
    reason: str = "Optimize the supplied workload cases without changing semantics."

    @property
    def logs(self) -> Path:
        return self.result_json.parent / "ave-iterations"


def read_prompts(path: Path) -> tuple[PromptStep, ...]:
    """Read ordered instructions independently of kernel selection."""
    return parse_prompts(json.loads(path.read_text(encoding="utf-8")))


def read_env_prompts() -> tuple[PromptStep, ...]:
    """Use the built-in prompt unless a file supplies replacement steps."""
    path = os.environ.get("AVE_PROMPTS_FILE", "").strip()
    if not path:
        return (PromptStep("optimize", DEFAULT_PROMPT),)
    try:
        return read_prompts(Path(path).expanduser())
    except (OSError, ValueError) as error:
        raise ValueError(f"AVE_PROMPTS_FILE: {error}") from error


def parse_prompts(payload: dict) -> tuple[PromptStep, ...]:
    """Validate the prompt file or an equivalent subprocess payload."""
    steps = TypeAdapter(tuple[PromptStep, ...]).validate_python(
        (payload.get("steps") if isinstance(payload, dict) else None)
    )
    if not steps:
        raise ValueError("prompts file requires a nonempty steps list")
    seen = set()
    for step in steps:
        if step.id in seen:
            raise ValueError(f"duplicate step id: {step.id}")
        seen.add(step.id)
    return tuple(replace(step, files=tuple(dict.fromkeys(step.files))) for step in steps)


def scope_prompts(steps: tuple[PromptStep, ...], sources: tuple[str, ...]) -> tuple[PromptStep, ...]:
    """Bind instructions to the selected kernel, respecting narrower per-step scopes."""
    if any(not set(step.files) <= set(sources) for step in steps):
        raise ValueError("step files must belong to the selected kernel's source_files")
    return tuple(replace(step, files=step.files or sources) for step in steps)
