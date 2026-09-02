# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filesystem layout and path validation for the kernel rewrite controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kernelforge.kernel_rewrite_controller.contracts import (
    DRIVER_FILENAME,
    STATE_FILENAME,
    TASK_FILENAME,
    TaskContractError,
)


def safe_relative_path(value: object, *, field_name: str) -> str:
    """Return a normalized POSIX path that cannot escape its future root."""
    if not isinstance(value, str):
        raise TaskContractError(f"{field_name} must be a string")
    raw = value.strip()
    if not raw:
        raise TaskContractError(f"{field_name} is required")
    if "\x00" in raw or "\\" in raw:
        raise TaskContractError(f"{field_name} must be a POSIX relative path")
    path = PurePosixPath(raw)
    if raw in {".", ".."} or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TaskContractError(f"{field_name} must not escape its root: {raw!r}")
    return path.as_posix()


def safe_operator_id(value: object) -> str:
    """Validate a canonical operator identity used as one directory segment."""
    if not isinstance(value, str):
        raise TaskContractError("operator_id must be a string")
    operator_id = value.strip()
    if (
        not operator_id
        or operator_id in {".", ".."}
        or "/" in operator_id
        or "\\" in operator_id
        or "\x00" in operator_id
    ):
        raise TaskContractError(f"unsafe operator_id: {operator_id!r}")
    return operator_id


@dataclass(frozen=True)
class ControllerLayout:
    """Canonical directories below one controller output root."""

    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())

    @property
    def controller_root(self) -> Path:
        return self.output_dir / "controller"

    @property
    def tasks_root(self) -> Path:
        return self.controller_root / "tasks"

    @property
    def workspaces_root(self) -> Path:
        return self.controller_root / "workspaces"

    @property
    def controller_state(self) -> Path:
        return self.controller_root / "state.json"

    @property
    def result_root(self) -> Path:
        return self.output_dir / "result"

    @property
    def patches_root(self) -> Path:
        return self.result_root / "patches"

    @property
    def summary_md(self) -> Path:
        return self.result_root / "summary.md"

    def task_dir(self, operator_id: str) -> Path:
        return self.tasks_root / safe_operator_id(operator_id)

    def workspace_dir(self, operator_id: str) -> Path:
        return self.workspaces_root / safe_operator_id(operator_id)

    def patch_dir(self, operator_id: str) -> Path:
        return self.patches_root / safe_operator_id(operator_id)

    @staticmethod
    def is_published_task_dir(path: Path) -> bool:
        """Return whether an entry is a complete, non-temporary task directory."""
        candidate = Path(path)
        return (
            candidate.is_dir()
            and not candidate.name.startswith(".")
            and (candidate / TASK_FILENAME).is_file()
            and (candidate / DRIVER_FILENAME).is_file()
        )


@dataclass(frozen=True)
class TaskLayout:
    """Paths owned by one published task directory."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def task_json(self) -> Path:
        return self.root / TASK_FILENAME

    @property
    def driver(self) -> Path:
        return self.root / DRIVER_FILENAME

    @property
    def state_json(self) -> Path:
        return self.root / STATE_FILENAME


__all__ = [
    "ControllerLayout",
    "TaskLayout",
    "safe_operator_id",
    "safe_relative_path",
]
