# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The acceptance step every path that can keep or adopt a kernel goes through."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from kernelforge.mcp_server.tools._subprocess import communicate_process_group

CANONICAL_CONFIG_FILENAME = "config.yaml"

# ``_DEFAULT_COMPILE_TIMEOUT_S`` and ``_DEFAULT_CORRECTNESS_TIMEOUT_S`` in the arena's evaluator.
ARENA_DEFAULT_COMPILE_TIMEOUT_SEC = 3600
ARENA_DEFAULT_CORRECTNESS_TIMEOUT_SEC = 3600

# The suite's failure text is what the agent reads instead of "SNR=33.4dB PASS", and the assertion that names the
# exceeded tolerance -- or the shape the kernel refused to compile for -- is the last thing a task runner prints.
_OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class CanonicalCorrectnessResult:
    """The arena's verdict on this candidate, or the reason there is none."""

    passed: bool
    detail: str
    output: str = ""
    unverified_reason: str = ""
    # "timeout" when the suite was killed, so the run's decision label separates a candidate the arena rejected from
    # one it never finished judging.
    outcome: str = ""


@dataclass(frozen=True)
class _CompileStep:
    """The arena's Step 1: ``evaluate_compilation``."""

    commands: tuple[str, ...]
    timeout_sec: int

    label: ClassVar[str] = "compilation"

    def reports_failure(self, output: str) -> bool:
        return False


@dataclass(frozen=True)
class _CorrectnessStep:
    """The arena's Step 2: ``evaluate_correctness``."""

    commands: tuple[str, ...]
    timeout_sec: int

    label: ClassVar[str] = "correctness"

    def reports_failure(self, output: str) -> bool:
        lowered = output.lower()
        return "fail" in lowered and "pass" not in lowered


@dataclass(frozen=True)
class _CanonicalSuite:
    """The steps the arena would run for this task, in the order it runs them."""

    compile_step: _CompileStep
    correctness_step: _CorrectnessStep

    @property
    def steps(self) -> tuple[_CompileStep | _CorrectnessStep, ...]:
        return (self.compile_step, self.correctness_step)


def _declared_commands(path: Path, document: dict[str, Any], key: str) -> tuple[str, ...] | str:
    """Return the declared command list, or the reason it is unusable."""
    declared = document.get(key)
    if not declared:
        # The arena's absent-command branch returns a failure, not a skip: "No compile_command specified" / "No
        # correctness_command specified".
        return f"{path} declares no {key!r}"
    # The arena iterates this value directly, so a bare string would be run one character at a time.
    if not isinstance(declared, (list, tuple)) or not all(
        isinstance(command, str) and command.strip() for command in declared
    ):
        return f"{path} declares {key!r} as {declared!r}; the arena runs it as a list of shell command strings"
    return tuple(str(command) for command in declared)


def _declared_timeout(path: Path, document: dict[str, Any], key: str, arena_default_sec: int) -> int | str:
    """Return the declared timeout in seconds, or the reason it is unusable."""
    raw_timeout = document.get(key, arena_default_sec)
    try:
        timeout_sec = int(raw_timeout)
    except (TypeError, ValueError):
        return f"{path} declares {key!r}: {raw_timeout!r}, which is not a number of seconds"
    if timeout_sec <= 0:
        return f"{path} declares a non-positive {key!r}: {raw_timeout!r}"
    return timeout_sec


def _load_suite(workspace_dir: str) -> _CanonicalSuite | str | None:
    """Return the declared suite, the reason it is unusable, or None if absent."""
    path = Path(workspace_dir) / CANONICAL_CONFIG_FILENAME
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        return f"{path} exists but could not be read: {error}"
    if not isinstance(document, dict):
        return f"{path} does not parse to a mapping of task settings"

    compile_commands = _declared_commands(path, document, "compile_command")
    if isinstance(compile_commands, str):
        return compile_commands
    compile_timeout = _declared_timeout(path, document, "compile_timeout", ARENA_DEFAULT_COMPILE_TIMEOUT_SEC)
    if isinstance(compile_timeout, str):
        return compile_timeout

    correctness_commands = _declared_commands(path, document, "correctness_command")
    if isinstance(correctness_commands, str):
        return correctness_commands
    correctness_timeout = _declared_timeout(
        path, document, "correctness_timeout", ARENA_DEFAULT_CORRECTNESS_TIMEOUT_SEC
    )
    if isinstance(correctness_timeout, str):
        return correctness_timeout

    return _CanonicalSuite(
        compile_step=_CompileStep(commands=compile_commands, timeout_sec=compile_timeout),
        correctness_step=_CorrectnessStep(commands=correctness_commands, timeout_sec=correctness_timeout),
    )


async def _run_canonical_suite(
    workspace_dir: str,
    *,
    timeout_cap_sec: int,
) -> CanonicalCorrectnessResult:
    """Run the arena's Step 1 then Step 2 and stop at the first failure."""
    suite = _load_suite(workspace_dir)
    if suite is None:
        return CanonicalCorrectnessResult(
            passed=True,
            detail="",
            unverified_reason=(
                f"this workspace ships no {CANONICAL_CONFIG_FILENAME}, so there "
                "is no canonical acceptance suite to judge this candidate "
                "against and only the SNR probe stands behind it"
            ),
        )
    if isinstance(suite, str):
        return CanonicalCorrectnessResult(passed=False, detail=suite)

    passed_steps: list[str] = []
    for step in suite.steps:
        timeout_sec = min(step.timeout_sec, timeout_cap_sec)
        for command in step.commands:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await communicate_process_group(proc, timeout=timeout_sec)
            except asyncio.TimeoutError:
                return CanonicalCorrectnessResult(
                    passed=False,
                    detail=(f"{step.label}: {command!r} timed out after {timeout_sec}s"),
                    output=f"canonical {step.label} command timed out: {command}",
                    outcome="timeout",
                )
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            if proc.returncode != 0:
                return CanonicalCorrectnessResult(
                    passed=False,
                    detail=f"{step.label}: {command!r} exited {proc.returncode}",
                    output=output[-_OUTPUT_TAIL_CHARS:],
                )
            if step.reports_failure(output):
                return CanonicalCorrectnessResult(
                    passed=False,
                    detail=(f"{step.label}: {command!r} reported failure in its output"),
                    output=output[-_OUTPUT_TAIL_CHARS:],
                )
        passed_steps.append(f"{step.label}: {len(step.commands)} command(s) under {timeout_sec}s")

    return CanonicalCorrectnessResult(passed=True, detail="; ".join(passed_steps))


async def accept_candidate(
    workspace_dir: str,
    *,
    timeout_cap_sec: int,
    candidate_label: str,
) -> CanonicalCorrectnessResult:
    """Judge a candidate every other gate has already accepted."""
    print(
        f"  [canonical] Running the arena's acceptance suite (compilation, then correctness) for {candidate_label}..."
    )
    result = await _run_canonical_suite(
        workspace_dir,
        timeout_cap_sec=timeout_cap_sec,
    )
    if result.unverified_reason:
        print(f"  [canonical] UNVERIFIED: {result.unverified_reason}")
    elif result.passed:
        print(f"  [canonical] PASS: {result.detail}")
    else:
        print(f"  [canonical] FAIL: {result.detail}")
        if result.output:
            print(result.output)
    return result
