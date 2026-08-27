# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The acceptance step every path that can keep or adopt a kernel goes through.

:func:`accept_candidate` is that step: it reproduces the arena's acceptance
verdict -- compilation, then correctness, stopping at the first failure, exactly
as ``AgentKernelArena/src/evaluator.py::evaluate_kernel`` does in its Step 1 and
Step 2 -- and it is the only entry point this module offers, so a new path that
promotes a kernel cannot reach that verdict by a route with its own rules. The
iteration loop's KEEP and the KB warm-start's adoption both call it; a warm start
reaching the arena's verdict by a different route is exactly how a kernel that
fails the task's tolerance once became a campaign's answer.

The gate covers Step 1 because a gate that only ran the correctness command is a
gate the agent can walk past. On ``tilelang_dsa_sparse_mla_glm5`` the agent turned
the kernel's hardcoded launch geometry into sweepable knobs and, soundly, added
``assert inner_iter >= 2`` to protect LDS from the knob values that corrupt it --
but the task's ``compile_command`` shrinks the case to ``num_seqs=2`` to keep the
smoke test cheap, and at that shape ``inner_iter`` is 1 for every knob value, so
the assertion fired unconditionally. Thirteen iterations and seven KEEPs all
checked the full shape only; the arena's Step 1 was the first thing to run the
shrunk one, and the run scored FAIL. No instruction to the agent prevents this
class of failure -- it writes new self-protection code every day -- so the only
durable defence is that what forge checks before a KEEP is what the arena runs.

The SNR probe is forge's, and no scorer uses it. The authority for Step 2 is
``evaluate_correctness``, which carries no numeric criterion at all: it executes
the ``correctness_command`` list from the task's ``config.yaml`` and lets the
task's own tolerances decide. Those tolerances differ per task -- one kernel
asserts ``cos > 0.9995`` and ``rel_max < 0.02``, another uses
``atol=0.08, rtol=0.08`` -- so a single global SNR threshold cannot stand in for
any of them. One run held 33.4 dB, well over forge's 30 dB gate, while the task's
own suite measured a normalized max error of 0.02468 against its 0.02 limit;
forge kept optimizing on that kernel for fifteen more hours and scored zero.

This module reproduces the arena's verdict, never a more permissive one. The
correctness step's output scan looks redundant next to the arena's, which also
tests for ``correctness: pass`` -- but that inner test sits under a condition
requiring ``pass`` to be absent from the output, so it can never fire.

One divergence is deliberate and remains open: the arena passes
``extra_env=force_jit_rebuild(...)`` to both steps, which sets ``AITER_REBUILD=1``
and deletes the task op's stale compiled ``.so`` (see the arena's
``src/jit_rebuild.py``). It applies to aiter C/C++ tasks only -- ``apply_jit_rebuild``
returns ``{}`` for ``.py`` sources. Forge does not reproduce it, so on an aiter C++
task this gate can load a prebuilt ``.so`` and pass on code it did not build. That
is scoped separately; do not read this module as a complete reproduction.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from kernelforge.mcp_server.tools._subprocess import communicate_process_group

CANONICAL_CONFIG_FILENAME = "config.yaml"

# ``_DEFAULT_COMPILE_TIMEOUT_S`` and ``_DEFAULT_CORRECTNESS_TIMEOUT_S`` in the
# arena's evaluator. They are two separate keys there and happen to share a
# value; a task that declares neither is judged under both, so forge has to know
# each one to reproduce the same run.
ARENA_DEFAULT_COMPILE_TIMEOUT_SEC = 3600
ARENA_DEFAULT_CORRECTNESS_TIMEOUT_SEC = 3600

# The suite's failure text is what the agent reads instead of "SNR=33.4dB PASS",
# and the assertion that names the exceeded tolerance -- or the shape the kernel
# refused to compile for -- is the last thing a task runner prints.
_OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class CanonicalCorrectnessResult:
    """The arena's verdict on this candidate, or the reason there is none.

    ``unverified_reason`` is empty when the suite actually ran. Otherwise no
    canonical suite exists for this workspace, ``passed`` carries no evidence,
    and the caller must say so rather than report a check that passed.
    """

    passed: bool
    detail: str
    output: str = ""
    unverified_reason: str = ""
    # "timeout" when the suite was killed, so the run's decision label separates
    # a candidate the arena rejected from one it never finished judging.
    outcome: str = ""


@dataclass(frozen=True)
class _CompileStep:
    """The arena's Step 1: ``evaluate_compilation``.

    Judged by exit status alone. ``evaluate_correctness`` scans the output for
    "fail" as well; this step does not, and adding a scan here would reject
    candidates the arena admits -- a compiler is entitled to print the word
    "failure" in a warning and still produce a binary.
    """

    commands: tuple[str, ...]
    timeout_sec: int

    label: ClassVar[str] = "compilation"

    def reports_failure(self, output: str) -> bool:
        return False


@dataclass(frozen=True)
class _CorrectnessStep:
    """The arena's Step 2: ``evaluate_correctness``.

    Judged by exit status and by the output scan below, because a task runner
    that prints per-case verdicts and exits 0 is common enough that the arena
    reads the text too.
    """

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
        # The arena's absent-command branch returns a failure, not a skip:
        # "No compile_command specified" / "No correctness_command specified".
        return f"{path} declares no {key!r}"
    # The arena iterates this value directly, so a bare string would be run one
    # character at a time. Refusing it is the same verdict by a clearer route.
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
    """Return the declared suite, the reason it is unusable, or None if absent.

    None means this workspace ships no ``config.yaml``: the task was not
    prepared by the arena and has no canonical suite to run. A string means the
    file is there and forge cannot get a suite out of it, which is the case the
    arena fails outright rather than skips.

    Both steps are resolved from the one parse before either runs, so a task
    whose ``correctness_command`` is malformed is rejected on the declaration
    rather than after paying for a compile that was never going to be judged.
    """
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
    """Run the arena's Step 1 then Step 2 and stop at the first failure.

    ``timeout_cap_sec`` bounds a declared timeout that would otherwise let one
    candidate consume most of a campaign; it clamps both steps, and clamping can
    only turn a pass into a failure, never the reverse.
    """
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
    """Judge a candidate every other gate has already accepted.

    Call this from anywhere a kernel can become the incumbent or be published as
    a run's result, and act on ``passed``: an iteration's KEEP, a KB warm-start's
    adoption, and anything later that joins them. ``candidate_label`` names the
    path in the printed verdict, since a campaign log holds verdicts from more
    than one of them.

    ``detail`` names the step that failed, because "your kernel does not
    compile" and "your kernel is not accurate enough" send the agent to
    completely different edits.

    The suite is the expensive check, so call it last -- only for a candidate
    that would otherwise be accepted, never as a pre-filter.
    """
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
