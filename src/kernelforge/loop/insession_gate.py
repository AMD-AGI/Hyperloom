# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-session self-correction gate for the forge-loop agent.

Turns each forge-loop iteration's edit into a *self-closing* Agent session: the
agent Edits -> builds/tests -> reads the error -> Edits again, all within one
session. When the agent tries to end its turn, a ``Stop`` hook decides, in this
order:

  1. HARNESS PROTECTION (security, first). If a protected measurement file (the
     driver / test harness / config) changed and was not restored, BLOCK and
     feed the diff back so the agent restores it. Bounded by ``max_stop_blocks``
     (mirrors ``profile_driver._AdaptStopGate``): past the cap the gate stops
     fighting a non-cooperating agent, allows the stop with
     ``end_reason = "harness_tampered"``, and hands off -- the outer
     IterationLoop force-REVERTs a tampered candidate (files the driver-only
     ``_validate_driver_integrity`` does not cover), so a gamed measurement can
     never be KEPT.

  2. SELF-CORRECTION. Otherwise the hook runs the loop's CANONICAL validation
     (correctness + benchmark) on whatever is on disk:
       * NOT correct                       -> BLOCK ("fix it, keep going")
       * correct but NOT faster than best  -> BLOCK ("try a different opt")
       * correct AND mean(measurements) >= best + t * sigma / sqrt(n) -> ALLOW
         (``end_reason="converged"``)
     In a correctness-only phase (``correctness_only=True``, e.g. the PORT phase)
     a correct kernel alone ALLOWS the stop; the perf gate is skipped entirely.
     This prevents "fake exits": the agent cannot end by merely claiming success
     -- this gate re-checks with the SAME measurement the outer loop uses.

  3. BUDGET. Bounded by ``max_blocks`` blocked stops: once the gate has BLOCKed
     this many non-converging stops it allows the next one
     (``end_reason="block_budget_exhausted"``) and hands off to the outer
     IterationLoop, which re-validates and keep/reverts. This clean allow-path is
     what lets the provider RESUME the session afterwards to write a full lesson
     (a session killed by the SDK turn cap raises instead, losing the resume
     handle). The gate never gets the final word and can never hang the session
     to the SDK turn cap. Edits are still counted (``edit_count``) for logging
     but no longer bound the budget — a session making steady real progress
     should not be cut off just for editing a lot.

Backends with lifecycle-hook support translate ``make_agent_hooks`` into their
native callback representation. ``make_agent_hooks(stop_check=False)`` installs
the harness protection above without the Stop hook, for a session that must not
benchmark: an Implementer lane runs beside its siblings while the device times
one thing at a time, so its candidate is measured later, once, by the loop.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import sys
from typing import Any

from kernelforge.llm.workspace_policy import (
    PROTECTED_DIRS,
    PROTECTED_GLOBS,
    is_protected_path,
    protected_path_inventory,
)
from kernelforge.llm.git import git
from kernelforge.loop.jit_rebuild import force_jit_rebuild_for_changes
from kernelforge.loop.scoring import (
    KEEP_MEASUREMENT_COUNT,
    keep_score,
    passes_keep_threshold,
    required_keep_speedup,
)
from kernelforge.mcp_server.tools.test import test_correctness
from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    calculate_measurement_case_speedups,
    measure_wallclock,
)


# Files the agent must NOT modify: the test harness / driver that MEASURES the
# kernel. Editing these would let the agent game the metric, so a PreToolUse
# hook denies any write to them. Protection is by EXACT PATH for the driver
# (passed in via --driver) plus the basename globs below, which catch the test
# harness / perf helpers that live next to the kernel.
_DEFAULT_PROTECTED_GLOBS = list(PROTECTED_GLOBS)

# Directories that belong to the benchmark harness rather than the kernel
# implementation. Source files listed as explicit targets remain editable.
_DEFAULT_PROTECTED_DIRS = set(PROTECTED_DIRS)

# Tools that modify files on disk (subject to the protected-file deny + counted
# as edits when they target the kernel).
_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
_EDIT_TOOL_MATCHER = "|".join(_EDIT_TOOLS)

# Shell verbs that WRITE their path arguments (vs reading / executing them). A
# protected path is only a real write target when it is an ARGUMENT to one of these
# within a simple command (matched per-command in _bash_deny_reason).
_BASH_WRITE_VERBS = frozenset(
    {
        "rm",
        "rmdir",
        "mv",
        "cp",
        "tee",
        "truncate",
        "install",
        "dd",
        "shred",
        "chmod",
        "chown",
        "ln",
    }
)
# Wrappers to see through when locating a simple command's real verb.
_BASH_CMD_WRAPPERS = frozenset(
    {
        "env",
        "timeout",
        "sudo",
        "nohup",
        "nice",
        "ionice",
        "stdbuf",
        "command",
        "exec",
        "time",
        "xargs",
    }
)
# A shell variable assignment, which is what precedes a command's verb. Anchored
# to the name grammar so an option that merely carries a value -- ``--unset=FOO``
# -- is not read as one.
_SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# In-process (python/perl) WRITE APIs — there is no shell verb to key off, so these
# are detected textually and paired with a protected-file mention below.
_INLINE_WRITE_INTENT = re.compile(
    r"\b(?:write_text|write_bytes)\b"
    r"|\bshutil\.(?:copy\w*|move|rmtree)\b"
    r"|\b(?:open|io\.open|Path\s*\([^)]*\)\.open)\s*\("
    r"|\b(?:os\.)?(?:rename|replace)\s*\("
    r"|\bPath\s*\([^)]*\)\.(?:rename|replace)\s*\(",
    re.IGNORECASE,
)
_BASH_REDIRECT_TARGET = re.compile(r"(?:^|\s)(?<!<)(?:\d*>>?|\d*>\||&>>?)\s*([^\s;&|]+)")
_PYTHON_HEREDOC = re.compile(
    r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\1(?:\s|$)",
    re.DOTALL,
)
_PYTHON_COMMAND_PREFIX = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_./-]*/)?python"
    r"(?:\d+(?:\.\d+)*)?\s+"
    r"(?:-[A-Za-z][^\s;&|]*\s+)*"
    r"-c(?=\s)",
)
# An interpreter as a simple command's verb, which _simple_commands has already
# reduced to a basename, so `/usr/bin/python3.11` arrives here as `python3.11`.
_PYTHON_VERB = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
# A shell whose ``-c`` argument is another command line rather than one of its
# own. Not a _BASH_CMD_WRAPPERS entry: those pass their trailing words through
# as a command, whereas this carries one inside a single word.
_SHELL_VERBS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


def _unquote(token: str) -> str:
    """Strip surrounding whitespace and one layer of shell quoting."""
    return token.strip().strip("'\"")


def _short_option_value(args: Sequence[str], letter: str) -> str:
    """What a short option carries, however it was written, or "".

    POSIX short options cluster and may carry their value attached, so one
    option arrives as any of ``-c cmd``, ``-ccmd``, ``-lc cmd`` and ``-lccmd``.
    Reading only the bare token sees ``bash -lc`` as a shell that was given no
    command and ``python3 -mforge_driver`` as an interpreter that was given no
    module, and in both cases the driver run inside is never looked at.
    """
    for index, arg in enumerate(args):
        if not arg.startswith("-") or arg.startswith("--"):
            continue
        cluster = arg[1:]
        position = cluster.find(letter)
        if position < 0:
            continue
        attached = cluster[position + 1 :]
        if attached:
            return attached
        return args[index + 1] if index + 1 < len(args) else ""
    return ""


def _operator_segments(text: str) -> Iterator[str]:
    """Split shell text on the operators that separate commands, respecting quotes.

    Falls back to the plain split when the text cannot be tokenized -- an
    unbalanced quote is not a shape any rule here can reason about, and a
    coarser split errs toward offering more verb positions rather than fewer.
    """
    for line in text.split("\n"):
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        segment: list[str] = []
        try:
            for token in lexer:
                if token and set(token) <= {"&", "|", ";"}:
                    yield shlex.join(segment) if segment else ""
                    segment = []
                    continue
                segment.append(token)
        except ValueError:
            yield from re.split(r"(?:&&|\|\||[;|&])", line)
            continue
        yield shlex.join(segment) if segment else ""


def _simple_commands(text: str) -> Iterator[tuple[str, list[str]]]:
    """Split shell text into simple commands, each as its verb and arguments.

    Splitting on the operators that separate commands is what lets a rule match
    a verb against ITS OWN arguments: ``rm -rf ~/.flydsl; python3 driver.py``
    clears a cache and RUNS the driver, and reading the line as one command
    would call it a write to the driver merely because the driver is named on
    it. Leading assignments and wrappers (``env X=1 timeout 60 sudo ...``) are
    skipped so the yielded verb is the one that acts.

    A segment that opened with one of those also yields every later word as a
    verb position, each carrying the words after it. Skipping to exactly one
    verb needs the option grammar of every wrapper -- ``env -u FOO tee driver``
    and ``timeout --signal=KILL 60 tee driver`` both hid the write behind an
    option whose argument is not an option -- and the caller acts only where a
    write verb meets a protected path, so offering the positions costs less than
    parsing each wrapper and misses nothing when a new wrapper is added.

    The operators are found with the shell's own quoting rules rather than by a
    regex over the raw text. ``pgrep -af "a.py|b.py"`` carries a pipe inside one
    argument, and cutting there turns the tail of that string into a command
    whose verb is a file the caller never ran.
    """
    for segment in _operator_segments(text):
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        index = 0
        while index < len(words) and (
            _SHELL_ASSIGNMENT.match(words[index]) or os.path.basename(_unquote(words[index])) in _BASH_CMD_WRAPPERS
        ):
            index += 1
        if index >= len(words):
            continue
        yield os.path.basename(_unquote(words[index])), words[index + 1 :]
        if index:
            for position in range(index + 1, len(words)):
                yield (
                    os.path.basename(_unquote(words[position])),
                    words[position + 1 :],
                )


def _python_command_payloads(command: str) -> list[tuple[str, int, int]]:
    """Extract ``python -c`` payloads with a linear quoted-string scan."""
    payloads: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        match = _PYTHON_COMMAND_PREFIX.search(command, cursor)
        if match is None:
            break
        start = match.end()
        while start < len(command) and command[start].isspace():
            start += 1
        if start >= len(command):
            break

        quote = command[start] if command[start] in {"'", '"'} else ""
        content_start = start + 1 if quote else start
        end = content_start
        escaped = False
        while end < len(command):
            char = command[end]
            if quote == "'":
                if char == "'":
                    break
            elif quote == '"':
                if char == '"' and not escaped:
                    break
                if char == "\\" and not escaped:
                    escaped = True
                    end += 1
                    continue
                escaped = False
            elif char.isspace() or char in ";&|\n":
                break
            end += 1

        if quote and (end >= len(command) or command[end] != quote):
            cursor = start + 1
            continue
        payload_end = end + 1 if quote else end
        raw = command[start:payload_end]
        try:
            source = shlex.split(raw)[0]
        except (ValueError, IndexError):
            cursor = max(payload_end, start + 1)
            continue
        payloads.append((source, content_start, end))
        cursor = payload_end
    return payloads


# Max times the Stop hook will BLOCK a stop for an unrestored protected-harness
# change before it gives up, allows the stop, and hands off to the outer loop's
# force-REVERT. Bounds the block loop so a non-cooperating agent can never burn
# the whole SDK turn budget. Mirrors ``profile_driver._ADAPT_MAX_STOP_BLOCKS``.
_MAX_STOP_BLOCKS = 3


@dataclass(frozen=True)
class _ProtectedFileState:
    """Restorable state for one protected filesystem path."""

    path: Path
    kind: str
    content: bytes
    mode: int
    digest: str
    error: str = ""


def _python_write_targets(source: str) -> tuple[set[str], bool, bool]:
    """Return resolved write targets, ambiguity, and whether writes were found."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), True, bool(_INLINE_WRITE_INTENT.search(source))

    constants: dict[str, str] = {}
    targets: set[str] = set()
    ambiguous = False
    found_write = False

    def resolve(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
            return resolve(node.args[0])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = resolve(node.left)
            right = resolve(node.right)
            if left is not None and right is not None:
                return str(Path(left) / right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"resolve", "absolute"}
        ):
            return resolve(node.func.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and node.func.value.attr == "path"
            and node.func.attr == "join"
        ):
            parts = [resolve(arg) for arg in node.args]
            if parts and all(part is not None for part in parts):
                return os.path.join(*(part for part in parts if part is not None))
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = resolve(node.value)
            names = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None:
                for target in names:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
        if not isinstance(node, ast.Call):
            continue
        is_builtin_open = isinstance(node.func, ast.Name) and node.func.id == "open"
        is_io_open = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "io"
            and node.func.attr == "open"
        )
        is_path_open = isinstance(node.func, ast.Attribute) and node.func.attr == "open" and not is_io_open
        if is_builtin_open or is_io_open or is_path_open:
            positional_mode_index = 0 if is_path_open else 1
            mode_node = (
                node.args[positional_mode_index]
                if len(node.args) > positional_mode_index
                else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "mode"),
                    None,
                )
            )
            mode = resolve(mode_node) if mode_node is not None else "r"
            if mode is None:
                found_write = True
                ambiguous = True
                continue
            if not any(flag in mode for flag in "wax+"):
                continue
            found_write = True
            target = resolve(node.func.value if is_path_open else (node.args[0] if node.args else None))
            if target is None:
                ambiguous = True
            else:
                targets.add(target)
            continue
        if isinstance(node.func, ast.Attribute):
            attribute = node.func.attr
            if attribute in {"write_text", "write_bytes", "unlink", "rmdir"}:
                found_write = True
                target = resolve(node.func.value)
                if target is None:
                    ambiguous = True
                else:
                    targets.add(target)
            elif (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "shutil"
                and attribute in {"copy", "copy2", "copyfile", "move", "rmtree"}
            ):
                found_write = True
                target_nodes = node.args[:1] if attribute == "rmtree" else node.args[:2]
                for target_node in target_nodes:
                    target = resolve(target_node)
                    if target is None:
                        ambiguous = True
                    else:
                        targets.add(target)
            elif (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and attribute in {"rename", "replace"}
            ):
                found_write = True
                for target_node in node.args[:2]:
                    target = resolve(target_node)
                    if target is None:
                        ambiguous = True
                    else:
                        targets.add(target)
            elif attribute in {"rename", "replace"}:
                found_write = True
                for target_node in (node.func.value, *node.args[:1]):
                    target = resolve(target_node)
                    if target is None:
                        ambiguous = True
                    else:
                        targets.add(target)
    return targets, ambiguous, found_write


class InSessionGate:
    """Per-iteration harness-protection gate driving the Stop hook.

    A fresh instance is created for every outer iteration (state is per-session).
    """

    def __init__(
        self,
        driver_script: str,
        snr_threshold: float,
        baseline_case_times: dict | None = None,
        best_mean_case_speedup: float | None = None,
        kernel_file: str = "",
        max_blocks: int = 10,
        stage_timeout_sec: int = 1800,
        bench_timeout_sec: int = 300,
        max_stop_blocks: int = _MAX_STOP_BLOCKS,
        protected_globs: list[str] | None = None,
        target_files: list[str] | None = None,
        extra_protected_globs: list[str] | None = None,
        extra_protected_paths: list[str] | None = None,
        correctness_only: bool = False,
        bench_repeat: int = 1,
        interposed_driver_path: str | None = None,
        workspace: str | Path | None = None,
    ):
        self.driver_script = driver_script
        # The wrapper script this session must reach the driver through, when a
        # caller interposed one. A path rather than a command line: it is
        # compared by basename and named in the prompt as an interpreter's
        # argument. Empty for every ordinary session, which runs the driver
        # itself and is left exactly as it was.
        self.interposed_driver_path = interposed_driver_path or ""
        self.snr_threshold = snr_threshold
        self.bench_repeat = bench_repeat
        self.baseline_case_times = dict(baseline_case_times or {})
        self.best_mean_case_speedup = best_mean_case_speedup
        # Correctness-only phase (e.g. PORT): the gate requires ONLY correctness and
        # never runs the perf gate (no benchmark; best score unused).
        self.correctness_only = correctness_only
        # Attempt budget for the correctness/perf self-correction loop. After
        # this many BLOCKed stops the gate allows the next one so the session
        # ends cleanly (resumable -> full lesson) instead of grinding to the SDK
        # turn cap. Edit count is observability only and never bounds the session.
        self.max_blocks = max_blocks
        self.stage_timeout_sec = stage_timeout_sec
        self.bench_timeout_sec = bench_timeout_sec
        # Upper bound on Stop-hook blocks for unrestored harness tampering.
        self.max_stop_blocks = max_stop_blocks

        # Target file set used to count edits for observability. A single-file
        # task passes only ``kernel_file``; a repository task passes the whole
        # ``target_files`` set. ``kernel_file`` is always included as the anchor.
        targets = list(target_files) if target_files else []
        if kernel_file:
            targets.append(kernel_file)
        self.target_abs = {os.path.normpath(os.path.abspath(f)) for f in targets if f}
        # Kept for logging/back-compat (the anchor file).
        self.kernel_abs = os.path.normpath(os.path.abspath(kernel_file)) if kernel_file else ""
        self.kernel_base = os.path.basename(kernel_file) if kernel_file else ""

        # Per-session mutable state.
        self.edit_count = 0
        self.block_count = 0
        # Harness-protection blocks only (bounded separately by max_stop_blocks),
        # so a non-cooperating tamperer is capped independently of the perf loop.
        self.harness_block_count = 0
        self.passed = False
        self.last_wall_ms: float | None = None
        self.last_mean_case_speedup: float | None = None
        self.last_bench_result: dict | None = None
        self.last_reason = ""
        # Why the gate ALLOWED the session to stop (set once, at an allow path):
        #   "converged"              — correct AND faster than best; a real win.
        #   "block_budget_exhausted" — max_blocks blocked stops spent; hand off to
        #                              the outer loop to re-validate + keep/revert.
        #   "harness_tampered"       — harness block cap hit on unrestored protected
        #                              changes; the outer loop force-REVERTs it.
        #   "validation_timeout"     — full-suite correctness timed out; the outer
        #                              loop performs the one authoritative retry.
        #   "gate_error"             — the gate itself raised; fail OPEN.
        # Stays "" if the SDK terminated the session before any Stop hook fired
        # (e.g. the turn cap), which the caller detects separately.
        self.end_reason = ""
        # Real failure signals seen this session (block reasons: compile errors,
        # "correct but not faster", …). Consumed by the ExperienceLedger so the
        # next iteration learns from them instead of repeating the mistake.
        self.findings: list[str] = []

        # Protected (measurement) files — the driver (passed in via --driver) is
        # matched by EXACT absolute path; the test harness / perf helpers next to
        # the kernel are caught by the basename globs below.
        self.protected_abs: set[str] = set()
        if driver_script:
            self.protected_abs.add(os.path.normpath(os.path.abspath(driver_script)))
        # Additional exact-path measurement files. The rewrite PORT phase adds the
        # source kernel it ports FROM here: the driver imports it as the live
        # correctness oracle + baseline, so it gets the SAME tier as the driver —
        # matched by exact absolute path (not a fragile basename glob) and always
        # snapshotted for the stop-time change check.
        for p in extra_protected_paths or []:
            if p:
                self.protected_abs.add(os.path.normpath(os.path.abspath(p)))
        self.workspace_root = self._infer_workspace_root(workspace, driver_script, kernel_file)

        globs = list(protected_globs) if protected_globs is not None else list(_DEFAULT_PROTECTED_GLOBS)
        # Repository tasks ship the reference/test implementation INSIDE the repo
        # tree (e.g. AITER's op_tests/.../test_*.py provides the correctness
        # reference), which the default globs above do not catch. The caller
        # passes extra globs so the agent cannot edit the reference to game the
        # SNR/allclose gate. Protected status always wins over source hints.
        if extra_protected_globs:
            globs += list(extra_protected_globs)
        self.protected_globs = list(dict.fromkeys(globs))  # dedup, keep order
        (
            self._protected_baseline,
            self._protected_snapshot_errors,
        ) = self._snapshot_protected_states()
        self._protected_snapshot = {
            key: state.digest
            for key, state in self._protected_baseline.items()
            if state.kind in {"file", "symlink"} and not state.error
        }
        self._last_protected_states = dict(self._protected_baseline)
        self.integrity_verdict = "violation" if self._protected_snapshot_errors else "unknown"
        self.integrity_reason = "; ".join(self._protected_snapshot_errors)
        self.integrity_violation = bool(self._protected_snapshot_errors)

    def findings_blob(self) -> str:
        """Joined findings for the experience ledger (most-recent-last)."""
        return "\n---\n".join(self.findings)

    @property
    def hook_timeout_sec(self) -> int:
        """Total Stop-hook ceiling for correctness, optional bench, and cleanup."""
        benchmark_budget = 0 if self.correctness_only else KEEP_MEASUREMENT_COUNT * self.bench_timeout_sec
        return self.stage_timeout_sec + benchmark_budget + 120

    def count_target_edits(self, cwd: str, file_changes: list[str]) -> int:
        """Count changed tracked implementation paths outside the protected set."""
        root = os.path.abspath(cwd or ".")
        total = 0
        for relative in file_changes:
            if not relative:
                continue
            candidate = relative if os.path.isabs(relative) else os.path.join(root, relative)
            if not self._is_protected(candidate):
                total += 1
        return total

    # ── hook wiring ──────────────────────────────────────────────────────────
    def make_agent_hooks(self, *, stop_check: bool = True):
        """Build provider-neutral lifecycle hooks for capable backends.

        The protection hooks are the same in both modes, and both read the one
        protected-path rule this instance was built with (:meth:`_is_protected`,
        which delegates to :func:`kernelforge.llm.workspace_policy.is_protected_path`).

        ``stop_check=False`` omits the Stop hook, so nothing in the session runs
        correctness or a benchmark. A caller whose sessions run concurrently
        needs that: the Stop hook times the kernel, and a device that is timing
        one session cannot also be timing another. Such a session is never sent
        back to keep improving -- the gate has no say in when it ends -- so its
        candidate is judged only by whoever measures it afterwards.
        """
        from kernelforge.agent_backends.base import AgentHook, AgentHooks

        return AgentHooks(
            pre_tool_use=[
                AgentHook(
                    matcher=_EDIT_TOOL_MATCHER,
                    callback=self._on_pre_edit,
                ),
                AgentHook(
                    matcher="Bash",
                    callback=self._on_pre_bash,
                ),
            ],
            # Count every non-protected implementation edit. Declared source files
            # are orientation hints, not the edit boundary.
            post_tool_use=[
                AgentHook(
                    matcher=_EDIT_TOOL_MATCHER,
                    callback=self._on_edit,
                )
            ],
            # Stop: harness protection, then canonical correctness+bench self-check.
            # Timeout covers the protected-path scan plus one GPU validation pass.
            stop=(
                [
                    AgentHook(
                        matcher="",
                        callback=self._on_stop,
                        timeout_sec=self.hook_timeout_sec,
                    )
                ]
                if stop_check
                else []
            ),
        )

    # ── path helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _edited_path(input_data: dict) -> str:
        ti = input_data.get("tool_input") or {}
        return ti.get("file_path") or ti.get("path") or ti.get("notebook_path") or ""

    @staticmethod
    def _bash_command(input_data: dict) -> str:
        ti = input_data.get("tool_input") or {}
        return ti.get("command") or ""

    @staticmethod
    def _infer_workspace_root(workspace: str | Path | None, driver_script: str, kernel_file: str) -> Path | None:
        """Resolve the tree this gate measures, preferring the declared one.

        The caller's ``--workspace`` is authoritative when it is given: it is
        the agent's own cwd, so it is the root the agent's relative tool paths
        are relative to, the root ``protected_path_inventory`` must scan, and
        the repository ``git diff HEAD`` runs in. Inferring it from the driver
        instead only happens to work when the driver sits inside that tree.
        ``forge-fuse`` writes its driver into the run's ``--output-dir``, so the
        inferred root was the output dir -- not a repository at all, which made
        ``git diff HEAD -- .`` fall into git's implicit ``--no-index`` mode and
        fail with ``Could not access 'HEAD'`` on every stop, and pointed the
        protected-file snapshot at a tree holding none of the protected files.
        """
        for candidate in (workspace, driver_script, kernel_file):
            if not candidate:
                continue
            try:
                p = Path(candidate).resolve()
                if p.exists():
                    return p.parent if p.is_file() else p
            except Exception:
                continue
        return None

    def _is_protected(self, fp: str) -> bool:
        if not fp:
            return False
        return is_protected_path(
            fp,
            workspace=self.workspace_root,
            exact_paths=self.protected_abs,
            extra_globs=self.protected_globs,
        )

    def _is_protected_dir_path(self, fp: str) -> bool:
        """Back-compatible directory-only protected-path probe."""

        if not fp:
            return False
        path = Path(fp)
        if self.workspace_root and path.is_absolute():
            try:
                path = path.resolve().relative_to(self.workspace_root)
            except ValueError:
                path = path.resolve()
        return any(part.lower() in _DEFAULT_PROTECTED_DIRS for part in path.parts[:-1])

    def _iter_snapshot_paths(self) -> list[Path]:
        root = self.workspace_root
        if root is None:
            return sorted(Path(path) for path in self.protected_abs)
        return list(
            protected_path_inventory(
                root,
                exact_paths=self.protected_abs,
                extra_globs=self.protected_globs,
            )
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _candidate_diff_sha256(self) -> str:
        """Fingerprint the exact tracked candidate measured by this gate."""
        if self.workspace_root is None:
            raise RuntimeError("candidate diff fingerprint requires a workspace")
        diff = git("diff", "HEAD", "--", ".", cwd=self.workspace_root).stdout
        return hashlib.sha256(diff.encode()).hexdigest()

    def _protected_key(self, path: Path) -> str:
        root = self.workspace_root
        try:
            return str(path.relative_to(root)) if root is not None else str(path)
        except ValueError:
            return str(path)

    @staticmethod
    def _capture_protected_state(path: Path) -> _ProtectedFileState:
        """Capture a protected path without following symbolic links."""

        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return _ProtectedFileState(path, "missing", b"", 0, "")
        except OSError as error:
            return _ProtectedFileState(
                path,
                "error",
                b"",
                0,
                "",
                f"could not inspect protected path {path}: {error}",
            )

        mode = stat.S_IMODE(metadata.st_mode)
        try:
            if stat.S_ISLNK(metadata.st_mode):
                content = os.readlink(path).encode(errors="surrogateescape")
                kind = "symlink"
            elif stat.S_ISREG(metadata.st_mode):
                content = path.read_bytes()
                kind = "file"
            elif stat.S_ISDIR(metadata.st_mode):
                content = b""
                kind = "directory"
            else:
                return _ProtectedFileState(
                    path,
                    "error",
                    b"",
                    mode,
                    "",
                    f"unsupported protected path type: {path}",
                )
        except OSError as error:
            return _ProtectedFileState(
                path,
                "error",
                b"",
                mode,
                "",
                f"could not read protected path {path}: {error}",
            )
        return _ProtectedFileState(
            path,
            kind,
            content,
            mode,
            hashlib.sha256(content).hexdigest(),
        )

    def _snapshot_protected_states(
        self,
        *,
        include_baseline: bool = False,
    ) -> tuple[dict[str, _ProtectedFileState], list[str]]:
        out: dict[str, _ProtectedFileState] = {}
        errors: list[str] = []
        try:
            paths = set(self._iter_snapshot_paths())
        except Exception as error:  # noqa: BLE001 - an incomplete scan is a verdict
            paths = set()
            errors.append(f"protected inventory scan failed: {type(error).__name__}: {error}")
        if include_baseline:
            paths.update(state.path for state in self._protected_baseline.values())
        for path in sorted(paths, key=str):
            state = self._capture_protected_state(path)
            key = self._protected_key(path)
            out[key] = state
            if state.error:
                errors.append(state.error)
        return out, errors

    def _snapshot_protected_files(self) -> dict[str, str]:
        """Return the current digest view retained for compatibility and tests."""

        states, _errors = self._snapshot_protected_states(
            include_baseline=hasattr(self, "_protected_baseline"),
        )
        return {
            key: state.digest for key, state in states.items() if state.kind in {"file", "symlink"} and not state.error
        }

    def _protected_changes(self) -> str:
        current, current_errors = self._snapshot_protected_states(
            include_baseline=True,
        )
        self._last_protected_states = current
        before = self._protected_baseline
        errors = [*self._protected_snapshot_errors, *current_errors]
        modified: list[str] = []
        deleted: list[str] = []
        added: list[str] = []

        for key in sorted(set(before) | set(current)):
            old = before.get(key)
            new = current.get(key)
            if old is None:
                if new is not None and new.kind != "missing":
                    added.append(key)
                continue
            if old.error:
                continue
            if new is None or new.kind == "missing":
                if old.kind != "missing":
                    deleted.append(key)
                continue
            if new.error:
                continue
            if old.kind == "missing":
                added.append(key)
            elif old.kind != new.kind or old.digest != new.digest or old.mode != new.mode:
                modified.append(key)

        if not (modified or deleted or added or errors):
            return ""
        parts = []
        if modified:
            parts.append(f"modified={modified[:5]}")
        if deleted:
            parts.append(f"deleted={deleted[:5]}")
        if added:
            parts.append(f"added={added[:5]}")
        if errors:
            parts.append(f"errors={errors[:5]}")
        return "; ".join(parts)

    def finalize_integrity(self) -> str:
        """Set the final protected-integrity verdict after an agent session."""

        try:
            reason = self._protected_changes()
        except Exception as error:  # noqa: BLE001 - scan failure is fail-closed
            reason = f"protected integrity scan failed: {type(error).__name__}: {error}"
        self.integrity_reason = reason
        self.integrity_violation = bool(reason)
        self.integrity_verdict = "violation" if reason else "clean"
        if reason:
            finding = f"Protected workspace integrity violation: {reason}"
            if finding not in self.findings:
                self.findings.append(finding[:1200])
        return reason

    @staticmethod
    def _remove_filesystem_path(path: Path) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()

    def restore_protected_files(self) -> None:
        """Restore the complete protected inventory to its pre-session state."""

        current, current_errors = self._snapshot_protected_states(
            include_baseline=True,
        )
        if self._protected_snapshot_errors or current_errors:
            raise RuntimeError(
                "cannot restore an incompletely snapshotted protected inventory: "
                + "; ".join([*self._protected_snapshot_errors, *current_errors])
            )

        baseline = self._protected_baseline
        for key in sorted(
            set(current) - set(baseline),
            key=lambda value: len(Path(value).parts),
            reverse=True,
        ):
            added = current[key]
            if added.kind != "missing":
                self._remove_filesystem_path(added.path)

        for key, state in baseline.items():
            if state.error:
                raise RuntimeError(state.error)
            current_state = current.get(key)
            if current_state == state:
                continue
            path = state.path
            if state.kind == "missing":
                self._remove_filesystem_path(path)
                continue
            self._remove_filesystem_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if state.kind == "file":
                path.write_bytes(state.content)
                path.chmod(state.mode)
            elif state.kind == "symlink":
                os.symlink(
                    state.content.decode(errors="surrogateescape"),
                    path,
                )
            elif state.kind == "directory":
                path.mkdir(parents=True, exist_ok=True)
                path.chmod(state.mode)
            else:
                raise RuntimeError(f"unsupported protected snapshot kind: {state.kind}")

        remaining = self.finalize_integrity()
        if remaining:
            raise RuntimeError(f"protected files remain inconsistent after restoration: {remaining}")

    def _bash_deny_reason(self, command: str) -> str:
        """WHY a Bash command is denied (a short trigger reason), or "" to allow.

        Denies ONLY when a protected measurement file is an actual WRITE TARGET:
          * an output-redirect destination (``> f``, ``2> f``, ``&> f``), or
          * a path ARGUMENT to a shell write verb (rm/mv/cp/sed -i/tee/...) within
            the SAME simple command, or
          * a protected file named alongside an in-process write API
            (``open(...,'w')`` / ``write_text`` / ``shutil.copy|move|rmtree``).

        Merely EXECUTING a protected file (``python3 rewrite_driver.py``), or writing
        a NON-protected path on the same line (``rm -rf ~/.flydsl`` cache;
        ``sed -i ... flydsl/kernel.py`` the editable kernel) is allowed here --
        a session given an interposed driver command is answered separately by
        :meth:`_bash_bypass_reason`. The Stop-hook
        protected-file hash check is the authoritative backstop for anything a text
        heuristic misses. Returning the reason (not a bool) lets ``_on_pre_bash`` log
        the exact trigger for false-positive review.
        """
        if not command:
            return ""

        def _safe_redirect_target(raw: str) -> bool:
            target = raw.strip().strip("'\"")
            return (
                not target
                or target == "/dev/null"
                or target.startswith("/tmp/")
                or target.startswith("$tmp")
                or target.startswith("${tmp")
            )

        # A path names a protected file if it resolves to one OR shares a basename
        # with one (agents `cd` into the workspace, so args are often relative and
        # would not resolve to the protected ABSPATH).
        protected_bases = {os.path.basename(p) for p in self.protected_abs}
        protected_bases |= {Path(k).name for k in self._protected_snapshot}

        def _names_protected(raw: str) -> bool:
            p = _unquote(raw)
            if not p:
                return False
            return self._is_protected(p) or os.path.basename(p) in protected_bases

        def _protected_mention(source: str) -> str:
            lowered = source.lower()
            for path in self.protected_abs:
                if path.lower() in lowered or Path(path).name.lower() in lowered:
                    return Path(path).name
            for relative in self._protected_snapshot:
                if relative.lower() in lowered or Path(relative).name.lower() in lowered:
                    return Path(relative).name
            return ""

        def _inspect_python(source: str) -> str:
            targets, ambiguous, found_write = _python_write_targets(source)
            if not found_write:
                return ""
            for target in targets:
                if _names_protected(target):
                    return f"inline write targets protected file '{os.path.basename(_unquote(target))}'"
            if ambiguous:
                mentioned = _protected_mention(source)
                if mentioned:
                    return f"inline write may modify protected file '{mentioned}'"
            return ""

        # Parse every Python payload independently. Heredoc bodies are removed
        # from the shell text after inspection so their Python tokens cannot be
        # mistaken for shell commands, and a safe payload cannot allow a later
        # unsafe command on the same Bash invocation.
        remainder = command
        heredoc_matches = list(_PYTHON_HEREDOC.finditer(command))
        for match in heredoc_matches:
            line_start = command.rfind("\n", 0, match.start()) + 1
            prefix = command[line_start : match.start()]
            if re.search(
                r"(?:^|\s)(?:[A-Za-z0-9_./-]*/)?python"
                r"(?:\d+(?:\.\d+)*)?(?:\s|$)",
                prefix,
            ):
                reason = _inspect_python(match.group(2))
                if reason:
                    return reason
        if heredoc_matches:
            chars = list(remainder)
            for match in heredoc_matches:
                for index in range(match.start(), match.end()):
                    if chars[index] != "\n":
                        chars[index] = " "
            remainder = "".join(chars)

        python_c_payloads = _python_command_payloads(remainder)
        for source, _start, _end in python_c_payloads:
            reason = _inspect_python(source)
            if reason:
                return reason
        if python_c_payloads:
            chars = list(remainder)
            for _source, start, end in python_c_payloads:
                chars[start:end] = " " * (end - start)
            remainder = "".join(chars)

        # (1) Output redirection whose DESTINATION is a protected file. Harmless
        # diagnostic redirects (`... 2>/dev/null | head`, `> /tmp/x`) are fine.
        for match in _BASH_REDIRECT_TARGET.finditer(remainder):
            target = match.group(1)
            if _safe_redirect_target(target):
                continue
            if _names_protected(target):
                return f"redirect writes protected path '{_unquote(target)}'"

        # (2) A shell write verb whose PATH ARGUMENT is a protected file. Split into
        # simple commands so each verb is matched to ITS OWN args — this is what keeps
        # `rm -rf ~/.flydsl; python3 rewrite_driver.py` (clear cache + RUN the driver)
        # and `sed -i ... flydsl/kernel.py` (edit the editable kernel) from being
        # misread as writing the driver just because the line also names it.
        for verb, args in _simple_commands(remainder):
            inplace_edit = verb in ("sed", "perl") and any(a == "-i" or a.startswith("-i") for a in args)
            if verb not in _BASH_WRITE_VERBS and not inplace_edit:
                continue
            for a in args:
                if a.startswith("-"):
                    continue
                if _names_protected(a):
                    return f"`{verb}` writes protected file '{os.path.basename(_unquote(a))}'"

        # (3) Non-Python in-process writes still need a conservative textual
        # backstop. Parsed Python source has been blanked out above.
        if _INLINE_WRITE_INTENT.search(remainder):
            mentioned = _protected_mention(remainder)
            if mentioned:
                return f"inline write may modify protected file '{mentioned}'"
        return ""

    def _bash_may_modify_protected(self, command: str) -> bool:
        """Back-compat boolean wrapper around :meth:`_bash_deny_reason`."""
        return bool(self._bash_deny_reason(command))

    def _bash_bypass_reason(self, command: str) -> str:
        """WHY a Bash command reaches the driver around its wrapper, or "".

        Only a session handed an interposed command has this rule. That command
        exists because nothing else in the chain can do what it does -- for a
        concurrent Implementer lane it takes the device lock, and the CLI, the
        shell it runs from and the driver are three separate processes, so the
        lock has to live in one of them. Naming it in the system prompt states
        the requirement; this is what holds it. A driver run that goes around it
        times this session against whichever sibling is benchmarking at that
        moment and corrupts that sibling's number too, which is the part no
        lesson can attribute to anything.

        Reading the driver stays allowed -- it is how a session learns what it
        is scored on -- so only a simple command that EXECUTES it is refused,
        whether as the verb itself, as an interpreter's script argument, or as
        the module an interpreter is pointed at with ``-m``.

        What no command rule reaches is a run that never names the driver: a
        script that invokes it, or a timing loop written inline. Those score
        nothing the loop reads, but they hold the device all the same.
        """
        driver_base = os.path.basename(self.driver_script)
        if not self.interposed_driver_path or not command or not driver_base:
            return ""
        driver_stem = os.path.splitext(driver_base)[0]
        run_base = os.path.basename(_unquote(self.interposed_driver_path))
        for verb, args in _simple_commands(command):
            if verb == run_base:
                continue
            if verb == driver_base:
                return f"`{verb}` runs the driver outside `{run_base}`"
            if verb in _SHELL_VERBS:
                # A nested shell carries its command line inside one word, so
                # the split above cannot see into it. Each nesting strips a
                # level, so the recursion is as deep as the command is nested.
                nested = self._bash_bypass_reason(_short_option_value(args, "c"))
                if nested:
                    return nested
                continue
            if not _PYTHON_VERB.match(verb):
                continue
            # An interpreter reaches the driver by module as readily as by
            # path, and `-m` names it without the suffix the path carries, so
            # the scan below cannot see it. Only the last component is compared
            # because a driver imported through a package is the same run.
            module = _short_option_value(args, "m")
            if module and module.rsplit(".", 1)[-1] == driver_stem:
                return f"`{verb} -m {module}` runs the driver outside `{run_base}`"
            # Every non-option word, not just the first. Reading only the first
            # needs the option grammar of the interpreter -- `python3 -W ignore
            # driver.py` and `python3 -X dev driver.py` each carry a value that
            # is not itself an option, and the driver sits one word further on
            # than the scan expected. The cost of offering every position is
            # refusing a command that merely named the driver, which costs the
            # session one retry against a message that says what to run
            # instead; the cost of missing one is a sibling lane's measurement.
            if any(os.path.basename(_unquote(arg)) == driver_base for arg in args if not arg.startswith("-")):
                return f"`{verb} {driver_base}` runs the driver outside `{run_base}`"
        return ""

    # ── hooks ────────────────────────────────────────────────────────────────
    async def _on_pre_edit(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """Deny any edit to a protected measurement file (harness/driver)."""
        if input_data.get("tool_name", "") not in _EDIT_TOOLS:
            return {}
        fp = self._edited_path(input_data)
        if self._is_protected(fp):
            base = os.path.basename(fp)
            self.findings.append(f"DENIED edit to protected measurement file: {base}")
            self._log(f"DENY edit to protected file {base}")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Editing `{base}` is NOT allowed — it is the test harness / driver that "
                        "measures your kernel. Modify the target kernel (and, if needed, its own "
                        "helper modules) only; changing the measurement is prohibited."
                    ),
                }
            }
        return {}

    def _deny_bash(self, command: str, *, reason: str, finding: str, told: str) -> dict:
        """One denial, logged with the command that triggered it.

        Observability: the ACTUAL command (single-lined + bounded) and the
        trigger reason, so a denied command can be reviewed later for false
        positives. Grep the run log for "DENY Bash".
        """
        cmd_1line = " ".join(command.split())[:500]
        self.findings.append(f"{finding}: {cmd_1line[:200]}")
        self._log(f"DENY Bash [{reason}]: {cmd_1line}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": told,
            }
        }

    async def _on_pre_bash(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """Deny shell writes to protected harness files, and driver runs that
        would go around the command this session must measure through."""
        if input_data.get("tool_name", "") != "Bash":
            return {}
        command = self._bash_command(input_data)
        reason = self._bash_deny_reason(command)
        if reason:
            return self._deny_bash(
                command,
                reason=reason,
                finding="DENIED Bash write to protected measurement files",
                told=(
                    "This shell command appears to modify protected benchmark "
                    "harness/config files. Run the harness for validation, but "
                    "do not edit it; modify only kernel implementation files."
                ),
            )
        reason = self._bash_bypass_reason(command)
        if reason:
            driver_base = os.path.basename(self.driver_script)
            return self._deny_bash(
                command,
                reason=reason,
                finding="DENIED driver run outside the interposed command",
                told=(
                    f"Run the driver as `python3 {self.interposed_driver_path}` "
                    f"instead. It passes every argument through to "
                    f"{driver_base} unchanged, writes nothing of its own and "
                    "returns its exit status — but it first takes a lock on the "
                    "GPU this session shares with others running right now. "
                    "Timing two kernels on one device at once corrupts both "
                    "numbers, including the ones this session is judged on. It "
                    "may sit silent before it starts; that wait is another "
                    "session's benchmark, not a hang."
                ),
            )
        return {}

    async def _on_edit(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        # Count every implementation edit outside the protected measurement set.
        if input_data.get("tool_name", "") in _EDIT_TOOLS and not self._is_protected(self._edited_path(input_data)):
            self.edit_count += 1
        return {}

    # ── decisions ────────────────────────────────────────────────────────────
    @staticmethod
    def _allow() -> dict:
        return {}

    def _block(self, reason: str) -> dict:
        self.block_count += 1
        self.last_reason = reason
        # Keep a trimmed record for the experience ledger (cap each entry).
        self.findings.append(reason.strip()[:1200])
        return {"decision": "block", "reason": reason}

    def _harness_block(self, reason: str) -> dict:
        """Block a stop for unrestored harness tampering (does not eat perf budget)."""
        self.last_reason = reason
        self.findings.append(reason.strip()[:1200])
        return {"decision": "block", "reason": reason}

    async def _on_stop(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        try:
            # ── 1) HARNESS PROTECTION (security, first) ───────────────────────
            protected_delta = self._protected_changes()
            self.integrity_reason = protected_delta
            self.integrity_violation = bool(protected_delta)
            self.integrity_verdict = "violation" if protected_delta else "clean"
            if protected_delta:
                # Keep fighting an unrestored harness change only up to the cap.
                # Past it, the agent is not cooperating: stop blocking (which would
                # otherwise burn turns to the SDK cap) and hand off with a signal
                # the outer loop turns into a forced REVERT — the tampered harness
                # can't be trusted to measure this candidate, so it must not KEEP.
                if self.harness_block_count >= self.max_stop_blocks:
                    self.end_reason = "harness_tampered"
                    self._log(
                        f"block cap {self.max_stop_blocks} reached on protected "
                        "harness changes -> allow stop, outer loop force-REVERTs"
                    )
                    return self._allow()
                self.harness_block_count += 1
                self._log(f"BLOCK {self.harness_block_count}/{self.max_stop_blocks} (protected harness changed)")
                return self._harness_block(
                    "Protected benchmark harness/config files changed. Restore "
                    "them before continuing; only kernel implementation files may "
                    f"be modified.\n\n{protected_delta}"
                )

            # ── 3) BUDGET (bounded so the gate never hangs the session) ───────
            # Checked before the (GPU-costly) canonical validation so an exhausted
            # session hands off immediately instead of paying for one more bench.
            # Budget is purely block-based: once the gate has BLOCKed max_blocks
            # non-converging stops, allow the next one. Ending on this clean
            # allow-path (rather than the SDK turn cap, which RAISES) is what keeps
            # the session resumable so the summarizer can write a full lesson.
            if self.block_count >= self.max_blocks:
                self.end_reason = "block_budget_exhausted"
                self._log(
                    f"block budget exhausted (blocks={self.block_count}/{self.max_blocks}, "
                    f"edits={self.edit_count}) -> allow stop, hand off to outer "
                    "canonical validation (session stays resumable for lesson)"
                )
                return self._allow()

            # ── 2) SELF-CORRECTION: canonical correctness + benchmark ─────────
            # Ensure the canonical check compiles the kernel the agent has on disk
            # RIGHT NOW: the SDK hook may run in a subprocess that did not inherit
            # the loop's AITER_REBUILD, so (re)assert it here (aiter HIP; no-op
            # otherwise).
            force_jit_rebuild_for_changes(
                self.workspace_root or Path.cwd(),
                [self.kernel_abs, *self.target_abs],
            )

            # 2a) Correctness — canonical driver, same call the pipeline uses.
            corr = await test_correctness(
                driver_script=self.driver_script,
                driver_args=[],
                snr_threshold=self.snr_threshold,
                timeout_sec=self.stage_timeout_sec,
            )
            if not corr.get("passed"):
                outcome = str(corr.get("outcome") or "correctness_failure")
                if outcome == "timeout":
                    self.end_reason = "validation_timeout"
                    finding = (
                        "Full-suite correctness timed out in the in-session gate; "
                        "handing the candidate to outer validation without blocking."
                    )
                    self.findings.append(finding)
                    self._log(f"ALLOW (validation timeout; outer loop will retry once) edit={self.edit_count}")
                    return self._allow()
                tail = corr.get("output") or corr.get("message") or "correctness failed"
                self._log(f"BLOCK (validation {outcome}) edit={self.edit_count}")
                goal = "correct" if self.correctness_only else "correct AND faster"
                return self._block(
                    "Your change is NOT finished: the kernel fails correctness. "
                    f"Fix the error below and keep going — do not stop until it is {goal}.\n\n"
                    f"{corr.get('message', '')}\n{str(tail)[-1400:]}"
                )

            # Correctness-only phase (e.g. PORT): there is no performance requirement,
            # so a CORRECT kernel is done. The perf gate does not apply here — skip the
            # benchmark entirely (running it would be wasted work, and a crashing bench
            # must never block a correct port).
            if self.correctness_only:
                self.passed = True
                self.end_reason = "converged"
                self._log(f"ALLOW (correct; correctness-only phase) edit={self.edit_count}")
                return self._allow()

            # 2b) Performance — three independent canonical measurements.
            self.last_bench_result = None
            bench = await measure_wallclock(
                driver_script=self.driver_script,
                driver_args=[],
                measurements=KEEP_MEASUREMENT_COUNT,
                timeout_sec=self.bench_timeout_sec,
                repeat=self.bench_repeat,
            )
            wall = bench.get("median_ms")
            if not bench.get("success"):
                self.last_wall_ms = None
                self.last_mean_case_speedup = None
                self.last_bench_result = None
                detail = bench.get("message") or "benchmark failed"
                self._log(f"BLOCK (benchmark failed: {detail})")
                return self._block(
                    "The kernel benchmark did not complete all three independent "
                    f"measurements: {detail}. Fix the failure and continue."
                )
            try:
                measurement_scores = calculate_measurement_case_speedups(
                    bench,
                    self.baseline_case_times,
                    expected_measurements=KEEP_MEASUREMENT_COUNT,
                )
            except CaseCoverageError as error:
                self.last_wall_ms = None
                self._log(f"BLOCK (case coverage failed: {error})")
                return self._block(
                    "The kernel benchmark did not report every baseline case, so "
                    f"the candidate cannot be scored safely: {error}. Restore full "
                    "suite coverage and continue."
                )
            mean_case_speedup = keep_score(measurement_scores)
            if mean_case_speedup is None or self.best_mean_case_speedup is None:
                self.last_wall_ms = wall
                self.last_mean_case_speedup = None
                self.last_bench_result = None
                self._log("BLOCK (pristine scoring state unavailable)")
                return self._block(
                    "Mean case scoring requires the fixed pristine baseline and "
                    "current best score. Restore canonical scoring state and continue."
                )
            self.last_wall_ms = wall
            self.last_mean_case_speedup = mean_case_speedup
            bench["mean_case_speedup"] = mean_case_speedup
            bench["measurement_mean_case_speedups"] = measurement_scores
            bench["candidate_diff_sha256"] = self._candidate_diff_sha256()
            bench["driver_sha256"] = self._sha256(Path(self.driver_script).resolve())
            bench["baseline_case_times"] = dict(self.baseline_case_times)
            bench["best_mean_case_speedup"] = self.best_mean_case_speedup
            bench["bench_repeat"] = self.bench_repeat
            self.last_bench_result = bench

            required = required_keep_speedup(self.best_mean_case_speedup, measurement_scores)
            if passes_keep_threshold(
                measurement_scores,
                best_mean_case_speedup=self.best_mean_case_speedup,
            ):
                self.passed = True
                self.end_reason = "converged"
                self._log(
                    "ALLOW (correct + faster: mean case speedup "
                    f"mean score={mean_case_speedup:.6f}x >= {required:.6f}x "
                    f"from scores="
                    f"{[round(score, 6) for score in measurement_scores]}; "
                    f"raw mean {wall} ms) "
                    f"edit={self.edit_count}"
                )
                return self._allow()

            speedup_txt = f"{mean_case_speedup:.6f}x"
            wall_txt = f"{wall:.6f}" if wall is not None else "unmeasured"

            self._log(
                "BLOCK (correct but not faster: mean case speedup "
                f"{speedup_txt}; required="
                f"{required:.6f}x; "
                f"raw mean {wall_txt} ms) "
                f"edit={self.edit_count}"
            )
            return self._block(
                "The kernel is CORRECT but NOT faster than the current best, so it "
                "is not good enough to finish.\n"
                f"Measured mean case speedup={speedup_txt}; required="
                f"{required:.6f}x; "
                f"raw mean={wall_txt} ms.\n"
                "Keep the kernel correct and try a DIFFERENT optimization to reduce "
                "wall time, then continue."
            )
        except Exception as e:  # noqa: BLE001 - end session but reject candidate
            self.finalize_integrity()
            self.integrity_violation = True
            self.integrity_verdict = "violation"
            self.integrity_reason = f"in-session gate failed: {type(e).__name__}: {e}"
            self.end_reason = "gate_error"
            self._log(f"gate error ({type(e).__name__}: {e}) -> allow stop, outer loop will reject candidate")
            return self._allow()

    def _log(self, msg: str) -> None:
        sys.stderr.write(f"  [in-session-gate] {msg}\n")
        sys.stderr.flush()
