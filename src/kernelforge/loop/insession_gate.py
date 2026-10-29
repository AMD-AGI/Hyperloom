# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-session self-correction gate for the forge-loop agent."""

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
import traceback
from typing import Any

from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS, PROTECTED_GLOBS
from kernelforge.tools.git import git
from kernelforge.loop.gate_settings import GateMeasurement, GateScope
from kernelforge.evaluation.jit_rebuild import JitRebuildUnavailable, force_jit_rebuild_for_changes
from kernelforge.evaluation.scoring import (
    KEEP_MEASUREMENT_COUNT,
    keep_score,
    passes_keep_threshold,
    required_keep_speedup,
)
from kernelforge.evaluation.correctness.driver import test_correctness
from kernelforge.evaluation.benchmark.runner import (
    CaseCoverageError,
    calculate_measurement_case_speedups,
    BenchRun,
    measure_wallclock,
)


# Files the agent must NOT modify: the test harness / driver that MEASURES the kernel.
_DEFAULT_PROTECTED_GLOBS = list(PROTECTED_GLOBS)

# Tools that modify files on disk (subject to the protected-file deny + counted as edits when they target the kernel).
_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
_EDIT_TOOL_MATCHER = "|".join(_EDIT_TOOLS)

# Shell verbs that WRITE their path arguments (vs reading / executing them).
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
# A shell variable assignment, which is what precedes a command's verb.
_SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# In-process (python/perl) WRITE APIs — there is no shell verb to key off, so these are detected textually and paired
# with a protected-file mention below.
_INLINE_WRITE_INTENT = re.compile(
    r"\b(?:write_text|write_bytes)\b"
    r"|\bshutil\.(?:copy\w*|move|rmtree)\b"
    r"|\b(?:open|io\.open|Path\s*\([^)]*\)\.open)\s*\("
    r"|\b(?:os\.)?(?:rename|replace)\s*\("
    r"|\bPath\s*\([^)]*\)\.(?:rename|replace)\s*\(",
    re.IGNORECASE,
)
# ``shutil`` helpers that write their path arguments; ``rmtree`` writes only the first.
_SHUTIL_WRITE_CALLS = frozenset({"copy", "copy2", "copyfile", "move", "rmtree"})
_BASH_REDIRECT_TARGET = re.compile(r"(?:^|\s)(?<!<)(?:\d*>>?|\d*>\||&>>?)\s*([^\s;&|]+)")
_PYTHON_HEREDOC = re.compile(
    r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\1(?:\s|$)",
    re.DOTALL,
)
# The interpreter a heredoc is fed to, as it appears on the line that opens the document.
_HEREDOC_PYTHON_VERB = re.compile(r"(?:^|\s)(?:[A-Za-z0-9_./-]*/)?python(?:\d+(?:\.\d+)*)?(?:\s|$)")
_PYTHON_COMMAND_PREFIX = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_./-]*/)?python"
    r"(?:\d+(?:\.\d+)*)?\s+"
    r"(?:-[A-Za-z][^\s;&|]*\s+)*"
    r"-c\s+",
)
# Where a ``python -c`` payload word ends, by the quote it opens with: a single-quoted word at the next quote, a
# double-quoted one at the next quote no backslash escapes, a bare word at whitespace or a command separator.
_PAYLOAD_WORD_BODY = {
    "'": re.compile(r"[^']*"),
    '"': re.compile(r'(?:[^"\\]|\\.)*', re.DOTALL),
    "": re.compile(r"[^\s;&|]*"),
}
# An interpreter as a simple command's verb, which _simple_commands has already reduced to a basename, so
# `/usr/bin/python3.11` arrives here as `python3.11`.
_PYTHON_VERB = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
# A shell whose ``-c`` argument is another command line rather than one of its own.
_SHELL_VERBS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


def _unquote(token: str) -> str:
    """Strip surrounding whitespace and one layer of shell quoting."""
    return token.strip().strip("'\"")


def _short_option_value(args: Sequence[str], letter: str) -> str:
    """What a short option carries, however it was written, or \"\"."""
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
    """Split shell text on the operators that separate commands, respecting quotes."""
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
    """Split shell text into simple commands, each as its verb and arguments."""
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
        if start >= len(command):
            break

        quote = command[start] if command[start] in {"'", '"'} else ""
        content_start = start + 1 if quote else start
        end = _PAYLOAD_WORD_BODY[quote].match(command, content_start).end()
        if quote and not command.startswith(quote, end):
            cursor = start + 1
            continue
        payload_end = end + 1 if quote else end
        try:
            source = shlex.split(command[start:payload_end])[0]
        except (ValueError, IndexError):
            cursor = max(payload_end, start + 1)
            continue
        payloads.append((source, content_start, end))
        cursor = payload_end
    return payloads


def _heredoc_payloads(command: str) -> list[tuple[str, int, int]]:
    """Extract every heredoc body with its span, as ``(source, start, end)``.

    A heredoc the surrounding line does not feed to an interpreter yields an
    empty source: there is no program to parse, but its span is still reported
    so the caller blanks the document instead of scanning it as shell text.
    """
    payloads: list[tuple[str, int, int]] = []
    for match in _PYTHON_HEREDOC.finditer(command):
        line_start = command.rfind("\n", 0, match.start()) + 1
        prefix = command[line_start : match.start()]
        source = match.group(2) if _HEREDOC_PYTHON_VERB.search(prefix) else ""
        payloads.append((source, match.start(), match.end()))
    return payloads


def _blank_heredocs(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Blank heredoc bodies, keeping their newlines.

    A heredoc body occupies whole lines of the command, so the newlines around
    it are the command's own: dropping them would join the line before the
    ``<<`` to the line after the terminator.
    """
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _blank_quoted_payloads(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Blank quoted payloads whole, newlines included.

    A ``python -c`` program sits inside one quoted word, so its newlines belong
    to the program, not to the shell. Keeping them would split the rest of the
    command across lines and leave each half with an unmatched quote -- which is
    how a write verb gets separated from the protected path it names.
    """
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _is_disposable_redirect_target(raw: str) -> bool:
    """Whether a redirect destination is a scratch sink that no protected file can hide behind."""
    target = raw.strip().strip("'\"")
    return not target or target == "/dev/null" or target.startswith(("/tmp/", "$tmp", "${tmp"))


@dataclass(frozen=True)
class _ProtectedFileState:
    """Restorable state for one protected filesystem path."""

    path: Path
    kind: str
    content: bytes
    mode: int
    digest: str
    error: str = ""


def _fold_literal_path(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    """Fold an expression into the literal path it names, or None when it names none.

    ``constants`` carries the names already bound to a literal path earlier in
    the same module, so a script that assigns its output path once and writes
    through the name still resolves.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _fold_literal_path(node.left, constants)
        right = _fold_literal_path(node.right, constants)
        return str(Path(left) / right) if left is not None and right is not None else None
    if isinstance(node, ast.Call):
        return _fold_literal_path_call(node, constants)
    return None


def _fold_literal_path_call(node: ast.Call, constants: dict[str, str]) -> str | None:
    """Fold the call shapes that still name a literal path: ``Path``, ``.resolve``, ``os.path.join``."""
    if isinstance(node.func, ast.Name):
        if node.func.id == "Path" and node.args:
            return _fold_literal_path(node.args[0], constants)
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr in {"resolve", "absolute"}:
        return _fold_literal_path(node.func.value, constants)
    if _is_os_path_join(node.func):
        parts = [_fold_literal_path(arg, constants) for arg in node.args]
        if parts and all(part is not None for part in parts):
            return os.path.join(*(part for part in parts if part is not None))
    return None


def _is_os_path_join(func: ast.Attribute) -> bool:
    """Whether an attribute call target is literally ``os.path.join``."""
    return (
        func.attr == "join"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "os"
    )


def _bind_literal_path(node: ast.Assign | ast.AnnAssign, constants: dict[str, str]) -> None:
    """Bind each name this assignment targets to the literal path it was given."""
    value = _fold_literal_path(node.value, constants)
    if value is None:
        return
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            constants[target.id] = value


def _open_write_targets(node: ast.Call, constants: dict[str, str]) -> list[ast.AST | None] | None:
    """Nodes naming what an ``open``-family call writes, or None when it opens nothing for writing.

    A ``[None]`` return says the call writes somewhere the analysis could not
    name -- the mode itself did not resolve, so the path cannot be cleared.
    """
    is_builtin_open = isinstance(node.func, ast.Name) and node.func.id == "open"
    is_io_open = (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "io"
        and node.func.attr == "open"
    )
    is_path_open = isinstance(node.func, ast.Attribute) and node.func.attr == "open" and not is_io_open
    if not (is_builtin_open or is_io_open or is_path_open):
        return None
    positional_mode_index = 0 if is_path_open else 1
    mode_node = (
        node.args[positional_mode_index]
        if len(node.args) > positional_mode_index
        else next((keyword.value for keyword in node.keywords if keyword.arg == "mode"), None)
    )
    mode = _fold_literal_path(mode_node, constants) if mode_node is not None else "r"
    if mode is None:
        return [None]
    if not any(flag in mode for flag in "wax+"):
        return None
    return [node.func.value if is_path_open else (node.args[0] if node.args else None)]


def _method_write_targets(node: ast.Call) -> list[ast.AST] | None:
    """Nodes naming what a write method or ``shutil``/``os`` helper writes, or None when it writes nothing."""
    if not isinstance(node.func, ast.Attribute):
        return None
    attribute = node.func.attr
    receiver = node.func.value
    if attribute in {"write_text", "write_bytes", "unlink", "rmdir"}:
        return [receiver]
    if isinstance(receiver, ast.Name) and receiver.id == "shutil" and attribute in _SHUTIL_WRITE_CALLS:
        return list(node.args[:1] if attribute == "rmtree" else node.args[:2])
    if isinstance(receiver, ast.Name) and receiver.id == "os" and attribute in {"rename", "replace"}:
        return list(node.args[:2])
    if attribute in {"rename", "replace"}:
        return [receiver, *node.args[:1]]
    return None


class _WriteTargetSet:
    """The paths one script writes, and whether any write stayed unresolved."""

    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.ambiguous = False
        self.found_write = False

    def record(self, nodes: Sequence[ast.AST | None], constants: dict[str, str]) -> None:
        """Book one write call: its targets when they resolve, ambiguity when they do not."""
        self.found_write = True
        for node in nodes:
            resolved = _fold_literal_path(node, constants)
            if resolved is None:
                self.ambiguous = True
            else:
                self.paths.add(resolved)


def _python_write_targets(source: str) -> tuple[set[str], bool, bool]:
    """Return resolved write targets, ambiguity, and whether writes were found."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), True, bool(_INLINE_WRITE_INTENT.search(source))

    constants: dict[str, str] = {}
    written = _WriteTargetSet()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            _bind_literal_path(node, constants)
        if not isinstance(node, ast.Call):
            continue
        targets = _open_write_targets(node, constants)
        if targets is None:
            targets = _method_write_targets(node)
        if targets is not None:
            written.record(targets, constants)
    return written.paths, written.ambiguous, written.found_write


class InSessionGate:
    """Per-iteration harness-protection gate driving the Stop hook."""

    def __init__(
        self,
        measurement: GateMeasurement,
        scope: GateScope = GateScope(),
        *,
        baseline_case_times: dict | None = None,
        best_mean_case_speedup: float | None = None,
    ):
        driver_script, kernel_file = measurement.driver_script, scope.kernel_file
        self.driver_script = driver_script
        # The wrapper script this session must reach the driver through, when a caller interposed one.
        self.interposed_driver_path = measurement.interposed_driver_path or ""
        self.snr_threshold = measurement.snr_threshold
        self.bench_repeat = measurement.bench_repeat
        self.baseline_case_times = dict(baseline_case_times or {})
        self.best_mean_case_speedup = best_mean_case_speedup
        # Correctness-only phase (e.g. PORT): the gate requires ONLY correctness and never runs the perf gate (no
        # benchmark; best score unused).
        self.correctness_only = measurement.correctness_only
        # Attempt budget for the correctness/perf self-correction loop.
        self.max_blocks = measurement.max_blocks
        self.stage_timeout_sec = measurement.stage_timeout_sec
        self.bench_timeout_sec = measurement.bench_timeout_sec
        # Upper bound on Stop-hook blocks for unrestored harness tampering.
        self.max_stop_blocks = measurement.max_stop_blocks

        # Target file set used to count edits for observability.
        targets = list(scope.target_files) if scope.target_files else []
        if kernel_file:
            targets.append(kernel_file)
        self.target_abs = {os.path.normpath(os.path.abspath(f)) for f in targets if f}
        # Kept for logging/back-compat (the anchor file).
        self.kernel_abs = os.path.normpath(os.path.abspath(kernel_file)) if kernel_file else ""

        # Per-session mutable state.
        self.edit_count = 0
        self.block_count = 0
        # Harness-protection blocks only (bounded separately by max_stop_blocks), so a non-cooperating tamperer is
        # capped independently of the perf loop.
        self.harness_block_count = 0
        self.passed = False
        self.last_wall_ms: float | None = None
        self.last_mean_case_speedup: float | None = None
        self.last_bench_result: dict | None = None
        # Why the gate ALLOWED the session to stop (set once, at an allow path): "converged" — correct AND faster than
        # best; a real win. "block_budget_exhausted" — max_blocks blocked stops spent; hand off to the outer loop to
        # re-validate + keep/revert. "harness_tampered" — harness block cap hit on unrestored protected changes; the
        # outer loop force-REVERTs it. "validation_timeout" — full-suite correctness timed out; the outer loop
        # performs the one authoritative retry. "jit_rebuild_unavailable" — the workspace could not assert a rebuild
        # of the edited sources, so nothing was measured in-session; the candidate is untouched and the outer loop
        # measures it itself. "gate_error" — the gate itself raised; fail OPEN.
        self.end_reason = ""
        # Real failure signals seen this session (block reasons: compile errors, "correct but not faster", …).
        self.findings: list[str] = []

        # Protected (measurement) files — the driver (passed in via --driver) is matched by EXACT absolute path; the
        # test harness / perf helpers next to the kernel are caught by the basename globs below.
        self.protected_abs: set[str] = set()
        if driver_script:
            self.protected_abs.add(os.path.normpath(os.path.abspath(driver_script)))
        # Additional exact-path measurement files.
        for p in scope.extra_protected_paths or []:
            if p:
                self.protected_abs.add(os.path.normpath(os.path.abspath(p)))
        self.workspace_root = self._infer_workspace_root(scope.workspace, driver_script, kernel_file)

        globs = list(_DEFAULT_PROTECTED_GLOBS)
        # Repository tasks ship the reference/test implementation INSIDE the repo tree (e.g. AITER's
        # op_tests/.../test_*.py provides the correctness reference), which the default globs above do not catch.
        if scope.extra_protected_globs:
            globs += list(scope.extra_protected_globs)
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
        """Build provider-neutral lifecycle hooks for capable backends."""
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
            # Count every non-protected implementation edit.
            post_tool_use=[
                AgentHook(
                    matcher=_EDIT_TOOL_MATCHER,
                    callback=self._on_edit,
                )
            ],
            # Stop: harness protection, then canonical correctness+bench self-check.
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
        """Resolve the tree this gate measures, preferring the declared one."""
        for candidate in (workspace, driver_script, kernel_file):
            if not candidate:
                continue
            try:
                p = Path(candidate).resolve()
                if p.exists():
                    return p.parent if p.is_file() else p
            except OSError:
                continue
        return None

    def _is_protected(self, fp: str) -> bool:
        if not fp:
            return False
        return FORGE_PROTECTED_PATHS.is_protected(
            fp,
            workspace=self.workspace_root,
            exact_paths=self.protected_abs,
            extra_globs=self.protected_globs,
        )

    def _iter_snapshot_paths(self) -> list[Path]:
        root = self.workspace_root
        if root is None:
            return sorted(Path(path) for path in self.protected_abs)
        return list(
            FORGE_PROTECTED_PATHS.inventory(
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
        except (OSError, RuntimeError, ValueError) as error:  # an incomplete scan is a verdict
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

    def _protected_changes(self) -> str:
        current, current_errors = self._snapshot_protected_states(
            include_baseline=True,
        )
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
        except Exception as error:  # noqa: BLE001 - isolation boundary: one session's protected-integrity scan
            reason = f"protected integrity scan failed: {type(error).__name__}: {error}"
            self._log(f"{reason}\n{traceback.format_exc().rstrip()}")
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

    def _protected_basenames(self) -> set[str]:
        """Basenames of every protected file.

        Agents ``cd`` into the workspace, so a command's path arguments are
        usually relative and would never resolve to the protected ABSPATH.
        """
        return {os.path.basename(path) for path in self.protected_abs} | {
            Path(relative).name for relative in self._protected_snapshot
        }

    def _names_protected(self, raw: str) -> bool:
        """Whether a command argument names a protected file, by path or by basename."""
        path = _unquote(raw)
        if not path:
            return False
        return self._is_protected(path) or os.path.basename(path) in self._protected_basenames()

    def _protected_mention(self, text: str) -> str:
        """Basename of the first protected file this text mentions anywhere, or \"\"."""
        lowered = text.lower()
        for path in (*self.protected_abs, *self._protected_snapshot):
            if path.lower() in lowered or Path(path).name.lower() in lowered:
                return Path(path).name
        return ""

    def _python_write_deny_reason(self, source: str) -> str:
        """WHY an inline Python payload writes a protected file, or \"\"."""
        targets, ambiguous, found_write = _python_write_targets(source)
        if not found_write:
            return ""
        for target in targets:
            if self._names_protected(target):
                return f"inline write targets protected file '{os.path.basename(_unquote(target))}'"
        if not ambiguous:
            return ""
        mentioned = self._protected_mention(source)
        return f"inline write may modify protected file '{mentioned}'" if mentioned else ""

    def _redirect_deny_reason(self, text: str) -> str:
        """WHY an output redirection in this command writes a protected file, or \"\"."""
        for match in _BASH_REDIRECT_TARGET.finditer(text):
            target = match.group(1)
            if _is_disposable_redirect_target(target):
                continue
            if self._names_protected(target):
                return f"redirect writes protected path '{_unquote(target)}'"
        return ""

    def _write_verb_deny_reason(self, text: str) -> str:
        """WHY a shell write verb in this command takes a protected file as a path argument, or \"\"."""
        for verb, args in _simple_commands(text):
            inplace_edit = verb in ("sed", "perl") and any(arg == "-i" or arg.startswith("-i") for arg in args)
            if verb not in _BASH_WRITE_VERBS and not inplace_edit:
                continue
            for arg in args:
                if arg.startswith("-"):
                    continue
                if self._names_protected(arg):
                    return f"`{verb}` writes protected file '{os.path.basename(_unquote(arg))}'"
        return ""

    def _inline_write_deny_reason(self, text: str) -> str:
        """WHY a non-Python in-process write may touch a protected file, or \"\".

        The conservative textual backstop for the write APIs no shell verb and no
        Python parse can reach.
        """
        if not _INLINE_WRITE_INTENT.search(text):
            return ""
        mentioned = self._protected_mention(text)
        return f"inline write may modify protected file '{mentioned}'" if mentioned else ""

    def _bash_deny_reason(self, command: str) -> str:
        """WHY a Bash command is denied (a short trigger reason), or \"\" to allow."""
        if not command:
            return ""
        # Every inline Python payload is parsed on its own, then blanked out so the
        # textual scans below never read a quoted program as shell text.
        remainder = command
        for find_payloads, blank in (
            (_heredoc_payloads, _blank_heredocs),
            (_python_command_payloads, _blank_quoted_payloads),
        ):
            payloads = find_payloads(remainder)
            for source, _start, _end in payloads:
                reason = self._python_write_deny_reason(source)
                if reason:
                    return reason
            remainder = blank(remainder, [(start, end) for _source, start, end in payloads])
        return (
            self._redirect_deny_reason(remainder)
            or self._write_verb_deny_reason(remainder)
            or self._inline_write_deny_reason(remainder)
        )

    def _bash_bypass_reason(self, command: str) -> str:
        """WHY a Bash command reaches the driver around its wrapper, or \"\"."""
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
                # A nested shell carries its command line inside one word, so the split above cannot see into it.
                nested = self._bash_bypass_reason(_short_option_value(args, "c"))
                if nested:
                    return nested
                continue
            if not _PYTHON_VERB.match(verb):
                continue
            # An interpreter reaches the driver by module as readily as by path, and `-m` names it without the suffix
            # the path carries, so the scan below cannot see it.
            module = _short_option_value(args, "m")
            if module and module.rsplit(".", 1)[-1] == driver_stem:
                return f"`{verb} -m {module}` runs the driver outside `{run_base}`"
            # Every non-option word, not just the first.
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
        """One denial, logged with the command that triggered it."""
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
        """Deny shell writes to protected harness files, and driver runs that would go around the command this session must measure through."""
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
        # Keep a trimmed record for the experience ledger (cap each entry).
        self.findings.append(reason.strip()[:1200])
        return {"decision": "block", "reason": reason}

    def _harness_block(self, reason: str) -> dict:
        """Block a stop for unrestored harness tampering (does not eat perf budget)."""
        self.findings.append(reason.strip()[:1200])
        return {"decision": "block", "reason": reason}

    def _stop_for_protected_harness(self, protected_delta: str) -> dict:
        """Allow or block a stop that still has protected harness changes on disk."""
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

    def _record_insession_bench(
        self,
        bench: dict,
        wall,
        mean_case_speedup: float,
        measurement_scores: list[float],
    ) -> None:
        """Publish the in-session bench the outer loop may reuse for this stop."""
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

    async def _stop_performance_verdict(self) -> dict:
        """Bench the correct candidate and allow only when it clears the KEEP bar."""
        self.last_wall_ms = None
        self.last_mean_case_speedup = None
        self.last_bench_result = None
        bench = await measure_wallclock(
            BenchRun(self.driver_script, driver_args=[], repeat=self.bench_repeat),
            measurements=KEEP_MEASUREMENT_COUNT,
            timeout_sec=self.bench_timeout_sec,
        )
        wall = bench.get("median_ms")
        if not bench.get("success"):
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
            self._log(f"BLOCK (case coverage failed: {error})")
            return self._block(
                "The kernel benchmark did not report every baseline case, so "
                f"the candidate cannot be scored safely: {error}. Restore full "
                "suite coverage and continue."
            )
        mean_case_speedup = keep_score(measurement_scores)
        if mean_case_speedup is None or self.best_mean_case_speedup is None:
            self.last_wall_ms = wall
            self._log("BLOCK (pristine scoring state unavailable)")
            return self._block(
                "Mean case scoring requires the fixed pristine baseline and "
                "current best score. Restore canonical scoring state and continue."
            )
        self._record_insession_bench(bench, wall, mean_case_speedup, measurement_scores)

        required = required_keep_speedup(self.best_mean_case_speedup, measurement_scores)
        if passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=self.best_mean_case_speedup,
        ):
            self.passed = True
            self.end_reason = "converged"
            self._log(
                f"ALLOW (correct + faster: mean case speedup mean score={mean_case_speedup:.6f}x >= {required:.6f}x "
                f"from scores={[round(score, 6) for score in measurement_scores]}; raw mean {wall} ms) "
                f"edit={self.edit_count}"
            )
            return self._allow()

        speedup_txt = f"{mean_case_speedup:.6f}x"
        wall_txt = f"{wall:.6f}" if wall is not None else "unmeasured"
        self._log(
            f"BLOCK (correct but not faster: mean case speedup {speedup_txt}; required={required:.6f}x; "
            f"raw mean {wall_txt} ms) edit={self.edit_count}"
        )
        return self._block(
            "The kernel is CORRECT but NOT faster than the current best, so it is not good enough to finish.\n"
            f"Measured mean case speedup={speedup_txt}; required={required:.6f}x; raw mean={wall_txt} ms.\n"
            "Keep the kernel correct and try a DIFFERENT optimization to reduce wall time, then continue."
        )

    async def _stop_candidate_verdict(self) -> dict:
        """Run in-session correctness (and performance) and allow or block the stop."""
        # Ensure the canonical check compiles the kernel the agent has on disk RIGHT NOW: the SDK hook may run in a
        # subprocess that did not inherit the loop's AITER_REBUILD, so (re)assert it here. Its measurement is the one
        # the outer loop reuses, so a shard chosen from a stale source list would decide keep/revert on the previous
        # iteration's binary.
        try:
            force_jit_rebuild_for_changes(
                self.workspace_root or Path.cwd(),
                [self.kernel_abs, *self.target_abs],
            )
        except JitRebuildUnavailable as error:
            # The workspace, not the agent: the candidate on disk is intact, and outer canonical validation
            # asserts the rebuild itself before measuring. Blocking here would only spend turns on a failure no
            # edit can clear.
            self.end_reason = "jit_rebuild_unavailable"
            self.findings.append(f"In-session validation skipped; workspace unavailable: {error}")
            self._log(
                f"ALLOW (rebuild unavailable: {error}; outer loop measures this candidate) edit={self.edit_count}"
            )
            return self._allow()

        # Correctness — canonical driver, same call the pipeline uses.
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

        # Correctness-only phase (e.g. PORT): there is no performance requirement, so a CORRECT kernel is done.
        if self.correctness_only:
            self.passed = True
            self.end_reason = "converged"
            self._log(f"ALLOW (correct; correctness-only phase) edit={self.edit_count}")
            return self._allow()
        return await self._stop_performance_verdict()

    async def _on_stop(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        try:
            # Harness protection (security, first).
            protected_delta = self._protected_changes()
            self.integrity_reason = protected_delta
            self.integrity_violation = bool(protected_delta)
            self.integrity_verdict = "violation" if protected_delta else "clean"
            if protected_delta:
                return self._stop_for_protected_harness(protected_delta)

            # Budget (bounded so the gate never hangs the session). Checked before the (GPU-costly) canonical
            # validation so an exhausted session hands off immediately instead of paying for one more bench.
            if self.block_count >= self.max_blocks:
                self.end_reason = "block_budget_exhausted"
                self._log(
                    f"block budget exhausted (blocks={self.block_count}/{self.max_blocks}, "
                    f"edits={self.edit_count}) -> allow stop, hand off to outer "
                    "canonical validation (session stays resumable for lesson)"
                )
                return self._allow()

            return await self._stop_candidate_verdict()
        except Exception as e:  # noqa: BLE001 - end session but reject candidate
            self.finalize_integrity()
            self.integrity_violation = True
            self.integrity_verdict = "violation"
            self.integrity_reason = f"in-session gate failed: {type(e).__name__}: {e}"
            self.end_reason = "gate_error"
            self._log(
                f"gate error ({type(e).__name__}: {e}) -> allow stop, outer loop will reject candidate\n"
                f"{traceback.format_exc().rstrip()}"
            )
            return self._allow()

    def _log(self, msg: str) -> None:
        sys.stderr.write(f"  [in-session-gate] {msg}\n")
        sys.stderr.flush()
