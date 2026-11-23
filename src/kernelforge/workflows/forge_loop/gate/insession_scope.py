# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shell and inline-Python command parsing that finds the paths a Bash command writes or runs."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
import os
from pathlib import Path
import re
import shlex
import string


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
# Verbs that choose the processes they signal by name or pattern, out of every process on the host.
_PATTERN_KILL_VERBS = frozenset({"pkill", "killall", "killall5", "skill"})
# Verbs whose output is a list of processes found by name; a ``kill`` in the same command is a pattern kill in two
# steps, however the list reaches it: a substitution, quoted or not, a pipe, a loop or a variable.
_PROCESS_SEARCH_VERBS = frozenset({"pgrep", "pidof", "ps"})
# Words that open a command without being one; the command is what follows them.
_SHELL_KEYWORDS = frozenset({"if", "then", "elif", "else", "while", "until", "do", "!", "{", "("})
# ``kill`` options that only ask a question, so naming no process with them signals nothing.
_KILL_QUERY_OPTIONS = frozenset({"-l", "-L", "--list", "--table", "--help", "--version", "-V"})
# The same verbs named inside an inline program, where no shell word carries them.
_INLINE_PATTERN_KILL = re.compile(r"\b(?:pkill|killall)\b")
# A process search run from inside an inline program: as a word of an argument list or a command string, or psutil's.
_INLINE_PROCESS_SEARCH = re.compile(
    r"\b(?:pgrep|pidof)\b|[\"']ps[\"']|\bps\s+(?:-|aux\b|ax\b)|\bpsutil\.(?:process_iter|pids)\b"
)
# A signal sent from inside an inline program.
_INLINE_SIGNAL = re.compile(r"\bkill(?:pg)?\b|\.send_signal\b|\.terminate\b")
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
# An interpreter word and the whitespace after it; the options that may follow it are walked by
# _python_c_payload_start, not matched here.
_PYTHON_INTERPRETER_WORD = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_./-]*/)?python(?:\d+(?:\.\d+)*)?\s+")
_COMMAND_SEPARATORS = frozenset(";&|")
_ASCII_LETTERS = frozenset(string.ascii_letters)
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


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _python_c_payload_start(command: str, cursor: int) -> int | None:
    """Where the payload of the next ``python [options] -c`` at or after ``cursor`` starts, or None.

    The options between the interpreter and ``-c`` (``-`` and an ASCII letter, then anything up to whitespace
    or a command separator, then whitespace) are walked one word at a time, so a long run of them costs linear
    time. When the run holds ``-c`` more than once, the payload follows the last one.
    """
    for interpreter in _PYTHON_INTERPRETER_WORD.finditer(command, cursor):
        position, payload_start = interpreter.end(), None
        while command.startswith("-", position) and command[position + 1 : position + 2] in _ASCII_LETTERS:
            word_end = position + 1
            while word_end < len(command) and not (
                command[word_end].isspace() or command[word_end] in _COMMAND_SEPARATORS
            ):
                word_end += 1
            next_word = _skip_whitespace(command, word_end)
            if next_word == word_end:
                break
            if command[position:word_end] == "-c":
                payload_start = next_word
            position = next_word
        if payload_start is not None:
            return payload_start
    return None


def _python_command_payloads(command: str) -> list[tuple[str, int, int]]:
    """Extract ``python -c`` payloads with a linear quoted-string scan."""
    payloads: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        start = _python_c_payload_start(command, cursor)
        if start is None:
            break
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


def inline_programs(command: str) -> tuple[list[str], str]:
    """The inline Python programs a shell command carries, and the command with each of them blanked out.

    Heredoc bodies come first and ``python -c`` payloads are then found in what is left, so a quoted program is
    never read as shell text and a heredoc is never read as a quoted word.
    """
    sources: list[str] = []
    remainder = command
    for find_payloads, blank in (
        (_heredoc_payloads, _blank_heredocs),
        (_python_command_payloads, _blank_quoted_payloads),
    ):
        payloads = find_payloads(remainder)
        sources.extend(source for source, _start, _end in payloads)
        remainder = blank(remainder, [(start, end) for _source, start, end in payloads])
    return sources, remainder


def _kill_names_no_process(args: Sequence[str]) -> bool:
    """Whether a ``kill`` is given no process of its own: its targets then come from a pipe or are every process."""
    named = list(args[args.index("--") + 1 :]) if "--" in args else [arg for arg in args if not arg.startswith("-")]
    return not named


def _closing_parenthesis(text: str, start: int) -> int:
    """Where the parenthesis opened just before ``start`` closes, or the end of the text when it never does."""
    depth = 1
    for position in range(start, len(text)):
        depth += (text[position] == "(") - (text[position] == ")")
        if not depth:
            return position
    return len(text)


def _substitution_payloads(text: str) -> list[tuple[str, int, int]]:
    """The shell text a command runs to produce one of its words or streams, as ``(source, start, end)``.

    These are the outermost ``$(...)``, ``<(...)``, ``>(...)`` and backtick spans, whether or not double quotes
    surround them; a nested one is part of its parent's source. Single quotes make the text literal, so nothing
    inside them is one.
    """
    payloads: list[tuple[str, int, int]] = []
    position, in_single, in_double = 0, False, False
    while position < len(text):
        character = text[position]
        if character == "'" and not in_double:
            in_single = not in_single
        elif character == '"' and not in_single:
            in_double = not in_double
        elif character == "`" and not in_single:
            closing = text.find("`", position + 1)
            closing = len(text) if closing < 0 else closing
            payloads.append((text[position + 1 : closing], position, min(closing + 1, len(text))))
            position = closing
        elif character in "$<>" and text.startswith("(", position + 1) and not in_single:
            closing = _closing_parenthesis(text, position + 2)
            payloads.append((text[position + 2 : closing], position, min(closing + 1, len(text))))
            position = closing
        position += 1
    return payloads


def _shell_commands(text: str) -> Iterator[tuple[str, list[str]]]:
    """Every simple command shell text runs: after a keyword, inside a substitution and in a ``sh -c`` word included."""
    payloads = _substitution_payloads(text)
    # Each substitution stands in the outer command as one opaque word, so the outer command still has the word it
    # was given and the substitution's own punctuation does not run into the operators around it.
    outer = list(text)
    for _source, start, end in payloads:
        outer[start:end] = "x" * (end - start)
    for verb, args in _simple_commands("".join(outer)):
        while verb in _SHELL_KEYWORDS and args:
            verb, args = os.path.basename(_unquote(args[0])), args[1:]
        yield verb, args
        if verb in _SHELL_VERBS:
            yield from _shell_commands(_short_option_value(args, "c"))
    for source, _start, _end in payloads:
        yield from _shell_commands(source)


def _shell_kill_reason(text: str) -> str:
    commands = list(_shell_commands(text))
    pattern_kill = next((verb for verb, _args in commands if verb in _PATTERN_KILL_VERBS), "")
    if pattern_kill:
        return f"`{pattern_kill}` chooses its targets by name out of every process on the host"
    kills = [args for verb, args in commands if verb == "kill" and not _KILL_QUERY_OPTIONS.intersection(args)]
    if not kills:
        return ""
    search = next((verb for verb, _args in commands if verb in _PROCESS_SEARCH_VERBS), "")
    if search:
        return (
            f"this command runs `kill` and `{search}`, a search by name over every process on the host, so what it "
            "signals is not limited to this session"
        )
    if any(_kill_names_no_process(args) for args in kills):
        return "`kill` names no process, so its targets come from a search or are every process"
    return ""


def host_wide_kill_reason(command: str) -> str:
    """WHY a shell command signals processes it did not name one by one, or \"\" when it does not."""
    if not command:
        return ""
    sources, remainder = inline_programs(command)
    for source in sources:
        found = _INLINE_PATTERN_KILL.search(source)
        if found:
            return f"an inline program runs `{found.group(0)}`, which chooses its targets by name"
        search = _INLINE_PROCESS_SEARCH.search(source)
        if search and _INLINE_SIGNAL.search(source):
            return (
                f"an inline program searches processes by name (`{search.group(0)}`) and sends a signal, so what "
                "it signals is not limited to this session"
            )
    return _shell_kill_reason(remainder)
