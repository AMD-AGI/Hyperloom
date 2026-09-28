# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX server script a recipe boots through, and what it does to the launch it is handed."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}`` guard: the recipe's own value
# wins over anything the caller exported under that name.
_UNGUARDED_EXPORT_RE = re.compile(r"^[^\S\n]*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)

# Spelled out rather than imported from ``agentx.deploy``: this module is on the
# default benchmark path, which is pinned not to import the agentx package.
_AGENTX_CLIENT_SCRIPT = "aiperf_client.sh"

# Pattern for the server command array declaration: NAME=(
_CMD_ARRAY_OPEN_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=\(", re.MULTILINE)
# Pattern for a spliced sub-array inside the command: "${NAME[@]}"
_SPLICED_ARRAY_RE = re.compile(r'"\$\{[A-Za-z_][A-Za-z0-9_]*\[@\]\}"')
# Pattern for the array launch line: "${CMD_ARRAY[@]}" ...
_ARRAY_LAUNCH_RE = re.compile(r'"?\$\{?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[@\]\}"?\s')
# Normalize flag names: leading dashes, convert underscores to dashes, lowercase.
_FLAG_NAME_RE = re.compile(r"^--?")


class RecipeLeverUnavailableError(ValueError):
    """Raised when the recipe cannot carry a lever the variant depends on.

    On the agentic-recipe surface this means the edit is structurally
    inexpressible (e.g. the flag lives in a spliced sub-array, or the recipe
    has no single server-command array).
    """


def resolve_launch_server_script(bench: Mapping[str, Any]) -> str:
    """Path of the script that boots the server, or ``""`` when unresolvable.

    ``benchmark_script`` names a server launcher on every non-AgentX run. The
    AgentX switch pins the aiperf client there instead, so for that one recipe
    shape the launcher is the builtin the client delegates to. Resolution
    mirrors ``aiperf_client.sh``: the same ``AGENTX_SERVER_SCRIPT`` override,
    the same ``{framework}_{gpu}.sh`` fallback, the same ``<checkout>/benchmarks/``
    directory and no recursive search. Recipe-recorded values beat the ambient
    env because the recipe is the record of what actually ran.
    """
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    script = Path(str(bench.get("benchmark_script") or "").strip()).name
    if not script:
        return ""

    if script == _AGENTX_CLIENT_SCRIPT:
        script = str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
        if not script:
            framework = str(bench.get("framework") or envs.get("FRAMEWORK") or "").strip().lower()
            if not framework:
                return ""
            gpu = (
                str(
                    envs.get("GPU_TYPE")
                    or envs.get("RUNNER_TYPE")
                    or bench.get("runner_type")
                    or os.environ.get("GPU_TYPE")
                    or os.environ.get("RUNNER_TYPE")
                    or "mi300x"
                )
                .strip()
                .lower()
            )
            script = f"{framework}_{gpu}.sh"

    for root in (
        str(bench.get("inferencex_path") or "").strip(),
        os.environ.get("INFERENCEX_PATH", "").strip(),
    ):
        if not root:
            continue
        benchmarks = Path(root) / "benchmarks"
        candidate = benchmarks / script
        # The builtin sources benchmark_lib.sh from its own directory and dies
        # without it, so a half-populated checkout resolves to nothing.
        if candidate.is_file() and (benchmarks / "benchmark_lib.sh").is_file():
            return str(candidate)
    return ""


def recipe_owns_argv(bench: Mapping[str, Any]) -> bool:
    """True when the resolved server script lives under an ``agentic/`` directory.

    On this surface the recipe hardcodes its own argv and env; Hyperloom
    delivers levers by rendering an edited copy rather than through
    ``EXTRA_*_ARGS``.
    """
    path = resolve_launch_server_script(bench)
    if not path:
        return False
    return "agentic" in Path(path).parts


def launcher_overwritten_envs(bench: Mapping[str, Any]) -> frozenset[str]:
    """Env names the resolved server script re-exports unconditionally.

    Returns an empty set when the recipe owns its argv (where Hyperloom writes
    env levers directly into the script copy rather than through the YAML).
    Returns an empty set when the script cannot be read (safe default: no drop).
    """
    if recipe_owns_argv(bench):
        return frozenset()
    path = resolve_launch_server_script(bench)
    if not path:
        return frozenset()
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("recipe: could not read the server script %s; assuming it constrains nothing", path)
        return frozenset()
    return frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


def _normalize_flag_name(flag: str) -> str:
    """Lowercase flag name with dashes (--flag-name -> flag-name)."""
    return _FLAG_NAME_RE.sub("", flag).lower().replace("_", "-")


def _parse_flag_tokens(argv_str: str) -> list[str]:
    """Split an argv string into tokens, handling single-quoted values."""
    import shlex

    try:
        return shlex.split(argv_str)
    except ValueError:
        return argv_str.split()


def _find_server_cmd_array(text: str, framework: str) -> tuple[str, int, int] | None:
    """Return ``(array_name, open_line_index, close_line_index)`` for the server command array.

    Looks for the framework-appropriate array name pattern: VLLM_CMD, SGLANG_CMD, ATOM_CMD,
    or any NAME=( pattern whose expansion appears on a launch line.

    Returns None when the array cannot be identified unambiguously.
    """
    fw = framework.lower()
    if "vllm" in fw:
        preferred = "VLLM_CMD"
    elif "sglang" in fw:
        preferred = "SGLANG_CMD"
    elif "atom" in fw:
        preferred = "ATOM_CMD"
    else:
        preferred = None

    lines = text.splitlines(keepends=True)

    # Find array declaration spans.
    arrays: dict[str, tuple[int, int]] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _CMD_ARRAY_OPEN_RE.match(line.lstrip())
        if m:
            name = m.group("name")
            # Scan for the closing paren on its own line.
            depth = 1
            j = i + 1
            while j < len(lines) and depth > 0:
                stripped = lines[j].strip()
                if stripped == ")":
                    depth -= 1
                elif stripped.endswith("(") and not stripped.startswith("#"):
                    depth += 1
                j += 1
            if depth == 0:
                arrays[name] = (i, j - 1)
        i += 1

    if preferred and preferred in arrays:
        open_idx, close_idx = arrays[preferred]
        return preferred, open_idx, close_idx

    # Fall back to the array whose expansion appears first on a launch line.
    launch_names = [m.group("name") for m in _ARRAY_LAUNCH_RE.finditer(text)]
    for name in launch_names:
        if name in arrays:
            open_idx, close_idx = arrays[name]
            return name, open_idx, close_idx

    return None


def _array_lines_contain_spliced(lines: list[str], open_idx: int, close_idx: int) -> list[str]:
    """Return spliced sub-array names found inside the command array body."""
    spliced = []
    for line in lines[open_idx + 1 : close_idx]:
        if _SPLICED_ARRAY_RE.search(line):
            spliced.append(line.strip())
    return spliced


def _flag_in_array_body(lines: list[str], open_idx: int, close_idx: int, flag_name: str) -> bool:
    """True when ``flag_name`` appears as a flag token in the array body lines."""
    for line in lines[open_idx + 1 : close_idx]:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Token may be bare (--flag) or combined with value (--flag value or --flag=value).
        token = stripped.split()[0] if stripped.split() else ""
        if not token.startswith("-"):
            continue
        tok_name = _normalize_flag_name(token.split("=")[0])
        if tok_name == flag_name:
            return True
    return False


def _remove_flag_from_body(
    lines: list[str],
    open_idx: int,
    close_idx: int,
    flag_name: str,
) -> list[str]:
    """Remove all occurrences of ``flag_name`` (and its value line) from the array body."""
    result = list(lines)
    to_delete: set[int] = set()
    i = open_idx + 1
    while i < close_idx:
        line = result[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            i += 1
            continue
        parts = stripped.split()
        if not parts or not parts[0].startswith("-"):
            i += 1
            continue
        tok = parts[0]
        tok_name = _normalize_flag_name(tok.split("=")[0])
        if tok_name == flag_name:
            to_delete.add(i)
            # If the flag is on one line and value is on the next (no = sign),
            # also remove the value line unless it starts with a dash.
            if "=" not in tok and len(parts) == 1 and i + 1 < close_idx:
                next_stripped = result[i + 1].strip()
                if next_stripped and not next_stripped.startswith("-") and not next_stripped.startswith(")"):
                    to_delete.add(i + 1)
                    i += 2
                    continue
        i += 1
    return [line for idx, line in enumerate(result) if idx not in to_delete]


def _set_flag_in_body(
    lines: list[str],
    open_idx: int,
    close_idx: int,
    flag_name: str,
    flag_line: str,
) -> list[str]:
    """Remove existing occurrences of flag and append the new value before the closing paren."""
    after_remove = _remove_flag_from_body(lines, open_idx, close_idx, flag_name)
    # Find ) on its own line from open_idx onward.
    new_close_idx = open_idx + 1
    depth = 1
    while new_close_idx < len(after_remove) and depth > 0:
        stripped = after_remove[new_close_idx].strip()
        if stripped == ")":
            depth -= 1
        new_close_idx += 1
    close_real = new_close_idx - 1
    indent = "    "
    new_line = f"{indent}{flag_line}\n"
    return after_remove[:close_real] + [new_line] + after_remove[close_real:]


def _export_block(env_sets: dict[str, str], env_unsets: list[str]) -> str:
    """Render a shell export block for the given env lever state."""
    lines = []
    for name in sorted(env_unsets):
        lines.append(f"unset {name}")
    for name in sorted(env_sets):
        val = env_sets[name]
        # Shell-quote the value if it contains spaces or special chars.
        if any(c in val for c in (" ", '"', "'", "$", "\\", "`")):
            import shlex

            quoted = shlex.quote(val)
        else:
            quoted = val
        lines.append(f"export {name}={quoted}")
    return "\n".join(lines) + "\n" if lines else ""


def apply_recipe_levers(
    bench: Mapping[str, Any],
    *,
    inherited_script: str,
    argv: list[str],
    remove_args: list[str],
    env_sets: dict[str, str],
    env_unsets: list[str],
) -> str:
    """Return the path (relative to ``benchmarks/``) of a rendered recipe copy.

    The copy is a sibling of the official recipe under
    ``benchmarks/single_node/agentic/``, named
    ``<official-stem>.hl-<sha256[:12]>.sh``.  It is written atomically only
    when absent.  When there are no levers and no inherited copy the official
    recipe path (relative form) is returned unchanged so the baseline runs
    byte-for-byte on the official recipe.

    Args:
        bench: Materialised benchmark dict (must have ``inferencex_path``).
        inherited_script: Current value of ``AGENTX_SERVER_SCRIPT`` in the
            bench envs, or ``""`` when ``args_mode=replace``.
        argv: Declared server-arg tokens to apply (set/append semantics).
        remove_args: Flag names to delete from the recipe array.
        env_sets: Env names to set (export) in the copy.
        env_unsets: Env names to unset in the copy.

    Returns:
        Path relative to ``benchmarks/``, suitable for ``AGENTX_SERVER_SCRIPT``.

    Raises:
        RecipeLeverUnavailableError: when the recipe structure cannot express
            the requested edit (no server-cmd array, a flag in a spliced
            sub-array, or the recipe file is unreadable).
    """
    from hyperloom.common.io import atomic_write_text

    has_levers = bool(argv or remove_args or env_sets or env_unsets)
    has_inherited = bool(inherited_script)

    if not has_levers and not has_inherited:
        # No edits and no prior copy: run the official recipe as-is.
        official_path = resolve_launch_server_script(bench)
        if not official_path:
            return ""
        # Return relative to benchmarks/.
        benchmarks = Path(official_path).parent
        while benchmarks.name != "benchmarks" and benchmarks.parent != benchmarks:
            benchmarks = benchmarks.parent
        try:
            return str(Path(official_path).relative_to(benchmarks))
        except ValueError:
            return ""

    # Resolve the base script to edit from.
    if has_inherited:
        # Inherited copy exists: build on top of it.
        for root in (
            str(bench.get("inferencex_path") or "").strip(),
            os.environ.get("INFERENCEX_PATH", "").strip(),
        ):
            if not root:
                continue
            base_candidate = Path(root) / "benchmarks" / inherited_script
            if base_candidate.is_file():
                base_path = base_candidate
                benchmarks_root = Path(root) / "benchmarks"
                break
        else:
            base_path = None
            benchmarks_root = None
    else:
        base_path = None
        benchmarks_root = None

    if base_path is None:
        # Fall back to the official recipe.
        official_path = resolve_launch_server_script(bench)
        if not official_path:
            raise RecipeLeverUnavailableError("cannot resolve the agentic server script for this bench")
        base_path = Path(official_path)
        benchmarks_root = base_path.parent
        while benchmarks_root.name != "benchmarks" and benchmarks_root.parent != benchmarks_root:
            benchmarks_root = benchmarks_root.parent

    try:
        text = base_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RecipeLeverUnavailableError(f"cannot read the agentic server script {base_path}: {exc}") from exc

    framework = str(bench.get("framework") or "").lower()
    result = _apply_edits(text, framework, argv=argv, remove_args=remove_args)

    # Append env block immediately after the launch call line.
    if env_sets or env_unsets:
        result = _insert_env_block(result, env_sets=env_sets, env_unsets=env_unsets)

    # Content-addressed name so identical lever combinations share one copy.
    digest = hashlib.sha256(result.encode("utf-8", "replace")).hexdigest()[:12]
    # Find the official recipe to derive the stem name.
    official_path = resolve_launch_server_script(bench)
    if official_path:
        stem = Path(official_path).stem
    else:
        stem = base_path.stem.split(".hl-")[0]
    copy_name = f"{stem}.hl-{digest}.sh"
    copy_path = base_path.parent / copy_name
    if not copy_path.exists():
        atomic_write_text(copy_path, result, preserve_mode=True)
    # Return path relative to benchmarks/.
    try:
        return str(copy_path.relative_to(benchmarks_root))
    except ValueError:
        return str(copy_path)


def _apply_edits(text: str, framework: str, *, argv: list[str], remove_args: list[str]) -> str:
    """Apply argv set/append and remove_args to the server command array in ``text``."""
    if not argv and not remove_args:
        return text

    result = _find_server_cmd_array(text, framework)
    if result is None:
        if not argv and not remove_args:
            return text
        raise RecipeLeverUnavailableError(
            f"the agentic recipe has no recognisable server-command array for framework={framework!r}; "
            "cannot deliver server-arg levers"
        )
    array_name, open_idx, close_idx = result
    lines = text.splitlines(keepends=True)

    # Check for spliced sub-arrays that might contain the flags being changed.
    spliced = _array_lines_contain_spliced(lines, open_idx, close_idx)

    # Build (flag_name, raw_line) pairs from the declared argv.
    from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

    parsed = tokenize_server_args_preserving_json(" ".join(argv)) if argv else None
    flag_pairs: list[tuple[str, str]] = []
    if parsed is not None:
        _, tokens = parsed
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("-"):
                flag_name = _normalize_flag_name(tok.split("=")[0])
                if "=" in tok:
                    raw_line = tok
                    flag_pairs.append((flag_name, raw_line))
                    i += 1
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    raw_line = f"{tok} {tokens[i + 1]}"
                    flag_pairs.append((flag_name, raw_line))
                    i += 2
                else:
                    raw_line = tok
                    flag_pairs.append((flag_name, raw_line))
                    i += 1
            else:
                i += 1

    # Validate: no spliced sub-array should contain the flags we're editing.
    all_flag_names = {fn for fn, _ in flag_pairs} | {_normalize_flag_name(r) for r in remove_args}
    if spliced and all_flag_names:
        # We cannot safely check which flags are inside spliced arrays without
        # following the variable, so refuse conservatively when any spliced
        # array exists alongside flag edits.
        raise RecipeLeverUnavailableError(
            f"the agentic recipe uses spliced sub-arrays ({spliced[0]}); "
            "cannot safely locate flags to edit without evaluating shell variables"
        )

    # Apply removals first, then set/append.
    for flag_spec in remove_args:
        flag_name = _normalize_flag_name(flag_spec.lstrip("-"))
        lines = _remove_flag_from_body(lines, open_idx, close_idx, flag_name)
        # close_idx may have shrunk; re-locate it.
        close_idx, open_idx = _relocate_array_bounds(lines, array_name, open_idx)

    for flag_name, raw_line in flag_pairs:
        lines = _set_flag_in_body(lines, open_idx, close_idx, flag_name, raw_line)
        close_idx, open_idx = _relocate_array_bounds(lines, array_name, open_idx)

    return "".join(lines)


def _relocate_array_bounds(lines: list[str], array_name: str, hint_open: int) -> tuple[int, int]:
    """Re-find the array bounds after edits that may have shifted lines."""
    # Scan from hint forward/backward to find open and close.
    # The array name is stable; search near hint.
    for i in range(max(0, hint_open - 2), min(len(lines), hint_open + 3)):
        if _CMD_ARRAY_OPEN_RE.match(lines[i].lstrip()) and array_name in lines[i]:
            open_idx = i
            break
    else:
        open_idx = hint_open
    # Find close from open.
    depth = 1
    j = open_idx + 1
    while j < len(lines) and depth > 0:
        stripped = lines[j].strip()
        if stripped == ")":
            depth -= 1
        j += 1
    return j - 1, open_idx


def _insert_env_block(text: str, *, env_sets: dict[str, str], env_unsets: list[str]) -> str:
    """Insert the env block after the line that expands the server command array."""
    block = _export_block(env_sets, env_unsets)
    if not block:
        return text
    lines = text.splitlines(keepends=True)
    # Find the first line that expands the array: "${CMD_ARRAY[@]}" ...
    insert_after = len(lines)
    for i, line in enumerate(lines):
        if _ARRAY_LAUNCH_RE.search(line) and ">" in line:
            insert_after = i + 1
            break
    # Also accept a write_command or run call immediately before the launch.
    return (
        "".join(lines[:insert_after]) + "# env levers from Hyperloom config\n" + block + "".join(lines[insert_after:])
    )


__all__ = [
    "RecipeLeverUnavailableError",
    "apply_recipe_levers",
    "launcher_overwritten_envs",
    "recipe_owns_argv",
    "resolve_launch_server_script",
]
