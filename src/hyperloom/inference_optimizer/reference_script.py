# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reference launch-recipe parsing and rendering."""

from __future__ import annotations

import logging
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import is_allowed_external_env_key, is_secret_shaped_env_name
from hyperloom.common.overlay import validate_overlay_pythonpath

log = logging.getLogger(__name__)

# Flags that never belong in the lifted base: the optimizer's env seeding owns the workload + I/O, so drop these even
# when fully resolved.
_DROP_FLAGS = frozenset(
    {
        "--port",
        "--host",
        "--served-model-name",
        "--result-dir",
        "--result-filename",
    }
)
# Flags dropped by prefix (result-*, served-model-* variants, log redirection).
_DROP_PREFIXES = ("--result-", "--served-model")


@dataclass(frozen=True)
class ReferenceRecipe:
    """Static facts lifted from a reference launch recipe."""

    server_args: str = ""
    envs: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    launch_controls: dict[str, Any] = field(default_factory=dict)


_CONTROLS_PREFIX = "# hyperloom-launch-controls: "


def _validate_launch_controls(controls: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(controls, dict) or set(controls) - {
        "overlay_pythonpath",
        "unset_envs",
        "remove_args",
        "args_mode",
    }:
        raise ValueError("invalid reference launch controls")
    for key in ("unset_envs", "remove_args"):
        values = controls.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"reference {key} must be a list of strings")
    if any(not is_allowed_external_env_key(name) for name in controls.get("unset_envs", [])):
        raise ValueError("reference unset_envs contains a forbidden environment name")
    mode = controls.get("args_mode", "append")
    if not isinstance(mode, str) or mode not in {"append", "replace"}:
        raise ValueError("invalid reference args_mode")
    overlay = controls.get("overlay_pythonpath", "")
    if not isinstance(overlay, str):
        raise ValueError("reference overlay_pythonpath must be a string")
    validate_overlay_pythonpath(overlay)
    return controls


def _read_source(source: str) -> str:
    """Return the recipe text; raises when the source cannot be read."""
    s = str(source or "").strip()
    if s.startswith(("http://", "https://")):
        from .baseline_comparison.inferencex_client import _fetch_raw

        return _fetch_raw(s).decode("utf-8", errors="replace")
    return Path(s).read_text(encoding="utf-8", errors="replace")


def _entrypoint_markers(framework: str) -> tuple[str, ...]:
    fw = str(framework or "").strip().lower()
    if "atom" in fw:
        return ("atom.entrypoints",)
    if "vllm" in fw:
        return ("vllm serve",)
    return ("sglang.launch_server",)


def _join_continuations(text: str) -> list[str]:
    """Collapse backslash line-continuations into single logical lines."""
    logical: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            buf += line
            logical.append(buf)
            buf = ""
    if buf:
        logical.append(buf)
    return logical


def _find_entrypoint_line(text: str, framework: str) -> str | None:
    markers = _entrypoint_markers(framework)
    for line in _join_continuations(text):
        if any(m in line for m in markers):
            return line
    return None


def _strip_redirection(tokens: list[str]) -> list[str]:
    """Drop shell redirection / backgrounding tail (``> log 2>&1 &``)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("&", ";"):
            break
        if t.startswith(">") or t.startswith("<") or t.startswith("2>") or "2>&1" in t:
            # redirection target may be the next token
            if t in (">", "<", "2>") and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _has_var(s: str) -> bool:
    return "$" in s


def _is_flag(tok: str) -> bool:
    return tok.startswith("--")


def _flag_name(tok: str) -> str:
    return tok.split("=", 1)[0]


def _should_drop_flag(name: str) -> bool:
    if name in _DROP_FLAGS:
        return True
    return any(name.startswith(p) for p in _DROP_PREFIXES)


def parse_reference_script(source: str, *, framework: str) -> ReferenceRecipe:
    """Lift static launch settings from an untrusted local or remote recipe.

    Executable overlay imports require resuming the owning session or explicitly
    running its exported launcher; a recipe comment cannot authorize Python code.
    """
    text = _read_source(source)
    controls: dict[str, Any] = {}
    for raw in text.splitlines():
        if raw.startswith(_CONTROLS_PREFIX):
            if controls:
                raise ValueError("duplicate reference launch controls")
            controls = _validate_launch_controls(json.loads(raw[len(_CONTROLS_PREFIX) :]))
            if controls.get("overlay_pythonpath"):
                raise ValueError(
                    "reference overlay_pythonpath imports executable code; resume the owning session "
                    "or review and run the exported launcher directly"
                )
    envs = _extract_envs(text)
    line = _find_entrypoint_line(text, framework)
    if not line:
        log.warning(
            "reference-script: no %s entrypoint in %r; carrying exports only", _entrypoint_markers(framework), source
        )
        return ReferenceRecipe(server_args="", envs=envs, model=None, launch_controls=controls)

    server_args, model = _extract_server_args(shlex.split(line), framework)
    return ReferenceRecipe(server_args=server_args, envs=envs, model=model, launch_controls=controls)


def _extract_envs(text: str) -> dict[str, str]:
    """Pull literal exports the denylist allows, resolving self-referential defaults."""
    envs: dict[str, str] = {}
    dropped: list[str] = []
    pat = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
    lines = iter(text.splitlines(keepends=True))
    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if not is_allowed_external_env_key(key):
            dropped.append(key)
            continue
        while True:
            try:
                tokens = shlex.split(val)
                break
            except ValueError:
                continuation = next(lines, None)
                if continuation is None:
                    tokens = []
                    break
                val += continuation
        if len(tokens) != 1 and val.strip():
            dropped.append(key)
            continue
        literal = tokens[0] if tokens else ""
        if _has_shell_expansion(val):
            resolved = _resolve_self_default(key, literal)
            if resolved is None:
                dropped.append(key)
                continue
            literal = resolved
        envs[key] = literal
    if dropped:
        log.info("reference recipe: dropped %d export(s): %s", len(dropped), ", ".join(sorted(set(dropped))))
    return envs


def _has_shell_expansion(value: str) -> bool:
    """Recognize dynamic shell syntax while preserving quoted literal values."""
    quote = ""
    escaped = False
    for char in value.strip():
        if escaped:
            escaped = False
        elif quote == "'":
            if char == "'":
                quote = ""
        elif char == "\\":
            escaped = True
        elif char == quote:
            quote = ""
        elif not quote and char in ("'", '"'):
            quote = char
        elif char in ("$", "`") or (not quote and char in ";&|<>()"):
            return True
    return False


# ``${FOO:-1}`` / ``${FOO-1}``, capturing the name and the default.
_SELF_DEFAULT_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*):?-(.*)\}$")


def _resolve_self_default(key: str, val: str) -> str | None:
    """Return the literal default of ``${key:-default}``, else ``None``."""
    m = _SELF_DEFAULT_RE.match(val)
    if not m or m.group(1) != key:
        return None
    default = m.group(2)
    return None if _has_shell_expansion(default) else default


def _extract_server_args(
    tokens: list[str],
    framework: str,
) -> tuple[str, str | None]:
    """Walk entrypoint tokens as (flag, value) pairs; keep static flags only."""
    tokens = _strip_redirection(tokens)
    # Skip the entrypoint prefix itself.
    fw = str(framework or "").strip().lower()
    start = 0
    if "vllm" in fw and "atom" not in fw:
        # ``vllm serve <model> ...`` → entrypoint is the first two tokens.
        for i, t in enumerate(tokens):
            if t == "serve":
                start = i + 1
                break
    else:
        # ``python3 -m <module> ...``: flags follow the ``-m module`` run.
        for i, t in enumerate(tokens):
            if t == "-m" and i + 1 < len(tokens):
                start = i + 2
                break

    model: str | None = None
    kept: list[str] = []
    i = start
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _is_flag(tok):
            # positional (the model for ``vllm serve $MODEL``); capture, drop.
            if model is None and tok not in ("serve",):
                model = None if _has_var(tok) else Path(tok).name
            i += 1
            continue
        name = _flag_name(tok)
        # capture model from --model / --model-path even though we drop it.
        is_model_flag = name in ("--model", "--model-path")
        if "=" in tok:  # --flag=value (self-contained)
            value = tok.split("=", 1)[1]
            if is_model_flag:
                if not _has_var(value):
                    model = Path(value).name
                i += 1
                continue
            if _has_var(value) or _should_drop_flag(name):
                i += 1
                continue
            kept.append(tok)
            i += 1
            continue
        # ``--flag value`` or bare ``--flag``
        has_value = i + 1 < n and not tokens[i + 1].startswith("-")
        if has_value:
            value = tokens[i + 1]
            if is_model_flag:
                if not _has_var(value):
                    model = Path(value).name
                i += 2
                continue
            if _has_var(value) or _should_drop_flag(name):
                i += 2  # drop BOTH flag and its value (no orphan flag)
                continue
            kept.append(tok)
            kept.append(value)
            i += 2
            continue
        # bare store-true flag
        if _should_drop_flag(name):
            i += 1
            continue
        kept.append(tok)
        i += 1

    return " ".join(kept), model


#: ``vcs`` value of a framework root with no version control of its own.
#: Duplicates ``orchestrator.bringup.trees.VCS_NONE``, which this layer cannot
#: import without inverting the package layering.
VCS_NONE = "none"

# git -C resolves a relative patch path against the target tree, not the caller, so the script dir is baked in.
_APPLY_PATCH_GIT = """\
apply_patch() {
  local patch_file="$SCRIPT_DIR/$1"
  for lvl in 1 0 2 3 4 5 6 7 8; do
    if git -C "$FRAMEWORK_ROOT" apply --check -p"$lvl" "$patch_file" 2>/dev/null; then
      git -C "$FRAMEWORK_ROOT" apply -p"$lvl" "$patch_file"
      return 0
    fi
  done
  echo "ERROR: could not apply $patch_file at any strip level" >&2
  return 1
}"""

# For a root with no git of its own -- an installed wheel -- where ``git
# apply`` has nothing to run against.
_APPLY_PATCH_NO_GIT = """\
apply_patch() {
  local patch_file="$SCRIPT_DIR/$1"
  for lvl in 1 0 2 3 4 5 6 7 8; do
    if patch -p"$lvl" --fuzz=0 --dry-run -d "$FRAMEWORK_ROOT" -i "$patch_file" >/dev/null 2>&1; then
      patch -p"$lvl" --fuzz=0 -d "$FRAMEWORK_ROOT" -i "$patch_file"
      return 0
    fi
  done
  echo "ERROR: could not apply $patch_file at any strip level" >&2
  return 1
}"""


def _apply_patch_func(framework_root_vcs: str) -> str:
    """Return the ``apply_patch`` helper that matches the target tree's kind.

    Args:
        framework_root_vcs: The framework root's vcs discriminant. Only
            :data:`VCS_NONE` selects the POSIX ``patch`` channel.

    Returns:
        str: The shell function body.
    """
    return _APPLY_PATCH_NO_GIT if framework_root_vcs == VCS_NONE else _APPLY_PATCH_GIT


def render_reference_script(
    *,
    framework: str,
    server_args: str,
    envs: dict[str, str] | None = None,
    overlay_pythonpath: str | None = None,
    unset_envs: list[str] | None = None,
    remove_args: list[str] | None = None,
    args_mode: str = "append",
    model: str | None = None,
    tp: int | None = None,
    max_model_len: int | None = None,
    gpu_type: str | None = None,
    setup_commands: list[str] | None = None,
    framework_root: str | None = None,
    framework_root_vcs: str = "",
    runtime: str | None = None,
    rounds: list[dict[str, Any]] | None = None,
) -> str:
    """Render a runnable ``*.sh`` artifact from a launch recipe."""
    fw = str(framework or "sglang").strip().lower()
    has_enablement = bool(setup_commands or framework_root or rounds)

    lines: list[str] = ["#!/usr/bin/env bash"]
    controls = _validate_launch_controls(
        {
            **({"overlay_pythonpath": overlay_pythonpath} if overlay_pythonpath else {}),
            **({"unset_envs": unset_envs} if unset_envs else {}),
            **({"remove_args": remove_args} if remove_args else {}),
            **({"args_mode": args_mode} if args_mode == "replace" else {}),
        }
    )
    if controls:
        lines.append(_CONTROLS_PREFIX + json.dumps(controls, sort_keys=True))
    for name in unset_envs or []:
        lines.append(f"unset {shlex.quote(name)}")
    if has_enablement:
        lines.append("# Auto-generated by hyperloom — enablement fix replay script.")
        lines.append("set -euo pipefail")
        lines.append('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"')
        lines.append('cd "$SCRIPT_DIR"')
    else:
        lines.append("# Auto-generated by hyperloom — current best launch recipe.")

    exported_model = bool(model) and "/" in str(model)
    if exported_model:
        lines.append(f"export MODEL={shlex.quote(str(model))}")
    elif model:
        # A bare basename cannot be launched; the parser records one when the operator recipe named the model without
        # a path.
        lines.append(f"# model: {model}")
    if has_enablement and not exported_model:
        # The launch line dereferences $MODEL, which set -u would kill first.
        lines.append(': "${MODEL:?set MODEL to the model path before running}"')
    if tp and int(tp) > 0:
        lines.append(f"export TP={int(tp)}")
    if max_model_len and int(max_model_len) > 0:
        lines.append(f"export MAX_MODEL_LEN={int(max_model_len)}")
    if gpu_type:
        lines.append(f"export GPU_TYPE={shlex.quote(str(gpu_type))}")
    if framework_root:
        lines.append(f"export FRAMEWORK_ROOT={shlex.quote(str(framework_root))}")
    for k, v in (envs or {}).items():
        if not str(k).strip():
            continue
        # The artifact is archived and uploaded, so a credential-shaped value is named but never written out.
        if is_secret_shaped_env_name(k):
            lines.append(f"# export {k}=<redacted; supply manually>")
        else:
            lines.append(f"export {k}={shlex.quote(str(v))}")
    if overlay_pythonpath:
        prefix = shlex.quote(str(overlay_pythonpath))
        lines.append(f'export PYTHONPATH={prefix}"${{PYTHONPATH:+:$PYTHONPATH}}"')

    if runtime:
        lines.append("")
        lines.append(f"# NOTE: this enablement round used an isolated attempt venv at {runtime!r}.")
        lines.append("# That layer is not archived and cannot be reproduced by this script.")
        lines.append("# The script reproduces only the install commands, patches, and server args.")

    if setup_commands:
        lines.append("")
        for cmd in setup_commands:
            lines.append(cmd)

    if rounds:
        if any(rnd.get("patches") for rnd in rounds):
            lines.append("")
            lines.append(_apply_patch_func(framework_root_vcs))
        for rnd in rounds:
            if rnd.get("patches"):
                lines.append("")
                for patch in rnd["patches"]:
                    lines.append(f"apply_patch {shlex.quote(str(patch))}")
            if rnd.get("artifacts"):
                lines.append("")
                for art in rnd["artifacts"]:
                    # $SCRIPT_DIR stays outside the quotes so the shell still expands it.
                    src = f'"$SCRIPT_DIR"/{shlex.quote(art["archive_path"])}'
                    lines.append(f"install -D {src} {shlex.quote(art['target'])}")

    args = str(server_args or "").strip()
    lines.append("")
    if "atom" in fw:
        entry = f"python3 -m atom.entrypoints.openai_server {args}".rstrip()
    elif "vllm" in fw:
        entry = f"vllm serve $MODEL {args}".rstrip()
    else:
        entry = f"python3 -m sglang.launch_server --model-path=$MODEL {args}".rstrip()
    lines.append(entry)
    return "\n".join(lines) + "\n"


__all__ = [
    "ReferenceRecipe",
    "parse_reference_script",
    "render_reference_script",
]
