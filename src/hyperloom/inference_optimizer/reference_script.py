# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reference launch-recipe parsing and rendering.

A *reference recipe* is an operator-supplied launch script (local path or URL).
The optimizer lifts its **static, fully-resolved** server flags plus the
``export`` lines the denylist allows, and uses them as the lowest-priority base
for the baseline server args (EXPLORE can still override). The shell is never
executed — anything dynamic (``$VARS``: TP/CONC/ISL/OSL/model/port) is skipped,
because the optimizer's normal env seeding already owns those.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.env_safety import is_allowed_external_env_key

log = logging.getLogger(__name__)

# Flags that never belong in the lifted base: the optimizer's env seeding owns
# the workload + I/O, so drop these even when fully resolved.
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
    """Lift ``(server_args, envs, model)`` from a reference recipe; raises on a
    source that cannot be read or shell-parsed."""
    text = _read_source(source)

    envs = _extract_envs(text)
    line = _find_entrypoint_line(text, framework)
    if not line:
        log.warning("reference-script: no %s entrypoint in %r; carrying exports only", _entrypoint_markers(framework), source)
        return ReferenceRecipe(server_args="", envs=envs, model=None)

    server_args, model = _extract_server_args(shlex.split(line), framework)
    return ReferenceRecipe(server_args=server_args, envs=envs, model=model)


def _extract_envs(text: str) -> dict[str, str]:
    """Pull literal exports the denylist allows, resolving self-referential defaults."""
    envs: dict[str, str] = {}
    dropped: list[str] = []
    pat = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if not is_allowed_external_env_key(key):
            dropped.append(key)
            continue
        # strip surrounding quotes if present
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if _has_var(val):
            resolved = _resolve_self_default(key, val)
            if resolved is None:
                dropped.append(key)
                continue
            val = resolved
        envs[key] = val
    if dropped:
        log.info("reference recipe: dropped %d export(s): %s", len(dropped), ", ".join(sorted(set(dropped))))
    return envs


# ``${FOO:-1}`` / ``${FOO-1}``, capturing the name and the default.
_SELF_DEFAULT_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*):?-(.*)\}$")


def _resolve_self_default(key: str, val: str) -> str | None:
    """Return the literal default of ``${key:-default}``, else ``None``.

    Only the *self*-referential form counts: ``export FOO=${BAR:-1}`` depends on
    an unrelated variable, so its default is not FOO's effective value here.
    """
    m = _SELF_DEFAULT_RE.match(val)
    if not m or m.group(1) != key:
        return None
    default = m.group(2)
    return None if _has_var(default) else default


def _extract_server_args(
    tokens: list[str],
    framework: str,
) -> tuple[str, str | None]:
    """Walk entrypoint tokens as (flag, value) pairs; keep static flags only.

    Returns ``(server_args, model_basename_or_None)``. A flag whose value
    contains ``$`` is dropped together with its value (no orphan flags). The
    positional model and drop-listed flags are removed from ``server_args`` but
    the model is captured for caller-side model-gating.
    """
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


_APPLY_PATCH_FUNC = """\
apply_patch() {
  local patch_file="$1"
  for lvl in 1 0 2 3 4 5 6 7 8; do
    git -C "$FRAMEWORK_ROOT" apply --check -p"$lvl" "$patch_file" 2>/dev/null && \
      git -C "$FRAMEWORK_ROOT" apply -p"$lvl" "$patch_file" && return 0
  done
  echo "ERROR: could not apply $patch_file at any strip level" >&2
  return 1
}"""


def render_reference_script(
    *,
    framework: str,
    server_args: str,
    envs: dict[str, str] | None = None,
    model: str | None = None,
    tp: int | None = None,
    max_model_len: int | None = None,
    gpu_type: str | None = None,
    setup_commands: list[str] | None = None,
    patches: list[str] | None = None,
    framework_root: str | None = None,
    runtime: str | None = None,
) -> str:
    """Render a runnable ``*.sh`` artifact from a launch recipe.

    With only the base parameters, renders the ``current_setting.sh`` summary
    for the current optimization best.  When ``setup_commands``, ``patches``,
    or ``framework_root`` are supplied, renders an ``enablement_setting.sh``
    that additionally installs dependencies, applies patches, and launches the
    server.

    Args:
        framework: Framework identifier (``sglang``, ``vllm``, ``atom``, …).
        server_args: Extra server-arg CLI fragment.
        envs: Extra environment variable exports (values containing shell
            variable references are skipped).
        model: Model path, emitted as ``export MODEL=<path>`` so the launch
            line can reference ``$MODEL``.
        tp: Tensor-parallel degree, emitted as ``export TP=<n>``.
        max_model_len: Context length cap, emitted as ``export MAX_MODEL_LEN=<n>``.
        gpu_type: GPU type string, emitted as ``export GPU_TYPE=<s>``.
        setup_commands: Ordered install commands to run before launching.
            Only emit when generating an enablement artifact.
        patches: Ordered patch file paths (relative to the script or absolute)
            to apply via ``git apply``.  Requires ``framework_root``.
        framework_root: Framework source tree root where patches are applied.
            Emitted as ``export FRAMEWORK_ROOT=<path>`` and used in the
            ``apply_patch`` helper.
        runtime: If non-empty, a note is appended warning that this enablement
            round relied on an isolated attempt venv at the given path and the
            script does not reproduce that layer.

    Returns:
        The script text, always terminated by a newline.
    """
    fw = str(framework or "sglang").strip().lower()
    has_enablement = bool(setup_commands or patches or framework_root)

    lines: list[str] = ["#!/usr/bin/env bash"]
    if has_enablement:
        lines.append("# Auto-generated by hyperloom — enablement fix replay script.")
        lines.append("set -euo pipefail")
        lines.append('cd "$(dirname "${BASH_SOURCE[0]}")"')
    else:
        lines.append("# Auto-generated by hyperloom — current best launch recipe.")

    if model:
        lines.append(f"export MODEL={model}")
    if tp and int(tp) > 0:
        lines.append(f"export TP={int(tp)}")
    if max_model_len and int(max_model_len) > 0:
        lines.append(f"export MAX_MODEL_LEN={int(max_model_len)}")
    if gpu_type:
        lines.append(f"export GPU_TYPE={gpu_type}")
    if framework_root:
        lines.append(f"export FRAMEWORK_ROOT={framework_root}")
    for k, v in (envs or {}).items():
        if str(k).strip() and not _has_var(str(v)):
            lines.append(f"export {k}={v}")

    if runtime:
        lines.append("")
        lines.append(
            f"# NOTE: this enablement round used an isolated attempt venv at {runtime!r}."
        )
        lines.append("# That layer is not archived and cannot be reproduced by this script.")
        lines.append("# The script reproduces only the install commands, patches, and server args.")

    if setup_commands:
        lines.append("")
        for cmd in setup_commands:
            lines.append(cmd)

    if patches:
        lines.append("")
        lines.append(_APPLY_PATCH_FUNC)
        lines.append("")
        for patch in patches:
            lines.append(f'apply_patch "{patch}"')

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
