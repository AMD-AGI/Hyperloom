# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a runnable launch script (``current_setting.sh``, the enablement replay) from a launch recipe.

The operator's ``--reference-script`` is read by ``inference_suite``'s user file parser, which rejects the
``# hyperloom-launch-controls:`` line this renderer records, so a rendered script with launch controls is an audit
record rather than a launch file.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.env_safety import is_allowed_external_env_key, is_credential_name
from hyperloom.common.overlay import validate_overlay_pythonpath

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


def _launch_command(framework: str) -> str:
    """The serving entrypoint the rendered script runs for ``framework``."""
    name = str(framework or "").strip().lower()
    if "atom" in name:
        return "python3 -m atom.entrypoints.openai_server"
    if "vllm" in name:
        return "vllm serve $MODEL"
    return "python3 -m sglang.launch_server --model-path=$MODEL"


def _shell_ready_server_args(server_args: Any) -> str:
    """Return ``server_args`` with every token quoted for the shell that will run it.

    The script this renders is executed by a shell, and the args were stored as
    a command line, not as shell source. ``--compilation-config
    {"max_cudagraph_capture_size":8,"cudagraph_mode":"NONE"}`` interpolated raw
    is brace-expanded and quote-stripped into three words -- the flag, a value
    that is no longer JSON, and a stray operand -- so the one setting that kept
    the server from segfaulting silently did not reach it.

    Tokenized by the splitter the launch path itself uses, so the script hands
    the server the same argv every other consumer got, rather than a second
    opinion about where the tokens are.
    """
    text = str(server_args or "").strip()
    if not text:
        return ""
    from inference_suite.argv import split_args_preserving_json

    tokens = split_args_preserving_json(text)
    if tokens is None:
        # Unparseable to the canonical splitter: quote it whole rather than
        # guess where it breaks. A single odd operand beats three wrong ones.
        return shlex.quote(text)
    return " ".join(shlex.quote(tok) for tok in tokens)


@dataclass(frozen=True)
class LaunchSettings:
    """The launch configuration a script replays: server args, envs and the launch controls applied with them."""

    server_args: str
    envs: dict[str, str] = field(default_factory=dict)
    overlay_pythonpath: str | None = None
    unset_envs: list[str] | None = None
    remove_args: list[str] | None = None
    args_mode: str = "append"


@dataclass(frozen=True)
class ServingTarget:
    """What a rendered script serves: the model and the TP, context length and GPU type it exports."""

    model: str | None = None
    tp: int | None = None
    max_model_len: int | None = None
    gpu_type: str | None = None


def render_reference_script(
    *,
    framework: str,
    launch: LaunchSettings,
    target: ServingTarget = ServingTarget(),
    setup_script: str | None = None,
) -> str:
    """Render a runnable ``*.sh`` artifact from a launch recipe.

    ``setup_script`` names a script beside this one that prepares the
    environment. It runs before the launch envs are exported, as the session's
    own setup did, and the launch follows only when it succeeds.
    """
    overlay_pythonpath, unset_envs = launch.overlay_pythonpath, launch.unset_envs
    model, tp, max_model_len, gpu_type = target.model, target.tp, target.max_model_len, target.gpu_type

    lines: list[str] = ["#!/usr/bin/env bash"]
    controls = _validate_launch_controls(
        {
            **({"overlay_pythonpath": overlay_pythonpath} if overlay_pythonpath else {}),
            **({"unset_envs": unset_envs} if unset_envs else {}),
            **({"remove_args": launch.remove_args} if launch.remove_args else {}),
            **({"args_mode": launch.args_mode} if launch.args_mode == "replace" else {}),
        }
    )
    if controls:
        lines.append(_CONTROLS_PREFIX + json.dumps(controls, sort_keys=True))
    for name in controls.get("unset_envs", []):
        lines.append(f"unset {shlex.quote(name)}")
    if setup_script:
        lines.append("# Auto-generated by hyperloom — enablement fix replay script.")
        lines.append("set -euo pipefail")
        lines.append('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"')
        lines.append('cd "$SCRIPT_DIR"')
    else:
        lines.append("# Auto-generated by hyperloom — current best launch recipe.")

    exported_model = "/" in str(model)
    if exported_model:
        lines.append(f"export MODEL={shlex.quote(str(model))}")
    elif model:
        # A model named without a path cannot be launched, so it is recorded rather than exported.
        lines.append(f"# model: {model}")
    if setup_script:
        if not exported_model:
            # The launch line dereferences $MODEL, which set -u would kill first.
            lines.append(': "${MODEL:?set MODEL to the model path before running}"')
        # $SCRIPT_DIR stays outside the quotes so the shell still expands it.
        lines.append(f'bash "$SCRIPT_DIR"/{shlex.quote(setup_script)}')
    for name, value in (("TP", tp), ("MAX_MODEL_LEN", max_model_len)):
        if value and int(value) > 0:
            lines.append(f"export {name}={int(value)}")
    if gpu_type:
        lines.append(f"export GPU_TYPE={shlex.quote(str(gpu_type))}")
    for k, v in launch.envs.items():
        if not str(k).strip():
            continue
        # The artifact is archived and uploaded, so a credential-shaped value is named but never written out.
        if is_credential_name(k):
            lines.append(f"# export {k}=<redacted; supply manually>")
        else:
            lines.append(f"export {k}={shlex.quote(str(v))}")
    if overlay_pythonpath:
        prefix = shlex.quote(str(overlay_pythonpath))
        lines.append(f'export PYTHONPATH={prefix}"${{PYTHONPATH:+:$PYTHONPATH}}"')

    args = _shell_ready_server_args(launch.server_args)
    lines.append("")
    lines.append(f"{_launch_command(framework)} {args}".rstrip())
    return "\n".join(lines) + "\n"


__all__ = [
    "LaunchSettings",
    "ServingTarget",
    "render_reference_script",
]
