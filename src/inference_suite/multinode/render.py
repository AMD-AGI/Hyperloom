# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A role's args and env as the PD launcher runs them: the keys the launcher owns taken out and named.

The launcher sets the model path, bind address, ports, tensor-parallel size, disaggregation mode and bootstrap,
multi-node rendezvous, served name, context length, metrics switch and (for ATOM) the KV transfer config itself, so a
source's value for any of them is dropped and recorded. ``--disaggregation-transfer-backend`` is dropped too and its
value becomes the launch's transfer backend. Env values that name the fabric (``@fabric.*``) or host paths are the
hosts' own and are dropped the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inference_suite.argv import canonical_flag_name, is_flag
from inference_suite.frameworks import Framework, framework_facts
from inference_suite.recipe.inferencex import srtctl_cli_args
from inference_suite.recipe.model import ATOM_RUNNER_OWNED, SGLANG_RUNNER_OWNED

TRANSFER_BACKEND_FLAG = "disaggregation-transfer-backend"
_FABRIC_PLACEHOLDER = "@fabric."
_HOST_PATH_ENV = frozenset({"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"})


def _context_flag(framework: Framework) -> str:
    flags = framework_facts(framework).server_flags
    return canonical_flag_name(flags.context_length) if flags is not None else ""


#: The flags the PD launcher owns, by framework, in canonical spelling.
PD_RUNNER_OWNED = {
    Framework.SGLANG: SGLANG_RUNNER_OWNED
    | {"tp-size", "tensor-parallel-size", "tp", "disaggregation-mode", "disaggregation-bootstrap-port"}
    | {"disaggregation-ib-device", "dist-init-addr", "nnodes", "node-rank", TRANSFER_BACKEND_FLAG}
    | {_context_flag(Framework.SGLANG)},
    Framework.ATOM: ATOM_RUNNER_OWNED
    | {"tp", "tensor-parallel-size", "kv-transfer-config", _context_flag(Framework.ATOM)},
}


@dataclass(frozen=True)
class RenderedRole:
    """A role's args and env with the launcher's keys taken out, what was taken out, and the transfer backend the
    args named (empty when they named none)."""

    args: tuple[str, ...]
    env: Mapping[str, str]
    stripped: tuple[str, ...]
    transfer_backend: str


def _flag_end(args: Sequence[str], index: int) -> int:
    """The index past ``args[index]``'s values: an ``=`` value, else every following token that is no flag."""
    end = index + 1
    if "=" in args[index]:
        return end
    while end < len(args) and not is_flag(args[end]):
        end += 1
    return end


def flag_groups(args: Sequence[str]) -> list[tuple[str, ...]]:
    """``args`` split into a flag with its values, or a token that follows no flag, each."""
    groups: list[tuple[str, ...]] = []
    index = 0
    while index < len(args):
        end = _flag_end(args, index) if is_flag(args[index]) else index + 1
        groups.append(tuple(args[index:end]))
        index = end
    return groups


def group_flag(group: Sequence[str]) -> str:
    """The canonical flag a group sets; empty for a token that follows no flag."""
    return canonical_flag_name(group[0].partition("=")[0]) if is_flag(group[0]) else ""


def _strip_args(framework: Framework, args: Sequence[str]) -> RenderedRole:
    """``args`` with the launcher's flags taken out; no env."""
    owned = PD_RUNNER_OWNED[framework]
    kept: list[str] = []
    stripped: list[str] = []
    backend = ""
    for group in flag_groups(args):
        if group_flag(group) not in owned:
            kept.extend(group)
            continue
        name, _, assigned = group[0].partition("=")
        stripped.append(name)
        if group_flag(group) == TRANSFER_BACKEND_FLAG:
            backend = assigned or (group[1] if len(group) > 1 else "")
    return RenderedRole(tuple(kept), {}, tuple(stripped), backend)


def render_role(framework: Framework, args: Sequence[str], env: Mapping[str, Any]) -> RenderedRole:
    """``args`` and ``env`` with the launcher's keys taken out."""
    rendered = _strip_args(framework, args)
    stripped = list(rendered.stripped)
    kept_env: dict[str, str] = {}
    for key, value in env.items():
        text = str(value)
        if key in _HOST_PATH_ENV or text.startswith(_FABRIC_PLACEHOLDER):
            stripped.append(f"env {key}")
        else:
            kept_env[key] = text
    return RenderedRole(rendered.args, kept_env, tuple(stripped), rendered.transfer_backend)


def recipe_role_args(framework: Framework, role: Mapping[str, Any]) -> list[str]:
    """A srt-slurm recipe role's ``args`` mapping as srtctl puts it on the engine command."""
    return list(srtctl_cli_args(framework.value, dict(role.get("args") or {})))


__all__ = [
    "PD_RUNNER_OWNED",
    "TRANSFER_BACKEND_FLAG",
    "RenderedRole",
    "flag_groups",
    "group_flag",
    "recipe_role_args",
    "render_role",
]
