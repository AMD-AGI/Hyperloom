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
from inference_suite.recipe.model import ATOM_RUNNER_OWNED, ATOM_TOPOLOGY, SGLANG_RUNNER_OWNED, SGLANG_TOPOLOGY

TRANSFER_BACKEND_FLAG = "disaggregation-transfer-backend"
_FABRIC_PLACEHOLDER = "@fabric."
_HOST_PATH_ENV = frozenset({"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"})


def _context_flag(framework: Framework) -> str:
    """The canonical flag that sets the context ``framework``'s server serves; empty when it takes none."""
    flags = framework_facts(framework).server_flags
    return canonical_flag_name(flags.context_length) if flags is not None else ""


def _canonical(flags: Sequence[str]) -> set[str]:
    """``flags`` in canonical spelling."""
    return {canonical_flag_name(flag) for flag in flags}


#: The flags the PD launcher owns, by framework, in canonical spelling.
PD_RUNNER_OWNED = {
    Framework.SGLANG: SGLANG_RUNNER_OWNED
    | _canonical(SGLANG_TOPOLOGY.tp)
    | {"disaggregation-mode", "disaggregation-bootstrap-port", "disaggregation-ib-device", "dist-init-addr"}
    | {"nnodes", "node-rank", TRANSFER_BACKEND_FLAG, _context_flag(Framework.SGLANG)},
    Framework.ATOM: ATOM_RUNNER_OWNED
    | _canonical(ATOM_TOPOLOGY.tp)
    | {"kv-transfer-config", _context_flag(Framework.ATOM)},
}


@dataclass(frozen=True)
class RenderedRole:
    """A role's args and env with the launcher's keys taken out, what was taken out, and the transfer backend the
    args named (empty when they named none)."""

    args: tuple[str, ...]
    env: Mapping[str, str]
    stripped: tuple[str, ...]
    transfer_backend: str


def flag_groups(args: Sequence[str]) -> list[tuple[str, ...]]:
    """``args`` split into a flag with its values (an ``=`` value, else every following token that is no flag), or a
    token that follows no flag, each."""
    groups: list[list[str]] = []
    for token in args:
        if groups and not is_flag(token) and is_flag(groups[-1][0]) and "=" not in groups[-1][0]:
            groups[-1].append(token)
        else:
            groups.append([token])
    return [tuple(group) for group in groups]


def group_flag(group: Sequence[str]) -> str:
    """The canonical flag a group sets; empty for a token that follows no flag."""
    return canonical_flag_name(group[0].partition("=")[0]) if is_flag(group[0]) else ""


def render_role(framework: Framework, args: Sequence[str], env: Mapping[str, Any]) -> RenderedRole:
    """``args`` and ``env`` with the launcher's keys taken out."""
    owned = PD_RUNNER_OWNED[framework]
    kept: list[str] = []
    stripped: list[str] = []
    backend = ""
    for group in flag_groups(args):
        flag = group_flag(group)
        if flag not in owned:
            kept.extend(group)
            continue
        name, _, assigned = group[0].partition("=")
        stripped.append(name)
        if flag == TRANSFER_BACKEND_FLAG:
            backend = assigned or (group[1] if len(group) > 1 else "")
    kept_env: dict[str, str] = {}
    for key, value in env.items():
        text = str(value)
        if key in _HOST_PATH_ENV or text.startswith(_FABRIC_PLACEHOLDER):
            stripped.append(f"env {key}")
        else:
            kept_env[key] = text
    return RenderedRole(tuple(kept), kept_env, tuple(stripped), backend)


def recipe_role_args(framework: Framework, role: Mapping[str, Any]) -> list[str]:
    """A srt-slurm recipe role's ``args`` mapping as srtctl puts it on the engine command."""
    return list(srtctl_cli_args(framework.value, role.get("args") or {}))


__all__ = [
    "RenderedRole",
    "flag_groups",
    "group_flag",
    "recipe_role_args",
    "render_role",
]
