# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One identity for a specialist proposal, its deltas, and the explore variant it becomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from hyperloom.common.coerce import to_str_list

from ._canonical_fingerprint import canonical_fingerprint

if TYPE_CHECKING:
    from ._grid_base import GridVariant


__all__ = [
    "ArtifactDelta",
    "ConfigDelta",
    "PatchDelta",
    "SetupDelta",
    "coerce_args",
    "controls_of",
    "effective_fingerprint",
    "is_executable",
    "normalize_proposal",
    "to_grid_variant",
]


@dataclass(frozen=True)
class ConfigDelta:
    """A server-arg / env change that requires no workspace mutation."""

    extra_args: str = ""
    extra_envs: dict[str, str] = field(default_factory=dict)
    remove_args: list[str] = field(default_factory=list)
    unset_envs: list[str] = field(default_factory=list)
    args_mode: str = "append"


@dataclass(frozen=True)
class PatchDelta:
    """A diff the specialist wrote into its worktree."""

    patches_written: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    artifacts_written: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactDelta:
    """A non-diff tuned artifact (e.g. an autotuned config JSON)."""

    source: str = ""
    target: str = ""
    kind: str = ""
    description: str = ""


@dataclass(frozen=True)
class SetupDelta:
    """A setup command sequence (e.g. install a wheel)."""

    commands: list[str] = field(default_factory=list)


def coerce_args(value: Any) -> str:
    """Coerce a payload ``extra_args`` / ``extra_server_args`` value to a shell-arg string."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _args_mode_of(value: Any) -> str:
    """Coerce an args-mode to ``"replace"`` or ``"append"``."""
    return "replace" if str(value or "").strip().lower() == "replace" else "append"


def normalize_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Project a ``proposal_set`` entry onto the variant field set."""
    envs = proposal.get("extra_envs")
    return {
        "name": str(proposal.get("name") or "").strip(),
        "extra_args": coerce_args(proposal.get("extra_args") or proposal.get("extra_server_args")),
        "extra_envs": {str(k): str(v) for k, v in envs.items()} if isinstance(envs, Mapping) else {},
        "remove_args": to_str_list(proposal.get("remove_args")),
        "unset_envs": to_str_list(proposal.get("unset_envs")),
        "args_mode": _args_mode_of(proposal.get("args_mode")),
        "atomic": bool(proposal.get("atomic")),
        "reason": str(proposal.get("reason") or "").strip(),
    }


def to_grid_variant(proposal: Mapping[str, Any]) -> "GridVariant":
    """Project a proposal entry onto a GridVariant for the explore executor."""
    from ._grid_base import GridVariant

    fields = normalize_proposal(proposal)
    gv = GridVariant(
        name=fields["name"],
        extra_server_args=fields["extra_args"],
        extra_envs=dict(fields["extra_envs"]),
        note=str(proposal.get("reason") or ""),
        remove_args=list(fields["remove_args"]),
        unset_envs=list(fields["unset_envs"]),
        args_mode=fields["args_mode"],
    )
    gv.provenance = str(proposal.get("provenance") or "specialist")  # type: ignore[attr-defined]
    gv.scope = str(proposal.get("scope") or "")  # type: ignore[attr-defined]
    gv.overlay_pythonpath = str(proposal.get("overlay_pythonpath") or "")  # type: ignore[attr-defined]
    gv.accepted_kernels = [  # type: ignore[attr-defined]
        str(k).strip() for k in (proposal.get("accepted_kernels") or []) if str(k).strip()
    ]
    gv.kb_evidence = list(proposal.get("kb_evidence") or [])  # type: ignore[attr-defined]
    gv.pr_evidence = list(proposal.get("pr_evidence") or [])  # type: ignore[attr-defined]
    gv.source_evidence = list(proposal.get("source_evidence") or [])  # type: ignore[attr-defined]
    return gv


def is_executable(fields: Mapping[str, Any]) -> bool:
    """Whether a server restart could apply these fields."""
    return bool(
        fields["extra_args"]
        or fields["extra_envs"]
        or fields["remove_args"]
        or fields["unset_envs"]
        or fields["args_mode"] == "replace"
    )


def controls_of(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the non-default removal/replacement controls."""
    out: dict[str, Any] = {}
    if fields["remove_args"]:
        out["remove_args"] = list(fields["remove_args"])
    if fields["unset_envs"]:
        out["unset_envs"] = list(fields["unset_envs"])
    if fields["args_mode"] == "replace":
        out["args_mode"] = "replace"
    return out


def effective_fingerprint(
    extra_args: Any,
    extra_envs: Any,
    *,
    controls: Mapping[str, Any] | None = None,
    base_remove_args: Any = None,
    base_unset_envs: Any = None,
    base_args_mode: Any = None,
) -> str:
    """Fingerprint a variant against the stack it will be launched on."""
    identity = dict(controls or {})
    remove_args = list(dict.fromkeys(to_str_list(base_remove_args) + to_str_list(identity.get("remove_args"))))
    unset_envs = list(dict.fromkeys(to_str_list(base_unset_envs) + to_str_list(identity.get("unset_envs"))))
    if remove_args:
        identity["remove_args"] = remove_args
    if unset_envs:
        identity["unset_envs"] = unset_envs
    if _args_mode_of(base_args_mode) == "replace":
        identity["args_mode"] = "replace"
    return canonical_fingerprint(extra_args, extra_envs, **identity)
