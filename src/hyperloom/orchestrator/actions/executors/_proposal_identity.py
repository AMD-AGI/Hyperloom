# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One identity for a specialist proposal and the explore variant it becomes.

The proposal is a mapping keyed ``extra_args``; the variant is a ``GridVariant``
keyed ``extra_server_args``. Both fingerprint into ``explore_search["tested"]``
under the variant's own args and envs folded together with the union of its
removal controls and the current stack's, so the key is stack-relative and a
proposal hashed on its own would not match the ledger.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.common.coerce import to_str_list

from ._canonical_fingerprint import canonical_fingerprint


__all__ = [
    "coerce_args",
    "controls_of",
    "effective_fingerprint",
    "is_executable",
    "normalize_proposal",
]


def coerce_args(value: Any) -> str:
    """Coerce a payload ``extra_args`` / ``extra_server_args`` value to a shell-arg string.

    The LLM sometimes emits the flags as a JSON list; ``str(list)`` would yield
    a Python repr the server rejects, so lists are space-joined into tokens.

    Args:
        value: The raw payload value (string, list/tuple, or ``None``).

    Returns:
        The coerced shell-arg string.
    """
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _args_mode_of(value: Any) -> str:
    """Coerce an args-mode to ``"replace"`` or ``"append"``."""
    return "replace" if str(value or "").strip().lower() == "replace" else "append"


def normalize_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Project a ``proposal_set`` entry onto the variant field set.

    Args:
        proposal: One ``specialist_done.proposal_set`` entry.

    Returns:
        ``name`` / ``extra_args`` / ``extra_envs`` / ``remove_args`` /
        ``unset_envs`` / ``args_mode`` / ``atomic`` / ``reason``, with the
        ``extra_args`` / ``extra_server_args`` alias resolved.
    """
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


def is_executable(fields: Mapping[str, Any]) -> bool:
    """Whether a server restart could apply these fields.

    A removal-only entry qualifies; a research-only entry does not.

    Args:
        fields: A :func:`normalize_proposal` result.

    Returns:
        ``True`` when the entry carries args, envs, or a removal/replacement
        control.
    """
    return bool(
        fields["extra_args"]
        or fields["extra_envs"]
        or fields["remove_args"]
        or fields["unset_envs"]
        or fields["args_mode"] == "replace"
    )


def controls_of(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the non-default removal/replacement controls.

    Args:
        fields: A :func:`normalize_proposal` result.

    Returns:
        The controls that differ from the default; all-default yields ``{}`` so
        a plain variant fingerprints unchanged.
    """
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
    """Fingerprint a variant against the stack it will be launched on.

    Removals union base-first with order preserved; a ``replace`` base
    args-mode wins over the variant's, since the base is what it launches on.

    Args:
        extra_args: The variant's own server-args string.
        extra_envs: The variant's own env mapping.
        controls: The variant's own non-default controls.
        base_remove_args: ``base_remove_args`` from the current stack.
        base_unset_envs: ``base_unset_envs`` from the current stack.
        base_args_mode: ``base_args_mode`` from the current stack.

    Returns:
        The 16-char fingerprint ``explore_search["tested"]`` is keyed on.
    """
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
