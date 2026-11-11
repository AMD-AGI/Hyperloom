# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declared deliverable a specialist round hands back."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _clean_seq(values: Any) -> tuple[str, ...]:
    """Return a de-duplicated tuple of non-empty strings, order preserved."""
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def declared_targets(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the tree-relative paths a ``specialist_done`` payload declares it touched.

    The nested ``deliverable`` mapping is authoritative; a payload without one
    is read flat.
    """
    if not payload:
        return ()
    declared = payload.get("deliverable")
    source: Mapping[str, Any] = declared if isinstance(declared, Mapping) else payload
    return _clean_seq(source.get("targets"))


def declared_launch(payload: Mapping[str, Any] | None) -> tuple[dict[str, str], str]:
    """Return the envs and server args a ``specialist_done`` payload declares its patch runs under.

    Read like :func:`declared_targets`, falling back to the payload's own ``extra_envs`` and ``extra_server_args``.
    """
    if not payload:
        return {}, ""
    declared = payload.get("deliverable")
    source: Mapping[str, Any] = declared if isinstance(declared, Mapping) else payload
    envs = source.get("envs")
    if not isinstance(envs, Mapping):
        envs = payload.get("extra_envs")
    server_args = source.get("server_args")
    if server_args is None:
        server_args = payload.get("extra_server_args")
    return (
        {str(k): str(v) for k, v in envs.items() if str(k).strip()} if isinstance(envs, Mapping) else {},
        server_args.strip() if isinstance(server_args, str) else "",
    )


__all__ = ["declared_launch", "declared_targets"]
