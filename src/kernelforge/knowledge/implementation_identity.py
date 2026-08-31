# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic logical and implementation identities for Forge experience."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


_UNKNOWN = "unknown"
_NO_FRAMEWORK_SENTINELS = {"", "standalone", "none", "unknown"}
_OWNER_ALIASES = {
    "aiter": "aiter",
    "aiter_meta": "aiter",
    "sglang": "sglang",
    "vllm": "vllm",
}
_STABLE_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ITANIUM_MANGLED_RE = re.compile(r"^_Z\d")


def canonical_owner_framework(value: str) -> str:
    """Canonicalize source-owner names shared by page and path identity."""
    owner = str(value or "").strip().lower().replace("-", "_")
    if owner in _NO_FRAMEWORK_SENTINELS:
        return _UNKNOWN
    return _OWNER_ALIASES.get(owner, owner)


def _strip_balanced_template_arguments(value: str) -> str:
    """Remove balanced C++-style template argument groups, including nesting."""
    out: list[str] = []
    depth = 0
    for character in value:
        if character == "<":
            depth += 1
            continue
        if character == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            out.append(character)
    return "".join(out) if depth == 0 else value


def normalize_operator_name(value: str) -> str:
    """Return the stable logical operator component used by kernel page keys."""
    name = str(value or "").strip()
    if "::" in name:
        name = name.rsplit("::", 1)[-1]
    name = _strip_balanced_template_arguments(name)
    name = name.lower().replace(".", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"_kernel$", "", name)
    return name or _UNKNOWN


def _workspace_relative(path: str, workspace: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(workspace) / resolved
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _strip_optional_src(parts: tuple[str, ...]) -> tuple[str, ...]:
    return parts[1:] if parts and parts[0].lower() == "src" else parts


def _canonical_source_path(path: str, workspace: str, framework: str) -> str:
    """Canonicalize one editable path across roots, aliases, and ``src/``."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(workspace) / resolved
    resolved = resolved.resolve()
    owner = canonical_owner_framework(framework)
    aliases = {alias for alias, canonical in _OWNER_ALIASES.items() if canonical == owner}
    lowered = [part.lower() for part in resolved.parts]
    owner_indexes = [index for index, part in enumerate(lowered) if part in aliases]
    if owner != _UNKNOWN and owner_indexes:
        suffix = _strip_optional_src(tuple(resolved.parts[owner_indexes[-1] + 1 :]))
        return Path(owner, *suffix).as_posix()

    relative = Path(_workspace_relative(str(resolved), workspace))
    relative_parts = _strip_optional_src(relative.parts)
    if owner != _UNKNOWN:
        if relative_parts and relative_parts[0].lower() in aliases:
            relative_parts = relative_parts[1:]
        return Path(owner, *relative_parts).as_posix()
    return Path(*relative_parts).as_posix()


def canonical_editable_source_map(
    *,
    workspace: str,
    kernel_path: str,
    source_files: Iterable[str] | None,
    framework: str,
) -> dict[str, str]:
    """Map declared source hints to canonical consumer-relative paths.

    The map supports cross-repository KB matching; it is not an edit allowlist.
    """
    mapping: dict[str, str] = {}
    for raw in [kernel_path, *(source_files or [])]:
        if not raw:
            continue
        canonical = _canonical_source_path(str(raw), workspace, framework)
        relative = _workspace_relative(str(raw), workspace)
        previous = mapping.get(canonical)
        if previous is not None and previous != relative:
            raise ValueError(f"ambiguous canonical editable source path: {canonical}")
        mapping[canonical] = relative
    return dict(sorted(mapping.items()))


def canonical_editable_source_paths(
    *,
    workspace: str,
    kernel_path: str,
    source_files: Iterable[str] | None,
    framework: str,
) -> list[str]:
    """Return sorted package-relative paths for the declared source hints."""
    return list(
        canonical_editable_source_map(
            workspace=workspace,
            kernel_path=kernel_path,
            source_files=source_files,
            framework=framework,
        )
    )


def derive_implementation_symbols(
    *,
    kernel_path: str,
    source_files: Iterable[str] | None,
    workspace: str = "",
    source_contents: dict[str, str] | None = None,
) -> list[str]:
    """Derive stable symbols from the declared implementation entry points."""

    def stable(names: Iterable[str]) -> set[str]:
        return {
            value
            for name in names
            if (value := str(name or "").strip())
            and _STABLE_SYMBOL_RE.fullmatch(value)
            and not _ITANIUM_MANGLED_RE.match(value)
        }

    source_symbols: set[str] = set()
    try:
        from kernelforge.mcp_server.tools.pmc import derive_kernel_names

        seen_paths: set[str] = set()
        for raw in [kernel_path, *(source_files or [])]:
            if not raw or str(raw) in seen_paths:
                continue
            seen_paths.add(str(raw))
            try:
                source = None
                if source_contents is not None:
                    source = source_contents.get(str(raw))
                if source is None:
                    path = Path(raw)
                    if not path.is_absolute() and workspace:
                        path = Path(workspace) / path
                    source = path.read_text(errors="replace")
            except OSError:
                continue
            source_symbols.update(stable(derive_kernel_names(source)))
    except Exception:
        # Identity extraction is best-effort; callers safely fall back to path identity.
        pass
    return sorted(source_symbols)


def implementation_signature(
    *,
    workspace: str,
    kernel_path: str,
    source_files: Iterable[str] | None,
    framework: str,
    source_contents: dict[str, str] | None = None,
) -> tuple[str, dict]:
    """Hash the canonical editable implementation contract."""
    payload = {
        "source_paths": canonical_editable_source_paths(
            workspace=workspace,
            kernel_path=kernel_path,
            source_files=source_files,
            framework=framework,
        ),
        "implementation_symbols": derive_implementation_symbols(
            kernel_path=kernel_path,
            source_files=source_files,
            workspace=workspace,
            source_contents=source_contents,
        ),
    }
    return hash_implementation_identity(payload), payload


def hash_implementation_identity(payload: dict) -> str:
    """Hash one canonical implementation identity payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "canonical_editable_source_map",
    "canonical_editable_source_paths",
    "canonical_owner_framework",
    "derive_implementation_symbols",
    "hash_implementation_identity",
    "implementation_signature",
    "normalize_operator_name",
]
