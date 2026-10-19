# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic logical and implementation identities for Forge experience."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from packaging.version import InvalidVersion, Version


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

#: One run of letters and digits, the unit camel-case boundaries are found in.
#: Whatever separates two runs -- ``_``, ``.``, ``-``, ``::`` -- is a boundary
#: the author already wrote, and is left exactly as it is.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

#: ``HTTPServer`` -> ``HTTP|Server``: the tail of a capitalized run starts the
#: next word when a lowercase letter follows it. Applied before
#: :data:`_LOWER_UPPER_BOUNDARY` so the run is cut once, at its real end.
_ACRONYM_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

#: ``fusedAdd`` -> ``fused|Add``, the ordinary camel-case boundary.
_LOWER_UPPER_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Longest piece a trailing capital is absorbed back into. ``Mo`` + ``E`` is an
#: acronym the boundaries cut in half; ``Kernel`` + ``O`` is a word followed by
#: another word. Two letters is what separates them across the kernel names in
#: the store and in this tree.
_ACRONYM_FRAGMENT_MAX = 2

#: Every spelling of "no version was observed". The three code paths that answer
#: that question answer it in three different words -- the framework's
#: distribution is not installed, no framework owns the source, or the campaign
#: declared nothing -- and the store holds pages under all of them. They name one
#: absence, so they are one value of the dimension.
_UNKNOWN_VERSIONS = frozenset({"", "unknown", "unspecified", "none", "unknown_version"})

#: ``v0.24.0`` is the tag spelling of release ``0.24.0``.
_TAG_V_PREFIX = re.compile(r"^v(?=\d)")


def canonical_owner_framework(value: str) -> str:
    """Canonicalize source-owner names shared by page and path identity."""
    owner = str(value or "").strip().lower().replace("-", "_")
    if owner in _NO_FRAMEWORK_SENTINELS:
        return _UNKNOWN
    return _OWNER_ALIASES.get(owner, owner)


def canonical_framework_version(value: str) -> str:
    """Return the release a framework version string names.

    One installed framework is named several ways at once. ``importlib.metadata``
    reports whatever the wheel was built as -- ``0.24.0+rocm723``,
    ``0.5.15.post1.dev20260724+g3d91a569ce`` -- while a campaign reading the
    package or the image it arrived in writes ``v0.24.0`` or
    ``v0.5.15.post1-rocm720-mi35x-20260724``. Every one of those names the source
    a port was written against; what follows the release names the machine that
    compiled it, which a port does not depend on. Keeping the difference gives
    one release a page per spelling, and a campaign reads only the page its own
    spelling addresses.
    """
    raw = str(value or "").strip().lower()
    if raw in _UNKNOWN_VERSIONS:
        return _UNKNOWN
    candidate = _TAG_V_PREFIX.sub("", raw)
    # An image tag joins the build to the release with the same character a
    # pre-release uses, so it is only distinguishable by failing to parse whole.
    for text in (candidate, candidate.split("-", 1)[0]):
        try:
            parsed = Version(text)
        except InvalidVersion:
            continue
        release = parsed.base_version
        if parsed.pre is not None:
            release += "".join(str(part) for part in parsed.pre)
        if parsed.post is not None:
            release += f".post{parsed.post}"
        return release
    return raw


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


def _split_camel_case(value: str) -> str:
    """Underscore-delimit camel-case boundaries inside each word of ``value``.

    A one-character piece is given back to the word it was cut from only when
    that word is itself a fragment of at most two letters, which is what an
    acronym written with a lowercase letter inside looks like once cut:
    ``MoE`` -> ``Mo|E`` -> ``MoE``, so ``FusedMoE`` reaches the same page as
    ``fused_moe``. After a whole word the capital is a word of its own and
    keeps its boundary -- ``ChunkFwdKernelO`` is ``chunk_fwd_kernel_o``, the
    spelling the source that declares it uses. Only boundaries found here are
    undone, never a single letter the author underscored themselves.
    """

    def split_word(match: re.Match[str]) -> str:
        pieces = _LOWER_UPPER_BOUNDARY.sub("_", _ACRONYM_BOUNDARY.sub("_", match.group())).split("_")
        merged: list[str] = []
        for piece in pieces:
            if merged and len(piece) == 1 and len(merged[-1]) <= _ACRONYM_FRAGMENT_MAX:
                merged[-1] += piece
            else:
                merged.append(piece)
        return "_".join(merged)

    return _WORD_RE.sub(split_word, value)


def normalize_operator_name(value: str) -> str:
    """Return the stable logical operator component used by kernel page keys.

    Case and word separators are not part of what a name means, and a kernel is
    spelled both ways across a source tree -- declared ``KdaPackedDecodeKernel``
    in a header and ``kda_packed_decode_kernel`` in the module that binds it.
    Both spellings have to land on one page, or the same kernel accumulates two
    half-filled histories and each campaign starts from the emptier one.
    """
    name = str(value or "").strip()
    if "::" in name:
        name = name.rsplit("::", 1)[-1]
    name = _strip_balanced_template_arguments(name)
    name = _split_camel_case(name)
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
    """Map declared source hints to canonical consumer-relative paths."""
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

    sources = _declared_source_texts(
        kernel_path=kernel_path,
        source_files=source_files,
        workspace=workspace,
        source_contents=source_contents,
    )
    source_symbols: set[str] = set()
    try:
        from kernelforge.mcp_server.tools.pmc import derive_kernel_names

        for source in sources:
            source_symbols.update(stable(derive_kernel_names(source)))
    except OSError:
        # Reading the sources is strict above; naming symbols inside them is best-effort, and a source that names
        # none leaves the path identity to tell the implementations apart.
        pass
    return sorted(source_symbols)


def _declared_source_texts(
    *,
    kernel_path: str,
    source_files: Iterable[str] | None,
    workspace: str,
    source_contents: dict[str, str] | None,
) -> list[str]:
    """Read every declared implementation source, in declaration order.

    A path that is not a file is one the declaration outruns and it contributes nothing. A file that is present but
    unreadable is not the same thing: dropping it would hash a subset of the sources the signature claims to cover,
    giving two different implementations one address, so it raises instead.
    """
    texts: list[str] = []
    seen_paths: set[str] = set()
    for raw in [kernel_path, *(source_files or [])]:
        if not raw or str(raw) in seen_paths:
            continue
        seen_paths.add(str(raw))
        source = source_contents.get(str(raw)) if source_contents is not None else None
        if source is None:
            path = Path(raw)
            if not path.is_absolute() and workspace:
                path = Path(workspace) / path
            if not path.is_file():
                continue
            source = path.read_text(errors="replace")
        texts.append(source)
    return texts


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
    "canonical_framework_version",
    "canonical_owner_framework",
    "derive_implementation_symbols",
    "hash_implementation_identity",
    "implementation_signature",
    "normalize_operator_name",
]
