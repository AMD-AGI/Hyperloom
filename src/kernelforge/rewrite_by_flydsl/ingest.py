# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ingest — resolve a cross-language rewrite task into a :class:`RewriteSpec`.

Also provides host-entry auto-discovery: when the task does not name the source
host callable (``source_entry``), find the function that launches the target
kernel (e.g. the ``softmax(x)`` wrapper that calls
``softmax_kernel_online[grid](...)``, or the ``__host__`` launcher that calls
``attention_kernel<<<...>>>``). This is a best-effort convenience; tasks should
prefer to state ``source_entry`` explicitly.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from kernelforge.rewrite_by_flydsl import protocol
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

log = logging.getLogger(__name__)

# Curated candidate kinds that name a language this producer reads.
_KIND_LANGUAGE = {"hip_cpp": "hip"}

_SUFFIX_LANGUAGE = {
    ".hip": "hip",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
}

# A top-level C function signature, e.g. ``void attention(const float* q) {``.
# Anchored at column 0, which is what separates a definition from the ``if`` and
# ``for`` lines inside its body.
_C_SIGNATURE_RE = re.compile(r"^[A-Za-z_][\w\s\*&:<>]*?\b(\w+)\s*\(")


def resolve_source_language(source_path: str, declared: str = "") -> str:
    """Resolve the language a source kernel is written in.

    A caller's declaration wins, since it comes from a profiler that saw the
    kernel run. Reports ``""`` rather than defaulting to Triton when neither the
    declaration nor the file settles it.

    Args:
        source_path: Path to the source kernel.
        declared: Language or curated kind the caller named, if any.

    Returns:
        One of :data:`protocol.SUPPORTED_SOURCE_LANGUAGES`, or ``""``.
    """
    stated = str(declared or "").strip().lower().replace("-", "_")
    if stated in protocol.SUPPORTED_SOURCE_LANGUAGES:
        return stated
    if stated in _KIND_LANGUAGE:
        return _KIND_LANGUAGE[stated]
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix != ".py":
        return _SUFFIX_LANGUAGE.get(suffix, "")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "triton" if "triton" in text else ""


def _discover_c_source_entry(source_path: str, target_functions: list[str]) -> str:
    """Find the function performing a ``kernel<<<grid, block>>>(...)`` launch."""
    try:
        lines = Path(source_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        log.debug("source-entry discovery: cannot read %s: %s", source_path, error)
        return ""
    launches = tuple(f"{target}<<<" for target in target_functions)
    enclosing = ""
    for line in lines:
        signature = _C_SIGNATURE_RE.match(line)
        if signature and signature.group(1) not in target_functions:
            enclosing = signature.group(1)
        if enclosing and any(launch in line.replace(" ", "") for launch in launches):
            return enclosing
    return ""


def discover_source_entry(
    source_path: str,
    target_functions: list[str],
    *,
    source_language: str = "triton",
) -> str:
    """Find the function that launches one of ``target_functions``.

    For a Python source the heuristic parses with ``ast`` and returns the first
    top-level ``def`` whose body references ``<target_fn>`` as a subscript/call
    (a Triton ``kernel[grid](...)`` launch shows up as a ``Subscript`` on the
    kernel name, or a plain ``Call``), preferring a wrapper that takes a single
    positional arg (the classic ``op(x) -> y`` shape). A C-like source is scanned
    textually instead, since ``ast`` can only raise ``SyntaxError`` on it.
    Returns "" if none is found.
    """
    if not target_functions:
        return ""
    if source_language and source_language != "triton":
        return _discover_c_source_entry(source_path, target_functions)
    try:
        tree = ast.parse(Path(source_path).read_text())
    except (OSError, SyntaxError) as e:
        log.debug("source-entry discovery: cannot parse %s: %s", source_path, e)
        return ""

    targets = set(target_functions)

    def _references_target(fn_node: ast.FunctionDef) -> bool:
        for node in ast.walk(fn_node):
            # Triton launch: kernel[grid](...) -> Subscript with the kernel Name.
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in targets:
                    return True
            # Plain call: kernel(...) or launch helper referencing the name.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in targets:
                    return True
            if isinstance(node, ast.Name) and node.id in targets:
                return True
        return False

    candidates: list[tuple[int, str]] = []  # (num_pos_args, name)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _references_target(node):
            n_pos = len(node.args.args)
            candidates.append((n_pos, node.name))
    if not candidates:
        return ""
    # Prefer the simplest wrapper (fewest positional args -> closest to op(x)->y).
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def build_spec(
    *,
    op_name: str,
    source_kernel: str,
    flydsl_kernel: str,
    workspace: str,
    target_functions: list[str],
    source_entry: str = "",
    source_language: str = "",
    shapes: list[dict] | None = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
) -> RewriteSpec:
    """Resolve paths, the source language, and an auto-discovered entry."""
    source_kernel = str(Path(source_kernel).resolve())
    flydsl_kernel = str(Path(flydsl_kernel).resolve())
    language = resolve_source_language(source_kernel, source_language)
    if not language:
        log.warning(
            "rewrite: unresolved source language for %s (caller said %r); the port "
            "prompt and entry discovery stay language-neutral",
            Path(source_kernel).name,
            source_language,
        )

    entry = source_entry.strip()
    if not entry:
        entry = discover_source_entry(
            source_kernel,
            target_functions,
            source_language=language,
        )
        if entry:
            log.info("rewrite: auto-discovered source entry '%s' in %s", entry, Path(source_kernel).name)
    # The source host entry is only a HINT shown to the port agent — the supplied
    # or rewrite-prepared measurement driver owns how the reference/baseline is
    # invoked, so an unresolved entry does not block the pipeline (no fail-fast).
    if not entry:
        log.warning(
            "rewrite: no source host entry for op '%s' (not provided, not "
            "auto-discovered from %s); the port prompt will omit it. The driver "
            "still defines the reference/baseline.",
            op_name,
            Path(source_kernel).name,
        )

    return RewriteSpec(
        op_name=op_name,
        source_kernel=source_kernel,
        target_functions=list(target_functions or []),
        source_entry=entry,
        source_language=language,
        flydsl_kernel=flydsl_kernel,
        shapes=list(shapes or []),
        snr_threshold=snr_threshold,
        workspace=str(Path(workspace).resolve()),
    )
