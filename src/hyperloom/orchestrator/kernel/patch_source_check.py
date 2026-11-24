# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks that a whole-file kernel patch is a complete, drop-in replacement for its target source."""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

from .patch_rebuild import COMPILED_SOURCE_SUFFIXES

TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}


def _source_text_looks_complete(text: str, suffix: str) -> bool:
    """Heuristically decide whether ``text`` is a complete source file."""
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(marker in stripped for marker in ("def ", "class ", "import ", "@triton.jit", "torch."))
    if suffix in COMPILED_SOURCE_SUFFIXES:
        return any(
            marker in stripped
            for marker in (
                "#include",
                "__global__",
                "__device__",
                "extern ",
                "namespace ",
                "template",
                "void ",
                "int ",
                "float ",
                "half",
                "torch::",
            )
        )
    return False


def validate_patch_source(patch: Path, target: Path) -> None:
    """Validate that ``patch`` is a complete drop-in replacement for ``target``.

    Checks that the patch is not a text artifact, that its suffix matches the
    target's, that it reads as text and looks like a complete source file, that
    it is compatible with the target (see
    :func:`_validate_replacement_compatibility`), and that Python patches
    compile.

    Args:
        patch (Path): The candidate replacement source file.
        target (Path): The file that would be replaced.

    Raises:
        ValueError: If the patch is incompatible or not a complete source file.
    """
    patch_suffix = patch.suffix.lower()
    target_suffix = target.suffix.lower()
    if patch_suffix in TEXT_ARTIFACT_SUFFIXES:
        raise ValueError(f"patch_path is not a complete source file: {patch}")
    if patch_suffix != target_suffix:
        raise ValueError(
            f"patch suffix {patch_suffix or '<none>'} does not match target suffix {target_suffix or '<none>'}"
        )
    text = patch.read_text(encoding="utf-8", errors="replace")
    if not _source_text_looks_complete(text, target_suffix):
        raise ValueError(f"patch_path does not look like a complete {target_suffix} source file: {patch}")
    target_text = target.read_text(encoding="utf-8", errors="replace")
    _validate_replacement_compatibility(text, target_text, target)
    if target_suffix == ".py":
        py_compile.compile(str(patch), doraise=True)


def _host_entry_functions(source_text: str) -> set[str]:
    """Extract public host (non-``__global__``) entry function names.

    Scans for C/C++ free-function definitions with common return types,
    excluding ``__global__`` kernels, leading-underscore helpers, and ``main``.

    Args:
        source_text (str): The C/C++/HIP source text to scan.

    Returns:
        set[str]: The set of discovered host entry-function names.
    """
    names: set[str] = set()
    pattern = re.compile(
        r"(?m)^\s*(?!__global__)(?:template\s*<[^>]+>\s*)?"
        r"(?:void|int|bool|float|double|auto|at::Tensor|torch::Tensor)\s+"
        r"([A-Za-z_]\w*)\s*\("
    )
    for match in pattern.finditer(source_text):
        name = match.group(1)
        if name.startswith("_") or name in {"main"}:
            continue
        names.add(name)
    return names


def _validate_replacement_compatibility(patch_text: str, target_text: str, target: Path) -> None:
    """Guard against patches that break the target's integration contract.

    Rejects patches that introduce standalone PYBIND11/TORCH_LIBRARY
    registrations absent from the target, drop a required ``namespace aiter``,
    or omit any host entry function the target exposes.

    Args:
        patch_text (str): The candidate replacement source text.
        target_text (str): The current target source text.
        target (Path): The target path (used in error messages).

    Raises:
        ValueError: If the patch is incompatible with the target's contract.
    """
    if "PYBIND11_MODULE" in patch_text and "PYBIND11_MODULE" not in target_text:
        raise ValueError("patch creates a standalone PYBIND11 module but target is a framework source file")
    if "TORCH_LIBRARY" in patch_text and "TORCH_LIBRARY" not in target_text:
        raise ValueError("patch creates standalone TORCH_LIBRARY registration absent from target")
    if "namespace aiter" in target_text and "namespace aiter" not in patch_text:
        raise ValueError("patch does not preserve namespace aiter")

    required = _host_entry_functions(target_text)
    if required:
        present = _host_entry_functions(patch_text)
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"patch does not preserve target host entry function(s) for {target}: " + ", ".join(missing[:12])
            )
