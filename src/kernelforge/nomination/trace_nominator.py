# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trace-driven nominator: the three things the placeholder does not do.

``nomination.stub`` exists to make the contract runnable, and says so: it
"trusts Hyperloom's already-resolved rows and never looks at the trace". That
is enough for a demo and not enough for a serving target, because on a real
decode trace the rows that matter most are frequently the *unresolved* ones --
a hot AITER kernel arrives as a mangled HIP symbol with no ``source_file``, and
the placeholder drops it silently by way of ``is_resolved``.

This module adds, in order:

1. **Trace evidence.** ``gpu_pct`` on an incoming row is whatever the producer
   measured, which may be a whole-run average. When ``request.trace_path``
   points at a candidate manifest carrying per-kernel shares, those shares
   replace the row's, so ranking reflects the profiled window rather than the
   run that happened to contain it.
2. **Source resolution.** An unresolved row is searched for across configured
   source roots and promoted to ``resolved`` when exactly one plausible home is
   found. This is the "rescue" the placeholder's docstring points at.
3. **Evidence-weighted budget.** The placeholder splits the lane budget evenly.
   Here it is split in proportion to measured GPU share, subject to a floor, so
   a 30%-of-GPU kernel is not given the same wall clock as a 3% one.

Ranking stays a pure function of the inputs: same manifest and same roots give
the same targets, because a nominator that reorders between runs makes a
campaign impossible to reproduce.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernelforge.nomination import Candidate, NominationRequest, Target

#: Rows below this share are not worth a session: even a perfect rewrite of a
#: 1%-of-GPU kernel cannot move an end-to-end latency target.
MIN_GPU_PCT = 1.0

#: No target gets less than this, regardless of proportional split -- a session
#: too short to finish a measure/keep cycle produces nothing at all.
MIN_BUDGET_SEC = 600

#: Where to look for a kernel's implementation, in priority order. Overridable
#: with KERNELFORGE_SOURCE_ROOTS (os.pathsep-separated) so a deployment can
#: point at its own checkouts without a code change.
DEFAULT_SOURCE_ROOTS: tuple[str, ...] = (
    "/forge/aiter-amd",
    "/work/aiter-amd",
    "/opt/rocm/share/aiter",
)

#: Extensions worth searching. A kernel lives in one of these or is vendor
#: binary, in which case no amount of grepping will resolve it.
SOURCE_SUFFIXES = frozenset({".py", ".hip", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc"})

#: Directories that never contain editable kernel source.
SKIP_DIRS = frozenset({".git", "build", "__pycache__", "third_party", "dist", ".pytest_cache"})


def source_roots() -> tuple[Path, ...]:
    """Resolve the search roots, honouring the env override."""
    raw = os.environ.get("KERNELFORGE_SOURCE_ROOTS", "")
    names = [part for part in raw.split(os.pathsep) if part.strip()] if raw else list(DEFAULT_SOURCE_ROOTS)
    return tuple(Path(name) for name in names if Path(name).is_dir())


def _identifier_candidates(kernel_name: str) -> list[str]:
    """Reduce a device symbol to identifiers worth grepping for.

    HIP kernel names arrive templated and mangled
    (``void aiter::fmoe_kernel<...>(...)``). The searchable part is the bare
    identifier, so template arguments, namespaces, arguments and the leading
    return type are stripped, longest first.
    """
    name = kernel_name.strip()
    name = re.sub(r"\(.*$", "", name)  # drop argument list
    name = re.sub(r"<.*$", "", name)  # drop template arguments
    name = name.split("::")[-1]  # drop namespaces
    name = re.sub(r"^\s*(void|int|float|bool)\s+", "", name)
    name = name.strip()
    out: list[str] = []
    if len(name) >= 4 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        out.append(name)
        # Torch/aiter wrappers often expose the kernel without its
        # _kernel/_impl suffix; try that too, but never a fragment so short it
        # would match half the tree.
        trimmed = re.sub(r"_(kernel|impl|launcher|fwd|bwd)$", "", name)
        if trimmed != name and len(trimmed) >= 6:
            out.append(trimmed)
    return out


def _definition_pattern(identifier: str) -> re.Pattern[bytes]:
    """Match a *definition* of ``identifier``, not a mention of it.

    Substring matching is not good enough and actively dangerous here: a bare
    ``in path.read_bytes()`` test resolves a name like ``blind`` to whatever
    file happens to contain those five letters, and a campaign then spends its
    whole budget editing an unrelated file. So the identifier has to appear in
    a position that can only be a definition:

    - ``def name(``                         -- python / triton
    - ``__global__ ... name(``              -- HIP / CUDA device entry
    - ``void|static|inline|auto name(``     -- C++ host or device function
    - ``template<...> ... name(``           -- templated kernel
    - ``name = triton.jit`` / ``@triton.jit`` decorated defs are covered by
      the ``def`` form.
    """
    name = re.escape(identifier).encode("utf-8")
    return re.compile(
        rb"(?:^|\n)\s*(?:"
        rb"def\s+" + name + rb"\s*\("
        rb"|(?:template\s*<[^\n>]*>\s*)?(?:__global__|__device__|__launch_bounds__\([^\n)]*\))"
        rb"[^\n;{]*?\b" + name + rb"\s*\("
        rb"|(?:static\s+|inline\s+|extern\s+\"C\"\s+)*"
        rb"(?:void|auto|int|float|bool|__half|hipError_t)\s+\**" + name + rb"\s*\("
        rb")",
        re.MULTILINE,
    )


def resolve_source(kernel_name: str, roots: tuple[Path, ...]) -> tuple[str, str]:
    """Find the file that *defines* ``kernel_name``.

    Returns:
        ``(source_file, reason_class)``. ``source_file`` is empty when no
        definition site could be identified, and ``reason_class`` then says
        which way it failed so the caller can report it rather than guess.
    """
    identifiers = _identifier_candidates(kernel_name)
    if not identifiers:
        return "", "non_patchable_name"
    for identifier in identifiers:
        pattern = _definition_pattern(identifier)
        needle = identifier.encode("utf-8")
        hits: list[Path] = []
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                    continue
                if SKIP_DIRS & set(path.parts):
                    continue
                try:
                    blob = path.read_bytes()
                except OSError:
                    continue
                # Cheap reject before the regex: most files do not mention it.
                if needle not in blob:
                    continue
                if pattern.search(blob):
                    hits.append(path)
            if hits:
                break
        if len(hits) == 1:
            return str(hits[0]), "resolved"
        if len(hits) > 1:
            # Several definition sites (a header plus its .hip, or an arch
            # fan-out). Prefer a file whose stem is the identifier, else the
            # shallowest path, so the choice is deterministic.
            exact = [p for p in hits if p.stem == identifier]
            if len(exact) == 1:
                return str(exact[0]), "resolved"
            shallow = sorted(hits, key=lambda p: (len(p.parts), str(p)))
            return str(shallow[0]), "resolved"
    return "", "source_not_resolved"


def _trace_shares(trace_path: str) -> dict[str, float]:
    """Read per-kernel GPU shares from the trace artifact, if it carries them.

    Accepts a candidate manifest (the shape ``analyze_trace.py`` writes) and
    returns an empty mapping for anything else -- a raw chrome trace is not
    parsed here, because ranking must not depend on re-deriving numbers the
    producer already measured.
    """
    if not trace_path:
        return {}
    path = Path(trace_path)
    if not path.is_file():
        return {}
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("hot_kernels")
    if not isinstance(rows, list):
        return {}
    shares: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("kernel_name") or "").strip()
        try:
            pct = float(row.get("gpu_pct"))
        except (TypeError, ValueError):
            continue
        if name:
            shares[name] = pct
    return shares


def nominate_from_trace(
    request: NominationRequest,
    candidates: list[Candidate],
) -> list[Target]:
    """Rank on trace evidence, rescue unresolved rows, split budget by share.

    Args:
        request: The lane brief, for the trace path, budget and ceiling.
        candidates: Rows from :func:`kernelforge.nomination.read_candidates`.

    Returns:
        At most ``request.max_kernels`` targets, strongest first. Empty when
        nothing clears :data:`MIN_GPU_PCT`, which is a valid outcome: it says
        this trace has no kernel worth a session.
    """
    from kernelforge.nomination import Target

    shares = _trace_shares(request.trace_path)
    roots = source_roots()

    ranked: list[tuple[float, Candidate, str, str]] = []
    for candidate in candidates:
        if candidate.rejected:
            continue
        pct = shares.get(candidate.kernel_name, candidate.gpu_pct)
        if pct < MIN_GPU_PCT:
            continue
        source_file, reason_class = candidate.source_file, candidate.reason_class
        if not source_file:
            source_file, reason_class = resolve_source(candidate.kernel_name, roots)
        if not source_file:
            continue
        ranked.append((pct, candidate, source_file, reason_class))

    # Sort by measured share, then by name so ties are stable across runs.
    ranked.sort(key=lambda row: (-row[0], row[1].kernel_name))
    picked = ranked[: request.max_kernels]
    if not picked:
        return []

    total = sum(row[0] for row in picked) or 1.0
    targets: list[Target] = []
    for pct, candidate, source_file, reason_class in picked:
        budget = int(request.lane_budget_sec * (pct / total))
        targets.append(
            Target(
                kernel_name=candidate.kernel_name,
                source_file=source_file,
                budget_sec=max(MIN_BUDGET_SEC, budget),
                gpu_pct=pct,
                reason=(
                    f"trace nominator: {pct:.2f}% of profiled GPU time"
                    + ("" if reason_class == candidate.reason_class else f"; rescued via {reason_class}")
                ),
            )
        )
    return targets
