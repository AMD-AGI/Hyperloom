# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shapes JSON and token list a forge tuning run is pointed at."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .gemm_shape_coverage import canonical_dtype, traced_gemm_shapes

log = logging.getLogger(__name__)


def _resolve_trace_shape_manifest(state, session_dir: Path) -> str:
    """Find the newest TraceShapeManifest this session produced.

    ``bypass_trace_analysis`` writes ``trace_shape_manifest.json`` next to its
    other bypass artifacts; forge calls it the preferred dense-shape source but
    nothing forwarded it, so the file was written and never read. Newest wins:
    a later trace reflects the currently resolved server args.
    """
    # Deduplicate: state.session_dir is usually the same path we were handed,
    # and an empty session then paid for two full-tree walks to find nothing.
    seen_roots: set[str] = set()
    roots: list[Path] = []
    for raw in (session_dir, Path(str(getattr(state, "session_dir", "") or session_dir))):
        if raw is None or not Path(raw).is_dir():
            continue
        key = str(Path(raw).resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(Path(raw))
    for root in roots:
        best: tuple[float, str] | None = None
        for found in Path(root).glob("**/trace_shape_manifest.json"):
            try:
                mtime = found.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, str(found))
        if best is not None:
            return best[1]
    return ""


def _is_forge_compatible_shapes_json(path: Path) -> bool:
    """Validate that a shapes JSON file matches forge's expected format.

    Forge expects: [{"M": int, "N": int, "K": int}, ...]
    or {"shapes": [{"M": int, "N": int, "K": int}, ...]}
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("shapes", [])
        if not isinstance(data, list) or not data:
            return False
        sample = data[0]
        if not isinstance(sample, dict):
            return False
        # Must have M/N/K keys (case-insensitive).
        keys = {k.upper() for k in sample}
        return {"M", "N", "K"}.issubset(keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _profile_shapes_are_fresh(state: Any) -> bool:
    """Return whether the latest profile matches the active workload/config."""
    return bool(state.profile_trace_matches_workload())


def _resolve_forge_shapes(
    state,
    session_dir: Path,
    *,
    require_fresh_profile: bool = False,
    precision: str = "",
) -> str:
    """Find TraceLens shapes JSON if available and in forge-compatible format.

    Forge dense tuners expect: [{"M": int, "N": int, "K": int}, ...]
    Only passes files that match this schema; incompatible formats are
    silently skipped so forge falls back to config.json shape derivation.

    ``precision`` scopes the traced shapes to the dtype the tuner will actually
    tune (a trace carries every GEMM dtype the model runs, e.g. FP8 projections
    alongside BF16 router heads). Empty means "no dtype scoping".

    When scoping is requested the candidate extraction runs first, because it is
    the only source whose dtype can be checked: a pre-rendered shapes artifact is
    a bare ``[{M,N,K}]`` list carrying no dtype or provenance, so an artifact
    recorded for another dtype would otherwise be handed to the tuner ahead of
    correctly-scoped shapes. Artifacts stay the fallback for the unscoped case
    and for when the trace yields nothing for this precision.
    """
    if require_fresh_profile and not _profile_shapes_are_fresh(state):
        log.info(
            "Forge GEMM shapes: latest TraceLens profile does not match the "
            "active workload/config; ignoring its shape artifacts"
        )
        return ""
    last_trace = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(last_trace, dict):
        return ""

    candidates: list[str] = []

    # Prefer explicit artifact fields when TraceLens exposes them.
    artifact_paths = last_trace.get("artifact_paths")
    for source, keys in (
        (last_trace, ("shapes_json", "shapes_path")),
        (artifact_paths if isinstance(artifact_paths, dict) else {}, ("shapes_json", "shapes", "gemm_shapes_json")),
    ):
        for key in keys:
            raw = str(source.get(key) or "").strip()
            if raw:
                candidates.append(raw)
    # Fallback: check beside candidates_path.
    candidates_path_str = last_trace.get("candidates_path") or ""
    if candidates_path_str:
        cand_file = Path(candidates_path_str)
        if cand_file.is_file():
            shapes_file = cand_file.parent / "shapes.json"
            candidates.append(str(shapes_file))

    # Extract the GEMM shapes observed by the latest TraceLens analysis. Older
    # traces can describe a backend that is no longer active.
    def _extracted() -> str:
        return _extract_gemm_shapes_from_candidates(
            str(last_trace.get("candidates_path") or ""),
            session_dir,
            precision=precision,
        )

    if canonical_dtype(precision):
        scoped = _extracted()
        if scoped:
            return scoped

    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and _is_forge_compatible_shapes_json(p):
            return str(p)

    return _extracted()


def _extract_gemm_shapes_from_candidates(candidates_path_str: str, session_dir: Path, *, precision: str = "") -> str:
    """Extract M,N,K from kernel_candidates.json hot_kernels input_shapes.

    Derives the GEMM dimensions actually observed during serving and writes a
    forge-compatible shapes JSON beside the candidates file, returning its path.

    ``precision`` scopes the result to one traced dtype. A trace records every
    GEMM the model runs, so an FP8 tuner would otherwise also receive the BF16
    router/head shapes; those rows are never looked up at serve time and they
    displace real FP8 shapes in the call-count ordering below. Empty keeps every
    dtype (historical behaviour).
    """
    import json as _json

    if not candidates_path_str:
        return ""
    cand_file = Path(candidates_path_str)
    if not cand_file.is_file():
        return ""

    try:
        data = _json.loads(cand_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    hot_kernels = data.get("hot_kernels", [])
    if not isinstance(hot_kernels, list):
        return ""

    # ``weight`` is the observed call count: decode GEMMs are invoked far more
    # often than prefill ones, so ordering by it puts the throughput-dominant
    # shapes first and they still get tuned when the tuner runs out of budget.
    # The same (M,N,K) can be reported by several kernels; keep the largest
    # count, otherwise a rare first sighting would outrank the hot one.
    weights: dict[tuple[int, int, int], int] = {}
    order: dict[tuple[int, int, int], int] = {}

    for kernel in hot_kernels:
        for key, weight in traced_gemm_shapes(kernel, precision=precision):
            if key not in order:
                order[key] = len(order)
            weights[key] = max(weights.get(key, 0), weight)

    if not weights:
        return ""

    # Most-called first; ties keep discovery order so output stays deterministic.
    ranked = sorted(weights, key=lambda key: (-weights[key], order[key]))
    shapes = [{"M": M, "N": N, "K": K} for M, N, K in ranked]

    out_path = cand_file.parent / "traced_gemm_shapes.json"
    try:
        out_path.write_text(_json.dumps(shapes, indent=2), encoding="utf-8")
    except OSError:
        return ""

    log.info(
        "extracted %d unique GEMM shapes from kernel_candidates.json -> %s",
        len(shapes),
        out_path,
    )
    return str(out_path)


def _path_is_existing_file(value: str) -> bool:
    """Safe ``Path.is_file()`` that never raises on an over-long pathname.

    A caller may hand us inline JSON content instead of a path; ``is_file()``
    raises ``OSError(ENAMETOOLONG)`` on such input. Treat any OSError as
    "not a file".
    """
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _normalize_tokens(value: Any) -> str:
    """Return a clean comma-separated token string for forge's ``--tokens``.

    forge parses ``--tokens`` as ``int(t) for t in value.split(",")``, so accept
    lists and bracketed strings and emit a bare comma-separated list.
    """
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip().strip("[](){}")
        if not text:
            return ""
        items = list(text.split(","))
    out: list[str] = []
    for it in items:
        s = str(it).strip().strip("'\"")
        if not s:
            continue
        try:
            out.append(str(int(float(s))))
        except (TypeError, ValueError):
            continue
    return ",".join(out)


def _normalize_forge_shapes_json(value: Any, workspace: Path) -> str:
    """Return a usable shapes-JSON *file path*, materializing inline content.

    Callers sometimes pass GEMM shapes as inline JSON in ``shapes_json`` instead
    of a file path; forge treats it strictly as a path. Normalize here:

    - existing file path -> returned unchanged
    - list/dict, or a string that parses as JSON -> written to
      ``<workspace>/forge_shapes.json`` and that path returned
    - anything else (empty / unparseable / non-existent path) -> ""
    """
    if value in (None, ""):
        return ""

    # Already-parsed inline content.
    if isinstance(value, (list, dict)):
        parsed: Any = value
    else:
        text = str(value).strip()
        if not text:
            return ""
        if _path_is_existing_file(text):
            return text
        # Inline JSON content (possibly Python-repr with single quotes).
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                try:
                    import ast

                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return ""
        else:
            # Non-JSON string that is not an existing file.
            return ""

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        out = workspace / "forge_shapes.json"
        out.write_text(json.dumps(parsed), encoding="utf-8")
        return str(out)
    except (OSError, TypeError, ValueError):
        return ""
