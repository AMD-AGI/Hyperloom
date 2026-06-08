#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""
Transform a (possibly-legacy) session_breakdown.json into the
`session_summary_v2.json` shape that SessionBreakdownPage.vue reads via
`GET /stats-v2/data/session_summary_v2.json?session_id=<sessionId>`.

Top-level response shape::

    {
      "source": "hyperloom_v2" | "claw_legacy_phased",
      "data": <SessionBreakdown>,
      "message": null,
      "error": null,
      "hint": null
    }

We auto-detect:
- "hyperloom_v2"        if all 4 V2 fields are already present (upstream fixed)
- "claw_legacy_phased"  otherwise -> we backfill the 4 gaps:
    1. baseline.extra_server_args / baseline.extra_envs
    2. capability_summary.<phase>.best_gain_pct
    3. phase_timeline[].extras renamed: candidate_extra_server_args -> best_extra_server_args
    4. kernel_lifecycle.detected[].geak / .oob structured as {decision, best_speedup}

Usage
-----
    # Single file (writes <input>.v2.json next to it)
    python scripts/transform_to_session_summary_v2.py session_breakdown.json

    # Explicit output path
    python scripts/transform_to_session_summary_v2.py session_breakdown.json -o out.json

    # Batch: every *.json under a directory, output into another directory
    python scripts/transform_to_session_summary_v2.py --in-dir ./remote_sessions --out-dir ./v2_out

    # Stdout (one file only)
    python scripts/transform_to_session_summary_v2.py session_breakdown.json -o -

The output file name in batch mode is `<session_id>.json` when we can read a
session id, otherwise it mirrors the input file's relative path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts by a sequence of keys.

    Args:
        d (Any): The starting object (typically a dict).
        *keys (str): Keys to descend through in order.
        default (Any): Value returned if any key is missing or a non-dict is
            encountered, or if the final value is None.

    Returns:
        Any: The nested value, or ``default`` when unreachable/None.
    """
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


# ---------------------------------------------------------------------------
# Gap 1: baseline.extra_server_args / extra_envs
# ---------------------------------------------------------------------------

def _patch_baseline(data: Dict) -> List[str]:
    """
    `final` already exposes top-level `extra_server_args` / `extra_envs`.
    `baseline` does not. Mirror them up from baseline.invocation when missing.

    Read-tolerant for the legacy ``extra_sglang_args`` key on input
    (renamed to ``extra_server_args``); the output always writes the
    canonical name only.

    Args:
        data (Dict): The session breakdown dict, mutated in place.

    Returns:
        List[str]: Human-readable notes describing each patch applied.
    """
    notes = []
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        return notes

    if "extra_server_args" not in baseline:
        # Back-compat: prefer the legacy key's value if the
        # input session_breakdown.json predates the rename.
        if "extra_sglang_args" in baseline:
            baseline["extra_server_args"] = baseline.pop("extra_sglang_args")
            notes.append("baseline.extra_server_args <- extra_sglang_args (legacy)")
        else:
            baseline["extra_server_args"] = ""  # legacy baseline ran with no extra flags
            notes.append("baseline.extra_server_args=''")

    if "extra_envs" not in baseline:
        inv_envs = safe_get(baseline, "invocation", "extra_envs", default=None)
        baseline["extra_envs"] = inv_envs if isinstance(inv_envs, dict) else {}
        notes.append("baseline.extra_envs <- baseline.invocation.extra_envs")

    return notes


# ---------------------------------------------------------------------------
# Gap 2: capability_summary.<phase>.best_gain_pct
# ---------------------------------------------------------------------------

def _best_gain_for_phase(data: Dict, phase: str) -> Optional[float]:
    """
    Derive a best_gain_pct from auxiliary structures when missing.

    Order of preference:
      1. param_search.<phase>.top_by_gain[0].gain_pct  (params, backends)
      2. phase_timeline[].extras.best_gain_pct_vs_base where action == phase
      3. None

    Args:
        data (Dict): The session breakdown dict.
        phase (str): The optimization phase name to look up.

    Returns:
        Optional[float]: The derived best gain percentage, or None if absent.
    """
    top_by_gain = safe_get(data, "param_search", phase, "top_by_gain", default=None)
    if isinstance(top_by_gain, list) and top_by_gain:
        v = safe_get(top_by_gain[0], "gain_pct")
        if isinstance(v, (int, float)):
            return float(v)

    pt = data.get("phase_timeline")
    if isinstance(pt, list):
        candidates = []
        for entry in pt:
            if isinstance(entry, dict) and entry.get("action") == phase:
                v = safe_get(entry, "extras", "best_gain_pct_vs_base")
                if isinstance(v, (int, float)):
                    candidates.append(float(v))
        if candidates:
            return max(candidates)

    return None


def _patch_capability_summary(data: Dict) -> List[str]:
    """Backfill ``best_gain_pct`` for each capability_summary phase.

    Derives the value via :func:`_best_gain_for_phase` for known phases and
    mirrors ``last_validated_gain_pct`` for ``validate_stack``.

    Args:
        data (Dict): The session breakdown dict, mutated in place.

    Returns:
        List[str]: Notes describing each patch applied.
    """
    notes = []
    cs = data.get("capability_summary")
    if not isinstance(cs, dict):
        return notes

    for phase in ("params", "backends", "sweep", "geak", "oob"):
        entry = cs.get(phase)
        if not isinstance(entry, dict):
            continue
        if "best_gain_pct" not in entry:
            entry["best_gain_pct"] = _best_gain_for_phase(data, phase)
            notes.append(f"capability_summary.{phase}.best_gain_pct")

    vs = cs.get("validate_stack")
    if isinstance(vs, dict) and "best_gain_pct" not in vs:
        # Mirror last_validated_gain_pct -> best_gain_pct so the frontend can
        # read either uniformly.
        v = vs.get("last_validated_gain_pct")
        vs["best_gain_pct"] = float(v) if isinstance(v, (int, float)) else None
        notes.append("capability_summary.validate_stack.best_gain_pct <- last_validated_gain_pct")

    return notes


# ---------------------------------------------------------------------------
# Gap 3: phase_timeline[].extras key rename
# ---------------------------------------------------------------------------

def _patch_phase_timeline(data: Dict) -> List[str]:
    """
    Frontend reads one of: `extra_server_args`, `best_extra_server_args`, `sglang_args`.
    Legacy puts the value under `candidate_extra_server_args`. Add an alias
    `best_extra_server_args` without removing the original key.

    Args:
        data (Dict): The session breakdown dict, mutated in place.

    Returns:
        List[str]: Notes describing each alias added.
    """
    notes = []
    pt = data.get("phase_timeline")
    if not isinstance(pt, list):
        return notes

    for entry in pt:
        if not isinstance(entry, dict):
            continue
        extras = entry.get("extras")
        if not isinstance(extras, dict):
            continue
        if (
            "best_extra_server_args" not in extras
            and "extra_server_args" not in extras
            and "sglang_args" not in extras
        ):
            # Back-compat: accept both ``candidate_extra_server_args``
            # (post-rename) and ``candidate_extra_sglang_args`` (legacy
            # pre-rename JSONs). Canonical key wins when both are present.
            cand = extras.get("candidate_extra_server_args")
            if cand is None:
                cand = extras.get("candidate_extra_sglang_args")
            if isinstance(cand, str):
                extras["best_extra_server_args"] = cand
                notes.append("phase_timeline.extras.best_extra_server_args <- candidate_extra_server_args")
    return notes


# ---------------------------------------------------------------------------
# Gap 4: detected[].geak / .oob structured aggregates
# ---------------------------------------------------------------------------

def _aggregate_backend(data: Dict, kernel_id: str, backend: str) -> Dict[str, Any]:
    """
    Walk kernel_decision_path[kid].steps and pick the best speedup + the most
    informative decision (priority: KEEP > NEEDS_REVIEW > PARTIAL > REVERT > FAILED).

    Args:
        data (Dict): The session breakdown dict.
        kernel_id (str): The kernel id (``kid``) to aggregate steps for.
        backend (str): The backend name to match (e.g. ``"geak"``, ``"oob"``).

    Returns:
        Dict[str, Any]: ``{"decision": str|None, "best_speedup": float|None}``.
    """
    paths = data.get("kernel_decision_path")
    if not isinstance(paths, list):
        return {"decision": None, "best_speedup": None}

    decision_rank = {
        "KEEP": 5,
        "NEEDS_REVIEW": 4,
        "PARTIAL": 3,
        "REVERT": 2,
        "FAILED": 1,
    }

    best_speedup: Optional[float] = None
    best_decision: Optional[str] = None
    best_rank = -1

    for entry in paths:
        if not isinstance(entry, dict):
            continue
        if entry.get("kid") != kernel_id:
            continue
        for step in entry.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if (step.get("backend") or "").lower() != backend:
                continue
            sp = step.get("speedup")
            if isinstance(sp, (int, float)) and (best_speedup is None or sp > best_speedup):
                best_speedup = float(sp)
            dec = step.get("outcome") or step.get("decision")
            if isinstance(dec, str):
                r = decision_rank.get(dec.upper(), 0)
                if r > best_rank:
                    best_rank = r
                    best_decision = dec.upper()

    return {"decision": best_decision, "best_speedup": best_speedup}


def _patch_detected_kernels(data: Dict) -> List[str]:
    """Backfill ``geak`` / ``oob`` aggregates on detected kernels.

    Fills any missing per-kernel backend aggregate using
    :func:`_aggregate_backend`.

    Args:
        data (Dict): The session breakdown dict, mutated in place.

    Returns:
        List[str]: Notes describing each kernel field filled.
    """
    notes = []
    detected = safe_get(data, "kernel_lifecycle", "detected", default=None)
    if not isinstance(detected, list):
        return notes

    for k in detected:
        if not isinstance(k, dict):
            continue
        kid = k.get("kernel_id") or ""

        # Only fill if missing or null
        if k.get("geak") is None:
            k["geak"] = _aggregate_backend(data, kid, "geak")
            notes.append(f"kernel[{kid}].geak")
        if k.get("oob") is None:
            k["oob"] = _aggregate_backend(data, kid, "oob")
            notes.append(f"kernel[{kid}].oob")

    return notes


# ---------------------------------------------------------------------------
# Master transform
# ---------------------------------------------------------------------------

def is_already_v2(data: Dict) -> bool:
    """Cheap heuristic: presence of all 4 V2 fields means it's already V2.

    Args:
        data (Dict): The session breakdown dict to inspect.

    Returns:
        bool: True if baseline, capability_summary, phase_timeline, and
        detected-kernel fields are all already in their V2 shape.
    """
    # Back-compat: a legacy session_breakdown.json carries
    # ``extra_sglang_args`` instead of ``extra_server_args``; treat
    # either as evidence the field is present.
    baseline_ok = isinstance(data.get("baseline"), dict) and (
        "extra_server_args" in data["baseline"]
        or "extra_sglang_args" in data["baseline"]
    )

    cs = data.get("capability_summary") or {}
    cs_ok = all(
        isinstance(cs.get(p), dict) and "best_gain_pct" in cs[p]
        for p in ("params", "backends", "sweep", "geak", "oob")
    )

    pt = data.get("phase_timeline") or []
    pt_ok = True
    for entry in pt:
        extras = (entry or {}).get("extras") or {}
        # Back-compat: accept either canonical or legacy key.
        has_candidate = (
            "candidate_extra_server_args" in extras
            or "candidate_extra_sglang_args" in extras
        )
        if has_candidate and not (
            "best_extra_server_args" in extras
            or "extra_server_args" in extras
            or "sglang_args" in extras
        ):
            pt_ok = False
            break

    detected = safe_get(data, "kernel_lifecycle", "detected", default=[]) or []
    detected_ok = all(
        isinstance(k, dict) and isinstance(k.get("geak"), dict) and isinstance(k.get("oob"), dict)
        for k in detected
    )

    return baseline_ok and cs_ok and pt_ok and detected_ok


def transform(data: Dict) -> Dict:
    """Produce the wrapped V2 response object for a session breakdown.

    The input is deep-copied and never mutated. If already V2 it is returned
    verbatim; otherwise the four backfill patches are applied and recorded
    under ``_v2_patches``.

    Args:
        data (Dict): The (possibly legacy) session breakdown dict.

    Returns:
        Dict: The wrapped response with ``source``, ``data``, ``message``,
        ``error``, and ``hint`` keys.
    """
    copy_data = copy.deepcopy(data)

    if is_already_v2(copy_data):
        return {
            "source": "hyperloom_v2",
            "data": copy_data,
            "message": None,
            "error": None,
            "hint": None,
        }

    patches: List[str] = []
    patches += _patch_baseline(copy_data)
    patches += _patch_capability_summary(copy_data)
    patches += _patch_phase_timeline(copy_data)
    patches += _patch_detected_kernels(copy_data)

    copy_data.setdefault("_v2_patches", patches)

    return {
        "source": "claw_legacy_phased",
        "data": copy_data,
        "message": None,
        "error": None,
        "hint": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def transform_file(in_path: Path, out_path: Optional[Path]) -> Path:
    """Read a session breakdown file, transform it, and write the V2 output.

    Args:
        in_path (Path): Path to the input session_breakdown JSON file.
        out_path (Optional[Path]): Output path; ``None`` writes ``<input>.v2.json``
            next to the input, and ``Path("-")`` writes to stdout.

    Returns:
        Path: The path written to, or ``Path("-")`` for stdout.

    Raises:
        ValueError: If the input file is not a JSON object.
    """
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{in_path} is not a JSON object")

    wrapped = transform(data)

    if out_path is None:
        out_path = in_path.with_suffix(".v2.json")

    if str(out_path) == "-":
        json.dump(wrapped, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return Path("-")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(wrapped, f, ensure_ascii=False, indent=2)

    return out_path


def _output_name_for(input_file: Path, in_dir: Path) -> Path:
    """Derive the batch-mode output file name for an input file.

    Prefers ``<session_id>.json`` when a session id can be read; otherwise
    mirrors the input's relative path with a ``.json`` suffix.

    Args:
        input_file (Path): The input session breakdown file.
        in_dir (Path): The batch input root directory.

    Returns:
        Path: The relative output file name to use.
    """
    rel = input_file.relative_to(in_dir)
    session_id = None
    try:
        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        session_id = safe_get(data, "session", "session_id")
    except Exception:
        session_id = None

    if session_id:
        return Path(f"{session_id}.json")
    # fallback: mirror input layout, force .json
    return rel.with_suffix(".json")


def main():
    """Parse CLI arguments and transform one or more session breakdown files.

    Supports single-file, explicit-output, stdout, and recursive batch
    (``--in-dir`` / ``--out-dir``) modes. Per-file errors are reported to
    stderr without aborting the batch.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="Input session_breakdown JSON file(s).")
    parser.add_argument("-o", "--out", help="Output path (single input only). Use '-' for stdout.")
    parser.add_argument("--in-dir", help="Directory containing *.json to transform recursively.")
    parser.add_argument("--out-dir", help="Directory to write transformed files into.")
    parser.add_argument("--pattern", default="session_breakdown.json",
                        help="When using --in-dir, file name glob (default: session_breakdown.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.in_dir:
        if not args.out_dir:
            parser.error("--in-dir requires --out-dir")
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.rglob(args.pattern))
        if not files:
            print(f"[no input] no files matching '{args.pattern}' under {in_dir}", file=sys.stderr)
            return
        for f in files:
            try:
                target = out_dir / _output_name_for(f, in_dir)
                final = transform_file(f, target)
                if not args.quiet:
                    print(f"[ok] {f} -> {final}")
            except Exception as e:
                print(f"[err] {f}: {e}", file=sys.stderr)
        return

    if not args.inputs:
        parser.error("provide one or more input files, or use --in-dir / --out-dir")

    if args.out and len(args.inputs) != 1:
        parser.error("--out can only be used with a single input file")

    for inp in args.inputs:
        in_path = Path(inp)
        out_path = Path(args.out) if args.out else None
        if args.out == "-":
            out_path = Path("-")
        try:
            final = transform_file(in_path, out_path)
            if not args.quiet and str(final) != "-":
                print(f"[ok] {in_path} -> {final}")
        except Exception as e:
            print(f"[err] {in_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
