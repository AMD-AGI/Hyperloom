#!/usr/bin/env python3
###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Independent (TraceLens-free) trace analysis backend for the bypass route.

This tool is the runtime target of ``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``.
It replaces the TraceLens agent / TraceLens deterministic scripts entirely:
it never imports or shells out to TraceLens. It reads the torch-profiler
Kineto trace produced by the ``profile`` step and emits the same downstream
artifact contract the Coordinator / kernel-agent expect:

    - ``<run_dir>/bypass/analysis.md``       (human-readable report)
    - ``<run_dir>/kernel_candidates.json``   (hot kernels + skipped + groups)
    - ``<run_dir>/bypass/summary.json``      (routed vs skipped audit)
    - ``<workspace>/reports/<roofline>.json``(per-kernel roofline sidecar)
    - ``<run_dir>/trace_input_manifest.json``(input record)

and prints a single JSON result object to stdout in the shape
``kernel_request_handlers._shape_tool_result`` consumes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Sibling modules live next to this tool (invoked by absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bypass_report as _report  # noqa: E402
import _bypass_trace_reader as _reader  # noqa: E402
import _trace_shape_manifest as _tsm  # noqa: E402

# Shared provenance builder (WP-0). Optional import: this tool is also invoked
# standalone by absolute path, where the ``hyperloom`` package may not be on the
# path -- in that case we fall back to a minimal env-derived stub.
try:
    from hyperloom.common.provenance import build_provenance as _shared_build_provenance
except Exception:  # noqa: BLE001 — standalone invocation without the package installed.
    _shared_build_provenance = None
from _idle_gate import (  # noqa: E402
    build_graph_under_recorded_warning,
    build_high_idle_warning,
    resolve_idle_pct_threshold,
)
from _denoise_steps import count_profiler_steps, resolve_perstep_divisor  # noqa: E402
from _io_utils import atomic_write_json, utc_now, write_text  # noqa: E402


AGGREGATION_SCOPE_FULL = "full_trace"
AGGREGATION_SCOPE_STEADY = "steady_state"


def _run_stamp() -> str:
    """Return a compact UTC timestamp used to name the per-run output dir."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _maybe_enrich_rocprof(
    kernel_roofline_path: Path,
    candidates_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Dead rocprof-compute enrichment hook; kept only for the summary key.

    Gated off by default via ``HYPERLOOM_ROCPROF_ROOFLINE_ENRICH``. No
    ``rocprof_roofline`` module exists, so opting in cannot enrich anything —
    the import raises and the except branch returns an error status.

    Args:
        kernel_roofline_path: Path to the written ``kernel_roofline.json``.
        candidates_path: Path to the written ``kernel_candidates.json``.
        run_dir: Per-run output directory used as the profiling workdir.

    Returns:
        ``{"status": "disabled"}`` when the env gate is off (the only reachable
        non-error result), otherwise ``{"status": "error: ..."}``. Progress is
        logged to stderr so stdout stays a single result-JSON line.
    """
    enrich_value = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", "0").strip().lower()
    if enrich_value not in {"1", "true", "yes", "on"}:
        return {"status": "disabled"}
    try:
        from rocprof_roofline import enrich_kernel_roofline_sidecar

        timeout_sec = int(os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800") or 1800)
        enrich_summary = enrich_kernel_roofline_sidecar(
            sidecar_path=str(kernel_roofline_path),
            candidates_path=str(candidates_path),
            workdir=str(run_dir),
            timeout_sec_per_kernel=timeout_sec,
            log_fn=None,
        )
        print(
            "[rocprof_enrich] "
            f"matched={enrich_summary.get('matched', 0)} "
            f"skipped={enrich_summary.get('skipped', 0)} "
            f"failed={enrich_summary.get('failed', 0)} "
            f"rows={enrich_summary.get('rows', 0)}",
            file=sys.stderr,
        )
        return enrich_summary
    except Exception as exc:  # noqa: BLE001 — enrichment must never break bypass
        msg = f"error: {type(exc).__name__}: {exc}"
        print(f"[rocprof_enrich] skipped: {msg}", file=sys.stderr)
        return {"status": msg}


def _emit_quality_warnings(analyze: dict[str, Any], warnings: list[dict[str, Any]]) -> None:
    """Append analysis-quality health signals so weak analyses are never silent.

    Emits (all non-fatal) when:
      * too much GPU time is unclassified (``Others`` share high) -> taxonomy gap;
      * op-attribution coverage is near-zero -> correlation chain broken;
      * steady-state windowing was requested but fell back to the full trace;
      * only CUDA-graph capture shards were found (no main profiler trace).

    Thresholds are env-tunable (``HYPERLOOM_BYPASS_OTHERS_WARN_PCT`` default 40,
    ``HYPERLOOM_BYPASS_CORR_WARN_PCT`` default 10). Only called when the trace
    yielded GPU kernels (otherwise ``bypass_no_gpu_kernels`` already fired).

    Args:
        analyze: Result of :func:`_bypass_trace_reader.analyze_trace`.
        warnings: The ``trace_health_warnings`` list to append to (mutated).
    """
    try:
        others_thr = float(os.environ.get("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", "40") or 40)
    except ValueError:
        others_thr = 40.0
    try:
        corr_thr = float(os.environ.get("HYPERLOOM_BYPASS_CORR_WARN_PCT", "10") or 10)
    except ValueError:
        corr_thr = 10.0

    rollup = _report._category_rollup(analyze)
    others_pct = next((r["gpu_pct"] for r in rollup if r["category"] == "Others"), 0.0)
    if others_pct >= others_thr:
        warnings.append(
            {
                "code": "bypass_high_unclassified_share",
                "severity": "warning",
                "message": (
                    f"{others_pct}% of GPU time is unclassified (category=Others); "
                    "the kernel-name taxonomy likely needs extension for this workload."
                ),
            }
        )

    corr_pct = float((analyze.get("attribution") or {}).get("attributed_pct") or 0.0)
    if corr_pct < corr_thr:
        warnings.append(
            {
                "code": "bypass_low_op_correlation",
                "severity": "info",
                "message": (
                    f"op-attribution coverage is {corr_pct}% (< {corr_thr}%); kernel-name "
                    "classification still applies, but op names/shapes are largely unresolved "
                    "(expected under cudagraph/torch.compile replay)."
                ),
            }
        )

    if analyze.get("steady_window_status"):
        warnings.append(
            {
                "code": "bypass_steady_fallback_full_trace",
                "severity": "info",
                "message": (
                    "steady-state windowing requested but no repeating window found; "
                    "fell back to full-trace share aggregation."
                ),
            }
        )

    if analyze.get("selected_capture_fragment"):
        warnings.append(
            {
                "code": "bypass_only_capture_fragments",
                "severity": "warning",
                "message": (
                    "no main profiler trace found; analysis ran on a sglang CUDA-graph "
                    "capture shard (device-kernel sparse). Ensure the main "
                    "*-TP-*.trace.json.gz (not just capture_traces/bs_*) was captured."
                ),
            }
        )


#: Opt-in env gate for the variant-discriminating TraceShapeManifest (P0-A/WP-1).
_SHAPE_MANIFEST_ENV = "HYPERLOOM_TRACE_SHAPE_MANIFEST"
#: Optional gfx-arch provenance override (WP-1 stub; superseded by WP-0/WP-7).
_GFX_ENV = "HYPERLOOM_GFX_ARCH"
#: sglang capture shard filename -> ``bs_<batch>`` variant. vLLM instead emits
#: ``graph_capture_rank_*`` files whose batch/mode live in execution_details.json.
#: Searched rather than matched from the start: an SGLang without the profiler
#: patch prefixes the runner name, as in ``DecodeCudaGraphRunner_bs_104_rank0``,
#: and an anchored read falls through to the bare stem -- which differs per rank,
#: so the label dedup below stops collapsing the ranks of one batch.
_VARIANT_RE = re.compile(r"(bs_\d+)", re.IGNORECASE)
#: Which files carry an indexable per-variant shape. Deliberately NOT the shared
#: ``_capture_shapes`` classifier: that one answers "is this a sidecar rather
#: than a workload trace", which is a wider question than this one. A
#: whole-capture-per-rank file such as an unpatched SGLang's
#: ``cuda_graph_capture-<runner>-TP-<n>`` is certainly a sidecar, but it holds
#: every batch size at once and carries no variant identity, so admitting it
#: mints one bogus variant per rank -- TP=8 would inflate ``variant_count`` by
#: eight shape-identical entries and defeat the dedup a few lines down. Admit a
#: ``bs_<batch>`` token anywhere (both SGLang layouts) or the vLLM prefix whose
#: batch/mode arrive via execution_details.json.
_CAPTURE_FILE_RE = re.compile(r"bs_\d+|\Agraph_capture", re.IGNORECASE)
#: Optional cap on how many capture files to index (0 = all). Logged when hit.
_MAX_CAPTURES_ENV = "HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES"


def _sha256_file(path: str | Path) -> str:
    """Return the sha256 hex of a file, or ``""`` on any I/O error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _load_execution_details(capdir: Path) -> dict[str, dict[str, Any]]:
    """Map ``capture filename -> {batch_size, mode}`` from vLLM's
    ``execution_details.json`` (a list of ``{file, batch_size, mode}``).

    Returns an empty map when absent/unreadable (sglang shards or older
    captures), so callers fall back to filename-derived labels.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        data = json.loads((capdir / "execution_details.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for e in data if isinstance(data, list) else []:
        if isinstance(e, dict) and e.get("file"):
            out[str(e["file"])] = {"batch_size": e.get("batch_size"), "mode": e.get("mode")}
    return out


def _discover_capture_shards(trace_input: str, capture_folder: str) -> list[tuple[Path, str, str | None]]:
    """Return ``(file, variant_label, mode)`` for each CUDA-graph capture shard.

    Handles both capture layouts:
      * sglang ``bs_<batch>_rank<n>`` shards -> variant from the filename;
      * vLLM ``graph_capture_rank_*`` files -> variant (``bs_<batch>``) and mode
        from the sibling ``execution_details.json`` batch mapping.

    Looks in ``capture_folder`` when given, else under the trace-input tree.
    """
    roots: list[Path] = []
    if capture_folder:
        roots.append(Path(capture_folder))
    ti = Path(trace_input)
    roots.append(ti if ti.is_dir() else ti.parent)
    seen: set[str] = set()
    seen_labels: set[str] = set()
    exec_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    out: list[tuple[Path, str, str | None]] = []
    for root in roots:
        if not root.exists():
            continue
        for cand in _reader._trace_candidates(root):
            name = cand.name
            if not _CAPTURE_FILE_RE.search(name):
                continue
            key = str(cand.resolve())
            if key in seen:
                continue
            seen.add(key)
            pdir = cand.parent
            if pdir not in exec_cache:
                exec_cache[pdir] = _load_execution_details(pdir)
            # vLLM only: its shards are named ``graph_capture_*`` and keep
            # batch/mode in execution_details.json. An SGLang name that merely
            # *contains* the token writes no such file, so routing it here would
            # yield bs=None and fall back to the per-rank stem.
            if name.lower().startswith("graph_capture"):
                meta = exec_cache[pdir].get(name, {})
                bs = meta.get("batch_size")
                mode = meta.get("mode")
                # vLLM captures each batch in >1 graph mode (PIECEWISE + FULL);
                # mode is part of the variant identity or they collide.
                if bs not in (None, ""):
                    label = f"bs_{bs}_{str(mode).lower()}" if mode else f"bs_{bs}"
                else:
                    label = cand.stem
            else:  # sglang bs_<batch>_rank<n>, with or without a runner prefix
                m = _VARIANT_RE.search(name)
                label = m.group(1).lower() if m else cand.stem
                mode = None
            # TP>1 emits one capture shard per rank with the SAME variant label
            # (bs_<batch>[_mode]); the ranks carry identical shapes, so keep only
            # the first (representative rank). Otherwise duplicate labels
            # overwrite each other's hash/meta downstream and inflate
            # variant_count (risking the multi-variant unresolved path).
            if label in seen_labels:
                continue
            seen_labels.add(label)
            out.append((cand, label, mode))
    return out


def _build_manifest_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Provenance block for the TraceShapeManifest.

    Delegates to the shared ``hyperloom.common.provenance.build_provenance``
    (WP-0) so the trace manifest and the session manifest never drift. Falls
    back to a minimal env-derived stub only when the shared module is not
    importable (standalone tool invocation without the package installed); the
    ``_provenance_source`` tag distinguishes the two.
    """
    if _shared_build_provenance is not None:
        try:
            return _shared_build_provenance(args, env=os.environ, probe=True)
        except Exception:  # noqa: BLE001 — provenance must never break the manifest.
            pass

    def _env(*names: str) -> Any:
        for n in names:
            v = os.environ.get(n)
            if v:
                return v
        return None

    return {
        "_provenance_source": "wp1_stub",
        "model_name": args.model_name or None,
        "model_path": getattr(args, "model_path", "") or None,
        "framework": args.framework or None,
        "target_platform": args.target_platform or None,
        "gfx_arch": _env(_GFX_ENV),
        "dtype": args.precision or _env("PRECISION"),
        "tp": _env("TP"),
        "ep": _env("EP"),
        "concurrency": _env("CONC", "CONCURRENCY"),
        "isl": _env("ISL"),
        "osl": _env("OSL"),
        "graph_mode": _env("HYPERLOOM_GRAPH_MODE"),
    }


def _maybe_build_shape_manifest(
    args: argparse.Namespace,
    analyze: dict[str, Any],
    bypass_dir: Path,
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Optionally build + write the variant-discriminating TraceShapeManifest.

    Opt-in via ``HYPERLOOM_TRACE_SHAPE_MANIFEST`` (off by default -> returns
    ``{"status": "disabled"}`` and writes nothing, so a run without the flag is
    byte-for-byte unchanged). When enabled, capture shards are indexed per
    ``bs_<batch>`` variant; with no capture shards it falls back to an eager
    manifest built from the main analysis. Never raises -- any failure degrades
    to ``{"status": "error: ..."}`` and is logged to stderr.
    """
    flag = os.environ.get(_SHAPE_MANIFEST_ENV, "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return {"status": "disabled"}
    try:
        main_trace = analyze.get("trace_file", "") or ""
        main_hash = _sha256_file(main_trace) if main_trace else ""
        shards = _discover_capture_shards(args.trace_input, args.capture_folder or "")
        try:
            max_caps = int(os.environ.get(_MAX_CAPTURES_ENV, "0") or 0)
        except ValueError:
            max_caps = 0
        if max_caps > 0 and len(shards) > max_caps:
            print(
                f"[trace_shape_manifest] capping capture files {len(shards)}->{max_caps} "
                f"(set {_MAX_CAPTURES_ENV}=0 to index all)",
                file=sys.stderr,
            )
            shards = shards[:max_caps]
        capture_variants: list[tuple[str, dict[str, Any]]] = []
        capture_hashes: dict[str, str] = {}
        variant_meta: dict[str, dict[str, Any]] = {}
        for path, label, mode in shards:
            shard_an = _reader.analyze_trace(path, top_k=0, steady_state=False, emit_launches=True)
            if shard_an.get("status") != "ok":
                continue
            capture_variants.append((label, shard_an))
            capture_hashes[label] = _sha256_file(path)
            variant_meta[label] = {
                "batch_size": label.split("_")[1] if label.startswith("bs_") else None,
                "mode": mode,
                "file": path.name,
            }

        # phase hint from the analysis mode the coordinator forwards.
        phase_hint = (args.analysis_mode or "mixed").lower() or "mixed"
        manifest = _tsm.build_shape_manifest(
            main_analysis=analyze,
            capture_variants=capture_variants,
            provenance=_build_manifest_provenance(args),
            main_trace_hash=main_hash,
            capture_trace_hashes=capture_hashes,
            variant_meta=variant_meta,
            analysis_route="bypass",
            generated_at=generated_at,
            phase_hint=phase_hint,
        )
        out_path = bypass_dir / "trace_shape_manifest.json"
        atomic_write_json(out_path, manifest, ensure_ascii=False, sort_keys=False, trailing_newline=False)
        print(
            f"[trace_shape_manifest] variants={len(capture_variants) or 'eager'} "
            f"rows={len(manifest.get('rows', []))} path={out_path}",
            file=sys.stderr,
        )
        return {
            "status": "ok",
            "path": str(out_path),
            "variant_count": len(capture_variants),
            "row_count": len(manifest.get("rows", [])),
            "warnings": manifest.get("warnings", []),
        }
    except Exception as exc:  # noqa: BLE001 — manifest must never break bypass
        msg = f"error: {type(exc).__name__}: {exc}"
        print(f"[trace_shape_manifest] skipped: {msg}", file=sys.stderr)
        return {"status": msg}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser mirroring the flags the handler forwards."""
    p = argparse.ArgumentParser(description="Hyperloom bypass trace analysis (TraceLens-free)")
    p.add_argument("--trace-input", required=True)
    p.add_argument("--session-id", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--workspace-path", default=os.environ.get("USER_DATA_PATH", "/workspace/hyperloom"))
    p.add_argument("--model-name", default="")
    p.add_argument("--framework", default="")
    p.add_argument("--target-platform", default="")
    p.add_argument("--analysis-mode", default="")
    p.add_argument("--split-conc", default="")
    p.add_argument("--split-osl", default="")
    p.add_argument("--split-r", default="")
    p.add_argument("--capture-folder", default="")
    p.add_argument(
        "--steady-state-mode",
        default="",
        help="Steady-state windowing mode. Any non-off value enables windowing, "
        "including the TraceLens splitter chunk types the coordinator forwards "
        "(mixed / decode_only / prefilldecode); off values: '', 0, false, off, none.",
    )
    p.add_argument("--roofline-output-name", default="kernel_roofline.json")
    # Denoise-step count for scriptable/diffusion workloads; 0 = infer. The env
    # fallback matches the TraceLens CLI, so the divisor cannot differ by route.
    p.add_argument(
        "--num-denoise-steps",
        type=int,
        default=int(os.environ.get("HYPERLOOM_NUM_DENOISE_STEPS", "0") or 0),
    )
    # Diffusion analytic-ceiling inputs shared with the TraceLens CLI surface;
    # parsed but unused on this route.
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    p.add_argument("--precision", default="")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--cfg-batch", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


#: ``--steady-state-mode`` values that mean "do NOT window" (analyze the full trace).
_STEADY_OFF_VALUES = frozenset({"", "0", "false", "off", "no", "none"})


def _should_enable_steady(*, steady_state_mode: str, framework: str, env_steady: bool) -> bool:
    """Whether to run steady-state windowing for this trace analysis.

    On for: the env opt-in; xDiT (homogeneous denoise steps -> one representative
    step); OR any non-off ``--steady-state-mode``. The last clause covers the
    TraceLens splitter chunk types the coordinator forwards (``mixed`` /
    ``decode_only`` / ``prefilldecode``) plus the legacy opt-in aliases, so a
    text-gen bypass run windows the requested steady chunk instead of silently
    aggregating the full trace (parity with the TraceLens route). bypass's
    repeat-based windowing then anchors a representative step, or degrades to
    full-trace (with the existing warning) when no window is found.
    """
    mode = (steady_state_mode or "").strip().lower()
    return bool(env_steady) or (framework or "").lower() == "xdit" or mode not in _STEADY_OFF_VALUES


#: Mirrors of the ``framework_registry``, used only when that package is not
#: importable (standalone invocation). Keep in sync when a framework is added;
#: tests assert both against the registry so a divergence cannot land.
_STANDALONE_UNITS = {"xdit": "img/s", "custom": "unit/s"}
_STANDALONE_SCRIPTABLE = frozenset({"xdit", "custom"})


def _is_scriptable_framework(framework: str | None) -> bool:
    """Return whether ``framework`` is a server-less scriptable workload.

    Args:
        framework: Framework name (matched case-insensitively).

    Returns:
        bool: ``True`` for ``kind=scriptable`` frameworks.
    """
    try:
        from hyperloom.inference_optimizer.framework_registry import is_scriptable

        return is_scriptable(framework)
    except ImportError:  # standalone invocation without the package installed.
        return str(framework or "").strip().lower() in _STANDALONE_SCRIPTABLE


def _throughput_unit(framework: str | None) -> str:
    """Return the throughput unit ``framework`` reports, per the registry.

    The registry is the single source of truth here, so an operator's ``custom``
    workload reports its own neutral ``unit/s`` instead of being mislabelled
    ``tok/s``. Falls back to ``_STANDALONE_UNITS`` for standalone invocation,
    where the ``hyperloom`` package may not be importable (see the provenance
    import above).

    Args:
        framework: Framework name (matched case-insensitively).

    Returns:
        str: The unit string, e.g. ``"tok/s"``, ``"img/s"`` or ``"unit/s"``.
    """
    try:
        from hyperloom.inference_optimizer.framework_registry import throughput_unit

        return throughput_unit(framework)
    except ImportError:  # standalone invocation without the package installed.
        return _STANDALONE_UNITS.get(str(framework or "").strip().lower(), "tok/s")


def main(argv: list[str] | None = None) -> int:
    """Entry point: emit the minimal bypass artifact set and a result JSON.

    Returns:
        Process exit code (``0`` on success). The structured result is printed
        to stdout as a single JSON object regardless of exit code.
    """
    args = _build_arg_parser().parse_args(argv)

    workspace = Path(args.workspace_path)
    session_id = args.session_id or workspace.name
    run_dir = workspace / "kernel-agent" / "runs" / session_id / f"{_run_stamp()}_bypass"
    bypass_dir = run_dir / "bypass"
    reports_dir = workspace / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    bypass_dir.mkdir(parents=True, exist_ok=True)

    trace_health_warnings: list[dict[str, Any]] = []
    top_k = args.top_k if args.top_k and args.top_k > 0 else 15

    # Mirrors the ``run_meta`` block tracelens_analysis.py returns so the caller's
    # SBD V6 roofline event carries one shape for both tools. The ladder is
    # genuinely shorter here: this reader picks its steady-state window in memory,
    # so there is no TraceLens install, no on-disk split, and no chunk selection.
    # Tool-specific analysis output goes under ``route_ext`` instead of widening
    # the shared envelope.
    run_meta: dict[str, Any] = {
        "preflight": {},
        "split": {},
        "selection": {},
        "steps": [],
        "route_ext": {},
    }

    def _note_step(
        step_id: str,
        *,
        category: str,
        status: str,
        skip_reason: str | None = None,
        **detail: Any,
    ) -> None:
        """Append one step to the run's ladder.

        Args:
            step_id: Stable step name.
            category: Ladder phase (``preflight`` / ``analyze`` / ``emit``).
            status: ``ok`` / ``failed`` / ``skipped``.
            skip_reason: Why the step did not run, when skipped.
            **detail: Measured values for the step.
        """
        run_meta["steps"].append(
            {
                "step_id": step_id,
                "order": len(run_meta["steps"]) + 1,
                "category": category,
                "status": status,
                "skip_reason": skip_reason,
                "detail": detail,
            }
        )

    framework_l = (args.framework or "").lower()
    # Steady-state windowing: opt-in via --steady-state-mode / env, always on
    # for xDiT. Falls back to full-trace shares when no repeating window found.
    env_steady = os.environ.get("HYPERLOOM_BYPASS_STEADY_STATE", "").strip().lower() in {"1", "true", "yes", "on"}
    enable_steady = _should_enable_steady(
        steady_state_mode=args.steady_state_mode or "",
        framework=args.framework or "",
        env_steady=env_steady,
    )

    # --- analyze the trace (independent streaming reader) ---
    analyze: dict[str, Any]
    if args.dry_run:
        analyze = {
            "status": "ok",
            "timeline": {},
            "attribution": {},
            "kernels": [],
            "ops": [],
            "aggregation_scope": "full_trace",
        }
    else:
        try:
            # top_k=0 -> keep all device-kernel aggregates; candidate slicing uses top_k.
            analyze = _reader.analyze_trace(
                args.trace_input,
                top_k=0,
                steady_state=enable_steady,
                framework=args.framework,
                emit_launches=True,
            )
        except Exception as exc:  # noqa: BLE001 — never abort the pipeline
            analyze = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    run_meta["preflight"].update(
        {
            "trace_input": str(args.trace_input),
            "steady_state_requested": bool(enable_steady),
            "steady_state_mode": str(args.steady_state_mode or ""),
        }
    )
    _note_step(
        "discover_inputs",
        category="preflight",
        status="ok",
        trace_input=str(args.trace_input),
    )
    for _skipped_step, _skip_reason in (
        ("install_tracelens", "the TraceLens-free reader needs no TraceLens checkout"),
        ("split_trace", "the reader windows the trace in memory and writes no split chunks"),
        ("select_chunk", "no split chunks exist to select from"),
    ):
        _note_step(_skipped_step, category="split", status="skipped", skip_reason=_skip_reason)
    _note_step(
        "run_analysis",
        category="analyze",
        status="ok" if analyze.get("status") == "ok" else "failed",
        reader="bypass_trace_reader",
        steady_state_requested=bool(enable_steady),
        error=str(analyze.get("error") or ""),
    )
    # ``analysis_degraded`` distinguishes an analysis failure (bad/unparsable
    # trace) from a genuine empty result.
    analysis_degraded = False
    if analyze.get("status") != "ok":
        analysis_degraded = True
        trace_health_warnings.append(
            {
                "code": "bypass_trace_parse_failed",
                "severity": "warning",
                "message": f"bypass reader could not analyze trace: {analyze.get('error', 'unknown')}",
            }
        )
        analyze = {
            "status": "ok",
            "timeline": {},
            "attribution": {},
            "kernels": [],
            "ops": [],
            "aggregation_scope": "full_trace",
        }
    else:
        stream_errors = [str(error) for error in (analyze.get("stream_errors") or [])]
        if stream_errors:
            analysis_degraded = True
            trace_health_warnings.append(
                {
                    "code": "bypass_trace_stream_incomplete",
                    "severity": "warning",
                    "message": ("bypass reader recovered an incomplete trace stream: " + "; ".join(stream_errors[:3])),
                }
            )
        if not analyze.get("kernels"):
            trace_health_warnings.append(
                {
                    "code": "bypass_no_gpu_kernels",
                    "severity": "warning",
                    "message": "bypass reader found no GPU kernel events in the trace",
                }
            )

    # Aggregation scope is driven by the reader (steady_state vs full_trace).
    scope = analyze.get("aggregation_scope", AGGREGATION_SCOPE_FULL)
    steady_window = analyze.get("steady_window")

    # ``estimated`` marks shares not anchored to a real per-step window (steady
    # windowing requested but fell back to the full trace).
    estimated = enable_steady and scope != AGGREGATION_SCOPE_STEADY
    if framework_l == "xdit" and estimated:
        trace_health_warnings.append(
            {
                "code": "bypass_xdit_estimated",
                "severity": "info",
                "message": (
                    "xDiT analysis fell back to full-trace shares (no per-step denoising "
                    "window found; trace lacks step annotations such as ProfilerStep) — "
                    "treat kernel shares / roofline as estimated."
                ),
            }
        )
    elif framework_l == "xdit":
        trace_health_warnings.append(
            {
                "code": "bypass_xdit_steady_anchored",
                "severity": "info",
                "message": (
                    f"xDiT analysis anchored to a real per-step denoising window "
                    f"({(steady_window or {}).get('step_name', 'step')}×"
                    f"{(steady_window or {}).get('step_count', 0)}); per-step kernel "
                    "shares are trace-anchored (not estimated)."
                ),
            }
        )

    # Multi-rank provenance: xDiT TP>1 produces one trace per rank; the reader
    # analyzes one representative rank.
    analyzed_rank = analyze.get("analyzed_rank")
    rank_count = analyze.get("rank_count", 1)
    if isinstance(rank_count, int) and rank_count > 1:
        trace_health_warnings.append(
            {
                "code": "bypass_multi_rank_single_analyzed",
                "severity": "info",
                "message": (
                    f"trace dir has {rank_count} per-rank traces; analyzed rank {analyzed_rank} "
                    "as representative (each rank runs the same kernels on sharded data under "
                    "sequence/tensor parallel)."
                ),
            }
        )

    # Corrupt durations outrank real kernels, since ranking is by summed time.
    stream_overlap = (analyze.get("timeline") or {}).get("stream_overlap") or {}
    if stream_overlap:
        _severity = str(stream_overlap.get("severity") or "info")
        _share_pct = float(stream_overlap.get("excess_share") or 0.0) * 100.0
        _advice = (
            "Kernel ranking is by summed duration, so treat the hot-kernel table as "
            "unreliable until the trace is recaptured or the durations are repaired."
            if _severity == "warning"
            else "Kernel ranking is by summed duration, so the affected entries are "
            "overstated by this much; the overall ordering is probably still usable."
        )
        trace_health_warnings.append(
            {
                "code": "bypass_impossible_kernel_durations",
                "severity": _severity,
                "message": (
                    f"device stream (pid={stream_overlap.get('pid')}, tid={stream_overlap.get('tid')}): "
                    f"{stream_overlap.get('overlapping_events')} event(s) run past the start of the "
                    f"event after them, by {stream_overlap.get('excess_ms')} ms in total -- "
                    f"{_share_pct:.1f}% of the {stream_overlap.get('device_ms')} ms of device time "
                    "booked on that stream. That is impossible on a serial stream, so these "
                    f"durations are corrupt. Worst offender: '{stream_overlap.get('worst_event')}' "
                    f"overruns by {stream_overlap.get('worst_event_excess_ms')} ms. " + _advice
                ),
            }
        )

    # Steps present in the analyzed data. ``annotation_window_count`` is
    # deliberately not a fallback: it counts user_annotation ranges, not steps.
    requested_denoise_steps = int(getattr(args, "num_denoise_steps", 0) or 0)
    inferred_denoise_steps = int((steady_window or {}).get("step_count", 0) or 0)
    if inferred_denoise_steps <= 0:
        # Count in the file the reader resolved; a directory input may glob a
        # different one.
        inferred_denoise_steps = count_profiler_steps(str(analyze.get("trace_file") or args.trace_input))
    if requested_denoise_steps > 0 and inferred_denoise_steps > 0 and requested_denoise_steps != inferred_denoise_steps:
        trace_health_warnings.append(
            {
                "code": "bypass_denoise_steps_mismatch",
                "severity": "info",
                "message": (
                    f"requested --num-denoise-steps={requested_denoise_steps} differs from the "
                    f"{inferred_denoise_steps} step(s) found in the analyzed data; the requested "
                    "count wins as the per-step divisor."
                ),
            }
        )

    # Analysis-quality health signals (observability only; never fatal).
    if analyze.get("kernels"):
        _emit_quality_warnings(analyze, trace_health_warnings)

    # --- build downstream artifacts from classified device kernels ---
    candidates = _report.build_candidates(
        analyze,
        framework=args.framework,
        target_platform=args.target_platform,
        top_k=top_k,
        discover_benchmarks=False,
    )

    analysis_md_path = bypass_dir / "analysis.md"
    candidates_path = run_dir / "kernel_candidates.json"
    summary_path = bypass_dir / "summary.json"
    manifest_path = run_dir / "trace_input_manifest.json"
    roofline_name = args.roofline_output_name or "kernel_roofline.json"
    kernel_roofline_path = reports_dir / roofline_name
    kernel_metrics_csv_path = bypass_dir / "kernel_metrics.csv"
    kernel_summary_csv_path = bypass_dir / "kernel_summary.csv"

    # High-idle gate: when GPU idle exceeds the threshold, per-kernel rewriting
    # cannot move end-to-end latency, so suppress every candidate list and
    # surface a high_gpu_idle_pct warning for the Coordinator.
    idle_pct_value = (analyze.get("timeline") or {}).get("idle_pct")
    idle_pct_threshold = resolve_idle_pct_threshold()
    # Graph under-recording makes idle% unreliable: skip the idle gate (keep
    # candidates ranked by recorded-kernel GPU share) and surface a health warning.
    graph_coverage = analyze.get("graph_coverage") or {}
    graph_under_recorded = bool(graph_coverage.get("graph_under_recorded"))
    if graph_under_recorded:
        trace_health_warnings.append(
            build_graph_under_recorded_warning(
                graph_launch_count=int(graph_coverage.get("graph_launch_count", 0) or 0),
                idle_pct=float(idle_pct_value) if isinstance(idle_pct_value, (int, float)) else None,
            )
        )
    elif isinstance(idle_pct_value, (int, float)) and float(idle_pct_value) > idle_pct_threshold:
        for _cand_key in ("hot_kernels", "routable_kernels", "skipped_kernels", "task_groups"):
            candidates[_cand_key] = []
        trace_health_warnings.append(
            build_high_idle_warning(
                idle_pct=float(idle_pct_value),
                threshold_pct=idle_pct_threshold,
                report_path=analysis_md_path,
            )
        )

    # Stamp the report path onto each candidate (downstream reads it).
    for cand in candidates.get("hot_kernels", []):
        cand["trace_report_path"] = str(analysis_md_path)

    throughput_unit = _throughput_unit(args.framework)
    write_text(
        analysis_md_path,
        _report.render_analysis_md(
            candidates,
            analyze,
            model_name=args.model_name,
            framework=args.framework,
            target_platform=args.target_platform,
            throughput_unit=throughput_unit,
            metrics_csv_path=str(kernel_metrics_csv_path),
            summary_csv_path=str(kernel_summary_csv_path),
        ),
    )
    atomic_write_json(candidates_path, candidates, ensure_ascii=False, sort_keys=False, trailing_newline=False)
    # CSV exports: full per-kernel metrics + category summary.
    write_text(kernel_metrics_csv_path, _report.build_metrics_csv(candidates))
    write_text(kernel_summary_csv_path, _report.build_category_summary_csv(candidates))

    summary = _report.build_summary(
        candidates,
        framework=args.framework,
        target_platform=args.target_platform,
        generated_at=utc_now(timespec="seconds"),
        trace_health_warnings=trace_health_warnings,
    )
    summary["estimated"] = estimated
    summary["analysis_degraded"] = analysis_degraded
    # Always present (may be null) so the summary/manifest/result schemas match.
    summary["steady_window"] = steady_window  # always present (may be null)

    atomic_write_json(
        manifest_path,
        {
            "source": "bypass",
            "trace_input": str(args.trace_input),
            "trace_file": analyze.get("trace_file", ""),
            "capture_folder": args.capture_folder or None,
            "aggregation_scope": scope,
            "steady_window": steady_window,
            "estimated": estimated,
            "analysis_degraded": analysis_degraded,
            "analyzed_rank": analyzed_rank,
            "rank_count": rank_count,
            "event_total": analyze.get("event_total", 0),
            "created_at": utc_now(timespec="seconds"),
        },
        ensure_ascii=False,
        sort_keys=False,
        trailing_newline=False,
    )

    kernel_roofline = _report.build_kernel_roofline(
        candidates,
        analysis_md_path=str(analysis_md_path),
        kernel_candidates_path=str(candidates_path),
    )
    atomic_write_json(
        kernel_roofline_path, kernel_roofline, ensure_ascii=False, sort_keys=False, trailing_newline=False
    )

    # Optional rocprof-compute enrichment (opt-in; enriches the sidecar in
    # place). Skipped in --dry-run.
    rocprof_enrich: dict[str, Any] = (
        {"status": "disabled"}
        if args.dry_run
        else _maybe_enrich_rocprof(kernel_roofline_path, candidates_path, run_dir)
    )
    summary["rocprof_enrich"] = rocprof_enrich
    atomic_write_json(summary_path, summary, ensure_ascii=False, sort_keys=False, trailing_newline=False)

    # Kernel-fusion opportunities: launch adjacency -> fusable clusters.
    fusion = _report.build_fusion(analyze)

    # Diffusion / scriptable workload-level roofline: aggregate the per-kernel
    # analytical roofline into an end-to-end workload roofline + per-denoise-step
    # split. Best-effort sidecar over all device kernels, independent of the
    # per-kernel high-idle gate, so still emitted in the high-idle regime.
    # Gated on scriptable, not on a framework name: this sidecar is built purely
    # from the trace, so it needs no denoiser config the operator may not supply.
    diffusion_roofline_path: str | None = None
    if _is_scriptable_framework(args.framework):
        try:
            from diffusion_roofline import build_report_from_bypass  # noqa: E402

            _diff_steps = resolve_perstep_divisor(
                requested_steps=requested_denoise_steps,
                inferred_steps=inferred_denoise_steps,
            )
            # Workload totals cover all analyzed device kernels (not just top-k).
            _workload_totals = _report.build_workload_roofline_totals(analyze, target_platform=args.target_platform)
            _all_kernels = [k for k in (analyze.get("kernels") or []) if float(k.get("gpu_time_us") or 0.0) > 0]
            _diff_report = build_report_from_bypass(
                candidates.get("hot_kernels", []),
                analyze.get("timeline") or {},
                _diff_steps,
                top_k,
                totals=_workload_totals,
                kernels_aggregated=len(_all_kernels),
            )
            _diff_path = run_dir / "diffusion_roofline.json"
            atomic_write_json(_diff_path, _diff_report, ensure_ascii=False, sort_keys=False, trailing_newline=False)
            diffusion_roofline_path = str(_diff_path)
        except Exception:  # noqa: BLE001 - best-effort sidecar, never blocks the run
            diffusion_roofline_path = None

    # Optional variant-discriminating TraceShapeManifest (P0-A / WP-1; opt-in via
    # HYPERLOOM_TRACE_SHAPE_MANIFEST). Off by default -> disabled, writes nothing.
    shape_manifest = _maybe_build_shape_manifest(args, analyze, bypass_dir, generated_at=utc_now(timespec="seconds"))

    hot_kernels = candidates.get("hot_kernels", [])
    result: dict[str, Any] = {
        "status": "ok",
        "route": "bypass",
        "aggregation_scope": scope,
        "steady_window": steady_window,
        "estimated": estimated,
        "analysis_degraded": analysis_degraded,
        "analyzed_rank": analyzed_rank,
        "rank_count": rank_count,
        "num_denoise_steps": requested_denoise_steps or inferred_denoise_steps,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "hot_kernels": hot_kernels,
        "hot_kernels_top15": hot_kernels[:15],
        "routable_kernels": candidates.get("routable_kernels", []),
        "skipped_kernels": candidates.get("skipped_kernels", []),
        "task_groups": candidates.get("task_groups", []),
        "candidates_path": str(candidates_path),
        "trace_report_path": str(analysis_md_path),
        "kernel_roofline_path": str(kernel_roofline_path),
        "tracelens_summary_path": str(summary_path),
        "kernel_metrics_csv_path": str(kernel_metrics_csv_path),
        "kernel_summary_csv_path": str(kernel_summary_csv_path),
        "fusion": {
            "launch_count": fusion.get("launch_count", 0),
            "fusable_cluster_count": fusion.get("fusable_cluster_count", 0),
            "fusable_time_us": fusion.get("fusable_time_us", 0.0),
        },
        "orchestrator_mode": "bypass",
        "timeline": analyze.get("timeline") or {},
        "attribution": analyze.get("attribution") or {},
        "graph_coverage": analyze.get("graph_coverage") or {},
        "trace_health_warnings": trace_health_warnings,
        "artifact_paths": {
            "trace_report_path": str(analysis_md_path),
            "kernel_candidates": str(candidates_path),
            "kernel_roofline": str(kernel_roofline_path),
            "tracelens_summary": str(summary_path),
            "kernel_metrics_csv": str(kernel_metrics_csv_path),
            "kernel_summary_csv": str(kernel_summary_csv_path),
            "trace_input_manifest": str(manifest_path),
        },
    }
    run_meta["selection"].update(
        {
            "requested_mode": str(args.steady_state_mode or ""),
            "steady_window": steady_window,
            "aggregation_scope": scope,
            "fell_back_to_full_trace": bool(estimated),
        }
    )
    # Analysis output that only this reader produces. Kept out of the shared
    # envelope so a consumer reading ``analysis`` across routes never has to
    # branch on which tool ran to find a field it needs.
    run_meta["route_ext"] = {
        "target_platform": str(args.target_platform or ""),
        "analyzed_rank": analyzed_rank,
        "rank_count": rank_count,
        "num_denoise_steps": requested_denoise_steps or inferred_denoise_steps,
        "estimated": bool(estimated),
        "analysis_degraded": bool(analysis_degraded),
        "graph_coverage": analyze.get("graph_coverage") or {},
        "timeline": analyze.get("timeline") or {},
        "attribution": analyze.get("attribution") or {},
        "fusion": {
            "launch_count": fusion.get("launch_count", 0),
            "fusable_cluster_count": fusion.get("fusable_cluster_count", 0),
            "fusable_time_us": fusion.get("fusable_time_us", 0.0),
        },
        # Counts rather than the lists: the lists are already in the candidates
        # artifact, and it is the partition that says whether dispatch has
        # anything to work with.
        "routable_kernel_count": len(candidates.get("routable_kernels", []) or []),
        "skipped_kernel_count": len(candidates.get("skipped_kernels", []) or []),
        "task_group_count": len(candidates.get("task_groups", []) or []),
        "trace_shape_manifest_status": str(shape_manifest.get("status") or ""),
        "diffusion_roofline_path": str(diffusion_roofline_path or ""),
    }
    _note_step(
        "extract_hot_kernels",
        category="emit",
        status="ok",
        hot_kernel_count=len(hot_kernels),
        routable_kernel_count=run_meta["route_ext"]["routable_kernel_count"],
    )
    _note_step("write_reports", category="emit", status="ok", report_path=str(analysis_md_path))
    result["run_meta"] = run_meta
    # Surfaced only when the opt-in manifest was produced (P0-A / WP-1).
    result["trace_shape_manifest"] = shape_manifest
    if shape_manifest.get("status") == "ok" and shape_manifest.get("path"):
        result["artifact_paths"]["trace_shape_manifest"] = shape_manifest["path"]

    # Surfaced only when produced (xDiT/scriptable).
    if diffusion_roofline_path:
        result["diffusion_roofline_path"] = diffusion_roofline_path
        result["artifact_paths"]["diffusion_roofline"] = diffusion_roofline_path
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
