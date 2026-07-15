# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from ._common import (
    _rel,
    _to_float,
    _to_int,
)



def _geak_accepted_kernels_from_journey(
    result: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Derive the accepted (KEEP/integrated) kernels from ``kernel_journey.json``.

    Projects each kernel whose ``e2e`` sub-object was integrated (or decided
    ``KEEP``/``ADOPTED``) into a compact accepted-kernel descriptor. Used to
    back-fill an empty ``accepted_kernels`` in ``result.json``. Best-effort: a
    missing/partial file yields ``[]`` and never raises.

    Args:
        result (dict[str, Any]): The normalized ``result.json`` (carries
            ``kernel_journey_path`` / ``eval_dir`` used to locate the journey).
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        list[dict[str, Any]]: The accepted-kernel descriptors, or ``[]`` when the
            journey is absent, unreadable, or holds no integrated kernel.
    """
    kj_path = str(result.get("kernel_journey_path") or "")
    if not kj_path:
        eval_dir = str(result.get("eval_dir") or "")
        if eval_dir:
            kj_path = str(Path(eval_dir) / "kernel_journey.json")
    if not kj_path:
        return []
    try:
        if not Path(kj_path).is_file():
            return []
        journey = json.loads(Path(kj_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"geak: kernel_journey read failed for backfill: {exc}")
        return []
    if not isinstance(journey, dict):
        return []

    accepted: list[dict[str, Any]] = []
    for k in journey.get("kernels") or []:
        if not isinstance(k, dict):
            continue
        e2e = k.get("e2e")
        if not isinstance(e2e, dict):
            continue
        decision = str(e2e.get("decision") or "").strip().upper()
        integrated = e2e.get("integrated") is True
        if not (integrated or decision in ("KEEP", "ADOPTED")):
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        br = k.get("backend_result") if isinstance(k.get("backend_result"), dict) else {}
        verification = br.get("verification") if isinstance(br.get("verification"), dict) else {}
        dispatch = k.get("dispatch") if isinstance(k.get("dispatch"), dict) else {}
        backend = str(
            verification.get("best_backend")
            or (dispatch.get("backends") or [None])[0]
            or ""
        )
        accepted.append(
            {
                "kernel_id": kid,
                "name": str(k.get("name") or kid),
                "gpu_pct": _to_float(k.get("gpu_pct")),
                "micro_speedup": _to_float(
                    k.get("micro_speedup") or verification.get("micro_speedup")
                ),
                "e2e_gain_pct": _to_float(e2e.get("e2e_gain_pct")),
                "validated": e2e.get("validated") if isinstance(e2e.get("validated"), bool) else None,
                "decision": decision or "KEEP",
                "backend": backend,
                "target_file": e2e.get("target_file"),
                "extra_server_args": str(e2e.get("extra_server_args") or ""),
                "source": "kernel_journey_backfill",
            }
        )
    return accepted


def _geak_reconstruct_from_disk(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Best-effort reconstruction of a GEAK run from on-disk survivors.

    Scans the runner's working tree under ``<session>/geak/`` to recover what
    actually ran when ``state.geak_result`` is empty/missing: the handoff, the
    e2e ``exp_root`` and the stages it reached (baseline / kernels / opbench /
    strategy), any flushed-but-unpromoted ``result.json`` status, and the
    per-kernel ``kernel_journey`` accepted kernels.

    Returns ``None`` when nothing usable is on disk (caller keeps the legacy
    ``missing`` section). Never raises — failures append to ``warnings``.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any] | None: The recovered evidence, or ``None``.
    """
    pf = session_dir / "geak"
    try:
        if not pf.is_dir():
            return None
    except OSError:
        return None

    def _load_json(p: Path) -> dict[str, Any]:
        try:
            if p.is_file():
                obj = json.loads(p.read_text(encoding="utf-8"))
                return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError) as exc:
            warnings.append(
                f"geak: reconstruct read failed for {p.name}: {exc}"
            )
        return {}

    stages: list[str] = []
    recon: dict[str, Any] = {}

    # 1) handoff.json — proves HL built + handed the e2e contract off.
    handoff = _load_json(pf / "handoff.json")
    if handoff:
        stages.append("handoff")
        recon["handoff"] = {
            "model_path": handoff.get("model_path"),
            "framework": handoff.get("framework"),
            "gpu_type": handoff.get("gpu_type"),
            "tp": handoff.get("tp"),
            "workload": handoff.get("workload"),
            "accepted_flags": handoff.get("accepted_flags"),
            "raw_baseline_tput": _to_float(handoff.get("raw_baseline_tput")),
        }

    # 2) a flushed-but-unpromoted result.json (absent or non-ok status).
    flushed = _load_json(pf / "result.json")
    if flushed:
        stages.append("result_json")
        recon["flushed_result_status"] = flushed.get("status")

    # 3) the e2e exp_root (newest ``e2e_*`` dir) + the stages it reached.
    exp_root: Path | None = None
    try:
        e2e_dirs = sorted(
            (d for d in pf.iterdir() if d.is_dir() and d.name.startswith("e2e_")),
            key=lambda d: d.name,
        )
        if e2e_dirs:
            exp_root = e2e_dirs[-1]
    except OSError as exc:
        warnings.append(f"geak: reconstruct iterdir failed: {exc}")

    kernels_attempted: list[dict[str, Any]] = []
    if exp_root is not None:
        recon["exp_root"] = _rel(exp_root, session_dir)
        for name, label in (
            ("baseline", "baseline"),
            ("baseline_rerun", "baseline_rerun"),
            ("strategy.md", "strategy"),
            ("kernel_journey.json", "kernel_journey"),
        ):
            try:
                if (exp_root / name).exists():
                    stages.append(label)
            except OSError:
                continue
        kdir = exp_root / "kernels"
        try:
            if kdir.is_dir():
                stages.append("kernels")
                for d in sorted(kdir.iterdir(), key=lambda p: p.name):
                    if d.is_dir() and not d.name.startswith("_"):
                        kernels_attempted.append({"name": d.name})
                # ``_exp`` holds the per-team op-bench / recursive kernel work.
                if (kdir / "_exp").is_dir():
                    stages.append("opbench")
        except OSError as exc:
            warnings.append(f"geak: reconstruct kernels scan failed: {exc}")
    recon["kernels_attempted"] = kernels_attempted

    # 4) per-kernel accepted kernels from the journey (reuse the projection so
    #    the recovered section's shape matches the producer-populated one).
    if exp_root is not None:
        kj = exp_root / "kernel_journey.json"
        try:
            if kj.is_file():
                recon["accepted_kernels"] = (
                    _geak_accepted_kernels_from_journey(
                        {"kernel_journey_path": str(kj)}, warnings
                    )
                )
        except OSError as exc:
            warnings.append(f"geak: accepted kernels journey unreadable: {exc}")

    # 5) newest-artifact timestamp (how far the run got in wall-clock). Bounded
    #    to a handful of key paths to avoid a full rglob of the exp_root tree.
    candidates = [pf / "handoff.json", pf / "result.json"]
    if exp_root is not None:
        candidates += [
            exp_root,
            exp_root / "logs",
            exp_root / "kernels",
            exp_root / "strategy.md",
            exp_root / "kernel_journey.json",
        ]
    newest = 0.0
    for p in candidates:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    if newest > 0:
        recon["last_artifact_ts"] = datetime.fromtimestamp(
            newest, tz=timezone.utc
        ).isoformat()

    # 6) op-bench verdicts — each per-kernel ``opbench_result.json`` records
    #    whether the backend bake-off found a deployable winner
    #    (``winner_editable`` + ``isolated_speedup`` > 1). Bounded to the
    #    top-level per-task files (the deep ``_exp`` tree is skipped).
    opbench_results: list[dict[str, Any]] = []
    if exp_root is not None:
        try:
            task_dirs = [
                d for d in (exp_root / "kernels").iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ] if (exp_root / "kernels").is_dir() else []
            for task_dir in sorted(task_dirs, key=lambda p: p.name)[:12]:
                ob = _load_json(task_dir / "opbench_result.json")
                if ob:
                    opbench_results.append({
                        "task": ob.get("task") or task_dir.name,
                        "winner_backend": ob.get("winner_backend"),
                        "isolated_speedup": _to_float(ob.get("isolated_speedup")),
                        "winner_editable": bool(ob.get("winner_editable")),
                        "winner_kind": ob.get("winner_kind"),
                    })
        except OSError as exc:
            warnings.append(f"geak: reconstruct opbench scan failed: {exc}")
    if opbench_results:
        recon["opbench_results"] = opbench_results

    # 7) runner log tails — run_e2e stdout/stderr survivors under
    #    ``exp_root/logs/``, the recoverable proxy for how far / why the run
    #    got. Bounded to the newest handful, tail-only.
    log_tails: dict[str, str] = {}
    if exp_root is not None:
        logs_dir = exp_root / "logs"
        try:
            if logs_dir.is_dir():
                logs = sorted(
                    (p for p in logs_dir.iterdir() if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                )
                for p in logs[-4:]:
                    try:
                        log_tails[p.name] = p.read_text(
                            encoding="utf-8", errors="replace",
                        )[-1500:]
                    except OSError:
                        continue
        except OSError as exc:
            warnings.append(f"geak: reconstruct log-tail read failed: {exc}")
    if log_tails:
        recon["runner_log_tails"] = log_tails

    # 8) likely_cause — a conservative classification of WHY no result reached
    #    state, so a reader does not have to re-derive it from the raw survivors:
    #      * ``runner_reported_failure``  — a non-ok result.json was flushed.
    #      * ``ran_no_deployable_winner`` — op-bench ran but found no editable
    #        winner > 1.0x, so there was simply nothing to flush as a win.
    #      * ``killed_before_flush``      — stages were reached but neither a
    #        kernel_journey nor a result.json landed (the in-flight result died
    #        with the process — the incident pattern: SIGKILL / budget / hang).
    #      * ``indeterminate``            — not enough on-disk signal to classify.
    has_journey = "kernel_journey" in stages
    ran_opbench = "opbench" in stages or bool(opbench_results)
    any_deployable = any(
        (r.get("isolated_speedup") or 0.0) > 1.0 and r.get("winner_editable")
        for r in opbench_results
    )
    if flushed and flushed.get("status") and flushed.get("status") != "ok":
        likely_cause = "runner_reported_failure"
    elif ran_opbench and not any_deployable and not has_journey:
        likely_cause = "ran_no_deployable_winner"
    elif stages and not has_journey and "result_json" not in stages:
        likely_cause = "killed_before_flush"
    else:
        likely_cause = "indeterminate"
    recon["likely_cause"] = likely_cause

    recon["stages_reached"] = stages
    # Nothing meaningful recovered → let the caller emit ``missing``.
    if not (handoff or flushed or exp_root):
        return None
    return recon


def collect_geak(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the GEAK/GEAK e2e KERNEL-phase section.

    Maps ``state.geak_result`` (the normalized ``result.json`` plus runner
    metadata) into the session-breakdown data contract: what the optimizer did
    (per-kernel / per-head), the accepted config, the validated regimes, the
    gain attribution, and — on a miss — the normalized failure reason.

    Returns an empty ``{}`` when GEAK was never engaged.

    Args:
        session_dir (Path): Absolute session root (used to relativize paths).
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The GEAK section, or ``{}`` when not engaged.
    """
    optimizer = str(state.get("kernel_optimizer") or "").strip().lower()
    result = state.get("geak_result")
    # An empty ``geak_result`` dict must NOT count as engaged; engage only when
    # the optimizer flag selected geak or a non-empty result was recorded.
    has_result = isinstance(result, dict) and bool(result)
    engaged = optimizer == "geak" or has_result
    if not engaged:
        return {}
    if not has_result:
        # Engaged via the flag but no result recorded; reconstruct from the
        # on-disk ``geak/`` working tree before surfacing ``missing``.
        recon = _geak_reconstruct_from_disk(session_dir, warnings)
        if recon is None:
            return {
                "engaged": True,
                "status": "missing",
                "error_class": "no_result",
                "error": (
                    "kernel_optimizer=geak but no geak_result "
                    "recorded"
                ),
                "accepted_kernels": [],
                "accepted_heads": [],
            }
        recovered_kernels = recon.get("accepted_kernels") or []
        return {
            "engaged": True,
            "status": "no_result_recovered_from_disk",
            "error_class": "no_result",
            "error": (
                "kernel_optimizer=geak but no geak_result was "
                "committed to state; reconstructed the run from on-disk "
                "geak/ artifacts. The runner handed off and produced "
                "intermediate output, but the normalized result.json was never "
                "folded into state — typically an external kill (SIGKILL / OOM "
                "/ budget) before the tick-boundary state.save, then a resume "
                "past KERNEL."
            ),
            "recovered_from_disk": True,
            "handoff": recon.get("handoff"),
            "exp_root": recon.get("exp_root"),
            "stages_reached": recon.get("stages_reached") or [],
            "kernels_attempted": recon.get("kernels_attempted") or [],
            "opbench_results": recon.get("opbench_results") or [],
            "runner_log_tails": recon.get("runner_log_tails") or {},
            "likely_cause": recon.get("likely_cause"),
            "flushed_result_status": recon.get("flushed_result_status"),
            "last_artifact_ts": recon.get("last_artifact_ts"),
            "accepted_kernels": recovered_kernels,
            "accepted_kernels_source": (
                "kernel_journey_backfill" if recovered_kernels else None
            ),
            "accepted_heads": [],
            "kernels_optimized": len(recovered_kernels),
        }

    def _rel_if_under(p: Any) -> Any:
        """Relativize ``p`` against the session dir when it lives under it."""
        if not p:
            return p
        try:
            pp = Path(str(p))
            if pp.is_absolute() and str(pp).startswith(str(session_dir)):
                return _rel(pp, session_dir)
        except (ValueError, OSError) as exc:
            # Relativizing is cosmetic: keep the absolute path on failure.
            warnings.append(f"geak: failed to relativize path {p!r}: {exc}")
        return p

    status = str(result.get("status") or "unknown")
    base = _to_float(result.get("baseline_throughput_tok_s"))
    final = _to_float(result.get("final_throughput_tok_s"))
    speedup = _to_float(result.get("throughput_speedup"))
    gain_pct: float | None = None
    if isinstance(base, (int, float)) and base > 0 and isinstance(final, (int, float)):
        gain_pct = (final - base) / base * 100.0
    elif isinstance(speedup, (int, float)) and speedup > 0:
        gain_pct = (speedup - 1.0) * 100.0

    accepted_kernels = result.get("accepted_kernels") or []
    accepted_heads = result.get("accepted_heads") or []
    if not isinstance(accepted_kernels, list):
        accepted_kernels = []
        warnings.append("geak: accepted_kernels was not a list")
    if not isinstance(accepted_heads, list):
        accepted_heads = []

    # Back-fill per-kernel attribution from ``kernel_journey.json`` when
    # ``result.json`` shipped the aggregate win but an empty ``accepted_kernels``.
    # Only fires on a successful run with an empty list; a producer-populated
    # list is always preserved verbatim.
    accepted_kernels_source = "result" if accepted_kernels else None
    if not accepted_kernels and status == "ok":
        backfilled = _geak_accepted_kernels_from_journey(result, warnings)
        if backfilled:
            accepted_kernels = backfilled
            accepted_kernels_source = "kernel_journey_backfill"

    section: dict[str, Any] = {
        "engaged": True,
        "status": status,
        # Failure provenance (None on success).
        "error_class": result.get("error_class"),
        "error": result.get("error"),
        "returncode": result.get("returncode"),
        # Throughput / gain attribution (aggregate output tok/s).
        "baseline_throughput_tok_s": base,
        "final_throughput_tok_s": final,
        "throughput_speedup": speedup,
        "gain_pct": gain_pct,
        "metric_basis": result.get("metric_basis"),
        "bench_client": result.get("bench_client"),
        # Latency (median ms), field names aligned with the native sweep.
        "ttft_mean_ms": _to_float(result.get("ttft_ms")),
        "tpot_mean_ms": _to_float(result.get("tpot_ms")),
        "output_parity": result.get("output_parity"),
        # What the optimizer actually changed (per-kernel / head / config).
        "accepted_kernels": accepted_kernels,
        # Provenance of ``accepted_kernels``: ``result``,
        # ``kernel_journey_backfill``, or ``None``.
        "accepted_kernels_source": accepted_kernels_source,
        "accepted_heads": accepted_heads,
        "kernels_optimized": len(accepted_kernels),
        "accepted_config": dict(result.get("accepted_config") or {}),
        # Regimes the kernels were validated at (sweep points outside need reparity).
        "validated_regimes": list(result.get("validated_regimes") or []),
        # Reusable deliverables + human report (relativized when under the session).
        "eval_dir": _rel_if_under(result.get("eval_dir")),
        "report_path": _rel_if_under(result.get("report_path")),
        "final_launch_script": _rel_if_under(result.get("final_launch_script")),
        "bench_script": _rel_if_under(result.get("bench_script")),
        "final_patch": _rel_if_under(result.get("final_patch")),
        # Budget audit (present when the runner was budget-capped/skipped).
        "runner_timeout_s": _to_int(result.get("runner_timeout_s")),
        "kill_timeout_s": _to_int(result.get("kill_timeout_s")),
    }
    return section

