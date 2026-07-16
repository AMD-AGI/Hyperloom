# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.state.optimization_journal import (
    classify_change_kind,
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _benchmark_report_metrics,
    _find_benchmark_report,
    _load_json_safe,
    _rel,
    _to_float,
    _to_int,
)



# Explore search ledger
def _shape_ledger(
    ledger: dict[str, Any] | None,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Normalize a search ledger (params / backends / explore) for the report.

    Shapes the ``accepted`` / ``rejected`` / ``tested`` entries to a stable
    field set and derives a ``top_by_gain`` ranking from the tested entries.

    Args:
        ledger (dict[str, Any] | None): A raw search ledger from state, or
            ``None``.
        top_n (int): Maximum number of entries in ``top_by_gain``. Defaults to
            20.

    Returns:
        dict[str, Any]: ``{"schema_version", "tested_count", "accepted",
        "rejected", "top_by_gain"}``. An empty shell is returned when
        ``ledger`` is not a dict.
    """
    if not isinstance(ledger, dict):
        return {
            "schema_version": 0,
            "tested_count": 0,
            "accepted": [],
            "rejected": [],
            "top_by_gain": [],
        }

    def _shape_entry(e: Any) -> dict[str, Any]:
        """Coerce one ledger entry to the stable field set.

        Args:
            e (Any): A raw ledger entry.

        Returns:
            dict[str, Any]: The shaped entry, or ``{}`` when ``e`` is not a
            dict.
        """
        if not isinstance(e, dict):
            return {}
        args = str(e.get("extra_server_args") or "")
        envs = dict(e.get("extra_envs") or {})
        provenance = str(e.get("provenance") or "")
        # Classify the variant into a stable operation_kind (backend/param/env).
        change_kind = classify_change_kind(
            "explore",
            {"extra_server_args": args, "extra_envs": envs},
        )
        shaped = {
            "name": str(e.get("name") or ""),
            "fingerprint": str(e.get("fingerprint") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            "output_throughput": _to_float(e.get("output_throughput") or e.get("tput")),
            "gain_pct": _to_float(e.get("gain_pct")),
            "ts": str(e.get("ts") or ""),
            "operation_kind": operation_kind_for("explore", change_kind),
        }
        if provenance:
            shaped["provenance"] = provenance
            shaped["proposer"] = proposer_for(provenance)
        scope = str(e.get("scope") or "")
        if scope:
            shaped["scope"] = scope
        return shaped

    accepted = [_shape_entry(e) for e in ledger.get("accepted") or []]
    rejected = [_shape_entry(e) for e in ledger.get("rejected") or []]
    tested = list((ledger.get("tested") or {}).values()) if isinstance(ledger.get("tested"), dict) else []
    tested_shaped = [_shape_entry(e) for e in tested]
    top_by_gain = sorted(
        (e for e in tested_shaped if e.get("gain_pct") is not None),
        key=lambda e: e.get("gain_pct") or 0.0,
        reverse=True,
    )[:top_n]
    return {
        "schema_version": int(ledger.get("schema_version") or 0),
        "tested_count": len(tested),
        "accepted": accepted,
        "rejected": rejected,
        "top_by_gain": top_by_gain,
    }


def _patch_winners_history(
    rows: list[Any],
    baseline_tput: float | None,
) -> list[dict[str, Any]]:
    """Fix ``backend_winners_history`` data gaps: fall back to session ``baseline_tput`` for a 0 ``base_tput`` and compute missing per-winner ``gain_pct``.

    Args:
        rows (list[Any]): Raw ``backend_winners_history`` rows.
        baseline_tput (float | None): Session baseline throughput used to
            patch a zero / missing ``base_tput``.

    Returns:
        list[dict[str, Any]]: The patched rows with repaired ``base_tput`` and
        back-filled per-winner ``gain_pct`` where derivable.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        try:
            bt = float(row.get("base_tput") or 0.0)
        except (TypeError, ValueError):
            bt = 0.0
        if bt <= 0 and baseline_tput and baseline_tput > 0:
            row["base_tput"] = float(baseline_tput)
            row["base_tput_source"] = "session_baseline"
            bt = float(baseline_tput)
        if bt > 0:
            winners = list(row.get("winners") or [])
            new_winners: list[dict[str, Any]] = []
            for w in winners:
                if not isinstance(w, dict):
                    continue
                w2 = dict(w)
                if w2.get("gain_pct") in (None, "") and w2.get("tput") is not None:
                    try:
                        w2["gain_pct"] = (float(w2["tput"]) - bt) / bt * 100.0
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                new_winners.append(w2)
            row["winners"] = new_winners
            best = row.get("best")
            if isinstance(best, dict) and best.get("gain_pct") in (None, "") and best.get("tput") is not None:
                try:
                    best = dict(best)
                    best["gain_pct"] = (float(best["tput"]) - bt) / bt * 100.0
                    row["best"] = best
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        out.append(row)
    return out


def _shape_winners_history(
    explore_search: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Persist ``explore_search.winners_history`` rows with their join key + source.

    Re-emits the fingerprint→provenance map (dropped by ``_shape_ledger``) so
    downstream can reconstruct ``phase_breakdown.explore.by_domain`` offline.

    Args:
        explore_search (dict[str, Any] | None): The ``explore_search`` ledger
            from state, or ``None``.

    Returns:
        list[dict[str, Any]]: The shaped ``winners_history`` rows (round id,
        variant, fingerprint, provenance, scope, gain, args/envs, ts). Empty
        when no history is present.
    """
    if not isinstance(explore_search, dict):
        return []
    rows = explore_search.get("winners_history")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for w in rows:
        if not isinstance(w, dict):
            continue
        out.append(
            {
                "round_id": str(w.get("round_id") or ""),
                "variant_name": str(w.get("variant_name") or w.get("name") or ""),
                "fingerprint": str(w.get("fingerprint") or ""),
                "provenance": str(w.get("provenance") or ""),
                "scope": str(w.get("scope") or ""),
                "gain_pct": _to_float(w.get("gain_pct")),
                "extra_args": str(w.get("extra_args") or w.get("extra_server_args") or ""),
                "extra_envs": dict(w.get("extra_envs") or {}),
                "ts": str(w.get("ts") or ""),
            }
        )
    return out


def collect_explore_search(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the explore-phase search summary for the breakdown.

    Args:
        state: Session state mapping.
        warnings: Mutable list that collected warnings are appended to.

    Returns:
        A dict summarizing the explore-phase search activity and outcomes.
    """
    # Emit all three ledgers (unified explore + legacy params/backends);
    # unused ones shape to empty shells.
    explore_ledger = _shape_ledger(state.get("explore_search"))
    # Persist provenance+fingerprint winners_history for offline recompute.
    explore_ledger["winners_history"] = _shape_winners_history(state.get("explore_search"))
    explore_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    explore_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    params_ledger = _shape_ledger(state.get("params_search"))
    params_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    params_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    backends_ledger = _shape_ledger(state.get("backends_search"))

    baseline_tput = _to_float(state.get("baseline_tput"))
    return {
        "explore": explore_ledger,
        "params": params_ledger,
        "backends": backends_ledger,
        "synergy_attempted": list(state.get("synergy_attempted") or []),
        "discovered_flags": dict(state.get("discovered_flags") or {}),
        "backend_winners_history": _patch_winners_history(
            state.get("backend_winners_history") or [],
            baseline_tput,
        ),
    }


# Sweep
_VARIANT_NAME_RE = re.compile(r"variant_(\d+)_conc(\d+)_isl(\d+)_osl(\d+)", re.IGNORECASE)


def _scan_sweep_variants(session_dir: Path) -> list[Path]:
    """List variant directories under runs/sweep/<task>/variant_*/.

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        list[Path]: Every ``variant_*`` directory across all sweep tasks,
        sorted. Empty when no ``runs/sweep/`` tree exists.
    """
    root = session_dir / "runs" / "sweep"
    if not root.exists():
        return []
    out: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for v in sorted(task_dir.glob("variant_*")):
            if v.is_dir():
                out.append(v)
    return out


def _shape_sweep_point(
    variant_dir: Path,
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Shape one sweep variant directory into a sweep-point row.

    Parses the ``conc`` / ``isl`` / ``osl`` from the variant directory name,
    reads its benchmark report for throughput / latency metrics, and derives a
    status (``ok`` / ``failed`` / ``skipped``).

    Args:
        variant_dir (Path): A ``variant_*`` directory.
        session_dir (Path): Absolute session root (used to relativize paths).
        warnings (list[str]): Shared warnings list (mutated in place when a
            report fails to parse).

    Returns:
        dict[str, Any]: A sweep-point row with the parsed workload knobs,
        metrics, status, and relative report path.
    """
    name = variant_dir.name
    m = _VARIANT_NAME_RE.search(name)
    conc: int | None = None
    isl_: int | None = None
    osl_: int | None = None
    if m:
        try:
            conc = int(m.group(2))
            isl_ = int(m.group(3))
            osl_ = int(m.group(4))
        except ValueError:
            pass
    report = _find_benchmark_report(variant_dir)
    report_data = _load_json_safe(report, warnings) if report else None
    status = "ok"
    out_tput = ttft = tpot = e2el = None
    if isinstance(report_data, dict):
        out_tput, ttft, tpot, e2el = _benchmark_report_metrics(report_data)
        if report_data.get("success") is False:
            status = "failed"
    elif report is None:
        status = "skipped"
    return {
        "variant_name": name,
        "conc": conc,
        "isl": isl_,
        "osl": osl_,
        "output_throughput_tok_s": out_tput,
        "ttft_mean_ms": ttft,
        "tpot_mean_ms": tpot,
        "e2el_mean_ms": e2el,
        "status": status,
        "benchmark_report_path": _rel(report, session_dir) if report else None,
    }


def collect_sweep(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the sweep section.

    Merges the in-state ``last_sweep`` summary (grid size, best-overall,
    per-conc bests, pareto front) with the variant points discovered on disk.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The sweep section (grid size, best_overall,
        best_for_each_conc, pareto_front, all_variants, config_path).
    """
    ls = state.get("last_sweep") or {}
    if not isinstance(ls, dict):
        ls = {}

    variants_on_disk = [_shape_sweep_point(v, session_dir, warnings) for v in _scan_sweep_variants(session_dir)]
    return {
        "grid_size": _to_int(ls.get("grid_size")) or len(variants_on_disk),
        "best_overall": dict(ls.get("best_overall") or {}),
        "best_for_each_conc": list(ls.get("best_for_each_conc") or []),
        "pareto_front": list(ls.get("pareto_front") or []),
        "all_variants": variants_on_disk,
        "config_path": ls.get("config_path"),
    }

