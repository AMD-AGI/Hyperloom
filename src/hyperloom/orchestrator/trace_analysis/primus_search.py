# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Training kernel targets from the tuning agent's top configurations, not only its best.

The seed search's best configuration is the projection's pick, and the
projection can misrank configurations whose times are close. Kernel targets
taken from the top ``k`` legal configurations still hold when the pick is wrong:
each is projected with ``--kernel-report`` and the reports are merged into one,
each configuration weighted equally.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
TOP_K_ENV = "HYPERLOOM_PRIMUS_SEARCH_TOP_K"
_OBJECTIVE = "tokens_per_s_per_gpu"
_CONFIG = "--config"
_EST = "est_total_ms"


def top_k_setting() -> int:
    """``HYPERLOOM_PRIMUS_SEARCH_TOP_K``, or 5; 1 (or less) keeps only the best configuration."""
    raw = os.environ.get(TOP_K_ENV, "")
    try:
        return int(raw) if raw else DEFAULT_TOP_K
    except ValueError:
        log.warning("ignoring %s=%r: not an integer", TOP_K_ENV, raw)
        return DEFAULT_TOP_K


@dataclasses.dataclass(frozen=True)
class Trial:
    """One legal trial of the seed search: its YAML and what the search projected for it."""

    idx: int
    config: str
    tokens_per_s_per_gpu: float
    iteration_ms: float | None
    memory_gb: float | None
    knobs: dict[str, Any]

    @property
    def label(self) -> str:
        k = self.knobs
        if not k:
            return Path(self.config).stem
        text = f"#{self.idx} TP{k.get('tp')} PP{k.get('pp')} EP{k.get('ep')} CP{k.get('cp')} MBS{k.get('mbs')}"
        if (k.get("vpp") or 1) > 1:
            text += f" VPP{k['vpp']}"
        if k.get("recompute_granularity"):
            text += f" recompute {k['recompute_granularity']}"
            if k.get("recompute_num_layers"):
                text += f" x{k['recompute_num_layers']}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"label": self.label}


def _trial(record: dict[str, Any]) -> Trial | None:
    result = record.get("result") or {}
    cmd = result.get("cmd") or []
    score = result.get(_OBJECTIVE) or 0.0
    if not result.get("legal") or score <= 0 or _CONFIG not in cmd[:-1]:
        return None
    return Trial(
        idx=int(record.get("idx", -1)),
        config=cmd[cmd.index(_CONFIG) + 1],
        tokens_per_s_per_gpu=float(score),
        iteration_ms=result.get("iteration_ms"),
        memory_gb=result.get("memory_per_gpu_gb_upper") or result.get("memory_per_gpu_gb"),
        knobs=record.get("config") or result.get("config") or {},
    )


def ranked_trials(trials_path: Path) -> list[Trial]:
    """The search's legal trials, best first by its objective (tokens/s/GPU), one per YAML."""
    trials: dict[str, Trial] = {}
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        trial = _trial(json.loads(line)) if line.strip() else None
        if trial is not None and trial.config not in trials:
            trials[trial.config] = trial
    return sorted(trials.values(), key=lambda t: -t.tokens_per_s_per_gpu)


def chosen(ranked: list[Trial], winner_config: str, top_k: int = DEFAULT_TOP_K) -> list[Trial]:
    """The top ``k`` configurations, the search's best first (on its own when the trials don't list it)."""
    best = [t for t in ranked if t.config == winner_config] or [Trial(-1, winner_config, 0.0, None, None, {})]
    return (best + [t for t in ranked if t.config != winner_config])[: max(1, top_k)]


def _ms(row: dict[str, Any]) -> float:
    return float(row.get(_EST) or 0.0)


def _stage_totals(report: dict[str, Any]) -> dict[int, float]:
    stages: dict[int, float] = {}
    for row in report.get("kernels") or []:
        stages[row["stage"]] = stages.get(row["stage"], 0.0) + _ms(row)
    return stages


def _busiest_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Each term's rows on the stage it costs the most: the rank a kernel target is sized on."""
    by_term: dict[tuple, dict[int, list[dict[str, Any]]]] = {}
    for row in report.get("kernels") or []:
        term = (row["scope"], row["op"], row["phase"], row["kind"])
        by_term.setdefault(term, {}).setdefault(row["stage"], []).append(row)
    return [row for stages in by_term.values() for row in max(stages.values(), key=lambda rows: sum(map(_ms, rows)))]


def _row_key(row: dict[str, Any]) -> tuple:
    dims = json.dumps(row.get("dims") or {}, sort_keys=True)
    return (row["scope"], row["op"], row["phase"], row["kind"], dims)


def merge_reports(reports: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """One kernel report for several configurations, each weighted equally; the first's metadata describes it.

    Configurations differ in pipeline stages, so stages are not merged by index.
    Each configuration contributes every term on the stage it costs the most,
    rescaled to that stage's share of the first report's busiest stage. A term's
    time is then its mean share over the configurations (0 where it doesn't
    run), and its shapes are the union of theirs. The merged report has one
    stage; each row keeps the stages it came from.
    """
    if not reports:
        raise ValueError("no kernel reports to merge")
    base = max(_stage_totals(reports[0]).values(), default=0.0) or 1.0
    merged: dict[tuple, dict[str, Any]] = {}
    for report, label in zip(reports, labels):
        totals = _stage_totals(report)
        for row in _busiest_rows(report):
            weight = base / (totals[row["stage"]] or base) / len(reports)
            into = merged.setdefault(_row_key(row), {**row, "stage": 0, "calls": 0, _EST: 0.0, "configs": []})
            into["calls"] += round(weight * int(row.get("calls") or 0))
            into[_EST] += weight * _ms(row)
            into["configs"].append(f"{label} (stage {row['stage']})")
    first = reports[0]
    return {
        **first,
        "provenance": {**(first.get("provenance") or {}), "merged_configs": labels},
        "kernels": sorted(merged.values(), key=lambda r: -r[_EST]),
    }


@dataclasses.dataclass(frozen=True)
class Projected:
    """The chosen configurations' kernel reports (the best first) and what became of each."""

    reports: list[dict[str, Any]]
    labels: list[str]
    configs: list[dict[str, Any]]


def project_top(trials: list[Trial], project: Callable[[Trial, int], dict[str, Any]]) -> Projected:
    """Project each chosen configuration; one that fails is recorded and left out, unless it is the best."""
    reports, labels, configs = [], [], []
    for i, trial in enumerate(trials):
        row = trial.to_dict()
        try:
            report = project(trial, i)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            if i == 0:
                raise
            log.warning("projection of %s failed; left out of the targets: %s", trial.label, exc)
            configs.append(row | {"error": str(exc)[-500:]})
            continue
        reports.append(report)
        labels.append(trial.label)
        configs.append(row | {"projected_iteration_ms": (report.get("iteration") or {}).get("iteration_time_ms")})
    return Projected(reports, labels, configs)
