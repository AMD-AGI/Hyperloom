# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Price each change a finished campaign kept.

The best kernel is timed in pairs beside itself with one kept change reverted, for every change, and beside the
kernel the campaign started from (every change off). A change's cost is how much slower the best kernel runs
without it. Changes that only pay together each look cheap when removed alone, so the single costs are combined
and set beside the all-off cost: where the two disagree, the singles cannot be read one at a time.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from kernelforge.evaluation.benchmark.paired import measure_pairs
from kernelforge.evaluation.benchmark.runner import BenchRun
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.evaluation.scoring import keep_t_critical, noise_floor_from_null, paired_keep_floor
from kernelforge.loop.campaign_config import CampaignConfigStore
from kernelforge.loop.direction_ledger import keep_gains
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.worktrees import DetachedWorktree, _git, sha256_of, tree_copy_refusal
from kernelforge.tools.durable_io import atomic_write_text

ABLATION_JSON, ABLATION_MD = "ablation.json", "ablation.md"
ALL_OFF = "all_off"


class ArmStatus(str, Enum):
    MEASURED = "measured"
    # Reverting the change conflicts with a later one, so best-without-it is not a kernel anyone wrote.
    NOT_SEPARABLE = "not_separable"
    # The arm's tree holds a different driver, so its times would not be comparable.
    DRIVER_CHANGED = "driver_changed"
    REQUIRED_FOR_CORRECTNESS = "required_for_correctness"
    FAILED = "failed"
    SKIPPED = "skipped_budget"


# An arm with one of these outcomes is not measured again on a rerun.
_SETTLED = {
    ArmStatus.MEASURED,
    ArmStatus.NOT_SEPARABLE,
    ArmStatus.DRIVER_CHANGED,
    ArmStatus.REQUIRED_FOR_CORRECTNESS,
}


class AblationError(RuntimeError):
    """The campaign cannot be ablated as it stands."""


@dataclass(frozen=True)
class Component:
    """One commit between the start and the best, and what the campaign recorded when it kept it."""

    commit: str
    iteration: int | None
    title: str
    direction_id: str
    gain_when_kept: float | None


def _plan_title(plan: str) -> str:
    """The plan's first line that says what it changes; its ``Direction:`` claim names the direction instead."""
    for line in plan.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.lower().startswith("direction:"):
            return line[:100]
    return ""


def kept_components(
    workspace: Path, base: str, best: str, results: list[dict], start_score: float | None
) -> list[Component]:
    """Every commit from ``base`` to ``best``, matched to the KEEP that made it; largest recorded gain first."""
    gains = keep_gains(results, start_score)
    keeps = {e["commit_hash"]: e for e in results if e.get("decision") == "KEEP" and e.get("commit_hash")}
    components = []
    for commit in _git(workspace, "rev-list", "--reverse", f"{base}..{best}").split():
        event = keeps.get(commit)
        if event is None:
            title = _git(workspace, "log", "-1", "--format=%s", commit).strip()
            components.append(Component(commit, None, title, "", None))
            continue
        iteration = int(event["iter"])
        title = _plan_title(str(event.get("plan") or "")) or _git(workspace, "log", "-1", "--format=%s", commit).strip()
        components.append(
            Component(commit, iteration, title, str(event.get("direction_id") or ""), gains.get(iteration))
        )
    return sorted(components, key=lambda component: component.gain_when_kept or 0.0, reverse=True)


def _cost(scores: list[float]) -> dict:
    """Variant time over best time, minus one, with one-sided 95% bounds on each side."""
    mean = statistics.fmean(scores)
    margin = keep_t_critical(len(scores)) * statistics.stdev(scores) / math.sqrt(len(scores))
    return {"mean": mean - 1.0, "lower": mean - margin - 1.0, "upper": mean + margin - 1.0, "scores": scores}


def _record(
    component: Component,
    status: ArmStatus,
    *,
    cost: dict | None = None,
    within_noise: bool | None = None,
    detail: str = "",
) -> dict:
    return {
        "commit": component.commit,
        "iteration": component.iteration,
        "title": component.title,
        "direction_id": component.direction_id,
        "gain_when_kept": component.gain_when_kept,
        "status": status.value,
        "cost": cost,
        "within_noise": within_noise,
        "detail": detail,
    }


def _summary(arms: dict, floor: float) -> dict:
    """Combine the single costs as independent speedups would compose, and set them beside the all-off cost."""
    singles = [arm for key, arm in arms.items() if key != ALL_OFF]
    priced = [arm["cost"]["mean"] for arm in singles if arm["status"] == ArmStatus.MEASURED]
    combined = math.prod(1.0 + cost for cost in priced) - 1.0 if priced else None
    all_off = arms.get(ALL_OFF) or {}
    all_off_cost = all_off["cost"]["mean"] if all_off.get("status") == ArmStatus.MEASURED else None
    meaningful = combined is not None and all_off_cost is not None and all_off_cost > floor
    return {
        "floor": floor,
        "singles_combined": combined,
        "unpriced": len(singles) - len(priced),
        "all_off_cost": all_off_cost,
        "additivity": combined / all_off_cost if meaningful else None,
        "completed": all(arm["status"] != ArmStatus.SKIPPED for arm in arms.values()),
    }


class _Ablation:
    """One ablation of one campaign, saved to ``ablation.json`` as each arm finishes."""

    def __init__(self, workspace: Path, store: LoopStateStore, *, pairs: int, null_pairs: int, timeout_sec: int):
        self.workspace, self.root = workspace, store.root
        config = CampaignConfigStore(str(workspace)).load()
        driver = (workspace / config.driver_path).resolve()
        refusal = tree_copy_refusal(workspace=workspace, driver=driver, source_files=list(config.source_files))
        if refusal:
            raise AblationError(f"a second checkout cannot time this kernel: {refusal}")
        state = store.load()
        self.base = config.base_commit
        self.best = state.best.commit_hash or _git(workspace, "rev-parse", "HEAD").strip()
        if not self.base or self.base == self.best:
            raise AblationError("the campaign kept no change to ablate")
        results = [event for event in store.read_events() if event.get("type") == "iteration_result"]
        start = state.search_start_mean_case_speedup
        self.components = kept_components(workspace, self.base, self.best, results, start)
        overall = state.best.mean_case_speedup / start - 1.0 if state.best.mean_case_speedup and start else None
        self.all_off = Component(self.base, None, "every kept change (the start)", "", overall)
        self.scored = frozenset(state.baseline_case_times) - frozenset(state.unscored_cases) or None
        self.snr, self.bench_repeat = config.snr_threshold, int(config.bench_repeat)
        self.pairs, self.null_pairs, self.timeout = pairs, null_pairs, timeout_sec
        self.driver_rel = driver.relative_to(workspace)
        sibling = workspace.parent / f"{workspace.name}.forge-ablate"
        self.best_tree = DetachedWorktree(workspace, f"{sibling}-best")
        self.variant_tree = DetachedWorktree(workspace, f"{sibling}-variant")
        previous = self._read()
        self.report = previous if previous.get("best_commit") == self.best else {}
        self.report |= {"best_commit": self.best, "base_commit": self.base, "pairs": pairs}
        self.report.setdefault("arms", {})

    def _read(self) -> dict:
        try:
            return json.loads((self.root / ABLATION_JSON).read_text())
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        atomic_write_text(self.root / ABLATION_JSON, json.dumps(self.report, indent=2))
        atomic_write_text(self.root / ABLATION_MD, render_ablation_md(self.report))

    def _bench(self, tree: DetachedWorktree) -> BenchRun:
        return BenchRun(str(tree.path / self.driver_rel), driver_args=[], repeat=self.bench_repeat, cwd=str(tree.path))

    async def _pairs(self, count: int, key: str):
        """Pairs of best beside the variant; each scores variant time over best time."""
        return await measure_pairs(
            self._bench(self.best_tree),
            self._bench(self.variant_tree),
            pairs=count,
            rng=random.Random(f"{self.best}:{key}"),
            timeout_sec=self.timeout,
            scored=self.scored,
        )

    async def _floor(self) -> float:
        if self.report.get("noise_floor") is None:
            self.variant_tree.ensure_at(self.best)
            measured = await self._pairs(self.null_pairs, "null")
            if measured.failure:
                raise AblationError(f"the best kernel could not be timed beside itself: {measured.failure}")
            self.report["noise_floor"] = noise_floor_from_null(measured.scores)
            self._save()
        return paired_keep_floor(self.report["noise_floor"])

    def _place_variant(self, component: Component) -> ArmStatus | None:
        """Check the variant tree out as best without ``component``, or say why that kernel cannot be timed."""
        if component is self.all_off:
            self.variant_tree.ensure_at(self.base)
        else:
            self.variant_tree.ensure_at(self.best)
            path = str(self.variant_tree.path)
            reverted = subprocess.run(
                ["git", "-C", path, "revert", "--no-commit", component.commit], capture_output=True
            )
            if reverted.returncode != 0:
                subprocess.run(["git", "-C", path, "revert", "--abort"], capture_output=True)
                return ArmStatus.NOT_SEPARABLE
        variant_driver = self.variant_tree.path / self.driver_rel
        if not variant_driver.is_file() or sha256_of(variant_driver) != sha256_of(
            self.best_tree.path / self.driver_rel
        ):
            return ArmStatus.DRIVER_CHANGED
        return None

    async def _arm(self, key: str, component: Component, floor: float, deadline: float) -> dict:
        if time.time() > deadline:
            return _record(component, ArmStatus.SKIPPED)
        blocked = self._place_variant(component)
        if blocked is not None:
            return _record(component, blocked)
        validation = await run_validation_pipeline(
            driver_script=str(self.variant_tree.path / self.driver_rel),
            snr_threshold=self.snr,
            timeout_per_stage=self.timeout,
        )
        if not validation.all_passed:
            return _record(component, ArmStatus.REQUIRED_FOR_CORRECTNESS, detail=validation.summary())
        measured = await self._pairs(self.pairs, key)
        if measured.failure:
            return _record(component, ArmStatus.FAILED, detail=measured.failure)
        cost = _cost(measured.scores)
        return _record(
            component,
            ArmStatus.MEASURED,
            cost=cost,
            within_noise=cost["lower"] <= floor and cost["upper"] >= -floor,
        )

    async def run(self, deadline: float) -> dict:
        try:
            self.best_tree.ensure_at(self.best)
            floor = await self._floor()
            arms = self.report["arms"]
            for key, component in [(ALL_OFF, self.all_off), *((c.commit, c) for c in self.components)]:
                if arms.get(key, {}).get("status") in _SETTLED:
                    continue
                arms[key] = await self._arm(key, component, floor, deadline)
                self._save()
        finally:
            self.best_tree.remove()
            self.variant_tree.remove()
        self.report |= _summary(self.report["arms"], floor)
        self._save()
        return self.report


async def ablate(
    workspace: str | Path, *, pairs: int = 3, null_pairs: int = 5, max_hours: float = 1.0, timeout_sec: int = 900
) -> dict:
    """Ablate the campaign in ``workspace`` within ``max_hours``, holding its lock; return what ablation.json holds.

    Raises ``WorkspaceLockError`` while forge-loop runs on the workspace, ``AblationError`` when it cannot be ablated.
    """
    workspace = Path(workspace).resolve()
    deadline = time.time() + max_hours * 3600
    store = LoopStateStore(str(workspace))
    with store.workspace_lock():
        ablation = _Ablation(workspace, store, pairs=pairs, null_pairs=null_pairs, timeout_sec=timeout_sec)
        return await ablation.run(deadline)


def _pct(value, spec: str = "+.2f") -> str:
    return f"{value * 100:{spec}}%" if isinstance(value, (int, float)) else "—"


def render_ablation_md(report: dict) -> str:
    """The report as a reviewer reads it: what each kept change is worth, and whether the singles add up."""
    lines = [
        f"# Ablation of {str(report.get('best_commit', ''))[:12]}",
        "",
        f"Each row times the best kernel beside itself with that change reverted, in {report.get('pairs')} pairs;",
        "`all_off` reverts every change (the kernel the campaign started from). Cost = how much slower the best",
        f"kernel runs without it. Noise floor (best beside best): {_pct(report.get('noise_floor'), '.2f')}.",
        "",
        "| change | iteration | direction | gain when kept | status | cost without it | one-sided 95% bounds "
        "| within noise |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, arm in (report.get("arms") or {}).items():
        cost = arm.get("cost") or {}
        bounds = f"{_pct(cost['lower'])} … {_pct(cost['upper'])}" if cost else "—"
        name = "all off" if key == ALL_OFF else f"`{key[:12]}` {arm.get('title', '')}"
        noise = {True: "yes", False: "no"}.get(arm.get("within_noise"), "")
        lines.append(
            f"| {name} | {arm.get('iteration') or ''} | {arm.get('direction_id') or ''} "
            f"| {_pct(arm.get('gain_when_kept'))} | {arm.get('status')} | {_pct(cost.get('mean'))} | {bounds} "
            f"| {noise} |"
        )
    if "singles_combined" in report:
        additivity = report.get("additivity")
        ratio = f"; ratio {additivity:.2f}" if isinstance(additivity, float) else ""
        lines += [
            "",
            f"Single costs combined: {_pct(report.get('singles_combined'))} ({report.get('unpriced')} change(s) "
            f"not priced) against all-off {_pct(report.get('all_off_cost'))}{ratio}.",
        ]
    if report.get("completed") is False:
        lines += ["", "The budget ran out: rerun forge-ablate to price the skipped changes."]
    return "\n".join([*lines, ""])
