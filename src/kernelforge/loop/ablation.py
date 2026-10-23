# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Price each change a finished campaign kept, telling the rounds that carry the kernel from the passengers.

The ladder is each kept version beside the version before it. Under paired KEEP every round was kept on exactly that
timing, read in one window, so the ladder is read from those records; a round whose recorded lower bound is low is
timed again before it is believed, and rounds kept without a paired timing are timed here. Then what does not pay
is reverted from the best, alone, and every change at once (the start): a cost is how much slower the kernel runs
without it. Arms run side by side, one per device given.
"""

from __future__ import annotations

import asyncio
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
from kernelforge.loop.run_state import NOISE_FLOOR_FILE, LoopStateStore
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


def _changed_lines(workspace: Path, parent: str, commit: str) -> int:
    lines = 0
    for row in _git(workspace, "diff", "--numstat", parent, commit).splitlines():
        added, deleted, _ = row.split("\t", 2)
        lines += int(added) + int(deleted) if added.isdigit() and deleted.isdigit() else 0
    return lines


def _record(
    component: Component,
    status: ArmStatus,
    *,
    cost: dict | None = None,
    cases: dict[str, float] | None = None,
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
        "cases": cases,
        "within_noise": within_noise,
        "detail": detail,
    }


def _combined(arms) -> tuple[float | None, int]:
    """The measured costs composed as independent speedups compose, and how many arms had none."""
    arms = list(arms)
    priced = [arm["cost"]["mean"] for arm in arms if arm["status"] == ArmStatus.MEASURED]
    return (math.prod(1.0 + cost for cost in priced) - 1.0 if priced else None), len(arms) - len(priced)


def _summary(report: dict, floor: float) -> dict:
    """Combine the single costs, and the ladder's steps, and set each beside the all-off cost."""
    arms, ladder = report["arms"], report["ladder"]
    combined, unpriced = _combined(arm for key, arm in arms.items() if key != ALL_OFF)
    ladder_combined, ladder_unpriced = _combined(ladder.values())
    all_off = arms.get(ALL_OFF) or {}
    all_off_cost = all_off["cost"]["mean"] if all_off.get("status") == ArmStatus.MEASURED else None
    meaningful = combined is not None and all_off_cost is not None and all_off_cost > floor
    return {
        "floor": floor,
        "singles_combined": combined,
        "unpriced": unpriced,
        "ladder_combined": ladder_combined,
        "ladder_unpriced": ladder_unpriced,
        "all_off_cost": all_off_cost,
        "additivity": combined / all_off_cost if meaningful else None,
        "completed": all(arm["status"] != ArmStatus.SKIPPED for arm in [*arms.values(), *ladder.values()]),
    }


@dataclass(frozen=True)
class AblationOptions:
    """How much an ablation measures, and on which devices."""

    pairs: int = 3
    null_pairs: int = 5
    # A round whose recorded one-sided lower bound sits below this gain is timed again before it is believed.
    verify_below: float = 0.02
    # One value of HIP_VISIBLE_DEVICES per arm timed at once; none times one arm at a time on the inherited device.
    devices: tuple[str, ...] = ()
    timeout_sec: int = 900


def recorded_rung(event: dict) -> tuple[dict, dict | None] | None:
    """The gain a paired KEEP recorded over the version before it, timed in one window, and its per-case gains."""
    bar = event.get("keep_bar") or {}
    if bar.get("mode") != "paired" or bar.get("mean") is None or bar.get("lower") is None:
        return None
    mean, lower = float(bar["mean"]), float(bar["lower"])
    upper = float(bar["upper"]) if bar.get("upper") is not None else 2 * mean - lower
    cases = {case: float(ratio) - 1.0 for case, ratio in (bar.get("cases") or {}).items()} or None
    return {"mean": mean - 1.0, "lower": lower - 1.0, "upper": upper - 1.0, "scores": []}, cases


class _Slot:
    """One device, and the two trees an arm on it is timed in."""

    def __init__(self, workspace: Path, index: int, device: str):
        sibling = workspace.parent / f"{workspace.name}.forge-ablate-{index}"
        self.best_tree = DetachedWorktree(workspace, f"{sibling}-best")
        self.variant_tree = DetachedWorktree(workspace, f"{sibling}-variant")
        self.env = {"HIP_VISIBLE_DEVICES": device} if device else None

    def remove(self) -> None:
        self.best_tree.remove()
        self.variant_tree.remove()


class _Ablation:
    """One ablation of one campaign, saved to ``ablation.json`` as each arm finishes."""

    def __init__(self, workspace: Path, store: LoopStateStore, options: AblationOptions):
        self.workspace, self.root, self.options = workspace, store.root, options
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
        self.chain = _git(workspace, "rev-list", "--reverse", f"{self.base}..{self.best}").split()
        self.by_commit = {component.commit: component for component in self.components}
        self.keeps = {e["commit_hash"]: e for e in results if e.get("decision") == "KEEP" and e.get("commit_hash")}
        overall = state.best.mean_case_speedup / start - 1.0 if state.best.mean_case_speedup and start else None
        self.all_off = Component(self.base, None, "every kept change (the start)", "", overall)
        self.scored = frozenset(state.baseline_case_times) - frozenset(state.unscored_cases) or None
        self.snr, self.bench_repeat = config.snr_threshold, int(config.bench_repeat)
        self.driver_rel = driver.relative_to(workspace)
        self.slots = [_Slot(workspace, index, device) for index, device in enumerate(options.devices or ("",))]
        previous = self._read()
        self.report = previous if previous.get("best_commit") == self.best else {}
        self.report |= {"best_commit": self.best, "base_commit": self.base, "pairs": options.pairs}
        self.report.setdefault("arms", {})
        self.report.setdefault("ladder", {})

    def _read(self) -> dict:
        try:
            return json.loads((self.root / ABLATION_JSON).read_text())
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        atomic_write_text(self.root / ABLATION_JSON, json.dumps(self.report, indent=2))
        atomic_write_text(self.root / ABLATION_MD, render_ablation_md(self.report))

    def _bench(self, slot: _Slot, tree: DetachedWorktree) -> BenchRun:
        return BenchRun(
            str(tree.path / self.driver_rel),
            driver_args=[],
            repeat=self.bench_repeat,
            cwd=str(tree.path),
            env=slot.env,
        )

    async def _pairs(self, slot: _Slot, count: int, key: str):
        """Pairs of the slot's best tree beside its variant tree; each scores variant time over best time."""
        return await measure_pairs(
            self._bench(slot, slot.best_tree),
            self._bench(slot, slot.variant_tree),
            pairs=count,
            rng=random.Random(f"{self.best}:{key}"),
            timeout_sec=self.options.timeout_sec,
            scored=self.scored,
        )

    async def _floor(self) -> float:
        """The best beside itself: measured once here, unless the campaign's own null control already did."""
        if self.report.get("noise_floor") is None:
            campaign = _read_json(self.root / NOISE_FLOOR_FILE).get("value")
            if campaign is not None:
                self.report["noise_floor"], self.report["noise_floor_source"] = campaign, "campaign"
            else:
                slot = self.slots[0]
                slot.best_tree.ensure_at(self.best)
                slot.variant_tree.ensure_at(self.best)
                measured = await self._pairs(slot, self.options.null_pairs, "null")
                if measured.failure:
                    raise AblationError(f"the best kernel could not be timed beside itself: {measured.failure}")
                self.report["noise_floor"] = noise_floor_from_null(measured.scores)
                self.report["noise_floor_source"] = "measured"
            self._save()
        return paired_keep_floor(self.report["noise_floor"])

    def _driver_changed(self, slot: _Slot) -> bool:
        variant_driver = slot.variant_tree.path / self.driver_rel
        return not variant_driver.is_file() or sha256_of(variant_driver) != sha256_of(
            slot.best_tree.path / self.driver_rel
        )

    def _place_variant(self, slot: _Slot, component: Component) -> ArmStatus | None:
        """Check the variant tree out as best without ``component``, or say why that kernel cannot be timed."""
        slot.best_tree.ensure_at(self.best)
        if component is self.all_off:
            slot.variant_tree.ensure_at(self.base)
        else:
            slot.variant_tree.ensure_at(self.best)
            path = str(slot.variant_tree.path)
            reverted = subprocess.run(
                ["git", "-C", path, "revert", "--no-commit", component.commit], capture_output=True
            )
            if reverted.returncode != 0:
                subprocess.run(["git", "-C", path, "revert", "--abort"], capture_output=True)
                return ArmStatus.NOT_SEPARABLE
        return ArmStatus.DRIVER_CHANGED if self._driver_changed(slot) else None

    async def _measured(self, slot: _Slot, key: str, component: Component, floor: float) -> dict:
        measured = await self._pairs(slot, self.options.pairs, key)
        if measured.failure:
            return _record(component, ArmStatus.FAILED, detail=measured.failure)
        cost = _cost(measured.scores)
        return _record(
            component,
            ArmStatus.MEASURED,
            cost=cost,
            cases={case: ratio - 1.0 for case, ratio in measured.case_means().items()},
            within_noise=cost["lower"] <= floor and cost["upper"] >= -floor,
        )

    async def _arm(self, slot: _Slot, key: str, component: Component, floor: float, deadline: float) -> dict:
        if time.time() > deadline:
            return _record(component, ArmStatus.SKIPPED)
        blocked = self._place_variant(slot, component)
        if blocked is not None:
            return _record(component, blocked)
        validation = await run_validation_pipeline(
            driver_script=str(slot.variant_tree.path / self.driver_rel),
            snr_threshold=self.snr,
            timeout_per_stage=self.options.timeout_sec,
            env=slot.env,
        )
        if not validation.all_passed:
            return _record(component, ArmStatus.REQUIRED_FOR_CORRECTNESS, detail=validation.summary())
        return await self._measured(slot, key, component, floor)

    async def _step(self, slot: _Slot, parent: str, commit: str, floor: float, deadline: float) -> dict:
        """A rung timed here: the kept version beside the one before it; every rung passed correctness when kept."""
        component = self.by_commit[commit]
        if time.time() > deadline:
            return _record(component, ArmStatus.SKIPPED)
        slot.best_tree.ensure_at(commit)
        slot.variant_tree.ensure_at(parent)
        if self._driver_changed(slot):
            return _record(component, ArmStatus.DRIVER_CHANGED)
        return await self._measured(slot, f"step:{commit}", component, floor)

    def _rung_from_record(self, commit: str, floor: float) -> dict | None:
        recorded = recorded_rung(self.keeps.get(commit) or {})
        if recorded is None:
            return None
        cost, cases = recorded
        within = cost["lower"] <= floor and cost["upper"] >= -floor
        return _record(self.by_commit[commit], ArmStatus.MEASURED, cost=cost, cases=cases, within_noise=within)

    def _ladder_work(self, floor: float, deadline: float) -> list:
        """Fill the ladder from what each KEEP recorded; return the rungs to time here, the doubtful ones first."""
        ladder, work = self.report["ladder"], []
        for parent, commit in zip([self.base, *self.chain[:-1]], self.chain):
            settled = ladder.get(commit, {})
            if settled.get("status") in _SETTLED and settled.get("source", "measured") != "recorded":
                continue
            shape = {"parent": parent, "lines": _changed_lines(self.workspace, parent, commit)}
            recorded = self._rung_from_record(commit, floor)
            if recorded is not None:
                ladder[commit] = recorded | shape | {"source": "recorded"}
                if recorded["cost"]["lower"] >= self.options.verify_below:
                    continue
            extra = {"recorded": recorded["cost"]} if recorded is not None else {}
            source = "remeasured" if recorded is not None else "measured"

            async def job(slot, parent=parent, commit=commit, shape=shape, extra=extra, source=source):
                return await self._step(slot, parent, commit, floor, deadline) | shape | extra | {"source": source}

            work.append((ladder, commit, job))
        return work

    def _revert_work(self, floor: float, deadline: float) -> list:
        """Revert from the best only what the ladder says does not pay, and every change at once."""
        arms, ladder = self.report["arms"], self.report["ladder"]
        doubtful = [commit for commit in self.chain if verdict(ladder.get(commit, {})) in {"within noise", "hurts"}]
        work = []
        for key, component in [(ALL_OFF, self.all_off), *((commit, self.by_commit[commit]) for commit in doubtful)]:
            if arms.get(key, {}).get("status") in _SETTLED:
                continue

            async def job(slot, key=key, component=component):
                return await self._arm(slot, key, component, floor, deadline)

            work.append((arms, key, job))
        return work

    async def _run_on_slots(self, work: list) -> None:
        """Run each (table, key, job) on the first free slot, saving the report as each one lands."""
        pending = list(work)

        async def worker(slot: _Slot) -> None:
            while pending:
                table, key, job = pending.pop(0)
                table[key] = await job(slot)
                self._save()

        await asyncio.gather(*(worker(slot) for slot in self.slots))

    async def run(self, deadline: float) -> dict:
        try:
            floor = await self._floor()
            await self._run_on_slots(self._ladder_work(floor, deadline))
            self._save()
            await self._run_on_slots(self._revert_work(floor, deadline))
        finally:
            for slot in self.slots:
                slot.remove()
        self.report |= _summary(self.report, floor)
        self._save()
        return self.report


async def ablate(
    workspace: str | Path, *, max_hours: float = 2.0, options: AblationOptions = AblationOptions()
) -> dict:
    """Ablate the campaign in ``workspace`` within ``max_hours``, holding its lock; return what ablation.json holds.

    Raises ``WorkspaceLockError`` while forge-loop runs on the workspace, ``AblationError`` when it cannot be ablated.
    """
    workspace = Path(workspace).resolve()
    deadline = time.time() + max_hours * 3600
    store = LoopStateStore(str(workspace))
    with store.workspace_lock():
        return await _Ablation(workspace, store, options).run(deadline)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _pct(value, spec: str = "+.2f") -> str:
    return f"{value * 100:{spec}}%" if isinstance(value, (int, float)) else "—"


def verdict(arm: dict) -> str:
    """Whether a measured change pays for itself, sits within the noise floor, or makes the kernel slower."""
    if arm.get("status") != ArmStatus.MEASURED:
        return str(arm.get("status") or "")
    if arm.get("within_noise"):
        return "within noise"
    return "pays" if arm["cost"]["mean"] > 0 else "hurts"


def _ladder_lines(report: dict) -> list[str]:
    ladder = report.get("ladder") or {}
    if not ladder:
        return []
    lines = [
        "## Round by round",
        "",
        "Each kept version beside the version before it: what each round gained. `recorded` is the paired timing the",
        "round was kept on; `remeasured` rounds were timed again here because that bound was low. A round within noise",
        "or one that hurts is not paying for its code, and only those are reverted from the best below.",
        "",
        "| change | iteration | lines | gain when kept | gain measured | one-sided 95% bounds | cases up / down "
        "| verdict | gain per 100 lines | from |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for commit, step in ladder.items():
        cost, cases = step.get("cost") or {}, (step.get("cases") or {}).values()
        bounds = f"{_pct(cost['lower'])} … {_pct(cost['upper'])}" if cost else "—"
        per_line = cost["mean"] * 100 / step["lines"] if cost and step.get("lines") else None
        lines.append(
            f"| `{commit[:12]}` {step.get('title', '')} | {step.get('iteration') or ''} | {step.get('lines', '')} "
            f"| {_pct(step.get('gain_when_kept'))} | {_pct(cost.get('mean'))} | {bounds} "
            f"| {sum(c > 0 for c in cases)} / {sum(c < 0 for c in cases)} | {verdict(step)} | {_pct(per_line)} "
            f"| {step.get('source', 'measured')} |"
        )
    measured = sorted(
        (step for step in ladder.values() if step.get("cost")), key=lambda step: step["cost"]["mean"], reverse=True
    )
    lines.append("")
    for label in ("pays", "within noise", "hurts"):
        names = [f"`{step['commit'][:12]}`" for step in measured if verdict(step) == label]
        lines.append(f"- {label}: {', '.join(names) or 'none'}")
    return [*lines, ""]


def render_ablation_md(report: dict) -> str:
    """The report as a reviewer reads it: what each round gained, what each kept change is worth, and whether the
    singles add up."""
    lines = [
        f"# Ablation of {str(report.get('best_commit', ''))[:12]}",
        "",
        "Cost or gain = how much slower the kernel runs without the change. Noise floor (best beside best): "
        f"{_pct(report.get('noise_floor'), '.2f')}.",
        "",
        *_ladder_lines(report),
        "## Revert one change",
        "",
        f"Each row times the best kernel beside itself with that change reverted, in {report.get('pairs')} pairs;",
        "`all_off` reverts every change (the kernel the campaign started from).",
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
            f"not priced) against all-off {_pct(report.get('all_off_cost'))}{ratio}. The ladder's rounds combined: "
            f"{_pct(report.get('ladder_combined'))} ({report.get('ladder_unpriced')} not priced).",
        ]
    if report.get("completed") is False:
        lines += ["", "The budget ran out: rerun forge-ablate to price the skipped changes."]
    return "\n".join([*lines, ""])
