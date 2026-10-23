# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deliver a finished campaign as one reviewed commit.

The kept commits are reviewed as one change, the reviewed kernel is checked for correctness and timed beside the
campaign's best, and the commits are squashed into one whose message carries what each round was worth, what the
review changed, and the checks. A review that breaks the kernel or slows it is discarded rather than delivered; the
per-round history stays on its own branch.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import random
import re
import shlex
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kernelforge.evaluation.benchmark.paired import PairedMeasurement, measure_pairs
from kernelforge.evaluation.benchmark.runner import BenchRun
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.evaluation.scoring import PAIRED_MAX_PAIRS, PAIRED_PAIRS, paired_keep_floor, paired_lower_bound
from kernelforge.finalize.review import ReviewBrief, report_of
from kernelforge.finalize.style_gate import style_findings
from kernelforge.workspace.edit_policy import TOOL_OWNED_UNTRACKED_GLOBS
from kernelforge.loop.ablation import ABLATION_JSON, verdict
from kernelforge.loop.campaign_config import CampaignConfigStore
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.worktrees import DetachedWorktree, tree_copy_refusal
from kernelforge.tools.durable_io import atomic_write_text

FINALIZE_JSON, FINALIZE_MD = "finalize.json", "finalize.md"
HISTORY_BRANCH = "forge-history"
REVIEW_ROUNDS = 2
_LINT_TAIL = 1500
_IDENTITY = re.compile(r"\s*(.+?)\s*<([^<>\s]+)>\s*")


class FinalizeError(RuntimeError):
    """The campaign cannot be delivered as it stands."""


def _git(workspace: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(workspace), *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def _paths(output: str) -> list[str]:
    return [path for path in output.split("\0") if path]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def code_of(source: str) -> str | None:
    """The source's syntax tree without docstrings, comments or layout; None when it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        documented = isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if documented and node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def lint_failures(cwd: Path, commands: tuple[str, ...], files: tuple[str, ...]) -> dict[str, str]:
    """Each lint command that fails on ``files`` in ``cwd``, with the end of what it said."""
    failures = {}
    for command in commands:
        try:
            ran = subprocess.run([*shlex.split(command), *files], cwd=cwd, capture_output=True, text=True)
        except OSError as error:
            failures[command] = f"could not run: {error}"
            continue
        if ran.returncode != 0:
            failures[command] = (ran.stdout + ran.stderr).strip()[-_LINT_TAIL:]
    return failures


def pricing_text(ablation: dict) -> str:
    """The ablation's ladder as the review reads it: each round's verdict, gain and size."""
    ladder = ablation.get("ladder") or {}
    if not ladder:
        return "   No ablation priced the rounds: judge each part by reading it, and remove nothing for speed alone."
    lines = [
        "   forge-ablate timed each kept version beside the one before it (gain, one-sided 95% bounds; noise floor "
        f"{float(ablation.get('noise_floor') or 0):.2%}; full table in forge_experiments/ablation.md):"
    ]
    for commit, step in ladder.items():
        cost = step.get("cost") or {}
        measured = f" {cost['mean']:+.2%} [{cost['lower']:+.2%}, {cost['upper']:+.2%}]" if cost else ""
        lines.append(
            f"   - {commit[:12]} (iteration {step.get('iteration')}, {step.get('lines')} lines): "
            f"{verdict(step)}{measured} -- {step.get('title', '')}"
        )
    return "\n".join(lines)


def _timing(measurement: PairedMeasurement, floor: float) -> dict:
    """Best time over reviewed time per pair: slower only when the upper bound sits below 1 - floor."""
    scores = measurement.scores
    record = {"pairs": len(scores), "scores": scores, "floor": floor, "failure": measurement.failure}
    lower = paired_lower_bound(scores)
    if measurement.failure or lower is None:
        return record | {"slower": None}
    mean = statistics.fmean(scores)
    upper = 2 * mean - lower
    return record | {"mean": mean, "lower": lower, "upper": upper, "slower": upper < 1.0 - floor}


@dataclass(frozen=True)
class DeliveryOptions:
    """How a delivery is reviewed, checked and committed."""

    author: str = ""
    lint_commands: tuple[str, ...] = ()
    guides: tuple[str, ...] = ()
    review_rounds: int = REVIEW_ROUNDS
    pairs: int = PAIRED_PAIRS
    timeout_sec: int = 900


class _Delivery:
    """One delivery of one campaign; ``finalize.json`` says how far it got."""

    def __init__(self, workspace: Path, store: LoopStateStore, options: DeliveryOptions):
        self.workspace, self.root = workspace, store.root
        self.config = CampaignConfigStore(str(workspace)).load()
        self.state = store.load()
        self.base = self.config.base_commit
        self.best = self.state.best.commit_hash or _git(workspace, "rev-parse", "HEAD").strip()
        self._refuse_unless_deliverable()
        driver = (workspace / self.config.driver_path).resolve()
        refusal = tree_copy_refusal(workspace=workspace, driver=driver, source_files=list(self.config.source_files))
        if refusal:
            raise FinalizeError(f"a second checkout cannot time this kernel: {refusal}")
        self.driver_rel = driver.relative_to(workspace)
        self.files = tuple(self.config.source_files or [self.config.kernel_path])
        self.allowed = {*self.files, *self.config.commit_new_paths}
        self.options = options
        ablation = _read_json(self.root / ABLATION_JSON)
        self.ablation = ablation if ablation.get("best_commit") == self.best else {}
        # The best beside itself, timed by the ablation just before; without one, the fixed minimum margin.
        self.floor = paired_keep_floor(self.ablation.get("noise_floor"))
        self.scored = frozenset(self.state.baseline_case_times) - frozenset(self.state.unscored_cases) or None
        self.tree = DetachedWorktree(workspace, f"{workspace}.forge-finalize")
        self.report: dict = {"base": self.base, "best": self.best, "files": list(self.files), "floor": self.floor}

    def _refuse_unless_deliverable(self) -> None:
        head = _git(self.workspace, "rev-parse", "HEAD").strip()
        if not self.base or self.base == self.best:
            raise FinalizeError("the campaign kept no change to deliver")
        if head != self.best:
            delivered = _read_json(self.root / FINALIZE_JSON).get("final_commit") == head
            raise FinalizeError(
                f"already delivered as {head[:12]}"
                if delivered
                else f"the workspace is at {head[:12]}, not at the campaign's best {self.best[:12]}"
            )
        if _git(self.workspace, "status", "--porcelain", "--untracked-files=no").strip():
            raise FinalizeError("the workspace has uncommitted changes")

    def _save(self) -> None:
        atomic_write_text(self.root / FINALIZE_JSON, json.dumps(self.report, indent=2))
        atomic_write_text(self.root / FINALIZE_MD, render_finalize_md(self.report))

    def _changed(self) -> list[str]:
        """The files the working tree changed against the best, tracked or new."""
        tracked = _paths(_git(self.workspace, "diff", "-z", "--name-only", "HEAD"))
        untracked = _paths(_git(self.workspace, "ls-files", "-z", "--others", "--exclude-standard"))
        tool_owned = [path for path in untracked if any(fnmatch.fnmatch(path, g) for g in TOOL_OWNED_UNTRACKED_GLOBS)]
        return [*tracked, *(path for path in untracked if path not in tool_owned)]

    def _restore(self, paths: list[str]) -> None:
        """Put ``paths`` back as the best has them: tracked ones checked out, new ones deleted."""
        tracked = set(_paths(_git(self.workspace, "ls-files", "-z")))
        if any(path in tracked for path in paths):
            _git(self.workspace, "checkout", "HEAD", "--", *(path for path in paths if path in tracked))
        for path in paths:
            if path not in tracked:
                (self.workspace / path).unlink(missing_ok=True)

    def _confine(self) -> list[str]:
        """Undo whatever the review touched outside the files it may edit; return those paths."""
        stray = [path for path in self._changed() if path not in self.allowed]
        self._restore(stray)
        return stray

    def _findings(self) -> tuple[list[str], dict[str, str]]:
        style = [str(finding) for finding in style_findings(self.workspace, self.base, list(self.files))]
        failing = lint_failures(self.workspace, self.options.lint_commands, self.files)
        regressions = {command: output for command, output in failing.items() if command not in self.lint_at_base}
        return style, regressions

    async def _review(self, review_fn) -> None:
        kept = int(_git(self.workspace, "rev-list", "--count", f"{self.base}..{self.best}").strip())
        style, lint = self._findings()
        self.report["before_review"] = {"style": style, "lint": lint, "lint_at_base": self.lint_at_base}
        rounds, open_findings = [], (*style, *(f"`{c}` fails: {o}" for c, o in lint.items()))
        for number in range(1, self.options.review_rounds + 1):
            brief = ReviewBrief(
                base=self.base,
                best=self.best,
                kept_commits=kept,
                files=self.files,
                driver=str(self.driver_rel),
                pricing=pricing_text(self.ablation),
                lint_commands=self.options.lint_commands,
                guides=self.options.guides,
                open_findings=open_findings,
                round_number=number,
            )
            reply = await review_fn(
                workspace=str(self.workspace), brief=brief, record=self.root / "finalize" / f"review_{number}.md"
            )
            stray = self._confine()
            style, lint = self._findings()
            rounds.append(
                {
                    "round": number,
                    "failed": reply is None,
                    "report": report_of(reply) if reply is not None else "",
                    "stray": stray,
                    "style": style,
                    "lint": lint,
                }
            )
            open_findings = (*style, *(f"`{c}` fails: {o}" for c, o in lint.items()))
            if reply is None or not open_findings:
                break
        self.report["review"] = {"rounds": rounds, "changed": self._changed()}
        self._save()

    async def _time(self) -> dict:
        self.tree.ensure_at(self.best)
        reviewed = BenchRun(str(self.workspace / self.driver_rel), driver_args=[], repeat=1, cwd=str(self.workspace))
        best = BenchRun(str(self.tree.path / self.driver_rel), driver_args=[], repeat=1, cwd=str(self.tree.path))
        timing = {
            "rng": random.Random(f"{self.best}:review"),
            "timeout_sec": self.options.timeout_sec,
            "scored": self.scored,
        }
        measured = await measure_pairs(reviewed, best, pairs=self.options.pairs, **timing)
        while not measured.failure and len(measured.pairs) < PAIRED_MAX_PAIRS:
            lower = paired_lower_bound(measured.scores)
            if lower is None or lower >= 1.0 - self.floor:
                break
            measured = measured.extended(await measure_pairs(reviewed, best, pairs=1, **timing))
        return _timing(measured, self.floor)

    def _only_words_changed(self, changed: list[str]) -> bool:
        """Whether the review changed comments, docstrings and layout only: the same code, so nothing to check."""
        for path in changed:
            shown = subprocess.run(
                ["git", "-C", str(self.workspace), "show", f"HEAD:{path}"], capture_output=True, text=True
            )
            file = self.workspace / path
            if not path.endswith(".py") or shown.returncode != 0 or not file.is_file():
                return False
            before = code_of(shown.stdout)
            if before is None or before != code_of(file.read_text(encoding="utf-8")):
                return False
        return True

    async def _gate(self) -> None:
        """Keep the review only if the kernel it leaves is still correct and no slower than the best."""
        changed = self.report["review"]["changed"]
        if not changed:
            self.report["gates"] = {"kept": False, "reason": "the review changed nothing"}
            return
        if self._only_words_changed(changed):
            self.report["gates"] = {"kept": True, "reason": "", "same_code": True}
            self._save()
            return
        validation = await run_validation_pipeline(
            driver_script=str(self.workspace / self.driver_rel),
            snr_threshold=self.config.snr_threshold,
            timeout_per_stage=self.options.timeout_sec,
        )
        gates: dict = {"correctness": validation.summary(), "correct": validation.all_passed}
        if validation.all_passed:
            gates["timing"] = await self._time()
        timing = gates.get("timing") or {}
        reason = (
            "the reviewed kernel fails correctness"
            if not validation.all_passed
            else f"the reviewed kernel could not be timed: {timing['failure']}"
            if timing.get("slower") is None
            else "the reviewed kernel is slower than the best"
            if timing["slower"]
            else ""
        )
        if reason:
            self._restore(changed)
        self.report["gates"] = gates | {"kept": not reason, "reason": reason}
        self._save()

    def _commit_message(self, paths: list[str]) -> str:
        start, best = self.state.search_start_mean_case_speedup, self.state.best.mean_case_speedup
        scores = f" ({start:.4f}x -> {best:.4f}x)" if start and best else ""
        lines = [f"forge-loop: deliver {self.config.kernel_path}{scores}", "", pricing_text(self.ablation).strip(), ""]
        gates = self.report["gates"]
        reports = [entry["report"] for entry in self.report["review"]["rounds"] if entry["report"]]
        lines.append(f"Review: {'kept' if gates['kept'] else 'not kept: ' + gates['reason']}")
        lines += [line for report in reports for line in report.splitlines() if line.strip()] if gates["kept"] else []
        timing = gates.get("timing") or {}
        if "mean" in timing:
            lines.append(
                f"Timed beside the best in {timing['pairs']} pairs: best/reviewed {timing['mean']:.4f} "
                f"[{timing['lower']:.4f}, {timing['upper']:.4f}], noise floor {timing['floor']:.2%}."
            )
        final_round = self.report["review"]["rounds"][-1] if self.report["review"]["rounds"] else {}
        left = len(final_round.get("style", [])) + len(final_round.get("lint", {}))
        lines += [f"Style and lint: {left} finding(s) left." if left else "Style and lint: clean.", ""]
        lines.append(f"Files: {', '.join(paths)}. Per-round history: {HISTORY_BRANCH} ({self.best[:12]}).")
        return "\n".join(lines) + "\n"

    def _commit_args(self) -> list[str]:
        if not self.options.author:
            return []
        match = _IDENTITY.fullmatch(self.options.author)
        if match is None:
            raise FinalizeError(f"--author must read 'Name <email>', not {self.options.author!r}")
        name, email = match.groups()
        return ["-c", f"user.name={name}", "-c", f"user.email={email}"]

    def _squash(self) -> None:
        """One commit from the base to the delivered kernel, on the campaign's branch; the rounds stay on history."""
        workspace = self.workspace
        _git(workspace, "branch", "-f", HISTORY_BRANCH, self.best)
        if self.report["gates"]["kept"]:
            _git(workspace, "add", "-A", "--", *self.report["review"]["changed"])
        _git(workspace, "reset", "-q", "--soft", self.base)
        paths = _paths(_git(workspace, "diff", "-z", "--cached", "--name-only", self.base))
        if not paths:
            _git(workspace, "reset", "-q", "--soft", self.best)
            raise FinalizeError("the delivered kernel is the one the campaign started from")
        message = self._commit_message(paths)
        _git(workspace, *self._commit_args(), "commit", "-q", "-F", "-", stdin=message)
        final = _git(workspace, "rev-parse", "HEAD").strip()
        self.report |= {
            "final_commit": final,
            "history_branch": HISTORY_BRANCH,
            "paths": paths,
            "outside_sources": [path for path in paths if path not in self.allowed],
            "diff_stat": _git(workspace, "diff", "--stat", self.base, final),
            "message": message,
        }

    async def run(self, review_fn) -> dict:
        try:
            self.tree.ensure_at(self.base)
            self.lint_at_base = lint_failures(self.tree.path, self.options.lint_commands, self.files)
            await self._review(review_fn)
            await self._gate()
        finally:
            self.tree.remove()
        self._squash()
        self._save()
        return self.report


async def finalize_campaign(workspace: str | Path, *, review_fn, options: DeliveryOptions = DeliveryOptions()) -> dict:
    """Review, check and squash the finished campaign in ``workspace``, holding its lock; return finalize.json.

    Raises ``WorkspaceLockError`` while forge-loop runs on it, ``FinalizeError`` when it cannot be delivered.
    """
    workspace = Path(workspace).resolve()
    store = LoopStateStore(str(workspace))
    with store.workspace_lock():
        return await _Delivery(workspace, store, options).run(review_fn)


def render_finalize_md(report: dict) -> str:
    """The delivery as a reviewer reads it: the commit, what the review changed, and the checks."""
    gates = report.get("gates") or {}
    lines = [f"# Delivery of {str(report.get('best', ''))[:12]}", ""]
    if report.get("final_commit"):
        lines += [
            f"Final commit: `{report['final_commit'][:12]}`; per-round history on `{report.get('history_branch')}`.",
            "",
        ]
        if report.get("outside_sources"):
            lines += [f"Files outside the kernel's sources in the commit: {', '.join(report['outside_sources'])}", ""]
    if gates:
        lines += [f"Review {'kept' if gates.get('kept') else 'not kept: ' + str(gates.get('reason'))}.", ""]
    timing = gates.get("timing") or {}
    if "mean" in timing:
        lines += [
            f"Timed beside the best in {timing['pairs']} pairs: best/reviewed {timing['mean']:.4f} "
            f"[{timing['lower']:.4f}, {timing['upper']:.4f}], noise floor {timing['floor']:.2%}.",
            "",
        ]
    for entry in (report.get("review") or {}).get("rounds", []):
        lines += [f"## Review round {entry['round']}{' (failed)' if entry.get('failed') else ''}", ""]
        lines += [entry.get("report") or "(no report)", ""]
        if entry.get("stray"):
            lines += [f"Edits outside the kernel's sources, undone: {', '.join(entry['stray'])}", ""]
        left = [*entry.get("style", []), *(f"{c} fails" for c in entry.get("lint", {}))]
        lines += [
            ("Left after this round:\n" + "\n".join(f"- {item}" for item in left))
            if left
            else "Style and lint: clean.",
            "",
        ]
    if report.get("diff_stat"):
        lines += ["```", report["diff_stat"].rstrip(), "```", ""]
    return "\n".join(lines)
