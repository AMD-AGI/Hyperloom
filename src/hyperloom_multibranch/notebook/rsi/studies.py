# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB2 studies on what MB1 sessions left: advice for the scheduler's owner, never a changed parameter.

* :func:`branch_cap_study` replays the archived sessions similar to a task under each of :data:`CAPS` branch caps
  (Q7). The scheduler's LLM judge, not a threshold, stops branches, so the replay never stops one on a stall: only the
  cap prunes. It recommends the cap of the best similarity-weighted utility, the smaller of equals, once the similar
  sessions are worth :data:`frontier.MIN_EFFECTIVE_SESSIONS`.

``python -m hyperloom_multibranch.notebook.rsi.studies`` runs each study and prints its report, in markdown or, with
``--json``, as JSON:

* ``frontier --archive DIR [--log SESSION_ID=frontier.jsonl ...]``: the frontier calibration (:mod:`rsi.calibration`),
  with the island bandit's rows (:mod:`rsi.bandit`), the scorers of :func:`rsi.scorers.custom_scorers` and the
  expressions of the archive's pool beside it, each judged on the whole archive: for reference, since a session
  counts only the sessions that started after a scorer was registered;
* ``branch-cap --archive DIR --like SESSION_ID [--caps 2,3,4]``: the branch cap replay, for sessions like that one;
* ``learnings --notebook DIR --log learnings.jsonl [--archive DIR]``: one session's learnings retrieval
  (:mod:`rsi.learning_eval`), and with an archive what it missed of the lesson pool beside it;
* ``dispersion --shadow consistency.jsonl [--labels FILE] [--archive DIR] --model M --tp N --gpu G [--agentx]``: the
  dispersion prior with the shadow runs' noise (:mod:`rsi.shadow`); ``--labels`` is a JSON object of test id to model,
  TP, GPU, and ``--agentx`` asks for an AgentX trace replay's prior;
* ``stalls --archive DIR [--archive DIR ...]``: the stall odds for each model type and workload the archives hold,
  and each archived session's stop audit (:mod:`rsi.stops`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_jsonl
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeArchive
from hyperloom_multibranch.notebook.rsi.bandit import bandit_baselines
from hyperloom_multibranch.notebook.rsi.calibration import (
    ROUND_SECONDS,
    Baseline,
    CalibrationSettings,
    calibrate,
    scorer_baseline,
)
from hyperloom_multibranch.notebook.rsi.expression import ExpressionError, ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import MIN_EFFECTIVE_SESSIONS
from hyperloom_multibranch.notebook.rsi.learning_eval import read_session
from hyperloom_multibranch.notebook.rsi.priors import HOSTED, DispersionPrior, DispersionTask, dispersion_prior
from hyperloom_multibranch.notebook.rsi.replay import TICK_SECONDS, PolicyEvaluation, ReplayPolicy, evaluate
from hyperloom_multibranch.notebook.rsi.report import number, table
from hyperloom_multibranch.notebook.rsi.scorers import ScorerPool, custom_scorers
from hyperloom_multibranch.notebook.rsi.shadow import ShadowNoise, read_shadow, shadow_noise
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.stops import stall_study

log = logging.getLogger(__name__)

CAPS = (2, 3, 4, 6, 8)
#: A stop threshold no stall falls under: the replay leaves stopping to the cap alone.
NO_STOP = -math.inf
_SAME = 1e-9
_ARCHIVE = "--archive"


@dataclass(frozen=True)
class BranchCapStudy:
    """Each cap's replayed utility over the similar sessions; ``recommended`` only when they are worth enough."""

    rows: tuple[PolicyEvaluation, ...]
    recommended: int | None
    note: str

    def render(self) -> str:
        """The study as markdown: a table of the caps, then the advice."""

        rows = [(row.policy.branch_cap, number(row.utility), number(row.effective_sessions, 1)) for row in self.rows]
        lines = table(("branch cap", "utility", "similar sessions' worth"), rows)
        return "\n".join(["Branch cap replay (Q7), no stop threshold:", "", *lines, "", self.note])


def branch_cap_study(
    history: Sequence[SearchTree],
    task: TaskConditions,
    *,
    caps: Sequence[int] = CAPS,
    tick_seconds: float = TICK_SECONDS,
    min_sessions: float = MIN_EFFECTIVE_SESSIONS,
) -> BranchCapStudy:
    """How the archived sessions similar to ``task`` would have fared under each of ``caps``, and which to consider."""

    rows = tuple(evaluate(ReplayPolicy(NO_STOP, cap), history, task, tick_seconds=tick_seconds) for cap in caps)
    worth = max((row.effective_sessions for row in rows), default=0.0)
    if worth < min_sessions:
        return BranchCapStudy(
            rows,
            None,
            f"Too little evidence for advice: the similar sessions are worth {worth:.1f}; {min_sessions} needed.",
        )
    best = max(row.utility for row in rows)
    cap = min(row.policy.branch_cap for row in rows if row.utility >= best - _SAME)
    return BranchCapStudy(
        rows,
        cap,
        f"Consider a branch cap of {cap}: the best replayed utility, {best:.2f}, over sessions worth {worth:.1f}. "
        "Advice only; the cap is the scheduler's (Q7).",
    )


@dataclass(frozen=True)
class DispersionReport:
    """The dispersion prior for one task, beside the shadow runs' noise it was given."""

    prior: DispersionPrior
    shadow: ShadowNoise

    def render(self) -> str:
        """The prior and the shadow noise as markdown."""

        prior, shadow = self.prior, self.shadow
        rows = [
            ("prior", f"{number(prior.dispersion_pct)}% from {prior.source.value} ({prior.environment})"),
            ("single-host floor", number(prior.lower_bound_pct)),
            ("shadow hosted / local cv", f"{number(shadow.hosted_cv_pct)}% / {number(shadow.local_cv_pct)}%"),
            ("hosted minus local", f"{number(shadow.offset_pct)}% over {len(shadow.test_ids)} shadow tests"),
        ]
        return "\n".join(["Dispersion prior (Q1):", "", *table(("measure", "value"), rows)])


class StudyError(ValueError):
    """A study cannot run on what it was given."""


def _session_log(text: str) -> tuple[str, Path]:
    session_id, separator, path = text.partition("=")
    if not separator or not session_id or not path:
        raise argparse.ArgumentTypeError(f"expected SESSION_ID=PATH, got {text!r}")
    return session_id, Path(path)


def _caps(text: str) -> tuple[int, ...]:
    try:
        caps = tuple(int(item) for item in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected caps such as 2,3,4, got {text!r}") from None
    return caps


def _scorer_rows(archive: Path) -> list[Baseline]:
    """The scorers of :func:`custom_scorers` and the expressions of ``archive``'s pool, as rows judged on the whole
    archive."""

    rows = [scorer_baseline(scorer) for scorer in custom_scorers()]
    for record in ScorerPool(archive).expressions():
        try:
            rows.append(scorer_baseline(ExpressionScorer.read(record.label, record.expression)))
        except ExpressionError as exc:
            log.warning("left out the pooled frontier scorer %s: %s", record.label, exc)
    return rows


def _frontier(args: argparse.Namespace) -> Any:
    logs = {session_id: read_jsonl(path, skip_malformed=True, skip_non_dict=True) for session_id, path in args.log}
    trees = TreeArchive(args.archive).load()
    settings = CalibrationSettings(round_seconds=args.round_seconds, horizon_seconds=args.horizon_seconds)
    baselines = [*bandit_baselines(trees), *_scorer_rows(args.archive)]
    return calibrate(trees, logs=logs, settings=settings, baselines=baselines)


def _branch_cap(args: argparse.Namespace) -> Any:
    trees = TreeArchive(args.archive).load()
    like = next((tree for tree in trees if tree.session_id == args.like), None)
    if like is None:
        raise StudyError(f"the archive {args.archive} holds no session {args.like!r}")
    return branch_cap_study(trees, like.task, caps=args.caps)


def _learnings(args: argparse.Namespace) -> Any:
    return asyncio.run(read_session(args.notebook, args.log, args.archive))


def _dispersion(args: argparse.Namespace) -> Any:
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    noise = shadow_noise(read_shadow(args.shadow), labels=labels, model=args.model, tp=args.tp, gpu=args.gpu)
    history = TreeArchive(args.archive).load() if args.archive else []
    task = DispersionTask(args.model, args.tp, args.gpu, agentx=args.agentx)
    prior = dispersion_prior(history, task, environment=args.environment, shadow=noise)
    return DispersionReport(prior, noise)


def _stalls(args: argparse.Namespace) -> Any:
    trees: dict[str, SearchTree] = {}
    for archive in args.archive:
        for tree in TreeArchive(archive).load():
            trees.setdefault(tree.session_id, tree)
    return stall_study(tuple(trees.values()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hyperloom_multibranch.notebook.rsi.studies", description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, run: Callable[[argparse.Namespace], Any], text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=text, description=text)
        sub.set_defaults(run=run)
        return sub

    frontier = command("frontier", _frontier, "calibrate the frontier scores on the archived sessions")
    frontier.add_argument(_ARCHIVE, type=Path, required=True)
    frontier.add_argument("--log", type=_session_log, action="append", default=[], metavar="SESSION_ID=PATH")
    frontier.add_argument("--round-seconds", type=float, default=ROUND_SECONDS)
    frontier.add_argument("--horizon-seconds", type=float, default=None)
    cap = command("branch-cap", _branch_cap, "replay the sessions like one under each branch cap (Q7)")
    cap.add_argument(_ARCHIVE, type=Path, required=True)
    cap.add_argument("--like", required=True, metavar="SESSION_ID")
    cap.add_argument("--caps", type=_caps, default=CAPS)
    learnt = command("learnings", _learnings, "evaluate one session's learnings retrieval")
    learnt.add_argument("--notebook", type=Path, required=True)
    learnt.add_argument("--log", type=Path, required=True)
    learnt.add_argument(_ARCHIVE, type=Path, default=None, help="an archive whose lesson pool the session drew on")
    noise = command("dispersion", _dispersion, "the dispersion prior with the shadow runs' noise (Q1)")
    noise.add_argument("--shadow", type=Path, required=True)
    noise.add_argument("--labels", type=Path, default=None)
    noise.add_argument(_ARCHIVE, type=Path, default=None)
    noise.add_argument("--model", required=True)
    noise.add_argument("--tp", type=int, required=True)
    noise.add_argument("--gpu", required=True)
    noise.add_argument("--environment", default=HOSTED)
    noise.add_argument("--agentx", action="store_true", help="the prior of an AgentX trace replay")
    stalls = command("stalls", _stalls, "the stall odds and every archived session's stop audit")
    stalls.add_argument(_ARCHIVE, type=Path, action="append", required=True, help="an archive; give more than one")
    return parser


def _finite(value: Any) -> Any:
    """``value`` with every infinite or NaN number as ``None``, so that it is strict JSON."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study ``argv`` names and print its report; 2 when it cannot run on what it was given."""

    args = _parser().parse_args(argv)
    try:
        report = args.run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_finite(encode(report)), indent=2, sort_keys=True) if args.json else report.render())
    return 0


__all__ = [
    "CAPS",
    "NO_STOP",
    "BranchCapStudy",
    "DispersionReport",
    "StudyError",
    "branch_cap_study",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
