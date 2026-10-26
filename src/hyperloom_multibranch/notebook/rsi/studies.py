# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The studies on what past sessions left: advice for the scheduler's owner, never a changed parameter.

``python -m hyperloom_multibranch.notebook.rsi.studies`` runs each study and prints its report, in markdown or, with
``--json``, as JSON:

* ``frontier --archive DIR [--log SESSION_ID=frontier.jsonl ...]``: the frontier calibration (:mod:`rsi.calibration`),
  with the island bandit's rows (:mod:`rsi.bandit`), the scorers of :func:`rsi.scorers.custom_scorers` and the
  expressions of the archive's pool beside it, each judged on the whole archive: for reference, since a session
  counts only the sessions that started after a scorer was registered;
* ``scheduling --archive DIR [--archive DIR ...] [--runs N] [--beyond optimistic|pessimistic]``: the scheduling
  playbooks of :func:`rsi.replay.variants` replayed on every archived session beside the archive (:mod:`rsi.replay`),
  the playbook's offline evaluation;
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
from hyperloom_multibranch.notebook.rsi.learning_eval import read_session
from hyperloom_multibranch.notebook.rsi.priors import HOSTED, DispersionPrior, DispersionTask, dispersion_prior
from hyperloom_multibranch.notebook.rsi.replay import OPTIMISTIC, PESSIMISTIC, REPLAY_RUNS, compare, variants
from hyperloom_multibranch.notebook.rsi.report import number, table
from hyperloom_multibranch.notebook.rsi.scorers import ScorerPool, custom_scorers
from hyperloom_multibranch.notebook.rsi.shadow import ShadowNoise, read_shadow, shadow_noise
from hyperloom_multibranch.notebook.rsi.stops import stall_study

log = logging.getLogger(__name__)

_ARCHIVE = "--archive"


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


def _learnings(args: argparse.Namespace) -> Any:
    return asyncio.run(read_session(args.notebook, args.log, args.archive))


def _dispersion(args: argparse.Namespace) -> Any:
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    noise = shadow_noise(read_shadow(args.shadow), labels=labels, model=args.model, tp=args.tp, gpu=args.gpu)
    history = TreeArchive(args.archive).load() if args.archive else []
    task = DispersionTask(args.model, args.tp, args.gpu, agentx=args.agentx)
    prior = dispersion_prior(history, task, environment=args.environment, shadow=noise)
    return DispersionReport(prior, noise)


def _archived(archives: Sequence[Path]) -> tuple[SearchTree, ...]:
    """Every tree the ``archives`` hold, a session in more than one counted once."""

    trees: dict[str, SearchTree] = {}
    for archive in archives:
        for tree in TreeArchive(archive).load():
            trees.setdefault(tree.session_id, tree)
    return tuple(trees.values())


def _stalls(args: argparse.Namespace) -> Any:
    return stall_study(_archived(args.archive))


def _scheduling(args: argparse.Namespace) -> Any:
    trees = _archived(args.archive)
    if not trees:
        raise StudyError("the archives hold no session to replay")
    return compare(trees, variants(), runs=args.runs, beyond=args.beyond)


def _archives(command: argparse.ArgumentParser) -> None:
    """``--archive``, given once for each archive a study reads."""

    command.add_argument(_ARCHIVE, type=Path, action="append", required=True, help="an archive; give more than one")


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
    replayed = command("scheduling", _scheduling, "replay the scheduling playbooks on every archived session")
    _archives(replayed)
    replayed.add_argument("--runs", type=int, default=REPLAY_RUNS)
    replayed.add_argument("--beyond", choices=(OPTIMISTIC, PESSIMISTIC), default=OPTIMISTIC)
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
    _archives(stalls)
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
    "DispersionReport",
    "StudyError",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
