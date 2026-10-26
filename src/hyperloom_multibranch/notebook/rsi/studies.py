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
  the playbook's offline evaluation; ``replay-check`` with the same archives checks that replay
  (:func:`rsi.replay.check`), and ``playbook-generations --model M [--count 5]`` lets that model change the
  playbook offline, a generation at a time, and replays the result held out (:mod:`rsi.generations`);
* ``learnings --notebook DIR --log learnings.jsonl [--archive DIR]``: one session's learnings retrieval
  (:mod:`rsi.learning_eval`), and with an archive what it missed of the lesson pool beside it;
* ``stalls --archive DIR [--archive DIR ...]``: the stall odds for each model type and workload the archives hold,
  and each archived session's stop audit (:mod:`rsi.stops`); ``families`` with the same archives: the direction
  families with the most evidence, and how well their odds tell committing tests apart (:mod:`rsi.family_stats`);
  ``outcomes`` with the same archives: the specialists' outcomes CLOSE kept, added up over the sessions, the menu's
  branches beside its control (:mod:`rsi.outcomes`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_jsonl
from hyperloom.common.llm_config import AnthropicCompletionRequest, aanthropic_completion
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
from hyperloom_multibranch.notebook.rsi.decisions import decision_calls
from hyperloom_multibranch.notebook.rsi.family_stats import family_report
from hyperloom_multibranch.notebook.rsi.generations import GENERATION_RUNS, GENERATIONS, Evolving, generations
from hyperloom_multibranch.notebook.rsi.ladder import Complete
from hyperloom_multibranch.notebook.rsi.learning_eval import read_session
from hyperloom_multibranch.notebook.rsi.outcomes import OutcomePool, pooled
from hyperloom_multibranch.notebook.rsi.replay import (
    OPTIMISTIC,
    PESSIMISTIC,
    REPLAY_RUNS,
    Draws,
    check,
    compare,
    variants,
)
from hyperloom_multibranch.notebook.rsi.scorers import ScorerPool, custom_scorers
from hyperloom_multibranch.notebook.rsi.stops import stall_study

log = logging.getLogger(__name__)

_ARCHIVE = "--archive"
_RUNS = "--runs"
#: The component an LLM call of these studies is tagged with, and the length and time a reply may take.
STUDIES = "notebook-studies"
COMPLETION_TOKENS = 4096
COMPLETION_SECONDS = 300.0


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


def _archived(archives: Sequence[Path]) -> tuple[SearchTree, ...]:
    """Every tree the ``archives`` hold, a session in more than one counted once."""

    trees: dict[str, SearchTree] = {}
    for archive in archives:
        for tree in TreeArchive(archive).load():
            trees.setdefault(tree.session_id, tree)
    return tuple(trees.values())


def _stalls(args: argparse.Namespace) -> Any:
    return stall_study(_archived(args.archive))


def _families(args: argparse.Namespace) -> Any:
    return family_report(_archived(args.archive))


def _outcomes(args: argparse.Namespace) -> Any:
    kept = {
        item.session_id: item
        for archive in args.archive
        for item in OutcomePool.beside(TreeArchive(archive)).sessions()
    }
    if not kept:
        raise StudyError("the archives hold no session's specialist outcomes")
    return pooled(tuple(kept.values()))


def _replayable(args: argparse.Namespace) -> tuple[SearchTree, ...]:
    trees = _archived(args.archive)
    if not trees:
        raise StudyError("the archives hold no session to replay")
    return trees


def _scheduling(args: argparse.Namespace) -> Any:
    return compare(_replayable(args), variants(), drawn=Draws(runs=args.runs, beyond=args.beyond))


def _replay_check(args: argparse.Namespace) -> Any:
    return check(_replayable(args), runs=args.runs)


def _completion(model: str) -> Complete:
    """One Anthropic completion of ``model`` a prompt, over the transport the environment's credential allows."""

    async def complete(system: str, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        request = AnthropicCompletionRequest(
            model=model, messages=messages, max_tokens=COMPLETION_TOKENS, system=system, timeout_s=COMPLETION_SECONDS
        )
        result = await aanthropic_completion(request=request, component=STUDIES, operation="playbook_generations")
        return result.text

    return complete


def _decisions(archives: Sequence[Path], trees: Sequence[SearchTree]) -> dict[str, tuple[Mapping[str, object], ...]]:
    """The judge's calls each of ``trees`` left, from the first of ``archives`` that kept them."""

    found: dict[str, tuple[Mapping[str, object], ...]] = {}
    for tree in trees:
        for archive in archives:
            calls = decision_calls(TreeArchive(archive), tree.session_id)
            if calls:
                found[tree.session_id] = calls
                break
    return found


def _generations(args: argparse.Namespace) -> Any:
    trees = _replayable(args)
    pool = args.pool if args.pool is not None else Path(tempfile.mkdtemp(prefix="playbook-generations-"))
    evolving = Evolving(model=args.model, count=args.count, runs=args.runs)
    run = generations(trees, _completion(args.model), pool, evolving, decisions=_decisions(args.archive, trees))
    return asyncio.run(run)


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
    replayed.add_argument(_RUNS, type=int, default=REPLAY_RUNS)
    replayed.add_argument("--beyond", choices=(OPTIMISTIC, PESSIMISTIC), default=OPTIMISTIC)
    checked = command("replay-check", _replay_check, "check the scheduling replay against the archived sessions")
    _archives(checked)
    checked.add_argument(_RUNS, type=int, default=REPLAY_RUNS)
    evolved = command("playbook-generations", _generations, "evolve the playbook offline and test it held out")
    _archives(evolved)
    evolved.add_argument("--model", required=True, help="the Anthropic model that writes each generation")
    evolved.add_argument("--count", type=int, default=GENERATIONS)
    evolved.add_argument(_RUNS, type=int, default=GENERATION_RUNS)
    evolved.add_argument("--pool", type=Path, default=None, help="where the generations' pool goes; a temporary one")
    learnt = command("learnings", _learnings, "evaluate one session's learnings retrieval")
    learnt.add_argument("--notebook", type=Path, required=True)
    learnt.add_argument("--log", type=Path, required=True)
    learnt.add_argument(_ARCHIVE, type=Path, default=None, help="an archive whose lesson pool the session drew on")
    stalls = command("stalls", _stalls, "the stall odds and every archived session's stop audit")
    _archives(stalls)
    _archives(command("families", _families, "the direction families' evidence and the menu's offline check"))
    _archives(command("outcomes", _outcomes, "the specialists' outcomes over the archived sessions, menu and control"))
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
    "StudyError",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
