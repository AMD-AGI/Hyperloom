# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m inference_suite``: ``run`` one round, ``stop`` the server a round left running, ``resolve`` a base,
``stage`` the tools a round runs, ``publish`` them where several hosts share them.

``run --config <config.yaml> --run-dir <dir>`` runs the round into ``<dir>``, prints its ``report.json`` path and
exits 0 when the round completed, 1 when it did not; a relative path there or in the config is read below the
caller's directory, and a relative ``model_path`` only when it names something there. SIGTERM and SIGINT cancel the
round, which still reaps what it started and writes its report. ``stop --pid-dir <dir>`` prints what it reaped as JSON and exits 1 while a process
outlives SIGKILL. ``resolve --query <query.json> --gpus <n> --out <base_launch.json> [--inferencex-commit <sha>]``
reads the sources once, InferenceX at ``<sha>`` instead of the head of its ``main`` when given, and writes the base
they give, or exits 2 naming why each source holds none, or why ``<sha>`` is refused: InferenceX has no such commit, or
it lacks code the resolver runs, or a source breaks its own format. ``stage --tool-cache <dir> [--source
<InferenceX repository>]`` stages the pinned tools (:func:`inference_suite.staging.ensure_staged`) and prints the
manifest's sha256; it stages nothing into a cache that already verifies, and exits 2 with ``staging_missing`` when it
cannot stage or the cache is damaged. ``publish --stagings <dir> [--source <InferenceX repository>]`` publishes the
staging of the pins in force as ``<dir>/<key>`` (:func:`inference_suite.staging.publish`) and prints that directory,
or exits 2 with ``staging_missing``. Exit 2 is also a document that breaks its contract. ``multinode`` resolves and
runs a multi-node PD launch (:mod:`inference_suite.multinode.cli`).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import FrameType

import yaml

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.contracts.config import SuiteConfig
from inference_suite.contracts.layout import RunDirError
from inference_suite.contracts.report import FailedStep, RunStatus
from inference_suite.multinode.cli import add_commands as add_multinode_commands
from inference_suite.recipe.model import RecipeSourceError
from inference_suite.recipe.resolve import Miss, resolve_base
from inference_suite.recipe.sources import InferencexCommitRefused, fetch_sources
from inference_suite.recipe.user_file import parse_user_file
from inference_suite.run import run_round
from inference_suite.runtime.process import PidfileError
from inference_suite.runtime.server import stop
from inference_suite.staging import StagingError, ensure_staged, publish

PROG = "python -m inference_suite"
EXIT_OK = 0
EXIT_NOT_COMPLETED = 1
EXIT_REFUSED = 2
#: The signals that cancel a running round.
CANCEL_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_UTF8 = "utf-8"


class _SignalCancel:
    """The round's cancel: empty until a cancelling signal arrives, then why the round was cancelled."""

    def __init__(self) -> None:
        self.reason = ""

    def __call__(self) -> str:
        return self.reason

    def hear(self, signum: int, _frame: FrameType | None) -> None:
        self.reason = f"received {signal.Signals(signum).name}"


def _refused(detail: str) -> int:
    print(detail, file=sys.stderr)
    return EXIT_REFUSED


def read_config(path: Path, run_dir: Path) -> SuiteConfig:
    """The config at ``path``, writing its round into ``run_dir``."""
    config = SuiteConfig.from_document(yaml.safe_load(path.read_text(encoding=_UTF8)))
    return replace(config, paths=replace(config.paths, run_dir=str(run_dir)))


def _run(args: argparse.Namespace) -> int:
    try:
        config = read_config(args.config, args.run_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _refused(f"{FailedStep.CONFIG_INVALID.value}: {exc}")
    cancel = _SignalCancel()
    for signum in CANCEL_SIGNALS:
        signal.signal(signum, cancel.hear)
    try:
        result = run_round(config, cancel=cancel)
    except RunDirError as exc:
        return _refused(str(exc))
    print(result.layout.report)
    return EXIT_OK if result.report.run_status is RunStatus.COMPLETED else EXIT_NOT_COMPLETED


def _stop(args: argparse.Namespace) -> int:
    try:
        outcome = stop(args.pid_dir)
    except PidfileError as exc:
        return _refused(str(exc))
    print(json.dumps({"reaped": list(outcome.reaped), "survivors": list(outcome.survivors)}))
    return EXIT_NOT_COMPLETED if outcome.survivors else EXIT_OK


def _resolve(args: argparse.Namespace) -> int:
    try:
        query = LaunchQuery.from_document(json.loads(args.query.read_text(encoding=_UTF8)))
        user_file = (
            None
            if args.user_file is None
            else parse_user_file(str(args.user_file), args.user_file.read_bytes(), framework=query.framework)
        )
    except (OSError, ValueError) as exc:
        return _refused(f"{FailedStep.CONFIG_INVALID.value}: {exc}")
    try:
        snapshot = fetch_sources(query, user_file=user_file, inferencex_commit=args.inferencex_commit)
        resolved = resolve_base(query, snapshot, gpus=args.gpus)
    except (InferencexCommitRefused, RecipeSourceError) as exc:
        return _refused(str(exc))
    if isinstance(resolved, Miss):
        for reason in resolved.reasons():
            print(reason, file=sys.stderr)
        return EXIT_REFUSED
    args.out.write_text(json.dumps(resolved.to_document(), indent=2) + "\n", encoding=_UTF8)
    print(resolved.digest())
    return EXIT_OK


def _stage(args: argparse.Namespace) -> int:
    try:
        digest = ensure_staged(args.tool_cache, args.source)
    except StagingError as exc:
        return _refused(f"{FailedStep.STAGING_MISSING.value}: {exc}")
    print(digest)
    return EXIT_OK


def _publish(args: argparse.Namespace) -> int:
    try:
        published = publish(args.stagings, args.source)
    except StagingError as exc:
        return _refused(f"{FailedStep.STAGING_MISSING.value}: {exc}")
    print(published)
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one round into an empty run_dir")
    run_parser.add_argument("--config", type=Path, required=True, help="the round's config.yaml")
    run_parser.add_argument("--run-dir", type=Path, required=True, help="where the round writes; absent or empty")
    run_parser.set_defaults(handler=_run)
    stop_parser = commands.add_parser("stop", help="reap the server a pid_dir's server.pid records")
    stop_parser.add_argument("--pid-dir", type=Path, required=True)
    stop_parser.set_defaults(handler=_stop)
    resolve_parser = commands.add_parser("resolve", help="resolve a session's base launch from its sources")
    resolve_parser.add_argument("--query", type=Path, required=True, help="the LaunchQuery document")
    resolve_parser.add_argument("--gpus", type=int, required=True, help="the GPUs the session may use")
    resolve_parser.add_argument("--out", type=Path, required=True, help="where to write base_launch.json")
    resolve_parser.add_argument("--user-file", type=Path, help="the operator's own launch file, which is the base")
    resolve_parser.add_argument(
        "--inferencex-commit",
        default="",
        metavar="SHA",
        help=(
            "read InferenceX at this commit (a full sha or a prefix of 7 or more hex digits) instead of the head of"
            " its main; the cookbooks, vLLM recipes and ATOM catalog are still read at main"
        ),
    )
    resolve_parser.set_defaults(handler=_resolve)
    stage_parser = commands.add_parser("stage", help="stage the pinned tools into a tool cache unless it holds them")
    stage_parser.add_argument("--tool-cache", type=Path, required=True, help="the tool cache rounds verify")
    stage_parser.add_argument(
        "--source", type=Path, help="an InferenceX repository holding the pinned commit; fetched when omitted"
    )
    stage_parser.set_defaults(handler=_stage)
    publish_parser = commands.add_parser(
        "publish", help="publish the staging of the pins in force in a shared directory"
    )
    publish_parser.add_argument("--stagings", type=Path, required=True, help="the directory published stagings live in")
    publish_parser.add_argument(
        "--source", type=Path, help="an InferenceX repository holding the pinned commit; fetched when omitted"
    )
    publish_parser.set_defaults(handler=_publish)
    add_multinode_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["CANCEL_SIGNALS", "EXIT_NOT_COMPLETED", "EXIT_OK", "EXIT_REFUSED", "PROG", "main", "read_config"]
