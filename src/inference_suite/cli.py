# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m inference_suite``: ``run`` one round, ``stop`` the server a round left running, ``resolve`` a base.

``run --config <config.yaml> --run-dir <dir>`` runs the round into ``<dir>``, prints its ``report.json`` path and
exits 0 when the round completed, 1 when it did not. SIGTERM and SIGINT cancel the round, which still reaps what it
started and writes its report. ``stop --pid-dir <dir>`` prints what it reaped as JSON and exits 1 while a process
outlives SIGKILL. ``resolve --query <query.json> --gpus <n> --out <base_launch.json>`` reads the sources once and
writes the base they give, or exits 2 naming why each source holds none. Exit 2 is also a document that breaks its
contract.
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
from inference_suite.recipe.resolve import Miss, resolve_base
from inference_suite.recipe.sources import fetch_sources
from inference_suite.recipe.user_file import parse_user_file
from inference_suite.run import run_round
from inference_suite.runtime.process import PidfileError
from inference_suite.runtime.server import stop

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
    resolved = resolve_base(query, fetch_sources(query, user_file=user_file), gpus=args.gpus)
    if isinstance(resolved, Miss):
        for reason in resolved.reasons():
            print(reason, file=sys.stderr)
        return EXIT_REFUSED
    args.out.write_text(json.dumps(resolved.to_document(), indent=2) + "\n", encoding=_UTF8)
    print(resolved.digest())
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
    resolve_parser.set_defaults(handler=_resolve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["CANCEL_SIGNALS", "EXIT_NOT_COMPLETED", "EXIT_OK", "EXIT_REFUSED", "PROG", "main", "read_config"]
