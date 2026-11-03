"""The flags an aiter dense tuner is run with, checked against what its script accepts."""

from __future__ import annotations

from pathlib import Path

from ..script_probe import ArgFilter, filter_args, probe_script
from .base import TuneContext


def dense_tuner_args(script: Path, ctx: TuneContext, profile_csv: Path, extra_args: tuple[str, ...]) -> ArgFilter:
    """Flags shared by every shape (everything except -i/-o), filtered by the script's argparse surface before
    minutes are spent on it. aiter's --timeout is added per run, to turn on mp_tuner's per-candidate GPU-fault
    isolation."""
    args = [
        "-o2",
        str(profile_csv),
        "--mp",
        str(ctx.mp),
        "--compare",
        "--iters",
        str(ctx.iters),
        "--warmup",
        str(ctx.warmup),
        "--min_improvement_pct",
        str(ctx.min_improvement_pct),
        "-v",
        *extra_args,
    ]
    return filter_args(args, probe_script(script))
