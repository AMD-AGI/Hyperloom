# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.runner.native.knob_tune``: sweep a kernel knob of the framework a single-branch session
builds from source (:mod:`hyperloom.measure.runner.native.knob_sweep`), on this process's GPU.

The builds are the session's, over the base it recorded and on its build site, into a store of its sweeps
(:func:`~hyperloom.measure.runner.native.session_builds.tune_native_builds`). The GPU is the one this process is given
(``ROCR_VISIBLE_DEVICES``); ``--gpu-lock`` names a file whose exclusive lock each round holds, for a GPU other processes
take under the same lock. The sweep's document is printed as JSON and written with the winning patch (``winning.patch``,
empty when nothing beat the default beyond the noise) to ``--out``. The patch is a git patch over the session's base:
apply it to your worktree and hand it to the normal integration path; only a test of it says whether it helps.

Exit status: 0 when the sweep ran (with or without a winner), 1 when the request is not one the knob takes, 2 when the
session builds no server from source, the pinned commit could not be built or timed, or no value was timed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path

from hyperloom.common.env_safety import is_secret_shaped_env_name
from hyperloom.common.outside_facts.kernel_knob import KnobSpecError
from hyperloom.common.visible_devices import HIP_LEVEL_VARS
from hyperloom.inference_optimizer.session.paths import session_dir
from hyperloom.measure.runner.native.backend_ops import check_cases, time_cases
from hyperloom.measure.runner.native.knob_sweep import (
    DEFAULT_COPIES,
    SweepError,
    SweepHooks,
    SweepPlan,
    SweepTiming,
    plan_sweep,
    run_sweep,
)
from hyperloom.inference_optimizer.native_builds import NativeBuildsRefused
from hyperloom.measure.runner.native.session_builds import SessionNativeBuilds, tune_native_builds

__all__ = ["main"]

_RUN_TIMEOUT_SEC = 300
_REPORT, _PATCH = "report.json", "winning.patch"


def _shape(text: str) -> tuple[int, int]:
    m, sep, k = text.lower().partition("x")
    if not sep or not m.isdigit() or not k.isdigit():
        raise argparse.ArgumentTypeError(f"a shape is MxK (the weight's rows and columns), not {text!r}")
    return int(m), int(k)


def _axis(text: str) -> tuple[str, list[str]]:
    name, sep, values = text.partition("=")
    if not sep or not name or not values:
        raise argparse.ArgumentTypeError(f"an axis is NAME=V1,V2,..., not {text!r}")
    return name, values.split(",")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.runner.native.knob_tune", description=__doc__)
    parser.add_argument("--knob", default="", help="the knob to sweep, one the rule pack declares")
    parser.add_argument(
        "--axis", type=_axis, action="append", default=[], help="NAME=V1,V2: the keys to sweep (an axis not named: all)"
    )
    parser.add_argument("--values", default="", help="V1,V2: the values to force (the knob's all when omitted)")
    parser.add_argument("--shape", type=_shape, action="append", required=True, help="MxK weight shape (repeatable)")
    parser.add_argument("--rounds", type=int, default=5, help="timed rounds of every build")
    parser.add_argument(
        "--copies", type=int, default=DEFAULT_COPIES, help="weights of each shape in one launch, past the GPU's cache"
    )
    parser.add_argument("--parallel", type=int, default=2, help="builds at once on the build CPUs")
    parser.add_argument("--session", type=Path, default=None, help="the session directory (this run's when omitted)")
    parser.add_argument("--gpu-lock", type=Path, default=None, help="a file whose exclusive lock each round holds")
    parser.add_argument("--out", type=Path, default=None, help="where the report and the patch go")
    return parser


@contextlib.contextmanager
def _held(lock: Path | None, env: Mapping[str, str]) -> Iterator[Mapping[str, str]]:
    if lock is None:
        yield env
        return
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield env


def _bench_env() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name not in HIP_LEVEL_VARS and not is_secret_shaped_env_name(name)
    }


def _refuse_arch(plan: SweepPlan, builds: SessionNativeBuilds) -> None:
    gfx = builds.session.planner.site.machine.gfx_arch
    if gfx not in plan.knob.archs:
        raise KnobSpecError(f"{plan.name} sets a branch compiled for {list(plan.knob.archs)}; the session builds {gfx}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        values = [value for value in args.values.split(",") if value]
        timing = SweepTiming(args.shape, args.rounds, args.copies)
        plan = plan_sweep(args.knob, dict(args.axis), values, timing)
        builds = tune_native_builds(args.session or session_dir())
        _refuse_arch(plan, builds)
    except KnobSpecError as refused:
        print(json.dumps({"error": str(refused)}))
        return 1
    except NativeBuildsRefused as refused:
        print(json.dumps({"error": str(refused)}))
        return 2
    out = args.out or builds.root / "sweeps" / f"{plan.name}-{int(time.time())}"
    try:
        result = run_sweep(
            plan,
            builds.read_upstream,
            out,
            SweepHooks(
                build=builds.build_patch,
                hold=partial(_held, args.gpu_lock, _bench_env()),
                time=partial(time_cases, timeout=_RUN_TIMEOUT_SEC),
                check=partial(check_cases, timeout=_RUN_TIMEOUT_SEC),
            ),
            parallel=args.parallel,
        )
    except (SweepError, KnobSpecError) as failed:
        print(json.dumps({"error": str(failed)}))
        return 2
    (out / _PATCH).write_bytes(result.patch)
    (out / _REPORT).write_text(json.dumps(result.document, indent=2), encoding="utf-8")
    print(json.dumps({**result.document, "report": str(out / _REPORT), "patch_file": str(out / _PATCH)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
