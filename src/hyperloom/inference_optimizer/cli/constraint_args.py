# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The KEEP-constraint flags: ``--max-latency-ms``, ``--max-power-w`` and ``--max-per-gpu-power-w``."""

from __future__ import annotations

import argparse
from math import isfinite


def _positive_ms_arg(value: str) -> float:
    """argparse type for a millisecond ceiling.

    The gate this feeds fails closed, so its switch must not fail open: an
    unusable value has to stop the launch rather than resolve to "no budget" and
    leave the operator believing an SLA is enforced.
    """
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive number of milliseconds, got {value!r}") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number of milliseconds, got {value!r}")
    return parsed


def positive_watts_arg(value: str) -> float:
    """argparse type for a power in watts; an unusable value stops the launch rather than skipping the check."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive number of watts, got {value!r}") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number of watts, got {value!r}")
    return parsed


def _non_negative_gpu_id_arg(value: str) -> int:
    """argparse type for a non-negative GPU id."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected an integer GPU id, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative GPU id, got {value!r}")
    return parsed


class _PerGpuPowerAction(argparse.Action):
    """Collect ``--max-per-gpu-power-w GPU_ID W`` pairs into ``{gpu_id: watts}``; a bad or repeated id stops the launch."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            limits = self._with(dict(getattr(namespace, self.dest, None) or {}), *values)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"{option_string}: {exc}")
        else:
            setattr(namespace, self.dest, limits)

    @staticmethod
    def _with(limits: dict[str, float], raw_id: str, raw_w: str) -> dict[str, float]:
        gpu = str(_non_negative_gpu_id_arg(raw_id))
        if gpu in limits:
            raise argparse.ArgumentTypeError(f"GPU {gpu} is given more than once")
        limits[gpu] = positive_watts_arg(raw_w)
        return limits


def add_constraint_arguments(opt: argparse._ActionsContainer) -> None:
    """Add the constraint flags to the optimize command.

    These sit outside the target group because they are constraints, not
    objectives: they say which winners are admissible rather than when to
    stop, so they compose with whichever target is in use.
    """
    opt.add_argument(
        "--max-latency-ms",
        type=_positive_ms_arg,
        default=None,
        help=(
            "Scriptable frameworks (xdit, custom) only. Refuse any KEEP whose mean "
            "end-to-end latency exceeds N ms. Off by default. A candidate that "
            "reported no end-to-end latency is refused too, since an unmeasured "
            "constraint is not a satisfied one."
        ),
    )
    opt.add_argument(
        "--max-power-w",
        type=positive_watts_arg,
        default=None,
        help=(
            "Refuse any KEEP whose serving GPUs together draw more than N W, summed over their mean power across "
            "the measured round (e.g. 2800 for four cards budgeted at 700 W each). Off by default. A candidate "
            "whose round reported no per-GPU power is refused too, since an unmeasured constraint is not a "
            "satisfied one, and the session refuses to start where GPU power cannot be sampled."
        ),
    )
    opt.add_argument(
        "--max-per-gpu-power-w",
        nargs=2,
        action=_PerGpuPowerAction,
        default=None,
        metavar=("GPU_ID", "W"),
        help=(
            "Refuse any KEEP in which GPU_ID (the physical amd-smi index) draws more than W watts, as its mean "
            "power over the measured round. Repeat once per card; a serving card with no entry is bound only by "
            "--max-power-w. The session refuses to start if a listed GPU is not one it can use, if the "
            "per-GPU limits add up to more than --max-power-w, or if GPU power cannot be sampled."
        ),
    )
