# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPU power-settings flags: ``--gpu-power-cap-w``, ``--gpu-perf-level``, ``--gpu-sclk-max-mhz`` and
``--apply-gpu-power-settings``."""

from __future__ import annotations

import argparse

from math import isfinite

from .constraint_args import positive_watts_arg


def _positive_mhz_arg(value: str) -> float:
    """A finite, positive clock frequency in MHz."""
    try:
        mhz = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a frequency in MHz, got {value!r}") from exc
    if not isfinite(mhz) or mhz <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive frequency in MHz, got {value!r}")
    return mhz


def add_gpu_power_arguments(opt: argparse._ActionsContainer) -> None:
    """Add the power-settings flags to the optimize command.

    By default the cap and perf level are assertions an operator satisfies before launch; the apply flag lets the
    session set them itself, on its masked cards only, for its whole lifetime.
    """
    opt.add_argument(
        "--gpu-power-cap-w",
        type=positive_watts_arg,
        default=None,
        help=(
            "Declare the power cap (W) the GPUs run at. By default an assertion: set it with "
            "`amd-smi set --power-cap` before launch, and the session refuses to start if any card it uses is at a "
            "different cap. With --apply-gpu-power-settings the session sets it itself. The observed cap is "
            "recorded whether or not this flag is passed."
        ),
    )
    opt.add_argument(
        "--gpu-perf-level",
        type=str,
        default=None,
        metavar="LEVEL",
        help=(
            "Declare the DPM performance level the GPUs run at (e.g. auto, high, determinism). Asserted like "
            "--gpu-power-cap-w, or set by the session with --apply-gpu-power-settings."
        ),
    )
    opt.add_argument(
        "--gpu-sclk-max-mhz",
        type=_positive_mhz_arg,
        default=None,
        metavar="MHZ",
        help=(
            "Declare the graphics clock (sclk) maximum the GPUs are limited to, in MHz. Asserted like "
            "--gpu-power-cap-w (set it with `amd-smi set --clk-limit sclk max MHZ` before launch), or set by the "
            "session with --apply-gpu-power-settings. A clock limit holds only at the manual perf level, so it "
            "refuses with any other --gpu-perf-level."
        ),
    )
    opt.add_argument(
        "--apply-gpu-power-settings",
        action="store_true",
        default=False,
        help=(
            "Permit the session to set --gpu-power-cap-w / --gpu-perf-level / --gpu-sclk-max-mhz on the cards of "
            "its integer ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES mask, once at launch and for the whole "
            "session; the "
            "optimizer never changes them afterwards. Needs amd-smi set privileges. The originals are recorded on "
            "host-local storage before anything is set and restored when the session ends, including on SIGTERM "
            "or SIGHUP; after a SIGKILL the cards stay at the applied settings until the next launch that passes "
            "this flag restores them. Refuses without an integer mask, on multi-node, and on a card that holds "
            "someone else's resident model or whose VRAM cannot be read, since a cap is card-wide, and on a "
            "starting perf level a restore could not bring back (determinism). Must be passed "
            "again on resume."
        ),
    )
