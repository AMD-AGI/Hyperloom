# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPU power-settings flags: ``--gpu-power-cap-w``, ``--gpu-perf-level``, ``--gpu-sclk-max-mhz``,
``--apply-gpu-power-settings``, ``--node-power-management`` and ``--node-power-limit-w``."""

from __future__ import annotations

import argparse

from hyperloom.common.gpu_power_settings import NPM_STATES, PERF_LEVELS

from .constraint_args import positive_watts_arg


def _positive_mhz_arg(value: str) -> int:
    """A positive whole number of MHz, the only clock limit ``amd-smi set -L`` takes."""
    try:
        mhz = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number of MHz, got {value!r}") from exc
    if mhz <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number of MHz, got {value!r}")
    return mhz


def add_gpu_power_arguments(opt: argparse._ActionsContainer) -> None:
    """Add the power-settings flags to the optimize command.

    By default every setting is an assertion an operator satisfies before launch; the apply flag lets the session set
    the cap and perf level itself, on its masked cards only, for its whole lifetime.
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
        type=str.lower,
        choices=PERF_LEVELS,
        default=None,
        metavar="LEVEL",
        help=(
            f"Declare the DPM performance level the GPUs run at, one of {', '.join(PERF_LEVELS)}. Asserted like "
            "--gpu-power-cap-w, or set by the session with --apply-gpu-power-settings."
        ),
    )
    opt.add_argument(
        "--gpu-sclk-max-mhz",
        type=_positive_mhz_arg,
        default=None,
        metavar="MHZ",
        help=(
            "Declare the graphics clock (sclk) maximum the GPUs are limited to, in MHz, as `amd-smi metric --clock` "
            "reports it. Always an assertion: set it with `amd-smi set -L sclk max MHZ` before launch; "
            "--apply-gpu-power-settings does not set it. A ceiling, not a locked clock: under load the cards may run "
            "below it. The observed limit is recorded whether or not this flag is passed."
        ),
    )
    opt.add_argument(
        "--apply-gpu-power-settings",
        action="store_true",
        default=False,
        help=(
            "Permit the session to set --gpu-power-cap-w / --gpu-perf-level on the cards of its integer "
            "ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES mask, once at launch and for the whole "
            "session; the "
            "optimizer never changes them afterwards. Needs root, as amd-smi set does. The originals are recorded on "
            "host-local storage before anything is set and restored when the session ends, including on SIGTERM "
            "or SIGHUP; after a SIGKILL the cards stay at the applied settings until the next launch that passes "
            "this flag with them in its mask restores them. Refuses without an integer mask, on multi-node, and on a card that holds "
            "someone else's resident model, whose VRAM cannot be read, or that is not in SPX compute-partition "
            "mode, since a cap is card-wide, and on a "
            "starting perf level a restore could not bring back (determinism or manual, when --gpu-perf-level is "
            "applied). "
            "Must be passed "
            "again on resume."
        ),
    )
    _add_node_power_arguments(opt)


def _add_node_power_arguments(opt: argparse._ActionsContainer) -> None:
    """The node power management flags: assertions only, since NPM is set on the BMC."""
    opt.add_argument(
        "--node-power-management",
        type=str.lower,
        choices=NPM_STATES,
        default=None,
        metavar="STATE",
        help=(
            "Declare whether Node Power Management (MI350 series) is enabled or disabled on this node, as "
            "`amd-smi node -p` reports it. Always an assertion: NPM is set on the BMC, never by the session, and "
            "--apply-gpu-power-settings does not touch it. The observed state is recorded whether or not this flag "
            "is passed."
        ),
    )
    opt.add_argument(
        "--node-power-limit-w",
        type=positive_watts_arg,
        default=None,
        help=(
            "Declare the node power limit (W) firmware enforces under NPM, as `amd-smi node -p` reports it; this can "
            "sit below the limit the BMC was asked for. Asserted like --node-power-management."
        ),
    )
