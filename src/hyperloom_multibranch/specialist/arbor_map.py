# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor's map, made by Arbor's own command lines in its checkout: generated from the root's profile with no GPU time
(:func:`generate_directions`), split into the shares the session's Arbor branches start on (:func:`split_directions`),
and, once a worker hands its share back, the slot's next order dealt from it by Arbor's master (:func:`deal_order`).
Each runs as Arbor's other commands with no site settings file, and raises :class:`SplitFailed` with what Arbor printed
when it cannot."""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.specialist.arbor_adapter import PYTHON, SETTINGS_ENV

#: The command lines of Arbor's map: ``arbor.directions`` splits and generates one, the master in ``arbor.roles``
#: deals from it.
DIRECTIONS_MODULE, ROLES_MODULE = "arbor.directions", "arbor.roles"
#: Where each of those command lines writes what it makes.
OUT_FLAG = "--out"
#: The lines of a command's output a failure keeps, its end.
_TAIL_LINES = 20


class SplitFailed(RuntimeError):
    """Arbor could not generate a map, split one into shares or deal an order; the message says what it printed."""


@dataclass(frozen=True)
class DealRequest:
    """What the master deals from: the session's map, the reports its workers handed back, the shares and orders
    running workers hold, and where the order goes (``out``), of at most ``size`` directions."""

    map_file: Path
    reports: tuple[Path, ...]
    held: tuple[Path, ...]
    out: Path
    size: int


async def split_directions(checkout: Path, map_file: Path, shares: int, out: Path) -> tuple[Path, ...]:
    """Split the open directions of ``map_file`` into ``shares`` disjoint plans in ``out``, with Arbor's own command
    line in ``checkout``; the plan files, in order, each a ``component.plan``. Raise :class:`SplitFailed` with what
    Arbor printed when it cannot: a map with fewer independent groups than shares, or one that does not validate."""

    paths = tuple(out / f"share-{number}.json" for number in range(1, shares + 1))
    await _arbor_cli(
        checkout, paths, DIRECTIONS_MODULE, "split", str(map_file), "--shares", str(shares), OUT_FLAG, str(out)
    )
    return paths


async def generate_directions(checkout: Path, profile: Path, out: Path) -> Path:
    """A map generated from the root profile ``profile``, the file ``component.profile`` names, by Arbor's own
    command line in ``checkout``, with no GPU time: one open direction per op family, ranked by its share. Written to
    ``out``, which :func:`split_directions` splits as any map. Raise :class:`SplitFailed` with what Arbor printed when
    it cannot: a profile with no share above its cut, or an Arbor older than ``generate``."""

    await _arbor_cli(checkout, (out,), DIRECTIONS_MODULE, "generate", "--profile", str(profile), OUT_FLAG, str(out))
    return out


async def deal_order(checkout: Path, deal: DealRequest) -> Path:
    """Have the master of the Arbor checkout ``checkout`` write the order ``deal.out``: at most ``deal.size`` open
    directions of ``deal.map_file`` that no report settled and no held share or order holds, as Arbor picks them
    (``python3 -m arbor.roles deal``, Arbor ``74933a2``). An order of no directions means nothing is left to deal.
    Raise :class:`SplitFailed` with what Arbor printed when it cannot."""

    args = ["deal", "--map", str(deal.map_file), "--size", str(deal.size)]
    if deal.reports:
        args += ["--reports", *(str(report) for report in deal.reports)]
    for held in deal.held:
        args += ["--held-order", str(held)]
    await _arbor_cli(checkout, (deal.out,), ROLES_MODULE, *args, OUT_FLAG, str(deal.out))
    return deal.out


async def _arbor_cli(checkout: Path, written: tuple[Path, ...], module: str, *args: str) -> None:
    """Run ``python3 -m <module>`` with ``args`` in ``checkout``, as Arbor's other commands with no site settings file
    (:data:`SETTINGS_ENV`); raise :class:`SplitFailed` with what it printed when it fails or leaves any of ``written``
    unwritten."""

    argv = [PYTHON, "-m", module, *args]
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=checkout,
        env={**os.environ, SETTINGS_ENV: os.devnull},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    said = " | ".join(line.strip() for line in output.decode(errors="replace").strip().splitlines()[-_TAIL_LINES:])
    if process.returncode != 0:
        raise SplitFailed(f"`{shlex.join(argv)}` exited {process.returncode}: {said}")
    missing = [path.name for path in written if not path.is_file()]
    if missing:
        raise SplitFailed(f"`{shlex.join(argv)}` wrote no {', '.join(missing)}: {said}")


__all__ = [
    "DealRequest",
    "SplitFailed",
    "deal_order",
    "generate_directions",
    "split_directions",
]
