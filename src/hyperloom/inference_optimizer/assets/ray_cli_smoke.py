# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Certify the Ray an interpreter has: the version preflight wants, and a CLI whose `ray start` parses Hyperloom's argv.

Usage: ``ray_cli_smoke.py MIN_VERSION CLICK_HOLD_VERSION CLICK_HOLD_EXIT START_ARG... RAY_VERSION``. An empty
RAY_VERSION accepts any Ray at or above MIN_VERSION. Prints the Ray version and exits 0 when certified. Exits
CLICK_HOLD_EXIT only when CLICK_HOLD_VERSION's CLI fails to import with click's Sentinel error, the one failure
a click hold repairs; every other failure exits 1.
"""

from __future__ import annotations

import re
import sys


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def _version_refusal(installed: str, wanted: str, floor: str) -> str | None:
    if wanted:
        return None if installed == wanted else f"ray version mismatch: {installed} != {wanted}"
    if _version_tuple(installed) < _version_tuple(floor):
        return f"ray too old: {installed} < {floor}"
    return None


def main(argv: list[str]) -> int:
    floor, hold_version, hold_exit, *start_argv, wanted = argv
    try:
        import ray
    except Exception as exc:  # noqa: BLE001 - any import failure means this interpreter has no usable Ray
        print(f"ray import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    refusal = _version_refusal(ray.__version__, wanted, floor)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    try:
        import click
        from ray.scripts.scripts import cli as ray_cli  # `main` only wraps this group
    except Exception as exc:  # noqa: BLE001 - the Sentinel error is a ValueError raised while Ray builds its click group
        print(f"ray CLI import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sentinel = "is not a valid Sentinel" in str(exc) and ray.__version__ == hold_version
        return int(hold_exit) if sentinel else 1
    try:
        ray_cli.commands["start"].make_context("start", list(start_argv), resilient_parsing=False).close()
    except click.UsageError as exc:
        print(f"ray start does not accept {' '.join(start_argv)}: {exc.format_message()}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a missing or broken `start` command is a refusal, reported with its cause
        print(f"ray CLI has no usable start command: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(ray.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
