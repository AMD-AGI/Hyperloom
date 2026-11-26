# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.inspector``: the ``inspect`` subcommand without the optimizer CLI.

It needs the Hyperloom it ships with: it imports Hyperloom's own redaction (``hyperloom.common``) and price sheet
and LLM client (``hyperloom_common``), and no package the ``hyperloom`` command does not need. A source tree copied
into a pod therefore carries ``src/hyperloom/common`` and ``src/hyperloom_common`` beside ``src/hyperloom/inspector``
(and ``src/hyperloom/__init__.py``); with that, the Hyperloom installed in the pod is not used. This module and the
command line it parses import only the standard library and the Inspector, so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import sys

from hyperloom.inspector.cli import add_inspect_arguments, run_inspect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hyperloom.inspector",
        description="Serve the read-only Hyperloom Inspector for a session.",
    )
    add_inspect_arguments(parser)
    return run_inspect(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
