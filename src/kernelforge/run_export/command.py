# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge export-runs``: Kernel Forge's run logs as a document for ``hyperloom-kb-collect``."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.argument(
    "log_roots",
    metavar="LOG_ROOT...",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the JSON document.",
)
def export_runs(log_roots: tuple[Path, ...], output: Path) -> None:
    """Write every run found under the LOG_ROOT directories as one JSON document.

    A run is a plain forge-loop run's forge_experiments directory. Collect the document with
    `hyperloom-kb-collect --mapping kernelforge-v1 --document <output>`.
    """

    from kernelforge.run_export.document import normalize

    document = normalize(log_roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    click.echo(f"wrote {len(document['runs'])} runs to {output}")
