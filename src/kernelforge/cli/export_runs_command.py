# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge export-runs``: Kernel Forge's run logs as a document for ``hyperloom-kb-collect``."""

from __future__ import annotations

from pathlib import Path

import click


def _unreadable(error: OSError) -> str:
    """What to tell the user about a file under the log roots the export could not read."""

    message = f"cannot read {error.filename or 'a file under the log roots'}: {error.strerror or error}"
    if isinstance(error, PermissionError):
        message += (
            ". Files written in a container may be owned by root: fix their ownership, for example with chown, "
            "or move them out of the log root, then export again"
        )
    return message


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
    help=(
        "Where to write the JSON document. An empty raw-logs-not-collected.log is written beside it, "
        "replacing a file of that name already there."
    ),
)
def export_runs(log_roots: tuple[Path, ...], output: Path) -> None:
    """Write every run found under the LOG_ROOT directories as one JSON document.

    A run is a plain forge-loop run's forge_experiments directory. Beside the document it also writes
    raw-logs-not-collected.log, empty, and truncates a file of that name already there. Every Experience links that
    file as its raw logs: the global KB refuses a complete Experience without raw logs, and the export collects none.
    Collect the document with `hyperloom-kb-collect --mapping kernelforge-v1 --document <output>`.
    """

    from kernelforge.run_export.document import normalize, write_document

    try:
        document = normalize(log_roots)
    except OSError as error:
        raise click.ClickException(_unreadable(error)) from error
    write_document(document, output)
    click.echo(f"wrote {len(document['runs'])} runs to {output}")
