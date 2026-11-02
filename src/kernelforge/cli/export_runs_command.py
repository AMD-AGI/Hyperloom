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
@click.option(
    "--collect",
    "collect_into_kb",
    is_flag=True,
    help=(
        "Then collect the document through the kernelforge-v1 mapping into the Experience KB HYPERLOOM_KB_URL names, "
        "as a run is collected as it ends. The receipt and the writes the KB has not taken yet are kept beside the "
        "document, and collecting a run again writes nothing new."
    ),
)
def export_runs(log_roots: tuple[Path, ...], output: Path, collect_into_kb: bool) -> None:
    """Write every run found under the LOG_ROOT directories as one JSON document.

    A run is a plain forge-loop run's forge_experiments directory. Beside the document it also writes
    raw-logs-not-collected.log, empty, and truncates a file of that name already there. Every Experience links that
    file as its raw logs: the global KB refuses a complete Experience without raw logs, and the export collects none.
    Collect the document with `hyperloom-kb-collect --mapping kernelforge-v1 --document <output>`, or with --collect.
    """

    from kernelforge.workflows.forge_loop.export.document import normalize, write_document

    try:
        document = normalize(log_roots)
    except OSError as error:
        raise click.ClickException(_unreadable(error)) from error
    write_document(document, output)
    click.echo(f"wrote {len(document['runs'])} runs to {output}")
    if collect_into_kb:
        _collect(output)


def _collect(document: Path) -> None:
    """Collect ``document`` into the configured Experience KB, failing the command when anything was not written."""

    from kernelforge.workflows.forge_loop.export.collect import (
        COLLECTION_ERRORS,
        RECEIPT,
        URL_ENV,
        collect_document,
        describe,
        push_to_global,
    )

    try:
        report = collect_document(document, document.parent)
    except COLLECTION_ERRORS as error:
        raise click.ClickException(f"collection failed: {type(error).__name__}: {error}") from error
    if report is None:
        raise click.ClickException(f"--collect needs the Experience KB {URL_ENV} names, and it is not set")
    click.echo(describe(report, document.parent / RECEIPT))
    if pushed := push_to_global():
        click.echo(pushed)
    if report.errors:
        raise click.ClickException(f"{len(report.errors)} candidates were not written; {RECEIPT} says why")
