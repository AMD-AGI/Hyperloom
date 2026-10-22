# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge forge-dashboard``: serve a campaign's dashboard and take operator messages from it."""

from __future__ import annotations

import click


@click.command("forge-dashboard")
@click.option(
    "--workspace",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Workspace of a forge-loop campaign, running or not",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to listen on")
@click.option("--port", default=8765, show_default=True, type=click.IntRange(min=0), help="Port to listen on")
def forge_dashboard(workspace: str, host: str, port: int) -> None:
    """Serve the dashboard of the campaign at --workspace, rendered fresh on every visit.

    Its form posts operator messages into the campaign's inbox, as forge-say does. Stop with Ctrl-C.
    """
    from kernelforge.loop.dashboard.server import make_server

    server = make_server(workspace, host, port)
    click.echo(f"dashboard: http://{host}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
