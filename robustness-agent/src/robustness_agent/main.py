# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Entry point for the standalone Robustness Agent reactor.

The package console script runs the symptom -> intent reactor in standalone
form, polling sources every ``standalone_tick_interval_s`` seconds and
writing findings to disk. Production hosts drive the same reactor via
:mod:`robustness_agent.runtime.cli` in a subprocess instead.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

from .config import Config
from .factory import build_reactor_components
from .role.prompt_inputs import ReactorContext, SharedStateSnapshot


def _setup_logging() -> None:
    """Configure root logging for the daemon.

    Sets up a basic stderr handler at INFO level with a timestamped
    format shared by both run modes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


async def _run_reactor_mode(config: Config) -> None:
    """Standalone reactor loop for dev / debugging.

    Polls the configured sources at ``standalone_tick_interval_s`` and
    writes findings to disk. Coordinator integration drives the same
    reactor through ``robustness_agent.runtime.cli tick`` in a
    subprocess (mirroring critic-agent's transport).

    Args:
        config (Config): Resolved agent configuration controlling the
            tick interval, server URL, and session directory.
    """
    log = logging.getLogger("robustness_agent")
    bundle = build_reactor_components(config)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        """Signal handler that requests a graceful loop shutdown.

        Args:
            sig (signal.Signals): The received signal triggering shutdown.
        """
        log.info("Received %s, shutting down", sig.name)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass

    log.info(
        "Reactor mode running tick=%.1fs server=%s session_dir=%s",
        config.standalone_tick_interval_s,
        config.robustness_server_url or "(local-only)",
        config.session_dir,
    )

    try:
        while not stop.is_set():
            ctx = ReactorContext(
                tick_index=0,
                shared_state=SharedStateSnapshot(session_id=config.session_dir.name),
                inbox=[],
                now_unix=time.time(),
            )
            try:
                intents = await bundle.reactor.tick(ctx)
                log.debug("tick=%d emitted %d intents", bundle.reactor.tick_index, len(intents))
            except Exception:
                log.exception("standalone reactor tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=config.standalone_tick_interval_s)
            except asyncio.TimeoutError:
                pass
    finally:
        await bundle.aclose()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse daemon command-line arguments.

    Args:
        argv (list[str] | None): Argument vector to parse. Defaults to
            ``None``, which uses ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments including the selected
        ``mode``.
    """
    parser = argparse.ArgumentParser(prog="robustness-agent")
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    """Discover configuration and run the selected mode.

    Args:
        argv (list[str] | None): Argument vector forwarded to
            :func:`_parse_args`. Defaults to ``None``.
    """
    args = _parse_args(argv)
    config = await Config.discover()
    await _run_reactor_mode(config)


def main() -> None:
    """Synchronous process entry point for the daemon.

    Configures logging and runs the async main loop, treating a
    keyboard interrupt as a clean exit.
    """
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logging.getLogger("robustness_agent").info("KeyboardInterrupt, exiting")


if __name__ == "__main__":
    main()
