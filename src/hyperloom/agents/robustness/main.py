# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Entry point for the standalone Robustness Agent reactor."""

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
    """Configure root logging for the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


async def _run_reactor_mode(config: Config) -> None:
    """Run the standalone reactor loop for dev / debugging."""
    log = logging.getLogger("robustness_agent")
    bundle = build_reactor_components(config)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        """Signal handler that requests a graceful loop shutdown."""
        log.info("Received %s, shutting down", sig.name)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            # Signal handlers are unavailable on this platform/loop; skip them.
            pass

    log.info(
        "Reactor mode running tick=%.1fs session_dir=%s",
        config.standalone_tick_interval_s,
        config.session_dir,
    )

    try:
        while not stop.is_set():
            ctx = ReactorContext(
                tick_index=0,
                shared_state=SharedStateSnapshot(),
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
                # Tick interval elapsed with no stop request; loop again.
                pass
    finally:
        await bundle.aclose()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse daemon command-line arguments."""
    parser = argparse.ArgumentParser(prog="robustness-agent")
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    """Discover configuration and run the reactor loop."""
    _parse_args(argv)
    config = Config.discover()
    await _run_reactor_mode(config)


def main() -> None:
    """Synchronous process entry point for the daemon."""
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logging.getLogger("robustness_agent").info("KeyboardInterrupt, exiting")


if __name__ == "__main__":
    main()
