"""Entry point for the Robustness Agent daemon."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from .agent import RobustnessAgent
from .config import Config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


async def _async_main() -> None:
    log = logging.getLogger("robustness_agent")

    config = await Config.discover()

    agent = RobustnessAgent(config)

    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        log.info("Received %s, shutting down", sig.name)
        loop.create_task(agent.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass

    await agent.run_forever()


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logging.getLogger("robustness_agent").info("KeyboardInterrupt, exiting")


if __name__ == "__main__":
    main()
