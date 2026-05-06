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


def main() -> None:
    _setup_logging()
    log = logging.getLogger("robustness_agent")

    config = Config.from_env()
    log.info("Config: session_dir=%s robust_url=%s",
             config.session_dir, config.robust_analyzer_url or "(local mode)")

    agent = RobustnessAgent(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(sig: signal.Signals) -> None:
        log.info("Received %s, shutting down", sig.name)
        loop.create_task(agent.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(agent.run_forever())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt, shutting down")
        loop.run_until_complete(agent.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
