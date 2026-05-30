"""Entry point for the Robustness Agent daemon.

Two run modes:

* ``--mode reactor`` (default, M1): runs the new symptom -> intent
  pipeline in standalone form, polling sources every
  ``standalone_tick_interval_s`` seconds and writing findings to disk.
  This is mainly used for dev / smoke testing; production hosts (the
  Coordinator) drive the reactor via :mod:`robustness_agent.runtime.cli`
  in a subprocess instead.
* ``--mode legacy``: keeps the previous RobustnessAgent loop available
  for environments that still depend on direct conductor.db writes.
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
    """
    log = logging.getLogger("robustness_agent")
    bundle = build_reactor_components(config)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
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


async def _run_legacy_mode(config: Config) -> None:
    """Original event-driven RobustnessAgent loop, kept for compatibility."""
    from .agent import RobustnessAgent

    log = logging.getLogger("robustness_agent")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="robustness-agent")
    parser.add_argument(
        "--mode",
        choices=("reactor", "legacy"),
        default="reactor",
        help="reactor: M1 standalone loop (default); legacy: previous agent.py loop",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = await Config.discover()
    if args.mode == "reactor":
        await _run_reactor_mode(config)
    else:
        await _run_legacy_mode(config)


def main() -> None:
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logging.getLogger("robustness_agent").info("KeyboardInterrupt, exiting")


if __name__ == "__main__":
    main()
