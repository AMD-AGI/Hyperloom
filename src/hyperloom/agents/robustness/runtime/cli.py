# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness runtime CLI.

Hosts (Coordinator, smoke harness, operator tooling) drive the reactor
through a single subprocess command::

    python -m hyperloom.agents.robustness.runtime.cli tick \\
        --request request.json [--out emit.json]

``request.json`` carries the same fields the critic-agent runtime
accepts on its ``coordinator_inbox`` shape::

    {
        "kind": "coordinator_inbox",
        "session_id": "<session id>",
        "raw_prompt": "=== Shared session state ===\\n...",
        "context":   {"tick_index": 0, "now_unix": 1700000000.0},
        "options":   {"session_dir": "/tmp/sess-1",
                      "llm_rca_enabled": false,
                      "disable_local_probe": false}
    }

``raw_prompt`` is parsed by :func:`from_coordinator_prompt`; ``context``
provides a deterministic ``tick_index`` / ``now_unix`` for repeatable
host-driven ticks; ``options`` are non-default :class:`Config` overrides
the host wants to apply (env-var equivalents are honoured for
production paths).

``emit.json`` carries the validated envelope plus per-tick metadata so
hosts can audit what each tick did without rerunning the reactor::

    {
        "intent_envelope": {"intents": [{"intent_type": "alert",
                                         "payload": {...}}, ...]},
        "session_id":   "<session id>",
        "tick_index":   <int>,
        "parse_warnings": ["..."]
    }

Exit codes (mirror critic-agent contract):
    0 — logical success (zero or more intents emitted)
    2 — adapter / configuration bug (caller should treat as fatal)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from hyperloom.common.llm_attribution import set_current_phase
from hyperloom.common.subprocess_bridge import RuntimeAdapterError, emit_json, read_json

from ..config import Config
from ..factory import build_reactor_components
from ..role.envelope import build_envelope_dict
from ..role.prompt_inputs import from_coordinator_prompt


log = logging.getLogger("robustness_agent.runtime.cli")


COORDINATOR_INBOX = "coordinator_inbox"
REQUEST_KINDS: frozenset[str] = frozenset({COORDINATOR_INBOX})


def _coerce_request(raw: Any) -> dict[str, Any]:
    """Validate and normalise a raw request payload.

    Enforces the ``coordinator_inbox`` contract: the payload must be an
    object with a recognised ``kind``, a non-empty ``session_id`` and
    ``raw_prompt``, and optional object-typed ``context`` / ``options``.

    Args:
        raw (Any): The parsed request payload to validate.

    Returns:
        dict[str, Any]: The validated request dictionary (returned
        unchanged).

    Raises:
        RuntimeAdapterError: If any required field is missing or has the
            wrong type.
    """
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(f"request top-level must be an object, got {type(raw).__name__}")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise RuntimeAdapterError("request.kind missing or empty")
    if kind not in REQUEST_KINDS:
        raise RuntimeAdapterError(f"request.kind={kind!r} not in {sorted(REQUEST_KINDS)!r}")
    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise RuntimeAdapterError("request.session_id must be non-empty string")
    raw_prompt = raw.get("raw_prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise RuntimeAdapterError("coordinator_inbox: raw_prompt must be a non-empty string")
    context = raw.get("context")
    if context is not None and not isinstance(context, dict):
        raise RuntimeAdapterError(f"request.context must be an object when present, got {type(context).__name__}")
    options = raw.get("options")
    if options is not None and not isinstance(options, dict):
        raise RuntimeAdapterError(f"request.options must be an object when present, got {type(options).__name__}")
    return raw


async def _run_tick(request: dict[str, Any]) -> dict[str, Any]:
    """Drive a single reactor tick from a normalised ``request`` dict.

    Discovers configuration, applies any host-supplied ``options``
    overrides, builds the reactor components, runs one tick, and returns
    the resulting emit payload.

    Args:
        request (dict[str, Any]): Validated request dict carrying
            ``session_id``, ``raw_prompt``, and optional ``context`` /
            ``options``.

    Returns:
        dict[str, Any]: Emit payload with ``intent_envelope``,
        ``session_id``, ``tick_index``, and ``parse_warnings``.
    """
    session_id = str(request["session_id"]).strip()
    raw_prompt = str(request["raw_prompt"])
    context = dict(request.get("context") or {})
    options = dict(request.get("options") or {})

    config = Config.discover()
    if "session_dir" in options:
        config.session_dir = Path(str(options["session_dir"]))
    if "llm_rca_enabled" in options:
        config.llm_rca_enabled = bool(options["llm_rca_enabled"])
    if "disable_local_probe" in options:
        config.disable_local_probe = bool(options["disable_local_probe"])
    if "nodes" in options:
        try:
            config.nodes = max(1, int(options["nodes"]))
        except (TypeError, ValueError):
            # Invalid --nodes value; keep the existing default.
            pass
    # Opt out of the default inference-server health probe.
    if "auto_probe_inference_server" in options:
        config.auto_probe_inference_server = bool(options["auto_probe_inference_server"])
    # Disable the per-tick ``ray status`` probe on hosts without a Ray head.
    if "ray_probe_enabled" in options:
        config.ray_probe_enabled = bool(options["ray_probe_enabled"])
    # Disable the ``external_deps`` probe (TraceLens CLI / WekaFS mount) on inert CI hosts.
    if "external_deps_enabled" in options:
        config.external_deps_enabled = bool(options["external_deps_enabled"])
    # ``no_levers_found`` floor knobs override the default window.
    if "progress_no_levers_min_minutes" in options:
        config.progress_no_levers_min_minutes = float(options["progress_no_levers_min_minutes"])
    if "progress_no_levers_min_ticks" in options:
        config.progress_no_levers_min_ticks = int(options["progress_no_levers_min_ticks"])

    # Advertise session_dir so co-deployed Critic ``prepare-review`` finds the findings jsonl.
    os.environ.setdefault(
        "ROBUSTNESS_AGENT_SESSION_DIR",
        str(config.session_dir),
    )

    tick_index_raw = context.get("tick_index", 0)
    tick_index = int(tick_index_raw) if isinstance(tick_index_raw, (int, float)) else 0
    now_unix_raw = context.get("now_unix")
    now_unix = float(now_unix_raw) if isinstance(now_unix_raw, (int, float)) else time.time()

    reactor_ctx = from_coordinator_prompt(
        raw_prompt,
        tick_index=tick_index,
        now_unix=now_unix,
    )

    # This runs one process below the orchestrator, where the published phase is
    # empty, so the RCA calls this tick makes would reach the gateway with no
    # phase on them. The Coordinator prompt carries the phase for exactly this
    # tick, which makes it the only honest value here -- and the reactor is
    # re-entered per tick, so re-publishing tracks a run that moves on. An
    # absent block parses to "", which suppresses the field rather than pinning
    # a stale one.
    set_current_phase(reactor_ctx.phase)

    bundle = build_reactor_components(config, session_id=session_id)
    try:
        intents = await bundle.reactor.tick(reactor_ctx)
        # Surface any RCA-LLM token spend this tick for the host's trace ledger.
        llm_usage = None
        rca = getattr(bundle.components, "rca", None)
        drain = getattr(rca, "drain_usage", None)
        if callable(drain):
            try:
                llm_usage = drain()
            except Exception:  # noqa: BLE001 — telemetry must never fail a tick
                llm_usage = None
    finally:
        await bundle.aclose()

    emit: dict[str, Any] = {
        "intent_envelope": build_envelope_dict(intents),
        "session_id": session_id,
        "tick_index": bundle.reactor.tick_index,
        "parse_warnings": list(reactor_ctx.parse_warnings),
    }
    if llm_usage:
        emit["llm_usage"] = llm_usage
    return emit


def _cmd_tick(args: argparse.Namespace) -> None:
    """Handle the ``tick`` subcommand.

    Reads and validates the request file, runs a single reactor tick,
    and emits the resulting envelope JSON.

    Args:
        args (argparse.Namespace): Parsed CLI arguments with ``request``
            and ``out`` attributes.
    """
    request = _coerce_request(read_json(args.request))
    emit = asyncio.run(_run_tick(request))
    emit_json(emit, args.out)


def _build_parser() -> argparse.ArgumentParser:
    """Build the runtime CLI argument parser.

    Returns:
        argparse.ArgumentParser: The parser carrying the ``tick`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="robustness-agent-runtime",
        description="Robustness reactor runtime CLI (subprocess transport).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tick = sub.add_parser(
        "tick",
        help="Run one reactor tick and emit an intent envelope JSON.",
    )
    tick.add_argument(
        "--request",
        required=True,
        help="Path to a JSON file containing the per-tick request.",
    )
    tick.add_argument(
        "--out",
        default="-",
        help="Path to write the emit JSON (default: stdout).",
    )
    tick.set_defaults(func=_cmd_tick)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runtime CLI process entry point.

    Configures logging, parses arguments, dispatches to the selected
    subcommand handler, and maps failures to the contract exit codes.

    Args:
        argv (list[str] | None): Argument vector to parse. Defaults to
            ``None``, which uses ``sys.argv``.

    Returns:
        int: ``0`` on logical success, ``2`` on adapter/configuration
        errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeAdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.exception("robustness-agent runtime CLI failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
