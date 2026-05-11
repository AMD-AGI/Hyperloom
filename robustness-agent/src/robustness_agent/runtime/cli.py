"""Robustness runtime CLI.

Hosts (Coordinator, smoke harness, operator tooling) drive the reactor
through a single subprocess command::

    python -m robustness_agent.runtime.cli tick \\
        --request request.json [--out emit.json]

``request.json`` carries the same fields the critic-agent runtime
accepts on its ``coordinator_inbox`` shape::

    {
        "kind": "coordinator_inbox",
        "session_id": "<session id>",
        "raw_prompt": "=== Shared session state ===\\n...",
        "context":   {"tick_index": 0, "now_unix": 1700000000.0},
        "options":   {"session_dir": "/tmp/sess-1",
                      "robustness_server_url": "http://...",
                      "llm_rca_enabled": false,
                      "metrics_window_s": 300}
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

Exit codes (mirror critic-agent contract §6):
    0 — logical success (zero or more intents emitted)
    2 — adapter / configuration bug (caller should treat as fatal)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ..config import Config
from ..factory import build_reactor_components
from ..role.envelope import build_envelope_dict
from ..role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
    from_coordinator_prompt,
)


log = logging.getLogger("robustness_agent.runtime.cli")


COORDINATOR_INBOX = "coordinator_inbox"
REQUEST_KINDS: frozenset[str] = frozenset({COORDINATOR_INBOX})


class RuntimeAdapterError(RuntimeError):
    """Raised on contract violations the host should surface as exit 2."""


def _read_json(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def _emit_json(obj: Any, out: str | None) -> None:
    serialised = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        Path(out).write_text(serialised + "\n", encoding="utf-8")
    sys.stdout.write(serialised + "\n")
    sys.stdout.flush()


def _coerce_request(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeAdapterError(
            f"request top-level must be an object, got {type(raw).__name__}"
        )
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise RuntimeAdapterError("request.kind missing or empty")
    if kind not in REQUEST_KINDS:
        raise RuntimeAdapterError(
            f"request.kind={kind!r} not in {sorted(REQUEST_KINDS)!r}"
        )
    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise RuntimeAdapterError("request.session_id must be non-empty string")
    raw_prompt = raw.get("raw_prompt")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise RuntimeAdapterError(
            "coordinator_inbox: raw_prompt must be a non-empty string"
        )
    context = raw.get("context")
    if context is not None and not isinstance(context, dict):
        raise RuntimeAdapterError(
            f"request.context must be an object when present, got {type(context).__name__}"
        )
    options = raw.get("options")
    if options is not None and not isinstance(options, dict):
        raise RuntimeAdapterError(
            f"request.options must be an object when present, got {type(options).__name__}"
        )
    return raw


async def _run_tick(request: dict[str, Any]) -> dict[str, Any]:
    """Drive a single reactor tick from a normalised ``request`` dict."""
    session_id = str(request["session_id"]).strip()
    raw_prompt = str(request["raw_prompt"])
    context = dict(request.get("context") or {})
    options = dict(request.get("options") or {})

    config = await Config.discover()
    if "session_dir" in options:
        config.session_dir = Path(str(options["session_dir"]))
    if "robustness_server_url" in options:
        config.robustness_server_url = str(options["robustness_server_url"] or "")
    if "llm_rca_enabled" in options:
        config.llm_rca_enabled = bool(options["llm_rca_enabled"])
    if "metrics_window_s" in options:
        config.metrics_window_s = int(options["metrics_window_s"])

    tick_index_raw = context.get("tick_index", 0)
    tick_index = int(tick_index_raw) if isinstance(tick_index_raw, (int, float)) else 0
    now_unix_raw = context.get("now_unix")
    now_unix = (
        float(now_unix_raw)
        if isinstance(now_unix_raw, (int, float))
        else time.time()
    )

    reactor_ctx = from_coordinator_prompt(
        raw_prompt, tick_index=tick_index, now_unix=now_unix,
    )
    if not reactor_ctx.shared_state.session_id:
        reactor_ctx = ReactorContext(
            tick_index=reactor_ctx.tick_index,
            shared_state=SharedStateSnapshot(
                session_id=session_id,
                model_name=reactor_ctx.shared_state.model_name,
                model_class=reactor_ctx.shared_state.model_class,
                baseline_tput=reactor_ctx.shared_state.baseline_tput,
                cumulative_gain=reactor_ctx.shared_state.cumulative_gain,
                crash_count=reactor_ctx.shared_state.crash_count,
                current_action=reactor_ctx.shared_state.current_action,
            ),
            inbox=list(reactor_ctx.inbox),
            now_unix=reactor_ctx.now_unix,
            parse_warnings=list(reactor_ctx.parse_warnings),
        )

    bundle = build_reactor_components(config, session_id=session_id)
    try:
        intents = await bundle.reactor.tick(reactor_ctx)
    finally:
        await bundle.aclose()

    return {
        "intent_envelope": build_envelope_dict(intents),
        "session_id": session_id,
        "tick_index": bundle.reactor.tick_index,
        "parse_warnings": list(reactor_ctx.parse_warnings),
    }


def _cmd_tick(args: argparse.Namespace) -> None:
    request = _coerce_request(_read_json(args.request))
    emit = asyncio.run(_run_tick(request))
    _emit_json(emit, args.out)


def _build_parser() -> argparse.ArgumentParser:
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
