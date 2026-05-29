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
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..config import Config
from ..factory import build_reactor_components
from ..finalize.postmortem import finalize_session
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
    # Auto-probe knob lets hosts/tests opt out of the default
    # inference-server health probe without having to configure
    # ``health_probe_targets`` from scratch. Useful for the heartbeat
    # tests (which run on hosts with no inference server) and for
    # sandbox environments that audit health out-of-band.
    if "auto_probe_inference_server" in options:
        config.auto_probe_inference_server = bool(
            options["auto_probe_inference_server"]
        )
    # The ``ray status`` probe was introduced in PR #239 and runs on
    # every tick of the agent backend. Heartbeat tests (and any host
    # without an in-sandbox Ray head) need to disable it to keep the
    # default heartbeat envelope clean of false-positive
    # ``ray_head_dead`` alerts. The Coordinator wiring mirrors
    # ``auto_probe_*`` above.
    if "ray_probe_enabled" in options:
        config.ray_probe_enabled = bool(options["ray_probe_enabled"])
    # Whole-probe disable for the J ``external_deps`` signal (TraceLens
    # CLI / WekaFS mount). Heartbeat e2e tests run on inert CI hosts
    # where neither dependency is provisioned, so the default probe
    # fires ``tracelens_cli_missing`` + ``wekafs_degraded`` alerts that
    # otherwise mask the expected ``send_message{heartbeat}`` envelope.
    if "external_deps_enabled" in options:
        config.external_deps_enabled = bool(options["external_deps_enabled"])
    # B3 ``no_levers_found`` floor knobs let hosts override the
    # default 45 min / 8 tick observation window without forking the
    # whole Config. Multi-node large-model setups need a longer floor
    # because sglang cold start + baseline + profile + turnaround
    # alone consume 35-50 min before any explore family runs;
    # inference_optimizer's _build_robustness_options injects 60.0
    # when args.nodes >= 2 (single-node defaults stay at 45.0).
    if "progress_no_levers_min_minutes" in options:
        config.progress_no_levers_min_minutes = float(
            options["progress_no_levers_min_minutes"]
        )
    if "progress_no_levers_min_ticks" in options:
        config.progress_no_levers_min_ticks = int(
            options["progress_no_levers_min_ticks"]
        )

    # L4 — advertise our session_dir to co-deployed Critic processes so
    # their ``prepare-review`` can find ``agents/robustness/findings/
    # <session>.jsonl`` without explicit configuration. Setdefault keeps
    # an operator-supplied override intact.
    os.environ.setdefault(
        "ROBUSTNESS_AGENT_SESSION_DIR", str(config.session_dir),
    )

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


def _cmd_finalize(args: argparse.Namespace) -> None:
    """Run the L1+L2 postmortem finalizer as a one-shot operator tool.

    Use when the reactor never observed ``stop_reason`` going
    non-empty (e.g. Coordinator killed by SIGKILL before the wind-down
    intent landed). Idempotent — re-running has no effect once the
    ``.robustness_finalized`` marker exists, matching the in-reactor
    behaviour.
    """
    session_dir = Path(str(args.session_dir)).expanduser()
    if not session_dir.is_dir():
        raise RuntimeAdapterError(
            f"--session-dir does not point to a directory: {session_dir}"
        )
    session_id = (args.session_id or session_dir.name or "default").strip()
    stop_reason = (args.stop_reason or "manual_finalize").strip()
    wrote = finalize_session(
        session_dir,
        session_id=session_id,
        stop_reason=stop_reason,
    )
    payload = {
        "session_dir": str(session_dir),
        "session_id": session_id,
        "stop_reason": stop_reason,
        "wrote_new_files": bool(wrote),
        "reports_dir": str(session_dir / "reports"),
    }
    _emit_json(payload, args.out)


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

    finalize = sub.add_parser(
        "finalize",
        help=(
            "Run the L1+L2 postmortem finalizer post-hoc "
            "(for sessions whose Coordinator died before stop_reason "
            "was written)."
        ),
    )
    finalize.add_argument(
        "--session-dir",
        required=True,
        help="Path to the session directory to finalize.",
    )
    finalize.add_argument(
        "--session-id",
        default="",
        help="Session id (default: basename of --session-dir).",
    )
    finalize.add_argument(
        "--stop-reason",
        default="manual_finalize",
        help="stop_reason to record in the postmortem (default: manual_finalize).",
    )
    finalize.add_argument(
        "--out",
        default="-",
        help="Path to write the finalize summary JSON (default: stdout).",
    )
    finalize.set_defaults(func=_cmd_finalize)
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
