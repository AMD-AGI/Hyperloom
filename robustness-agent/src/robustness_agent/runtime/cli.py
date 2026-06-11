# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    """Read and parse a JSON file.

    Args:
        path (str | Path): Path to the JSON file to read.

    Returns:
        Any: The parsed JSON value, or ``None`` if the file is empty or
        contains only whitespace.
    """
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def _emit_json(obj: Any, out: str | None) -> None:
    """Serialise an object to JSON and emit it.

    The serialised JSON is always written to stdout; when ``out`` is a
    real path it is additionally written to that file.

    Args:
        obj (Any): JSON-serialisable object to emit.
        out (str | None): Destination path, or ``"-"``/``None`` to write
            only to stdout.
    """
    serialised = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        Path(out).write_text(serialised + "\n", encoding="utf-8")
    sys.stdout.write(serialised + "\n")
    sys.stdout.flush()


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

    config = await Config.discover()
    if "session_dir" in options:
        config.session_dir = Path(str(options["session_dir"]))
    if "robustness_server_url" in options:
        config.robustness_server_url = str(options["robustness_server_url"] or "")
    if "llm_rca_enabled" in options:
        config.llm_rca_enabled = bool(options["llm_rca_enabled"])
    if "metrics_window_s" in options:
        config.metrics_window_s = int(options["metrics_window_s"])
    if "disable_local_probe" in options:
        config.disable_local_probe = bool(options["disable_local_probe"])
    if "enable_cluster_pod_metrics" in options:
        config.enable_cluster_pod_metrics = bool(options["enable_cluster_pod_metrics"])
    if "pod_metrics_categories" in options:
        raw_cats = options["pod_metrics_categories"]
        if isinstance(raw_cats, str):
            cats = tuple(
                part.strip() for part in raw_cats.split(",") if part.strip()
            )
        elif isinstance(raw_cats, (list, tuple)):
            cats = tuple(str(c).strip() for c in raw_cats if str(c).strip())
        else:
            cats = ()
        if cats:
            config.pod_metrics_categories = cats
    if "workload_uid" in options:
        config.workload_uid = str(options["workload_uid"] or "")
    if "nodes" in options:
        try:
            config.nodes = max(1, int(options["nodes"]))
        except (TypeError, ValueError):
            pass
    # Opt out of the default inference-server health probe (heartbeat tests /
    # sandboxes auditing health out-of-band) without reconfiguring targets.
    if "auto_probe_inference_server" in options:
        config.auto_probe_inference_server = bool(
            options["auto_probe_inference_server"]
        )
    # Disable the per-tick ``ray status`` probe on hosts without a Ray head to
    # avoid false-positive ``ray_head_dead`` alerts.
    if "ray_probe_enabled" in options:
        config.ray_probe_enabled = bool(options["ray_probe_enabled"])
    # Disable the ``external_deps`` probe (TraceLens CLI / WekaFS mount) on inert
    # CI hosts that would otherwise fire ``tracelens_cli_missing`` / ``wekafs_degraded``.
    if "external_deps_enabled" in options:
        config.external_deps_enabled = bool(options["external_deps_enabled"])
    # B3 ``no_levers_found`` floor knobs override the default 45 min / 8 tick window;
    # multi-node setups inject 60.0 (single-node stays 45.0) since cold start +
    # baseline + profile consume 35-50 min before any explore family runs.
    if "progress_no_levers_min_minutes" in options:
        config.progress_no_levers_min_minutes = float(
            options["progress_no_levers_min_minutes"]
        )
    if "progress_no_levers_min_ticks" in options:
        config.progress_no_levers_min_ticks = int(
            options["progress_no_levers_min_ticks"]
        )

    # L4 — advertise session_dir so co-deployed Critic ``prepare-review`` finds
    # the findings jsonl; setdefault keeps an operator override intact.
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
    """Handle the ``tick`` subcommand.

    Reads and validates the request file, runs a single reactor tick,
    and emits the resulting envelope JSON.

    Args:
        args (argparse.Namespace): Parsed CLI arguments with ``request``
            and ``out`` attributes.
    """
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

    Args:
        args (argparse.Namespace): Parsed CLI arguments with
            ``session_dir``, ``session_id``, ``stop_reason``, and ``out``
            attributes.

    Raises:
        RuntimeAdapterError: If ``--session-dir`` does not point to an
            existing directory.
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
    """Build the runtime CLI argument parser.

    Registers the ``tick`` and ``finalize`` subcommands with their
    respective options and handler functions.

    Returns:
        argparse.ArgumentParser: The configured parser.
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
