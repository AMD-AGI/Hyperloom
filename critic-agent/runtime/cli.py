"""Critic runtime CLI.

The Critic SKILL shells out to this module via:

```
python -m runtime.cli init-session     --request request.json
python -m runtime.cli prepare-review   --request request.json [--out judge.json]
python -m runtime.cli commit-review    --request request.json --review review.json [--out emit.json]
python -m runtime.cli close-session    --request request.json [--kb-draft draft.json]

# Low-level KB ops (kept for backward compat / tooling).
python -m runtime.cli list-priors      --packet packet.json [--kind ...] [--topic ...]
python -m runtime.cli write-verdict    --packet packet.json --verdict verdict.json --ctx ctx.json
python -m runtime.cli write-kb-drafts  --packet packet.json --kb-draft kb_draft.json --ctx ctx.json
python -m runtime.cli add-contradiction --new-id ID --old-ids id1,id2 --ctx ctx.json
python -m runtime.cli replay-dead-letter [--dir DIR] [--keep-on-success]
```

Every command writes a single JSON object to stdout (or to ``--out``)
and returns exit code 0 on logical success — including
``dead_lettered`` outcomes per contract §6. Exit code 2 is reserved for
adapter bugs that should propagate back to the SKILL caller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .dead_letter import DeadLetter
from .decision_reviewer import DecisionReviewer
from .errors import RuntimeAdapterError
from .in_memory_kb_client import InMemoryKBClient
from .kb_client import HTTPKBClient, KBClient
from .kb_writer import KBWriter, WriteContext
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory


# ---------------------------------------------------------------------------
def _read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file, returning ``None`` for an empty file.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        Any: The decoded JSON value, or ``None`` if the file is blank.
    """
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def _emit_json(obj: Any, out: str | None) -> None:
    """Serialise ``obj`` to JSON, writing to a file and/or stdout.

    Always writes to stdout; additionally writes to ``out`` when it is a path
    other than ``"-"``.

    Args:
        obj (Any): A JSON-serialisable value.
        out (str | None): Output path, ``"-"``/``None`` for stdout only.
    """
    serialised = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        Path(out).write_text(serialised + "\n", encoding="utf-8")
    sys.stdout.write(serialised + "\n")
    sys.stdout.flush()


def _resolve_kb_client() -> KBClient:
    """Build the KB client selected by ``CRITIC_KB_CLIENT_MODE``.

    ``live`` builds an :class:`HTTPKBClient` from the ``KB_*`` environment
    variables; any other value yields an :class:`InMemoryKBClient`.

    Returns:
        KBClient: The resolved KB client.

    Raises:
        RuntimeAdapterError: If ``CRITIC_KB_CLIENT_MODE=live`` but
            ``KB_BASE_URL`` is unset.
    """
    mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if mode == "live":
        base_url = os.environ.get("KB_BASE_URL")
        if not base_url:
            raise RuntimeAdapterError(
                "CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not set"
            )
        return HTTPKBClient(
            base_url=base_url,
            timeout_ms=int(os.environ.get("KB_TIMEOUT_MS", "10000")),
            retry_max=int(os.environ.get("KB_RETRY_MAX", "3")),
            token=os.environ.get("KB_SERVICE_TOKEN"),
        )
    return InMemoryKBClient()


def _resolve_reviewer() -> DecisionReviewer:
    """Build a fully wired :class:`DecisionReviewer` for CLI commands.

    Returns:
        DecisionReviewer: A reviewer backed by a fresh ``SessionMemory``, the
        env-selected KB client and a matching ``KBWriter``.
    """
    sm = SessionMemory()
    client = _resolve_kb_client()
    writer = KBWriter(client, session_memory=sm)
    return DecisionReviewer(session_memory=sm, kb_client=client, kb_writer=writer)


# ---------------------------------------------------------------------------
def _cmd_init_session(args: argparse.Namespace) -> None:
    """Handle ``init-session``: merge a request's context and emit it.

    Args:
        args (argparse.Namespace): Parsed CLI args (``request``, ``out``).
    """
    request = _read_json(args.request)
    reviewer = _resolve_reviewer()
    out = reviewer.init_session(request)
    _emit_json(out, args.out)


def _cmd_prepare_review(args: argparse.Namespace) -> None:
    """Handle ``prepare-review``: emit the phase-1 judge bundle.

    Args:
        args (argparse.Namespace): Parsed CLI args (``request``, ``out``).
    """
    request = _read_json(args.request)
    reviewer = _resolve_reviewer()
    bundle = reviewer.prepare_review(request)
    _emit_json(bundle.to_dict(), args.out)


def _cmd_commit_review(args: argparse.Namespace) -> None:
    """Handle ``commit-review``: validate a review and emit the outcome.

    Args:
        args (argparse.Namespace): Parsed CLI args (``request``, ``review``,
            ``out``).

    Raises:
        RuntimeAdapterError: If ``--review`` is not a JSON object.
    """
    request = _read_json(args.request)
    review = _read_json(args.review)
    if not isinstance(review, dict):
        raise RuntimeAdapterError("--review must be a JSON object")
    reviewer = _resolve_reviewer()
    outcome = reviewer.commit_review(request, review)
    _emit_json(outcome.to_dict(), args.out)


def _cmd_close_session(args: argparse.Namespace) -> None:
    """Handle ``close-session``: close a session, optionally flushing drafts.

    Args:
        args (argparse.Namespace): Parsed CLI args (``request``, ``kb_draft``,
            ``out``).
    """
    request = _read_json(args.request)
    kb_draft = _read_json(args.kb_draft) if args.kb_draft else None
    reviewer = _resolve_reviewer()
    outcome = reviewer.close_session(request, kb_draft)
    _emit_json(outcome.to_dict(), args.out)


# ---------------------------------------------------------------------------
def _cmd_list_priors(args: argparse.Namespace) -> None:
    """Handle ``list-priors``: look up KB priors for a packet's scope.

    Args:
        args (argparse.Namespace): Parsed CLI args (``packet``, ``kind``,
            ``topic``, ``limit``, ``session``, ``out``).
    """
    packet = _read_json(args.packet) or {}
    context = packet.get("context") or packet.get("environment") or {}
    scope = build_scope(context, require_critical=False)
    scope_filter = {k: v for k, v in scope.items() if v != "unknown"}
    client = _resolve_kb_client()
    writer = KBWriter(client)
    priors = writer.list_priors(
        scope=scope_filter,
        kind=args.kind,
        topic=args.topic,
        limit=args.limit,
        ctx=WriteContext(session_id=args.session or "cli", review_id="cli"),
    )
    priors["scope_cache_key"] = scope_cache_key(scope_filter, topic=args.topic)
    _emit_json(priors, args.out)


def _cmd_write_verdict(args: argparse.Namespace) -> None:
    """Handle ``write-verdict``: write a single verdict lesson to KB.

    Args:
        args (argparse.Namespace): Parsed CLI args (``packet``, ``verdict``,
            ``ctx``, ``out``).
    """
    packet = _read_json(args.packet) or {}
    verdict = _read_json(args.verdict) or {}
    ctx_raw = _read_json(args.ctx) or {}
    client = _resolve_kb_client()
    writer = KBWriter(client)
    ctx = WriteContext(
        session_id=ctx_raw.get("session_id") or "cli",
        review_id=ctx_raw.get("review_id"),
        source_role=ctx_raw.get("source_role", "critic"),
        source_type=ctx_raw.get("source_type", "critic_decision_review"),
        topic=ctx_raw.get("topic"),
        extra_metadata=ctx_raw.get("metadata") or {},
    )
    res = writer.write_verdict(
        verdict=verdict,
        packet_context=packet.get("context") or packet.get("environment") or {},
        session_context=ctx_raw.get("session_context") or {},
        ctx=ctx,
    )
    _emit_json(res.to_dict(), args.out)


def _cmd_write_kb_drafts(args: argparse.Namespace) -> None:
    """Handle ``write-kb-drafts``: batch-write KB drafts from a packet.

    Args:
        args (argparse.Namespace): Parsed CLI args (``packet``, ``kb_draft``,
            ``ctx``, ``out``).
    """
    packet = _read_json(args.packet) or {}
    kb_draft = _read_json(args.kb_draft) or {}
    ctx_raw = _read_json(args.ctx) or {}
    client = _resolve_kb_client()
    writer = KBWriter(client)
    ctx = WriteContext(
        session_id=ctx_raw.get("session_id") or "cli",
        review_id=ctx_raw.get("review_id"),
        source_role=ctx_raw.get("source_role", "critic"),
        source_type=ctx_raw.get("source_type", "critic_kb_draft"),
        extra_metadata=ctx_raw.get("metadata") or {},
    )
    res = writer.write_kb_drafts(
        kb_drafts=kb_draft.get("kb_drafts") or [],
        packet_context=packet.get("context") or packet.get("environment") or {},
        session_context=ctx_raw.get("session_context") or {},
        ctx=ctx,
    )
    _emit_json(res.to_dict(), args.out)


def _cmd_add_contradiction(args: argparse.Namespace) -> None:
    """Handle ``add-contradiction``: add contradicts edges between rows.

    Args:
        args (argparse.Namespace): Parsed CLI args (``new_id``, ``old_ids``
            comma-separated, ``ctx``, ``out``).
    """
    ctx_raw = _read_json(args.ctx) or {}
    client = _resolve_kb_client()
    writer = KBWriter(client)
    ctx = WriteContext(
        session_id=ctx_raw.get("session_id") or "cli",
        review_id=ctx_raw.get("review_id"),
        source_role=ctx_raw.get("source_role", "critic"),
    )
    old_ids = [oid.strip() for oid in args.old_ids.split(",") if oid.strip()]
    res = writer.add_contradiction(new_id=args.new_id, old_ids=old_ids, ctx=ctx)
    _emit_json(res.to_dict(), args.out)


def _cmd_replay_dead_letter(args: argparse.Namespace) -> None:
    """Handle ``replay-dead-letter``: re-dispatch queued failed KB writes.

    Args:
        args (argparse.Namespace): Parsed CLI args (``dir``,
            ``keep_on_success``, ``out``).
    """
    dlq = DeadLetter(root=args.dir or os.environ.get("KB_DEAD_LETTER_DIR"))
    client = _resolve_kb_client()
    summary = dlq.replay(
        lambda endpoint, payload: _replay_dispatch(client, endpoint, payload),
        delete_on_success=not args.keep_on_success,
    )
    _emit_json(summary.to_dict(), args.out)


def _replay_dispatch(client: KBClient, endpoint: str, payload: dict[str, Any]) -> None:
    """Re-dispatch a single dead-lettered KB request to ``client``.

    Args:
        client (KBClient): The KB client to replay against.
        endpoint (str): The original endpoint name (``upsert``,
            ``batch_insert``, ``edges/add`` or ``list``).
        payload (dict[str, Any]): The stored request payload.

    Raises:
        RuntimeAdapterError: If ``endpoint`` is unknown.
    """
    if endpoint == "upsert":
        client.upsert(payload)
    elif endpoint == "batch_insert":
        client.batch_insert(payload.get("items") or [], on_conflict=payload.get("on_conflict") or "upsert")
    elif endpoint == "edges/add":
        client.add_edges(payload.get("edges") or [])
    elif endpoint == "list":
        client.list(**payload)
    else:
        raise RuntimeAdapterError(f"replay-dead-letter: unknown endpoint {endpoint!r}")


# ---------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all Critic CLI subcommands.

    Returns:
        argparse.ArgumentParser: A parser whose subcommands each set ``func``
        to the matching handler.
    """
    p = argparse.ArgumentParser(prog="runtime.cli", description="Critic runtime CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Phase 0
    init = sub.add_parser("init-session")
    init.add_argument("--request", required=True)
    init.add_argument("--out", default="-")
    init.set_defaults(func=_cmd_init_session)

    close = sub.add_parser("close-session")
    close.add_argument("--request", required=True)
    close.add_argument("--kb-draft", default=None)
    close.add_argument("--out", default="-")
    close.set_defaults(func=_cmd_close_session)

    # Phase 1 + 2
    prep = sub.add_parser("prepare-review")
    prep.add_argument("--request", required=True)
    prep.add_argument("--out", default="-")
    prep.set_defaults(func=_cmd_prepare_review)

    commit = sub.add_parser("commit-review")
    commit.add_argument("--request", required=True)
    commit.add_argument("--review", required=True)
    commit.add_argument("--out", default="-")
    commit.set_defaults(func=_cmd_commit_review)

    # Low-level
    listp = sub.add_parser("list-priors")
    listp.add_argument("--packet", required=True)
    listp.add_argument("--kind", default=None)
    listp.add_argument("--topic", default=None)
    listp.add_argument("--limit", type=int, default=10)
    listp.add_argument("--session", default=None)
    listp.add_argument("--out", default="-")
    listp.set_defaults(func=_cmd_list_priors)

    wv = sub.add_parser("write-verdict")
    wv.add_argument("--packet", required=True)
    wv.add_argument("--verdict", required=True)
    wv.add_argument("--ctx", required=True)
    wv.add_argument("--out", default="-")
    wv.set_defaults(func=_cmd_write_verdict)

    wd = sub.add_parser("write-kb-drafts")
    wd.add_argument("--packet", required=True)
    wd.add_argument("--kb-draft", required=True)
    wd.add_argument("--ctx", required=True)
    wd.add_argument("--out", default="-")
    wd.set_defaults(func=_cmd_write_kb_drafts)

    ac = sub.add_parser("add-contradiction")
    ac.add_argument("--new-id", required=True)
    ac.add_argument("--old-ids", required=True, help="Comma-separated KB ids.")
    ac.add_argument("--ctx", required=True)
    ac.add_argument("--out", default="-")
    ac.set_defaults(func=_cmd_add_contradiction)

    rd = sub.add_parser("replay-dead-letter")
    rd.add_argument("--dir", default=None)
    rd.add_argument("--keep-on-success", action="store_true")
    rd.add_argument("--out", default="-")
    rd.set_defaults(func=_cmd_replay_dead_letter)

    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected subcommand.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv`` when
            ``None``.

    Returns:
        int: ``0`` on logical success (including dead-lettered outcomes) or
        ``2`` when a :class:`RuntimeAdapterError` propagates from the handler.
    """
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeAdapterError as exc:
        sys.stderr.write(f"runtime.cli: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
