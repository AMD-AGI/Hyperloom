# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic runtime CLI."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from hyperloom.common.subprocess_bridge import emit_json, read_json

from .dead_letter import DeadLetter
from .decision_reviewer import DecisionReviewer
from .errors import RuntimeAdapterError
from .in_memory_kb_client import InMemoryKBClient
from .kb_client import HTTPKBClient, KBClient
from .kb_writer import KBWriter, WriteContext
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory


def _resolve_kb_client() -> KBClient:
    """Build the KB client selected by ``CRITIC_KB_CLIENT_MODE``."""
    mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    timeout_ms = int(os.environ.get("KB_TIMEOUT_MS", "10000"))
    retry_max = int(os.environ.get("KB_RETRY_MAX", "3"))
    token = os.environ.get("KB_SERVICE_TOKEN")
    if mode == "live":
        base_url = os.environ.get("KB_BASE_URL")
        if not base_url:
            raise RuntimeAdapterError("CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not set")
        return HTTPKBClient(base_url=base_url, timeout_ms=timeout_ms, retry_max=retry_max, token=token)
    return InMemoryKBClient()


def _resolve_reviewer() -> DecisionReviewer:
    """Build a fully wired :class:`DecisionReviewer` for CLI commands."""
    sm = SessionMemory()
    client = _resolve_kb_client()
    writer = KBWriter(client, session_memory=sm)
    return DecisionReviewer(session_memory=sm, kb_client=client, kb_writer=writer)


def _cmd_init_session(args: argparse.Namespace) -> None:
    """Handle ``init-session``: merge a request's context and emit it."""
    request = read_json(args.request)
    reviewer = _resolve_reviewer()
    out = reviewer.init_session(request)
    emit_json(out, args.out)


def _cmd_prepare_review(args: argparse.Namespace) -> None:
    """Handle ``prepare-review``: emit the phase-1 judge bundle."""
    request = read_json(args.request)
    reviewer = _resolve_reviewer()
    bundle = reviewer.prepare_review(request)
    emit_json(bundle.to_dict(), args.out)


def _cmd_commit_review(args: argparse.Namespace) -> None:
    """Handle ``commit-review``: validate a review and emit the outcome."""
    request = read_json(args.request)
    review = read_json(args.review)
    if not isinstance(review, dict):
        raise RuntimeAdapterError("--review must be a JSON object")
    reviewer = _resolve_reviewer()
    outcome = reviewer.commit_review(request, review)
    emit_json(outcome.to_dict(), args.out)


def _cmd_close_session(args: argparse.Namespace) -> None:
    """Handle ``close-session``: close a session, optionally flushing drafts."""
    request = read_json(args.request)
    kb_draft = read_json(args.kb_draft) if args.kb_draft else None
    reviewer = _resolve_reviewer()
    outcome = reviewer.close_session(request, kb_draft)
    emit_json(outcome.to_dict(), args.out)


def _cmd_list_priors(args: argparse.Namespace) -> None:
    """Handle ``list-priors``: look up KB priors for a packet's scope."""
    packet = read_json(args.packet) or {}
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
    priors["scope_cache_key"] = scope_cache_key(scope_filter, topic=args.topic, kind=args.kind, limit=args.limit)
    emit_json(priors, args.out)


def _cmd_write_verdict(args: argparse.Namespace) -> None:
    """Handle ``write-verdict``: write a single verdict lesson to KB."""
    packet = read_json(args.packet) or {}
    verdict = read_json(args.verdict) or {}
    ctx_raw = read_json(args.ctx) or {}
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
        packet_context=packet.get("context") or {},
        session_context=ctx_raw.get("session_context") or {},
        ctx=ctx,
    )
    emit_json(res.to_dict(), args.out)


def _cmd_write_kb_drafts(args: argparse.Namespace) -> None:
    """Handle ``write-kb-drafts``: batch-write KB drafts from a packet."""
    packet = read_json(args.packet) or {}
    kb_draft = read_json(args.kb_draft) or {}
    ctx_raw = read_json(args.ctx) or {}
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
        packet_context=packet.get("context") or {},
        session_context=ctx_raw.get("session_context") or {},
        ctx=ctx,
    )
    emit_json(res.to_dict(), args.out)


def _cmd_add_contradiction(args: argparse.Namespace) -> None:
    """Handle ``add-contradiction``: add contradicts edges between rows."""
    ctx_raw = read_json(args.ctx) or {}
    client = _resolve_kb_client()
    writer = KBWriter(client)
    ctx = WriteContext(
        session_id=ctx_raw.get("session_id") or "cli",
        review_id=ctx_raw.get("review_id"),
        source_role=ctx_raw.get("source_role", "critic"),
    )
    old_ids = [oid.strip() for oid in args.old_ids.split(",") if oid.strip()]
    res = writer.add_contradiction(new_id=args.new_id, old_ids=old_ids, ctx=ctx)
    emit_json(res.to_dict(), args.out)


def _cmd_replay_dead_letter(args: argparse.Namespace) -> None:
    """Handle ``replay-dead-letter``: re-dispatch queued failed KB writes."""
    dlq = DeadLetter(root=args.dir or os.environ.get("KB_DEAD_LETTER_DIR"))
    client = _resolve_kb_client()
    summary = dlq.replay(
        lambda endpoint, payload: _replay_dispatch(client, endpoint, payload),
        delete_on_success=not args.keep_on_success,
    )
    emit_json(summary.to_dict(), args.out)


def _replay_dispatch(client: KBClient, endpoint: str, payload: dict[str, Any]) -> None:
    """Re-dispatch a single dead-lettered KB request to ``client``."""
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


def _make_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all Critic CLI subcommands."""
    p = argparse.ArgumentParser(prog="hyperloom.agents.critic.runtime.cli", description="Critic runtime CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-session")
    init.add_argument("--request", required=True)
    init.add_argument("--out", default="-")
    init.set_defaults(func=_cmd_init_session)

    close = sub.add_parser("close-session")
    close.add_argument("--request", required=True)
    close.add_argument("--kb-draft", default=None)
    close.add_argument("--out", default="-")
    close.set_defaults(func=_cmd_close_session)

    prep = sub.add_parser("prepare-review")
    prep.add_argument("--request", required=True)
    prep.add_argument("--out", default="-")
    prep.set_defaults(func=_cmd_prepare_review)

    commit = sub.add_parser("commit-review")
    commit.add_argument("--request", required=True)
    commit.add_argument("--review", required=True)
    commit.add_argument("--out", default="-")
    commit.set_defaults(func=_cmd_commit_review)

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
    """Parse arguments and run the selected subcommand."""
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
