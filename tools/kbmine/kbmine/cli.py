#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Estimate Hyperloom uplift from the Recipe KB without running a session.

Pulls per-session documents (or reads a local JSON list) and prints what prior
sessions already settled for a scope: the distribution of validated gain, and
which parallelism layout won inside a fixed GPU count. Scoped by identity and
by the full tp/conc/isl/osl replay scope, because a pooled median across
models or shapes is not a prior for any of them.

Credentials are supplied at runtime and are never literals in this file or in
the repository. Set ``KB_STORE_TOKEN`` in the environment; a token file and a
flag also work, resolved as ``--kb-store-token`` then
``--kb-store-token-file`` then ``KB_STORE_TOKEN``. Avoid the flag outside CI,
since an argument is visible in ``ps`` and in shell history. The report echoes
the store URL but never the token.

Examples
--------

::

    export KB_STORE_URL=... KB_STORE_TOKEN=...   # runtime only; never committed
    python -m kbmine.cli --hardware mi355x --framework-name sglang

    export PULSE_URL=...                         # Pulse API base
    python -m kbmine.cli --pulse-url --hardware mi355x

    python -m kbmine.cli \\
        --input prior_sessions.json --tp 8 --isl 1024 --osl 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .mine import (
    estimate_from_sessions,
    fetch_session_documents,
)
from .kb_store_client import (
    KBStoreClient,
    KBStoreError,
)
from .pulse import PulseClient, PulseError, project_pulse_row


def _load_input(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: cannot read a session JSON file: {exc}") from exc
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
        return [item for item in raw["sessions"] if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    raise SystemExit(f"{path}: expected a session object, a list, or {{sessions: [...]}}")


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve store URL + token from flags, then a token file, then env."""
    url = (args.kb_store_url or os.environ.get("KB_STORE_URL") or "").strip().rstrip("/")
    token = (args.kb_store_token or "").strip()
    if not token and args.kb_store_token_file is not None:
        path = Path(args.kb_store_token_file).expanduser()
        try:
            token = path.read_text(encoding="utf-8").strip()
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read --kb-store-token-file {path}: {exc}") from exc
    if not token:
        token = (os.environ.get("KB_STORE_TOKEN") or "").strip()
    return url, token


_FROM_ENV = object()

_WINNERS_ONLY = (
    "historical gains are conditional on a session having won: Hyperloom writes a Recipe KB record only for a "
    "session that kept a change and beat the scope's champion, so no 0% outcome can appear; read p50/p90 as what a "
    "winning session reached, not the odds of winning (Pulse rows include sessions that did not win)"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Offline JSON of session envelopes (skips the live store).",
    )
    parser.add_argument(
        "--kb-store-url",
        dest="kb_store_url",
        default="",
        help="KB Store base URL; falls back to $KB_STORE_URL.",
    )
    parser.add_argument(
        "--kb-store-token",
        dest="kb_store_token",
        default="",
        help="Bearer token. Visible in ps/history; prefer --kb-store-token-file.",
    )
    parser.add_argument(
        "--kb-store-token-file",
        dest="kb_store_token_file",
        type=Path,
        default=None,
        help="File holding only the bearer token; falls back to $KB_STORE_TOKEN.",
    )
    parser.add_argument("--hardware", help="Search match: hardware (e.g. mi355x).")
    parser.add_argument("--framework-name", dest="framework_name", help="Search match: framework_name.")
    parser.add_argument("--model", help="Search match: model.")
    parser.add_argument("--precision", help="Search match: precision.")
    parser.add_argument("--canonical-id", action="append", default=[], help="Fetch this identity (repeatable).")
    parser.add_argument(
        "--max-identities",
        type=int,
        default=50,
        help="Read at most this many matching identities from the KB (default 50; the search reads at most 1000 "
        "per run). The report's coverage and limitations say when the store holds more.",
    )
    for key, helptext in (
        ("tp", "tensor parallelism"),
        ("conc", "concurrency"),
        ("isl", "input sequence length"),
        ("osl", "output sequence length"),
    ):
        parser.add_argument(
            f"--{key}",
            type=int,
            default=None,
            help=f"Restrict the pool to this {helptext} (replay scope dimension).",
        )
    parser.add_argument(
        "--target-tp",
        dest="target_tp",
        type=int,
        default=None,
        help="Project this TP from observed per-GPU throughput (prior, not a replay).",
    )
    parser.add_argument("--output", type=Path, help="Write JSON here; default stdout.")
    pulse = parser.add_argument_group("pulse (fleet session evidence)")
    pulse.add_argument(
        "--pulse-url",
        dest="pulse_url",
        nargs="?",
        const=_FROM_ENV,
        default="",
        help="Read session evidence from Pulse instead of the Recipe KB. Takes the API base, or $PULSE_URL when "
        "given without a value.",
    )
    pulse.add_argument(
        "--ca-bundle",
        dest="ca_bundle",
        default=None,
        help="PEM bundle for an internally-signed host; required over VPN.",
    )
    pulse.add_argument("--start", help="Pulse filter: ISO date lower bound.")
    pulse.add_argument("--end", help="Pulse filter: ISO date upper bound.")
    pulse.add_argument("--data-source", dest="data_source", default=None, help="Pulse filter: data_source.")
    pulse.add_argument("--pipeline-tag", dest="pipeline_tag", default=None, help="Pulse filter: pipeline_tag.")
    pulse.add_argument("--max-rows", dest="max_rows", type=int, default=1000, help="Pulse row cap.")
    return parser


def _pulse_documents(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch Pulse rows, filtering identity dimensions client-side.

    Server-side filters are inconsistent (``prec`` is ignored, ``gpu_type``
    matches far fewer rows than carry that value), so identity narrowing is
    done here where it is verifiable.
    """
    _, token = resolve_credentials(args)
    client = PulseClient(args.pulse_url, token, ca_bundle=args.ca_bundle)
    rows = list(
        client.session_breakdowns(
            max_rows=max(1, args.max_rows),
            start=args.start,
            end=args.end,
            data_source=args.data_source,
            pipeline_tag=args.pipeline_tag,
        )
    )
    wanted = {
        "gpu_type": args.hardware,
        "framework": args.framework_name,
        "model_name": args.model,
        "prec": args.precision,
    }
    kept = [
        row
        for row in rows
        if all(not want or str(row.get(key) or "").lower() == want.lower() for key, want in wanted.items())
    ]
    notes = [f"pulse: fetched {len(rows)} rows, {len(kept)} matched the requested identity", *client.walk_notes]
    return kept, notes


def _note_identity_coverage(report: dict[str, Any], counts: dict[str, Any], *, cap: int) -> None:
    """Say how much of the store's match the report stands on; a truncated pool is the server's first N, not a sample."""
    fetched, matched = counts.get("fetched", 0), counts.get("matched")
    report["coverage"]["identities_fetched"] = fetched
    report["coverage"]["identities_matched"] = matched
    if not counts.get("searched"):
        return
    page_capped = bool(counts.get("page_capped"))
    binding = (
        f"the search's {counts.get('page_limit')}-identity page limit" if page_capped else f"--max-identities {cap}"
    )
    if matched is not None and matched > fetched:
        report["limitations"].append(
            f"read {fetched} of the {matched} identities matching the search ({binding}); the rest are not in this "
            "report, and the first ones in server order are not a random sample"
        )
    elif matched is None and (page_capped or fetched >= cap):
        report["limitations"].append(
            f"stopped at {binding} and the store reported no total, so more matching identities may exist"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.pulse_url is _FROM_ENV:
        args.pulse_url = (os.environ.get("PULSE_URL") or "").strip()
        if not args.pulse_url:
            print("Pulse is not configured: pass --pulse-url URL or set PULSE_URL", file=sys.stderr)
            return 2
    errors: list[str] = []
    store_url = ""
    projector = None
    identity_counts: dict[str, Any] = {}
    if args.input is not None:
        documents = _load_input(args.input)
    elif args.pulse_url:
        projector = project_pulse_row
        store_url = args.pulse_url
        try:
            documents, errors = _pulse_documents(args)
        except PulseError as exc:
            print(f"Pulse fetch failed: {exc}", file=sys.stderr)
            return 1
    else:
        store_url, token = resolve_credentials(args)
        if not store_url:
            print(
                "KB store is not configured: pass --kb-store-url or set KB_STORE_URL",
                file=sys.stderr,
            )
            return 2
        try:
            store = KBStoreClient(store_url, token, ca_bundle=args.ca_bundle)
        except KBStoreError as exc:
            print(f"KB store is not configured: {exc}", file=sys.stderr)
            return 2
        match: dict[str, str] = {}
        if args.model:
            match["model"] = args.model
        if args.hardware:
            match["hardware"] = args.hardware
        if args.framework_name:
            match["framework_name"] = args.framework_name
        if args.precision:
            match["precision"] = args.precision
        try:
            documents, errors = fetch_session_documents(
                store,
                match=match or None,
                max_identities=max(1, args.max_identities),
                canonical_ids=list(args.canonical_id) or None,
                counts=identity_counts,
            )
        except KBStoreError as exc:
            print(f"KB fetch failed: {exc}", file=sys.stderr)
            return 1
    shape = {key: getattr(args, key) for key in ("tp", "conc", "isl", "osl")}
    if projector is None:
        report = estimate_from_sessions(documents, shape=shape, target_tp=args.target_tp)
    else:
        report = estimate_from_sessions(documents, shape=shape, target_tp=args.target_tp, projector=projector)
        # Pulse rows carry no accepted server args, so there is no layout to
        # rank. Saying so beats publishing arms labelled from a default.
        report["sharding_whatif"] = {
            "available": False,
            "reason": "pulse session evidence carries no accepted server args; layout ranking needs the Recipe KB",
        }
        report["evidence_source"] = "pulse:/v1/session-breakdowns"
    if identity_counts:
        _note_identity_coverage(report, identity_counts, cap=max(1, args.max_identities))
    if projector is None:
        report["limitations"].append(_WINNERS_ONLY)
    report["fetch_errors"] = errors
    # One key per service, so a reader never takes the Pulse base for a KB Store address.
    report["pulse_url" if projector is not None else "kb_store_url"] = store_url
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"cannot write --output {args.output}: {exc}", file=sys.stderr)
            return 1
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
