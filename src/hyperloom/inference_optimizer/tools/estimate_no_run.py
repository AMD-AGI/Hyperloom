#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Estimate Hyperloom uplift from the Recipe KB without running a session.

Pulls per-session documents (or reads a local JSON list) and prints what prior
sessions already settled for a scope: the distribution of validated gain, and
which parallelism layout won inside a fixed GPU count. Scoped by identity and
by the full tp/conc/isl/osl replay scope, because a pooled median across
models or shapes is not a prior for any of them.

Credentials are parameters, never literals in this file. Resolution order is
``--kb-store-token`` then ``--kb-store-token-file`` then ``KB_STORE_TOKEN``;
prefer the file form so the secret stays out of shell history and ``ps``.
The report echoes the store URL but never the token.

Examples
--------

::

    python -m hyperloom.inference_optimizer.tools.estimate_no_run \\
        --kb-store-url https://host/knowledge-base \\
        --kb-store-token-file ~/.secrets/kb_store_token \\
        --hardware mi355x --framework-name sglang

    export KB_STORE_URL=... KB_STORE_TOKEN=...
    python -m hyperloom.inference_optimizer.tools.estimate_no_run --hardware mi355x

    python -m hyperloom.inference_optimizer.tools.estimate_no_run \\
        --input prior_sessions.json --tp 8 --isl 1024 --osl 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.knowledge.remote_recipe.no_run import (
    estimate_from_sessions,
    fetch_session_documents,
)
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KBStoreClient,
    KBStoreError,
)


def _load_input(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
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
        except OSError as exc:
            raise SystemExit(f"cannot read --kb-store-token-file {path}: {exc}")
    if not token:
        token = (os.environ.get("KB_STORE_TOKEN") or "").strip()
    return url, token


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
    parser.add_argument("--max-identities", type=int, default=50)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors: list[str] = []
    store_url = ""
    if args.input is not None:
        documents = _load_input(args.input)
    else:
        store_url, token = resolve_credentials(args)
        if not store_url:
            print(
                "KB store is not configured: pass --kb-store-url or set KB_STORE_URL",
                file=sys.stderr,
            )
            return 2
        try:
            store = KBStoreClient(store_url, token)
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
        hardware_in = [args.hardware] if args.hardware and not match else None
        # Prefer exact match; hardware_in is for multi-board scans without a match dict.
        if match:
            hardware_in = None
        try:
            documents, errors = fetch_session_documents(
                store,
                match=match or None,
                hardware_in=hardware_in,
                max_identities=max(1, args.max_identities),
                canonical_ids=list(args.canonical_id) or None,
            )
        except KBStoreError as exc:
            print(f"KB fetch failed: {exc}", file=sys.stderr)
            return 1
    report = estimate_from_sessions(
        documents,
        shape={key: getattr(args, key) for key in ("tp", "conc", "isl", "osl")},
        target_tp=args.target_tp,
    )
    report["fetch_errors"] = errors
    report["kb_store_url"] = store_url
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
