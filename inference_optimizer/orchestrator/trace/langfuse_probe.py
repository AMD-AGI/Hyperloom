# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tiny CLI to fire a Langfuse debug probe for a session (link-check aid).

Pushes one ``probe:session-start`` Generation and flushes immediately, so you
can confirm the live Langfuse pipe works within seconds — without waiting for
a real run to produce traffic or reach its session-end flush.

Usage::

    # Needs HYPERLOOM_LANGFUSE_ENABLE=1 + LANGFUSE_* creds + the langfuse SDK.
    python -m inference_optimizer.orchestrator.trace.langfuse_probe <session_dir>
    python -m inference_optimizer.orchestrator.trace.langfuse_probe <session_dir> --note "smoke test"

Exit codes: 0 = probe sent; 1 = push disabled (a warning explains which gate
tripped) or the probe failed. The session dir only needs to exist; a
``manifest.json`` inside it makes the trace id / session correlate to the real
run, but it is optional for a bare link check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .langfuse_emitter import emit_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="langfuse_probe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session_dir", type=Path,
        help="Session directory (the trace correlates on its manifest.json if present).",
    )
    parser.add_argument(
        "--note", default=None,
        help="Free-text note stored on the probe's metadata.",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose >= 1 else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sd = args.session_dir.resolve()
    sd.mkdir(parents=True, exist_ok=True)
    sent = emit_probe(sd, note=args.note)
    if sent:
        print(f"probe sent + flushed for {sd}; check your Langfuse UI.")
        return 0
    print(
        "probe NOT sent — live push is disabled (see the warning above for the "
        "reason). Verify HYPERLOOM_LANGFUSE_ENABLE=1, the three LANGFUSE_* vars, "
        "and that the langfuse SDK is installed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
