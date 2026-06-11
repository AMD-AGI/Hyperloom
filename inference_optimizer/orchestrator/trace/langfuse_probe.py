# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tiny CLI to fire a Langfuse debug probe for a session (link-check aid).

Pushes one ``probe:session-start`` Generation and flushes immediately, so you
can confirm the live Langfuse pipe works within seconds — without waiting for
a real run to produce traffic or reach its session-end flush.

Needs ``HYPERLOOM_LANGFUSE_ENABLE=1`` + the ``LANGFUSE_*`` creds. The langfuse
SDK is auto-installed on demand when push is enabled but the SDK is missing
(running this module directly bypasses ``scripts/install.sh``, which would
otherwise install it); pass ``--no-install-sdk`` to opt out.

Usage::

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
import subprocess
import sys
from pathlib import Path

from .langfuse_emitter import _sdk_available, emit_probe
from .trace_env import langfuse_live_enabled

log = logging.getLogger(__name__)

#: Mirror scripts/install.sh::ensure_langfuse_when_enabled.
_LANGFUSE_SPEC = "langfuse>=2.0"


def _ensure_sdk_when_enabled() -> None:
    """If live push is switched on but the SDK is missing, install it with the
    current interpreter (import-probe first, pip only on miss).

    The full session path installs langfuse via ``scripts/install.sh``, but
    running this probe module directly bypasses that — so a bare
    ``python -m ...langfuse_probe`` would always report ``sdk_missing``.
    Mirror the installer here so the probe is self-sufficient. Fail-soft:
    a failed install just warns (emit_probe then reports the gate) and never
    raises. Skipped entirely when the master switch is off.
    """
    if not langfuse_live_enabled():
        return
    if _sdk_available():
        return
    log.info("langfuse: SDK missing but push is enabled — installing %s", _LANGFUSE_SPEC)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"]
    if sys.prefix == sys.base_prefix:
        # Outside a venv: tolerate PEP 668 externally-managed environments.
        cmd.append("--break-system-packages")
    cmd.append(_LANGFUSE_SPEC)
    try:
        subprocess.run(cmd, check=True)
    except Exception:  # noqa: BLE001 — install failure must not break the probe
        log.warning(
            "langfuse: SDK auto-install failed; live push will no-op. "
            "Preinstall it: %s -m pip install '%s'",
            sys.executable, _LANGFUSE_SPEC, exc_info=True,
        )


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
    parser.add_argument(
        "--no-install-sdk", dest="install_sdk", action="store_false",
        help="Do NOT auto-install the langfuse SDK when it is missing "
             "(default: install it on demand if live push is enabled).",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose >= 1 else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.install_sdk:
        _ensure_sdk_when_enabled()

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
