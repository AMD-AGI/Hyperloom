#!/usr/bin/env python3
"""Write `packs/_shared/intents.md` from the intent registry.

The file a pack `include:`s instead of restating the intent rules in ~20 hand
written constraint lines. Generated so that adding an intent reaches every pack
by being a row, not by an edit in each of them.

Idempotent. `arbor/tests/test_intents.py::test_shared_doc_in_sync` fails if the
checked-in copy has drifted from the registry, so this is the only way to
change it.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from arbor.measure.intents import SHARED_DOC_RELPATH, render_shared_doc


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    out = root / SHARED_DOC_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    new = render_shared_doc()
    if out.exists() and out.read_text() == new:
        print(f"{SHARED_DOC_RELPATH}: already in sync")
        return 0
    out.write_text(new)
    print(f"{SHARED_DOC_RELPATH}: written ({len(new)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
