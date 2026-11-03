#!/usr/bin/env python3
"""Check a dossier, or one of its update rounds: ``arbor/dossier/check.py``, run from this skill's place.

    python3 check-dossier.py <dir> [--round uK] [--no-index] [--json]

The same as ``python3 -m arbor.dossier.check``. This file stays for the packs, callers and tests that name it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from arbor.dossier.check import *  # noqa: E402,F401,F403 - callers load this file and use its functions
from arbor.dossier.check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
