#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-verify Hyperloom's InferenceX patch anchors and refresh the contract record.

Hyperloom patches InferenceX by matching exact upstream text, so bumping
``INFERENCEX_REF`` -- or editing an anchor -- can silently take a patch offline.
``test_inferencex_anchor_contract.py`` fails whenever either happens, and this is
the tool it points at: it fetches the pinned files, confirms every anchor still
matches exactly one site, and records the result.

It refuses to record a broken contract. If an anchor no longer matches, re-anchor
it in ``_inferencex_patcher.py`` first, then run this again.

Requires ``gh`` authenticated against the (private) InferenceX repo.

Usage::

    python scripts/refresh_inferencex_anchor_contract.py
    python scripts/refresh_inferencex_anchor_contract.py --ref <commit>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The package is used from a source checkout here, not an installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hyperloom.inference_optimizer.cli.preflight import _INFERENCEX_REF_DEFAULT  # noqa: E402
from hyperloom.inference_optimizer.tests.test_inferencex_anchor_contract import (  # noqa: E402
    CONTRACT_PATH,
    build_record,
)


def main() -> int:
    """Refresh the anchor contract record.

    Returns:
        ``0`` on success, ``1`` when upstream could not be verified.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ref",
        default=_INFERENCEX_REF_DEFAULT,
        help="commit to verify against (default: the pin the code installs)",
    )
    args = parser.parse_args()

    try:
        record = build_record(args.ref)
    except RuntimeError as exc:
        print(f"refresh failed: {exc}", file=sys.stderr)
        return 1

    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"verified {len(record['files'])} file(s) at {record['ref']}")
    for rel_path, spec in sorted(record["files"].items()):
        for name, hits in sorted(spec["anchors"].items()):
            print(f"  {name:<20} {hits} site(s)  {rel_path}")
    print(f"wrote {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
