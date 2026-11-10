"""``hyperloom-kb-collect``: project a source document through a mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hyperloom_kb.collect.engine import SourceDocumentError, collect
from hyperloom_kb.collect.expressions import MappingError
from hyperloom_kb.config import ConfigurationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Experiences from one source document. Writes go to the KB configured by "
            "HYPERLOOM_KB_URL/HYPERLOOM_KB_TOKEN or the local collection settings."
        )
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="Packaged mapping name (for example hyperloom-sbd-v6) or mapping file path.",
    )
    parser.add_argument("--document", required=True, type=Path, help="Source JSON document.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Project and validate without writing; the report carries each Experience.",
    )
    parser.add_argument("--receipt", type=Path, help="Also write the report to this path.")
    args = parser.parse_args(argv)
    try:
        report = collect(args.mapping, args.document, dry_run=args.dry_run, receipt=args.receipt)
    except (ConfigurationError, MappingError, SourceDocumentError) as exc:
        print(f"hyperloom-kb-collect: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
