"""Re-publish a batch's normalized_results.json to luochen from a GHA artifact.

When the CI workflow's publish step fails (luochen down, asyncpg crash, etc.),
the ci-summary artifact still gets uploaded with summary-out/normalized_results.json
intact (45-day retention). This helper downloads that artifact and pipes it
through ci/publish_results.py so we can recover after luochen is back online,
without having to re-run the whole 2-3h optimization batch.

Usage:
    python .harvest/republish_from_artifact.py 25789927636
    python .harvest/republish_from_artifact.py 25789927636 --dry-run

Set HYPERLOOM_RESULTS_SERVICE_URL in env if you want a non-default endpoint
(defaults to the public ingress in publish_results.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="GitHub Actions run id (the numeric one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download but don't POST — useful for verifying the "
                         "artifact contents before pushing.")
    ap.add_argument("--repo", default="AMD-AGI/Hyperloom")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix=f"republish-{args.run_id}-") as tmp:
        tmp_path = Path(tmp)
        print(f"Downloading ci-summary artifact from run {args.run_id} into {tmp_path}")
        rc = subprocess.call([
            "gh", "run", "download", args.run_id,
            "--repo", args.repo,
            "--name", "ci-summary",
            "--dir", str(tmp_path),
        ])
        if rc != 0:
            print(f"ERROR: gh run download failed (exit {rc})", file=sys.stderr)
            return rc

        candidates = list(tmp_path.rglob("normalized_results.json"))
        if not candidates:
            print(f"ERROR: no normalized_results.json under {tmp_path}",
                  file=sys.stderr)
            return 2
        input_path = candidates[0]
        size = input_path.stat().st_size
        print(f"Found {input_path} ({size} bytes)")

        if args.dry_run:
            print("--dry-run set, skipping publish.")
            return 0

        ci_dir = Path(__file__).resolve().parent.parent / "ci"
        rc = subprocess.call(
            ["python3", str(ci_dir / "publish_results.py"),
             "--input", str(input_path)],
            env=os.environ.copy(),
        )
        return rc


if __name__ == "__main__":
    sys.exit(main())
