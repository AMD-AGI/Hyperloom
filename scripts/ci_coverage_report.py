# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render coverage once and defer its native exit status to a separate CI gate."""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from coverage.cmdline import main as coverage_main

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_SOURCE_SEGMENTS = {
    "src/hyperloom": ["/hyperloom/"],
    "OOB": ["/OOB/", "/agent_mcp_server/"],
}


def extract_total(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("TOTAL "):
            parts = line.split()
            return parts[-1] if parts else "unknown"
    return "unknown"


def _row_path_stmts_miss(line: str) -> tuple[str, int, int] | None:
    """Parse a coverage report row into its path, statement count, and misses."""
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        stmts = int(parts[-3])
        miss = int(parts[-2])
    except ValueError:
        return None
    path = " ".join(parts[:-3])
    return path, stmts, miss


def per_tree_totals(report_text: str) -> dict[str, str]:
    """Bucket report rows by source path segments without double-counting."""
    acc = {src: [0, 0] for src in _SOURCE_SEGMENTS}
    for line in report_text.splitlines():
        if line.startswith(("Name", "TOTAL", "-")):
            continue
        parsed = _row_path_stmts_miss(line)
        if parsed is None:
            continue
        path, stmts, miss = parsed
        norm = "/" + path.lstrip("/")
        for src, segs in _SOURCE_SEGMENTS.items():
            if any(seg in norm for seg in segs):
                acc[src][0] += stmts
                acc[src][1] += miss
                break
    out: dict[str, str] = {}
    for src, (stmts, miss) in acc.items():
        out[src] = f"{(stmts - miss) / stmts * 100:.2f}%" if stmts else "n/a"
    return out


def run_report() -> tuple[str, int]:
    """Keep coverage's configured threshold, precision, output, and exit status."""
    stdout, stderr = io.StringIO(), io.StringIO()
    print("=== Full coverage report (step log) ===")
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = coverage_main(["report", "--skip-empty"])
    finally:
        sys.stdout.write(stdout.getvalue())
        sys.stderr.write(stderr.getvalue())
    return stdout.getvalue(), status or 0


def write_summary() -> None:
    cfg = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sources = list(cfg["tool"]["coverage"]["run"]["source"])
    outcome = os.environ.get("PYTEST_OUTCOME", "unknown")
    if not Path(".coverage").exists():
        lines = ["## Coverage (UT)", "", "No `.coverage` data."]
        status = 1
    else:
        full_txt, status = run_report()
        total = extract_total(full_txt)
        tree_totals = per_tree_totals(full_txt)
        rows = [f"| `{src}` | {tree_totals.get(src, 'n/a')} |" for src in sources]
        lines = [
            "## Coverage (UT)",
            "",
            f"Python {sys.version_info.major}.{sys.version_info.minor}; "
            "roots and CI pytest argv from `pyproject.toml`. "
            "Combined across sharded jobs.",
            "",
            "### Combined measured source (all configured trees)",
            f"**TOTAL (line): {total}**",
            "",
            "### Per-tree (narrow)",
            "| Tree | Line coverage (TOTAL) |",
            "|------|----------------------|",
            *rows,
        ]
        if outcome != "success":
            lines.extend(
                [
                    "",
                    "> Tests did not all pass; coverage can be **partial or misleading**.",
                ]
            )

    block = "\n".join(lines) + "\n"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(block)
    else:
        print(block)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"report_exit_code={status}\n")


def enforce_gate() -> int:
    try:
        status = int(os.environ.get("REPORT_EXIT_CODE", ""))
    except ValueError:
        print("::error::No valid coverage report exit status; see the Coverage summary step.")
        return 1
    relax = (os.environ.get("COVERAGE_RELAX_FAIL_UNDER") or "").strip().lower()
    if status == 2 and relax in {"1", "true", "yes", "on"}:
        print("Relaxed mode: skip coverage fail_under enforcement.")
        return 0
    if status not in {0, 2}:
        print(f"::error::Coverage report failed (exit {status}); see the Coverage summary step.")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["report", "gate"], default="report")
    args = parser.parse_args(argv)
    if args.command == "gate":
        return enforce_gate()
    write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
