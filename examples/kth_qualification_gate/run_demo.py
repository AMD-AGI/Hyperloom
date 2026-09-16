#!/usr/bin/env python3
"""Run the deterministic KTH-before-performance Hyperloom demo."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name

from hyperloom.orchestrator.kernel.controller_patch_integration import (
    integrate_controller_patches,
)
from hyperloom.orchestrator.kernel.kth_qualification import KthQualificationProvider
from hyperloom.orchestrator.state.shared_state import SharedState

PLAN_ID = "fixture/cpu-rne-correction-v1"
KERNEL_PATH = "kernels/quantize.py"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "kth-hyperloom-demo",
    "GIT_AUTHOR_EMAIL": "demo@local",
    "GIT_COMMITTER_NAME": "kth-hyperloom-demo",
    "GIT_COMMITTER_EMAIL": "demo@local",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_patch(repo: Path, content: str) -> str:
    path = repo / KERNEL_PATH
    original = path.read_text(encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    patch = _git(repo, "diff", "--binary", "--", KERNEL_PATH)
    path.write_text(original, encoding="utf-8")
    return patch + "\n"


def _publish(
    patches_root: Path,
    repo: Path,
    base_commit: str,
    *,
    ordinal: str,
    content: str,
    evidence: str,
) -> None:
    kernel_name = f"{ordinal}-quant-rne"
    operator_id = f"kernel:forge-loop:{kernel_name}:standalone:demo:python-fixture:fixture-cpu"
    patch_dir = patches_root / operator_directory_name(operator_id)
    patch_dir.mkdir(parents=True)
    (patch_dir / "change.patch").write_text(
        _make_patch(repo, content),
        encoding="utf-8",
    )
    (patch_dir / "report.md").write_text(
        "\n".join(
            (
                f"# Candidate {ordinal}",
                "",
                "- micro_validated: true",
                "- tolerance evidence: fixture-labelled",
                f"- claim: {evidence}",
                "",
            )
        ),
        encoding="utf-8",
    )
    publication = {
        "schema_version": 2,
        "operator_id": operator_id,
        "identity": {
            "producer": "forge-loop",
            "kernel_name": kernel_name,
            "framework": "standalone",
            "framework_version": "demo",
            "backend": "python-fixture",
            "gpu": "fixture-cpu",
        },
        "base_commit": base_commit,
        "best_commit": "b" * 40,
        "repo_root": str(repo),
        "kernel_path": KERNEL_PATH,
        "operator_name": kernel_name,
        "micro_validated": True,
        "manifest": {"changed_files": [KERNEL_PATH]},
        "kth_qualification": {"plan_id": PLAN_ID},
    }
    (patch_dir / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n",
        encoding="utf-8",
    )


def _kth_wrapper(root: Path, output: Path) -> tuple[Path, str]:
    kth_sha = _git(root, "rev-parse", "HEAD")
    wrapper = output / "kth-qualify"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH="{root / "src"}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec "{sys.executable}" -m kth.provider_cli "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper, kth_sha


def _timeline(summary: dict[str, Any], performance_calls: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "step": 1,
            "event": "micro_validation",
            "outcome": "both candidates published with fixture tolerance evidence",
        }
    ]
    for result in summary["results"]:
        if result["status"] == "kth_blocked":
            events.append(
                {
                    "step": 2,
                    "event": "kth_qualification",
                    "candidate": "round-up drift",
                    "outcome": "Blocked",
                    "detector": result["kth_primary_detector"],
                    "performance_reached": result["performance_reached"],
                    "repair_feedback": result["repair_feedback"],
                }
            )
            events.append(
                {
                    "step": 3,
                    "event": "hyperloom_action",
                    "outcome": "evaluation skipped; patch reverted",
                }
            )
        elif result["status"] == "kept":
            events.append(
                {
                    "step": 4,
                    "event": "kth_qualification",
                    "candidate": "round-to-nearest-even correction",
                    "outcome": result["kth_verdict"],
                    "performance_reached": result["performance_reached"],
                }
            )
            events.append(
                {
                    "step": 5,
                    "event": "hyperloom_action",
                    "outcome": "fixture performance evaluated; KEEP committed",
                }
            )
    events.append(
        {
            "step": 6,
            "event": "invariant",
            "outcome": f"performance validator called {performance_calls} time",
        }
    )
    return events


def _write_terminal_svg(path: Path, events: list[dict[str, Any]]) -> None:
    lines = [
        "$ hyperloom-kth-demo",
        *[f"{event['step']}. {event['event']}: {event['outcome']}" for event in events],
        "$ verdict: completed",
    ]
    text = "\n".join(
        f'<text x="28" y="{48 + index * 27}">{html.escape(line)}</text>' for index, line in enumerate(lines)
    )
    path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="300" '
            'viewBox="0 0 1280 300">'
            '<rect width="1280" height="300" rx="12" fill="#111827"/>'
            '<g fill="#e5e7eb" font-family="monospace" font-size="17">'
            f"{text}</g></svg>\n"
        ),
        encoding="utf-8",
    )


async def _run(kth_root: Path, output: Path) -> dict[str, Any]:
    repo = output / "candidate_repo"
    session = output / "session"
    patches = output / "publication_artifacts"
    repo.mkdir(parents=True)
    session.mkdir()
    _git(repo, "init")
    kernel = repo / KERNEL_PATH
    kernel.parent.mkdir()
    kernel.write_text('ROUND_MODE = "baseline"\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "demo baseline")
    base_commit = _git(repo, "rev-parse", "HEAD")

    _publish(
        patches,
        repo,
        base_commit,
        ordinal="01-drift",
        content='ROUND_MODE = "ceil"\n',
        evidence="aggregate tolerance check passed, but rounding direction drifted",
    )
    _publish(
        patches,
        repo,
        base_commit,
        ordinal="02-corrected",
        content='ROUND_MODE = "rne"\n',
        evidence="round-to-nearest-even restored",
    )

    wrapper, kth_sha = _kth_wrapper(kth_root, output)
    os.environ["KTH_ALLOW_FIXTURE_PLANS"] = "1"
    state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
        framework_repo_path=str(repo),
    )
    state.save(session)
    performance_calls = 0

    async def fixture_performance(_publication) -> dict[str, Any]:
        nonlocal performance_calls
        performance_calls += 1
        assert kernel.read_text(encoding="utf-8") == 'ROUND_MODE = "rne"\n'
        return {
            "decision": "KEEP",
            "new_tput": 107.0,
            "gain_pct": 7.0,
            "performance_evidence": "fixture-labelled",
        }

    integration = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session,
        shared_state=state,
        validator=fixture_performance,
        kth_provider=KthQualificationProvider(
            executable=str(wrapper),
            expected_kth_sha=kth_sha,
        ),
    )
    summary = integration.to_dict()
    timeline = _timeline(summary, performance_calls)
    result = {
        "demo_schema_version": "1.0.0",
        "fixture_performance": True,
        "kth_sha": kth_sha,
        "base_commit": base_commit,
        "final_commit": _git(repo, "rev-parse", "HEAD"),
        "final_kernel": kernel.read_text(encoding="utf-8").strip(),
        "performance_validator_calls": performance_calls,
        "integration": summary,
        "timeline": timeline,
    }
    (output / "timeline.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "terminal_timeline.txt").write_text(
        "\n".join(f"{event['step']}. {event['event']}: {event['outcome']}" for event in timeline) + "\n",
        encoding="utf-8",
    )
    _write_terminal_svg(output / "demo_terminal.svg", timeline)
    if performance_calls != 1 or summary["kept_count"] != 1:
        raise RuntimeError("demo invariants failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kth-root",
        type=Path,
        required=True,
        help="KTH checkout containing the integration/hyperloom-provider branch",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    kth_root = args.kth_root.expanduser().resolve()
    if not (kth_root / "src" / "kth" / "provider_cli.py").is_file():
        parser.error(f"{kth_root} does not contain the KTH provider")
    output = args.out.expanduser().resolve() if args.out else Path(tempfile.mkdtemp(prefix="hyperloom-kth-demo-"))
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(_run(kth_root, output))
    except Exception:
        print(f"Demo failed; artifacts preserved at {output}", file=sys.stderr)
        raise
    print((output / "terminal_timeline.txt").read_text(), end="")
    print(f"Artifacts: {output}")
    print(f"Machine-readable result: {output / 'timeline.json'}")
    print(f"Final verdict: {result['integration']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
