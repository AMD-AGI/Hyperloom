"""The gate pipeline: integrity → correctness → read bench output → decide.

The harness never runs the bench. Claude owns the bench — it runs the bench script
via Bash, watches output, debugs failures, waits as long as needed. The gate only
reads bench_output.json from disk after Claude reports DONE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arbor.execute.bench import run_correctness
from arbor.measure.gate import GateResult, run_gate
from arbor.measure.integrity import snapshot_digests, verify_unchanged


def _read_bench_output(iter_dir: Path) -> dict:
    """Read bench_output.json from the iteration directory. No subprocess, no timeout."""
    bench_path = iter_dir / "bench_output.json"
    if not bench_path.exists():
        return {"status": "failed", "error": "bench_output.json not found — agent must run bench and write output before calling gate"}
    try:
        raw = json.loads(bench_path.read_text())
    except json.JSONDecodeError as e:
        return {"status": "failed", "error": f"invalid JSON in bench_output.json: {e}"}
    raw.setdefault("status", "ok")
    return raw


def run_full_gate(
    campaign,
    state,
    session_dir: str | Path,
    iteration: int,
    extra_env: dict[str, str] | None = None,
    reported_metric: float | None = None,
    accuracy: dict | None = None,
) -> GateResult:
    """Integrity → correctness → read bench output → decide.

    The harness never runs the bench and never imposes a timeout on it.
    Claude runs bench, writes bench_output.json, then calls the gate.
    If Claude reported a metric directly (DONE: metric=X), the harness
    uses that. Otherwise it reads bench_output.json from the iter dir.
    Correctness is always run independently — the tamper-evident check.
    ``accuracy`` is a full-set eval of a numerics change, as the loop passes it
    (arbor/measure/accuracy.py).
    """
    session_dir = Path(session_dir)
    iter_dir = session_dir / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    config = campaign.config

    # 1. Integrity — validator files unchanged since campaign start
    if not state.validator_digests:
        print(
            "[arbor] INTEGRITY ERROR: validator_digests is empty. "
            "Re-run 'arbor iterate init' with the container running.",
            file=sys.stderr,
        )
        return GateResult(
            "REVERT", 0.0, state.best_metric, 0.0,
            False, False,
            "validator_digests empty — init must be re-run with container running",
        )
    container = config["docker_container"]
    current = snapshot_digests(container, list(state.validator_digests.keys()))
    integrity = verify_unchanged(state.validator_digests, current)
    if not integrity.ok:
        return GateResult(
            "REVERT", 0.0, state.best_metric, 0.0,
            False, False, f"integrity: {integrity.reason}",
        )

    # 2. Correctness — always run independently (tamper-evident)
    correctness = run_correctness(config, str(iter_dir), extra_env=extra_env)

    # 3. Bench result — read from disk, never run by the harness
    if reported_metric is not None:
        metric_key = campaign.pack.objective.get("metric", "output_throughput")
        bench_result = {
            "status": "ok",
            metric_key: reported_metric,
            "source": "claude_reported",
        }
        print(f"[arbor] Bench: Claude-reported metric={reported_metric}", file=sys.stderr)
    else:
        bench_result = _read_bench_output(iter_dir)
        if bench_result.get("status") == "ok":
            print(f"[arbor] Bench: read from {iter_dir}/bench_output.json", file=sys.stderr)
        else:
            print(f"[arbor] Bench: {bench_result.get('error', 'failed')}", file=sys.stderr)

    # 4. Gate decision
    return run_gate(
        campaign.pack.objective,
        state.best_metric,
        bench_result,
        correctness,
        integrity.ok,
        accuracy=accuracy,
    )
