# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-loop driver that republishes ``kernel.py`` as a SIKL candidate and reports what
``sikl.benchmarks.api`` returns, the code SIKL's acceptance gate runs; it measures nothing itself.
``hyperloom.sikl.workspace`` fills the placeholders below."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

BUNDLE = Path("@BUNDLE@")
SIKL_ROOT = Path("@SIKL_ROOT@")
DEFINITION = "@DEFINITION@"
OP_TYPE = "@OP_TYPE@"
TARGET = json.loads("""@TARGET@""")
LANGUAGE = "@LANGUAGE@"

CANDIDATE_AUTHOR = "@CANDIDATE_AUTHOR@"
CANDIDATE_NAME = "@CANDIDATE_NAME@"
KERNEL = Path(__file__).resolve().parent / "kernel.py"
# Outside the workspace: a file rewritten there on every measurement reads to the guard as tampering.
BASELINE_RECORD = BUNDLE.parent / "sikl_in_window_baseline.json"
# A kernel may not load under another version of its language, so the candidate records the installed one.
LANGUAGE_DISTRIBUTIONS = {"flydsl": "flydsl", "triton": "triton", "gluon": "triton"}


def _dependencies() -> list[str]:
    distribution = LANGUAGE_DISTRIBUTIONS.get(LANGUAGE)
    if distribution is None:
        return []
    try:
        return [f"{distribution}=={metadata.version(distribution)}"]
    except metadata.PackageNotFoundError:
        return []


def _publish_candidate() -> None:
    """Rewrite the loop's current kernel as this definition's candidate solution."""
    solution = {
        "name": CANDIDATE_NAME,
        "definition": DEFINITION,
        "author": CANDIDATE_AUTHOR,
        "spec": {
            "language": LANGUAGE,
            "target": TARGET,
            "entry_point": "kernel.py::run",
            "dependencies": _dependencies(),
            "destination_passing_style": False,
            "binding": None,
        },
        "sources": [{"path": "kernel.py", "content": KERNEL.read_text()}],
        "description": f"forge-loop candidate for {DEFINITION}.",
    }
    path = BUNDLE / "solutions" / CANDIDATE_AUTHOR / OP_TYPE / DEFINITION / f"{CANDIDATE_NAME}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(solution), encoding="utf-8")


def _score(*, warmup: int, iterations: int, time_baseline: bool) -> list[dict]:
    """Run the harness over every workload of this definition and return its records."""
    _publish_candidate()
    handle, report_name = tempfile.mkstemp(suffix=".json", prefix="sikl_forge_")
    os.close(handle)
    report = Path(report_name)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SIKL_ROOT), env.get("PYTHONPATH", "")]))
    command = [
        sys.executable,
        "-m",
        "sikl.benchmarks.api",
        str(BUNDLE),
        "--definition",
        DEFINITION,
        "--solution",
        CANDIDATE_NAME,
        "--warmup-runs",
        str(warmup),
        "--iterations",
        str(iterations),
        "--num-trials",
        "1",
        "--time-baseline" if time_baseline else "--no-time-baseline",
        "--json-report",
        str(report),
    ]
    try:
        completed = subprocess.run(command, cwd=SIKL_ROOT, env=env, text=True, capture_output=True)
        # A candidate that does not qualify exits nonzero; only a missing report means the run did not happen.
        if not report.stat().st_size:
            sys.stderr.write(completed.stdout[-2000:] + completed.stderr[-2000:])
            raise SystemExit(f"sikl.benchmarks.api wrote no report (exit {completed.returncode})")
        return json.loads(report.read_text(encoding="utf-8"))
    finally:
        report.unlink(missing_ok=True)


def _case_id(record: dict) -> str:
    return str(record["workload"]["uuid"])


def _run_correctness() -> int:
    """Report SIKL's own per-case verdict; the loop gates on ``allclose``."""
    records = _score(warmup=0, iterations=1, time_baseline=False)
    passed, diffs = [], []
    for record in records:
        evaluation = record["evaluation"]
        case_passed = evaluation.get("status") == "PASSED"
        passed.append(case_passed)
        print(f"case_allclose: {_case_id(record)} {case_passed}")
        error = (evaluation.get("correctness") or {}).get("max_absolute_error")
        if error is not None:
            diffs.append(float(error))
    # SIKL compares by matched ratio, not SNR, so the loop takes its allclose fallback rather than a made-up SNR.
    print(f"allclose: {bool(passed) and all(passed)}")
    print(f"max_diff: {max(diffs) if diffs else 0.0:.6g}")
    return 0


def _run_bench(warmup: int, iterations: int) -> int:
    """Emit one ``case_ms`` line per workload plus the suite aggregate."""
    records = _score(warmup=warmup, iterations=iterations, time_baseline=True)
    baseline, unmeasured, latencies = {}, [], []
    for record in records:
        case_id = _case_id(record)
        performance = record["evaluation"].get("performance") or {}
        latency = performance.get("latency_ms")
        if latency is None:
            unmeasured.append(f"{case_id} ({record['evaluation'].get('status')})")
            continue
        print(f"case_ms: {case_id} {float(latency):.6f}")
        latencies.append(float(latency))
        reference = performance.get("reference_latency_ms")
        if reference is not None:
            baseline[case_id] = float(reference)
    if unmeasured:
        sys.stderr.write("cases without a latency: " + ", ".join(unmeasured) + "\n")
        return 1
    # A definition is scored over its whole workload set; ``median_ms`` is the aggregate field the loop reads.
    print(f"median_ms: {sum(latencies):.6f}")
    # The loop's anchor is measured once; SIKL's same-window baseline shows afterwards whether it drifted.
    BASELINE_RECORD.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return 0


def _run_profile() -> int:
    """Dispatch only the candidate on the largest workload, so a profiler sees its kernels. Inputs come
    from SIKL's own trial construction: a profile over a different draw describes a kernel nobody scores."""
    sys.path.insert(0, str(SIKL_ROOT))
    import torch
    from sikl.benchmarks.api.config import EvalConfig
    from sikl.benchmarks.api.evaluation.trials import TrialRequest, draw_trials
    from sikl.core.schema import SchemaCollection

    collection = SchemaCollection.from_path(str(BUNDLE))
    definition = collection.definitions[DEFINITION]
    workload = max(collection.workloads[DEFINITION], key=lambda record: sum(record.workload.axes.values()))
    request = TrialRequest(
        definition=definition,
        workload=workload.workload,
        eval_config=EvalConfig(num_trials=1),
        seed=0,
        schema_collection_root=str(BUNDLE),
    )
    values = draw_trials(request, "cuda").inputs[0]
    keywords = dict(zip(definition.inputs.keys(), values)) if isinstance(values, (list, tuple)) else dict(values)

    spec = importlib.util.spec_from_file_location("forge_candidate", KERNEL)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    for _ in range(6):
        candidate.run(**keywords)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"forge-loop driver for {DEFINITION}")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _unknown = parser.parse_known_args()
    if args.profile_run:
        return _run_profile()
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters)
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
