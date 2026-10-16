# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-loop driver that republishes ``kernel.py`` as a SIKL candidate, reports what ``sikl.benchmarks.api``
returns and times nothing itself, and fails a candidate whose GPU work is not kernels ``kernel.py`` defines.
``hyperloom.sikl.workspace`` fills the placeholders below."""

from __future__ import annotations

import argparse
import ast
import hashlib
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
KERNEL_DECORATOR = "@KERNEL_DECORATOR@"
# The definition's own reference and baseline anchor the loop's first measurement; they are not candidates.
STARTING_DIGESTS = frozenset(json.loads("""@STARTING_DIGESTS@"""))

CANDIDATE_AUTHOR = "@CANDIDATE_AUTHOR@"
CANDIDATE_NAME = "@CANDIDATE_NAME@"
KERNEL = Path(__file__).resolve().parent / "kernel.py"
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


def _score(*, warmup: int, iterations: int) -> list[dict]:
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
        # Timing the baseline too doubles a pass, past forge-loop's per-measurement limit; finalize times it once.
        "--no-time-baseline",
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
    """Report SIKL's own per-case verdict, failing any case whose kernels ``kernel.py`` does not define in
    LANGUAGE; the loop gates on ``allclose``."""
    records = _score(warmup=0, iterations=1)
    passed, diffs = {}, []
    for record in records:
        evaluation = record["evaluation"]
        passed[_case_id(record)] = evaluation.get("status") == "PASSED"
        error = (evaluation.get("correctness") or {}).get("max_absolute_error")
        if error is not None:
            diffs.append(float(error))
    if passed and all(passed.values()) and not _is_starting_kernel():
        for case_id, names in _foreign_kernels().items():
            if names:
                print(f"foreign_kernels: {case_id} {' '.join(names)}")
                passed[case_id] = False
    for case_id, case_passed in passed.items():
        print(f"case_allclose: {case_id} {case_passed}")
    # SIKL compares by matched ratio, not SNR, so the loop takes its allclose fallback rather than a made-up SNR.
    print(f"allclose: {bool(passed) and all(passed.values())}")
    print(f"max_diff: {max(diffs) if diffs else 0.0:.6g}")
    return 0


def _is_starting_kernel() -> bool:
    return hashlib.sha256(KERNEL.read_bytes()).hexdigest() in STARTING_DIGESTS


def _dotted(node: ast.expr, imports: dict[str, str]) -> str:
    """The import path an expression such as ``flyc.kernel`` names, or "" if it names no import."""
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in imports:
        return ""
    return ".".join([imports[node.id], *reversed(attributes)])


def _kernels_defined_here(source: str) -> set[str]:
    """Names a profiler shows for the kernels ``source`` defines with KERNEL_DECORATOR: the function's own,
    or a literal ``name=`` that replaces it."""
    tree = ast.parse(source)
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.partition(".")[0]
                imports[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            called = isinstance(decorator, ast.Call)
            if _dotted(decorator.func if called else decorator, imports) != KERNEL_DECORATOR:
                continue
            literals = [
                keyword.value.value
                for keyword in (decorator.keywords if called else [])
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
            ]
            names.add(literals[0] if literals else node.name)
    return names


def _is_defined_here(launched: str, kernels: set[str]) -> bool:
    # FlyDSL suffixes a kernel's name with its index in the module.
    return any(launched == name or launched.startswith(f"{name}_") for name in kernels)


def _foreign_kernels() -> dict[str, list[str]]:
    """Per case, the GPU activity one ``run`` launches that is not a kernel ``kernel.py`` defines."""
    import torch

    kernels = _kernels_defined_here(KERNEL.read_text(encoding="utf-8"))
    definition, workloads = _schema()
    candidate = _load_candidate()
    foreign = {}
    for record in workloads:
        keywords = _inputs(definition, record)
        # Compilation and autotuning launch their own work; the window holds one steady-state call.
        candidate.run(**keywords)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            candidate.run(**keywords)
            torch.cuda.synchronize()
        launched = {event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
        if not launched:
            raise SystemExit(f"the profiler recorded no GPU activity for case {record.workload.uuid}")
        foreign[str(record.workload.uuid)] = sorted(name for name in launched if not _is_defined_here(name, kernels))
    return foreign


def _schema():
    sys.path.insert(0, str(SIKL_ROOT))
    from sikl.core.schema import SchemaCollection

    collection = SchemaCollection.from_path(str(BUNDLE))
    return collection.definitions[DEFINITION], collection.workloads[DEFINITION]


def _inputs(definition, record) -> dict:
    """One workload's inputs from SIKL's own trial construction: inputs from a different draw describe a
    call nobody scores."""
    from sikl.benchmarks.api.config import EvalConfig
    from sikl.benchmarks.api.evaluation.trials import TrialRequest, draw_trials

    request = TrialRequest(
        definition=definition,
        workload=record.workload,
        eval_config=EvalConfig(num_trials=1),
        seed=0,
        schema_collection_root=str(BUNDLE),
    )
    values = draw_trials(request, "cuda").inputs[0]
    return dict(zip(definition.inputs.keys(), values)) if isinstance(values, (list, tuple)) else dict(values)


def _load_candidate():
    spec = importlib.util.spec_from_file_location("forge_candidate", KERNEL)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    return candidate


def _run_bench(warmup: int, iterations: int) -> int:
    """Emit one ``case_ms`` line per workload plus the suite aggregate."""
    records = _score(warmup=warmup, iterations=iterations)
    unmeasured, latencies = [], []
    for record in records:
        case_id = _case_id(record)
        latency = (record["evaluation"].get("performance") or {}).get("latency_ms")
        if latency is None:
            unmeasured.append(f"{case_id} ({record['evaluation'].get('status')})")
            continue
        print(f"case_ms: {case_id} {float(latency):.6f}")
        latencies.append(float(latency))
    if unmeasured:
        sys.stderr.write("cases without a latency: " + ", ".join(unmeasured) + "\n")
        return 1
    # A definition is scored over its whole workload set; ``median_ms`` is the aggregate field the loop reads.
    print(f"median_ms: {sum(latencies):.6f}")
    return 0


def _run_profile() -> int:
    """Dispatch only the candidate on the largest workload, so a profiler sees its kernels."""
    import torch

    definition, workloads = _schema()
    keywords = _inputs(definition, max(workloads, key=lambda record: sum(record.workload.axes.values())))
    candidate = _load_candidate()
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
