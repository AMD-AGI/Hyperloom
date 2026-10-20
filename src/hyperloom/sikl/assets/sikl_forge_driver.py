# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-loop driver that republishes ``kernel.py`` as the kernel of one SIKL schema, reports what
``python -m sikl validate`` returns and times nothing itself, and fails a candidate whose GPU work is not
kernels ``kernel.py`` defines. ``hyperloom.sikl.workspace`` fills the placeholders below."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SIKL_ROOT = Path("@SIKL_ROOT@")
SCHEMA_JSON = Path("@SCHEMA_JSON@")
OP_DIR = "@OP_DIR@"
LANGUAGE = "@LANGUAGE@"
KERNEL_DECORATOR = "@KERNEL_DECORATOR@"
# One validate run per kernel, read by every mode: it times as well as checks, which takes longer than
# forge-loop allows a single benchmark.
VALIDATE_DIR = Path("@VALIDATE_DIR@")
# The run's own starting kernel anchors the loop's first measurement; it is not a candidate.
STARTING_DIGESTS = frozenset(json.loads("""@STARTING_DIGESTS@"""))
KERNEL = Path(__file__).resolve().parent / "kernel.py"

# Findings that mean a workload's output cannot be trusted; NOT_FASTER and REGRESSED only grade speed.
INCORRECT_CODES = frozenset(
    {
        "schema_invalid",
        "kernel_missing",
        "source_rejected",
        "build_failed",
        "candidate_crash",
        "candidate_timeout",
        "capture_unsafe",
        "input_mutation",
        "numeric_mismatch",
        "output_caching",
        "nondeterministic",
        "post_timing_mismatch",
    }
)


def _sikl():
    sys.path.insert(0, str(SIKL_ROOT))
    from sikl.core.contract.locate import schema_relative_path, schema_uuid
    from sikl.core.contract.op import OpContract
    from sikl.core.contract.registry import ContractRegistry

    return ContractRegistry.default(), OpContract, schema_relative_path, schema_uuid


def _candidate_text() -> tuple[str, str]:
    """The schema text SIKL expects for this kernel, and the file name SIKL derives from it."""
    registry, op_contract, _, schema_uuid = _sikl()
    raw = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    raw["kernel"]["language"] = LANGUAGE
    raw["kernel"]["sources"] = [{"path": KERNEL.name, "content": KERNEL.read_text(encoding="utf-8")}]
    contract = registry.for_schema_json(raw)
    point = contract.point_from_json(raw, raw["workloads"][0]["axes"])
    text = contract.narrow(point, workloads=raw["workloads"], kernel=raw["kernel"]).schema_text()
    return text, schema_uuid(raw["name"], op_contract.authored_parts(json.loads(text)))


def _publish() -> Path:
    """Write the candidate where SIKL keeps kernels; return its path relative to the checkout."""
    _, _, schema_relative_path, _ = _sikl()
    text, uuid = _candidate_text()
    raw = json.loads(text)
    path = Path("sikl/ops", OP_DIR, schema_relative_path(raw["name"], LANGUAGE, uuid))
    (SIKL_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (SIKL_ROOT / path).write_text(text, encoding="utf-8")
    return path


def _report() -> dict:
    """SIKL's validation report for the current kernel, running validate once per kernel."""
    out = VALIDATE_DIR / hashlib.sha256(KERNEL.read_bytes()).hexdigest()[:16]
    report = out / "report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    path = _publish()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(SIKL_ROOT), os.environ.get("PYTHONPATH")])))
    command = [sys.executable, "-m", "sikl", "validate", "--stages", "audit,bench", "--devices", "0", "--out", str(out)]
    try:
        completed = subprocess.run([*command, str(path)], cwd=SIKL_ROOT, env=env, text=True, capture_output=True)
    finally:
        # Another kernel's file left in the checkout would be served beside the next one.
        (SIKL_ROOT / path).unlink(missing_ok=True)
    if not report.is_file():
        sys.stderr.write(completed.stdout[-3000:] + completed.stderr[-3000:])
        raise SystemExit(f"sikl validate wrote no report (exit {completed.returncode})")
    return json.loads(report.read_text(encoding="utf-8"))


def _stage(report: dict, name: str) -> dict:
    return next((stage for stage in report["stages"] if stage["stage_name"] == name), {})


def _case_id(axes: dict) -> str:
    return ",".join(f"{name}={value}" for name, value in axes.items())


def _outcomes(report: dict) -> dict[str, dict]:
    """Per workload: whether SIKL found its output wrong, and the candidate's and baseline's graph times."""
    workloads = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))["workloads"]
    outcomes = {
        _case_id(w["axes"]): {"incorrect": [], "candidate_ms": None, "baseline_ms": None, "speedup_bound": None}
        for w in workloads
    }
    for stage in ("audit", "bench"):
        for finding in _stage(report, stage).get("findings", []):
            if finding["code"] not in INCORRECT_CODES:
                continue
            # An audit finding names no workload: it rejects the kernel everywhere.
            cases = [_case_id(finding["axes"])] if finding.get("axes") else list(outcomes)
            for case_id in cases:
                outcomes[case_id]["incorrect"].append(finding["code"])
    for record in _stage(report, "bench").get("workload_records", []):
        outcome = outcomes[_case_id(record["axes"])]
        outcome["violation"] = record.get("worst_violation_fraction")
        for timing in record["timings"]:
            if timing["mode"] == "graph" and timing["versus"] == "baseline":
                outcome["candidate_ms"] = statistics.median(timing["candidate_milliseconds"])
                outcome["baseline_ms"] = statistics.median(timing["versus_milliseconds"])
                outcome["speedup_bound"] = timing["ratio_lower_bound"]
    return outcomes


def _run_correctness() -> int:
    """Report SIKL's per-workload verdict, failing any workload whose kernels ``kernel.py`` does not define in
    LANGUAGE; the loop gates on ``allclose``."""
    outcomes = _outcomes(_report())
    passed = {case_id: not outcome["incorrect"] for case_id, outcome in outcomes.items()}
    for case_id, outcome in outcomes.items():
        if outcome["incorrect"]:
            print(f"sikl_findings: {case_id} {' '.join(sorted(set(outcome['incorrect'])))}")
    if all(passed.values()) and not _is_starting_kernel():
        for case_id, names in _foreign_kernels().items():
            if names:
                print(f"foreign_kernels: {case_id} {' '.join(names)}")
                passed[case_id] = False
    for case_id, case_passed in passed.items():
        print(f"case_allclose: {case_id} {case_passed}")
    # SIKL grades the fraction of elements outside the error bound, not SNR, so the loop takes its allclose path.
    violations = [outcome.get("violation") or 0.0 for outcome in outcomes.values()]
    print(f"allclose: {bool(passed) and all(passed.values())}")
    print(f"max_diff: {max(violations, default=0.0):.6g}")
    return 0


def _run_bench() -> int:
    """Emit one ``case_ms`` line per workload plus the suite aggregate."""
    outcomes = _outcomes(_report())
    unmeasured = [case_id for case_id, outcome in outcomes.items() if outcome["candidate_ms"] is None]
    for case_id, outcome in outcomes.items():
        if outcome["candidate_ms"] is not None:
            print(f"case_ms: {case_id} {outcome['candidate_ms']:.6f}")
    if unmeasured:
        sys.stderr.write("workloads SIKL did not time: " + ", ".join(unmeasured) + "\n")
        return 1
    # A schema is served workload by workload, but the loop reads one aggregate: the whole set's time.
    print(f"median_ms: {sum(outcome['candidate_ms'] for outcome in outcomes.values()):.6f}")
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


def _calls() -> dict[str, dict]:
    """One call per workload, built by SIKL's own trial code: inputs from another draw describe a call
    nobody scores."""
    import torch

    registry, *_ = _sikl()
    from sikl.core.contract import semantics
    from sikl.core.validation.stages._bench._evaluation._trials import TrialRequest

    text, _ = _candidate_text()
    schema = registry.parse_schema(text.encode(), str(SCHEMA_JSON))
    contract = registry.get(schema.contract_id)
    calls = {}
    for workload in schema.workloads:
        request = TrialRequest.of_workload(
            str(SCHEMA_JSON), schema, contract, workload, thresholds={}, incumbent=None, baseline=None
        )
        call = request.assemble_call(request.compose_inputs(None), "cuda")
        for name in contract.allocated_outputs:
            dtype = semantics.torch_dtypes(request.point.dtypes[name])[0]
            call[name] = torch.empty(contract.storage_shape_of(name, request.point), dtype=dtype, device="cuda")
        calls[_case_id(workload.axes)] = call
    return calls


def _load_candidate():
    spec = importlib.util.spec_from_file_location("forge_candidate", KERNEL)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    return candidate


def _foreign_kernels() -> dict[str, list[str]]:
    """Per workload, the GPU activity one ``run`` launches that is not a kernel ``kernel.py`` defines."""
    import torch

    kernels = _kernels_defined_here(KERNEL.read_text(encoding="utf-8"))
    candidate = _load_candidate()
    foreign = {}
    for case_id, call in _calls().items():
        # Compilation launches its own work; the window holds one steady-state call.
        candidate.run(**call)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            candidate.run(**call)
            torch.cuda.synchronize()
        launched = {event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
        if not launched:
            raise SystemExit(f"the profiler recorded no GPU activity for workload {case_id}")
        foreign[case_id] = sorted(name for name in launched if not _is_defined_here(name, kernels))
    return foreign


def _run_profile() -> int:
    """Dispatch only the candidate on the largest workload, so a profiler sees its kernels."""
    import torch

    calls = _calls()
    largest = max(calls, key=lambda case_id: sum(int(part.split("=")[1]) for part in case_id.split(",")))
    candidate = _load_candidate()
    for _ in range(6):
        candidate.run(**calls[largest])
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"forge-loop driver for {SCHEMA_JSON.stem}")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--outcomes-json", type=Path, help="write SIKL's per-workload outcome for the summary")
    # forge-loop passes its own sampling flags; SIKL decides its own.
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    args, _unknown = parser.parse_known_args()
    if args.profile_run:
        return _run_profile()
    if args.outcomes_json:
        args.outcomes_json.write_text(json.dumps(_outcomes(_report()), indent=2), encoding="utf-8")
        return 0
    if args.bench_mode:
        return _run_bench()
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
