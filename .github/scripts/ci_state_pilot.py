# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded, paired CPU pilot; never a claim about full-workflow acceleration."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

BASELINE = "09d1ff5649f187b6c9d0322bf8e6441fde5ff0cb"
STATE = "src/hyperloom/orchestrator/state/shared_state.py"
HELPER = "src/hyperloom/common/dataclass_serde.py"
CONTRACTS = "src/hyperloom/common/tests/test_dataclass_serde.py"
TEST_ROOT = "src/hyperloom/inference_optimizer/tests/"
TESTS = (
    CONTRACTS,
    TEST_ROOT + "test_shared_state_persistence.py::test_save_load_round_trip",
    TEST_ROOT + "test_shared_state_persistence.py::test_save_is_atomic",
    TEST_ROOT + "test_phase_state_machine.py::test_record_phase_transition_writes_row_and_updates_phase",
    TEST_ROOT + "test_promote_shared_state_lock.py::test_promote_baseline_writes_state_and_audit",
    TEST_ROOT + "test_promote_shared_state_lock.py::test_lift_does_not_double_append_same_fingerprint",
    TEST_ROOT + "test_shared_state_evolution.py::test_enablement_accepted_config_path_roundtrips",
    TEST_ROOT + "test_shared_state_evolution.py::test_v4_nested_enablement_roundtrips",
)
WALKTHROUGH = (
    TEST_ROOT
    + "test_optimize_loop_walkthrough.py::test_a_baseline_carries_the_run_into_the_optimisation_phase_with_work"
)
RUN_BUDGET_SECONDS = 780
PAIRS = ("AB", "BA", "AB")
MICRO_PAIRS = ("AB", "BA", "AB", "BA", "AB")
MICRO_ITERATIONS = 30
MANIFEST = "ci-state-pilot.json"
ATOMIC_TYPES = (type(None), bool, int, float, str, bytes, complex)


class Incomplete(RuntimeError):
    """The pilot cannot provide valid evidence."""


class SemanticFailure(RuntimeError):
    """A measured semantic contract failed."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
    ).stdout


def check_callsite_only(baseline: bytes, candidate: bytes) -> None:
    """Allow only the serializer import and SharedState.to_dict call replacement."""
    before, after = ast.parse(baseline), ast.parse(candidate)
    replacements = 0
    for tree in (before, after):
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "hyperloom.common.dataclass_serde")
        ]
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                node.names = [alias for alias in node.names if alias.name != "asdict"]
    for node in ast.walk(after):
        if isinstance(node, ast.ClassDef) and node.name == "SharedState":
            method = next(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "to_dict"
            )
            for call in ast.walk(method):
                if isinstance(call, ast.Name) and call.id == "fast_asdict":
                    call.id = "asdict"
                    replacements += 1
    if replacements != 1 or ast.dump(before) != ast.dump(after):
        raise Incomplete("Candidate changes more than the serializer import and SharedState.to_dict call")


def prepare(subject: Path) -> None:
    facility = Path(__file__).resolve().parents[2]
    if subject.exists() or subject.is_relative_to(facility) or facility.is_relative_to(subject):
        raise Incomplete("Subject must be a new disposable directory outside the facility checkout")
    candidate = git(facility, "rev-parse", "HEAD").decode().strip()
    blobs = {name: git(facility, "show", f"{candidate}:{name}") for name in (STATE, HELPER, CONTRACTS)}
    baseline_state = git(facility, "show", f"{BASELINE}:{STATE}")
    check_callsite_only(baseline_state, blobs[STATE])
    subject.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(facility), str(subject)], check=True, timeout=60
    )
    git(subject, "-c", "core.autocrlf=false", "checkout", "--detach", BASELINE)
    names = git(subject, "ls-files", "-z").decode().split("\0")
    hashes = {name: digest((subject / name).read_bytes()) for name in names if name}
    for name in (HELPER, CONTRACTS):
        if name in hashes:
            raise Incomplete(f"Unexpected helper/test already present in baseline: {name}")
        (subject / name).parent.mkdir(parents=True, exist_ok=True)
        (subject / name).write_bytes(blobs[name])
        hashes[name] = digest(blobs[name])
    manifest = {
        "baseline": BASELINE,
        "candidate": candidate,
        "subject": str(subject),
        "facility": str(facility),
        "source_hashes": hashes,
        "arm_hashes": {"A": digest(baseline_state), "B": digest(blobs[STATE])},
    }
    # Arm blobs and measurement metadata stay in .git, never in source discovery.
    for arm, blob in (("A", baseline_state), ("B", blobs[STATE])):
        (subject / ".git" / f"ci-state-{arm}.blob").write_bytes(blob)
    dump(subject / ".git" / MANIFEST, manifest)
    print(json.dumps({"prepared": str(subject), "baseline": BASELINE, "candidate": candidate}))


def source_guard(subject: Path, manifest: dict, arm: str) -> str:
    mismatches = []
    for name, expected in manifest["source_hashes"].items():
        expected = manifest["arm_hashes"][arm] if name == STATE else expected
        path = subject / name
        if not path.is_file() or digest(path.read_bytes()) != expected:
            mismatches.append(name)
    if mismatches:
        raise Incomplete(f"Source guard failed: {mismatches}")
    return digest(json.dumps(manifest["source_hashes"], sort_keys=True).encode())


def switch_arm(subject: Path, manifest: dict, arm: str) -> None:
    current_hash = digest((subject / STATE).read_bytes())
    previous = next((key for key, value in manifest["arm_hashes"].items() if value == current_hash), None)
    if previous is None:
        raise Incomplete("Refusing to overwrite an unrecognized SharedState source")
    source_guard(subject, manifest, previous)
    blob = (subject / ".git" / f"ci-state-{arm}.blob").read_bytes()
    if digest(blob) != manifest["arm_hashes"][arm]:
        raise Incomplete("Stored arm blob changed")
    (subject / STATE).write_bytes(blob)
    # Only delete this disposable subject's bytecode; stale same-size/same-second
    # .pyc files otherwise make a source hash insufficient as an import guard.
    for directory in (subject / "src").rglob("__pycache__"):
        shutil.rmtree(directory)
    source_guard(subject, manifest, arm)


def import_probe(subject: Path, expected: str) -> dict:
    from hyperloom.orchestrator.state import shared_state
    from hyperloom.common import dataclass_serde

    actual = Path(shared_state.__file__).resolve()
    helper_path = Path(dataclass_serde.__file__).resolve()
    if actual != subject / STATE or helper_path != subject / HELPER:
        raise Incomplete(f"Wrong editable import: {actual}; helper: {helper_path}")
    if digest(actual.read_bytes()) != expected:
        raise Incomplete("Imported SharedState file has the wrong arm hash")
    return {"shared_state": str(actual), "helper": str(helper_path), "sha256": expected, "python": sys.executable}


def environment_metadata() -> dict:
    selected = ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH", "PYTHONHASHSEED", "LANG", "LC_ALL")
    memory = None
    if hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "python": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory_bytes": memory,
        "environment": {key: os.environ[key] for key in selected if key in os.environ},
        "temp_environment_changed": False,
        "distributions": sorted(
            ({"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions()),
            key=lambda item: item["name"].lower(),
        ),
    }


def rich_state():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(
        session_id="serde-pilot",
        start_ts="2026-09-22T12:00:00+00:00",
        framework="sglang",
        phase="FRAMEWORK_AGENT",
        baseline_tput=100.0,
        current_best={
            "tput": 110.0,
            "extra_server_args": "--chunked-prefill-size 4096",
            "extra_envs": {"TEST_MODE": "1"},
        },
        attempts=[
            {"task_id": f"task-{index}", "metrics": {"samples": [1.25, 2.5, 3.75]}, "kept": index % 2 == 0}
            for index in range(120)
        ],
        optimization_stack=[
            {"action": "explore", "params": {"extra_envs": {"TEST_MODE": str(index)}}, "gain_pct": 0.5}
            for index in range(24)
        ],
        phase_history=[{"from_phase": "PRELUDE", "to_phase": "FRAMEWORK_AGENT", "ts": "2026-09-22T12:00:00+00:00"}],
        explore_search={"tested": {}, "accepted": [], "rejected": []},
    )
    state.enablement.kept_rounds = [{"patches": ["fix.patch"], "artifacts": [{"path": "out.so"}]}]
    state.enablement.accepted_config = {"extra_envs": {"TEST_MODE": "1"}}
    state.enablement.active_runtime = {"task_id": "enable-1", "elapsed_sec": 1.5}
    return state


def type_census(value: object, counts: Counter) -> None:
    kind = type(value)
    if kind in ATOMIC_TYPES:
        counts["atomic_leaves"] += 1
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        counts["dataclasses"] += 1
        for field in dataclasses.fields(value):
            type_census(getattr(value, field.name), counts)
    elif kind is dict:
        counts["dicts"] += 1
        for key, item in value.items():
            type_census(key, counts)
            type_census(item, counts)
    elif kind in (list, tuple):
        counts[kind.__name__ + "s"] += 1
        for item in value:
            type_census(item, counts)
    else:
        counts["fallback_roots"] += 1
        counts[f"fallback_type:{kind.__module__}.{kind.__qualname__}"] += 1


def paired_statistics(samples: list[dict]) -> dict:
    arms = {arm: [sample[arm] for sample in samples] for arm in "AB"}
    medians = {arm: statistics.median(values) for arm, values in arms.items()}
    ranges = {arm: max(values) - min(values) for arm, values in arms.items()}
    difference = medians["A"] - medians["B"]
    return {
        "samples_seconds": samples,
        "median_seconds": medians,
        "range_seconds": ranges,
        "paired_a_minus_b_seconds": [sample["A"] - sample["B"] for sample in samples],
        "median_a_minus_b_seconds": difference,
        "directional_gate": all(sample["B"] < sample["A"] for sample in samples) and difference > max(ranges.values()),
    }


def microbenchmark(output: Path) -> None:
    from hyperloom.common.dataclass_serde import fast_asdict

    started = time.perf_counter()
    state = rich_state()
    construction = time.perf_counter() - started
    reference = dataclasses.asdict(state)
    encoded = json.dumps(reference, indent=2, sort_keys=True).encode()
    census = Counter()
    type_census(state, census)
    functions = {"A": dataclasses.asdict, "B": fast_asdict}
    for function in functions.values():
        result = function(state)
        if result != reference or json.dumps(result, indent=2, sort_keys=True).encode() != encoded:
            raise SemanticFailure("Micro input differs by serializer or JSON bytes")
        result["attempts"][0]["metrics"]["samples"].append(-1)
        if dataclasses.asdict(state) != reference:
            raise SemanticFailure("Serializer leaked a mutable alias into its source")
    samples = []
    report = {
        "status": "incomplete",
        "construction_seconds": construction,
        "fixture": "deterministic SharedState: 120 attempts, 24 stack entries, nested EnablementRound",
        "iterations_per_sample": MICRO_ITERATIONS,
        "json_bytes": len(encoded),
        "json_sha256": digest(encoded),
        "type_census": dict(census),
        "fallback_reporting": "type census outside timing; actual exotic fallbacks exercised by contracts",
        "measurement": "conversion calls only; no save, JSON encoding, construction or instrumentation in timer",
    }
    dump(output, report)
    for order in MICRO_PAIRS:
        sample = {"order": order}
        for arm in order:
            function = functions[arm]
            elapsed = 0.0
            for _ in range(MICRO_ITERATIONS):
                started = time.perf_counter()
                result = function(state)
                elapsed += time.perf_counter() - started
                if result != reference or json.dumps(result, indent=2, sort_keys=True).encode() != encoded:
                    raise SemanticFailure("Micro output changed during sampling")
                result["attempts"][0]["metrics"]["samples"].append(-1)
                if dataclasses.asdict(state) != reference:
                    raise SemanticFailure("Micro input changed or mutable alias escaped during sampling")
            sample[arm] = elapsed
        samples.append(sample)
        report["statistics"] = paired_statistics(samples)
        dump(output, report)
    report["status"] = "passed"
    dump(output, report)


class Measurements:
    def __init__(self, subject: Path, output: Path, python: str):
        self.subject, self.output, self.python = subject, output, python
        self.started = time.perf_counter()
        self.deadline = self.started + RUN_BUDGET_SECONDS
        self.env = dict(os.environ)
        for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            self.env.pop(key, None)
        self.env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        self.summary = {
            "status": "incomplete",
            "commands": [],
            "budget_seconds": RUN_BUDGET_SECONDS,
            "scope": "Directional pilot only, not full CI or coverage evidence",
            "walkthrough": WALKTHROUGH,
            "walkthrough_internal_passes": 300,
            "pass_count_basis": "source-derived sum(1..24): Coordinator.tick(n) performs n passes; not instrumented",
            "no_automatic_retry": True,
        }
        self.save()

    def save(self) -> None:
        self.summary["elapsed_seconds"] = time.perf_counter() - self.started
        dump(self.output / "summary.json", self.summary)

    def command(self, label: str, command: list[str], timeout: float) -> dict:
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise Incomplete("Total run budget exhausted; no retry or additional samples")
        number = len(self.summary["commands"])
        prefix = self.output / f"{number:02d}-{label}"
        record = {
            "label": label,
            "command": command,
            "status": "running",
            "timeout_seconds": min(timeout, remaining),
            "stdout": str(prefix.with_suffix(".stdout.log")),
            "stderr": str(prefix.with_suffix(".stderr.log")),
        }
        self.summary["commands"].append(record)
        self.save()
        print(f"Starting {label} (limit {record['timeout_seconds']:.1f}s)", flush=True)
        started = time.perf_counter()
        process = None
        try:
            with open(record["stdout"], "wb") as stdout, open(record["stderr"], "wb") as stderr:
                process = subprocess.Popen(
                    command, cwd=self.subject, env=self.env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                try:
                    record["returncode"] = process.wait(timeout=record["timeout_seconds"])
                    record["status"] = "passed" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout"
        except OSError as error:
            record.update(status="launch_error", error=str(error))
            raise
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
                record["returncode"] = process.returncode
            record["wall_seconds"] = time.perf_counter() - started
            self.save()
        if record["status"] == "timeout":
            raise Incomplete(f"Command timed out: {label}")
        return record

    def internal(self, mode: str, destination: Path, *extra: str, timeout: int = 30) -> dict:
        command = [self.python, "-B", str(Path(__file__).resolve()), mode, "--output", str(destination), *extra]
        record = self.command(mode.lstrip("_"), command, timeout)
        if record["returncode"] == 2:
            raise SemanticFailure(f"Target semantic check failed: {mode}; see raw logs")
        if record["returncode"] != 0:
            raise Incomplete(f"Target subprocess failed: {mode}; see raw logs")
        return json.loads(destination.read_text(encoding="utf-8"))

    def pytest(self, label: str, nodes: tuple[str, ...], *, collect: bool = False, timeout: int = 150) -> dict:
        scratch = self.output / "scratch" / f"run-{len(self.summary['commands']):02d}"
        scratch.mkdir(parents=True)
        command = [
            self.python,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "pytest_asyncio.plugin",
            "-o",
            "addopts=",
            "-o",
            f"cache_dir={scratch / 'cache'}",
            "--basetemp",
            str(scratch / "tmp"),
        ]
        if collect:
            command.append("--collect-only")
        else:
            command.extend(["-o", "junit_family=xunit2", f"--junitxml={self.output / (label + '.xml')}"])
        return self.command(label, command + list(nodes), timeout)


def collected_nodes(record: dict) -> list[str]:
    if record["returncode"] != 0:
        raise Incomplete("Test collection failed")
    nodes = [
        line.strip()
        for line in Path(record["stdout"]).read_text(encoding="utf-8").splitlines()
        if line.startswith("src/") and "::" in line
    ]
    if len(set(nodes)) != len(nodes) or not nodes:
        raise Incomplete("Empty or duplicate collected test catalogue")
    if not all(node in nodes for node in TESTS[1:]) or not any(node.startswith(CONTRACTS + "::") for node in nodes):
        raise Incomplete("Required regression or helper contract absent from catalogue")
    return nodes


def parse_junit(path: Path, expected: list[str]) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        issues = [
            {"kind": issue.tag, "message": issue.get("message", ""), "text": issue.text or ""}
            for issue in case
            if issue.tag in ("failure", "error", "skipped")
        ]
        cases.append(
            {
                "classname": case.get("classname", ""),
                "name": case.get("name", ""),
                "issues": issues,
                "seconds": float(case.get("time", "0")),
            }
        )
    expected_keys = Counter()
    for node in expected:
        parts = node.split("::")
        expected_keys[(".".join([parts[0][:-3].replace("/", "."), *parts[1:-1]]), parts[-1])] += 1
    actual_keys = Counter((case["classname"], case["name"]) for case in cases)
    counts = Counter(issue["kind"] for case in cases for issue in case["issues"])
    counts["tests"] = len(cases)
    counts["passed"] = sum(not case["issues"] for case in cases)
    for key in ("failure", "error", "skipped"):
        counts.setdefault(key, 0)
    return {
        "counts": dict(counts),
        "cases": cases,
        "catalogue_matches": actual_keys == expected_keys,
        "all_passed": bool(cases) and not any(case["issues"] for case in cases) and actual_keys == expected_keys,
    }


def checked_pytest(
    measure: Measurements,
    manifest: dict,
    arm: str,
    label: str,
    nodes: tuple[str, ...],
    expected: list[str],
    timeout: int,
) -> float:
    switch_arm(measure.subject, manifest, arm)
    probe = measure.internal(
        "_probe",
        measure.output / f"{label}-import.json",
        "--subject",
        str(measure.subject),
        "--expected",
        manifest["arm_hashes"][arm],
    )
    record = measure.pytest(label, nodes, timeout=timeout)
    record["import_probe"] = probe
    xml_path = measure.output / f"{label}.xml"
    if xml_path.exists():
        record["junit"] = parse_junit(xml_path, expected)
    measure.save()
    source_guard(measure.subject, manifest, arm)
    if "junit" not in record:
        raise Incomplete(f"Missing JUnit: {label}")
    counts = record["junit"]["counts"]
    if counts["failure"] or counts["error"]:
        raise SemanticFailure(f"Regression failed: {label}; exact JUnit failure/error details preserved")
    if record["returncode"] != 0 or not record["junit"]["all_passed"]:
        raise Incomplete(f"Skipped, incomplete, mismatched catalogue, or unsuccessful pytest: {label}")
    return record["wall_seconds"]


def execute_pilot(measure: Measurements, manifest: dict) -> None:
    measure.summary["baseline"] = manifest["baseline"]
    measure.summary["candidate"] = manifest["candidate"]
    measure.summary["arm_hashes"] = manifest["arm_hashes"]
    measure.summary["source_manifest_sha256"] = digest(json.dumps(manifest["source_hashes"], sort_keys=True).encode())
    metadata = measure.internal("_metadata", measure.output / "environment.json")
    measure.summary["environment"] = metadata
    catalogues = {}
    for arm in "AB":
        switch_arm(measure.subject, manifest, arm)
        measure.internal(
            "_probe",
            measure.output / f"collect-{arm}-import.json",
            "--subject",
            str(measure.subject),
            "--expected",
            manifest["arm_hashes"][arm],
        )
        record = measure.pytest(f"collect-{arm}", TESTS, collect=True, timeout=60)
        catalogues[arm] = collected_nodes(record)
        source_guard(measure.subject, manifest, arm)
        measure.summary["catalogues"] = catalogues
        measure.save()
    if catalogues["A"] != catalogues["B"]:
        raise Incomplete("A/B collected test catalogues differ")
    for arm in "AB":
        checked_pytest(measure, manifest, arm, f"contracts-{arm}", TESTS, catalogues[arm], timeout=150)
    measure.summary["contracts"] = "passed"
    measure.summary["save_count_evidence"] = {
        "nodeid": CONTRACTS + "::test_shared_state_save_keeps_atomic_write_count_options_and_bytes",
        "method": "Both external arms run the original contract: three saves per internal serializer, one atomic write per save",
        "writes_per_internal_serializer": 3,
        "atomic_write_options": {"indent": 2, "sort_keys": True},
        "bytes_equal": True,
        "clock": "fixed 115, initial anchor 100, initial elapsed 5; elapsed remains 20 after first save",
        "walkthrough_save_count": None,
        "walkthrough_save_count_note": "Not instrumented; prior profile count is not treated as a measurement here",
    }
    switch_arm(measure.subject, manifest, "B")
    measure.summary["micro"] = measure.internal(
        "_micro",
        measure.output / "micro.json",
        "--subject",
        str(measure.subject),
        "--expected",
        manifest["arm_hashes"]["B"],
        timeout=60,
    )
    source_guard(measure.subject, manifest, "B")
    if measure.summary["micro"]["status"] != "passed":
        raise Incomplete("Microbenchmark did not finish")
    measure.save()
    samples = []
    for pair_index, order in enumerate(PAIRS, 1):
        sample = {"order": order}
        for arm in order:
            sample[arm] = checked_pytest(
                measure, manifest, arm, f"walk-{pair_index}-{arm}", (WALKTHROUGH,), [WALKTHROUGH], timeout=150
            )
            measure.summary["walkthrough_partial_pair"] = sample.copy()
            measure.save()
        samples.append(sample)
        measure.summary["walkthrough_statistics"] = paired_statistics(samples)
        measure.save()
    supported = measure.summary["walkthrough_statistics"]["directional_gate"]
    measure.summary["status"] = "supported" if supported else "not_supported"
    measure.summary["reason"] = (
        "All semantic checks passed; all three B samples beat A and median improvement exceeds both arm ranges"
        if supported
        else "Semantic checks passed, but the preregistered directional/noise gate was not met"
    )
    measure.summary["next_step"] = (
        "Request a separately approved full-CI experiment; no extrapolated speedup claim"
        if supported
        else "Stop: no automatic extra samples or full-CI run"
    )


def run(subject: Path, output: Path, python: str) -> int:
    facility = Path(__file__).resolve().parents[2]
    if output.exists() or output.is_relative_to(subject) or output.is_relative_to(facility):
        raise Incomplete("Output must be a new directory outside subject and facility")
    if str(output) != str(output).lower():
        raise Incomplete("Use an all-lowercase output path for arm-neutral pytest temporary paths")
    output.mkdir(parents=True)
    measure = Measurements(subject, output, python)
    try:
        if os.name != "posix":
            raise Incomplete("Pilot execution requires a POSIX CPU runner for owned process-group cleanup")
        if Path(python).resolve() != Path(sys.executable).resolve():
            raise Incomplete("Invoke run with the same target environment interpreter supplied as --python")
        manifest = json.loads((subject / ".git" / MANIFEST).read_text(encoding="utf-8"))
        if manifest["baseline"] != BASELINE or Path(manifest["subject"]) != subject:
            raise Incomplete("Invalid prepared subject identity")
        execute_pilot(measure, manifest)
    except SemanticFailure as error:
        measure.summary.update(status="not_supported", reason=str(error), stopped_on="semantic_failure")
    except (Incomplete, OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        measure.summary.update(status="incomplete", reason=str(error), stopped_on=type(error).__name__)
    finally:
        measure.save()
    print(json.dumps({"status": measure.summary["status"], "output": str(output)}, sort_keys=True), flush=True)
    return 0 if measure.summary["status"] == "supported" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Clone the pinned baseline and add the two shared candidate files"
    )
    prepare_parser.add_argument("--subject", type=Path, required=True)
    run_parser = commands.add_parser(
        "run", help="Use the already installed target environment for a bounded paired pilot"
    )
    run_parser.add_argument("--subject", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--python", required=True)
    for name in ("_metadata", "_probe", "_micro"):
        child = commands.add_parser(name, help=argparse.SUPPRESS)
        child.add_argument("--output", type=Path, required=True)
        if name != "_metadata":
            child.add_argument("--subject", type=Path, required=True)
            child.add_argument("--expected", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.subject.resolve())
    elif args.command == "run":
        return run(args.subject.resolve(), args.output.resolve(), shutil.which(args.python) or args.python)
    elif args.command == "_metadata":
        dump(args.output, environment_metadata())
    elif args.command == "_probe":
        dump(args.output, import_probe(args.subject.resolve(), args.expected))
    elif args.command == "_micro":
        import_probe(args.subject.resolve(), args.expected)
        microbenchmark(args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SemanticFailure:
        traceback.print_exc()
        sys.exit(2)
    except (Incomplete, OSError, ValueError, subprocess.SubprocessError):
        traceback.print_exc()
        sys.exit(1)
