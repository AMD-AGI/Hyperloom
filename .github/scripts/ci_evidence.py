# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect fixed-baseline CI evidence without changing the measured test suite."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import pstats
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

BASELINE = "b8761298c0413b08937554077ad567a764a4e1dc"
REPO = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PROJECT = "hyperloom-inference-optimizer"
ALLOWED_ADDITIONS = {
    ".github/workflows/ci-evidence-experiments.yml",
    ".github/scripts/ci_evidence.py",
    ".github/scripts/ci_evidence_gate.py",
    ".github/scripts/tests/test_ci_evidence.py",
}
SHELL_FILE = "src/hyperloom/inference_optimizer/tests/test_aiperf_client_sh.py"
SHELL_NODEIDS = [
    f"{SHELL_FILE}::{name}"
    for name in (
        "test_a_stalled_flush_says_the_files_are_probably_truncated",
        "test_a_missing_rank_is_not_accepted_as_settled",
        "test_a_capture_that_produces_nothing_gives_up_early",
        "test_the_first_file_bound_never_exceeds_the_flush_budget",
        "test_the_client_waits_for_the_trace_to_stop_growing",
        "test_client_only_profiles_measured_phase_without_server_cleanup[0-0-profiler_output_unconfigured]",
        "test_client_only_profiles_measured_phase_without_server_cleanup[22-0-start_profile_failed]",
        "test_client_only_profiles_measured_phase_without_server_cleanup[0-22-stop_profile_failed]",
        "test_trace_flush_never_scans_without_remaining_budget[0-False]",
        "test_trace_flush_never_scans_without_remaining_budget[0-True]",
        "test_trace_flush_never_scans_without_remaining_budget[5-False]",
        "test_trace_flush_never_scans_without_remaining_budget[5-True]",
        "test_trace_flush_never_scans_without_remaining_budget[40-False]",
        "test_trace_flush_never_scans_without_remaining_budget[40-True]",
        "test_pre_stop_check_consumes_the_same_flush_budget",
        "test_trace_checks_reuse_capture_cache_and_remaining_flush_budget[False]",
        "test_trace_checks_reuse_capture_cache_and_remaining_flush_budget[True]",
    )
]
WALKTHROUGH_NODEIDS = [
    "src/hyperloom/inference_optimizer/tests/test_optimize_loop_walkthrough.py::"
    "test_a_baseline_carries_the_run_into_the_optimisation_phase_with_work"
]
PYLINT_PACKAGES = [
    "hyperloom.inference_optimizer",
    "hyperloom.orchestrator",
    "hyperloom.agents.framework",
    "hyperloom.agents.critic.runtime",
    "hyperloom.agents.quantization",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def constraints_text(packages: dict[str, str]) -> str:
    for name, version in packages.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
            raise ValueError("Invalid package name or version in constraints")
    return "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = normalize_name(distribution.metadata["Name"])
        if name == PROJECT:
            continue
        if distribution.read_text("direct_url.json") is not None:
            raise ValueError(f"Non-portable direct installation for {name}; URL intentionally omitted")
        if name in packages and packages[name] != distribution.version:
            raise ValueError(f"Conflicting installed versions for {name}")
        packages[name] = distribution.version
    constraints_text(packages)
    return packages


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def check_source_guard(repo: Path, baseline: str = BASELINE) -> dict[str, Any]:
    git(repo, "cat-file", "-e", f"{baseline}^{{commit}}")
    changed = git(repo, "diff", "--no-renames", "--name-status", "-z", baseline, "--").split("\0")
    entries = list(zip(changed[0::2], changed[1::2]))
    if any(status != "A" or path not in ALLOWED_ADDITIONS for status, path in entries):
        raise ValueError("Tracked source differs from baseline outside the four approved additions")
    untracked = set(git(repo, "ls-files", "--others", "--exclude-standard", "-z").split("\0")) - {""}
    if untracked - ALLOWED_ADDITIONS:
        raise ValueError("Unexpected untracked files in the measured checkout")
    return {
        "baseline": baseline,
        "baseline_tree": git(repo, "rev-parse", f"{baseline}^{{tree}}"),
        "head": git(repo, "rev-parse", "HEAD"),
        "approved_additions": sorted({path for _, path in entries} | untracked),
    }


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def capture(output: Path) -> None:
    cpuinfo = read_optional("/proc/cpuinfo") or ""
    models = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")})
    memory = read_optional("/proc/meminfo") or ""
    cgroup_files = (
        "cpu.max",
        "cpu.stat",
        "cpuset.cpus.effective",
        "memory.max",
        "memory.current",
        "memory.events",
        "cpu/cpu.cfs_quota_us",
        "cpu/cpu.cfs_period_us",
        "memory/memory.limit_in_bytes",
    )
    image_keys = (
        "ImageOS",
        "ImageVersion",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    )
    write_json(
        output / "environment.json",
        {
            "schema_version": SCHEMA_VERSION,
            "git": check_source_guard(REPO),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "image": {key: os.environ[key] for key in image_keys if key in os.environ},
            "os_release": {
                key: value
                for key, value in platform.freedesktop_os_release().items()
                if key in {"ID", "VERSION_ID", "PRETTY_NAME"}
            },
            "cpu_models": models,
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "memory": {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in memory.splitlines()
                if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:"))
            },
            "cgroup": {name: read_optional(f"/sys/fs/cgroup/{name}") for name in cgroup_files},
            "packages": installed_packages(),
        },
    )


def freeze(output: Path) -> None:
    check_source_guard(REPO)
    packages = installed_packages()
    (output / "constraints.txt").write_text(constraints_text(packages), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline": BASELINE,
            "python_version": platform.python_version(),
            "packages": packages,
        },
    )


def verify(manifest_path: Path, baseline: str = BASELINE) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("baseline") != baseline:
        raise ValueError("Manifest schema or baseline mismatch")
    if manifest.get("python_version") != platform.python_version():
        raise ValueError("Python version differs from frozen environment")
    if manifest.get("packages") != installed_packages():
        raise ValueError("Installed package versions differ from frozen environment")
    if manifest_path.with_name("constraints.txt").read_text(encoding="utf-8") != constraints_text(manifest["packages"]):
        raise ValueError("Constraints do not match the frozen manifest")
    check_source_guard(REPO, baseline)


def summarize_samples(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Samples must be nonempty, finite and nonnegative")
    return {"samples": values, "median": statistics.median(values), "min": min(values), "max": max(values)}


def normalize_diagnostics(messages: list[dict], repo: Path) -> list[tuple]:
    if not isinstance(messages, list):
        raise ValueError("Pylint JSON must be a list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or any(
            key not in message for key in ("path", "line", "column", "message-id", "symbol", "message")
        ):
            raise ValueError("Incomplete Pylint diagnostic")
        if any(not isinstance(message[key], str) for key in ("path", "message-id", "symbol", "message")):
            raise ValueError("Invalid Pylint diagnostic text")
        location = [message.get(key) for key in ("line", "column", "endLine", "endColumn")]
        if (
            any(value is not None and (type(value) is not int or value < 0) for value in location)
            or None in location[:2]
        ):
            raise ValueError("Invalid Pylint diagnostic location")
        path = (repo / message["path"]).resolve().relative_to(repo.resolve()).as_posix()
        text = message["message"].replace(str(repo.resolve()), "<repo>").replace(repo.resolve().as_posix(), "<repo>")
        normalized.append(
            (
                path,
                *(value if value is not None else -1 for value in location),
                message["message-id"],
                message["symbol"],
                text,
            )
        )
    return sorted(normalized)


def validate_pylint_result(returncode: int, diagnostics: list[tuple]) -> None:
    if type(returncode) is not int or returncode not in (0, 2):
        raise ValueError("Pylint fatal, usage, signal or other infrastructure failure")
    if bool(diagnostics) != (returncode == 2) or any(not re.fullmatch(r"E\d{4}", item[-3]) for item in diagnostics):
        raise ValueError("Pylint status and errors-only diagnostics disagree")


def validate_pytest_result(data: dict, expected: list[str]) -> None:
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != SCHEMA_VERSION
        or type(data.get("exitstatus")) is not int
        or data["exitstatus"] != 0
    ):
        raise ValueError("Pytest did not finish successfully with the expected schema")
    if not expected or len(set(expected)) != len(expected) or sorted(data.get("nodeids", [])) != sorted(expected):
        raise ValueError("Pytest collection differs from the fixed nodeids")
    phases = data.get("phases")
    if not isinstance(phases, list):
        raise ValueError("Missing pytest phase records")
    observed = Counter()
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("outcome") != "passed" or phase.get("wasxfail"):
            raise ValueError("Unexpected skip, xfail or failure in pytest phases")
        duration = phase.get("duration")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid pytest phase duration")
        observed[(phase.get("nodeid"), phase.get("when"))] += 1
    if observed != Counter((nodeid, when) for nodeid in expected for when in ("setup", "call", "teardown")):
        raise ValueError("Missing, duplicate or unexpected pytest phases")


class PhaseRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "nodeids": [], "phases": [], "exitstatus": None}
        write_json(self.path, self.data)

    def pytest_collection_finish(self, session: Any) -> None:
        self.data["nodeids"] = [item.nodeid for item in session.items]
        write_json(self.path, self.data)

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.data["phases"].append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "duration": report.duration,
                "wasxfail": getattr(report, "wasxfail", None),
            }
        )
        write_json(self.path, self.data)

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.data["exitstatus"] = int(exitstatus)
        write_json(self.path, self.data)


def pytest_configure(config: Any) -> None:
    if path := os.environ.get("CI_EVIDENCE_PHASES"):
        config.pluginmanager.register(PhaseRecorder(Path(path)), "ci-evidence-phases")


def oom_kills() -> int | None:
    events = read_optional("/sys/fs/cgroup/memory.events")
    return next((int(line.split()[1]) for line in (events or "").splitlines() if line.startswith("oom_kill ")), None)


def stop_process(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            # Keep the leader unreaped until both signals, avoiding process-group ID reuse.
            time.sleep(1)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait(timeout=5)


def measure_command(
    command: list[str], *, repo: Path, output: Path, env: dict[str, str], timeout: float, sample_rss: bool = False
) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Command timeout must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": "starting",
        "returncode": None,
        "elapsed_seconds": None,
        "peak_tree_rss_bytes": None,
        "rss_sampling_seconds": 0.1 if sample_rss else None,
        "rss_samples": 0,
        "rss_sample_errors": 0,
        "rss_caveat": "Approximate process-tree RSS; shared pages may be counted twice and short peaks missed."
        if sample_rss
        else None,
        "timeout_seconds": timeout,
        "oom_kills_before": oom_kills(),
    }
    write_json(output / "result.json", record)
    if sample_rss:
        import psutil
    start = time.monotonic()
    process = None
    try:
        with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name == "posix",
            )
            record.update(status="running", pid=process.pid)
            write_json(output / "result.json", record)
            while process.poll() is None:
                if time.monotonic() - start >= timeout:
                    record["status"] = "timed_out"
                    stop_process(process)
                    break
                if sample_rss:
                    try:
                        parent = psutil.Process(process.pid)
                        total = 0
                        for member in [parent, *parent.children(recursive=True)]:
                            try:
                                total += member.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                record["rss_sample_errors"] += 1
                        record["peak_tree_rss_bytes"] = max(record["peak_tree_rss_bytes"] or 0, total)
                        record["rss_samples"] += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        record["rss_sample_errors"] += 1
                time.sleep(0.1)
            if record["status"] == "running":
                record["status"] = "completed"
    finally:
        if process is not None:
            if process.poll() is None:
                record["status"] = "interrupted"
                stop_process(process)
            record["returncode"] = process.returncode
        else:
            record["status"] = "launch_failed"
        record["elapsed_seconds"] = time.monotonic() - start
        record["oom_kills_after"] = oom_kills()
        write_json(output / "result.json", record)
    return record


def child_environment(temporary: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COVERAGE_", "COV_CORE_", "CI_EVIDENCE_"))
    }
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(REPO / ".github/scripts"), str(REPO))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYLINTHOME": str(temporary / "pylint"),
            "XDG_CACHE_HOME": str(temporary / "cache"),
            "HYPOTHESIS_STORAGE_DIRECTORY": str(temporary / "hypothesis"),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
        }
    )
    return env


def profile_summary(path: Path) -> None:
    stats = pstats.Stats(str(path))
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:50]:
        filename = filename.replace(str(REPO), "<repo>").replace(str(Path(sys.prefix)), "<python>")
        if Path(filename).is_absolute():
            filename = f"<external>/{Path(filename).name}"
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "calls": calls,
                "own_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
    write_json(
        path.with_name("profile-top.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "caveat": "cProfile includes instrumentation and await/I/O effects; cumulative time is not CPU time.",
            "total_calls": stats.total_calls,
            "total_seconds": stats.total_tt,
            "top_cumulative": rows,
        },
    )


def run_sample(
    track: str, python: str, output: Path, label: str, jobs: int | None, profile: bool, timeout: float
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"ci-evidence-{label}-", dir=os.environ["RUNNER_TEMP"]) as temporary:
        env = child_environment(Path(temporary))
        if track == "pylint":
            command = [
                python,
                "-m",
                "pylint",
                "--errors-only",
                f"--jobs={jobs}",
                "--output-format=json",
                *PYLINT_PACKAGES,
            ]
        else:
            env["CI_EVIDENCE_PHASES"] = str(output / "phases.json")
            nodeids = SHELL_NODEIDS if track == "shell" else WALKTHROUGH_NODEIDS
            command = [python]
            if profile:
                command += ["-m", "cProfile", "-o", str(output / "profile.pstats")]
            command += [
                "-m",
                "pytest",
                "-p",
                "ci_evidence",
                "-p",
                "no:pytest_cov",
                "-p",
                "no:xdist",
                "-p",
                "no:xdist.looponfail",
                "--durations=0",
                "-o",
                "addopts=",
                "-o",
                "junit_duration_report=total",
                "-o",
                f"cache_dir={temporary}/pytest-cache",
                f"--basetemp={temporary}/pytest",
                f"--junitxml={output / 'junit.xml'}",
                *nodeids,
            ]
        record = measure_command(
            command, repo=REPO, output=output, env=env, timeout=timeout, sample_rss=track == "pylint"
        )
    record.update(label=label, jobs=jobs, profiled=profile, valid=False, environment="../environment.json")
    try:
        if record["status"] != "completed":
            raise ValueError("Command did not complete; no evidence of success")
        if record["oom_kills_before"] is not None and record["oom_kills_after"] > record["oom_kills_before"]:
            raise ValueError("Cgroup recorded an OOM kill during the measurement")
        if track == "pylint":
            messages = json.loads((output / "stdout.txt").read_text(encoding="utf-8"))
            diagnostics = normalize_diagnostics(messages, REPO)
            validate_pylint_result(record["returncode"], diagnostics)
            if not record["rss_samples"] or not record["peak_tree_rss_bytes"]:
                raise ValueError("No process-tree RSS samples were obtained")
            record["diagnostics"] = diagnostics
        else:
            if record["returncode"] != 0:
                raise ValueError("Pytest process returned a nonzero status")
            validate_pytest_result(json.loads((output / "phases.json").read_text(encoding="utf-8")), nodeids)
            if not (output / "junit.xml").is_file():
                raise ValueError("JUnit artifact is missing")
            if profile:
                profile_summary(output / "profile.pstats")
        check_source_guard(REPO)
        record["valid"] = True
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        record["validation_error"] = str(error)
        raise
    finally:
        write_json(output / "result.json", record)
    return record


def run_track(track: str, output: Path, python: str) -> None:
    if sys.platform != "linux":
        raise ValueError("Measurements require the approved Linux hosted runner")
    if os.path.abspath(python) != os.path.abspath(sys.executable):
        raise ValueError("Run the harness using the same target-environment Python as --python")
    capture(output)
    plan = [(f"baseline-{i}", None, False) for i in range(1, 4)]
    if track == "walkthrough":
        plan.append(("profile-1", None, True))
    elif track == "pylint":
        plan = [
            (f"pair-{pair}-{jobs}", jobs, False)
            for pair, order in enumerate(((1, 2), (2, 1), (1, 2)), 1)
            for jobs in order
        ]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "track": track,
        "baseline": BASELINE,
        "status": "incomplete",
        "samples": [{"label": label, "status": "missing"} for label, _, _ in plan],
        "cache_note": "First sample may include cold-start effects; later OS caches are warm/unknown. Every sample has fresh temporary/cache directories.",
        "expected_nodeids": SHELL_NODEIDS
        if track == "shell"
        else WALKTHROUGH_NODEIDS
        if track == "walkthrough"
        else [],
    }
    write_json(output / "summary.json", summary)
    deadline = time.monotonic() + {"shell": 900, "walkthrough": 1200, "pylint": 1200}[track]
    try:
        for index, (label, jobs, profile) in enumerate(plan):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                raise ValueError("Track time budget exhausted; remaining samples are missing")
            result = run_sample(
                track,
                python,
                output / label,
                label,
                jobs,
                profile,
                min(600 if profile else 300 if track == "pylint" else 360, remaining),
            )
            summary["samples"][index] = result
            write_json(output / "summary.json", summary)
            if track == "pylint" and index % 2:
                first, second = summary["samples"][index - 1 : index + 1]
                if first["returncode"] != second["returncode"] or first["diagnostics"] != second["diagnostics"]:
                    raise ValueError("Paired Pylint diagnostic multisets or exit codes differ")
        samples = summary["samples"]
        if track == "pylint":
            summary["wall_seconds"] = {
                str(jobs): summarize_samples(
                    [sample["elapsed_seconds"] for sample in samples if sample["jobs"] == jobs]
                )
                for jobs in (1, 2)
            }
            summary["paired_deltas_seconds"] = [
                next(sample["elapsed_seconds"] for sample in samples[start : start + 2] if sample["jobs"] == 2)
                - next(sample["elapsed_seconds"] for sample in samples[start : start + 2] if sample["jobs"] == 1)
                for start in (0, 2, 4)
            ]
        else:
            summary["wall_seconds"] = summarize_samples(
                [sample["elapsed_seconds"] for sample in samples if not sample["profiled"]]
            )
        summary["status"] = "complete"
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        for index, (label, _, _) in enumerate(plan):
            result_path = output / label / "result.json"
            if result_path.is_file():
                summary["samples"][index] = json.loads(result_path.read_text(encoding="utf-8"))
        write_json(output / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "freeze", "run"):
        command = commands.add_parser(action)
        command.add_argument("--output", required=True, type=Path)
        if action == "run":
            command.add_argument("--track", required=True, choices=("shell", "walkthrough", "pylint"))
            command.add_argument("--python", required=True)
    command = commands.add_parser("verify")
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--baseline", default=BASELINE, choices=(BASELINE,))
    args = parser.parse_args()
    try:
        if args.action == "verify":
            verify(args.manifest, args.baseline)
        else:
            output = args.output.resolve()
            runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
            if not output.is_relative_to(runner_temp) or output.is_relative_to(REPO):
                raise ValueError("Evidence output must be under RUNNER_TEMP and outside the repository")
            output.mkdir(parents=True, exist_ok=True)
            if args.action == "run":
                run_track(args.track, output, args.python)
            elif args.action == "capture":
                capture(output)
            else:
                freeze(output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"CI evidence failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
