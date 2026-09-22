# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one frozen, paired CPU-suite comparison in disposable baseline subjects."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from typing import Any

from ci_evidence import SHELL_NODEIDS, constraints_text, measure_command

BASELINE = "b8761298c0413b08937554077ad567a764a4e1dc"
FACILITY = "5101c6ea1afad1f49558be2f1f039cd60037972f"
REPO = Path(__file__).resolve().parents[2]
SHELL_FILE = "src/hyperloom/inference_optimizer/tests/test_aiperf_client_sh.py"
ADDITIONS = {
    ".github/scripts/ci_shell_full.py",
    ".github/scripts/ci_shell_full_plugin.py",
    ".github/scripts/tests/test_ci_shell_full.py",
    ".github/workflows/ci-shell-full-comparison.yml",
}
SCHEMA = 1


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_digest(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True, timeout=60).stdout


def facility_guard() -> dict:
    entries = git("diff", "--no-renames", "--name-status", "-z", FACILITY, "--").decode().split("\0")
    changes = list(zip(entries[0::2], entries[1::2]))
    if any(
        not ((path == SHELL_FILE and status == "M") or (path in ADDITIONS and status == "A"))
        for status, path in changes
    ):
        raise ValueError("Facility checkout differs outside the five approved files")
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""}
    if untracked - ADDITIONS:
        raise ValueError("Unexpected untracked facility files")
    return {"baseline": BASELINE, "facility": FACILITY, "head": git("rev-parse", "HEAD").decode().strip()}


def baseline_archive() -> tarfile.TarFile:
    return tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", BASELINE)), mode="r:")


def baseline_manifest() -> dict:
    result = {}
    with baseline_archive() as archive:
        for member in archive.getmembers():
            if member.isfile():
                result[member.name] = {
                    "sha256": digest(archive.extractfile(member).read()),
                    "mode": member.mode & 0o111,
                }
            elif member.issym():
                result[member.name] = {"sha256": digest(member.linkname.encode()), "symlink": True}
            elif not member.isdir():
                raise ValueError(f"Unsupported baseline archive entry: {member.name}")
    return result


def subject_guard(subject: Path, manifest: dict, arm: str) -> None:
    for name, entry in manifest["files"].items():
        path = subject / name
        expected = manifest["candidate_sha256"] if name == SHELL_FILE and arm == "B" else entry["sha256"]
        if entry.get("symlink"):
            data = os.readlink(path).encode() if path.is_symlink() else b""
        else:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or substituted subject file: {name}")
            data = path.read_bytes()
            if os.name == "posix" and path.stat().st_mode & 0o111 != entry["mode"]:
                raise ValueError(f"Subject executable mode changed: {name}")
        if digest(data) != expected:
            raise ValueError(f"Subject source guard failed: {name}")


def create_subject(path: Path) -> None:
    facility_guard()
    if path.exists():
        raise ValueError("Subject path must not already exist")
    # Source-contract tests require .git and git ls-files; an archive would silently skip them.
    subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", str(REPO), str(path)], check=True, timeout=120)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", BASELINE], check=True, timeout=60)


def validate_seed(path: Path) -> None:
    # Use the official timing merger's duplicate/value rules without loading its pytest plugin.
    command = [sys.executable, str(REPO / "scripts/ci_durations.py")]
    with tempfile.TemporaryDirectory(prefix="ci-shell-seed-") as temporary:
        root = Path(temporary)
        shard = root / "artifacts/durations-shard1"
        shard.mkdir(parents=True)
        shutil.copyfile(path, shard / ".test_durations.shard1")
        config = root / "config.toml"
        config.write_text("[tool.hyperloom.tests_coverage]\ntotal_shards = 1\n", encoding="utf-8")
        subprocess.run(
            command
            + ["--config", str(config), "--artifacts", str(shard.parent), "--output", str(root / "merged.json")],
            check=True,
            timeout=30,
        )


def prepare(output: Path, seed: Path) -> None:
    guard = facility_guard()
    validate_seed(seed)
    files = baseline_manifest()
    candidate = (REPO / SHELL_FILE).read_bytes()
    if digest(candidate) == files[SHELL_FILE]["sha256"]:
        raise ValueError("Candidate test file is identical to the baseline")
    shutil.copyfile(seed, output / "seed.json")
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            **guard,
            "files": files,
            "baseline_code_sha256": object_digest(files),
            "candidate_sha256": digest(candidate),
            "seed_sha256": digest(seed.read_bytes()),
            "seed_cache_key": "test-durations-v1-35682438624",
        },
    )


def load_prepared(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    facility_guard()
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("baseline") != BASELINE
        or manifest.get("facility") != FACILITY
    ):
        raise ValueError("Prepared artifact has the wrong baseline/schema")
    if manifest["files"] != baseline_manifest() or object_digest(manifest["files"]) != manifest["baseline_code_sha256"]:
        raise ValueError("Prepared baseline file manifest differs from Git")
    if digest((REPO / SHELL_FILE).read_bytes()) != manifest["candidate_sha256"]:
        raise ValueError("Candidate differs from the prepared artifact")
    if digest((directory / "seed.json").read_bytes()) != manifest["seed_sha256"]:
        raise ValueError("Duration seed differs from the prepared artifact")
    return manifest


TARGET_PROBE = r"""
import importlib.metadata as m, json, os, platform, re, sys
packages = {}
for dist in m.distributions():
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
    if name == "hyperloom-inference-optimizer":
        continue
    if dist.read_text("direct_url.json") is not None:
        raise ValueError("Non-portable direct installation: " + name + "; URL omitted")
    if name in packages and packages[name] != dist.version:
        raise ValueError("Conflicting installed package: " + name)
    packages[name] = dist.version
print(json.dumps({"python_version": platform.python_version(), "implementation": platform.python_implementation(),
    "packages": packages, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
    "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}))
"""


def target_environment(python: str) -> dict:
    result = subprocess.run([python, "-c", TARGET_PROBE], capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(result.stdout)
    constraints_text(data["packages"])
    if data["python_version"].rsplit(".", 1)[0] not in {"3.10", "3.11"}:
        raise ValueError("Only the approved Python versions may be measured")
    image_keys = ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    data["image"] = {key: os.environ[key] for key in image_keys if key in os.environ}
    for name, path in {
        "cpu": "/proc/cpuinfo",
        "memory": "/proc/meminfo",
        "cgroup_memory": "/sys/fs/cgroup/memory.max",
        "cgroup_cpu": "/sys/fs/cgroup/cpu.max",
    }.items():
        data[name] = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else None
    return data


def verify_environment(python: str, bootstrap: Path) -> dict:
    frozen = read_json(bootstrap / "manifest.json")
    current = target_environment(python)
    if any(frozen[key] != current[key] for key in ("python_version", "implementation", "packages")):
        raise ValueError("Target Python or package versions differ from the frozen environment")
    if (bootstrap / "constraints.txt").read_text(encoding="utf-8") != constraints_text(frozen["packages"]):
        raise ValueError("Frozen constraints disagree with the package manifest")
    return current


def official_args(subject: Path) -> list[str]:
    config = tomllib.loads((subject / "pyproject.toml").read_text(encoding="utf-8"))
    settings = config["tool"]["hyperloom"]["tests_coverage"]
    if (
        settings["total_shards"] != 6
        or settings["xdist_workers"] != 2
        or config["tool"]["coverage"]["report"]["fail_under"] != 90
    ):
        raise ValueError("Official shard/worker/coverage configuration changed")
    return settings["pytest_ci_args"]


def phase_outcome(phases: dict) -> str:
    if "setup" not in phases or "teardown" not in phases:
        raise ValueError("Missing setup or teardown report")
    setup, teardown = phases["setup"], phases["teardown"]
    if (setup["outcome"] == "passed") != ("call" in phases):
        raise ValueError("Call report disagrees with setup outcome")
    if setup["outcome"] == "failed" or teardown["outcome"] == "failed":
        return "error"
    call = phases.get("call", setup)
    if call["outcome"] == "failed":
        return "xpass" if (call.get("longrepr") or "").startswith("[XPASS(strict)]") else "fail"
    skipped = next((item for item in phases.values() if item["outcome"] == "skipped"), None)
    if skipped is not None:
        return "xfail" if skipped.get("wasxfail") is not None else "skip"
    return "xpass" if call.get("wasxfail") is not None else "pass"


def validate_stream(
    path: Path, expected: list[str] | None = None, workers: int = 2, collect_only: bool = False
) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(events) < 3 or events[0].get("event") != "session_start" or events[-1].get("event") != "session_finish":
        raise ValueError("Incomplete pytest event stream")
    start, finish = events[0], events[-1]
    if start.get("schema_version") != SCHEMA or start.get("collectonly") is not collect_only:
        raise ValueError("Unexpected pytest event schema or collection mode")
    if start.get("workers") != workers or finish.get("exitstatus") not in (0, 1):
        raise ValueError("Pytest worker count or exit status is invalid")
    collections, down, reports = {}, set(), {}
    for event in events[1:-1]:
        kind = event.get("event")
        if kind == "collection":
            worker = event.get("worker_id")
            if worker in collections:
                raise ValueError("Duplicate worker collection")
            collections[worker] = event["nodeids"]
        elif kind == "worker_down":
            worker = event.get("worker_id")
            if event.get("error") is not None or worker in down:
                raise ValueError("Worker crash or duplicate worker shutdown")
            down.add(worker)
        elif kind == "report" and not collect_only:
            nodeid, when = event.get("nodeid"), event.get("when")
            duration = event.get("duration")
            if when not in {"setup", "call", "teardown"} or event.get("outcome") not in {"passed", "failed", "skipped"}:
                raise ValueError("Unknown pytest phase or outcome")
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid phase duration")
            phases = reports.setdefault(nodeid, {})
            if when in phases:
                raise ValueError("Duplicate test phase (reruns are not allowed)")
            phases[when] = event
        else:
            raise ValueError(f"Invalid pytest event: {kind}")
    worker_ids = {f"gw{index}" for index in range(workers)} if workers else {None}
    if set(collections) != worker_ids or down != (worker_ids if workers else set()):
        raise ValueError("Missing, unexpected, or unfinished pytest worker")
    nodeids = next(iter(collections.values()))
    if (
        not nodeids
        or any(not isinstance(node, str) or not node for node in nodeids)
        or len(set(nodeids)) != len(nodeids)
    ):
        raise ValueError("Empty or duplicate collection nodeids")
    if any(ids != nodeids for ids in collections.values()) or (expected is not None and nodeids != expected):
        raise ValueError("Worker or expected collections differ")
    if finish.get("testscollected") != len(nodeids):
        raise ValueError("Session collection count disagrees with worker collections")
    if collect_only:
        if finish["exitstatus"] != 0:
            raise ValueError("Collection failed")
        return {"nodeids": nodeids, "outcomes": {}, "signatures": {}, "exitstatus": 0, "passed": True}
    if set(reports) != set(nodeids):
        raise ValueError("Missing or unexpected test reports")
    outcomes = {node: phase_outcome(reports[node]) for node in nodeids}
    failures = any(value in {"fail", "error", "xpass"} for value in outcomes.values())
    if (finish["exitstatus"] == 1) != any(
        value in {"fail", "error"}
        or (value == "xpass" and any(p["outcome"] == "failed" for p in reports[node].values()))
        for node, value in outcomes.items()
    ):
        raise ValueError("Pytest exit status disagrees with reported failures")
    root = start.get("rootpath", "")
    signatures = {}
    for node, phases in reports.items():
        signatures[node] = {
            when: {
                key: value.replace(root, "<subject>") if isinstance(value, str) and root else value
                for key, value in phase.items()
                if key in {"outcome", "wasxfail", "skipreason"}
            }
            for when, phase in phases.items()
        }
    return {
        "nodeids": nodeids,
        "outcomes": outcomes,
        "signatures": signatures,
        "exitstatus": finish["exitstatus"],
        "passed": not failures and finish["exitstatus"] == 0,
    }


def merge_outcomes(shards: list[dict], catalog: list[str]) -> dict:
    if len(shards) != 6 or not catalog or len(set(catalog)) != len(catalog):
        raise ValueError("Exactly six shards and a unique nonempty catalog are required")
    outcomes = {}
    for shard in shards:
        if set(shard["nodeids"]) != set(shard["outcomes"]) or len(shard["nodeids"]) != len(shard["outcomes"]):
            raise ValueError("Shard collection and outcome nodeids disagree")
        if set(outcomes) & set(shard["outcomes"]):
            raise ValueError("Duplicate nodeids across shards")
        outcomes.update(shard["outcomes"])
    if set(outcomes) != set(catalog):
        raise ValueError("Shard union differs from the frozen catalog")
    return outcomes


def compare_coverage(a: dict, b: dict) -> dict:
    files_a, files_b = a["files"], b["files"]
    differences, valid, no_drop = {}, bool(files_a) and set(files_a) == set(files_b), True
    for name in sorted(set(files_a) | set(files_b)):
        if name not in files_a or name not in files_b:
            differences[name] = {"source_set_changed": True}
            continue
        left, right = files_a[name], files_b[name]
        lines_a, lines_b = set(left["executed_lines"]), set(right["executed_lines"])
        missing_a, missing_b = set(left["missing_lines"]), set(right["missing_lines"])
        denominator_a, denominator_b = lines_a | missing_a, lines_b | missing_b
        same_denominator = denominator_a == denominator_b
        valid &= same_denominator and not (lines_a & missing_a or lines_b & missing_b)
        valid &= left["summary"]["num_statements"] == len(denominator_a) and right["summary"]["num_statements"] == len(
            denominator_b
        )
        valid &= left["summary"]["covered_lines"] == len(lines_a) and right["summary"]["covered_lines"] == len(lines_b)
        lost, gained = sorted(lines_a - lines_b), sorted(lines_b - lines_a)
        no_drop &= not lost
        if lost or gained or not same_denominator:
            differences[name] = {
                "lost_lines": lost,
                "gained_lines": gained,
                "same_denominator": same_denominator,
                "baseline_statements": len(denominator_a),
                "candidate_statements": len(denominator_b),
            }
    for report in (a, b):
        valid &= report["totals"]["num_statements"] == sum(
            row["summary"]["num_statements"] for row in report["files"].values()
        )
        valid &= report["totals"]["covered_lines"] == sum(
            len(set(row["executed_lines"])) for row in report["files"].values()
        )
    return {
        "valid": bool(valid),
        "no_drop": bool(no_drop and valid),
        "file_differences": differences,
        "baseline_totals": a["totals"],
        "candidate_totals": b["totals"],
    }


def swap_arm(subject: Path, manifest: dict, arm: str) -> None:
    current_hash = digest((subject / SHELL_FILE).read_bytes())
    baseline_hash = manifest["files"][SHELL_FILE]["sha256"]
    if current_hash not in {baseline_hash, manifest["candidate_sha256"]}:
        raise ValueError("Refusing to overwrite an unrecognized subject test file")
    subject_guard(subject, manifest, "A" if current_hash == baseline_hash else "B")
    content = git("show", f"{BASELINE}:{SHELL_FILE}") if arm == "A" else (REPO / SHELL_FILE).read_bytes()
    (subject / SHELL_FILE).write_bytes(content)
    subject_guard(subject, manifest, arm)


def test_environment(temporary: Path, output: Path, python: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "COVERAGE_", "COV_CORE_", "CI_EVIDENCE_", "CI_SHELL_FULL_"))
    }
    env.update(
        PYTHONPATH=str(REPO / ".github/scripts"),
        PYTHONDONTWRITEBYTECODE="1",
        PATH=str(Path(python).parent) + os.pathsep + os.environ.get("PATH", ""),
        CI_SHELL_FULL_EVENTS=str(output / "events.jsonl"),
        COVERAGE_FILE=str(output / ".coverage"),
        HYPOTHESIS_STORAGE_DIRECTORY=str(temporary / "hypothesis"),
        XDG_CACHE_HOME=str(temporary / "cache"),
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
    )
    return env


def run_pytest(
    python: str,
    subject: Path,
    output: Path,
    prepared: Path,
    manifest: dict,
    arm: str,
    *,
    mode: str,
    timeout: float,
    shard: int | None = None,
    expected: list[str] | None = None,
) -> dict:
    output.mkdir(parents=True)
    swap_arm(subject, manifest, arm)
    seed = subject / ".test_durations"
    seed.chmod(0o644) if seed.exists() else None
    shutil.copyfile(prepared / "seed.json", seed)
    seed.chmod(0o444)
    with tempfile.TemporaryDirectory(prefix=f"ci-shell-{arm}-", dir=os.environ["RUNNER_TEMP"]) as directory:
        temporary = Path(directory)
        command = [
            python,
            "-m",
            "pytest",
            *official_args(subject),
            "--cov-fail-under=0",
            "-p",
            "ci_shell_full_plugin",
            "-o",
            "addopts=",
            "-o",
            f"cache_dir={temporary / 'pytest-cache'}",
            f"--basetemp={temporary / 'pytest'}",
            f"--junitxml={output / 'junit.xml'}",
        ]
        workers = 2 if mode == "paired" else 0
        command += ["-n", str(workers)]
        if mode == "collect":
            command += ["--collect-only", "--no-cov"]
        elif mode == "preflight":
            command += SHELL_NODEIDS
        else:
            command += [
                "--splits",
                "6",
                "--group",
                str(shard),
                "--splitting-algorithm",
                "least_duration",
                "--durations-path",
                str(seed),
            ]
        record = measure_command(
            command,
            repo=subject,
            output=output,
            env=test_environment(temporary, output, python),
            timeout=max(1, timeout),
            sample_rss=True,
        )
    record.update(arm=arm, mode=mode, valid=False)
    try:
        if record["status"] != "completed" or record["returncode"] not in (0, 1):
            raise ValueError("Pytest timed out or failed outside normal test outcomes")
        if record["oom_kills_before"] is not None and record["oom_kills_after"] > record["oom_kills_before"]:
            raise ValueError("The cgroup recorded an OOM kill")
        parsed = validate_stream(output / "events.jsonl", expected, workers, mode == "collect")
        if parsed["exitstatus"] != record["returncode"]:
            raise ValueError("Pytest process and stream exit statuses disagree")
        if not (output / "junit.xml").is_file() or (mode == "paired" and not (output / ".coverage").is_file()):
            raise ValueError("Missing JUnit or coverage artifact")
        record.update(parsed, valid=True)
    except (ValueError, OSError, KeyError, TypeError) as error:
        record["validation_error"] = str(error)
    finally:
        write_json(output / "results.json", record)
    # Source failures invalidate the experiment and stop the other arm, unlike ordinary failures.
    subject_guard(subject, manifest, arm)
    facility_guard()
    if digest(seed.read_bytes()) != manifest["seed_sha256"]:
        raise ValueError("A test arm changed the immutable duration seed")
    return record


def bootstrap(output: Path, prepared: Path, python: str, subject: Path) -> None:
    manifest = load_prepared(prepared)
    subject_guard(subject, manifest, "A")
    environment = target_environment(python)
    write_json(output / "environment.json", environment)
    (output / "constraints.txt").write_text(constraints_text(environment["packages"]), encoding="utf-8")
    manifest.update({key: environment[key] for key in ("python_version", "implementation", "packages")})
    manifest["status"] = "incomplete"
    write_json(output / "manifest.json", manifest)
    deadline, catalog = time.monotonic() + 850, None
    for arm in ("A", "B"):
        result = run_pytest(
            python,
            subject,
            output / f"catalog-{arm}",
            prepared,
            manifest,
            arm,
            mode="collect",
            timeout=min(150, deadline - time.monotonic()),
            expected=catalog,
        )
        if not result["valid"] or not result["passed"]:
            raise ValueError(f"{arm} full catalog collection failed")
        catalog = result["nodeids"]
    write_json(output / "catalog.json", catalog)
    manifest["catalog_sha256"] = object_digest(catalog)
    write_json(output / "manifest.json", manifest)
    for arm in ("A", "B"):
        result = run_pytest(
            python,
            subject,
            output / f"preflight-{arm}",
            prepared,
            manifest,
            arm,
            mode="preflight",
            timeout=min(360, deadline - time.monotonic()),
            expected=SHELL_NODEIDS,
        )
        if not result["valid"] or not result["passed"] or set(result["outcomes"].values()) != {"pass"}:
            raise ValueError(f"{arm} preflight did not pass all 17 fixed controls")
    verify_environment(python, output)
    manifest.update(status="passed", preflight_nodeids=SHELL_NODEIDS)
    write_json(output / "manifest.json", manifest)


def frozen_catalog(bootstrap: Path) -> tuple[dict, list[str]]:
    manifest, catalog = read_json(bootstrap / "manifest.json"), read_json(bootstrap / "catalog.json")
    if (
        manifest.get("status") != "passed"
        or manifest.get("baseline") != BASELINE
        or manifest.get("facility") != FACILITY
    ):
        raise ValueError("Bootstrap artifact did not pass the approved baseline preflight")
    if object_digest(catalog) != manifest["catalog_sha256"] or manifest.get("preflight_nodeids") != SHELL_NODEIDS:
        raise ValueError("Frozen catalog or preflight controls changed")
    return manifest, catalog


def paired(output: Path, prepared: Path, bootstrap_path: Path, python: str, subject: Path, shard: int) -> None:
    manifest = load_prepared(prepared)
    frozen, _ = frozen_catalog(bootstrap_path)
    for key in ("baseline_code_sha256", "candidate_sha256", "seed_sha256"):
        if manifest[key] != frozen[key]:
            raise ValueError("Prepared and bootstrap artifacts disagree")
    environment = verify_environment(python, bootstrap_path)
    write_json(output / "environment.json", environment)
    py_index = 0 if environment["python_version"].startswith("3.10.") else 1
    order = ["A", "B"] if (shard + py_index) % 2 == 0 else ["B", "A"]
    summary = {
        "schema_version": SCHEMA,
        "shard": shard,
        "python_version": environment["python_version"],
        "order": order,
        "status": "incomplete",
        "arms": {},
        **{key: frozen[key] for key in ("baseline_code_sha256", "candidate_sha256", "seed_sha256", "catalog_sha256")},
    }
    write_json(output / "pair.json", summary)
    started = time.monotonic()
    try:
        for arm in order:
            result = run_pytest(
                python, subject, output / arm, prepared, manifest, arm, mode="paired", timeout=600, shard=shard
            )
            summary["arms"][arm] = {
                key: result.get(key) for key in ("valid", "passed", "returncode", "elapsed_seconds", "validation_error")
            }
            write_json(output / "pair.json", summary)
            verify_environment(python, bootstrap_path)
        a, b = (read_json(output / arm / "results.json") for arm in ("A", "B"))
        if not a["valid"] or not b["valid"]:
            raise ValueError("At least one arm produced invalid or incomplete evidence")
        if a["nodeids"] != b["nodeids"]:
            raise ValueError("A/B shard membership or order differs")
        summary["outcome_differences"] = {
            node: [a["outcomes"][node], b["outcomes"][node]]
            for node in a["nodeids"]
            if a["outcomes"][node] != b["outcomes"][node] or a["signatures"][node] != b["signatures"][node]
        }
        if not a["passed"] or not b["passed"] or summary["outcome_differences"]:
            raise ValueError("A/B tests failed or have different outcomes")
        summary["status"] = "passed"
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "pair.json", summary)


def combine_coverage(
    python: str, subject: Path, output: Path, inputs: list[Path], manifest: dict, passed: bool
) -> dict:
    output.mkdir(parents=True)
    subject_guard(subject, manifest, "A")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("COVERAGE_", "COV_CORE_"))}
    env.update(
        COVERAGE_FILE=str(subject / ".coverage"),
        COVERAGE_RELAX_FAIL_UNDER="false",
        GITHUB_OUTPUT=str(output / "report-output.txt"),
        PYTEST_OUTCOME="success" if passed else "failure",
    )
    env.pop("GITHUB_STEP_SUMMARY", None)
    (subject / ".coverage").unlink(missing_ok=True)
    combine = measure_command(
        [python, "-m", "coverage", "combine", "--keep", *map(str, inputs)],
        repo=subject,
        output=output / "combine",
        env=env,
        timeout=90,
    )
    if combine["returncode"] != 0 or combine["status"] != "completed":
        raise ValueError("Coverage combine failed")
    shutil.copyfile(subject / ".coverage", output / ".coverage")
    inspected = subprocess.run(
        [
            python,
            "-c",
            "import coverage,json; c=coverage.Coverage(); c.load(); print(json.dumps(sorted(c.get_data().measured_files())))",
        ],
        cwd=subject,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    measured = json.loads(inspected.stdout)
    write_json(output / "measured-files.json", measured)
    if not measured or any(name not in manifest["files"] for name in measured):
        raise ValueError("Raw coverage data contains external or unknown source files")
    command = [python, str(subject / "scripts/ci_coverage_report.py")]
    report = measure_command(command + ["report"], repo=subject, output=output / "report", env=env, timeout=90)
    status = re.fullmatch(r"report_exit_code=(\d+)\n?", (output / "report-output.txt").read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["returncode"] != 0 or status is None:
        raise ValueError("Official coverage reporting failed")
    env["REPORT_EXIT_CODE"] = status.group(1)
    gate = measure_command(command + ["gate"], repo=subject, output=output / "gate", env=env, timeout=30)
    exported = measure_command(
        [python, "-m", "coverage", "json", "--fail-under=0", "-o", str(output / "coverage.json")],
        repo=subject,
        output=output / "json",
        env=env,
        timeout=90,
    )
    if exported["status"] != "completed" or exported["returncode"] != 0:
        raise ValueError("Coverage JSON export failed")
    coverage = read_json(output / "coverage.json")
    for name in coverage["files"]:
        if name not in manifest["files"]:
            raise ValueError(f"Coverage contains an external or unknown source file: {name}")
    subject_guard(subject, manifest, "A")
    return {
        "gate_passed": gate["status"] == "completed" and gate["returncode"] == 0,
        "report_exit_code": int(status.group(1)),
        "coverage": coverage,
    }


def aggregate(output: Path, inputs: Path, bootstrap_path: Path, python: str, subject: Path) -> None:
    manifest, catalog = frozen_catalog(bootstrap_path)
    facility_guard()
    if (
        manifest["files"] != baseline_manifest()
        or digest((REPO / SHELL_FILE).read_bytes()) != manifest["candidate_sha256"]
    ):
        raise ValueError("Aggregate sources differ from the frozen baseline/candidate")
    environment = target_environment(python)
    if (
        environment["python_version"] != manifest["python_version"]
        or environment["packages"].get("coverage") != manifest["packages"]["coverage"]
    ):
        raise ValueError("Aggregate Python or coverage version differs from bootstrap")
    write_json(output / "environment.json", environment)
    pairs = {path.parent: read_json(path) for path in inputs.rglob("pair.json")}
    if len(pairs) != 6 or {pair["shard"] for pair in pairs.values()} != set(range(1, 7)):
        raise ValueError("Exactly one paired artifact for each of the six shards is required")
    for pair in pairs.values():
        if pair["schema_version"] != SCHEMA or any(
            pair[key] != manifest[key]
            for key in ("python_version", "baseline_code_sha256", "candidate_sha256", "seed_sha256", "catalog_sha256")
        ):
            raise ValueError("Paired artifacts belong to a different frozen campaign")
    ordered = sorted(pairs, key=lambda directory: pairs[directory]["shard"])
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "python_version": manifest["python_version"],
        "status": "failed",
        "arms": {},
        "errors": [],
        "sample_size": 1,
        "timing_caveat": "Maximum shard duration is an experimental critical-path estimate, not production workflow wall time. N=1; no significance claim.",
    }
    outcomes, signatures, coverages = {}, {}, {}
    for arm in ("A", "B"):
        try:
            records = [read_json(directory / arm / "results.json") for directory in ordered]
            parsed = [validate_stream(directory / arm / "events.jsonl") for directory in ordered]
            if any(
                not row["valid"] or row["returncode"] != stream["exitstatus"] for row, stream in zip(records, parsed)
            ):
                raise ValueError("Invalid shard measurement")
            outcomes[arm] = merge_outcomes(parsed, catalog)
            signatures[arm] = {node: value for stream in parsed for node, value in stream["signatures"].items()}
            durations = [row["elapsed_seconds"] for row in records]
            passed = all(stream["passed"] for stream in parsed)
            summary["arms"][arm] = {
                "passed": passed,
                "status_counts": dict(Counter(outcomes[arm].values())),
                "failures": {
                    node: status for node, status in outcomes[arm].items() if status in {"fail", "error", "xpass"}
                },
                "shard_seconds": durations,
                "runner_seconds": sum(durations),
                "critical_path_estimate_seconds": max(durations),
                "peak_tree_rss_bytes": [row["peak_tree_rss_bytes"] for row in records],
            }
            if not passed:
                summary["errors"].append(f"{arm}: test failures (including identical failures) are not a pass")
        except (ValueError, OSError, KeyError, TypeError) as error:
            summary["errors"].append(f"{arm} outcomes: {error}")
        try:
            result = combine_coverage(
                python,
                subject,
                output / arm,
                [directory / arm / ".coverage" for directory in ordered],
                manifest,
                summary["arms"].get(arm, {}).get("passed", False),
            )
            coverages[arm] = result["coverage"]
            summary["arms"].setdefault(arm, {})["coverage_gate_passed"] = result["gate_passed"]
            if not result["gate_passed"]:
                summary["errors"].append(f"{arm}: strict baseline coverage gate failed")
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
            summary["errors"].append(f"{arm} coverage: {error}")
        write_json(output / "summary.json", summary)
    if len(outcomes) == 2:
        summary["outcome_differences"] = {
            node: {
                "A": outcomes["A"][node],
                "B": outcomes["B"][node],
                "A_phases": signatures["A"][node],
                "B_phases": signatures["B"][node],
            }
            for node in catalog
            if outcomes["A"][node] != outcomes["B"][node] or signatures["A"][node] != signatures["B"][node]
        }
        if summary["outcome_differences"]:
            summary["errors"].append("A/B test outcomes differ")
    if len(coverages) == 2:
        summary["coverage_comparison"] = compare_coverage(coverages["A"], coverages["B"])
        if not summary["coverage_comparison"]["no_drop"]:
            summary["errors"].append("Coverage source/denominator changed or executed lines were lost")
    if any(pair["status"] != "passed" for pair in pairs.values()):
        summary["errors"].append("At least one paired job failed its membership/outcome guards")
    summary["status"] = "passed" if not summary["errors"] else "failed"
    summary["paired_measurement_runner_seconds"] = sum(pair["elapsed_seconds"] for pair in pairs.values())
    write_json(output / "summary.json", summary)
    if summary["errors"]:
        raise ValueError("Aggregate comparison failed; see summary.json and raw artifacts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    subject_command = commands.add_parser("subject")
    subject_command.add_argument("--path", required=True, type=Path)
    for action in ("prepare", "bootstrap", "paired", "aggregate"):
        command = commands.add_parser(action)
        command.add_argument("--output", required=True, type=Path)
        if action == "prepare":
            command.add_argument("--seed", required=True, type=Path)
            continue
        command.add_argument("--python", required=True)
        command.add_argument("--subject", type=Path)
        if action != "aggregate":
            command.add_argument("--prepared", required=True, type=Path)
        if action in {"paired", "aggregate"}:
            command.add_argument("--bootstrap", required=True, type=Path)
        if action == "paired":
            command.add_argument("--shard", required=True, type=int, choices=range(1, 7))
        if action == "aggregate":
            command.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args(argv)
    validated_output = None
    try:
        runner = Path(os.environ["RUNNER_TEMP"]).resolve()
        subject = (getattr(args, "subject", None) or runner / "subject").resolve()
        owned = args.path.resolve() if args.action == "subject" else args.output.resolve()
        if not owned.is_relative_to(runner) or owned == runner or owned.is_relative_to(REPO):
            raise ValueError("Disposable subject/output must be under RUNNER_TEMP, outside the facility checkout")
        if args.action == "subject":
            create_subject(owned)
            return 0
        if owned.is_relative_to(subject) or subject.is_relative_to(owned):
            raise ValueError("Output and subject trees must be disjoint")
        owned.mkdir(parents=True, exist_ok=True)
        if any(owned.iterdir()):
            raise ValueError("Output directory must be empty; samples cannot be overwritten")
        validated_output = owned
        if args.action == "prepare":
            prepare(owned, args.seed.resolve())
        else:
            if sys.platform != "linux" or sys.version_info[:2] != (3, 11):
                raise ValueError("Measurements require Linux and a separate Python 3.11 tools interpreter")
            if not subject.is_relative_to(runner) or subject.is_relative_to(REPO):
                raise ValueError("Measured subject must be disposable and outside the facility checkout")
            if args.action == "bootstrap":
                bootstrap(owned, args.prepared.resolve(), args.python, subject)
            elif args.action == "paired":
                paired(owned, args.prepared.resolve(), args.bootstrap.resolve(), args.python, subject, args.shard)
            else:
                aggregate(owned, args.inputs.resolve(), args.bootstrap.resolve(), args.python, subject)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"CI shell comparison failed: {error}", file=sys.stderr)
        if validated_output is not None:
            write_json(validated_output / "failure.json", {"error": str(error), "action": args.action})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
