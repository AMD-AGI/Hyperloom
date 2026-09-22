# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay the pinned pre-release decision step against isolated, synthetic Git histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BASELINE_SHA = "b8761298c0413b08937554077ad567a764a4e1dc"
WORKFLOW_PATH = ".github/workflows/pre-release-e2e-test.yml"
OUTPUT_FIELDS = frozenset({"run", "run_scope", "tasks", "reuse", "base_version", "ci_version"})
DECIDE_ENV = {
    "EVENT": "${{ github.event_name }}",
    "REUSE_IN": "${{ inputs.reuse_ci_version }}",
    "TASKS_IN": "${{ inputs.tasks }}",
    "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
    "BASE_REF": "${{ github.event.pull_request.base.ref }}",
}
SHELL_TIMEOUT_SECONDS = 30
_REUSE_VERSION = "1.2.3.dev202609210000+ci"
_SELECTED_TASKS = "baremetal-vllm-3h,docker-sglang-3h"
_PROJECT = '[project]\nname = "ci-evidence-fixture"\nversion = "1.2.3"\n\n[project.urls]\nChangelog = "https://example.invalid/changelog"\n'
_NO_RUN = {
    "run": "false",
    "run_scope": "none",
    "tasks": "",
    "reuse": "",
    "base_version": "1.2.3",
    "ci_version": None,
}
_FULL = {
    "run": "true",
    "run_scope": "full",
    "tasks": "",
    "reuse": "",
    "base_version": "1.2.3",
    "ci_version": None,
}
_SCRIPTS = {
    "run": "true",
    "run_scope": "scripts-only",
    "tasks": "baremetal-vllm-3h,baremetal-sglang-3h,docker-vllm-3h,docker-sglang-3h",
    "reuse": "",
    "base_version": "1.2.3",
    "ci_version": None,
}
_MANUAL_TASKS = {
    "run": "true",
    "run_scope": "full",
    "tasks": "baremetal-vllm-3h,docker-sglang-3h",
    "reuse": "",
    "base_version": "1.2.3",
    "ci_version": None,
}
_MANUAL_REUSE = {
    "run": "true",
    "run_scope": "full",
    "tasks": "",
    "reuse": "1.2.3.dev202609210000+ci",
    "base_version": "1.2.3",
    "ci_version": "1.2.3.dev202609210000+ci",
}
_MANUAL_BOTH = {
    "run": "true",
    "run_scope": "full",
    "tasks": "baremetal-vllm-3h,docker-sglang-3h",
    "reuse": "1.2.3.dev202609210000+ci",
    "base_version": "1.2.3",
    "ci_version": "1.2.3.dev202609210000+ci",
}


class GateValidationError(ValueError):
    """The replay input or observed decision does not satisfy the pinned contract."""


@dataclass(frozen=True)
class GateCase:
    """Synthetic edits and a literal expected decision, independent of the shell logic."""

    name: str
    changes: Mapping[str, str]
    expected: Mapping[str, str | None]
    base_project_version: str = "1.2.3"
    event: str = "pull_request"
    tasks_in: str = ""
    reuse_in: str = ""


CASES = (
    GateCase("pyproject_comment", {"pyproject.toml": _PROJECT + "# A harmless fixture comment.\n"}, _NO_RUN),
    GateCase("changelog_url", {"pyproject.toml": _PROJECT.replace("/changelog", "/release-notes")}, _NO_RUN),
    GateCase("unrelated_file", {"unrelated.txt": "A harmless unrelated edit.\n"}, _NO_RUN),
    GateCase("version_bump", {"pyproject.toml": _PROJECT}, _FULL, base_project_version="1.2.2"),
    GateCase("ci_workflow", {WORKFLOW_PATH: "# Synthetic workflow edit; never executed.\n"}, _SCRIPTS),
    GateCase("ci_script", {".github/scripts/pre-release-e2e-dispatch.sh": "# Synthetic script edit.\n"}, _SCRIPTS),
    GateCase("ci_prompt", {".github/pre-release/prompts/pre-release/demo-3h.md": "Synthetic prompt edit.\n"}, _SCRIPTS),
    GateCase(
        "version_and_ci",
        {"pyproject.toml": _PROJECT, WORKFLOW_PATH: "# Synthetic workflow edit.\n"},
        _FULL,
        base_project_version="1.2.2",
    ),
    GateCase("manual_default", {}, _FULL, event="workflow_dispatch"),
    GateCase("manual_tasks", {}, _MANUAL_TASKS, event="workflow_dispatch", tasks_in=_SELECTED_TASKS),
    GateCase("manual_reuse", {}, _MANUAL_REUSE, event="workflow_dispatch", reuse_in=_REUSE_VERSION),
    GateCase(
        "manual_tasks_and_reuse",
        {},
        _MANUAL_BOTH,
        event="workflow_dispatch",
        tasks_in=_SELECTED_TASKS,
        reuse_in=_REUSE_VERSION,
    ),
)


def _check_yaml_mapping_keys(node: object, yaml: object) -> None:
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if not isinstance(key, yaml.ScalarNode) or key.value == "<<":
                raise GateValidationError("Workflow keys must be scalar and cannot use YAML merge keys")
            if key.value in seen:
                raise GateValidationError(f"Duplicate YAML key: {key.value}")
            seen.add(key.value)
            _check_yaml_mapping_keys(value, yaml)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _check_yaml_mapping_keys(value, yaml)


def extract_decide_script(workflow_text: str) -> str:
    """Safely extract one unconditional Bash step, without evaluating Actions expressions."""
    import yaml

    try:
        for token in yaml.scan(workflow_text):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                raise GateValidationError("Workflow aliases and anchors are not allowed in this replay")
        _check_yaml_mapping_keys(yaml.compose(workflow_text, Loader=yaml.SafeLoader), yaml)
        workflow = yaml.safe_load(workflow_text)
    except yaml.YAMLError as exc:
        raise GateValidationError("Invalid workflow YAML") from exc
    if not isinstance(workflow, dict) or not isinstance(workflow.get("jobs"), dict):
        raise GateValidationError("Workflow must contain a jobs mapping")
    resolve = workflow["jobs"].get("resolve")
    if not isinstance(resolve, dict) or not isinstance(resolve.get("steps"), list):
        raise GateValidationError("Workflow must contain jobs.resolve.steps")
    if any(key in resolve for key in ("if", "needs", "uses", "continue-on-error")):
        raise GateValidationError("Conditional or reusable resolve jobs cannot be replayed")
    steps = resolve["steps"]
    if any(not isinstance(step, dict) for step in steps):
        raise GateValidationError("Every resolve step must be a mapping")
    ids = [step["id"] for step in steps if "id" in step]
    if any(not isinstance(step_id, str) for step_id in ids) or len(ids) != len(set(ids)):
        raise GateValidationError("Resolve step IDs must be unique strings")
    matches = [step for step in steps if step.get("id") == "decide"]
    if len(matches) != 1:
        raise GateValidationError("Expected exactly one resolve step with id=decide")
    step = matches[0]
    if set(step) - {"name", "id", "run", "env", "shell"}:
        raise GateValidationError("Decide must be an unconditional run step with no extra behavior")
    if step.get("shell", "bash") != "bash":
        raise GateValidationError("Decide must use Bash")
    if "env" in step and step["env"] != DECIDE_ENV:
        raise GateValidationError("Decide environment does not match the pinned five-input contract")
    script = step.get("run")
    if not isinstance(script, str) or not script.strip():
        raise GateValidationError("Decide must contain a nonempty run string")
    if "${{" in script or "\x00" in script:
        raise GateValidationError("Decide script contains unresolved templating or a NUL character")
    return script


def parse_outputs(text: str) -> dict[str, str]:
    """Parse the six single-line GITHUB_OUTPUT fields without silently discarding anything."""
    outputs = {}
    for number, line in enumerate(text.splitlines(), 1):
        key, separator, value = line.partition("=")
        if not separator or key not in OUTPUT_FIELDS or "\x00" in value:
            raise GateValidationError(f"Invalid output field on line {number}")
        if key in outputs:
            raise GateValidationError(f"Duplicate output field: {key}")
        outputs[key] = value
    if set(outputs) != OUTPUT_FIELDS:
        raise GateValidationError(f"Missing output fields: {sorted(OUTPUT_FIELDS - outputs.keys())}")
    return outputs


def validate_outputs(
    result: Mapping[str, str],
    expected: Mapping[str, str | None],
    start_utc: datetime,
    end_utc: datetime,
) -> None:
    """Compare literal fields and check generated versions against the measured UTC minute window.

    An expected ci_version of None denotes a freshly generated version. Reused versions
    must instead be specified literally, and must survive unchanged in both output fields.
    """
    if set(result) != OUTPUT_FIELDS or set(expected) != OUTPUT_FIELDS:
        raise GateValidationError("Observed and expected decisions must contain exactly six fields")
    if any(not isinstance(value, str) for value in result.values()):
        raise GateValidationError("Every observed output must be a string")
    for key in sorted(OUTPUT_FIELDS - {"ci_version"}):
        if result[key] != expected[key]:
            raise GateValidationError(f"Unexpected {key}: expected {expected[key]!r}, got {result[key]!r}")
    if start_utc.utcoffset() is None or end_utc.utcoffset() is None or start_utc > end_utc:
        raise GateValidationError("The observation window must be timezone-aware and ordered")
    if expected["ci_version"] is not None:
        if result["ci_version"] != expected["ci_version"]:
            raise GateValidationError("ci_version did not preserve the expected literal value")
        if result["reuse"] and result["ci_version"] != result["reuse"]:
            raise GateValidationError("Reused ci_version was not preserved exactly")
        return
    if result["reuse"]:
        raise GateValidationError("A reused version requires a literal expected ci_version")
    version = re.fullmatch(rf"{re.escape(result['base_version'])}\.dev([0-9]{{12}})\+ci", result["ci_version"])
    if version is None:
        raise GateValidationError("Generated ci_version has an invalid prefix or timestamp format")
    try:
        generated_at = datetime.strptime(version[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise GateValidationError("Generated ci_version has an invalid calendar timestamp") from exc
    lower = start_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    upper = end_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if not lower <= generated_at <= upper:
        raise GateValidationError("Generated ci_version is outside the observed UTC minute window")


def _clean_environment(home: Path) -> dict[str, str]:
    names = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP")
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_DATE": "2026-09-21T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-09-21T00:00:00+00:00",
        }
    )
    return environment


def _git(repo: Path, environment: dict[str, str], *arguments: str, history: list | None = None) -> str:
    command = [
        "git",
        "-c",
        "user.name=CI Evidence Fixture",
        "-c",
        "user.email=ci-evidence@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.autocrlf=false",
        "-c",
        "protocol.allow=never",
        "-C",
        str(repo),
        *arguments,
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
    if history is not None:
        history.append(
            {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        )
    if result.returncode:
        raise GateValidationError(f"Git fixture command {arguments[0]!r} failed with exit code {result.returncode}")
    return result.stdout.strip()


def _write_fixture_file(repo: Path, name: str, content: str) -> None:
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


def _build_fixture(
    repo: Path,
    case: GateCase,
    environment: dict[str, str],
    history: list,
) -> dict[str, object]:
    repo.mkdir()
    empty_template = repo.parent / "empty-template"
    empty_template.mkdir(exist_ok=True)
    _git(repo, environment, "init", "--initial-branch=main", f"--template={empty_template}", history=history)
    base_files = {
        "pyproject.toml": _PROJECT.replace('version = "1.2.3"', f'version = "{case.base_project_version}"'),
        WORKFLOW_PATH: "# Synthetic baseline workflow; never executed.\n",
        ".github/scripts/pre-release-e2e-dispatch.sh": "# Synthetic baseline script; never executed.\n",
        ".github/pre-release/prompts/pre-release/demo-3h.md": "Synthetic baseline prompt.\n",
        "unrelated.txt": "Synthetic baseline text.\n",
    }
    for name, content in base_files.items():
        _write_fixture_file(repo, name, content)
    _git(repo, environment, "add", ".", history=history)
    _git(repo, environment, "commit", "-m", "Synthetic base", history=history)
    base_sha = _git(repo, environment, "rev-parse", "HEAD", history=history)
    _git(repo, environment, "update-ref", "refs/remotes/origin/main", base_sha, history=history)
    _git(repo, environment, "checkout", "-b", "feature", history=history)
    for name, content in case.changes.items():
        _write_fixture_file(repo, name, content)
    _git(repo, environment, "add", ".", history=history)
    _git(repo, environment, "commit", "--allow-empty", "-m", f"Synthetic feature: {case.name}", history=history)
    feature_sha = _git(repo, environment, "rev-parse", "HEAD", history=history)
    _git(repo, environment, "checkout", "--detach", base_sha, history=history)
    _git(repo, environment, "merge", "--no-ff", "feature", "-m", "Synthetic PR merge", history=history)
    head_sha = _git(repo, environment, "rev-parse", "HEAD", history=history)
    _git(repo, environment, "update-ref", "refs/pull/1/merge", head_sha, history=history)
    _git(repo, environment, "checkout", "--detach", "refs/pull/1/merge", history=history)
    parents = _git(repo, environment, "rev-list", "--parents", "-n", "1", "HEAD", history=history).split()[1:]
    if parents != [base_sha, feature_sha]:
        raise GateValidationError("Fixture HEAD must be a merge with base first and feature second")
    changed = _git(repo, environment, "diff", "--name-only", base_sha, "HEAD", history=history).splitlines()
    if sorted(changed) != sorted(case.changes):
        raise GateValidationError("Fixture changed paths differ from the case's declared edits")
    return {
        "base_sha": base_sha,
        "feature_sha": feature_sha,
        "head_sha": head_sha,
        "parents": parents,
        "changed_paths": changed,
    }


def _timeout_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _run_case(case: GateCase, root: Path, output: Path, script: Path, environment: dict[str, str]) -> dict:
    destination = output / case.name
    destination.mkdir()
    record = {
        "name": case.name,
        "expected": dict(case.expected),
        "actual": None,
        "valid": False,
        "exit_code": None,
        "timed_out": False,
        "error": None,
        "setup_commands": [],
        "stdout_file": f"{case.name}/stdout.log",
        "stderr_file": f"{case.name}/stderr.log",
        "output_file": f"{case.name}/github-output.txt",
    }
    stdout = stderr = output_text = ""
    try:
        fixture = _build_fixture(root / case.name, case, environment, record["setup_commands"])
        record["fixture"] = fixture
        output_file = root / f"{case.name}-github-output.txt"
        output_file.touch()
        inputs = {
            "EVENT": case.event,
            "REUSE_IN": case.reuse_in,
            "TASKS_IN": case.tasks_in,
            "BASE_SHA": fixture["base_sha"] if case.event == "pull_request" else "",
            "BASE_REF": "main" if case.event == "pull_request" else "",
        }
        record["inputs"] = inputs
        command = ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", script.as_posix()]
        record["command"] = command
        start = datetime.now(timezone.utc)
        clock = time.perf_counter()
        record["started_at_utc"] = start.isoformat()
        try:
            completed = subprocess.run(
                command,
                cwd=root / case.name,
                env={**environment, **inputs, "GITHUB_OUTPUT": output_file.as_posix()},
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=SHELL_TIMEOUT_SECONDS,
            )
            record["exit_code"] = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            record["timed_out"] = True
            stdout, stderr = _timeout_text(exc.stdout), _timeout_text(exc.stderr)
        finally:
            end = datetime.now(timezone.utc)
            record["ended_at_utc"] = end.isoformat()
            record["wall_seconds"] = time.perf_counter() - clock
            output_text = output_file.read_text(encoding="utf-8")
        if record["timed_out"]:
            raise GateValidationError(f"Decision script exceeded {SHELL_TIMEOUT_SECONDS} seconds")
        if record["exit_code"] != 0:
            raise GateValidationError(f"Decision script failed with exit code {record['exit_code']}")
        record["actual"] = parse_outputs(output_text)
        validate_outputs(record["actual"], case.expected, start, end)
        record["valid"] = True
    except (GateValidationError, OSError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)
    for name, text in (("stdout.log", stdout), ("stderr.log", stderr), ("github-output.txt", output_text)):
        (destination / name).write_text(text, encoding="utf-8", newline="\n")
    return record


def _save_results(output: Path, result: dict) -> None:
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gate(repo: Path, output: Path) -> int:
    """Run every required case, preserving failed attempts instead of retrying or skipping."""
    repo, output = repo.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned_names = {"results.json", "decide.sh", *(case.name for case in CASES)}
    if any((output / name).exists() for name in owned_names):
        raise GateValidationError("Gate output already contains replay artifacts; refusing to overwrite")
    result = {
        "schema_version": 1,
        "track": "gate",
        "baseline_sha": BASELINE_SHA,
        "workflow_path": WORKFLOW_PATH,
        "planned_cases": [case.name for case in CASES],
        "cases": [],
        "missing_cases": [case.name for case in CASES],
        "valid": False,
        "error": None,
        "python": sys.version,
        "platform": platform.platform(),
        "shell_timeout_seconds": SHELL_TIMEOUT_SECONDS,
        "scope": "Pure decision replay only; no build, dispatch, polling, cleanup, or GPU work.",
    }
    try:
        temporary_parent = Path(tempfile.gettempdir()).resolve()
        if temporary_parent.is_relative_to(repo) or temporary_parent.is_relative_to(output):
            raise GateValidationError("Temporary fixtures must live outside the repository and artifact directory")
        with tempfile.TemporaryDirectory(prefix="ci-evidence-gate-", dir=temporary_parent) as temporary:
            root = Path(temporary)
            home = root / "home"
            home.mkdir()
            environment = _clean_environment(home)
            workflow = _git(repo, environment, "show", f"{BASELINE_SHA}:{WORKFLOW_PATH}") + "\n"
            if (repo / WORKFLOW_PATH).read_text(encoding="utf-8") != workflow:
                raise GateValidationError("Current workflow differs from the fixed baseline workflow")
            script = extract_decide_script(workflow)
            script_path = output / "decide.sh"
            script_path.write_text(script, encoding="utf-8", newline="\n")
            result["script_sha256"] = hashlib.sha256(script.encode("utf-8")).hexdigest()
            for case in CASES:
                result["cases"].append(_run_case(case, root, output, script_path, environment))
                completed = {entry["name"] for entry in result["cases"]}
                result["missing_cases"] = [case.name for case in CASES if case.name not in completed]
                _save_results(output, result)
        result["valid"] = not result["missing_cases"] and all(case["valid"] for case in result["cases"])
    except (GateValidationError, ImportError, OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    finally:
        _save_results(output, result)
    print(
        json.dumps({"valid": result["valid"], "cases": len(result["cases"]), "results": str(output / "results.json")})
    )
    return 0 if result["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_gate(args.repo, args.output)
    except (GateValidationError, OSError) as exc:
        print(f"Gate replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
