# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that a hosted measurement measures the same thing as a local one (design 4.4.5 and 12.4 stage 1).

``python -m hyperloom.measure.consistency --checklist FILE --config FILE`` runs every test of the checklist twice, one
after the other: with the local backend on this machine, then with the hosted backend in a fresh sandbox. Both sides are
read with ``extract_benchmark_measurement``; the check compares output throughput, completed requests and error rate,
and, for a test with a patched recipe, the hashes of the files the runner rebuilt against the locally compiled ones.
Every test is one line of ``consistency.jsonl``; the last line is the summary against the graduation criteria.

The verdict is ``PASS`` only when every test was compared and every criterion holds. A test that could not be compared
(either side failed, a report is missing, the rebuilt-file hashes were not available) counts against the verdict, and a
checklist with no test is ``NO_DATA``: nothing is ever reported as consistent because nothing was looked at.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.measure.hosted import HOSTED_DIR_NAME
from hyperloom.measure.returncodes import HOSTED_RESULT_REJECTED_RC
from hyperloom.measure.runner.inventory import FARM_INVENTORY_FILE
from hyperloom.measure.socket import MeasurementBackend, RoundOutcome, TestCall, TestOutcome
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

__all__ = [
    "Criteria",
    "ItemReport",
    "ChecklistItem",
    "Side",
    "Summary",
    "FARM_INVENTORY_SCHEMA",
    "RootExpectation",
    "load_checklist",
    "main",
    "run_checklist",
    "summarize",
]

#: The schema of the file the runner lists in its result bundle (``FARM_INVENTORY_FILE``) with what it rebuilt.
FARM_INVENTORY_SCHEMA = "hyperloom.farm-inventory/v1"
REPORT_NAME = "benchmark_report.json"
REPORT_FILE = "consistency.jsonl"
DEFAULT_REPORT_DIR = Path("reports/hosted")
PASS, FAIL, NO_DATA = "PASS", "FAIL", "NO_DATA"
_PERCENT = 100.0
_UTF8 = "utf-8"
LOCAL, HOSTED = "local", "hosted"


@dataclass(frozen=True)
class Criteria:
    """The graduation criteria of design 12.4 stage 1 (stage 2 raises ``p90_pct`` to 3.5)."""

    median_pct: float = 0.5
    p90_pct: float = 1.5


@dataclass(frozen=True)
class RootExpectation:
    """What the local compile produced for one environment root: the sha256 of each upserted file (path relative to the
    package root) and the deleted paths."""

    upserts: Mapping[str, str] = field(default_factory=dict)
    deletes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChecklistItem:
    """One test of the checklist.

    ``patched_roots`` maps a root id to what the local compile made of it; an empty mapping means the test has no
    patch, and then the runner must not have rebuilt anything either.
    """

    id: str
    call: Callable[[Path, str], TestCall]
    patched_roots: Mapping[str, RootExpectation] = field(default_factory=dict)


@dataclass(frozen=True)
class Side:
    """What one backend produced for a test, reduced to the values that are compared."""

    returncode: int | None
    throughput: float | None = None
    completed: tuple[int | None, ...] = ()
    error_rate: tuple[float | None, ...] = ()
    problem: str | None = None
    log_dir: str | None = None

    @property
    def usable(self) -> bool:
        return self.problem is None and self.throughput is not None


@dataclass(frozen=True)
class ItemReport:
    """One line of ``consistency.jsonl``."""

    id: str
    local: Side
    hosted: Side
    throughput_diff_pct: float | None
    completed_equal: bool | None
    error_rate_equal: bool | None
    rebuilt_files_equal: bool | None
    signature_verified: bool
    problems: tuple[str, ...]

    @property
    def compared(self) -> bool:
        return (
            self.local.usable and self.hosted.usable and not any(p.startswith("not_comparable") for p in self.problems)
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "type": "item",
            "id": self.id,
            "compared": self.compared,
            LOCAL: _side_document(self.local),
            HOSTED: _side_document(self.hosted),
            "throughput_diff_pct": self.throughput_diff_pct,
            "completed_equal": self.completed_equal,
            "error_rate_equal": self.error_rate_equal,
            "rebuilt_files_equal": self.rebuilt_files_equal,
            "signature_verified": self.signature_verified,
            "problems": list(self.problems),
        }


def _side_document(side: Side) -> dict[str, Any]:
    return {
        "returncode": side.returncode,
        "output_throughput": side.throughput,
        "completed_requests": list(side.completed),
        "request_error_rate": list(side.error_rate),
        "problem": side.problem,
        "log_dir": side.log_dir,
    }


@dataclass(frozen=True)
class Summary:
    verdict: str
    items: int
    compared: int
    median_diff_pct: float | None
    p90_diff_pct: float | None
    completed_all_equal: bool
    rebuilt_files_all_equal: bool
    signatures_all_verified: bool
    failures: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "type": "summary",
            "verdict": self.verdict,
            "items": self.items,
            "compared": self.compared,
            "median_diff_pct": self.median_diff_pct,
            "p90_diff_pct": self.p90_diff_pct,
            "completed_all_equal": self.completed_all_equal,
            "rebuilt_files_all_equal": self.rebuilt_files_all_equal,
            "signatures_all_verified": self.signatures_all_verified,
            "failures": list(self.failures),
        }


# -- reading a side -------------------------------------------------------------------------------------------------


def _report_of(round_: RoundOutcome) -> tuple[dict[str, Any] | None, Path]:
    directory = Path(round_.output_dir)
    for path in sorted(directory.rglob(REPORT_NAME)) if directory.is_dir() else []:
        try:
            document = json.loads(path.read_text(encoding=_UTF8))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict):
            return document, directory
    return None, directory


def _error_rate(measurement: Mapping[str, Any]) -> float | None:
    """The fraction of requests that did not complete: the reported rate, else derived from the request counts.

    ``None`` when neither is available; a missing rate on both sides would otherwise compare as "equal".
    """
    reported = measurement.get("request_error_rate")
    if reported is not None:
        return float(reported)
    requested, completed = measurement.get("requested_requests"), measurement.get("completed_requests")
    if not requested or completed is None:
        return None
    return max(0.0, round(1.0 - completed / requested, 12))


def read_side(outcome: TestOutcome, log_dir: Path) -> Side:
    """Reduce an outcome to the compared values; anything missing is a problem, never a default."""
    log_dir_text = str(log_dir)
    if outcome.returncode != 0:
        reason = outcome.failure_reason or outcome.stderr.strip().splitlines()[-1:] or ["no reason"]
        return Side(outcome.returncode, problem=f"test_failed: rc={outcome.returncode} {reason}", log_dir=log_dir_text)
    rounds = [r for r in outcome.rounds if r.phase == "measure"]
    if not rounds:
        return Side(0, problem="no_measure_round", log_dir=log_dir_text)
    values = []
    for round_ in rounds:
        report, workspace = _report_of(round_)
        if report is None:
            return Side(0, problem=f"no_report: {workspace.name}", log_dir=log_dir_text)
        measurement = extract_benchmark_measurement(report, workspace=workspace)
        if not measurement.get("valid_measurement") or measurement.get("output_throughput") is None:
            return Side(0, problem=f"invalid_measurement: {workspace.name}", log_dir=log_dir_text)
        if _error_rate(measurement) is None:
            return Side(0, problem=f"no_error_rate: {workspace.name}", log_dir=log_dir_text)
        values.append(measurement)
    throughputs = [float(v["output_throughput"]) for v in values]
    return Side(
        0,
        throughput=statistics.fmean(throughputs),
        completed=tuple(v.get("completed_requests") for v in values),
        error_rate=tuple(_error_rate(v) for v in values),
        log_dir=log_dir_text,
    )


def _farm_roots(hosted_dir: Path) -> dict[str, Any] | None:
    """The ``roots`` of the runner's farm inventory, or ``None`` when there is no readable inventory of that schema."""
    try:
        document = json.loads((hosted_dir / FARM_INVENTORY_FILE).read_text(encoding=_UTF8))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != FARM_INVENTORY_SCHEMA:
        return None
    roots = document.get("roots")
    return roots if isinstance(roots, dict) else None


def _root_problems(root_id: str, expected: RootExpectation, reported: Mapping[str, Any]) -> list[str]:
    upserts = reported.get("upserts") if isinstance(reported.get("upserts"), dict) else {}
    problems = []
    for path in sorted(set(expected.upserts) | set(upserts)):
        if path not in upserts:
            problems.append(f"rebuilt_file_missing: {root_id}/{path}")
        elif path not in expected.upserts:
            problems.append(f"rebuilt_file_unexpected: {root_id}/{path}")
        elif upserts[path] != expected.upserts[path]:
            problems.append(f"rebuilt_file_differs: {root_id}/{path}")
    deletes = reported.get("deletes") if isinstance(reported.get("deletes"), list) else []
    if sorted(deletes) != sorted(expected.deletes):
        problems.append(f"rebuilt_deletes_differ: {root_id}")
    return problems


def _compare_files(item: ChecklistItem, hosted_dir: Path) -> tuple[bool | None, list[str]]:
    """Compare the runner's rebuilt files with the local compile; ``(None, [])`` when neither side has a patch."""
    roots = _farm_roots(hosted_dir)
    if roots is None:
        if not item.patched_roots:
            return None, []
        return False, [f"not_comparable: the runner listed no usable {FARM_INVENTORY_FILE}"]
    problems = []
    for root_id, expected in sorted(item.patched_roots.items()):
        if root_id not in roots or not isinstance(roots[root_id], dict):
            problems.append(f"not_comparable: the runner did not report root {root_id}")
        else:
            problems.extend(_root_problems(root_id, expected, roots[root_id]))
    problems.extend(f"rebuilt_root_unexpected: {root_id}" for root_id in sorted(set(roots) - set(item.patched_roots)))
    if not item.patched_roots and not problems:
        return None, []
    return not problems, problems


# -- one test, both ways -------------------------------------------------------------------------------------------


def _run_side(backend: MeasurementBackend, call: TestCall, log_dir: Path) -> tuple[Side, TestOutcome | None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        outcome = backend.run_test(call)
    except subprocess.TimeoutExpired as exc:
        return Side(None, problem=f"timeout: {exc.timeout}s", log_dir=str(log_dir)), None
    except Exception as exc:  # noqa: BLE001 - one broken side must not stop the checklist; it is recorded
        return Side(None, problem=f"raised: {type(exc).__name__}: {exc}", log_dir=str(log_dir)), None
    (log_dir / "outcome.json").write_text(
        json.dumps(
            {
                "returncode": outcome.returncode,
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "reason": outcome.failure_reason,
            }
        ),
        encoding=_UTF8,
    )
    return read_side(outcome, log_dir), outcome


def _signature_verified(hosted: Any, request_id: str, outcome: TestOutcome | None) -> bool:
    if outcome is None or outcome.returncode == HOSTED_RESULT_REJECTED_RC:
        return False
    record = hosted.journal.latest(request_id)
    return bool(record is not None and record.state == "finished" and record.extra.get("verified") is True)


def _percent_diff(local: float, hosted: float) -> float | None:
    if local == 0:
        return None
    return abs(hosted - local) / abs(local) * _PERCENT


def check_item(item: ChecklistItem, root: Path, local: MeasurementBackend, hosted: Any) -> ItemReport:
    """Run one test locally, then in a fresh sandbox, and compare."""
    local_call = item.call(root / item.id / LOCAL, f"{item.id}-{LOCAL}")
    hosted_id = f"{item.id}-{HOSTED}"
    hosted_call = item.call(root / item.id / HOSTED, hosted_id)
    local_side, _ = _run_side(local, local_call, root / item.id / LOCAL)
    hosted_side, hosted_outcome = _run_side(hosted, hosted_call, root / item.id / HOSTED)
    verified = _signature_verified(hosted, hosted_id, hosted_outcome)
    problems = [
        f"{name}: {side.problem}" for name, side in ((LOCAL, local_side), (HOSTED, hosted_side)) if side.problem
    ]
    if not verified:
        problems.append("signature_not_verified")
    files_equal, file_problems = _compare_files(item, root / item.id / HOSTED)
    problems.extend(file_problems if hosted_side.usable else [])
    diff = completed_equal = error_equal = None
    if local_side.usable and hosted_side.usable:
        diff = _percent_diff(local_side.throughput, hosted_side.throughput)  # type: ignore[arg-type]
        completed_equal = local_side.completed == hosted_side.completed
        error_equal = local_side.error_rate == hosted_side.error_rate
        if diff is None:
            problems.append("not_comparable: the local throughput is zero")
        if not completed_equal:
            problems.append("completed_requests_differ")
        if not error_equal:
            problems.append("error_rate_differs")
    return ItemReport(
        item.id, local_side, hosted_side, diff, completed_equal, error_equal, files_equal, verified, tuple(problems)
    )


# -- the verdict ---------------------------------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value that at least ``fraction`` of the values do not exceed."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(reports: Sequence[ItemReport], criteria: Criteria = Criteria()) -> Summary:
    """The verdict of 12.4 stage 1; ``PASS`` needs every test compared and every criterion met."""
    if not reports:
        return Summary(NO_DATA, 0, 0, None, None, False, False, False, ("the checklist has no test",))
    diffs = [r.throughput_diff_pct for r in reports if r.compared and r.throughput_diff_pct is not None]
    median = statistics.median(diffs) if diffs else None
    p90 = percentile(diffs, 0.9) if diffs else None
    failures = [f"{r.id}: {problem}" for r in reports for problem in r.problems]
    completed_ok = all(r.completed_equal is True for r in reports)
    files_ok = all(r.rebuilt_files_equal is not False for r in reports)
    signatures_ok = all(r.signature_verified for r in reports)
    uncompared = [r.id for r in reports if not r.compared]
    if uncompared:
        failures.append(f"not compared: {sorted(uncompared)}")
    if median is None or median > criteria.median_pct:
        failures.append(f"median throughput difference {median} exceeds {criteria.median_pct}")
    if p90 is None or p90 > criteria.p90_pct:
        failures.append(f"p90 throughput difference {p90} exceeds {criteria.p90_pct}")
    passed = not failures and completed_ok and files_ok and signatures_ok
    return Summary(
        PASS if passed else FAIL,
        len(reports),
        sum(1 for r in reports if r.compared),
        median,
        p90,
        completed_ok,
        files_ok,
        signatures_ok,
        tuple(dict.fromkeys(failures)),
    )


def run_checklist(
    items: Sequence[ChecklistItem],
    *,
    local: MeasurementBackend,
    hosted: Any,
    report_dir: Path,
    criteria: Criteria = Criteria(),
) -> Summary:
    """Check every item, one after the other, writing ``consistency.jsonl`` as it goes; returns the summary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[ItemReport] = []
    lines: list[str] = []
    for item in items:
        report = check_item(item, report_dir, local, hosted)
        reports.append(report)
        lines.append(json.dumps(report.to_document(), sort_keys=True))
        (report_dir / REPORT_FILE).write_text("\n".join(lines) + "\n", encoding=_UTF8)
    summary = summarize(reports, criteria)
    lines.append(json.dumps(summary.to_document(), sort_keys=True))
    (report_dir / REPORT_FILE).write_text("\n".join(lines) + "\n", encoding=_UTF8)
    return summary


# -- the command line -----------------------------------------------------------------------------------------------


def _factory(spec: str) -> Callable[..., Any]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)  # type: ignore[no-any-return]


def load_checklist(path: Path) -> list[ChecklistItem]:
    """Read a checklist file: ``{"items": [{"id", "points", "timeout_sec", ...}]}``; every field is checked."""
    document = json.loads(Path(path).read_text(encoding=_UTF8))
    entries = document.get("items") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected an object with an 'items' list")
    return [_item_of(entry, index) for index, entry in enumerate(entries)]


def _patched_roots(value: Any, index: int) -> dict[str, RootExpectation]:
    where = f"items[{index}].patched_roots"
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected {{root id: {{upserts, deletes}}}}")
    roots = {}
    for root_id, body in value.items():
        upserts = body.get("upserts", {}) if isinstance(body, dict) else None
        deletes = body.get("deletes", []) if isinstance(body, dict) else None
        if not isinstance(upserts, dict) or not all(isinstance(h, str) for h in upserts.values()):
            raise ValueError(f"{where}.{root_id}.upserts: expected {{path: sha256}}")
        if not isinstance(deletes, list) or not all(isinstance(p, str) for p in deletes):
            raise ValueError(f"{where}.{root_id}.deletes: expected a list of paths")
        roots[str(root_id)] = RootExpectation(dict(upserts), tuple(deletes))
    return roots


def _item_of(entry: Any, index: int) -> ChecklistItem:
    if not isinstance(entry, dict) or not entry.get("id") or not entry.get("points"):
        raise ValueError(f"items[{index}]: needs an 'id' and a non-empty 'points' list")
    unknown = set(entry) - {
        "id",
        "points",
        "timeout_sec",
        "kind",
        "prep",
        "accuracy",
        "env_overrides",
        "patched_roots",
        "environment",
    }
    if unknown:
        raise ValueError(f"items[{index}]: unknown keys {sorted(unknown)}")
    patched = _patched_roots(entry.get("patched_roots"), index)
    points = tuple(Path(p) for p in entry["points"])
    environment = _factory(entry["environment"])() if entry.get("environment") else None

    def call(output_dir: Path, task_id: str) -> TestCall:
        extra: dict[str, Any] = {} if environment is None else {"environment": environment}
        return TestCall(
            points=points,
            output_dir=output_dir,
            timeout_sec=int(entry.get("timeout_sec", 3600)),
            cwd=str(output_dir),
            magpie_python=sys.executable,
            task_id=task_id,
            kind=entry.get("kind", "test"),
            prep=entry.get("prep", "grid"),
            accuracy=bool(entry.get("accuracy", False)),
            env_overrides=dict(entry.get("env_overrides") or {}),
            **extra,
        )

    return ChecklistItem(str(entry["id"]), call, patched)


@dataclass(frozen=True)
class Backends:
    """The two sides of the check and the call that shuts the hosted side down."""

    local: MeasurementBackend
    hosted: Any
    close: Callable[[], None]


def build_backends(config_path: Path) -> Backends:
    """The local backend and a started hosted backend over the local sandbox provider, from a config file."""
    from hyperloom.measure.archive import DirResultArchive
    from hyperloom.measure.compile import LocalBlobStore
    from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate

    cfg = json.loads(Path(config_path).read_text(encoding=_UTF8))
    session = Path(cfg["session_dir"])
    templates = [LocalTemplate.from_mapping(name, body) for name, body in cfg["templates"].items()]
    provider = LocalSandboxProvider(
        templates, gpus=cfg["gpus"], base_dir=Path(cfg["sandbox_dir"]) if cfg.get("sandbox_dir") else None
    )
    blobs_dir = session / HOSTED_DIR_NAME / "blobs"
    context = SpecContext(
        image_ref=cfg["image_ref"],
        image_digest=cfg["image_digest"],
        gpu_type=cfg["gpu_type"],
        magpie_commit=cfg["magpie_commit"],
        inferencex_commit=cfg["inferencex_commit"],
        blobs=LocalBlobStore(blobs_dir),
        power_cap_w=cfg.get("power_cap_w"),
        partition_mode=cfg.get("partition_mode"),
        model_hashes=cfg.get("model_hashes") or {},
        shared_roots=tuple(Path(p) for p in cfg.get("shared_roots", ())),
        model_dir=Path(cfg["model_dir"]) if cfg.get("model_dir") else None,
        blob_root=str(blobs_dir),
    )
    hosted_config = HostedConfig(
        session_dir=session,
        run_id=cfg.get("run_id", "consistency"),
        spec=context,
        runner_templates={int(k): v for k, v in cfg["runner_templates"].items()},
        archive=DirResultArchive(Path(cfg["archive_dir"])),
        max_parallel_tests=1,
        poll_interval_sec=float(cfg.get("poll_interval_sec", 5.0)),
        runner_version=cfg.get("runner_version"),
    )
    hosted = HostedMeasurementBackend(provider, hosted_config)
    hosted.start()
    local_cls = _factory(cfg.get("local_backend", "hyperloom.measure.local:LocalMeasurementBackend"))

    def close() -> None:
        hosted.close()
        provider.close()

    return Backends(local_cls(), hosted, close)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.measure.consistency", description=__doc__)
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--median-pct", type=float, default=Criteria.median_pct)
    parser.add_argument("--p90-pct", type=float, default=Criteria.p90_pct)
    args = parser.parse_args(argv)
    try:
        items = load_checklist(args.checklist)
    except (OSError, ValueError) as exc:
        print(f"checklist unusable: {exc}", file=sys.stderr)
        return 2
    backends = build_backends(args.config)
    try:
        summary = run_checklist(
            items,
            local=backends.local,
            hosted=backends.hosted,
            report_dir=args.report_dir,
            criteria=Criteria(args.median_pct, args.p90_pct),
        )
    finally:
        backends.close()
    print(json.dumps(summary.to_document(), indent=2, sort_keys=True))
    return 0 if summary.verdict == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
