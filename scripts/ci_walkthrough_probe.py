# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Temporary, target-only inclusive wall timers for the two walkthrough tests."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path

import pytest


_TARGET_FILE = "src/hyperloom/inference_optimizer/tests/test_optimize_loop_walkthrough.py"
_TARGET_NAMES = {
    "test_a_baseline_carries_the_run_into_the_optimisation_phase_with_work",
    "test_both_arms_dry_walks_the_rest_of_the_chain",
}
_TARGETS = (
    ("loop.coordinator", "Coordinator", "__init__", "coordinator.init", ""),
    ("loop.coordinator", "Coordinator", "tick", "coordinator.tick", "tick"),
    ("loop.coordinator", "Coordinator", "stop", "coordinator.stop", ""),
    ("loop.coordinator", "Coordinator", "_reactor_pass", "reactor", "agent"),
    ("loop.dispatcher", "DispatcherCollaborator", "_pump_dispatcher_once", "dispatcher.pump", ""),
    ("loop.dispatcher", "DispatcherCollaborator", "_reclaim_stale_dispatch_state", "dispatcher.reclaim", ""),
    ("loop.dispatcher", "DispatcherCollaborator", "_spawn_fitting_queued", "dispatcher.spawn", "spawn"),
    ("loop.dispatcher", "DispatcherCollaborator", "_reap_dispatched_task", "dispatcher.reap", ""),
    ("phases.framework", "FrameworkPhase", "_pump_framework_agent_phase_safely", "framework.pump", ""),
    ("enablement.lane", "EnablementLane", "_pump_enablement_safely", "enablement.pump", ""),
    ("phases.machine", "MachinePhase", "_advance_phase_if_needed", "phase.advance", ""),
    ("bringup.reconcile", "Reconciler", "run", "reconciler.run", ""),
    ("loop.writeback", "WritebackCollaborator", "_replay_resume_if_needed", "resume.replay", ""),
    ("state.shared_state", "SharedState", "save", "state.save", ""),
    ("state.shared_state", "SharedState", "to_dict", "state.to_dict", ""),
    ("loop.conversation", "ConversationCollaborator", "_compose_prompt", "prompt.compose", "agent"),
)


def _is_target(nodeid: str) -> bool:
    path, separator, name = nodeid.replace("\\", "/").partition("::")
    return bool(separator) and path.endswith(_TARGET_FILE) and name in _TARGET_NAMES


def _snapshot(obj) -> dict:
    state = vars(obj).get("shared_state")
    fields = vars(state) if state is not None else {}
    return {"phase": fields.get("phase"), "tick": fields.get("tick")}


class Probe:
    """Wrap methods without changing their return, exception, or await behavior."""

    def __init__(self, nodeid: str) -> None:
        self.record = {
            "nodeid": nodeid,
            "metrics": {},
            "ticks": [],
            "spawned_tasks": 0,
            "errors": [],
            "restored": False,
        }
        self._patches = []

    def error(self, where: str, exc: Exception) -> None:
        if len(self.record["errors"]) < 10:
            self.record["errors"].append(f"{where}: {type(exc).__name__}: {exc}")

    def _begin(self, label, mode, args, kwargs):
        context = None
        try:
            if mode == "agent":
                agent = args[1] if len(args) > 1 else kwargs.get("agent_name", "unknown")
                label = f"{label}.{agent}"
            elif mode == "tick":
                context = {
                    "n": args[1] if len(args) > 1 else kwargs.get("n", 1),
                    "before": _snapshot(args[0]),
                    "spawned_before": self.record["spawned_tasks"],
                }
        except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            self.error("timer begin", exc)
        return label, context

    def _finish(self, label, mode, args, context, elapsed, succeeded, result):
        try:
            metric = self.record["metrics"].setdefault(label, {"count": 0, "total_s": 0.0, "max_s": 0.0, "raised": 0})
            metric["count"] += 1
            metric["total_s"] += elapsed
            metric["max_s"] = max(metric["max_s"], elapsed)
            metric["raised"] += int(not succeeded)
            if mode == "spawn" and succeeded:
                self.record["spawned_tasks"] += len(result)
            elif mode == "tick" and context is not None:
                context.update(
                    elapsed_s=elapsed,
                    after=_snapshot(args[0]),
                    succeeded=succeeded,
                    spawned_tasks=self.record["spawned_tasks"] - context.pop("spawned_before"),
                )
                self.record["ticks"].append(context)
        except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            self.error("timer finish", exc)

    def wrap(self, owner, name: str, label: str, mode: str = "") -> None:
        original = getattr(owner, name)
        owned = name in vars(owner)
        previous = vars(owner).get(name)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapped(*args, **kwargs):
                key, context = self._begin(label, mode, args, kwargs)
                started = time.perf_counter()
                succeeded = False
                result = None
                try:
                    result = await original(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    elapsed = time.perf_counter() - started
                    self._finish(key, mode, args, context, elapsed, succeeded, result)

        else:

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                key, context = self._begin(label, mode, args, kwargs)
                started = time.perf_counter()
                succeeded = False
                result = None
                try:
                    result = original(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    elapsed = time.perf_counter() - started
                    self._finish(key, mode, args, context, elapsed, succeeded, result)

        setattr(owner, name, wrapped)
        self._patches.append((owner, name, owned, previous))

    def install(self) -> None:
        for module, class_name, method, label, mode in _TARGETS:
            owner = getattr(importlib.import_module(f"hyperloom.orchestrator.{module}"), class_name)
            self.wrap(owner, method, label, mode)

    def restore(self) -> None:
        while self._patches:
            owner, name, owned, previous = self._patches.pop()
            if owned:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.record["restored"] = True


def pytest_configure(config) -> None:
    workerinput = getattr(config, "workerinput", {})
    config._walkthrough_probe_run_id = workerinput.get("walkthrough_probe_run_id", uuid.uuid4().hex)
    config._walkthrough_probe_records = []
    config._walkthrough_probe_reports = {}


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    node.workerinput["walkthrough_probe_run_id"] = node.config._walkthrough_probe_run_id


def _write_records(config) -> None:
    worker = getattr(config, "workerinput", {}).get("workerid", "controller")
    path = Path(config.rootpath) / f"walkthrough-probe-{worker}.json"
    document = {
        "run_id": config._walkthrough_probe_run_id,
        "worker": worker,
        "python": sys.version,
        "platform": platform.platform(),
        "github_sha": os.environ.get("GITHUB_SHA", ""),
        "timing": "Inclusive wall seconds; nested timers overlap; instrumentation has overhead.",
        "tests": config._walkthrough_probe_records,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    if not _is_target(item.nodeid):
        yield
        return
    probe = Probe(item.nodeid)
    probe.record["pytest_reports"] = item.config._walkthrough_probe_reports.setdefault(item.nodeid, {})
    started = time.perf_counter()
    try:
        probe.install()
    except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        probe.error("probe install", exc)
    probe.record["probe_setup_s"] = time.perf_counter() - started
    body_started = time.perf_counter()
    try:
        yield
    finally:
        probe.record["wrapped_call_s"] = time.perf_counter() - body_started
        try:
            probe.restore()
        except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            probe.error("probe restore", exc)
        item.config._walkthrough_probe_records.append(probe.record)
        try:
            _write_records(item.config)
        except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            probe.error("probe output", exc)
            print(f"walkthrough probe output failed: {exc}", file=sys.stderr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not _is_target(item.nodeid):
        return
    report = outcome.get_result()
    item.config._walkthrough_probe_reports.setdefault(item.nodeid, {})[report.when] = {
        "duration_s": report.duration,
        "outcome": report.outcome,
    }
    for record in item.config._walkthrough_probe_records:
        if record["nodeid"] == item.nodeid:
            try:
                _write_records(item.config)
            except (OSError, TypeError, ValueError) as exc:
                print(f"walkthrough probe report output failed: {exc}", file=sys.stderr)
            break


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if hasattr(config, "workerinput"):
        return
    documents = []
    for path in sorted(Path(config.rootpath).glob("walkthrough-probe-*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            terminalreporter.write_line(f"walkthrough probe cannot read {path.name}: {exc}")
            continue
        if document.get("run_id") == config._walkthrough_probe_run_id:
            documents.append(document)
    if not documents:
        return
    terminalreporter.section("walkthrough probe: inclusive wall timers (nested totals overlap)")
    for document in documents:
        terminalreporter.write_line(
            f"worker={document['worker']} sha={document['github_sha']} python={document['python']}"
        )
        for record in document["tests"]:
            terminalreporter.write_line(record["nodeid"])
            terminalreporter.write_line(
                f"  setup={record['probe_setup_s']:.6f}s wrapped_call={record['wrapped_call_s']:.6f}s "
                f"spawned={record['spawned_tasks']} restored={record['restored']} errors={record['errors']}"
            )
            for label, metric in sorted(record["metrics"].items(), key=lambda pair: pair[1]["total_s"], reverse=True):
                terminalreporter.write_line(
                    f"  {label}: count={metric['count']} total={metric['total_s']:.6f}s "
                    f"max={metric['max_s']:.6f}s raised={metric['raised']}"
                )
            for tick in record["ticks"]:
                terminalreporter.write_line(
                    f"  tick(n={tick['n']}): {tick['elapsed_s']:.6f}s {tick['before']} -> "
                    f"{tick['after']} spawned={tick['spawned_tasks']} succeeded={tick['succeeded']}"
                )
