# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 1 (session reader, entry point, server): the session files the Inspector finds and reads are
the ones Hyperloom's own writers produce, and the names it maps to agents and views (task kinds, LLM components,
bus topics, state keys, ledger and journal fields) are the ones Hyperloom uses.

Each test writes a session artifact with the real producer (see :mod:`.producers`) or compares the Inspector's
names with the producer's constants, then reads it the way the Inspector does. A failure means a producer changed
what it writes: update the Inspector reader the failure message names, then this test. A test marked
``xfail(strict=True)`` pins a known break; it starts failing ("XPASS(strict)") once the Inspector is fixed, and then
the marker comes off.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs this interpreter on a fixed argv.
import sys
from pathlib import Path

import pytest

from hyperloom.inspector import derive, graph
from hyperloom.inspector.collectors import parse_ts
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.conftest import FAKE_SECRET
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


_WATCHER = "watcher.py"


_REDACT = "redact.py"


_GRAPH = "graph.py"


def test_inspector_reads_the_paths_session_paths_names(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session import paths, session_paths
    from hyperloom.inspector import session_layout

    session = producers.new_session(tmp_path)
    watcher = producers.watch(session)
    pairs = {
        "paths.shared_db_path_for": (watcher.db.path, paths.shared_db_path_for(session)),
        "session_paths.llm_calls_path": (watcher._llm_tail.path, session_paths.llm_calls_path(session)),
        "session_paths.sbd_v6_timeline_dir": (watcher._timeline_files.root, session_paths.sbd_v6_timeline_dir(session)),
        "session_paths.state_path": (session / session_layout.STATE_FILE, session_paths.state_path(session)),
    }
    for producer, (inspector, written) in pairs.items():
        assert inspector == written, contract_message(producer, _WATCHER, f"Inspector reads {inspector}, not {written}")
    event = session_paths.sbd_v6_timeline_event_path(session, 1, "baseline")
    assert event.match(watcher._timeline_files.pattern), contract_message(
        "session_paths.sbd_v6_timeline_event_path", _WATCHER, f"{event.name} is outside the Inspector's glob"
    )


async def test_coordinator_db_written_by_bus_and_registry_is_read(tmp_path: Path) -> None:
    from dataclasses import fields

    from hyperloom.inspector.collectors import CoordinatorDb
    from hyperloom.orchestrator.bus.message_bus import Message
    from hyperloom.orchestrator.state.task_registry import Task

    session = producers.new_session(tmp_path)
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="baseline", params={"domain": "d"}))
        await producers.announce_result(coord, task, {"status": "succeeded", "output_throughput": 10.0})
        db = CoordinatorDb(session / "storage" / "coordinator.db")
        events, tasks = db.new_events(), db.tasks()
    where = "collectors.py (CoordinatorDb)"
    assert [t["task_id"] for t in tasks] == [task.task_id], contract_message("TaskRegistry", where, f"tasks: {tasks}")
    assert set(tasks[0]) <= {f.name for f in fields(Task)}, contract_message("Task / tasks DDL", where, tasks[0])
    assert tasks[0]["params"] == {"domain": "d"} and tasks[0]["history"], contract_message("Task", where, tasks[0])
    assert set(events[0]) == {f.name for f in fields(Message)}, contract_message("Message / events DDL", where, events)
    payload = events[0]["payload"]
    assert payload["task_id"] == task.task_id and payload["result"]["output_throughput"] == 10.0, contract_message(
        "MessageBus.append_and_seq", where, f"payload not decoded: {payload!r}"
    )


async def test_session_with_private_ledger_is_read_or_reported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_LEDGER_DIR", str(tmp_path / "ledgers"))
    session = producers.new_session(tmp_path)
    state = producers.new_state(operator_extra_env={"HF_TOKEN": "hf_" + "a" * 30})
    async with producers.coordinator_db(session, state) as coord:
        state.save(session)
        await producers.run_task(coord, producers.TaskRun(kind="baseline", states=()))
    watcher = producers.watch(session)
    badges = [b["text"] for b in watcher.snapshot(include_streams=False)["badges"]]
    assert watcher.tasks or any("ledger" in b for b in badges), contract_message(
        "state_file.database_target / paths.db_path_for", "watcher.py, cli.py", f"no tasks and no badge: {badges}"
    )


async def test_session_with_private_ledger_is_read(tmp_path: Path, monkeypatch) -> None:
    # As a run starts: the manifest (with the session id the ledger is keyed by) before the Coordinator opens it.
    from hyperloom.inference_optimizer.session import paths

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(paths.ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    session = producers.new_session(tmp_path)
    producers.write_manifest(session, "/models/Qwen3-8B")
    state = producers.new_state(operator_extra_env={"HF_TOKEN": "hf_" + "a" * 30})
    async with producers.coordinator_db(session, state) as coord:
        state.save(session)
        await producers.run_task(coord, producers.TaskRun(kind="baseline", states=()))
    db_path = coord.path
    watcher = producers.watch(session)
    where = "watcher.py (session_layout.coordinator_db)"
    assert watcher.db.path == paths.db_path_for(session) == db_path, contract_message(
        "paths.db_path_for", where, f"Inspector reads {watcher.db.path}, Hyperloom writes {db_path}"
    )
    assert [t["kind"] for t in watcher.tasks] == ["baseline"], contract_message(
        "paths.db_path_for", where, watcher.tasks
    )


@pytest.mark.parametrize("ledger_dir", ["named", "tmpdir"])
@pytest.mark.parametrize("recorded", ["manifest", "state", "none"])
def test_the_inspector_s_private_db_path_is_hyperloom_s(
    tmp_path: Path, monkeypatch, ledger_dir: str, recorded: str
) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    if ledger_dir == "named":
        monkeypatch.setenv(paths.ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    else:
        monkeypatch.delenv(paths.ENV_LEDGER_DIR, raising=False)
    session = producers.new_session(tmp_path)
    if recorded == "manifest":
        producers.write_manifest(session, "/models/Qwen3-8B")
    elif recorded == "state":
        producers.new_state().save(session)
    pairs = {
        "paths.private_db_path_for": (session_layout.private_db_path(session), paths.private_db_path_for(session)),
        "paths.ledger_mark_path": (session / session_layout.LEDGER_MARK, paths.ledger_mark_path(session)),
        "paths.ENV_LEDGER_DIR": (session_layout.ENV_LEDGER_DIR, paths.ENV_LEDGER_DIR),
    }
    for producer, (inspector, hyperloom) in pairs.items():
        assert inspector == hyperloom, contract_message(producer, "session_layout.py", f"{inspector} != {hyperloom}")


def test_a_relative_ledger_dir_is_refused_as_hyperloom_refuses_it(tmp_path: Path, monkeypatch) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    monkeypatch.setenv(paths.ENV_LEDGER_DIR, "ledgers")
    with pytest.raises(ValueError):
        paths.private_db_path_for(tmp_path)
    with pytest.raises(ValueError):
        session_layout.private_db_path(tmp_path)


# A credential of each shape Hyperloom's ``env_safety.redact_secret_values`` masks, as (text, the secret in it).
_VALUE = FAKE_SECRET


_SECRET_TEXTS = {
    "authorization_header": (f"Authorization: Basic {_VALUE}", _VALUE),
    "bearer": (f"Bearer {_VALUE}", _VALUE),
    "sk_key": (f"sk-{_VALUE}{_VALUE}", _VALUE),
    "ak_key": (f"ak-{_VALUE}", _VALUE),
    "langfuse_pk": (f"pk-lf-{_VALUE}", _VALUE),
    "github_token": (f"ghp_{_VALUE}{_VALUE}", _VALUE),
    "github_pat": (f"github_pat_{_VALUE}{_VALUE}", _VALUE),
    "aws_key_id": ("AKIA" + "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNOP"),
    "jwt": ("eyJ" + f"{_VALUE}.{_VALUE}.{_VALUE}", _VALUE),
    "apim_key": (f"Ocp-Apim-Subscription-Key: {_VALUE}", _VALUE),
    "authorization_assign": (f"authorization={_VALUE}", _VALUE),
    "custom_headers": (f"OPENAI_CUSTOM_HEADERS=x-key: {_VALUE}", _VALUE),
    "api_key_assign": (f"MY_API_KEY={_VALUE}", _VALUE),
    "password_assign": (f'DB_PASSWORD="{_VALUE}"', _VALUE),
    "credential_assign": (f"GCP_CREDENTIAL={_VALUE}", _VALUE),
    "auth_assign": (f"AUTH={_VALUE}", _VALUE),
    "token_assign": (f"HF_TOKEN={_VALUE}", _VALUE),
}


def test_every_hyperloom_redaction_pattern_has_a_sample() -> None:
    from hyperloom.common.env_safety import _SECRET_REDACTION_PATTERNS

    texts = [text for text, _ in _SECRET_TEXTS.values()]
    unsampled = [p.pattern for p, _ in _SECRET_REDACTION_PATTERNS if not any(p.search(t) for t in texts)]
    assert not unsampled, contract_message(
        "env_safety._SECRET_REDACTION_PATTERNS", _REDACT, f"add a sample to _SECRET_TEXTS for: {unsampled}"
    )


@pytest.mark.parametrize("shape", list(_SECRET_TEXTS))
def test_redact_text_masks_what_hyperloom_masks(shape: str) -> None:
    from hyperloom.common.env_safety import redact_secret_values
    from hyperloom.inspector.redact import redact_text

    text, secret = _SECRET_TEXTS[shape]
    assert secret not in redact_secret_values(text), f"sample {shape!r} is not one Hyperloom masks; fix the sample"
    assert secret not in redact_text(text), contract_message(
        "env_safety.redact_secret_values", _REDACT, f"{shape}: served as {redact_text(text)!r}"
    )


def test_redact_masks_every_hyperloom_secret_env_name() -> None:
    from hyperloom.common.env_safety import BENCHMARK_SECRET_ENV_NAMES
    from hyperloom.inspector.redact import redact, redact_text

    as_key = [n for n in sorted(BENCHMARK_SECRET_ENV_NAMES) if redact({n: _VALUE})[n] == _VALUE]
    as_text = [n for n in sorted(BENCHMARK_SECRET_ENV_NAMES) if _VALUE in redact_text(f"{n}={_VALUE}")]
    assert not as_key and not as_text, contract_message(
        "env_safety.BENCHMARK_SECRET_ENV_NAMES", _REDACT, f"unmasked as a key: {as_key}; as NAME=value: {as_text}"
    )


@pytest.mark.parametrize(
    "field", ["env", "extra_envs", "launch_env", "operator_extra_env", "environment", "environ", "envoy", "ENV", "x"]
)
def test_the_inspector_s_environment_fields_are_hyperloom_s(field: str) -> None:
    from hyperloom.common.env_safety import _is_env_field
    from hyperloom.inspector.redact import is_env_field

    assert is_env_field(field) == _is_env_field(field), contract_message(
        "env_safety._is_env_field", _REDACT, f"{field!r}: {is_env_field(field)}"
    )


def test_conversation_rows_written_by_append_conversation_are_found(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    producers.append_conversation_row(session, component="orchestration", call_id="c1", prompt="p", response="r")
    producers.append_conversation_row(session, component="specialist", task_id="t1", turn=1, prompt="sp")
    watcher = producers.watch(session)
    by_call, by_turn = watcher.conversation(call_id="c1"), watcher.conversation(task_id="t1", turn="1")
    where = "watcher.py (Watcher.conversation)"
    assert by_call and by_call["prompt"] == "p" and by_call["response"] == "r", contract_message(
        "conversation_trace.append_conversation", where, f"call c1: {by_call}"
    )
    assert by_turn and by_turn["prompt"] == "sp", contract_message(
        "conversation_trace.append_conversation", where, f"task t1 turn 1: {by_turn}"
    )


def test_lock_and_pod_history_written_by_session_lock(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    lock = producers.acquire_lock(session)
    try:
        watcher = producers.watch(session)
        status, history = watcher.optimizer_status(), watcher.pod_history
    finally:
        lock.release()
    assert status == {"alive": True, "pid": os.getpid()}, contract_message("SessionLock.acquire", _WATCHER, status)
    assert [h.get("pid") for h in history] == [os.getpid()] and history[0].get("hostname"), contract_message(
        "SessionLock._append_pod_history", _WATCHER, f"pod history: {history}"
    )


def test_manifest_keys_the_inspector_reads_are_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    session = producers.new_session(tmp_path)
    written = producers.write_manifest(session, "/models/Qwen3-8B")
    watcher = producers.watch(session)
    watcher.manifest = recording = producers.RecordingDict(watcher.manifest)
    info = watcher.run_info()
    assert info["model"] == "Qwen3-8B", contract_message("manifest.build_manifest", _WATCHER, info["model"])
    unwritten = sorted(recording.read - set(written))
    assert not unwritten, contract_message("manifest.build_manifest", _WATCHER, f"read, never written: {unwritten}")


def test_manifest_without_a_model_name_reads_no_other_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    session = producers.new_session(tmp_path)
    written = producers.write_manifest(session, "/models/Qwen3-8B")
    watcher = producers.watch(session)
    watcher.manifest = recording = producers.RecordingDict({k: v for k, v in written.items() if k != "model_name"})
    watcher.state = {}
    assert watcher.run_info()["model"] is None
    unwritten = sorted(recording.read - set(written))
    assert not unwritten, contract_message("manifest.build_manifest", _WATCHER, f"read, never written: {unwritten}")


@pytest.mark.parametrize("name", ["manifest.json", "state.json"])
def test_a_started_session_is_found_from_the_workspace(tmp_path: Path, monkeypatch, name: str) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector.cli import resolve_session_dir

    assert name in paths.SESSION_FILES, contract_message("paths.SESSION_FILES", "cli.py", paths.SESSION_FILES)
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.setenv(paths.ENV_CURRENT_SESSION_DIR, "")
    session = producers.new_session(tmp_path)
    if name == "manifest.json":
        producers.write_manifest(session, "/models/Qwen3-8B")
    else:
        producers.new_state().save(session)
    found = None
    try:
        found = resolve_session_dir(tmp_path)
    except FileNotFoundError:
        pass
    assert found == session, contract_message("paths.make_session_dir / SESSION_FILES", "cli.py", f"found {found}")


def test_artifact_index_leaves_out_private_session_dirs(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import artifacts

    session = producers.new_session(tmp_path)
    archive = paths.archive_root(session)
    archive.mkdir(parents=True)
    (archive / "package.tar").write_text("x", encoding="utf-8")
    listed = [a["path"] for a in artifacts.index(session, {})]
    private = [p for p in listed if p.split("/", 1)[0] in paths.PRIVATE_SESSION_SUBDIRS]
    assert not private, contract_message("paths.PRIVATE_SESSION_SUBDIRS", "artifacts.py", f"served: {private}")


def test_the_inspector_s_private_subdirs_are_hyperloom_s() -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    assert session_layout.PRIVATE_SESSION_SUBDIRS == paths.PRIVATE_SESSION_SUBDIRS, contract_message(
        "paths.PRIVATE_SESSION_SUBDIRS", "session_layout.py", session_layout.PRIVATE_SESSION_SUBDIRS
    )


def test_the_inspector_s_session_files_are_hyperloom_s() -> None:
    from hyperloom.inference_optimizer.session import paths
    from hyperloom.inspector import session_layout

    assert session_layout.SESSION_FILES == paths.SESSION_FILES, contract_message(
        "paths.SESSION_FILES", "session_layout.py (cli._is_session)", session_layout.SESSION_FILES
    )


def test_agent_resource_readings_are_not_an_agent() -> None:
    from hyperloom.inference_optimizer.session.session_paths import RESOURCES_RELATIVE_PATH
    from hyperloom.inspector import graph

    node, _task = graph.producer_for(RESOURCES_RELATIVE_PATH.as_posix(), {})
    assert node != RESOURCES_RELATIVE_PATH.name, contract_message("session_paths.RESOURCES_RELATIVE_PATH", _GRAPH, node)


def test_retried_run_dirs_are_attributed_to_their_task(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir
    from hyperloom.inspector import graph

    session = producers.new_session(tmp_path)
    first, retry = (unique_runs_dir(session, "integrate", "t-int") for _ in range(2))
    attributed = [
        graph.producer_for(d.relative_to(session).as_posix() + "/x.log", {"t-int": "n"}) for d in (first, retry)
    ]
    assert attributed == [("n", "t-int")] * 2, contract_message("session_paths.unique_runs_dir", _GRAPH, attributed)


_VARIANT_NAMES = ["mnbt_8192", "max-num-seqs=64 tp=2", "a/b", "ünï.cödé", "x" * 80]


@pytest.mark.parametrize("name", _VARIANT_NAMES)
def test_variant_dir_names_are_the_grid_runner_s(name: str) -> None:
    from hyperloom.inspector.session_layout import variant_dir_name
    from hyperloom.orchestrator.actions.executors.grid._grid_base import _safe

    assert variant_dir_name(name) == _safe(name), contract_message(
        "_grid_base._safe", "session_layout.py (variant_dir_name)", variant_dir_name(name)
    )


# The node each task kind of ACTION_CATALOGUE is drawn as; a kind the catalogue gains fails until it is placed here.
_NODE_OF_KIND = {
    "baseline": "exec:baseline",
    "explore": "exec:explore",
    "gemm_tuning": "exec:gemm_tuning",
    "integrate": "exec:integrate",
    "integrate_patch": "exec:integrate_patch",
    "kernel_agent": graph.KERNEL_AGENT,
    "profile": "exec:profile",
    "replay_warm_recipe": "exec:replay_warm_recipe",
    "report": "exec:report",
    "roofline": "exec:roofline",
    "session_breakdown": "exec:session_breakdown",
    "specialist": "spec:specialist",
    "target_analysis": "exec:target_analysis",
}


# Task kinds graph.task_node special-cases that Hyperloom no longer creates. kernel_opt was retired with the kernel
# harness (d501561ef) and framework_agent with its action (636f3ae0f); no producer of ``kernel`` or ``framework`` as a task
# kind was found. Kept so older sessions draw the same.
_LEGACY_KINDS = frozenset({"kernel_opt", "kernel", "framework_agent", "framework"})


# The node each LLM component of the ledger's closed vocabulary (llm_trace.VALID_COMPONENTS) is drawn on when its row
# names no task; a component the vocabulary gains fails until it is placed here.
_NODE_OF_COMPONENT = {
    "orchestration": "orchestration",
    "critic": "critic",
    "kernel_agent": graph.KERNEL_AGENT,
    "framework": graph.FRAMEWORK_AGENT,
    "specialist": "specialist",
    "geak": "geak",
    "forge": "forge",
    "tracelens": "tracelens",
    "breakdown": "breakdown",
    "quantization": "quantization",
}


def test_every_catalogue_task_kind_is_placed() -> None:
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

    assert set(_NODE_OF_KIND) == set(ACTION_CATALOGUE), contract_message(
        "action_surfaces.ACTION_CATALOGUE",
        _GRAPH,
        f"new kinds: {sorted(set(ACTION_CATALOGUE) - set(_NODE_OF_KIND))}; "
        f"gone: {sorted(set(_NODE_OF_KIND) - set(ACTION_CATALOGUE))} — decide their node in graph.task_node",
    )


def test_catalogue_task_kinds_map_to_their_nodes() -> None:
    wrong = {
        kind: graph.task_node({"kind": kind})
        for kind, node in _NODE_OF_KIND.items()
        if kind != "kernel_agent" and graph.task_node({"kind": kind}) != node
    }
    assert not wrong, contract_message("action_surfaces.ACTION_CATALOGUE", "graph.py (task_node)", wrong)


def test_kernel_agent_tasks_share_the_kernel_agent_node() -> None:
    from hyperloom.inference_optimizer.session.paths import _SESSION_SKELETON

    kernel_dirs = [d for d in _SESSION_SKELETON if d.startswith("kernel-agent")]
    nodes = {
        "task kind kernel_agent": graph.task_node({"kind": "kernel_agent"}),
        "LLM component kernel_agent": graph.agent_node("kernel_agent"),
        **{f"{d}/": graph.producer_for(f"{d}/x", {})[0] for d in kernel_dirs},
    }
    assert len(set(nodes.values())) == 1, contract_message("ACTION_CATALOGUE / VALID_COMPONENTS", _GRAPH, nodes)


def test_kinds_the_inspector_special_cases_are_produced_or_legacy() -> None:
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

    special = {"kernel_opt", "kernel", "framework_agent", "framework", "specialist", "baseline"}
    unknown = sorted(special - set(ACTION_CATALOGUE) - _LEGACY_KINDS)
    assert not unknown, contract_message("action_surfaces.ACTION_CATALOGUE", "graph.py, watcher.py", unknown)


def test_llm_components_map_to_deliberate_nodes() -> None:
    from hyperloom.inference_optimizer.trace.llm_trace import VALID_COMPONENTS

    assert set(_NODE_OF_COMPONENT) == set(VALID_COMPONENTS), contract_message(
        "llm_trace.VALID_COMPONENTS",
        "graph.py (agent_node)",
        f"new: {sorted(set(VALID_COMPONENTS) - set(_NODE_OF_COMPONENT))}; "
        f"gone: {sorted(set(_NODE_OF_COMPONENT) - set(VALID_COMPONENTS))}",
    )
    wrong = {c: graph.agent_node(c) for c, node in _NODE_OF_COMPONENT.items() if graph.agent_node(c) != node}
    assert not wrong, contract_message("llm_trace.VALID_COMPONENTS", "graph.py (agent_node)", wrong)


def test_llm_components_the_inspector_branches_on_are_written() -> None:
    from hyperloom.inference_optimizer.trace.llm_trace import VALID_COMPONENTS

    branched = {*graph.LLM_ROLES}
    unknown = sorted(branched - VALID_COMPONENTS)
    assert not unknown, contract_message("llm_trace.VALID_COMPONENTS", "graph.py", unknown)


def test_state_keys_are_shared_state_fields() -> None:
    from dataclasses import fields

    from hyperloom.orchestrator.state.shared_state import SharedState

    unwritten = sorted(set(derive.STATE_KEYS) - {f.name for f in fields(SharedState)})
    assert not unwritten, contract_message("SharedState", "derive.py (STATE_KEYS)", f"never written: {unwritten}")


def test_bus_vocabulary() -> None:
    from hyperloom.orchestrator.bus.message_bus import ROLE_SUBSCRIPTIONS, TOPIC_ALLOWLIST

    assert "delegated_result" in TOPIC_ALLOWLIST, contract_message(
        "message_bus.TOPIC_ALLOWLIST", "watcher.py (_poll_db)", sorted(TOPIC_ALLOWLIST)
    )
    known = {graph.COORDINATOR, *graph.LLM_ROLES}
    assert set(ROLE_SUBSCRIPTIONS) <= known, contract_message(
        "message_bus.ROLE_SUBSCRIPTIONS", "graph.py (LLM_ROLES, agent_node)", sorted(set(ROLE_SUBSCRIPTIONS) - known)
    )


async def test_task_history_written_by_registry_drives_progress(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    notes = producers.variant_progress_notes(["a", "b"], ["succeeded", "succeeded"], [100.0, 110.0])
    async with producers.coordinator_db(session) as coord:
        await producers.run_task(coord, producers.TaskRun(kind="explore", progress=notes))
    row = producers.watch(session).tasks[0]
    progress = derive.task_progress(row["history"], None, parse_ts)
    labels = [(u["label"], u["status"], u["output_throughput"]) for u in progress.get("units", [])]
    assert labels == [("a", "succeeded", 100.0), ("b", "succeeded", 110.0)], contract_message(
        "TaskRegistry.record_progress / _grid_runner._variant_progress_note", "derive.py (task_progress)", progress
    )


async def test_task_history_written_by_registry_drives_replay(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="explore"))
    row = producers.watch(session).tasks[0]
    created, running, ended = (parse_ts(x) for x in (row["created_at"], task.history[0]["ts"], task.updated_at))
    where = "derive.py (task_state_at)"
    assert derive.task_state_at(row, created - 1, parse_ts) is None, contract_message("Task.created_at", where, row)
    states = [derive.task_state_at(row, t, parse_ts)["state"] for t in (created, running, ended)]
    assert states == ["queued", "running", "succeeded"], contract_message("TaskRegistry.transition", where, states)


def test_state_json_written_by_shared_state_drives_the_summary(tmp_path: Path) -> None:
    from hyperloom.inspector.watcher import _phases_at

    session = producers.new_session(tmp_path)
    state = producers.new_state(model_name="m")
    producers.record_phase(state, "BASELINE", "start", 1000.0)
    producers.record_phase(state, "EXPLORE", "baseline measured", 2000.0)
    state.record_specialist_round({"task_id": "t1", "domain": "d", "reason": "r", "summary": "s", "proposals_total": 1})
    state.save(session)
    summary = producers.watch(session).summary_at(0)
    history = summary["phase_history"]
    assert _phases_at(summary, 1500.0)["phase"] == "BASELINE", contract_message(
        "phase_chain.make_history_row / machine_state.record_phase_transition", "watcher.py (_phases_at)", history
    )
    rounds = summary["specialist_rounds"]
    assert rounds[0]["task_id"] == "t1" and rounds[0]["completed_at"], contract_message(
        "_PhaseStateMixin.record_specialist_round", "derive.py (state_summary)", rounds
    )


def test_journal_written_by_journal_drives_best_and_baseline(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session.optimization_journal import KIND_BASELINE, OUTCOME_KEEP

    session = producers.new_session(tmp_path)
    base = {"phase": "BASELINE", "iter": 0, "kind": KIND_BASELINE, "change": "baseline", "outcome": OUTCOME_KEEP}
    keep = {"phase": "EXPLORE", "iter": 1, "kind": "param", "change": "--x 1", "outcome": OUTCOME_KEEP}
    base.update(throughput_after=100.0, ts="2026-11-25T10:00:00Z")
    keep.update(throughput_after=120.0, ts="2026-11-25T11:00:00Z")
    producers.write_journal(session, [base, keep])
    watcher = producers.watch(session, replay=True)
    where = "derive.py (throughput_series, best_throughput) and watcher.py (_baseline_by)"
    series = [(e["kind"], e["throughput"]) for e in watcher.streams["journal"].items]
    assert series == [("baseline", 100.0), ("param", 120.0)], contract_message("Journal.append_entry", where, series)
    best = watcher._best_at({}, None)
    assert (best["hot_tput"], best["action"]) == (120.0, "--x 1"), contract_message("OUTCOME_KEEP", where, best)
    at_base = parse_ts(base["ts"])
    assert not watcher._baseline_by(at_base - 1) and watcher._baseline_by(at_base), contract_message(
        "optimization_journal.KIND_BASELINE", where, series
    )


def test_journal_fields_the_inspector_reads_are_journal_entry_fields(tmp_path: Path) -> None:
    from dataclasses import fields

    from hyperloom.inference_optimizer.session.optimization_journal import JournalEntry

    session = producers.new_session(tmp_path)
    journal = producers.write_journal(
        session, [{"phase": "EXPLORE", "iter": 1, "kind": "param", "change": "-x", "outcome": "KEEP"}]
    )
    entries = [producers.RecordingDict(e.to_dict()) for e in journal.entries]
    derive.throughput_series({"entries": entries})
    unknown = sorted(entries[0].read - {f.name for f in fields(JournalEntry)})
    assert not unknown, contract_message("JournalEntry", "derive.py (throughput_series)", f"never written: {unknown}")


def _write_ledger(session: Path) -> None:
    counts = {"input_tokens": 10, "output_tokens": 5}
    producers.append_ledger_row(session, component="orchestration", call_id="c1", latency_ms=2000, **counts)
    producers.append_ledger_row(session, component="critic", model="gpt-5", reasoning_output_tokens=2, **counts)
    producers.append_ledger_row(session, component="specialist", task_id="t-sp", turn=1, model="m", **counts)
    producers.append_ledger_row(session, component="critic", status="error", error_type="Timeout")


def test_ledger_rows_written_by_append_llm_call_are_counted(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    _write_ledger(session)
    watcher = producers.watch(session)
    llm = derive.llm_summary(watcher.streams["llm_call"].items)
    assert llm["critic"]["errors"] == 1 and llm["orchestration"]["latency_ms_avg"] == 2000, contract_message(
        "LLMCallRecord.status / latency_ms", "derive.py (llm_summary)", llm
    )


def test_ledger_fields_the_inspector_reads_are_in_the_closed_schema(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.trace.llm_trace import _ROW_FIELDS

    session = producers.new_session(tmp_path)
    _write_ledger(session)
    path = session / "reports" / "trace" / "llm_calls.jsonl"
    rows = [producers.RecordingDict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    derive.llm_summary(rows)
    graph.build_graph([], [], rows, [])
    read = set().union(*(r.read for r in rows))
    unknown = sorted(read - _ROW_FIELDS)
    assert not unknown, contract_message("llm_trace._ROW_FIELDS", "derive.py, graph.py", unknown)


def test_ledger_status_vocabulary() -> None:
    from hyperloom.inference_optimizer.trace.llm_trace import LLM_STATUS_ERROR, LLM_STATUS_OK, VALID_STATUSES

    assert VALID_STATUSES == {LLM_STATUS_OK, LLM_STATUS_ERROR} == {"ok", "error"}, contract_message(
        "llm_trace.VALID_STATUSES", "derive.py (llm_summary)", sorted(VALID_STATUSES)
    )


_SRC = Path(__file__).resolve().parents[3]


def _loaded_modules(*imports: str) -> list[str]:
    """The modules a fresh interpreter has loaded after importing *imports* (none: what it loads by itself)."""
    probe = "".join(f"import {name}\n" for name in ("json", "sys", *imports)) + "print(json.dumps(sorted(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True).stdout
    return json.loads(out)


def _third_party(modules: list[str]) -> set[str]:
    """The top-level packages of *modules* that are neither the standard library's nor this source tree's."""
    ours = {p.name for p in _SRC.iterdir() if p.is_dir()}
    tops = {m.split(".")[0] for m in modules}
    return {t for t in tops if t not in sys.stdlib_module_names and t not in ours and not t.startswith("_")}


def test_the_inspector_command_line_loads_only_the_stdlib_and_itself() -> None:
    # ``hyperloom --help`` and ``python -m hyperloom.inspector --help`` import this; the rest waits for run_inspect.
    # What the interpreter loads before any import of the probe's (``__main__``, ``sitecustomize``, the modules a
    # ``.pth`` file pulls in, such as setuptools' ``_distutils_hack``) is no import of the Inspector's.
    bare = set(_loaded_modules())
    loaded = _loaded_modules("hyperloom.inspector.__main__")
    others = [
        m
        for m in loaded
        if m.split(".")[0] not in sys.stdlib_module_names
        and m not in bare
        and m != "hyperloom"
        and not m.startswith("hyperloom.inspector")
    ]
    assert not others, contract_message("(the Inspector's own promise)", "cli.py, __main__.py", f"loads: {others[:8]}")


def test_the_inspector_needs_no_package_the_hyperloom_command_does_not() -> None:
    # The Inspector runs with the Hyperloom it ships with: it may import that, never a package beyond what the
    # ``hyperloom`` command already needs.
    inspector = _third_party(_loaded_modules("hyperloom.inspector.__main__", "hyperloom.inspector.server"))
    hyperloom = _third_party(_loaded_modules("hyperloom.inference_optimizer.cli"))
    assert inspector <= hyperloom, contract_message(
        "(the Inspector's own promise)", "__main__.py", f"needs beyond the hyperloom command: {inspector - hyperloom}"
    )
