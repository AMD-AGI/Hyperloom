# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 1 (session reader): the session files the Inspector finds and reads are
the ones Hyperloom's own writers produce, and the task kinds and LLM components it maps to agents are the ones
Hyperloom uses.

Each test writes a session artifact with the real producer (see :mod:`.producers`) or compares the Inspector's
names with the producer's constants, then reads it the way the Inspector does. A failure means a producer changed
what it writes: update the Inspector reader the failure message names, then this test. A test marked
``xfail(strict=True)`` pins a known break; it starts failing ("XPASS(strict)") once the Inspector is fixed, and then
the marker comes off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inspector import graph
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.conftest import FAKE_SECRET
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


_REDACT = "redact.py"


_GRAPH = "graph.py"


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
        "paths.SESSION_FILES", "session_layout.py", session_layout.SESSION_FILES
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


# The node each task kind of ACTION_CATALOGUE maps to; a kind the catalogue gains fails until it is placed here.
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
# kind was found. Kept so older sessions map the same.
_LEGACY_KINDS = frozenset({"kernel_opt", "kernel", "framework_agent", "framework"})


# The node each LLM component of the ledger's closed vocabulary (llm_trace.VALID_COMPONENTS) belongs to when its row
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
    assert not unknown, contract_message("action_surfaces.ACTION_CATALOGUE", "graph.py", unknown)


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
