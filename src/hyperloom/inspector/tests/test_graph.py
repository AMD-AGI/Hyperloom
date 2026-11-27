# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector.graph import agent_node, build_graph, producer_for, task_node

TASK_NODES = {"t-1": "spec:research_scout", "t-2": "exec:baseline"}


@pytest.mark.parametrize(
    ("task", "node"),
    [
        ({"kind": "specialist", "params": {"domain": "research_scout_specialist"}}, "spec:research_scout"),
        ({"kind": "specialist", "params": {"scope": "freeform"}}, "spec:freeform"),
        ({"kind": "specialist", "params": {"scope": "domain", "tags": ["serving_specialist"]}}, "spec:serving"),
        ({"kind": "specialist"}, "spec:specialist"),
        ({"kind": "kernel_agent"}, "agent:kernel"),
        # Kinds of retired actions: older sessions still hold their tasks.
        ({"kind": "framework_agent"}, "agent:framework"),
        ({"kind": "framework"}, "agent:framework"),
        ({"kind": "kernel_opt"}, "agent:kernel"),
        ({"kind": "kernel"}, "agent:kernel"),
        ({"kind": "roofline"}, "exec:roofline"),
    ],
)
def test_task_node(task, node):
    assert task_node(task) == node


@pytest.mark.parametrize(
    ("name", "node"),
    [
        ("*", "bus"),
        ("", "bus"),
        ("critic", "critic"),
        ("framework_worker", "agent:framework"),
        ("kernel", "agent:kernel"),
    ],
)
def test_agent_node(name, node):
    assert agent_node(name) == node


@pytest.mark.parametrize(
    ("path", "producer"),
    [
        ("runs/specialist/t-1/process.log", ("spec:research_scout", "t-1")),
        ("runs/specialist/t-9/prompt.md", ("spec:specialist", "t-9")),
        ("runs/baseline/t-2.attempt1/server.log", ("exec:baseline", "t-2")),
        ("runs/roofline/t-3/x.yaml", ("exec:roofline", "t-3")),
        ("agents/critic/system_prompt.md", ("critic", None)),
        ("agents/kernel_agent/notes/plan.md", ("agent:kernel", None)),
        ("agents/resources.jsonl", ("session", None)),
        ("agents/notes.md", ("session", None)),
        ("critic-workdir/3/out.json", ("critic", None)),
        ("kernel-agent/log.txt", ("agent:kernel", None)),
        ("state.json", ("coordinator", None)),
        ("reports/summary.md", ("session", None)),
    ],
)
def test_producer_for(path, producer):
    assert producer_for(path, TASK_NODES) == producer


# ``t`` and ``t-2`` are both tasks, so the directory ``t-2`` is the second one's own, not the first one's retry.
RETRY_TASK_NODES = {"t": "exec:a", "t-2": "exec:b"}


@pytest.mark.parametrize(
    ("path", "producer"),
    [
        ("runs/integrate/t/x.log", ("exec:a", "t")),
        ("runs/integrate/t-2/x.log", ("exec:b", "t-2")),
        # ``t-3`` is no task, so it is a retry of ``t``.
        ("runs/integrate/t-3/x.log", ("exec:a", "t")),
        # One suffix is dropped, not all: the retry of ``t-2`` is ``t-2-2``.
        ("runs/integrate/t-2-2/x.log", ("exec:b", "t-2")),
        ("runs/integrate/t-2-3.attempt1/x.log", ("exec:b", "t-2")),
        # A name that is no task, and no retry of one, keeps its own id: its suffix may belong to the name.
        ("runs/integrate/revalidate_geak-2/x.log", ("exec:integrate", "revalidate_geak-2")),
        ("runs/integrate/t-x/x.log", ("exec:integrate", "t-x")),
    ],
)
def test_producer_for_run_dir_of_a_retried_task(path, producer):
    assert producer_for(path, RETRY_TASK_NODES) == producer


def test_build_graph_edges_counts_and_health():
    tasks = [
        {"task_id": "t1", "node": "spec:research_scout", "health": "crashed"},
        {"node": "exec:baseline", "health": "ok"},
        {"node": "exec:baseline", "health": "running"},
    ]
    events = [
        {"from_agent": "orchestration", "to_agent": "critic", "topic": "proposal"},
        {"from_agent": "critic", "to_agent": "orchestration", "topic": "review_verdict"},
        {"from_agent": "critic", "to_agent": "orchestration", "topic": "review_verdict"},
        {"from_agent": "coordinator", "to_agent": "*", "topic": "observation"},
    ]
    calls = [
        {"component": "critic", "input_tokens": 3, "output_tokens": 1},
        {"component": "specialist", "task_id": "t1"},
        {"component": "specialist", "task_id": "gone"},
    ]
    artifacts = [{"producer": "exec:baseline"}, {"producer": "session"}]
    artifacts += [{"producer": "exec:baseline", "bulk": "runs/baseline/t1/tree"}] * 3
    g = build_graph(tasks, events, calls, artifacts)
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = {(e["source"], e["target"], e["label"]): e["count"] for e in g["edges"]}
    assert nodes["spec:research_scout"]["health"] == "crashed"
    assert nodes["exec:baseline"]["health"] == "running" and nodes["exec:baseline"]["tasks"] == {"ok": 1, "running": 1}
    assert nodes["exec:baseline"]["artifacts"] == 1 and nodes["session"]["artifacts"] == 1
    assert nodes["exec:baseline"]["trees"] == {"runs/baseline/t1/tree": 3}
    assert nodes["critic"]["llm_calls"] == 1 and nodes["critic"]["llm_tokens"] == 4
    assert nodes["spec:research_scout"]["llm_calls"] == 1 and nodes["specialist"]["llm_calls"] == 1
    assert nodes["coordinator"]["health"] == "idle"
    assert edges[("coordinator", "exec:baseline", "dispatch")] == 2
    assert edges[("critic", "orchestration", "review_verdict")] == 2
    assert edges[("coordinator", "bus", "observation")] == 1
    layers = [n["layer"] for n in g["nodes"]]
    assert layers == sorted(layers)
