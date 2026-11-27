# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The harness graph: which agents exist, how they talk, and what they produced.

Node ids are stable strings so the page can keep a node's position across
updates: ``coordinator``, the LLM roles (``orchestration``, ``critic``),
``exec:<task kind>`` for programmatic executors, ``spec:<domain>`` for
specialists, ``bus`` for broadcast traffic and ``session`` for session-wide
reports.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath

COORDINATOR = "coordinator"
CRITIC = "critic"
BUS = "bus"
SESSION = "session"
KERNEL_AGENT = "agent:kernel"
FRAMEWORK_AGENT = "agent:framework"
LLM_ROLES = ("orchestration", CRITIC)
_SPECIALIST = "specialist"

# Layer index per node group, used by the fixed layered layout.
LAYER = {"llm": 0, "core": 1, "exec": 2, "spec": 2, "agent": 2, "session": 3}

# The ``-<n>`` that ``session_paths.unique_runs_dir`` appends to the workspace of a retried task.
_RETRY_SUFFIX = re.compile(r"-[0-9]+$")

_ROOT_PRODUCERS = {
    "critic-workdir": CRITIC,
    "critic-session-memory": CRITIC,
    "target_analysis": "exec:target_analysis",
    "kernel-agent": KERNEL_AGENT,
    "kernel-agent-workspace": KERNEL_AGENT,
    "research_hints.json": "spec:research_scout",
    "research_hints.md": "spec:research_scout",
    "storage": COORDINATOR,
    "state.json": COORDINATOR,
    "current_setting.sh": COORDINATOR,
    "runtime": COORDINATOR,
}

# The agent node a task kind is drawn on, so that a kernel agent's task, its LLM calls and its ``kernel-agent*/``
# directories share one node. ``kernel_agent`` is the kind Hyperloom creates (the action catalogue's);
# ``test_kernel_agent_tasks_share_the_kernel_agent_node`` and ``test_every_catalogue_task_kind_is_placed`` fail when
# the catalogue renames it. The legacy kinds are no longer created; they stay so that older sessions draw the same.
_LEGACY_AGENT_TASK_KINDS = {
    "kernel_opt": KERNEL_AGENT,  # retired with the kernel harness
    "kernel": KERNEL_AGENT,  # no producer of this kind was ever found
    "framework_agent": FRAMEWORK_AGENT,  # retired with its action
    "framework": FRAMEWORK_AGENT,  # no producer of this kind was ever found
}
_AGENT_OF_TASK_KIND = {"kernel_agent": KERNEL_AGENT, **_LEGACY_AGENT_TASK_KINDS}


def specialist_domain(task: dict) -> str:
    params = task.get("params") or {}
    # FRAMEWORK_AGENT specialists have no ``domain``: ``scope: domain`` ones name it in ``tags``,
    # freeform ones only have their ``scope``.
    tags = params.get("tags") if isinstance(params.get("tags"), list) else []
    tagged = next((str(t) for t in tags if str(t).endswith("_specialist")), None)
    domain = str(params.get("domain") or tagged or params.get("scope") or _SPECIALIST)
    return domain[: -len("_specialist")] if domain.endswith("_specialist") else domain


def task_node(task: dict) -> str:
    kind = str(task.get("kind") or "task")
    if kind == _SPECIALIST:
        return f"spec:{specialist_domain(task)}"
    if kind in _AGENT_OF_TASK_KIND:
        return _AGENT_OF_TASK_KIND[kind]
    return f"exec:{kind}"


def agent_node(name: str) -> str:
    """Bus agent name -> node id."""
    if name in ("*", ""):
        return BUS
    if name in (COORDINATOR, *LLM_ROLES):
        return name
    if name.startswith("framework"):
        return FRAMEWORK_AGENT
    if name.startswith("kernel"):
        return KERNEL_AGENT
    return name


def _run_task_id(seg: str, task_nodes: dict[str, str]) -> str:
    """The task id of a ``runs/<kind>/<seg>`` directory: ``<id>``, ``<id>.<suffix>`` or a retry ``<id>-<n>``.

    Hyperloom names the workspace of a retried task ``<id>-2``, ``<id>-3``, ... (``unique_runs_dir``). A name that
    is a known task is kept as it is, so that a task whose own id ends in ``-<n>`` is not taken for the retry of
    another; only otherwise is the suffix dropped. A name that matches no task is left alone: its suffix may be part
    of it.
    """
    task_id = seg.split(".", 1)[0]
    if task_id in task_nodes:
        return task_id
    retried = _RETRY_SUFFIX.sub("", task_id)
    return retried if retried in task_nodes else task_id


def producer_for(rel_path: str, task_nodes: dict[str, str]) -> tuple[str, str | None]:
    """Map a session-relative path to (producer node, task id or None)."""
    match PurePosixPath(rel_path).parts:
        case ("runs", kind, seg, *_):
            task_id = _run_task_id(seg, task_nodes)
            if task_id in task_nodes:
                return task_nodes[task_id], task_id
            return (f"spec:{_SPECIALIST}" if kind == _SPECIALIST else f"exec:{kind}"), task_id
        case ("agents", name, _, *_):
            # Only a directory ``agents/<name>/`` is an agent; a file directly below ``agents/`` (the runner's
            # resource readings) belongs to no agent and falls through to the session's own reports.
            return agent_node(name), None
        case (head, *_) if head in _ROOT_PRODUCERS:
            return _ROOT_PRODUCERS[head], None
    return SESSION, None


def _node(node_id: str) -> dict:
    if node_id in LLM_ROLES:
        group, label = "llm", node_id
    elif node_id in (COORDINATOR, BUS):
        group, label = "core", ("broadcast" if node_id == BUS else node_id)
    elif node_id.startswith("exec:"):
        group, label = "exec", node_id[5:]
    elif node_id.startswith("spec:"):
        group, label = "spec", node_id[5:]
    elif node_id.startswith("agent:"):
        group, label = "agent", f"{node_id[6:]} agent"
    elif node_id == SESSION:
        group, label = "session", "session reports"
    else:
        group, label = "agent", node_id
    return {
        "id": node_id,
        "label": label,
        "group": group,
        "layer": LAYER[group],
        "tasks": Counter(),
        "llm_calls": 0,
        "llm_tokens": 0,
        "artifacts": 0,
        "trees": {},
        "health": "idle",
    }


_HEALTH_RANK = {"crashed": 5, "failed": 4, "stalled": 3, "running": 2, "queued": 1, "ok": 0, "cancelled": 0, "idle": -1}


class _Graph:
    """The nodes and edges counted for one view; each ``add_*`` counts one kind of record into them."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: Counter = Counter()
        for base in (COORDINATOR, *LLM_ROLES):
            self.node(base)

    def node(self, node_id: str) -> dict:
        if node_id not in self.nodes:
            self.nodes[node_id] = _node(node_id)
        return self.nodes[node_id]

    def add_tasks(self, tasks: list[dict]) -> None:
        for t in tasks:
            n = self.node(t["node"])
            health = t.get("health") or "ok"
            n["tasks"][health] += 1
            if _HEALTH_RANK.get(health, 0) > _HEALTH_RANK.get(n["health"], 0):
                n["health"] = health
            elif n["health"] == "idle":
                n["health"] = "ok"
            self.edges[(COORDINATOR, n["id"], "dispatch")] += 1

    def add_events(self, events: list[dict]) -> None:
        for e in events:
            src, dst = agent_node(str(e.get("from_agent"))), agent_node(str(e.get("to_agent")))
            self.node(src), self.node(dst)
            self.edges[(src, dst, str(e.get("topic")))] += 1

    def add_llm_calls(self, llm_calls: list[dict], tasks: list[dict]) -> None:
        task_nodes = {t["task_id"]: t["node"] for t in tasks if t.get("task_id")}
        for c in llm_calls:
            # Specialist calls name the task, not the domain; count them on that task's node.
            node_id = task_nodes.get(c.get("task_id")) or agent_node(str(c.get("component") or c.get("role") or "llm"))
            n = self.node(node_id)
            n["llm_calls"] += 1
            n["llm_tokens"] += (
                (c.get("input_tokens") or 0)
                + (c.get("output_tokens") or 0)
                + (c.get("cache_read_input_tokens") or 0)
                + (c.get("cache_creation_input_tokens") or 0)
            )

    def add_artifacts(self, artifacts: list[dict]) -> None:
        for a in artifacts:
            n = self.node(a["producer"])
            # A large subtree (a source-tree copy) counts as one tree, not thousands of files.
            if a.get("bulk"):
                n["trees"][a["bulk"]] = n["trees"].get(a["bulk"], 0) + 1
            else:
                n["artifacts"] += 1

    def view(self) -> dict:
        out_nodes = [{**n, "tasks": dict(n["tasks"])} for n in self.nodes.values()]
        out_edges = [
            {"source": s, "target": d, "label": lbl, "count": c} for (s, d, lbl), c in sorted(self.edges.items())
        ]
        return {"nodes": sorted(out_nodes, key=lambda n: (n["layer"], n["id"])), "edges": out_edges}


def build_graph(tasks: list[dict], events: list[dict], llm_calls: list[dict], artifacts: list[dict]) -> dict:
    """Nodes and edges for the current view. ``tasks`` carry ``node`` and ``health`` (see watcher)."""
    graph = _Graph()
    graph.add_tasks(tasks)
    graph.add_events(events)
    graph.add_llm_calls(llm_calls, tasks)
    graph.add_artifacts(artifacts)
    return graph.view()
