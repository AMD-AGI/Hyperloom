# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Node ids for the harness's agents: which agents exist and what they produced.

Node ids are stable strings: ``coordinator``, the LLM roles (``orchestration``, ``critic``),
``exec:<task kind>`` for programmatic executors, ``spec:<domain>`` for
specialists, ``bus`` for broadcast traffic and ``session`` for session-wide
reports.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

COORDINATOR = "coordinator"
CRITIC = "critic"
BUS = "bus"
SESSION = "session"
KERNEL_AGENT = "agent:kernel"
FRAMEWORK_AGENT = "agent:framework"
LLM_ROLES = ("orchestration", CRITIC)
_SPECIALIST = "specialist"

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

# The agent node a task kind belongs to, so that a kernel agent's task, its LLM calls and its ``kernel-agent*/``
# directories share one node. ``kernel_agent`` is the kind Hyperloom creates (the action catalogue's);
# ``test_kernel_agent_tasks_share_the_kernel_agent_node`` and ``test_every_catalogue_task_kind_is_placed`` fail when
# the catalogue renames it. The legacy kinds are no longer created; they stay so that older sessions map the same.
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
