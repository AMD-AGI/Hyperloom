# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A small synthetic session directory shaped like a real v1.1.2 run.

Times are fixed unix seconds from ``T0``. The layout mirrors the MiMo smoke: a baseline, a roofline that is
still running, and two specialists that crashed but landed as ``succeeded``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.inspector.tests import producers

T0 = 1_790_000_000.0
FAKE_KEY = "sk-ant-api03-" + "x" * 40


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _task(task_id, kind, state, created, transitions, domain=None):
    params = {"domain": domain} if domain else {}
    history = [{"from": a, "to": b, "ts": iso(t)} for a, b, t in transitions]
    updated = transitions[-1][2] if transitions else created
    return (
        task_id,
        kind,
        state,
        json.dumps(params),
        task_id,
        "[]",
        0,
        json.dumps(history),
        iso(created),
        iso(updated),
    )


TASKS = [
    _task(
        "t-base", "baseline", "succeeded", T0 + 10, [("queued", "running", T0 + 12), ("running", "succeeded", T0 + 600)]
    ),
    _task(
        "t-scout",
        "specialist",
        "succeeded",
        T0 + 20,
        [("queued", "running", T0 + 21), ("running", "succeeded", T0 + 300)],
        "research_scout",
    ),
    _task(
        "t-recon",
        "specialist",
        "succeeded",
        T0 + 20,
        [("queued", "running", T0 + 22), ("running", "succeeded", T0 + 310)],
        "static_recon_specialist",
    ),
    _task(
        "t-perf",
        "specialist",
        "succeeded",
        T0 + 20,
        [("queued", "running", T0 + 23), ("running", "succeeded", T0 + 400)],
        "perf_model",
    ),
    _task("t-roof", "roofline", "running", T0 + 700, [("queued", "running", T0 + 710)]),
]

EVENTS = [
    ("coordinator", "*", "observation", {"note": "baseline started"}, T0 + 12),
    (
        "coordinator",
        "*",
        "delegated_result",
        {"task_id": "t-scout", "state": "succeeded", "result": {"runner_status": "stale"}},
        T0 + 300,
    ),
    (
        "coordinator",
        "*",
        "delegated_result",
        {"task_id": "t-recon", "state": "succeeded", "result": {"runner_status": "stale"}},
        T0 + 310,
    ),
    (
        "coordinator",
        "*",
        "delegated_result",
        {"task_id": "t-perf", "state": "succeeded", "result": {"runner_status": "ok", "proposals": 2}},
        T0 + 400,
    ),
    ("orchestration", "*", "proposal", {"change": "--max-num-seqs 64"}, T0 + 620),
    ("critic", "orchestration", "review_verdict", {"verdict": "approve"}, T0 + 640),
    (
        "coordinator",
        "*",
        "delegated_result",
        {"task_id": "t-base", "state": "succeeded", "result": {"runner_status": "ok", "throughput": 1685.16}},
        T0 + 600,
    ),
]

LLM_CALLS = [
    {
        "component": "orchestration",
        "model": "claude-opus-5-5",
        "input_tokens": 1000,
        "output_tokens": 200,
        "latency_ms": 4000,
        "status": "ok",
        "call_id": "c1",
        "ts": iso(T0 + 610),
    },
    {
        "component": "critic",
        "model": "claude-opus-5-5",
        "input_tokens": 800,
        "output_tokens": 100,
        "latency_ms": 3000,
        "status": "ok",
        "call_id": "c2",
        "ts": iso(T0 + 635),
    },
    {
        "component": "orchestration",
        "model": "claude-opus-5-5",
        "input_tokens": 500,
        "output_tokens": 50,
        "latency_ms": 2000,
        "status": "error",
        "call_id": "c3",
        "ts": iso(T0 + 720),
    },
]


def _phase(frm, to, t, reason="advance"):
    return {"from_phase": frm, "to_phase": to, "reason": reason, "ts": iso(t), "ts_unix": t, "cycle": 0}


STATE = {
    "session_id": "20260925T105719Z-test",
    "model_name": "XiaomiMiMo/MiMo-V2.6-Pro-RL",
    "phase": "SYSTEM",
    "tick": 42,
    "start_ts": iso(T0),
    "elapsed_charged_sec": 1800,
    "max_minutes": 120,
    "baseline_tput": 1685.16,
    "crash_count": 0,
    "current_best": {"action": "baseline", "hot_tput": 1685.16},
    "phase_history": [
        _phase("", "PRELUDE", T0 + 5),
        _phase("PRELUDE", "PRELUDE", T0 + 200, "marker"),
        _phase("PRELUDE", "SYSTEM", T0 + 650),
    ],
    "specialist_rounds": [
        {
            "task_id": "t-scout",
            "domain": "research_scout",
            "reason": "subprocess_exit_code:1",
            "proposals_total": 0,
            "completed_at": iso(T0 + 300),
        },
        {
            "task_id": "t-recon",
            "domain": "static_recon",
            "reason": "subprocess_exit_code:1",
            "proposals_total": 0,
            "completed_at": iso(T0 + 310),
        },
        {
            "task_id": "t-perf",
            "domain": "perf_model",
            "reason": "done",
            "proposals_total": 2,
            "completed_at": iso(T0 + 400),
        },
    ],
    "agent_last_active": {"orchestration": T0 + 720, "critic": T0 + 640},
    "api_key": FAKE_KEY,
}


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    root = tmp_path / "20260925T105719Z-test"
    db_path = root / "storage" / "coordinator.db"
    db_path.parent.mkdir(parents=True)
    producers.create_coordinator_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, "
        "lease_ttl_sec, history, created_at, updated_at) VALUES "
        "(?,?,?,?,?,?,?,?,?,?)",
        TASKS,
    )
    conn.executemany(
        "INSERT INTO events (msg_id, from_agent, to_agent, topic, payload, ts) VALUES (?,?,?,?,?,?)",
        [(f"m{i}", a, b, topic, json.dumps(p), iso(t)) for i, (a, b, topic, p, t) in enumerate(EVENTS)],
    )
    conn.commit()
    conn.close()
    _write(root / "state.json", STATE)
    _write(
        root / "manifest.json",
        {
            "objective": "throughput",
            "gfx_arch": "gfx950",
            "host": "pod-a",
            "workload": {"isl": 8192, "osl": 1024, "conc": 32},
            "max_minutes": 120,
        },
    )
    _write(
        root / "runtime" / "pod_history.jsonl",
        json.dumps({"acquired_at": iso(T0), "hostname": "pod-a", "pid": 99999999}) + "\n",
    )
    trace = root / "reports" / "trace"
    _write(trace / "llm_calls.jsonl", "".join(json.dumps(c) + "\n" for c in LLM_CALLS) + '{"torn": ')
    _write(
        trace / "conversations.jsonl",
        json.dumps({"call_id": "c1", "prompt": f"use key {FAKE_KEY}", "response": "ok"})
        + "\n"
        + json.dumps({"call_id": None, "task_id": "spec1", "turn": 1, "prompt": "p", "response": "spec"})
        + "\n",
    )
    _write(
        root / "reports" / "optimization_journal.json",
        {
            "entries": [
                {
                    "ts": iso(T0 + 600),
                    "phase": "PRELUDE",
                    "kind": "baseline",
                    "change": "baseline",
                    "outcome": "keep",
                    "throughput_after": 1685.16,
                    "task_id": "t-base",
                }
            ]
        },
    )
    _write(
        root / "critic-workdir" / "turn-0001" / "review.json",
        {"review_verdicts": [{"target_proposal_msg_id": "m4", "verdict": "approve", "reasoning": "fine"}]},
    )
    runs = root / "runs"
    _write(runs / "baseline" / "t-base" / "server.log", "INFO started\nINFO throughput 1685.16\n")
    _write(runs / "baseline" / "t-base" / "baseline_config.with_envs.yaml", "tp: 4\nenv:\n  A: '1'\n")
    _write(runs / "roofline" / "t-roof" / "server.log", "INFO profiling\n")
    _write(runs / "roofline" / "t-roof" / "baseline_config.with_envs.yaml", "tp: 4\nenv:\n  A: '2'\n")
    _write(runs / "roofline" / "t-roof" / "rank0.pt.trace.json.gz", "\x1f\x8b fake")
    _write(runs / "specialist" / "t-scout" / "process.log", f"Error: model refused\nANTHROPIC_API_KEY={FAKE_KEY}\n")
    _write(root / ".env", f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
    _set_mtimes(root)
    return root


def _set_mtimes(root: Path) -> None:
    """Session-time mtimes; the default is the end of the fixture run."""
    special = {
        "runs/baseline": T0 + 600,
        "runs/specialist": T0 + 300,
        "critic-workdir": T0 + 640,
    }
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            t = next((v for k, v in special.items() if rel.startswith(k)), T0 + 800)
            os.utime(path, (t, t))
