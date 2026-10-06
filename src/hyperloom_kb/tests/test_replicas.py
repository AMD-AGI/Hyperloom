# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Any number of services serve one KB from one database and one home, and every one of them answers alike."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hyperloom_kb import (
    ExperienceHTTPService,
    HTTPServiceConfig,
    ImmutableExperienceConflict,
    LLMQueryPlanner,
    PlannerConfiguration,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
)
from hyperloom_kb.database import Database
from hyperloom_kb.http_service import DATABASE_URL_ENV
from hyperloom_kb.tests.conftest import fresh_database
from hyperloom_kb.tests.test_http_service import (
    DECISION,
    FakePlannerBackend,
    _declaration,
    _experience,
    _read_context,
)

# Spawned services run their own embedded database under ``tmp_path``.
pytestmark = pytest.mark.usefixtures("reachable_tmp_path")

TOKEN = "replica-token"
SCHEMA = _declaration()


def _replica(home: Path, database: Database) -> ExperienceHTTPService:
    planner = LLMQueryPlanner(FakePlannerBackend(), PlannerConfiguration.create("test-planner"))
    return ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, planner, database=database)


def _read_ids(replica: ExperienceHTTPService) -> set[str]:
    read = replica.read(decision=DECISION, context=_read_context(), limit=100)
    return {str(reference["id"]) for reference in read["rendered_refs"]}


def test_replicas_sharing_a_database_write_in_one_order_and_read_one_state(tmp_path: Path) -> None:
    database = fresh_database()
    other_pool = Database(database.pool.conninfo)
    first, second = _replica(tmp_path / "home", database), _replica(tmp_path / "home", other_pool)
    try:
        experiences = [_experience(SCHEMA, seq=seq, knob=f"knob_{seq}") for seq in range(24)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda item: (first, second)[item[0] % 2].write(item[1]), enumerate(experiences)))
        # The same new Experience reaching both replicas at once is stored once.
        racing = _experience(SCHEMA, seq=99, knob="knob_99")
        with ThreadPoolExecutor(max_workers=2) as pool:
            raced = sorted(
                str(result["status"]) for result in pool.map(lambda replica: replica.write(racing), (first, second))
            )
        conflicting = _experience(SCHEMA, seq=99, knob="knob_99", outcome_value=1.0)
        with pytest.raises(ImmutableExperienceConflict):
            second.write(conflicting)

        pages = [replica.records_after(0, 100) for replica in (first, second)]
        sequences = [sequence for sequence, _ in pages[0][0]]
        before = (_read_ids(first), _read_ids(second))
        second.exclude(experiences[0].id, "measured on a noisy node")
        after = (_read_ids(first), _read_ids(second))
        health = (first.health(), second.health())
    finally:
        other_pool.close()

    assert {result["status"] for result in results} == {"created"}
    assert raced == ["created", "unchanged"]
    assert [experience.id for _, experience in pages[0][0]] == [experience.id for _, experience in pages[1][0]]
    assert sequences == list(range(1, 26))
    assert before[0] == before[1] and experiences[0].id in before[0]
    assert after[0] == after[1] and experiences[0].id not in after[0]
    assert health[0]["schemas"] == health[1]["schemas"] == {SCHEMA.schema_ref: 24}
    assert health[0]["kb_id"] == health[1]["kb_id"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_service_processes_sharing_a_database_and_a_home_serve_one_kb(tmp_path: Path, database_url: str) -> None:
    home = tmp_path / "home"
    env = {
        **{key: value for key, value in os.environ.items() if not key.startswith(("ANTHROPIC_", "HYPERLOOM_"))},
        DATABASE_URL_ENV: database_url,
        "HYPERLOOM_KB_TOKEN": TOKEN,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    ports = [_free_port(), _free_port()]
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "hyperloom_kb", "--home", str(home), "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=(tmp_path / f"service-{port}.log").open("wb"),
        )
        for port in ports
    ]
    try:
        clients = [RemoteClient(RemoteConfig(f"http://127.0.0.1:{port}", TOKEN)) for port in ports]
        deadline = time.monotonic() + 60
        for client in clients:
            while True:
                try:
                    client.health()
                    break
                except RemoteClientError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
        experiences = [_experience(SCHEMA, seq=seq, knob=f"knob_{seq}") for seq in range(12)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda item: clients[item[0] % 2].write(item[1], declaration=SCHEMA), enumerate(experiences)))
        exports = [client.export_page(schema_ref=SCHEMA.schema_ref) for client in clients]
        identities = {str(client.health()["kb_id"]) for client in clients}
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=30)

    assert len(identities) == 1
    assert [item["experience"]["id"] for item in exports[0].items] == [
        item["experience"]["id"] for item in exports[1].items
    ]
    assert sorted(item["experience"]["id"] for item in exports[0].items) == sorted(e.id for e in experiences)
    assert exports[0].head == exports[1].head == 12
    assert exports[0].state == exports[1].state
