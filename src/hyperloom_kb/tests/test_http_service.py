# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import (
    Alternative,
    Change,
    ConstraintResult,
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    HTTPServiceConfig,
    HTTPServiceError,
    LLMQueryPlanner,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    PlannerConfiguration,
    Provenance,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    RemoteExperienceKB,
    create_http_server,
    derive_experience_id,
    experience_kb_from_env,
    load_declaration,
)
from hyperloom_kb.config import PACKAGED_DECLARATION

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)
TOKEN = "service-secret"
DECISION = "Select the next framework optimization to benchmark."


def _declaration() -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(
            FieldDeclaration("model", "Model."),
            FieldDeclaration("gpu", "GPU."),
        ),
        baseline_identity=(FieldDeclaration("config", "Baseline."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(
            ObjectiveDeclaration(
                "throughput@v1",
                ObjectiveDirection.HIGHER_IS_BETTER,
                "Throughput.",
            ),
        ),
        decisions=("keep", "revert"),
    )


def _experience(
    schema: ExperienceDeclaration,
    *,
    run_id: str = "run-1",
    seq: int = 0,
    knob: str = "page_size",
    decision: str = "keep",
    outcome_value: float = 110.0,
) -> Experience:
    return Experience(
        id=derive_experience_id("service-test", run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=NOW,
        completed_at=NOW + timedelta(minutes=seq),
        identity={"model": "qwen3", "gpu": "mi325x"},
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("service-test", "1"),
        schema_ref=schema.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=f"Prior evidence supports testing {knob}.",
        change=Change({"knob": knob}, f"Test {knob}.", kind="config"),
        outcome=Outcome(decision, outcome_value),
        reflection=f"The {knob} result was {decision}.",
    )


class FakePlannerBackend:
    model = "test-planner"

    def complete(self, *, user_payload: str, **_kwargs: Any) -> str:
        context = json.loads(user_payload)["request"]["context"]
        return json.dumps(
            {
                "signals": [
                    {
                        "source_path": "context.identity.model",
                        "field": "identity.model",
                        "value": context["identity"]["model"],
                        "weight": 1.0,
                    },
                    {
                        "source_path": "context.observations.bottleneck",
                        "text": context["observations"]["bottleneck"],
                        "weight": 0.9,
                    },
                ]
            }
        )


def _read_context() -> dict[str, Any]:
    return {
        "identity": {"model": "qwen3", "gpu": "mi325x"},
        "workload": {"conc": 64, "isl": 1024, "osl": 128},
        "objective": {"id": "throughput@v1", "direction": "higher_is_better"},
        "observations": {"bottleneck": "Tune page size for qwen3."},
    }


def _app(
    home: Path,
    schema: ExperienceDeclaration | None = None,
) -> ExperienceHTTPService:
    return ExperienceHTTPService(
        HTTPServiceConfig(home, TOKEN),
        schema or _declaration(),
        LLMQueryPlanner(FakePlannerBackend(), PlannerConfiguration.create("test-planner")),
    )


class RunningServer:
    def __init__(self, app: ExperienceHTTPService) -> None:
        self.server = create_http_server(app, "127.0.0.1", 0)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.server.server_address[:2]
        return f"http://{host!s}:{port}"

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _http(
    url: str,
    method: str,
    path: str,
    body: object = None,
    *,
    token: str | None = TOKEN,
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{url}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.loads(exc.read())


def _client(url: str, tmp_path: Path) -> RemoteClient:
    return RemoteClient(RemoteConfig(url, TOKEN, spool_root=tmp_path / "spool"))


def _record_json(prompt_block: str, experience_id: str) -> dict[str, Any]:
    block = prompt_block.split(f"Experience {experience_id}\n", 1)[1]
    record = block.split("\nRecord:\n", 1)[1]
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(record)
    assert isinstance(value, dict)
    return value


def test_write_is_immediately_readable_immutable_and_rendered_losslessly(
    tmp_path: Path,
) -> None:
    schema = _declaration()
    app = _app(tmp_path / "service", schema)
    experience = replace(
        _experience(schema),
        reasoning="Decode stalls on page-table walks. " * 2_000,
        preconditions=("measured_baseline_tput=100.0",),
        alternatives=(Alternative("chunk", "Chunk sizes were already swept."),),
        change=Change(
            {"knob": "page_size"},
            "Test page_size.",
            kind="config",
            content="--page-size 32\n--max-num-seqs 256",
            resource_refs=("artifact:config.yaml",),
        ),
        outcome=Outcome(
            "keep",
            110.0,
            constraints=(ConstraintResult("accuracy", True, 0.991),),
        ),
    )

    with RunningServer(app) as url:
        client = _client(url, tmp_path)
        created = client.publish(experience)
        replayed = client.publish(experience)
        with pytest.raises(RemoteClientError, match="HTTP 409") as conflict:
            client.publish(replace(experience, reflection="Different content."))
        read = client.read(DECISION, _read_context())
        page = client.list_experiences()
        health = client.health()

    assert (created.status, replayed.status) == ("created", "unchanged")
    assert created.content_hash == replayed.content_hash
    assert conflict.value.retryable is False
    assert not (tmp_path / "spool").exists()
    assert read.status == "completed"
    assert read.read_id.startswith("read-")
    assert [item.id for item in read.rendered_refs] == [experience.id]
    assert _record_json(read.prompt_block, experience.id) == experience.to_dict()
    assert "… [truncated]" not in read.prompt_block
    assert read.experiences[0]["experience_id"] == experience.id
    assert read.experiences[0]["decision"] == "keep"
    assert read.experiences[0]["source_run_id"] == "run-1"
    assert read.experiences[0]["score"] > 0
    assert read.experiences[0]["why_matched"]
    assert [item["experience_id"] for item in page.items] == [experience.id]
    assert "score" not in page.items[0]
    assert health == {
        "status": "ok",
        "schema_ref": schema.schema_ref,
        "experience_count": 1,
        "schemas": {schema.schema_ref: 1},
        "pid": os.getpid(),
        "config_digest": "",
    }


def test_read_defaults_to_ten_mixed_and_filters_by_outcome(tmp_path: Path) -> None:
    schema = _declaration()
    app = _app(tmp_path / "service", schema)
    experiences = tuple(
        _experience(
            schema,
            seq=seq,
            knob=f"knob_{seq}",
            decision="keep" if seq % 2 == 0 else "revert",
            outcome_value=110.0 if seq % 2 == 0 else 95.0,
        )
        for seq in range(14)
    )
    context = _read_context()

    with RunningServer(app) as url:
        client = _client(url, tmp_path)
        for experience in experiences:
            client.publish(experience)
        _, mixed = _http(url, "POST", "/v1/read", {"decision": DECISION, "context": context})
        _, kept = _http(
            url,
            "POST",
            "/v1/read",
            {"decision": DECISION, "context": context, "outcome": "keep"},
        )
        reverted = client.read(DECISION, context, outcome="revert", limit=3)
        invalid = client.read(DECISION, context, outcome="maybe")
        too_many = _http(
            url,
            "POST",
            "/v1/read",
            {"decision": DECISION, "context": context, "limit": 101},
        )

    decisions = {item.id: item.outcome.decision for item in experiences if item.outcome}
    assert (mixed["outcome"], mixed["limit"], mixed["eligible_count"]) == ("mixed", 10, 14)
    assert mixed["rendered_count"] == 10
    assert {decisions[item["id"]] for item in mixed["rendered_refs"]} == {"keep", "revert"}
    assert (kept["outcome"], kept["eligible_count"], kept["rendered_count"]) == ("keep", 7, 7)
    assert {decisions[item["id"]] for item in kept["rendered_refs"]} == {"keep"}
    assert len(reverted.rendered_refs) == 3
    assert {decisions[item.id] for item in reverted.rendered_refs} == {"revert"}
    assert all(item["decision"] == "revert" for item in reverted.experiences)
    assert invalid.status == "unavailable"
    assert "outcome must be one of: keep, revert, mixed" in invalid.warnings[0]
    assert too_many[0] == 400


def test_outcome_filter_keeps_full_repeat_group_annotations(tmp_path: Path) -> None:
    schema = _declaration()
    app = _app(tmp_path / "service", schema)
    kept = _experience(schema, run_id="run-keep")
    reverted = _experience(schema, run_id="run-revert", decision="revert", outcome_value=90.0)

    with RunningServer(app) as url:
        client = _client(url, tmp_path)
        client.publish(kept)
        client.publish(reverted)
        read = client.read(DECISION, _read_context(), outcome="keep")

    assert [item.id for item in read.rendered_refs] == [kept.id]
    annotations = json.loads(read.prompt_block.split("Repeat Group Annotations:\n", 1)[1].split("\nRecord:\n", 1)[0])
    assert annotations["member_count"] == 2
    assert annotations["decision_counts"] == {"keep": 1, "revert": 1}


def test_empty_service_reads_without_planning(tmp_path: Path) -> None:
    app = _app(tmp_path / "service")

    with RunningServer(app) as url:
        status, read = _http(url, "POST", "/v1/read", {"decision": DECISION, "context": {}})

    assert status == 200
    assert (read["status"], read["prompt_block"], read["eligible_count"]) == ("completed", "", 0)


def test_list_pages_in_write_order_and_rebuilds_index_after_restart(tmp_path: Path) -> None:
    schema = _declaration()
    home = tmp_path / "service"
    experiences = tuple(_experience(schema, seq=seq, knob=f"knob_{seq}") for seq in (3, 0, 4, 1, 2))

    def all_ids(url: str) -> tuple[list[str], list[int]]:
        client = _client(url, tmp_path)
        ids: list[str] = []
        sequences: list[int] = []
        cursor = 0
        while True:
            page = client.list_experiences(after=cursor, limit=2)
            ids.extend(str(item["experience_id"]) for item in page.items)
            sequences.extend(int(item["sequence"]) for item in page.items)
            if not page.has_more:
                return ids, sequences
            cursor = page.next_cursor

    with RunningServer(_app(home, schema)) as url:
        for experience in experiences:
            _client(url, tmp_path).publish(experience)
        written, sequences = all_ids(url)
        bad_limit = _http(url, "GET", "/v1/list?limit=501")
        bad_cursor = _http(url, "GET", "/v1/list?after=-1")

    with RunningServer(_app(home, schema)) as url:
        reopened, _ = all_ids(url)

    for path in home.glob("kb.sqlite3*"):
        path.unlink()
    with RunningServer(_app(home, schema)) as url:
        rebuilt, _ = all_ids(url)
        health = _client(url, tmp_path).health()

    assert written == [item.id for item in experiences]
    assert sequences == sorted(sequences)
    assert reopened == written
    assert rebuilt == [item.id for item in sorted(experiences, key=lambda item: item.seq)]
    assert health["experience_count"] == 5
    assert (bad_limit[0], bad_cursor[0]) == (400, 400)


def test_invalid_requests_are_rejected(tmp_path: Path) -> None:
    schema = _declaration()
    other = replace(schema, decisions=("keep", "revert", "failed"))
    experience = _experience(schema)
    incomplete = replace(
        experience,
        status=ExperienceStatus.IN_PROGRESS,
        completed_at=None,
        outcome=None,
        reflection="",
    )

    with RunningServer(_app(tmp_path / "service", schema)) as url:
        path = f"/v1/experiences/{experience.id}"
        mismatched = _http(url, "PUT", "/v1/experiences/exp-other", {"experience": experience.to_dict()})
        unfinished = _http(url, "PUT", path, {"experience": incomplete.to_dict()})
        foreign = _experience(other)
        wrong_schema = _http(
            url,
            "PUT",
            f"/v1/experiences/{foreign.id}",
            {"experience": foreign.to_dict()},
        )
        unknown_write_field = _http(
            url,
            "PUT",
            path,
            {"experience": experience.to_dict(), "correlation": {}},
        )
        unknown_read_field = _http(
            url,
            "POST",
            "/v1/read",
            {"decision": DECISION, "context": {}, "outcomes": "keep"},
        )
        missing_decision = _http(url, "POST", "/v1/read", {"context": {}})
        unknown_path = _http(url, "GET", "/v1/events")

    for status, body in (
        mismatched,
        unfinished,
        wrong_schema,
        unknown_write_field,
        unknown_read_field,
        missing_decision,
    ):
        assert status == 400
        assert body["error"] == "invalid_request"
        assert body["detail"]
    assert "unknown request fields: correlation" in unknown_write_field[1]["detail"]
    assert "unknown request fields: outcomes" in unknown_read_field[1]["detail"]
    assert unknown_path == (404, {"error": "not_found"})


def test_storage_failure_is_retryable_and_spooled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _declaration()
    app = _app(tmp_path / "service", schema)

    def unavailable_disk(_experience: Experience) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(app._store, "insert_complete", unavailable_disk)
    experience = _experience(schema)
    with RunningServer(app) as url:
        status, body = _http(
            url,
            "PUT",
            f"/v1/experiences/{experience.id}",
            {"experience": experience.to_dict()},
        )
        write = _client(url, tmp_path).publish(experience)

    assert (status, body) == (500, {"error": "internal_error", "detail": "OSError"})
    assert write.status == "spooled"
    assert len(tuple((tmp_path / "spool").glob("spool-*.json"))) == 1


def test_every_endpoint_requires_the_service_token(tmp_path: Path) -> None:
    with pytest.raises(RemoteClientError, match="HYPERLOOM_KB_TOKEN"):
        RemoteConfig.from_env({"HYPERLOOM_KB_URL": "https://kb.example"})
    with pytest.raises(HTTPServiceError, match="HYPERLOOM_KB_TOKEN"):
        HTTPServiceConfig(tmp_path / "service", "")

    experience = _experience(_declaration())
    requests: tuple[tuple[str, str, object], ...] = (
        ("GET", "/health", None),
        ("GET", "/v1/list", None),
        ("POST", "/v1/read", {"decision": DECISION, "context": {}}),
        ("PUT", f"/v1/experiences/{experience.id}", {"experience": experience.to_dict()}),
        ("GET", "/v1/unknown", None),
    )
    with RunningServer(_app(tmp_path / "service")) as url:
        responses = [
            _http(url, method, path, body, token=token)
            for method, path, body in requests
            for token in (None, "wrong-token")
        ]

    assert responses == [(401, {"error": "unauthorized"})] * len(responses)


def test_unavailable_service_fails_open_and_flushes_spool_idempotently(tmp_path: Path) -> None:
    schema = _declaration()
    spool = tmp_path / "spool"

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    offline = RemoteClient(
        RemoteConfig("http://service.invalid", TOKEN, spool_root=spool),
        opener=unavailable,
    )
    experience = _experience(schema)
    read = offline.read(DECISION, _read_context())
    spooled = offline.publish(experience)
    assert offline.flush_spool() == ()

    with RunningServer(_app(tmp_path / "service", schema)) as url:
        online = RemoteClient(RemoteConfig(url, TOKEN, spool_root=spool))
        flushed = online.flush_spool()
        replay = online.publish(experience)

    assert (read.status, read.prompt_block, read.rendered_refs) == ("unavailable", "", ())
    assert spooled.status == "spooled"
    assert [item.status for item in flushed] == ["created"]
    assert replay.status == "unchanged"
    assert not tuple(spool.glob("spool-*.json"))


def test_rejected_spool_file_does_not_block_later_writes(tmp_path: Path) -> None:
    schema = _declaration()
    spool = tmp_path / "spool"
    first = _experience(schema, seq=0, knob="first")
    second = _experience(schema, seq=1, knob="second")
    poison, valid = sorted(
        (first, second),
        key=lambda item: hashlib.sha256(item.id.encode()).hexdigest(),
    )

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    offline = RemoteClient(
        RemoteConfig("http://service.invalid", TOKEN, spool_root=spool),
        opener=unavailable,
    )
    offline.publish(replace(poison, reflection="Conflicting spooled content."))
    offline.publish(valid)
    (spool / "spool-corrupt.json").write_text("{", encoding="utf-8")

    with RunningServer(_app(tmp_path / "service", schema)) as url:
        online = RemoteClient(RemoteConfig(url, TOKEN, spool_root=spool))
        online.publish(poison)
        flushed = online.flush_spool()
        listed = online.list_experiences()

    assert [item.experience_id for item in flushed] == [valid.id]
    assert {item["experience_id"] for item in listed.items} == {poison.id, valid.id}
    assert not tuple(spool.glob("spool-*.json"))
    assert len(tuple((spool / "rejected").glob("spool-*.json"))) == 2


def test_sdk_uses_only_url_and_token_with_packaged_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    schema = load_declaration(PACKAGED_DECLARATION)
    app = _app(tmp_path / "service", schema)

    with RunningServer(app) as url:
        kb = experience_kb_from_env({"HYPERLOOM_KB_URL": url, "HYPERLOOM_KB_TOKEN": TOKEN})
        assert isinstance(kb, RemoteExperienceKB)
        assert kb.client.config.spool_root == tmp_path / "home" / ".cache/hyperloom/kb-spool"
        session = kb.begin(
            run_id="run-remote",
            seq=0,
            identity={
                "model": "qwen3",
                "gpu": "mi325x",
                "framework": "vllm",
                "model_type": "dense",
                "architecture": "Qwen3ForCausalLM",
                "framework_version": "0.10.0",
                "precision": "bf16",
            },
            objective="e2e_throughput@v1",
            baseline_identity={"baseline_fingerprint": "baseline-1"},
            baseline_value=100.0,
            provenance=Provenance("service-test", "1"),
            created_at=NOW,
        )
        session.decide(
            reasoning="Prior evidence supports tuning page size.",
            change=Change(
                {"change_family": "config", "change_fingerprint": "change-1"},
                "Tune page size.",
                kind="config",
            ),
        )
        session.complete(
            outcome=Outcome("keep", 110.0),
            reflection="The result improved.",
            completed_at=NOW,
        )
        receipt = session.publish()
        read = kb.read(DECISION, _read_context(), outcome="keep")

    assert kb.declaration.schema_ref == app.declaration.schema_ref
    assert receipt.status == "created"
    assert [item.id for item in read.rendered_refs] == [session.record.id]


def test_seed_imports_jsonl_idempotently(tmp_path: Path) -> None:
    schema = _declaration()
    app = _app(tmp_path / "service", schema)
    first = _experience(schema, seq=0, knob="first")
    second = _experience(schema, seq=1, knob="second")
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        "\n".join(
            (
                json.dumps({"experience": first.to_dict()}),
                "",
                json.dumps(second.to_dict()),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert app.seed(seed) == {"created": 2, "unchanged": 0}
    assert app.seed(seed) == {"created": 0, "unchanged": 2}
    assert [item["experience_id"] for item in app.list_experiences()["items"]] == [
        first.id,
        second.id,
    ]
