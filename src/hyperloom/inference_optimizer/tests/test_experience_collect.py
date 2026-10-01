# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework attempts reach the Experience KB through the packaged hyperloom-sbd-v6 mapping."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import hyperloom_kb.collect as kb_collect
from hyperloom.inference_optimizer import experience_collect, experience_kb_service
from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.schema import SCHEMA_VERSION_V6
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework import _patch_material
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
from hyperloom_kb import (
    PACKAGED_DECLARATION,
    ConfigurationError,
    ExperienceHTTPService,
    HTTPServiceConfig,
    RemoteClient,
    RemoteConfig,
    create_http_server,
    load_declaration,
)

_AUTHORING_REF = {"id": "exp-00000000000000000000000000000002", "purpose": "representative"}


@pytest.fixture
def session_dir(tmp_path: Path):
    path = tmp_path / "session"
    path.mkdir()
    with session_scope(path):
        yield path


def _coordinator(session_dir: Path) -> Coordinator:
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=session_dir,
        backends={"orchestration": MockBackend(idle), "critic": MockBackend(idle)},
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


def test_patch_material_keeps_only_publishable_session_patches(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "patches").mkdir(parents=True)
    kept = session / "patches" / "a.diff"
    kept.write_text("+a = 2\n")
    big = "+" * (3 * 1024 * 1024)
    (session / "patches" / "big.diff").write_text(big)
    (session / "patches" / "secret.diff").write_text("+OPENAI_API_KEY=sk-abcdefghijkl\n")
    (session / "patches" / "binary.diff").write_bytes(b"\xff\xfe")
    outside = tmp_path / "outside.diff"
    outside.write_text("+b = 3\n")

    material = _patch_material(
        session,
        [
            str(kept),
            "patches/a.diff",
            "patches/big.diff",
            "patches/secret.diff",
            "patches/binary.diff",
            "patches/missing.diff",
            str(outside),
            "",
        ],
    )

    assert material == [
        {"path": "patches/a.diff", "sha256": hashlib.sha256(b"+a = 2\n").hexdigest(), "content": "+a = 2\n"},
        {"path": "patches/big.diff", "sha256": hashlib.sha256(big.encode()).hexdigest(), "content": big},
    ]


def test_unconfigured_collection_is_a_no_op(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    monkeypatch.setattr(experience_collect, "collect", _unexpected_collect)
    monkeypatch.setattr(experience_collect, "experience_kb_from_env", _unexpected_collect)

    experience_collect.validate_config()
    experience_collect.collect_session(tmp_path, {"timeline": []})


def _unexpected_collect(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("an unconfigured Experience KB must not be contacted")


def _configured_kb(monkeypatch, *, collect: Any, schema_ref: str | None = None) -> SimpleNamespace:
    target = SimpleNamespace(schema_ref=schema_ref or experience_collect.mapping_schema_ref())
    monkeypatch.setattr(experience_collect, "collect", collect)
    monkeypatch.setattr(experience_collect, "experience_kb_from_env", lambda **_kwargs: target)
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://kb.invalid")
    for key in ("HYPERLOOM_KB_AUTO_PUSH", "HYPERLOOM_GLOBAL_KB_URL", "HYPERLOOM_GLOBAL_KB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    return target


def test_breakdown_write_collects_through_the_packaged_mapping(monkeypatch, session_dir: Path) -> None:
    seen: dict[str, Any] = {}

    def collect(mapping: str, document: dict[str, Any], *, kb: Any, receipt: Path) -> Any:
        seen.update(mapping=mapping, document=document, kb=kb, receipt=receipt)
        return SimpleNamespace(to_dict=lambda: {"counts": {"collected": 0}})

    target = _configured_kb(monkeypatch, collect=collect)

    written = exporter.write_breakdown_json(session_dir)

    assert written.is_file()
    assert seen["mapping"] == "hyperloom-sbd-v6"
    assert seen["document"]["schema_version"] == SCHEMA_VERSION_V6
    assert seen["kb"] is target
    assert seen["receipt"] == session_dir / "reports" / "experience_collect.json"


def test_collection_failure_leaves_the_breakdown_written(monkeypatch, session_dir: Path) -> None:
    def collect(*_args: Any, **_kwargs: Any) -> Any:
        raise kb_collect.MappingError("broken mapping")

    _configured_kb(monkeypatch, collect=collect)

    assert exporter.write_breakdown_json(session_dir).is_file()


def test_startup_accepts_a_kb_that_validates_the_mapping_declaration(monkeypatch) -> None:
    _configured_kb(monkeypatch, collect=_unexpected_collect)

    experience_collect.validate_config()


def test_startup_rejects_a_kb_that_validates_another_declaration(monkeypatch) -> None:
    _configured_kb(monkeypatch, collect=_unexpected_collect, schema_ref="schema:sha256:other")

    with pytest.raises(ConfigurationError, match="hyperloom-sbd-v6 produces"):
        experience_collect.validate_config()


# A dotenv value keeps an inline comment, which is how a first run once lost its push after three hours.
_UNREADABLE_SWITCH = {"HYPERLOOM_KB_AUTO_PUSH": "1        # push after every run"}
_NO_GLOBAL_KB = {"HYPERLOOM_KB_AUTO_PUSH": "1"}


@pytest.mark.parametrize(
    ("env", "reason"),
    [(_UNREADABLE_SWITCH, "is not a boolean"), (_NO_GLOBAL_KB, "HYPERLOOM_GLOBAL_KB_URL is not configured")],
)
def test_an_unusable_auto_push_setting_is_a_warning_at_launch(monkeypatch, caplog, env, reason) -> None:
    _configured_kb(monkeypatch, collect=_unexpected_collect)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with caplog.at_level(logging.WARNING):
        experience_collect.validate_config()

    assert "auto push is off for this run" in caplog.text
    assert reason in caplog.text


def test_a_run_with_an_unusable_auto_push_setting_still_exports_and_collects(
    monkeypatch, caplog, session_dir: Path
) -> None:
    collected: list[str] = []

    def collect(mapping: str, *_args: Any, **_kwargs: Any) -> Any:
        collected.append(mapping)
        return SimpleNamespace(to_dict=lambda: {"counts": {"collected": 0}})

    _configured_kb(monkeypatch, collect=collect)
    monkeypatch.setattr(experience_kb_service, "sync_with_global", _unexpected_collect)
    monkeypatch.setenv(*next(iter(_UNREADABLE_SWITCH.items())))

    with caplog.at_level(logging.WARNING):
        written = exporter.write_breakdown_json(session_dir)

    assert written.is_file()
    assert collected == ["hyperloom-sbd-v6"]
    assert "auto push is off for this run" in caplog.text


def test_auto_push_with_a_global_kb_is_quiet_at_launch(monkeypatch, caplog) -> None:
    _configured_kb(monkeypatch, collect=_unexpected_collect)
    monkeypatch.setenv("HYPERLOOM_KB_AUTO_PUSH", "1")
    monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_URL", "http://global.invalid")
    monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_TOKEN", "global-token")

    with caplog.at_level(logging.WARNING):
        experience_collect.validate_config()

    assert "auto push" not in caplog.text


def _record_framework_attempts(session_dir: Path) -> list[dict[str, Any]]:
    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.phase_framework._open_framework_timeline()
    patch = session_dir / "artifacts" / "source.patch"
    patch.parent.mkdir()
    patch.write_text("diff --git a/vllm/attention.py b/vllm/attention.py\n+optimized = True\n")
    coord.shared_state.framework_agent_specialist_candidate_map = {"t-auth-1": "https://x/pr/1"}
    coord.phase_framework._record_framework_agent_authored_outcome(
        task=SimpleNamespace(
            task_id="t-int-1",
            kind="integrate_patch",
            params={
                "framework_agent_authoring": True,
                "specialist_task_id": "t-auth-1",
                "framework_agent_candidate_id": "https://x/pr/1",
                "audit_step": "author_via_specialist",
                "lever_kind": "upstream_pr",
                "reasoning": "Profiling shows redundant attention setup on every request.",
                # What the authoring specialist was shown, forwarded from its dispatch.
                "kb_read_id": "read-authoring",
                "kb_rendered_refs": [_AUTHORING_REF],
            },
        ),
        result={
            "status": "kept",
            "base_tput": 100.0,
            "output_throughput": 108.0,
            "delta_pct": 8.0,
            "keep_threshold_pct": 3.0,
            "accuracy_pass": True,
            "accuracy_value": 0.83,
            "accuracy_reference": 0.80,
            "source_realized_patch": str(patch),
            "patches_applied": [str(patch)],
            "target_files": ["vllm/attention.py"],
            "measured_against": {"throughput": 100.0, "extra_server_args": "--already-kept 1"},
        },
    )
    import asyncio

    asyncio.run(
        coord._fact_write_hook(
            task=SimpleNamespace(task_id="t-exp-1", kind="explore", params={"proposal_msg_id": "p-config"}),
            result={
                "round_id": "explore-001",
                "per_variant_outcomes": [
                    {
                        "variant_name": "chunked-prefill",
                        "outcome": "REVERT",
                        "reason": "gain_below_threshold",
                        "fingerprint": "fp1",
                        "metrics": {"base_tput": 108.0, "tput": 107.5, "gain_pct": -0.46},
                        "variant": {
                            "extra_server_args": "--enable-chunked-prefill",
                            "extra_envs": {},
                            "note": "Prefill dominates at isl=8192; chunked prefill should overlap decode.",
                            "reasoning_origin": "action_payload.reasoning",
                        },
                        "measured_against": {"throughput": 108.0, "extra_server_args": "--already-kept 1"},
                        "gates": [{"gate": "keep_threshold", "passed": False, "observed": -0.46, "threshold": 3.0}],
                    }
                ],
            },
            kept=False,
        )
    )
    coord.phase_framework.close_timeline(exit_reason="optimize_no_more_leverage")
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "framework_agent"]


def test_recorded_framework_rows_only_carry_declared_fields(session_dir: Path) -> None:
    from hyperloom.inference_optimizer.breakdown import schema

    ext = _record_framework_attempts(session_dir)[0]["ext"]

    for attempt in ext["attempts"]:
        assert set(attempt) <= set(schema.V6FrameworkAttempt.__annotations__)
        assert set(attempt["failure"]) <= set(schema.V6FrameworkAttemptFailure.__annotations__)
        for patch in attempt.get("patch_material", []):
            assert set(patch) <= set(schema.V6FrameworkPatch.__annotations__)
    for proposal in ext["proposals"]:
        assert set(proposal) <= set(schema.V6FrameworkProposal.__annotations__)


def _breakdown(session_dir: Path) -> dict[str, Any]:
    return _document(_record_framework_attempts(session_dir))


def _document(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata": {
            "session": {"session_id": "session-42"},
            "task_config": {
                "model_name": "qwen3-8b",
                "gpu_type": "mi355x",
                "framework_name": "sglang",
                "framework_version": "0.5.18",
                "precision": "bf16",
                "architecture": {"model_type": "qwen3", "model_class": "Qwen3ForCausalLM"},
            },
            "grading": {"benchmark_mode": "synthetic", "objective": "output_throughput"},
        },
        "timeline": timeline,
    }


def test_writes_the_service_cannot_take_wait_under_user_data_path(
    monkeypatch, session_dir: Path, tmp_path: Path
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    monkeypatch.setenv("HYPERLOOM_KB_URL", f"http://127.0.0.1:{closed_port}")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "workspace-token")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "data"))

    experience_collect.collect_session(session_dir, _breakdown(session_dir))

    receipt = json.loads((session_dir / experience_collect.RECEIPT).read_text(encoding="utf-8"))
    assert {row["status"] for row in receipt["collected"]} == {"spooled"}
    spooled = list((tmp_path / "data" / "experience-kb" / "spool").glob("spool-*.json"))
    assert len(spooled) == len(receipt["collected"])


def test_a_later_run_reads_what_an_earlier_run_wrote_after_the_service_restarts(
    monkeypatch, session_dir: Path, tmp_path: Path
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("HYPERLOOM_KB_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "workspace-token")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "data"))
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    first_run = experience_kb_service.ensure_service()
    assert first_run is not None and first_run.process is not None
    try:
        experience_collect.collect_session(session_dir, _breakdown(session_dir))
    finally:
        first_run.process.terminate()
        first_run.process.wait(timeout=10)
    receipt = json.loads((session_dir / experience_collect.RECEIPT).read_text(encoding="utf-8"))
    written = {row["experience_id"] for row in receipt["collected"]}
    assert {row["status"] for row in receipt["collected"]} == {"created"}

    second_run = experience_kb_service.ensure_service()
    assert second_run is not None and second_run.process is not None
    try:
        client = RemoteClient(RemoteConfig.from_env(spool_root=tmp_path / "unused-spool"))
        listed = {str(item["experience_id"]) for item in client.list_experiences().items}
    finally:
        second_run.process.terminate()
        second_run.process.wait(timeout=10)

    # ``experience_count`` counts the corpus every read searches.
    assert second_run.health["experience_count"] == len(written) > 0
    assert listed == written


def _workspace(monkeypatch, root: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("HYPERLOOM_KB_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", f"{root.name}-token")
    monkeypatch.setenv("USER_DATA_PATH", str(root))


def _stop_workspace_service() -> None:
    service = experience_kb_service.ensure_service()
    assert service is not None
    os.kill(int(service.health["pid"]), signal.SIGTERM)


def test_an_auto_pushed_run_reaches_another_workspace_that_pulls(
    monkeypatch, session_dir: Path, tmp_path: Path
) -> None:
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    declaration = load_declaration(PACKAGED_DECLARATION)
    global_kb = ExperienceHTTPService(HTTPServiceConfig(tmp_path / "global", "global-token"), declaration, None)
    server = create_http_server(global_kb, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_TOKEN", "global-token")
        monkeypatch.setenv("HYPERLOOM_KB_AUTO_PUSH", "1")

        _workspace(monkeypatch, tmp_path / "first")
        try:
            experience_collect.collect_session(session_dir, _breakdown(session_dir))
        finally:
            _stop_workspace_service()
        receipt = json.loads((session_dir / experience_collect.RECEIPT).read_text(encoding="utf-8"))
        written = {row["experience_id"] for row in receipt["collected"]}
        shared = {str(item["experience_id"]) for item in global_kb.list_experiences()["items"]}

        _workspace(monkeypatch, tmp_path / "second")
        try:
            assert experience_kb_service.main(["pull"]) == 0
            config = RemoteConfig.from_env()
            assert config is not None
            pulled = {str(item["experience_id"]) for item in RemoteClient(config).list_experiences().items}
        finally:
            _stop_workspace_service()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert written and shared == written
    assert pulled == written


def test_recorded_framework_attempts_satisfy_the_packaged_mapping(session_dir: Path) -> None:
    breakdown = _breakdown(session_dir)

    report = kb_collect.collect(experience_collect.MAPPING, breakdown, dry_run=True).to_dict()

    assert report["skipped"] == []
    experiences = {row["unit_id"]: row["experience"] for row in report["collected"]}
    source = experiences["t-int-1"]
    assert source["change"]["kind"] == "source_patch"
    assert source["change"]["resource_refs"] == ["artifacts/source.patch"]
    assert "optimized = True" in source["change"]["content"]
    assert source["reasoning"] == "Profiling shows redundant attention setup on every request."
    assert source["rendered_refs"] == [_AUTHORING_REF]
    assert source["provenance"]["extra"]["kb_read_id"] == "read-authoring"
    config = experiences["t-exp-1:explore-001:fp1"]
    assert config["change"]["kind"] == "config_variant"
    assert config["outcome"]["decision"] == "revert"
    assert '"extra_server_args":"--already-kept 1"' in config["preconditions"][2]


def test_a_specialists_config_only_deliverable_is_published_as_a_config_experience(session_dir: Path) -> None:
    import asyncio

    coord = _coordinator(session_dir)
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.phase_framework._open_framework_timeline()
    discovery_reasoning = "The PR routes MoE through the fused kernel this profile shows idle."
    coord.phase_framework._ingest_candidate_discovery(
        task=SimpleNamespace(
            task_id="t-disc-7", params={"candidate_discovery": True, "domain": "candidate_discovery_specialist"}
        ),
        done_payload={
            "proposal_set": [
                {
                    "pr_url": "https://x/pr/7",
                    "title": "Route MoE through the fused kernel",
                    "repo": "vllm",
                    "verdict": "worth_a_bench",
                    "reasoning": discovery_reasoning,
                }
            ]
        },
    )
    authoring = SimpleNamespace(
        task_id="t-auth-7",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "https://x/pr/7",
            "domain": "serving_specialist",
            "kb_read_id": "read-authoring",
            "kb_rendered_refs": [],
        },
    )
    asyncio.run(
        coord.phase_framework.maybe_autosubmit_config(
            task=authoring,
            done_payload={
                "proposal_set": [
                    {
                        "name": "fused-moe-routing",
                        "extra_args": "--enable-fused-moe",
                        "extra_envs": {"VLLM_FUSED_MOE": "1"},
                        "reason": "The PR reduces to this server flag on the installed version.",
                    }
                ]
            },
        )
    )
    [pending] = coord.state.pending_proposals.values()
    coord.phase_framework._record_framework_agent_authored_outcome(
        task=SimpleNamespace(task_id="t-int-7", kind="integrate_patch", params=dict(pending.payload["params"])),
        result={
            "status": "kept",
            "base_tput": 100.0,
            "output_throughput": 106.0,
            "delta_pct": 6.0,
            "keep_threshold_pct": 3.0,
            "patches_applied": [],
            "measured_against": {"throughput": 100.0, "extra_server_args": "--already-kept 1"},
        },
    )
    coord.phase_framework.close_timeline(exit_reason="optimize_no_more_leverage")
    timeline = [event for event in read_timeline_events(session_dir) if event.get("type") == "framework_agent"]

    report = kb_collect.collect(experience_collect.MAPPING, _document(timeline), dry_run=True).to_dict()

    assert report["skipped"] == []
    [row] = report["collected"]
    experience = row["experience"]
    assert experience["change"]["kind"] == "config_variant"
    assert json.loads(experience["change"]["content"])["extra_server_args"] == "--enable-fused-moe"
    assert json.loads(experience["change"]["content"])["extra_envs"] == {"VLLM_FUSED_MOE": "1"}
    assert experience["outcome"]["decision"] == "keep"
    assert experience["reasoning"] == discovery_reasoning
    assert experience["provenance"]["extra"]["arm"] == "source"
    assert experience["provenance"]["extra"]["kb_read_id"] == "read-authoring"
