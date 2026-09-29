# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework attempts reach the Experience KB through the packaged hyperloom-sbd-v6 mapping."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer import experience_collect
from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.schema import SCHEMA_VERSION_V6
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework import _patch_material
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan

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
    (session / "patches" / "big.diff").write_text("+" * (129 * 1024))
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
        {"path": "patches/a.diff", "sha256": hashlib.sha256(b"+a = 2\n").hexdigest(), "content": "+a = 2\n"}
    ]


def test_unconfigured_collection_never_imports_the_sdk(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    monkeypatch.setitem(sys.modules, "hyperloom_kb", None)
    monkeypatch.setitem(sys.modules, "hyperloom_kb.collect", None)

    experience_collect.validate_config()
    experience_collect.collect_session(tmp_path, {"timeline": []})


def _fake_sdk(monkeypatch, *, collect: Any, schema_ref: str = "schema:sha256:" + "a" * 64) -> None:
    class ConfigurationError(ValueError):
        pass

    class MappingError(ValueError):
        pass

    class SourceDocumentError(ValueError):
        pass

    class RemoteClientError(RuntimeError):
        pass

    sdk = ModuleType("hyperloom_kb")
    sdk.ConfigurationError = ConfigurationError
    sdk.RemoteClientError = RemoteClientError
    sdk.experience_kb_from_env = lambda: SimpleNamespace(schema_ref=schema_ref)
    collect_module = ModuleType("hyperloom_kb.collect")
    collect_module.MappingError = MappingError
    collect_module.SourceDocumentError = SourceDocumentError
    collect_module.collect = collect
    collect_module.load_mapping = lambda name: SimpleNamespace(declaration=SimpleNamespace(schema_ref=schema_ref))
    monkeypatch.setitem(sys.modules, "hyperloom_kb", sdk)
    monkeypatch.setitem(sys.modules, "hyperloom_kb.collect", collect_module)
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://kb.invalid")


def test_breakdown_write_collects_through_the_packaged_mapping(monkeypatch, session_dir: Path) -> None:
    seen: dict[str, Any] = {}

    def collect(mapping: str, document: dict[str, Any], *, receipt: Path) -> Any:
        seen.update(mapping=mapping, document=document, receipt=receipt)
        return SimpleNamespace(to_dict=lambda: {"counts": {"collected": 0}})

    _fake_sdk(monkeypatch, collect=collect)

    target = exporter.write_breakdown_json(session_dir)

    assert target.is_file()
    assert seen["mapping"] == "hyperloom-sbd-v6"
    assert seen["document"]["schema_version"] == SCHEMA_VERSION_V6
    assert seen["receipt"] == session_dir / "reports" / "experience_collect.json"


def test_collection_failure_leaves_the_breakdown_written(monkeypatch, session_dir: Path) -> None:
    def collect(*_args: Any, **_kwargs: Any) -> Any:
        raise sys.modules["hyperloom_kb.collect"].MappingError("broken mapping")

    _fake_sdk(monkeypatch, collect=collect)

    assert exporter.write_breakdown_json(session_dir).is_file()


def test_startup_rejects_a_kb_that_validates_another_declaration(monkeypatch) -> None:
    _fake_sdk(monkeypatch, collect=lambda *a, **k: None)
    sys.modules["hyperloom_kb"].experience_kb_from_env = lambda: SimpleNamespace(schema_ref="schema:sha256:other")

    with pytest.raises(ValueError, match="hyperloom-sbd-v6 produces"):
        experience_collect.validate_config()


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
    coord._close_framework_timeline(exit_reason="optimize_no_more_leverage")
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


def test_recorded_framework_attempts_satisfy_the_packaged_mapping(session_dir: Path) -> None:
    collect_module = pytest.importorskip("hyperloom_kb.collect")
    breakdown = {
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
            "grading": {"benchmark_mode": "synthetic"},
        },
        "timeline": _record_framework_attempts(session_dir),
    }

    report = collect_module.collect(experience_collect.MAPPING, breakdown, dry_run=True).to_dict()

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
