# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import experience_v1
from hyperloom.inference_optimizer.breakdown import exporter


@dataclass(frozen=True)
class FakeChange:
    identity: dict
    summary: str
    kind: str
    content: str
    resource_refs: tuple


@dataclass(frozen=True)
class FakeConstraint:
    name: str
    passed: bool
    value: object


@dataclass(frozen=True)
class FakeOutcome:
    decision: str
    value: float | None
    constraints: tuple
    error_class: str


@dataclass(frozen=True)
class FakeProvenance:
    producer: str
    producer_version: str
    snapshot_version: str
    source_ref: str
    extra: dict


@dataclass(frozen=True)
class FakeRenderedRef:
    id: str
    purpose: str

    @classmethod
    def from_dict(cls, value):
        return cls(str(value["id"]), str(value.get("purpose") or ""))


class FakeSession:
    def __init__(self, experience_id: str, publish_status: str = "") -> None:
        self.record = SimpleNamespace(
            id=experience_id,
            status=SimpleNamespace(value="in_progress"),
            change=None,
        )
        self.decisions: list[dict] = []
        self.completions: list[dict] = []
        self.publish_status = publish_status

    def decide(self, **kwargs):
        self.decisions.append(kwargs)
        self.record = SimpleNamespace(
            id=self.record.id,
            status=SimpleNamespace(value="in_progress"),
            change=kwargs["change"],
        )

    def complete(self, **kwargs):
        self.completions.append(kwargs)
        self.record = SimpleNamespace(
            id=self.record.id,
            status=SimpleNamespace(value="complete"),
            change=self.record.change,
        )

    def publish(self):
        return SimpleNamespace(status=self.publish_status)


class FakeKB:
    enabled = True

    def __init__(self, publish_status: str = "") -> None:
        self.sessions: dict[int, FakeSession] = {}
        self.begin_calls: list[dict] = []
        self.publish_status = publish_status

    def begin(self, **kwargs):
        self.begin_calls.append(kwargs)
        return self.sessions.setdefault(
            kwargs["seq"],
            FakeSession(f"exp-{kwargs['seq']:032x}", self.publish_status),
        )


def fake_module(kb: FakeKB):
    return SimpleNamespace(
        Change=FakeChange,
        ConstraintResult=FakeConstraint,
        Outcome=FakeOutcome,
        Provenance=FakeProvenance,
        RenderedRef=FakeRenderedRef,
        experience_kb_from_env=lambda *_args, **_kwargs: kb,
    )


def breakdown(*, benchmark_mode: str = "synthetic") -> dict:
    return {
        "schema_version": "hyperloom.session_breakdown.v6",
        "metadata": {
            "session": {"session_id": "framework-run-1"},
            "task_config": {
                "model_name": "qwen3-8b",
                "gpu_type": "mi355x",
                "framework_name": "sglang",
                "framework_version": "0.5.18",
                "precision": "bf16",
                "tp": 8,
                "ep": 4,
                "conc": 64,
                "isl": 8192,
                "osl": 1024,
                "max_model_len": 16384,
                "compute_partition": {"mode": "CPX", "partitions": 8},
                "architecture": {
                    "model_type": "qwen3",
                    "model_class": "Qwen3ForCausalLM",
                    "architectures": ["Qwen3ForCausalLM"],
                },
            },
            "grading": {"benchmark_mode": benchmark_mode},
        },
        "timeline": [
            {
                "type": "framework_agent",
                "event_id": "framework_agent:0:framework",
                "start_time": "2026-09-17T12:00:00Z",
                "ext": {
                    "proposals": [
                        {
                            "proposal_id": "proposal-1",
                            "reasoning": "Larger prefill chunks should reduce scheduler overhead.",
                        }
                    ],
                    "attempts": [
                        {
                            "attempt_id": "attempt-keep",
                            "arm": "config",
                            "task_id": "task-keep",
                            "proposal_ref": "proposal-1",
                            "reasoning": "Larger prefill chunks should reduce scheduler overhead.",
                            "reasoning_origin": "action_payload.reasoning",
                            "variant_name": "chunk-8192",
                            "outcome": "KEEP",
                            "ts": "2026-09-17T12:10:00Z",
                            "measurement": {
                                "before_tput": 800.0,
                                "after_tput": 860.0,
                                "gain_pct": 7.5,
                            },
                            "measured_against": {
                                "extra_server_args": "--base",
                                "extra_envs": {"BASE": "1"},
                            },
                            "config_delta": {
                                "extra_server_args": "--max-num-batched-tokens 8192",
                                "extra_envs": {},
                                "remove_args": ["--old-batch-limit"],
                                "unset_envs": ["OLD_SCHEDULER_MODE"],
                                "args_mode": "replace",
                            },
                            "accuracy": {
                                "required": True,
                                "passed": True,
                                "value": 0.82,
                            },
                            "gates": [
                                {
                                    "gate": "accuracy",
                                    "passed": True,
                                    "observed": 0.82,
                                }
                            ],
                        },
                        {
                            "attempt_id": "attempt-failed",
                            "arm": "config",
                            "task_id": "task-failed",
                            "reasoning": "Test whether compilation removes repeated Python dispatch overhead.",
                            "reasoning_origin": "action_payload.reasoning",
                            "variant_name": "compile",
                            "outcome": "FAILED",
                            "ts": "2026-09-17T12:20:00Z",
                            "measurement": {"before_tput": 860.0},
                            "measured_against": {
                                "extra_server_args": "--max-num-batched-tokens 8192",
                                "extra_envs": {"BASE": "1"},
                            },
                            "config_delta": {
                                "extra_server_args": "--enable-torch-compile",
                                "extra_envs": {},
                            },
                            "failure": {
                                "error_class": "capability_unsupported",
                                "error_excerpt": "the requested compile path is unsupported",
                                "attribution": "candidate_caused",
                            },
                        },
                        {
                            "attempt_id": "attempt-no-reasoning",
                            "arm": "config",
                            "variant_name": "page-size",
                            "outcome": "REVERT",
                            "ts": "2026-09-17T12:30:00Z",
                            "measurement": {
                                "before_tput": 860.0,
                                "after_tput": 850.0,
                            },
                            "measured_against": {
                                "extra_server_args": "--max-num-batched-tokens 8192",
                                "extra_envs": {"BASE": "1"},
                            },
                            "config_delta": {
                                "extra_server_args": "--page-size 64",
                                "extra_envs": {},
                            },
                        },
                    ],
                },
            }
        ],
    }


def test_review_selects_only_fidelity_complete_attempts(tmp_path: Path) -> None:
    review = experience_v1.build_framework_experience_review(tmp_path, breakdown())

    assert review["blocked_reason"] == ""
    assert review["framework_attempts"] == 3
    assert {row["attempt_id"] for row in review["ready"]} == {
        "attempt-keep",
        "attempt-failed",
    }
    assert review["skipped"] == [
        {
            "attempt_id": "attempt-no-reasoning",
            "reason": "decision reasoning is missing or non-specific",
        }
    ]
    keep = next(row for row in review["ready"] if row["attempt_id"] == "attempt-keep")
    assert keep["baseline_value"] == 800.0
    assert keep["identity"]["ep"] == 4
    assert keep["identity"]["max_model_len"] == 16384
    assert keep["identity"]["compute_partition_mode"] == "CPX"
    assert keep["identity"]["partitions"] == 8
    assert keep["baseline_configuration"] == {
        "extra_server_args": "--base",
        "extra_envs": {"BASE": "1"},
        "remove_args": [],
        "unset_envs": [],
        "args_mode": "append",
    }
    assert len(keep["baseline_identity"]["baseline_fingerprint"]) == 64
    assert keep["outcome_value"] == 860.0
    assert keep["reasoning_origin"] == "action_payload.reasoning"
    assert keep["constraints"] == [{"name": "accuracy", "passed": True, "value": 0.82}]
    projected, _ = experience_v1._project_all(tmp_path, breakdown())
    keep_projected = next(item for item in projected if item.attempt_id == "attempt-keep")
    assert json.loads(keep_projected.change_content) == {
        "extra_server_args": "--max-num-batched-tokens 8192",
        "extra_envs": {},
        "remove_args": ["--old-batch-limit"],
        "unset_envs": ["OLD_SCHEDULER_MODE"],
        "args_mode": "replace",
    }
    failed = next(row for row in review["ready"] if row["attempt_id"] == "attempt-failed")
    assert failed["failure_attribution"] == "candidate_caused"


def test_review_keeps_runtime_failure_in_sbd_but_not_experience(tmp_path: Path) -> None:
    value = breakdown()
    attempt = value["timeline"][0]["ext"]["attempts"][1]
    attempt["failure"] = {
        "error_class": "magpie_nonzero_invalid_measurement",
        "error_excerpt": "benchmark subprocess returned no valid measurement",
        "attribution": "harness",
    }

    review = experience_v1.build_framework_experience_review(tmp_path, value)

    assert {row["attempt_id"] for row in review["ready"]} == {"attempt-keep"}
    assert {
        "attempt_id": "attempt-failed",
        "reason": "failed attempt is not candidate-attributed (failure_attribution=harness)",
    } in review["skipped"]


def test_review_rejects_reasoning_not_traceable_to_action_time(tmp_path: Path) -> None:
    value = breakdown()
    value["timeline"][0]["ext"]["proposals"] = []
    attempt = value["timeline"][0]["ext"]["attempts"][0]
    attempt["reasoning_origin"] = "post_action_result.reasoning"

    review = experience_v1.build_framework_experience_review(tmp_path, value)

    assert {
        "attempt_id": "attempt-keep",
        "reason": "decision reasoning is not traceable to the action-time proposal",
    } in review["skipped"]


def test_review_blocks_agentx_until_identity_is_real(tmp_path: Path) -> None:
    review = experience_v1.build_framework_experience_review(
        tmp_path,
        breakdown(benchmark_mode="agentx"),
    )

    assert review["blocked_reason"] == "agentx_experience_identity_not_supported"
    assert review["ready"] == []
    assert len(review["skipped"]) == 3
    assert {row["reason"] for row in review["skipped"]} == {"agentx_experience_identity_not_supported"}


def test_review_rejects_keep_without_required_accuracy(tmp_path: Path) -> None:
    value = breakdown()
    attempt = value["timeline"][0]["ext"]["attempts"][0]
    attempt["accuracy"]["passed"] = False

    review = experience_v1.build_framework_experience_review(tmp_path, value)

    assert {
        "attempt_id": "attempt-keep",
        "reason": "kept attempt did not pass required accuracy",
    } in review["skipped"]


def test_review_requires_verifiable_source_patch_and_preserves_resource(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "artifacts" / "change.patch"
    patch.parent.mkdir()
    patch.write_text("diff --git a/a.py b/a.py\n+optimized = True\n")
    value = breakdown()
    ext = value["timeline"][0]["ext"]
    ext["proposals"] = [
        {
            "proposal_id": "source-proposal",
            "title": "Remove redundant scheduler work.",
            "reasoning": "Profiling shows redundant scheduler work on every request.",
            "source_ref": "https://example.test/pr/1",
        }
    ]
    ext["attempts"] = [
        {
            "attempt_id": "source-attempt",
            "arm": "source",
            "proposal_ref": "source-proposal",
            "outcome": "REVERT",
            "ts": "2026-09-17T12:10:00Z",
            "patch_path": "artifacts/change.patch",
            "target_files": ["scheduler.py"],
            "measurement": {
                "before_tput": 800.0,
                "after_tput": 790.0,
                "gain_pct": -1.25,
            },
            "measured_against": {
                "extra_server_args": "--base",
                "extra_envs": {"BASE": "1"},
            },
        }
    ]

    projected, skipped = experience_v1._project_all(tmp_path, value)

    assert skipped == []
    assert projected[0].change_family == "source_patch"
    assert projected[0].resource_refs == ("artifacts/change.patch",)
    assert len(projected[0].change_fingerprint) == 64
    assert json.loads(projected[0].change_content)["patches"][0]["content"].startswith("diff --git")


def test_review_rejects_source_patch_too_large_for_record_eval(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "artifacts" / "change.patch"
    patch.parent.mkdir()
    patch.write_text("x" * (experience_v1._MAX_PATCH_BYTES + 1))
    value = breakdown()
    ext = value["timeline"][0]["ext"]
    ext["proposals"] = [
        {
            "proposal_id": "source-proposal",
            "title": "Large source change.",
            "reasoning": "Profiling suggests replacing a large generated implementation.",
        }
    ]
    ext["attempts"] = [
        {
            "attempt_id": "source-attempt",
            "arm": "source",
            "proposal_ref": "source-proposal",
            "outcome": "REVERT",
            "ts": "2026-09-17T12:10:00Z",
            "patch_path": "artifacts/change.patch",
            "measurement": {"before_tput": 800.0, "after_tput": 790.0},
            "measured_against": {
                "extra_server_args": "--base",
                "extra_envs": {},
            },
        }
    ]

    _, skipped = experience_v1._project_all(tmp_path, value)

    assert skipped == [
        {
            "attempt_id": "source-attempt",
            "reason": "source attempt has no durable patch material",
        }
    ]


def test_publish_maps_ready_attempts_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kb = FakeKB()
    monkeypatch.setenv("HYPERLOOM_KB_URL", "https://kb.example")
    monkeypatch.setattr(experience_v1, "import_module", lambda _name: fake_module(kb))

    receipt = experience_v1.publish_framework_experiences(tmp_path, breakdown())

    assert receipt["selected"] == 2
    assert receipt["published"] == 2
    assert receipt["errors"] == []
    assert len(kb.begin_calls) == 2
    keep = next(session for session in kb.sessions.values() if session.completions[0]["outcome"].decision == "keep")
    keep_begin = next(call for call in kb.begin_calls if call["baseline_value"] == 800.0)
    assert any(item.startswith("materialized_baseline_configuration=") for item in keep_begin["preconditions"])
    assert keep.decisions[0]["rendered_refs"] == ()
    assert keep.decisions[0]["change"].identity["change_family"] == "config_variant"
    assert keep_begin["provenance"].extra["reasoning_origin"] == "action_payload.reasoning"
    assert keep_begin["provenance"].extra["action_ref"] == "proposal-1"
    assert keep.completions[0]["outcome"].constraints == (FakeConstraint("accuracy", True, 0.82),)
    report = json.loads((tmp_path / "reports" / "experience_v1_publish.json").read_text())
    assert report["source"] == "session_breakdown.timeline[type=framework_agent].ext.attempts"


def test_publish_preserves_kb_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = breakdown()
    proposal = value["timeline"][0]["ext"]["proposals"][0]
    proposal["kb_read_id"] = "read-1"
    proposal["rendered_refs"] = [
        {
            "id": "exp-00000000000000000000000000000001",
            "purpose": "representative",
        }
    ]
    kb = FakeKB()
    monkeypatch.setenv("HYPERLOOM_KB_URL", "https://kb.example")
    monkeypatch.setattr(experience_v1, "import_module", lambda _name: fake_module(kb))

    experience_v1.publish_framework_experiences(tmp_path, value)

    keep = next(session for session in kb.sessions.values() if session.completions[0]["outcome"].decision == "keep")
    assert keep.decisions[0]["rendered_refs"] == (
        FakeRenderedRef(
            "exp-00000000000000000000000000000001",
            "representative",
        ),
    )
    keep_begin = next(call for call in kb.begin_calls if call["baseline_value"] == 800.0)
    assert keep_begin["provenance"].extra["kb_read_id"] == "read-1"


def test_publish_receipt_distinguishes_remote_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kb = FakeKB("spooled")
    monkeypatch.setenv("HYPERLOOM_KB_URL", "https://kb.example")
    monkeypatch.setattr(experience_v1, "import_module", lambda _name: fake_module(kb))

    receipt = experience_v1.publish_framework_experiences(tmp_path, breakdown())

    assert receipt["selected"] == 2
    assert receipt["published"] == 0
    assert receipt["spooled"] == 2
    assert {item["status"] for item in receipt["experiences"]} == {"spooled"}


def test_baseline_fingerprint_ignores_unordered_map_and_set_order(
    tmp_path: Path,
) -> None:
    first = breakdown()
    second = breakdown()
    first_baseline = first["timeline"][0]["ext"]["attempts"][0]["measured_against"]
    second_baseline = second["timeline"][0]["ext"]["attempts"][0]["measured_against"]
    first_baseline.update(
        {
            "extra_envs": {"Z_TUNE": "2", "A_TUNE": "1"},
            "remove_args": ["--z", "--a", "--z"],
            "unset_envs": ["Z_ENV", "A_ENV", "Z_ENV"],
        }
    )
    second_baseline.update(
        {
            "extra_envs": {"A_TUNE": "1", "Z_TUNE": "2"},
            "remove_args": ["--a", "--z"],
            "unset_envs": ["A_ENV", "Z_ENV"],
        }
    )

    first_projected, _ = experience_v1._project_all(tmp_path, first)
    second_projected, _ = experience_v1._project_all(tmp_path, second)

    assert first_projected[0].baseline_configuration == second_projected[0].baseline_configuration
    assert first_projected[0].baseline_identity == second_projected[0].baseline_identity


def test_baseline_material_rejects_credential_envs(tmp_path: Path) -> None:
    value = breakdown()
    attempt = value["timeline"][0]["ext"]["attempts"][0]
    attempt["measured_against"]["extra_envs"]["ANTHROPIC_API_KEY"] = "not-safe"

    _, skipped = experience_v1._project_all(tmp_path, value)

    assert {
        "attempt_id": "attempt-keep",
        "reason": "measured-against environment contains unsafe key: ANTHROPIC_API_KEY",
    } in skipped


def test_disabled_publisher_does_not_import_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    monkeypatch.setattr(
        experience_v1,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("unexpected import")),
    )

    assert experience_v1.publish_framework_experiences("/tmp", breakdown())["enabled"] is False


def test_breakdown_write_triggers_optional_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = breakdown()
    seen: list[tuple[Path, dict]] = []
    monkeypatch.setattr(exporter, "build", lambda _path: value)
    monkeypatch.setattr(
        experience_v1,
        "publish_framework_experiences",
        lambda path, document: seen.append((Path(path), document)) or {},
    )

    target = exporter.write_breakdown_json(tmp_path)

    assert target.exists()
    assert seen == [(tmp_path.resolve(), value)]
