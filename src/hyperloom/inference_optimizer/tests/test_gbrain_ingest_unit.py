# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the GBrain Recipe page codec."""

from __future__ import annotations

import json

import yaml

from hyperloom.orchestrator.knowledge.recipe_kb import gbrain_ingest as gi
from hyperloom.orchestrator.knowledge.recipe_kb.replay_bundle import (
    refresh_bundle_digest,
    validate_replay_bundle,
)


# -- _emit_yaml ------------------------------------------------------------
def test_emit_yaml_shapes() -> None:
    out = gi._emit_yaml(
        {
            "empty_list": [],
            "empty_map": {},
            "nested": {"a": 1},
            "scalar_list": ["x", "y"],
            "flat": "v",
        }
    )
    assert "empty_list: []" in out
    assert "empty_map: {}" in out
    assert "- x" in out and "- y" in out
    assert "flat: v" in out


# -- recipe_to_page --------------------------------------------------------
def test_recipe_to_page_none_without_canonical() -> None:
    assert gi.recipe_to_page({"model": "m"}) is None


def test_recipe_to_page_emits_fingerprint_and_negatives(monkeypatch) -> None:
    slug, content = gi.recipe_to_page(
        {
            "canonical_id": "sglang:qwen:mi300x:bf16:v1",
            "model": "qwen",
            "hardware": "mi300x",
            "framework": "sglang",
            "best_config": {"extra_server_args": "--tp 1", "extra_envs": {"A": "1"}},
            "best_throughput": 1000.0,
            "what_worked": ["aiter"],
            "pitfalls": [{"knob": "x"}],
            "stack_fingerprint": {"rocm": "6.2", "aiter": "abc"},
        }
    )
    assert slug.startswith("hyperloom-recipe-kb/")
    assert "stack_fingerprint" in content
    assert "what_worked" in content
    assert "kind:recipe" in content


def test_recipe_json_strips_secrets_and_internal_paths() -> None:
    _slug, content = gi.recipe_to_page(
        {
            "canonical_id": "inference:qwen:mi300x:sglang:qwen:qwen:1.0:fp8",
            "model": "qwen",
            "hardware": "mi300x",
            "best_config": {
                "extra_server_args": (
                    "--tp 8 --token top-secret --token-budget 4096 --tokenizer /workspace/replay/tokenizer"
                ),
                "extra_envs": {
                    "ROCM_VERSION": "7.0",
                    "AITER_CONFIG": "/workspace/replay/tuned.csv",
                    "API_TOKEN": "env-secret",
                },
            },
            "provenance": {
                "generator": "close",
                "authorization": "Bearer provenance-secret",
                "session_path": "/workspace/hyperloom/sessions/s1",
                "details": {"safe": "kept", "password": "nested-secret"},
            },
            "evidence_refs": [
                {
                    "kind": "report",
                    "url": "https://reports.example/safe",
                    "local_path": "/workspace/hyperloom/report.json",
                }
            ],
            "safe_metadata": {
                "note": "shareable",
                "workspace": "/home/operator/private",
                "references": ["paper", "/tmp/session/private.json"],
            },
        }
    )

    frontmatter = yaml.safe_load(content.split("---", 2)[1])
    decoded = json.loads(frontmatter["attrs"]["recipe_json"])
    assert decoded["best_config"]["extra_server_args"] == "--tp 8 --token-budget 4096"
    assert decoded["best_config"]["extra_envs"] == {
        "ROCM_VERSION": "7.0",
    }
    assert decoded["provenance"] == {"generator": "close", "details": {"safe": "kept"}}
    assert decoded["evidence_refs"] == [{"kind": "report", "url": "https://reports.example/safe"}]
    assert decoded["safe_metadata"] == {"note": "shareable", "references": ["paper"]}
    for secret_or_path in (
        "top-secret",
        "env-secret",
        "provenance-secret",
        "nested-secret",
        "/workspace/hyperloom/sessions/",
        "/workspace/hyperloom/report.json",
        "/home/operator/",
        "/tmp/session/",
    ):
        assert secret_or_path not in content


def test_replay_argv_is_sanitized_and_bundle_is_resigned_fail_closed() -> None:
    bundle = refresh_bundle_digest(
        {
            "schema_version": 1,
            "replayable": True,
            "reason": "",
            "config": {
                "argv": [
                    "--tp",
                    "8",
                    "--api-key",
                    "top-secret",
                    "--tokenizer",
                    "/workspace/private/tokenizer",
                ],
                "extra_envs": {
                    "USE_AITER": "1",
                    "HF_TOKEN": "env-secret",
                },
            },
            "source_artifacts": [],
            "measurement": {
                "optimized_throughput": 1000.0,
                "measured_with_complete_bundle": True,
            },
        }
    )
    _slug, content = gi.recipe_to_page(
        {
            "canonical_id": "inference:qwen:mi300x:vllm:qwen:qwen:1.0:fp8",
            "model": "qwen",
            "hardware": "mi300x",
            "framework_name": "vllm",
            "best_config": {
                "argv": bundle["config"]["argv"],
                "extra_envs": bundle["config"]["extra_envs"],
            },
            "best_throughput": 1000.0,
            "replay_bundle": bundle,
        }
    )
    frontmatter = yaml.safe_load(content.split("---", 2)[1])
    decoded = json.loads(frontmatter["attrs"]["recipe_json"])
    cleaned = decoded["replay_bundle"]
    assert cleaned["config"]["argv"] == ["--tp", "8"]
    assert cleaned["config"]["extra_envs"] == {"USE_AITER": "1"}
    assert cleaned["replayable"] is False
    assert cleaned["reason"] == "gbrain_replay_data_sanitized"
    assert cleaned["measurement"]["measured_with_complete_bundle"] is False
    checked = validate_replay_bundle(cleaned)
    assert checked["reason"] == "gbrain_replay_data_sanitized"
    for forbidden in ("top-secret", "env-secret", "/workspace/private"):
        assert forbidden not in content
