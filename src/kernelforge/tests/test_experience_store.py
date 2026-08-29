"""Local/remote experience-store configuration and integration tests."""

from __future__ import annotations


import pytest
from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.config import Config
from kernelforge.knowledge.experience_reader import read_best_solution
from kernelforge.knowledge.experience_sink import write_run_experience
from kernelforge.knowledge.experience_store import (
    KnowledgeConfig,
    KnowledgeStoreMode,
    knowledge_config_from_runtime,
)


def test_default_mode_is_local_and_root_uses_user_data_path(tmp_path):
    config = KnowledgeConfig.from_env({"USER_DATA_PATH": str(tmp_path)})

    assert config.mode is KnowledgeStoreMode.LOCAL
    assert config.local_root == tmp_path / "knowledge"
    assert config.experience_root == (tmp_path / "knowledge" / "kernelforge" / "experiences")
    assert config.gbrain_base_url == ""
    assert config.gbrain_token == ""


def test_default_root_without_user_data_path_uses_hyperloom_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = KnowledgeConfig.from_env({})

    assert config.local_root == tmp_path / ".cache" / "hyperloom" / "knowledge"


def test_explicit_local_ignores_ambient_remote_credentials(tmp_path):
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KNOWLEDGE_LOCAL_ROOT": str(tmp_path),
            "GBRAIN_BASE_URL": "https://ambient.invalid",
            "GBRAIN_TOKEN": "ambient-secret",
        }
    )

    # Blanked, not merely unused: a later reader of this config cannot reach the
    # network with credentials that are not there.
    assert config.mode is KnowledgeStoreMode.LOCAL
    assert config.gbrain_base_url == ""
    assert config.gbrain_token == ""
    assert config.kb_store_url == ""
    assert config.kb_store_token == ""


@pytest.mark.parametrize("mode", ["", "hybrid", "LOCAL", "LOCAL_REMOTE"])
def test_unknown_mode_fails_strict_validation(mode):
    with pytest.raises(ValueError, match="KNOWLEDGE_STORE_MODE"):
        KnowledgeConfig.from_env({"KNOWLEDGE_STORE_MODE": mode})


@pytest.mark.parametrize(
    "env",
    [
        {"KNOWLEDGE_STORE_MODE": "remote"},
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "GBRAIN_BASE_URL": "https://gbrain",
        },
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "GBRAIN_TOKEN": "token",
        },
    ],
)
def test_remote_requires_both_gbrain_values(env):
    with pytest.raises(ValueError, match="requires"):
        KnowledgeConfig.from_env(env)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_local_root_is_rejected_rather_than_defaulted(blank):
    with pytest.raises(ValueError, match="KNOWLEDGE_LOCAL_ROOT"):
        KnowledgeConfig.from_env({"KNOWLEDGE_LOCAL_ROOT": blank})


def test_blank_local_root_override_is_rejected_too():
    with pytest.raises(ValueError, match="KNOWLEDGE_LOCAL_ROOT"):
        KnowledgeConfig.from_env({}, local_root="   ")


def test_an_unknown_remote_backend_is_a_programming_error():
    with pytest.raises(ValueError, match="remote_backend must be"):
        KnowledgeConfig.from_env({}, remote_backend="gbrian")


def test_a_runtime_config_without_knowledge_falls_back_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("KNOWLEDGE_STORE_MODE", raising=False)
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))

    config = knowledge_config_from_runtime(object())

    assert config.mode is KnowledgeStoreMode.LOCAL
    assert config.local_root == tmp_path


def test_sink_reader_end_to_end_local_warm_start(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "vllm" / "ops" / "local_kernel.py"
    kernel.parent.mkdir(parents=True)
    source = "@triton.jit\ndef local_kernel(x):\n    return x\n"
    kernel.write_text(source)
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="local",
        local_root=tmp_path / "knowledge",
    )
    producer_config = Config(workspace=str(workspace), knowledge_config=knowledge, gpu_type="mi355x")

    written = write_run_experience(
        config=producer_config,
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=source,
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id="local-run",
        baseline_wall_ms=10.0,
        best_wall_ms=5.0,
        mean_case_speedup=2.0,
        cumulative_diff=(
            "diff --git a/vllm/ops/local_kernel.py b/vllm/ops/local_kernel.py\n"
            "--- a/vllm/ops/local_kernel.py\n"
            "+++ b/vllm/ops/local_kernel.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
        digest="local warm start",
        framework="vllm",
        summary_override={
            "category": "others",
            "strategy": "use local tiles",
            "recipe": "Increase the tile.",
            "lessons": "Persist the winner.",
        },
    )

    consumer_config = Config(workspace=str(workspace), knowledge_config=knowledge, gpu_type="mi355x")
    solution = read_best_solution(
        config=consumer_config,
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=source,
        kernel_backend="triton",
        framework="vllm",
    )

    assert written["written"] is True
    assert solution is not None
    assert solution["solution_slug"] == written["solution"]
    assert solution["strategy"] == "use local tiles"
    assert solution["speedup"] == 2.0
    # The GPU is part of the address: a run on another card resolves elsewhere
    # rather than reading this record and filtering it out afterwards.
    # ``local_kernel`` normalizes to ``local`` because the operator name drops
    # its ``_kernel`` suffix.
    assert written["kernel"].startswith("kernel:forge-loop:local:vllm:")
    assert written["kernel"].endswith(":triton:mi355x")
    assert solution["patch_content"].endswith("@@ -1 +1 @@\n-old\n+new\n")


def test_forge_loop_rejects_invalid_remote_config_before_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "must-not-be-created"
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--kernel",
            "k.py",
            "--driver",
            "d.py",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code != 0
    assert "KNOWLEDGE_STORE_MODE=remote requires" in result.output
    assert not workspace.exists()
