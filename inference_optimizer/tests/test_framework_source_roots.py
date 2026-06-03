"""Framework source-root resolution + prompt injection tests."""

from __future__ import annotations



from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.framework_paths import (
    probe_framework_source_roots_for_env,
    resolve_source_file_allowlist,
    resolve_sglang_server_args_path,
    resolve_vllm_arg_utils_path,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


def test_resolve_source_file_allowlist_unions_env_override(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "/custom/vllm/:/extra/pkg/",
    )
    roots = resolve_source_file_allowlist()
    assert "/sgl-workspace/vllm/" in roots
    assert "/custom/vllm/" in roots
    assert "/extra/pkg/" in roots


def test_find_spec_fallback_returns_note_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VLLM_ARG_UTILS", "")
    path, note = resolve_sglang_server_args_path()
    assert path.name == "server_args.py"
    assert "not found" in note.lower() or str(path) in note
    vpath, vnote = resolve_vllm_arg_utils_path()
    assert vpath.name == "arg_utils.py"
    assert vnote


def test_prompt_renders_framework_source_roots(registry=None):
    registry = registry or ActionRegistry().load()
    custom = ("/custom/sglang/", "/opt/venv/lib/python3.12/site-packages/vllm/")
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        max_minutes=60,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
        framework_source_roots=custom,
    )
    assert "framework_source_roots:" in text
    assert "/custom/sglang/" in text
    assert "site-packages/vllm/" in text


def test_probe_framework_source_roots_includes_defaults(tmp_path, monkeypatch):
    ws = tmp_path / "sgl-workspace" / "sglang"
    ws.mkdir(parents=True)
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.framework_paths._DEFAULT_SOURCE_ROOTS",
        (str(ws) + "/",),
    )
    out = probe_framework_source_roots_for_env()
    assert str(ws) in out or (str(ws) + "/") in out
