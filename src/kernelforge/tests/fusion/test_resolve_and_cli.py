# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for framework source-file resolution and the CLI end-to-end."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError

from kernelforge.fusion import discover as discover_module
from kernelforge.fusion import author as author_module
from kernelforge.fusion import command as cli_module
from kernelforge.fusion.author import (
    AUTHOR_RC_SAFETY,
    run_author,
)
from kernelforge.fusion.command import (
    _create_agent_backend,
    _framework_repo_root,
    _package_root,
    _resolve_agent_choice,
    _reset_fusion_source,
    _snapshot_fusion_source,
    main,
)
from kernelforge.fusion.emit import export_artifacts
from kernelforge.fusion.llm_failure import AUTH, LlmUnavailableError
from kernelforge.fusion.locate import resolve_framework_source_file


_AGENT_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "FORGE_API_KEY",
    "FORGE_AGENT_SANDBOX_MODE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "SAFE_API_KEY",
)


@pytest.fixture
def clean_agent_env(monkeypatch):
    for name in _AGENT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "OPENAI_API_KEY": "openai-key",
            },
            "codex",
        ),
        (
            {
                "ANTHROPIC_BASE_URL": "https://anthropic.example",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            "claude",
        ),
        (
            {
                "OPENAI_BASE_URL": "https://openai.example/v1",
                "OPENAI_API_KEY": "openai-key",
                "ANTHROPIC_BASE_URL": "https://anthropic.example",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            "claude",
        ),
    ],
)
def test_auto_agent_backend_uses_credential_shape(
    clean_agent_env,
    monkeypatch,
    env,
    expected,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    provider, _model = _resolve_agent_choice("auto", None)
    assert provider == expected


def test_auto_agent_backend_rejects_unconfigured_environment(clean_agent_env):
    with pytest.raises(click.UsageError, match="no OpenAI or Anthropic credentials"):
        _resolve_agent_choice("auto", None)


def test_explicit_agent_backend_wins_over_credential_shape(clean_agent_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    provider, model = _resolve_agent_choice("codex", None)
    assert provider == "codex"
    assert model == "gpt-5.6"


def test_explicit_claude_wins_over_openai_credentials(clean_agent_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    provider, model = _resolve_agent_choice("claude", None)
    assert provider == "claude"
    assert model == "claude-opus-5"


def test_agent_model_precedence(clean_agent_env, monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-env")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-env")
    assert _resolve_agent_choice("codex", "gpt-explicit") == ("codex", "gpt-explicit")
    assert _resolve_agent_choice("codex", None) == ("codex", "gpt-env")
    assert _resolve_agent_choice("claude", None) == ("claude", "claude-env")


def test_explicit_backend_disables_cross_provider_fallback(
    clean_agent_env,
    monkeypatch,
):
    captured = {}

    def fake_resolve(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return SimpleNamespace(provider=provider, model=kwargs["model"])

    backend = SimpleNamespace(
        name="codex",
        runtime=SimpleNamespace(provider="codex", model="gpt-explicit"),
    )
    monkeypatch.setattr(cli_module, "resolve_agent_runtime", fake_resolve)
    monkeypatch.setattr(cli_module, "create_registered_backend", lambda runtime: backend)

    assert _create_agent_backend("codex", "gpt-explicit") is backend
    assert captured["fallback_provider"] == ""
    assert captured["sandbox_mode"] == "workspace-write"


@pytest.mark.parametrize("mode", ["workspace-write", "read-only"])
def test_explicit_secure_sandbox_mode_is_wired(
    clean_agent_env,
    monkeypatch,
    mode,
):
    captured = {}

    def fake_resolve(provider, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(provider=provider, model=kwargs["model"], sandbox_mode=mode)

    backend = SimpleNamespace(
        name="codex",
        runtime=SimpleNamespace(provider="codex", model="gpt-explicit", sandbox_mode=mode),
    )
    monkeypatch.setattr(cli_module, "resolve_agent_runtime", fake_resolve)
    monkeypatch.setattr(cli_module, "create_registered_backend", lambda runtime: backend)

    assert _create_agent_backend("codex", "gpt-explicit", mode) is backend
    assert captured["sandbox_mode"] == mode


def test_bypass_is_wired(clean_agent_env, monkeypatch):
    captured = {}

    def fake_resolve(provider, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(provider=provider, model=kwargs["model"], sandbox_mode="bypass")

    backend = SimpleNamespace(
        name="codex",
        runtime=SimpleNamespace(provider="codex", model="gpt-explicit", sandbox_mode="bypass"),
    )
    monkeypatch.setattr(cli_module, "resolve_agent_runtime", fake_resolve)
    monkeypatch.setattr(cli_module, "create_registered_backend", lambda runtime: backend)

    assert _create_agent_backend("codex", "gpt-explicit", "bypass") is backend
    assert captured["sandbox_mode"] == "bypass"


def test_bypass_is_wired_without_retired_external_sandbox_env(clean_agent_env, monkeypatch):
    """forge-fuse bypass no longer gates on HYPERLOOM_CODEX_EXTERNAL_SANDBOX."""
    monkeypatch.delenv("HYPERLOOM_CODEX_EXTERNAL_SANDBOX", raising=False)
    captured = {}

    def fake_resolve(provider, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(provider=provider, model=kwargs["model"], sandbox_mode="bypass")

    backend = SimpleNamespace(
        name="codex",
        runtime=SimpleNamespace(provider="codex", model="gpt-explicit", sandbox_mode="bypass"),
    )
    monkeypatch.setattr(cli_module, "resolve_agent_runtime", fake_resolve)
    monkeypatch.setattr(cli_module, "create_registered_backend", lambda runtime: backend)

    assert _create_agent_backend("codex", "gpt-explicit", "bypass") is backend
    assert captured["sandbox_mode"] == "bypass"


def test_retired_external_sandbox_env_does_not_confirm_bypass(clean_agent_env, monkeypatch):
    """Legacy HYPERLOOM_CODEX_EXTERNAL_SANDBOX=1 must not substitute bypass mode."""
    monkeypatch.setenv("HYPERLOOM_CODEX_EXTERNAL_SANDBOX", "1")
    assert cli_module._resolve_agent_sandbox_mode(None) == "workspace-write"


def test_sandbox_mode_explicit_value_wins_over_environment(clean_agent_env, monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_SANDBOX_MODE", "read-only")
    assert cli_module._resolve_agent_sandbox_mode(None) == "read-only"
    assert cli_module._resolve_agent_sandbox_mode("workspace-write") == "workspace-write"


def test_invalid_sandbox_mode_is_rejected(clean_agent_env):
    with pytest.raises(click.UsageError, match="unsupported agent sandbox mode"):
        cli_module._resolve_agent_sandbox_mode("unconfined")


def test_author_harness_is_staged_inside_worktree_then_published(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    staging = cli_module._author_harness_target(str(repo), out)
    assert Path(staging).resolve().is_relative_to(repo.resolve())

    ready, reason, _deterministic = cli_module._prepare_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
    )
    assert ready, reason
    Path(staging).write_text("print('validated')\n", encoding="utf-8")
    ok, reason = cli_module._finish_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
        author_ok=True,
    )

    assert ok, reason
    assert (out / "kernel_harness.py").read_text(encoding="utf-8") == "print('validated')\n"
    assert not Path(staging).exists()


def test_author_harness_staging_survives_running_the_harness(tmp_path):
    """Let the interpreter's own bytecode cache go with the staging directory.

    Staging exists so the harness can be run from inside the worktree, and
    running it makes the interpreter write __pycache__ beside the module. That
    byproduct then blocked the rmdir, so an authoring turn that had produced a
    working fusion was failed by its own cleanup -- and failed identically on
    every retry, because each retry ran the harness again. Five attempts, ~25
    minutes each, the whole kernel budget.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    staging = cli_module._author_harness_target(str(repo), out)
    ready, reason, _deterministic = cli_module._prepare_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
    )
    assert ready, reason
    Path(staging).write_text("print('validated')\n", encoding="utf-8")
    cache = Path(staging).parent / "__pycache__"
    cache.mkdir()
    (cache / "kernel_harness.cpython-310.pyc").write_bytes(b"\x00bytecode")

    ok, reason = cli_module._finish_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
        author_ok=True,
    )

    assert ok, reason
    assert (out / "kernel_harness.py").read_text(encoding="utf-8") == "print('validated')\n"
    assert not Path(staging).parent.exists()


def test_author_harness_staging_tolerates_another_harness(tmp_path):
    """Share the staging directory without one harness failing another's run.

    The directory is per-repo while the digest in the file name is per-output
    dir, so a second run -- or the author writing its own validation harness --
    leaves a sibling ``kernel_harness_*.py`` behind. That sibling blocked the
    rmdir and turned a wired, 1.81x fusion into AUTHOR FAILED. Deleting it is not
    an option either: a concurrent run may still be using it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    staging = cli_module._author_harness_target(str(repo), out)
    ready, reason, _deterministic = cli_module._prepare_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
    )
    assert ready, reason
    Path(staging).write_text("print('validated')\n", encoding="utf-8")
    sibling = Path(staging).parent / "kernel_harness_c262b47917a2.py"
    sibling.write_text("print('other run')\n", encoding="utf-8")

    ok, reason = cli_module._finish_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
        author_ok=True,
    )

    assert ok, reason
    assert (out / "kernel_harness.py").read_text(encoding="utf-8") == "print('validated')\n"
    assert not Path(staging).exists()
    assert sibling.is_file()


def test_author_harness_staging_still_reports_foreign_leftovers(tmp_path):
    """Name anything else left in staging instead of deleting it broadly."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    staging = cli_module._author_harness_target(str(repo), out)
    ready, reason, _deterministic = cli_module._prepare_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
    )
    assert ready, reason
    Path(staging).write_text("print('validated')\n", encoding="utf-8")
    (Path(staging).parent / "stray.py").write_text("STRAY = True\n", encoding="utf-8")

    ok, reason = cli_module._finish_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
        author_ok=True,
    )

    assert ok is False
    assert "stray.py" in reason
    assert Path(staging).parent.exists()


def test_inherited_harness_staging_detects_author_mutation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    final = out / "kernel_harness.py"
    final.parent.mkdir()
    final.write_text("print('inherited')\n", encoding="utf-8")
    staging = cli_module._author_harness_target(str(repo), out)
    ready, reason, _deterministic = cli_module._prepare_author_harness(
        staging,
        str(final),
        inherited=True,
    )
    assert ready, reason
    Path(staging).write_text("print('mutated')\n", encoding="utf-8")

    ok, reason = cli_module._finish_author_harness(
        staging,
        str(final),
        inherited=True,
        author_ok=True,
    )

    assert ok is False
    assert reason == "author modified the inherited harness"
    assert final.read_text(encoding="utf-8") == "print('inherited')\n"
    assert not Path(staging).exists()


def test_registered_discovery_returns_final_text_without_bare_openai(
    tmp_path,
    monkeypatch,
):
    captured = {}
    source = tmp_path / "model.py"
    original = "def forward(x):\n    return x\n"
    source.write_text(original, encoding="utf-8")

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
            sandbox_mode="bypass",
        )

        async def run(self, spec, usage=None):
            captured["spec"] = spec
            return AgentRunResult(text='[{"name":"fused"}]')

    monkeypatch.setattr(
        discover_module,
        "default_llm_fn",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("registered discovery must not construct a bare OpenAI client")
        ),
    )
    fn = discover_module.registered_agent_llm_fn(
        Backend(),
        model="gpt-test",
        timeout_s=10,
        workdir=str(tmp_path),
        protected_files=[str(source)],
    )

    assert fn("DISCOVERY PROMPT") == '[{"name":"fused"}]'
    spec = captured["spec"]
    assert spec.system_prompt != spec.user_prompt
    assert spec.user_prompt == "DISCOVERY PROMPT"
    assert spec.writable is False
    assert spec.tool_policy.read is True and spec.tool_policy.search is True
    assert spec.tool_policy.write is False and spec.tool_policy.shell is False
    assert spec.protected_globs == ["*"]
    # Discovery tolerates a dirty worktree, but must NOT claim the read-only-resume
    # contract: that flag disqualifies the session from the workspace guard's
    # read-only fast path, which then demands cwd be a git worktree. Discovery's cwd
    # is the framework repo root, routinely a pip install root with no .git, so the
    # guard rejected every LLM discovery against a pip-installed framework.
    assert spec.read_only_resume is False
    assert spec.allow_dirty_baseline is True
    assert WorkspaceGuard.is_read_only_session(spec) is True


def test_registered_discovery_rejects_and_restores_source_edits(tmp_path):
    source = tmp_path / "model.py"
    original = "def forward(x):\n    return x\n"
    source.write_text(original, encoding="utf-8")

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
            sandbox_mode="bypass",
        )

        async def run(self, spec, usage=None):
            source.write_text("MUTATED\n", encoding="utf-8")
            return AgentRunResult(text="[]")

    fn = discover_module.registered_agent_llm_fn(
        Backend(),
        model="gpt-test",
        timeout_s=10,
        workdir=str(tmp_path),
        protected_files=[str(source)],
    )

    with pytest.raises(discover_module.DiscoverySafetyError, match="modified protected source"):
        fn("DISCOVERY PROMPT")
    assert source.read_text(encoding="utf-8") == original


def test_registered_discovery_retries_transient_timeout(tmp_path):
    calls = {"count": 0}

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("temporary request timeout")
            return AgentRunResult(text="[]")

    fn = discover_module.registered_agent_llm_fn(
        Backend(),
        timeout_s=10,
        workdir=str(tmp_path),
        attempts=2,
        sleep=lambda _delay: None,
    )

    assert fn("DISCOVERY PROMPT") == "[]"
    assert calls["count"] == 2


def test_registered_discovery_does_not_retry_auth_failure(tmp_path):
    calls = {"count": 0}

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            calls["count"] += 1
            raise RuntimeError("401 unauthorized")

    fn = discover_module.registered_agent_llm_fn(
        Backend(),
        timeout_s=10,
        workdir=str(tmp_path),
        attempts=3,
        sleep=lambda _delay: None,
    )

    with pytest.raises(LlmUnavailableError) as exc_info:
        fn("DISCOVERY PROMPT")
    assert exc_info.value.kind == AUTH
    assert calls["count"] == 1


def test_registered_discovery_does_not_retry_safety_violation(tmp_path):
    calls = {"count": 0}

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
            sandbox_mode="bypass",
        )

        async def run(self, spec, usage=None):
            calls["count"] += 1
            raise WorkspaceSafetyError("the read-only session changed the workspace")

    fn = discover_module.registered_agent_llm_fn(
        Backend(),
        timeout_s=10,
        workdir=str(tmp_path),
        attempts=5,
        sleep=lambda _delay: None,
    )

    with pytest.raises(
        discover_module.DiscoverySafetyError,
        match="the read-only session changed the workspace",
    ):
        fn("DISCOVERY PROMPT")
    assert calls["count"] == 1


def test_framework_transaction_restores_non_target_claude_shell_edits(tmp_path):
    """A Claude Bash write outside target_files must not survive the transaction."""
    import subprocess

    repo = tmp_path / "framework"
    repo.mkdir()
    source = repo / "model.py"
    unrelated = repo / "unrelated.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")
    unrelated.write_text("UNRELATED = 'baseline'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "init", "-q"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "model.py", "unrelated.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(
            provider="claude",
            model="claude-test",
            sandbox_mode="workspace-write",
        )

        async def run(self, spec, usage=None):
            # This models Claude using its allowed Bash tool instead of Edit.
            source.write_text("SOURCE = 'authored'\n", encoding="utf-8")
            unrelated.write_text(
                "UNRELATED = 'claude-shell-edit'\n",
                encoding="utf-8",
            )
            return AgentRunResult(
                text="AUTHORING_RESULT: completed",
                tool_calls=[("Bash", {"command": "redacted mutation"})],
            )

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-shell-gap.log"),
        backend=Backend(),
        target_files=[str(source)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert unrelated.read_text(encoding="utf-8") == "UNRELATED = 'baseline'\n"
    assert source.read_text(encoding="utf-8") == "SOURCE = 'authored'\n"


def _autoloop_recipe(source: Path):
    """The one recipe the autoloop tests drive a campaign for."""
    from kernelforge.fusion.models import Recipe

    return Recipe(
        pattern_id="residual_add_rmsnorm",
        description="Fuse residual add into RMSNorm",
        env_flag="QWEN3_FUSED",
        source_file=str(source),
        source_hints=["+ residual"],
        fusion_math="y = rmsnorm(x + residual)",
        eager_reference_hint="import RMSNorm",
        shapes={},
        matched_categories=["add", "rmsnorm"],
        trigger_share=0.34,
    )


def _autoloop_result(tmp_path, monkeypatch, *, source: Path, author: bool = True):
    """Run ``_run_fusion_autoloop`` end to end and return its LoopResult.

    Harness authoring is stubbed out; what these tests are about starts after it.
    """
    monkeypatch.setattr(cli_module, "_author_baseline_harness", lambda *a, **k: (True, ""))
    return cli_module._run_fusion_autoloop(
        [_autoloop_recipe(source)],
        framework="vllm",
        out=tmp_path,
        repo_root=str(source.parent),
        author=author,
        gpu="0",
        llm_model="m",
        target_speedup=1.03,
        ab_isl=512,
        ab_osl=64,
        max_turns=1,
        agent_factory=object,
    )


def _autoloop_campaign_fn(tmp_path, monkeypatch, *, source: Path, author: bool = True, experience: str = ""):
    """Drive ``_run_fusion_autoloop`` for one recipe and report what it did.

    The campaign_fn is invoked INSIDE the loop rather than handed back, because
    it restores the shadow repository before every campaign and the autoloop
    disposes of that repository as soon as the loop returns.
    """
    captured: dict[str, object] = {}

    def fake_run_fusion_loop(recipes, *, framework, campaign_fn, config):
        captured["config"] = config
        captured["verdict"] = campaign_fn(recipes[0], experience)
        return cli_module.LoopResult(kept=False, best=None, best_recipe=None)

    monkeypatch.setattr(cli_module, "run_fusion_loop", fake_run_fusion_loop)

    recipe = _autoloop_recipe(source)
    _autoloop_result(tmp_path, monkeypatch, source=source, author=author)
    return recipe, captured


def test_autoloop_runs_one_forge_loop_campaign_per_recipe(tmp_path, monkeypatch):
    """Repeated authoring belongs to the campaign, not to a second loop here."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    seen: list[dict] = []

    def fake_campaign(recipe, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            result=SimpleNamespace(kept=True, kernel_speedup=1.2),
            experiment_id="exp-1",
        )

    monkeypatch.setattr(cli_module, "run_recipe_campaign", fake_campaign)
    _autoloop_campaign_fn(tmp_path, monkeypatch, source=source, experience="prior experience")

    assert len(seen) == 1
    # Harness path is now per-recipe: kernel_harness_{stem}.py
    assert "kernel_harness_" in seen[0]["harness_path"]
    assert seen[0]["harness_path"].endswith(".py")
    assert seen[0]["experience"] == "prior experience"
    assert seen[0]["target_speedup"] == 1.03
    # The shadow repository is visible through the .git pointer file in the tree,
    # so shadow_env may be empty (pointer case) or carry GIT_DIR (env fallback).
    # Either way, the workspace is the shadow root.
    shadow_env = seen[0]["shadow_env"]
    assert isinstance(shadow_env, dict)
    # The author is given a module that is already tracked, not left to create
    # one that ``git add -u`` could never commit.
    assert seen[0]["fused_module"].endswith("qwen3_fused_residual_add_rmsnorm.py")


def test_autoloop_gives_the_loop_a_git_workspace(tmp_path, monkeypatch):
    """forge-loop keeps and reverts with git, which a pip install does not have."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "run_recipe_campaign",
        lambda *a, **k: SimpleNamespace(result=SimpleNamespace(kept=False, kernel_speedup=None), experiment_id=""),
    )
    _recipe, captured = _autoloop_campaign_fn(tmp_path, monkeypatch, source=source)

    assert "verdict" in captured, "the loop must have been reached"
    # The repository lives under the run's output directory, never in the
    # framework tree, and is disposed of once the loop returns.
    assert not (model_dir / ".git").exists()
    assert not (model_dir / ".gitignore").exists()
    assert not (tmp_path / "shadow.git").exists()
    # Nor is the loop's own campaign state left in the framework install.
    assert not (model_dir / "forge_experiments").exists()


def test_autoloop_restores_the_baseline_before_every_campaign(tmp_path, monkeypatch):
    """A later recipe must not measure its baseline on the previous one's code."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    on_entry: list[tuple[str, str]] = []

    def fake_campaign(recipe, **kwargs):
        fused = Path(kwargs["fused_module"])
        on_entry.append((source.read_text(encoding="utf-8"), fused.read_text(encoding="utf-8")))
        # Stand in for what the loop commits on its own smaller margin, which
        # fusion then judges a miss.
        fused.write_text("FUSED = 1\n", encoding="utf-8")
        source.write_text("SOURCE = 'fused'\n", encoding="utf-8")
        return SimpleNamespace(
            result=SimpleNamespace(kept=False, kernel_speedup=1.01),
            experiment_id="exp-1",
        )

    def fake_run_fusion_loop(recipes, *, framework, campaign_fn, config):
        # Two recipes in sequence is the only shape where the leak was visible:
        # the loop returns the instant one KEEPs.
        campaign_fn(recipes[0], "")
        campaign_fn(recipes[0], "")
        return cli_module.LoopResult(kept=False, best=None, best_recipe=None)

    monkeypatch.setattr(cli_module, "run_recipe_campaign", fake_campaign)
    monkeypatch.setattr(cli_module, "run_fusion_loop", fake_run_fusion_loop)

    _autoloop_result(tmp_path, monkeypatch, source=source)

    # Both campaigns opened on the pristine tree. Without the reset the second
    # would have measured its baseline on the first one's rejected fusion.
    assert on_entry == [("SOURCE = 'baseline'\n", ""), ("SOURCE = 'baseline'\n", "")]


def test_autoloop_refuses_to_run_without_the_harness_the_loop_benches(tmp_path, monkeypatch):
    """The harness anchors the speedup benchmark; failure aborts the whole run.

    Harness authoring happens inside campaign_fn (after reset_to_base, before
    run_recipe_campaign) so the harness bench runs on the unfused baseline.
    A failed author raises FusionAbort, which _run_fusion_autoloop catches and
    converts into termination_reason="harness_author_failed" without recording
    a per-recipe history entry.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    started: list[str] = []
    monkeypatch.setattr(cli_module, "run_recipe_campaign", lambda *a, **k: started.append("campaign"))
    monkeypatch.setattr(cli_module, "_author_baseline_harness", lambda *a, **k: (False, "author exited 3"))

    result = cli_module._run_fusion_autoloop(
        [_autoloop_recipe(source)],
        framework="vllm",
        out=tmp_path,
        repo_root=str(source.parent),
        author=True,
        gpu="0",
        llm_model="m",
        target_speedup=1.03,
        ab_isl=512,
        ab_osl=64,
        max_turns=1,
        agent_factory=object,
    )

    assert started == []
    assert result.kept is False
    assert result.termination_reason == "harness_author_failed"
    # The shadow is disposed of on this path too, not left in the framework.
    assert not (tmp_path / "shadow.git").exists()


def test_autoloop_fails_a_recipe_whose_baseline_could_not_be_restored(tmp_path, monkeypatch):
    """Running anyway is the measurement the restore exists to prevent."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    started: list[str] = []
    monkeypatch.setattr(cli_module, "run_recipe_campaign", lambda *a, **k: started.append("campaign"))
    from kernelforge.fusion.shadow_repo import ShadowRepo

    monkeypatch.setattr(ShadowRepo, "reset_to_base", lambda _self: False, raising=True)

    _recipe, captured = _autoloop_campaign_fn(tmp_path, monkeypatch, source=source)

    assert started == [], "a campaign ran on a tree that could not be restored"
    verdict = captured["verdict"]
    assert verdict.kept is False
    assert "could not restore the unfused baseline" in verdict.note


def test_autoloop_clears_the_loop_state_that_would_reject_the_next_recipe(tmp_path, monkeypatch):
    """The loop anchors its campaign store to the workspace, not to --experiments-dir.

    It refuses to start where a campaign already left state, so without this the
    second recipe is rejected outright rather than run.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    seen_state: list[bool] = []

    def fake_campaign(recipe, *, workspace, **_kwargs):
        state = Path(workspace) / "forge_experiments"
        seen_state.append(state.exists())
        (state / "candidates").mkdir(parents=True, exist_ok=True)
        (state / "run_state.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            result=SimpleNamespace(kept=False, kernel_speedup=1.01),
            experiment_id="exp-1",
        )

    def fake_run_fusion_loop(recipes, *, framework, campaign_fn, config):
        campaign_fn(recipes[0], "")
        campaign_fn(recipes[0], "")
        return cli_module.LoopResult(kept=False, best=None, best_recipe=None)

    monkeypatch.setattr(cli_module, "run_recipe_campaign", fake_campaign)
    monkeypatch.setattr(cli_module, "run_fusion_loop", fake_run_fusion_loop)

    _autoloop_result(tmp_path, monkeypatch, source=source)

    assert seen_state == [False, False]
    assert not (model_dir / "forge_experiments").exists()


def test_autoloop_records_which_forge_loop_run_answered_each_recipe(tmp_path, monkeypatch):
    """A manifest entry with no experiment id cannot be investigated."""
    from kernelforge.fusion.loop import LoopIteration

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    def fake_campaign(recipe, **_kwargs):
        return SimpleNamespace(
            result=SimpleNamespace(kept=True, kernel_speedup=1.2),
            experiment_id="exp-42",
        )

    def fake_run_fusion_loop(recipes, *, framework, campaign_fn, config):
        recipe = recipes[0]
        campaign_fn(recipe, "")
        return cli_module.LoopResult(
            kept=True,
            best=None,
            best_recipe=recipe,
            history=[
                LoopIteration(
                    recipe_index=0,
                    attempt=1,
                    pattern_id=recipe.pattern_id,
                    env_flag=recipe.env_flag,
                    kept=True,
                    correctness_passed=True,
                    kernel_speedup=1.2,
                    max_abs_err=None,
                    note="",
                )
            ],
        )

    monkeypatch.setattr(cli_module, "run_recipe_campaign", fake_campaign)
    monkeypatch.setattr(cli_module, "run_fusion_loop", fake_run_fusion_loop)
    monkeypatch.setattr(cli_module, "apply_serving_gate", lambda *a, **k: None)

    result = _autoloop_result(tmp_path, monkeypatch, source=source)

    assert result.history[0].experiment_id == "exp-42"
    assert result.to_dict()["best_experiment_id"] == "exp-42"


def test_autoloop_without_author_leaves_the_fusion_it_is_scoring(tmp_path, monkeypatch):
    """--no-author scores what is on disk, so nothing may be staged over it.

    There is no campaign to keep or revert either, so the shadow repository and
    its empty placeholders have no reason to exist on this path.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")
    fused = model_dir / "qwen3_fused_residual_add_rmsnorm.py"
    fused.write_text("FUSED = 1\n", encoding="utf-8")

    monkeypatch.setattr(cli_module, "validate_recipe", lambda *a, **k: SimpleNamespace(kept=False))
    monkeypatch.setattr(cli_module, "HarnessKernelRunner", lambda **_k: object())
    monkeypatch.setattr(
        cli_module,
        "run_recipe_campaign",
        lambda *a, **k: None,
    )

    _autoloop_campaign_fn(tmp_path, monkeypatch, source=source, author=False)

    assert fused.read_text(encoding="utf-8") == "FUSED = 1\n"
    assert not (tmp_path / "shadow.git").exists()


def test_autoloop_without_author_scores_the_source_as_it_stands(tmp_path, monkeypatch):
    """--no-author must not start a campaign; there is nothing to author."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    source = model_dir / "qwen3.py"
    source.write_text("SOURCE = 'baseline'\n", encoding="utf-8")

    started: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "run_recipe_campaign",
        lambda *a, **k: started.append("campaign"),
    )
    monkeypatch.setattr(cli_module, "validate_recipe", lambda *a, **k: SimpleNamespace(kept=False))
    monkeypatch.setattr(cli_module, "HarnessKernelRunner", lambda **_k: object())

    _autoloop_campaign_fn(tmp_path, monkeypatch, source=source, author=False)

    assert started == []


def test_prepare_author_harness_marks_an_occupied_staging_target_deterministic(
    tmp_path,
):
    """The discriminator has to come from the function that knows the reason."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    staging = cli_module._author_harness_target(str(repo), out)
    Path(staging).parent.mkdir(parents=True, exist_ok=True)
    Path(staging).write_text("print('left behind')\n", encoding="utf-8")

    ready, reason, deterministic = cli_module._prepare_author_harness(
        staging,
        str(out / "kernel_harness.py"),
        inherited=False,
    )

    assert ready is False
    assert "already exists" in reason
    assert deterministic is True


class TestAgentTimeoutSetting:
    def test_default_is_two_hours(self, monkeypatch):
        monkeypatch.delenv("FORGE_FUSION_AGENT_TIMEOUT_SEC", raising=False)
        assert cli_module._agent_timeout_sec() == 7200

    def test_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("FORGE_FUSION_AGENT_TIMEOUT_SEC", "900")
        assert cli_module._agent_timeout_sec() == 900

    @pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
    def test_unusable_override_fails_loudly(self, monkeypatch, value):
        """A silently ignored budget would leave the two-hour default in place."""
        monkeypatch.setenv("FORGE_FUSION_AGENT_TIMEOUT_SEC", value)
        with pytest.raises(click.UsageError, match="FORGE_FUSION_AGENT_TIMEOUT_SEC"):
            cli_module._agent_timeout_sec()


def _venv_in_git_layout(tmp_path):
    """A git project whose UNTRACKED pip vLLM lives under .venv/site-packages.

    Returns (site_packages_root, source_file). The source file is not git-tracked.
    """
    import subprocess

    project = tmp_path / "myproj"
    site = project / ".venv" / "lib" / "python3.11" / "site-packages"
    d = site / "vllm" / "model_executor" / "models"
    d.mkdir(parents=True)
    for p in (site / "vllm", site / "vllm" / "model_executor", d):
        (p / "__init__.py").write_text("", encoding="utf-8")
    src = d / "qwen3.py"
    src.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True, capture_output=True, text=True)
    (project / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", ".gitignore"], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return site, src


class TestNonGitRepoRoot:
    def test_package_root_walks_up_to_install_dir(self, tmp_path):
        # <site>/pkg/sub/mod.py with __init__.py up the package -> root == <site>
        site = tmp_path / "site"
        d = site / "pkg" / "sub"
        d.mkdir(parents=True)
        for p in (site / "pkg", d):
            (p / "__init__.py").write_text("", encoding="utf-8")
        src = d / "mod.py"
        src.write_text("x = 1\n", encoding="utf-8")
        assert _package_root(str(src)) == str(site.resolve())

    def test_framework_repo_root_nongit_falls_back_to_package_root(self, tmp_path):
        # A non-git framework (pip install) must yield a NON-EMPTY root so the
        # caller runs the snapshot-based export instead of skipping it.
        site = tmp_path / "site"
        d = site / "vllm" / "model_executor" / "models"
        d.mkdir(parents=True)
        for p in (site / "vllm", site / "vllm" / "model_executor", d):
            (p / "__init__.py").write_text("", encoding="utf-8")
        src = d / "qwen3.py"
        src.write_text("x = 1\n", encoding="utf-8")
        root = _framework_repo_root(str(src), "")
        assert root == str(site.resolve()), "non-git must fall back to package install root"

    def test_framework_repo_root_venv_inside_git_uses_package_root(self, tmp_path):
        """Repro: a pip framework under a git project's .venv is UNTRACKED.

        `git rev-parse --show-toplevel` returns the project root, but git diff of
        that untracked file is empty -> patch=null. The root must instead be the
        package install dir (site-packages) so export takes the snapshot path and
        emits package-relative paths that apply at site-packages.
        """
        import subprocess

        project = tmp_path / "myproj"
        site = project / ".venv" / "lib" / "python3.11" / "site-packages"
        d = site / "vllm" / "model_executor" / "models"
        d.mkdir(parents=True)
        for p in (site / "vllm", site / "vllm" / "model_executor", d):
            (p / "__init__.py").write_text("", encoding="utf-8")
        src = d / "qwen3.py"
        src.write_text("x = 1\n", encoding="utf-8")
        # git project that does NOT track the venv
        subprocess.run(["git", "-C", str(project), "init", "-q"], check=True, capture_output=True, text=True)
        (project / ".gitignore").write_text(".venv/\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(project), "add", ".gitignore"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "-c",
                "user.email=a@b.c",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "base",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        root = _framework_repo_root(str(src), "")
        assert root == str(site.resolve()), "venv-in-git must use the package root, not the git project toplevel"

    def test_reset_venv_in_git_restores_from_pristine(self, tmp_path):
        """Repro: an UNTRACKED pip source under a git work tree must be reset from the
        pristine snapshot (git checkout is a no-op on untracked files)."""
        site, src = _venv_in_git_layout(tmp_path)
        pristine_text = src.read_text()
        out = tmp_path / "out"
        pdir = _snapshot_fusion_source(str(site), str(src), out)
        assert pdir
        # a failed author attempt leaves a broken edit behind
        src.write_text("SYNTAX ERROR broken edit\n", encoding="utf-8")
        _reset_fusion_source(str(site), str(src), pristine_dir=pdir)
        assert src.read_text() == pristine_text, "untracked venv source must be reverted via pristine snapshot"

    def test_export_venv_in_git_produces_patch(self, tmp_path):
        """End-to-end: venv-in-git layout must still yield a non-empty, package-relative
        patch (not fall into the empty git-diff path)."""
        site, src = _venv_in_git_layout(tmp_path)
        out = tmp_path / "out"
        pdir = _snapshot_fusion_source(str(site), str(src), out)
        src.write_text(
            "import os\nFUSED = os.environ.get('QWEN3_FUSED', '0') == '1'\ndef forward(x):\n    return x\n",
            encoding="utf-8",
        )
        arts = export_artifacts(str(site), str(src), out, pristine_dir=pdir)
        assert arts.patch is not None, "venv-in-git export must still produce a patch"
        rel = "vllm/model_executor/models/qwen3.py"
        patch_text = (out / "fusion.patch").read_text()
        assert f"diff --git a/{rel} b/{rel}" in patch_text
        assert arts.repo_root == str(site.resolve())


class TestResolveFrameworkSourceFile:
    def test_sglang_editable_layout(self, tmp_path):
        # editable checkout: <root>/python/sglang/srt/models/<mt>.py
        mdir = tmp_path / "python" / "sglang" / "srt" / "models"
        mdir.mkdir(parents=True)
        (mdir / "lfm2.py").write_text("# model", encoding="utf-8")
        path, note = resolve_framework_source_file(
            str(tmp_path), "sglang", framework_root=str(tmp_path), model_type="lfm2"
        )
        assert path == str(mdir / "lfm2.py")

    def test_sglang_site_packages_layout(self, tmp_path):
        # site-packages: <root>/sglang/srt/models/<mt>.py (no python/ dir)
        mdir = tmp_path / "sglang" / "srt" / "models"
        mdir.mkdir(parents=True)
        (mdir / "zaya.py").write_text("# model", encoding="utf-8")
        path, note = resolve_framework_source_file(
            str(tmp_path), "sglang", framework_root=str(tmp_path), model_type="zaya"
        )
        assert path == str(mdir / "zaya.py")

    def test_vllm_layout(self, tmp_path):
        mdir = tmp_path / "vllm" / "model_executor" / "models"
        mdir.mkdir(parents=True)
        (mdir / "llama.py").write_text("# model", encoding="utf-8")
        path, note = resolve_framework_source_file(
            str(tmp_path), "vllm", framework_root=str(tmp_path), model_type="llama"
        )
        assert path == str(mdir / "llama.py")

    def test_unresolvable_returns_empty(self, tmp_path):
        path, note = resolve_framework_source_file(
            str(tmp_path), "sglang", framework_root=str(tmp_path), model_type="nope"
        )
        assert path == ""

    def test_unknown_framework_returns_empty(self, tmp_path):
        path, note = resolve_framework_source_file(str(tmp_path), "tensorrt", model_type="x")
        assert path == ""


def _write_trace(path, events, gz=False):
    payload = {"traceEvents": events}
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _launch_bound_events():
    return [
        {"cat": "kernel", "name": "Cijk_gemm", "ts": 0, "dur": 40},
        {"cat": "kernel", "name": "add_rmsnorm_quant_kernel", "ts": 200, "dur": 12},
        {"cat": "kernel", "name": "vectorized_elementwise CUDAFunctor_add", "ts": 400, "dur": 10},
        {"cat": "kernel", "name": "vectorized_elementwise silu", "ts": 600, "dur": 8},
    ]


class TestCliDryRun:
    def test_cli_accepts_explicit_secure_sandbox_mode(self, tmp_path):
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "toylm",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "out"

        result = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--dry-run",
                "--agent-sandbox-mode",
                "read-only",
            ],
        )

        assert result.exit_code == 0, result.output

    def test_dry_run_candidate_writes_manifest(self, tmp_path):
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        # Synthetic model_type -> source unresolved (hermetic, no source filtering).
        (model / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
            encoding="utf-8",
        )
        out = tmp_path / "out"

        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--dry-run",
            ],
        )
        assert res.exit_code == 0, res.output
        manifest = json.loads((out / "fusion_manifest.json").read_text())
        assert manifest["verdict"] == "candidate"
        assert manifest["fusion"]["pattern"] == "residual_add_rmsnorm"
        assert manifest["fusion_candidates"]  # populated by build_manifest
        assert manifest["validation"] is None and manifest["artifacts"] is None

    def test_dry_run_no_opportunity(self, tmp_path):
        trace = tmp_path / "decode.trace.json"
        # GEMM-dominated, GPU busy -> compute-bound -> no opportunity.
        _write_trace(
            trace,
            [
                {"cat": "kernel", "name": "Cijk_gemm", "ts": 0, "dur": 100},
                {"cat": "kernel", "name": "Cijk_gemm2", "ts": 100, "dur": 100},
                {"cat": "kernel", "name": "add_rmsnorm", "ts": 200, "dur": 5},
            ],
        )
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
        out = tmp_path / "out"
        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--dry-run",
            ],
        )
        assert res.exit_code == 0, res.output
        manifest = json.loads((out / "fusion_manifest.json").read_text())
        assert manifest["verdict"] == "no_opportunity"
        assert manifest["fusion"] is None

    def test_discovery_targets_the_framework_root_it_was_given(self, tmp_path, monkeypatch):
        """``--framework-root`` pins WHICH install is being optimized.

        Discovery reaches it too, because a proposal is checked against that
        install's compile-pass config to decide whether the framework already
        fuses the chain. Left unset, that check probes whichever vLLM happens to
        be importable, and its verdict rewrites the pattern id -- so the run can
        both judge the wrong install and store under a different key than a run
        that passed the flag. The pattern route has always forwarded it.
        """
        from kernelforge.fusion import command as cli_module

        seen: dict[str, object] = {}

        def spy(diagnosis, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(cli_module, "discover_recipes", spy)
        backend = SimpleNamespace(
            name="codex",
            runtime=SimpleNamespace(model="gpt-test", sandbox_mode="workspace-write"),
        )
        monkeypatch.setattr(cli_module, "_create_agent_backend", lambda *_args: backend)
        monkeypatch.setattr(cli_module, "registered_agent_llm_fn", lambda *_a, **_k: lambda _p: "[]")
        source = tmp_path / "toylm.py"
        source.write_text("def forward(x):\n    return x\n", encoding="utf-8")
        monkeypatch.setattr(
            cli_module, "resolve_framework_source_file", lambda *a, **k: (str(source), "path convention")
        )
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
            encoding="utf-8",
        )
        root = tmp_path / "pinned-vllm"
        root.mkdir()

        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "vllm",
                "--output-dir",
                str(tmp_path / "out"),
                "--framework-root",
                str(root),
                "--dry-run",
                "--discover",
                "llm",
            ],
        )

        assert res.exit_code == 0, res.output
        assert seen.get("framework_root") == str(root)

    def test_unreachable_llm_is_not_written_as_no_opportunity(self, tmp_path, monkeypatch):
        """The incident, end to end: launch-bound trace + a dead gateway.

        The old code wrote ``no_opportunity`` and exited 0 here, publishing a
        wrong optimization conclusion about a model it never analyzed.
        """
        from kernelforge.fusion import command as cli_module
        from kernelforge.fusion.llm_failure import API_ERROR, LlmUnavailableError

        def dead_agent(_backend, **_kwargs):
            def _fn(_prompt):
                raise LlmUnavailableError(
                    "discovery LLM unreachable after 5 attempts: BadRequestError: Error code: 400",
                    kind=API_ERROR,
                    attempts=5,
                )

            return _fn

        source = tmp_path / "toylm.py"
        source.write_text("def forward(x):\n    return x\n", encoding="utf-8")
        backend = SimpleNamespace(
            name="codex",
            runtime=SimpleNamespace(model="gpt-test", sandbox_mode="workspace-write"),
        )
        monkeypatch.setattr(cli_module, "_create_agent_backend", lambda *_args: backend)
        monkeypatch.setattr(cli_module, "registered_agent_llm_fn", dead_agent)
        monkeypatch.setattr(
            cli_module, "resolve_framework_source_file", lambda *a, **k: (str(source), "path convention")
        )
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
            encoding="utf-8",
        )
        out = tmp_path / "out"

        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--dry-run",
                "--discover",
                "llm",
            ],
        )

        assert res.exit_code == cli_module.EXIT_LLM_UNAVAILABLE, res.output
        manifest = json.loads((out / "fusion_manifest.json").read_text())
        assert manifest["diagnosis"]["is_candidate"] is True
        assert manifest["verdict"] == "llm_unavailable"
        assert manifest["error"]["kind"] == API_ERROR
        assert manifest["error"]["attempts"] == 5

    def test_explicit_codex_discovery_uses_one_registered_backend(
        self,
        tmp_path,
        monkeypatch,
    ):
        captured = {"runs": 0}

        class Backend:
            name = "codex"
            capabilities = AgentCapabilities(
                sandbox=True,
                requires_workspace_cwd=True,
            )
            runtime = AgentRuntimeConfig(
                provider="codex",
                model="gpt-explicit",
                sandbox_mode="workspace-write",
            )

            async def run(self, spec, usage=None):
                captured["runs"] += 1
                captured["spec"] = spec
                return AgentRunResult(
                    text=json.dumps(
                        [
                            {
                                "name": "residual_norm",
                                "env_flag": "FUSED_RESIDUAL",
                                "op_chain": "residual add + rmsnorm",
                                "source_anchors": ["forward"],
                                "fusion_math": "Fuse residual add and RMSNorm.",
                                "eager_reference": "Import the eager RMSNorm.",
                                "candidate_kind": "new_fusion",
                                "existing_operator": "",
                                "priority": 0.9,
                                "rationale": "The trace shows tiny add and norm kernels.",
                            }
                        ]
                    )
                )

        def fake_create(provider, model, sandbox_mode):
            captured["provider"] = provider
            captured["model"] = model
            captured["sandbox_mode"] = sandbox_mode
            return Backend()

        source = tmp_path / "toylm.py"
        source.write_text(
            "def forward(x, residual):\n    return rmsnorm(x + residual)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(cli_module, "_create_agent_backend", fake_create)
        monkeypatch.setattr(
            cli_module,
            "resolve_framework_source_file",
            lambda *_args, **_kwargs: (str(source), "path convention"),
        )
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "toylm",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "out"

        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--dry-run",
                "--discover",
                "llm",
                "--agent-backend",
                "codex",
                "--model",
                "gpt-explicit",
            ],
        )

        assert res.exit_code == 0, res.output
        assert captured["provider"] == "codex"
        assert captured["model"] == "gpt-explicit"
        assert captured["sandbox_mode"] == "workspace-write"
        assert captured["runs"] == 1
        manifest = json.loads((out / "fusion_manifest.json").read_text())
        assert manifest["agent_backend"] == "codex"
        assert manifest["agent_model"] == "gpt-explicit"
        assert manifest["agent_sandbox_mode"] == "workspace-write"

    def test_non_dry_run_no_author_no_validate(self, tmp_path):
        # Non-dry-run with author+validate disabled must NOT invoke the LLM/GPU;
        # it just emits the manifest (validation null).
        trace = tmp_path / "decode.trace.json"
        _write_trace(trace, _launch_bound_events())
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
            encoding="utf-8",
        )
        out = tmp_path / "out"
        res = CliRunner().invoke(
            main,
            [
                "--trace",
                str(trace),
                "--model-path",
                str(model),
                "--framework",
                "sglang",
                "--output-dir",
                str(out),
                "--no-author",
                "--no-validate",
            ],
        )
        assert res.exit_code == 0, res.output
        manifest = json.loads((out / "fusion_manifest.json").read_text())
        assert manifest["verdict"] == "candidate"
        assert manifest["validation"] is None


@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (author_module.AUTHOR_RC_OK, author_module.AUTHOR_RC_FAILED),
        (author_module.AUTHOR_RC_TIMEOUT, author_module.AUTHOR_RC_FAILED),
        (author_module.AUTHOR_RC_FAILED, author_module.AUTHOR_RC_FAILED),
        (author_module.AUTHOR_RC_SAFETY, author_module.AUTHOR_RC_SAFETY),
    ],
)
def test_a_failed_harness_finalization_keeps_a_safety_verdict(rc, expected):
    """Fold a harness failure into the code without erasing a verdict.

    The fold is retryable on purpose: the bucket mixes an author that rewrote the
    inherited harness with a plain OSError while publishing it, and only the
    first is deterministic. A safety stop is neither -- the author already
    decided, identically on every attempt, so replacing it with a retryable code
    sends the loop back to re-run a recipe that is rejected the same way and
    spends the budget proving it.
    """
    assert cli_module._author_rc_after_harness(rc, harness_ok=False) == expected


@pytest.mark.parametrize(
    "rc",
    [
        author_module.AUTHOR_RC_OK,
        author_module.AUTHOR_RC_TIMEOUT,
        author_module.AUTHOR_RC_SAFETY,
    ],
)
def test_a_successful_harness_finalization_changes_nothing(rc):
    """Leave the author's own code alone when the harness published cleanly."""
    assert cli_module._author_rc_after_harness(rc, harness_ok=True) == rc


def test_the_fuse_cli_names_its_model_and_target_as_the_loop_does():
    """Two spellings of one concept per pipeline is a support cost, not a feature."""
    help_text = CliRunner().invoke(main, ["--help"]).output

    assert "--model TEXT" in help_text
    assert "--gpu-target TEXT" in help_text
    assert "--llm-model" not in help_text
    assert "--gpu-arch" not in help_text
