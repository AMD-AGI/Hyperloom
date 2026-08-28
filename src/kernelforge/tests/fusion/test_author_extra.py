# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for run_author registered-backend path and multi-recipe prompt assembly."""

from __future__ import annotations

import asyncio
import shutil
import stat
import subprocess

import pytest
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRuntimeConfig,
)

from kernelforge.fusion import author, emit
from kernelforge.fusion.author import (
    AUTHOR_RC_FAILED,
    AUTHOR_RC_SAFETY,
    AUTHOR_RC_TIMEOUT,
    build_multi_author_prompt,
    run_author,
)
from kernelforge.fusion.llm_failure import AGENT_SAFETY_REJECTION_ATTR


def _recipe(flag: str = "FUSED"):
    return {
        "pattern": "residual_add_rmsnorm",
        "description": "Fold residual-add into RMSNorm.",
        "env_flag": flag,
        "source_file": "/sgl/models/lfm2.py",
        "source_hints": ["+ residual", "RMSNorm("],
        "fusion_math": "y, residual = norm(x, residual)",
        "eager_reference_hint": "Import RMSNorm.",
        "shapes": {"hidden_size": 2048, "T": 16},
        "rocm_native": True,
    }


def test_multi_prompt_single_recipe_delegates():
    p = build_multi_author_prompt([_recipe()], framework="sglang", ab_hint="run")
    # single recipe path returns the same shape as build_author_prompt
    assert "residual_add_rmsnorm" in p
    assert "Fusion 1:" not in p


def test_multi_prompt_multiple_recipes():
    r1, r2 = _recipe("FLAG_A"), _recipe("FLAG_B")
    p = build_multi_author_prompt([r1, r2], framework="sglang", ab_hint="run", harness_path="/tmp/h.py")
    assert "Fusion 1:" in p and "Fusion 2:" in p
    assert "FLAG_A" in p and "FLAG_B" in p
    assert "env_flags=FLAG_A FLAG_B" in p
    assert "/tmp/h.py" in p  # harness block wired in


def test_multi_prompt_rocm_absent_when_none_native():
    r1, r2 = _recipe("FLAG_A"), _recipe("FLAG_B")
    r1["rocm_native"] = r2["rocm_native"] = False
    p = build_multi_author_prompt([r1, r2], framework="sglang", ab_hint="run")
    assert "TARGET IS ROCm" not in p


def test_registered_author_uses_backend_contract(tmp_path):
    captured = {}
    repo, source, _non_target = _author_repo(tmp_path)

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            captured["spec"] = spec
            captured["usage"] = usage
            assert spec.progress_log is not None
            spec.progress_log.append("tool: Edit model.py")
            return AgentRunResult(text="AUTHORING_RESULT: ok", end_reason="agent_stopped")

    log_path = tmp_path / "logs" / "author.log"

    rc = run_author(
        "USER AUTHORING PROMPT",
        workdir=str(repo),
        log_path=str(log_path),
        gpu="3",
        max_turns=17,
        timeout_s=33,
        backend=Backend(),
        target_files=[str(source)],
    )

    assert rc == 0
    spec = captured["spec"]
    assert spec.system_prompt != spec.user_prompt
    assert spec.user_prompt == "USER AUTHORING PROMPT"
    assert spec.cwd == str(repo)
    assert spec.model == "gpt-test"
    assert spec.timeout_sec == 33
    assert spec.tool_policy.max_turns == 17
    assert spec.target_files == [str(source)]
    assert spec.writable is True
    assert spec.tool_policy.write is True and spec.tool_policy.shell is True
    hook = spec.hooks.pre_tool_use[0].callback
    assert (
        asyncio.run(
            hook(
                {"tool_input": {"file_path": str(source)}},
                "tool-id",
                None,
            )
        )
        == {}
    )
    denied = asyncio.run(
        hook(
            {"tool_input": {"file_path": str(tmp_path / "not-a-target.py")}},
            "tool-id",
            None,
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "tool: Edit model.py" in log_path.read_text(encoding="utf-8")
    assert "AUTHORING_RESULT: ok" in log_path.read_text(encoding="utf-8")


def test_registered_author_timeout_keeps_return_code_contract(tmp_path, monkeypatch):
    repo, target, _non_target = _author_repo(tmp_path)

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            raise TimeoutError("agent timed out")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author.log"),
        backend=Backend(),
        target_files=[str(target)],
        timeout_s=1,
    )

    assert rc == AUTHOR_RC_TIMEOUT
    assert "timed out" in (tmp_path / "author.log").read_text(encoding="utf-8")


def test_registered_author_reports_provider_safety_stop_as_deterministic(tmp_path):
    """A provider safety stop must not reach the loop as a retryable failure."""
    repo, target, _non_target = _author_repo(tmp_path)

    class ProviderSafetyError(RuntimeError):
        """Stand in for WorkspaceSafetyError without importing a provider package."""

        def __init__(self, message):
            super().__init__(message)
            setattr(self, AGENT_SAFETY_REJECTION_ATTR, True)

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            raise ProviderSafetyError("Codex session changed the workspace")

    log_path = tmp_path / "author-provider-safety.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=Backend(),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert "changed the workspace" in log_path.read_text(encoding="utf-8")


def test_registered_author_reports_transport_failure_as_retryable(tmp_path):
    """Keep an ordinary backend failure in the class the loop retries."""
    repo, target, _non_target = _author_repo(tmp_path)

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            raise RuntimeError("gateway reset the connection")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-transport.log"),
        backend=Backend(),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_FAILED


def _io_safety_backend(error):
    """A backend whose safety class carries an I/O failure, not a verdict."""

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            raise error

    return Backend()


def _provider_safety_error(message, *, rejection):
    """Stand in for WorkspaceSafetyError without importing a provider package."""

    class ProviderSafetyError(RuntimeError):
        pass

    error = ProviderSafetyError(message)
    setattr(error, AGENT_SAFETY_REJECTION_ATTR, rejection)
    return error


def test_registered_author_retries_a_provider_safety_class_raised_for_io(tmp_path):
    """The provider raises its safety class for I/O too, and that is weather.

    ``Could not snapshot`` says the guard could not read a file, not that the
    session touched one. Classifying it by class name abandoned the recipe on the
    first attempt for a condition the next attempt would very likely not see.
    """
    repo, target, _non_target = _author_repo(tmp_path)

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-provider-io.log"),
        backend=_io_safety_backend(
            _provider_safety_error(
                "Could not snapshot /repo/models/qwen3.py: [Errno 5] Input/output error",
                rejection=False,
            )
        ),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_FAILED


def test_registered_author_reports_a_timeout_a_failed_rollback_wrapped(tmp_path):
    """A hiccuping restore on the way out of a timeout is still a timeout.

    The backend rolls back while unwinding an expired clock, and a rollback that
    itself fails replaces the timeout with its own exception. Reading only the
    outermost error called that a deterministic rejection and abandoned the
    recipe over a session that had merely run out of time.
    """
    repo, target, _non_target = _author_repo(tmp_path)

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            # Raised from the handler, exactly as the backend's rollback is, so the
            # expired clock survives only in __context__.
            try:
                raise TimeoutError("Codex timed out after 3600s")
            except TimeoutError:
                raise _provider_safety_error(
                    "Codex run ended and the inherited workspace state could not "
                    "be restored: [Errno 30] Read-only file system",
                    rejection=False,
                )

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-timeout-rollback.log"),
        backend=Backend(),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_TIMEOUT


def test_registered_author_retries_a_locked_git_index_before_the_run(tmp_path):
    """``index.lock`` is held for milliseconds by any other git command.

    Reporting it as a workspace-safety rejection made the textbook retryable
    condition fatal: the loop abandoned the recipe without authoring anything.
    """
    repo, target, _non_target = _author_repo(tmp_path)
    (repo / ".git" / "index.lock").write_text("", encoding="utf-8")

    log_path = tmp_path / "author-index-locked.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(lambda: None),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_FAILED
    assert "index is locked" in log_path.read_text(encoding="utf-8")


def test_registered_author_retries_a_git_index_locked_during_restoration(tmp_path):
    """Same condition, reached from the restoration half of the transaction."""
    repo, target, non_target = _author_repo(tmp_path)

    def mutate():
        non_target.write_text("NON_TARGET = 'rejected'\n", encoding="utf-8")
        (repo / ".git" / "index.lock").write_text("", encoding="utf-8")

    log_path = tmp_path / "author-index-locked-late.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_FAILED
    assert "index became locked" in log_path.read_text(encoding="utf-8")


def test_registered_author_rejects_a_nominated_module_directory_that_is_absent(
    tmp_path,
):
    """An empty inventory is the most permissive scope there is.

    Every name reads as absent in it, so failing open turned a mis-nominated
    directory into the one place the author could create anything -- and the
    prompt then advertised that absent path.
    """
    repo, target, _non_target = _author_repo(tmp_path)

    log_path = tmp_path / "author-absent-module-dir.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(lambda: None),
        target_files=[str(target)],
        new_module_dirs=[str(repo / "models")],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert "does not exist" in log_path.read_text(encoding="utf-8")


def test_registered_author_names_the_run_failure_beside_a_workspace_rejection(
    tmp_path,
):
    """A rejected turn that also ran out of clock must report both.

    ``enforce()`` is judged before the run error is examined and returns from
    there, so the operator saw the violation and no sign the session never
    finished -- with ``result`` unset, the log had no agent text either.
    """
    repo, target, non_target = _author_repo(tmp_path)

    class Backend:
        name = "codex"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test")

        async def run(self, spec, usage=None):
            non_target.write_text("NON_TARGET = 'rejected'\n", encoding="utf-8")
            raise TimeoutError("Codex timed out after 3600s")

    log_path = tmp_path / "author-violation-and-timeout.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=Backend(),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    log_text = log_path.read_text(encoding="utf-8")
    assert "non_target.py" in log_text
    assert "timed out after 3600s" in log_text


def test_registered_author_rejects_a_permitted_new_module_that_was_staged(tmp_path):
    """Staging a permitted creation drops it out of the exported patch.

    Export reaches an untracked new module through ``git diff --no-index``, so an
    indexed one silently disappears from the handoff. The prompt states the rule;
    nothing pinned that the guard enforces it.
    """
    repo, target, _non_target = _author_repo(tmp_path)
    created = repo / "qwen3_fused_ops.py"

    def mutate():
        created.write_text("def fused():\n    return 'authored'\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repo), "add", "qwen3_fused_ops.py"],
            check=True,
            capture_output=True,
        )

    log_path = tmp_path / "author-staged-new-module.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
        new_module_dirs=[str(repo)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert not created.exists()
    assert "qwen3_fused_ops.py" in log_path.read_text(encoding="utf-8")


def _author_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "target.py"
    non_target = repo / "non_target.py"
    target.write_text("TARGET = 'baseline'\n", encoding="utf-8")
    non_target.write_text("NON_TARGET = 'baseline-secret'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "init", "-q"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "target.py", "non_target.py"],
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
    return repo, target, non_target


def _mutating_backend(action):
    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(
            provider="claude",
            model="claude-test",
            sandbox_mode="workspace-write",
        )

        async def run(self, spec, usage=None):
            action()
            return AgentRunResult(
                text="AUTHORING_RESULT: completed",
                end_reason="agent_stopped",
                tool_calls=[("Bash", {"command": "redacted mutation"})],
            )

    return Backend()


@pytest.mark.parametrize(
    "mutation",
    [
        "modify",
        "create",
        "delete",
        "rename",
        "chmod",
        "symlink",
    ],
)
def test_registered_author_restores_non_target_git_mutations(tmp_path, mutation):
    repo, target, non_target = _author_repo(tmp_path)
    baseline = non_target.read_bytes()
    baseline_mode = stat.S_IMODE(non_target.stat().st_mode)
    created = repo / "created.py"
    renamed = repo / "renamed.py"

    def mutate():
        if mutation == "modify":
            non_target.write_text("NON_TARGET = 'changed'\n", encoding="utf-8")
        elif mutation == "create":
            created.write_text("CREATED = True\n", encoding="utf-8")
        elif mutation == "delete":
            non_target.unlink()
        elif mutation == "rename":
            non_target.rename(renamed)
        elif mutation == "chmod":
            non_target.chmod(baseline_mode ^ stat.S_IXUSR)
        elif mutation == "symlink":
            non_target.unlink()
            non_target.symlink_to(target.name)

    log_path = tmp_path / f"author-{mutation}.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert non_target.is_file() and not non_target.is_symlink()
    assert non_target.read_bytes() == baseline
    assert stat.S_IMODE(non_target.stat().st_mode) == baseline_mode
    assert not created.exists() and not renamed.exists()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    log_text = log_path.read_text(encoding="utf-8")
    assert "rejected" in log_text and "restored" in log_text
    assert "baseline-secret" not in log_text


def test_registered_author_preserves_allowed_target_changes(tmp_path):
    repo, target, _non_target = _author_repo(tmp_path)

    def mutate():
        target.write_text("TARGET = 'authored'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-target.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == 0
    assert target.read_text(encoding="utf-8") == "TARGET = 'authored'\n"


def test_registered_author_may_write_the_staged_harness(tmp_path):
    """Let the author write the harness the pipeline staged inside the worktree.

    ``_author_harness_target`` puts the validation harness under
    ``<repo>/.forge_fusion/`` because the author sandbox is workspace-write and
    cannot reach outside the tree, so writing there is what the directory exists
    for. The guard counted it as an out-of-scope creation, restored it and
    returned a safety verdict -- discarding a wired 1.81x fusion, and doing so
    identically on every retry.
    """
    repo, target, _non_target = _author_repo(tmp_path)
    staged = repo / ".forge_fusion" / "kernel_harness_5a45e46212fc.py"

    def mutate():
        target.write_text("TARGET = 'authored'\n", encoding="utf-8")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("print('harness')\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-staging.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == 0
    assert target.read_text(encoding="utf-8") == "TARGET = 'authored'\n"
    assert staged.read_text(encoding="utf-8") == "print('harness')\n"


def test_edit_hook_is_no_stricter_than_the_guard_on_staging(tmp_path):
    """Keep the PreToolUse hook exactly as permissive as the transaction.

    The hook consults the guard's own predicate precisely so it can never block
    a path the transaction would keep. If only ``enforce`` learns about the
    staging directory, Edit/Write stay denied there and the author can get its
    harness written only through Bash.
    """
    repo, target, _non_target = _author_repo(tmp_path)
    models = repo / "models"
    models.mkdir()
    guard = author._AuthorWorkspaceGuard(
        str(repo),
        [str(target)],
        new_module_dirs=[str(models)],
    )
    staged = repo / ".forge_fusion" / "kernel_harness_5a45e46212fc.py"

    assert guard.permits_new_path(str(staged))


def test_registered_author_restores_after_backend_timeout(tmp_path):
    repo, target, non_target = _author_repo(tmp_path)
    baseline = non_target.read_bytes()

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            target.write_text("TARGET = 'timeout-authored'\n", encoding="utf-8")
            non_target.write_text("NON_TARGET = 'timeout-edit'\n", encoding="utf-8")
            raise TimeoutError("agent timed out")

    log_path = tmp_path / "author-timeout-restore.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=Backend(),
        target_files=[str(target)],
        timeout_s=1,
    )

    assert rc != 0
    assert target.read_text(encoding="utf-8") == "TARGET = 'timeout-authored'\n"
    assert non_target.read_bytes() == baseline
    assert "restored" in log_path.read_text(encoding="utf-8")


def test_registered_author_preserves_preexisting_dirty_index_and_untracked_state(tmp_path):
    repo, target, non_target = _author_repo(tmp_path)
    staged = repo / "staged.py"
    untracked = repo / "operator_notes.py"
    staged.write_text("STAGED = 'base'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "staged.py"],
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
            "add staged fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    non_target.write_text("NON_TARGET = 'operator-unstaged'\n", encoding="utf-8")
    staged.write_text("STAGED = 'operator-staged'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "staged.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    untracked.write_text("OPERATOR = 'untracked'\n", encoding="utf-8")

    def git_output(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
        ).stdout

    baseline_status = git_output("status", "--porcelain=v2", "-z")
    baseline_diff = git_output("diff", "--binary")
    baseline_cached = git_output("diff", "--cached", "--binary")
    baseline_files = {path: path.read_bytes() for path in (non_target, staged, untracked)}

    def mutate():
        non_target.write_text("NON_TARGET = 'agent-overwrite'\n", encoding="utf-8")
        staged.write_text("STAGED = 'agent-overwrite'\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repo), "add", "non_target.py"],
            check=True,
            capture_output=True,
        )
        untracked.write_text("OPERATOR = 'agent-overwrite'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-dirty.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert git_output("status", "--porcelain=v2", "-z") == baseline_status
    assert git_output("diff", "--binary") == baseline_diff
    assert git_output("diff", "--cached", "--binary") == baseline_cached
    for path, content in baseline_files.items():
        assert path.read_bytes() == content


def test_registered_author_accepts_target_edit_with_preexisting_dirty_state(tmp_path):
    repo, target, non_target = _author_repo(tmp_path)
    staged = repo / "staged.py"
    staged.write_text("STAGED = 'base'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "staged.py"],
        check=True,
        capture_output=True,
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
            "add staged fixture",
        ],
        check=True,
        capture_output=True,
    )
    non_target.write_text("NON_TARGET = 'operator-dirty'\n", encoding="utf-8")
    staged.write_text("STAGED = 'operator-staged'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "staged.py"],
        check=True,
        capture_output=True,
    )
    untracked = repo / "operator_notes.py"
    untracked.write_text("OPERATOR = 'untracked'\n", encoding="utf-8")
    baseline_non_target = non_target.read_bytes()
    baseline_staged = staged.read_bytes()
    baseline_untracked = untracked.read_bytes()
    baseline_cached = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--binary"],
        check=True,
        capture_output=True,
    ).stdout

    def mutate():
        target.write_text("TARGET = 'authored'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-dirty-target.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == 0
    assert target.read_text(encoding="utf-8") == "TARGET = 'authored'\n"
    assert non_target.read_bytes() == baseline_non_target
    assert staged.read_bytes() == baseline_staged
    assert untracked.read_bytes() == baseline_untracked
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--binary"],
            check=True,
            capture_output=True,
        ).stdout
        == baseline_cached
    )


def _git_commit(repo, message):
    """Commit the current index in a fixture repository."""
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
            message,
        ],
        check=True,
        capture_output=True,
    )


def test_registered_author_keeps_permitted_new_fused_module(tmp_path):
    """Let the author add the fused-kernel module the target file imports."""
    repo, target, non_target = _author_repo(tmp_path)
    created = repo / "qwen3_fused_ops.py"
    baseline_non_target = non_target.read_bytes()

    def mutate():
        target.write_text(
            "from qwen3_fused_ops import fused\n\nTARGET = fused()\n",
            encoding="utf-8",
        )
        created.write_text("def fused():\n    return 'authored'\n", encoding="utf-8")

    log_path = tmp_path / "author-new-module.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
        new_module_dirs=[str(repo)],
    )

    assert rc == 0
    assert created.read_text(encoding="utf-8") == "def fused():\n    return 'authored'\n"
    assert non_target.read_bytes() == baseline_non_target
    log_text = log_path.read_text(encoding="utf-8")
    assert "created" in log_text and "qwen3_fused_ops.py" in log_text


@pytest.mark.parametrize(
    "relative",
    [
        "nested/qwen3_fused_ops.py",
        "qwen3_helper.py",
        "qwen3_fused_ops.txt",
        "diffusion_qwen3.py",
    ],
)
def test_registered_author_restores_new_file_outside_permitted_scope(tmp_path, relative):
    """Keep every creation outside the bounded fused-module scope rejected."""
    repo, target, _non_target = _author_repo(tmp_path)
    created = repo / relative

    def mutate():
        created.parent.mkdir(parents=True, exist_ok=True)
        created.write_text("CREATED = True\n", encoding="utf-8")

    log_path = tmp_path / "author-outside-scope.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
        new_module_dirs=[str(repo)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert not created.exists()
    assert "rejected" in log_path.read_text(encoding="utf-8")


def test_registered_author_restores_edit_to_existing_fused_module(tmp_path):
    """A framework file that merely matches the fused marker stays read-only."""
    repo, target, _non_target = _author_repo(tmp_path)
    framework = repo / "fused_moe.py"
    framework.write_text("MOE = 'framework'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "fused_moe.py"],
        check=True,
        capture_output=True,
    )
    _git_commit(repo, "add framework fused module")

    def mutate():
        framework.write_text("MOE = 'agent overwrite'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-existing-fused.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
        new_module_dirs=[str(repo)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert framework.read_text(encoding="utf-8") == "MOE = 'framework'\n"


def test_registered_author_hook_and_prompt_match_the_permitted_scope(tmp_path):
    """Keep the SDK edit hook, the system prompt, and the guard on one allowlist."""
    repo, target, non_target = _author_repo(tmp_path)
    captured = {}

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            captured["spec"] = spec
            return AgentRunResult(text="ok", end_reason="agent_stopped")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-hook-scope.log"),
        backend=Backend(),
        target_files=[str(target)],
        new_module_dirs=[str(repo)],
    )

    assert rc == 0
    spec = captured["spec"]
    assert spec.allow_dirty_baseline is True
    # Every fragment the guard's predicate matches on has to be in the prompt, or
    # adding one to emit leaves the author obeying the previous rule and being
    # rejected for it.
    for fragment in (*emit._FUSED_MODULE_MARKERS, *emit._FUSED_MODULE_PREFIXES):
        assert fragment in spec.system_prompt, fragment
    hook = spec.hooks.pre_tool_use[0].callback

    def decide(path):
        return asyncio.run(hook({"tool_input": {"file_path": str(path)}}, "tool-id", None))

    assert decide(repo / "qwen3_fused_ops.py") == {}
    assert decide(target) == {}
    for rejected in (non_target, repo / "qwen3_helper.py", repo / "nested" / "x_fused.py"):
        decision = decide(rejected)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_registered_author_preserves_allowed_target_index_change_while_restoring(
    tmp_path,
):
    repo, target, non_target = _author_repo(tmp_path)

    def mutate():
        target.write_text("TARGET = 'staged-authored'\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repo), "add", "target.py"],
            check=True,
            capture_output=True,
        )
        non_target.write_text("NON_TARGET = 'reject-me'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-target-index.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert target.read_text(encoding="utf-8") == "TARGET = 'staged-authored'\n"
    staged_target = subprocess.run(
        ["git", "-C", str(repo), "show", ":target.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert staged_target == "TARGET = 'staged-authored'\n"
    assert non_target.read_text(encoding="utf-8") == "NON_TARGET = 'baseline-secret'\n"


def test_registered_author_restores_assume_unchanged_non_target(tmp_path):
    repo, target, non_target = _author_repo(tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--assume-unchanged", "non_target.py"],
        check=True,
        capture_output=True,
    )

    def mutate():
        non_target.write_text("NON_TARGET = 'hidden-edit'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "author-assume-unchanged.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert non_target.read_text(encoding="utf-8") == "NON_TARGET = 'baseline-secret'\n"
    flag = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-v", "non_target.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout[:1]
    assert flag == "h"


@pytest.mark.parametrize("target_kind", ["traversal", "symlink-escape"])
def test_registered_author_rejects_unsafe_target_paths(tmp_path, target_kind):
    repo, target, _non_target = _author_repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n", encoding="utf-8")
    unsafe_target = repo / ".." / "outside.py"
    if target_kind == "symlink-escape":
        unsafe_target = repo / "escape.py"
        unsafe_target.symlink_to(outside)
    called = {"value": False}

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities()
        runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

        async def run(self, spec, usage=None):
            called["value"] = True
            return AgentRunResult(text="unexpected")

    log_path = tmp_path / f"unsafe-{target_kind}.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=Backend(),
        target_files=[str(unsafe_target), str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert called["value"] is False
    assert "outside the git worktree" in log_path.read_text(encoding="utf-8").lower()


def test_registered_author_restores_target_symlink_escape(tmp_path):
    repo, target, _non_target = _author_repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n", encoding="utf-8")

    def mutate():
        target.unlink()
        target.symlink_to(outside)

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "target-symlink.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert target.is_file() and not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "TARGET = 'baseline'\n"
    assert outside.read_text(encoding="utf-8") == "OUTSIDE = True\n"


def test_registered_author_never_follows_non_target_parent_symlink(tmp_path):
    repo, target, _non_target = _author_repo(tmp_path)
    nested = repo / "nested"
    tracked = nested / "tracked.py"
    nested.mkdir()
    tracked.write_text("TRACKED = 'baseline'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "nested/tracked.py"],
        check=True,
        capture_output=True,
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
            "add nested fixture",
        ],
        check=True,
        capture_output=True,
    )
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    outside_file = outside_dir / "tracked.py"
    outside_file.write_text("OUTSIDE = 'must-survive'\n", encoding="utf-8")

    def mutate():
        shutil.rmtree(nested)
        nested.symlink_to(outside_dir, target_is_directory=True)

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "parent-symlink.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert nested.is_dir() and not nested.is_symlink()
    assert tracked.read_text(encoding="utf-8") == "TRACKED = 'baseline'\n"
    assert outside_file.read_text(encoding="utf-8") == "OUTSIDE = 'must-survive'\n"


def test_registered_author_restores_file_replaced_by_directory(tmp_path):
    repo, target, non_target = _author_repo(tmp_path)
    child = non_target / "child.py"

    def mutate():
        non_target.unlink()
        non_target.mkdir()
        child.write_text("CHILD = 'new'\n", encoding="utf-8")

    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(tmp_path / "file-to-directory.log"),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    assert non_target.is_file()
    assert non_target.read_text(encoding="utf-8") == "NON_TARGET = 'baseline-secret'\n"


def test_registered_author_fails_closed_when_git_head_changes(tmp_path):
    repo, target, _non_target = _author_repo(tmp_path)

    def mutate():
        target.write_text("TARGET = 'committed'\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repo), "add", "target.py"],
            check=True,
            capture_output=True,
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
                "forbidden agent commit",
            ],
            check=True,
            capture_output=True,
        )

    log_path = tmp_path / "head-change.log"
    rc = run_author(
        "P",
        workdir=str(repo),
        log_path=str(log_path),
        backend=_mutating_backend(mutate),
        target_files=[str(target)],
    )

    assert rc == AUTHOR_RC_SAFETY
    log_text = log_path.read_text(encoding="utf-8")
    assert "changed Git HEAD or branch" in log_text
    assert "<git-head>" in log_text


def test_capture_path_state_reports_io_failures_as_transient(tmp_path, monkeypatch):
    """A stat that failed says nothing about what the author did.

    ``AuthorSafetyError`` carries a verdict about the worktree's CONTENT, which
    the guard reaches identically on the next attempt, so the loop abandons the
    recipe on one. Reading the worktree is not that: an NFS blip while snapshotting
    is weather, and marking it a verdict throws away a recipe a retry would have
    finished. The Git-command and index-lock paths beside these already say so.
    """
    path = tmp_path / "kernel.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")

    def failing_lstat(_self):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(author.Path, "lstat", failing_lstat)

    with pytest.raises(author.AuthorSafetyError) as raised:
        author._capture_path_state(path)

    assert raised.value.transient is True


def test_capture_path_state_reports_an_unreadable_file_as_transient(tmp_path, monkeypatch):
    """Same for the read that follows the stat."""
    path = tmp_path / "kernel.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")

    def failing_read(_self):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(author.Path, "read_bytes", failing_read)

    with pytest.raises(author.AuthorSafetyError) as raised:
        author._capture_path_state(path)

    assert raised.value.transient is True


def test_capture_path_state_reports_an_unreadable_symlink_as_transient(tmp_path, monkeypatch):
    """And for the readlink on the symlink branch."""
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)

    def failing_readlink(_path):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(author.os, "readlink", failing_readlink)

    with pytest.raises(author.AuthorSafetyError) as raised:
        author._capture_path_state(link)

    assert raised.value.transient is True


def test_module_directory_inventory_reports_io_failures_as_transient(tmp_path, monkeypatch):
    """Inventorying the creatable-module directory is bookkeeping too.

    A missing directory stays a verdict -- it is the same on every attempt and
    would otherwise advertise a scope the author cannot write into -- but a
    listdir that failed for any other reason is not.
    """
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")

    def failing_listdir(_path):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(author.os, "listdir", failing_listdir)

    with pytest.raises(author.AuthorSafetyError) as raised:
        author._AuthorWorkspaceGuard(
            str(repo),
            [str(repo / "kernel.py")],
            new_module_dirs=[str(repo / "models")],
        )

    assert raised.value.transient is True


def test_declared_harness_target_survives_the_default_name_globs(tmp_path):
    """A target the caller allowlisted must not be reclaimed by ``*harness*.py``.

    The harness-author turn's whole deliverable is one
    ``.forge_fusion/kernel_harness_<digest>.py``, declared as its sole
    ``target_files`` entry (``command.py::_author_baseline_harness``). The
    default measurement globs match it by name, and the shadow repo keeps the
    staging directory Git-ignored, so before the exemption the guard rejected the
    file the agent had just been told to write -- ``forge-fuse --author`` died
    with ``protected ignored files changed`` on every real run.
    """
    import subprocess

    from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
    from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError

    repo = tmp_path / "framework"
    repo.mkdir()
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".forge_fusion/\n", encoding="utf-8")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)

    staging = repo / ".forge_fusion" / "kernel_harness_e0120d9d95c2.py"

    def run(targets):
        spec = AgentRunSpec(
            system_prompt="",
            user_prompt="",
            cwd=str(repo),
            writable=True,
            target_files=[str(path) for path in targets],
            allow_dirty_targets=True,
            allow_untracked=True,
            allow_dirty_baseline=True,
            protected_globs=[],
            tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True),
        )
        guard = WorkspaceGuard(spec, dirty_baseline_default=True)
        guard.prepare()
        staging.parent.mkdir(exist_ok=True)
        staging.write_text("BENCH = True\n", encoding="utf-8")
        return guard

    assert run([staging]).verify() == []

    staging.unlink()
    # Undeclared, and the name protection is back on: the implementer turn, whose
    # targets are framework sources, still may not touch the harness.
    with pytest.raises(WorkspaceSafetyError, match="protected"):
        run([repo / "kernel.py"]).verify()
