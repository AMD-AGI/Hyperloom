"""Hermetic tests for framework apply-back patch generation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest

import kernelforge.agent_backends.registry as agent_registry
from kernelforge.config import Config
from kernelforge.rewrite_by_flydsl import applyback, protocol
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


@pytest.fixture(autouse=True)
def isolated_agent_provider_registry(monkeypatch):
    """Give every test in this module its own copy of the provider registry.

    ``register_agent_provider`` writes into module-level state that outlives
    the test that called it, and the registry offers no way to unregister, so
    the provider registered below would otherwise stay visible to every later
    test in the same worker process. Discovery runs first so the snapshot
    already holds the built-ins and any installed plugin; the module globals
    are then rebound to copies that monkeypatch drops during teardown.
    """
    agent_registry.discover_agent_providers()
    monkeypatch.setattr(
        agent_registry,
        "_providers",
        dict(agent_registry._providers),
    )
    monkeypatch.setattr(
        agent_registry,
        "_plugin_errors",
        dict(agent_registry._plugin_errors),
    )


@pytest.fixture
def available_agent_provider(isolated_agent_provider_registry):
    """Register one available provider so ``auto`` backend selection resolves.

    ``Config.agent_backend`` defaults to ``auto``, and both built-in providers
    report themselves unavailable unless their optional SDK is installed, so a
    test that reaches provider selection has to supply an available provider
    itself instead of inheriting whichever one another test left behind. Every
    caller replaces backend construction, so the factory only has to exist.
    """

    def factory(runtime):
        """Refuse construction; callers monkeypatch create_registered_backend."""
        raise AssertionError("create_registered_backend must be monkeypatched by the test")

    agent_registry.register_agent_provider(
        agent_registry.AgentProvider(
            name="applybackcli",
            factory=factory,
            default_model="applyback-model",
        )
    )


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_spec(tmp_path):
    repo = tmp_path / "framework"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "softmax.py"
    source.write_text("def softmax(x):\n    return x\n")
    _git(repo, "add", "softmax.py")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    kernel = repo / "kernel.py"
    kernel.write_text(
        "import flydsl\n\ndef build_softmax_module(*args):\n    return lambda *launch_args, **launch_kwargs: None\n"
    )
    spec = RewriteSpec(
        op_name="softmax",
        source_kernel=str(source),
        target_functions=["softmax"],
        flydsl_kernel=str(kernel),
        workspace=str(repo),
    )
    return repo, base, spec


@pytest.mark.parametrize("framework", protocol.SUPPORTED_FRAMEWORKS)
def test_applyback_agent_patch_is_published_in_forge_compatible_bundle(
    tmp_path,
    monkeypatch,
    framework,
):
    repo, base, spec = _repo_spec(tmp_path)

    async def fake_agent(**kwargs):
        worktree = kwargs["worktree"]
        (worktree / "softmax.py").write_text(
            "from flydsl_softmax import run_softmax\n\ndef softmax(x):\n    return run_softmax(x)\n"
        )
        (worktree / "flydsl_softmax.py").write_text("def run_softmax(x):\n    return x\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", fake_agent)
    experiments = tmp_path / "experiments"
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(experiments),
        framework=framework,
        best_commit="verified-best",
        source_ms=2.0,
        flydsl_best_ms=1.0,
        reference_snr_db=61.5,
    )

    assert result.ok is True
    campaign = repo / "forge_experiments"
    namespace = campaign / "rewrite_applyback"
    patch = namespace / "best" / "iter_000" / "forge.patch"
    assert patch.is_file()
    assert "flydsl_softmax.py" in patch.read_text()
    manifest = json.loads((namespace / "best" / "manifest.json").read_text())
    assert manifest["patch_path"] == "rewrite_applyback/best/iter_000/forge.patch"
    assert manifest["artifact_dir"] == "rewrite_applyback/best/iter_000"
    assert manifest["schema_version"] == 2
    assert manifest["artifact_kind"] == "framework_applyback"
    assert manifest["validation_scope"] == "reference"
    assert manifest["logical_op_name"] == "softmax"
    assert manifest["builder_symbol"] == "build_softmax_module"
    assert manifest["reference_correctness_passed"] is True
    assert manifest["reference_snr_db"] == 61.5
    assert manifest["integration_validation_required"] is True
    assert manifest["integration_validation_status"] == "pending"
    assert manifest["framework"] == framework
    # The ambiguous key that read as "the framework patch is proven" is gone.
    assert "correctness_passed" not in manifest
    assert manifest["commit_hash"] == result.best_commit
    assert manifest["flydsl_best_commit"] == "verified-best"
    assert _git(repo, "rev-parse", result.commit_ref) == result.best_commit
    assert sorted(manifest["changed_files"]) == ["flydsl_softmax.py", "softmax.py"]
    assert result.canonical_patch_path == str(patch)
    assert result.canonical_files_root == str(namespace / "best" / "iter_000" / "files")
    assert result.canonical_result_path == str(namespace / "result.json")
    assert json.loads((namespace / "result.json").read_text()) == manifest
    assert result.forge_workspace == str(repo)
    assert result.artifacts == [str(patch)]
    assert result.import_validation_modules == ["softmax"]
    assert (namespace / "best" / "iter_000" / "files" / "flydsl_softmax.py").is_file()
    # The nested standalone forge-loop namespace stays entirely untouched.
    assert not (campaign / "best").exists()
    assert not (campaign / "best_result.json").exists()


@pytest.mark.parametrize("framework", protocol.SUPPORTED_FRAMEWORKS)
def test_framework_is_inferred_from_the_source_path(tmp_path, framework):
    source = tmp_path / framework / "ops" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("def kernel(x):\n    return x\n")
    spec = RewriteSpec(
        op_name="kernel",
        source_kernel=str(source),
        target_functions=["kernel"],
        workspace=str(tmp_path),
    )

    assert applyback._infer_framework(spec, "") == framework


def _seed_standalone_forge_loop_best(repo) -> dict:
    """Write a nested standalone forge-loop publication into its own namespace."""
    campaign = repo / "forge_experiments"
    version = campaign / "best" / "iter_007"
    version.mkdir(parents=True)
    (version / "forge.patch").write_text("standalone flydsl patch\n")
    manifest = {
        "schema_version": 1,
        "iteration": 7,
        "commit_hash": "standalone-flydsl-best",
        "correctness_passed": True,
        "artifact_dir": "best/iter_007",
        "patch_path": "best/iter_007/forge.patch",
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (campaign / "best" / "manifest.json").write_text(payload)
    (campaign / "best_result.json").write_text(payload)
    return manifest


def test_applyback_publication_leaves_the_standalone_best_intact(
    tmp_path,
    monkeypatch,
):
    repo, base, spec = _repo_spec(tmp_path)
    standalone = _seed_standalone_forge_loop_best(repo)

    async def fake_agent(**kwargs):
        (kwargs["worktree"] / "softmax.py").write_text("def softmax(x):\n    return x * 1\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", fake_agent)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
        best_commit="standalone-flydsl-best",
    )

    assert result.ok is True
    campaign = repo / "forge_experiments"
    assert json.loads((campaign / "best_result.json").read_text()) == standalone
    assert json.loads((campaign / "best" / "manifest.json").read_text()) == standalone
    assert (campaign / "best" / "iter_007" / "forge.patch").read_text() == ("standalone flydsl patch\n")
    # A standalone iteration far ahead must not shift the apply-back numbering.
    published = json.loads((campaign / "rewrite_applyback" / "result.json").read_text())
    assert published["iteration"] == 0
    assert published["commit_hash"] == result.best_commit
    assert published["flydsl_best_commit"] == "standalone-flydsl-best"


def test_applyback_failure_publishes_no_canonical_result(tmp_path, monkeypatch):
    repo, base, spec = _repo_spec(tmp_path)
    standalone = _seed_standalone_forge_loop_best(repo)

    async def idle_agent(**kwargs):
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", idle_agent)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
        best_commit="standalone-flydsl-best",
    )

    assert result.ok is False
    assert "no repository changes" in result.error
    # The standalone best is neither republished as an apply-back result nor
    # echoed back as the framework best commit.
    assert result.best_commit == ""
    assert result.canonical_result_path == ""
    namespace = repo / "forge_experiments" / "rewrite_applyback"
    assert not namespace.exists()
    assert json.loads((repo / "forge_experiments" / "best_result.json").read_text()) == standalone
    assert Path(result.diagnostic_path).is_dir()


def test_applyback_retries_from_a_fresh_worktree_with_prior_failure(
    tmp_path,
    monkeypatch,
):
    repo, base, spec = _repo_spec(tmp_path)
    worktrees: list[Path] = []
    prior_failures: list[str] = []

    async def second_attempt_integrates(**kwargs):
        worktrees.append(kwargs["worktree"])
        prior_failures.append(kwargs["prior_failure"])
        if len(worktrees) == 1:
            return "fake", "fake-model"
        (kwargs["worktree"] / "softmax.py").write_text("def softmax(x):\n    return x * 1\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", second_attempt_integrates)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
        max_attempts=2,
    )

    assert result.ok is True
    assert result.attempts == 2
    assert len(set(worktrees)) == 2
    assert prior_failures[0] == ""
    assert "no repository changes" in prior_failures[1]


def test_applyback_publications_increment_within_their_own_namespace(tmp_path):
    repo, base, spec = _repo_spec(tmp_path)
    _seed_standalone_forge_loop_best(repo)

    for index in range(2):
        patch_path, manifest_path, files_root, result_path = applyback._publish_patch(
            spec=spec,
            framework="vllm",
            base_commit=base,
            applyback_commit=base,
            flydsl_best_commit="standalone-flydsl-best",
            commit_ref="refs/forge-rewrite/applyback/softmax-abcdef123456",
            source_ms=2.0,
            flydsl_best_ms=1.0,
            reference_snr_db=45.0,
            patch=f"framework patch {index}\n",
            changed_files=["softmax.py"],
        )

    namespace = repo / "forge_experiments" / "rewrite_applyback"
    assert patch_path == str(namespace / "best" / "iter_001" / "forge.patch")
    assert manifest_path == str(namespace / "best" / "manifest.json")
    assert files_root == str(namespace / "best" / "iter_001" / "files")
    assert result_path == str(namespace / "result.json")
    # Each published version is immutable; the pointers move, the bundles do not.
    assert (namespace / "best" / "iter_000" / "forge.patch").read_text() == ("framework patch 0\n")
    assert Path(patch_path).read_text() == "framework patch 1\n"
    published = json.loads(Path(result_path).read_text())
    assert published["iteration"] == 1
    assert published["artifact_dir"] == "rewrite_applyback/best/iter_001"
    assert json.loads(Path(manifest_path).read_text()) == published
    assert (namespace / "best" / "iter_001" / "files" / "softmax.py").read_text() == "def softmax(x):\n    return x\n"


def test_applyback_pins_the_commit_under_a_slug_derived_ref(tmp_path, monkeypatch):
    repo, base, spec = _repo_spec(tmp_path)
    spec.op_name = "vllm::softmax"

    async def integrate(**kwargs):
        (kwargs["worktree"] / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", integrate)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
    )

    assert result.ok is True
    # One normalization rule, shared with the builder symbol.
    assert result.commit_ref == (f"refs/forge-rewrite/applyback/{spec.operator_slug}-{result.best_commit[:12]}")
    assert "::" not in result.commit_ref
    assert _git(repo, "rev-parse", result.commit_ref) == result.best_commit


def test_applyback_requires_a_pristine_framework_commit(tmp_path):
    repo, _base, spec = _repo_spec(tmp_path)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit="",
        experiments_dir=str(tmp_path / "experiments"),
    )
    assert result.ok is False
    assert "base commit" in result.error


def test_applyback_rejects_an_unresolved_framework_before_agent_work(
    tmp_path,
    monkeypatch,
):
    repo, base, spec = _repo_spec(tmp_path)

    async def unexpected_agent(**kwargs):
        raise AssertionError("unknown framework must fail before agent work")

    monkeypatch.setattr(applyback, "_run_agent", unexpected_agent)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
    )

    assert result.ok is False
    assert "could not be resolved" in result.error


def test_import_plan_infers_package_module_and_python_roots(tmp_path):
    worktree = tmp_path / "framework"
    package = worktree / "python" / "sample" / "ops"
    package.mkdir(parents=True)
    (worktree / "python" / "sample" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "softmax.py").write_text("VALUE = 1\n")

    plan = applyback._build_import_validation_plan(
        worktree=worktree,
        source_relative="python/sample/ops/softmax.py",
        import_modules=(),
    )

    assert plan.modules == ("sample.ops.softmax",)
    assert str(worktree / "python") in plan.python_roots


def test_baseline_import_failure_prevents_the_applyback_agent(
    tmp_path,
    monkeypatch,
):
    repo, _base, spec = _repo_spec(tmp_path)
    (repo / "softmax.py").write_text("import dependency_that_is_not_installed\n")
    _git(repo, "add", "softmax.py")
    _git(repo, "commit", "-qm", "break baseline import")
    base = _git(repo, "rev-parse", "HEAD")

    async def unexpected_agent(**kwargs):
        raise AssertionError("agent must not run when the pristine import fails")

    monkeypatch.setattr(applyback, "_run_agent", unexpected_agent)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
    )

    assert result.ok is False
    assert "baseline apply-back import validation failed for softmax" in result.error
    assert result.attempts == 1


def test_patched_import_failure_rejects_a_syntax_valid_patch(
    tmp_path,
    monkeypatch,
):
    repo, base, spec = _repo_spec(tmp_path)

    async def breaks_import(**kwargs):
        (kwargs["worktree"] / "softmax.py").write_text(
            "from dependency_that_is_not_installed import run\n\ndef softmax(x):\n    return run(x)\n"
        )
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", breaks_import)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
    )

    assert result.ok is False
    assert "patched apply-back import validation failed for softmax" in result.error
    assert not (repo / "forge_experiments" / "rewrite_applyback").exists()


def test_applyback_timeout_preserves_non_publishable_diagnostics(
    tmp_path,
    monkeypatch,
):
    repo, base, spec = _repo_spec(tmp_path)

    async def timing_out_agent(**kwargs):
        worktree = kwargs["worktree"]
        (worktree / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
        kwargs["progress_log"].append("tool:Bash focused smoke test")
        raise asyncio.TimeoutError

    monkeypatch.setattr(applyback, "_run_agent", timing_out_agent)
    experiments = tmp_path / "experiments"
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(
            workspace=str(repo),
            agent_precheck=False,
            agent_timeout_sec=60,
        ),
        base_commit=base,
        experiments_dir=str(experiments),
        framework="vllm",
        deadline_unix=10_000_000_000,
    )

    assert result.ok is False
    assert result.error == "apply-back agent timed out after 60s"
    diagnostic = Path(result.diagnostic_path)
    assert diagnostic.is_dir()
    assert "return x + 1" in (diagnostic / "partial.patch").read_text()
    progress = json.loads((diagnostic / "progress.json").read_text())
    assert progress["events"] == ["tool:Bash focused smoke test"]
    assert not (experiments / "rewrite_applyback" / "best").exists()
    assert not (repo / "forge_experiments" / "rewrite_applyback").exists()
    assert not (repo / "forge_experiments" / "best_result.json").exists()


def test_applyback_host_validation_rejects_test_edits(tmp_path, monkeypatch):
    repo, base, spec = _repo_spec(tmp_path)
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_softmax.py").write_text("def test_softmax():\n    pass\n")
    _git(repo, "add", "tests/test_softmax.py")
    _git(repo, "commit", "-qm", "add test")
    base = _git(repo, "rev-parse", "HEAD")

    async def edits_test(**kwargs):
        worktree = kwargs["worktree"]
        (worktree / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
        (worktree / "tests" / "test_softmax.py").write_text("def test_softmax():\n    assert True\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", edits_test)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
    )

    assert result.ok is False
    assert "protected validation files" in result.error


def test_applyback_refuses_to_publish_producer_owned_state(tmp_path, monkeypatch):
    repo, base, spec = _repo_spec(tmp_path)

    async def leaks_forge_state(**kwargs):
        worktree = kwargs["worktree"]
        (worktree / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
        campaign = worktree / "forge_experiments"
        campaign.mkdir()
        (campaign / "run_state.json").write_text("{}\n")
        return "fake", "fake-model"

    monkeypatch.setattr(applyback, "_run_agent", leaks_forge_state)
    result = applyback.generate_applyback_patch(
        spec,
        Config.from_env(workspace=str(repo), agent_precheck=False),
        base_commit=base,
        experiments_dir=str(tmp_path / "experiments"),
        framework="vllm",
    )

    assert result.ok is False
    assert "producer-owned forge state" in result.error
    assert "forge_experiments/run_state.json" in result.error
    assert not (repo / "forge_experiments" / "rewrite_applyback").exists()


class _StoppedBackend:
    """Minimal stand-in for a registered agent backend."""

    name = "fake"

    def __init__(self, end_reason: str):
        self._end_reason = end_reason
        self.runtime = type("_Runtime", (), {"model": "fake-model"})()

    async def run(self, _run_spec):
        from kernelforge.agent_backends.base import AgentRunResult

        return AgentRunResult(end_reason=self._end_reason)


@pytest.mark.parametrize("end_reason", ["turn_cap", "sdk_error", ""])
def test_an_abnormal_agent_end_is_not_mistaken_for_a_finished_integration(
    tmp_path,
    monkeypatch,
    available_agent_provider,
    end_reason,
):
    # A turn cap or an SDK error leaves the worktree at whatever partial state
    # the agent reached -- routinely "kernel swapped, dispatch not yet rewired",
    # which passes host validation and every gate after it. Only the end reason
    # separates that from a finished integration.
    repo, _base, spec = _repo_spec(tmp_path)
    monkeypatch.setattr(
        applyback,
        "create_registered_backend",
        lambda *_args, **_kwargs: _StoppedBackend(end_reason),
    )

    with pytest.raises(RuntimeError, match="did not finish normally"):
        asyncio.run(
            applyback._run_agent(
                spec=spec,
                config=Config.from_env(workspace=str(repo), agent_precheck=False),
                worktree=repo,
                reference_path=Path(spec.flydsl_kernel),
                framework="vllm",
                source_relative="softmax.py",
                timeout_sec=60,
                progress_log=[],
            )
        )


def test_a_normal_agent_end_reports_the_backend_that_ran(
    tmp_path,
    monkeypatch,
    available_agent_provider,
):
    repo, _base, spec = _repo_spec(tmp_path)
    monkeypatch.setattr(
        applyback,
        "create_registered_backend",
        lambda *_args, **_kwargs: _StoppedBackend("agent_stopped"),
    )

    backend_name, backend_model = asyncio.run(
        applyback._run_agent(
            spec=spec,
            config=Config.from_env(workspace=str(repo), agent_precheck=False),
            worktree=repo,
            reference_path=Path(spec.flydsl_kernel),
            framework="vllm",
            source_relative="softmax.py",
            timeout_sec=60,
            progress_log=[],
        )
    )

    assert (backend_name, backend_model) == ("fake", "fake-model")


@pytest.mark.parametrize(
    "leaked",
    [
        "forge_experiments/events.jsonl",
        ".forge_rewrite/attempt/kernel.py",
        ".forge_driver_9182.py",
    ],
)
def test_host_validation_rejects_every_producer_owned_pattern(tmp_path, leaked):
    repo, _base, _spec = _repo_spec(tmp_path)
    (repo / "kernel.py").unlink()
    (repo / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
    leak = repo / leaked
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_text("forge state\n")

    with pytest.raises(RuntimeError, match="producer-owned forge state"):
        applyback._validate_worktree_changes(worktree=repo, timeout_sec=60)


def test_host_validation_publishes_a_framework_file_with_a_similar_name(tmp_path):
    repo, _base, _spec = _repo_spec(tmp_path)
    (repo / "kernel.py").unlink()
    lookalike = repo / "framework" / "forge_experiments_reader.py"
    lookalike.parent.mkdir(parents=True)
    lookalike.write_text("READER = True\n")

    changed = applyback._validate_worktree_changes(worktree=repo, timeout_sec=60)

    assert changed == ["framework/forge_experiments_reader.py"]


def test_host_validation_rejects_forge_state_created_by_a_hook(tmp_path, monkeypatch):
    repo, _base, _spec = _repo_spec(tmp_path)
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n")
    _git(repo, "add", ".pre-commit-config.yaml")
    _git(repo, "commit", "-qm", "add pre-commit config")
    (repo / "kernel.py").unlink()
    (repo / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")

    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] != "/fake/pre-commit":
            return real_run(command, *args, **kwargs)
        cache = repo / ".forge_driver_cache"
        cache.write_text("hook output\n")
        return subprocess.CompletedProcess(command, 0, stdout="passed", stderr="")

    monkeypatch.setattr(applyback.shutil, "which", lambda name: "/fake/pre-commit")
    monkeypatch.setattr(applyback.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="producer-owned forge state"):
        applyback._validate_worktree_changes(worktree=repo, timeout_sec=60)


def test_host_validation_rejects_a_hook_that_reverts_every_change(
    tmp_path,
    monkeypatch,
):
    repo, _base, _spec = _repo_spec(tmp_path)
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n")
    _git(repo, "add", ".pre-commit-config.yaml")
    _git(repo, "commit", "-qm", "add pre-commit config")
    (repo / "kernel.py").unlink()
    (repo / "softmax.py").write_text("def softmax(x):\n return x\n")

    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] != "/fake/pre-commit":
            return real_run(command, *args, **kwargs)
        (repo / "softmax.py").write_text("def softmax(x):\n    return x\n")
        return subprocess.CompletedProcess(command, 0, stdout="passed", stderr="")

    monkeypatch.setattr(applyback.shutil, "which", lambda name: "/fake/pre-commit")
    monkeypatch.setattr(applyback.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="reverted every repository change"):
        applyback._validate_worktree_changes(worktree=repo, timeout_sec=60)


def test_applyback_shell_hook_enforces_convergence_budget():
    hooks = applyback._make_applyback_hooks(
        deadline_monotonic=time.monotonic() + 240,
    )
    callback = hooks.pre_tool_use[0].callback

    benchmark = asyncio.run(
        callback(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "python benchmark.py"},
            },
            None,
            None,
        )
    )
    assert benchmark["hookSpecificOutput"]["permissionDecision"] == "deny"

    too_long = asyncio.run(
        callback(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": "timeout 900 pytest tests/unit/test_op.py",
                    "timeout": 1_000_000,
                },
            },
            None,
            None,
        )
    )
    assert too_long["hookSpecificOutput"]["permissionDecision"] == "deny"

    focused = asyncio.run(
        callback(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": "pytest tests/unit/test_op.py -q",
                    "timeout": 60_000,
                },
            },
            None,
            None,
        )
    )
    assert focused == {}

    finalizing_callback = (
        applyback._make_applyback_hooks(
            deadline_monotonic=time.monotonic() + 60,
        )
        .pre_tool_use[0]
        .callback
    )
    late_read = asyncio.run(
        finalizing_callback(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "operator.py"},
            },
            None,
            None,
        )
    )
    assert late_read["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_applyback_host_validation_rechecks_formatter_fixes(
    tmp_path,
    monkeypatch,
):
    repo, _base, _spec = _repo_spec(tmp_path)
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n")
    _git(repo, "add", ".pre-commit-config.yaml")
    _git(repo, "commit", "-qm", "add pre-commit config")
    (repo / "kernel.py").unlink()
    (repo / "softmax.py").write_text("def softmax(x):\n return x + 1\n")

    real_run = subprocess.run
    precommit_calls = []

    def fake_run(command, *args, **kwargs):
        if command[0] != "/fake/pre-commit":
            return real_run(command, *args, **kwargs)
        precommit_calls.append(command)
        if len(precommit_calls) == 1:
            (repo / "softmax.py").write_text("def softmax(x):\n    return x + 1\n")
            return subprocess.CompletedProcess(command, 1, stdout="fixed", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="passed", stderr="")

    monkeypatch.setattr(applyback.shutil, "which", lambda name: "/fake/pre-commit")
    monkeypatch.setattr(applyback.subprocess, "run", fake_run)

    changed = applyback._validate_worktree_changes(
        worktree=repo,
        timeout_sec=60,
    )

    assert changed == ["softmax.py"]
    assert len(precommit_calls) == 2
    assert _git(repo, "diff", "--cached", "--check") == ""
