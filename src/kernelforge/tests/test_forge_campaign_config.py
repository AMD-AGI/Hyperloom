"""Tests for immutable Forge campaign configuration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace

import pytest

from kernelforge.knowledge.kernel_identity import kernel_recipe_canonical_id
from kernelforge.knowledge.loop_identity import resolve_loop_identity
from kernelforge.loop.campaign_config import (
    CampaignConfig,
    CampaignConfigStore,
    create_campaign_config,
    derive_campaign_implementation_contract,
    detect_gpu_target,
    infer_kernel_backend,
    resolve_kernel_backend_override,
    validate_pending_campaign_head,
)
from kernelforge.llm.git import GitError


def _git_workspace(tmp_path, name="workspace"):
    workspace = tmp_path / name
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "-b", "feature/test-campaign"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "KernelForge Tests"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
    )
    kernel = workspace / "src" / "kernel.py"
    helper = workspace / "src" / "helper.py"
    driver = workspace / "driver.py"
    kernel.parent.mkdir()
    kernel.write_text("import triton\n\n@triton.jit\ndef fused_kernel(x):\n    return x\n")
    helper.write_text("VALUE = 1\n")
    driver.write_text("pass\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace, kernel, helper, driver


def test_create_save_load_normalizes_and_persists_campaign(tmp_path, monkeypatch):
    workspace, kernel, helper, driver = _git_workspace(tmp_path)
    program = tmp_path / "program.md"
    program.write_text("# Optimize fused kernel\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[str(kernel), str(helper)],
        program_md_file=str(program),
        operator_name="fused",
        gpu_type="MI300X",
    )
    store = CampaignConfigStore(str(workspace))
    store.save(config, program_md=program.read_text())
    loaded = store.load()

    assert loaded == config
    assert loaded.kernel_path == "src/kernel.py"
    assert loaded.driver_path == "driver.py"
    assert loaded.source_files == ["src/kernel.py", "src/helper.py"]
    assert loaded.git_branch == "feature/test-campaign"
    assert loaded.base_commit
    assert loaded.gpu_target == "gfx950"
    assert loaded.gpu_type == "mi300x"
    assert loaded.kernel_backend == "triton"
    assert loaded.task_type == "repository"
    assert "fused_kernel" in loaded.target_functions
    assert loaded.operator_name == "fused"
    assert len(loaded.implementation_signature) == 64
    assert loaded.implementation_identity["implementation_symbols"] == ["fused_kernel"]
    assert loaded.program_md_path == "forge_experiments/program.md"
    assert loaded.program_md_sha256 == hashlib.sha256(program.read_bytes()).hexdigest()
    assert (workspace / loaded.program_md_path).read_text() == program.read_text()
    assert store.read_program_md(loaded) == program.read_text()


def test_the_operator_is_settled_before_the_loop_can_rename_it(tmp_path, monkeypatch):
    """The address must not move when the loop writes its first GPU kernel.

    A run that turns eager code into a kernel would otherwise file its result
    under the name of the kernel it just invented, at an address no read
    resolves to, and the write would report success while the experience became
    unreachable.
    """
    workspace, kernel, _helper, driver = _git_workspace(tmp_path, "eager")
    # Eager source declares no GPU kernel at all, so the operator can only come
    # from the entry point the driver calls.
    kernel.write_text("def dynamic_quant(x):\n    return x / x.abs().max()\n")
    program = tmp_path / "eager-program.md"
    program.write_text("# Optimize dynamic_quant\n")
    _commit_paths(workspace, "eager baseline")
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[str(kernel)],
        program_md_file=str(program),
        target_functions=["dynamic_quant"],
        kernel_backend="triton",
    )

    assert config.operator_name == "dynamic_quant"
    store = CampaignConfigStore(str(workspace))
    store.save(config, program_md=program.read_text())

    # The loop now does what it exists to do: it writes a kernel, under a name
    # nobody declared. Neither the campaign nor the address may follow it.
    optimized = "import triton\n\n\n@triton.jit\ndef _partial_amax_kernel(x):\n    return x\n"
    kernel.write_text(optimized)

    assert store.load().operator_name == "dynamic_quant"

    def address(source: str) -> str:
        identity, _op, _fw = resolve_loop_identity(
            kernel_path=str(kernel),
            kernel_source=source,
            kernel_backend="triton",
            gpu_type="mi355x",
            target_functions=list(config.target_functions),
            framework=config.framework,
            operator_name=config.operator_name,
        )
        return kernel_recipe_canonical_id(identity)

    # What a later run reads with, and what this run writes with.
    assert address("def dynamic_quant(x):\n    return x\n") == address(optimized)
    assert ":dynamic_quant:" in address(optimized)


def _commit_paths(workspace, message):
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=workspace,
        check=True,
        capture_output=True,
    )


def test_campaign_infers_direct_source_owner_before_signature(tmp_path, monkeypatch):
    workspace, _kernel, _helper, driver = _git_workspace(tmp_path)
    kernel = workspace / "vllm" / "ops" / "direct.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n\n@triton.jit\ndef direct_kernel(x):\n    return x\n")
    _commit_paths(workspace, "add direct vllm kernel")
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )

    assert config.framework == "vllm"
    assert config.implementation_identity["source_paths"] == ["vllm/ops/direct.py"]


def test_campaign_infers_owner_from_cross_package_defining_file(
    tmp_path,
    monkeypatch,
):
    workspace, _kernel, _helper, driver = _git_workspace(tmp_path)
    anchor = workspace / "vllm" / "attention" / "entry.py"
    defining = workspace / "aiter" / "ops" / "triton" / "attention.py"
    anchor.parent.mkdir(parents=True)
    defining.parent.mkdir(parents=True)
    anchor.write_text("def attention_entry(x):\n    return unified_attention_kernel(x)\n")
    defining.write_text("import triton\n\n@triton.jit\ndef unified_attention_kernel(x):\n    return x\n")
    _commit_paths(workspace, "add cross-package kernel")
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(anchor),
        driver=str(driver),
        source_files=[str(defining)],
        program_md_file=None,
    )

    assert config.framework == "aiter"
    assert "unified_attention_kernel" in config.target_functions
    assert "aiter/ops/triton/attention.py" in config.implementation_identity["source_paths"]
    assert all(path.startswith("aiter/") for path in config.implementation_identity["source_paths"])


def test_campaign_explicit_framework_overrides_defining_path(
    tmp_path,
    monkeypatch,
):
    workspace, _kernel, _helper, driver = _git_workspace(tmp_path)
    kernel = workspace / "aiter" / "ops" / "explicit.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n\n@triton.jit\ndef explicit_kernel(x):\n    return x\n")
    _commit_paths(workspace, "add explicit override kernel")
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
        framework="vllm",
    )

    assert config.framework == "vllm"
    assert config.implementation_identity["source_paths"] == ["vllm/aiter/ops/explicit.py"]


def test_campaign_persists_unknown_when_source_owner_is_unrecognized(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    store = CampaignConfigStore(str(workspace))
    store.save(config)

    assert config.framework == "unknown"
    assert store.load().framework == "unknown"
    assert config.implementation_identity["source_paths"] == ["kernel.py"]


@pytest.mark.parametrize("staged", [False, True])
def test_fresh_campaign_rejects_tracked_changes_before_recording_base(
    tmp_path,
    monkeypatch,
    staged,
):
    workspace, kernel, helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    helper.write_text("VALUE = 2\n")
    if staged:
        subprocess.run(["git", "add", str(helper)], cwd=workspace, check=True)

    with pytest.raises(ValueError, match="uncommitted tracked changes"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[str(helper)],
            program_md_file=None,
        )
    with pytest.raises(ValueError, match="uncommitted tracked changes"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[str(helper)],
            program_md_file=None,
            base_commit="legacy-base",
        )


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_program_context_must_match_persisted_digest(
    tmp_path,
    monkeypatch,
    mutation,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    program = tmp_path / "program.md"
    program.write_text("# Optimize fused kernel\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=str(program),
    )
    store = CampaignConfigStore(str(workspace))
    store.save(config, program_md=program.read_text())

    if mutation == "missing":
        store.program_path.unlink()
    else:
        store.program_path.write_text("# Changed task\n")

    with pytest.raises(ValueError, match="program context"):
        store.read_program_md(config)


def test_store_rejects_replacing_immutable_campaign_config(tmp_path, monkeypatch):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    store = CampaignConfigStore(str(workspace))
    store.save(config)

    with pytest.raises(ValueError, match="immutable"):
        store.save(replace(config, snr_threshold=40.0))


def test_store_rejects_future_schema(tmp_path):
    root = tmp_path / "forge_experiments"
    root.mkdir()
    (root / "campaign_config.json").write_text(json.dumps({"schema_version": 999}))

    with pytest.raises(ValueError, match="schema"):
        CampaignConfigStore(str(tmp_path)).load()


def test_store_rejects_unknown_campaign_fields(tmp_path, monkeypatch):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    store = CampaignConfigStore(str(workspace))
    store.root.mkdir()
    payload = config.to_dict()
    payload["removed_field"] = "unsupported"
    store.path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported campaign config fields"):
        store.load()


def test_store_rejects_non_authoritative_schema_two(tmp_path, monkeypatch):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    store = CampaignConfigStore(str(workspace))
    store.root.mkdir()
    payload = config.to_dict()
    payload["schema_version"] = 2
    store.path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported campaign config schema"):
        store.load()


def test_infer_kernel_backend_requires_unambiguous_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_KERNEL_BACKEND", raising=False)
    triton_kernel = tmp_path / "triton_kernel.py"
    triton_kernel.write_text("import triton\n@triton.jit\ndef kernel():\n    pass\n")
    hip_kernel = tmp_path / "kernel.hip"
    hip_kernel.write_text('extern "C" __global__ void kernel() {}\n')
    unknown_kernel = tmp_path / "kernel.py"
    unknown_kernel.write_text("def kernel():\n    pass\n")

    assert infer_kernel_backend([triton_kernel]) == "triton"
    assert infer_kernel_backend([hip_kernel]) == "hip"
    with pytest.raises(ValueError, match="infer"):
        infer_kernel_backend([unknown_kernel])


def test_infer_kernel_backend_falls_back_from_unknown_environment_override(monkeypatch):
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "tilelang")

    assert infer_kernel_backend([]) == "flydsl"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("hip", "hip"),
        ("hip", "hip"),
        ("triton", "triton"),
        ("tilelang", "flydsl"),
        ("tilelang", "flydsl"),
    ],
)
def test_resolve_kernel_backend_override(requested, expected):
    assert resolve_kernel_backend_override(requested) == expected


@pytest.mark.parametrize("kernel_backend", ["tilelang", "tilelang"])
def test_create_campaign_falls_back_from_unsupported_kernel_backend(
    tmp_path,
    monkeypatch,
    kernel_backend,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
        kernel_backend=kernel_backend,
    )

    assert config.kernel_backend == "flydsl"


def test_external_driver_is_accepted_and_stored_absolute(tmp_path, monkeypatch):
    """A driver outside the workspace must not be rejected at config time.

    Task preparation stages/publishes external drivers transactionally, so the
    fresh-campaign CLI has to let that path through; rejecting it here aborted
    the run before prep could stage anything ("driver must be inside workspace").
    """
    workspace, kernel, helper, _ = _git_workspace(tmp_path)
    external = tmp_path / "forge-run" / "forge_autogen_driver.py"
    external.parent.mkdir()
    external.write_text("pass\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(external),
        source_files=[str(kernel)],
        program_md_file=None,
    )

    assert config.driver_path == external.resolve().as_posix()
    # The digest must be of the external file, and `workspace / driver_path`
    # (how every consumer rebuilds the path) must still land on it.
    assert config.driver_sha256 == hashlib.sha256(external.read_bytes()).hexdigest()
    assert (workspace / config.driver_path).resolve() == external.resolve()


def test_missing_external_driver_still_fails(tmp_path, monkeypatch):
    workspace, kernel, _, _ = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    with pytest.raises(ValueError, match="driver is not a file"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(tmp_path / "nope" / "driver.py"),
            source_files=[str(kernel)],
            program_md_file=None,
        )


def test_external_source_file_is_still_rejected(tmp_path, monkeypatch):
    """Only the driver gets the external allowance — sources stay git-tracked."""
    workspace, kernel, _, driver = _git_workspace(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 2\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    with pytest.raises(ValueError, match="source file must be inside workspace"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[str(kernel), str(outside)],
            program_md_file=None,
        )


def _campaign_payload(tmp_path, monkeypatch, name="payload"):
    """A freshly created, valid campaign config as its persisted dict."""
    workspace, kernel, _helper, driver = _git_workspace(tmp_path, name)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    return config.to_dict()


_DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"program_md_path": "forge_experiments/program.md"}, "digest is missing"),
        ({"program_md_sha256": _DIGEST}, "path is missing"),
        ({"driver_sha256": ""}, "canonical driver digest"),
        ({"driver_sha256": "not-a-digest"}, "canonical driver digest"),
        ({"snr_threshold": 0.0}, "positive finite"),
        ({"snr_threshold": float("inf")}, "positive finite"),
        ({"implementation_signature": ""}, "signature is missing or invalid"),
        ({"implementation_signature": _DIGEST}, "does not match its signature"),
        ({"implementation_identity": {}}, "does not match its signature"),
    ],
)
def test_from_dict_rejects_incoherent_campaign_snapshot(
    tmp_path,
    monkeypatch,
    mutation,
    match,
):
    """Resuming on a half-valid snapshot would measure a different campaign."""
    payload = _campaign_payload(tmp_path, monkeypatch)
    payload.update(mutation)

    with pytest.raises(ValueError, match=match):
        CampaignConfig.from_dict(payload)


def test_from_dict_requires_a_json_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        CampaignConfig.from_dict([{"schema_version": 6}])


def test_from_dict_rejects_the_retired_pre_rename_key(tmp_path, monkeypatch):
    """The old backend key is a hard error now, not a silent migration.

    ``from_dict`` rejects unknown fields on purpose. A config carrying the old
    key therefore refuses to load and names the field, which is the outcome we
    want once the back-compat shim is gone: the operator is told what to edit
    rather than watching the campaign resume on the fallback backend.
    """
    payload = _campaign_payload(tmp_path, monkeypatch)
    retired_key = "fel" + "low"
    payload[retired_key] = payload.pop("kernel_backend")

    with pytest.raises(ValueError, match="unsupported campaign config fields"):
        CampaignConfig.from_dict(payload)


def test_from_dict_round_trips_measurement_semantics(tmp_path, monkeypatch):
    """nproc/bench_repeat decide what a number MEANS, so they must survive."""
    payload = _campaign_payload(tmp_path, monkeypatch)
    payload["nproc_per_node"] = 4
    payload["bench_repeat"] = 25

    restored = CampaignConfig.from_dict(payload)

    assert (restored.nproc_per_node, restored.bench_repeat) == (4, 25)
    assert CampaignConfig.from_dict(restored.to_dict()) == restored
    # Absent/zero values clamp up to one rank and one shot, never to zero.
    payload["nproc_per_node"] = 0
    del payload["bench_repeat"]
    clamped = CampaignConfig.from_dict(payload)
    assert (clamped.nproc_per_node, clamped.bench_repeat) == (1, 1)


@pytest.mark.parametrize(
    ("content", "error", "match"),
    [
        (None, FileNotFoundError, "campaign config not found"),
        ("{not json", ValueError, "invalid campaign config"),
        ("[1, 2]", ValueError, "must be a JSON object"),
    ],
)
def test_store_load_reports_unusable_campaign_files(
    tmp_path,
    content,
    error,
    match,
):
    store = CampaignConfigStore(str(tmp_path))
    if content is not None:
        store.root.mkdir()
        store.path.write_text(content)

    with pytest.raises(error, match=match):
        store.load()


@pytest.mark.parametrize(
    ("program_md", "preexisting", "match"),
    [
        (None, None, "program context content is required"),
        ("# Different task\n", None, "digest does not match"),
        ("", "# Someone else's task\n", "program context is immutable"),
    ],
)
def test_save_refuses_to_bind_the_wrong_program_context(
    tmp_path,
    monkeypatch,
    program_md,
    preexisting,
    match,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    program = tmp_path / "program.md"
    program.write_text("# Optimize fused kernel\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=str(program),
    )
    store = CampaignConfigStore(str(workspace))
    if preexisting is not None:
        store.root.mkdir(parents=True, exist_ok=True)
        store.program_path.write_text(preexisting)

    # An empty parametrization means "hand over the genuine content".
    content = program.read_text() if program_md == "" else program_md
    with pytest.raises(ValueError, match=match):
        store.save(config, program_md=content)

    # A rejected save must leave no campaign anchor behind.
    assert not store.exists()


def test_read_program_md_rejects_a_path_escaping_the_workspace(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    outside = tmp_path / "escape.md"
    outside.write_text("# Elsewhere\n")
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    escaping = replace(
        config,
        program_md_path="../escape.md",
        program_md_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )

    store = CampaignConfigStore(str(workspace))
    with pytest.raises(ValueError, match="escapes workspace"):
        store.read_program_md(escaping)
    # A campaign without a program context reads as empty rather than failing.
    assert store.read_program_md(config) == ""


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("GFX950", "gfx950"), (" gfx942 ", "gfx942")],
)
def test_detect_gpu_target_normalizes_the_environment_override(
    monkeypatch,
    configured,
    expected,
):
    monkeypatch.setenv("GPU_TARGET", configured)

    assert detect_gpu_target() == expected


@pytest.mark.parametrize("configured", ["gfx", "nvidia-h100", "gfx950 gfx942"])
def test_detect_gpu_target_rejects_a_malformed_override(monkeypatch, configured):
    monkeypatch.setenv("GPU_TARGET", configured)

    with pytest.raises(ValueError, match="invalid GPU_TARGET"):
        detect_gpu_target()


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "  Name:  gfx942\n  Name: gfx942\n", "gfx942"),
        (0, "Name: gfx942\nName: gfx90a\n", None),
        (0, "no amd device here\n", None),
        (1, "Name: gfx942\n", None),
    ],
)
def test_detect_gpu_target_requires_exactly_one_architecture(
    monkeypatch,
    returncode,
    stdout,
    expected,
):
    """An ambiguous or absent rocminfo answer must not be guessed at."""
    monkeypatch.delenv("GPU_TARGET", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["rocminfo"],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        ),
    )

    if expected is None:
        with pytest.raises(ValueError, match="exactly one GPU target"):
            detect_gpu_target()
    else:
        assert detect_gpu_target() == expected


def test_detect_gpu_target_reports_a_missing_rocminfo(monkeypatch):
    monkeypatch.delenv("GPU_TARGET", raising=False)

    def _missing(*args, **kwargs):
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(subprocess, "run", _missing)

    with pytest.raises(ValueError, match="ensure rocminfo is available"):
        detect_gpu_target()


@pytest.mark.parametrize("branch", ["main", "master"])
def test_fresh_campaign_requires_a_development_branch(tmp_path, monkeypatch, branch):
    """The loop rewrites tracked sources, so it may not sit on the trunk."""
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    subprocess.run(["git", "branch", "-m", branch], cwd=workspace, check=True)
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    with pytest.raises(ValueError, match="non-main development branch"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[],
            program_md_file=None,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_fresh_campaign_rejects_a_meaningless_snr_threshold(
    tmp_path,
    monkeypatch,
    threshold,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    with pytest.raises(ValueError, match="positive finite float"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[],
            program_md_file=None,
            snr_threshold=threshold,
        )


def test_fresh_campaign_rejects_a_program_context_that_is_not_a_file(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    with pytest.raises(ValueError, match="program context is not a file"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[],
            program_md_file=str(tmp_path / "absent-program.md"),
        )


def test_workspace_relative_inputs_resolve_like_absolute_ones(tmp_path, monkeypatch):
    """Callers may pass paths relative to the workspace, or a directory by mistake."""
    workspace, kernel, helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    relative = create_campaign_config(
        workspace_dir=str(workspace),
        kernel="src/kernel.py",
        driver="driver.py",
        source_files=["src/helper.py"],
        program_md_file=None,
    )
    absolute = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[str(helper)],
        program_md_file=None,
    )

    assert relative == absolute
    assert relative.source_files == ["src/kernel.py", "src/helper.py"]
    with pytest.raises(ValueError, match="kernel is not a file"):
        create_campaign_config(
            workspace_dir=str(workspace),
            kernel="src",
            driver="driver.py",
            source_files=[],
            program_md_file=None,
        )


@pytest.mark.parametrize(
    ("filename", "source", "expected"),
    [
        ("gemm.py", "import hipblaslt\n@triton.jit\ndef k(): pass\n", "hipblaslt"),
        ("attn.py", "import aiter\n@triton.jit\ndef k(): pass\n", "aiter"),
        ("dsl.py", "from cutlass import cute\n@triton.jit\n", "flydsl"),
        ("ck_op.cpp", "#include <composable_kernel/foo.hpp>\n", "ck"),
        ("plain.cu", "__global__ void k() {}\n", "hip"),
    ],
)
def test_infer_kernel_backend_prefers_the_more_specific_backend(
    tmp_path,
    monkeypatch,
    filename,
    source,
    expected,
):
    """Backend detection is ordered; a generic marker must not win over a library."""
    monkeypatch.delenv("FORGE_KERNEL_BACKEND", raising=False)
    path = tmp_path / filename
    path.write_text(source)

    assert infer_kernel_backend([path]) == expected


def test_infer_kernel_backend_falls_back_to_the_path_when_content_is_unreadable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("FORGE_KERNEL_BACKEND", raising=False)
    unreadable = tmp_path / "aiter" / "ops"
    unreadable.mkdir(parents=True)

    assert infer_kernel_backend([unreadable]) == "aiter"


def test_implementation_contract_is_rederived_from_the_pristine_lineage(
    tmp_path,
    monkeypatch,
):
    """A resume re-derives the contract; the loop's own edits must not move it."""
    workspace, kernel, _helper, driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")
    config = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    # The loop does its job and rewrites the kernel under a brand-new symbol.
    kernel.write_text("import triton\n\n@triton.jit\ndef rewritten_kernel(x):\n    return x\n")

    signature, identity = derive_campaign_implementation_contract(
        workspace_dir=str(workspace),
        kernel_path=config.kernel_path,
        source_files=config.source_files,
        framework=config.framework,
        base_commit=config.base_commit,
    )

    assert signature == config.implementation_signature
    assert identity == config.implementation_identity
    # Without the lineage there is nothing pristine to read, so the working tree
    # wins -- which is exactly why the campaign snapshots base_commit.
    drifted, _identity = derive_campaign_implementation_contract(
        workspace_dir=str(workspace),
        kernel_path=config.kernel_path,
        source_files=config.source_files,
        framework=config.framework,
    )
    assert drifted != signature


def test_pending_campaign_head_accepts_only_its_own_lineage(tmp_path, monkeypatch):
    """A pending retry may advance by one `kb warm-start:` commit and no further."""
    workspace, _kernel, helper, _driver = _git_workspace(tmp_path)
    monkeypatch.setenv("GPU_TARGET", "gfx950")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    validate_pending_campaign_head(str(workspace), base)

    helper.write_text("VALUE = 2\n")
    _commit_paths(workspace, "KB warm-start: seed prior experience")
    validate_pending_campaign_head(str(workspace), base)

    helper.write_text("VALUE = 3\n")
    _commit_paths(workspace, "unrelated work")
    with pytest.raises(ValueError, match="pending campaign HEAD mismatch"):
        validate_pending_campaign_head(str(workspace), base)

    with pytest.raises(GitError, match="git rev-parse .* failed"):
        validate_pending_campaign_head(str(workspace), "no-such-commit")
