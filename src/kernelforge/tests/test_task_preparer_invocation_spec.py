"""Tests for invocation-spec assisted task preparation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from kernelforge.agent_backends.base import AgentRuntimeConfig
from kernelforge.cli import main
from kernelforge.loop import task_preparer


@pytest.mark.asyncio
async def test_preflight_requires_complete_profiling_contract(
    tmp_path,
    monkeypatch,
):
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    captured = {}

    async def fake_correctness(**_kwargs):
        return {"passed": True, "snr_db": 60.0, "message": "ok"}

    async def fake_bench(**_kwargs):
        return {
            "success": True,
            "median_ms": 1.0,
            "message": "ok",
            "case_times": {"case_001": 1.0},
        }

    async def fake_graph(*_args, **_kwargs):
        return 10, ""

    async def fake_profile(path, **_kwargs):
        captured["driver"] = path
        return True, "verified"

    monkeypatch.setattr(task_preparer, "test_correctness", fake_correctness)
    monkeypatch.setattr(task_preparer, "bench_wallclock", fake_bench)
    monkeypatch.setattr(task_preparer, "_count_graph_replays", fake_graph)
    monkeypatch.setattr(task_preparer, "_check_profile_contract", fake_profile)

    result = await task_preparer._preflight_async(
        str(driver),
        30.0,
        3,
        10,
        require_graph=True,
        require_profile=True,
    )

    assert result.ok is True
    assert result.profile_ok is True
    assert captured == {"driver": str(driver)}


def _passing_stages(monkeypatch, case_times: dict[str, float]):
    """Wire preflight's stages so only the per-case bench result matters."""

    async def fake_correctness(**_kwargs):
        return {"passed": True, "snr_db": 60.0, "message": "ok"}

    async def fake_bench(**_kwargs):
        return {
            "success": True,
            "median_ms": 1.0,
            "message": "BENCH: mean=1.0 ms",
            "case_times": dict(case_times),
        }

    monkeypatch.setattr(task_preparer, "test_correctness", fake_correctness)
    monkeypatch.setattr(task_preparer, "bench_wallclock", fake_bench)


@pytest.mark.asyncio
async def test_preflight_rejects_driver_that_skips_declared_cases(
    tmp_path,
    monkeypatch,
):
    """The declared suite is the contract; a subset of it is not conforming."""
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    _passing_stages(monkeypatch, {"case_001": 1.0})

    result = await task_preparer._preflight_async(
        str(driver),
        30.0,
        3,
        10,
        expected_case_ids=["case_001", "case_002"],
    )

    assert result.ok is False
    assert result.bench_ok is False
    assert any("case_002" in reason for reason in result.reasons)
    assert result.details["bench"]["missing_cases"] == ["case_002"]


@pytest.mark.asyncio
async def test_preflight_rejects_driver_that_benchmarks_undeclared_cases(
    tmp_path,
    monkeypatch,
):
    """The declared suite is the contract in both directions.

    A driver that measures cases the task never declared is certified here and
    then scored on all of them: the baseline takes the case table from what the
    driver prints, so the extra cases enter the mean the KEEP/REVERT decision is
    made against. The suite being optimized is then not the suite that was asked
    for, and nothing downstream can tell.
    """
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    _passing_stages(monkeypatch, {"case_001": 1.0, "case_002": 2.0, "case_099": 3.0})

    result = await task_preparer._preflight_async(
        str(driver),
        30.0,
        3,
        10,
        expected_case_ids=["case_001", "case_002"],
    )

    assert result.ok is False
    assert result.bench_ok is False
    assert any("case_099" in reason for reason in result.reasons)
    assert result.details["bench"]["undeclared_cases"] == ["case_099"]


@pytest.mark.asyncio
async def test_preflight_accepts_the_complete_declared_suite(tmp_path, monkeypatch):
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    _passing_stages(monkeypatch, {"case_001": 1.0, "case_002": 2.0})

    result = await task_preparer._preflight_async(
        str(driver),
        30.0,
        3,
        10,
        expected_case_ids=["case_001", "case_002"],
    )

    assert result.ok is True
    assert result.bench_ok is True
    assert result.details["bench"]["case_count"] == 2


def test_declared_case_ids_reads_the_driver_contract(tmp_path):
    spec = tmp_path / "invocation_spec_gemm.json"
    spec.write_text(
        json.dumps(
            {
                "tests": {
                    "driver_contract": {
                        "case_selectors": [
                            {"CASE_ID": "case_002", "M": 1},
                            {"CASE_ID": "case_001", "M": 2},
                            {"M": 3},
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert task_preparer.declared_case_ids(spec) == ["case_001", "case_002"]
    # No spec at all is the documented "no declared suite", not a failure.
    assert task_preparer.declared_case_ids(None) == []
    assert task_preparer.declared_case_ids("") == []


def test_a_spec_that_declares_no_suite_disables_the_gate(tmp_path):
    """Distinguish a task that declares no suite from one nobody could read."""
    spec = tmp_path / "invocation_spec_gemm.json"
    spec.write_text(json.dumps({"schema_version": 1, "tests": {}}), encoding="utf-8")

    assert task_preparer.declared_case_ids(spec) == []


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("missing.json", None),
        ("corrupt.json", "{not json"),
        ("array.json", "[]"),
    ],
)
def test_an_unusable_explicit_spec_fails_instead_of_disabling_the_gate(
    tmp_path,
    name,
    contents,
):
    """Refuse to run when the operator named a suite that cannot be read.

    An empty result means "this task declares no suite", which switches the
    driver's case check off entirely. Returning it for a spec that was supplied
    and could not be used spends the whole run optimizing and scoring a case set
    nobody verified, and says so in one log line among thousands. The operator
    named the file; a name that does not resolve is an error, not a default.
    """
    spec = tmp_path / name
    if contents is not None:
        spec.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=str(spec.name)):
        task_preparer.declared_case_ids(spec)


def test_prepare_agent_does_not_receive_shell_access(tmp_path, monkeypatch):
    """Map preparation policy through the provider-neutral backend contract."""
    captured: dict[str, object] = {}

    class FakeBackend:
        """Capture one normalized preparation request."""

        capabilities = SimpleNamespace(requires_workspace_cwd=False)

        async def run(self, spec, usage=None):
            """Return one successful preparation result."""
            captured["spec"] = spec
            captured["usage"] = usage
            return SimpleNamespace(text="prepared")

    def fake_factory(runtime):
        """Capture the sandboxed runtime selected for preparation."""
        captured["runtime"] = runtime
        return FakeBackend()

    monkeypatch.setattr(task_preparer, "create_registered_backend", fake_factory)
    usage = object()
    configured = AgentRuntimeConfig(provider="codex", model="gpt-test")
    result = asyncio.run(
        task_preparer._run_prepare_agent(
            config=SimpleNamespace(
                agent_runtime=lambda: configured,
            ),
            workspace=tmp_path,
            system_prompt="system",
            prompt="prompt",
            timeout_sec=10,
            additional_dirs=[str(tmp_path / "read-only")],
            allow_shell=False,
            target_files=[str(tmp_path / "driver.py")],
            protected_files=[str(tmp_path / "kernel.py")],
            usage=usage,
        )
    )

    assert result == "prepared"
    assert captured["runtime"].sandbox_mode == configured.sandbox_mode
    spec = captured["spec"]
    assert spec.tool_policy.read is True
    assert spec.tool_policy.search is True
    assert spec.tool_policy.write is True
    assert spec.tool_policy.shell is False
    assert spec.additional_directories == [str(tmp_path / "read-only")]
    assert spec.target_files == [str(tmp_path / "driver.py")]
    assert spec.protected_globs == ["kernel.py"]
    assert spec.allow_dirty_targets is True
    assert spec.allow_untracked is True
    assert captured["usage"] is usage


def test_prepare_agent_owns_the_driver_it_is_asked_to_author(
    tmp_path,
    monkeypatch,
):
    """Declare the driver as this turn's target, never as protected state.

    Preparation exists to write the driver, and it materializes its own
    scaffolding -- the reference bundle's harness and the durable invocation
    spec -- before the agent starts. Two declarations used to contradict that
    job. Naming the driver as ``driver_script`` marked the file being authored
    as one whose content must survive the turn, so the agent's rewrite was
    reported as a protected file changed and rolled back. Judging the worktree
    against HEAD read the scaffolding as files this turn had created, which
    failed every attempt with "protected files created:
    .forge_task_reference/...". Three attempts, no driver, budget spent.
    """
    captured: dict[str, object] = {}

    class FakeBackend:
        """Capture one normalized preparation request."""

        capabilities = SimpleNamespace(requires_workspace_cwd=False)

        async def run(self, spec, usage=None):
            """Record the spec and return a successful preparation result."""
            captured["spec"] = spec
            return SimpleNamespace(text="prepared")

    monkeypatch.setattr(
        task_preparer,
        "create_registered_backend",
        lambda runtime: FakeBackend(),
    )
    driver = tmp_path / "driver.py"

    asyncio.run(
        task_preparer._run_prepare_agent(
            config=SimpleNamespace(
                agent_runtime=lambda: AgentRuntimeConfig(
                    provider="codex",
                    model="gpt-test",
                ),
            ),
            workspace=tmp_path,
            system_prompt="system",
            prompt="prompt",
            timeout_sec=10,
            target_files=[str(driver)],
            protected_files=[str(tmp_path / "graph_harness.py")],
        )
    )

    spec = captured["spec"]
    assert spec.allow_dirty_baseline is True
    assert spec.driver_script == ""
    assert spec.target_files == [str(driver)]


def test_prepare_agent_initializes_required_git_workspace(
    tmp_path,
    monkeypatch,
):
    """Initialize one private baseline for backends that require a git cwd."""
    calls = 0
    (tmp_path / "driver.py").write_text("BROKEN = True\n")

    class FakeBackend:
        """Require and verify a git-backed preparation workspace."""

        capabilities = SimpleNamespace(requires_workspace_cwd=True)

        async def run(self, _spec, usage=None):
            """Confirm the temporary baseline exists before execution."""
            nonlocal calls
            calls += 1
            assert task_preparer._git_head(tmp_path)
            return SimpleNamespace(text="prepared")

    def fake_factory(_runtime):
        """Return the git-requiring fake backend."""
        return FakeBackend()

    monkeypatch.setattr(task_preparer, "create_registered_backend", fake_factory)
    config = SimpleNamespace(
        agent_runtime=lambda: AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
        ),
    )

    first = asyncio.run(
        task_preparer._run_prepare_agent(
            config=config,
            workspace=tmp_path,
            system_prompt="system",
            prompt="prompt",
            timeout_sec=10,
            target_files=[str(tmp_path / "driver.py")],
        )
    )
    second = asyncio.run(
        task_preparer._run_prepare_agent(
            config=config,
            workspace=tmp_path,
            system_prompt="system",
            prompt="prompt",
            timeout_sec=10,
            target_files=[str(tmp_path / "driver.py")],
        )
    )

    assert first == "prepared"
    assert second == "prepared"
    assert calls == 2


@pytest.mark.parametrize(
    ("failed_command", "message"),
    [
        ("init", "initialize"),
        ("add", "stage"),
        ("commit", "commit"),
    ],
)
def test_required_git_workspace_reports_setup_failures(
    tmp_path,
    monkeypatch,
    failed_command,
    message,
):
    """Report each failed temporary Git baseline setup stage."""

    def fake_git(_workspace, *args):
        """Fail the selected setup command after a missing-repository probe."""
        if args[0] == "rev-parse":
            return 1, ""
        command = "commit" if "commit" in args else args[0]
        if command == failed_command:
            return 1, "denied"
        return 0, ""

    monkeypatch.setattr(task_preparer, "_git", fake_git)

    with pytest.raises(RuntimeError, match=message):
        task_preparer._ensure_agent_git_workspace(tmp_path)


def test_a_conforming_driver_still_gets_its_spec_beside_it(tmp_path):
    """Persist the declared spec even when preparation is skipped.

    A driver that already conforms skips preparation, and preparation is what
    placed the spec next to the driver. The spec is a durable runtime input --
    a driver that derives its cases from the task reads it while benchmarking --
    so skipping the copy leaves that driver reading whatever path the operator
    passed, on a machine and at a time nobody controls. An external spec edited
    later then silently changes the measured suite, and a resumed campaign
    measures something its own baseline never did.
    """
    from kernelforge.cli import _persist_declared_spec

    driver_dir = tmp_path / "artifacts"
    driver_dir.mkdir()
    driver = driver_dir / "driver.py"
    driver.write_text("# already conforms\n", encoding="utf-8")
    source = tmp_path / "invocation_spec_gemm.json"
    payload = {"schema_version": 1, "kernel": {"name": "gemm"}}
    source.write_text(json.dumps(payload), encoding="utf-8")

    _persist_declared_spec(str(source), str(driver))

    assert json.loads((driver_dir / "invocation_spec_gemm.json").read_text()) == payload


def test_a_refused_destination_is_reported_and_not_fatal(tmp_path, capsys):
    """Say so and carry on: the driver conformed without the spec beside it."""
    from kernelforge.cli import _persist_declared_spec

    driver_dir = tmp_path / "artifacts"
    driver_dir.mkdir()
    driver = driver_dir / "driver.py"
    driver.write_text("# already conforms\n", encoding="utf-8")
    occupied = driver_dir / "invocation_spec_gemm.json"
    occupied.write_text("the caller's own notes\n", encoding="utf-8")
    source = tmp_path / "invocation_spec_gemm.json"
    source.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    _persist_declared_spec(str(source), str(driver))

    assert "could not place" in capsys.readouterr().out
    assert occupied.read_text(encoding="utf-8") == "the caller's own notes\n"


def test_persisting_the_spec_is_optional_when_none_was_supplied(tmp_path):
    """Do nothing at all when the operator named no spec."""
    from kernelforge.cli import _persist_declared_spec

    driver = tmp_path / "driver.py"
    driver.write_text("# already conforms\n", encoding="utf-8")

    _persist_declared_spec("", str(driver))

    assert list(tmp_path.iterdir()) == [driver]


def test_forge_loop_help_exposes_invocation_spec_option():
    result = CliRunner().invoke(main, ["forge-loop", "--help"])

    assert result.exit_code == 0
    assert "--invocation-spec-file" in result.output
    assert "--deadline-unix" in result.output
    assert "--aiter-cache-max-gb" in result.output


def test_materializes_only_valid_object_specs(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    source = tmp_path / "operator.json"
    payload = {"schema_version": 1, "kernel": {"name": "scaled_gemm"}}
    source.write_text(json.dumps(payload), encoding="utf-8")

    destination, canonical = task_preparer._materialize_invocation_spec(
        str(source),
        ref_dir,
    )

    assert destination == ref_dir / task_preparer.INVOCATION_SPEC_FILENAME
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert json.loads(canonical) == payload

    source.write_text("[]", encoding="utf-8")
    destination, canonical = task_preparer._materialize_invocation_spec(
        str(source),
        ref_dir,
    )
    assert destination is None
    assert canonical == ""


def test_existing_durable_spec_with_the_same_payload_is_left_untouched(tmp_path):
    """An external bundle already carries its spec beside the driver.

    Rewriting it canonically would change bytes the external transaction guards
    as a read-only caller input, so an equivalent payload already in place is
    authoritative as-is.
    """
    durable_dir = tmp_path / "artifacts"
    durable_dir.mkdir()
    source = tmp_path / "invocation_spec_gemm.json"
    payload = {"schema_version": 1, "kernel": {"name": "gemm"}}
    source.write_text(json.dumps(payload), encoding="utf-8")
    existing = durable_dir / "invocation_spec_gemm.json"
    existing.write_text('{"kernel": {"name": "gemm"}, "schema_version": 1}', encoding="utf-8")

    destination, canonical = task_preparer._materialize_invocation_spec(
        str(source),
        durable_dir,
    )

    assert destination == existing
    assert existing.read_text(encoding="utf-8") == ('{"kernel": {"name": "gemm"}, "schema_version": 1}')
    assert canonical == '{"kernel": {"name": "gemm"}, "schema_version": 1}'


def test_a_conflicting_file_beside_the_driver_is_never_overwritten(tmp_path):
    """Refuse the destination rather than replace a caller's own file.

    The destination is the driver's directory, which belongs to the caller, and
    the name is taken from the source. A file already there holding something
    else is not this function's to replace: preparation's rollback restores the
    driver and Git-tracked state, so an untracked file overwritten here is gone
    for good. Preparation continues without the spec, which the caller already
    reports.
    """
    durable_dir = tmp_path / "artifacts"
    durable_dir.mkdir()
    source = tmp_path / "invocation_spec_gemm.json"
    source.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    occupied = durable_dir / "invocation_spec_gemm.json"
    occupied.write_text("the caller's own notes, not JSON\n", encoding="utf-8")

    destination, canonical = task_preparer._materialize_invocation_spec(
        str(source),
        durable_dir,
    )

    assert destination is None
    assert canonical == ""
    assert occupied.read_text(encoding="utf-8") == "the caller's own notes, not JSON\n"


def test_a_symlinked_destination_is_never_written_through(tmp_path):
    """Refuse a symlink rather than write to wherever it points.

    Writing through it would edit a file outside the directory this function was
    given, which nothing in preparation can restore.
    """
    durable_dir = tmp_path / "artifacts"
    durable_dir.mkdir()
    outside = tmp_path / "somebody_elses.json"
    outside.write_text('{"kernel": "not ours"}\n', encoding="utf-8")
    (durable_dir / "invocation_spec_gemm.json").symlink_to(outside)
    source = tmp_path / "invocation_spec_gemm.json"
    source.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    destination, canonical = task_preparer._materialize_invocation_spec(
        str(source),
        durable_dir,
    )

    assert destination is None
    assert canonical == ""
    assert outside.read_text(encoding="utf-8") == '{"kernel": "not ours"}\n'


def test_prepare_agent_gets_the_spec_inline_and_it_is_restored(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    driver = workspace / "driver.py"
    source_spec = tmp_path / "invocation_spec_scaled_gemm.json"
    payload = {
        "schema_version": 1,
        "kernel": {"name": "scaled_gemm"},
        "invocation": {
            "arguments": [
                {"path": "args[0]", "shape": [64, 17408], "dtype": "fp8"},
            ]
        },
        "tests": {
            "driver_contract": {
                "case_selectors": [{"CASE_ID": "case_001", "M": 64}],
            },
        },
    }
    source_spec.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict = {}
    git_state = {"committed": False}
    materialized = workspace / source_spec.name

    def fake_materialize_reference(_workspace):
        ref_dir = workspace / task_preparer.REFERENCE_SUBDIR
        ref_dir.mkdir(exist_ok=True)
        (ref_dir / "CONTRACT.md").write_text("driver contract\n", encoding="utf-8")
        return ref_dir

    async def fake_agent(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        assert json.loads(materialized.read_text(encoding="utf-8")) == payload
        materialized.write_text('{"tampered": true}\n', encoding="utf-8")
        driver.write_text("# prepared driver\n", encoding="utf-8")
        return "prepared"

    async def fake_preflight(*_args, **kwargs):
        captured["expected_case_ids"] = kwargs.get("expected_case_ids")
        assert json.loads(materialized.read_text(encoding="utf-8")) == payload
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", fake_materialize_reference)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", fake_preflight)
    monkeypatch.setattr(
        task_preparer,
        "_git_head",
        lambda _workspace: "new-head" if git_state["committed"] else "base-head",
    )
    monkeypatch.setattr(task_preparer, "_git_untracked", lambda _workspace: set())
    monkeypatch.setattr(task_preparer, "_git_diff_patch", lambda *_args: "")
    monkeypatch.setattr(task_preparer, "_git_changed_since", lambda *_args: ["driver.py"])

    def fake_git(_workspace, *args):
        if args and args[0] == "commit":
            git_state["committed"] = True
        return 0, ""

    monkeypatch.setattr(task_preparer, "_git", fake_git)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model"),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(kernel)],
            preflight=task_preparer.PreflightResult(
                ok=False,
                correctness_ok=False,
                bench_ok=False,
                reasons=["driver missing"],
            ),
            invocation_spec_file=str(source_spec),
            expected_case_ids=["case_001"],
        )
    )

    assert result.ok is True
    prompt = captured["prompt"]
    assert "BUILD THE DRIVER FROM THIS" in prompt
    # Inlined, so the agent has the evidence without spending a tool call, but
    # the path stays: the driver may read the same file at runtime.
    assert "### The specification, verbatim" in prompt
    for token in ('"scaled_gemm"', '"CASE_ID": "case_001"', "17408"):
        assert token in prompt
    assert "./invocation_spec_scaled_gemm.json" in prompt
    # The durable spec has to be introduced before the temporary bundle, wherever
    # either block ends up: it is the authoritative input, so a reference-bundle
    # path above it competes for the agent's attention.
    assert prompt.index("Invocation specification") < prompt.index(task_preparer.REFERENCE_SUBDIR)
    assert captured["expected_case_ids"] == ["case_001"]
    # The specification is a durable task artifact: the driver may read it at
    # runtime, so it outlives preparation and enters the pristine commit.
    assert json.loads(materialized.read_text(encoding="utf-8")) == payload
    assert not (workspace / task_preparer.REFERENCE_SUBDIR).exists()


def test_failed_external_preparation_rolls_back_driver_helpers_and_source(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "forge_attempt"
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    driver = output_dir / "driver.py"
    driver.write_text("ORIGINAL_DRIVER\n", encoding="utf-8")
    helper = output_dir / "helper.py"
    helper.write_text("ORIGINAL_HELPER\n", encoding="utf-8")
    program = output_dir / "program.md"
    program.write_text("# Original program\n", encoding="utf-8")
    source_spec = output_dir / "invocation_spec_debug_op.json"
    source_spec.write_text('{"schema_version": 1}\n', encoding="utf-8")

    async def fake_agent(**kwargs):
        staged = Path(kwargs["workspace"])
        assert staged != output_dir
        (staged / "workspace" / "kernel.py").write_text(
            "ILLEGAL_KERNEL_EDIT\n",
            encoding="utf-8",
        )
        (staged / "driver.py").write_text("FAILED_PREP_DRIVER\n", encoding="utf-8")
        (staged / "helper.py").write_text("FAILED_HELPER_EDIT\n", encoding="utf-8")
        (staged / "new_helper.py").write_text("FAILED_NEW_HELPER\n", encoding="utf-8")
        return "attempted"

    async def failed_preflight(*_args, **_kwargs):
        return task_preparer.PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            graph_ok=False,
            reasons=["still invalid"],
        )

    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", failed_preflight)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=tmp_path / "experiments",
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(kernel)],
            invocation_spec_file=str(source_spec),
            read_only_files=[str(program), str(source_spec)],
        )
    )

    assert result.ok is False
    assert result.rolled_back is True
    assert kernel.read_text(encoding="utf-8") == "ORIGINAL_KERNEL\n"
    assert driver.read_text(encoding="utf-8") == "ORIGINAL_DRIVER\n"
    assert helper.read_text(encoding="utf-8") == "ORIGINAL_HELPER\n"
    assert not (output_dir / "new_helper.py").exists()
    assert not (workspace / task_preparer.REFERENCE_SUBDIR).exists()


def test_external_driver_uses_add_dir_without_kernel_repo_commit(tmp_path, monkeypatch):
    output_dir = tmp_path / "forge_attempt"
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    driver = output_dir / "driver.py"
    driver.write_text("BROKEN_DRIVER\n", encoding="utf-8")
    helper = output_dir / "helper.py"
    helper.write_text("ORIGINAL_HELPER\n", encoding="utf-8")
    run_log = output_dir / "artifacts" / "run.log"
    run_log.parent.mkdir()
    run_log.write_text("RUNNING\n", encoding="utf-8")
    captured: dict = {}

    async def fake_agent(**kwargs):
        staged = Path(kwargs["workspace"])
        captured["workspace"] = staged
        captured["additional_dirs"] = kwargs.get("additional_dirs")
        assert kwargs["allow_shell"] is False
        assert staged != output_dir
        assert (staged / "workspace").is_symlink()
        assert not (staged / "artifacts").exists()
        assert driver.read_text(encoding="utf-8") == "BROKEN_DRIVER\n"
        run_log.write_text("RUNNING\nMORE OUTPUT\n", encoding="utf-8")
        staged_artifacts = staged / "artifacts"
        staged_artifacts.mkdir()
        (staged_artifacts / "run.log").write_text(
            "TAMPERED STAGED LOG\n",
            encoding="utf-8",
        )
        (staged / "driver.py").write_text("PREPARED_DRIVER\n", encoding="utf-8")
        (staged / "helper.py").write_text("PREPARED_HELPER\n", encoding="utf-8")
        (staged / "new_helper.py").write_text("NEW_HELPER\n", encoding="utf-8")
        return "prepared"

    async def passing_preflight(staged_driver, *_args, **_kwargs):
        checked_driver = Path(staged_driver)
        if checked_driver == driver:
            assert driver.read_text(encoding="utf-8") == "PREPARED_DRIVER\n"
        else:
            assert checked_driver == captured["workspace"] / "driver.py"
            assert driver.read_text(encoding="utf-8") == "BROKEN_DRIVER\n"
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", passing_preflight)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=output_dir / "artifacts" / "forge_experiments",
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(kernel)],
        )
    )

    assert result.ok is True
    assert result.message == "external task prepared"
    assert captured["additional_dirs"] == [str(workspace)]
    assert driver.read_text(encoding="utf-8") == "PREPARED_DRIVER\n"
    assert helper.read_text(encoding="utf-8") == "PREPARED_HELPER\n"
    assert (output_dir / "new_helper.py").read_text(encoding="utf-8") == "NEW_HELPER\n"
    assert set(result.wrote_files) == {
        str(driver),
        str(output_dir / "graph_harness.py"),
        str(helper),
        str(output_dir / "new_helper.py"),
    }
    assert set(result.created_files) == {
        str(output_dir / "graph_harness.py"),
        str(output_dir / "new_helper.py"),
    }
    assert kernel.read_text(encoding="utf-8") == "ORIGINAL_KERNEL\n"
    assert run_log.read_text(encoding="utf-8") == "RUNNING\nMORE OUTPUT\n"
    assert (output_dir / "graph_harness.py").is_file()
    assert not (workspace / "graph_harness.py").exists()


def test_external_staging_is_discarded_when_preflight_is_cancelled(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "forge_attempt"
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    driver = output_dir / "driver.py"
    driver.write_text("BROKEN_DRIVER\n", encoding="utf-8")

    async def fake_agent(**kwargs):
        staged = Path(kwargs["workspace"])
        (staged / "driver.py").write_text("PREPARED_DRIVER\n", encoding="utf-8")
        (staged / "new_helper.py").write_text("NEW_HELPER\n", encoding="utf-8")
        return "prepared"

    async def cancelled_preflight(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(
        task_preparer,
        "_preflight_async",
        cancelled_preflight,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            task_preparer.prepare_task(
                config=SimpleNamespace(
                    model="test-model",
                    experiments_dir=tmp_path / "experiments",
                ),
                workspace_dir=str(workspace),
                kernel=str(kernel),
                driver=str(driver),
                program_md="# Task",
                target_functions=[],
                source_files=[str(kernel)],
            )
        )

    assert driver.read_text(encoding="utf-8") == "BROKEN_DRIVER\n"
    assert not (output_dir / "new_helper.py").exists()


def test_timeout_salvages_driver_when_preflight_passes(tmp_path, monkeypatch):
    output_dir = tmp_path / "forge_attempt"
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    driver = output_dir / "driver.py"
    driver.write_text("BROKEN_DRIVER\n", encoding="utf-8")

    async def timing_out_agent(**kwargs):
        staged_driver = Path(kwargs["workspace"]) / "driver.py"
        staged_driver.write_text(
            "VALID_DRIVER_WRITTEN_BEFORE_TIMEOUT\n",
            encoding="utf-8",
        )
        raise asyncio.TimeoutError

    async def passing_preflight(staged_driver, *_args, **_kwargs):
        checked_driver = Path(staged_driver)
        assert "VALID_DRIVER" in checked_driver.read_text(encoding="utf-8")
        if checked_driver != driver:
            assert driver.read_text(encoding="utf-8") == "BROKEN_DRIVER\n"
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_run_prepare_agent", timing_out_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", passing_preflight)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=tmp_path / "experiments",
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(kernel)],
        )
    )

    assert result.ok is True
    assert result.attempts == 1
    assert driver.read_text(encoding="utf-8") == "VALID_DRIVER_WRITTEN_BEFORE_TIMEOUT\n"
    audit = Path(result.audit_dir)
    assert json.loads((audit / "attempt_01" / "agent_event.json").read_text(encoding="utf-8"))["status"] == "timeout"
    assert json.loads((audit / "attempt_01" / "preflight.json").read_text(encoding="utf-8"))["ok"] is True


def test_the_note_carries_the_specification_verbatim(tmp_path):
    """Handed over whole rather than summarised.

    Every selective rendering has to decide what an absent field looks like, and
    both ways of deciding mislead: a heading over nothing claims the field is
    known and empty, while dropping the heading leaves no trace it exists. In the
    raw JSON an absent key is unambiguously absent.
    """
    spec = {
        "invocation": {
            "launcher_locator": "aiter/ops/gemm_op_a8w8.py(651): gemm_a8w8_blockscale",
            "arguments": [{"position": 0, "shape": [3118, 5120], "dtype": "fp8"}],
        },
        "tests": {"related_files": ["/sgl-workspace/aiter/op_tests/test_gemm.py"]},
        "deployment": {"batch": {"serving_concurrency": 64}},
    }
    spec_path = tmp_path / "invocation_spec_gemm.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert "### The specification, verbatim" in note
    # Every field reaches the agent, including ones no extractor selected for.
    assert json.dumps(spec, indent=2) in note
    assert "serving_concurrency" in note
    assert "gemm_a8w8_blockscale" in note


def test_the_note_demands_the_deployment_shapes_not_a_toy_size(tmp_path):
    """A kernel tuned at a size the workload never serves can report a large
    speedup that disappears end to end; that is the failure this text targets.
    """
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text(json.dumps({"invocation": {"arguments": []}}), encoding="utf-8")

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert "BUILD THE DRIVER FROM THIS" in note
    assert "A toy size is not a smaller version of the real measurement" in note
    assert "do not fall back to round numbers of your own choosing" in note


def test_an_absent_field_is_shown_as_absent_rather_than_as_an_empty_one(tmp_path):
    """A graph replay has no CPU-side parent op, so the profiler records no
    arguments and the key is simply missing. The agent has to be able to see
    that it is missing, which is what the raw document gives it.
    """
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text(
        json.dumps({"schema_version": 2, "missing_fields": ["inputs"]}, indent=2),
        encoding="utf-8",
    )

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert '"missing_fields"' in note
    assert "Arguments (call order):" not in note


def test_an_oversized_spec_is_referenced_rather_than_inlined(tmp_path):
    """A prompt is the wrong place to discover a producer grew without warning."""
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text(
        json.dumps({"pad": "x" * (task_preparer._SPEC_INLINE_MAX_BYTES + 1)}),
        encoding="utf-8",
    )

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert "larger than the 64 KB inline limit" in note
    assert "Use `Read` on it before you touch the driver." in note
    assert "xxxx" not in note


def test_an_empty_spec_says_it_is_empty_rather_than_too_large(tmp_path):
    """Three states, three messages.

    ``_invocation_spec_text`` refuses for three unrelated reasons and used to
    return a bare ``""`` for all of them, so the note called an empty file too
    large. That is the same defect this branch removed from the quick
    reference -- a renderer that cannot tell absent from empty states one when
    it means the other -- reappearing one level up.
    """
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text("", encoding="utf-8")

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert "is empty, so it declares no invocation evidence at all" in note
    assert "inline limit" not in note, "an empty file is not an oversized one"
    assert "Recover the public callable" in note


def test_a_whitespace_only_spec_counts_as_empty(tmp_path):
    """The text is stripped before the size test, so it must be before the empty one."""
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text("   \n\t\n  ", encoding="utf-8")

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert "is empty" in note
    assert "inline limit" not in note


def test_a_malformed_spec_is_inlined_verbatim_rather_than_dropped(tmp_path):
    """Content is deliberately NOT validated as JSON.

    The old quick reference parsed the document and rendered nothing when
    ``json.loads`` failed, which told the agent only that no evidence arrived.
    Handing the bytes over unchanged is the point of this branch: they are the
    same bytes the driver will read at runtime, so a corrupt document is worth
    more to the agent visible than hidden. Pinned because the reasoning is not
    obvious from the code, and because "validate it" is the natural review
    instinct.
    """
    spec_path = tmp_path / "invocation_spec.json"
    spec_path.write_text('{"invocation": {"arguments": [', encoding="utf-8")

    note = task_preparer._invocation_spec_note(spec_path, tmp_path)

    assert '{"invocation": {"arguments": [' in note
    assert "The specification, verbatim" in note
    # Not mistaken for one of the refusal states.
    assert "is empty" not in note
    assert "inline limit" not in note


def test_an_unreadable_spec_still_produces_a_usable_note(tmp_path):
    """The document is evidence, not a precondition; losing it must not take the
    instruction with it.

    It must also not be described as too large, which is what a single empty
    return value made the note say.
    """
    missing = tmp_path / "gone.json"

    note = task_preparer._invocation_spec_note(missing, tmp_path)

    assert "BUILD THE DRIVER FROM THIS" in note
    assert "could not be read (FileNotFoundError)" in note
    assert "recover the public callable" in note
    assert "inline limit" not in note, "a missing file is not an oversized one"


def test_no_spec_at_all_yields_no_note(tmp_path):
    assert task_preparer._invocation_spec_note(None, tmp_path) == ""
