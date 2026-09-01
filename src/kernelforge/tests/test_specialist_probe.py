"""A planning specialist may measure one variant, in a scratch tree only.

On `dynamic-quant` the whole 12.88% deficit reduced to one geometry constant
that the specialists could only argue about, because they run read-only. These
tests pin the two halves of the fix: the probe reaches nothing but its own
scratch root, and every attempt it makes -- refused, failed, or over budget --
comes back into the analysis labelled, so an unmeasured question never reads
like one nobody asked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

import pytest

from kernelforge.llm.process_reaping import ReapReport

from kernelforge.agent_backends import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.mcp_server import probe_stdio_server as probe_server
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    OrchestrationContext,
    SpecialistAssignment,
    SpecialistDefinition,
)
from kernelforge.orchestrator import specialists
from kernelforge.orchestrator.orchestration import _specialist_probe_config
from kernelforge.orchestrator.specialists import (
    SpecialistAgent,
    SpecialistProbeConfig,
)


def _refusing_load_sandbox(env):
    """Stand in for a server that would refuse the session it was handed."""
    raise probe_server.ProbeSandboxError("the ledger volume went away")


def _patch_primitive(monkeypatch, primitive) -> None:
    """Stand PR-1's seam in or out, strictly.

    The patch targets the resolver this branch owns rather than the attribute in
    ``tools.bench``, which the earlier revision patched with ``raising=False`` --
    and so kept passing while the probe named a primitive PR-1 never wrote.
    ``test_the_seam_is_callable_the_way_the_probe_calls_it`` is what checks the
    name and the signature against whatever this build actually provides.
    """
    monkeypatch.setattr(probe_server, "resolve_probe_primitive", lambda: primitive)


def _definition() -> SpecialistDefinition:
    return SpecialistDefinition(
        role_id="memory",
        description="Memory optimization specialist",
        instructions="Analyze memory layout and cache behavior.",
        capabilities=("memory", "cache"),
    )


def _assignment() -> SpecialistAssignment:
    return SpecialistAssignment(
        assignment_id="memory-1",
        role_id="memory",
        target_case_ids=("case-a",),
        evidence_refs=(),
        reason="Decide the replicated block width.",
    )


def _context(workspace: Path) -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize the operator.",
        source_map_path="analysis/abc123/source_map.json",
        cases=(
            CaseEvidence(
                case_id="case-a",
                latency_ms=1.0,
                bottleneck="memory",
                profile_summary_path="analysis/abc123/profiles/case-a/summary.json",
            ),
        ),
    )


class _McpBackend:
    """A backend that serves MCP tools and records the spec it was given."""

    capabilities = AgentCapabilities(mcp=True)

    def __init__(self, result: str = "Widen the block.", ledger=(), corrupt: bool = False) -> None:
        self.result = result
        self.ledger = list(ledger)
        self.corrupt = corrupt
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        server = spec.mcp_servers.get("specialist_probe")
        if server is not None and self.ledger:
            # Stand in for the probe server the session would have spawned.
            path = Path(server.env[probe_server.LEDGER_ENV])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for record in self.ledger:
                    handle.write(json.dumps(record) + "\n")
                if self.corrupt:
                    handle.write('{"probe_index": 2, "label": "trun\n')
        return AgentRunResult(text=self.result, end_reason="agent_stopped")


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "canonical"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "kernel.py").write_text("BLOCK = 256\n", encoding="utf-8")
    (workspace / "driver.py").write_text("print('wall_ms: 1.0')\n", encoding="utf-8")
    return workspace


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _probe(tmp_path: Path, **kwargs) -> SpecialistProbeConfig:
    return SpecialistProbeConfig(
        scratch_root=str(tmp_path / "experiments" / "specialist_probe"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_probe_is_offered_without_reaching_the_canonical_tree(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    workspace = _workspace(tmp_path)
    before = _fingerprint(workspace)
    backend = _McpBackend(
        ledger=[
            {
                "probe_index": 1,
                "label": "block-1024-vs-256",
                "case_id": "case-a",
                "status": probe_server.MEASURED,
                "case_ms": 0.87,
                "detail": "",
            }
        ],
    )
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(workspace))

    spec = backend.specs[0]
    server = spec.mcp_servers["specialist_probe"]
    scratch = Path(server.env[probe_server.SCRATCH_ENV])

    assert spec.writable is False
    assert spec.tool_policy.write is False
    assert spec.tool_policy.shell is False
    assert spec.protected_globs == ["*"]
    assert spec.tool_policy.extra_tools == ("mcp__specialist_probe__probe_variant",)
    assert workspace not in scratch.parents and scratch != workspace
    assert scratch.is_dir()
    assert server.env[probe_server.WORKSPACE_ENV] == str(workspace.resolve())
    assert "mcp__specialist_probe__probe_variant" in spec.system_prompt
    assert _fingerprint(workspace) == before
    assert outcome.succeeded
    assert "block-1024-vs-256" in outcome.content
    assert "measured" in outcome.content
    assert "0.87 ms" in outcome.content


@pytest.mark.asyncio
async def test_unused_probe_is_distinguished_from_an_unavailable_one(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert "offered and never called" in outcome.content


@pytest.mark.asyncio
async def test_an_earlier_rounds_probes_are_not_reported_as_this_rounds(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    probe = _probe(tmp_path)
    stale = Path(probe.scratch_root) / "memory-1" / "probe_ledger.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps(
            {
                "probe_index": 1,
                "label": "last-round",
                "case_id": "case-a",
                "status": probe_server.MEASURED,
                "case_ms": 9.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent = SpecialistAgent(
        definition=_definition(),
        backend=_McpBackend(),
        timeout_sec=1800,
        max_turns=4,
        probe=probe,
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert "last-round" not in outcome.content
    assert "offered and never called" in outcome.content


@pytest.mark.asyncio
async def test_missing_measurement_primitive_is_reported_not_hidden(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, None)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert backend.specs[0].mcp_servers == {}
    assert backend.specs[0].tool_policy.extra_tools == ()
    assert probe_server.PRIMITIVE_PATH in outcome.content
    assert "argued, not measured" in outcome.content


@pytest.mark.asyncio
async def test_scratch_root_inside_the_canonical_tree_disables_the_probe(tmp_path, monkeypatch, caplog) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    workspace = _workspace(tmp_path)
    before = _fingerprint(workspace)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=SpecialistProbeConfig(scratch_root=str(workspace / "scratch")),
    )

    with caplog.at_level("WARNING"):
        outcome = await agent.run(_assignment(), _context(workspace))

    assert backend.specs[0].mcp_servers == {}
    assert _fingerprint(workspace) == before
    assert "overlaps the canonical tree" in outcome.content
    assert "specialist probe not offered for memory-1" in caplog.text


@pytest.mark.asyncio
async def test_probe_record_survives_a_failed_specialist(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend(
        result="",
        ledger=[
            {
                "probe_index": 1,
                "label": "block-1024-vs-256",
                "case_id": "case-a",
                "status": probe_server.FAILED,
                "detail": "compile error",
            }
        ],
    )
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert outcome.failure is not None
    assert outcome.failure.kind == "empty_output"
    assert "0 of 1 probe attempts measured" in outcome.failure.message


@pytest.mark.asyncio
async def test_specialist_without_probe_keeps_its_read_only_spec(tmp_path) -> None:
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    spec = backend.specs[0]
    assert spec.mcp_servers == {}
    assert spec.tool_policy.extra_tools == ()
    assert "Bounded measurement" not in spec.system_prompt
    assert "probe" not in outcome.content.lower()


@pytest.mark.asyncio
async def test_a_primitive_the_probe_cannot_call_is_reported_not_offered(tmp_path, monkeypatch, caplog) -> None:
    async def _wrong_signature(*, workspace, scratch_dir, case_id):
        raise AssertionError("the probe must not call a primitive it cannot call")

    _patch_primitive(monkeypatch, _wrong_signature)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    with caplog.at_level("WARNING"):
        outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert backend.specs[0].mcp_servers == {}
    assert "does not accept driver_script" in outcome.content
    assert "argued, not measured" in outcome.content
    assert "does not accept driver_script" in caplog.text


@pytest.mark.asyncio
async def test_a_sandbox_the_server_would_refuse_is_never_offered(tmp_path, monkeypatch, caplog) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setattr(
        specialists,
        "load_sandbox",
        _refusing_load_sandbox,
    )
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    with caplog.at_level("WARNING"):
        outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert backend.specs[0].mcp_servers == {}
    assert "No probe ran: the ledger volume went away" in outcome.content
    assert "offered and never called" not in outcome.content
    assert "specialist probe not offered" in caplog.text


@pytest.mark.asyncio
async def test_an_unreadable_probe_ledger_is_reported_as_partial(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend(
        ledger=[
            {
                "probe_index": 1,
                "label": "widen",
                "case_id": "case-a",
                "status": probe_server.MEASURED,
                "case_ms": 0.87,
            }
        ],
        corrupt=True,
    )
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert "its record is incomplete" in outcome.content
    assert "unreadable entry" in outcome.content


# --- the probe server itself ------------------------------------------------


async def _stub_primitive(*, driver_script, case_id, constants, timeout_sec, prefix_constants=True):
    """Stand in for PR-1's ``sweep_case``, with its result shape."""
    return {
        "success": True,
        "kind": "exploratory",
        "case_id": case_id,
        "case_ms": 0.87,
        "constants": {str(k): str(v) for k, v in constants.items()},
        "narrowed": True,
        "message": "SWEEP (EXPLORATORY, NOT AN ACCEPTANCE RESULT): probe ok",
    }


async def _failing_primitive(*, driver_script, case_id, constants, timeout_sec, prefix_constants=True):
    """PR-1 reports a configuration that did not run with no timing at all."""
    return {
        "success": False,
        "kind": "exploratory",
        "message": "SWEEP: CONFIGURATION DID NOT RUN (exit 1)",
    }


async def _crashing_primitive(**_kwargs):
    raise RuntimeError("hipcc exited 1")


def _sandbox(tmp_path, **overrides) -> probe_server.ProbeSandbox:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    workspace = tmp_path / "canonical"
    workspace.mkdir(exist_ok=True)
    (workspace / "driver.py").write_text("print('wall_ms: 1.0')\n", encoding="utf-8")
    defaults = {
        "scratch_root": scratch,
        "workspace": workspace,
        "ledger_path": scratch / "probe_ledger.jsonl",
        "max_probes": 2,
        "budget_sec": 60.0,
        # Every real sandbox has one, and it exists: a probe with no device
        # sentinel refuses to measure rather than timing against whatever else
        # holds the GPU, and one it created itself would serialize nothing.
        "device_lock": tmp_path / "device.lock",
    }
    (tmp_path / "device.lock").touch()
    defaults.update(overrides)
    return probe_server.ProbeSandbox(**defaults)


def _ledger(sandbox) -> list[dict]:
    return [json.loads(line) for line in sandbox.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_server_probe_records_a_measurement_without_editing_the_workspace(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path)
    (sandbox.workspace / "kernel.py").write_text("BLOCK = 256\n", encoding="utf-8")
    before = _fingerprint(sandbox.workspace)

    result = await probe_server.probe_variant(
        {
            "label": "widen",
            "driver_script": "driver.py",
            "case_id": "case-a",
            "constants": {"BLOCK": 1024},
        },
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.MEASURED
    assert result["case_ms"] == 0.87
    assert "not an acceptance-gate result" in result["evidence"]
    assert result["kind"] == "exploratory"
    assert result["driver_script"] == str(sandbox.workspace / "driver.py")
    assert _fingerprint(sandbox.workspace) == before
    assert sandbox.ledger_path.is_relative_to(sandbox.scratch_root)
    assert _ledger(sandbox)[0]["status"] == probe_server.MEASURED


@pytest.mark.asyncio
async def test_server_passes_verbatim_names_and_records_what_was_read(tmp_path, monkeypatch) -> None:
    """A knob the source named itself is unreachable under FORGE_SWEEP_."""
    seen: dict[str, object] = {}

    async def _recording_primitive(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "kind": "exploratory",
            "case_id": kwargs["case_id"],
            "case_ms": 0.87,
            "narrowed": False,
            "case_selection": "whole_suite",
            "override_consumption": {"GPTOSS_BOUND": "unread"},
            "message": "SWEEP: unconfirmed",
        }

    _patch_primitive(monkeypatch, _recording_primitive)
    sandbox = _sandbox(tmp_path)

    result = await probe_server.probe_variant(
        {
            "label": "bound",
            "driver_script": "driver.py",
            "case_id": "case-a",
            "constants": {"GPTOSS_BOUND": 512},
            "prefix_constants": False,
        },
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert seen["prefix_constants"] is False
    assert result["status"] == probe_server.MEASURED
    assert result["case_selection"] == "whole_suite"
    # The ledger has to keep the difference between a confirmed number and one
    # nothing was seen to read.
    assert _ledger(sandbox)[0]["override_consumption"] == {"GPTOSS_BOUND": "unread"}


@pytest.mark.asyncio
async def test_server_defaults_to_the_sweep_prefix(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _recording_primitive(**kwargs):
        seen.update(kwargs)
        return await _stub_primitive(**kwargs)

    _patch_primitive(monkeypatch, _recording_primitive)

    await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=_sandbox(tmp_path),
        budget=probe_server.ProbeBudget(),
    )

    assert seen["prefix_constants"] is True


@pytest.mark.asyncio
async def test_server_refuses_a_non_boolean_prefix_constants(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)

    with pytest.raises(probe_server.InvalidParamsError):
        await probe_server.probe_variant(
            {
                "label": "widen",
                "driver_script": "driver.py",
                "case_id": "case-a",
                "prefix_constants": "false",
            },
            sandbox=_sandbox(tmp_path),
            budget=probe_server.ProbeBudget(),
        )


@pytest.mark.asyncio
async def test_server_reports_a_crashed_probe(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _crashing_primitive)
    sandbox = _sandbox(tmp_path)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.FAILED
    assert "hipcc exited 1" in result["detail"]
    assert _ledger(sandbox)[0]["status"] == probe_server.FAILED


@pytest.mark.asyncio
async def test_server_reports_an_exhausted_count_budget(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path, max_probes=1)
    budget = probe_server.ProbeBudget()

    await probe_server.probe_variant(
        {"label": "first", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )
    result = await probe_server.probe_variant(
        {"label": "second", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.BUDGET_EXHAUSTED
    assert "probe count budget of 1 is spent" in result["detail"]
    assert [record["status"] for record in _ledger(sandbox)] == [
        probe_server.MEASURED,
        probe_server.BUDGET_EXHAUSTED,
    ]


@pytest.mark.asyncio
async def test_server_reports_an_exhausted_wallclock_budget(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path, budget_sec=30.0)
    budget = probe_server.ProbeBudget(attempts=1, seconds_used=30.0)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.BUDGET_EXHAUSTED
    assert "wall-clock budget of 30s is spent" in result["detail"]


@pytest.mark.asyncio
async def test_server_reports_the_missing_primitive(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, None)
    sandbox = _sandbox(tmp_path)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.UNAVAILABLE
    assert probe_server.PRIMITIVE_PATH in result["detail"]
    assert _ledger(sandbox)[0]["status"] == probe_server.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_refused_sandbox_reaches_the_ledger_the_parent_reads(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "scratch" / "memory-1" / "probe_ledger.jsonl"
    monkeypatch.setenv(probe_server.LEDGER_ENV, str(ledger))
    monkeypatch.delenv(probe_server.SCRATCH_ENV, raising=False)
    monkeypatch.delenv(probe_server.WORKSPACE_ENV, raising=False)
    server = probe_server.ProbeServer()

    for label in ("widen", "narrow"):
        payload = await server.handle_tool_call(
            "probe_variant",
            {"label": label, "driver_script": "driver.py", "case_id": "case-a"},
        )
    result = json.loads(payload["content"][0]["text"])

    assert result["status"] == probe_server.REFUSED
    assert "probe sandbox unusable" in result["detail"]
    recorded = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["status"] for record in recorded] == [
        probe_server.REFUSED,
        probe_server.REFUSED,
    ]
    assert [record["label"] for record in recorded] == ["widen", "narrow"]


@pytest.mark.asyncio
async def test_a_refusal_with_no_ledger_to_reach_says_so(monkeypatch) -> None:
    monkeypatch.delenv(probe_server.LEDGER_ENV, raising=False)
    monkeypatch.delenv(probe_server.SCRATCH_ENV, raising=False)
    monkeypatch.delenv(probe_server.WORKSPACE_ENV, raising=False)
    server = probe_server.ProbeServer()

    payload = await server.handle_tool_call(
        "probe_variant",
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
    )
    result = json.loads(payload["content"][0]["text"])

    assert result["status"] == probe_server.REFUSED
    assert "reaches no ledger" in result["detail"]


@pytest.mark.asyncio
async def test_the_refusals_the_server_records_are_reported_as_refusals(tmp_path, monkeypatch) -> None:
    """The whole point of the fallback ledger: six refusals are not zero calls."""
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend(
        ledger=[
            {
                "probe_index": index,
                "label": f"probe-{index}",
                "case_id": "case-a",
                "status": probe_server.REFUSED,
                "detail": "probe sandbox unusable: the ledger volume went away",
            }
            for index in (1, 2)
        ],
    )
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert "offered and never called" not in outcome.content
    assert outcome.content.count("refused") == 2
    assert "0 of 2 probe attempts produced a measurement" in outcome.content


@pytest.mark.asyncio
async def test_a_driver_outside_the_workspace_is_refused_and_recorded(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path)
    outside = tmp_path / "elsewhere" / "driver.py"
    outside.parent.mkdir()
    outside.write_text("print('wall_ms: 1.0')\n", encoding="utf-8")

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": str(outside), "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.REFUSED
    assert "is not a file inside the canonical workspace" in result["detail"]
    assert _ledger(sandbox)[0]["status"] == probe_server.REFUSED


@pytest.mark.asyncio
async def test_a_primitive_that_returns_no_timing_is_a_failed_probe(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _failing_primitive)
    sandbox = _sandbox(tmp_path)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.FAILED
    assert "case_ms" not in result
    assert "CONFIGURATION DID NOT RUN" in result["detail"]
    assert _ledger(sandbox)[0]["status"] == probe_server.FAILED


@pytest.mark.asyncio
async def test_a_probe_that_overruns_its_ceiling_is_a_failed_probe(tmp_path, monkeypatch) -> None:
    async def _hanging_primitive(*, driver_script, case_id, constants, timeout_sec, prefix_constants=True):
        await asyncio.sleep(30)

    _patch_primitive(monkeypatch, _hanging_primitive)
    sandbox = _sandbox(tmp_path, budget_sec=0.01)
    budget = probe_server.ProbeBudget()

    async def _immediate(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(probe_server.asyncio, "wait_for", _immediate)
    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.FAILED
    assert "ceiling" in result["detail"]
    assert budget.attempts == 1
    assert _ledger(sandbox)[0]["status"] == probe_server.FAILED


def test_the_seam_is_callable_the_way_the_probe_calls_it() -> None:
    """Whatever this build provides under PR-1's name must take the probe's call.

    Absent, this is the branch's stated dependency and the probe says so. Present
    with a signature the probe cannot call, the second branch fails -- which is
    the check the earlier ``raising=False`` monkeypatch removed.
    """
    primitive, unusable = probe_server.probe_primitive_status()

    assert probe_server.PRIMITIVE_PATH.endswith(".sweep_case")
    if primitive is None:
        assert probe_server.PRIMITIVE_PATH in unusable
    else:
        assert unusable == ""


def test_load_sandbox_rejects_a_scratch_root_inside_the_canonical_tree(
    tmp_path,
) -> None:
    workspace = tmp_path / "canonical"
    scratch = workspace / "scratch"
    scratch.mkdir(parents=True)

    with pytest.raises(probe_server.ProbeSandboxError, match="overlaps"):
        probe_server.load_sandbox(
            {
                probe_server.SCRATCH_ENV: str(scratch),
                probe_server.WORKSPACE_ENV: str(workspace),
                probe_server.MAX_PROBES_ENV: "4",
                probe_server.BUDGET_SEC_ENV: "60",
            }
        )


def test_load_sandbox_rejects_a_missing_budget(tmp_path) -> None:
    workspace = tmp_path / "canonical"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()

    with pytest.raises(probe_server.ProbeSandboxError, match="greater than zero"):
        probe_server.load_sandbox(
            {
                probe_server.SCRATCH_ENV: str(scratch),
                probe_server.WORKSPACE_ENV: str(workspace),
                probe_server.MAX_PROBES_ENV: "0",
                probe_server.BUDGET_SEC_ENV: "60",
            }
        )


def test_probe_config_rejects_an_empty_budget() -> None:
    with pytest.raises(ValueError, match="max_probes"):
        SpecialistProbeConfig(scratch_root="/tmp/scratch", max_probes=0)


@pytest.mark.asyncio
async def test_the_default_campaign_layout_still_gets_a_usable_probe(tmp_path, monkeypatch) -> None:
    """The CLI default puts experiments_dir inside the workspace; the probe runs anyway.

    ``config.experiments_dir = campaign_root`` is ``<workspace>/forge_experiments``
    on every default campaign, and a scratch root under it is the one placement
    the probe refuses. Placing it there disabled the feature everywhere.
    """
    _patch_primitive(monkeypatch, _stub_primitive)
    workspace = _workspace(tmp_path)
    config = Config(workspace=str(workspace))
    config.experiments_dir = workspace / "forge_experiments"

    probe = _specialist_probe_config(config)

    assert probe is not None
    assert not Path(probe.scratch_root).is_relative_to(workspace)

    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=probe,
    )
    outcome = await agent.run(_assignment(), _context(workspace))

    assert backend.specs[0].mcp_servers != {}
    assert "overlaps the canonical tree" not in outcome.content


def test_factory_prefers_an_experiments_dir_the_operator_moved_out(tmp_path) -> None:
    config = Config(workspace=str(tmp_path / "canonical"))
    config.experiments_dir = tmp_path / "hyperloom_out"

    probe = _specialist_probe_config(config)

    assert probe is not None
    assert Path(probe.scratch_root).is_relative_to(tmp_path / "hyperloom_out")


def test_factory_says_when_it_cannot_place_a_scratch_root(caplog) -> None:
    with caplog.at_level("WARNING"):
        probe = _specialist_probe_config(Config(workspace=""))

    assert probe is None
    assert "specialist probe disabled" in caplog.text


# --- the environment, the device and the clock the probe actually runs under --


@pytest.mark.asyncio
async def test_the_probe_child_is_given_the_environment_its_driver_needs(tmp_path, monkeypatch) -> None:
    """The MCP client forwards a six-name allow-list, not this process's env.

    So the child would start with no import path -- this repo is not installed
    -- and the driver ``sweep_case`` re-runs would compile and dispatch with no
    ROCm and no device selection. The allow-list is forwarded explicitly, and
    only the allow-list.
    """
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setenv("PYTHONPATH", "/repo/src")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/rocm/lib")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HSA_XNACK", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/cache/triton")
    monkeypatch.setenv("SOME_UNRELATED_SECRET", "do-not-forward")
    monkeypatch.setenv(probe_server.MAX_PROBES_ENV, "999")
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path, max_probes=4),
    )

    await agent.run(_assignment(), _context(_workspace(tmp_path)))

    env = backend.specs[0].mcp_servers["specialist_probe"].env
    assert env["PYTHONPATH"] == "/repo/src"
    assert env["ROCM_PATH"] == "/opt/rocm"
    assert env["LD_LIBRARY_PATH"] == "/opt/rocm/lib"
    assert env["HIP_VISIBLE_DEVICES"] == "3"
    assert env["HSA_XNACK"] == "1"
    assert env["TRITON_CACHE_DIR"] == "/cache/triton"
    assert "SOME_UNRELATED_SECRET" not in env
    # The sandbox definition wins over anything inherited under the same name.
    assert env[probe_server.MAX_PROBES_ENV] == "4"


@pytest.mark.asyncio
async def test_a_probe_locks_the_same_device_sentinel_the_fanout_lanes_do(tmp_path, monkeypatch) -> None:
    """One GPU, one sentinel: a probe queues behind a lane and a lane behind it."""
    from kernelforge.loop.fanout import campaign_device_lock_path

    _patch_primitive(monkeypatch, _stub_primitive)
    workspace = _workspace(tmp_path)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    await agent.run(_assignment(), _context(workspace))

    env = backend.specs[0].mcp_servers["specialist_probe"].env
    assert env[probe_server.DEVICE_LOCK_ENV] == str(campaign_device_lock_path(workspace))
    assert not Path(env[probe_server.DEVICE_LOCK_ENV]).is_relative_to(workspace)


@pytest.mark.asyncio
async def test_specialist_setup_creates_the_campaign_sentinel(tmp_path, monkeypatch) -> None:
    """Naming the sentinel is not enough; the analysis round has to find it.

    A probe opens the sentinel without creating it, and the only other maker is
    the fan-out lane path, which a campaign reaches after the analysis round.
    Left uncreated here, every probe of every specialist is refused and the
    round plans against nothing it measured.
    """
    from kernelforge.loop.fanout import campaign_device_lock_path

    _patch_primitive(monkeypatch, _stub_primitive)
    workspace = _workspace(tmp_path)
    sentinel = campaign_device_lock_path(workspace)
    assert not sentinel.exists()

    agent = SpecialistAgent(
        definition=_definition(),
        backend=_McpBackend(),
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )
    await agent.run(_assignment(), _context(workspace))

    assert sentinel.is_file()


@pytest.mark.asyncio
async def test_a_probe_with_no_device_sentinel_measures_nothing(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path, device_lock=None)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.UNAVAILABLE
    assert probe_server.DEVICE_LOCK_ENV in result["detail"]


@pytest.mark.asyncio
async def test_waiting_for_a_busy_device_is_charged_and_bounded(tmp_path, monkeypatch) -> None:
    """A specialist that blocked on the device would spend its session idle."""
    import fcntl

    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setattr(probe_server, "DEVICE_LOCK_POLL_SEC", 0.01)
    sandbox = _sandbox(tmp_path, budget_sec=0.2)
    budget = probe_server.ProbeBudget()
    sandbox.device_lock.touch()
    holder = sandbox.device_lock.open("a+", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    try:
        result = await probe_server.probe_variant(
            {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
            sandbox=sandbox,
            budget=budget,
        )
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

    assert result["status"] == probe_server.DEVICE_BUSY
    assert "nothing was measured" in result["detail"]
    assert budget.attempts == 1
    assert budget.seconds_used > 0.0
    assert _ledger(sandbox)[0]["status"] == probe_server.DEVICE_BUSY


def test_a_probe_budget_never_outlasts_the_session_that_pays_for_it() -> None:
    """The configured budget is a ceiling, not an entitlement."""
    # Plenty of session: the configured budget stands.
    assert probe_server.probe_budget_sec(configured_remaining=600.0, session_remaining=1800.0) == 600.0
    # Little session left: half of what remains, not the configured 600.
    assert probe_server.probe_budget_sec(configured_remaining=600.0, session_remaining=300.0) == 150.0
    # Overspent, or no session left at all: nothing.
    assert probe_server.probe_budget_sec(configured_remaining=-5.0, session_remaining=1800.0) == 0.0
    assert probe_server.probe_budget_sec(configured_remaining=600.0, session_remaining=0.0) == 0.0
    # No declared session deadline leaves the configured budget alone.
    assert probe_server.probe_budget_sec(configured_remaining=600.0, session_remaining=math.inf) == 600.0


def test_a_probe_ceiling_leaves_the_session_time_to_write_its_analysis() -> None:
    """``int(budget_sec)`` truncated a fractional budget to an instant timeout."""
    assert probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=600.0) == 600
    # The session, not the budget, is what is short here.
    assert probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=300.0, requested=600.0) == int(
        300 - probe_server.ANALYSIS_RESERVE_SEC
    )
    # Never zero, which some backends read as "time out immediately".
    assert probe_server.probe_timeout_sec(budget_remaining=0.5, session_remaining=1800.0, requested=0.5) == 1
    assert probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=10.0, requested=600.0) == 1


@pytest.mark.asyncio
async def test_a_probe_that_would_leave_no_time_for_the_analysis_is_refused(tmp_path, monkeypatch) -> None:
    """A specialist killed mid-probe returns no analysis, and the round raises.

    Driven with a fake clock rather than a sleep: what is under test is the
    arithmetic on the session deadline, not the passage of time.
    """
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setattr(probe_server, "wall_clock", lambda: 1_000.0)
    sandbox = _sandbox(
        tmp_path,
        # 90s of session left, and 120s of that is reserved for the analysis.
        session_deadline=1_090.0,
    )
    budget = probe_server.ProbeBudget()

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.BUDGET_EXHAUSTED
    assert "reserved for writing the analysis" in result["detail"]
    assert "stop probing" in result["detail"]
    # Refusing costs an attempt, so the same question is not asked all session.
    assert budget.attempts == 1
    assert _ledger(sandbox)[0]["status"] == probe_server.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_a_probe_still_runs_while_the_session_has_room_for_both(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setattr(probe_server, "wall_clock", lambda: 1_000.0)
    sandbox = _sandbox(tmp_path, session_deadline=1_000.0 + 1800.0)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.MEASURED


@pytest.mark.asyncio
async def test_the_mcp_tool_timeout_is_not_the_raw_budget(tmp_path, monkeypatch) -> None:
    """A tool timeout of the whole budget can outlive the session that pays it."""
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=300,
        max_turns=4,
        probe=_probe(tmp_path, budget_sec=600.0),
    )

    await agent.run(_assignment(), _context(_workspace(tmp_path)))

    server = backend.specs[0].mcp_servers["specialist_probe"]
    # Plus the grace the server itself allows the primitive: a client that
    # timed out first would kill the call before ``_record`` wrote anything.
    assert server.tool_timeout_sec == (int(300 - probe_server.ANALYSIS_RESERVE_SEC) + probe_server.PROBE_TOOL_GRACE_SEC)
    assert float(server.env[probe_server.SESSION_DEADLINE_ENV]) > 0.0


# --- the round, not the assignment, is what the budget bounds ----------------


def test_the_probe_budget_is_shared_by_the_specialists_of_one_round(
    tmp_path,
) -> None:
    """Each server process has its own budget object; the round has one budget."""
    shared = tmp_path / "round_budget.json"
    first = probe_server.ProbeBudget(path=shared)
    second = probe_server.ProbeBudget(path=shared)

    first.spend(attempts=1, seconds=12.0)
    second.spend(attempts=1, seconds=8.0)
    first.refresh()

    assert first.attempts == 2
    assert first.seconds_used == 20.0
    assert second.attempts == 2


@pytest.mark.asyncio
async def test_a_rounds_scratch_tree_does_not_outlive_the_round(tmp_path, monkeypatch) -> None:
    """Nothing removed the per-assignment scratch trees, one per round forever."""
    _patch_primitive(monkeypatch, _stub_primitive)
    probe = _probe(tmp_path)
    workspace = _workspace(tmp_path)
    agent = SpecialistAgent(
        definition=_definition(),
        backend=_McpBackend(
            ledger=[
                {
                    "probe_index": 1,
                    "label": "widen",
                    "case_id": "case-a",
                    "status": probe_server.MEASURED,
                    "case_ms": 0.87,
                }
            ]
        ),
        probe=probe,
        timeout_sec=1800,
        max_turns=4,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)

    outcomes = (await pool.run((_assignment(),), _context(workspace))).outcomes

    assert outcomes[0].succeeded
    assert "0.87 ms" in outcomes[0].content
    assert list(Path(probe.scratch_root).iterdir()) == []


@pytest.mark.asyncio
async def test_a_rounds_scratch_tree_is_removed_when_the_round_fails(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)

    class _Exploding(_McpBackend):
        async def run(self, spec, usage=None):
            self.specs.append(spec)
            raise RuntimeError("the provider went away")

    probe = _probe(tmp_path)
    agent = SpecialistAgent(
        definition=_definition(),
        backend=_Exploding(),
        probe=probe,
        timeout_sec=1800,
        max_turns=4,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)

    outcomes = (await pool.run((_assignment(),), _context(_workspace(tmp_path)))).outcomes

    assert outcomes[0].failure is not None
    assert list(Path(probe.scratch_root).iterdir()) == []


# --- a probe outliving its specialist held the device unnoticed -------------


def _fake_reaper(monkeypatch, report_for):
    """Stand in for the reaper, recording where and when it was asked.

    Never a real process and never a real signal: what these tests pin is that
    the round asks at all, that it asks while its tree is still there to be
    surveyed, and that what comes back travels.
    """
    calls: list[Path] = []

    async def _reap(directory, *, description):
        directory = Path(directory)
        # Before the removal, not after: the reaper identifies a process by
        # what it holds open under this directory, so a tree already gone
        # would answer "nothing is running" for every leak.
        assert directory.is_dir(), "the round was reaped after its tree was gone"
        assert str(directory) in description
        calls.append(directory)
        return report_for(directory)

    monkeypatch.setattr(specialists, "reap_processes_under", _reap)
    return calls


async def _run_one_round(tmp_path, monkeypatch, *, probe=None):
    """One ordinary round, whose backend writes one measured probe record."""
    _patch_primitive(monkeypatch, _stub_primitive)
    probe = _probe(tmp_path) if probe is None else probe
    agent = SpecialistAgent(
        definition=_definition(),
        backend=_McpBackend(
            ledger=[
                {
                    "probe_index": 1,
                    "label": "widen",
                    "case_id": "case-a",
                    "status": probe_server.MEASURED,
                    "case_ms": 0.87,
                }
            ]
        ),
        probe=probe,
        timeout_sec=1800,
        max_turns=4,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)
    return await pool.run((_assignment(),), _context(_workspace(tmp_path)))


@pytest.mark.asyncio
async def test_a_probe_that_outlived_its_round_is_reported_to_the_caller(tmp_path, monkeypatch) -> None:
    """A specialist killed mid-probe left a benchmark on the shared GPU.

    The tree was removed and nothing else was done, so the lanes queued behind
    the leftover probe on the device sentinel while the canonical measurement --
    which takes no lock -- ran straight into it. The report has to reach the
    caller, blockers included: those pids are all that is left to ask about once
    the round's tree is gone.
    """
    calls = _fake_reaper(
        monkeypatch,
        lambda directory: ReapReport(
            directory=str(directory),
            unkillable=(4321,),
            holding_device=(4321,),
        ),
    )

    run = await _run_one_round(tmp_path, monkeypatch)

    assert len(calls) == 1
    assert run.contended is True
    assert run.reaped.blockers == (4321,)
    assert "4321" in run.reaped.describe()
    # The round still analysed; the contention is about the device, not this
    # round's own answer.
    assert run.outcomes[0].succeeded
    assert list(Path(_probe(tmp_path).scratch_root).iterdir()) == []


@pytest.mark.asyncio
async def test_an_ordinary_round_is_not_made_to_look_contended(tmp_path, monkeypatch) -> None:
    """A clean teardown must cost the round nothing."""
    calls = _fake_reaper(monkeypatch, lambda directory: ReapReport(str(directory)))

    run = await _run_one_round(tmp_path, monkeypatch)

    assert len(calls) == 1
    assert run.contended is False
    assert run.reaped.blockers == ()
    assert run.outcomes[0].succeeded


@pytest.mark.asyncio
async def test_a_round_that_never_got_a_tree_is_not_reaped(tmp_path, monkeypatch) -> None:
    """There is no directory to survey, and the specialists still analyse."""
    _patch_primitive(monkeypatch, _stub_primitive)
    calls = _fake_reaper(monkeypatch, lambda directory: ReapReport(str(directory)))

    def _no_tree(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(specialists.tempfile, "mkdtemp", _no_tree)
    probe = _probe(tmp_path)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        probe=probe,
        timeout_sec=1800,
        max_turns=4,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)

    run = await pool.run((_assignment(),), _context(_workspace(tmp_path)))

    assert calls == []
    assert run.reaped is None
    assert run.contended is False
    assert backend.specs[0].mcp_servers == {}
    assert "No probe ran" in run.outcomes[0].content


@pytest.mark.asyncio
async def test_a_round_without_a_probe_is_not_reaped(tmp_path, monkeypatch) -> None:
    """No probe means no scratch tree and nothing of ours to have leaked."""
    calls = _fake_reaper(monkeypatch, lambda directory: ReapReport(str(directory)))
    agent = SpecialistAgent(
        definition=_definition(),
        backend=_McpBackend(),
        timeout_sec=1800,
        max_turns=4,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)

    run = await pool.run((_assignment(),), _context(_workspace(tmp_path)))

    assert calls == []
    assert run.reaped is None
    assert run.outcomes[0].succeeded


def test_an_operator_can_turn_the_probe_off_and_resize_it(tmp_path) -> None:
    """max_probes/budget_sec were dataclass defaults no operator could reach."""
    workspace = tmp_path / "canonical"
    off = Config(workspace=str(workspace), specialist_probe=False)
    off.experiments_dir = tmp_path / "out"

    assert _specialist_probe_config(off) is None

    resized = Config(
        workspace=str(workspace),
        specialist_probe_max=2,
        specialist_probe_budget_sec=45.0,
    )
    resized.experiments_dir = tmp_path / "out"
    probe = _specialist_probe_config(resized)

    assert probe is not None
    assert probe.max_probes == 2
    assert probe.budget_sec == 45.0


def test_an_operator_can_place_the_scratch_root(tmp_path) -> None:
    config = Config(
        workspace=str(tmp_path / "canonical"),
        specialist_probe_scratch_root=str(tmp_path / "elsewhere"),
    )
    config.experiments_dir = tmp_path / "out"

    probe = _specialist_probe_config(config)

    assert probe is not None
    assert Path(probe.scratch_root) == tmp_path / "elsewhere"


def test_a_scratch_root_that_contains_the_workspace_is_refused(tmp_path, caplog) -> None:
    """The containment check ran one way only; a tree removed per round is worse."""
    config = Config(
        workspace=str(tmp_path / "campaign" / "canonical"),
        specialist_probe_scratch_root=str(tmp_path / "campaign"),
    )
    config.experiments_dir = tmp_path / "out"

    with caplog.at_level("WARNING"):
        probe = _specialist_probe_config(config)

    assert probe is None
    assert "overlaps the canonical tree" in caplog.text


# --- an attempt that cost nothing could be made all session ------------------


@pytest.mark.asyncio
async def test_a_refused_probe_costs_one_of_the_round_count(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    sandbox = _sandbox(tmp_path, max_probes=2)
    budget = probe_server.ProbeBudget()
    outside = tmp_path / "elsewhere" / "driver.py"
    outside.parent.mkdir()
    outside.write_text("print('wall_ms: 1.0')\n", encoding="utf-8")

    statuses = []
    for _ in range(3):
        result = await probe_server.probe_variant(
            {"label": "widen", "driver_script": str(outside), "case_id": "case-a"},
            sandbox=sandbox,
            budget=budget,
        )
        statuses.append(result["status"])

    assert statuses == [
        probe_server.REFUSED,
        probe_server.REFUSED,
        probe_server.BUDGET_EXHAUSTED,
    ]
    assert budget.attempts == 3


@pytest.mark.asyncio
async def test_an_unavailable_primitive_costs_one_of_the_round_count(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, None)
    sandbox = _sandbox(tmp_path, max_probes=1)
    budget = probe_server.ProbeBudget()

    first = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )
    second = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert first["status"] == probe_server.UNAVAILABLE
    assert second["status"] == probe_server.BUDGET_EXHAUSTED


def test_a_ledger_stops_growing_at_its_cap(tmp_path) -> None:
    """The parent reads this file back in full; refusals are unbounded in nothing."""
    ledger = tmp_path / "probe_ledger.jsonl"

    for index in range(probe_server.MAX_LEDGER_RECORDS + 5):
        probe_server._append_line(ledger, {"probe_index": index, "status": "refused"})

    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == probe_server.MAX_LEDGER_RECORDS + 1
    assert lines[-1]["label"] == "ledger-full"
    assert "dropped and unrecorded" in lines[-1]["detail"]


def test_a_primitive_that_explodes_on_import_is_reported_unavailable(
    monkeypatch,
) -> None:
    """Only ImportError was caught, so any other import-time error killed the server."""

    def _explode():
        raise RuntimeError("the ROCm runtime is not installed")

    monkeypatch.setattr(probe_server, "resolve_probe_primitive", _explode)

    primitive, unusable = probe_server.probe_primitive_status()

    assert primitive is None
    assert "could not be imported" in unusable
    assert "the ROCm runtime is not installed" in unusable


# --- round two: what the child inherits, and what the clocks really allow -----


@pytest.mark.asyncio
async def test_the_probe_child_inherits_the_campaigns_aiter_cache_isolation(tmp_path, monkeypatch) -> None:
    """A probe that misses these times a binary built from other source.

    ``aiter_cache.configure_aiter_cache_isolation`` puts five variables in the
    environment, and aiter's ``get_module`` imports the ``.so`` out of
    ``AITER_JIT_DIR`` by name without checking it against the source. A child
    that fell back to the shared default cache would therefore report a number
    labelled ``measured`` for a binary it did not build -- and, on a cold
    default cache, spend the whole probe budget rebuilding while holding the
    device lock.

    FlyDSL is the third compiler behind that isolation and reaches the child the
    same way. Unforwarded, aiter's own default puts its cache inside the
    workspace, which is both the wrong binary and a git-visible write. It is
    named rather than swept in by a ``FLYDSL_`` prefix because the same family
    holds ``FLYDSL_RUNTIME_RUN_ONLY`` and ``FLYDSL_RUNTIME_ENABLE_CACHE``, which
    would change what the probe measures rather than where it builds.
    """
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setenv("AITER_ROOT_DIR", "/cache/aiter/root")
    monkeypatch.setenv("AITER_JIT_DIR", "/cache/aiter/jit")
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", "/cache/aiter")
    monkeypatch.setenv("FORGE_AITER_CACHE_OWNER_PID", "4242")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", "/cache/aiter/flydsl_cache")
    monkeypatch.setenv("FORGE_NPROC_PER_NODE", "4")
    # Same family, but knobs that change what is measured rather than where it
    # is built: forwarding these is what a "FLYDSL_" prefix would have cost.
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    # Neither is a variable this campaign sets: AITER_HOME appears nowhere in
    # this repository, and AITER_REBUILD is popped by the cache isolation.
    monkeypatch.setenv("AITER_HOME", "/somewhere/else")
    monkeypatch.setenv("AITER_REBUILD", "1")
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path),
    )

    await agent.run(_assignment(), _context(_workspace(tmp_path)))

    env = backend.specs[0].mcp_servers["specialist_probe"].env
    assert env["AITER_ROOT_DIR"] == "/cache/aiter/root"
    assert env["AITER_JIT_DIR"] == "/cache/aiter/jit"
    assert env["FORGE_AITER_CACHE_ROOT"] == "/cache/aiter"
    assert env["FORGE_AITER_CACHE_OWNER_PID"] == "4242"
    assert env["FLYDSL_RUNTIME_CACHE_DIR"] == "/cache/aiter/flydsl_cache"
    assert env["FORGE_NPROC_PER_NODE"] == "4"
    assert "FLYDSL_RUNTIME_RUN_ONLY" not in env
    assert "FLYDSL_RUNTIME_ENABLE_CACHE" not in env
    assert "FLYDSL_" not in specialists._PROBE_CHILD_ENV_PREFIXES
    assert "AITER_HOME" not in env
    assert "AITER_REBUILD" not in env
    assert "AITER_HOME" not in specialists._PROBE_CHILD_ENV_VARS
    assert "AITER_REBUILD" not in specialists._PROBE_CHILD_ENV_VARS


def test_a_non_positive_requested_ceiling_falls_back_to_the_default() -> None:
    """``requested <= 0`` matched the outer test and failed the inner one."""
    assert (
        probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=None)
        == probe_server.DEFAULT_PROBE_TIMEOUT_SEC
    )
    assert (
        probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=0)
        == probe_server.DEFAULT_PROBE_TIMEOUT_SEC
    )
    assert (
        probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=-5)
        == probe_server.DEFAULT_PROBE_TIMEOUT_SEC
    )
    assert (
        probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=True)
        == probe_server.DEFAULT_PROBE_TIMEOUT_SEC
    )
    assert (
        probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested="60")
        == probe_server.DEFAULT_PROBE_TIMEOUT_SEC
    )


@pytest.mark.asyncio
async def test_a_session_too_small_for_one_probe_is_never_offered_one(tmp_path, monkeypatch, caplog) -> None:
    """Offered, promised six probes, and refused from the first call."""
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=int(probe_server.ANALYSIS_RESERVE_SEC),
        max_turns=4,
        probe=_probe(tmp_path),
    )

    with caplog.at_level("WARNING"):
        outcome = await agent.run(_assignment(), _context(_workspace(tmp_path)))

    assert backend.specs[0].mcp_servers == {}
    assert backend.specs[0].tool_policy.extra_tools == ()
    assert "Bounded measurement" not in backend.specs[0].system_prompt
    assert "No probe ran" in outcome.content
    assert "specialist probe not offered" in caplog.text


@pytest.mark.asyncio
async def test_the_prompt_states_the_ceiling_the_client_enforces(tmp_path, monkeypatch) -> None:
    """The tool timeout must outlive the server's own grace, and be stated."""
    _patch_primitive(monkeypatch, _stub_primitive)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=_probe(tmp_path, budget_sec=600.0),
    )

    await agent.run(_assignment(), _context(_workspace(tmp_path)))

    spec = backend.specs[0]
    server = spec.mcp_servers["specialist_probe"]
    ceiling = probe_server.probe_timeout_sec(budget_remaining=600.0, session_remaining=1800.0, requested=600.0)
    # The client must outlast the server, or ``_record`` never writes the
    # ledger line that is the only channel back to the parent.
    assert server.tool_timeout_sec == ceiling + probe_server.PROBE_TOOL_GRACE_SEC
    assert server.tool_timeout_sec > ceiling
    assert f"{ceiling}s" in spec.system_prompt


def test_an_unreachable_round_budget_is_reported_not_silently_per_process(tmp_path, caplog) -> None:
    """Falling back per process gives every specialist a full budget of its own."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    budget = probe_server.ProbeBudget(path=blocker / "round_budget.json")

    with caplog.at_level("WARNING"):
        budget.spend(attempts=1, seconds=5.0)
        budget.spend(attempts=1, seconds=5.0)

    assert budget.shared_error
    assert "round" in budget.shared_error
    assert caplog.text.count("shared probe budget") == 1


@pytest.mark.asyncio
async def test_a_probe_whose_shared_budget_is_unreachable_measures_nothing(tmp_path, monkeypatch) -> None:
    _patch_primitive(monkeypatch, _stub_primitive)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    sandbox = _sandbox(tmp_path)
    budget = probe_server.ProbeBudget(path=blocker / "round_budget.json")

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.UNAVAILABLE
    assert "shared probe budget" in result["detail"]
    assert _ledger(sandbox)[0]["status"] == probe_server.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_non_utf8_round_budget_does_not_escape_the_handler(tmp_path, monkeypatch) -> None:
    """A UnicodeDecodeError is not an OSError; it reached JSON-RPC -32603."""
    _patch_primitive(monkeypatch, _stub_primitive)
    shared = tmp_path / "round_budget.json"
    shared.write_bytes(b"\xff\xfe garbage \x00")
    sandbox = _sandbox(tmp_path)
    budget = probe_server.ProbeBudget(path=shared)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert result["status"] == probe_server.MEASURED
    assert budget.attempts == 1
    assert _ledger(sandbox)[0]["status"] == probe_server.MEASURED


@pytest.mark.asyncio
async def test_the_gate_is_re_checked_after_waiting_for_the_device(tmp_path, monkeypatch) -> None:
    """A full-length wait can push the session under the analysis reserve.

    The old code recomputed the ceiling but not the gate, so the probe started
    with the ``max(1, ...)`` clamp and the ledger recorded "the probe exceeded
    its 1s ceiling" for a session that had simply run out.
    """
    clock = {"wall": 1_000.0, "mono": 0.0}
    monkeypatch.setattr(probe_server, "wall_clock", lambda: clock["wall"])
    monkeypatch.setattr(probe_server, "monotonic_clock", lambda: clock["mono"])

    called = []

    async def _never_called(**kwargs):
        called.append(kwargs)
        raise AssertionError("no probe may start under an impossible ceiling")

    _patch_primitive(monkeypatch, _never_called)

    async def _slow_lock(path, *, timeout_sec):
        clock["wall"] += 125.0
        clock["mono"] += 125.0
        return path.open("r+", encoding="utf-8")

    monkeypatch.setattr(probe_server, "acquire_device_lock", _slow_lock)
    # 240s of session: the gate passes before the wait and fails after it.
    sandbox = _sandbox(tmp_path, budget_sec=600.0, session_deadline=1_240.0)
    budget = probe_server.ProbeBudget()

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=budget,
    )

    assert called == []
    assert result["status"] == probe_server.BUDGET_EXHAUSTED
    assert "waiting for the device" in result["detail"]
    assert "ceiling" not in result["detail"]
    assert result["duration_sec"] == 125.0
    assert budget.attempts == 1
    assert _ledger(sandbox)[0]["status"] == probe_server.BUDGET_EXHAUSTED


def test_a_session_deadline_that_is_not_a_time_is_refused(tmp_path) -> None:
    """``nan`` made the session constraint vanish; ``0`` refused every probe."""
    scratch = tmp_path / "scratch"
    workspace = tmp_path / "canonical"
    scratch.mkdir()
    workspace.mkdir()
    base = {
        probe_server.SCRATCH_ENV: str(scratch),
        probe_server.WORKSPACE_ENV: str(workspace),
        probe_server.MAX_PROBES_ENV: "4",
        probe_server.BUDGET_SEC_ENV: "60",
    }

    for raw in ("nan", "inf", "-inf", "0.000", "-5"):
        with pytest.raises(probe_server.ProbeSandboxError, match="Unix timestamp"):
            probe_server.load_sandbox({**base, probe_server.SESSION_DEADLINE_ENV: raw})

    # Absent stays fail-open: the configured probe budget still bounds it.
    sandbox = probe_server.load_sandbox(base)
    assert sandbox.session_deadline is None
    assert sandbox.session_remaining_sec() == math.inf


def test_a_setup_with_no_deadline_omits_the_variable(tmp_path) -> None:
    """A default of 0.0 that is always formatted disables the feature silently."""
    setup = specialists._ProbeSetup(
        enabled=True,
        scratch_dir=tmp_path / "scratch" / "memory-1",
        ledger_path=tmp_path / "scratch" / "memory-1" / "probe_ledger.jsonl",
        workspace=str(tmp_path / "canonical"),
        config=_probe(tmp_path),
    )

    assert setup.session_deadline is None
    assert probe_server.SESSION_DEADLINE_ENV not in setup.server_env()


@pytest.mark.asyncio
async def test_every_result_carries_the_three_numbers_promised(tmp_path, monkeypatch) -> None:
    """The description promised a third number the record never carried."""
    _patch_primitive(monkeypatch, _stub_primitive)
    monkeypatch.setattr(probe_server, "wall_clock", lambda: 1_000.0)
    sandbox = _sandbox(tmp_path, session_deadline=1_000.0 + 1800.0)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["probes_remaining"] == 1
    assert result["seconds_remaining"] > 0.0
    # The round's budget being gone is a different thing from this session
    # nearly being over, and the model is told it can tell them apart.
    assert result["session_seconds_remaining"] == pytest.approx(1800.0)
    assert _ledger(sandbox)[0]["session_seconds_remaining"] == pytest.approx(1800.0)
    description = probe_server.TOOL_DEFINITIONS[0]["description"]
    assert "YOUR OWN session" in description


@pytest.mark.asyncio
async def test_the_ledger_full_marker_is_written_before_the_refusals_go_quiet(tmp_path, monkeypatch) -> None:
    """A truncated ledger that reads like a short one is what the marker prevents."""
    ledger = tmp_path / "scratch" / "memory-1" / "probe_ledger.jsonl"
    monkeypatch.setenv(probe_server.LEDGER_ENV, str(ledger))
    monkeypatch.delenv(probe_server.SCRATCH_ENV, raising=False)
    monkeypatch.delenv(probe_server.WORKSPACE_ENV, raising=False)
    server = probe_server.ProbeServer()

    for _ in range(probe_server.MAX_LEDGER_RECORDS + 3):
        await server.handle_tool_call(
            "probe_variant",
            {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        )

    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == probe_server.MAX_LEDGER_RECORDS + 1
    assert lines[-1]["label"] == "ledger-full"


@pytest.mark.asyncio
async def test_a_device_sentinel_that_does_not_exist_measures_nothing(tmp_path, monkeypatch) -> None:
    """Opening it ``a+`` locked a fresh private file and serialized nothing."""
    _patch_primitive(monkeypatch, _stub_primitive)
    missing = tmp_path / "no-such-sentinel.lock"
    sandbox = _sandbox(tmp_path, device_lock=missing)

    result = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=probe_server.ProbeBudget(),
    )

    assert result["status"] == probe_server.UNAVAILABLE
    assert str(missing) in result["detail"]
    assert not missing.exists()


@pytest.mark.asyncio
async def test_the_probe_index_counts_this_assignments_own_attempts(tmp_path, monkeypatch) -> None:
    """The ledger is one assignment's; a round-global counter read 1, 3, 4."""
    _patch_primitive(monkeypatch, _stub_primitive)
    shared = tmp_path / "round_budget.json"
    sandbox = _sandbox(tmp_path, max_probes=6)
    mine = probe_server.ProbeBudget(path=shared)
    sibling = probe_server.ProbeBudget(path=shared)

    sibling.spend(attempts=1, seconds=1.0)
    first = await probe_server.probe_variant(
        {"label": "widen", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=mine,
    )
    sibling.spend(attempts=1, seconds=1.0)
    second = await probe_server.probe_variant(
        {"label": "narrow", "driver_script": "driver.py", "case_id": "case-a"},
        sandbox=sandbox,
        budget=mine,
    )

    assert [first["probe_index"], second["probe_index"]] == [1, 2]
    # The count is still the round's.
    assert second["probes_remaining"] == 2


@pytest.mark.asyncio
async def test_a_round_whose_tree_cannot_be_made_disables_the_probe(tmp_path, monkeypatch) -> None:
    """``None`` meant both "no round" and "no tree", so the probe fell back to it."""
    _patch_primitive(monkeypatch, _stub_primitive)

    def _no_tree(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(specialists.tempfile, "mkdtemp", _no_tree)
    probe = _probe(tmp_path)
    backend = _McpBackend()
    agent = SpecialistAgent(
        definition=_definition(),
        backend=backend,
        timeout_sec=1800,
        max_turns=4,
        probe=probe,
    )
    pool = specialists.SpecialistPool({"memory": agent}, max_parallel=1)

    outcomes = (await pool.run((_assignment(),), _context(_workspace(tmp_path)))).outcomes

    assert backend.specs[0].mcp_servers == {}
    assert "No probe ran" in outcomes[0].content
    assert "no space left on device" in outcomes[0].content
    # Nothing was written under the root the round could not use.
    assert list(Path(probe.scratch_root).iterdir()) == []


def test_a_relative_scratch_root_is_refused(tmp_path) -> None:
    """A relative value resolves against whatever the process CWD happens to be."""
    with pytest.raises(ValueError, match="absolute"):
        Config(
            workspace=str(tmp_path / "canonical"),
            specialist_probe_scratch_root="relative/scratch",
        )


def test_the_scratch_root_fallback_is_said_at_warning_level(tmp_path, caplog) -> None:
    """``forge_loop`` never calls ``basicConfig``, so ``log.info`` is discarded."""
    workspace = tmp_path / "canonical"
    config = Config(workspace=str(workspace))
    config.experiments_dir = workspace / "forge_experiments"

    with caplog.at_level("WARNING"):
        probe = _specialist_probe_config(config)

    assert probe is not None
    assert "specialist probe scratch root placed at" in caplog.text


def test_the_probe_env_overrides_are_reachable_from_the_forge_loop_path(
    monkeypatch,
) -> None:
    """Concrete click defaults meant ``from_env`` never saw the environment."""
    from kernelforge.cli import main

    params = {param.name: param for param in main.commands["forge-loop"].params}
    for name in (
        "specialist_probe",
        "specialist_probe_max",
        "specialist_probe_budget_sec",
        "specialist_probe_scratch_root",
    ):
        assert params[name].default is None, name

    monkeypatch.setenv("FORGE_SPECIALIST_PROBE", "0")
    monkeypatch.setenv("FORGE_SPECIALIST_PROBE_MAX", "3")
    monkeypatch.setenv("FORGE_SPECIALIST_PROBE_BUDGET_SEC", "120")
    config = Config.from_env(workspace="/tmp/canonical")

    assert config.specialist_probe is False
    assert config.specialist_probe_max == 3
    assert config.specialist_probe_budget_sec == 120.0
