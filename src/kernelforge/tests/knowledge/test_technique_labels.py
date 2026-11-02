"""The Implementer names the techniques its change applies in a knowledge base's vocabulary, and each archived
candidate keeps what was declared."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.config import Config
from kernelforge.knowledge.techniques import TechniqueVocabulary
from kernelforge.workflows.forge_loop.campaign.archive import CandidateArchive, CandidateRecord
from kernelforge.workflows.forge_loop.iteration.fanout import LaneResult
from kernelforge.workflows.forge_loop.iteration import keep_checkpoint
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop, IterationResult, LoopAgents
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _counting_plan,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)
from kernelforge.workflows.forge_loop.campaign.experiment import ExperimentTracker

TAGS = """\
techniques:
  - split-k
  - software-pipelining   # a comment the vocabulary ignores
  - vectorized-loads
architectures:
  - cdna3
"""
ALIASES = """\
split-k:
  - "split K"
  - splitk
cdna3:
  - MI300X
"""
SPLIT_K = {"listed": ["split-k"], "unlisted": ["atomic epilogue"]}


def _tags(tmp_path) -> Path:
    data = tmp_path / "wiki" / "data"
    data.mkdir(parents=True)
    (data / "tags.yaml").write_text(TAGS)
    (data / "aliases.yaml").write_text(ALIASES)
    return data / "tags.yaml"


def _vocabulary(tmp_path) -> TechniqueVocabulary:
    return TechniqueVocabulary.load(_tags(tmp_path))


def test_a_declared_line_is_read_in_the_vocabularys_own_terms(tmp_path):
    text = (
        "Split the reduction across workgroups.\n"
        "- **TECHNIQUES:** Split K, software pipelining, `vectorized-loads`, splitk, LDS swizzle\n"
        "PLAN: split the reduction"
    )

    assert _vocabulary(tmp_path).labels(text) == {
        "listed": ["split-k", "software-pipelining", "vectorized-loads"],
        "unlisted": ["LDS swizzle"],
    }


def test_a_change_declared_to_apply_none_is_told_from_one_that_declared_nothing(tmp_path):
    vocabulary = _vocabulary(tmp_path)

    assert vocabulary.labels("TECHNIQUES: none\nPLAN: only a comment changed") == {"listed": [], "unlisted": []}
    assert vocabulary.labels("PLAN: no techniques line") is None


def test_only_the_last_line_counts_and_names_of_its_own_are_bounded(tmp_path):
    names = ", ".join(f"idea {index} " + "x" * 80 for index in range(8))

    labels = _vocabulary(tmp_path).labels(f"TECHNIQUES: split-k\nwent on working\nTECHNIQUES: {names}")

    assert labels["listed"] == []
    assert [name[:7] for name in labels["unlisted"]] == [f"idea {index} " for index in range(5)]
    assert {len(name) for name in labels["unlisted"]} == {60}


def test_the_request_names_every_technique_and_how_to_declare_none(tmp_path):
    request = _vocabulary(tmp_path).request()

    assert request.startswith("Just before your `PLAN:` line")
    assert "split-k, software-pipelining, vectorized-loads." in request
    assert "cdna3" not in request
    assert "`TECHNIQUES: none`" in request


def test_a_vocabulary_that_lists_no_techniques_is_refused(tmp_path):
    path = tmp_path / "tags.yaml"
    path.write_text("architectures:\n  - cdna3\n")

    with pytest.raises(ValueError, match="lists no techniques"):
        TechniqueVocabulary.load(path)


def test_a_campaign_without_a_vocabulary_asks_for_nothing_and_reads_nothing(tmp_path):
    own = TechniqueVocabulary.from_config(Config(technique_vocabulary=str(_tags(tmp_path))))
    off = TechniqueVocabulary.from_config(Config())

    assert own.terms == ("split-k", "software-pipelining", "vectorized-loads")
    assert (off.request_suffix(), off.labels("TECHNIQUES: split-k")) == ("", None)


def test_the_vocabulary_comes_from_the_environment_and_must_be_an_absolute_file(tmp_path, monkeypatch):
    tags = _tags(tmp_path)
    monkeypatch.setenv("FORGE_TECHNIQUE_VOCABULARY", f" {tags} ")

    assert Config.from_env().technique_vocabulary == str(tags)
    with pytest.raises(ValueError, match="technique_vocabulary"):
        Config(technique_vocabulary="data/tags.yaml")
    with pytest.raises(ValueError, match="technique_vocabulary"):
        Config(technique_vocabulary=str(tmp_path / "missing.yaml"))


def _implementer_session(monkeypatch, config, *, kernel, text, driver_script=None):
    """Run one Implementer session whose last message is ``text``; return its run spec and session sink."""
    stub = types.ModuleType("claude_agent_sdk")
    stub.ClaudeAgentOptions = object
    stub.query = None
    stub.HookMatcher = object
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", stub)

    import kernelforge.workflows.forge_loop.roles.agent as agent_mod
    from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult, AgentRuntimeConfig

    runtime = AgentRuntimeConfig(provider="claude", model="m", timeout_sec=1800)
    backend = types.SimpleNamespace(
        name="claude",
        runtime=runtime,
        capabilities=AgentCapabilities(
            writable=True, resumable=True, stop_hooks=True, native_subagents=True, mcp=False
        ),
        fallback_reason="",
    )
    seen: dict = {}

    async def run_session(_backend, spec, usage=None, **_kwargs):
        seen["spec"] = spec
        return AgentRunResult(text=text, subtype="success")

    monkeypatch.setattr(agent_mod, "create_registered_backend", lambda *a, **k: backend)
    monkeypatch.setattr(agent_mod, "run_session_with_api_resume", run_session)
    config.workspace = ""
    config.agent_runtime = lambda: runtime
    agent_fn = agent_mod.make_agent_fn(
        config,
        agent_mod.ImplementerSetup(
            agent_mod.ImplementerTask(program_md="PROGRAM BODY"),
            measurement=agent_mod.GateMeasurement(driver_script=driver_script),
            session=agent_mod.SessionOptions(session_timeout_sec=3600, insession_gate=driver_script is not None),
        ),
    )
    sink: dict = {}
    asyncio.run(agent_fn(str(kernel), "(history)", session_sink=sink))
    return seen["spec"], sink


@pytest.mark.parametrize("gated", [False, True])
def test_an_implementer_with_a_vocabulary_is_asked_for_its_techniques_and_they_reach_its_sink(
    tmp_path, monkeypatch, gated
):
    tags = _tags(tmp_path)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("# kernel\n")
    driver = tmp_path / "driver.py"
    driver.write_text("print('driver')\n")
    text = "Split it.\nTECHNIQUES: split K, atomic epilogue\nPLAN: split the reduction"
    run = {"kernel": kernel, "text": text, "driver_script": str(driver) if gated else None}

    asked, sink = _implementer_session(monkeypatch, Config(technique_vocabulary=str(tags)), **run)
    plain, plain_sink = _implementer_session(monkeypatch, Config(), **run)

    suffix = TechniqueVocabulary.load(tags).request_suffix()
    assert suffix in asked.system_prompt
    assert asked.system_prompt.replace(suffix, "") == plain.system_prompt
    assert sink["techniques"] == SPLIT_K
    assert plain_sink["techniques"] is None


def test_meta_json_carries_declared_techniques_only_for_a_candidate_that_declared_them(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    for iteration, techniques in ((1, SPLIT_K), (2, {"listed": [], "unlisted": []}), (3, None)):
        archive.record(
            CandidateRecord(iteration=iteration, decision="KEEP", kernel_source="k\n", techniques=techniques)
        )

    assert (archive.load_meta(1)["techniques"], archive.load_meta(1)["techniques_unlisted"]) == (
        ["split-k"],
        ["atomic epilogue"],
    )
    assert (archive.load_meta(2)["techniques"], archive.load_meta(2)["techniques_unlisted"]) == ([], [])
    assert "techniques" not in archive.load_meta(3) and "techniques_unlisted" not in archive.load_meta(3)


async def _kept(iteration, plan=""):
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=True,
        validation_summary="canonical validation passed",
        wall_ms=0.9,
        mean_case_speedup=1.1,
        snr_db=40.0,
        kept=True,
    )


async def _declaring_agent(kernel_path, _history, session_sink):
    session_sink["plan"] = "split the reduction"
    session_sink["techniques"] = SPLIT_K
    Path(kernel_path).write_text("def kernel():\n    return 2\n")
    return "split-k fills the idle CUs"


def _archived(loop, iteration) -> tuple:
    meta = loop.archive.load_meta(iteration)
    return meta["decision"], meta["techniques"], meta["techniques_unlisted"]


def test_a_kept_candidate_is_archived_with_the_techniques_its_session_declared(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    monkeypatch.setattr(loop, "run_one_iteration", _kept)
    asyncio.run(loop.run(LoopAgents(agent_fn=_declaring_agent, supervisor_fn=_unused_supervisor)))

    assert _archived(loop, 1) == ("KEEP", ["split-k"], ["atomic epilogue"])


def test_a_keep_finished_after_the_process_died_keeps_its_declared_techniques(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)

    def die_after_the_commit(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(first, "run_one_iteration", _kept)
    monkeypatch.setattr(first, "_finalize_keep_checkpoint", die_after_the_commit)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(first.run(LoopAgents(agent_fn=_declaring_agent, supervisor_fn=_unused_supervisor)))
    assert (workspace / "forge_experiments" / "pending_keep.json").is_file()
    assert first.archive.load_meta(1) == {}

    resumed = IterationLoop(
        first.ic,
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(gpu_target="gfx942"),
        resume=True,
    )
    monkeypatch.setattr(resumed, "_time_remaining", lambda: 0.0)
    asyncio.run(resumed.run(LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    assert _archived(resumed, 1) == ("KEEP", ["split-k"], ["atomic epilogue"])


def test_a_checkpoint_written_before_techniques_were_kept_still_loads(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop._pending_keep_path.parent.mkdir(parents=True, exist_ok=True)
    loop._persist_pending_keep(dict.fromkeys(keep_checkpoint.PENDING_KEEP_FIELDS - {"techniques"}))

    assert loop._load_pending_keep()["techniques"] is None


def test_a_queued_candidate_keeps_its_declared_techniques_across_processes():
    declared = LaneResult(lane_id="1", plan="widen the loads", diff="patch-1", techniques=SPLIT_K)
    earlier = {key: value for key, value in declared.to_dict().items() if key != "techniques"}

    assert LaneResult.from_dict(declared.to_dict()).techniques == SPLIT_K
    assert LaneResult.from_dict(earlier).techniques is None


def test_each_lane_candidate_is_measured_with_the_techniques_its_own_session_declared(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    monkeypatch.setattr(loop, "_run_orchestration", _counting_plan(loop, workspace, [], plans=("split", "pipeline")))

    def factory(lane_dir, _serialized_driver):
        async def agent(_kernel_file, _prompt, session_sink):
            lane = Path(lane_dir).name
            path = Path(lane_dir) / "kernel.py"
            path.write_text(path.read_text().replace("return 1", f"return 1{lane}"))
            session_sink["techniques"] = {"listed": ["split-k"], "unlisted": [f"lane {lane} idea"]}
            return f"lane {lane} expects this to beat best"

        return agent

    async def measured(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.0,
            mean_case_speedup=1.0,
            kept=False,
            bench_detail={"case_times": {"case": 1.0}},
        )

    monkeypatch.setattr(loop, "run_one_iteration", measured)
    agents = LoopAgents(
        agent_fn=_no_change_agent,
        agent_factory=factory,
        orchestration_service=object(),
        supervisor_fn=_unused_supervisor,
    )
    asyncio.run(loop.run(agents))

    assert loop.archive.load_meta(1)["techniques_unlisted"] == ["lane 1 idea"]
    assert [(item.lane_id, item.techniques) for item in loop._lane_queue] == [
        ("2", {"listed": ["split-k"], "unlisted": ["lane 2 idea"]})
    ]
