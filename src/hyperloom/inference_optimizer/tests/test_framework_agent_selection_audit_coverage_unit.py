# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for FRAMEWORK agent-ranked selection, semantic-audit routing,
ranker-client plumbing, config-lever extraction, and the cyclic phase-budget
dispatch guard added on the framework-enhance branch.

These exercise the pure/sync helpers and the small async helpers directly
(stubbing the LLM client / fa phase-audit / KB writeback) so no event-loop GPU
work or network is needed."""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop import coordinator as coord_mod
from hyperloom.orchestrator.framework import client as fa_client_mod
from hyperloom.orchestrator.framework import paths as fp_mod
from hyperloom.orchestrator.framework import artifacts as fpa_mod
from hyperloom.orchestrator.knowledge import kb_writeback as kb_mod
from hyperloom.orchestrator.phases import machine_state as ps_mod
from hyperloom.orchestrator.actions.executors import framework_agent as fpr_mod
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {
        name: MockBackend(_silent_plan(), name=name)
        for name in ("orchestration", "kernel_agent", "critic", "robustness")
    }


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


# --------------------------------------------------------------------------
# _framework_config_levers_from_done
# --------------------------------------------------------------------------
def test_config_levers_non_dict_and_patch_precedence() -> None:
    f = coord_mod._framework_config_levers_from_done
    assert f(None) == {}
    # A patch deliverable is NOT a config-only outcome.
    assert f({"patches_written": ["a.patch"], "proposal_set": [{"extra_envs": {"X": "1"}}]}) == {}
    # No proposal_set / wrong type.
    assert f({"proposal_set": "nope"}) == {}
    assert f({}) == {}


def test_config_levers_flatten_envs_and_args() -> None:
    f = coord_mod._framework_config_levers_from_done
    levers = f(
        {
            "proposal_set": [
                {
                    "extra_envs": {"VLLM_FOO": 1, "  ": "skipped"},
                    "extra_args": "--enable-x --max-num-seqs=256 --tp 4 --bare",
                }
            ]
        }
    )
    assert levers["VLLM_FOO"] == "1"
    assert levers["--max-num-seqs"] == "256"
    assert levers["--tp"] == "4"
    assert levers["--enable-x"] == ""  # followed by another flag -> bare
    assert levers["--bare"] == ""  # trailing bare flag
    assert "  " not in levers


def test_config_levers_args_as_list() -> None:
    f = coord_mod._framework_config_levers_from_done
    levers = f({"proposal_set": [{"extra_args": ["--flag", "val"]}]})
    assert levers["--flag"] == "val"


# --------------------------------------------------------------------------
# _framework_agent_audit_skip_confident
# --------------------------------------------------------------------------
def test_audit_skip_confident_paths(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_SKIP_MIN_CONFIDENCE", raising=False)
    # No evidence -> not confident.
    assert coord._framework_agent_audit_skip_confident({"confidence": 0.99}) is False
    assert coord._framework_agent_audit_skip_confident(None) is False
    # Evidence + high confidence -> confident.
    assert coord._framework_agent_audit_skip_confident({"evidence": [{"x": 1}], "confidence": 0.9}) is True
    # Evidence but low confidence -> not.
    assert coord._framework_agent_audit_skip_confident({"evidence": [{"x": 1}], "confidence": 0.5}) is False
    # Bad confidence value -> treated as 0.0.
    assert coord._framework_agent_audit_skip_confident({"evidence": [{"x": 1}], "confidence": "NaNish"}) is False


def test_audit_skip_confident_bad_floor_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_SKIP_MIN_CONFIDENCE", "not-a-float")
    # Floor falls back to 0.8; confidence 0.85 clears it.
    assert coord._framework_agent_audit_skip_confident({"evidence": [{"x": 1}], "confidence": 0.85}) is True


# --------------------------------------------------------------------------
# _framework_agent_roots_have_git
# --------------------------------------------------------------------------
def test_roots_have_git_true(coord: Coordinator, monkeypatch, tmp_path) -> None:
    repo = tmp_path / "fwrepo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(fp_mod, "resolve_source_file_allowlist", lambda: [str(missing), str(repo)])
    assert coord._framework_agent_roots_have_git() is True


def test_roots_have_git_false(coord: Coordinator, monkeypatch, tmp_path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(fp_mod, "resolve_source_file_allowlist", lambda: [str(plain)])
    assert coord._framework_agent_roots_have_git() is False


# --------------------------------------------------------------------------
# _audit_framework_agent_candidate
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_audit_candidate_cached(coord: Coordinator) -> None:
    cand = {"candidate_id": "c1", "_audit": {"recommended_next_step": "skip", "cached": True}}
    out = await coord._audit_framework_agent_candidate(cand)
    assert out.get("cached") is True


@pytest.mark.asyncio
async def test_audit_candidate_success(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(fp_mod, "resolve_source_file_allowlist", lambda: [])

    async def _phase_audit(**kwargs):
        return {"recommended_next_step": "direct_framework", "semantic_status": "applies", "evidence": []}

    monkeypatch.setattr(fa_client_mod, "phase_audit", _phase_audit)
    cand = {"candidate_id": "c2", "pr_url": "http://x/1"}
    out = await coord._audit_framework_agent_candidate(cand)
    assert out["recommended_next_step"] == "direct_framework"
    assert cand["_audit"]["recommended_next_step"] == "direct_framework"  # cached onto candidate


@pytest.mark.asyncio
async def test_audit_candidate_failure_degrades_unknown(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(fp_mod, "resolve_source_file_allowlist", lambda: [])

    async def _boom(**kwargs):
        raise RuntimeError("fa missing")

    monkeypatch.setattr(fa_client_mod, "phase_audit", _boom)
    out = await coord._audit_framework_agent_candidate({"candidate_id": "c3"})
    assert out["semantic_status"] == "unknown"
    assert out["recommended_next_step"] == ""


@pytest.mark.asyncio
async def test_audit_candidate_non_dict_result_coerced(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(fp_mod, "resolve_source_file_allowlist", lambda: [])

    async def _weird(**kwargs):
        return ["not", "a", "dict"]

    monkeypatch.setattr(fa_client_mod, "phase_audit", _weird)
    out = await coord._audit_framework_agent_candidate({"candidate_id": "c4"})
    assert out["semantic_status"] == "unknown"


# --------------------------------------------------------------------------
# _framework_agent_audit_seed_lines
# --------------------------------------------------------------------------
def test_audit_seed_lines_empty(coord: Coordinator) -> None:
    assert coord._framework_agent_audit_seed_lines(None) == []
    assert coord._framework_agent_audit_seed_lines({}) == []


def test_audit_seed_lines_populated(coord: Coordinator) -> None:
    lines = coord._framework_agent_audit_seed_lines(
        {
            "semantic_status": "needs_rewrite",
            "applicability": "needs_rewrite",
            "evidence": [
                {"local_file": "vllm/x.py", "symbol": "foo", "reason": "moved"},
                "not-a-dict",
            ],
            "risks": ["r1", "r2"],
        }
    )
    blob = "\n".join(lines)
    assert "AUDIT EVIDENCE" in blob
    assert "vllm/x.py" in blob and "[foo]" in blob and "moved" in blob
    assert "risks: r1; r2" in blob


# --------------------------------------------------------------------------
# _record_framework_agent_audit_skip
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_record_audit_skip_already_present(coord: Coordinator, monkeypatch) -> None:
    seen: list = []

    async def _kb(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(kb_mod, "write_framework_record", _kb)
    coord.shared_state.framework_agent_phase_progress = None  # exercise the list-init branch
    cand = {"candidate_id": "c1", "pr_url": "http://x/1", "batch_id": "b1", "head_sha": "deadbeef"}
    await coord._record_framework_agent_audit_skip(cand, {"semantic_status": "already_merged", "confidence": 0.95})
    prog = coord.shared_state.framework_agent_phase_progress
    assert any(p.get("status") == "already_present" for p in prog)
    assert seen  # KB writeback fired for already_present


@pytest.mark.asyncio
async def test_record_audit_skip_not_applicable(coord: Coordinator, monkeypatch) -> None:
    async def _kb(**kwargs):  # should NOT be called for not_applicable
        raise AssertionError("KB writeback must not fire for not_applicable")

    monkeypatch.setattr(kb_mod, "write_framework_record", _kb)
    cand = {"candidate_id": "c2", "batch_id": "b1"}
    await coord._record_framework_agent_audit_skip(cand, {"semantic_status": "not_relevant", "risks": ["x"]})
    prog = coord.shared_state.framework_agent_phase_progress
    assert any(p.get("status") == "not_applicable" for p in prog)


# --------------------------------------------------------------------------
# _collect_framework_agent_candidate_priors
# --------------------------------------------------------------------------
def test_collect_framework_agent_candidate_priors(coord: Coordinator) -> None:
    coord.shared_state.framework_agent_critic_decisions = [
        "not-a-dict",  # skipped via the continue branch
        {"candidate_id": "c1", "verdict": "approve", "rationale": "looks good"},
    ]
    coord.shared_state.framework_agent_phase_progress = [
        {"candidate_id": "c1", "status": "kept", "gain_pct": 3.2},
        {"candidate_id": "c2", "status": "in_flight"},  # non-terminal -> excluded
        {"candidate_id": "c3", "status": "critic_denied"},
    ]
    priors = coord._collect_framework_agent_candidate_priors()
    assert priors["recent_decisions"] == [
        {"candidate_id": "c1", "verdict": "approve", "rationale": "looks good"}
    ]
    statuses = {o["status"] for o in priors["recent_outcomes"]}
    assert statuses == {"kept", "critic_denied"}


# --------------------------------------------------------------------------
# _match_framework_agent_candidate
# --------------------------------------------------------------------------
def test_match_candidate(coord: Coordinator) -> None:
    cands = [
        {"candidate_id": "id-a", "pr_number": "1015"},
        {"pr_url": "http://x/2", "pr_number": "2020"},
    ]
    assert coord._match_framework_agent_candidate("", cands) is None
    assert coord._match_framework_agent_candidate("id-a", cands)["candidate_id"] == "id-a"
    assert coord._match_framework_agent_candidate("http://x/2", cands)["pr_number"] == "2020"
    # Bare PR-number fallback.
    assert coord._match_framework_agent_candidate("PR:2020", cands)["pr_number"] == "2020"
    assert coord._match_framework_agent_candidate("nope", cands) is None


# --------------------------------------------------------------------------
# _framework_agent_ranker_model / _framework_agent_ranker_client
# --------------------------------------------------------------------------
def test_ranker_model_env_override(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_RANKER_MODEL", "my-model")
    assert coord._framework_agent_ranker_model() == "my-model"


def test_ranker_model_from_backend(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_RANKER_MODEL", raising=False)
    coord.backends["orchestration"].model = "backend-model"  # type: ignore[attr-defined]
    assert coord._framework_agent_ranker_model() == "backend-model"


def test_ranker_client_cached(coord: Coordinator) -> None:
    sentinel = object()
    coord._fa_ranker_client = sentinel  # type: ignore[attr-defined]
    assert coord._framework_agent_ranker_client() is sentinel


def test_ranker_client_from_scorer(coord: Coordinator) -> None:
    coord._fa_ranker_client = None  # type: ignore[attr-defined]
    fake_client = object()

    class _Scorer:
        def _ensure_client(self):
            return fake_client

    coord._proposal_scorer = _Scorer()  # type: ignore[attr-defined]
    assert coord._framework_agent_ranker_client() is fake_client
    # Now cached.
    assert coord._framework_agent_ranker_client() is fake_client


def test_ranker_client_none_without_key(coord: Coordinator, monkeypatch) -> None:
    coord._fa_ranker_client = None  # type: ignore[attr-defined]

    class _Scorer:
        def _ensure_client(self):
            raise RuntimeError("no scorer client")

    coord._proposal_scorer = _Scorer()  # type: ignore[attr-defined]
    monkeypatch.delenv("SAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert coord._framework_agent_ranker_client() is None


def test_ranker_client_builds_from_env(coord: Coordinator, monkeypatch) -> None:
    pytest.importorskip("openai")
    coord._fa_ranker_client = None  # type: ignore[attr-defined]
    coord._proposal_scorer = None  # type: ignore[attr-defined]
    monkeypatch.setenv("SAFE_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.example/v1")
    client = coord._framework_agent_ranker_client()
    assert client is not None
    # Cached on the instance after a successful build.
    assert coord._framework_agent_ranker_client() is client


# --------------------------------------------------------------------------
# _select_best_framework_agent_candidate / _rank_framework_agent_candidates_llm
# --------------------------------------------------------------------------
class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChunkChoice(content)] if content is not None else []
        self.usage = None


class _FakeStream:
    """Async-iterable stream of completion chunks (mirrors the streaming proxy)."""

    def __init__(self, content: str) -> None:
        # Split the content into a couple of deltas to exercise accumulation.
        mid = max(1, len(content) // 2)
        self._chunks = [_FakeChunk(content[:mid]), _FakeChunk(content[mid:])] if content else [_FakeChunk("")]

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:  # noqa: PERF203
            raise StopAsyncIteration from None


class _FakeCompletions:
    def __init__(self, content: str, raise_exc: bool = False) -> None:
        self._content = content
        self._raise = raise_exc

    async def create(self, **kwargs):
        # The ranker now streams; assert the streaming flags are set.
        assert kwargs.get("stream") is True
        if self._raise:
            raise RuntimeError("llm down")
        return _FakeStream(self._content)


class _FakeClient:
    def __init__(self, content: str, raise_exc: bool = False) -> None:
        self.chat = type("_C", (), {"completions": _FakeCompletions(content, raise_exc)})()


def _install_ranker(coord: Coordinator, monkeypatch, client) -> None:
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_client", lambda: client)
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_model", lambda: "m")


@pytest.mark.asyncio
async def test_select_best_empty_and_single(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.phase_framework, "_unprocessed_framework_agent_candidates", lambda: [])
    assert await coord._select_best_framework_agent_candidate() is None
    only = {"candidate_id": "solo"}
    monkeypatch.setattr(coord.phase_framework, "_unprocessed_framework_agent_candidates", lambda: [only])
    assert await coord._select_best_framework_agent_candidate() is only


@pytest.mark.asyncio
async def test_select_best_ranker_picks(coord: Coordinator, monkeypatch) -> None:
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    monkeypatch.setattr(coord.phase_framework, "_unprocessed_framework_agent_candidates", lambda: cands)
    _install_ranker(coord, monkeypatch, _FakeClient('{"candidate_id": "c2", "reason": "moe"}'))
    chosen = await coord._select_best_framework_agent_candidate()
    assert chosen["candidate_id"] == "c2"


@pytest.mark.asyncio
async def test_select_best_ranker_failure_falls_back(coord: Coordinator, monkeypatch) -> None:
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    monkeypatch.setattr(coord.phase_framework, "_unprocessed_framework_agent_candidates", lambda: cands)

    async def _raise(_c):
        raise RuntimeError("rank boom")

    monkeypatch.setattr(coord.phase_framework, "_rank_framework_agent_candidates_llm", _raise)
    chosen = await coord._select_best_framework_agent_candidate()
    assert chosen["candidate_id"] == "c1"  # deterministic fallback (discovery order)


@pytest.mark.asyncio
async def test_rank_llm_no_client_or_model(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_client", lambda: None)
    assert await coord._rank_framework_agent_candidates_llm([{"candidate_id": "c1"}]) is None
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_client", lambda: _FakeClient("{}"))
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_model", lambda: "")
    assert await coord._rank_framework_agent_candidates_llm([{"candidate_id": "c1"}]) is None


@pytest.mark.asyncio
async def test_rank_llm_success_with_audit_and_context(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.best_throughput = 1234.5  # type: ignore[attr-defined]
    cands = [
        {"candidate_id": "c1", "title": "t1", "repo": "r", "_audit": {"applicability": "applies"}},
        {"candidate_id": "c2", "title": "t2"},
    ]
    _install_ranker(coord, monkeypatch, _FakeClient('```json\n{"candidate_id": "c1", "reason": "ok"}\n```'))
    chosen = await coord._rank_framework_agent_candidates_llm(cands)
    assert chosen["candidate_id"] == "c1"


@pytest.mark.asyncio
async def test_rank_llm_call_raises(coord: Coordinator, monkeypatch) -> None:
    _install_ranker(coord, monkeypatch, _FakeClient("", raise_exc=True))
    assert await coord._rank_framework_agent_candidates_llm([{"candidate_id": "c1"}, {"candidate_id": "c2"}]) is None


@pytest.mark.asyncio
async def test_rank_llm_empty_and_unknown_id(coord: Coordinator, monkeypatch) -> None:
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    # Empty reply -> None.
    _install_ranker(coord, monkeypatch, _FakeClient("   "))
    assert await coord._rank_framework_agent_candidates_llm(cands) is None
    # Valid JSON but unknown id -> None.
    _install_ranker(coord, monkeypatch, _FakeClient('{"candidate_id": "nope"}'))
    assert await coord._rank_framework_agent_candidates_llm(cands) is None
    # Non-JSON text (no braces) -> None.
    _install_ranker(coord, monkeypatch, _FakeClient("no json here"))
    assert await coord._rank_framework_agent_candidates_llm(cands) is None


# --------------------------------------------------------------------------
# _dispatch_paused_for_phase_budget
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# framework._materialize_pr_diff_via_worktree (scripted _run_git)
# --------------------------------------------------------------------------
def _scripted_run_git(diff_text: str = "diff --git a b\n+x\n", fetch_ok: bool = True, worktree_add_ok: bool = True):
    def _fake(args, timeout=None):  # noqa: ANN001
        sub = args[2] if len(args) > 2 else ""
        if sub == "fetch":
            return (fetch_ok, "", "" if fetch_ok else "remote hung up")
        if sub == "rev-parse":
            return (True, "abc123headsha", "")
        if sub == "worktree":
            if "add" in args:
                return (worktree_add_ok, "", "" if worktree_add_ok else "worktree add failed")
            return (True, "", "")  # remove (best-effort)
        if sub == "merge-base":
            return (True, "basesha", "")
        if sub == "diff":
            return (True, diff_text, "")
        return (True, "", "")

    return _fake


def test_materialize_pr_diff_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git())
    dest = tmp_path / "out" / "cand.patch"
    ok, err = fpr_mod._materialize_pr_diff_via_worktree(
        tmp_path / "root", {"pr_number": 1015}, dest, timeout_sec=30.0
    )
    assert ok is True and err == ""
    assert dest.read_text().startswith("diff --git")


def test_materialize_pr_diff_fetch_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(fetch_ok=False))
    ok, err = fpr_mod._materialize_pr_diff_via_worktree(
        tmp_path / "root", {"ref": "refs/pull/1/head"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "git fetch" in err


def test_materialize_pr_diff_empty_diff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(diff_text="   \n"))
    ok, err = fpr_mod._materialize_pr_diff_via_worktree(
        tmp_path / "root", {"head_sha": "deadbeef"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "empty diff" in err


def test_materialize_pr_diff_no_head_resolvable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git())
    ok, err = fpr_mod._materialize_pr_diff_via_worktree(
        tmp_path / "root", {}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "cannot resolve PR head" in err


def test_materialize_pr_diff_worktree_add_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(worktree_add_ok=False))
    ok, err = fpr_mod._materialize_pr_diff_via_worktree(
        tmp_path / "root", {"head_sha": "deadbeef"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "worktree add failed" in err


# --------------------------------------------------------------------------
# framework_agent_client.phase_audit optional-field plumbing
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_phase_audit_request_optional_fields(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    async def _invoke(*, subcommand, request, session_dir, timeout_sec):
        captured["request"] = request
        captured["subcommand"] = subcommand
        return {"recommended_next_step": "skip"}

    monkeypatch.setattr(fa_client_mod, "_invoke_fa_phase", _invoke)
    out = await fa_client_mod.phase_audit(
        candidate={"candidate_id": "c1", "diff_url": "http://x/d"},
        framework="sglang",
        framework_source_roots=["/root"],
        session_dir=tmp_path,
        diff_text="--- a\n+++ b\n",
        primus_cortex_url="http://cortex/v1",
        model="my-model",
    )
    assert out["recommended_next_step"] == "skip"
    req = captured["request"]
    assert req["diff_text"].startswith("--- a")
    assert req["primus_cortex_url"] == "http://cortex/v1"
    assert req["model"] == "my-model"
    assert req["diff_url"] == "http://x/d"


# --------------------------------------------------------------------------
# framework_agent_artifacts.write_semantic_audit error path
# --------------------------------------------------------------------------
def test_write_semantic_audit_error_returns_none(tmp_path) -> None:
    # Pass a FILE as the session dir so the runs_dir mkdir fails -> exception path.
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    out = fpa_mod.write_semantic_audit(file_path, candidate_id="c1", verdict={"semantic_status": "x"})
    assert out is None


# --------------------------------------------------------------------------
# _dispatch_paused_for_phase_budget
# --------------------------------------------------------------------------
def test_dispatch_pause_phase_not_gated(coord: Coordinator) -> None:
    coord.shared_state.phase = "PRELUDE"
    assert coord._dispatch_paused_for_phase_budget() is False


def test_dispatch_pause_not_long_run(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.phase = "EXPLORE"
    monkeypatch.setattr(coord_mod._phase_state, "is_cyclic_phases_enabled", lambda: True)
    monkeypatch.setattr(coord_mod._phase_state, "is_long_run", lambda _s: False)
    assert coord._dispatch_paused_for_phase_budget() is False


def test_dispatch_pause_budget_spent(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.phase = "EXPLORE"
    monkeypatch.setattr(coord_mod._phase_state, "is_cyclic_phases_enabled", lambda: True)
    monkeypatch.setattr(coord_mod._phase_state, "is_long_run", lambda _s: True)
    monkeypatch.setattr(
        coord_mod._phase_state,
        "phase_budget_remaining_seconds",
        lambda _s, budget_pct=None: 0.0,
    )
    assert coord._dispatch_paused_for_phase_budget() is True


def test_dispatch_pause_budget_remaining(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.phase = "KERNEL"
    monkeypatch.setattr(coord_mod._phase_state, "is_cyclic_phases_enabled", lambda: True)
    monkeypatch.setattr(coord_mod._phase_state, "is_long_run", lambda _s: True)
    monkeypatch.setattr(
        coord_mod._phase_state,
        "phase_budget_remaining_seconds",
        lambda _s, budget_pct=None: 123.0,
    )
    assert coord._dispatch_paused_for_phase_budget() is False


# --------------------------------------------------------------------------
# _maybe_autosubmit_framework_config
# --------------------------------------------------------------------------
def _authoring_task(task_id: str = "spec-1") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        task_id=task_id,
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "cand-1",
            "framework_batch_id": "batch-1",
        },
    )


@pytest.mark.asyncio
async def test_autosubmit_config_not_authoring_returns(coord: Coordinator) -> None:
    task = types.SimpleNamespace(task_id="x", params={})
    await coord._maybe_autosubmit_framework_config(task=task, done_payload={})
    assert not coord.state.pending_proposals


@pytest.mark.asyncio
async def test_autosubmit_config_patch_deliverable_returns(coord: Coordinator) -> None:
    await coord._maybe_autosubmit_framework_config(
        task=_authoring_task(),
        done_payload={"patches_written": ["p.patch"]},
    )
    assert not coord.state.pending_proposals


@pytest.mark.asyncio
async def test_autosubmit_config_no_levers_returns(coord: Coordinator) -> None:
    await coord._maybe_autosubmit_framework_config(
        task=_authoring_task(),
        done_payload={"proposal_set": [{"name": "n"}]},  # no extra_args/envs -> no levers
    )
    assert not coord.state.pending_proposals


@pytest.mark.asyncio
async def test_autosubmit_config_routes_to_integrate_patch(coord: Coordinator) -> None:
    done = {
        "proposal_set": [
            {"name": "mtp-toggle", "extra_envs": {"VLLM_MTP": "1"}, "extra_args": "--speculative 4"}
        ]
    }
    before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_framework_config(task=_authoring_task(), done_payload=done)
    assert len(coord.state.pending_proposals) == before + 1
    prop = next(iter(coord.state.pending_proposals.values()))
    assert prop.action_name == "integrate_patch"
    params = (prop.payload or {}).get("params") or {}
    assert params["framework_agent_authoring"] is True
    assert params["framework_agent_candidate_id"] == "cand-1"
    assert params["config_changes"]["VLLM_MTP"] == "1"
    assert params["config_changes"]["--speculative"] == "4"


@pytest.mark.asyncio
async def test_autosubmit_config_idempotent_on_existing_verdict(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(
        coord.shared_state, "get_specialist_patch_verdict", lambda _sid: "approve", raising=False
    )
    done = {"proposal_set": [{"name": "n", "extra_envs": {"X": "1"}}]}
    await coord._maybe_autosubmit_framework_config(task=_authoring_task(), done_payload=done)
    assert not coord.state.pending_proposals


# --------------------------------------------------------------------------
# _record_framework_agent_authored_outcome
# --------------------------------------------------------------------------
def test_record_authored_outcome_non_dict_and_empty_status(coord: Coordinator) -> None:
    # result.result not a dict -> no-op.
    coord._record_framework_agent_authored_outcome(
        task=types.SimpleNamespace(task_id="t", params={}),
        result=types.SimpleNamespace(result=None),
    )
    # empty status -> no-op.
    coord._record_framework_agent_authored_outcome(
        task=types.SimpleNamespace(task_id="t", params={}),
        result=types.SimpleNamespace(result={"status": ""}),
    )
    assert not (coord.shared_state.framework_agent_phase_progress or [])


def test_record_authored_outcome_kept_rolls_batch_stat(coord: Coordinator) -> None:
    coord.shared_state.framework_agent_batches = [{"batch_id": "batch-1"}]
    coord.shared_state.framework_agent_phase_progress = []
    task = types.SimpleNamespace(
        task_id="ip-1",
        params={
            "specialist_task_id": "spec-1",
            "framework_agent_candidate_id": "cand-1",
            "framework_batch_id": "batch-1",
        },
    )
    result = types.SimpleNamespace(
        result={
            "status": "kept",
            "delta_pct": 4.5,
            "output_throughput": 130.0,
            "reason": "win",
            "accuracy_pass": True,
        }
    )
    coord._record_framework_agent_authored_outcome(task=task, result=result)
    rows = coord.shared_state.framework_agent_phase_progress
    assert rows[-1]["candidate_id"] == "cand-1"
    assert rows[-1]["status"] == "kept" and rows[-1]["kept"] is True
    assert rows[-1]["gain_pct"] == 4.5
    assert coord.shared_state.framework_agent_batches[0]["max_gain_pct_observed_in_batch"] == 4.5


def test_record_authored_outcome_uses_candidate_map_and_batch_fallback(coord: Coordinator) -> None:
    coord.shared_state.framework_agent_specialist_candidate_map = {"spec-9": "cand-from-map"}
    coord.shared_state.framework_agent_batches = [{"batch_id": "latest-batch"}]
    coord.shared_state.framework_agent_phase_progress = []
    task = types.SimpleNamespace(task_id="ip-9", params={"specialist_task_id": "spec-9"})
    result = types.SimpleNamespace(result={"status": "reverted", "delta_pct": -1.0})
    coord._record_framework_agent_authored_outcome(task=task, result=result)
    row = coord.shared_state.framework_agent_phase_progress[-1]
    assert row["candidate_id"] == "cand-from-map"
    assert row["batch_id"] == "latest-batch"
    assert row["status"] == "reverted"


# --------------------------------------------------------------------------
# _record_framework_agent_authoring_empty_outcome
# --------------------------------------------------------------------------
def _enter_fpr(coord: Coordinator) -> None:
    coord.shared_state.phase = ps_mod.PHASE_FRAMEWORK_AGENT


def test_record_authoring_empty_guards(coord: Coordinator) -> None:
    _enter_fpr(coord)
    # Not authoring -> no-op.
    coord._record_framework_agent_authoring_empty_outcome(
        task=types.SimpleNamespace(task_id="t", params={}), done_payload={}
    )
    # Patch present -> no-op (integrate_patch will own the row).
    coord._record_framework_agent_authoring_empty_outcome(
        task=_authoring_task(),
        done_payload={"patches_written": ["p.patch"]},
    )
    # Config-lever deliverable -> no-op (config autosubmit owns the row).
    coord._record_framework_agent_authoring_empty_outcome(
        task=_authoring_task(),
        done_payload={"proposal_set": [{"extra_envs": {"X": "1"}}]},
    )
    assert not (coord.shared_state.framework_agent_phase_progress or [])


def test_record_authoring_empty_already_present(coord: Coordinator) -> None:
    _enter_fpr(coord)
    task = types.SimpleNamespace(
        task_id="spec-2",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "cand-2",
            "framework_batch_id": "b1",
            "framework_audit": {"semantic_status": "already_equivalent"},
        },
    )
    coord._record_framework_agent_authoring_empty_outcome(
        task=task, done_payload={"payload": {"patches_written": [], "summary": "already there"}}
    )
    row = coord.shared_state.framework_agent_phase_progress[-1]
    assert row["candidate_id"] == "cand-2"
    assert row["status"] == "already_present"
    # Idempotent: a second call does not append a duplicate.
    coord._record_framework_agent_authoring_empty_outcome(
        task=task, done_payload={"payload": {"patches_written": [], "summary": "again"}}
    )
    assert sum(1 for p in coord.shared_state.framework_agent_phase_progress if p["candidate_id"] == "cand-2") == 1


def test_record_authoring_empty_status_variants(coord: Coordinator) -> None:
    _enter_fpr(coord)
    coord.shared_state.framework_agent_phase_progress = []
    # not_present -> not_applicable.
    coord._record_framework_agent_authoring_empty_outcome(
        task=types.SimpleNamespace(
            task_id="s-a",
            params={
                "framework_agent_authoring": True,
                "framework_agent_candidate_id": "na-1",
                "framework_audit": {"semantic_status": "not_present"},
            },
        ),
        done_payload={"patches_written": [], "summary": "missing"},
    )
    # no audit -> author_empty.
    coord._record_framework_agent_authoring_empty_outcome(
        task=types.SimpleNamespace(
            task_id="s-b",
            params={"framework_agent_authoring": True, "framework_agent_candidate_id": "ae-1"},
        ),
        done_payload={"patches_written": []},
    )
    statuses = {p["candidate_id"]: p["status"] for p in coord.shared_state.framework_agent_phase_progress}
    assert statuses["na-1"] == "not_applicable"
    assert statuses["ae-1"] == "author_empty"


# ---------------------------------------------------------------------------
# Lenient ranker: an "applicable: false" reply never vetoes the phase
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ranker_applicable_false_falls_back_to_discovery_order(
    coord: Coordinator, monkeypatch
) -> None:
    cands = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    monkeypatch.setattr(coord.phase_framework, "_unprocessed_framework_agent_candidates", lambda: cands)
    _install_ranker(
        coord,
        monkeypatch,
        _FakeClient('{"applicable": false, "reason": "all off-arch"}'),
    )
    result = await coord._select_best_framework_agent_candidate()
    assert result is cands[0]
