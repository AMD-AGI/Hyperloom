# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for FRAMEWORK agent-ranked selection, semantic-audit routing,
ranker-client plumbing, config-lever extraction, and the cyclic phase-budget
dispatch guard.

These exercise the pure/sync helpers and the small async helpers directly
(stubbing the LLM client / fa phase-audit / KB writeback) so no event-loop GPU
work or network is needed."""

from __future__ import annotations

import types
import pytest

from hyperloom.orchestrator.loop import coordinator as coord_mod
from hyperloom.orchestrator.phases import machine_state as ps_mod
from hyperloom.orchestrator.actions.executors import _patch_source_pr as fpr_mod
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic", "robustness")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


# --------------------------------------------------------------------------
# _framework_config_levers_from_done
# --------------------------------------------------------------------------
def test_config_levers_non_dict_and_patch_precedence() -> None:
    f = coord_mod._framework_config_levers_from_done
    assert f(None) == {}
    # A patch deliverable is not a config-only outcome.
    assert f({"patches_written": ["a.patch"], "proposal_set": [{"extra_envs": {"X": "1"}}]}) == {}
    assert f({"proposal_set": "nope"}) == {}
    assert f({}) == {}


def test_config_levers_preserve_envs_and_args() -> None:
    f = coord_mod._framework_config_levers_from_done
    extra_args = '--enable-x --compilation-config \'{"mode": "max-autotune"}\' --bare'
    levers = f(
        {
            "proposal_set": [
                {
                    "extra_envs": {"VLLM_FOO": 1, "  ": "skipped"},
                    "extra_args": extra_args,
                }
            ]
        }
    )
    assert levers == {
        "extra_server_args": '--enable-x --compilation-config {"mode":"max-autotune"} --bare',
        "extra_envs": {"VLLM_FOO": "1"},
    }


def test_config_levers_args_as_list() -> None:
    f = coord_mod._framework_config_levers_from_done
    levers = f({"proposal_set": [{"extra_args": ["--flag", "value with space"]}]})
    assert levers == {}


def test_invalid_config_args_preserve_independent_env_overrides() -> None:
    f = coord_mod._framework_config_levers_from_done
    levers = f(
        {
            "proposal_set": [
                {
                    "extra_args": ["--flag", "value with space"],
                    "extra_envs": {"SAFE_ENV": "1"},
                }
            ]
        }
    )
    assert levers == {
        "extra_server_args": "",
        "extra_envs": {"SAFE_ENV": "1"},
    }


def test_config_levers_json_args_as_list_stay_unquoted() -> None:
    f = coord_mod._framework_config_levers_from_done
    levers = f(
        {
            "proposal_set": [
                {
                    "extra_args": [
                        "--json-model-override-args",
                        '{"rope_scaling":null}',
                    ],
                }
            ]
        }
    )
    assert levers == {
        "extra_server_args": '--json-model-override-args {"rope_scaling":null}',
        "extra_envs": {},
    }


# --------------------------------------------------------------------------
# _framework_agent_audit_skip_confident
# --------------------------------------------------------------------------


def test_collect_framework_agent_candidate_priors(coord: Coordinator) -> None:
    coord.shared_state.framework_agent_phase_progress = [
        "not-a-dict",  # skipped via the isinstance filter
        {"candidate_id": "c1", "status": "kept", "gain_pct": 3.2},
        {"candidate_id": "c2", "status": "in_flight"},  # non-terminal -> excluded
        {"candidate_id": "c3", "status": "critic_denied", "rationale": "off the bottleneck"},
    ]
    priors = coord._collect_framework_agent_candidate_priors()
    statuses = {o["status"] for o in priors["recent_outcomes"]}
    assert statuses == {"kept", "critic_denied"}
    # The denial reason has to reach the Critic, or the priors carry the
    # verdict without the argument behind it.
    denied = next(o for o in priors["recent_outcomes"] if o["status"] == "critic_denied")
    assert denied["rationale"] == "off the bottleneck"


# --------------------------------------------------------------------------
# _match_framework_agent_candidate
# --------------------------------------------------------------------------


def _stub_sanctioned_async_client(monkeypatch, recorder: list[dict] | None = None):
    """Patch ``llm_config``'s async-client contract with a resolving stub.

    The stub still runs ``llm_config.resolve_openai_client_config`` so the
    returned credentials remain the ones the call site actually asked for, then
    exposes them on a plain object. ``raising=False``: the contract is owned by
    ``llm_config``, and patching it keeps these tests independent of whether the
    ``openai`` SDK is installed.
    """
    from hyperloom.common import llm_config

    def _fake(**kwargs):
        kwargs.pop("timeout", None)
        cfg = llm_config.resolve_openai_client_config(**kwargs)
        if recorder is not None:
            recorder.append(kwargs)
        return types.SimpleNamespace(api_key=cfg.api_key, base_url=cfg.base_url, chat=object())

    monkeypatch.setattr(llm_config, "get_async_openai_client", _fake, raising=False)


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
        # Split into a couple of deltas to exercise accumulation.
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
        # The ranker streams; assert the streaming flag is set.
        assert kwargs.get("stream") is True
        if self._raise:
            raise RuntimeError("llm down")
        return _FakeStream(self._content)


class _FakeClient:
    def __init__(self, content: str, raise_exc: bool = False) -> None:
        self.chat = type("_C", (), {"completions": _FakeCompletions(content, raise_exc)})()


def _scripted_run_git(diff_text: str = "diff --git a b\n+x\n", fetch_ok: bool = True, seen: list | None = None):
    def _fake(args, timeout=None):  # noqa: ANN001
        sub = args[2] if len(args) > 2 else ""
        if seen is not None:
            seen.append(sub)
        if sub == "fetch":
            return (fetch_ok, "", "" if fetch_ok else "remote hung up")
        if sub == "rev-parse":
            return (True, "abc123headsha", "")
        if sub == "merge-base":
            return (True, "basesha", "")
        if sub == "diff":
            return (True, diff_text, "")
        return (True, "", "")

    return _fake


def test_materialize_pr_diff_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git())
    dest = tmp_path / "out" / "cand.patch"
    ok, err = fpr_mod._materialize_pr_diff_from_head(tmp_path / "root", {"pr_number": 1015}, dest, timeout_sec=30.0)
    assert ok is True and err == ""
    assert dest.read_text().startswith("diff --git")


def test_materialize_pr_diff_fetch_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(fetch_ok=False))
    ok, err = fpr_mod._materialize_pr_diff_from_head(
        tmp_path / "root", {"ref": "refs/pull/1/head"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "git fetch" in err


def test_materialize_pr_diff_empty_diff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(diff_text="   \n"))
    ok, err = fpr_mod._materialize_pr_diff_from_head(
        tmp_path / "root", {"head_sha": "deadbeef"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "empty diff" in err


def test_materialize_pr_diff_no_head_resolvable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git())
    ok, err = fpr_mod._materialize_pr_diff_from_head(tmp_path / "root", {}, tmp_path / "c.patch", timeout_sec=30.0)
    assert ok is False and "cannot resolve PR head" in err


def test_materialize_pr_diff_checks_out_nothing(monkeypatch, tmp_path) -> None:
    """Both ends of the diff range are shas, so no tree has to be materialized.

    The mode checked the head out into a worktree and diffed the bare repo
    anyway, paying a full checkout of a multi-gigabyte tree nothing read.
    """
    seen: list[str] = []
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git(seen=seen))
    ok, _err = fpr_mod._materialize_pr_diff_from_head(
        tmp_path / "root", {"head_sha": "deadbeef"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is True
    assert "worktree" not in seen


def test_materialize_pr_diff_ignores_an_unusable_pr_number(monkeypatch, tmp_path) -> None:
    """The row reaches us from the KB and from LLM-authored proposals alike.

    A non-numeric number used to reach ``int()`` and raise out of the
    executor; it now reads as "no PR number".
    """
    monkeypatch.setattr(fpr_mod, "_run_git", _scripted_run_git())
    ok, err = fpr_mod._materialize_pr_diff_from_head(
        tmp_path / "root", {"pr_number": "not-a-number"}, tmp_path / "c.patch", timeout_sec=30.0
    )
    assert ok is False and "cannot resolve PR head" in err


# --------------------------------------------------------------------------
# Dispatch pause on a spent phase budget
# --------------------------------------------------------------------------
def test_dispatch_pause_phase_not_gated(coord: Coordinator) -> None:
    coord.shared_state.phase = "PRELUDE"
    assert coord._dispatch_paused_for_phase_budget() is False


def test_dispatch_pause_budget_spent(coord: Coordinator, monkeypatch) -> None:
    # The pause is length-agnostic: it fires whenever the phase budget is spent,
    # regardless of is_long_run (the dispatcher no longer reads it), so short and
    # long runs both pause new dispatch — consistent with the phase-advance gates.
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    monkeypatch.setattr(
        coord_mod._phase_state,
        "phase_budget_remaining_seconds",
        lambda _s, budget_pct=None: 0.0,
    )
    assert coord._dispatch_paused_for_phase_budget() is True


def test_dispatch_pause_budget_remaining(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.phase = "KERNEL_AGENT"
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
    done = {"proposal_set": [{"name": "mtp-toggle", "extra_envs": {"VLLM_MTP": "1"}, "extra_args": "--speculative 4"}]}
    before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_framework_config(task=_authoring_task(), done_payload=done)
    assert len(coord.state.pending_proposals) == before + 1
    prop = next(iter(coord.state.pending_proposals.values()))
    assert prop.action_name == "integrate_patch"
    params = (prop.payload or {}).get("params") or {}
    assert params["framework_agent_authoring"] is True
    assert params["framework_agent_candidate_id"] == "cand-1"
    assert params["extra_server_args"] == "--speculative 4"
    assert params["extra_envs"] == {"VLLM_MTP": "1"}
    assert "config_changes" not in params


@pytest.mark.asyncio
async def test_autosubmit_config_idempotent_on_existing_verdict(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.shared_state, "get_specialist_patch_verdict", lambda _sid: "approve", raising=False)
    done = {"proposal_set": [{"name": "n", "extra_envs": {"X": "1"}}]}
    await coord._maybe_autosubmit_framework_config(task=_authoring_task(), done_payload=done)
    assert not coord.state.pending_proposals


def _enablement_authoring_task(task_id: str = "spec-enable-1") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        task_id=task_id,
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": "cand-e",
            "framework_batch_id": "batch-e",
            "enablement": True,
            "enablement_before_signature": {"kind": "unregistered_arch"},
            "enablement_setup_commands": ["pip install -U vllm==0.21.0"],
        },
    )


@pytest.mark.asyncio
async def test_autosubmit_config_enablement_propagates_marker_and_setup(coord: Coordinator) -> None:
    """Regression: a config-lever ENABLEMENT deliverable must carry the
    ``enablement`` marker + setup commands into integrate_patch, otherwise the
    integrate result never gets ``enablement=True`` and ``_maybe_rearm_enablement``
    no-ops, the stall streak never advances, and the run spins until wall-clock.
    """
    done = {
        "proposal_set": [{"name": "v4-serve-flags", "extra_args": "--tokenizer-mode deepseek_v4"}],
        # NEW setup command proposed by the specialist in this deliverable.
        "setup_commands": ["pip install -U aiter"],
    }
    before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_framework_config(task=_enablement_authoring_task(), done_payload=done)
    assert len(coord.state.pending_proposals) == before + 1
    prop = next(iter(coord.state.pending_proposals.values()))
    params = (prop.payload or {}).get("params") or {}
    assert params.get("enablement") is True
    assert params["enablement_before_signature"] == {"kind": "unregistered_arch"}
    # Base setup (from spec_params) + new setup (from done_payload), deduped/merged.
    assert params["enablement_setup_commands"] == ["pip install -U vllm==0.21.0", "pip install -U aiter"]


@pytest.mark.asyncio
async def test_autosubmit_config_enablement_setup_only_still_routes(coord: Coordinator) -> None:
    """An enablement deliverable with NO config levers (setup-only stack upgrade)
    must still reach integrate_patch so the stall accounting can advance."""
    done = {"proposal_set": [], "setup_commands": ["pip install -U vllm==0.21.0"]}
    before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_framework_config(task=_enablement_authoring_task(), done_payload=done)
    assert len(coord.state.pending_proposals) == before + 1
    prop = next(iter(coord.state.pending_proposals.values()))
    params = (prop.payload or {}).get("params") or {}
    assert params.get("enablement") is True
    assert params.get("extra_server_args") == ""
    assert params.get("extra_envs") == {}
    assert "config_changes" not in params


@pytest.mark.asyncio
async def test_autosubmit_config_build_only_skips_integrate(coord: Coordinator) -> None:
    done = {
        "proposal_set": [{"name": "build-aiter"}],
        "needs_targeted_build": {
            "component": "aiter",
            "capability": "deepseek_v4_decode",
            "ref": "v0.1.15.post2",
        },
    }

    await coord._maybe_autosubmit_framework_config(
        task=_enablement_authoring_task(),
        done_payload=done,
    )

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
            "framework_agent_authoring": True,
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
    task = types.SimpleNamespace(
        task_id="ip-9",
        params={"framework_agent_authoring": True, "specialist_task_id": "spec-9"},
    )
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
