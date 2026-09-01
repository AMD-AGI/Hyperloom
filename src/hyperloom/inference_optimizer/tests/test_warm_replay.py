# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Warm-recipe replay tests (enqueue skip/enqueue paths, promote decision logic, resume safety)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator


@dataclass
class _StubTask:
    task_id: str = "task-warm-1"
    kind: str = "replay_warm_recipe"
    params: dict = field(default_factory=dict)
    state: str = "succeeded"


@dataclass
class _StubSharedState:
    """Minimal SharedState surface the warm-replay helpers read/write."""

    framework: str = "sglang"
    model_name: str = "DeepSeek-R1"
    gpu_type: str = "MI300X"
    baseline_tput: float = 600.0
    baseline_config_path: str = "/tmp/baseline.yaml"
    warm_start_recipe: dict = field(default_factory=dict)
    warm_start_context: dict = field(default_factory=dict)
    warm_replay_attempted: bool = False
    warm_replay_outcome: dict = field(default_factory=dict)
    warm_history_injected: bool = False
    auto_roofline_pending_task_id: str = ""
    stop_reason: str = ""
    enable_roofline: bool = True
    last_baseline: dict = field(default_factory=dict)
    explore_search: dict = field(default_factory=dict)
    optimization_stack: list = field(default_factory=list)
    gain_per_stack_entry: list = field(default_factory=list)
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    cumulative_gain_validated_stack_len: int = 0
    current_best: dict = field(default_factory=dict)
    baseline_perf: dict | None = None
    tick: int = 0
    phase: str = "PRELUDE"
    conc: int = 64
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0

    def save(self, *args, **kwargs):  # noqa: D401 — stub
        pass

    def append_stack_gain_entry(
        self, *, action, variant_name, new_tput, extra_server_args="", ts=None, candidate=None
    ):
        from hyperloom.common.perf_metric import session_gain_from_measurement

        entry_gain_pct, _used = session_gain_from_measurement(
            float(new_tput or 0.0),
            state=self,
            candidate=candidate,
            base_tput=float(self.baseline_tput or 0.0),
        )
        self.gain_per_stack_entry.append(entry_gain_pct)
        return entry_gain_pct

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason


class _StubTaskRegistry:
    """Captures ``create_or_return_existing`` calls so tests can assert."""

    def __init__(self):
        self.calls: list[dict] = []

    async def create_or_return_existing(
        self,
        *,
        kind,
        params,
        idempotency_key,
        **kwargs,
    ):
        self.calls.append(
            {
                "kind": kind,
                "params": dict(params),
                "idempotency_key": idempotency_key,
            }
        )
        task = _StubTask(
            task_id=f"task-{idempotency_key}",
            kind=kind,
            params=dict(params),
        )
        return task, False


def _make_coord(
    tmp_path: Path,
    *,
    warm_start_recipe: dict | None = None,
    warm_start_context: dict | None = None,
    warm_replay_enabled: bool = True,
    warm_replay_min_confidence: float = 0.7,
    warm_replay_min_reproduce_pct: float = 0.8,
    warm_replay_attempted: bool = False,
) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = _StubSharedState(
        warm_start_recipe=warm_start_recipe or {},
        warm_start_context=warm_start_context or {},
        warm_replay_attempted=warm_replay_attempted,
    )
    coord.tasks = _StubTaskRegistry()
    coord._warm_replay_enabled = warm_replay_enabled
    coord._warm_replay_min_confidence = warm_replay_min_confidence
    coord._warm_replay_min_reproduce_pct = warm_replay_min_reproduce_pct
    coord._journal = None
    return coord


def _warm_recipe_t1(
    *,
    extra_server_args: str = "--attention-backend AITER",
    extra_envs: dict | None = None,
    expected_gain_pct: float = 25.0,
    confidence: float = 0.85,
    tier: str = "exact",
    sessions: list | None = None,
    what_failed: list | None = None,
) -> dict:
    """Build a fake warm_start_recipe payload; ``expected_gain_pct`` lands in ``attrs.sessions[0].gain_pct``."""
    recipe_sessions = (
        sessions
        if sessions is not None
        else [
            {"session_id": "prior-session-A", "gain_pct": expected_gain_pct, "stack_len": 1},
        ]
    )
    attrs: dict = {
        "model": "DeepSeek-R1",
        "hardware": "MI300X",
        "framework": "sglang",
        "best_config": {
            "extra_server_args": extra_server_args,
            "extra_envs": dict(extra_envs or {}),
        },
        "sessions": recipe_sessions,
    }
    if what_failed is not None:
        attrs["what_failed"] = what_failed
    return {
        "tier": tier,
        "confidence": confidence,
        "recipe": {
            "id": 1,
            "canonical_id": "recipe:deepseek-r1:sglang:mi300x",
            "kind": "recipe",
            "attrs": attrs,
        },
    }


def _warm_recipe_v2_arbor(
    *,
    extra_server_args: str = "--x",
    extra_envs: dict | None = None,
    expected_gain_pct: float = 25.0,
    tier: str = "exact",
    confidence: float = 1.0,
) -> dict:
    """v2 RecipeKB arbor shape: ``best_config`` / ``sessions`` at the TOP LEVEL of ``recipe`` (no ``attrs`` wrapper)."""
    return {
        "tier": tier,
        "confidence": confidence,
        "recipe": {
            "canonical_id": "inference:deepseek-r1:mi300x:sglang:0.4.5:fp8",
            "model": "deepseek-r1",
            "hardware": "mi300x",
            "framework": "sglang",
            "best_config": {
                "extra_server_args": extra_server_args,
                "extra_envs": dict(extra_envs or {}),
            },
            "sessions": [
                {"session_id": "prior-A", "gain_pct": expected_gain_pct, "stack_len": 1},
            ],
        },
    }


@pytest.mark.asyncio
async def test_current_recipe_replay_uses_sdk_sections_and_global_order(
    tmp_path,
    monkeypatch,
):
    warm_dir = tmp_path / "runtime" / "remote_recipe"
    refs = [
        "framework/overlays/000001/00-framework.patch",
        "explore/overlays/000002/00-explore.patch",
    ]
    targets = ["src/framework_fix.py", "src/explore_fix.py"]
    for ref, target in zip(refs, targets, strict=True):
        patch = warm_dir / "files" / ref
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(
            f"diff --git a/{target} b/{target}\n--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-old\n+new\n",
            encoding="utf-8",
        )
    table_ref = "kernel/gemm/table.json"
    table = warm_dir / "files" / table_ref
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("{}", encoding="utf-8")
    warm_dir.mkdir(parents=True, exist_ok=True)
    (warm_dir / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": 1,
                "record_kind": "hyperloom_recipe",
                "value": {
                    "explore": {
                        "extra_server_args": "--explore --shared",
                        "extra_envs": {"EXPLORE": "1", "SHARED": "same"},
                        "patches": [refs[1]],
                    },
                    "framework": {
                        "extra_server_args": "--framework",
                        "extra_envs": {"FRAMEWORK": "1", "SHARED": "same"},
                        "patches": [refs[0]],
                    },
                    "patch_timeline": refs,
                    "kernel": {
                        "gemm": {
                            "optimizations": [
                                {
                                    "tuned_file": table_ref,
                                    "extra_server_args": "--shared --kernel",
                                    "extra_envs": {
                                        "SHARED": "same",
                                        "KERNEL": "1",
                                    },
                                }
                            ]
                        },
                        "fusion": {},
                        "rewrite": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_DRAFT_DIR", str(tmp_path / "runtime" / "draft"))
    monkeypatch.setenv("KB_WARM_START_DIR", str(warm_dir))
    framework_root = tmp_path / "framework"
    framework_root.mkdir()
    for target in targets:
        path = framework_root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_session_framework_root",
        lambda: str(framework_root),
    )
    coord = _make_coord(
        tmp_path,
        warm_start_recipe={
            "tier": "exact",
            "confidence": 1.0,
            "recipe": {
                "canonical_id": "inference:test",
                "record_kind": "hyperloom_recipe",
                "validated_gain_pct": 12.0,
            },
        },
        warm_start_context={
            "recommended_replay": {
                "extra_server_args": "--must-not-be-read",
                "patches": [{"patch_file": "legacy.patch"}],
            },
        },
    )

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is not None
    assert task.params["extra_server_args"] == "--explore --shared --framework --kernel"
    assert task.params["extra_envs"] == {
        "EXPLORE": "1",
        "SHARED": "same",
        "FRAMEWORK": "1",
        "KERNEL": "1",
    }
    assert [patch["patch_file"] for patch in task.params["patches"]] == refs
    assert task.params["required_patch_timeline"] is True


def _patch_current_sdk_readers(
    monkeypatch,
    tmp_path,
    *,
    timeline,
    explore_refs,
    framework_refs,
    explore_config=None,
    framework_config=None,
    gemm=None,
):
    from hyperloom.orchestrator.knowledge.agent_kb import (
        ExploreAgentKB,
        FrameworkAgentKB,
        KernelAgentKB,
        RecipeReplayKB,
    )

    refs = set(timeline) | set(explore_refs) | set(framework_refs)
    paths = {}
    for index, ref in enumerate(refs):
        path = tmp_path / f"member-{index}.patch"
        path.write_text("patch", encoding="utf-8")
        paths[ref] = path
    if gemm:
        for row in gemm.get("optimizations") or []:
            ref = str(row.get("tuned_file") or "")
            if ref and ref not in paths:
                path = tmp_path / f"member-{len(paths)}.json"
                path.write_text("{}", encoding="utf-8")
                paths[ref] = path

    class _Owner:
        active = True

        def __init__(self, config, refs):
            self.config = config
            self.refs = refs

        def read_config(self):
            return dict(self.config or {})

        def read_patches(self):
            return list(self.refs)

        def read_patch_roots(self):
            return {}

        def prior_file(self, ref):
            return paths.get(ref)

    class _Kernel:
        active = True

        def read_gemm(self):
            return dict(gemm or {})

        def read_fusion(self):
            return {}

        def read_rewrite(self):
            return {}

        def prior_file(self, ref):
            return paths.get(ref)

    class _Replay:
        active = True

        def read_config(self):
            configs = [explore_config or {}, framework_config or {}]
            return {
                "extra_server_args": " ".join(
                    str(config.get("extra_server_args") or "").strip()
                    for config in configs
                    if str(config.get("extra_server_args") or "").strip()
                ),
                "extra_envs": {
                    str(key): str(value)
                    for config in configs
                    for key, value in (config.get("extra_envs") or {}).items()
                },
            }

        def read_patch_timeline(self):
            return list(timeline)

    monkeypatch.setattr(
        ExploreAgentKB,
        "open",
        classmethod(lambda cls: _Owner(explore_config or {}, explore_refs)),
    )
    monkeypatch.setattr(
        FrameworkAgentKB,
        "open",
        classmethod(lambda cls: _Owner(framework_config or {}, framework_refs)),
    )
    monkeypatch.setattr(
        KernelAgentKB,
        "open",
        classmethod(lambda cls: _Kernel()),
    )
    monkeypatch.setattr(
        RecipeReplayKB,
        "open",
        classmethod(lambda cls: _Replay()),
    )


@pytest.mark.asyncio
async def test_current_recipe_skips_undersized_context_for_target_workload(
    tmp_path,
    monkeypatch,
):
    _patch_current_sdk_readers(
        monkeypatch,
        tmp_path,
        timeline=[],
        explore_refs=[],
        framework_refs=[],
        explore_config={"extra_server_args": "--context-length 6144"},
    )
    coord = _make_coord(
        tmp_path,
        warm_start_recipe={
            "tier": "exact",
            "confidence": 1.0,
            "recipe": {
                "canonical_id": "inference:test",
                "record_kind": "hyperloom_recipe",
                "validated_gain_pct": 12.0,
            },
        },
    )
    coord.shared_state.isl = 8192
    coord.shared_state.osl = 1024
    coord.shared_state.max_model_len = 32768

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert coord.tasks.calls == []
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert "context_length=6144 < isl+osl=9216" in (coord.shared_state.warm_replay_outcome["reason"])


@pytest.mark.asyncio
async def test_legacy_recipe_skips_undersized_context_for_target_workload(
    tmp_path,
):
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(extra_server_args="--context-length 6144 --watchdog-timeout 1800"),
    )
    coord.shared_state.isl = 8192
    coord.shared_state.osl = 1024
    coord.shared_state.max_model_len = 32768

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert coord.tasks.calls == []
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert "context_length=6144 < isl+osl=9216" in (coord.shared_state.warm_replay_outcome["reason"])


@pytest.mark.asyncio
async def test_warm_replay_does_not_misclassify_preflight_code_bug(
    tmp_path,
    monkeypatch,
):
    from hyperloom.orchestrator.actions.executors import _grid_server_args

    def _bug(*_args, **_kwargs):
        raise AttributeError("preflight implementation bug")

    monkeypatch.setattr(
        _grid_server_args,
        "validate_warm_replay_context_length",
        _bug,
    )
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
    )

    with pytest.raises(AttributeError, match="preflight implementation bug"):
        await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert coord.tasks.calls == []
    assert "reason" not in coord.shared_state.warm_replay_outcome


@pytest.mark.asyncio
async def test_current_recipe_patch_skips_without_active_framework_root(
    tmp_path,
    monkeypatch,
):
    _patch_current_sdk_readers(
        monkeypatch,
        tmp_path,
        timeline=["explore/p.patch"],
        explore_refs=["explore/p.patch"],
        framework_refs=[],
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_session_framework_root",
        lambda: "",
    )
    coord = _make_coord(
        tmp_path,
        warm_start_recipe={
            "tier": "exact",
            "confidence": 1.0,
            "recipe": {
                "canonical_id": "inference:test",
                "record_kind": "hyperloom_recipe",
            },
        },
    )
    prepared = 0

    async def _prepare(*_args, **_kwargs):
        nonlocal prepared
        prepared += 1
        return {"status": "prepared"}

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert prepared == 0
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert coord.shared_state.warm_replay_outcome["reason"] == "patch_targets_invalid"
    assert coord.shared_state.warm_replay_outcome["framework_patch_root_allowlist"]


@pytest.mark.parametrize(
    ("timeline", "explore_refs", "framework_refs", "match"),
    [
        (["explore/a.patch", "explore/a.patch"], ["explore/a.patch"], [], "duplicate"),
        (["explore/a.patch"], ["explore/a.patch", "explore/a.patch"], [], "duplicate"),
        (["explore/a.patch"], ["explore/a.patch"], ["framework/b.patch"], "exactly equal"),
        (
            ["explore/a.patch", "framework/b.patch"],
            ["explore/a.patch"],
            [],
            "exactly equal",
        ),
    ],
)
def test_current_recipe_timeline_requires_exact_unique_owner_ref_set(
    tmp_path,
    monkeypatch,
    timeline,
    explore_refs,
    framework_refs,
    match,
):
    _patch_current_sdk_readers(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        explore_refs=explore_refs,
        framework_refs=framework_refs,
    )
    coord = _make_coord(tmp_path)

    with pytest.raises(ValueError, match=match):
        coord.phase_prelude._read_current_recipe_replay()


@pytest.mark.asyncio
async def test_current_kernel_conflict_fails_before_preparation(
    tmp_path,
    monkeypatch,
):
    _patch_current_sdk_readers(
        monkeypatch,
        tmp_path,
        timeline=[],
        explore_refs=[],
        framework_refs=[],
        explore_config={"extra_envs": {"SHARED": "recipe"}},
        gemm={
            "optimizations": [
                {
                    "tuned_file": "kernel/gemm/table.json",
                    "extra_envs": {"SHARED": "kernel"},
                }
            ]
        },
    )
    coord = _make_coord(
        tmp_path,
        warm_start_recipe={
            "tier": "exact",
            "confidence": 1.0,
            "recipe": {
                "canonical_id": "inference:test",
                "record_kind": "hyperloom_recipe",
            },
        },
    )
    prepared = 0

    async def _prepare(*_args, **_kwargs):
        nonlocal prepared
        prepared += 1
        return {"status": "prepared"}

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert prepared == 0
    assert "env conflict for SHARED" in coord.shared_state.warm_replay_outcome["reason"]


@pytest.mark.asyncio
async def test_current_history_only_view_never_auto_replays(tmp_path):
    coord = _make_coord(
        tmp_path,
        warm_start_recipe={
            "tier": "exact",
            "confidence": 1.0,
            "recipe": {
                "canonical_id": "inference:test",
                "record_kind": "hyperloom_recipe",
                "replayable": False,
                "replay_disabled_reason": "legacy_history_only",
                "what_worked": [{"description": "old win"}],
            },
        },
    )
    prepared = 0

    async def _prepare(*_args, **_kwargs):
        nonlocal prepared
        prepared += 1
        return {"status": "prepared"}

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert prepared == 0
    assert coord.shared_state.warm_replay_outcome == {
        "status": "skipped",
        "reason": "legacy_history_only",
        "view_source": "",
    }


@pytest.mark.asyncio
async def test_warm_replay_skips_when_disabled_by_flag(tmp_path):
    """``--no-warm-replay`` → skip + flip the one-shot guard so a flag-less resume can't trigger replay."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=False,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert "disabled_by_flag" in coord.shared_state.warm_replay_outcome["reason"]
    assert coord.tasks.calls == []
    assert coord.shared_state.warm_replay_attempted is True


@pytest.mark.asyncio
async def test_warm_replay_resume_with_lost_disable_flag_is_still_blocked(
    tmp_path,
):
    """Resume safety: after a disabled launch flips warm_replay_attempted, a flag-less resume still short-circuits."""
    coord1 = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=False,
    )
    await coord1._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord1.shared_state.warm_replay_attempted is True
    coord2 = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_enabled=True,
        warm_replay_attempted=True,  # restored from state.json
    )
    task = await coord2._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord2.tasks.calls == []


@pytest.mark.asyncio
async def test_warm_replay_skips_when_already_attempted(tmp_path):
    """Resume safety: a prior boot already ran the replay; no second enqueue."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
        warm_replay_attempted=True,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.tasks.calls == []


@pytest.mark.asyncio
async def test_warm_replay_skips_when_no_warm_start_recipe(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe={})
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_attempted is True
    assert coord.shared_state.warm_replay_outcome["status"] == "skipped"
    assert coord.shared_state.warm_replay_outcome["reason"] == "no_warm_start_recipe"


@pytest.mark.asyncio
async def test_warm_replay_skips_when_confidence_below_threshold(tmp_path):
    """Only T1/T2 fire by default; lower-tier hits aren't worth a verify spend."""
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(
            confidence=0.55,
            tier="T3_same_family",
        ),
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "skipped"
    assert "below_threshold" in outcome["reason"]
    assert outcome["warm_recipe_tier"] == "T3_same_family"


@pytest.mark.asyncio
async def test_warm_replay_skips_when_best_config_empty(tmp_path):
    """A seed-only recipe (no actual args) isn't worth replaying."""
    recipe = _warm_recipe_t1(extra_server_args="", extra_envs={})
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is None
    assert coord.shared_state.warm_replay_outcome["reason"] == "best_config_empty"


@pytest.mark.asyncio
async def test_warm_replay_enqueues_with_warm_best_config_args_envs(tmp_path):
    """Happy path: a high-confidence T1 hit creates a task carrying the warm config in ``params``."""
    recipe = _warm_recipe_t1(
        extra_server_args="--attention-backend AITER --kv-cache-dtype fp8",
        extra_envs={"VLLM_ROCM_USE_AITER": "1"},
        expected_gain_pct=25.0,
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is not None
    assert task.kind == "replay_warm_recipe"
    assert len(coord.tasks.calls) == 1
    call = coord.tasks.calls[0]
    assert call["kind"] == "replay_warm_recipe"
    assert call["idempotency_key"] == "warm-replay-prelude"
    params = call["params"]
    assert params["extra_server_args"] == "--attention-backend AITER --kv-cache-dtype fp8"
    assert params["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert params["config_path"] == "/tmp/baseline.yaml"
    assert params["warm_expected_gain_pct"] == 25.0
    assert params["warm_recipe_tier"] == "exact"
    assert params["warm_recipe_conf"] == 0.85
    assert params["baseline_tput_anchor"] == 600.0
    assert coord.shared_state.warm_replay_attempted is True
    assert coord.shared_state.warm_replay_outcome["status"] == "in_flight"
    assert coord.shared_state.warm_replay_outcome["replay_task_id"] == task.task_id


@pytest.mark.asyncio
async def test_warm_replay_enqueues_with_v2_arbor_top_level_best_config(tmp_path):
    """Regression: warm-replay must read best_config from the v2 arbor TOP LEVEL, else it skips with best_config_empty."""
    recipe = _warm_recipe_v2_arbor(
        extra_server_args="--attention-backend AITER",
        extra_envs={"VLLM_ROCM_USE_AITER": "1"},
        expected_gain_pct=25.0,
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is not None, "v2 arbor top-level best_config not read"
    params = coord.tasks.calls[0]["params"]
    assert params["extra_server_args"] == "--attention-backend AITER"
    assert params["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert params["warm_expected_gain_pct"] == 25.0


@pytest.mark.asyncio
async def test_warm_replay_prefers_warm_start_context_recommended_replay(tmp_path):
    """status=hit WarmStartContext: warm-replay launches from its ``recommended_replay`` champion (args/envs) over the raw recipe row."""
    recipe = _warm_recipe_t1(
        extra_server_args="--from-recipe-row",
        extra_envs={"RECIPE": "1"},
        expected_gain_pct=25.0,
    )
    context = {
        "status": "hit",
        "match": {"tier": "exact", "confidence": 0.85, "source": "gbrain"},
        "recommended_replay": {
            "extra_server_args": "--from-context --cuda-graph-max-bs 256",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
            "expected_gain_pct": 25.0,
            "best_throughput": 5430.9,
            "donor_canonical_id": "inference:donor:h:f:v:p",
            "donor_model": "donor-model",
            "donor_session_id": "donor-session",
            "donor_family_tags": ["moe"],
            "donor_gain_pct": 25.0,
            "donor_breakdown_link": "https://example.test/session/donor-session",
        },
    }
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=recipe,
        warm_start_context=context,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is not None
    params = coord.tasks.calls[0]["params"]
    assert params["extra_server_args"] == "--from-context --cuda-graph-max-bs 256"
    assert params["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert coord.shared_state.warm_replay_outcome["donor_model"] == "donor-model"
    assert coord.shared_state.warm_replay_outcome["donor_session_id"] == "donor-session"
    assert coord.shared_state.warm_replay_outcome["donor_family_tags"] == ["moe"]


@pytest.mark.asyncio
async def test_warm_replay_falls_back_to_recipe_when_context_not_hit(tmp_path):
    """A non-hit (e.g. seed_only) WarmStartContext must NOT override the recipe-derived champion."""
    recipe = _warm_recipe_t1(
        extra_server_args="--from-recipe-row",
        extra_envs={"RECIPE": "1"},
        expected_gain_pct=25.0,
    )
    context = {"status": "seed_only", "recommended_replay": {}}
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=recipe,
        warm_start_context=context,
    )
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert task is not None
    params = coord.tasks.calls[0]["params"]
    assert params["extra_server_args"] == "--from-recipe-row"
    assert params["extra_envs"] == {"RECIPE": "1"}


def test_promote_warm_replay_reproduced_pushes_stack_and_updates_gain(
    tmp_path,
):
    """When measured gain ≥ expected × min_reproduce, push the warm config onto the stack and bump the validated gain."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "warm_recipe_tier": "exact",
        "warm_recipe_conf": 0.85,
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        }
    )
    # Measured 23% gain (600 -> 738), above the 20% threshold.
    result = {"status": "succeeded", "output_throughput": 738.0}
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert outcome["actual_gain_pct"] == 23.0
    assert outcome["throughput_after"] == 738.0
    assert len(coord.shared_state.optimization_stack) == 1
    entry = coord.shared_state.optimization_stack[0]
    assert entry["action"] == "replay_warm_recipe"
    assert entry["extra_server_args"] == "--attention-backend AITER"
    assert entry["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert entry["tput"] == 738.0
    assert coord.shared_state.gain_per_stack_entry == [23.0]
    assert coord.shared_state.cumulative_gain_validated == 23.0
    assert coord.shared_state.cumulative_gain_validated_ts
    assert coord.shared_state.cumulative_gain_validated_stack_len == 1
    assert coord.shared_state.current_best["action"] == "replay_warm_recipe"
    assert coord.shared_state.current_best["tput"] == 738.0


def test_promote_warm_replay_keeps_prebaseline_enablement_as_zero_gain_anchor(
    tmp_path,
):
    """A PRELUDE enablement config stays reproducible but contributes no gain."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.optimization_stack = [
        {
            "action": "integrate_patch",
            "baseline_enablement": True,
            "attribution_eligible": False,
            "tput": 600.0,
        }
    ]
    coord.shared_state.gain_per_stack_entry = [None]
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }

    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 738.0},
        task=_StubTask(params={"extra_server_args": "--attention-backend AITER"}),
    )

    assert [entry["action"] for entry in coord.shared_state.optimization_stack] == [
        "integrate_patch",
        "replay_warm_recipe",
    ]
    assert coord.shared_state.gain_per_stack_entry == [None, 23.0]
    assert coord.shared_state.cumulative_gain_validated == 23.0
    assert coord.shared_state.cumulative_gain_validated_stack_len == 2


def test_promote_warm_replay_rejected_by_failed_quality_gate(tmp_path):
    """A faster warm config that FAILS the image-quality gate vs the baseline
    reference must NOT be promoted (no stack push, no current_best), even though
    its throughput beats baseline.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "warm_recipe_tier": "exact",
        "warm_recipe_conf": 0.85,
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        }
    )
    # +23% throughput but the quality gate FAILED (mse above the ceiling).
    result = {
        "status": "succeeded",
        "output_throughput": 738.0,
        "quality_gate": {
            "passed": False,
            "mse": 0.0295,
            "mse_max": 0.002,
            "ssim": 1.0,
            "lpips": 0.0,
        },
    }
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "quality_failed"
    assert outcome["quality_gate"]["passed"] is False
    assert coord.shared_state.optimization_stack == []
    assert coord.shared_state.current_best == {}
    assert coord.shared_state.cumulative_gain_validated == 0.0


@pytest.mark.parametrize(
    "result",
    [
        {"status": "failed", "error": "launch failed"},
        {"status": "succeeded", "output_throughput": 0.0},
        {
            "status": "succeeded",
            "output_throughput": 700.0,
            "quality_gate": {"passed": False},
        },
        {"status": "succeeded", "output_throughput": 600.0},
    ],
)
def test_all_revert_branches_retain_pending_on_rollback_failure(
    tmp_path,
    result,
):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_pending = {"task_id": "warm"}
    coord.shared_state.warm_replay_outcome = {"status": "in_flight"}
    coord.phase_prelude._rollback_combined_warm = (  # type: ignore[method-assign]
        lambda *_args: {"ok": False, "errors": ["restore failed"]}
    )
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "extra_server_args": "--warm",
        }
    )

    coord._promote_warm_replay(result, task=task)

    assert coord.shared_state.warm_replay_outcome["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_pending == {"task_id": "warm"}
    assert coord.shared_state.stop_reason == "warm_replay_rollback_failed"


def test_promote_warm_replay_passes_quality_gate_is_promoted(tmp_path):
    """A warm config that beats baseline AND clears the quality gate (mse within
    the ceiling) is promoted normally."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        }
    )
    result = {
        "status": "succeeded",
        "output_throughput": 738.0,
        "quality_gate": {"passed": True, "mse": 0.0005, "mse_max": 0.002},
    }
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert len(coord.shared_state.optimization_stack) == 1
    assert coord.shared_state.current_best["action"] == "replay_warm_recipe"


def test_promote_warm_replay_double_run_uses_hot_measure_round(tmp_path):
    """Double-run replay uses the hot measure round for gain/current_best.

    The discarded warmup round is retained only under ``cold_tput`` for audit.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "warm_recipe_tier": "exact",
        "warm_recipe_conf": 0.85,
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        }
    )
    result = {
        "status": "succeeded",
        "output_throughput": 738.0,
        "warmup_round_tput": 690.0,
    }
    coord._promote_warm_replay(result, task=task)

    cb = coord.shared_state.current_best
    assert cb["action"] == "replay_warm_recipe"
    assert cb["tput"] == 738.0
    # The measured rounds are audit metadata on the stack entry, not config.
    assert "hot_tput" not in cb
    assert "cold_tput" not in cb
    entry = coord.shared_state.optimization_stack[0]
    assert entry["tput"] == 738.0
    assert entry["hot_tput"] == 738.0
    assert entry["cold_tput"] == 690.0
    assert entry["gain_pct"] == 23.0
    assert coord.shared_state.cumulative_gain_validated == 23.0


def test_promote_warm_replay_adopts_on_any_positive_gain(tmp_path):
    """Any replay tput above baseline seeds the stack, even below the historical reproduce bar."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "baseline_tput_anchor": 600.0,
        }
    )
    # +10% vs baseline; below the historical bar but still adopted.
    result = {"status": "succeeded", "output_throughput": 660.0}
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert outcome["actual_gain_pct"] == 10.0
    assert outcome.get("below_historical_reproduce_pct") is True
    assert len(coord.shared_state.optimization_stack) == 1
    assert coord.shared_state.current_best["action"] == "replay_warm_recipe"


def test_promote_warm_replay_composite_adopts_flat_output_input_lift(tmp_path, monkeypatch):
    """Flag on: +20% input / flat output reproduces; the old tput bar would have drifted."""
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    baseline_perf = {
        "output_throughput": 600.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_perf = dict(baseline_perf)
    coord.shared_state.current_best = {"action": "baseline", "tput": 600.0, **baseline_perf}
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    result = {
        "status": "succeeded",
        "output_throughput": 600.0,
        "input_throughput": 12000.0,
        "intvty_p90": 700.0,
    }
    coord._promote_warm_replay(
        result,
        task=_StubTask(
            params={
                "extra_server_args": "--attention-backend AITER",
                "baseline_tput_anchor": 600.0,
            }
        ),
    )

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert outcome["used_composite"] is True
    # S = 0.55 * 20% input = 11%; output did not move.
    assert outcome["actual_gain_pct"] == pytest.approx(11.0)
    assert outcome["throughput_after"] == 600.0
    assert outcome.get("below_historical_reproduce_pct") is not True
    assert coord.shared_state.current_best["action"] == "replay_warm_recipe"
    assert coord.shared_state.current_best["tput"] == 600.0
    assert coord.shared_state.current_best["input_throughput"] == 12000.0
    assert len(coord.shared_state.optimization_stack) == 1
    assert coord.shared_state.gain_per_stack_entry == [pytest.approx(11.0)]
    assert coord.shared_state.cumulative_gain_validated == pytest.approx(11.0)


def test_promote_warm_replay_composite_combined_contract_uses_score_bar(tmp_path, monkeypatch):
    """Flag on + combined 1% bar: the same flat-output input lift still clears KEEP."""
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    baseline_perf = {
        "output_throughput": 600.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_perf = dict(baseline_perf)
    coord.shared_state.current_best = {"action": "baseline", "tput": 600.0, **baseline_perf}
    coord.shared_state.warm_replay_outcome = {"status": "in_flight", "expected_gain_pct": 0.0}
    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 600.0,
            "input_throughput": 12000.0,
            "intvty_p90": 700.0,
        },
        task=_StubTask(
            params={
                "extra_server_args": "--current",
                "baseline_tput_anchor": 600.0,
                "combined_current_contract": True,
                "combined_keep_threshold_pct": 1.0,
            }
        ),
    )

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "reproduced"
    assert outcome["used_composite"] is True
    assert outcome["actual_gain_pct"] == pytest.approx(11.0)


def test_promote_warm_replay_composite_falls_back_without_triple(tmp_path, monkeypatch):
    """Flag on but no intvty: still the output-tput path, so flat output is drift."""
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_perf = {
        "output_throughput": 600.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    coord.shared_state.warm_replay_outcome = {"status": "in_flight", "expected_gain_pct": 25.0}
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 600.0, "input_throughput": 12000.0},
        task=_StubTask(
            params={
                "extra_server_args": "--attention-backend AITER",
                "baseline_tput_anchor": 600.0,
            }
        ),
    )

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "drift"
    assert outcome["used_composite"] is False
    assert outcome["actual_gain_pct"] == 0.0
    assert coord.shared_state.optimization_stack == []


def test_promote_warm_replay_no_gain_is_drift(tmp_path):
    """Zero or negative measured gain → ``drift``, no stack push."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "baseline_tput_anchor": 600.0,
        }
    )
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=task)

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "drift"
    assert coord.shared_state.optimization_stack == []
    assert coord.shared_state.cumulative_gain_validated == 0.0


def test_promote_warm_replay_succeeded_but_zero_gain_is_drift(tmp_path):
    """``expected_gain_pct=0`` → any positive measurement is reproduced; zero/negative falls to drift."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 0.0,
    }
    task = _StubTask(params={"extra_server_args": "--foo"})
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.warm_replay_outcome["status"] == "drift"
    assert coord.shared_state.warm_replay_outcome["actual_gain_pct"] == 0.0


def test_promote_warm_replay_failed_records_outcome(tmp_path):
    """Subprocess failure → tag as ``failed`` with the error_class verbatim."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }
    result = {
        "status": "failed",
        "error_class": "crash",
        "error": "GPU OOM during prefill",
    }
    coord._promote_warm_replay(result, task=_StubTask())

    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "failed"
    assert outcome["error_class"] == "crash"
    assert "GPU OOM" in outcome["reason"]
    assert coord.shared_state.optimization_stack == []


# A FAILED replay_warm_recipe must route to _promote_warm_replay (which clears
# in_flight); otherwise PRELUDE never exits.
def test_failed_replay_is_routed_to_promote_not_unpromotable(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    assert (
        coord._is_promotable_result(
            "replay_warm_recipe",
            {"status": "failed", "error_class": "crash"},
        )
        is True
    ), (
        "failed replay must route to _promote_warm_replay so the in_flight "
        "flag is cleared; otherwise PRELUDE never exits"
    )
    assert (
        coord._is_promotable_result(
            "replay_warm_recipe",
            {"status": "succeeded", "output_throughput": 700.0},
        )
        is True
    )


def test_multi_file_kernel_targets_share_one_framework_root(
    tmp_path,
    monkeypatch,
):
    coord = _make_coord(tmp_path)
    framework_root = tmp_path / "framework"
    existing = framework_root / "python/sglang/srt/models/qwen3.py"
    added = framework_root / "python/sglang/srt/models/qwen3_fused_ops.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("original\n", encoding="utf-8")
    patch = tmp_path / "fusion.patch"
    patch.write_text(
        "diff --git a/python/sglang/srt/models/qwen3.py "
        "b/python/sglang/srt/models/qwen3.py\n"
        "--- a/python/sglang/srt/models/qwen3.py\n"
        "+++ b/python/sglang/srt/models/qwen3.py\n"
        "@@ -1 +1 @@\n"
        "-original\n"
        "+patched\n"
        "diff --git a/python/sglang/srt/models/qwen3_fused_ops.py "
        "b/python/sglang/srt/models/qwen3_fused_ops.py\n"
        "--- /dev/null\n"
        "+++ b/python/sglang/srt/models/qwen3_fused_ops.py\n"
        "@@ -0,0 +1 @@\n"
        "+new\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths._warm_replay_kernel_patch_roots",
        lambda: (str(framework_root),),
    )

    targets = coord.phase_prelude._resolve_kernel_target_paths(
        {
            "patch_path": str(patch),
        }
    )

    assert targets == [str(existing), str(added)]


def test_kernel_target_uses_allowlist_when_framework_root_does_not_match(
    tmp_path,
    monkeypatch,
):
    coord = _make_coord(tmp_path)
    active_root = tmp_path / "active"
    stale_root = tmp_path / "stale"
    stale_target = stale_root / "src/kernel.py"
    active_root.mkdir()
    stale_target.parent.mkdir(parents=True)
    stale_target.write_text("old\n", encoding="utf-8")
    patch = tmp_path / "kernel.patch"
    patch.write_text(
        "diff --git a/src/kernel.py b/src/kernel.py\n"
        "--- a/src/kernel.py\n"
        "+++ b/src/kernel.py\n"
        "@@ -1 +1 @@\n-old\n+new\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_session_framework_root",
        lambda: str(active_root),
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths._warm_replay_kernel_patch_roots",
        lambda: (str(stale_root),),
    )

    entry = {
        "patch_path": str(patch),
        "resolution_error": "old failure",
        "resolution_reason": "explicit_root_target_mismatch",
    }
    assert coord.phase_prelude._resolve_kernel_target_paths(entry) == [str(stale_target)]
    assert "resolution_error" not in entry
    assert "resolution_reason" not in entry


def test_create_only_kernel_target_requires_known_kernel_root(tmp_path, monkeypatch):
    coord = _make_coord(tmp_path)
    patch = tmp_path / "create.patch"
    patch.write_text(
        "diff --git a/src/new.py b/src/new.py\n--- /dev/null\n+++ b/src/new.py\n@@ -0,0 +1 @@\n+new\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_session_framework_root",
        lambda: "",
    )
    # Pinned to the shape a clean CI host has: the allowlist names roots, so the
    # operator still sees them, but none of them exists, so nothing survives as a
    # candidate. Leaving this to the host let a real tree such as /opt/flydsl
    # stand in as a candidate and mask the reason under test.
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths._warm_replay_kernel_patch_roots",
        lambda: (str(tmp_path / "absent_kernel_root"),),
    )

    entry = {"patch_path": str(patch)}
    assert coord.phase_prelude._resolve_kernel_target_paths(entry) == []
    assert entry["resolution_reason"] == "pure_create_requires_explicit_root"
    assert "allowlist=" in entry["resolution_error"]


def test_restored_kernel_plan_reresolves_root_before_blocking(
    tmp_path,
    monkeypatch,
):
    from hyperloom.orchestrator.framework.paths import WarmReplayRootResolution

    coord = _make_coord(tmp_path)
    entry = {
        "column": "fusion",
        "patch_path": str(tmp_path / "fusion.patch"),
        "resolution_reason": "active_kernel_patch_root_missing",
    }
    coord.shared_state.warm_kernel_kb_plan = [entry]
    calls = []

    def _resolve(*, patch_entries, precomputed_allowlist=None):
        calls.append(patch_entries)
        return WarmReplayRootResolution(
            root=str(tmp_path / "restored-framework"),
            source="session_framework_root",
            reason="",
            allowlist=(str(tmp_path / "restored-framework"),),
        )

    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_warm_replay_kernel_root",
        _resolve,
    )

    assert coord.phase_prelude._warm_replay_kernel_root_block_reason(coord.shared_state) is None
    assert calls == [[entry]]


def test_kernel_plan_blocks_on_any_unresolved_patch_root(tmp_path, monkeypatch):
    from hyperloom.orchestrator.framework.paths import WarmReplayRootResolution

    coord = _make_coord(tmp_path)
    entry = {
        "column": "fusion",
        "patch_path": str(tmp_path / "fusion.patch"),
    }
    coord.shared_state.warm_kernel_kb_plan = [entry]
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_warm_replay_kernel_root",
        lambda *, patch_entries=None, precomputed_allowlist=None: WarmReplayRootResolution(
            root="",
            source="",
            reason="ambiguous_root",
            allowlist=("/aiter-a", "/aiter-b"),
        ),
    )

    outcome = coord.phase_prelude._warm_replay_kernel_root_block_reason(coord.shared_state)

    assert outcome is not None
    assert outcome["reason"] == "ambiguous_root"


def test_multi_file_kernel_snapshot_restores_modify_and_create(tmp_path):
    coord = _make_coord(tmp_path)
    existing = tmp_path / "framework/existing.py"
    created = tmp_path / "framework/created.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("original\n", encoding="utf-8")
    snapshots = [
        coord.phase_prelude._snapshot_warm_kernel_target(str(existing), 0),
        coord.phase_prelude._snapshot_warm_kernel_target(str(created), 1),
    ]
    existing.write_text("patched\n", encoding="utf-8")
    created.write_text("new\n", encoding="utf-8")

    result = coord.phase_prelude._restore_warm_kernel_snapshots(snapshots)

    assert result == {"ok": True, "errors": []}
    assert existing.read_text(encoding="utf-8") == "original\n"
    assert not created.exists()


@pytest.mark.asyncio
async def test_failed_replay_clears_in_flight_via_full_routing(tmp_path):
    """A failed replay must leave ``warm_replay_in_flight`` False so PRELUDE can exit."""
    from hyperloom.orchestrator.phases.machine_state import (
        warm_replay_in_flight,
    )

    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    assert warm_replay_in_flight(coord.shared_state) is True

    failed = {"status": "failed", "error_class": "timeout", "error": "killed"}
    task = _StubTask(kind="replay_warm_recipe")
    if coord._is_promotable_result(task.kind, failed):
        await coord._promote_to_shared_state(task.kind, failed, task=task)
    else:
        await coord._handle_unpromotable_result(task, failed)

    assert warm_replay_in_flight(coord.shared_state) is False, (
        "failed replay left warm_replay_in_flight True → PRELUDE would never exit"
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


@pytest.mark.asyncio
async def test_dispatch_failure_rolls_back_preapplied_warm_kernel(tmp_path):
    """A dispatch-time policy failure must restore the live framework target."""
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
    from hyperloom.orchestrator.phases.machine_state import (
        warm_replay_in_flight,
    )

    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    dispatcher = DispatcherCollaborator(coord)

    class _Bus:
        async def append_and_seq(self, _message):
            return 1

    dispatcher.bus = _Bus()
    coord.shared_state.baseline_tput = 600.0
    target = tmp_path / "site-packages/vllm/prefix_prefill.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("patched\n", encoding="utf-8")
    backup = tmp_path / "warm_kernel_snapshots/0000.bin"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("original\n", encoding="utf-8")
    snapshots = [
        {
            "target": str(target),
            "existed": True,
            "backup": str(backup),
            "mode": 0o644,
        }
    ]
    applied = [{"status": "ok", "target_file": str(target)}]
    coord.shared_state.warm_replay_pending = {
        "status": "in_flight",
        "task_id": "task-warm-1",
        "kernel_apply_results": applied,
        "kernel_snapshots": snapshots,
    }
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-1",
    }
    task = _StubTask(
        task_id="task-warm-1",
        params={
            "warm_kernel_apply_results": applied,
            "warm_kernel_snapshots": snapshots,
        },
    )

    await dispatcher._reap_dispatched_task(
        task,
        SubAgentResult(
            task_id=task.task_id,
            state="failed",
            result={},
            error=("replay_warm_recipe target_file='/usr/local/vllm.py' escapes session_dir"),
        ),
        None,
    )

    assert target.read_text(encoding="utf-8") == "original\n"
    assert coord.shared_state.warm_replay_pending == {}
    assert warm_replay_in_flight(coord.shared_state) is False
    outcome = coord.shared_state.warm_replay_outcome
    assert outcome["status"] == "failed"
    assert outcome["error_class"] == "dispatch_failed"
    assert "escapes session_dir" in outcome["reason"]


@pytest.mark.asyncio
async def test_prelude_initial_analysis_deferred_while_warm_replay_in_flight(
    tmp_path,
):
    """Initial roofline must not enqueue while KB replay is still running."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.shared_state.warm_replay_outcome["status"] == "in_flight"
    assert len(coord.tasks.calls) == 1

    await coord._maybe_enqueue_prelude_initial_analysis_after_baseline(
        baseline_tput=600.0,
    )
    assert len(coord.tasks.calls) == 1
    assert not coord.shared_state.auto_roofline_pending_task_id


@pytest.mark.asyncio
async def test_prelude_initial_analysis_enqueued_after_warm_replay_finishes(
    tmp_path,
):
    """Deferred initial roofline enqueues once warm-replay outcome settles."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    coord._promote_warm_replay(
        {"status": "failed", "error_class": "crash", "error": "killed"},
        task=_StubTask(),
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"

    await coord._maybe_enqueue_prelude_initial_analysis_after_baseline()
    assert len(coord.tasks.calls) == 2
    assert coord.tasks.calls[1]["idempotency_key"] == ("internal-analysis-prelude_initial")
    assert coord.shared_state.auto_roofline_pending_task_id


@pytest.mark.asyncio
async def test_prelude_initial_analysis_dropped_when_it_would_cost_the_optimization_phases(
    tmp_path,
):
    """A roofline is worth an hour only if the session can still use what it finds.

    The Qwen3.5-397B shape: 51 minutes of baseline, then an 81-minute TraceLens
    arm that left FRAMEWORK_AGENT 46 minutes against its 108-minute threshold.
    """
    coord = _make_coord(tmp_path)
    state = coord.shared_state
    state.baseline_tput = 600.0
    state.max_minutes = 180
    state.baseline_runtime_sec = 2705.7
    state.phase_elapsed_totals = {"PRELUDE": 3090.0}
    state.phase_history = [{"to_phase": "PRELUDE", "evidence": {}}]
    state.session_budget_usable_sec = lambda: 7700.0

    await coord._maybe_enqueue_prelude_initial_analysis_after_baseline()

    assert coord.tasks.calls == []
    assert not coord.shared_state.auto_roofline_pending_task_id
    dropped = state.phase_history[-1]["evidence"]["budget_dropped_arms"]
    assert dropped[0]["arm"] == "initial_analysis"
    assert dropped[0]["expected_cost_sec"] == pytest.approx(2705.7)


@pytest.mark.asyncio
async def test_prelude_initial_analysis_runs_when_the_budget_covers_it(tmp_path):
    """Same wiring, ordinary session: the arm is not dropped just because the guard exists."""
    coord = _make_coord(tmp_path)
    state = coord.shared_state
    state.baseline_tput = 600.0
    state.max_minutes = 180
    state.baseline_runtime_sec = 300.0
    state.phase_elapsed_totals = {"PRELUDE": 320.0}
    state.phase_history = [{"to_phase": "PRELUDE", "evidence": {}}]
    state.session_budget_usable_sec = lambda: 10_300.0

    await coord._maybe_enqueue_prelude_initial_analysis_after_baseline()

    assert len(coord.tasks.calls) == 1
    assert coord.shared_state.auto_roofline_pending_task_id


def test_prelude_bootstrap_runs_on_positive_baseline(tmp_path):
    coord = _make_coord(tmp_path)
    assert coord._should_run_prelude_bootstrap(600.0) is True


def test_prelude_bootstrap_skipped_without_throughput(tmp_path):
    coord = _make_coord(tmp_path)
    assert coord._should_run_prelude_bootstrap(0.0) is False
    assert coord._should_run_prelude_bootstrap(None) is False


def test_prelude_bootstrap_skipped_when_roofline_pending(tmp_path):
    coord = _make_coord(tmp_path)
    coord.shared_state.auto_roofline_pending_task_id = "task-roofline"
    assert coord._should_run_prelude_bootstrap(600.0) is False


def test_prelude_bootstrap_skipped_when_stop_pending(tmp_path):
    """A baseline that halted the run (e.g. baseline_accuracy_failed) must not
    enqueue/dispatch any post-baseline bootstrap work before the halt fires."""
    coord = _make_coord(tmp_path)
    coord.shared_state.stop_reason = "baseline_accuracy_failed"
    assert coord._should_run_prelude_bootstrap(600.0) is False


def test_inject_warm_recipe_history_skips_when_no_recipe(tmp_path):
    """No warm_start_recipe → nothing to inject; flag still flipped to prevent retries."""
    coord = _make_coord(tmp_path, warm_start_recipe={})
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    assert coord.shared_state.warm_history_injected is True
    assert coord.shared_state.explore_search.get("rejected", []) == []


def test_inject_warm_recipe_history_adds_what_failed_rows(tmp_path):
    """Every what_failed row carries a canonical fingerprint into the
    rejected ledger, with ``source=warm_start_recipe``."""
    recipe = _warm_recipe_t1(
        what_failed=[
            {
                "name": "fp4_kv_cache",
                "extra_server_args": "--kv-cache-dtype fp4",
                "extra_envs": {},
                "gain_pct": -8.0,
                "error_class": "regress",
            },
            {
                "name": "tilelang_mla",
                "extra_server_args": "",
                "extra_envs": {"SGLANG_HACK_FLASHMLA_BACKEND": "tilelang"},
                "gain_pct": None,
                "error_class": "crash",
            },
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 2
    rejected = coord.shared_state.explore_search["rejected"]
    assert len(rejected) == 2
    for row in rejected:
        assert isinstance(row.get("fingerprint"), str) and len(row["fingerprint"]) == 16
        assert row["reason"] == "warm_recipe_what_failed"
        assert row["source"] == "warm_start_recipe"
        assert row["source_tier"] == "exact"
    assert any(r["error_class"] == "regress" for r in rejected)
    assert any(r["error_class"] == "crash" for r in rejected)
    assert coord.shared_state.warm_history_injected is True


def test_inject_warm_recipe_history_v2_arbor_top_level(tmp_path):
    """Regression: the injector must read v2 ``what_failed`` at the TOP LEVEL, else negative-history injection no-ops."""
    recipe = {
        "tier": "exact",
        "confidence": 1.0,
        "recipe": {
            "canonical_id": "inference:deepseek-r1:mi300x:sglang:0.4.5:fp8",
            "model": "deepseek-r1",
            "what_failed": [
                {
                    "name": "fp4_kv_cache",
                    "extra_server_args": "--kv-cache-dtype fp4",
                    "extra_envs": {},
                    "gain_pct": -8.0,
                    "error_class": "regress",
                },
            ],
        },
    }
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 1, "v2 arbor top-level what_failed not read"
    rejected = coord.shared_state.explore_search["rejected"]
    assert len(rejected) == 1
    assert rejected[0]["source"] == "warm_start_recipe"


def test_inject_warm_recipe_history_is_idempotent(tmp_path):
    """Resume safety: re-invoking the injector after the one-shot flag is set must not re-append rows."""
    recipe = _warm_recipe_t1(
        what_failed=[
            {
                "name": "x",
                "extra_server_args": "--bad-flag",
                "extra_envs": {},
                "gain_pct": -10.0,
            }
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    coord._inject_warm_recipe_history_into_ledger()
    first = list(coord.shared_state.explore_search["rejected"])
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    assert coord.shared_state.explore_search["rejected"] == first


def test_inject_warm_recipe_history_dedupes_with_existing_ledger(tmp_path):
    """A ledger row with the same fingerprint is not duplicated."""
    from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
        canonical_fingerprint,
    )

    failed_args = "--kv-cache-dtype fp4"
    pre_existing_fp = canonical_fingerprint(failed_args, {})
    recipe = _warm_recipe_t1(
        what_failed=[
            {
                "name": "fp4",
                "extra_server_args": failed_args,
                "extra_envs": {},
                "gain_pct": -8.0,
            }
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {
        "rejected": [
            {
                "name": "explore_round_1_X",
                "fingerprint": pre_existing_fp,
                "reason": "stack_unstable",
            }
        ],
    }
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 0
    assert len(coord.shared_state.explore_search["rejected"]) == 1
    assert coord.shared_state.explore_search["rejected"][0]["reason"] == "stack_unstable"


def test_inject_warm_recipe_history_skips_empty_rows(tmp_path):
    """A what_failed row with neither args nor envs is unreplayable; skip silently."""
    recipe = _warm_recipe_t1(
        what_failed=[
            {"name": "bogus", "extra_server_args": "", "extra_envs": {}},
            {"name": "real", "extra_server_args": "--actual-flag", "extra_envs": {}},
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    coord.shared_state.explore_search = {}
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 1
    assert coord.shared_state.explore_search["rejected"][0]["name"] == "real"


@pytest.mark.asyncio
async def test_warm_replay_pulls_expected_gain_from_sessions_max(tmp_path):
    """The historical gain anchor is the MAX of ``attrs.sessions[].gain_pct``."""
    recipe = _warm_recipe_t1(
        sessions=[
            {"session_id": "older", "gain_pct": 12.0, "stack_len": 1},
            {"session_id": "best", "gain_pct": 28.0, "stack_len": 4},
            {"session_id": "newer", "gain_pct": 20.0, "stack_len": 2},
        ],
    )
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 28.0


@pytest.mark.asyncio
async def test_warm_replay_zero_expected_when_no_sessions(tmp_path):
    """Recipes without sessions[] → expected_gain falls to 0 (``_promote`` accepts any positive measurement)."""
    recipe = _warm_recipe_t1(sessions=[])
    coord = _make_coord(tmp_path, warm_start_recipe=recipe)
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 0.0


@pytest.mark.asyncio
async def test_warm_replay_falls_back_to_flat_gain_pct_for_arbor_seed(tmp_path):
    """Arbor seeds with a flat ``gain_pct`` attr (no sessions[]) are still read as the expected anchor."""
    coord = _make_coord(tmp_path)
    coord.shared_state.warm_start_recipe = {
        "tier": "relative",
        "confidence": 0.75,
        "recipe": {
            "attrs": {
                "best_config": {"extra_server_args": "--foo", "extra_envs": {}},
                "gain_pct": 18.0,  # flat, no sessions[]
            },
        },
    }
    await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)
    assert coord.tasks.calls[0]["params"]["warm_expected_gain_pct"] == 18.0


def test_promote_warm_replay_cumulative_gain_uses_tput_ratio(tmp_path):
    """Flag off: cumulative gain after warm-replay is (tput / baseline_tput - 1) × 100."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
        }
    )
    # baseline 600, measured 738 -> gain = 23% via tput ratio.
    result = {"status": "succeeded", "output_throughput": 738.0}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.cumulative_gain_validated == 23.0


def test_promote_warm_replay_zero_baseline_tput_is_failure(tmp_path):
    """Defense in depth: an invalid baseline_tput must not divide-by-zero — tag as failed."""
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 0.0
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=_StubTask())
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"
    assert "invalid_tput" in coord.shared_state.warm_replay_outcome["reason"]


@pytest.mark.asyncio
async def test_combined_replay_prepares_kernel_without_separate_validation(
    tmp_path,
):
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(
            extra_server_args="--recipe",
            extra_envs={"RECIPE": "1"},
        ),
    )
    prepared_calls = 0

    async def _prepare():
        nonlocal prepared_calls
        prepared_calls += 1
        return {
            "status": "prepared",
            "pending": [{"column": "gemm", "decision": "PENDING"}],
            "applied": [{"status": "ok", "manifest_path": "/tmp/m"}],
            "extra_envs": {"KERNEL": "1"},
            "extra_server_args": "--kernel",
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert prepared_calls == 1
    assert len(coord.tasks.calls) == 1
    assert task.params["extra_envs"] == {"RECIPE": "1", "KERNEL": "1"}
    assert "--recipe" in task.params["extra_server_args"]
    assert "--kernel" in task.params["extra_server_args"]
    assert len(task.params["warm_kernel_plan"]) == 1


@pytest.mark.asyncio
async def test_dirty_kernel_preparation_stops_recipe_enqueue(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_pending = {"kernel_snapshots": [{"target": "/tmp/kernel.py"}]}

    async def _prepare():
        return {
            "status": "rollback_failed",
            "dirty": True,
            "reason": "restore failed",
            "rollback": {"ok": False, "errors": ["restore failed"]},
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert coord.tasks.calls == []
    assert coord.shared_state.warm_replay_outcome["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_pending["status"] == "rollback_failed"


@pytest.mark.asyncio
async def test_enqueue_failure_rolls_back_prepared_kernel(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    applied = [{"manifest_path": "/tmp/m"}]
    snapshots = [{"target": "/tmp/kernel.py"}]
    coord.shared_state.warm_replay_pending = {
        "kernel_apply_results": applied,
        "kernel_snapshots": snapshots,
    }

    async def _prepare():
        return {
            "status": "prepared",
            "pending": [{"column": "rewrite"}],
            "applied": applied,
            "snapshots": snapshots,
        }

    async def _raise(**_kwargs):
        raise RuntimeError("registry unavailable")

    rollbacks: list[tuple[list[dict], list[dict]]] = []
    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]
    coord.phase_prelude._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda got_applied, got_snapshots=None: (
            rollbacks.append((got_applied, got_snapshots or [])) or {"ok": True, "errors": []}
        )
    )
    coord.tasks.create_or_return_existing = _raise  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="registry unavailable"):
        await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert rollbacks == [(applied, snapshots)]
    assert coord.shared_state.warm_replay_pending == {}
    assert coord.shared_state.warm_replay_outcome["status"] == "enqueue_failed"


@pytest.mark.asyncio
async def test_enqueue_failure_retains_pending_when_kernel_restore_fails(
    tmp_path,
):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_pending = {
        "kernel_apply_results": [{"manifest_path": "/tmp/m"}],
        "kernel_snapshots": [{"target": "/tmp/kernel.py"}],
    }

    async def _prepare():
        return {
            "status": "prepared",
            "pending": [{"column": "rewrite"}],
            "applied": [{"manifest_path": "/tmp/m"}],
            "snapshots": [{"target": "/tmp/kernel.py"}],
        }

    async def _raise(**_kwargs):
        raise RuntimeError("registry unavailable")

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]
    coord.phase_prelude._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda *_args: {"ok": False, "errors": ["restore failed"]}
    )
    coord.tasks.create_or_return_existing = _raise  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="registry unavailable"):
        await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert coord.shared_state.warm_replay_pending["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_outcome["status"] == "rollback_failed"


def test_combined_replay_revert_rolls_back_recipe_and_kernel(tmp_path, monkeypatch):
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(),
    )
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 5.0}
    recipe_rollbacks: list[tuple[str, str]] = []
    kernel_rollbacks: list[list[dict]] = []

    import hyperloom.orchestrator.actions.executors.baseline as baseline_module

    monkeypatch.setattr(
        baseline_module,
        "_revert_patches",
        lambda target, sha, manifest=None: recipe_rollbacks.append((target, sha)) or {"ok": True, "errors": []},
    )
    coord.phase_prelude._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda applied, snapshots=None: kernel_rollbacks.append(applied) or {"ok": True, "errors": []}
    )
    coord.shared_state.warm_replay_pending = {
        "recipe_patch_target": "/repo",
        "recipe_patch_pre_sha": "abc",
        "recipe_patch_snapshot_manifest": {"manifest_path": "/repo.json"},
    }
    task = _StubTask(
        task_id="combined",
        params={
            "baseline_tput_anchor": 600.0,
            "extra_server_args": "--recipe --kernel",
            "extra_envs": {"RECIPE": "1", "KERNEL": "1"},
            "warm_kernel_plan": [{"column": "rewrite"}],
            "warm_kernel_apply_results": [{"manifest_path": "/tmp/m"}],
        },
    )

    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 500.0,
        },
        task=task,
    )

    assert recipe_rollbacks == [("/repo", "abc")]
    assert kernel_rollbacks == [[{"manifest_path": "/tmp/m"}]]
    assert coord.shared_state.warm_replay_outcome["kernel"]["status"] == "reverted"


@pytest.mark.asyncio
async def test_kernel_only_replay_enqueues_without_recipe(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe={})

    async def _prepare():
        return {
            "status": "prepared",
            "pending": [{"column": "gemm"}],
            "applied": [],
            "extra_envs": {"KERNEL_ONLY": "1"},
            "extra_server_args": "",
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is not None
    assert task.params["recipe_extra_envs"] == {}
    assert task.params["extra_envs"] == {"KERNEL_ONLY": "1"}
    assert task.params["combined_current_contract"] is True


@pytest.mark.asyncio
async def test_no_recipe_after_loaded_kernel_clears_stale_pending(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe={})
    coord.shared_state.warm_replay_pending = {"status": "preparing_kernel"}

    async def _prepare():
        return {
            "status": "loaded",
            "pending": [],
            "applied": [],
            "snapshots": [],
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]

    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is None
    assert coord.shared_state.warm_replay_pending == {}


@pytest.mark.asyncio
async def test_combined_threshold_uses_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_WARM_KERNEL_KEEP_PCT", "2.5")
    coord = _make_coord(tmp_path, warm_start_recipe={})

    async def _prepare():
        return {
            "status": "prepared",
            "pending": [{"column": "gemm"}],
            "applied": [],
            "extra_envs": {"KERNEL_ONLY": "1"},
            "extra_server_args": "",
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task.params["combined_keep_threshold_pct"] == 2.5


@pytest.mark.asyncio
async def test_low_confidence_recipe_does_not_suppress_kernel(tmp_path):
    coord = _make_coord(
        tmp_path,
        warm_start_recipe=_warm_recipe_t1(
            confidence=0.1,
            extra_server_args="--untrusted-recipe",
            extra_envs={"UNTRUSTED": "1"},
        ),
    )

    async def _prepare():
        return {
            "status": "prepared",
            "pending": [{"column": "fusion"}],
            "applied": [{"manifest_path": "/tmp/m"}],
            "extra_envs": {"KERNEL": "1"},
            "extra_server_args": "--kernel",
        }

    coord.phase_prelude._prepare_warm_kernel_kb = _prepare  # type: ignore[method-assign]
    task = await coord._maybe_enqueue_warm_replay(baseline_tput=600.0)

    assert task is not None
    assert task.params["recipe_extra_server_args"] == ""
    assert task.params["recipe_extra_envs"] == {}
    assert task.params["extra_server_args"] == "--kernel"
    assert coord.shared_state.warm_replay_outcome["recipe_suppressed"] is True
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 612.0},
        task=task,
    )
    assert coord.shared_state.optimization_stack[-1]["recipe_delta"] == {
        "extra_server_args": "",
        "extra_envs": {},
        "remove_args": [],
        "unset_envs": [],
        "args_mode": "replace",
    }
    assert coord.shared_state.current_best["extra_server_args"] == "--kernel"


def _git_repo_for_required_patch(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "canonical-ix"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target = repo / "vllm" / "fp8.py"
    target.parent.mkdir()
    target.write_text("# fp8 module\noriginal = True\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    patch = (
        "diff --git a/vllm/fp8.py b/vllm/fp8.py\n"
        "--- a/vllm/fp8.py\n"
        "+++ b/vllm/fp8.py\n"
        "@@ -1,2 +1,3 @@\n"
        " # fp8 module\n"
        " original = True\n"
        "+persisted = True\n"
    )
    return repo, patch


def test_combined_keep_retains_validated_framework_root_without_reapply(
    tmp_path,
):
    checkout, patch_content = _git_repo_for_required_patch(tmp_path)
    subprocess.run(
        ["git", "apply", "-"],
        cwd=checkout,
        input=patch_content.encode(),
        check=True,
        capture_output=True,
    )
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "required_patch_timeline": True,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "patches": [
                {
                    "patch_file": "explore/overlays/000000/00-p.patch",
                    "patch_content": patch_content,
                }
            ],
            "extra_server_args": "--recipe --kernel",
            "extra_envs": {"VLLM_RECIPE": "1", "KERNEL_ONLY": "1"},
            "recipe_extra_server_args": "--recipe",
            "recipe_extra_envs": {"VLLM_RECIPE": "1"},
            "warm_kernel_plan": [],
            "warm_kernel_apply_results": [],
        }
    )

    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 612.0,
            "warm_patch_target": str(checkout),
            "warm_patch_pre_sha": "base-sha",
            "warm_patch_snapshot_manifest": {
                "repo_path": str(checkout),
                "manifest_path": str(tmp_path / "manifest.json"),
            },
            "warm_patches_applied": [
                {
                    "patch_file": "explore/overlays/000000/00-p.patch",
                    "status": "applied",
                }
            ],
        },
        task=task,
    )

    assert "persisted = True" in (checkout / "vllm" / "fp8.py").read_text()
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"
    assert coord.shared_state.warm_replay_outcome["active_framework_root"] == str(checkout.resolve())
    assert coord.shared_state.optimization_stack[-1]["framework_source_root"] == str(checkout.resolve())
    entry = coord.shared_state.optimization_stack[-1]
    assert entry["recipe_delta"] == {
        "extra_server_args": "--recipe",
        "extra_envs": {"VLLM_RECIPE": "1"},
        "remove_args": [],
        "unset_envs": [],
        "args_mode": "replace",
    }
    assert entry["candidate_extra_server_args"] == "--recipe --kernel"
    assert entry["candidate_extra_envs"] == {
        "VLLM_RECIPE": "1",
        "KERNEL_ONLY": "1",
    }
    assert coord.shared_state.current_best["extra_envs"]["KERNEL_ONLY"] == "1"
    assert coord.shared_state.warm_replay_pending == {}


def test_checkout_promotion_failure_rejects_keep_and_rolls_kernel(tmp_path, monkeypatch):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    kernel_rollbacks: list[list[dict]] = []
    coord.phase_prelude._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda applied, snapshots=None: kernel_rollbacks.append(applied) or {"ok": True, "errors": []}
    )
    import hyperloom.orchestrator.actions.executors.baseline as baseline_module

    monkeypatch.setattr(
        baseline_module,
        "_revert_patches",
        lambda *_args: {"ok": True, "errors": []},
    )
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "required_patch_timeline": True,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "patches": [{"patch_file": "p.patch", "patch_content": "diff"}],
            "extra_server_args": "--recipe",
            "extra_envs": {},
            "warm_kernel_plan": [{"column": "rewrite"}],
            "warm_kernel_apply_results": [{"manifest_path": "/tmp/m"}],
        }
    )
    mirror = tmp_path / "mirror"
    other = tmp_path / "other"
    mirror.mkdir()
    other.mkdir()

    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 612.0,
            "warm_patch_target": str(mirror),
            "warm_patch_pre_sha": "abc",
            "warm_patch_snapshot_manifest": {
                "repo_path": str(other),
                "manifest_path": str(tmp_path / "manifest.json"),
            },
        },
        task=task,
    )

    assert coord.shared_state.warm_replay_outcome["status"] == "promotion_failed"
    assert coord.shared_state.optimization_stack == []
    assert kernel_rollbacks == [[{"manifest_path": "/tmp/m"}]]


def test_checkout_promotion_failure_retains_pending_when_rollback_fails(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    coord.shared_state.warm_replay_pending = {"task_id": "warm"}
    coord.phase_prelude._resolve_promoted_recipe_checkout = (  # type: ignore[method-assign]
        lambda *_args: (False, {"failure": "persist failed"})
    )
    coord.phase_prelude._rollback_combined_warm = (  # type: ignore[method-assign]
        lambda *_args: {"ok": False, "errors": ["restore failed"]}
    )
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "required_patch_timeline": True,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "patches": [{"patch_file": "p.patch", "patch_content": "diff"}],
            "extra_server_args": "--recipe",
        }
    )

    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 612.0},
        task=task,
    )

    assert coord.shared_state.warm_replay_outcome["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_pending == {"task_id": "warm"}


def test_current_contract_threshold_preserves_local_legacy_positive_gain(tmp_path):
    current = _make_coord(
        tmp_path / "current",
        warm_start_recipe=_warm_recipe_t1(),
    )
    current.shared_state.baseline_tput = 600.0
    current.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    current._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 603.0},
        task=_StubTask(
            params={
                "baseline_tput_anchor": 600.0,
                "combined_current_contract": True,
                "combined_keep_threshold_pct": 1.0,
                "extra_server_args": "--current",
            }
        ),
    )
    assert current.shared_state.warm_replay_outcome["status"] == "drift"

    legacy = _make_coord(tmp_path / "legacy", warm_start_recipe=_warm_recipe_t1())
    legacy.shared_state.baseline_tput = 600.0
    legacy.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    legacy._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 603.0},
        task=_StubTask(
            params={
                "baseline_tput_anchor": 600.0,
                "extra_server_args": "--legacy",
            }
        ),
    )
    assert legacy.shared_state.warm_replay_outcome["status"] == "reproduced"


def test_zero_and_nonfinite_combined_thresholds(tmp_path, monkeypatch):
    zero = _make_coord(tmp_path / "zero", warm_start_recipe=_warm_recipe_t1())
    zero.shared_state.baseline_tput = 600.0
    zero.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    zero._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 600.0},
        task=_StubTask(
            params={
                "baseline_tput_anchor": 600.0,
                "combined_current_contract": True,
                "combined_keep_threshold_pct": 0.0,
                "extra_server_args": "--zero",
            }
        ),
    )
    assert zero.shared_state.warm_replay_outcome["status"] == "reproduced"
    assert zero.shared_state.warm_replay_outcome["keep_threshold_pct"] == 0.0

    monkeypatch.setenv("HYPERLOOM_WARM_KERNEL_KEEP_PCT", "nan")
    nonfinite = _make_coord(
        tmp_path / "nan",
        warm_start_recipe=_warm_recipe_t1(),
    )
    nonfinite.shared_state.baseline_tput = 600.0
    nonfinite.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    nonfinite._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 603.0},
        task=_StubTask(
            params={
                "baseline_tput_anchor": 600.0,
                "combined_current_contract": True,
                "combined_keep_threshold_pct": float("inf"),
                "extra_server_args": "--nonfinite",
            }
        ),
    )
    assert nonfinite.shared_state.warm_replay_outcome["status"] == "drift"
    assert nonfinite.shared_state.warm_replay_outcome["keep_threshold_pct"] == 1.0


def test_already_present_required_patch_is_not_republished(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "extra_server_args": "--recipe",
            "required_patch_timeline": True,
            "patches": [],
        }
    )
    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 612.0,
            "warm_patches_applied": [{"patch_file": "old.patch", "status": "already_present"}],
        },
        task=task,
    )

    assert "replayed_patch_refs" not in coord.shared_state.warm_replay_outcome
    assert "replayed_patch_refs" not in coord.shared_state.optimization_stack[-1]


def test_dirty_worktree_required_patch_is_republished(tmp_path):
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_tput = 600.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 0.0}
    task = _StubTask(
        params={
            "baseline_tput_anchor": 600.0,
            "combined_current_contract": True,
            "combined_keep_threshold_pct": 1.0,
            "extra_server_args": "--recipe",
            "required_patch_timeline": True,
            "patches": [],
        }
    )
    coord._promote_warm_replay(
        {
            "status": "succeeded",
            "output_throughput": 612.0,
            "warm_patches_applied": [
                {
                    "patch_file": "old.patch",
                    "status": "present_in_dirty_worktree",
                }
            ],
        },
        task=task,
    )

    assert coord.shared_state.warm_replay_outcome["replayed_patch_refs"] == ["old.patch"]
    assert coord.shared_state.optimization_stack[-1]["replayed_patch_refs"] == ["old.patch"]


def _warm_replay_ledger(session_dir):
    """Assemble the recorded parts plus the baseline reading gains are measured against."""
    from hyperloom.inference_optimizer.breakdown.collectors import (
        collect_recorded_optimizations,
    )
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

    parts = assemble_parts(session_dir)
    operations = list(parts.get("operations") or [])
    measurements = list(parts.get("measurements") or [])
    operations.append({"operation_id": "op-base", "kind": "baseline", "measurement_refs": ["m-base"]})
    measurements.append({"measurement_id": "m-base", "name": "throughput", "value": 600.0})
    return collect_recorded_optimizations(
        "s1",
        operations,
        measurements,
        list(parts.get("adoptions") or []),
        list(parts.get("artifacts") or []),
        [],
        [],
        [],
    )


def test_reproduced_replay_reaches_the_canonical_ledger(tmp_path):
    """A promoted warm replay must be an adopted step in the recorded ledger.

    The replay executor settles on ``succeeded`` whether or not the recipe
    reproduced, and the keep decision is reached only by ``_promote_warm_replay``.
    Mirroring the action before that ruling published every replay as
    discarded, so a reproduced one was pushed onto the stack and moved
    ``cumulative_gain_validated`` while the canonical streams held no adoption
    for it -- ``optimizations.entries`` came back empty on a session that had
    measurably gained. The verdict-aware mirror closes that gap.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "baseline_tput_anchor": 600.0,
        }
    )
    result = {"status": "succeeded", "output_throughput": 660.0}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"

    WritebackCollaborator(coord)._mirror_warm_replay_verdict(result, task)

    ledger = _warm_replay_ledger(tmp_path)
    entry = ledger["entries"][0]
    assert entry["source"] == "warm_replay"
    assert entry["optimization_kind"] == "replay_warm_recipe"
    # The gain chains from the recorded session baseline, not the enqueue anchor.
    assert entry["gain_pct"] == pytest.approx(10.0, abs=0.01)
    assert ledger["validation"]["ledger_total_gain_pct"] == pytest.approx(10.0, abs=0.01)
    assert ledger["validation"]["keep_count"] == 1
    # The executor's real status survives; the keep rides on the decision.
    assert ledger["attempts"][0]["status"] == "succeeded"


def test_drifted_replay_stays_out_of_the_canonical_ledger(tmp_path):
    """A replay that missed the bar must not be credited any gain.

    The fix for the discarded-reproduced replay must not reach the other way
    and let a drift claim a keep it never earned. The rejected attempt must
    still carry the evidence that explains the rejection.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "baseline_tput_anchor": 600.0,
        }
    )
    result = {"status": "succeeded", "output_throughput": 600.0}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.warm_replay_outcome["status"] == "drift"

    WritebackCollaborator(coord)._mirror_warm_replay_verdict(result, task)

    ledger = _warm_replay_ledger(tmp_path)
    assert ledger["entries"] == []
    assert ledger["validation"]["ledger_total_gain_pct"] == 0.0
    attempt = ledger["attempts"][0]
    assert attempt["adopted"] is False
    # The rejected replay carries why it was dropped: the measured gain, the bar
    # it missed, and the reason -- not a blank row.
    assert attempt["local_gain_pct"] == pytest.approx(0.0, abs=0.01)
    assert attempt["keep_threshold_pct"] == pytest.approx(0.0, abs=0.01)
    assert "below keep threshold" in attempt["decision_reason"]
    # Its status is normalized outside the executor-adoption verdict set.
    assert attempt["status"] not in ("kept", "kept_inert", "promoted", "adopted")


def test_replay_admitted_without_a_score_is_unscored_not_validated(tmp_path):
    """An eval that ran but returned no score is adopted, not accuracy-validated.

    ``_warm_replay_accuracy_ok`` admits a replay whose eval ran but produced no
    usable number (``eval_ran`` true, ``replay_accuracy`` None). ``eval_ran``
    alone must not be read as a passed gate: the step is adopted on the keep
    verdict alone and has to record ``keep_verdict_unscored`` rather than dress
    up an absent score as ``accuracy_pass``.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    # Reproduced, but the accuracy eval ran without yielding a score.
    coord.shared_state.warm_replay_outcome = {
        "status": "reproduced",
        "actual_gain_pct": 10.0,
        "throughput_after": 660.0,
        "keep_threshold_pct": 0.0,
        "eval_ran": True,
        "replay_accuracy": None,
    }
    task = _StubTask(params={"extra_server_args": "--attention-backend AITER"})
    result = {"status": "succeeded", "output_throughput": 660.0}

    WritebackCollaborator(coord)._mirror_warm_replay_verdict(result, task)

    ledger = _warm_replay_ledger(tmp_path)
    attempt = ledger["attempts"][0]
    assert attempt["adopted"] is True
    assert attempt["validation_basis"] == "keep_verdict_unscored"
    assert ledger["entries"][0]["gain_pct"] == pytest.approx(10.0, abs=0.01)
    assert ledger["validation"]["unscored_keep_count"] == 1


def test_replay_with_a_passing_score_is_accuracy_validated(tmp_path):
    """A scored, passing replay records ``accuracy_pass`` -- the counterpart.

    The unscored path must not swallow the case where a real score exists: a
    numeric ``replay_accuracy`` that reached ``reproduced`` cleared the gate and
    has to read as validated.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "reproduced",
        "actual_gain_pct": 10.0,
        "throughput_after": 660.0,
        "keep_threshold_pct": 0.0,
        "eval_ran": True,
        "replay_accuracy": 0.87,
    }
    task = _StubTask(params={"extra_server_args": "--attention-backend AITER"})
    result = {"status": "succeeded", "output_throughput": 660.0}

    WritebackCollaborator(coord)._mirror_warm_replay_verdict(result, task)

    ledger = _warm_replay_ledger(tmp_path)
    attempt = ledger["attempts"][0]
    assert attempt["adopted"] is True
    assert attempt["validation_basis"] == "accuracy_pass"
    assert ledger["validation"]["unscored_keep_count"] == 0


def test_scored_replay_drives_accuracy_pass_through_the_real_promote_path(tmp_path):
    """The accuracy provenance must ride on what the run actually stamped.

    The unscored/scored pair above hand-build ``warm_replay_outcome``; on their
    own they would stay green even if ``_promote_warm_replay`` stopped stamping
    ``replay_accuracy`` and every replay silently degraded to unscored. This
    drives the real promote path so the ``replay_accuracy`` (not ``eval_ran``)
    keying is anchored to a genuine producer.
    """
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    task = _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "baseline_tput_anchor": 600.0,
        }
    )
    # A numeric accuracy makes _warm_replay_accuracy_ok stamp replay_accuracy.
    result = {"status": "succeeded", "output_throughput": 660.0, "accuracy": 0.9}
    coord._promote_warm_replay(result, task=task)
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"
    assert coord.shared_state.warm_replay_outcome["replay_accuracy"] == 0.9

    WritebackCollaborator(coord)._mirror_warm_replay_verdict(result, task)

    ledger = _warm_replay_ledger(tmp_path)
    attempt = ledger["attempts"][0]
    assert attempt["adopted"] is True
    assert attempt["validation_basis"] == "accuracy_pass"
    assert ledger["validation"]["unscored_keep_count"] == 0


# ---- Phase 4: recipe carries apply root -----------------------------------
def test_agent_kb_read_patch_roots_with_recorded_root(tmp_path):
    from hyperloom.orchestrator.knowledge.agent_kb import _ConfigPatchAgentKB

    root = str(tmp_path / "sglang_root")
    ref = "framework/overlays/000000/00-p.patch"

    class _FakeKB(_ConfigPatchAgentKB):
        SECTION = "framework"

        def read(self):
            return {
                "patches": [ref],
                "patch_roots": {ref: root},
            }

    kb = _FakeKB(None)
    assert kb.read_patch_roots() == {ref: root}
    assert kb.read_patches() == [ref]


def test_read_patch_roots_returns_empty_for_legacy_records():
    from hyperloom.orchestrator.knowledge.agent_kb import _ConfigPatchAgentKB

    class _LegacyKB(_ConfigPatchAgentKB):
        SECTION = "framework"

        def read(self):
            return {"patches": ["framework/overlays/000000/00-x.patch"]}

    kb = _LegacyKB(None)
    assert kb.read_patch_roots() == {}
