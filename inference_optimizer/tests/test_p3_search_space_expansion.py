"""P3 — search-space expansion (T1+T2) tests.

Pins the new T1/T2 contracts so the search space stays "free":

* ``backends.discover_backend_flags`` and the new
  ``discover_vllm_backend_flags`` AST-parse the right files and yield
  CLI-style flag names.
* ``params.discover_param_flags`` AST-parses parameter-tuning attrs.
* ``BackendsExecutor`` augments DEFAULT_BACKENDS_GRID with discovered
  boolean flags AND surfaces a ``discovered_flags_update`` for the
  Coordinator to persist.
* ``BackendsExecutor`` honors ``params.synergy_groups`` /
  ``params.synergy_mode='auto'`` overrides AND skips combos already
  recorded in ``params.synergy_attempted``.
* ``ParamsExecutor`` surfaces a ``discovered_flags_update`` payload.
* ``SharedState`` mutators (``record_discovered_flags``,
  ``push_backend_winners_round``, ``mark_synergy_attempted``) round-trip
  via ``save`` / ``load_or_init``.
* ``Coordinator._promote_to_shared_state`` propagates the executor
  output into SharedState fields.
* ``build_orchestration_prompt`` includes the IR-26 idea-generation
  block and the per-action grid-injection hints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors import (
    DEFAULT_BACKENDS_GRID,
    DEFAULT_PARAMS_GRID,
    BackendsExecutor,
    ParamsExecutor,
    discover_backend_flags,
    discover_param_flags,
    discover_vllm_backend_flags,
)
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
)
from inference_optimizer.orchestrator.action_executors.backends import (
    _augment_grid_with_discovered_flags,
    _auto_synergy_combos,
    _build_synergy_combos_from_groups,
)
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    GRID_INJECTABLE_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


# ---------------------------------------------------------------------------
# AST discovery
# ---------------------------------------------------------------------------
def test_discover_vllm_backend_flags_parses_attrs(tmp_path):
    fake = tmp_path / "arg_utils.py"
    fake.write_text(
        "class EngineArgs:\n"
        "    def __init__(self):\n"
        "        self.kv_cache_dtype = 'auto'\n"
        "        self.compilation_config = None\n"
        "        self.enable_aiter_moe = False\n"
        "        self.unrelated = 0\n"
    )
    flags = discover_vllm_backend_flags(arg_utils_path=fake)
    assert "--kv-cache-dtype" in flags
    assert "--compilation-config" in flags
    assert "--enable-aiter-moe" in flags
    assert "--unrelated" not in flags


def test_discover_vllm_backend_flags_returns_empty_when_missing(tmp_path):
    assert discover_vllm_backend_flags(arg_utils_path=tmp_path / "no.py") == []


def test_discover_param_flags_parses_param_keywords(tmp_path):
    fake = tmp_path / "server_args.py"
    fake.write_text(
        "class ServerArgs:\n"
        "    def __init__(self):\n"
        "        self.max_num_seqs = 256\n"
        "        self.cuda_graph_max_bs = 64\n"
        "        self.mem_fraction_static = 0.85\n"
        "        self.chunked_prefill_size = 8192\n"
        "        self.attention_backend = 'aiter'\n"
        "        self.unrelated_field = 0\n"
    )
    flags = discover_param_flags(server_args_path=fake)
    assert "--max-num-seqs" in flags
    assert "--cuda-graph-max-bs" in flags
    assert "--mem-fraction-static" in flags
    assert "--chunked-prefill-size" in flags
    # `attention_backend` is a backend-style toggle, not a param
    assert "--attention-backend" not in flags
    assert "--unrelated-field" not in flags


def test_discover_param_flags_returns_empty_when_missing(tmp_path):
    assert discover_param_flags(server_args_path=tmp_path / "no.py") == []


# ---------------------------------------------------------------------------
# Grid augmentation
# ---------------------------------------------------------------------------
def test_augment_grid_skips_value_typed_and_existing_flags():
    base = [
        GridVariant("baseline_x", "--enable-overlap-scheduler"),
        GridVariant("env_y", extra_envs={"VLLM_USE_TRITON_FLASH_ATTN": "1"}),
    ]
    discovered = [
        "--enable-overlap-scheduler",       # already in base — skipped
        "--enable-mixed-chunk",              # bool, new — KEPT
        "--max-num-seqs",                    # value-typed — skipped
        "--attention-backend",               # backend keyword — skipped (NEEDS_VALUE)
        "--enable-aiter-fused-moe",          # bool, new — KEPT
    ]
    out = _augment_grid_with_discovered_flags(base, discovered, framework="sglang")
    new_names = {v.name for v in out} - {b.name for b in base}
    assert "sglang_discovered_enable_mixed_chunk" in new_names
    assert "sglang_discovered_enable_aiter_fused_moe" in new_names
    # Skipped flags must NOT appear
    assert not any(v.name.endswith("max_num_seqs") for v in out)
    assert not any(v.name.endswith("attention_backend") for v in out)
    # Original base preserved
    assert {b.name for b in base} <= {v.name for v in out}


def test_augment_grid_returns_base_when_discovery_empty():
    base = list(DEFAULT_BACKENDS_GRID[:2])
    out = _augment_grid_with_discovered_flags(base, [], framework="sglang")
    assert out == base


# ---------------------------------------------------------------------------
# Synergy combos (override + auto + dedup)
# ---------------------------------------------------------------------------
def test_build_synergy_combos_from_groups_skips_unknown_members():
    grid = [
        GridVariant("a", "--flag-a"),
        GridVariant("b", "--flag-b"),
        GridVariant("c", "--flag-c"),
    ]
    combos = _build_synergy_combos_from_groups(
        grid, [["a", "b"], ["a", "missing"]],
        require_one_winner=None,
    )
    names = [c.name for c in combos]
    assert "combo_a+b" in names
    # Missing member ⇒ group dropped
    assert not any("missing" in n for n in names)


def test_auto_synergy_combos_pairs_winners_under_cap():
    grid = [GridVariant(f"v{i}", f"--flag-{i}") for i in range(5)]
    combos = _auto_synergy_combos(
        grid, winner_names={f"v{i}" for i in range(5)},
        max_combo_size=2, max_combos=4,
    )
    assert len(combos) == 4
    assert all(c.name.startswith("combo_") for c in combos)
    # All members must be valid winners
    for c in combos:
        members = c.name.removeprefix("combo_").split("+")
        assert all(m in {f"v{i}" for i in range(5)} for m in members)


# ---------------------------------------------------------------------------
# SharedState round-trip + new mutators
# ---------------------------------------------------------------------------
def test_shared_state_new_fields_round_trip(tmp_path):
    s = SharedState(session_id="t1")
    s.record_discovered_flags(
        framework="sglang",
        backend_flags=["--enable-aiter", "--enable-fused-moe"],
        param_flags=["--max-num-seqs"],
        source_path="/tmp/server_args.py",
    )
    s.push_backend_winners_round(
        action="backends",
        base_tput=1000.0,
        base_extra_args="",
        winners=[
            {"name": "aiter_kv_fp8", "output_throughput": 1100.0,
             "extra_sglang_args": "--kv-cache-dtype fp8"},
        ],
        best={"name": "aiter_kv_fp8", "output_throughput": 1100.0,
              "extra_sglang_args": "--kv-cache-dtype fp8"},
    )
    s.mark_synergy_attempted(["aiter_kv_fp8", "decode_steps_2"])
    s.mark_synergy_attempted(["decode_steps_2", "aiter_kv_fp8"])  # dedup

    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert s2.discovered_flags["sglang"]["backend_flags"] == [
        "--enable-aiter", "--enable-fused-moe",
    ]
    assert s2.discovered_flags["sglang"]["param_flags"] == ["--max-num-seqs"]
    assert len(s2.backend_winners_history) == 1
    entry = s2.backend_winners_history[0]
    assert entry["best"]["name"] == "aiter_kv_fp8"
    assert entry["winners"][0]["gain_pct"] is None
    # Two mark_synergy_attempted calls yield ONE attempt due to canonical key.
    assert len(s2.synergy_attempted) == 1
    assert s2.is_synergy_attempted(["decode_steps_2", "aiter_kv_fp8"])


def test_shared_state_prompt_summary_mentions_new_fields():
    s = SharedState(session_id="t2")
    s.record_discovered_flags(framework="sglang",
                                backend_flags=["--enable-aiter"])
    s.push_backend_winners_round(
        action="params", base_tput=900.0, base_extra_args="",
        winners=[{"name": "v1", "output_throughput": 950.0}],
        best={"name": "v1", "output_throughput": 950.0},
    )
    summary = s.to_prompt_summary()
    assert "discovered_flags=" in summary
    assert "backend_winners_history=" in summary
    assert "synergy_attempted=" in summary
    # Counts surface, not full lists
    assert "sglang:backend=1/param=0" in summary


# ---------------------------------------------------------------------------
# BackendsExecutor — discovery + synergy override + dedup wiring
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_backends_executor_emits_discovery_and_synergy_attempts(
    tmp_path, monkeypatch,
):
    """BackendsExecutor must surface discovered_flags_update + synergy_attempted_new
    in its result dict, and respect the params.synergy_attempted dedup set."""
    fake_args = tmp_path / "server_args.py"
    fake_args.write_text(
        "class ServerArgs:\n"
        "    def __init__(self):\n"
        "        self.attention_backend = 'flashinfer'\n"
        "        self.enable_overlap_schedule = False\n"
    )
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", str(fake_args))
    # We monkeypatch the backends module's path constant directly — the
    # env var is read at import time only.
    from inference_optimizer.orchestrator.action_executors import backends as be_mod
    monkeypatch.setattr(be_mod, "DEFAULT_SGLANG_SERVER_ARGS", fake_args)

    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  envs:\n"
        "    TP: 1\n"
        "    CONC: 8\n"
        "    ISL: 256\n"
        "    OSL: 256\n"
    )

    captured_grids: list[list[GridVariant]] = []
    call_count = {"n": 0}

    async def _fake_run_grid(*, base_yaml_path, base_extra_args, grid,
                              output_root, **_kw):
        captured_grids.append(list(grid))
        call_count["n"] += 1
        # Phase 1: declare two winners; phase 2: empty (covered separately).
        results: list[VariantResult] = []
        for v in grid:
            tput = 1100.0 if v.name in {
                "sglang_aiter_fused_moe", "sglang_kv_fp8",
            } else 900.0
            results.append(VariantResult(
                name=v.name, extra_sglang_args=v.extra_sglang_args,
                extra_envs=dict(v.extra_envs),
                status="succeeded", output_throughput=tput,
            ))
        return results

    monkeypatch.setattr(be_mod, "run_grid", _fake_run_grid)
    # Replace the shipped grid with a tiny synthetic one we can reason about.
    synth_grid = [
        GridVariant("sglang_aiter_fused_moe", "--enable-aiter-fused-moe"),
        GridVariant("sglang_kv_fp8", "--kv-cache-dtype fp8"),
        GridVariant("sglang_baseline_extra", "--enable-overlap-scheduler"),
    ]

    exec_ = BackendsExecutor(
        default_grid=synth_grid,
        default_vllm_grid=[],
        default_config_path=base_yaml,
        session_dir=tmp_path,
    )

    class _Task:
        params = {
            "synergy_mode": "auto",
            "max_combo_size": 2,
            "max_combos": 4,
            "synergy_attempted": ["sglang_aiter_fused_moe+sglang_kv_fp8"],
        }
        task_id = "t-be-1"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    result = await exec_(_Ctx())
    assert result["status"] == "succeeded"

    # Discovery surfaced for Coordinator
    upd = result.get("discovered_flags_update")
    assert upd and upd["framework"] == "sglang"
    assert "--enable-overlap-schedule" in upd["backend_flags"]

    # synergy_attempted_new must EXCLUDE the pre-attempted combo (dedup)
    new_attempts = result.get("synergy_attempted_new") or []
    flat_keys = {"+".join(sorted(c)) for c in new_attempts}
    assert "sglang_aiter_fused_moe+sglang_kv_fp8" not in flat_keys

    # Phase 1 grid was augmented by the AST-discovered booleans (the
    # `enable_overlap_schedule` flag is already in base; only NEW bools
    # appear). At minimum the call ran phase 1 + phase 2.
    assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# ParamsExecutor — discovery surfaces in result
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_params_executor_emits_discovery_update(tmp_path, monkeypatch):
    fake_args = tmp_path / "server_args.py"
    fake_args.write_text(
        "class ServerArgs:\n"
        "    def __init__(self):\n"
        "        self.max_num_seqs = 128\n"
        "        self.cuda_graph_max_bs = 64\n"
    )
    from inference_optimizer.orchestrator.action_executors import params as p_mod
    monkeypatch.setattr(p_mod, "DEFAULT_SGLANG_SERVER_ARGS", fake_args)

    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  envs:\n"
        "    TP: 1\n"
        "    CONC: 8\n"
        "    ISL: 256\n"
        "    OSL: 256\n"
    )

    async def _fake_run_grid(*, base_yaml_path, base_extra_args, grid,
                              output_root, **_kw):
        # All variants succeed with tput slightly above base — keeps the
        # executor on the "winners exist, no combos" code path.
        return [VariantResult(
            name=v.name, extra_sglang_args=v.extra_sglang_args,
            extra_envs=dict(v.extra_envs),
            status="succeeded", output_throughput=950.0,
        ) for v in grid]

    monkeypatch.setattr(p_mod, "run_grid", _fake_run_grid)

    tiny_grid = [
        GridVariant("p_max_num_seqs_256", "--max-num-seqs 256"),
        GridVariant("p_cuda_graph_max_bs_128", "--cuda-graph-max-bs 128"),
    ]
    exec_ = ParamsExecutor(
        default_grid=tiny_grid,
        default_vllm_grid=[],
        default_nccl_grid=[],
        default_config_path=base_yaml,
        session_dir=tmp_path,
    )

    class _Task:
        params = {"base_tput": 900.0, "max_candidates_per_round": 0}
        task_id = "t-p-1"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    result = await exec_(_Ctx())
    upd = result.get("discovered_flags_update")
    assert upd is not None
    assert upd["framework"] == "sglang"
    assert "--max-num-seqs" in upd["param_flags"]
    assert "--cuda-graph-max-bs" in upd["param_flags"]


# ---------------------------------------------------------------------------
# Prompt builder — IR-26 + grid-injection hints
# ---------------------------------------------------------------------------
@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture
def rules_path() -> Path:
    return asset_system_prompts_dir() / "orchestration.md"


def test_prompt_includes_grid_injection_hints_for_grid_actions(
    registry, rules_path,
):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    assert "GRID OVERRIDE" in text
    # All three grid-injectable actions get a hint
    assert "delegate{action_name='backends'" in text
    assert "delegate{action_name='params'" in text
    assert "delegate{action_name='sweep'" in text
    # Sanity: GRID_INJECTABLE_ACTIONS frozen for upstream callers
    assert {"backends", "params", "sweep"} <= GRID_INJECTABLE_ACTIONS


def test_prompt_includes_idea_generation_block(registry, rules_path):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    assert "IDEA GENERATION" in text
    assert "Sub-actions" in text
    assert "Follow-ons" in text
    assert "Retry-with-alternate-strategy" in text
    assert "Synthetic fallback" in text
    assert "Self-reflection" in text


def test_prompt_section_count_unchanged(registry, rules_path):
    """IR-26 must be appended INSIDE section 5, not as a new top-level
    section — the seven-section contract is a hard public contract."""
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        objective_kind="time_only",
        objective_value=None,
        max_minutes=120,
        rules_fragment_path=rules_path,
    )
    top_headers = [
        line.strip() for line in text.splitlines()
        if line.startswith("## ") and not line.startswith("### ")
    ]
    assert top_headers == [
        "## 1. MISSION",
        "## 2. SESSION CONTEXT",
        "## 3. PIPELINE & TIME BUDGET",
        "## 4. ACTIONS YOU MAY USE",
        "## 5. DECISION FRAMEWORK (apply EVERY tick BEFORE emitting)",
        "## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)",
        "## 7. RULES & OUTPUT PROTOCOL",
    ]


# ---------------------------------------------------------------------------
# Per-round caps — backends.py max_candidates_per_round + max_synergy_combos
# ---------------------------------------------------------------------------
def _make_backends_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "base.yaml"
    p.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  envs:\n"
        "    TP: 1\n    CONC: 8\n    ISL: 256\n    OSL: 256\n"
    )
    return p


def _wire_be_module(monkeypatch, tmp_path, captured):
    """Replace run_grid + AST source paths with deterministic stubs."""
    from inference_optimizer.orchestrator.action_executors import backends as be_mod

    fake_args = tmp_path / "server_args.py"
    fake_args.write_text(
        "class ServerArgs:\n"
        "    def __init__(self):\n"
        "        self.enable_overlap_schedule = False\n"
    )
    monkeypatch.setattr(be_mod, "DEFAULT_SGLANG_SERVER_ARGS", fake_args)

    async def _fake_run_grid(*, base_yaml_path, base_extra_args, grid,
                              output_root, **_kw):
        captured.append([v.name for v in grid])
        return [VariantResult(
            name=v.name, extra_sglang_args=v.extra_sglang_args,
            extra_envs=dict(v.extra_envs),
            status="succeeded", output_throughput=1100.0,
        ) for v in grid]

    monkeypatch.setattr(be_mod, "run_grid", _fake_run_grid)
    return be_mod


@pytest.mark.asyncio
async def test_backends_default_cap_5_in_phase1(tmp_path, monkeypatch):
    captured: list[list[str]] = []
    _wire_be_module(monkeypatch, tmp_path, captured)

    synth_grid = [GridVariant(f"v{i}", f"--flag-{i}") for i in range(12)]
    exec_ = BackendsExecutor(
        default_grid=synth_grid,
        default_vllm_grid=[],
        default_config_path=_make_backends_yaml(tmp_path),
        session_dir=tmp_path,
    )

    class _Task:
        # Disable discovery so the cap-of-5 assertion is deterministic
        # (AST scan would otherwise append a few boolean variants).
        params = {"disable_discovery": True}
        task_id = "t-cap-default"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    await exec_(_Ctx())
    # Phase 1 should run exactly 5 variants (the cap), in original order.
    assert captured[0] == [f"v{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_backends_cap_0_disables_cap(tmp_path, monkeypatch):
    captured: list[list[str]] = []
    _wire_be_module(monkeypatch, tmp_path, captured)

    synth_grid = [GridVariant(f"v{i}", f"--flag-{i}") for i in range(12)]
    exec_ = BackendsExecutor(
        default_grid=synth_grid,
        default_vllm_grid=[],
        default_config_path=_make_backends_yaml(tmp_path),
        session_dir=tmp_path,
    )

    class _Task:
        params = {"max_candidates_per_round": 0, "disable_discovery": True}
        task_id = "t-cap-off"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    await exec_(_Ctx())
    # Phase 1 must run the WHOLE grid (12 variants).
    assert len(captured[0]) == 12


@pytest.mark.asyncio
async def test_backends_llm_grid_override_bypasses_cap(tmp_path, monkeypatch):
    captured: list[list[str]] = []
    _wire_be_module(monkeypatch, tmp_path, captured)

    exec_ = BackendsExecutor(
        default_grid=[],
        default_vllm_grid=[],
        default_config_path=_make_backends_yaml(tmp_path),
        session_dir=tmp_path,
        default_max_candidates_per_round=5,
    )

    llm_grid = [
        {"name": f"llm_v{i}", "extra_sglang_args": f"--llm-{i}"}
        for i in range(9)
    ]

    class _Task:
        params = {"grid": llm_grid}
        task_id = "t-llm-grid"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    await exec_(_Ctx())
    # All 9 LLM-specified variants must run despite default cap=5.
    assert len(captured[0]) == 9


@pytest.mark.asyncio
async def test_backends_skips_already_tested_when_capped(tmp_path, monkeypatch):
    captured: list[list[str]] = []
    _wire_be_module(monkeypatch, tmp_path, captured)

    synth_grid = [GridVariant(f"v{i}", f"--flag-{i}") for i in range(12)]
    exec_ = BackendsExecutor(
        default_grid=synth_grid,
        default_vllm_grid=[],
        default_config_path=_make_backends_yaml(tmp_path),
        session_dir=tmp_path,
    )

    # ``tested_variant_names`` was replaced by the ``backends_search``
    # ledger in Phase 2. Seed it with the first five variants' content
    # fingerprints so the executor's filter drops them before launch.
    from inference_optimizer.orchestrator.action_executors._grid_runner import (
        variant_fingerprint,
    )
    tested = {}
    name_index = {}
    for i in range(5):
        fp = variant_fingerprint(f"--flag-{i}", {})
        tested[fp] = {
            "name": f"v{i}", "extra_sglang_args": f"--flag-{i}",
            "extra_envs": {},
        }
        name_index[f"v{i}"] = fp

    class _Task:
        params = {
            "backends_search": {
                "schema_version": 1,
                "accepted": [],
                "rejected": [],
                "tested": tested,
                "name_index": name_index,
                "cursor": 5,
            },
            "disable_discovery": True,
        }
        task_id = "t-tested"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    await exec_(_Ctx())
    # First 5 are tested → next round must take v5..v9 instead.
    assert captured[0] == ["v5", "v6", "v7", "v8", "v9"]


@pytest.mark.asyncio
async def test_backends_synergy_cap_default_4(tmp_path, monkeypatch):
    captured: list[list[str]] = []
    _wire_be_module(monkeypatch, tmp_path, captured)

    # Tiny grid + many synergy groups so the cap matters.
    synth_grid = [GridVariant(f"v{i}", f"--flag-{i}") for i in range(6)]
    exec_ = BackendsExecutor(
        default_grid=synth_grid,
        default_vllm_grid=[],
        default_config_path=_make_backends_yaml(tmp_path),
        session_dir=tmp_path,
        default_max_candidates_per_round=6,  # let all 6 run as phase-1
    )

    class _Task:
        params = {
            "synergy_mode": "auto",
            "max_combo_size": 2,
            "disable_discovery": True,
            # Without an explicit cap, default_max_synergy_combos=4 wins.
        }
        task_id = "t-synergy-default"

    class _Ctx:
        task = _Task()
        extra: dict = {}

    await exec_(_Ctx())
    # captured[0] = phase-1 (6 variants), captured[1] = phase-2 (combos).
    assert len(captured) == 2
    assert len(captured[1]) <= 4, captured[1]


# ---------------------------------------------------------------------------
# Coordinator integration — promotion writes new SharedState fields
# ---------------------------------------------------------------------------
def test_promote_to_shared_state_propagates_discovered_and_synergy(tmp_path):
    """Direct unit test on _promote_to_shared_state without spinning up
    the full Coordinator reactor: we just need the SharedState side-effects."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    # The Coordinator constructor wants a session_dir + role_registry; we
    # build the smallest possible instance that lets us call the helper.
    # Most fields are only used by the reactor loop, so a default-constructed
    # SharedState attached after the fact is fine.
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(session_id="t-promote")

    fake_result = {
        "status": "succeeded",
        "base_tput": 1000.0,
        "best_variant": {
            "name": "x_winner",
            "extra_sglang_args": "--enable-aiter",
            "output_throughput": 1080.0,
        },
        "winners": [
            {"name": "x_winner", "output_throughput": 1080.0,
             "extra_sglang_args": "--enable-aiter"},
        ],
        "discovered_flags_update": {
            "framework": "sglang",
            "backend_flags": ["--enable-aiter", "--enable-fused-moe"],
            "source_path": "/tmp/server_args.py",
        },
        "synergy_attempted_new": [["x_winner", "y"]],
    }

    class _Task:
        kind = "backends"
        params = {"base_extra_args": "--prev-arg"}

    import asyncio

    # We patch save() to a no-op so the test doesn't depend on the on-disk
    # state.json layout.
    with patch.object(SharedState, "save", lambda *_a, **_k: None):
        asyncio.run(coord._promote_to_shared_state(
            "backends", fake_result, task=_Task(),
        ))

    assert "sglang" in coord.shared_state.discovered_flags
    assert (
        coord.shared_state.discovered_flags["sglang"]["backend_flags"]
        == ["--enable-aiter", "--enable-fused-moe"]
    )
    assert coord.shared_state.is_synergy_attempted(["x_winner", "y"])
    assert len(coord.shared_state.backend_winners_history) == 1
    rec = coord.shared_state.backend_winners_history[0]
    assert rec["base_extra_args"] == "--prev-arg"
    assert rec["winners"][0]["name"] == "x_winner"
