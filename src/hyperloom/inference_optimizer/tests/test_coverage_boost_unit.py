# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Focused unit coverage for small pure-logic helpers.

These modules are import-level libraries whose error / edge branches were
previously only exercised indirectly. Each test here pins one concrete
branch so the contract stays covered without standing up the full runtime.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------------- #
# orchestrator.gain_math                                                       #
# --------------------------------------------------------------------------- #
def test_gain_math_branches() -> None:
    from hyperloom.orchestrator import gain_math

    # gain_pct: non-positive new / non-positive base -> None.
    assert gain_math.gain_pct(0, 100.0) is None
    assert gain_math.gain_pct(120.0, 0.0) is None
    assert gain_math.gain_pct(110.0, 100.0) == pytest.approx(10.0)

    # gain_pct_or_zero: base <= 0 -> 0.0 (line 16).
    assert gain_math.gain_pct_or_zero(120.0, 0.0) == 0.0
    assert gain_math.gain_pct_or_zero(90.0, 100.0) == pytest.approx(-10.0)

    # incremental_gain_pct: ref <= 0 -> None (line 23).
    assert gain_math.incremental_gain_pct(120.0, 0.0) is None
    assert gain_math.incremental_gain_pct(110.0, 100.0) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# orchestrator.kb_writeback                                                    #
# --------------------------------------------------------------------------- #
def test_kb_writeback_default_root_override(monkeypatch, tmp_path) -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    root = kb_writeback._default_kb_root()
    assert root == tmp_path / "framework_optimization"


async def test_kb_writeback_rejects_unknown_outcome() -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    with pytest.raises(ValueError):
        await kb_writeback.write_framework_record(
            pr_url="u",
            pr_sha="s",
            patch_path="p",
            outcome="not_a_real_outcome",
            tps_delta_pct=1.0,
            session_id="sess",
        )


async def test_kb_writeback_appends_record(monkeypatch, tmp_path) -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path / "fa")
    path = await kb_writeback.write_framework_record(
        pr_url="https://x/pr/1",
        pr_sha="abc",
        patch_path="/tmp/p.patch",
        outcome=kb_writeback.OUTCOME_INTEGRATED,
        tps_delta_pct=3.5,
        session_id="sess-1",
    )
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["outcome"] == "integrated"
    assert rec["tps_delta_pct"] == 3.5


# --------------------------------------------------------------------------- #
# baseline_comparison.name_mapping                                            #
# --------------------------------------------------------------------------- #
def test_name_mapping_paths() -> None:
    from hyperloom.inference_optimizer.baseline_comparison import name_mapping as nm

    assert nm.to_inferencex_name("") is None
    # Whitespace-only collapses to empty after strip (line 76).
    assert nm.to_inferencex_name("   ") is None
    assert nm.to_inferencex_name("/wekafs/models/MiniMaxAI-MiniMax-M2.5") == "MiniMax-M2.5"
    assert nm.to_inferencex_name("totally-unknown-model") is None


# --------------------------------------------------------------------------- #
# baseline_comparison.types                                                   #
# --------------------------------------------------------------------------- #
def test_baseline_summary_from_dict_partial() -> None:
    from hyperloom.inference_optimizer.baseline_comparison.types import BaselineSummary

    summary = BaselineSummary.from_dict({"query": {"model": "m"}, "best": None})
    assert summary.query.model == "m"
    assert summary.best is None
    # Round-trips back to a dict.
    assert summary.to_dict()["query"]["model"] == "m"


# --------------------------------------------------------------------------- #
# framework_registry                                                          #
# --------------------------------------------------------------------------- #
def test_framework_registry_surface() -> None:
    from hyperloom.inference_optimizer import framework_registry as fr

    assert "sglang" in fr.names()
    assert fr.is_supported("SGLang") is True
    assert fr.is_supported("nope") is False
    spec = fr.get("sglang")  # lines 134-135
    assert spec.name == "sglang"
    assert fr.kind("xdit") == fr.SCRIPTABLE
    assert fr.is_scriptable("xdit") is True
    assert fr.is_scriptable("sglang") is False
    assert fr.extra_args_env("vllm") == "EXTRA_VLLM_ARGS"
    assert fr.throughput_unit("xdit") == "img/s"
    assert fr.supports_server_reuse("atom") is False
    assert fr.repo_url("vllm")
    assert fr.repo_url("does-not-exist") is None
    # Unknown name falls back to the default spec.
    assert fr.kind("unknown") == fr.get(fr.DEFAULT_FRAMEWORK).kind


# --------------------------------------------------------------------------- #
# orchestrator.quantization_schemes                                          #
# --------------------------------------------------------------------------- #
def test_quantization_join_and_prompt() -> None:
    from hyperloom.orchestrator.phases import quantization_schemes as qs

    assert qs._join_clauses([]) == ""  # line 131
    assert qs._join_clauses(["a"]) == "a"
    assert qs._join_clauses(["a", "b", "c"]) == "a, b and c"

    cfg = qs.QuantizationConfig(global_scheme="fp8")
    # model_path without skill_path -> "Quantize ..." intro (line 232).
    prompt = qs.build_quantization_prompt(cfg, model_path="/m", gpu_type="mi300x")
    assert "Quantize /m on an MI300X target." in prompt
    assert "Quantization strategy" in prompt


# --------------------------------------------------------------------------- #
# orchestrator.objective                                                      #
# --------------------------------------------------------------------------- #
def test_tput_objective_progress_zero() -> None:
    from hyperloom.orchestrator.state.objective import TargetTputObjective

    obj = TargetTputObjective(target_tput_per_gpu=100.0)
    state = SimpleNamespace(current_best={}, baseline_tput=0.0)
    assert obj.progress(state) == 0.0  # line 226
    assert obj.reached(state) is False
    assert obj.describe() == "target_tput_per_gpu=100.0"


def test_baseline_objective_progress_zero_ref(tmp_path) -> None:
    from hyperloom.orchestrator.state.objective import TargetBaselineObjective

    report = tmp_path / "benchmark_report.json"
    report.write_text(json.dumps({"throughput": {"output_throughput": 50.0}}), encoding="utf-8")
    obj = TargetBaselineObjective(baseline_dir=str(tmp_path))
    assert obj.kind() == "baseline"
    # Force the degenerate ref to exercise the guard (line 339).
    obj._ref_tput = 0.0
    state = SimpleNamespace(current_best={"tput": 10.0}, baseline_tput=0.0)
    assert obj.progress(state) == 0.0


def test_baseline_objective_invalid_dir() -> None:
    from hyperloom.orchestrator.state.objective import ObjectiveError, TargetBaselineObjective

    with pytest.raises(ObjectiveError):
        TargetBaselineObjective(baseline_dir="/nonexistent/path/zzz")


# --------------------------------------------------------------------------- #
# recipe_kb.canonical_id                                                      #
# --------------------------------------------------------------------------- #
def test_canonical_id_roundtrip_and_errors() -> None:
    from hyperloom.inference_optimizer.recipe_kb import canonical_id as cid

    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("")  # line 110
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("inference:only:three")
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("wrongprefix:a:b:c:d:e:f:g")
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("inference:a::c:d:e:f:g")

    full = "inference:model:hw:fw:mt:arch:fwv:prec"
    comps = cid.cid_to_path_components(full)
    assert comps == ("model", "hw", "fw", "mt", "arch", "fwv", "prec")


def test_canonical_id_for_path_errors(tmp_path) -> None:
    from hyperloom.inference_optimizer.recipe_kb import canonical_id as cid

    # recipe_dir not under root (lines 187-188).
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.canonical_id_for_path(root=tmp_path / "root", recipe_dir=tmp_path / "elsewhere")

    # Wrong depth under root (line 194).
    root = tmp_path / "root"
    shallow = root / "only_one_level"
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.canonical_id_for_path(root=root, recipe_dir=shallow)


# --------------------------------------------------------------------------- #
# recipe_kb.schema.Attempt                                                    #
# --------------------------------------------------------------------------- #
def test_attempt_from_dict_with_fitness() -> None:
    from hyperloom.inference_optimizer.recipe_kb.schema import Attempt

    a = Attempt.from_dict({"id": "7", "fitness": "1.5", "outcome": "win"})
    assert a.id == 7
    assert a.fitness == pytest.approx(1.5)
    assert a.outcome == "win"
    # Round-trip includes fitness only when set.
    assert "fitness" in a.to_dict()
    assert "fitness" not in Attempt(id=1).to_dict()


# --------------------------------------------------------------------------- #
# orchestrator.action_executors._file_lock                                    #
# --------------------------------------------------------------------------- #
def test_file_lock_no_fcntl(monkeypatch, tmp_path) -> None:
    import builtins

    from hyperloom.orchestrator.actions.executors import _file_lock

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Falls through without exclusion (lines 19-21); body still runs.
    with _file_lock.best_effort_file_lock(str(tmp_path / "lock")):
        ran = True
    assert ran


def test_file_lock_acquires(tmp_path) -> None:
    from hyperloom.orchestrator.actions.executors import _file_lock

    with _file_lock.best_effort_file_lock(str(tmp_path / "lock"), label="t"):
        pass


# --------------------------------------------------------------------------- #
# orchestrator.action_executors._framework_gap_composer                       #
# --------------------------------------------------------------------------- #
def test_framework_gap_bottleneck(tmp_path) -> None:
    from hyperloom.orchestrator.actions.executors import _framework_gap_composer as gc

    # top_kernels as list of strings (lines 91-92).
    bp = tmp_path / "bd.json"
    bp.write_text(json.dumps({"top_kernels": ["fused_moe_gemm_kernel"]}), encoding="utf-8")
    assert gc._extract_bottleneck_from_breakdown(str(bp)) == "moe"

    # No candidates -> "" (line 102).
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"top_kernels": []}), encoding="utf-8")
    assert gc._extract_bottleneck_from_breakdown(str(empty)) == ""

    assert gc._extract_bottleneck_from_breakdown(None) == ""
    assert gc._extract_bottleneck_from_breakdown(str(tmp_path / "missing.json")) == ""


def test_framework_gap_compose() -> None:
    from hyperloom.orchestrator.actions.executors import _framework_gap_composer as gc

    desc, keywords = gc.compose_gap(framework="sglang", gpu_type="MI300X", model_class="moe_mla")
    assert "sglang" in keywords
    assert "moe" in keywords
    assert isinstance(desc, str) and desc


# --------------------------------------------------------------------------- #
# manifest                                                                     #
# --------------------------------------------------------------------------- #
def test_gpu_specialist_capacity_coercions() -> None:
    from hyperloom.inference_optimizer import manifest

    assert manifest._gpu_specialist_capacity_from_args(argparse.Namespace(gpu_specialist_capacity=4)) == 4
    # Non-int coerces through the except branch (lines 370-371) then detects.
    val = manifest._gpu_specialist_capacity_from_args(argparse.Namespace(gpu_specialist_capacity="nope"))
    assert isinstance(val, int) and val >= 0


# --------------------------------------------------------------------------- #
# protocol.intent                                                             #
# --------------------------------------------------------------------------- #
def test_validate_envelope_structural_errors() -> None:
    from hyperloom.inference_optimizer.protocol.intent import IntentValidationError, validate_envelope

    with pytest.raises(IntentValidationError):
        validate_envelope("not a dict")  # type: ignore[arg-type]  # line 194
    with pytest.raises(IntentValidationError):
        validate_envelope({})  # missing intents -> line 196
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": "x"})  # line 199
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": ["x"]})  # item not dict -> line 204
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "alert"}]})  # missing payload -> 206
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "alert", "payload": "x"}]})  # 213
    with pytest.raises(IntentValidationError):
        validate_envelope({"intents": [{"intent_type": "bad_type", "payload": {}}]})


def test_validate_envelope_review_verdict_map_keys() -> None:
    from hyperloom.inference_optimizer.protocol.intent import IntentValidationError, validate_envelope

    # verdict_map with a non-string key (line 263).
    bad = {
        "intents": [
            {
                "intent_type": "review_verdict",
                "payload": {"target_proposal_msg_id": "m1", "verdict_map": {"": {"verdict": "approve"}}},
            }
        ]
    }
    with pytest.raises(IntentValidationError):
        validate_envelope(bad)


def test_validate_envelope_happy_path() -> None:
    from hyperloom.inference_optimizer.protocol.intent import IntentType, validate_envelope

    intents = validate_envelope(
        {"intents": [{"intent_type": "alert", "payload": {"severity": "high", "summary": "s"}}]}
    )
    assert intents[0].type is IntentType.ALERT


# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #
def test_paths_helpers(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer import paths

    # _sanitize_model_basename empty -> "session" (line 139).
    assert paths._sanitize_model_basename("   ") == "session"
    assert paths._sanitize_model_basename("/a/b/Model:X") == "Model_X"

    # asset_root override that exists (line 267).
    monkeypatch.setenv(paths.ENV_OVERRIDE_ASSET_ROOT, str(tmp_path))
    assert paths.asset_root() == tmp_path
    # Derived asset dirs (lines 277, 304).
    assert paths.asset_scripts_dir() == tmp_path / "scripts"
    assert paths.asset_kernel_opt_dir() == tmp_path / "kernel_opt"

    # find_latest returns None when workspace root is not a dir (line 179).
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "does_not_exist"))
    assert paths.find_latest_per_session_dir() is None

    sd = tmp_path / "sd"
    assert paths.kernel_agent_runs_root(sd) == sd / "kernel-agent"  # line 383
    assert paths.optimizer_runs_dir(sd) == sd / "optimizer_runs"  # line 396


def test_paths_asset_root_missing_override(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer import paths

    monkeypatch.setenv(paths.ENV_OVERRIDE_ASSET_ROOT, str(tmp_path / "nope"))
    with pytest.raises(paths.AssetRootNotFound):
        paths.asset_root()


# --------------------------------------------------------------------------- #
# orchestrator.backends._runtime_bridge                                       #
# --------------------------------------------------------------------------- #
def test_runtime_bridge_success(monkeypatch, tmp_path) -> None:
    from hyperloom.orchestrator.roles import _runtime_bridge as rb

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    call = rb.RuntimeCall(
        phase="tick",
        request_path=tmp_path / "req.json",
        out_path=tmp_path / "out.json",
        cwd=tmp_path,
        env={"X": "1"},
    )
    rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="robustness", timeout_sec=5.0)
    assert "runtime.cli" in captured["cmd"]
    assert "tick" in captured["cmd"]


def test_runtime_bridge_timeout(monkeypatch, tmp_path) -> None:
    from hyperloom.orchestrator.roles import _runtime_bridge as rb
    from hyperloom.orchestrator.roles.base import BackendError

    def fake_run(cmd, **kwargs):
        raise rb.subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    call = rb.RuntimeCall(
        phase="prepare-review",
        request_path=tmp_path / "req.json",
        out_path=tmp_path / "out.json",
        cwd=tmp_path,
        env={},
    )
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="critic", timeout_sec=5.0)


def test_runtime_bridge_not_found_and_nonzero(monkeypatch, tmp_path) -> None:
    from hyperloom.orchestrator.roles import _runtime_bridge as rb
    from hyperloom.orchestrator.roles.base import BackendError

    call = rb.RuntimeCall(
        phase="tick",
        request_path=tmp_path / "req.json",
        out_path=tmp_path / "out.json",
        cwd=tmp_path,
        env={},
    )

    def fake_run_missing(cmd, **kwargs):
        raise FileNotFoundError("python gone")

    monkeypatch.setattr(rb.subprocess, "run", fake_run_missing)
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="a", timeout_sec=1.0)

    def fake_run_rc(cmd, **kwargs):
        return SimpleNamespace(returncode=3, stderr="boom")

    monkeypatch.setattr(rb.subprocess, "run", fake_run_rc)
    with pytest.raises(BackendError):
        rb.invoke_runtime_cli(call, module="runtime.cli", agent_label="a", timeout_sec=1.0)
