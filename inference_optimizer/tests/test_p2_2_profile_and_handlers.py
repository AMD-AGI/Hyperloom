"""P2-2 tests: ProfileExecutor + kernel REQUEST programmatic handlers."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from inference_optimizer import cli as optimizer_cli
from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.action_executors.baseline import (
    BaselineExecutor,
    _default_baseline_config,
    _materialize_config_with_envs,
)
from inference_optimizer.orchestrator.action_executors.profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
    _default_profile_config,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
    MockTurn,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    RunnerContext, SubAgentRunner,
)
from inference_optimizer.manifest import build_manifest
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path))
    kernel_agent_root = Path(__file__).resolve().parents[2] / "kernel-agent"
    monkeypatch.setattr(krh, "HYPERLOOM_KERNEL_AGENT_ROOT", kernel_agent_root)
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n)
            for n in ("orchestration", "kernel", "critic", "robustness")}


def test_mi325x_keeps_real_gpu_type_but_uses_mi300x_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setenv("TARGET_GPU_TYPE", "mi325x")
    args = SimpleNamespace(
        model="/models/Qwen3",
        model_class="",
        target_summary="",
        max_hours=1,
        no_kernel=False,
        gpu_type="mi325x",
        target_gain=None,
        target_tput=None,
    )

    assert optimizer_cli._gpu_runner_type("mi325x") == "mi300x"
    assert optimizer_cli._GFX_TO_RUNNER.get("gfx1100") is None
    manifest = build_manifest(tmp_path, args=args, session_id="mi325x-session")
    state = optimizer_cli._seed_shared_state(
        tmp_path, args, session_id="mi325x-session",
    )

    assert manifest["gpu_type"] == "mi325x"
    assert state.gpu_type == "mi325x"
    assert os.environ["TARGET_GPU_TYPE"] == "mi325x"
    assert os.environ["GPU_TYPE"] == "mi300x"


# ===========================================================================
# ProfileExecutor
# ===========================================================================
def test_profile_default_config_path_is_in_assets():
    assert "profile_sglang.yaml" in str(PROFILE_DEFAULT_CONFIG)
    assert PROFILE_DEFAULT_CONFIG.exists(), \
        "profile YAML must ship as a package asset"


def test_profile_yaml_has_torch_profiler_enabled():
    """The whole point of the profile config is profiler ON."""
    import yaml
    with PROFILE_DEFAULT_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True


# ===========================================================================
# Regression: model_path injection beats the YAML's hardcoded fallback.
#
# Bug: the shipped baseline_sglang.yaml / profile_sglang.yaml pin
# `benchmark.model: /wekafs/models/Qwen-Qwen3-8B` as a fallback for offline
# Magpie use. The CLI's --model arg only flowed into SharedState.model_path;
# if the executor did not propagate it into the materialized YAML, Magpie
# silently benchmarked Qwen3-8B no matter what the user asked for.
# _materialize_config_with_envs(model_path=...) is the single seam that
# prevents this — locking it down here.
# ===========================================================================
def test_materialize_config_injects_model_path(tmp_path):
    """Default YAML's hardcoded Qwen3-8B must be overridden when caller
    passes ``model_path`` — otherwise the silent fallback bug returns."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        model_path="/wekafs/models/DeepSeek-R1-0528",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/wekafs/models/DeepSeek-R1-0528"


def test_materialize_config_leaves_model_alone_without_override(tmp_path, monkeypatch):
    """When no model_path is passed, the materialized YAML still has the
    original model field from the source YAML (not overwritten)."""
    import yaml
    # Clear ISL/OSL/MAX_MODEL_LEN env so they don't inject
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "PRECISION"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert "Qwen" in rendered["benchmark"]["model"]


def test_materialize_config_injects_model_with_other_overrides(tmp_path):
    """Co-existence: model_path + extra_envs should both land in the
    materialized YAML."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        extra_envs={"FOO": "bar"},
        model_path="/some/model",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/some/model"
    assert rendered["benchmark"]["envs"]["FOO"] == "bar"


# ===========================================================================
# Regression: gpu_type injection sets runner_type AND removes the legacy
# `benchmark_script` field so Magpie's runner_type -> script logic wins.
# ===========================================================================
def test_materialize_config_injects_runner_type(tmp_path):
    """gpu_type kwarg must land in benchmark.runner_type as-is."""
    import yaml
    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        gpu_type="mi355x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["runner_type"] == "mi355x"


def test_materialize_config_pops_legacy_benchmark_script(tmp_path):
    """If the source YAML still hardcodes a benchmark_script (priority 1
    in Magpie's resolver), gpu_type must remove it; otherwise runner_type
    is silently ignored and the run uses the wrong GPU's script."""
    import yaml
    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "sglang",
            "model": "/m",
            "benchmark_script": "sglang_mi300x.sh",  # legacy field
        },
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = _materialize_config_with_envs(
        src_yaml, out_dir, gpu_type="mi355x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["runner_type"] == "mi355x"
    assert "benchmark_script" not in rendered["benchmark"], \
        "legacy benchmark_script must be popped so runner_type wins"


# ===========================================================================
# Regression: TP / CONC env override yaml hardcode (DSR1-0528 verification
# was deadlooping because TP=8 env was silently ignored, vllm ran with
# yaml-hardcoded TP=1 and OOM-ed retry forever).
# ===========================================================================
def test_materialize_config_tp_env_overrides_yaml_hardcode(tmp_path, monkeypatch):
    """TP env var must override yaml hardcode (was 1, becomes 8)."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["TP"] == 8, f"TP not overridden: {envs.get('TP')}"


def test_materialize_config_conc_env_overrides_yaml_hardcode(tmp_path, monkeypatch):
    """CONC env var must override yaml hardcode."""
    import yaml
    monkeypatch.setenv("CONC", "64")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["CONC"] == 64, f"CONC not overridden: {envs.get('CONC')}"


def test_materialize_config_rocr_visible_devices_auto_expands_when_tp_overridden(
    tmp_path, monkeypatch,
):
    """When TP=8 is set via env but ROCR_VISIBLE_DEVICES isn't explicit,
    expand the GPU list to 0..TP-1 so vllm/sglang sees enough devices."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7", (
        f"ROCR_VISIBLE_DEVICES not auto-expanded: {envs.get('ROCR_VISIBLE_DEVICES')}"
    )


def test_materialize_config_rocr_visible_devices_explicit_env_wins_when_enough(
    tmp_path, monkeypatch,
):
    """Explicit ROCR_VISIBLE_DEVICES wins when it has at least TP devices."""
    import yaml
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "4,5,6,7"


def test_materialize_config_rocr_visible_devices_expands_when_under_tp(
    tmp_path, monkeypatch,
):
    """If explicit ROCR_VISIBLE_DEVICES has fewer devices than TP requires,
    `_workload_envs` auto-expands to 0..TP-1 and logs a warning, so SGLang
    actually sees enough GPUs to start."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_materialize_config_rocr_unchanged_when_tp1(tmp_path, monkeypatch):
    """When TP=1 (default), don't auto-touch ROCR_VISIBLE_DEVICES."""
    import yaml
    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "sglang",
            "model": "/m",
            "envs": {
                "TP": 1,
                "CONC": 8,
                "ISL": 256,
                "OSL": 256,
                "ROCR_VISIBLE_DEVICES": "1",
            },
        },
    }))
    for k in ("TP", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(src_yaml, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    # yaml default is "1" — should be preserved as-is when TP not overridden upward
    assert envs.get("ROCR_VISIBLE_DEVICES") == "1"


# ===========================================================================
# Regression #194: steady-state window must follow the TraceLens magpie
# skill formulas, not the old `delay = 5*CONC; max = clamp(16*OSL/CONC,4,64)`
# placeholders.
#
# Skill:
#   max_iters   = min(1024, max(256, OSL * 16 / CONC))
#   delay_iters = OSL * (RANDOM_RANGE_RATIO + 1) * 3 - max_iters / 2
#
# Worked example used in the issue: OSL=1024, CONC=32, R=1
#   max_iters   = min(1024, max(256, 1024*16/32))   = max(256, 512) = 512
#   delay_iters = 1024 * (1+1) * 3 - 512/2          = 6144 - 256    = 5888
#
# Previous Hyperloom code gave (160, 64) for the same inputs — roughly 1/8
# of the skill window and ignored R entirely, so issue #194 §1 flagged
# Optimizer-driven profiles as under-representing decode-heavy steady state.
# ===========================================================================
def _profile_yaml(tmp_path, framework: str, envs: dict) -> Path:
    """Synthesize a minimal profile YAML the materializer recognises as
    PROFILE=1 + torch_profiler.enabled=True.
    """
    import yaml as _yaml
    src = tmp_path / f"src_{framework}.yaml"
    src.write_text(_yaml.safe_dump({
        "benchmark": {
            "framework": framework,
            "model": "/m",
            "envs": {"PROFILE": "1", **envs},
            "profiler": {"torch_profiler": {"enabled": True}},
        },
    }))
    return src


def _clear_workload_env(monkeypatch):
    for k in (
        "CONC", "ISL", "OSL", "TP", "MAX_MODEL_LEN",
        "RANDOM_RANGE_RATIO", "ROCR_VISIBLE_DEVICES", "FRAMEWORK",
    ):
        monkeypatch.delenv(k, raising=False)


def test_materialize_profile_window_vllm_skill_formula_default_R(
    tmp_path, monkeypatch,
):
    """vLLM: OSL=1024, CONC=32, R unset → max=512, delay=5888 per skill."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 5888" in extra, extra
    assert "--profiler-config.max_iterations 512" in extra, extra


def test_materialize_profile_window_vllm_skill_formula_explicit_R(
    tmp_path, monkeypatch,
):
    """vLLM: explicit R=0.5 must shrink delay (skill: 3*OSL*(R+1) term)."""
    import yaml
    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("RANDOM_RANGE_RATIO", "0.5")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    # R=0.5: delay = 1024 * 1.5 * 3 - 512/2 = 4608 - 256 = 4352; max=512.
    extra = envs["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 4352" in extra, extra
    assert "--profiler-config.max_iterations 512" in extra, extra
    # And R must round-trip into the YAML as a float, not stringified-int.
    assert envs["RANDOM_RANGE_RATIO"] == 0.5


def test_materialize_profile_window_sglang_skill_formula(
    tmp_path, monkeypatch,
):
    """SGLang path writes the same window into PROFILE_EXTRA_BODY."""
    import json
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    body = json.loads(rendered["benchmark"]["envs"]["PROFILE_EXTRA_BODY"])
    assert body["start_step"] == 5888
    assert body["num_steps"] == 512


def test_materialize_profile_window_clamps_to_skill_floor(
    tmp_path, monkeypatch,
):
    """Skill: max_iters has a floor of 256 (not 4) and a ceiling of 1024.

    OSL=256, CONC=64 ⇒ 16*OSL/CONC = 64, so the floor must kick in.
    Old formula clamped to 64; new formula must clamp to 256.
    """
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 64, "ISL": 256, "OSL": 256})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 256" in extra, extra


# ===========================================================================
# Regression #194 §2: NUM_PROMPTS must be sized to cover the steady-state
# window. With the skill formulas, delay_iters reaches into the thousands
# for non-trivial OSL — an under-sized NUM_PROMPTS makes the engine exit
# before the profile window opens, yielding empty traces.
#
# The formula (from `_workload_envs`):
#   required_iters    = delay_iters + max_iters
#   iters_to_prompts  = ceil(required_iters * CONC / OSL)   # batch math
#   NUM_PROMPTS       = max(CONC, iters_to_prompts * 2)     # 2x buffer
#
# Profile mode FORCE-overrides any caller-supplied NUM_PROMPTS — we own
# the floor and an under-sized value silently kills the trace.
# ===========================================================================
def test_materialize_profile_num_prompts_covers_steady_state_window(
    tmp_path, monkeypatch,
):
    """OSL=1024 / CONC=32 / R=1 → delay+max = 6400 iters ⇒ NUM_PROMPTS=400."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # delay=5888, max=512 → required=6400; 6400*32/1024 = 200; *2 = 400.
    assert envs["NUM_PROMPTS"] == 400, envs.get("NUM_PROMPTS")


def test_materialize_profile_num_prompts_floors_at_conc_for_tiny_osl(
    tmp_path, monkeypatch,
):
    """Tiny OSL with skill floor max_iters=256 still produces a sane
    NUM_PROMPTS (covers the floor's delay+max window)."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 64, "OSL": 64})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # max=256, delay=64*2*3-128=256, required=512; 512*32/64=256; *2=512.
    assert envs["NUM_PROMPTS"] == 512, envs.get("NUM_PROMPTS")


def test_materialize_profile_force_overrides_user_num_prompts(
    tmp_path, monkeypatch,
):
    """Profile mode must IGNORE caller-supplied NUM_PROMPTS — an
    under-sized value (skill default `max_concurrency * 1`) would
    silently empty the trace."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(
        tmp_path, "vllm",
        # Caller deliberately under-sizes to trip the regression.
        {"CONC": 32, "ISL": 256, "OSL": 1024, "NUM_PROMPTS": 32},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # Hyperloom-computed 400 must win over the caller's 32.
    assert envs["NUM_PROMPTS"] == 400, envs.get("NUM_PROMPTS")


def test_materialize_non_profile_keeps_legacy_seq_cost_factor(
    tmp_path, monkeypatch,
):
    """The §2 override is profile-only. Baseline / sweep paths must
    still get the existing seq_cost-based NUM_PROMPTS (or honour a
    caller-supplied value), so the §2 fix can't accidentally explode
    baseline run lengths."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = tmp_path / "baseline.yaml"
    src.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "vllm",
            "model": "/m",
            "envs": {"CONC": 32, "ISL": 256, "OSL": 1024},
            # No profiler.torch_profiler.enabled, no PROFILE=1.
        },
    }))
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # seq_cost=1280 → factor=5 → CONC*5 = 160 (legacy baseline path).
    assert envs["NUM_PROMPTS"] == 160, envs.get("NUM_PROMPTS")


# ===========================================================================
# Regression #194 §4 / §5: when the runtime patcher (server_patcher)
# successfully applies TraceLens's patches to the in-container vLLM /
# SGLang install, materialize must auto-append the flags that only
# exist in the patched build:
#
#   * vLLM:   --profiler-config.capture_torch_profiler_dir <dir>
#   * vLLM:   --profiler-config.detailed_trace_annotation True
#   * SGLang: --enable-shape-discovery-for-cuda-graph-profile
#
# When the patcher fails-soft (returns False), materialize must NOT
# inject any of those — otherwise unpatched vLLM rejects them as
# unknown JSON keys and crashes the entire profile.
#
# The kill switch HYPERLOOM_ENABLE_PATCH=0 must short-circuit the
# patcher entirely so users can disable runtime patching when their
# image is a custom fork / read-only / under audit.
# ===========================================================================
def _mock_patchers(monkeypatch, *, vllm: bool, sglang: bool) -> dict[str, int]:
    """Replace the two patcher symbols on `_workload_envs` with stubs
    that record invocation counts so we can assert per-framework
    dispatch (vLLM path must not invoke the SGLang patcher and vice
    versa)."""
    from inference_optimizer.orchestrator.action_executors import _workload_envs
    counts = {"vllm": 0, "sglang": 0}

    def _vllm_stub() -> bool:
        counts["vllm"] += 1
        return vllm

    def _sglang_stub() -> bool:
        counts["sglang"] += 1
        return sglang

    monkeypatch.setattr(
        _workload_envs, "ensure_vllm_patched_for_tracelens", _vllm_stub,
    )
    monkeypatch.setattr(
        _workload_envs, "ensure_sglang_patched_for_tracelens", _sglang_stub,
    )
    return counts


def test_materialize_profile_vllm_injects_tracelens_flags_when_patched(
    tmp_path, monkeypatch,
):
    """Patcher returns True for vLLM ⇒ EXTRA_VLLM_ARGS gains
    capture_torch_profiler_dir + detailed_trace_annotation, on top of
    the §1 delay/max iterations."""
    import yaml
    _clear_workload_env(monkeypatch)
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 5888" in extra, extra
    assert "--profiler-config.max_iterations 512" in extra, extra
    assert "--profiler-config.capture_torch_profiler_dir " in extra, extra
    assert "--profiler-config.detailed_trace_annotation True" in extra, extra
    # Per-framework dispatch: the SGLang patcher must NOT be invoked
    # when the YAML's framework is vLLM (saves an unnecessary file
    # probe + lock acquisition).
    assert counts == {"vllm": 1, "sglang": 0}, counts


def test_materialize_profile_vllm_omits_tracelens_flags_when_patch_fails(
    tmp_path, monkeypatch,
):
    """Patcher returns False (unpatchable image) ⇒ EXTRA_VLLM_ARGS
    keeps only the §1 safe set. Otherwise unpatched vLLM would
    crash on `unknown JSON key`."""
    import yaml
    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 5888" in extra, extra
    assert "capture_torch_profiler_dir" not in extra, extra
    assert "detailed_trace_annotation" not in extra, extra


def test_materialize_profile_sglang_injects_shape_discovery_when_patched(
    tmp_path, monkeypatch,
):
    """Patcher returns True for SGLang ⇒ EXTRA_SGLANG_ARGS gains
    --enable-shape-discovery-for-cuda-graph-profile."""
    import yaml
    _clear_workload_env(monkeypatch)
    counts = _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS", "",
    )
    assert "--enable-shape-discovery-for-cuda-graph-profile" in extra, extra
    # Per-framework dispatch in reverse: the vLLM patcher must NOT be
    # invoked when the YAML's framework is SGLang.
    assert counts == {"vllm": 0, "sglang": 1}, counts


def test_materialize_profile_sglang_omits_shape_discovery_when_patch_fails(
    tmp_path, monkeypatch,
):
    """Patcher returns False ⇒ no shape-discovery flag (otherwise
    SGLang argparse errors on the unknown flag)."""
    import yaml
    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS", "",
    )
    assert "shape-discovery" not in extra, extra


def test_materialize_profile_kill_switch_skips_patcher_entirely(
    tmp_path, monkeypatch,
):
    """HYPERLOOM_ENABLE_PATCH=0 must short-circuit the patcher call
    entirely — neither vLLM nor SGLang patcher should be touched, and
    no TraceLens-only flags should land in the YAML. This is the
    escape hatch for users with custom forks / read-only filesystems
    / compliance requirements."""
    import yaml
    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=True)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    # Safe §1 flags still present.
    assert "--profiler-config.delay_iterations 5888" in extra, extra
    # TraceLens-only flags absent.
    assert "capture_torch_profiler_dir" not in extra, extra
    assert "detailed_trace_annotation" not in extra, extra
    # Patchers never invoked.
    assert counts == {"vllm": 0, "sglang": 0}, counts


def test_materialize_profile_kill_switch_default_is_on(
    tmp_path, monkeypatch,
):
    """Unset HYPERLOOM_ENABLE_PATCH == default-on. The patcher must
    be invoked so users on TraceLens-patched images get the enhanced
    flags without any opt-in step. Symmetric to the kill-switch test
    above."""
    import yaml
    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    _materialize_config_with_envs(src, tmp_path)
    assert counts["vllm"] == 1, counts


def test_materialize_profile_sglang_does_not_duplicate_shape_discovery(
    tmp_path, monkeypatch,
):
    """If EXTRA_SGLANG_ARGS already mentions
    --enable-shape-discovery-for-cuda-graph-profile (e.g. user
    pre-populated it in the YAML or via env), the materializer must
    NOT append a duplicate copy."""
    import yaml
    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(
        tmp_path, "sglang",
        {
            "CONC": 32, "ISL": 256, "OSL": 1024,
            "EXTRA_SGLANG_ARGS": (
                "--enable-profile-cuda-graph "
                "--enable-shape-discovery-for-cuda-graph-profile"
            ),
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert extra.count("--enable-shape-discovery-for-cuda-graph-profile") == 1, extra


def test_profile_executor_calls_benchmark_lib_patcher():
    """ProfileExecutor must call ensure_benchmark_lib_patched before
    launching Magpie, or the steady-state NUM_PROMPTS we just
    computed gets stomped by upstream `benchmark_lib.sh` and the
    trace is silently empty.

    We don't run the full subprocess machinery — that's gated by
    BaselineExecutor.__call__ which already has its own coverage.
    Here we just lock down the seam: the symbol is imported by name
    in profile.py and invoked unconditionally inside __call__.
    """
    import inference_optimizer.orchestrator.action_executors.profile as profile_mod
    # The symbol must be re-exportable from the module (so monkey-
    # patching in tests / integration sites is straightforward).
    assert profile_mod.ensure_benchmark_lib_patched is not None
    # And the source of __call__ must reference it; this is a cheap
    # regression guard against silent removal during refactors.
    import inspect
    src = inspect.getsource(profile_mod.ProfileExecutor.__call__)
    assert "ensure_benchmark_lib_patched" in src, (
        "ProfileExecutor.__call__ must invoke ensure_benchmark_lib_patched "
        "before super().__call__ — otherwise issue #194 §2 regresses."
    )


# ===========================================================================
# Regression: $FRAMEWORK env switches the default yaml between sglang/vllm
# without anyone passing config_path explicitly. Locks down the entry-layer
# fix for vLLM support — the optimizer used to be sglang-only because all 5
# executors hardcoded baseline_sglang.yaml.
# ===========================================================================
def test_default_baseline_config_resolves_sglang_by_default(monkeypatch):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_baseline_config_resolves_vllm_when_env_set(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_baseline_config().name == "baseline_vllm.yaml"


def test_default_baseline_config_falls_back_on_unknown_value(monkeypatch):
    """Unknown $FRAMEWORK is treated as sglang (matches CLI default).
    The CLI fail-fasts on unknown values, but if a user shell has a stale
    or weird FRAMEWORK env, we should not blow up — sglang is the safe
    default."""
    monkeypatch.setenv("FRAMEWORK", "tensorrt")
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_profile_config_tracks_framework(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_profile_config().name == "profile_vllm.yaml"
    monkeypatch.setenv("FRAMEWORK", "sglang")
    assert _default_profile_config().name == "profile_sglang.yaml"


def test_baseline_executor_picks_framework_yaml_at_call_time(tmp_path, monkeypatch):
    """No config_path override + FRAMEWORK=vllm => baseline_vllm.yaml is
    the resolved default, NOT baseline_sglang.yaml. This is the very
    regression that was blocking vllm users."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = BaselineExecutor()
    # Default constructor leaves default_config_path=None so the resolver
    # is consulted at call time.
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "baseline_vllm.yaml"


def test_profile_executor_picks_framework_yaml_at_call_time(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = ProfileExecutor()
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "profile_vllm.yaml"


@pytest.mark.asyncio
async def test_baseline_executor_keeps_valid_measurement_with_wrapper_failure(tmp_path):
    """A cleanup/profile wrapper failure must not discard completed requests."""
    db = SqliteConnection(tmp_path / "baseline.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    output_dir = tmp_path / "out"
    workspace = output_dir / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": False,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 1.8,
            "output_throughput": 1872.0,
            "total_token_throughput": 3744.0,
            "completed_requests": 320,
            "duration_seconds": 177.0,
        },
        "latency": {"ttft": {"mean_ms": 140}, "e2el": {"mean_ms": 2500}},
    }))

    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="cleanup failed",
    )

    task = await tr.create(
        kind="baseline",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="baseline-valid-warning",
    )
    sub.register_executor("baseline", BaselineExecutor(session_dir=tmp_path))
    with patch("subprocess.run", return_value=fake_completed):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result["status"] == "succeeded"
    assert res.result["reported_success"] is False
    assert res.result["output_throughput"] == 1872.0
    assert res.result["completed_requests"] == 320
    assert "benchmark_report_success_false" in res.result["nonfatal_warnings"]
    assert "magpie_nonzero_after_valid_measurement" in res.result["nonfatal_warnings"]
    db.close()


@pytest.mark.asyncio
async def test_coordinator_promotes_valid_baseline_even_with_failed_status(session_dir):
    c = Coordinator(session_dir, backends=_backends_silent())
    payload = {
        "status": "failed",
        "output_throughput": 1855.76,
        "completed_requests": 320,
        "workspace": "/tmp/baseline",
        "materialized_config": "/tmp/baseline/config.yaml",
    }
    assert c._is_promotable_result("baseline", payload)

    await c._promote_to_shared_state("baseline", payload)

    assert c.shared_state.baseline_tput == pytest.approx(1855.76)
    assert c.shared_state.current_best["tput"] == pytest.approx(1855.76)
    assert c.shared_state.baseline_config_path == "/tmp/baseline/config.yaml"


@pytest.mark.asyncio
async def test_profile_executor_extracts_trace_dir(tmp_path):
    """When the workspace contains torch_trace/*.trace.json.gz, the
    runner surfaces them in the result so downstream consumers can
    feed them into tracelens_analysis.py."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    # Build a fake workspace dir matching what Magpie would create.
    output_dir = tmp_path / "out"
    workspace = output_dir / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 3.2, "output_throughput": 800.0,
            "total_token_throughput": 1600.0, "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158},
                    "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
    }))
    trace_dir = workspace / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "TP-0_main.trace.json.gz").write_bytes(b"fake-trace")
    (trace_dir / "TP-0_aux.trace.json.gz").write_bytes(b"fake-trace")

    # Stub subprocess.run so we don't actually launch sglang.
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr="",
    )
    def _fake_run(*args, **kwargs):
        return fake_completed

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-1",
    )
    sub.register_executor("profile", pe)
    with patch("subprocess.run", side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["framework"] == "sglang"
    assert res.result["trace_dir"] == str(trace_dir)
    assert len(res.result["trace_files"]) == 2
    assert "main_trace_path" in res.result
    db.close()


# ===========================================================================
# kernel_request_handlers — direct unit
# ===========================================================================
@pytest.mark.asyncio
async def test_select_kernels_handler_dry_run_returns_structured_result(session_dir):
    """Tracelens tool always emits structured JSON (even on validation
    failure). Our handler must surface it verbatim — including ``status``
    + run_id + session_id — so callers can debug without parsing logs."""
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    payload = {
        "trace_input": str(fake_trace),
        "session_id": session_dir.name,
        "model_name": "Qwen3-8B",
        "framework": "sglang",
        "top_k": 5,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.select_kernels_handler(payload, session_dir=session_dir)
    # The tool will return failed because the dir has no trace files,
    # but the response must be structured (not generic returncode-only).
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "tool" in res or "run_id" in res or "error" in res
    assert res.get("session_id") == session_dir.name or "run_id" in res


@pytest.mark.asyncio
async def test_select_kernels_handler_surfaces_candidates_path(session_dir, monkeypatch):
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "artifact_paths": {
                "kernel_candidates": "/tmp/kernel_candidates.json",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {
            "trace_input": str(session_dir),
            "dry_run": True,
            "roofline_json": "/tmp/roofline.json",
            "capture_folder": "/tmp/capture_traces",
        },
        session_dir=session_dir,
    )
    assert res["candidates_path"] == "/tmp/kernel_candidates.json"
    assert "--roofline-json" in captured["cmd"]
    assert "/tmp/roofline.json" in captured["cmd"]
    assert "--capture-folder" in captured["cmd"]
    assert "/tmp/capture_traces" in captured["cmd"]


@pytest.mark.asyncio
async def test_select_kernels_handler_backfills_workload_context_from_state(
    session_dir, monkeypatch,
):
    """When the payload omits framework/gpu_type/model, the handler must
    fall back to SharedState so tracelens_analysis.py receives the real
    workload context (vllm/MI300X/Qwen3-30B-A3B/inference) instead of
    the script defaults (""/MI355X/default)."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "vllm"
    state.gpu_type = "mi300x"
    state.model_path = "/wekafs/models/Qwen3-30B-A3B"
    state.model_name = "Qwen3-30B-A3B"
    state.save(session_dir)

    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok"}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--framework" in cmd and "vllm" in cmd
    assert "--target-platform" in cmd and "mi300x" in cmd
    assert "--model-name" in cmd and "Qwen3-30B-A3B" in cmd
    assert "--analysis-mode" in cmd and "inference" in cmd


@pytest.mark.asyncio
async def test_select_kernels_handler_surfaces_analysis_report_path(
    session_dir, monkeypatch,
):
    """The handler must forward the TraceLens v0.3 ``analysis.md`` path so
    GEAK / Coordinator can ground their actions on the same final stakeholder
    report Hyperloom parsed for ``hot_kernels`` (PR #155 review, scheme C)."""
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        # Mimic both the explicit field set by tracelens_analysis.py and the
        # backwards-compatible nested location, so a partial-rollout SDK still
        # surfaces the path through the handler.
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "analysis_report_path": "/tmp/runs/abc/tracelens/analysis.md",
            "artifact_paths": {
                "tracelens_agent_report": "/tmp/runs/abc/tracelens/analysis.md",
                "kernel_candidates": "/tmp/runs/abc/kernel_candidates.json",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["analysis_report_path"] == "/tmp/runs/abc/tracelens/analysis.md"


@pytest.mark.asyncio
async def test_select_kernels_handler_falls_back_to_artifact_paths_for_report(
    session_dir, monkeypatch,
):
    """If the underlying tool only surfaces analysis.md inside artifact_paths
    (e.g. an older tracelens_analysis.py build that wasn't updated yet), the
    handler must still hoist it to the top-level ``analysis_report_path`` for
    Coordinator/GEAK consumers."""
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "artifact_paths": {
                "tracelens_agent_report": "/tmp/legacy/tracelens/analysis.md",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["analysis_report_path"] == "/tmp/legacy/tracelens/analysis.md"


@pytest.mark.asyncio
async def test_select_kernels_handler_missing_trace_input(session_dir):
    res = await krh.select_kernels_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "trace_input" in res["error"]


@pytest.mark.asyncio
async def test_select_kernels_handler_requires_kernel_agent_root(session_dir, monkeypatch):
    monkeypatch.setattr(krh, "HYPERLOOM_KERNEL_AGENT_ROOT", None)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir)},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert res["error_class"] == "kernel_agent_root_missing"
    assert "HYPERLOOM_KERNEL_AGENT_ROOT is not set" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_missing_kernel_id(session_dir):
    res = await krh.run_optimization_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "kernel_id" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_dry_run(session_dir):
    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.run_optimization_handler(payload, session_dir=session_dir)
    assert res.get("status") in ("ok", "succeeded", "failed")  # dry-run may still fail validation


@pytest.mark.asyncio
async def test_run_optimization_handler_forwards_extra_sglang_args(session_dir):
    captured: dict[str, object] = {}

    async def fake_run(cmd, *, timeout_sec):
        captured["cmd"] = cmd
        captured["timeout_sec"] = timeout_sec
        return 0, '{"status": "ok"}', ""

    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "source_file": "/sgl-workspace/sglang/python/sglang/fake.py",
        "extra_sglang_args": "--kv-cache-dtype fp8 --page-size 16",
        "dry_run": True,
        "_single_kernel": True,
    }
    with patch.object(krh, "_validate_reusable_native_kernel", return_value=None), \
         patch.object(krh, "_run_subprocess", side_effect=fake_run):
        res = await krh.run_optimization_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--extra-sglang-args" in cmd
    assert cmd[cmd.index("--extra-sglang-args") + 1] == "--kv-cache-dtype fp8 --page-size 16"


def test_run_optimization_handler_backfills_target_platform_from_state(session_dir):
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.gpu_type = "mi325x"
    state.save(session_dir)
    captured: dict[str, object] = {}

    async def fake_run(cmd, *, timeout_sec):
        captured["cmd"] = cmd
        return 0, '{"status": "ok"}', ""

    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "source_file": "/sgl-workspace/sglang/python/sglang/fake.py",
        "dry_run": True,
        "_single_kernel": True,
    }
    with patch.object(krh, "_validate_reusable_native_kernel", return_value=None), \
         patch.object(krh, "_run_subprocess", side_effect=fake_run):
        res = asyncio.run(
            krh.run_optimization_handler(payload, session_dir=session_dir),
        )

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--target-platform" in cmd
    assert cmd[cmd.index("--target-platform") + 1] == "mi325x"


def test_handlers_dispatch_table():
    """P2-2 only registered select_kernels + run_optimization. P2-4
    added apply_patch + integrate (covered in test_p2_4_integrate_report)."""
    assert krh.has_handler("select_kernels")
    assert krh.has_handler("run_optimization")
    assert not krh.has_handler("totally_unknown_kind")


# ===========================================================================
# PR-B §1: _batch_kernel_candidates collapses task_group members
# ===========================================================================
def _write_candidates_json(tmp_path, payload):
    p = tmp_path / "kernel_candidates.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_batch_kernel_candidates_collapses_task_group_to_primary(tmp_path):
    """Two reusable kernels in the same task_group must dispatch as ONE
    candidate (the primary), with the full group attached for
    build_prompt to render multi-row benchmark cases."""
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rms_norm_prefill",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
                "duration_us": 100.0,
            },
            {
                "kernel_id": "k002", "name": "rms_norm_decode",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
                "duration_us": 50.0,
            },
            {
                "kernel_id": "k003", "name": "other_kernel",
                "source_file": "/sgl-workspace/aiter/other.py",
                "reusable_native_kernel": True,
                "duration_us": 30.0,
            },
        ],
        "task_groups": [
            {
                "task_group_id": "tg001",
                "function_name": "rms_norm",
                "source_path": "/sgl-workspace/aiter/rmsnorm.py",
                "definition_line": 10,
                "primary_kernel_id": "k001",
                "kernel_ids": ["k001", "k002"],
                "rows": [
                    {"kernel_id": "k001", "name": "rms_norm_prefill"},
                    {"kernel_id": "k002", "name": "rms_norm_decode"},
                ],
                "aggregate_duration_us": 150.0,
            },
        ],
    })
    selected = krh._batch_kernel_candidates({"candidates_path": str(candidates_path)})
    # k001 (primary) + k003 (ungrouped) = 2 dispatches, not 3.
    kernel_ids = [c.get("kernel_id") for c in selected]
    assert kernel_ids == ["k001", "k003"]
    # The primary carries the full group dict so build_prompt can render
    # both rows as benchmark cases.
    assert selected[0]["task_group"]["task_group_id"] == "tg001"
    assert set(selected[0]["task_group"]["kernel_ids"]) == {"k001", "k002"}
    # The ungrouped kernel has no task_group attached.
    assert "task_group" not in selected[1]


def test_batch_kernel_candidates_falls_back_when_primary_is_non_reusable(tmp_path):
    """If the group's primary_kernel_id was rejected by classify_patchability
    (e.g. vendor BLAS name marker landed on the heaviest row), dispatch
    falls back to the first reusable member instead of dropping the
    whole group."""
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rocblas_sgemm_call",
                "source_file": "/sgl-workspace/aiter/foo.py",
                "reusable_native_kernel": False,  # primary rejected
                "duration_us": 200.0,
            },
            {
                "kernel_id": "k002", "name": "rms_norm_call",
                "source_file": "/sgl-workspace/aiter/foo.py",
                "reusable_native_kernel": True,
                "duration_us": 50.0,
            },
        ],
        "task_groups": [
            {
                "task_group_id": "tg001",
                "function_name": "foo",
                "primary_kernel_id": "k001",
                "kernel_ids": ["k001", "k002"],
                "rows": [
                    {"kernel_id": "k001"},
                    {"kernel_id": "k002"},
                ],
            },
        ],
    })
    selected = krh._batch_kernel_candidates({"candidates_path": str(candidates_path)})
    # k002 (the only reusable member) replaces the rejected primary.
    assert [c["kernel_id"] for c in selected] == ["k002"]
    assert selected[0]["task_group"]["task_group_id"] == "tg001"


def test_batch_kernel_candidates_legacy_path_unchanged_without_task_groups(tmp_path):
    """When kernel_candidates.json has no task_groups[] (older runs,
    raw-trace fallback, LLama70B fixture path), the candidate list is
    byte-identical to pre-PR-B behaviour."""
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rms_norm",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
            },
            {
                "kernel_id": "k002", "name": "vendor",
                "source_file": "/sgl-workspace/aiter/vendor.py",
                "reusable_native_kernel": False,
            },
        ],
    })
    selected = krh._batch_kernel_candidates({"candidates_path": str(candidates_path)})
    assert [c["kernel_id"] for c in selected] == ["k001"]
    assert "task_group" not in selected[0]


# ===========================================================================
# Coordinator — REQUEST programmatic handler integration
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_request_select_kernels_uses_handler(session_dir):
    """When Orchestration emits REQUEST{kind=select_kernels}, the Coordinator
    should run the registered handler programmatically and emit RESPONSE
    on the bus *without* waiting for the Kernel LLM."""
    c = Coordinator(session_dir, backends=_backends_silent())

    captured: dict = {}

    async def fake_handler(payload, *, session_dir):
        captured["payload"] = payload
        captured["session_dir"] = session_dir
        return {"status": "ok", "hot_kernels": ["kernel_a", "kernel_b"]}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "select_kernels",
                    "params": {"trace_input": "/tmp/fake-trace.json.gz"},
                },
            ))
            req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
            assert req_msgs, "request must be mirrored to kernel inbox"
            req_id = req_msgs[0].msg_id

            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs, "handler must emit RESPONSE without LLM"
            r = resp_msgs[0]
            assert r.from_agent == "kernel"
            assert r.payload["kind"] == "select_kernels_done"
            assert r.payload["status"] == "ok"
            assert r.payload["result"]["hot_kernels"] == ["kernel_a", "kernel_b"]
            assert r.payload["in_reply_to"] == req_id
            assert r.payload["source"] == "programmatic_handler"

            # And the handler did receive merged payload (params flattened in).
            assert captured["payload"].get("trace_input") == "/tmp/fake-trace.json.gz"
            assert captured["session_dir"] == session_dir
        finally:
            await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_unknown_kind_routes_to_llm(session_dir):
    """REQUEST whose kind has no handler is mirrored to kernel inbox
    (LLM responder path) — no auto-RESPONSE."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={
                "target_agent": "kernel",
                "kind": "invent_brand_new_kind",  # NOT in registry
            },
        ))
        req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
        assert req_msgs, "request must be mirrored even when no handler"
        # No auto-response should have been emitted.
        resp_msgs = await c.bus.tail(topic="response")
        assert not resp_msgs
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_handler_exception_recorded(session_dir):
    """Handler crashes → RESPONSE.status='failed' + error_class set."""
    c = Coordinator(session_dir, backends=_backends_silent())

    async def bad_handler(payload, *, session_dir):
        raise RuntimeError("boom")

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": bad_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel", "kind": "select_kernels"},
            ))
            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs
            r = resp_msgs[0]
            assert r.payload["status"] == "failed"
            assert r.payload["result"]["error_class"] == "handler_exception"
            assert "boom" in r.payload["result"]["error"]
        finally:
            await c.stop()
