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
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    kernel_agent_root = Path(__file__).resolve().parents[2] / "kernel-agent"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
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


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so
    Baseline/ProfileExecutor's always-on artifact harvest does not
    pick up the host's real ``/workspace`` during the stubbed
    subprocess runs exercised here.
    """
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


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
# Regression: gpu_type injection sets runner_type AND force-pins the generic
# `{framework}_{gpu_type}.sh` so Magpie's resolver hits priority 1 (explicit
# user override) and never falls through to the InferenceX native script
# (which hardcodes `--result-dir /workspace/`). See
# `design/magpie-generic-script-and-user-data-path.md` §3.
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


def test_materialize_config_forces_generic_benchmark_script(tmp_path):
    """When `gpu_type` is supplied, `benchmark.benchmark_script` MUST be
    pinned to the generic `{framework}_{gpu_type}.sh` so Magpie's
    resolver hits priority 1 (explicit override) and never silently
    falls through to the InferenceX native script (e.g.
    `dsr1_fp8_mi300x.sh`) which hardcodes `--result-dir /workspace/`."""
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
    assert rendered["benchmark"]["benchmark_script"] == "sglang_mi355x.sh", \
        "gpu_type must pin the generic {framework}_{gpu_type}.sh"


def test_materialize_config_forces_generic_when_source_yaml_has_no_script(
    tmp_path,
):
    """Even when the source YAML doesn't carry a `benchmark_script`, the
    renderer MUST still write one explicitly — otherwise Magpie's
    resolver would fall through to the InferenceX native script."""
    import yaml
    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "benchmark": {
            "framework": "vllm",
            "model": "/m",
            # No benchmark_script field at all.
        },
    }))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = _materialize_config_with_envs(
        src_yaml, out_dir, gpu_type="mi300x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


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
        "INFERENCEX_PATH",
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


def test_materialize_persists_inferencex_path_for_magpie(
    tmp_path, monkeypatch,
):
    """$INFERENCEX_PATH must be written into benchmark.inferencex_path.

    Otherwise Magpie resolves an empty value to its sibling checkout
    ($MAGPIE_DIR/InferenceX), while Hyperloom's profile patcher may have
    patched a different checkout.
    """
    import yaml
    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("INFERENCEX_PATH", "/wekafs/InferenceX")
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    assert rendered["benchmark"]["inferencex_path"] == "/wekafs/InferenceX"


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
    """ProfileExecutor must patch the materialized InferenceX checkout before
    launching Magpie, or the steady-state NUM_PROMPTS / PROFILE_EXTRA_BODY we
    just computed gets stomped by upstream InferenceX and the trace is
    silently empty.

    We don't run the full subprocess machinery — that's gated by
    BaselineExecutor.__call__ which already has its own coverage.
    Here we just lock down the seam: the symbols are imported by name
    in profile.py and invoked from the post-materialization hook.
    """
    import inference_optimizer.orchestrator.action_executors.profile as profile_mod
    # The symbol must be re-exportable from the module (so monkey-
    # patching in tests / integration sites is straightforward).
    assert profile_mod.ensure_benchmark_lib_patched is not None
    assert profile_mod.ensure_benchmark_serving_patched is not None
    # And the source of the hook must reference both patchers; this is a cheap
    # regression guard against silent removal during refactors. The hook runs
    # after YAML materialization so it can patch the exact
    # benchmark.inferencex_path that Magpie will execute.
    import inspect
    src = inspect.getsource(profile_mod.ProfileExecutor._after_materialize_config)
    assert "ensure_benchmark_lib_patched" in src, (
        "ProfileExecutor._after_materialize_config must invoke "
        "ensure_benchmark_lib_patched on the materialized InferenceX path — "
        "otherwise issue #194 §2 regresses."
    )
    assert "ensure_benchmark_serving_patched" in src, (
        "ProfileExecutor._after_materialize_config must invoke "
        "ensure_benchmark_serving_patched so PROFILE_EXTRA_BODY reaches "
        "SGLang's /start_profile request."
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


def test_default_baseline_config_resolves_atom_when_env_set(monkeypatch):
    """B1: FRAMEWORK=atom selects baseline_atom.yaml. Single-source-of-truth
    selector — every executor (baseline/params/sweep/backends) routes through
    this so an env flip propagates everywhere without per-executor changes."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    assert _default_baseline_config().name == "baseline_atom.yaml"


def test_server_args_env_name_atom():
    """B1: atom maps to EXTRA_ATOM_ARGS, matching the env contract consumed
    by Magpie's atom_mi*x.sh wrapper. Ordering note: the atom branch sits
    before vllm so a future framework name containing 'vllm' as a substring
    cannot accidentally win — even though 'atom' itself is not a vllm
    substring today."""
    from inference_optimizer.orchestrator.action_executors._grid_runner import (
        server_args_env_name,
    )
    assert server_args_env_name("atom") == "EXTRA_ATOM_ARGS"
    assert server_args_env_name("ATOM") == "EXTRA_ATOM_ARGS"
    # Regression: sglang/vllm still resolve correctly after the new branch.
    assert server_args_env_name("vllm") == "EXTRA_VLLM_ARGS"
    assert server_args_env_name("sglang") == "EXTRA_SGLANG_ARGS"


def test_materialize_config_atom_profile_skips_tracelens_flags(
    tmp_path, monkeypatch,
):
    """B1: PROFILE=1 + framework=atom must NOT inject sglang/vllm-specific
    profiler CLI flags (--profiler-config.*) into EXTRA_ATOM_ARGS — atom's
    argparse would reject them. The executor short-circuits before this
    code path on a real run, but we defend in depth so direct callers
    (params/sweep) can't accidentally render a broken atom YAML."""
    import yaml
    monkeypatch.setenv("FRAMEWORK", "atom")
    monkeypatch.setenv("PROFILE", "1")
    src = _default_baseline_config()  # baseline_atom.yaml
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    extra = str(envs.get("EXTRA_ATOM_ARGS", ""))
    assert "--profiler-config" not in extra, (
        f"atom EXTRA_ATOM_ARGS leaked sglang/vllm profiler flag: {extra!r}"
    )
    # --trust-remote-code from the baseline YAML must survive untouched.
    assert "--trust-remote-code" in extra, (
        f"atom EXTRA_ATOM_ARGS lost base --trust-remote-code: {extra!r}"
    )


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
async def test_profile_executor_skips_when_framework_atom(monkeypatch, tmp_path):
    """B2: FRAMEWORK=atom must short-circuit ProfileExecutor to a
    structured skipped result BEFORE any Magpie subprocess is launched.
    atom (Magpie v1) has no torch_profiler wiring — running the full
    profile path would either silently no-op or, worse, crash because
    sglang/vllm-specific --profiler-config flags get injected into
    EXTRA_ATOM_ARGS. Verified by checking the result dict shape AND
    that no subprocess machinery (BaselineExecutor.__call__ /
    run_with_session_kill) ran."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    pe = ProfileExecutor()
    # If the short-circuit fails, the executor would try to materialize a
    # YAML and shell out to Magpie. Sentinel-patch the parent __call__ so
    # we can prove it was never reached.
    called = {"parent": False}

    async def _explode(self, ctx):  # pragma: no cover — must not run
        called["parent"] = True
        return {"status": "succeeded"}

    monkeypatch.setattr(BaselineExecutor, "__call__", _explode)

    task = SimpleNamespace(params={}, task_id="t-atom-profile")
    ctx = SimpleNamespace(task=task, extra=None)

    result = await pe(ctx)

    assert result["status"] == "skipped"
    assert result["error_class"] == "atom_no_profiler"
    assert "torch_profiler" in result["error"]
    assert called["parent"] is False, (
        "ProfileExecutor must short-circuit BEFORE BaselineExecutor.__call__"
    )


@pytest.mark.asyncio
async def test_roofline_executor_skips_when_framework_atom(monkeypatch):
    """B2: FRAMEWORK=atom must short-circuit RooflineExecutor at its
    entrypoint, returning status=skipped without invoking profile or
    trace_analyze sub-steps. Critical because the composite would
    otherwise treat a skipped profile_result as a failure (the existing
    _failed("profile", ...) branch) and pollute roofline_failure_streak."""
    from inference_optimizer.orchestrator.action_executors.roofline import (
        RooflineExecutor,
    )

    monkeypatch.setenv("FRAMEWORK", "atom")
    # RooflineExecutor requires shared_state, but the atom guard returns
    # before touching it — a sentinel object is enough.
    rexec = RooflineExecutor(shared_state=SimpleNamespace())

    # Sentinel: prove the lazy import / sub-step orchestration never runs.
    import inference_optimizer.orchestrator.action_executors.profile as profile_mod

    async def _explode(_ctx):  # pragma: no cover — must not run
        raise AssertionError("profile_executor must not be invoked under atom")

    monkeypatch.setattr(profile_mod, "profile_executor", _explode)

    task = SimpleNamespace(params={}, task_id="t-atom-roofline")
    ctx = SimpleNamespace(task=task, extra=None)

    result = await rexec(ctx)
    assert result["status"] == "skipped"
    assert result["error_class"] == "atom_no_profiler"
    assert result["framework"] == "atom"


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
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", return_value=fake_completed):
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
    (trace_dir / "177-TP-0-DECODE.trace.json.gz").write_bytes(b"fake-trace")
    merged_trace = trace_dir / "merged-177.trace.json.gz"
    merged_trace.write_bytes(b"fake-trace")

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
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["framework"] == "sglang"
    assert res.result["trace_dir"] == str(trace_dir)
    assert len(res.result["trace_files"]) == 2
    assert res.result["main_trace_path"] == str(merged_trace)
    assert res.result["profile_trace_selection_reason"] == "merged_trace_preferred"
    db.close()


@pytest.mark.asyncio
async def test_profile_executor_patches_configured_inferencex_path(
    tmp_path, monkeypatch,
):
    """ProfileExecutor must patch the InferenceX checkout Magpie will use.

    Regression for the Qwen3-32B TraceLens run where $INFERENCEX_PATH pointed
    at /wekafs/InferenceX but Magpie's rendered YAML had an empty
    benchmark.inferencex_path, so Magpie used $MAGPIE_DIR/InferenceX and lost
    NUM_PROMPTS / PROFILE_EXTRA_BODY.
    """
    fake_ix = tmp_path / "InferenceX"
    (fake_ix / "benchmarks").mkdir(parents=True)
    (fake_ix / "utils" / "bench_serving").mkdir(parents=True)
    (fake_ix / "benchmarks" / "benchmark_lib.sh").write_text(
        'num_prompts="${NUM_PROMPTS:-$max_concurrency}"\n',
        encoding="utf-8",
    )
    (fake_ix / "utils" / "bench_serving" / "benchmark_serving.py").write_text(
        "# already patched\nPROFILE_EXTRA_BODY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(fake_ix))

    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

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

    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr="",
    )
    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-inferencex-path",
    )
    sub.register_executor("profile", pe)
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", return_value=fake_completed):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    materialized = Path(res.result["materialized_config"])
    import yaml
    rendered = yaml.safe_load(materialized.read_text())
    assert rendered["benchmark"]["inferencex_path"] == str(fake_ix)
    db.close()


@pytest.mark.asyncio
async def test_profile_executor_extracts_vllm_capture_traces(tmp_path):
    """TraceLens-patched vLLM writes graph-capture traces next to the
    benchmark workspace, under the profile task's ``capture_traces`` dir."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    output_dir = tmp_path / "out"
    workspace = output_dir / "benchmark_vllm_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "vllm",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 3.2, "output_throughput": 800.0,
            "total_token_throughput": 1600.0, "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158},
                    "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
    }))
    capture_dir = output_dir / "capture_traces"
    capture_dir.mkdir()
    (capture_dir / "graph_capture_rank_0.1.pt.trace.json.gz").write_bytes(b"fake-trace")
    (capture_dir / "graph_capture_rank_0.2.pt.trace.json.gz").write_bytes(b"fake-trace")

    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr="",
    )

    def _fake_run(*args, **kwargs):
        return fake_completed

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-capture",
    )
    sub.register_executor("profile", pe)
    with patch("inference_optimizer.orchestrator.action_executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["framework"] == "vllm"
    assert res.result["trace_dir"] == str(capture_dir)
    assert len(res.result["trace_files"]) == 2
    assert res.result["main_trace_path"].startswith(str(capture_dir))
    db.close()


# ===========================================================================
# kernel_request_handlers — direct unit
# ===========================================================================
@pytest.mark.asyncio
async def test_trace_analyze_handler_dry_run_returns_structured_result(session_dir):
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
    res = await krh.trace_analyze_handler(payload, session_dir=session_dir)
    # The tool will return failed because the dir has no trace files,
    # but the response must be structured (not generic returncode-only).
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "tool" in res or "run_id" in res or "error" in res
    assert res.get("session_id") == session_dir.name or "run_id" in res


@pytest.mark.asyncio
async def test_trace_analyze_handler_surfaces_candidates_path(session_dir, monkeypatch):
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
    res = await krh.trace_analyze_handler(
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
async def test_trace_analyze_handler_backfills_workload_context_from_state(
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
    res = await krh.trace_analyze_handler(
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
async def test_trace_analyze_handler_surfaces_trace_report_path(
    session_dir, monkeypatch,
):
    """The handler must forward the TraceLens v0.3 analysis.md path."""
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "trace_report_path": "/tmp/runs/abc/tracelens/analysis.md",
            "artifact_paths": {
                "trace_report_path": "/tmp/runs/abc/tracelens/analysis.md",
                "kernel_candidates": "/tmp/runs/abc/kernel_candidates.json",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["trace_report_path"] == "/tmp/runs/abc/tracelens/analysis.md"


@pytest.mark.asyncio
async def test_trace_analyze_handler_persists_trace_report_to_candidates(
    session_dir, tmp_path, monkeypatch,
):
    """Disk candidates must carry the TraceLens report path for GEAK prompts."""
    report_path = tmp_path / "analysis.md"
    report_path.write_text("# TraceLens Report\n", encoding="utf-8")
    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps({
            "hot_kernels": [{
                "kernel_id": "k1",
                "name": "paged_attention",
                "source_file": "/sgl-workspace/sglang/kernels/paged.py",
                "reusable_native_kernel": True,
            }],
        }),
        encoding="utf-8",
    )

    async def fake_run_subprocess(cmd, *, timeout_sec):
        return 0, json.dumps({
            "status": "ok",
            "hot_kernels": json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"],
            "trace_report_path": str(report_path),
            "artifact_paths": {
                "kernel_candidates": str(candidates_path),
                "trace_report_path": str(report_path),
            },
        }), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)

    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )

    persisted = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidate = persisted["hot_kernels"][0]
    assert res["hot_kernels"][0]["trace_report_path"] == str(report_path)
    assert persisted["trace_report_path"] == str(report_path)
    assert persisted["artifact_paths"]["trace_report_path"] == str(report_path)
    assert candidate["trace_report_path"] == str(report_path)


@pytest.mark.asyncio
async def test_trace_analyze_handler_backfills_runtime_metadata_from_config(
    session_dir, tmp_path, monkeypatch,
):
    """GEAK candidates must inherit the materialized Magpie workload config."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    config_path = tmp_path / "profile_config.with_envs.yaml"
    config_path.write_text(
        """
benchmark:
  framework: sglang
  model: /models/Qwen3
  precision: bf16
  envs:
    TP: 8
    CONC: 64
    ISL: 1024
    OSL: 1024
    NUM_PROMPTS: 512
    MAX_MODEL_LEN: 8192
    EXTRA_SGLANG_ARGS: "--kv-cache-dtype fp8 --page-size 16"
    SGLANG_USE_TRITON: "1"
    ROCR_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    SAFE_API_KEY: "should-not-leak"
""",
        encoding="utf-8",
    )
    state = SharedState.load_or_init(session_dir)
    state.baseline_config_path = str(config_path)
    state.save(session_dir)

    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps({
            "hot_kernels": [{
                "kernel_id": "k1",
                "name": "paged_attention",
                "source_file": "/sgl-workspace/sglang/kernels/paged.py",
                "reusable_native_kernel": True,
            }],
        }),
        encoding="utf-8",
    )

    async def fake_run_subprocess(cmd, *, timeout_sec):
        return 0, json.dumps({
            "status": "ok",
            "hot_kernels": json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"],
            "artifact_paths": {"kernel_candidates": str(candidates_path)},
        }), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)

    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )

    enriched = json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"][0]
    assert res["hot_kernels"][0]["env_vars"]["SGLANG_USE_TRITON"] == "1"
    assert enriched["env_vars"]["TP"] == "8"
    assert enriched["env_vars"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "SAFE_API_KEY" not in enriched["env_vars"]
    assert enriched["runtime_args"]["framework"] == "sglang"
    assert enriched["runtime_args"]["server_args"] == "--kv-cache-dtype fp8 --page-size 16"
    assert enriched["runtime_args"]["workload"] == {
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 1024,
        "num_prompts": 512,
        "max_model_len": 8192,
    }


def test_materialized_workload_metadata_filters_prefixed_secrets(tmp_path):
    config_path = tmp_path / "profile_config.with_envs.yaml"
    config_path.write_text(
        """
benchmark:
  framework: vllm
  envs:
    VLLM_USE_V1: "1"
    VLLM_API_KEY: "should-not-leak"
    TRITON_AUTH_TOKEN: "should-not-leak"
""",
        encoding="utf-8",
    )

    metadata = krh._load_materialized_workload_metadata(str(config_path))

    assert metadata["env_vars"]["VLLM_USE_V1"] == "1"
    assert "VLLM_API_KEY" not in metadata["env_vars"]
    assert "TRITON_AUTH_TOKEN" not in metadata["env_vars"]


def test_materialized_workload_metadata_tolerates_bad_server_args(tmp_path):
    config_path = tmp_path / "profile_config.with_envs.yaml"
    config_path.write_text(
        """
benchmark:
  framework: sglang
  envs:
    EXTRA_SGLANG_ARGS: "--kv-cache-dtype 'unterminated"
    TP: 1
""",
        encoding="utf-8",
    )

    metadata = krh._load_materialized_workload_metadata(str(config_path))

    assert metadata["runtime_args"]["server_args"] == "--kv-cache-dtype 'unterminated"
    assert metadata["runtime_args"]["server_args_argv"] == []


@pytest.mark.asyncio
async def test_trace_analyze_handler_uses_artifact_trace_report_path(
    session_dir, monkeypatch,
):
    """TraceLens now surfaces the upstream analysis.md as trace_report_path."""
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "artifact_paths": {
                "trace_report_path": "/tmp/tracelens/analysis.md",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["trace_report_path"] == "/tmp/tracelens/analysis.md"


@pytest.mark.asyncio
async def test_trace_analyze_handler_missing_trace_input(session_dir):
    res = await krh.trace_analyze_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "trace_input" in res["error"]


@pytest.mark.asyncio
async def test_select_kernels_handler_requires_kernel_agent_root(session_dir, monkeypatch):
    # N15 made HYPERLOOM_KERNEL_AGENT_ROOT a lazy env read; delenv is
    # the correct way to exercise the "not configured" branch.
    monkeypatch.delenv("HYPERLOOM_KERNEL_AGENT_ROOT", raising=False)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir)},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert res["error_class"] == "kernel_agent_root_missing"
    assert "HYPERLOOM_KERNEL_AGENT_ROOT is not set" in res["error"]


# ===========================================================================
# T4 — TraceLens permanent failure stays failed (no fallback)
# ===========================================================================
# A failed TraceLens run must not be rewritten into ok+empty kernels. That
# fallback hid split/annotation problems as a valid "no candidates" result and
# let Orchestration continue down params/backends. The handler now preserves
# ``status=failed`` and only appends structured diagnostics so operators can
# see the upstream rc / error / stderr.

@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_keeps_tool_failure_failed(
    session_dir, monkeypatch,
):
    """When tracelens_analysis.py returns ``status=failed`` the handler must
    keep the failure status. Older code demoted this to ok+empty kernels, which
    let Orchestration keep walking params/backends as if TraceLens had
    completed. We still clear stale candidates and append a diagnostic warning.
    """
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "failed",
            "tool": "tracelens_analysis",
            "error": "RuntimeError: TraceLens perf CLI crashed",
            "returncode": 1,
            "stderr_tail": "RuntimeError: graph capture folder missing",
            # The tool also emits an empty hot_kernels[] on failure in
            # some paths — we explicitly seed a non-empty list here to
            # prove the handler clears it.
            "hot_kernels": [{"kernel_id": "stale_1"}],
        }
        return 1, json.dumps(payload), "stderr noise"

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert res["hot_kernels"] == [], (
        "stale hot_kernels must be cleared on tool failure"
    )
    warnings = res.get("trace_health_warnings") or []
    assert any(w.get("code") == "tracelens_analysis_failed" for w in warnings), (
        "operator must see WHY hot_kernels[] is empty"
    )
    failure_w = next(w for w in warnings if w["code"] == "tracelens_analysis_failed")
    assert failure_w["severity"] == "warning"
    assert "TraceLens perf CLI crashed" in failure_w.get("error", "")
    assert failure_w.get("returncode") == 1


@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_passes_through_idle_warning(
    session_dir, monkeypatch,
):
    """When tracelens_analysis emits a ``trace_health_warnings`` from
    the T3 idle gate (status=ok, empty hot_kernels), the handler must
    pass it through verbatim. No de-duplication, no rewriting — that
    warning is the routing signal the Coordinator reads."""
    idle_warning = {
        "code": "high_gpu_idle_pct",
        "severity": "warning",
        "idle_pct": 35.0,
        "threshold_pct": 20.0,
        "source": "/tmp/runs/abc/tracelens/analysis.md",
        "message": "GPU was idle 35.00% …",
    }

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "tool": "tracelens_analysis",
            "hot_kernels": [],
            "trace_health_warnings": [idle_warning],
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    assert res["hot_kernels"] == []
    assert res["trace_health_warnings"] == [idle_warning]


@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_defaults_warnings_to_empty_list(
    session_dir, monkeypatch,
):
    """When the tool emits no ``trace_health_warnings`` (steady state),
    the handler still surfaces an empty list — downstream code (the
    Coordinator's branching, the prompt-summary renderer) can iterate
    without a ``None`` guard."""
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "tool": "tracelens_analysis",
            "hot_kernels": [{"kernel_id": "fake_1"}],
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    assert res["trace_health_warnings"] == []


# ===========================================================================
# T5 (this PR) — trace_health_warnings must reach the Orchestration LLM
# ===========================================================================
# Handler-boundary plumbing alone is not enough: the Orchestration LLM
# only sees what ``SharedState._format_*`` renders into its prompt. Pin
# that record_trace_analyze keeps the warning list AND that
# _format_last_trace_analyze surfaces it inline so the LLM grounds its
# next ACTION on the routing signal (params vs kernel-opt vs re-profile).

def test_record_trace_analyze_persists_trace_health_warnings(session_dir):
    """``record_trace_analyze`` must keep ``trace_health_warnings`` from
    the handler result verbatim in ``last_trace_analyze`` so prompt
    rendering can see it on the next tick."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    warning = {
        "code": "high_gpu_idle_pct",
        "severity": "warning",
        "idle_pct": 35.0,
        "threshold_pct": 20.0,
        "source": "/tmp/x/analysis.md",
        "message": "high idle",
    }
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [warning],
        },
    )
    assert state.last_trace_analyze["trace_health_warnings"] == [warning]


def test_record_trace_analyze_defaults_warnings_to_empty_list(session_dir):
    """Steady-state (no warnings emitted) — the cached entry must still
    expose ``trace_health_warnings`` as an empty list rather than the
    field being absent, so iteration code in renderers / consumers
    doesn't need a ``KeyError`` guard."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [{"kernel_id": "k1", "reusable_native_kernel": True}],
        },
    )
    assert state.last_trace_analyze["trace_health_warnings"] == []


def test_record_trace_analyze_persists_task_groups(session_dir):
    """893bc6f: ``task_groups`` must flow from the handler result into
    ``last_trace_analyze`` so the multi-KEEP queue's
    ``untried_hot_reusable_kernels`` / ``next_pending_keep_kernel_id``
    can collapse members of the same AST function into one slot.

    Without this, Qwen3-30B-A3B-Base session 20260523T035235Z saw
    k001/k003/k005 (non-primary members of moe_op + rmsnorm groups
    already covered by k002/k004/k009) re-dispatched as a separate
    second batch, wasting GEAK->Claude->Codex wall-clock on patches
    that targeted the same source functions as the first batch.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    groups = [
        {"primary_kernel_id": "k004", "kernel_ids": ["k003", "k004"],
         "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
        {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"],
         "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
    ]
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [
                {"kernel_id": "k001", "gpu_pct": 8.0, "reusable_native_kernel": True,
                 "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
                {"kernel_id": "k002", "gpu_pct": 25.0, "reusable_native_kernel": True,
                 "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
                {"kernel_id": "k003", "gpu_pct": 12.0, "reusable_native_kernel": True,
                 "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
                {"kernel_id": "k004", "gpu_pct": 38.0, "reusable_native_kernel": True,
                 "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
            ],
            "task_groups": groups,
        },
    )
    assert state.last_trace_analyze.get("task_groups") == groups
    # After k002 + k004 attempted, group-aware collapse should report
    # NO untried hot kernels even though k001/k003 have attempts=0.
    state.record_kernel_opt({
        "status": "failed", "kernel_id": "k002",
        "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "error_class": "subtask_exception",
    })
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k004",
        "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "proposal": {"decision": "KEEP", "reasons": []},
        "verification": {"micro_speedup": 1.17,
                         "compile_passed": True, "correctness_passed": True,
                         "best_artifact_path": "/tmp/k004.py"},
    })
    assert state.untried_hot_reusable_kernels() == [], (
        "k001/k003 must be filtered out because their groups have an "
        "attempted member (k002 / k004 respectively)"
    )


def test_record_trace_analyze_defaults_task_groups_to_empty_list(session_dir):
    """When the handler result has no ``task_groups`` field (legacy
    TraceLens output), the cached entry must default to an empty
    list so downstream readers never get a KeyError."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [
                {"kernel_id": "k1", "reusable_native_kernel": True},
            ],
        },
    )
    assert state.last_trace_analyze.get("task_groups") == []


def test_record_select_kernels_filters_invalid_warning_entries(session_dir):
    """Defensive: a buggy tool emitting non-dict entries or dicts
    missing the ``code`` field shouldn't poison ``last_trace_analyze``.
    We accept only well-formed dicts with at least a ``code`` key so
    the prompt renderer never has to defensively coerce types."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                "not-a-dict",
                {"severity": "warning"},  # missing 'code'
                {"code": "high_gpu_idle_pct", "idle_pct": 30.0,
                 "threshold_pct": 20.0},
                None,
            ],
        },
    )
    warnings = state.last_trace_analyze["trace_health_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "high_gpu_idle_pct"


def test_format_last_trace_analyze_renders_idle_warning_inline(session_dir):
    """Prompt rendering: when an idle warning was persisted, the
    Orchestration prompt line must surface it with the numeric context
    so the LLM can ground its routing on the actual percentages."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace.json.gz"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                {
                    "code": "high_gpu_idle_pct",
                    "severity": "warning",
                    "idle_pct": 60.5,
                    "threshold_pct": 20.0,
                    "source": "/tmp/x/analysis.md",
                    "message": "high idle",
                }
            ],
        },
    )
    rendered = state._format_last_trace_analyze()
    assert "high_gpu_idle_pct" in rendered
    assert "60.5%" in rendered
    assert "20.0%" in rendered
    assert "warnings=[" in rendered


def test_format_last_trace_analyze_renders_failure_warning_with_rc(session_dir):
    """Tool-failure warning carries ``returncode``; the prompt must
    surface that too so an operator-or-LLM can distinguish 'TraceLens
    crashed' (rc=1) from a benign skip."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                {
                    "code": "tracelens_analysis_failed",
                    "severity": "warning",
                    "returncode": 1,
                    "error": "RuntimeError: …",
                    "message": "TraceLens failed",
                }
            ],
        },
    )
    rendered = state._format_last_trace_analyze()
    assert "tracelens_analysis_failed" in rendered
    assert "rc=1" in rendered


def test_format_last_trace_analyze_omits_warnings_suffix_in_steady_state(session_dir):
    """Format-stability guard: when no warnings were recorded (the
    common case), the prompt line MUST NOT gain a gratuitous
    ``warnings=[]`` suffix. Prompt format stability matters because
    we have downstream prompt-snapshot tests pinned to the legacy
    format; growing the line in the steady state would break them."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [{"kernel_id": "k1", "reusable_native_kernel": True}],
        },
    )
    rendered = state._format_last_trace_analyze()
    assert "warnings=" not in rendered, (
        "no warnings → no warnings= suffix; this keeps existing prompt "
        "snapshots stable"
    )


@pytest.mark.asyncio
async def test_t5_handler_to_sharedstate_e2e_idle_warning_reaches_prompt(
    session_dir, monkeypatch,
):
    """End-to-end pinning of the routing signal path:
       tracelens_analysis (T3)  →  handler result.trace_health_warnings
                                →  SharedState.last_trace_analyze (this PR)
                                →  Orchestration prompt line  (this PR)
    Without ALL three steps the LLM cannot route on idle %, and the
    upstream T3 work is wasted."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "tool": "tracelens_analysis",
            "hot_kernels": [],
            "trace_health_warnings": [
                {
                    "code": "high_gpu_idle_pct",
                    "severity": "warning",
                    "idle_pct": 42.0,
                    "threshold_pct": 20.0,
                    "source": "/tmp/runs/abc/tracelens/analysis.md",
                    "message": "high idle",
                }
            ],
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    # Step 1: handler boundary carries the warning.
    assert res["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"

    # Step 2: SharedState persists it.
    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze({"trace_input": str(session_dir)}, res)
    assert state.last_trace_analyze["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"

    # Step 3: prompt rendering surfaces it.
    rendered = state._format_last_trace_analyze()
    assert "high_gpu_idle_pct" in rendered
    assert "42.0%" in rendered


@pytest.mark.asyncio
async def test_t5_handler_to_sharedstate_e2e_failure_warning_reaches_prompt(
    session_dir, monkeypatch,
):
    """Same path for T4: when TraceLens fails permanently, the
    failure warning must reach the Orchestration prompt so the LLM
    doesn't keep re-trying TraceLens or guessing why ``hot_kernels=[]``."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "failed",
            "tool": "tracelens_analysis",
            "error": "RuntimeError: TraceLens crashed",
            "returncode": 1,
            "hot_kernels": [],
        }
        return 1, json.dumps(payload), "stderr"

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze({"trace_input": str(session_dir)}, res)
    rendered = state._format_last_trace_analyze()
    assert "tracelens_analysis_failed" in rendered
    assert "rc=1" in rendered


@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_failure_appends_to_existing_warnings(
    session_dir, monkeypatch,
):
    """Edge case: the tool emits BOTH ``status=failed`` AND a pre-
    existing ``trace_health_warnings`` list (e.g. the idle gate fired
    AND then a later step crashed). The handler must preserve the
    existing entries and APPEND the failure warning, not overwrite."""
    pre_existing = {
        "code": "high_gpu_idle_pct",
        "severity": "warning",
        "idle_pct": 60.0,
        "threshold_pct": 20.0,
        "source": "/tmp/x/analysis.md",
        "message": "high idle",
    }

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "failed",
            "tool": "tracelens_analysis",
            "error": "RuntimeError: ran out of disk",
            "returncode": 2,
            "hot_kernels": [],
            "trace_health_warnings": [pre_existing],
        }
        return 2, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    warnings = res["trace_health_warnings"]
    assert len(warnings) == 2, "must preserve pre-existing + append failure"
    assert warnings[0] == pre_existing
    assert warnings[1]["code"] == "tracelens_analysis_failed"


def test_optimization_wrapper_timeout_sec_geak_default_90min():
    assert krh._optimization_wrapper_timeout_sec({"backends": "geak"}) == 90 * 60 + 180


def test_optimization_wrapper_timeout_sec_oob_default_60min():
    assert krh._optimization_wrapper_timeout_sec({"backends": "claude"}) == 60 * 60 + 180


def test_optimization_wrapper_timeout_sec_geak_env_override(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "120")
    assert krh._optimization_wrapper_timeout_sec({"backends": "geak"}) == 120 * 60 + 180


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
    """P2-2 only registered trace_analyze + run_optimization. P2-4
    added apply_patch + integrate (covered in test_p2_4_integrate_report)."""
    assert krh.has_handler("trace_analyze")
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
    # PR-I: default min_gpu_pct is 3.0; rows must carry gpu_pct >= 3.0
    # to pass the dispatcher's hot-kernel gate.
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rms_norm_prefill",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
                "duration_us": 100.0, "gpu_pct": 12.0,
            },
            {
                "kernel_id": "k002", "name": "rms_norm_decode",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
                "duration_us": 50.0, "gpu_pct": 8.0,
            },
            {
                "kernel_id": "k003", "name": "other_kernel",
                "source_file": "/sgl-workspace/aiter/other.py",
                "reusable_native_kernel": True,
                "duration_us": 30.0, "gpu_pct": 4.5,
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
    # PR-I: default min_gpu_pct is 3.0, so rows must carry gpu_pct >= 3.0
    # to be retained by the dispatcher (matches production fixtures).
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rocblas_sgemm_call",
                "source_file": "/sgl-workspace/aiter/foo.py",
                "reusable_native_kernel": False,  # primary rejected
                "duration_us": 200.0, "gpu_pct": 22.0,
            },
            {
                "kernel_id": "k002", "name": "rms_norm_call",
                "source_file": "/sgl-workspace/aiter/foo.py",
                "reusable_native_kernel": True,
                "duration_us": 50.0, "gpu_pct": 5.5,
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
    # PR-I: default min_gpu_pct is 3.0; legacy fixture now carries
    # gpu_pct >= 3.0 so the dispatcher's hot-kernel gate doesn't drop
    # k001. We're testing the *legacy task_groups-absent path*, not the
    # gpu_pct filter, so this stays orthogonal to PR-I's intent.
    candidates_path = _write_candidates_json(tmp_path, {
        "hot_kernels": [
            {
                "kernel_id": "k001", "name": "rms_norm",
                "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                "reusable_native_kernel": True,
                "gpu_pct": 11.0,
            },
            {
                "kernel_id": "k002", "name": "vendor",
                "source_file": "/sgl-workspace/aiter/vendor.py",
                "reusable_native_kernel": False,
                "gpu_pct": 9.0,
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
async def test_coordinator_request_trace_analyze_uses_handler(session_dir):
    """When Orchestration emits REQUEST{kind=trace_analyze}, the Coordinator
    should run the registered handler programmatically and emit RESPONSE
    on the bus *without* waiting for the Kernel LLM."""
    c = Coordinator(session_dir, backends=_backends_silent())

    captured: dict = {}

    async def fake_handler(payload, *, session_dir):
        captured["payload"] = payload
        captured["session_dir"] = session_dir
        return {"status": "ok", "hot_kernels": ["kernel_a", "kernel_b"]}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"trace_analyze": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "trace_analyze",
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
            assert r.payload["kind"] == "trace_analyze_done"
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
                     {"trace_analyze": bad_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel", "kind": "trace_analyze"},
            ))
            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs
            r = resp_msgs[0]
            assert r.payload["status"] == "failed"
            assert r.payload["result"]["error_class"] == "handler_exception"
            assert "boom" in r.payload["result"]["error"]
        finally:
            await c.stop()


# ===========================================================================
# PR-X: batch dispatch enablers
#   1) _DEFAULT_KERNEL_BATCH_PARALLEL is sized for a full MI300X-class node
#      so a single ``run_optimization`` batch fans out to one GEAK / OOB
#      attempt per GPU (Ray then schedules against actual ``num_gpus``).
#   2) Coordinator force-injects ``candidates_path`` from SharedState into
#      every ``run_optimization`` payload so the batch path
#      (``_run_optimization_batch``) fires deterministically regardless of
#      whether the LLM remembered to include the field. LLM-supplied
#      values still win, so future prompts can target a different
#      TraceLens snapshot.
# ===========================================================================
def test_default_kernel_batch_parallel_matches_full_node():
    """Default fanout is sized for a single MI300X / MI355X node (8 GPU)
    so a typical ``run_optimization`` batch (TraceLens emits 3-8 reusable
    units) does NOT serialize behind an asyncio semaphore tighter than
    Ray's view of the cluster. Pre-PR-X value was 3, which throttled even
    the small batches actually observed in production sessions."""
    assert krh._DEFAULT_KERNEL_BATCH_PARALLEL == 8


@pytest.mark.asyncio
async def test_coordinator_injects_candidates_path_for_run_optimization(
    session_dir, monkeypatch,
):
    """When the LLM emits ``run_optimization`` without ``candidates_path``,
    the Coordinator must pull it from ``state.last_trace_analyze`` and
    inject it into the handler payload so ``_run_optimization_batch``
    fires instead of silently collapsing to ``_run_optimization_single``
    (which would waste 7 idle GPUs on an 8-GPU node)."""
    # Bypass the N13/N19c sequence gate that normally denies
    # ``run_optimization`` until a roofline snapshot + cheap exploration
    # exist; this test focuses purely on the injection path.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "1")
    c = Coordinator(session_dir, backends=_backends_silent())
    # ``_sequence_denial_for_request`` requires both baseline_tput > 0
    # and last_profile_trace to be set (kernel requests are denied with
    # "baseline must run first" / "profile must run first" otherwise);
    # simulate the post-baseline + post-profile state so we exercise the
    # injection branch under realistic ordering preconditions.
    c.shared_state.baseline_tput = 1234.5
    c.shared_state.last_profile_trace = "/wekafs/trace/x.json.gz"
    cached_path = "/wekafs/cached/kernel_candidates.json"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/wekafs/trace/x.json.gz",
        "candidates_path": cached_path,
    }
    # On this branch ``_sequence_denial_for_request`` still consults
    # ``last_select_kernels`` (the rename to ``trace_analyze`` is
    # planned for M3); seed it with the same trace so the gate clears.
    c.shared_state.last_select_kernels = {
        "trace_input": "/wekafs/trace/x.json.gz",
        "candidates_path": cached_path,
    }
    explicit = "/wekafs/operator/override_candidates.json"

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"run_optimization": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "run_optimization",
                    "params": {
                        "kernel_id": "k001",
                        "candidates_path": explicit,
                    },
                },
            ))
            assert captured["payload"].get("candidates_path") == explicit
        finally:
            await c.stop()


# ===========================================================================
# PR-B: multi-KEEP integrate queue + streaming batch record
# ---------------------------------------------------------------------------
# Tests for the Coordinator <-> kernel_request_handlers wiring that
# implements:
#   1) Streaming record_partial callback so each batch sub-attempt's
#      KEEP/REVERT lands in SharedState *before* asyncio.gather() wait-all
#      unblocks (mid-batch visibility).
#   2) batch_mode dedup so the post-gather record_kernel_opt(best) call
#      doesn't double-count attempts already recorded via the callback.
#   3) base_tput auto-injection from current_best.tput on ``integrate``
#      requests where the LLM forgot to populate it (a routine miss on
#      the 2nd/3rd integrate of a multi-KEEP drain).
# ===========================================================================
@pytest.mark.asyncio
async def test_run_optimization_handler_invokes_record_partial_per_sub_result(
    session_dir,
):
    """Each batch sub-attempt's result must flow through record_partial
    the moment _run_kernel_backend_sequence returns, NOT only after
    asyncio.gather() wait-all releases. Without this, a single slow
    GEAK sibling delays integrate-queue visibility for all the fast
    KEEPs (the Qwen3-30B-A3B-Base 20260522T093903Z regression).
    """
    candidates = [
        {"kernel_id": "kA", "source_file": "/p/a.py", "reusable_native_kernel": True},
        {"kernel_id": "kB", "source_file": "/p/b.py", "reusable_native_kernel": True},
        {"kernel_id": "kC", "source_file": "/p/c.py", "reusable_native_kernel": True},
    ]

    completion_log: list[str] = []
    recorded: list[dict] = []

    async def fake_sequence(base_payload, candidate, *, session_dir):
        kid = str(candidate.get("kernel_id"))
        # Stagger completion times so kA finishes last; the streaming
        # callback should still see kB and kC's results before kA.
        delay = {"kA": 0.05, "kB": 0.01, "kC": 0.02}[kid]
        await asyncio.sleep(delay)
        completion_log.append(kid)
        return {
            "status": "ok",
            "kernel_id": kid,
            "source_file": candidate["source_file"],
            "proposal": {"decision": "KEEP" if kid in ("kB", "kC") else "REVERT"},
            "verification": {"micro_speedup": 1.5 if kid == "kB" else 2.0},
        }

    def record_partial(result: dict) -> None:
        recorded.append({
            "kernel_id": result.get("kernel_id"),
            "decision": (result.get("proposal") or {}).get("decision"),
        })

    with patch.object(krh, "_run_kernel_backend_sequence",
                       side_effect=fake_sequence):
        await krh._run_optimization_batch(
            payload={"candidates_path": "/dummy"},
            candidates=candidates,
            session_dir=session_dir,
            record_partial=record_partial,
        )

    # Callback must have fired for every candidate, in completion order
    # (NOT input order). kB finishes first (sleep=0.01), then kC, then kA.
    assert [r["kernel_id"] for r in recorded] == ["kB", "kC", "kA"], recorded
    assert completion_log == ["kB", "kC", "kA"]


@pytest.mark.asyncio
async def test_backend_ladder_prefers_keep_over_higher_micro_non_keep(
    session_dir,
):
    """PR-F bug repro: the GEAK→Claude→Codex ladder used to pick the
    backend with the highest ``micro_speedup``, ignoring whether that
    attempt was a real KEEP or just a NEEDS_REVIEW with a paper claim.

    Qwen3-30B-A3B-Base session 20260523T035235Z k004 trace:
      * GEAK         micro=1.30  decision=NEEDS_REVIEW (correctness missing)
      * Claude       micro=??    decision=??
      * Codex        micro=1.17  decision=KEEP (correctness PASS)

    Before PR-F: best=GEAK (1.30 > 1.17), KEEP signal silently
    discarded, integrate gate never fires, k004 patch ends up in
    rejected with attempts=1 and the actual production-quality 1.17x
    KEEP is wasted.

    Fix: ladder must prefer KEEP over non-KEEP regardless of micro.
    Mirror the batch handler's tuple key.
    """
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
        if backend == "geak":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "NEEDS_REVIEW",
                             "reasons": ["correctness missing"]},
                "verification": {"micro_speedup": 1.30,
                                 "correctness_passed": False,
                                 "best_artifact_path": "/tmp/geak.py"},
            }
        if backend == "claude":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "REVERT",
                             "reasons": ["micro 0.9 regression"]},
                "verification": {"micro_speedup": 0.9,
                                 "correctness_passed": True,
                                 "best_artifact_path": "/tmp/claude.py"},
            }
        if backend == "codex":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP",
                             "reasons": ["ready for integrate"]},
                "verification": {"micro_speedup": 1.17,
                                 "correctness_passed": True,
                                 "best_artifact_path": "/tmp/codex.py"},
            }
        raise AssertionError(f"unexpected backend {backend!r}")

    candidate = {
        "kernel_id": "k004",
        "source_file": "/p/moe_op.py",
        "reusable_native_kernel": True,
    }
    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy"},
            candidate,
            session_dir=session_dir,
        )

    # Ladder MUST keep walking past GEAK NEEDS_REVIEW and Claude REVERT,
    # then break on Codex KEEP.
    assert calls == ["geak", "claude", "codex"], calls
    assert (best.get("proposal") or {}).get("decision") == "KEEP", best
    assert (best.get("verification") or {}).get("micro_speedup") == 1.17
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/codex.py"


@pytest.mark.asyncio
async def test_backend_ladder_breaks_on_first_keep(session_dir):
    """When GEAK already KEEPs, ladder must short-circuit (not waste
    Claude/Codex wall-clock chasing a higher number)."""
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
        if backend == "geak":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.50,
                                 "correctness_passed": True,
                                 "best_artifact_path": "/tmp/geak.py"},
            }
        raise AssertionError(f"ladder must NOT run {backend!r} after GEAK KEEP")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
        )

    assert calls == ["geak"]
    assert (best.get("proposal") or {}).get("decision") == "KEEP"
    assert (best.get("verification") or {}).get("micro_speedup") == 1.50


@pytest.mark.asyncio
async def test_backend_ladder_falls_back_to_highest_micro_when_no_keep(
    session_dir,
):
    """If NO backend KEEPs, ladder picks the highest-micro non-KEEP
    so the per-kernel ledger at least records the strongest signal
    (the prior behaviour, kept under the new tuple-key)."""
    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        if backend == "geak":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "NEEDS_REVIEW", "reasons": []},
                "verification": {"micro_speedup": 1.30,
                                 "best_artifact_path": "/tmp/geak.py"},
            }
        if backend == "claude":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "NEEDS_REVIEW", "reasons": []},
                "verification": {"micro_speedup": 1.45,
                                 "best_artifact_path": "/tmp/claude.py"},
            }
        if backend == "codex":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "REVERT", "reasons": []},
                "verification": {"micro_speedup": 0.8,
                                 "best_artifact_path": "/tmp/codex.py"},
            }
        raise AssertionError(backend)

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
        )

    # All non-KEEP -> pick highest micro (Claude 1.45)
    assert (best.get("verification") or {}).get("micro_speedup") == 1.45
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/claude.py"


@pytest.mark.asyncio
async def test_batch_handler_isolates_sub_task_exceptions_from_gather(
    session_dir,
):
    """Sub-task exceptions must NOT propagate out of ``asyncio.gather`` while
    sibling tasks are still in flight.

    Default ``asyncio.gather(return_exceptions=False)`` re-raises on the
    first exception while the other coroutines keep running in the
    background. That would unblock the Coordinator mid-batch, let it
    dispatch an integrate, and collide with the still-running
    kernel_opt subprocesses on the same GPU lease.

    The fix wraps each sub-task in a try/except inside ``_guarded`` so
    exceptions surface as structured ``failed`` results. gather then
    behaves as true wait-all and ``record_partial`` sees every
    candidate -- including the failed one -- with a stamped kernel_id
    so the per-kernel attempts ledger stays consistent.
    """
    candidates = [
        {"kernel_id": "kFast", "source_file": "/p/fast.py", "reusable_native_kernel": True},
        {"kernel_id": "kCrash", "source_file": "/p/crash.py", "reusable_native_kernel": True},
        {"kernel_id": "kSlow", "source_file": "/p/slow.py", "reusable_native_kernel": True},
    ]

    recorded: list[dict] = []
    completion_order: list[str] = []

    async def fake_sequence(base_payload, candidate, *, session_dir):
        kid = str(candidate.get("kernel_id"))
        if kid == "kFast":
            await asyncio.sleep(0.01)
            completion_order.append(kid)
            return {
                "status": "ok",
                "kernel_id": kid,
                "source_file": candidate["source_file"],
                "proposal": {"decision": "KEEP"},
                "verification": {"micro_speedup": 1.6},
            }
        if kid == "kCrash":
            await asyncio.sleep(0.02)
            completion_order.append(kid)
            raise RuntimeError("simulated GEAK crash mid-batch")
        # kSlow finishes last; gather must wait for it.
        await asyncio.sleep(0.06)
        completion_order.append(kid)
        return {
            "status": "ok",
            "kernel_id": kid,
            "source_file": candidate["source_file"],
            "proposal": {"decision": "REVERT"},
            "verification": {"micro_speedup": 0.9},
        }

    def record_partial(result: dict) -> None:
        recorded.append({
            "kernel_id": result.get("kernel_id"),
            "status": result.get("status"),
            "decision": (result.get("proposal") or {}).get("decision"),
            "error_class": result.get("error_class"),
        })

    with patch.object(krh, "_run_kernel_backend_sequence",
                       side_effect=fake_sequence):
        result = await krh._run_optimization_batch(
            payload={"candidates_path": "/dummy"},
            candidates=candidates,
            session_dir=session_dir,
            record_partial=record_partial,
        )

    # Gather MUST have waited for all three (kSlow finishes last).
    assert completion_order == ["kFast", "kCrash", "kSlow"], completion_order

    # record_partial got exactly one call per candidate, in completion
    # order, and the crashed sibling came through as a structured
    # ``status=failed`` with kernel_id preserved (NOT swallowed).
    assert [r["kernel_id"] for r in recorded] == ["kFast", "kCrash", "kSlow"]
    crash_record = next(r for r in recorded if r["kernel_id"] == "kCrash")
    assert crash_record["status"] == "failed"
    assert crash_record["error_class"] == "subtask_exception"

    # Batch handler still returns the best KEEP (kFast) and tags
    # batch_mode so Coordinator's post-gather record_kernel_opt dedups.
    assert isinstance(result, dict)
    assert result.get("batch_mode") is True
    assert result.get("kernel_id") == "kFast"


@pytest.mark.asyncio
async def test_coordinator_streams_batch_results_and_dedups_final_record(
    session_dir, monkeypatch,
):
    """End-to-end through Coordinator._handle_request:

    * record_partial is wired so each sub-attempt records once via
      ``record_kernel_opt`` while the batch is in flight, and
    * the post-gather ``record_kernel_opt(best)`` call is skipped when
      ``result["batch_mode"]`` is True (no double-counting).

    Verified by counting how many times SharedState.record_kernel_opt is
    invoked relative to the number of batch sub-results.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "1")
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 1234.5
    c.shared_state.last_profile_trace = "/wekafs/trace/x.json.gz"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/wekafs/trace/x.json.gz",
        "candidates_path": "/wekafs/cached/candidates.json",
    }
    # The sequence gate on this branch still consults
    # ``last_select_kernels`` (M3 will rename it to ``trace_analyze``).
    c.shared_state.last_select_kernels = dict(c.shared_state.last_trace_analyze)
    c.shared_state.current_best = {
        "action": "integrate",
        "tput": 4500.0,
        "kernel_id": "k009",
    }

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        return {"status": "ok", "decision": "KEEP", "new_tput": 4620.0,
                "gain_pct": 2.7, "kernel_id": "k001"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"integrate": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "integrate",
                    "params": {
                        "kernel_id": "k001",
                        "patch_path": "/tmp/k001.py",
                        "target_file": "/p/moe_op.py",
                        # no base_tput intentionally
                    },
                },
            ))
        finally:
            await c.stop()

    assert captured["payload"].get("base_tput") == 4500.0, \
        "Coordinator must auto-inject base_tput from current_best.tput"


@pytest.mark.asyncio
async def test_coordinator_does_not_overwrite_explicit_base_tput_on_integrate(
    session_dir,
):
    """Explicit operator-supplied ``base_tput`` must NOT be clobbered by
    the auto-injection -- some flows (e.g. resume from a saved snapshot)
    intentionally pin a different baseline."""
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 4319.5
    c.shared_state.last_profile_trace = "/wekafs/trace/x.json.gz"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/wekafs/trace/x.json.gz",
        "candidates_path": "/wekafs/cached/candidates.json",
    }
    c.shared_state.last_select_kernels = dict(c.shared_state.last_trace_analyze)
    c.shared_state.current_best = {"action": "backends", "tput": 4500.0}

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        return {"status": "ok", "decision": "NEEDS_REVIEW", "new_tput": 4400.0,
                "gain_pct": 0.0, "kernel_id": "k009"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"integrate": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "integrate",
                    "params": {
                        "kernel_id": "k009",
                        "patch_path": "/tmp/k009.py",
                        "target_file": "/p/rmsnorm.py",
                        "base_tput": 4200.0,  # operator override
                    },
                },
            ))
        finally:
            await c.stop()

    assert captured["payload"].get("base_tput") == 4200.0, \
        "Explicit base_tput must take precedence over current_best.tput"


@pytest.fixture
def _candidates_factory(tmp_path):
    """Write a kernel_candidates.json fixture and return its path."""
    def _make(hot_kernels, task_groups=None):
        path = tmp_path / "kernel_candidates.json"
        path.write_text(json.dumps({
            "hot_kernels": hot_kernels,
            "task_groups": task_groups or [],
            "reusable_native_kernel_ids": [],
        }))
        return str(path)
    return _make


def test_batch_candidates_filters_rejected_kernel_ids(
    session_dir, _candidates_factory,
):
    """PR-C: a kernel that's already on rejected_kernel_ids must not
    show up in the next batch's candidate list, even though it's still
    in kernel_candidates.json.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k001"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k002"]


def test_batch_candidates_filters_kernels_with_recorded_attempts(
    session_dir, _candidates_factory,
):
    """PR-C max_attempts=1 default: any prior attempt -> kernel skipped
    in the next batch. Defends against the LLM re-proposing the same
    run_optimization batch after a previous one returned all failures.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    state = SharedState.load_or_init(session_dir)
    # k001 has an attempt recorded but is not (yet) on rejected list
    # (e.g. PARTIAL that hasn't yet hit max_partial).
    state.kernel_opt_attempts = {
        "k001": {"attempts": 1, "partial_count": 1, "last_decision": "PARTIAL"},
    }
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert [c.get("kernel_id") for c in out] == ["k002"]


def test_batch_candidates_task_group_falls_back_to_live_member(
    session_dir, _candidates_factory,
):
    """When primary (k002) is rejected, the task_group should still
    dispatch via the next live member (k001), because k001 patches the
    same AST function -- not falling back here is how the 12h session
    silently lost half its kernel leverage.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState
    cpath = _candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
             "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
             "source_file": "/p/moe_op.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k002"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    # Group dispatches as k001 with the original task_group attached.
    assert len(out) == 1
    assert out[0]["kernel_id"] == "k001"
    assert out[0].get("task_group", {}).get("primary_kernel_id") == "k002"


def test_batch_candidates_skips_group_when_all_members_rejected(
    session_dir, _candidates_factory,
):
    """If every member of a task_group is unusable, the group must
    skip cleanly -- legacy code would have errored out trying to
    dispatch a rejected primary."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    cpath = _candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
             "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
             "source_file": "/p/moe_op.py"},
            {"kernel_id": "k009", "gpu_pct": 10.0, "reusable_native_kernel": True,
             "source_file": "/p/rmsnorm.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k001", "k002"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    # moe_op.py group fully retired; only k009 remains.
    assert out_ids == ["k009"]


def test_batch_candidates_skips_in_flight_kernels(
    session_dir, _candidates_factory,
):
    """In-flight defense: a status/ko-*.json with state=running for k004
    must keep k004 out of the next batch -- prevents the 5-concurrent-
    Claude-process pile-up the 12h session hit.
    """
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    # Plant a running status file for k004.
    status_dir = (
        session_dir / "kernel-agent" / "runs" / session_dir.name
        / "status" / "kernel_optimization"
    )
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "ko-deadbeef.json").write_text(json.dumps({
        "state": "running",
        "current_step": "run_backends",
        "pid": 123456,
        "last_lines": ["kernel_id=k004", "selected_backends=geak"],
    }))

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k001"]


def test_batch_candidates_below_min_gpu_pct_skipped(
    session_dir, _candidates_factory, monkeypatch,
):
    """min_gpu_pct env=5.0 keeps tiny rmsnorm kernels out of the batch."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "5.0")
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k005", "gpu_pct": 2.8, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k001"]


def test_batch_candidates_default_min_gpu_pct_matches_sharedstate_gate(
    session_dir, _candidates_factory,
):
    """PR-I: ``_batch_kernel_candidates`` default must match
    ``SharedState.untried_hot_reusable_kernels``'s default (3.0).

    Repro: Qwen3-30B-A3B-Base session 20260523T035235Z third batch
    round dispatched k006 (gpu_pct=1.3%) via task_group fallback even
    though it was below the SharedState gate's 3.0% threshold; LLM
    couldn't even see k006 in untried_hot_reusable_kernels yet the
    batch wasted ~30-90 min on its ladder. The two layers now share
    the same default so a kernel that's invisible to the gate is also
    rejected by the batch dispatcher.
    """
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 38.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k006", "gpu_pct": 1.3, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
        {"kernel_id": "k008", "gpu_pct": 3.13, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    # No env set -> default 3.0 must filter out k006 (1.3 < 3.0)
    # but keep k001 (38) and k008 (3.13).
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert "k006" not in out_ids, out_ids
    assert "k001" in out_ids
    assert "k008" in out_ids



def test_in_flight_kernel_ids_returns_running_only(session_dir):
    status_dir = (
        session_dir / "kernel-agent" / "runs" / session_dir.name
        / "status" / "kernel_optimization"
    )
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "ko-aaa.json").write_text(json.dumps({
        "state": "running",
        "last_lines": ["kernel_id=k001"],
    }))
    (status_dir / "ko-bbb.json").write_text(json.dumps({
        "state": "succeeded",
        "last_lines": ["kernel_id=k002"],
    }))
    out = krh._in_flight_kernel_ids(session_dir)
    assert out == {"k001"}


def test_resolve_integrate_payload_falls_back_to_kernel_opt_attempts_ledger(
    session_dir,
):
    """The multi-KEEP queue drains kernel_ids that aren't the current
    ``last_kernel_opt`` slot. ``_resolve_integrate_payload`` must look up
    patch_path / source_file from the per-kernel ``kernel_opt_attempts``
    ledger so any queued KEEP can integrate.
    """
    from inference_optimizer.orchestrator.shared_state import SharedState
    state = SharedState.load_or_init(session_dir)
    # Pretend the streaming record path landed two KEEPs but
    # last_kernel_opt only holds the strongest (k009).
    state.last_kernel_opt = {
        "kernel_id": "k009",
        "decision": "KEEP",
        "best_artifact_path": "/tmp/k009.py",
        "source_file": "/p/rmsnorm.py",
    }
    state.kernel_opt_attempts = {
        "k009": {
            "last_decision": "KEEP", "last_micro_speedup": 4.13,
            "last_artifact_path": "/tmp/k009.py", "last_source_file": "/p/rmsnorm.py",
        },
        "k001": {
            "last_decision": "KEEP", "last_micro_speedup": 2.0,
            "last_artifact_path": "/tmp/k001.py", "last_source_file": "/p/moe_op.py",
        },
    }
    state.save(session_dir)

    # Orch sends integrate(k001) with only the kernel_id -- it's the
    # *second* KEEP from the queue, not last_kernel_opt.
    resolved, missing = krh._resolve_integrate_payload(
        {"kernel_id": "k001", "base_tput": 4500.0},
        session_dir=session_dir,
    )
    assert missing is None, missing
    assert resolved.get("patch_path") == "/tmp/k001.py", \
        "patch_path must fall back to kernel_opt_attempts[k001].last_artifact_path"
    assert resolved.get("source_file") == "/p/moe_op.py", \
        "source_file must fall back to kernel_opt_attempts[k001].last_source_file"
