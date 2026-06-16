# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    _sanitize_profile_server_args,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    SubAgentRunner,
)
from inference_optimizer.manifest import build_manifest
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# fixtures
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


def test_mi308x_keeps_real_gpu_type_but_uses_mi300x_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setenv("TARGET_GPU_TYPE", "mi308x")
    args = SimpleNamespace(
        model="/models/Qwen3",
        model_class="",
        target_summary="",
        max_hours=1,
        no_kernel=False,
        gpu_type="mi308x",
        target_gain=None,
        target_tput=None,
    )

    assert optimizer_cli._gpu_runner_type("mi308x") == "mi300x"
    manifest = build_manifest(tmp_path, args=args, session_id="mi308x-session")
    state = optimizer_cli._seed_shared_state(
        tmp_path, args, session_id="mi308x-session",
    )

    assert manifest["gpu_type"] == "mi308x"
    assert state.gpu_type == "mi308x"
    assert os.environ["TARGET_GPU_TYPE"] == "mi308x"
    assert os.environ["GPU_TYPE"] == "mi300x"


def test_cli_parser_accepts_mi308x():
    parser = optimizer_cli._build_parser()
    args = parser.parse_args([
        "optimize", "--model", "/tmp/model", "--gpu-type", "mi308x",
    ])
    assert args.gpu_type == "mi308x"


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so the artifact harvest doesn't pick up the host's ``/workspace``."""
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# ProfileExecutor
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


# Regression: model_path injection beats the YAML's hardcoded fallback (else Magpie silently benchmarks the shipped Qwen3-8B).
def test_materialize_config_injects_model_path(tmp_path):
    """Default YAML's hardcoded Qwen3-8B must be overridden when caller passes ``model_path``."""
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
    """When no model_path is passed, the materialized YAML keeps the source model field."""
    import yaml
    # Clear ISL/OSL/MAX_MODEL_LEN env so they don't inject
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "PRECISION"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(PROFILE_DEFAULT_CONFIG, tmp_path)
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert "Qwen" in rendered["benchmark"]["model"]


def test_materialize_config_injects_model_with_other_overrides(tmp_path):
    """model_path + extra_envs should both land in the materialized YAML."""
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


# Regression: gpu_type injection sets runner_type AND force-pins the generic `{framework}_{gpu_type}.sh` (Magpie priority 1). See `design/magpie-generic-script-and-user-data-path.md` §3.
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
    """`gpu_type` pins `benchmark_script` to the generic `{framework}_{gpu_type}.sh` (Magpie priority 1)."""
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
    """Even with no source `benchmark_script`, the renderer must write one explicitly."""
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


# Regression: TP / CONC env override yaml hardcode (DSR1-0528 deadlooped when TP=8 env was silently ignored).
def test_materialize_config_tp_env_overrides_yaml_hardcode(tmp_path, monkeypatch):
    """TP env var must override yaml hardcode (was 1, becomes 8)."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
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
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
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
    """When explicit ROCR_VISIBLE_DEVICES has fewer devices than TP requires, `_workload_envs` auto-expands to 0..TP-1."""
    import yaml
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
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


# Regression #194: steady-state window must follow the TraceLens magpie skill formulas (max_iters/delay_iters), not the old placeholders.
def _profile_yaml(tmp_path, framework: str, envs: dict) -> Path:
    """Synthesize a minimal profile YAML the materializer recognises as PROFILE=1 + torch_profiler.enabled=True."""
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
    """$INFERENCEX_PATH must be written into benchmark.inferencex_path so Magpie uses the patched checkout."""
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
    """Skill: max_iters floors at 256 (OSL=256, CONC=64 ⇒ 16*OSL/CONC=64, so the floor kicks in)."""
    import yaml
    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 64, "ISL": 256, "OSL": 256})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 256" in extra, extra


# Regression #194 §2: NUM_PROMPTS must be sized to cover the steady-state window (profile mode force-overrides any caller-supplied value).
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
    """Tiny OSL with skill floor max_iters=256 still produces a sane NUM_PROMPTS."""
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
    """The §2 override is profile-only; baseline / sweep paths keep the seq_cost-based NUM_PROMPTS."""
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


# Regression #194 §4 / §5: when the runtime patcher succeeds, materialize auto-appends the patched-build profiler flags; when it fails-soft, none are injected. HYPERLOOM_ENABLE_PATCH=0 short-circuits the patcher.
def _mock_patchers(monkeypatch, *, vllm: bool, sglang: bool) -> dict[str, int]:
    """Replace the two patcher symbols on `_workload_envs` with stubs that record invocation counts for per-framework dispatch asserts."""
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
    """Patcher True for vLLM ⇒ EXTRA_VLLM_ARGS gains capture_torch_profiler_dir + detailed_trace_annotation on top of §1 iterations."""
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
    # Per-framework dispatch: the SGLang patcher must NOT run for a vLLM YAML.
    assert counts == {"vllm": 1, "sglang": 0}, counts


def test_materialize_profile_vllm_omits_tracelens_flags_when_patch_fails(
    tmp_path, monkeypatch,
):
    """Patcher False ⇒ EXTRA_VLLM_ARGS keeps only the §1 safe set (else unpatched vLLM crashes on unknown JSON key)."""
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
    """HYPERLOOM_ENABLE_PATCH=0 short-circuits the patcher entirely; no TraceLens-only flags land in the YAML."""
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
    """Unset HYPERLOOM_ENABLE_PATCH == default-on; the patcher must be invoked."""
    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    _materialize_config_with_envs(src, tmp_path)
    assert counts["vllm"] == 1, counts


def test_materialize_profile_sglang_does_not_duplicate_shape_discovery(
    tmp_path, monkeypatch,
):
    """If EXTRA_SGLANG_ARGS already has --enable-shape-discovery-for-cuda-graph-profile, the materializer must NOT duplicate it."""
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


def _profile_yaml_model(tmp_path, framework: str, model: str, envs: dict) -> Path:
    """Like _profile_yaml but with an explicit model path (for Gemma2 gating)."""
    import yaml as _yaml
    src = tmp_path / f"src_{framework}_model.yaml"
    src.write_text(_yaml.safe_dump({
        "benchmark": {
            "framework": framework,
            "model": model,
            "envs": {"PROFILE": "1", **envs},
            "profiler": {"torch_profiler": {"enabled": True}},
        },
    }))
    return src


def test_materialize_profile_sglang_skips_shape_discovery_for_gemma2(
    tmp_path, monkeypatch,
):
    """Gemma2 + patched SGLang must NOT inject shape-discovery (it crashes
    CUDA-graph capture); --enable-profile-cuda-graph still applies."""
    import yaml
    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "gemma2_model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"],
    }), encoding="utf-8")
    src = _profile_yaml_model(
        tmp_path, "sglang", str(model), {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "shape-discovery" not in envs.get("EXTRA_SGLANG_ARGS", ""), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is False


def test_materialize_profile_sglang_keeps_shape_discovery_for_non_gemma2(
    tmp_path, monkeypatch,
):
    """A non-Gemma2 model still gets shape-discovery when patched."""
    import yaml
    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "llama_model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
    }), encoding="utf-8")
    src = _profile_yaml_model(
        tmp_path, "sglang", str(model), {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS", "",
    )
    assert "--enable-shape-discovery-for-cuda-graph-profile" in extra, extra


def test_profile_executor_calls_benchmark_lib_patcher():
    """ProfileExecutor must patch the materialized InferenceX checkout before launching Magpie (else the computed profile window is stomped and the trace is empty)."""
    import inference_optimizer.orchestrator.action_executors.profile as profile_mod
    # The symbols must be re-exportable for monkey-patching.
    assert profile_mod.ensure_benchmark_lib_patched is not None
    assert profile_mod.ensure_benchmark_serving_patched is not None
    # The hook source must reference both patchers (regression guard against silent removal).
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


def test_profile_server_args_sanitizer_drops_torch_compile_flags():
    raw = (
        "--enable-torch-compile --torch-compile-max-bs 32 "
        "--quantization fp8 --foo=bar --torch-compile-max-bs=64"
    )

    sanitized = _sanitize_profile_server_args(raw)

    assert "--enable-torch-compile" not in sanitized
    assert "--torch-compile-max-bs" not in sanitized
    assert "--quantization fp8" in sanitized
    assert "--foo=bar" in sanitized


# Regression: $FRAMEWORK env switches the default yaml between sglang/vllm without an explicit config_path (entry-layer fix for vLLM support).
def test_default_baseline_config_resolves_sglang_by_default(monkeypatch):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_baseline_config_resolves_vllm_when_env_set(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_baseline_config().name == "baseline_vllm.yaml"


def test_default_baseline_config_falls_back_on_unknown_value(monkeypatch):
    """Unknown $FRAMEWORK falls back to sglang (the safe default)."""
    monkeypatch.setenv("FRAMEWORK", "tensorrt")
    assert _default_baseline_config().name == "baseline_sglang.yaml"


def test_default_baseline_config_resolves_atom_when_env_set(monkeypatch):
    """B1: FRAMEWORK=atom selects baseline_atom.yaml (the single-source-of-truth selector for every executor)."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    assert _default_baseline_config().name == "baseline_atom.yaml"


def test_server_args_env_name_atom():
    """B1: atom maps to EXTRA_ATOM_ARGS (the atom branch sits before vllm to avoid substring collisions)."""
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
    """B1: PROFILE=1 + framework=atom must NOT inject sglang/vllm profiler CLI flags into EXTRA_ATOM_ARGS (atom's argparse rejects them)."""
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
    """No config_path override + FRAMEWORK=vllm resolves to baseline_vllm.yaml (the regression that blocked vllm users)."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = BaselineExecutor()
    # default_config_path=None so the resolver is consulted at call time.
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "baseline_vllm.yaml"


def test_profile_executor_picks_framework_yaml_at_call_time(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    pe = ProfileExecutor()
    assert pe.default_config_path is None
    assert pe._resolve_default_config().name == "profile_vllm.yaml"


@pytest.mark.asyncio
async def test_profile_executor_skips_when_framework_atom(monkeypatch, tmp_path):
    """FRAMEWORK=atom now falls through to the normal profile path.

    The atom Magpie wrapper bridges PROFILE=1 to atom's torch profiler,
    so the historical structured ``skipped`` short-circuit is retired.
    """
    monkeypatch.setenv("FRAMEWORK", "atom")
    # Anchor session/runs paths under the test tmp dir. Without this the
    # executor falls back to the ``/workspace/hyperloom`` default, which is
    # not writable on a clean CI runner (PermissionError on ``/workspace``).
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    pe = ProfileExecutor()
    # Sentinel-patch the parent __call__ so we can prove the normal path
    # is reached without launching Magpie in this unit test.
    called = {"parent": False}

    async def _fake_parent(self, ctx):
        called["parent"] = True
        return {"status": "succeeded"}

    monkeypatch.setattr(BaselineExecutor, "__call__", _fake_parent)

    task = SimpleNamespace(params={}, task_id="t-atom-profile")
    ctx = SimpleNamespace(task=task, extra=None)

    result = await pe(ctx)

    assert result["status"] == "succeeded"
    assert called["parent"] is True


def test_profile_executor_sanitizes_current_best_args(monkeypatch, tmp_path):
    """Profile must not inherit torch-compile flags that break profiler boot."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    captured: dict[str, str] = {}

    async def _fake_parent(self, ctx):
        captured.update(ctx.task.params)
        return {"status": "succeeded"}

    monkeypatch.setattr(BaselineExecutor, "__call__", _fake_parent)

    task = SimpleNamespace(
        params={
            "base_extra_args": (
                "--enable-torch-compile --torch-compile-max-bs 32 "
                "--quantization fp8"
            ),
        },
        task_id="t-profile-sanitize",
    )
    ctx = SimpleNamespace(task=task, extra={"workspace": str(tmp_path / "ws")})

    result = asyncio.run(ProfileExecutor()(ctx))

    assert result["status"] == "succeeded"
    merged = captured["extra_server_args"]
    assert "extra_sglang_args" not in captured
    assert "--enable-torch-compile" not in merged
    assert "--torch-compile-max-bs" not in merged
    assert "--quantization fp8" in merged


def test_profile_executor_sanitizes_canonical_extra_server_args(monkeypatch, tmp_path):
    """Canonical extra_server_args must not bypass the profile sanitizer."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    captured: dict[str, str] = {}

    async def _fake_parent(self, ctx):
        captured.update(ctx.task.params)
        return {"status": "succeeded"}

    monkeypatch.setattr(BaselineExecutor, "__call__", _fake_parent)

    task = SimpleNamespace(
        params={
            "base_extra_args": "--attention-backend AITER",
            "extra_server_args": (
                "--enable-torch-compile --torch-compile-max-bs 32 "
                "--quantization fp8"
            ),
            "extra_sglang_args": "--enable-torch-compile",
        },
        task_id="t-profile-canonical-sanitize",
    )
    ctx = SimpleNamespace(task=task, extra={"workspace": str(tmp_path / "ws")})

    result = asyncio.run(ProfileExecutor()(ctx))

    assert result["status"] == "succeeded"
    merged = captured["extra_server_args"]
    assert "extra_sglang_args" not in captured
    assert "--enable-torch-compile" not in merged
    assert "--torch-compile-max-bs" not in merged
    assert "--attention-backend AITER" in merged
    assert "--quantization fp8" in merged


@pytest.mark.asyncio
async def test_roofline_executor_skips_when_framework_atom(monkeypatch):
    """FRAMEWORK=atom now attempts the normal roofline profile sub-step."""
    from inference_optimizer.orchestrator.action_executors.roofline import (
        RooflineExecutor,
    )

    monkeypatch.setenv("FRAMEWORK", "atom")
    rexec = RooflineExecutor(shared_state=SimpleNamespace())

    # Sentinel: prove the lazy import / sub-step orchestration is reached.
    import inference_optimizer.orchestrator.action_executors.profile as profile_mod

    async def _explode(_ctx):
        raise AssertionError("profile_executor must not be invoked under atom")

    monkeypatch.setattr(profile_mod, "profile_executor", _explode)

    task = SimpleNamespace(
        params={},
        task_id="t-atom-roofline",
        idempotency_key="t-atom-roofline",
        requires_lanes=[],
        allowed_tools=[],
        side_effects=[],
        lease_ttl_sec=0,
    )
    ctx = SimpleNamespace(task=task, lease=None, extra=None)

    result = await rexec(ctx)
    assert result["status"] == "failed"
    assert result["phase"] == "profile"
    assert "profile_executor raised" in result["error"]


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
    """When the workspace has torch_trace/*.trace.json.gz, the runner surfaces them in the result."""
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
    """ProfileExecutor must patch the InferenceX checkout Magpie will use (Qwen3-32B regression: empty benchmark.inferencex_path lost NUM_PROMPTS)."""
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


# kernel_request_handlers — direct unit
@pytest.mark.asyncio
async def test_trace_analyze_handler_dry_run_returns_structured_result(session_dir):
    """The handler surfaces the tool's structured JSON verbatim (status + run_id + session_id)."""
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
    # No trace files → failed, but the response must still be structured.
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "tool" in res or "run_id" in res or "error" in res
    assert res.get("session_id") == session_dir.name or "run_id" in res


@pytest.mark.asyncio
async def test_trace_analyze_handler_tolerates_non_string_analysis_route(session_dir):
    """A non-string analysis_route (e.g. bool/list from an LLM payload) must not
    crash cmd construction with AttributeError; it is coerced and ignored."""
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    for bad_route in (True, ["deterministic"], {"route": "agent"}, 1):
        payload = {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
            "dry_run": True,
            "budget_minutes": 1,
            "analysis_route": bad_route,
        }
        res = await krh.trace_analyze_handler(payload, session_dir=session_dir)
        # Must return a structured result, never raise AttributeError.
        assert res["status"] in ("ok", "succeeded", "failed")


@pytest.mark.asyncio
async def test_trace_analyze_handler_records_bypass_discovery_success(
    session_dir, monkeypatch,
):
    """Deterministic route surfaces a kernel_journey discovery run labelled
    source="bypass" (with the real hot kernels), while version provenance stays
    under the tracelens toolchain (no junk versions["bypass"])."""
    from inference_optimizer.breakdown.recorder import assemble_parts

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()

    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        payload = {
            "status": "ok",
            "orchestrator_mode": "deterministic",
            "hot_kernels": [
                {"kernel_id": "k001", "name": "fused_moe", "gpu_pct": 42.0,
                 "bottleneck": "memory", "reusable_native_kernel": True},
                {"kernel_id": "k002", "name": "rms_norm", "gpu_pct": 7.5},
            ],
            "artifact_paths": {"kernel_candidates": "/tmp/kc.json"},
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "deterministic",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    # The deterministic route flag is forwarded to the tool.
    assert "--analysis-route" in captured["cmd"]
    assert "deterministic" in captured["cmd"]

    out = assemble_parts(session_dir)
    runs = out["kernel_journey"]["discovery_runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["source"] == "bypass"
    assert run["status"] == "ok"
    assert run["hot_kernel_count"] == 2
    assert {k["name"] for k in run["hot_kernels"]} == {"fused_moe", "rms_norm"}
    assert run["scan"]["analysis_route"] == "bypass"
    # Underlying toolchain is still tracelens; no empty versions["bypass"].
    assert "bypass" not in out.get("versions", {})


@pytest.mark.asyncio
async def test_trace_analyze_handler_records_bypass_discovery_failed(
    session_dir, monkeypatch,
):
    """Fail-loud deterministic pipeline -> discovery run status=failed with the
    error text and an empty hot-kernel list, still labelled source="bypass"."""
    from inference_optimizer.breakdown.recorder import assemble_parts

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "failed",
            "orchestrator_mode": "deterministic",
            "error": "deterministic: category script for gemm exited rc=1",
            "hot_kernels": [],
        }
        return 1, json.dumps(payload), "boom"

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "deterministic",
        },
        session_dir=session_dir,
    )
    assert res["status"] == "failed"

    out = assemble_parts(session_dir)
    run = out["kernel_journey"]["discovery_runs"][0]
    assert run["source"] == "bypass"
    assert run["status"] == "failed"
    assert run["hot_kernel_count"] == 0
    assert run["error"]


@pytest.mark.asyncio
async def test_trace_analyze_handler_records_bypass_discovery_high_idle_empty(
    session_dir, monkeypatch,
):
    """High-idle gate suppresses hot kernels but the run still succeeds -> a
    bypass discovery run with status=ok and hot_kernel_count=0."""
    from inference_optimizer.breakdown.recorder import assemble_parts

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "orchestrator_mode": "deterministic",
            "hot_kernels": [],
            "trace_health_warnings": [
                {"code": "high_gpu_idle", "severity": "warning"},
            ],
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "deterministic",
        },
        session_dir=session_dir,
    )
    assert res["status"] == "ok"

    out = assemble_parts(session_dir)
    run = out["kernel_journey"]["discovery_runs"][0]
    assert run["source"] == "bypass"
    assert run["status"] == "ok"
    assert run["hot_kernel_count"] == 0


@pytest.mark.asyncio
async def test_trace_analyze_handler_agent_route_stays_tracelens(
    session_dir, monkeypatch,
):
    """The LLM/agent route keeps source="tracelens" (regression guard for the
    bypass relabel)."""
    from inference_optimizer.breakdown.recorder import assemble_parts

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()

    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "orchestrator_mode": "claude_agent_sdk",
            "hot_kernels": [
                {"kernel_id": "k001", "name": "fused_moe", "gpu_pct": 30.0},
            ],
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "agent",
        },
        session_dir=session_dir,
    )

    out = assemble_parts(session_dir)
    run = out["kernel_journey"]["discovery_runs"][0]
    assert run["source"] == "tracelens"
    assert run["scan"]["analysis_route"] == "tracelens"


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
    assert "--roofline-json" not in captured["cmd"]
    assert "/tmp/roofline.json" not in captured["cmd"]
    assert "--capture-folder" in captured["cmd"]
    assert "/tmp/capture_traces" in captured["cmd"]


@pytest.mark.asyncio
async def test_trace_analyze_handler_backfills_workload_context_from_state(
    session_dir, monkeypatch,
):
    """When the payload omits framework/gpu_type/model, the handler falls back to SharedState for the real workload context."""
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
async def test_trace_analyze_handler_requires_kernel_agent_root(session_dir, monkeypatch):
    # N15: HYPERLOOM_KERNEL_AGENT_ROOT is a lazy env read; delenv exercises the "not configured" branch.
    monkeypatch.delenv("HYPERLOOM_KERNEL_AGENT_ROOT", raising=False)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir)},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert res["error_class"] == "kernel_agent_root_missing"
    assert "HYPERLOOM_KERNEL_AGENT_ROOT is not set" in res["error"]


# T4 — TraceLens permanent failure stays failed (no fallback): the handler preserves ``status=failed`` and appends structured diagnostics instead of rewriting to ok+empty kernels.

@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_keeps_tool_failure_failed(
    session_dir, monkeypatch,
):
    """When tracelens_analysis.py returns ``status=failed`` the handler keeps the failure status, clears stale candidates, and appends a diagnostic warning."""
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "failed",
            "tool": "tracelens_analysis",
            "error": "RuntimeError: TraceLens perf CLI crashed",
            "returncode": 1,
            "stderr_tail": "RuntimeError: graph capture folder missing",
            # Seed a non-empty list to prove the handler clears stale candidates on failure.
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
    """A T3 idle-gate ``trace_health_warnings`` (status=ok, empty hot_kernels) must pass through verbatim."""
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
    """With no ``trace_health_warnings`` (steady state), the handler still surfaces an empty list (no ``None`` guard needed downstream)."""
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


# T5 — trace_health_warnings must reach the Orchestration LLM: record_trace_analyze keeps the warning list and _format_last_trace_analyze surfaces it inline.

def test_record_trace_analyze_persists_trace_health_warnings(session_dir):
    """``record_trace_analyze`` keeps ``trace_health_warnings`` verbatim in ``last_trace_analyze`` for next-tick rendering."""
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
    """Steady-state: the cached entry exposes ``trace_health_warnings`` as an empty list, not an absent field."""
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
    """893bc6f: ``task_groups`` must flow into ``last_trace_analyze`` so the multi-KEEP queue collapses members of the same AST function into one slot."""
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
    # After k002 + k004 attempted, group-aware collapse reports no untried kernels.
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
    """With no ``task_groups`` field (legacy TraceLens output), the cached entry defaults to an empty list."""
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
    """Defensive: only well-formed warning dicts with a ``code`` key are accepted into ``last_trace_analyze``."""
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
    """Prompt rendering: a persisted idle warning surfaces inline with its numeric context."""
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
    """Tool-failure warning carries ``returncode``; the prompt must surface ``rc=N`` to distinguish a crash from a benign skip."""
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
    """Format-stability guard: with no warnings, the prompt line must NOT gain a ``warnings=[]`` suffix (snapshot tests pin the legacy format)."""
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
    """End-to-end: T3 idle warning flows handler → SharedState.last_trace_analyze → Orchestration prompt line."""
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
    """T4: a permanent TraceLens failure warning must reach the Orchestration prompt."""
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
    """When the tool emits ``status=failed`` plus a pre-existing warnings list, the handler appends the failure warning rather than overwriting."""
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


def test_optimization_wrapper_timeout_sec_geak_default_full_mode_130min(monkeypatch):
    # Default tracks ``$GEAK_RUN_MODE`` (full -> 130 min) to match the kernel-agent defaults (PR #301).
    monkeypatch.delenv("GEAK_RUN_MODE", raising=False)
    monkeypatch.delenv("HYPERLOOM_GEAK_BUDGET_MIN", raising=False)
    assert krh._optimization_wrapper_timeout_sec({"backends": "geak"}) == 130 * 60 + 180


def test_optimization_wrapper_timeout_sec_geak_quick_mode_70min(monkeypatch):
    monkeypatch.setenv("GEAK_RUN_MODE", "quick")
    monkeypatch.delenv("HYPERLOOM_GEAK_BUDGET_MIN", raising=False)
    assert krh._optimization_wrapper_timeout_sec({"backends": "geak"}) == 70 * 60 + 180


def test_optimization_wrapper_timeout_sec_oob_default_60min():
    assert krh._optimization_wrapper_timeout_sec({"backends": "claude"}) == 60 * 60 + 180


def test_optimization_wrapper_timeout_sec_geak_env_override(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "120")
    assert krh._optimization_wrapper_timeout_sec({"backends": "geak"}) == 120 * 60 + 180


@pytest.mark.asyncio
async def test_run_optimization_handler_missing_kernel_id(session_dir):
    # ``source_file`` short-circuits the ``missing_trace_analyze`` guard so the legacy missing-kernel_id path is exercised.
    res = await krh.run_optimization_handler(
        {"source_file": "/tmp/dummy.py"}, session_dir=session_dir,
    )
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
    """Dispatch table includes trace_analyze / run_gemm_tuning / run_optimization, not unknown kinds."""
    assert krh.has_handler("trace_analyze")
    assert krh.has_handler("run_gemm_tuning")
    assert krh.has_handler("run_optimization")
    assert not krh.has_handler("totally_unknown_kind")


# PR-B §1: _batch_kernel_candidates collapses task_group members
def _write_candidates_json(tmp_path, payload):
    p = tmp_path / "kernel_candidates.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_batch_kernel_candidates_collapses_task_group_to_primary(tmp_path):
    """Two reusable kernels in the same task_group dispatch as ONE candidate (the primary), with the full group attached."""
    # PR-I: rows must carry gpu_pct >= 3.0 to pass the default hot-kernel gate.
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
    """When the group's primary_kernel_id is non-reusable, dispatch falls back to the first reusable member instead of dropping the group."""
    # PR-I: rows must carry gpu_pct >= 3.0 to be retained by the dispatcher.
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
    """With no task_groups[] (legacy runs), the candidate list matches pre-PR-B behaviour."""
    # PR-I: legacy fixture carries gpu_pct >= 3.0 so the hot-kernel gate doesn't drop k001 (orthogonal to PR-I).
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


# Coordinator — REQUEST programmatic handler integration
@pytest.mark.asyncio
async def test_coordinator_request_trace_analyze_uses_handler(session_dir):
    """REQUEST{kind=trace_analyze} runs the registered handler programmatically and emits RESPONSE without the Kernel LLM."""
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
    """REQUEST with no handler is mirrored to the kernel inbox (LLM responder path), no auto-RESPONSE."""
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


# PR-X: batch dispatch enablers — _DEFAULT_KERNEL_BATCH_PARALLEL sized for a full node, and Coordinator force-injects candidates_path so the batch path fires deterministically (LLM values still win).
def test_default_kernel_batch_parallel_matches_full_node():
    """Default fanout is sized for a single MI300X / MI355X node (8 GPU)
    so a typical ``run_optimization`` batch (TraceLens emits 3-8 reusable
    units) does NOT serialize behind an asyncio semaphore tighter than
    Ray's view of the cluster. Pre-PR-X value was 3, which throttled even
    the small batches actually observed in production sessions."""
    assert krh._DEFAULT_KERNEL_BATCH_PARALLEL == 8


@pytest.mark.asyncio
async def test_coordinator_injects_candidates_path_for_run_optimization(
    session_dir,
):
    """When the LLM emits ``run_optimization`` without ``candidates_path``,
    the Coordinator must pull it from ``state.last_trace_analyze`` and
    inject it into the handler payload so ``_run_optimization_batch``
    fires instead of silently collapsing to ``_run_optimization_single``
    (which would waste 7 idle GPUs on an 8-GPU node)."""
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


# PR-B: multi-KEEP integrate queue + streaming batch record — streaming record_partial callback, batch_mode dedup, and base_tput auto-injection on integrate.
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

    async def fake_sequence(base_payload, candidate, *, session_dir, parallel_backends=False):
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
            payload={"candidates_path": "/dummy",
                     "backend_order": "geak,claude,codex",
                     "max_parallel": 3},
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
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
            candidate,
            session_dir=session_dir,
        )

    # Ladder walks past GEAK NEEDS_REVIEW + Claude REVERT, then breaks on Codex KEEP.
    assert calls == ["geak", "claude", "codex"], calls
    assert (best.get("proposal") or {}).get("decision") == "KEEP", best
    assert (best.get("verification") or {}).get("micro_speedup") == 1.17
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/codex.py"


@pytest.mark.asyncio
async def test_backend_ladder_breaks_on_first_keep(session_dir):
    """When GEAK already KEEPs, the ladder short-circuits (no Claude/Codex)."""
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
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
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
    """If NO backend KEEPs, the ladder picks the highest-micro non-KEEP."""
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
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
        )

    # All non-KEEP -> pick highest micro (Claude 1.45)
    assert (best.get("verification") or {}).get("micro_speedup") == 1.45
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/claude.py"


@pytest.mark.asyncio
async def test_backend_sequence_parallel_runs_oob_even_when_geak_keeps(session_dir):
    """GPU-rich mode: GEAK's clean KEEP must NOT short-circuit the OOB
    ladder. GEAK and OOB race concurrently and the higher micro_speedup
    wins -- the whole point of spending spare GPUs to chase a better
    rewrite than GEAK's first KEEP."""
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
        if backend == "geak":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.20,
                                 "correctness_passed": True,
                                 "best_artifact_path": "/tmp/geak.py"},
            }
        if backend == "claude":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.50,
                                 "correctness_passed": True,
                                 "best_artifact_path": "/tmp/claude.py"},
            }
        raise AssertionError(f"unexpected backend {backend!r}")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
            parallel_backends=True,
        )

    # GEAK KEEP no longer short-circuits: claude (OOB) must have run too.
    assert "geak" in calls and "claude" in calls, calls
    # Higher micro wins (claude 1.50 > geak 1.20).
    assert (best.get("proposal") or {}).get("decision") == "KEEP"
    assert (best.get("verification") or {}).get("micro_speedup") == 1.50
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/claude.py"
    # Attempt ledger records both ladders.
    logged = {a["backend"] for a in best["backend_fallback_attempts"]}
    assert "geak" in logged and "claude" in logged, logged


@pytest.mark.asyncio
async def test_backend_sequence_parallel_keeps_geak_when_oob_lower(session_dir):
    """GPU-rich mode races both, but if GEAK is the strongest it still
    wins the best-selection contest."""
    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        if backend == "geak":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.60,
                                 "best_artifact_path": "/tmp/geak.py"},
            }
        if backend == "claude":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.10,
                                 "best_artifact_path": "/tmp/claude.py"},
            }
        if backend == "codex":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "NEEDS_REVIEW", "reasons": []},
                "verification": {"micro_speedup": 1.05,
                                 "best_artifact_path": "/tmp/codex.py"},
            }
        raise AssertionError(f"unexpected backend {backend!r}")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
            parallel_backends=True,
        )

    assert (best.get("verification") or {}).get("micro_speedup") == 1.60
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/geak.py"


@pytest.mark.asyncio
async def test_backend_sequence_parallel_oob_ladder_still_falls_back(session_dir):
    """In GPU-rich mode the OOB group keeps its own internal
    break-on-KEEP fallback (claude REVERT -> codex KEEP), raced against
    GEAK, with the strongest overall result selected."""
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
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
                "proposal": {"decision": "REVERT", "reasons": []},
                "verification": {"micro_speedup": 0.9,
                                 "best_artifact_path": "/tmp/claude.py"},
            }
        if backend == "codex":
            return {
                "status": "ok", "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.45,
                                 "best_artifact_path": "/tmp/codex.py"},
            }
        raise AssertionError(f"unexpected backend {backend!r}")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy",
             "backend_order": "geak,claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
            parallel_backends=True,
        )

    # OOB group walked claude -> codex; geak raced alongside exactly once.
    assert calls.count("geak") == 1
    assert "claude" in calls and "codex" in calls, calls
    # codex KEEP (1.45) beats geak NEEDS_REVIEW (1.30).
    assert (best.get("proposal") or {}).get("decision") == "KEEP"
    assert (best.get("verification") or {}).get("micro_speedup") == 1.45
    assert (best.get("verification") or {}).get("best_artifact_path") == "/tmp/codex.py"


@pytest.mark.asyncio
async def test_backend_sequence_parallel_noop_without_geak(session_dir):
    """``parallel_backends`` is inert when GEAK isn't in the ladder
    (nothing to race) -- it behaves like the sequential OOB ladder and
    still short-circuits on the first KEEP."""
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
        return {
            "status": "ok", "kernel_id": child["kernel_id"],
            "proposal": {"decision": "KEEP", "reasons": []},
            "verification": {"micro_speedup": 1.2,
                             "best_artifact_path": f"/tmp/{backend}.py"},
        }

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy", "backend_order": "claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
            parallel_backends=True,
        )

    # No geak in the ladder -> sequential break-on-KEEP: only claude runs.
    assert calls == ["claude"]
    assert (best.get("verification") or {}).get("micro_speedup") == 1.2


@pytest.mark.asyncio
async def test_backend_sequence_forge_keep_short_circuits(session_dir):
    """Forge runs first and a KEEP short-circuits before GEAK/OOB.

    Regression coverage for Bugbot: _kernel_result_rank() returns a tuple, so
    the short-circuit must inspect the KEEP slot instead of comparing the tuple
    directly to int 0.
    """
    calls: list[str] = []

    async def fake_single(child, *, session_dir):
        backend = child["backends"]
        calls.append(backend)
        if backend == "forge":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.05,
                                 "best_artifact_path": "/tmp/forge.py"},
            }
        raise AssertionError(f"forge KEEP must short-circuit before {backend!r}")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy",
             "backend_order": "forge,geak,claude,codex"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py",
             "reusable_native_kernel": True},
            session_dir=session_dir,
            parallel_backends=True,
        )

    assert calls == ["forge"]
    assert (best.get("proposal") or {}).get("decision") == "KEEP"
    assert best["batch_kernel_id"] == "k004"
    assert {a["backend"] for a in best["backend_fallback_attempts"]} == {"forge"}


@pytest.mark.asyncio
async def test_batch_serializes_when_forge_in_ladder(session_dir, monkeypatch):
    """Forge in-place editing is repo-global, so batch concurrency is capped at 1.

    Even when GPU-rich mode says parallel backends are available, multiple
    kernels must not race forge against other backends in the same live repo.
    """
    active = 0
    max_active = 0
    seen_flags: list[bool] = []

    async def fake_sequence(base_payload, candidate, *, session_dir, parallel_backends=False):
        nonlocal active, max_active
        seen_flags.append(parallel_backends)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "status": "ok",
            "kernel_id": candidate["kernel_id"],
            "proposal": {"decision": "REVERT", "reasons": []},
            "verification": {"micro_speedup": 1.0},
        }

    monkeypatch.setattr(krh, "_should_parallelize_backends", lambda payload, n: True)
    monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fake_sequence)

    out = await krh._run_optimization_batch(
        {"candidates_path": "/dummy",
         "backend_order": "forge,geak,claude,codex",
         "max_parallel": 8},
        [
            {"kernel_id": "k001", "source_file": "/p/a.py"},
            {"kernel_id": "k002", "source_file": "/p/b.py"},
        ],
        session_dir=session_dir,
    )

    assert max_active == 1
    assert seen_flags == [True, True]
    assert out["parallel_backends"] is True


@pytest.mark.asyncio
async def test_batch_threads_parallel_backends_flag(session_dir, monkeypatch):
    """``_run_optimization_batch`` computes the GPU-rich decision once,
    threads it into every ``_run_kernel_backend_sequence`` call, and
    surfaces it on the aggregate result for observability."""
    seen_flags: list[bool] = []

    async def fake_sequence(
        base_payload, candidate, *, session_dir, parallel_backends=False,
    ):
        seen_flags.append(parallel_backends)
        return {
            "status": "ok",
            "kernel_id": candidate["kernel_id"],
            "source_file": candidate.get("source_file"),
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.3},
        }

    candidates = [
        {"kernel_id": "k1", "source_file": "/p/a.py", "reusable_native_kernel": True},
        {"kernel_id": "k2", "source_file": "/p/b.py", "reusable_native_kernel": True},
    ]
    # Force the decision deterministically (no real GPUs under CI); the
    # env override short-circuits the torch/GPU math in
    # ``_should_parallelize_backends``.
    monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "1")
    with patch.object(krh, "_run_kernel_backend_sequence", side_effect=fake_sequence):
        out = await krh._run_optimization_batch(
            payload={"candidates_path": "/dummy", "max_parallel": 2},
            candidates=candidates,
            session_dir=session_dir,
        )

    assert seen_flags == [True, True], seen_flags
    assert out["parallel_backends"] is True


@pytest.mark.asyncio
async def test_batch_handler_isolates_sub_task_exceptions_from_gather(
    session_dir,
):
    """Sub-task exceptions surface as structured ``failed`` results so ``gather`` stays true wait-all and doesn't unblock the Coordinator mid-batch."""
    candidates = [
        {"kernel_id": "kFast", "source_file": "/p/fast.py", "reusable_native_kernel": True},
        {"kernel_id": "kCrash", "source_file": "/p/crash.py", "reusable_native_kernel": True},
        {"kernel_id": "kSlow", "source_file": "/p/slow.py", "reusable_native_kernel": True},
    ]

    recorded: list[dict] = []
    completion_order: list[str] = []

    async def fake_sequence(base_payload, candidate, *, session_dir, parallel_backends=False):
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

    # record_partial got one call per candidate; the crash surfaced as a structured failed with kernel_id preserved.
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
    session_dir,
):
    """End-to-end: record_partial records each sub-attempt in flight, and the post-gather record_kernel_opt(best) is skipped in batch_mode (no double-counting)."""
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 1234.5
    c.shared_state.last_profile_trace = "/wekafs/trace/x.json.gz"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/wekafs/trace/x.json.gz",
        "candidates_path": "/wekafs/cached/candidates.json",
    }
    # The sequence gate still consults ``last_select_kernels`` (to be renamed ``trace_analyze`` in M3).
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
    """Explicit operator-supplied ``base_tput`` must NOT be clobbered by the auto-injection."""
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
    """PR-C: a kernel on rejected_kernel_ids must not appear in the next batch, even if still in kernel_candidates.json."""
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
    """PR-C max_attempts=1 default: any prior attempt skips the kernel in the next batch."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    state = SharedState.load_or_init(session_dir)
    # k001 has an attempt recorded but is not yet on the rejected list (PARTIAL below max_partial).
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
    """When the primary (k002) is rejected, the task_group still dispatches via the next live member (k001)."""
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
    """If every member of a task_group is unusable, the group skips cleanly."""
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
    """In-flight defense: a status/ko-*.json with state=running for k004 keeps it out of the next batch."""
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
    """PR-I: ``_batch_kernel_candidates`` default (3.0) must match ``SharedState.untried_hot_reusable_kernels``'s gate so a sub-threshold kernel can't sneak in via task_group fallback."""
    cpath = _candidates_factory([
        {"kernel_id": "k001", "gpu_pct": 38.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k006", "gpu_pct": 1.3, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
        {"kernel_id": "k008", "gpu_pct": 3.13, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    # Default 3.0 filters out k006 (1.3) but keeps k001 (38) and k008 (3.13).
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
    """``_resolve_integrate_payload`` looks up patch_path / source_file from the per-kernel ``kernel_opt_attempts`` ledger so any queued KEEP can integrate."""
    from inference_optimizer.orchestrator.shared_state import SharedState
    state = SharedState.load_or_init(session_dir)
    # Two KEEPs landed but last_kernel_opt only holds the strongest (k009).
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

    # integrate(k001) carries only the kernel_id (the second queued KEEP, not last_kernel_opt).
    resolved, missing = krh._resolve_integrate_payload(
        {"kernel_id": "k001", "base_tput": 4500.0},
        session_dir=session_dir,
    )
    assert missing is None, missing
    assert resolved.get("patch_path") == "/tmp/k001.py", \
        "patch_path must fall back to kernel_opt_attempts[k001].last_artifact_path"
    assert resolved.get("source_file") == "/p/moe_op.py", \
        "source_file must fall back to kernel_opt_attempts[k001].last_source_file"
