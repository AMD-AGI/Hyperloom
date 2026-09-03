# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProfileExecutor + kernel REQUEST programmatic handler tests."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperloom.inference_optimizer.cli import bootstrap as cli_bootstrap
from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.inference_optimizer.cli import parser as cli_parser
from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
    _default_baseline_config,
    _materialize_config_with_envs,
)
from hyperloom.orchestrator.actions.executors.profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
    _default_profile_config,
    _preferred_main_trace_path,
    _sanitize_profile_server_args,
    _trace_rank,
    _trace_files_for_dir,
)
from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.orchestrator.bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.sub_agent_runner import (
    SubAgentRunner,
)
from hyperloom.inference_optimizer.session.manifest import build_manifest
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.bus.storage import SqliteConnection


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    kernel_agent_root = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    tracelens_root = tmp_path / "TraceLens"
    # A usable checkout needs .git (completeness gate).
    (tracelens_root / ".git").mkdir(parents=True)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n) for n in ("orchestration", "critic", "robustness")}


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

    assert cli_model_gate._gpu_runner_type("mi325x") == "mi300x"
    assert cli_model_gate._GFX_TO_RUNNER.get("gfx1100") is None
    manifest = build_manifest(tmp_path, args=args, session_id="mi325x-session")
    state = cli_bootstrap._seed_shared_state(
        tmp_path,
        args,
        session_id="mi325x-session",
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

    assert cli_model_gate._gpu_runner_type("mi308x") == "mi300x"
    manifest = build_manifest(tmp_path, args=args, session_id="mi308x-session")
    state = cli_bootstrap._seed_shared_state(
        tmp_path,
        args,
        session_id="mi308x-session",
    )

    assert manifest["gpu_type"] == "mi308x"
    assert state.gpu_type == "mi308x"
    assert os.environ["TARGET_GPU_TYPE"] == "mi308x"
    assert os.environ["GPU_TYPE"] == "mi300x"


def test_cli_parser_accepts_mi308x():
    parser = cli_parser._build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--model",
            "/tmp/model",
            "--gpu-type",
            "mi308x",
        ]
    )
    assert args.gpu_type == "mi308x"


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so the artifact harvest doesn't pick up the host's ``/workspace``."""
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# ProfileExecutor
def test_profile_default_config_path_is_in_assets():
    assert "profile_sglang.yaml" in str(PROFILE_DEFAULT_CONFIG)
    assert PROFILE_DEFAULT_CONFIG.exists(), "profile YAML must ship as a package asset"


def test_profile_yaml_has_torch_profiler_enabled():
    """The whole point of the profile config is profiler ON."""
    import yaml

    with PROFILE_DEFAULT_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True


def test_materialize_config_injects_model_path(tmp_path):
    """Default YAML's hardcoded Qwen3-8B must be overridden when caller passes ``model_path``."""
    import yaml

    out = _materialize_config_with_envs(
        PROFILE_DEFAULT_CONFIG,
        tmp_path,
        model_path="/path/models/DeepSeek-R1-0528",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/path/models/DeepSeek-R1-0528"


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
        extra_envs={"BENCH_FOO": "bar"},
        model_path="/some/model",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["model"] == "/some/model"
    assert rendered["benchmark"]["envs"]["BENCH_FOO"] == "bar"


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
    src_yaml.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/m",
                    "benchmark_script": "sglang_mi300x.sh",  # legacy field
                },
            }
        )
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = _materialize_config_with_envs(
        src_yaml,
        out_dir,
        gpu_type="mi355x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["runner_type"] == "mi355x"
    assert rendered["benchmark"]["benchmark_script"] == "sglang_mi355x.sh", (
        "gpu_type must pin the generic {framework}_{gpu_type}.sh"
    )


def test_materialize_config_forces_generic_when_source_yaml_has_no_script(
    tmp_path,
):
    """Even with no source `benchmark_script`, the renderer must write one explicitly."""
    import yaml

    src_yaml = tmp_path / "src.yaml"
    src_yaml.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "/m",
                    # No benchmark_script field at all.
                },
            }
        )
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = _materialize_config_with_envs(
        src_yaml,
        out_dir,
        gpu_type="mi300x",
    )
    with out.open() as f:
        rendered = yaml.safe_load(f)
    assert rendered["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


# TP / CONC env must override yaml hardcode.
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
    tmp_path,
    monkeypatch,
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
    tmp_path,
    monkeypatch,
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
    tmp_path,
    monkeypatch,
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
    src_yaml.write_text(
        yaml.safe_dump(
            {
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
            }
        )
    )
    for k in ("TP", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    out = _materialize_config_with_envs(src_yaml, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    # yaml default is "1" — should be preserved as-is when TP not overridden upward
    assert envs.get("ROCR_VISIBLE_DEVICES") == "1"


# steady-state window follows the TraceLens magpie skill formulas.
def _profile_yaml(tmp_path, framework: str, envs: dict) -> Path:
    """Synthesize a minimal profile YAML the materializer recognises as PROFILE=1 + torch_profiler.enabled=True."""
    import yaml as _yaml

    src = tmp_path / f"src_{framework}.yaml"
    src.write_text(
        _yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "/m",
                    "envs": {"PROFILE": "1", **envs},
                    "profiler": {"torch_profiler": {"enabled": True}},
                },
            }
        )
    )
    return src


def _clear_workload_env(monkeypatch):
    for k in (
        "CONC",
        "ISL",
        "OSL",
        "TP",
        "MAX_MODEL_LEN",
        "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES",
        "FRAMEWORK",
        "INFERENCEX_PATH",
    ):
        monkeypatch.delenv(k, raising=False)


def test_materialize_profile_window_vllm_skill_formula_default_R(
    tmp_path,
    monkeypatch,
):
    """vLLM: OSL=1024, CONC=32, R unset → capture capped at 128, delay=6080.

    Capture is the serialization-safe cap (default 128); delay keeps the
    warmup formula OSL*(R+1)*3 - max_iters/2 = 1024*2*3 - 64 = 6080.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    assert "--profiler-config.max_iterations 128" in extra, extra


def test_materialize_profile_window_vllm_skill_formula_explicit_R(
    tmp_path,
    monkeypatch,
):
    """vLLM: explicit R=0.5 must shrink delay (warmup: 3*OSL*(R+1) term)."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("RANDOM_RANGE_RATIO", "0.5")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    # R=0.5: max capped at 128; delay = 1024 * 1.5 * 3 - 128/2 = 4608 - 64 = 4544.
    extra = envs["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 4544" in extra, extra
    assert "--profiler-config.max_iterations 128" in extra, extra
    # And R must round-trip into the YAML as a float, not stringified-int.
    assert envs["RANDOM_RANGE_RATIO"] == 0.5


def test_materialize_profile_bounds_survive_a_replacing_candidate(
    tmp_path,
    monkeypatch,
    caplog,
):
    """A candidate with args_mode="replace" must not strip the profiler bounds.

    Once ``current_best`` carries ``args_mode="replace"``, the candidate's flag
    string overwrote EXTRA_VLLM_ARGS wholesale and took the injected
    ``max_iterations`` with it. vLLM reads a missing ``max_iterations`` as
    "profile until stop_profile", which grew host RAM at ~60 MiB/s until the
    cgroup OOM-killer took the engine process out mid-roofline.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_VLLM_ARGS": "--profiler-config.ignore_frontend True",
        },
    )
    caplog.set_level("WARNING")
    # Verbatim from the gemma-4-26B-A4B roofline that was OOM-killed, JSON flag
    # included -- the restore has to survive a string the arg merger refuses to
    # tokenize.
    candidate = '--no-enable-prefix-caching --compilation-config {"cudagraph_capture_sizes":[17,34,1088]}'
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args=candidate,
        args_mode="replace",
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    assert "--profiler-config.max_iterations 128" in extra, extra
    # The frontend profiler tracks no iterations, so it has to come back too.
    assert "--profiler-config.ignore_frontend True" in extra, extra
    assert "--profiler-config.capture_torch_profiler True" in extra, extra
    assert "--profiler-config.detailed_trace_annotation True" in extra, extra
    # The candidate's own flags must still take effect, JSON value unmangled.
    assert "--no-enable-prefix-caching" in extra, extra
    assert '--compilation-config {"cudagraph_capture_sizes":[17,34,1088]}' in extra, extra
    assert "lost torch-profiler flags" in caplog.text


def test_materialize_profile_bounds_survive_an_extra_envs_override(
    tmp_path,
    monkeypatch,
):
    """``extra_envs`` is applied last and unconditionally, so an EXTRA_VLLM_ARGS entry there is the other way the bounds can vanish."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_envs={"EXTRA_VLLM_ARGS": "--quantization fp8"},
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert "--quantization fp8" in extra, extra


def test_materialize_profile_states_ignore_frontend_exactly_once(
    tmp_path,
    monkeypatch,
):
    """The bounds imply ignore_frontend, but the YAML usually already sets it and vLLM warns on duplicate keys."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_VLLM_ARGS": "--profiler-config.ignore_frontend True",
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert extra.count("--profiler-config.ignore_frontend True") == 1, extra


def test_materialize_profile_adds_ignore_frontend_when_yaml_omits_it(
    tmp_path,
    monkeypatch,
):
    """Bounding only the worker profiler leaves AsyncLLM capturing the whole range, so the flag is injected alongside the bounds."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.ignore_frontend True" in extra, extra


def test_materialize_profile_cap_wins_over_a_max_iterations_pinned_in_the_yaml(
    tmp_path,
    monkeypatch,
):
    """The computed cap has to override a YAML-pinned value, not defer to it.

    The cap is a serialization-safe budget (``HYPERLOOM_PROFILE_MAX_STEPS_CAP`` /
    steady-floor); a hand-written ``max_iterations 100000`` is unbounded in practice.
    Injecting unconditionally and letting the repeated flag win last is what enforces
    that -- skipping injection because the name is already present hands the run the
    YAML value and silently discards the budget. ``HYPERLOOM_PROFILE_MAX_ITERS`` is
    the operator override channel, not the YAML.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_VLLM_ARGS": "--profiler-config.max_iterations 100000",
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in extra, extra
    # Last occurrence wins in vLLM's argparse, so the injected cap must come after.
    assert extra.rindex("max_iterations 128") > extra.rindex("max_iterations 100000"), extra


def test_materialize_profile_capture_flag_wins_over_a_stale_yaml_value(
    tmp_path,
    monkeypatch,
):
    """A YAML that disables graph-capture profiling must not win over the injected True."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_VLLM_ARGS": "--profiler-config.capture_torch_profiler False",
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.capture_torch_profiler True" in extra, extra
    assert extra.rindex("capture_torch_profiler True") > extra.rindex("capture_torch_profiler False"), extra


def test_materialize_profile_annotation_flag_wins_over_a_stale_yaml_value(
    tmp_path,
    monkeypatch,
):
    """A YAML that disables the annotation would leave the trace unlabelled,
    so the injected value has to land after it and win the last-wins resolution."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_VLLM_ARGS": "--profiler-config.detailed_trace_annotation False",
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.detailed_trace_annotation True" in extra, extra
    assert extra.rindex("detailed_trace_annotation True") > extra.rindex("detailed_trace_annotation False"), extra


def test_materialize_profile_restore_rejects_a_zero_max_iterations(
    tmp_path,
    monkeypatch,
    caplog,
):
    """``max_iterations 0`` is vLLM's own spelling of "no limit".

    Matching the flag by name alone accepted it, so the guard logged that it had made
    the profiler bounded while the run stayed unbounded -- worse than not guarding,
    because the warning sends the next investigation the wrong way.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    caplog.set_level("WARNING")
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args="--profiler-config.max_iterations 0",
        args_mode="replace",
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert extra.rindex("max_iterations 128") > extra.rindex("max_iterations 0"), extra
    assert "lost torch-profiler flags" in caplog.text


def test_materialize_profile_restore_rejects_a_max_iterations_above_the_cap(
    tmp_path,
    monkeypatch,
):
    """Above the serialization-safe cap is unbounded in every way that matters."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args="--profiler-config.max_iterations 100000",
        args_mode="replace",
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert extra.rindex("max_iterations 128") > extra.rindex("max_iterations 100000"), extra


def test_materialize_profile_restore_rejects_ignore_frontend_false(
    tmp_path,
    monkeypatch,
):
    """A frontend profiler left on tracks no iterations and captures the whole range;
    that is how an API-server process became an OOM victim."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args="--profiler-config.ignore_frontend False",
        args_mode="replace",
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.ignore_frontend True" in extra, extra
    assert extra.rindex("ignore_frontend True") > extra.rindex("ignore_frontend False"), extra


def test_materialize_profile_restore_accepts_a_bound_that_already_holds(
    tmp_path,
    monkeypatch,
    caplog,
):
    """A candidate carrying a valid in-cap bound is left alone and logs nothing."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    caplog.set_level("WARNING")
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_envs={
            "EXTRA_VLLM_ARGS": (
                "--profiler-config.delay_iterations 6080 "
                "--profiler-config.max_iterations 64 "
                "--profiler-config.ignore_frontend True "
                "--profiler-config.capture_torch_profiler True "
                "--profiler-config.detailed_trace_annotation True"
            ),
        },
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert extra.count("--profiler-config.max_iterations") == 1, extra
    assert "--profiler-config.max_iterations 64" in extra, extra
    assert "lost torch-profiler flags" not in caplog.text


def test_materialize_profile_bounds_outlive_remove_args(
    tmp_path,
    monkeypatch,
    caplog,
):
    """``remove_args`` runs after the merges, so the re-assertion has to be the last write.

    The two arrive together in practice: ``args_mode="replace"`` exists precisely
    because the KEEP carried ``remove_args`` (profile.py copies both off
    ``base_*``), so a restore that lands before ``remove_server_args`` can be
    undone by it -- while still logging that it restored the bounds.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    caplog.set_level("WARNING")
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args="--no-enable-prefix-caching",
        args_mode="replace",
        remove_args=["--profiler-config.max_iterations"],
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert "lost torch-profiler flags" in caplog.text


def test_materialize_profile_restores_max_iterations_even_when_delay_survives(
    tmp_path,
    monkeypatch,
):
    """``delay_iterations`` is a bad sentinel: it is ``max_iterations`` that bounds the capture.

    A candidate that happens to carry a delay flag used to satisfy the guard and
    leave the run with no cap at all -- exactly the unbounded profiler this is
    meant to prevent.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_server_args="--profiler-config.delay_iterations 0",
        args_mode="replace",
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert "--profiler-config.ignore_frontend True" in extra, extra
    assert "--profiler-config.capture_torch_profiler True" in extra, extra
    assert "--profiler-config.detailed_trace_annotation True" in extra, extra
    # The candidate's own delay value is left alone; only the missing flags return.
    assert extra.count("--profiler-config.delay_iterations") == 1, extra


def test_materialize_profile_restore_does_not_duplicate_surviving_flags(
    tmp_path,
    monkeypatch,
):
    """The restore re-states only what is missing; vLLM warns on duplicate keys."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(
        src,
        tmp_path,
        extra_envs={
            "EXTRA_VLLM_ARGS": "--profiler-config.ignore_frontend True --quantization fp8",
        },
    )
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert extra.count("--profiler-config.ignore_frontend True") == 1, extra
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert "--quantization fp8" in extra, extra


def test_materialize_profile_window_sglang_skill_formula(
    tmp_path,
    monkeypatch,
):
    """SGLang path writes the same window into PROFILE_EXTRA_BODY."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    body = json.loads(rendered["benchmark"]["envs"]["PROFILE_EXTRA_BODY"])
    # max capped at 128; delay = 1024 * 2 * 3 - 128/2 = 6080.
    assert body["start_step"] == 6080
    assert body["num_steps"] == 128


def test_materialize_profile_agentx_clamp_warns_below_steady_floor(
    tmp_path,
    monkeypatch,
    caplog,
):
    """AgentX's tighter capture cap (8) must warn when it undercuts steady_floor.

    CONC=32/OSL=1024/R=1.0 -> steady_floor=ceil(1024*2/64)=32, far above the
    AgentX cap of 8. The manual HYPERLOOM_PROFILE_MAX_ITERS override already
    warns in this situation; the AgentX auto-clamp must match it instead of
    silently capturing a trace with no steady-state window.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    with caplog.at_level("WARNING"):
        out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 8" in extra, extra
    assert any("steady-state floor" in r.message for r in caplog.records)


def test_materialize_profile_agentx_clamp_warns_on_explicit_override(
    tmp_path,
    monkeypatch,
    caplog,
):
    """An explicit HYPERLOOM_PROFILE_MAX_STEPS_CAP must not be silently overridden.

    Without this, an operator who explicitly raised the cap (e.g. to widen the
    profiler's steady-state window) would see it clamped back to 8 by the
    AgentX branch with no indication their override had no effect.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_STEPS_CAP", "64")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    with caplog.at_level("WARNING"):
        out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 8" in extra, extra
    assert any("explicit HYPERLOOM_PROFILE_MAX_STEPS_CAP=64" in r.message for r in caplog.records)


def test_materialize_profile_max_iters_override_warns_it_undoes_the_agentx_bound(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Overriding the AgentX capture bound must say so -- 128 warns about nothing else.

    HYPERLOOM_PROFILE_MAX_ITERS is applied after the AgentX clamp and wins, so
    it restores exactly the host-RAM exposure the clamp exists to remove. The
    two pre-existing warnings cannot cover this: ``cap`` defaults to
    _DEFAULT_PROFILE_MAX_STEPS (128), so an override of 128 is neither below
    steady_floor's band nor above the cap, and the bound would be lifted in
    silence.
    """
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "128")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    with caplog.at_level("WARNING"):
        out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    # The override is still honored verbatim; this is a visibility fix only.
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert any(
        "HYPERLOOM_PROFILE_MAX_ITERS=128 overrides the AgentX capture bound of 8" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_materialize_profile_max_iters_override_is_quiet_without_agentx(
    tmp_path,
    monkeypatch,
    caplog,
):
    """The new warning is AgentX-only; the synthetic path has no host-RAM bound."""
    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "128")
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    with caplog.at_level("WARNING"):
        _materialize_config_with_envs(src, tmp_path)
    assert not any("AgentX capture bound" in r.message for r in caplog.records)


def test_materialize_persists_inferencex_path_for_magpie(
    tmp_path,
    monkeypatch,
):
    """$INFERENCEX_PATH must be written into benchmark.inferencex_path so Magpie uses the patched checkout."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("INFERENCEX_PATH", "/path/InferenceX")
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    assert rendered["benchmark"]["inferencex_path"] == "/path/InferenceX"


def test_materialize_profile_window_clamps_to_skill_floor(
    tmp_path,
    monkeypatch,
):
    """Capture is always the serialization cap (default 128), even for a small
    OSL whose steady floor is far below it (OSL=256, CONC=64 ⇒ floor=4)."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 64, "ISL": 256, "OSL": 256})
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    extra = rendered["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in extra, extra


# NUM_PROMPTS must cover the steady-state window (profile mode force-overrides caller values).
def test_materialize_profile_num_prompts_covers_steady_state_window(
    tmp_path,
    monkeypatch,
):
    """OSL=1024 / CONC=32 / R=1 → delay+max = 6208 iters ⇒ NUM_PROMPTS=388."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # delay=6080, max=128 → required=6208; floor(6208*32/1024)=194; *2 = 388.
    assert envs["NUM_PROMPTS"] == 388, envs.get("NUM_PROMPTS")


def test_materialize_profile_num_prompts_floors_at_conc_for_tiny_osl(
    tmp_path,
    monkeypatch,
):
    """Tiny OSL with the capped capture still produces a sane NUM_PROMPTS."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 64, "OSL": 64})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # max=128, delay=64*2*3-64=320, required=448; 448*32/64=224; *2=448.
    assert envs["NUM_PROMPTS"] == 448, envs.get("NUM_PROMPTS")


def test_materialize_profile_force_overrides_user_num_prompts(
    tmp_path,
    monkeypatch,
):
    """Profile mode must IGNORE caller-supplied NUM_PROMPTS — an
    under-sized value (skill default `max_concurrency * 1`) would
    silently empty the trace."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = _profile_yaml(
        tmp_path,
        "vllm",
        # Caller deliberately under-sizes to trip the regression.
        {"CONC": 32, "ISL": 256, "OSL": 1024, "NUM_PROMPTS": 32},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # Hyperloom-computed 388 must win over the caller's 32.
    assert envs["NUM_PROMPTS"] == 388, envs.get("NUM_PROMPTS")


def test_materialize_non_profile_keeps_legacy_seq_cost_factor(
    tmp_path,
    monkeypatch,
):
    """The profile-only NUM_PROMPTS override does not affect baseline / sweep paths (they keep the seq_cost-based value)."""
    import yaml

    _clear_workload_env(monkeypatch)
    src = tmp_path / "baseline.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "/m",
                    "envs": {"CONC": 32, "ISL": 256, "OSL": 1024},
                    # No profiler.torch_profiler.enabled, no PROFILE=1.
                },
            }
        )
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    # seq_cost=1280 → factor=5 → CONC*5 = 160 (legacy baseline path).
    assert envs["NUM_PROMPTS"] == 160, envs.get("NUM_PROMPTS")


def _mock_patchers(monkeypatch, *, vllm: bool, sglang: bool) -> dict[str, int]:
    """Replace the two patcher symbols on `_workload_envs` with stubs that record invocation counts for per-framework dispatch asserts."""
    from hyperloom.orchestrator.actions.executors import _workload_envs

    counts = {"vllm": 0, "sglang": 0}

    def _vllm_stub() -> bool:
        counts["vllm"] += 1
        return vllm

    def _sglang_stub() -> bool:
        counts["sglang"] += 1
        return sglang

    monkeypatch.setattr(
        _workload_envs,
        "ensure_vllm_patched_for_tracelens",
        _vllm_stub,
    )
    monkeypatch.setattr(
        _workload_envs,
        "ensure_sglang_patched_for_tracelens",
        _sglang_stub,
    )
    return counts


def test_materialize_profile_vllm_injects_tracelens_flags_when_patched(
    tmp_path,
    monkeypatch,
):
    """Patcher True for vLLM ⇒ EXTRA_VLLM_ARGS gains capture_torch_profiler and detailed_trace_annotation."""
    import yaml

    _clear_workload_env(monkeypatch)
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    assert "--profiler-config.max_iterations 128" in extra, extra
    assert "--profiler-config.capture_torch_profiler True" in extra, extra
    assert "--profiler-config.detailed_trace_annotation True" in extra, extra
    # Per-framework dispatch: the SGLang patcher must NOT run for a vLLM YAML.
    assert counts == {"vllm": 1, "sglang": 0}, counts


def test_materialize_profile_vllm_omits_tracelens_flags_when_patch_fails(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Patcher False ⇒ EXTRA_VLLM_ARGS keeps only the default safe set (else unpatched vLLM crashes on unknown JSON key)."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    caplog.set_level("WARNING")
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    extra = envs["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    assert "capture_torch_profiler" not in extra, extra
    assert "detailed_trace_annotation" not in extra, extra
    assert envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "unavailable"
    assert envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] == "tracelens_runtime_patch_unavailable"
    assert "TraceLens runtime patch unavailable" in caplog.text


def test_materialize_profile_sglang_injects_shape_discovery_when_patched(
    tmp_path,
    monkeypatch,
):
    """Patcher returns True for SGLang ⇒ EXTRA_SGLANG_ARGS gains
    --enable-shape-discovery-for-cuda-graph-profile."""
    import yaml

    _clear_workload_env(monkeypatch)
    counts = _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS",
        "",
    )
    assert "--enable-shape-discovery-for-cuda-graph-profile" in extra, extra
    # Per-framework dispatch in reverse: the vLLM patcher must NOT be
    # invoked when the YAML's framework is SGLang.
    assert counts == {"vllm": 0, "sglang": 1}, counts


def test_materialize_profile_sglang_omits_shape_discovery_when_patch_fails(
    tmp_path,
    monkeypatch,
):
    """Patcher returns False ⇒ no shape-discovery flag (otherwise
    SGLang argparse errors on the unknown flag)."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS",
        "",
    )
    assert "shape-discovery" not in extra, extra


def test_materialize_profile_sglang_drops_annotations_when_patch_fails(
    tmp_path,
    monkeypatch,
):
    """A failed patch also clears the annotation-only capture options.

    Without the server-side patch the trace carries no ``kernel_shape_profiler``
    events, so requesting shape discovery / detailed annotations only pays the
    capture cost. The vLLM branch already drops its equivalent flag."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] == "tracelens_runtime_patch_unavailable"
    body = json.loads(envs["PROFILE_EXTRA_BODY"])
    assert body["shape_discovery"] is False, body
    assert body["detailed_annotations"] is False, body


def test_materialize_profile_sglang_keeps_annotations_when_patch_succeeds(
    tmp_path,
    monkeypatch,
):
    """The healthy path is untouched: annotations stay on when the patch lands."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "HYPERLOOM_PROFILE_DEGRADED_REASON" not in envs
    body = json.loads(envs["PROFILE_EXTRA_BODY"])
    assert body["shape_discovery"] is True, body
    assert body["detailed_annotations"] is True, body


def test_materialize_profile_sglang_keeps_annotations_when_patch_not_attempted(
    tmp_path,
    monkeypatch,
):
    """HYPERLOOM_ENABLE_PATCH=0 must not degrade the capture options.

    Patching disabled is not the same as patching failed: the image may ship the
    TraceLens patch already applied, in which case the annotations still work."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    counts = _mock_patchers(monkeypatch, vllm=False, sglang=False)
    src = _profile_yaml(tmp_path, "sglang", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert counts == {"vllm": 0, "sglang": 0}, counts
    assert "HYPERLOOM_PROFILE_DEGRADED_REASON" not in envs
    body = json.loads(envs["PROFILE_EXTRA_BODY"])
    assert body["shape_discovery"] is True, body
    assert body["detailed_annotations"] is True, body


def test_materialize_profile_sglang_drops_graph_capture_flag_when_eager(
    tmp_path,
    monkeypatch,
):
    """``--disable-cuda-graph`` and ``--enable-profile-cuda-graph`` contradict.

    The eager flag arrives via ``extra_server_args`` while the graph-capture
    profiling flag comes from the profile YAML, so the two only meet after the
    merges. An eager server captures no graph, leaving nothing to profile."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(
        tmp_path,
        "sglang",
        {"CONC": 32, "ISL": 256, "OSL": 1024, "EXTRA_SGLANG_ARGS": "--enable-profile-cuda-graph"},
    )
    out = _materialize_config_with_envs(src, tmp_path, extra_server_args="--disable-cuda-graph")
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--disable-cuda-graph" in extra.split(), extra
    assert "--enable-profile-cuda-graph" not in extra.split(), extra


def test_materialize_profile_sglang_keeps_graph_capture_flag_without_eager(
    tmp_path,
    monkeypatch,
):
    """Graph-mode profiling keeps the capture flag (the healthy path)."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(
        tmp_path,
        "sglang",
        {"CONC": 32, "ISL": 256, "OSL": 1024, "EXTRA_SGLANG_ARGS": "--enable-profile-cuda-graph"},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--enable-profile-cuda-graph" in extra.split(), extra


def test_materialize_profile_kill_switch_skips_patcher_entirely(
    tmp_path,
    monkeypatch,
):
    """HYPERLOOM_ENABLE_PATCH=0 short-circuits the patcher entirely; no TraceLens-only flags land in the YAML."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=True)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    # Safe profiler flags still present.
    assert "--profiler-config.delay_iterations 6080" in extra, extra
    # TraceLens-only flags absent.
    assert "detailed_trace_annotation" not in extra, extra
    # Patchers never invoked.
    assert counts == {"vllm": 0, "sglang": 0}, counts


def test_materialize_profile_kill_switch_default_is_on(
    tmp_path,
    monkeypatch,
):
    """Unset HYPERLOOM_ENABLE_PATCH == default-on; the patcher must be invoked."""
    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    counts = _mock_patchers(monkeypatch, vllm=True, sglang=False)
    src = _profile_yaml(tmp_path, "vllm", {"CONC": 32, "ISL": 256, "OSL": 1024})
    _materialize_config_with_envs(src, tmp_path)
    assert counts["vllm"] == 1, counts


def test_materialize_profile_sglang_does_not_duplicate_shape_discovery(
    tmp_path,
    monkeypatch,
):
    """If EXTRA_SGLANG_ARGS already has --enable-shape-discovery-for-cuda-graph-profile, the materializer must NOT duplicate it."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    src = _profile_yaml(
        tmp_path,
        "sglang",
        {
            "CONC": 32,
            "ISL": 256,
            "OSL": 1024,
            "EXTRA_SGLANG_ARGS": ("--enable-profile-cuda-graph --enable-shape-discovery-for-cuda-graph-profile"),
        },
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert extra.count("--enable-shape-discovery-for-cuda-graph-profile") == 1, extra


def _profile_yaml_model(tmp_path, framework: str, model: str, envs: dict) -> Path:
    """Like _profile_yaml but with an explicit model path (for Gemma2 gating)."""
    import yaml as _yaml

    src = tmp_path / f"src_{framework}_model.yaml"
    src.write_text(
        _yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": model,
                    "envs": {"PROFILE": "1", **envs},
                    "profiler": {"torch_profiler": {"enabled": True}},
                },
            }
        )
    )
    return src


def test_materialize_profile_sglang_skips_shape_discovery_for_gemma2(
    tmp_path,
    monkeypatch,
):
    """Gemma2 + patched SGLang must NOT inject shape-discovery (it crashes
    CUDA-graph capture); --enable-profile-cuda-graph still applies."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "gemma2_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma2",
                "architectures": ["Gemma2ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        str(model),
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "shape-discovery" not in envs.get("EXTRA_SGLANG_ARGS", ""), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is False


def test_materialize_profile_sglang_keeps_shape_discovery_for_non_gemma2(
    tmp_path,
    monkeypatch,
):
    """A non-Gemma2 model still gets shape-discovery when patched."""
    import yaml

    _clear_workload_env(monkeypatch)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "llama_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        str(model),
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS",
        "",
    )
    assert "--enable-shape-discovery-for-cuda-graph-profile" in extra, extra


def test_materialize_profile_sglang_skips_shape_discovery_gemma2_by_path(
    tmp_path,
    monkeypatch,
):
    """No config.json but a gemma-2 path -> heuristic skips shape-discovery."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE", raising=False)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    # Path looks like Gemma2 but ships no config.json (not-yet-materialized).
    model = "/path/models/google-gemma-2-9b-it"
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        model,
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "shape-discovery" not in envs.get("EXTRA_SGLANG_ARGS", ""), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is False


def test_materialize_profile_sglang_no_config_non_gemma_keeps_shape_discovery(
    tmp_path,
    monkeypatch,
):
    """No config.json and a non-Gemma2 path -> shape-discovery stays on."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE", raising=False)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = "/path/models/meta-llama-3-8b-instruct"
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        model,
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    extra = yaml.safe_load(out.read_text())["benchmark"]["envs"].get(
        "EXTRA_SGLANG_ARGS",
        "",
    )
    assert "--enable-shape-discovery-for-cuda-graph-profile" in extra, extra


def test_materialize_profile_sglang_skips_shape_discovery_nested_gemma2(
    tmp_path,
    monkeypatch,
):
    """Gemma2 declared only in text_config still trips the shape-discovery gate."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE", raising=False)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "wrapper_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "wrapper",
                "text_config": {"model_type": "gemma2"},
            }
        ),
        encoding="utf-8",
    )
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        str(model),
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "shape-discovery" not in envs.get("EXTRA_SGLANG_ARGS", ""), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is False


def test_materialize_profile_sglang_residual_config_gemma2_path(
    tmp_path,
    monkeypatch,
):
    """Empty config.json + gemma-2 path -> heuristic still skips shape-discovery."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE", raising=False)
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "google-gemma-2-9b-it"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        str(model),
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "shape-discovery" not in envs.get("EXTRA_SGLANG_ARGS", ""), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is False


def test_materialize_profile_sglang_force_overrides_gemma2_gate(
    tmp_path,
    monkeypatch,
):
    """HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 keeps shape-discovery on for
    Gemma2 (escape hatch for debugging the TraceLens root-cause fix)."""
    import yaml

    _clear_workload_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE", "1")
    _mock_patchers(monkeypatch, vllm=False, sglang=True)
    model = tmp_path / "gemma2_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma2",
                "architectures": ["Gemma2ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    src = _profile_yaml_model(
        tmp_path,
        "sglang",
        str(model),
        {"CONC": 32, "ISL": 256, "OSL": 1024},
    )
    out = _materialize_config_with_envs(src, tmp_path)
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "--enable-shape-discovery-for-cuda-graph-profile" in envs.get(
        "EXTRA_SGLANG_ARGS",
        "",
    ), envs
    assert json.loads(envs["PROFILE_EXTRA_BODY"])["shape_discovery"] is True


def test_profile_executor_calls_benchmark_lib_patcher():
    """ProfileExecutor must patch the materialized InferenceX checkout before launching Magpie (else the computed profile window is stomped and the trace is empty)."""
    from hyperloom.orchestrator.actions.executors import profile as profile_mod

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
    raw = "--enable-torch-compile --torch-compile-max-bs 32 --quantization fp8 --foo=bar --torch-compile-max-bs=64"

    sanitized = _sanitize_profile_server_args(raw)

    assert "--enable-torch-compile" not in sanitized
    assert "--torch-compile-max-bs" not in sanitized
    assert "--quantization fp8" in sanitized
    assert "--foo=bar" in sanitized


def test_profile_server_args_sanitizer_preserves_json_value_quotes():
    """Regression: embedded JSON values (e.g. --speculative-config) must keep
    their inner double-quotes. POSIX shlex.split would strip them, yielding the
    unparseable {method:...} and failing every profile/roofline server boot."""
    spec = '--speculative-config {"method":"deepseek_mtp","num_speculative_tokens":1}'
    assert _sanitize_profile_server_args(spec) == spec

    mixed = spec + " --enable-torch-compile --torch-compile-max-bs 8"
    sanitized = _sanitize_profile_server_args(mixed)
    assert '{"method":"deepseek_mtp","num_speculative_tokens":1}' in sanitized
    assert "--enable-torch-compile" not in sanitized
    assert "--torch-compile-max-bs" not in sanitized


def test_profile_server_args_sanitizer_degrades_on_unbalanced_quote():
    """An unbalanced quote must not raise; the function falls back to whitespace split."""
    result = _sanitize_profile_server_args("--foo 'unterminated --bar baz")
    assert isinstance(result, str)
    assert "--foo" in result
    assert "--bar" in result


# $FRAMEWORK env switches the default yaml between sglang/vllm without an explicit config_path.
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
    """FRAMEWORK=atom selects baseline_atom.yaml (the single-source-of-truth selector for every executor)."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    assert _default_baseline_config().name == "baseline_atom.yaml"


def test_server_args_env_name_atom():
    """atom maps to EXTRA_ATOM_ARGS (the atom branch sits before vllm to avoid substring collisions)."""
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        server_args_env_name,
    )

    assert server_args_env_name("atom") == "EXTRA_ATOM_ARGS"
    assert server_args_env_name("ATOM") == "EXTRA_ATOM_ARGS"
    assert server_args_env_name("vllm") == "EXTRA_VLLM_ARGS"
    assert server_args_env_name("sglang") == "EXTRA_SGLANG_ARGS"


def test_materialize_config_atom_profile_skips_tracelens_flags(
    tmp_path,
    monkeypatch,
):
    """PROFILE=1 + framework=atom must NOT inject sglang/vllm profiler CLI flags into EXTRA_ATOM_ARGS (atom's argparse rejects them)."""
    import yaml

    monkeypatch.setenv("FRAMEWORK", "atom")
    monkeypatch.setenv("PROFILE", "1")
    src = _default_baseline_config()  # baseline_atom.yaml
    out = _materialize_config_with_envs(src, tmp_path)
    rendered = yaml.safe_load(out.read_text())
    envs = rendered["benchmark"]["envs"]
    extra = str(envs.get("EXTRA_ATOM_ARGS", ""))
    assert "--profiler-config" not in extra, f"atom EXTRA_ATOM_ARGS leaked sglang/vllm profiler flag: {extra!r}"
    # --trust-remote-code from the baseline YAML must survive untouched.
    assert "--trust-remote-code" in extra, f"atom EXTRA_ATOM_ARGS lost base --trust-remote-code: {extra!r}"


def test_default_profile_config_tracks_framework(monkeypatch):
    monkeypatch.setenv("FRAMEWORK", "vllm")
    assert _default_profile_config().name == "profile_vllm.yaml"
    monkeypatch.setenv("FRAMEWORK", "custom")
    assert _default_profile_config().name == "profile_custom.yaml"
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
    """FRAMEWORK=atom falls through to the normal profile path (the atom Magpie wrapper bridges PROFILE=1 to atom's torch profiler)."""
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
            "base_extra_args": ("--enable-torch-compile --torch-compile-max-bs 32 --quantization fp8"),
        },
        task_id="t-profile-sanitize",
    )
    ctx = SimpleNamespace(task=task, extra={"workspace": str(tmp_path / "ws")})

    result = asyncio.run(ProfileExecutor()(ctx))

    assert result["status"] == "succeeded"
    merged = captured["extra_server_args"]
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
            "extra_server_args": ("--enable-torch-compile --torch-compile-max-bs 32 --quantization fp8"),
        },
        task_id="t-profile-canonical-sanitize",
    )
    ctx = SimpleNamespace(task=task, extra={"workspace": str(tmp_path / "ws")})

    result = asyncio.run(ProfileExecutor()(ctx))

    assert result["status"] == "succeeded"
    merged = captured["extra_server_args"]
    assert "--enable-torch-compile" not in merged
    assert "--torch-compile-max-bs" not in merged
    assert "--attention-backend AITER" in merged
    assert "--quantization fp8" in merged


def test_profile_executor_merges_current_best_envs(monkeypatch, tmp_path):
    """A refreshed Roofline must launch with the backend env selected by Explore."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    captured: dict[str, object] = {}

    async def _fake_parent(self, ctx):
        captured.update(ctx.task.params)
        return {"status": "succeeded"}

    monkeypatch.setattr(BaselineExecutor, "__call__", _fake_parent)
    task = SimpleNamespace(
        params={
            "base_extra_envs": {
                "VLLM_ROCM_USE_AITER": "1",
                "SHARED": "base",
            },
            "extra_envs": {
                "VLLM_ROCM_USE_AITER_LINEAR": "1",
                "SHARED": "caller",
            },
        },
        task_id="t-profile-envs",
    )
    ctx = SimpleNamespace(task=task, extra={"workspace": str(tmp_path / "ws")})

    result = asyncio.run(ProfileExecutor()(ctx))

    assert result["status"] == "succeeded"
    assert captured["extra_envs"] == {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "SHARED": "caller",
    }


@pytest.mark.asyncio
async def test_roofline_executor_skips_when_framework_atom(monkeypatch):
    """FRAMEWORK=atom attempts the normal roofline profile sub-step."""
    from hyperloom.orchestrator.actions.executors.roofline import (
        RooflineExecutor,
    )

    monkeypatch.setenv("FRAMEWORK", "atom")
    rexec = RooflineExecutor(shared_state=SimpleNamespace())

    # Sentinel: prove the lazy import / sub-step orchestration is reached.
    from hyperloom.orchestrator.actions.executors import profile as profile_mod

    async def _explode(_ctx):
        raise AssertionError("profile_executor sentinel: sub-step reached under atom")

    monkeypatch.setattr(profile_mod, "profile_executor", _explode)

    task = SimpleNamespace(
        params={},
        task_id="t-atom-roofline",
        idempotency_key="t-atom-roofline",
        requires_lanes=[],
        side_effects=[],
        lease_ttl_sec=0,
    )
    ctx = SimpleNamespace(task=task, lease=None, extra=None)

    result = await rexec(ctx)
    assert result["status"] == "failed"
    assert result["phase"] == "profile"
    assert "profile_executor raised" in result["error"]


@pytest.mark.asyncio
async def test_baseline_executor_fails_on_nonzero_rc_despite_valid_measurement(tmp_path):
    """A parseable measurement must not launder a non-zero process exit into success.

    The round cannot be the number a later comparison anchors to. Same contract
    ``run_grid`` enforces for a variant.
    """
    db = SqliteConnection(tmp_path / "baseline.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    # The workspace must be created inside the fake to count as this run's output.
    report_body = json.dumps(
        {
            "success": False,
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "throughput": {
                "request_throughput": 1.8,
                "output_throughput": 1872.0,
                "total_token_throughput": 3744.0,
                "completed_requests": 320,
                "duration_seconds": 177.0,
            },
            "latency": {"ttft": {"mean_ms": 140}, "e2el": {"mean_ms": 2500}},
        }
    )

    def fake_run(cmd, *args, **kwargs):
        ws = output_dir / "benchmark_sglang_20260501_001122"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(report_body)
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="cleanup failed")

    task = await tr.create(
        kind="baseline",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="baseline-valid-warning",
    )
    sub.register_executor("baseline", BaselineExecutor(session_dir=tmp_path))
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=fake_run):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "magpie_nonzero_after_valid_measurement"
    assert res.result["returncode"] == 1
    assert "cleanup failed" in res.result["error"]
    assert res.result["reported_success"] is False
    assert "output_throughput" not in res.result
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

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    # The workspace must be created inside the fake to count as this run's output.
    ws_name = "benchmark_sglang_20260501_001122"

    async def _fake_baseline(_self, _ctx):
        workspace = output_dir / ws_name
        workspace.mkdir(parents=True, exist_ok=True)
        report_path = workspace / "benchmark_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "sglang",
                    "model": "/path/models/Qwen-Qwen3-8B",
                    "throughput": {
                        "request_throughput": 3.2,
                        "output_throughput": 800.0,
                        "total_token_throughput": 1600.0,
                        "completed_requests": 80,
                        "duration_seconds": 25.0,
                    },
                    "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158}, "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
                }
            )
        )
        trace_dir = workspace / "torch_trace"
        trace_dir.mkdir()
        (trace_dir / "177-TP-0-DECODE.trace.json.gz").write_bytes(b"fake-trace")
        (trace_dir / "merged-177.trace.json.gz").write_bytes(b"fake-trace")
        return {
            "status": "succeeded",
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "output_throughput": 800.0,
            "workspace": str(workspace),
            "report_path": str(report_path),
        }

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-1",
    )
    sub.register_executor("profile", pe)
    with patch.object(BaselineExecutor, "__call__", _fake_baseline):
        res = await sub.run_task(task)

    workspace = output_dir / ws_name
    trace_dir = workspace / "torch_trace"
    merged_trace = trace_dir / "merged-177.trace.json.gz"
    assert res.state == "succeeded"
    assert res.result["framework"] == "sglang"
    assert res.result["trace_dir"] == str(trace_dir)
    assert len(res.result["trace_files"]) == 2
    assert res.result["main_trace_path"] == str(merged_trace)
    assert res.result["profile_trace_selection_reason"] == "merged_trace_preferred"
    db.close()


@pytest.mark.asyncio
async def test_agentx_profile_executor_passes_rank_zero_not_merged(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("TP", "2")
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _fake_baseline(_self, _ctx):
        workspace = output_dir / "benchmark_sglang_agentx"
        trace_dir = workspace / "torch_trace"
        trace_dir.mkdir(parents=True)
        rank_zero = trace_dir / "177-TP-0-DECODE.trace.json.gz"
        rank_one = trace_dir / "177-TP-1-DECODE.trace.json.gz"
        merged = trace_dir / "merged-177.trace.json.gz"
        rank_zero.write_bytes(b"rank-zero")
        rank_one.write_bytes(b"rank-one")
        merged.write_bytes(b"merged")
        capture_status = Path(_ctx.task.params["extra_envs"]["AGENTX_CAPTURE_STATUS_PATH"])
        capture_status.write_text(
            json.dumps(
                {
                    "capture_id": _ctx.task.params["extra_envs"]["AGENTX_CAPTURE_ID"],
                    "status": "succeeded",
                    "reason": "capture_complete",
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "succeeded",
            "framework": "sglang",
            "workspace": str(workspace),
            "submission_valid": True,
        }

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-agentx-rank-zero",
    )
    sub.register_executor("profile", pe)
    with patch.object(BaselineExecutor, "__call__", _fake_baseline):
        res = await sub.run_task(task)

    trace_dir = output_dir / "benchmark_sglang_agentx" / "torch_trace"
    assert res.result["status"] == "succeeded"
    assert res.result["main_trace_path"] == str(trace_dir / "177-TP-0-DECODE.trace.json.gz")
    assert res.result["primary_rank"] == 0
    assert res.result["profile_trace_selection_reason"] == "primary_rank_trace"
    assert res.result["merged_trace_paths"] == [str(trace_dir / "merged-177.trace.json.gz")]
    assert sorted(res.result["rank_trace_paths"]) == ["0", "1"]
    assert res.result["trace_capture_status"] == "succeeded"
    capture_status_path = Path(res.result["trace_capture_status_path"])
    assert capture_status_path.parent.parent == output_dir / "agentx-profile"
    manifest = json.loads(Path(res.result["trace_manifest_path"]).read_text())
    assert manifest["capture_id"] == capture_status_path.parent.name
    assert manifest["primary_trace_path"] == res.result["main_trace_path"]
    db.close()


@pytest.mark.asyncio
async def test_profile_executor_surfaces_failed_agentx_capture_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _fake_baseline(_self, _ctx):
        workspace = output_dir / "benchmark_sglang_capture_failed"
        trace_dir = workspace / "torch_trace"
        trace_dir.mkdir(parents=True)
        (trace_dir / "rank-0.trace.json.gz").write_bytes(b"partial")
        capture_status = Path(_ctx.task.params["extra_envs"]["AGENTX_CAPTURE_STATUS_PATH"])
        capture_status.write_text(
            json.dumps(
                {
                    "capture_id": _ctx.task.params["extra_envs"]["AGENTX_CAPTURE_ID"],
                    "status": "failed",
                    "reason": "trace_flush_failed",
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "succeeded",
            "framework": "sglang",
            "workspace": str(workspace),
        }

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-capture-failed",
    )
    sub.register_executor("profile", pe)
    with patch.object(BaselineExecutor, "__call__", _fake_baseline):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "profile_capture_failed"
    assert res.result["measurement_status"] == "succeeded"
    assert res.result["trace_capture_status"] == "failed"
    assert res.result["trace_capture"]["reason"] == "trace_flush_failed"
    db.close()


@pytest.mark.asyncio
async def test_agentx_profile_executor_rejects_missing_capture_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _fake_baseline(_self, _ctx):
        workspace = output_dir / "benchmark_sglang_missing_status"
        trace_dir = workspace / "torch_trace"
        trace_dir.mkdir(parents=True)
        (trace_dir / "rank-0.trace.json.gz").write_bytes(b"trace")
        return {
            "status": "succeeded",
            "framework": "sglang",
            "workspace": str(workspace),
            "submission_valid": True,
        }

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-capture-status-missing",
    )
    sub.register_executor("profile", pe)
    with patch.object(BaselineExecutor, "__call__", _fake_baseline):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "profile_capture_failed"
    assert res.result["measurement_status"] == "succeeded"
    assert res.result["trace_capture_status"] == "missing"
    assert res.result["trace_capture"]["reason"] == "capture_status_missing"
    db.close()


@pytest.mark.asyncio
async def test_agentx_profile_preserves_pre_capture_failure_for_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def _fake_baseline(_self, _ctx):
        workspace = output_dir / "benchmark_sglang_failed"
        stale_trace_dir = workspace / "torch_trace"
        stale_trace_dir.mkdir(parents=True)
        stale_trace = stale_trace_dir / "rank-0.trace.json.gz"
        stale_trace.write_bytes(b"stale")
        os.utime(stale_trace, (1, 1))
        return {
            "status": "failed",
            "error_class": "cuda_graph_capture_failed",
            "error": "CUDA graph capture failed before AgentX capture started",
            "workspace": str(workspace),
        }

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-before-capture-failed",
    )
    sub.register_executor("profile", pe)
    with patch.object(BaselineExecutor, "__call__", _fake_baseline):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "cuda_graph_capture_failed"
    assert res.result["measurement_status"] == "failed"
    assert res.result["trace_capture_status"] == "not_reached"
    assert res.result["trace_input_ready"] is False
    assert "main_trace_path" not in res.result
    db.close()


@pytest.mark.asyncio
async def test_profile_executor_patches_configured_inferencex_path(
    tmp_path,
    monkeypatch,
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
    output_dir.mkdir(parents=True)

    report_body = json.dumps(
        {
            "success": True,
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "throughput": {
                "request_throughput": 3.2,
                "output_throughput": 800.0,
                "total_token_throughput": 1600.0,
                "completed_requests": 80,
                "duration_seconds": 25.0,
            },
            "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158}, "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
        }
    )

    def _fake_run_ix(cmd, *args, **kwargs):
        ws = output_dir / "benchmark_sglang_20260501_001122"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(report_body)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-inferencex-path",
    )
    sub.register_executor("profile", pe)
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run_ix):
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
    output_dir.mkdir(parents=True)

    def _fake_run(cmd, *args, **kwargs):
        workspace = output_dir / "benchmark_vllm_20260501_001122"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "vllm",
                    "model": "/path/models/Qwen-Qwen3-8B",
                    "throughput": {
                        "request_throughput": 3.2,
                        "output_throughput": 800.0,
                        "total_token_throughput": 1600.0,
                        "completed_requests": 80,
                        "duration_seconds": 25.0,
                    },
                    "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158}, "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
                }
            )
        )
        capture_dir = output_dir / "capture_traces"
        capture_dir.mkdir(exist_ok=True)
        (capture_dir / "graph_capture_rank_0.1.pt.trace.json.gz").write_bytes(b"fake-trace")
        (capture_dir / "graph_capture_rank_0.2.pt.trace.json.gz").write_bytes(b"fake-trace")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    pe = ProfileExecutor(session_dir=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-capture",
    )
    sub.register_executor("profile", pe)
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await sub.run_task(task)

    capture_dir = output_dir / "capture_traces"
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
        # Exercise the structured-result plumbing via the explicit bypass route
        # (the default is now the TraceLens agent route, which needs a real root).
        "analysis_route": "bypass",
    }
    res = await krh.trace_analyze_handler(payload, session_dir=session_dir)
    # Structured result surfaced verbatim by the bypass backend.
    assert res["status"] in ("ok", "succeeded", "failed")
    assert res.get("route") == "bypass"
    assert res.get("candidates_path") and "artifact_paths" in res


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
async def test_trace_analyze_handler_xdit_defaults_to_tracelens_agent(session_dir, monkeypatch):
    """With no explicit route, every framework (incl. xDiT) DEFAULTS to the
    TraceLens ``agent`` route (the shipped default); bypass is an explicit route."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "xdit",
            "model_name": "FLUX.1-dev",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert not any("bypass_trace_analysis.py" in c for c in cmd)
    assert "--analysis-route" in cmd and "agent" in cmd
    assert "--tracelens-root" in cmd
    assert "--skip-split" in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_xdit_state_overrides_stale_payload_framework(
    session_dir,
    monkeypatch,
):
    """A stale text-generation payload must not make xDiT split its raw trace."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "xdit"
    state.save(session_dir)

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
        },
        session_dir=session_dir,
    )

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--framework" in cmd and cmd[cmd.index("--framework") + 1] == "xdit"
    assert "--skip-split" in cmd
    assert "--analysis-mode" not in cmd
    warnings = res["trace_health_warnings"]
    assert warnings[0]["code"] == "stale_framework_overridden"
    assert warnings[0]["payload_framework"] == "sglang"
    assert warnings[0]["session_framework"] == "xdit"


@pytest.mark.asyncio
async def test_trace_analyze_handler_custom_state_overrides_stale_payload_framework(
    session_dir,
    monkeypatch,
):
    """All scriptable session frameworks preserve their raw trace."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "custom"
    state.save(session_dir)

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
        },
        session_dir=session_dir,
    )

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--framework" in cmd and cmd[cmd.index("--framework") + 1] == "custom"
    assert "--skip-split" in cmd
    assert res["trace_health_warnings"][0]["code"] == "stale_framework_overridden"


@pytest.mark.asyncio
async def test_trace_analyze_handler_payload_framework_overrides_serving_state(
    session_dir,
    monkeypatch,
):
    """Explicit serving-framework payloads keep their existing precedence."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "vllm"
    state.save(session_dir)

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
        },
        session_dir=session_dir,
    )

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--framework" in cmd and cmd[cmd.index("--framework") + 1] == "sglang"
    assert "--analysis-mode" in cmd and cmd[cmd.index("--analysis-mode") + 1] == "inference"
    assert "--skip-split" not in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_env_route_forces_bypass(session_dir, monkeypatch):
    """HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass forces the independent backend even
    for a text-gen framework (explicit env route wins over the default)."""
    monkeypatch.setenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "bypass")
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "orchestrator_mode": "bypass", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert any("bypass_trace_analysis.py" in c for c in cmd)
    assert "--tracelens-root" not in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_text_gen_defaults_to_tracelens_agent(session_dir, monkeypatch):
    """Text-gen with no explicit route DEFAULTS to the TraceLens ``agent`` route
    (the shipped default). Bypass is reached only via an explicit route."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert not any("bypass_trace_analysis.py" in c for c in cmd)
    assert "--analysis-route" in cmd and "agent" in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_invalid_route_falls_back_to_agent(session_dir, monkeypatch):
    """An unknown analysis_route (e.g. an LLM typo) must NOT silently mis-route;
    it falls back to the default TraceLens ``agent`` route and surfaces a warning."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "analysis_route": "foobar",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert "--analysis-route" in cmd and "agent" in cmd
    codes = {w.get("code") for w in res.get("trace_health_warnings", [])}
    assert "invalid_analysis_route" in codes


@pytest.mark.asyncio
async def test_trace_analyze_handler_scriptable_converges_route_params(session_dir, monkeypatch):
    """Scriptable (xDiT) params converge by route: --skip-split is TraceLens-only
    (must NOT reach bypass, which would crash argparse -> degraded), while
    --num-denoise-steps is forwarded to BOTH routes (bypass consumes it)."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    base = {
        "trace_input": str(fake_trace),
        "session_id": session_dir.name,
        "framework": "xdit",
        "num_denoise_steps": 20,
        "top_k": 5,
    }
    # Explicit bypass route: no --skip-split, but --num-denoise-steps forwarded.
    await krh.trace_analyze_handler({**base, "analysis_route": "bypass"}, session_dir=session_dir)
    cmd = captured["cmd"]
    assert any("bypass_trace_analysis.py" in c for c in cmd)
    assert "--skip-split" not in cmd
    assert "--num-denoise-steps" in cmd and "20" in cmd
    # TraceLens (deterministic) route: both flags present.
    await krh.trace_analyze_handler({**base, "analysis_route": "deterministic"}, session_dir=session_dir)
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert "--skip-split" in cmd
    assert "--num-denoise-steps" in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_text_gen_deterministic_escapes_to_tracelens(session_dir, monkeypatch):
    """TraceLens stays reachable as an explicit escape hatch: text-gen with
    analysis_route=deterministic runs the TraceLens tool, not bypass."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "sglang",
            "analysis_route": "deterministic",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert "--tracelens-root" in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_xdit_explicit_route_overrides_bypass(session_dir, monkeypatch):
    """An explicit route wins over the xDiT bypass default (e.g. forcing the
    TraceLens deterministic route)."""
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setattr(krh, "_resolve_tracelens_root", lambda: session_dir)
    monkeypatch.setattr(krh, "_tracelens_root_error", lambda root: None)
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "orchestrator_mode": "deterministic", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "framework": "xdit",
            "analysis_route": "deterministic",
            "top_k": 5,
        },
        session_dir=session_dir,
    )
    cmd = captured["cmd"]
    assert any("tracelens_analysis.py" in c for c in cmd)
    assert "--analysis-route" in cmd and "deterministic" in cmd


@pytest.mark.asyncio
async def test_trace_analyze_handler_records_bypass_discovery_success(
    session_dir,
    monkeypatch,
):
    """Deterministic route surfaces a kernel_journey discovery run labelled
    source="bypass" (with the real hot kernels), while version provenance stays
    under the tracelens toolchain (no junk versions["bypass"])."""
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()

    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        payload = {
            "status": "ok",
            "orchestrator_mode": "deterministic",
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "name": "fused_moe",
                    "gpu_pct": 42.0,
                    "bottleneck": "memory",
                    "reusable_native_kernel": True,
                },
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
async def test_trace_analyze_handler_omits_top_k_when_not_requested(
    session_dir,
    monkeypatch,
):
    """Without an explicit ``top_k`` the handler must NOT pass
    ``--top-k`` so tracelens_analysis.py applies its own large-pool default
    (candidate-build cap decoupled from the dispatch-side budget)."""
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "deterministic",
        },
        session_dir=session_dir,
    )
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "--top-k" not in captured["cmd"]


@pytest.mark.asyncio
async def test_trace_analyze_handler_does_not_forward_top_k(
    session_dir,
    monkeypatch,
):
    """``top_k`` is not a tool flag; the live dial is ``HYPERLOOM_KERNEL_CANDIDATES_TOP_K``.

    A payload still carrying the key must be ignored rather than reach an
    argparse that no longer defines it.
    """
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    captured: dict = {}

    async def fake_run_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = list(cmd)
        return 0, json.dumps({"status": "ok", "hot_kernels": []}), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.trace_analyze_handler(
        {
            "trace_input": str(fake_trace),
            "session_id": session_dir.name,
            "analysis_route": "deterministic",
            "top_k": 20,
        },
        session_dir=session_dir,
    )
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "--top-k" not in captured["cmd"]


@pytest.mark.asyncio
async def test_trace_analyze_handler_records_bypass_discovery_failed(
    session_dir,
    monkeypatch,
):
    """Fail-loud deterministic pipeline -> discovery run status=failed with the
    error text and an empty hot-kernel list, still labelled source="bypass"."""
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

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
    session_dir,
    monkeypatch,
):
    """High-idle gate suppresses hot kernels but the run still succeeds -> a
    bypass discovery run with status=ok and hot_kernel_count=0."""
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

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
    session_dir,
    monkeypatch,
):
    """The LLM/agent route keeps source="tracelens" (regression guard for the
    bypass relabel)."""
    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts

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
    session_dir,
    monkeypatch,
):
    """When the payload omits framework/gpu_type/model, the handler falls back to SharedState for the real workload context."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.framework = "vllm"
    state.gpu_type = "mi300x"
    state.model_path = "/path/models/Qwen3-30B-A3B"
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
    session_dir,
    monkeypatch,
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
    session_dir,
    tmp_path,
    monkeypatch,
):
    """Disk candidates must carry the TraceLens report path for GEAK prompts."""
    report_path = tmp_path / "analysis.md"
    report_path.write_text("# TraceLens Report\n", encoding="utf-8")
    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k1",
                        "name": "paged_attention",
                        "source_file": "/sgl-workspace/sglang/kernels/paged.py",
                        "reusable_native_kernel": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_subprocess(cmd, *, timeout_sec):
        return (
            0,
            json.dumps(
                {
                    "status": "ok",
                    "hot_kernels": json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"],
                    "trace_report_path": str(report_path),
                    "artifact_paths": {
                        "kernel_candidates": str(candidates_path),
                        "trace_report_path": str(report_path),
                    },
                }
            ),
            "",
        )

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
    session_dir,
    tmp_path,
    monkeypatch,
):
    """GEAK candidates must inherit the materialized Magpie workload config."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    OPENAI_API_KEY: "should-not-leak"
""",
        encoding="utf-8",
    )
    state = SharedState.load_or_init(session_dir)
    state.baseline_config_path = str(config_path)
    state.save(session_dir)

    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k1",
                        "name": "paged_attention",
                        "source_file": "/sgl-workspace/sglang/kernels/paged.py",
                        "reusable_native_kernel": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_subprocess(cmd, *, timeout_sec):
        return (
            0,
            json.dumps(
                {
                    "status": "ok",
                    "hot_kernels": json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"],
                    "artifact_paths": {"kernel_candidates": str(candidates_path)},
                }
            ),
            "",
        )

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)

    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )

    enriched = json.loads(candidates_path.read_text(encoding="utf-8"))["hot_kernels"][0]
    assert res["hot_kernels"][0]["env_vars"]["SGLANG_USE_TRITON"] == "1"
    assert enriched["env_vars"]["TP"] == "8"
    assert enriched["env_vars"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "OPENAI_API_KEY" not in enriched["env_vars"]
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
    session_dir,
    monkeypatch,
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
    # HYPERLOOM_KERNEL_AGENT_ROOT is a lazy env read; delenv exercises the "not configured" branch.
    monkeypatch.delenv("HYPERLOOM_KERNEL_AGENT_ROOT", raising=False)
    res = await krh.trace_analyze_handler(
        {"trace_input": str(session_dir)},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert res["error_class"] == "kernel_agent_root_missing"
    assert "HYPERLOOM_KERNEL_AGENT_ROOT is not set" in res["error"]


# TraceLens permanent failure stays failed (no fallback).


@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_keeps_tool_failure_failed(
    session_dir,
    monkeypatch,
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
    assert res["hot_kernels"] == [], "stale hot_kernels must be cleared on tool failure"
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
    session_dir,
    monkeypatch,
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
    session_dir,
    monkeypatch,
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


# trace_health_warnings must reach the Orchestration LLM.


def test_record_trace_analyze_persists_trace_health_warnings(session_dir):
    """``record_trace_analyze`` keeps ``trace_health_warnings`` verbatim in ``last_trace_analyze`` for next-tick rendering."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    """``task_groups`` must flow into ``last_trace_analyze`` so the multi-KEEP queue collapses members of the same AST function into one slot."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    groups = [
        {
            "primary_kernel_id": "k004",
            "kernel_ids": ["k003", "k004"],
            "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        },
        {
            "primary_kernel_id": "k002",
            "kernel_ids": ["k001", "k002"],
            "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        },
    ]
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "gpu_pct": 8.0,
                    "reusable_native_kernel": True,
                    "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
                },
                {
                    "kernel_id": "k002",
                    "gpu_pct": 25.0,
                    "reusable_native_kernel": True,
                    "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
                },
                {
                    "kernel_id": "k003",
                    "gpu_pct": 12.0,
                    "reusable_native_kernel": True,
                    "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
                },
                {
                    "kernel_id": "k004",
                    "gpu_pct": 38.0,
                    "reusable_native_kernel": True,
                    "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
                },
            ],
            "task_groups": groups,
        },
    )
    assert state.last_trace_analyze.get("task_groups") == groups
    # After k002 + k004 attempted, group-aware collapse reports no untried kernels.
    state.record_kernel_opt(
        {
            "status": "failed",
            "kernel_id": "k002",
            "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            "error_class": "subtask_exception",
        }
    )
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k004",
            "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            "proposal": {"decision": "KEEP", "reasons": []},
            "verification": {
                "micro_speedup": 1.17,
                "compile_passed": True,
                "correctness_passed": True,
                "best_artifact_path": "/tmp/k004.py",
            },
        }
    )
    assert state.untried_hot_reusable_kernels() == [], (
        "k001/k003 must be filtered out because their groups have an attempted member (k002 / k004 respectively)"
    )


def test_record_trace_analyze_defaults_task_groups_to_empty_list(session_dir):
    """With no ``task_groups`` field (legacy TraceLens output), the cached entry defaults to an empty list."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                "not-a-dict",
                {"severity": "warning"},  # missing 'code'
                {"code": "high_gpu_idle_pct", "idle_pct": 30.0, "threshold_pct": 20.0},
                None,
            ],
        },
    )
    warnings = state.last_trace_analyze["trace_health_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "high_gpu_idle_pct"


def test_format_last_trace_analyze_renders_idle_warning_inline(session_dir):
    """Prompt rendering: a persisted idle warning surfaces inline with its numeric context."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "high_gpu_idle_pct" in rendered
    assert "60.5%" in rendered
    assert "20.0%" in rendered
    assert "warnings=[" in rendered


def test_format_last_trace_analyze_renders_low_compute_warning_numbers(session_dir):
    """The compact line must carry the numbers the Coordinator routes on.

    ``low_gpu_compute_pct`` exists to send a run to comm/params instead of
    kernel rewriting, and telling a comm-bound window from a host-bound one
    needs ``exposed_comm_pct``. A per-field ``if`` chain that only knew
    ``idle_pct`` rendered this warning as a bare code with every number gone.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace.json.gz"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                {
                    "code": "low_gpu_compute_pct",
                    "severity": "warning",
                    "compute_pct": 3.99,
                    "threshold_pct": 10.0,
                    "exposed_comm_pct": 95.99,
                    "source": "/tmp/x/analysis.md",
                    "message": "low compute",
                }
            ],
        },
    )
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "low_gpu_compute_pct(" in rendered
    assert "compute=3.99%" in rendered
    assert "exposed_comm=95.99%" in rendered
    assert "threshold=10.0%" in rendered


def test_format_last_trace_analyze_renders_both_gate_warnings(session_dir):
    """Both gates can fire on one window; neither may erase the other's numbers."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace.json.gz"},
        {
            "status": "ok",
            "hot_kernels": [],
            "trace_health_warnings": [
                {"code": "high_gpu_idle_pct", "severity": "warning", "idle_pct": 95.0, "threshold_pct": 80.0},
                {"code": "low_gpu_compute_pct", "severity": "warning", "compute_pct": 3.0, "threshold_pct": 10.0},
            ],
        },
    )
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "high_gpu_idle_pct(idle=95.0%,threshold=80.0%)" in rendered
    assert "low_gpu_compute_pct(compute=3.0%,threshold=10.0%)" in rendered


def test_format_last_trace_analyze_renders_failure_warning_with_rc(session_dir):
    """Tool-failure warning carries ``returncode``; the prompt must surface ``rc=N`` to distinguish a crash from a benign skip."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "tracelens_analysis_failed" in rendered
    assert "rc=1" in rendered


def test_format_last_trace_analyze_omits_warnings_suffix_in_steady_state(session_dir):
    """Format-stability guard: with no warnings, the prompt line must NOT gain a ``warnings=[]`` suffix (snapshot tests pin the legacy format)."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace"},
        {
            "status": "ok",
            "hot_kernels": [{"kernel_id": "k1", "reusable_native_kernel": True}],
        },
    )
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "warnings=" not in rendered, "no warnings → no warnings= suffix; this keeps existing prompt snapshots stable"


@pytest.mark.asyncio
async def test_t5_handler_to_sharedstate_e2e_idle_warning_reaches_prompt(
    session_dir,
    monkeypatch,
):
    """End-to-end: T3 idle warning flows handler → SharedState.last_trace_analyze → Orchestration prompt line."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    # Handler boundary carries the warning.
    assert res["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"

    # SharedState persists it.
    state = SharedState.load_or_init(session_dir)
    state.record_trace_analyze({"trace_input": str(session_dir)}, res)
    assert state.last_trace_analyze["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"

    # Prompt rendering surfaces it.
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "high_gpu_idle_pct" in rendered
    assert "42.0%" in rendered


@pytest.mark.asyncio
async def test_t5_handler_to_sharedstate_e2e_failure_warning_reaches_prompt(
    session_dir,
    monkeypatch,
):
    """T4: a permanent TraceLens failure warning must reach the Orchestration prompt."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    rendered = state._format_trace_analyze_blob(state.last_trace_analyze)
    assert "tracelens_analysis_failed" in rendered
    assert "rc=1" in rendered


@pytest.mark.asyncio
async def test_trace_analyze_handler_t4_failure_appends_to_existing_warnings(
    session_dir,
    monkeypatch,
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


@pytest.mark.asyncio
async def test_run_optimization_handler_missing_kernel_id(session_dir):
    # ``source_file`` short-circuits the ``missing_trace_analyze`` guard so the legacy missing-kernel_id path is exercised.
    res = await krh.run_optimization_handler(
        {"source_file": "/tmp/dummy.py"},
        session_dir=session_dir,
    )
    assert res["status"] == "failed"
    assert "kernel_id" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_empty_queue_skips_cleanly(session_dir, tmp_path):
    # Empty eligible queue (all candidates non-reusable) with no specific kernel
    # named (the post-GEMM auto pass shape) must finish as a clean skip, not a
    # "missing 'kernel_id'" GEAK failure.
    candidates_path = _write_candidates_json(
        tmp_path,
        {
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "name": "fused_moe",
                    "source_file": "/sgl-workspace/aiter/moe.py",
                    "reusable_native_kernel": False,
                    "duration_us": 100.0,
                    "gpu_pct": 12.0,
                },
            ],
        },
    )
    assert krh._batch_kernel_candidates({"candidates_path": str(candidates_path)}) == []
    res = await krh.run_optimization_handler(
        {"candidates_path": str(candidates_path), "session_id": session_dir.name},
        session_dir=session_dir,
    )
    assert res["status"] == "skipped"
    assert res["reason"] == "no_eligible_kernels"
    assert res.get("error_class") is None
    assert res["kernels_considered"] == 1


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
async def test_run_optimization_handler_forwards_extra_server_args(session_dir):
    captured: dict[str, object] = {}

    async def fake_run(cmd, *, timeout_sec):
        captured["cmd"] = cmd
        captured["timeout_sec"] = timeout_sec
        return 0, '{"status": "ok"}', ""

    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "source_file": "/sgl-workspace/sglang/python/sglang/fake.py",
        "extra_server_args": "--kv-cache-dtype fp8 --page-size 16",
        "dry_run": True,
        "_single_kernel": True,
    }
    with (
        patch.object(krh, "_validate_reusable_native_kernel", return_value=None),
        patch.object(krh, "_run_subprocess", side_effect=fake_run),
    ):
        res = await krh.run_optimization_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    cmd = captured["cmd"]
    assert "--extra-sglang-args" in cmd
    assert cmd[cmd.index("--extra-sglang-args") + 1] == "--kv-cache-dtype fp8 --page-size 16"


def test_run_optimization_handler_backfills_target_platform_from_state(session_dir):
    from hyperloom.orchestrator.state.shared_state import SharedState

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
    with (
        patch.object(krh, "_validate_reusable_native_kernel", return_value=None),
        patch.object(krh, "_run_subprocess", side_effect=fake_run),
    ):
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


# _batch_kernel_candidates collapses task_group members.
def _write_candidates_json(tmp_path, payload):
    p = tmp_path / "kernel_candidates.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_batch_kernel_candidates_collapses_task_group_to_primary(tmp_path):
    """Two reusable kernels in the same task_group dispatch as ONE candidate (the primary), with the full group attached."""
    # Rows must carry gpu_pct >= 10.0 to pass the default hot-kernel gate. Both
    # group members are above it on purpose: the collapse must be what drops
    # k002, not the gate.
    candidates_path = _write_candidates_json(
        tmp_path,
        {
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "name": "rms_norm_prefill",
                    "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                    "reusable_native_kernel": True,
                    "duration_us": 100.0,
                    "gpu_pct": 12.0,
                },
                {
                    "kernel_id": "k002",
                    "name": "rms_norm_decode",
                    "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                    "reusable_native_kernel": True,
                    "duration_us": 50.0,
                    "gpu_pct": 11.0,
                },
                {
                    "kernel_id": "k003",
                    "name": "other_kernel",
                    "source_file": "/sgl-workspace/aiter/other.py",
                    "reusable_native_kernel": True,
                    "duration_us": 30.0,
                    "gpu_pct": 10.5,
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
        },
    )
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
    # Rows must carry gpu_pct >= 10.0 to be retained by the dispatcher, so the
    # fallback member is above the gate and can only be dropped by the rejection.
    candidates_path = _write_candidates_json(
        tmp_path,
        {
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "name": "rocblas_sgemm_call",
                    "source_file": "/sgl-workspace/aiter/foo.py",
                    "reusable_native_kernel": False,  # primary rejected
                    "duration_us": 200.0,
                    "gpu_pct": 22.0,
                },
                {
                    "kernel_id": "k002",
                    "name": "rms_norm_call",
                    "source_file": "/sgl-workspace/aiter/foo.py",
                    "reusable_native_kernel": True,
                    "duration_us": 50.0,
                    "gpu_pct": 12.5,
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
        },
    )
    selected = krh._batch_kernel_candidates({"candidates_path": str(candidates_path)})
    # k002 (the only reusable member) replaces the rejected primary.
    assert [c["kernel_id"] for c in selected] == ["k002"]
    assert selected[0]["task_group"]["task_group_id"] == "tg001"


def test_batch_kernel_candidates_legacy_path_unchanged_without_task_groups(tmp_path):
    """With no task_groups[] (legacy runs), the candidate list matches pre-PR-B behaviour."""
    # Legacy fixture carries gpu_pct >= 3.0 so the hot-kernel gate doesn't drop k001.
    candidates_path = _write_candidates_json(
        tmp_path,
        {
            "hot_kernels": [
                {
                    "kernel_id": "k001",
                    "name": "rms_norm",
                    "source_file": "/sgl-workspace/aiter/rmsnorm.py",
                    "reusable_native_kernel": True,
                    "gpu_pct": 11.0,
                },
                {
                    "kernel_id": "k002",
                    "name": "vendor",
                    "source_file": "/sgl-workspace/aiter/vendor.py",
                    "reusable_native_kernel": False,
                    "gpu_pct": 9.0,
                },
            ],
        },
    )
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

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"trace_analyze": fake_handler}):
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={
                        "target_agent": "kernel_agent",
                        "kind": "trace_analyze",
                        "params": {"trace_input": "/tmp/fake-trace.json.gz"},
                    },
                ),
            )
            req_msgs = await c.bus.tail(topic="request", to_agent="kernel_agent")
            assert req_msgs, "request must be mirrored to kernel inbox"
            req_id = req_msgs[0].msg_id

            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs, "handler must emit RESPONSE without LLM"
            r = resp_msgs[0]
            assert r.from_agent == "kernel_agent"
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
async def test_coordinator_request_unknown_kind_auto_rejected(session_dir):
    """REQUEST with no registered handler emits an auto-reject RESPONSE."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.kernel_enabled = True
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "invent_brand_new_kind",
                },
            ),
        )
        req_msgs = await c.bus.tail(topic="request", to_agent="kernel_agent")
        assert req_msgs, "request must be recorded on bus"
        resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
        assert resp_msgs, "auto-reject RESPONSE must be emitted"
        r = resp_msgs[0]
        assert r.from_agent == "kernel_agent"
        assert r.payload["status"] == "failed"
        assert r.payload["result"]["error_class"] == "unknown_kernel_kind"
        assert r.payload["source"] == "coordinator_auto_reject"
        assert "valid_kinds" in r.payload["result"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_kernel_disabled_auto_rejected(session_dir):
    """REQUEST to kernel_agent when kernel_enabled=False emits agent_disabled RESPONSE."""
    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.kernel_enabled = False
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "trace_analyze",
                    "params": {"trace_input": "/tmp/t.json.gz"},
                },
            ),
        )
        resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
        assert resp_msgs, "auto-reject RESPONSE must be emitted"
        r = resp_msgs[0]
        assert r.payload["status"] == "failed"
        assert r.payload["result"]["error_class"] == "agent_disabled"
        assert r.payload["source"] == "coordinator_auto_reject"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_handler_exception_recorded(session_dir):
    """Handler crashes → RESPONSE.status='failed' + error_class set."""
    c = Coordinator(session_dir, backends=_backends_silent())

    async def bad_handler(payload, *, session_dir):
        raise RuntimeError("boom")

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"trace_analyze": bad_handler}):
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={"target_agent": "kernel_agent", "kind": "trace_analyze"},
                ),
            )
            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs
            r = resp_msgs[0]
            assert r.payload["status"] == "failed"
            assert r.payload["result"]["error_class"] == "handler_exception"
            assert "boom" in r.payload["result"]["error"]
        finally:
            await c.stop()


# Batch dispatch enablers: batch-parallel sizing + candidates_path injection.
def test_default_kernel_batch_parallel_matches_full_node():
    """Default fanout is sized for a single MI300X / MI355X node (8 GPU) so a
    typical ``run_optimization`` batch does NOT serialize behind an asyncio
    semaphore tighter than Ray's view of the cluster."""
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
    # ``_sequence_denial_for_request`` needs baseline_tput > 0 and
    # last_profile_trace set; simulate the post-baseline + post-profile state.
    c.shared_state.baseline_tput = 1234.5
    c.shared_state.last_profile_trace = "/path/trace/x.json.gz"
    cached_path = "/path/cached/kernel_candidates.json"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/path/trace/x.json.gz",
        "candidates_path": cached_path,
    }
    # The gate also consults ``last_select_kernels``; seed it with the same trace.
    c.shared_state.last_select_kernels = {
        "trace_input": "/path/trace/x.json.gz",
        "candidates_path": cached_path,
    }
    explicit = "/path/operator/override_candidates.json"

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"run_optimization": fake_handler}):
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={
                        "target_agent": "kernel_agent",
                        "kind": "run_optimization",
                        "params": {
                            "kernel_id": "k001",
                            "candidates_path": explicit,
                        },
                    },
                ),
            )
            assert captured["payload"].get("candidates_path") == explicit
        finally:
            await c.stop()


# Multi-KEEP integrate queue: streaming record_partial, batch_mode dedup, base_tput auto-injection.
@pytest.mark.asyncio
async def test_run_optimization_handler_invokes_record_partial_per_sub_result(
    session_dir,
):
    """Each batch sub-attempt's result must flow through record_partial the
    moment _run_kernel_backend_sequence returns, NOT only after
    asyncio.gather() wait-all releases, so one slow GEAK sibling doesn't
    delay integrate-queue visibility for the fast KEEPs."""
    candidates = [
        {"kernel_id": "kA", "source_file": "/p/a.py", "reusable_native_kernel": True},
        {"kernel_id": "kB", "source_file": "/p/b.py", "reusable_native_kernel": True},
        {"kernel_id": "kC", "source_file": "/p/c.py", "reusable_native_kernel": True},
    ]

    completion_log: list[str] = []
    recorded: list[dict] = []

    # Deterministic completion order kB -> kC -> kA via an explicit gate chain
    # instead of real sleeps, so the assertion never depends on wall-clock
    # timing (which is flaky under a loaded parallel test run).
    _gates = {kid: asyncio.Event() for kid in ("kA", "kB", "kC")}
    _release_after = {"kB": "kC", "kC": "kA"}  # kB done -> release kC -> release kA

    async def fake_sequence(base_payload, candidate, *, session_dir, parallel_backends=False):
        kid = str(candidate.get("kernel_id"))
        if kid != "kB":
            # kC and kA wait until their predecessor signals completion.
            await _gates[kid].wait()
        completion_log.append(kid)
        nxt = _release_after.get(kid)
        if nxt:
            _gates[nxt].set()
        return {
            "status": "ok",
            "kernel_id": kid,
            "source_file": candidate["source_file"],
            "proposal": {"decision": "KEEP" if kid in ("kB", "kC") else "REVERT"},
            "verification": {"micro_speedup": 1.5 if kid == "kB" else 2.0},
        }

    def record_partial(result: dict) -> None:
        recorded.append(
            {
                "kernel_id": result.get("kernel_id"),
                "decision": (result.get("proposal") or {}).get("decision"),
            }
        )

    with patch.object(krh, "_run_kernel_backend_sequence", side_effect=fake_sequence):
        await krh._run_optimization_batch(
            payload={
                "candidates_path": "/dummy",
                # Synthetic order avoids the forge batch serialization path; the
                # monkeypatched sequence below is what this test exercises.
                "backend_order": "synthetic",
                "max_parallel": 3,
                "parallel_backends": False,
            },
            candidates=candidates,
            session_dir=session_dir,
            record_partial=record_partial,
        )

    # Callback must have fired for every candidate, in completion order
    # (NOT input order). kB runs ungated first, then releases kC, then kA.
    assert [r["kernel_id"] for r in recorded] == ["kB", "kC", "kA"], recorded
    assert completion_log == ["kB", "kC", "kA"]


@pytest.mark.asyncio
async def test_backend_ladder_breaks_on_first_keep(session_dir, monkeypatch):
    """When forge already KEEPs, the ladder short-circuits."""
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    calls: list[str] = []

    async def fake_single(child, *, session_dir, timeout_override_sec=None):
        backend = child["backends"]
        calls.append(backend)
        if backend == "forge":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {
                    "micro_speedup": 1.50,
                    "correctness_passed": True,
                    "best_artifact_path": "/tmp/forge.py",
                },
            }
        raise AssertionError(f"ladder must NOT run {backend!r} after forge KEEP")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy", "backend_order": "forge"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py", "reusable_native_kernel": True},
            session_dir=session_dir,
        )

    assert calls == ["forge"]
    assert (best.get("proposal") or {}).get("decision") == "KEEP"
    assert (best.get("verification") or {}).get("micro_speedup") == 1.50


@pytest.mark.asyncio
async def test_backend_sequence_forge_keep_short_circuits(session_dir, monkeypatch):
    """Forge runs first and a KEEP short-circuits before GEAK fallback.

    Regression coverage for Bugbot: _kernel_result_rank() returns a tuple, so
    the short-circuit must inspect the KEEP slot instead of comparing the tuple
    directly to int 0.
    """
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    calls: list[str] = []

    async def fake_single(child, *, session_dir, timeout_override_sec=None):
        backend = child["backends"]
        calls.append(backend)
        if backend == "forge":
            return {
                "status": "ok",
                "kernel_id": child["kernel_id"],
                "proposal": {"decision": "KEEP", "reasons": []},
                "verification": {"micro_speedup": 1.05, "best_artifact_path": "/tmp/forge.py"},
            }
        raise AssertionError(f"forge KEEP must short-circuit before {backend!r}")

    with patch.object(krh, "_run_optimization_single", side_effect=fake_single):
        best = await krh._run_kernel_backend_sequence(
            {"candidates_path": "/dummy", "backend_order": "forge"},
            {"kernel_id": "k004", "source_file": "/p/moe_op.py", "reusable_native_kernel": True},
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
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
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
        {"candidates_path": "/dummy", "backend_order": "forge", "max_parallel": 8},
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
        base_payload,
        candidate,
        *,
        session_dir,
        parallel_backends=False,
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
        recorded.append(
            {
                "kernel_id": result.get("kernel_id"),
                "status": result.get("status"),
                "decision": (result.get("proposal") or {}).get("decision"),
                "error_class": result.get("error_class"),
            }
        )

    with patch.object(krh, "_run_kernel_backend_sequence", side_effect=fake_sequence):
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
    c.shared_state.last_profile_trace = "/path/trace/x.json.gz"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/path/trace/x.json.gz",
        "candidates_path": "/path/cached/candidates.json",
    }
    # The sequence gate also consults ``last_select_kernels``.
    c.shared_state.last_select_kernels = dict(c.shared_state.last_trace_analyze)
    c.shared_state.current_best = {
        "action": "integrate",
        "tput": 4500.0,
        "kernel_id": "k009",
    }

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        return {"status": "ok", "decision": "KEEP", "new_tput": 4620.0, "gain_pct": 2.7, "kernel_id": "k001"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"integrate": fake_handler}):
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={
                        "target_agent": "kernel_agent",
                        "kind": "integrate",
                        "params": {
                            "kernel_id": "k001",
                            "patch_path": "/tmp/k001.py",
                            "target_file": "/p/moe_op.py",
                            # no base_tput intentionally
                        },
                    },
                ),
            )
        finally:
            await c.stop()

    assert captured["payload"].get("base_tput") == 4500.0, (
        "Coordinator must auto-inject base_tput from current_best.tput"
    )


@pytest.mark.asyncio
async def test_coordinator_does_not_overwrite_explicit_base_tput_on_integrate(
    session_dir,
):
    """Explicit operator-supplied ``base_tput`` must NOT be clobbered by the auto-injection."""
    c = Coordinator(session_dir, backends=_backends_silent())
    c.shared_state.baseline_tput = 4319.5
    c.shared_state.last_profile_trace = "/path/trace/x.json.gz"
    c.shared_state.last_trace_analyze = {
        "trace_input": "/path/trace/x.json.gz",
        "candidates_path": "/path/cached/candidates.json",
    }
    c.shared_state.last_select_kernels = dict(c.shared_state.last_trace_analyze)
    c.shared_state.current_best = {"action": "backends", "tput": 4500.0}

    captured: dict = {}

    async def fake_handler(payload, *, session_dir, **kwargs):
        captured["payload"] = dict(payload)
        return {"status": "ok", "decision": "NEEDS_REVIEW", "new_tput": 4400.0, "gain_pct": 0.0, "kernel_id": "k009"}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"integrate": fake_handler}):
        try:
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={
                        "target_agent": "kernel_agent",
                        "kind": "integrate",
                        "params": {
                            "kernel_id": "k009",
                            "patch_path": "/tmp/k009.py",
                            "target_file": "/p/rmsnorm.py",
                            "base_tput": 4200.0,  # operator override
                        },
                    },
                ),
            )
        finally:
            await c.stop()

    assert captured["payload"].get("base_tput") == 4200.0, (
        "Explicit base_tput must take precedence over current_best.tput"
    )


@pytest.fixture
def _candidates_factory(tmp_path):
    """Write a kernel_candidates.json fixture and return its path."""

    def _make(hot_kernels, task_groups=None):
        path = tmp_path / "kernel_candidates.json"
        path.write_text(
            json.dumps(
                {
                    "hot_kernels": hot_kernels,
                    "task_groups": task_groups or [],
                    "reusable_native_kernel_ids": [],
                }
            )
        )
        return str(path)

    return _make


def test_batch_candidates_filters_rejected_kernel_ids(
    session_dir,
    _candidates_factory,
):
    """A kernel on rejected_kernel_ids must not appear in the next batch, even if still in kernel_candidates.json."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    cpath = _candidates_factory(
        [
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
        ]
    )
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k001"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k002"]


def test_batch_candidates_filters_kernels_with_recorded_attempts(
    session_dir,
    _candidates_factory,
):
    """max_attempts=1 default: any prior attempt skips the kernel in the next batch."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    cpath = _candidates_factory(
        [
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
        ]
    )
    state = SharedState.load_or_init(session_dir)
    # k001 has an attempt recorded but is not yet on the rejected list (PARTIAL below max_partial).
    state.kernel_opt_attempts = {
        "k001": {"attempts": 1, "partial_count": 1, "last_decision": "PARTIAL"},
    }
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    assert [c.get("kernel_id") for c in out] == ["k002"]


def test_batch_candidates_task_group_falls_back_to_live_member(
    session_dir,
    _candidates_factory,
):
    """When the primary (k002) is rejected, the task_group still dispatches via the next live member (k001)."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    cpath = _candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k002"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    # Group dispatches as k001 with the original task_group attached.
    assert len(out) == 1
    assert out[0]["kernel_id"] == "k001"
    assert out[0].get("task_group", {}).get("primary_kernel_id") == "k002"


def test_batch_candidates_skips_group_when_all_members_rejected(
    session_dir,
    _candidates_factory,
):
    """If every member of a task_group is unusable, the group skips cleanly."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    cpath = _candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k009", "gpu_pct": 10.0, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    state = SharedState.load_or_init(session_dir)
    state.rejected_kernel_ids = ["k001", "k002"]
    state.save(session_dir)

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    # moe_op.py group fully retired; only k009 remains.
    assert out_ids == ["k009"]


def test_batch_candidates_skips_in_flight_kernels(
    session_dir,
    _candidates_factory,
):
    """In-flight defense: a status/ko-*.json with state=running for k004 keeps it out of the next batch."""
    cpath = _candidates_factory(
        [
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
        ]
    )
    # Plant a running status file for k004.
    status_dir = session_dir / "kernel-agent" / "runs" / session_dir.name / "status" / "kernel_optimization"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "ko-deadbeef.json").write_text(
        json.dumps(
            {
                "state": "running",
                "current_step": "run_backends",
                "pid": 123456,
                "last_lines": ["kernel_id=k004", "selected_backends=forge"],
            }
        )
    )

    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k001"]


def test_batch_candidates_below_min_gpu_pct_skipped(
    session_dir,
    _candidates_factory,
    monkeypatch,
):
    """min_gpu_pct env=5.0 keeps tiny rmsnorm kernels out of the batch."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "5.0")
    cpath = _candidates_factory(
        [
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k005", "gpu_pct": 2.8, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
        ]
    )
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath},
        session_dir=session_dir,
    )
    out_ids = sorted(c.get("kernel_id") for c in out)
    assert out_ids == ["k001"]


def test_batch_candidates_default_min_gpu_pct_matches_sharedstate_gate(
    session_dir,
    _candidates_factory,
):
    """The dispatch batch and the phase-advance gate must apply the same GPU floor.

    ``_batch_kernel_candidates`` decides what a dispatch actually runs;
    ``SharedState.untried_hot_reusable_kernels`` decides whether KERNEL still
    owes work and what the report calls unattempted. Drift either way is a live
    defect: a lower batch floor advances the phase past kernels it would still
    dispatch, a lower state floor holds the phase open on kernels no dispatch
    will ever pick up. So assert the two selections are equal, not merely that
    each contains what this test expected.

    The straddling pair is derived from the shipped default instead of written
    in. Hardcoding it is what let this test go on asserting a 10% boundary at a
    5% default, and pinning the default's own value is already
    ``test_untried_hot_kernels_returns_only_reusable_above_threshold``'s job.

    Every candidate gets its own source file. When two share one, the batch
    side's op-fanout dedup merges the weaker away and the exclusion passes for a
    reason the floor had no part in -- which is how the 10% assertion above kept
    passing at a 5% default.
    """
    from hyperloom.orchestrator.state.kernel_decision_settings import (
        _DEFAULT_HOT_KERNEL_MIN_GPU_PCT as floor,
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    margin = 0.13
    assert floor - margin > 0, f"floor {floor} too small to straddle by {margin}"
    hot_kernels = [
        {
            "kernel_id": "k001",
            "gpu_pct": round(floor + 30.0, 2),
            "reusable_native_kernel": True,
            "source_file": "/p/moe_op.py",
        },
        {
            "kernel_id": "k006",
            "gpu_pct": round(floor - margin, 2),
            "reusable_native_kernel": True,
            "source_file": "/p/rmsnorm.py",
        },
        {
            "kernel_id": "k008",
            "gpu_pct": round(floor + margin, 2),
            "reusable_native_kernel": True,
            "source_file": "/p/silu_and_mul.py",
        },
    ]
    cpath = _candidates_factory(hot_kernels)

    state = SharedState.load_or_init(session_dir)
    state.last_trace_analyze = {"hot_kernels": hot_kernels, "task_groups": []}
    state.save(session_dir)

    skipped: dict[str, str] = {}
    batch_ids = sorted(
        c.get("kernel_id")
        for c in krh._batch_kernel_candidates(
            {"candidates_path": cpath},
            session_dir=session_dir,
            skipped_out=skipped,
        )
    )

    assert batch_ids == ["k001", "k008"], (batch_ids, skipped)
    # The floor, not the dedup, has to be what dropped the sub-threshold row.
    assert "below_min_gpu_pct" in skipped.get("k006", ""), skipped
    assert sorted(state.untried_hot_reusable_kernels()) == batch_ids


def test_in_flight_kernel_ids_returns_running_only(session_dir):
    status_dir = session_dir / "kernel-agent" / "runs" / session_dir.name / "status" / "kernel_optimization"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "ko-aaa.json").write_text(
        json.dumps(
            {
                "state": "running",
                "last_lines": ["kernel_id=k001"],
            }
        )
    )
    (status_dir / "ko-bbb.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "last_lines": ["kernel_id=k002"],
            }
        )
    )
    out = krh._in_flight_kernel_ids(session_dir)
    assert out == {"k001"}


def test_resolve_integrate_payload_falls_back_to_kernel_opt_attempts_ledger(
    session_dir,
):
    """``_resolve_integrate_payload`` looks up patch_path / source_file from the per-kernel ``kernel_opt_attempts`` ledger so any queued KEEP can integrate."""
    from hyperloom.orchestrator.state.shared_state import SharedState

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
            "last_decision": "KEEP",
            "last_micro_speedup": 4.13,
            "last_artifact_path": "/tmp/k009.py",
            "last_source_file": "/p/rmsnorm.py",
        },
        "k001": {
            "last_decision": "KEEP",
            "last_micro_speedup": 2.0,
            "last_artifact_path": "/tmp/k001.py",
            "last_source_file": "/p/moe_op.py",
        },
    }
    state.save(session_dir)

    # integrate(k001) carries only the kernel_id (the second queued KEEP, not last_kernel_opt).
    resolved, missing = krh._resolve_integrate_payload(
        {"kernel_id": "k001", "base_tput": 4500.0},
        session_dir=session_dir,
    )
    assert missing is None, missing
    assert resolved.get("patch_path") == "/tmp/k001.py", (
        "patch_path must fall back to kernel_opt_attempts[k001].last_artifact_path"
    )
    assert resolved.get("source_file") == "/p/moe_op.py", (
        "source_file must fall back to kernel_opt_attempts[k001].last_source_file"
    )


def _gz_trace(path: Path, payload_bytes: int) -> Path:
    """Write a ``*.trace.json.gz`` of roughly the requested size."""
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": [{"n": "x" * payload_bytes}]}, fh)
    return path


def test_trace_files_for_dir_excludes_split_chunks_and_leads_with_the_capture(tmp_path):
    """Splitter chunks must never lead the discovered trace list.

    Two consumers fall back to ``trace_files[0]`` when ``main_trace_path`` is
    absent (the roofline trace extractor and the writeback path). Under
    alphabetical order a 900-byte ``trace_split/`` chunk sorted ahead of
    ``rank_0.trace.json.gz``, and a single chunk handed to ``--trace-input``
    takes the single-file branch of discovery, where the multi-candidate probing
    downstream cannot rescue it. This function already excludes ``capture_traces``
    sidecars for the same reason.
    """
    trace_dir = tmp_path / "torch_trace"
    chunk = _gz_trace(trace_dir / "trace_split" / "aaa_mixed_0.trace.json.gz", 32)
    capture = _gz_trace(trace_dir / "zzz_rank_0.trace.json.gz", 40_000)
    sidecar = _gz_trace(trace_dir / "capture_traces" / "aaa_graph_capture_0.pt.trace.json.gz", 32)

    found = _trace_files_for_dir(trace_dir)

    assert chunk not in found, "trace_split chunks must be excluded"
    assert sidecar not in found, "capture_traces sidecars must stay excluded"
    assert found[0] == capture


def test_trace_files_for_dir_orders_by_size_not_name(tmp_path):
    """Size ordering, so the fallback does not depend on a naming rule.

    The real discriminator between a fragment and a capture is that one is
    hundreds of bytes and the other is hundreds of kilobytes. Ranking on size
    gets this right without knowing any of the splitter's filename conventions,
    which it is free to change.
    """
    trace_dir = tmp_path / "torch_trace"
    small = _gz_trace(trace_dir / "aaa_first_by_name.trace.json.gz", 16)
    large = _gz_trace(trace_dir / "zzz_last_by_name.trace.json.gz", 60_000)

    found = _trace_files_for_dir(trace_dir)

    assert found == [large, small]


@pytest.mark.parametrize(
    ("relative_path", "rank"),
    [
        ("177-TP-0-DECODE.trace.json.gz", 0),
        ("worker-rank-3.pt.trace.json.gz", 3),
        ("worker-rank0.pt.trace.json.gz", 0),
        ("dp0_pp0_tp0_dcp0_ep0_rank0.1787293265778058798.pt.trace.json.gz", 0),
        ("dp0_pp0_tp7_dcp0_ep7_rank7.1787293266292722593.pt.trace.json.gz", 7),
        ("dp1_pp0_tp0_dcp0_ep0_rank8.1787293265008841647.pt.trace.json.gz", 8),
        ("dp1_pp0_tp3_dcp0_ep3_rank11.1787293275126074931.pt.trace.json.gz", 11),
        ("rank_5/trace.pt.trace.json.gz", 5),
        ("rank_5/worker-TP-3.pt.trace.json.gz", 3),
        ("model-tp8.trace.json.gz", None),
        ("benchmark_sglang_tp_8/torch_trace/trace.pt.trace.json.gz", None),
        ("merged-177.trace.json.gz", None),
    ],
)
def test_trace_rank_supports_framework_naming(relative_path, rank):
    assert _trace_rank(Path(relative_path)) == rank


def test_agentx_primary_trace_prefers_rank_zero_over_merged(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    merged = trace_dir / "merged-177.trace.json.gz"
    rank_zero_warmup = trace_dir / "100-TP-0-WARMUP.trace.json.gz"
    rank_zero = trace_dir / "900-TP-0-DECODE.trace.json.gz"
    rank_one = trace_dir / "177-TP-1-DECODE.trace.json.gz"
    merged.write_bytes(b"merged")
    rank_zero_warmup.write_bytes(b"x")
    rank_zero.write_bytes(b"x" * 100)
    rank_one.write_bytes(b"rank-one")

    selected = _preferred_main_trace_path(
        trace_dir,
        [rank_zero_warmup, merged, rank_one, rank_zero],
        require_single_rank=True,
        tensor_parallel_size=2,
    )

    assert selected == rank_zero


def test_agentx_primary_trace_does_not_fall_back_to_multi_rank_merge(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    merged = trace_dir / "merged-177.trace.json.gz"

    assert (
        _preferred_main_trace_path(
            trace_dir,
            [merged],
            require_single_rank=True,
            tensor_parallel_size=8,
        )
        is None
    )


def test_agentx_single_unranked_trace_is_safe_without_tp_environment(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    trace = trace_dir / "worker.pt.trace.json.gz"

    assert (
        _preferred_main_trace_path(
            trace_dir,
            [trace],
            require_single_rank=True,
            tensor_parallel_size=None,
        )
        == trace
    )


@pytest.mark.parametrize(
    "trace_name",
    [
        "worker-rank-3.pt.trace.json.gz",
        "worker.pt.trace.json.gz",
    ],
)
def test_agentx_tp8_does_not_substitute_only_non_primary_trace(tmp_path, trace_name):
    trace_dir = tmp_path / "torch_trace"
    trace = trace_dir / trace_name

    assert (
        _preferred_main_trace_path(
            trace_dir,
            [trace],
            require_single_rank=True,
            tensor_parallel_size=8,
        )
        is None
    )


def test_agentx_tp1_can_use_single_merged_trace_as_compatibility_fallback(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    merged = trace_dir / "merged-177.trace.json.gz"

    assert (
        _preferred_main_trace_path(
            trace_dir,
            [merged],
            require_single_rank=True,
            tensor_parallel_size=1,
        )
        == merged
    )


def test_trace_files_for_dir_survives_an_ancestor_named_trace_split(tmp_path):
    """An ancestor named ``trace_split`` must not empty the list.

    The exclusion is relative to the scanned directory. Tested absolutely, a
    capture that happened to live below such a directory would have every one of
    its traces excluded, and the caller reads an empty list as "no traces here".
    """
    trace_dir = tmp_path / "trace_split" / "run" / "torch_trace"
    capture = _gz_trace(trace_dir / "rank_0.trace.json.gz", 40_000)

    found = _trace_files_for_dir(trace_dir)

    assert found == [capture]
