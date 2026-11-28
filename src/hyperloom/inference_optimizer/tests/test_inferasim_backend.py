# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the InferaSim projection bridge.

No GPU and no Infera install: these verify benchmark-spec parsing, argv
construction, model preset resolution and the mapping of a projection's
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.common.benchmark_config import BenchmarkPoint
from hyperloom.orchestrator.actions.executors.explore_prefilter import inferasim_bridge as ib
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD, MI355X_CARD, publish_card


@pytest.fixture(autouse=True)
def _a_part_to_price(monkeypatch):
    """A projection is priced for a named part; a test about the session GPU unsets this one."""
    monkeypatch.setenv(ib.ENV_GPU_ARCH, "mi355x")
    monkeypatch.delenv(ib.ENV_HBM_GB, raising=False)


def _point_spec(bench: dict, **kwargs) -> ib.ServingSpec:
    return ib.spec_from_point(BenchmarkPoint.from_document({"benchmark": bench}), **kwargs)


def test_spec_from_point_parses_envs():
    bench = {
        "framework": "vllm",
        "model": "/models/Qwen-Qwen3-14B",
        "precision": "fp8",
        "envs": {"TP": 4, "CONC": 128, "ISL": 2048, "OSL": 256},
    }
    spec = _point_spec(bench)
    assert spec.framework == "vllm"
    assert spec.tp == 4
    assert spec.conc == 128
    assert spec.isl == 2048
    assert spec.osl == 256
    assert spec.weight_dtype == "fp8"


def test_spec_parses_ep_from_server_args():
    bench = {
        "framework": "sglang",
        "model": "/models/gpt-oss-120b",
        "envs": {"TP": 8, "EXTRA_SGLANG_ARGS": "--ep-size 8 --foo 1"},
    }
    spec = _point_spec(bench)
    assert spec.ep == 8


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--kv-cache-dtype fp8_e4m3", "fp8"),
        ("--kv-cache-dtype fp8", "fp8"),
        ("--kv-cache-dtype=fp8_e5m2", "fp8"),
        ("--kv-cache-dtype auto", "bf16"),  # auto follows the weights
        ("--foo 1", "bf16"),  # absent
    ],
)
def test_spec_parses_kv_cache_dtype_from_server_args(flag, expected):
    """An explore grid changes the KV dtype by passing this flag and nothing else.

    The projection prices KV dtype perfectly well, so dropping the flag made a
    lever the model *can* see look like one it cannot, and scored an fp8
    candidate as bf16 -- a candidate whose whole point is halving KV traffic.
    """
    spec = _point_spec(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "EXTRA_VLLM_ARGS": flag},
        }
    )
    assert spec.kv_cache_dtype == expected


def test_max_num_seqs_caps_the_running_batch(monkeypatch):
    """A scheduler cap below the offered load is the batch the step actually runs."""
    spec = _point_spec(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "CONC": 64, "EXTRA_VLLM_ARGS": "--max-num-seqs 32"},
        }
    )
    assert spec.conc == 32


def test_max_num_seqs_above_the_load_changes_nothing(monkeypatch):
    """Raising a cap nobody reaches is a no-op, and must not be reported as a win."""
    spec = _point_spec(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "CONC": 64, "EXTRA_VLLM_ARGS": "--max-num-seqs 512"},
        }
    )
    assert spec.conc == 64


def test_resolve_preset_heuristics_and_override(monkeypatch):
    monkeypatch.delenv(ib.ENV_MODEL, raising=False)
    assert ib.resolve_preset("/models/gpt-oss-120b") == "gpt_oss_120B"
    assert ib.resolve_preset("/data/Qwen-Qwen3-14B") == "qwen3_14B"
    assert ib.resolve_preset("/some/unknown-model") is None
    monkeypatch.setenv(ib.ENV_MODEL, "custom_preset")
    assert ib.resolve_preset("/models/gpt-oss-120b") == "custom_preset"


def test_resolve_workload_prefers_explicit_env(tmp_path, monkeypatch):
    wl = tmp_path / "custom_workload.yaml"
    wl.write_text("work_group: t\n", encoding="utf-8")
    monkeypatch.setenv(ib.ENV_WORKLOAD, str(wl))
    spec = ib.ServingSpec(framework="sglang", model_path="/m")
    workload, extra_env = ib._resolve_workload_and_env(spec)
    assert workload == str(wl.resolve())
    assert "INFERASIM_MODEL" not in extra_env


def test_resolve_workload_uses_template_for_preset(monkeypatch):
    monkeypatch.delenv(ib.ENV_WORKLOAD, raising=False)
    monkeypatch.setenv(ib.ENV_MODEL, "gpt_oss_120B")
    spec = ib.ServingSpec(framework="sglang", model_path="/models/gpt-oss-120b")
    workload, extra_env = ib._resolve_workload_and_env(spec)
    assert Path(workload).name == "inferasim_workload.yaml"
    assert extra_env["INFERASIM_MODEL"] == "gpt_oss_120B"


def test_recipe_from_spec_extracts_attention_backend():
    spec = ib.ServingSpec(
        framework="sglang",
        model_path="/models/x",
        extra_server_args="--attention-backend aiter --max-num-seqs 64",
    )
    recipe = ib.recipe_from_spec(spec)
    assert recipe["attention_backend"] == "aiter"
    assert recipe["weight_dtype"] == "bf16"


def _write_curve(path: Path, points: list[tuple[int, float]]) -> None:
    """Artifact carrying an explicit decode-vs-batch curve."""
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "sweep": [{"batch": b, "prefill_ms": 10.0, "decode_ms": d} for b, d in points],
                "meta": {"model": "m", "tp": 1, "input_len": 1024},
            }
        )
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ("", (None, 0)),
        ("--attention-backend triton", (None, 0)),
        ('--speculative-config \'{"method": "deepseek_mtp", "num_speculative_tokens": 3}\'', ("deepseek_mtp", 3)),
        ("--speculative-algorithm NEXTN --speculative-num-steps 3", ("NEXTN", 3)),
        ("--speculative-algorithm EAGLE3", ("EAGLE3", 1)),
        ("--method mtp --num-speculative-tokens 3", ("mtp", 3)),
        ("--method fp8", (None, 0)),
    ],
)
def test_parse_speculative_across_frameworks(args, expected):
    assert ib.parse_speculative(args) == expected


def test_recipe_marks_speculative_candidates_apart():
    """A speculating candidate must not share a regime with a plain one.

    Speculation changes how many tokens a step emits, so reusing a
    non-speculative anchor for it silently under-predicts throughput.
    """
    plain = ib.recipe_from_spec(ib.ServingSpec(framework="vllm", model_path="m"))
    mtp = ib.recipe_from_spec(
        ib.ServingSpec(
            framework="atom",
            model_path="m",
            extra_server_args="--method mtp --num-speculative-tokens 3",
        )
    )
    assert plain["speculative"] == "off"
    assert mtp["speculative"] == "spec:3"
    assert plain["speculative"] != mtp["speculative"]


def test_recipe_names_the_serving_engine():
    """Two engines serving one checkpoint are two regimes, not one.

    InferaSim scores an axis as matching when it is absent on either side, so
    leaving the engine off the recipe lets a vLLM anchor price an SGLang run.
    """
    engines = ("vllm", "sglang", "atom")
    recipes = [ib.recipe_from_spec(ib.ServingSpec(framework=fw, model_path="m")) for fw in engines]
    assert [r["engine"] for r in recipes] == list(engines)
    assert len({r["engine"] for r in recipes}) == len(engines)


def test_argv_names_the_serving_engine():
    """Tell InferaSim the engine too, so it can refuse a cross-engine anchor.

    Nothing in the analytical model reads it, so the projected number does not
    move; it decides which measured anchor calibration may read.
    """
    for fw in ("vllm", "sglang", "atom"):
        argv = ib._build_argv(ib.ServingSpec(framework=fw, model_path="m"), "w.yaml")
        assert argv[argv.index("--serving-engine") + 1] == fw


def test_argv_omits_the_engine_when_unknown():
    """An unnamed framework leaves the axis unset rather than emitting a blank."""
    argv = ib._build_argv(ib.ServingSpec(framework="", model_path="m"), "w.yaml")
    assert "--serving-engine" not in argv


def test_engine_build_keeps_its_name_and_still_reads_server_args(monkeypatch):
    """An engine build is its own regime, but shares its base engine's args env.

    Infera takes the engine as a free string, so ``mori-sglang`` must reach it
    verbatim rather than being folded into ``sglang``. Its server args still
    arrive in ``EXTRA_SGLANG_ARGS``, and dropping them would lose the very
    flags that decide the regime.
    """
    monkeypatch.delenv("MODEL", raising=False)
    bench = {
        "framework": "mori-sglang",
        "model": "/models/x",
        "envs": {"EXTRA_SGLANG_ARGS": "--attention-backend aiter", "TP": 4},
    }
    spec = _point_spec(bench)
    assert spec.framework == "mori-sglang"
    assert spec.extra_server_args == "--attention-backend aiter"
    assert ib.recipe_from_spec(spec)["attention_backend"] == "aiter"
    # Distinct from its base engine, matching Infera's regime axis.
    assert ib.recipe_from_spec(spec)["engine"] != "sglang"
    argv = ib._build_argv(spec, "w.yaml")
    assert argv[argv.index("--serving-engine") + 1] == "mori-sglang"


@pytest.mark.parametrize(
    "raw, mode",
    [(None, "auto"), ("", "auto"), ("simulate", "simulate"), ("Benchmark", "benchmark"), ("AUTO", "auto")],
)
def test_projection_mode_defaults_to_auto(monkeypatch, raw, mode):
    if raw is None:
        monkeypatch.delenv(ib.ENV_MODE, raising=False)
    else:
        monkeypatch.setenv(ib.ENV_MODE, raw)
    assert ib.projection_mode() == mode


def test_projection_mode_rejects_an_unknown_name(monkeypatch):
    """A typo for benchmark must not quietly fall back to uncalibrated numbers."""
    monkeypatch.setenv(ib.ENV_MODE, "benchmrak")
    with pytest.raises(ib.InferasimBridgeError):
        ib.projection_mode()


def _anchored_argv(tmp_path, monkeypatch, mode: str | None) -> list[str]:
    anchor = tmp_path / "anchor.json"
    _write_curve(anchor, [(16, 12.0), (64, 20.0)])
    scaling = tmp_path / "scaling.json"
    _write_curve(scaling, [(16, 6.0), (64, 10.0)])
    monkeypatch.setenv(ib.ENV_ANCHOR_SCALING, str(scaling))
    if mode is None:
        monkeypatch.delenv(ib.ENV_MODE, raising=False)
    else:
        monkeypatch.setenv(ib.ENV_MODE, mode)
    choice = ib.AnchorChoice(path=str(anchor), regime_distance=0, model="m")
    return ib._build_argv(ib.ServingSpec(framework="vllm", model_path="m"), "w.yaml", choice)


def test_simulate_mode_reads_no_anchor(tmp_path, monkeypatch):
    """The default projection is analytical even when anchors are configured."""
    argv = _anchored_argv(tmp_path, monkeypatch, None)
    assert "--load-benchmark" not in argv
    assert "--load-benchmark-scaling" not in argv
    assert argv[argv.index("--profiling-mode") + 1] == "simulate"


def test_benchmark_mode_calibrates_against_the_anchor(tmp_path, monkeypatch):
    argv = _anchored_argv(tmp_path, monkeypatch, "benchmark")
    assert argv[argv.index("--load-benchmark") + 1] == str(tmp_path / "anchor.json")
    assert argv[argv.index("--load-benchmark-scaling") + 1] == str(tmp_path / "scaling.json")


# ── reading a materialized config on its own ─────────────────────────────────


def test_ambient_env_is_ignored_when_asked(monkeypatch):
    """Comparing two materialized configs must not let an exported CONC flatten them."""
    monkeypatch.setenv("CONC", "8")
    monkeypatch.setenv("TP", "1")
    bench = {
        "framework": "vllm",
        "model": "/models/gpt-oss-120b",
        "envs": {"TP": 8, "CONC": 128, "EXTRA_VLLM_ARGS": "--kv-cache-dtype fp8"},
    }
    ambient = _point_spec(bench)
    alone = _point_spec(bench, ambient=False)
    assert (ambient.tp, ambient.conc) == (1, 8)
    assert (alone.tp, alone.conc, alone.kv_cache_dtype) == (8, 128, "fp8")


def _spec(**kw) -> ib.ServingSpec:
    base = {"framework": "vllm", "model_path": "/models/gpt-oss-120b", "tp": 8, "conc": 64}
    base.update(kw)
    return ib.ServingSpec(**base)


# ── closed-loop replay ───────────────────────────────────────────────────────


def test_the_replay_is_the_default_estimator(monkeypatch):
    monkeypatch.delenv(ib.ENV_ESTIMATOR, raising=False)
    spec = _spec(framework="sglang", conc=32, extra_server_args="--chunked-prefill-size 8192 --max-running-requests 64")
    argv = ib._build_argv(spec, "w.yaml")
    arg = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
    assert "--des-closed-loop" in argv
    assert arg("--des-num-requests") == "320"  # ten requests per client
    assert arg("--des-warmup-frac") == "0"
    assert arg("--chunked-prefill-size") == "8192"
    assert arg("--max-num-seqs") == "64"
    assert "--des-exclusive-prefill" in argv  # SGLang prefills alone


def test_the_replay_sizes_to_the_client_and_vllm_co_schedules_prefill(monkeypatch):
    monkeypatch.delenv(ib.ENV_ESTIMATOR, raising=False)
    argv = ib._build_argv(_spec(framework="vllm", conc=4, num_prompts=1000000), "w.yaml")
    assert argv[argv.index("--des-num-requests") + 1] == "4000"
    assert "--des-exclusive-prefill" not in argv
    argv = ib._build_argv(_spec(framework="vllm", conc=2), "w.yaml")
    assert argv[argv.index("--des-num-requests") + 1] == "64"


def test_the_closed_form_is_still_selectable(monkeypatch):
    monkeypatch.setenv(ib.ENV_ESTIMATOR, "analytical")
    assert "--des-closed-loop" not in ib._build_argv(_spec(), "w.yaml")
    monkeypatch.setenv(ib.ENV_ESTIMATOR, "replay")
    with pytest.raises(ib.InferasimBridgeError):
        ib.estimator()


def test_replayed_metrics_come_from_the_replay():
    from types import SimpleNamespace

    perf = SimpleNamespace(decode_throughput_tps=9999.0, ttft_ms=1.0, itl_ms=1.0, request_latency_ms=1.0, extras={})
    point = SimpleNamespace(
        system_throughput_tps=1200.0,
        ttft={"mean": 300.0},
        ttft_arrival={"mean": 450.0},
        tpot={"mean": 12.5},
        itl={"mean": 11.0},
        e2e={"mean": 13000.0},
    )
    m = ib._metrics_from_results(_spec(osl=1000, isl=1000), {"performance": perf, "des": {"point": point}})
    assert m.output_throughput == pytest.approx(1200.0)
    assert (m.ttft_ms, m.tpot_ms, m.itl_ms, m.e2el_ms) == (450.0, 12.5, 11.0, 13000.0)
    assert m.extras["estimator"] == "des"
    closed = ib._metrics_from_results(_spec(), {"performance": perf})
    assert closed.output_throughput == pytest.approx(9999.0)
    assert closed.extras["estimator"] == "analytical"


def test_gpu_arch_is_canonical(monkeypatch):
    monkeypatch.delenv(ib.ENV_GPU_ARCH)
    assert ib.gpu_arch() == ""
    assert ib.gpu_arch("AMD Instinct MI300X") == "mi300x"
    assert ib.recipe_from_spec(_spec())["gpu_arch"] == ""


def test_a_projection_with_no_part_to_price_is_refused(monkeypatch):
    monkeypatch.delenv(ib.ENV_GPU_ARCH)
    with pytest.raises(ib.InferasimBridgeError, match=ib.ENV_GPU_ARCH):
        ib._build_argv(_spec(), "w.yaml", mode=ib.MODE_SIMULATE, estimator_name=ib.ESTIMATOR_ANALYTICAL)


def test_the_projection_is_priced_for_the_session_gpu(monkeypatch):
    monkeypatch.delenv(ib.ENV_GPU_ARCH)
    publish_card(monkeypatch, MI325X_CARD)
    spec = _spec(gpu="MI325X")
    assert ib.recipe_from_spec(spec)["gpu_arch"] == "mi325x"
    argv = ib._build_argv(spec, "w.yaml", mode=ib.MODE_SIMULATE, estimator_name=ib.ESTIMATOR_ANALYTICAL)
    assert argv[argv.index("--gpu-arch") + 1] == "mi325x"
    assert argv[argv.index("--hbm-capacity-gb") + 1] == "256.0"


@pytest.mark.parametrize(("card", "hbm_gb"), [(MI355X_CARD, "288.0"), (None, None)])
def test_hbm_is_the_session_cards_memory(monkeypatch, card, hbm_gb):
    if card is not None:
        publish_card(monkeypatch, card)
    argv = ib._build_argv(_spec(), "w.yaml", mode=ib.MODE_SIMULATE, estimator_name=ib.ESTIMATOR_ANALYTICAL)
    assert (argv[argv.index("--hbm-capacity-gb") + 1] if "--hbm-capacity-gb" in argv else None) == hbm_gb


def test_an_operator_hbm_capacity_outranks_the_card(monkeypatch):
    publish_card(monkeypatch, MI355X_CARD)
    monkeypatch.setenv(ib.ENV_HBM_GB, "192")
    argv = ib._build_argv(_spec(), "w.yaml", mode=ib.MODE_SIMULATE, estimator_name=ib.ESTIMATOR_ANALYTICAL)
    assert argv[argv.index("--hbm-capacity-gb") + 1] == "192"


def test_the_arch_override_only_names_an_unknown_session_gpu(monkeypatch):
    monkeypatch.setenv(ib.ENV_GPU_ARCH, "mi325x")
    assert ib.recipe_from_spec(_spec())["gpu_arch"] == "mi325x"
    assert ib.recipe_from_spec(_spec(gpu="mi300x"))["gpu_arch"] == "mi300x"


@pytest.mark.parametrize(
    ("model", "preset"),
    [
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", None),
        ("deepseek-ai/DeepSeek-R1", "deepseek_v3"),
        ("deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek_v2_lite"),
        ("MiniMaxAI/MiniMax-M2.5", "minimax_m2.5"),
    ],
)
def test_resolve_preset_takes_the_most_specific_match_and_the_student_of_a_distill(monkeypatch, model, preset):
    monkeypatch.delenv(ib.ENV_MODEL, raising=False)
    assert ib.resolve_preset(model) == preset


def test_an_env_lever_the_projection_does_not_price_changes_the_launch_digest():
    def spec(aiter: str):
        bench = {
            "framework": "sglang",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": "8", "CONC": "64", "SGLANG_USE_AITER": aiter, "RESULT_DIR": "/out"},
        }
        return _point_spec(bench, ambient=False)

    on, off = spec("1"), spec("0")
    assert on.launch and on.launch != off.launch
