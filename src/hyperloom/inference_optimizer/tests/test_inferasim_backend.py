# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the InferaSim projection bridge.

No GPU and no Infera install: the projection and the harvest subprocess are
stubbed, so these verify benchmark-spec parsing, argv construction, model
preset resolution, anchor selection, harvest-on-miss, and that benchmark mode
fails closed rather than returning an uncalibrated number.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.orchestrator.actions.executors import inferasim_bridge as ib


def test_inferasim_is_not_a_benchmark_backend(monkeypatch):
    """Baseline, integrate and rebench must never be answered by a projection."""
    assert "inferasim" not in bb.KNOWN_BENCHMARK_BACKENDS
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "inferasim")
    assert bb.resolve_backend_name() == "magpie"


def test_spec_from_benchmark_parses_envs(monkeypatch):
    monkeypatch.delenv(ib.ENV_EP, raising=False)
    monkeypatch.delenv(ib.ENV_PP, raising=False)
    bench = {
        "framework": "vllm",
        "model": "/models/Qwen-Qwen3-14B",
        "precision": "fp8",
        "envs": {"TP": 4, "CONC": 128, "ISL": 2048, "OSL": 256},
    }
    spec = ib.spec_from_benchmark(bench)
    assert spec.framework == "vllm"
    assert spec.tp == 4
    assert spec.conc == 128
    assert spec.isl == 2048
    assert spec.osl == 256
    assert spec.weight_dtype == "fp8"


def test_spec_parses_ep_from_server_args(monkeypatch):
    monkeypatch.delenv(ib.ENV_EP, raising=False)
    bench = {
        "framework": "sglang",
        "model": "/models/gpt-oss-120b",
        "envs": {"TP": 8, "EXTRA_SGLANG_ARGS": "--ep-size 8 --foo 1"},
    }
    spec = ib.spec_from_benchmark(bench)
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
def test_spec_parses_kv_cache_dtype_from_server_args(monkeypatch, flag, expected):
    """An explore grid changes the KV dtype by passing this flag and nothing else.

    The projection prices KV dtype perfectly well, so dropping the flag made a
    lever the model *can* see look like one it cannot, and scored an fp8
    candidate as bf16 -- a candidate whose whole point is halving KV traffic.
    """
    monkeypatch.delenv(ib.ENV_KV_DTYPE, raising=False)
    spec = ib.spec_from_benchmark(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "EXTRA_VLLM_ARGS": flag},
        }
    )
    assert spec.kv_cache_dtype == expected


def test_kv_dtype_env_overrides_the_server_arg(monkeypatch):
    monkeypatch.setenv(ib.ENV_KV_DTYPE, "bf16")
    spec = ib.spec_from_benchmark(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "EXTRA_VLLM_ARGS": "--kv-cache-dtype fp8_e4m3"},
        }
    )
    assert spec.kv_cache_dtype == "bf16"


def test_max_num_seqs_caps_the_running_batch(monkeypatch):
    """A scheduler cap below the offered load is the batch the step actually runs."""
    spec = ib.spec_from_benchmark(
        {
            "framework": "vllm",
            "model": "/models/gpt-oss-120b",
            "envs": {"TP": 8, "CONC": 64, "EXTRA_VLLM_ARGS": "--max-num-seqs 32"},
        }
    )
    assert spec.conc == 32


def test_max_num_seqs_above_the_load_changes_nothing(monkeypatch):
    """Raising a cap nobody reaches is a no-op, and must not be reported as a win."""
    spec = ib.spec_from_benchmark(
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


def _write_anchor(
    path: Path, *, model: str, real_weights: bool, decode_ms: float, quant=None, kv="bf16", aiter=True
) -> None:
    """Minimal benchmark artifact in the shape benchmark_vllm.py emits."""
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "measured": {"model": {"prefill_ms": 10.0, "decode_ms": decode_ms}},
                "sweep": [{"batch": 16, "prefill_ms": 10.0, "decode_ms": decode_ms}],
                "meta": {
                    "model": model,
                    "batch": 16,
                    "input_len": 1024,
                    "tp": 1,
                    "quantization": quant,
                    "kv_cache_dtype": kv,
                    "use_aiter": aiter,
                    "real_weights": real_weights,
                    "load_format": "auto" if real_weights else "dummy",
                },
            }
        ),
        encoding="utf-8",
    )


def test_anchor_is_real_weights_detection(tmp_path):
    real, dummy = tmp_path / "r.json", tmp_path / "d.json"
    _write_anchor(real, model="m", real_weights=True, decode_ms=9.0)
    _write_anchor(dummy, model="m", real_weights=False, decode_ms=5.0)
    assert ib._anchor_is_real_weights(str(real)) is True
    assert ib._anchor_is_real_weights(str(dummy)) is False
    assert ib._anchor_is_real_weights(str(tmp_path / "missing.json")) is False


def test_recipe_from_spec_extracts_attention_backend():
    spec = ib.ServingSpec(
        framework="sglang",
        model_path="/models/x",
        extra_server_args="--attention-backend aiter --max-num-seqs 64",
    )
    recipe = ib.recipe_from_spec(spec)
    assert recipe["attention_backend"] == "aiter"
    assert recipe["weight_dtype"] == "bf16"


def test_select_anchor_prefers_explicit_env(tmp_path, monkeypatch):
    a = tmp_path / "explicit.json"
    _write_anchor(a, model="m", real_weights=True, decode_ms=9.0)
    monkeypatch.setenv(ib.ENV_ANCHOR, str(a))
    choice = ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m"))
    assert choice is not None
    assert choice.path == str(a)
    assert choice.regime_distance == 0


def test_select_anchor_none_without_store(monkeypatch):
    monkeypatch.delenv(ib.ENV_ANCHOR, raising=False)
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE, raising=False)
    assert ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m")) is None


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
    "points, sane",
    [
        ([(1, 4.0), (8, 6.4), (32, 12.0)], True),  # ordinary rising curve
        ([(16, 12.0)], True),  # single point: narrow, valid
        ([(8, 6.0), (16, 5.7)], True),  # -5%: run-to-run noise
        ([(16, 16.3), (64, 1.4)], False),  # differencing degenerated
        ([(4, 11.6), (32, 9.5)], False),  # decode faster at 8x batch
        ([(8, 0.0)], False),  # non-positive timing
        ([], False),  # nothing measured
    ],
)
def test_anchor_curve_sanity_gate(tmp_path, points, sane):
    p = tmp_path / "curve.json"
    _write_curve(p, points)
    assert ib.anchor_curve_is_sane(str(p)) is sane


def test_anchor_curve_sanity_gate_missing_file(tmp_path):
    assert ib.anchor_curve_is_sane(str(tmp_path / "nope.json")) is False


@pytest.mark.parametrize("payload", ["[]", '[{"batch": 1}]', '"text"', "12"])
def test_anchor_curve_sanity_gate_rejects_non_object_json(tmp_path, payload):
    """A JSON file that is not an artifact is rejected, not a crash.

    The anchor store sits next to analysis output, so the gate is pointed at
    whatever JSON is on disk; a list or scalar must not raise past the caller.
    """
    art = tmp_path / "not_an_artifact.json"
    art.write_text(payload)
    assert ib.anchor_curve_is_sane(str(art)) is False


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
    spec = ib.spec_from_benchmark(bench)
    assert spec.framework == "mori-sglang"
    assert spec.extra_server_args == "--attention-backend aiter"
    assert ib.recipe_from_spec(spec)["attention_backend"] == "aiter"
    # Distinct from its base engine, matching Infera's regime axis.
    assert ib.recipe_from_spec(spec)["engine"] != "sglang"
    argv = ib._build_argv(spec, "w.yaml")
    assert argv[argv.index("--serving-engine") + 1] == "mori-sglang"


def test_select_anchor_rejects_insane_anchor(tmp_path, monkeypatch):
    """A corrupt curve is worse than no anchor: fall back to pure analysis.

    Applies even to an operator-pinned anchor, which is the path most likely to
    point at a hand-picked artifact nobody re-validated.
    """
    bad = tmp_path / "bad.json"
    _write_curve(bad, [(16, 16.3), (64, 1.4)])
    monkeypatch.setenv(ib.ENV_ANCHOR, str(bad))
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE, raising=False)
    assert ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m")) is None

    good = tmp_path / "good.json"
    _write_curve(good, [(16, 12.0), (64, 20.0)])
    monkeypatch.setenv(ib.ENV_ANCHOR, str(good))
    choice = ib.select_anchor(ib.ServingSpec(framework="vllm", model_path="m"))
    assert choice is not None and choice.path == str(good)


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
    monkeypatch.setenv(ib.ENV_KV_DTYPE, "bf16")
    bench = {
        "framework": "vllm",
        "model": "/models/gpt-oss-120b",
        "envs": {"TP": 8, "CONC": 128, "EXTRA_VLLM_ARGS": "--kv-cache-dtype fp8"},
    }
    ambient = ib.spec_from_benchmark(bench)
    alone = ib.spec_from_benchmark(bench, ambient=False)
    assert (ambient.tp, ambient.conc, ambient.kv_cache_dtype) == (1, 8, "bf16")
    assert (alone.tp, alone.conc, alone.kv_cache_dtype) == (8, 128, "fp8")


# ── harvest on a miss, fail closed otherwise ─────────────────────────────────


def _infera_root(tmp_path: Path) -> Path:
    root = tmp_path / "infera"
    script = root / ib._HARVEST_SCRIPT_REL
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n")
    return root


def _spec(**kw) -> ib.ServingSpec:
    base = {"framework": "vllm", "model_path": "/models/gpt-oss-120b", "tp": 8, "conc": 64}
    base.update(kw)
    return ib.ServingSpec(**base)


def test_harvest_command_measures_the_candidates_regime(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_ROOT, str(_infera_root(tmp_path)))
    monkeypatch.delenv(ib.ENV_HARVEST_GPUS, raising=False)
    spec = _spec(
        framework="mori-sglang",
        ep=8,
        extra_server_args="--attention-backend aiter --tp-size 8 --max-running-requests 128",
    )
    cmd = ib.harvest_command(spec, "/store/a.json")
    assert cmd is not None
    arg = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    assert arg("--serving-backend") == "sglang"  # a build launches as its family
    assert arg("--tp") == "8"
    assert arg("--benchmark-gpus") == "4"  # reduced-width anchor, restored to TP8
    assert arg("--load-format") == "auto"  # real weights
    assert arg("--concurrency") == "128"
    assert "--enable-expert-parallel" in cmd
    server_args = next(c for c in cmd if c.startswith("--server-args="))
    assert "--attention-backend aiter" in server_args
    assert "--tp-size" not in server_args  # the harvest picks its own width


@pytest.mark.parametrize(
    "spec_kw, root_ok",
    [
        ({}, False),  # no Infera checkout
        ({"framework": "trtllm"}, True),  # engine the harvest cannot launch
        ({"model_path": ""}, True),  # nothing to load
    ],
)
def test_harvest_command_declines_when_it_cannot_run(tmp_path, monkeypatch, spec_kw, root_ok):
    if root_ok:
        monkeypatch.setenv(ib.ENV_ROOT, str(_infera_root(tmp_path)))
    else:
        monkeypatch.delenv(ib.ENV_ROOT, raising=False)
    assert ib.harvest_command(_spec(**spec_kw), "/store/a.json") is None


def test_benchmark_mode_fails_closed_without_an_anchor(monkeypatch):
    """No anchor and no way to harvest one must raise, not return an analytical number."""
    monkeypatch.setenv(ib.ENV_MODE, "benchmark")
    monkeypatch.delenv(ib.ENV_ANCHOR, raising=False)
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE, raising=False)
    with pytest.raises(ib.InferasimBridgeError, match="no in-regime anchor"):
        ib.resolve_anchor(_spec())


def test_harvest_disabled_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_ANCHOR_STORE, str(tmp_path / "store"))
    monkeypatch.setenv(ib.ENV_HARVEST, "0")
    monkeypatch.delenv(ib.ENV_ANCHOR, raising=False)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: None)
    with pytest.raises(ib.InferasimBridgeError):
        ib.resolve_anchor(_spec())


def test_an_in_regime_anchor_is_used_without_harvesting(monkeypatch):
    choice = ib.AnchorChoice(path="/a.json", regime_distance=0)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: choice)
    monkeypatch.setattr(ib, "harvest_anchor", lambda spec: pytest.fail("harvested despite a hit"))
    assert ib.resolve_anchor(_spec()) is choice


def test_an_out_of_regime_anchor_triggers_a_harvest(monkeypatch):
    near = ib.AnchorChoice(path="/near.json", regime_distance=1)
    fresh = ib.AnchorChoice(path="/fresh.json", regime_distance=0)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: near)
    monkeypatch.setattr(ib, "harvest_anchor", lambda spec: fresh)
    assert ib.resolve_anchor(_spec()) is fresh


def test_harvest_runs_once_indexes_and_selects(tmp_path, monkeypatch):
    """A miss boots one harvest; the artifact is indexed and then selected."""
    store = tmp_path / "store"
    monkeypatch.setenv(ib.ENV_ROOT, str(_infera_root(tmp_path)))
    monkeypatch.setenv(ib.ENV_ANCHOR_STORE, str(store))
    monkeypatch.delenv(ib.ENV_HARVEST, raising=False)
    monkeypatch.setattr(ib, "_ensure_infera_importable", lambda: None)

    indexed: list[str] = []

    class FakeStore:
        def __init__(self, root, discover=True):
            assert root == str(store)

        def add_artifact(self, path):
            indexed.append(path)

    import sys
    import types

    mod_name = "infera.projection.core.projection.inference_projection.search.anchor_store"
    for name in (
        "infera",
        "infera.projection",
        "infera.projection.core",
        "infera.projection.core.projection",
        "infera.projection.core.projection.inference_projection",
        "infera.projection.core.projection.inference_projection.search",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fake = types.ModuleType(mod_name)
    fake.AnchorStore = FakeStore
    monkeypatch.setitem(sys.modules, mod_name, fake)

    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        save = cmd[cmd.index("--save") + 1]
        Path(save).write_text(
            json.dumps({"sweep": [{"batch": 16, "decode_ms": 10.0}, {"batch": 64, "decode_ms": 14.0}], "meta": {}})
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ib.subprocess, "run", fake_run)
    selections = iter([None, ib.AnchorChoice(path="/harvested.json", regime_distance=0)])
    monkeypatch.setattr(ib, "select_anchor", lambda spec: next(selections))

    choice = ib.harvest_anchor(_spec())
    assert choice is not None and choice.path == "/harvested.json"
    assert len(runs) == 1
    assert len(indexed) == 1 and indexed[0].startswith(str(store / "harvested"))


def test_a_failed_harvest_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_ROOT, str(_infera_root(tmp_path)))
    monkeypatch.setenv(ib.ENV_ANCHOR_STORE, str(tmp_path / "store"))
    monkeypatch.delenv(ib.ENV_HARVEST, raising=False)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: None)
    monkeypatch.setattr(
        ib.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "engine failed to start")
    )
    assert ib.harvest_anchor(_spec()) is None


def test_a_corrupt_harvest_is_not_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv(ib.ENV_ROOT, str(_infera_root(tmp_path)))
    monkeypatch.setenv(ib.ENV_ANCHOR_STORE, str(tmp_path / "store"))
    monkeypatch.delenv(ib.ENV_HARVEST, raising=False)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: None)

    def fake_run(cmd, **kw):
        save = cmd[cmd.index("--save") + 1]
        Path(save).write_text(
            json.dumps({"sweep": [{"batch": 16, "decode_ms": 16.3}, {"batch": 64, "decode_ms": 1.4}]})
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ib.subprocess, "run", fake_run)
    monkeypatch.setattr(ib, "_ensure_infera_importable", lambda: pytest.fail("indexed a corrupt anchor"))
    assert ib.harvest_anchor(_spec()) is None


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
    m = ib._metrics_from_results(_spec(osl=1000, isl=1000), perf, None, des={"point": point})
    assert m.output_throughput == pytest.approx(1200.0)
    assert (m.ttft_ms, m.tpot_ms, m.itl_ms, m.e2el_ms) == (450.0, 12.5, 11.0, 13000.0)
    assert m.extras["estimator"] == "des"
    closed = ib._metrics_from_results(_spec(), perf, None)
    assert closed.output_throughput == pytest.approx(9999.0)
    assert closed.extras["estimator"] == "analytical"


# ── auto mode ────────────────────────────────────────────────────────────────


def test_auto_calibrates_only_on_an_in_regime_anchor(monkeypatch):
    monkeypatch.delenv(ib.ENV_MODE, raising=False)
    monkeypatch.setattr(ib, "select_anchor", lambda spec: None)
    assert ib.resolve_mode(_spec()) == ib.MODE_SIMULATE
    monkeypatch.setattr(ib, "select_anchor", lambda spec: ib.AnchorChoice(path="/a", regime_distance=1))
    assert ib.resolve_mode(_spec()) == ib.MODE_SIMULATE
    monkeypatch.setattr(ib, "select_anchor", lambda spec: ib.AnchorChoice(path="/a", regime_distance=0))
    assert ib.resolve_mode(_spec()) == ib.MODE_BENCHMARK
    monkeypatch.setenv(ib.ENV_MODE, "simulate")
    assert ib.resolve_mode(_spec()) == ib.MODE_SIMULATE


def test_a_miss_without_harvest_fails_closed_and_boots_nothing(monkeypatch):
    monkeypatch.setattr(ib, "select_anchor", lambda spec: ib.AnchorChoice(path="/a", regime_distance=1))
    monkeypatch.setattr(ib, "harvest_anchor", lambda spec: pytest.fail("auto mode harvested"))
    with pytest.raises(ib.InferasimBridgeError, match="could be found$"):
        ib.resolve_anchor(_spec(), harvest=False)


# ── decision rounds as anchors ───────────────────────────────────────────────


@pytest.fixture
def anchor_store(tmp_path, monkeypatch):
    store = tmp_path / "anchors"
    monkeypatch.setenv(ib.ENV_ANCHOR_STORE, str(store))
    indexed: list[str] = []
    monkeypatch.setattr(ib, "_index_artifact", lambda root, path: indexed.append(path))
    return store, indexed


def test_a_decision_round_is_recorded_as_a_served_single_point_anchor(anchor_store):
    store, indexed = anchor_store
    spec = _spec(
        framework="sglang", conc=64, extra_server_args="--attention-backend aiter", runtime={"image_digest": "x"}
    )
    path = ib.record_measured_anchor(spec, tpot_mean_ms=14.0, gpu="MI355X", source={"round_id": "explore-001"})
    assert path and indexed == [path]
    doc = json.loads(Path(path).read_text())
    assert doc["backend"] == "sglang"
    assert doc["sweep"] == [{"batch": 64, "decode_ms": 14.0}]
    meta = doc["meta"]
    assert (meta["tp"], meta["attention_backend"], meta["gpu_arch"]) == (8, "aiter", "mi355x")
    assert meta["runtime"] == {"image_digest": "x"}
    assert meta["sources"] == [{"batch": 64, "round_id": "explore-001"}]
    assert ib.anchor_curve_is_sane(path) and ib._anchor_is_served(path) and ib._anchor_is_real_weights(path)


def test_concurrencies_of_one_launch_build_a_ladder(anchor_store):
    first = ib.record_measured_anchor(_spec(conc=64), tpot_mean_ms=14.0)
    ib.record_measured_anchor(_spec(conc=16), tpot_mean_ms=9.0)
    again = ib.record_measured_anchor(_spec(conc=64), tpot_mean_ms=15.0)
    other = ib.record_measured_anchor(_spec(conc=16, extra_server_args="--max-num-seqs 16"), tpot_mean_ms=8.0)
    assert first == again != other
    sweep = json.loads(Path(first).read_text())["sweep"]
    assert sweep == [{"batch": 16, "decode_ms": 9.0}, {"batch": 64, "decode_ms": 15.0}]


def test_an_impossible_ladder_is_not_recorded(anchor_store):
    ib.record_measured_anchor(_spec(conc=16), tpot_mean_ms=20.0)
    assert ib.record_measured_anchor(_spec(conc=64), tpot_mean_ms=5.0) is None


def test_nothing_is_recorded_without_a_store_or_a_tpot(monkeypatch, anchor_store):
    assert ib.record_measured_anchor(_spec(), tpot_mean_ms=None) is None
    monkeypatch.delenv(ib.ENV_ANCHOR_STORE)
    assert ib.record_measured_anchor(_spec(), tpot_mean_ms=10.0) is None


def test_a_measured_anchor_from_another_image_conflicts(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"meta": {"runtime": {"image_digest": "old", "framework_version": "0.5"}}}))
    assert ib._runtime_conflicts(str(path), {"image_digest": "new"})
    assert not ib._runtime_conflicts(str(path), {"image_digest": "old"})
    assert not ib._runtime_conflicts(str(path), {"rocm": "7"})
    assert not ib._runtime_conflicts(str(path), {})


def test_gpu_arch_is_canonical(monkeypatch):
    monkeypatch.delenv(ib.ENV_GPU_ARCH, raising=False)
    assert ib.gpu_arch() == "mi355x"
    assert ib.gpu_arch("AMD Instinct MI300X") == "mi300x"
    assert ib.recipe_from_spec(_spec())["gpu_arch"] == "mi355x"
