# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``materialize_config_with_envs``: golden snapshots of
the real materialized YAML, plus RUN_EVAL warn-once and unset/extra_envs restore.

The snapshots carry ``AITER_LOG_TUNED_CONFIG`` because every serving run now gets
it: aiter logs a tuned-config MISS unconditionally but the matching HIT only
under that flag, and both GEMM shape discovery and apply verification read hits.
Its position differs between the snapshots because ``extra_server_args`` seeds
``EXTRA_*_ARGS`` earlier in the dict when a caller passes one.
"""

from __future__ import annotations

import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we

# Env names that leak process state into the materialized YAML. Cleared so the
# golden snapshots stay deterministic regardless of the runner's environment.
_LEAKY_ENV = (
    "TP",
    "EP",
    "ISL",
    "OSL",
    "CONC",
    "MAX_MODEL_LEN",
    "PRECISION",
    "RANDOM_RANGE_RATIO",
    "ROCR_VISIBLE_DEVICES",
    "RUN_EVAL",
    "PROFILE",
    "MODEL_PATH",
    "INFERENCEX_PATH",
    "PORT",
    "PROFILE_OSL",
    "HYPERLOOM_PROFILE_MAX_ITERS",
    "HYPERLOOM_PROFILE_DELAY_ITERS",
    "HYPERLOOM_PROFILE_MAX_STEPS_CAP",
    "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT",
    "INFERENCE_OPTIMIZER_SERVER_ARGS",
    "XDIT_QUALITY_REF",
    "XDIT_QUALITY_REF_WRITE",
)


def _isolate(monkeypatch):
    for k in _LEAKY_ENV:
        monkeypatch.delenv(k, raising=False)
    # Pin the two knobs that would otherwise depend on the host (GPU count and
    # runtime patching), keeping the golden output byte-stable.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")


def _write(path, bench):
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


# ── golden YAML snapshots ───────────────────────────────────────────────────

_GOLDEN_SGLANG_BASELINE = """\
benchmark:
  envs:
    TP: 1
    ROCR_VISIBLE_DEVICES: '0'
    NUM_PROMPTS: 320
    NUM_WARMUPS: 8
    EXTRA_SGLANG_ARGS: --variant 4 --watchdog-timeout 1800
    AITER_LOG_TUNED_CONFIG: '1'
    MAGPIE_TRUST_REMOTE_CODE: '1'
    BENCH_TRUST_REMOTE_CODE: '1'
    HF_HUB_TRUST_REMOTE_CODE: '1'
    RUN_EVAL: 'true'
  framework: sglang
  model: /models/foo
"""

_GOLDEN_VLLM = """\
benchmark:
  envs:
    TP: 1
    ROCR_VISIBLE_DEVICES: '0'
    NUM_PROMPTS: 320
    NUM_WARMUPS: 8
    EXTRA_VLLM_ARGS: --max-num-seqs 256
    AITER_LOG_TUNED_CONFIG: '1'
    MAGPIE_TRUST_REMOTE_CODE: '1'
    BENCH_TRUST_REMOTE_CODE: '1'
    HF_HUB_TRUST_REMOTE_CODE: '1'
    RUN_EVAL: 'true'
  framework: vllm
  model: /models/bar
"""

_GOLDEN_PROFILE = """\
benchmark:
  envs:
    PROFILE: '1'
    CONC: 64
    ISL: 1024
    OSL: 1024
    TP: 1
    ROCR_VISIBLE_DEVICES: '0'
    PROFILE_EXTRA_BODY: '{"start_step": 1536, "num_steps": 128, "shape_discovery":
      true, "detailed_annotations": true}'
    NUM_PROMPTS: 208
    NUM_WARMUPS: 8
    AITER_LOG_TUNED_CONFIG: '1'
    EXTRA_SGLANG_ARGS: --watchdog-timeout 1800
    MAGPIE_TRUST_REMOTE_CODE: '1'
    BENCH_TRUST_REMOTE_CODE: '1'
    HF_HUB_TRUST_REMOTE_CODE: '1'
    RUN_EVAL: 'true'
  framework: sglang
  model: /models/foo
"""


def test_golden_sglang_baseline(monkeypatch, tmp_path):
    _isolate(monkeypatch)
    src = _write(tmp_path / "sglang.yaml", {"framework": "sglang", "model": "/models/foo", "envs": {}})
    res = we.materialize_config_with_envs(
        src,
        tmp_path / "out",
        extra_server_args="--variant 4",
        args_mode="append",
    )
    assert res.read_text(encoding="utf-8") == _GOLDEN_SGLANG_BASELINE


def test_golden_vllm(monkeypatch, tmp_path):
    _isolate(monkeypatch)
    src = _write(tmp_path / "vllm.yaml", {"framework": "vllm", "model": "/models/bar", "envs": {}})
    res = we.materialize_config_with_envs(
        src,
        tmp_path / "out",
        extra_server_args="--max-num-seqs 256",
    )
    assert res.read_text(encoding="utf-8") == _GOLDEN_VLLM


def test_golden_profile_mode(monkeypatch, tmp_path):
    _isolate(monkeypatch)
    monkeypatch.setenv("ISL", "1024")
    monkeypatch.setenv("OSL", "1024")
    monkeypatch.setenv("CONC", "64")
    # PROFILE is read from benchmark.envs (YAML), not the process env: the loop
    # that copies process env into envs does not include PROFILE.
    src = _write(tmp_path / "profile.yaml", {"framework": "sglang", "model": "/models/foo", "envs": {"PROFILE": "1"}})
    res = we.materialize_config_with_envs(src, tmp_path / "out")
    assert res.read_text(encoding="utf-8") == _GOLDEN_PROFILE


# ── RUN_EVAL warn-once-per-process semantics ────────────────────────────────


def test_run_eval_disabled_warns_only_once_per_process(monkeypatch, tmp_path, caplog):
    _isolate(monkeypatch)
    monkeypatch.setenv("RUN_EVAL", "false")
    # Module-level global gates the warning; reset so this test owns the "once".
    monkeypatch.setattr(we, "_RUN_EVAL_DISABLED_WARN_EMITTED", False)
    src = _write(tmp_path / "cfg.yaml", {"framework": "sglang", "model": "/m", "envs": {}})
    with caplog.at_level("WARNING", logger=we.log.name):
        we.materialize_config_with_envs(src, tmp_path / "o1")
        we.materialize_config_with_envs(src, tmp_path / "o2")
    hits = [r for r in caplog.records if "RUN_EVAL is disabled" in r.message]
    assert len(hits) == 1
    assert we._RUN_EVAL_DISABLED_WARN_EMITTED is True


def test_run_eval_disabled_no_warn_when_already_emitted(monkeypatch, tmp_path, caplog):
    _isolate(monkeypatch)
    monkeypatch.setenv("RUN_EVAL", "false")
    # Simulate a process that already warned: the flag is set before the call.
    monkeypatch.setattr(we, "_RUN_EVAL_DISABLED_WARN_EMITTED", True)
    src = _write(tmp_path / "cfg.yaml", {"framework": "sglang", "model": "/m", "envs": {}})
    with caplog.at_level("WARNING", logger=we.log.name):
        we.materialize_config_with_envs(src, tmp_path / "o1")
    hits = [r for r in caplog.records if "RUN_EVAL is disabled" in r.message]
    assert hits == []


# ── unset_envs then restore-from-extra_envs semantics ───────────────────────


def test_unset_env_restored_from_extra_envs(monkeypatch, tmp_path):
    _isolate(monkeypatch)
    # KEEP exists in the YAML base, is overridden by extra_envs, then named in
    # unset_envs. The final restore loop re-injects the extra_envs value.
    src = _write(tmp_path / "cfg.yaml", {"framework": "sglang", "model": "/m", "envs": {"KEEP": "base"}})
    res = we.materialize_config_with_envs(
        src,
        tmp_path / "out",
        extra_envs={"KEEP": "from_extra"},
        unset_envs="KEEP",
    )
    envs = yaml.safe_load(res.read_text())["benchmark"]["envs"]
    assert envs["KEEP"] == "from_extra"


def test_unset_env_dropped_when_not_in_extra_envs(monkeypatch, tmp_path):
    _isolate(monkeypatch)
    # KEEP is only in the YAML base and named in unset_envs with no extra_envs
    # entry, so it is popped and never restored.
    src = _write(tmp_path / "cfg.yaml", {"framework": "sglang", "model": "/m", "envs": {"KEEP": "base"}})
    res = we.materialize_config_with_envs(
        src,
        tmp_path / "out",
        unset_envs="KEEP",
    )
    envs = yaml.safe_load(res.read_text())["benchmark"]["envs"]
    assert "KEEP" not in envs


# ── FLYDSL_EXTRA_SOURCE_DIRS is asked for, and never clobbers a set value ────


def _flydsl_env(monkeypatch, tmp_path, **kwargs):
    """Materialize with a real FlyDSL root on disk; return the rendered envs."""
    _isolate(monkeypatch)
    root = tmp_path / "flydsl"
    root.mkdir(parents=True)
    monkeypatch.setenv("FLYDSL_ROOT", str(root))
    monkeypatch.delenv("FLYDSL_EXTRA_SOURCE_DIRS", raising=False)
    src = _write(
        tmp_path / "cfg.yaml",
        {"framework": "vllm", "model": "/m", "envs": dict(kwargs.pop("yaml_envs", {}))},
    )
    res = we.materialize_config_with_envs(src, tmp_path / "out", **kwargs)
    return yaml.safe_load(res.read_text())["benchmark"]["envs"], str(root)


def test_an_ordinary_benchmark_is_not_given_the_flydsl_source_dirs(monkeypatch, tmp_path):
    # Widening FlyDSL's JIT cache key recompiles kernels, so a baseline, explore
    # or sweep run that patched nothing must not pay for it.
    envs, _root = _flydsl_env(monkeypatch, tmp_path)

    assert "FLYDSL_EXTRA_SOURCE_DIRS" not in envs


def test_the_run_that_applied_a_flydsl_patch_asks_for_the_source_dirs(monkeypatch, tmp_path):
    envs, root = _flydsl_env(monkeypatch, tmp_path, flydsl_source_dirs=True)

    # A host with its own FlyDSL install contributes further roots, so the
    # configured one being named is what matters, not the whole list.
    assert root in envs["FLYDSL_EXTRA_SOURCE_DIRS"].split(":")


def test_an_explicitly_configured_value_is_never_clobbered(monkeypatch, tmp_path):
    # The injection used to be the last writer of all, so it overwrote both the
    # YAML base and an extra_envs value the caller had threaded in on purpose.
    from_yaml, _root = _flydsl_env(
        monkeypatch,
        tmp_path,
        flydsl_source_dirs=True,
        yaml_envs={"FLYDSL_EXTRA_SOURCE_DIRS": "/operator/yaml"},
    )
    assert from_yaml["FLYDSL_EXTRA_SOURCE_DIRS"] == "/operator/yaml"

    from_extra, _root2 = _flydsl_env(
        monkeypatch,
        tmp_path / "second",
        flydsl_source_dirs=True,
        extra_envs={"FLYDSL_EXTRA_SOURCE_DIRS": "/operator/extra"},
    )
    assert from_extra["FLYDSL_EXTRA_SOURCE_DIRS"] == "/operator/extra"
